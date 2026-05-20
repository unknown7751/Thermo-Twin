"""
Phase 6 — Fleet Twin.

Manages N parallel digital twins (one TwinEngine per machine_id) and exposes
fleet-level aggregations:

  • get_fleet_health           — counts by status, average health, uptime %
  • get_dispatch_queue         — units sorted by RUL ascending + ROI estimate
  • detect_cross_unit_anomalies— pattern-matches divergence across units to
                                 surface multi-unit faults ("shared loop leak")

Each fleet unit owns its own SyntheticDataStreamer and an optional
fault-profile overlay so seeded units actually look different in real time
(different divergence patterns → cross-unit anomaly detection has signal).

Design notes
------------
• Fleet ticking is in-process (a background thread in app.py calls FleetManager.tick()
  each interval). Production fleets would push samples in over HTTP/MQTT —
  the FleetManager.process_sample(machine_id, sample) entry point already
  supports that path; tick() is just the demo driver.
• Heavy per-unit objects (Kalman, RUL, particle filter, physics) are spun up
  per twin. Fine for ~10 demo units; for hundreds you'd want shared physics.
• No locking on the registry — registration happens once at startup and the
  background tick loop and HTTP handlers all only *read* the dict afterwards.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from twin_engine import TwinEngine
from data_streamer import SyntheticDataStreamer
from physics.degradation_model import ComponentHealth

log = logging.getLogger("thermo-twin.fleet")

# Health → traffic-light bands (match the rest of the dashboard)
HEALTHY_THRESHOLD  = 80.0
WARNING_THRESHOLD  = 50.0

# Dispatch economics (₹) — coarse, demo-friendly figures
DISPATCH_COST_INR  = 22_000      # truck roll + tech time for proactive visit
INCIDENT_COST_INR  = 55_000      # average reactive-failure cost (downtime + emergency)
INR_PROACTIVE_SAVE = INCIDENT_COST_INR - DISPATCH_COST_INR


# ── Fault-profile overlays ────────────────────────────────────────────────────
# Applied to each raw sample so units with different profiles produce
# meaningfully different sensor traces and Kalman estimates diverge naturally.
def _apply_profile(sample: dict, profile: Optional[str]) -> dict:
    if profile is None:
        return sample
    s = dict(sample)
    if profile == "refrigerant_leak":
        s["discharge_pressure_psi"] *= 0.78
        s["supply_air_temp_c"]      += 5.0
    elif profile == "fan_failure":
        s["fan_rpm"]                *= 0.55
        s["discharge_pressure_psi"] *= 1.18
        s["compressor_power_kw"]    *= 1.25
        s["supply_air_temp_c"]      += 3.0
    elif profile == "compressor_wear":
        s["compressor_power_kw"]    *= 1.30
        s["discharge_pressure_psi"] *= 0.88
        s["supply_air_temp_c"]      += 2.0
    return s


@dataclass
class _FleetUnit:
    """Internal per-unit bundle: streamer + twin + metadata + optional fault profile."""
    twin: TwinEngine
    streamer: SyntheticDataStreamer
    metadata: dict
    fault_profile: Optional[str] = None
    sample_count: int = 0
    last_alert: Optional[dict] = field(default=None)


class FleetManager:
    def __init__(self, enable_influx: bool = False, enable_mqtt: bool = False, use_coolprop: bool = False):
        self._units: Dict[str, _FleetUnit] = {}
        self._enable_influx = enable_influx
        self._enable_mqtt   = enable_mqtt
        self._use_coolprop  = use_coolprop

    # ── Registry ──────────────────────────────────────────────────────────────
    def register_unit(
        self,
        machine_id: str,
        location: str = "",
        model: str = "Carrier-50000-Series",
        commissioned_date: Optional[str] = None,
        initial_health: Optional[ComponentHealth] = None,
        fault_profile: Optional[str] = None,
    ) -> bool:
        """
        Register a new HVAC unit in the fleet. Returns False if already registered.

        `fault_profile`, if set, biases the unit's generated samples so it
        produces a recognisable signature (refrigerant_leak / fan_failure /
        compressor_wear). The Kalman filter picks the bias up within ~30s.
        """
        if machine_id in self._units:
            log.warning("Unit %s already registered", machine_id)
            return False
        try:
            twin = TwinEngine(use_coolprop=self._use_coolprop, machine_id=machine_id)
            if initial_health is not None:
                # Seed Kalman state vector directly so it doesn't start at 100/100/100
                twin._kalman._x = np.array([
                    float(initial_health.refrigerant_charge_pct),
                    float(initial_health.compressor_efficiency_pct),
                    float(initial_health.fan_health_pct),
                ])
            streamer = SyntheticDataStreamer()
            metadata = {
                "machine_id":        machine_id,
                "location":          location,
                "model":             model,
                "commissioned_date": commissioned_date,
                "registered_at":     datetime.now(timezone.utc).isoformat(),
                "fault_profile":     fault_profile,
            }
            self._units[machine_id] = _FleetUnit(
                twin=twin, streamer=streamer,
                metadata=metadata, fault_profile=fault_profile,
            )
            log.info("Registered unit %s at %s (profile=%s)", machine_id, location, fault_profile)
            return True
        except Exception as exc:
            log.error("Failed to register %s: %s", machine_id, exc)
            return False

    def unregister_unit(self, machine_id: str) -> bool:
        if machine_id not in self._units:
            return False
        del self._units[machine_id]
        log.info("Unregistered unit %s", machine_id)
        return True

    def reset_unit(self, machine_id: str) -> bool:
        unit = self._units.get(machine_id)
        if unit is None:
            return False
        unit.twin.reset()
        unit.sample_count = 0
        log.info("Reset unit %s", machine_id)
        return True

    def list_units(self) -> List[str]:
        return list(self._units.keys())

    def get_metadata(self, machine_id: str) -> dict:
        return self._units[machine_id].metadata if machine_id in self._units else {}

    def get_twin(self, machine_id: str) -> TwinEngine:
        if machine_id not in self._units:
            raise KeyError(f"Unit {machine_id} not registered")
        return self._units[machine_id].twin

    # ── Driving the fleet ────────────────────────────────────────────────────
    def process_sample(self, machine_id: str, sample: dict) -> dict:
        """Run one sample through a specific unit's twin pipeline."""
        unit = self._units.get(machine_id)
        if unit is None:
            raise KeyError(f"Unit {machine_id} not registered")
        biased = _apply_profile(sample, unit.fault_profile)
        result = unit.twin.process_sample(biased)
        unit.sample_count += 1
        return result

    def tick(self) -> int:
        """
        Advance every registered unit by one sample. Returns number of units ticked.
        Called by the fleet background thread; safe to call from tests too.
        """
        n = 0
        for mid, unit in self._units.items():
            try:
                sample = unit.streamer.get_next_sample()
                biased = _apply_profile(sample, unit.fault_profile)
                unit.twin.process_sample(biased)
                unit.sample_count += 1
                n += 1
            except Exception as exc:
                log.error("Fleet tick failed for %s: %s", mid, exc)
        return n

    # ── Aggregations ─────────────────────────────────────────────────────────
    def _unit_avg_health(self, unit: _FleetUnit) -> Optional[float]:
        state = unit.twin._last_twin_state.get("state") if unit.twin._last_twin_state else None
        if not state:
            return None
        return float(np.mean([
            state["refrigerant_charge_pct"],
            state["compressor_efficiency_pct"],
            state["fan_health_pct"],
        ]))

    def get_fleet_health(self) -> dict:
        health_by_unit: Dict[str, float] = {}
        for mid, unit in self._units.items():
            avg = self._unit_avg_health(unit)
            if avg is not None:
                health_by_unit[mid] = avg

        healthy = [m for m, h in health_by_unit.items() if h > HEALTHY_THRESHOLD]
        warning = [m for m, h in health_by_unit.items() if WARNING_THRESHOLD < h <= HEALTHY_THRESHOLD]
        critical = [m for m, h in health_by_unit.items() if h <= WARNING_THRESHOLD]

        total   = len(self._units)
        scored  = len(health_by_unit)
        avg_pct = float(np.mean(list(health_by_unit.values()))) if health_by_unit else 100.0
        uptime  = ((len(healthy) + len(warning)) / scored * 100.0) if scored else 0.0

        return {
            "total_units":      total,
            "scored_units":     scored,
            "healthy":          len(healthy),
            "warning":          len(warning),
            "critical":         len(critical),
            "units_by_status":  {"healthy": healthy, "warning": warning, "critical": critical},
            "health_by_unit":   {m: round(h, 1) for m, h in health_by_unit.items()},
            "avg_health_pct":   round(avg_pct, 1),
            "fleet_uptime_pct": round(uptime, 1),
        }

    def get_dispatch_queue(self, top_n: Optional[int] = None) -> dict:
        """
        Units sorted by their most-critical RUL (ascending). For each unit
        we surface the most-critical component, its RUL central/CI, a
        fault probability, and a recommended action.
        """
        queue: List[dict] = []
        for mid, unit in self._units.items():
            rul = unit.twin._last_rul_state or {}
            if not rul:
                continue
            ruls = {
                "refrigerant": rul.get("refrigerant_rul_days", 9999.0),
                "compressor":  rul.get("compressor_rul_days",  9999.0),
                "fan":         rul.get("fan_rul_days",         9999.0),
            }
            critical_component = min(ruls, key=ruls.get)
            min_rul = float(ruls[critical_component])
            lo_key  = f"{critical_component}_rul_days_lower"
            hi_key  = f"{critical_component}_rul_days_upper"
            queue.append({
                "machine_id":              mid,
                "location":                unit.metadata.get("location", ""),
                "most_critical_component": critical_component,
                "rul_days":                round(min_rul, 1),
                "rul_days_lower":          round(float(rul.get(lo_key, min_rul * 0.8)), 1),
                "rul_days_upper":          round(float(rul.get(hi_key, min_rul * 1.2)), 1),
                "fault_probability":       round(self._fault_probability(min_rul), 3),
                "recommended_action":      self._recommendation(critical_component),
                "dispatch_cost_inr":       DISPATCH_COST_INR,
            })

        # Sort by urgency: lowest RUL first
        queue.sort(key=lambda x: x["rul_days"])
        for i, item in enumerate(queue):
            item["priority"] = i + 1

        sliced = queue if top_n is None else queue[:top_n]
        total_cost = sum(it["dispatch_cost_inr"] for it in sliced)
        # ROI: how much we save by handling proactively vs reactive failure
        save       = INR_PROACTIVE_SAVE * len(sliced)
        roi_pct    = round((save / total_cost) * 100.0, 1) if total_cost > 0 else 0.0

        return {
            "queue":                          sliced,
            "total_units_in_queue":           len(queue),
            "estimated_dispatch_cost_inr":    int(total_cost),
            "estimated_save_by_proactive_inr": int(save),
            "net_roi_pct":                    roi_pct,
        }

    def detect_cross_unit_anomalies(self, window_hours: int = 24) -> dict:
        """
        Pattern-match divergence signatures across units. When 2+ units show
        the same failure signature, flag it as a cross-unit anomaly and
        attribute a likely shared root cause (e.g., common supply manifold).
        """
        patterns = {}
        for mid, unit in self._units.items():
            tw = unit.twin._last_twin_state
            if not tw:
                continue
            div   = tw.get("divergence", {}) or {}
            state = tw.get("state", {}) or {}
            patterns[mid] = {
                "pressure_drop":       div.get("discharge_pressure_psi", 0) < -10,
                "pressure_rise":       div.get("discharge_pressure_psi", 0) > 10,
                "temp_rise":           div.get("supply_air_temp_c",      0) > 1.5,
                "rpm_drop":            div.get("fan_rpm",                0) < -100,
                "power_rise":          div.get("compressor_power_kw",    0) > 0.4,
                "refrigerant_health":  state.get("refrigerant_charge_pct", 100.0),
                "fan_health":          state.get("fan_health_pct",         100.0),
                "compressor_health":   state.get("compressor_efficiency_pct", 100.0),
                "location":            unit.metadata.get("location", ""),
            }

        anomalies = []

        # Rule 1 — Refrigerant loss across multiple units.
        # Either an actively-diverging unit (pressure drop + temp rise) OR a
        # steady-state already-degraded unit (refrigerant health < 85) counts.
        # The "already degraded" path catches fleets where the issue has been
        # present long enough that the Kalman estimate has converged.
        refrig_units = [
            m for m, p in patterns.items()
            if (p["pressure_drop"] and p["temp_rise"]) or p["refrigerant_health"] < 85
        ]
        if len(refrig_units) >= 2:
            cluster = self._location_cluster([patterns[m]["location"] for m in refrig_units])
            anomalies.append({
                "pattern":                 "refrigerant_loss",
                "affected_units":          refrig_units,
                "location_cluster":        cluster,
                "confidence":              round(min(0.99, 0.80 + 0.04 * len(refrig_units)), 2),
                "description":             f"{len(refrig_units)} units show refrigerant-loss pattern "
                                           f"(pressure drop + supply-temp rise)",
                "root_cause_hypothesis":   (f"Shared refrigerant loop / manifold in {cluster}"
                                            if cluster else "Distributed refrigerant system issue"),
                "recommended_action":      "Inspect common refrigerant manifold and compressor discharge "
                                           "line for leaks; perform pressure-decay test on shared loop.",
            })

        # Rule 2 — Condenser fan failure (active divergence OR degraded steady state)
        fan_units = [
            m for m, p in patterns.items()
            if (p["rpm_drop"] and p["pressure_rise"]) or p["fan_health"] < 75
        ]
        if len(fan_units) >= 2:
            cluster = self._location_cluster([patterns[m]["location"] for m in fan_units])
            anomalies.append({
                "pattern":                 "condenser_fan_failure",
                "affected_units":          fan_units,
                "location_cluster":        cluster,
                "confidence":              round(min(0.99, 0.75 + 0.04 * len(fan_units)), 2),
                "description":             f"{len(fan_units)} units show fan-failure pattern "
                                           f"(RPM drop + head-pressure rise)",
                "root_cause_hypothesis":   "Dust/debris in condenser fins, or electrical-supply issue "
                                           f"in {cluster}" if cluster else "Common electrical fault",
                "recommended_action":      "Clean condenser coils, check motor windings & contactors, "
                                           "audit shared electrical feed.",
            })

        # Rule 3 — Compressor wear (active power-rise OR degraded steady state)
        comp_units = [
            m for m, p in patterns.items()
            if (p["power_rise"] and p["compressor_health"] < 90) or p["compressor_health"] < 75
        ]
        if len(comp_units) >= 2:
            anomalies.append({
                "pattern":                 "compressor_wear",
                "affected_units":          comp_units,
                "location_cluster":        self._location_cluster([patterns[m]["location"] for m in comp_units]),
                "confidence":              round(min(0.95, 0.70 + 0.04 * len(comp_units)), 2),
                "description":             f"{len(comp_units)} units show progressive compressor wear "
                                           f"(power creep at constant load)",
                "root_cause_hypothesis":   "Same install batch / lubricant lifetime exhaustion",
                "recommended_action":      "Schedule compressor service window; consider lubricant "
                                           "analysis on the affected units.",
            })

        return {"anomalies": anomalies, "window_hours": window_hours}

    def _fault_probability(self, rul_days: float) -> float:
        """Sigmoid centred at 7 days — short RUL → high probability."""
        if rul_days >= 9000:   # MAX_RUL sentinel
            return 0.0
        if rul_days > 30:
            return 0.05
        return float(1.0 / (1.0 + np.exp(rul_days - 7.0)))

    def _recommendation(self, component: str) -> str:
        return {
            "refrigerant": "Dispatch with recharge kit + leak detector. Risk of cooling loss.",
            "compressor":  "Schedule compressor replacement within 2 weeks; monitor closely.",
            "fan":         "Replace fan motor; critical for condenser heat rejection.",
        }.get(component, "Schedule preventive maintenance.")

    @staticmethod
    def _location_cluster(locations: List[str]) -> str:
        """Common prefix (e.g. 'Building A' from a list of 'Building A, Floor N')."""
        prefixes = [(loc.split(",")[0].strip()) for loc in locations if loc]
        if prefixes and len(set(prefixes)) == 1:
            return prefixes[0]
        return ""


# ── Demo seed ────────────────────────────────────────────────────────────────
def seed_demo_fleet(mgr: FleetManager) -> None:
    """Register a small varied fleet so the UI has something interesting to show.

    Two units in Building A share a refrigerant_leak profile → cross-unit
    'refrigerant_loss' anomaly fires once their twins converge (~30s of ticks).
    """
    mgr.register_unit(
        "UNIT-001", location="Building A, Floor 3",
        fault_profile="refrigerant_leak",
        initial_health=ComponentHealth(refrigerant_charge_pct=78, compressor_efficiency_pct=98, fan_health_pct=96),
    )
    mgr.register_unit(
        "UNIT-002", location="Building A, Floor 5",
        fault_profile="refrigerant_leak",
        initial_health=ComponentHealth(refrigerant_charge_pct=82, compressor_efficiency_pct=99, fan_health_pct=97),
    )
    mgr.register_unit(
        "UNIT-003", location="Building B, Ground",
        fault_profile=None,
        initial_health=ComponentHealth(refrigerant_charge_pct=99, compressor_efficiency_pct=100, fan_health_pct=99),
    )
    mgr.register_unit(
        "UNIT-004", location="Building C, Floor 2",
        fault_profile="fan_failure",
        initial_health=ComponentHealth(refrigerant_charge_pct=96, compressor_efficiency_pct=92, fan_health_pct=58),
    )
