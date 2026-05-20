"""
Phase 4 — What-If Simulator

Given a set of operating conditions (compressor speed, ambient temp, load) and a
time horizon, project the machine forward in time and report:

    • predicted sensor trajectory (with a per-step anomaly score)
    • fault-risk probability + time-to-critical
    • energy cost vs the nominal baseline
    • a plain-English recommendation

Reuses the existing first-principles physics + degradation models so the
projection is consistent with the live digital twin (no separate model to drift).

Wear rates are derived from the operating conditions themselves: running the
compressor harder / hotter / under heavier load accelerates component decay.
"""

import logging
from dataclasses import dataclass

from physics.hvac_physics import HVACPhysicsModel, OperatingConditions
from physics.degradation_model import DegradationModel, ComponentHealth

log = logging.getLogger("thermo-twin.whatif")

# Energy tariff — identical to explainability/alert_payload.py
INR_PER_KWH = 8.0
USD_PER_KWH = 0.12

# Health thresholds below which a component is "critical" (match RULEngine)
CRITICAL_THRESHOLDS = {
    "refrigerant_charge_pct":    50.0,
    "compressor_efficiency_pct": 20.0,
    "fan_health_pct":            10.0,
}

# Baseline (nominal) operating point
BASELINE = {
    "compressor_speed_pct": 70.0,
    "ambient_temp_c":       35.0,
    "load_demand_pct":      50.0,
}


@dataclass
class WhatIfRequest:
    compressor_speed_pct: float = 70.0
    ambient_temp_c: float = 35.0
    load_demand_pct: float = 50.0
    simulation_duration_hours: float = 4.0
    # Optional starting health (defaults to perfect)
    refrigerant_charge_pct: float = 100.0
    compressor_efficiency_pct: float = 100.0
    fan_health_pct: float = 100.0


class WhatIfSimulator:
    """Projects a scenario forward using physics + degradation models."""

    _STEP_MINUTES = 15          # finest trajectory resolution (short runs)
    _MAX_POINTS   = 300         # cap trajectory length; long runs use coarser steps
    _ANOMALY_WARN = 0.15        # score thresholds (match frontend visualMap)
    _ANOMALY_CRIT = 0.30

    def __init__(self, use_coolprop: bool = False):
        self._physics     = HVACPhysicsModel(use_coolprop=use_coolprop)
        self._degradation = DegradationModel()

    # Wear floor at the design point — equipment is rated for nominal, so it
    # ages negligibly there (~years to fail). Component lifetimes at this rate:
    #   fan  ≈ 90 / 0.002  ≈ 45,000 h ≈ 5 years   → baseline is visually flat.
    _WEAR_FLOOR = 0.002  # %/hour

    # Nominal design point (no overstress below these values)
    _NOM_SPEED = 70.0
    _NOM_AMB   = 35.0
    _NOM_LOAD  = 50.0

    # ── Wear-rate model ────────────────────────────────────────────────────────
    def _wear_rates(self, c: WhatIfRequest) -> dict:
        """
        Component wear (pct/hour) driven by OVERSTRESS above the nominal design
        point — not absolute operating level. Equipment rated for 70/35/50
        barely ages there; wear accrues only as conditions exceed design.

          • Healthy baseline (70/35/50) → ~0.002 %/hr → effectively flat / years
          • Extreme max (100/50/100)    → ~7–10 %/hr  → critical in ~7–10 h (demo)

        Each overstress term is normalized to 0 at nominal and 1 at the slider
        maximum, then raised to a power for realistic non-linear acceleration.
        """
        f = self._WEAR_FLOOR

        # Normalized overstress (0 at design point, 1 at slider max), clamped ≥0
        speed_over = max(0.0, (c.compressor_speed_pct - self._NOM_SPEED) / (100.0 - self._NOM_SPEED))
        amb_over   = max(0.0, (c.ambient_temp_c       - self._NOM_AMB)   / (50.0  - self._NOM_AMB))
        load_over  = max(0.0, (c.load_demand_pct      - self._NOM_LOAD)  / (100.0 - self._NOM_LOAD))

        # Refrigerant leak: driven by head pressure (hot ambient) + heavy load
        leak_rate = f + 6.0 * (amb_over ** 2) + 1.5 * load_over

        # Compressor wear: mechanical fatigue ∝ speed-overstress², amplified by
        # load, plus thermal stress from high ambient
        comp_wear_rate = f + 7.0 * (speed_over ** 2) * (0.5 + 0.5 * load_over) + 3.0 * (amb_over ** 2)

        # Fan bearing wear: speed-overstress² + thermal stress
        fan_wear_rate = f + 6.0 * (speed_over ** 2) + 3.0 * (amb_over ** 2)

        return {
            "leak_rate":      round(max(0.0, leak_rate), 4),
            "comp_wear_rate": round(max(0.0, comp_wear_rate), 4),
            "fan_wear_rate":  round(max(0.0, fan_wear_rate), 4),
        }

    # ── Anomaly score ──────────────────────────────────────────────────────────
    def _anomaly_score(self, healthy, degraded) -> float:
        """Normalized deviation of degraded sensors from the healthy baseline (0–1)."""
        def rel(a, b, scale):
            return abs(a - b) / scale

        dev = (
            rel(degraded.compressor_power_kw,    healthy.compressor_power_kw,    3.5) * 0.35 +
            rel(degraded.discharge_pressure_psi, healthy.discharge_pressure_psi, 245.0) * 0.30 +
            rel(degraded.fan_rpm,                healthy.fan_rpm,                1190.0) * 0.20 +
            rel(degraded.supply_air_temp_c,      healthy.supply_air_temp_c,      11.0) * 0.15
        )
        return round(min(1.0, dev), 4)

    # ── Single scenario projection ─────────────────────────────────────────────
    def _project(self, req: WhatIfRequest) -> dict:
        conditions = OperatingConditions(
            ambient_temp_c       = req.ambient_temp_c,
            load_demand_pct      = req.load_demand_pct,
            compressor_speed_pct = req.compressor_speed_pct,
        )
        healthy = self._physics.predict(conditions)
        rates   = self._wear_rates(req)

        health = ComponentHealth(
            refrigerant_charge_pct    = req.refrigerant_charge_pct,
            compressor_efficiency_pct = req.compressor_efficiency_pct,
            fan_health_pct            = req.fan_health_pct,
        )

        duration_min = max(30.0, req.simulation_duration_hours * 60.0)
        # Adaptive step: 15-min resolution for short runs, but never more than
        # _MAX_POINTS samples so a 250,000-hour horizon stays fast & renderable.
        step_min = max(self._STEP_MINUTES, duration_min / self._MAX_POINTS)
        n_steps  = int(duration_min / step_min) + 1
        dt_hours = step_min / 60.0

        trajectory          = []
        time_to_critical_h  = None
        peak_score          = 0.0
        total_energy_kwh    = 0.0

        for step in range(n_steps):
            t_min   = step * step_min
            degraded = self._degradation.apply(healthy, health)
            score    = self._anomaly_score(healthy, degraded)
            peak_score = max(peak_score, score)

            trajectory.append({
                "t_minutes":              t_min,
                "t_hours":                round(t_min / 60.0, 2),
                "compressor_power_kw":    degraded.compressor_power_kw,
                "discharge_pressure_psi": degraded.discharge_pressure_psi,
                "fan_rpm":                degraded.fan_rpm,
                "supply_air_temp_c":      degraded.supply_air_temp_c,
                "anomaly_score":          score,
                "refrigerant_charge_pct":    round(health.refrigerant_charge_pct, 1),
                "compressor_efficiency_pct": round(health.compressor_efficiency_pct, 1),
                "fan_health_pct":            round(health.fan_health_pct, 1),
            })

            # Energy consumed this step (degraded power × hours)
            total_energy_kwh += degraded.compressor_power_kw * dt_hours

            # First time any component crosses its critical threshold
            if time_to_critical_h is None and (
                health.refrigerant_charge_pct    <= CRITICAL_THRESHOLDS["refrigerant_charge_pct"] or
                health.compressor_efficiency_pct <= CRITICAL_THRESHOLDS["compressor_efficiency_pct"] or
                health.fan_health_pct            <= CRITICAL_THRESHOLDS["fan_health_pct"]
            ):
                time_to_critical_h = round(t_min / 60.0, 1)

            # Advance health for next step
            health = self._degradation.advance_time(
                health, dt_hours,
                leak_rate      = rates["leak_rate"],
                fan_wear_rate  = rates["fan_wear_rate"],
                comp_wear_rate = rates["comp_wear_rate"],
            )

        return {
            "trajectory":         trajectory,
            "peak_anomaly_score": round(peak_score, 4),
            "time_to_critical_h": time_to_critical_h,
            "total_energy_kwh":   round(total_energy_kwh, 3),
            "wear_rates":         rates,
            "final_health":       health.to_dict(),
        }

    # ── Public API ─────────────────────────────────────────────────────────────
    def simulate(self, req: WhatIfRequest, include_baseline: bool = True) -> dict:
        scenario = self._project(req)

        # Baseline = a HEALTHY machine (100/100/100) run at the nominal
        # operating point for the same horizon. This is a fixed reference, so
        # any degraded/stressed scenario always shows a meaningful cost delta
        # (a unit pulling 190% power will correctly read as far costlier than
        # a brand-new unit run normally — not "₹0" against itself).
        baseline_req = WhatIfRequest(
            compressor_speed_pct      = BASELINE["compressor_speed_pct"],
            ambient_temp_c            = BASELINE["ambient_temp_c"],
            load_demand_pct           = BASELINE["load_demand_pct"],
            simulation_duration_hours = req.simulation_duration_hours,
            # health left at defaults (100/100/100) = healthy reference
        )
        baseline = self._project(baseline_req) if include_baseline else None

        # Energy cost
        scen_kwh = scenario["total_energy_kwh"]
        base_kwh = baseline["total_energy_kwh"] if baseline else scen_kwh
        scen_inr = round(scen_kwh * INR_PER_KWH, 1)
        base_inr = round(base_kwh * INR_PER_KWH, 1)
        save_inr = round(base_inr - scen_inr, 1)
        save_pct = round((save_inr / base_inr) * 100.0, 1) if base_inr > 0 else 0.0

        # Fault risk: map peak anomaly score → probability %
        peak = scenario["peak_anomaly_score"]
        fault_risk_pct = round(min(100.0, peak / self._ANOMALY_CRIT * 50.0), 1)

        ttc = scenario["time_to_critical_h"]
        if peak >= self._ANOMALY_CRIT or (ttc is not None and ttc <= req.simulation_duration_hours):
            status = "critical"
        elif peak >= self._ANOMALY_WARN:
            status = "warning"
        else:
            status = "safe"

        recommendation = self._recommend(req, status, fault_risk_pct, save_pct, ttc)

        return {
            "scenario": {
                "inputs": {
                    "compressor_speed_pct":      req.compressor_speed_pct,
                    "ambient_temp_c":            req.ambient_temp_c,
                    "load_demand_pct":           req.load_demand_pct,
                    "simulation_duration_hours": req.simulation_duration_hours,
                },
                "trajectory": scenario["trajectory"],
            },
            "baseline": (
                {"trajectory": baseline["trajectory"]} if baseline else None
            ),
            "summary": {
                "status":                      status,
                "fault_risk_probability_pct":  fault_risk_pct,
                "time_to_critical_hours":      ttc if ttc is not None else -1,
                "peak_anomaly_score":          peak,
                "wear_rates":                  scenario["wear_rates"],
                "energy_cost": {
                    "scenario_cost_inr":  scen_inr,
                    "baseline_cost_inr":  base_inr,
                    "savings_inr":        save_inr,
                    "savings_pct":        save_pct,
                    "scenario_cost_usd":  round(scen_kwh * USD_PER_KWH, 1),
                    "baseline_cost_usd":  round(base_kwh * USD_PER_KWH, 1),
                    "savings_usd":        round((base_kwh - scen_kwh) * USD_PER_KWH, 1),
                },
                "recommendation": recommendation,
            },
        }

    def _recommend(self, req, status, risk_pct, save_pct, ttc) -> str:
        speed = req.compressor_speed_pct
        if status == "critical":
            if ttc is not None and ttc > 0:
                return (
                    f"High risk: a component reaches critical condition in ~{ttc} h "
                    f"under these settings. Reduce compressor speed and/or lower ambient "
                    f"load before extended operation."
                )
            return (
                "Unsafe scenario: predicted anomaly risk is high. Do not deploy "
                "these conditions without maintenance first."
            )
        if status == "warning":
            return (
                f"Moderate risk ({risk_pct:.0f}%). Acceptable for short runs, but monitor "
                f"closely. Trimming compressor speed below {speed:.0f}% improves the "
                f"safety margin."
            )
        # safe
        if save_pct > 1:
            return (
                f"Safe to operate. Running at {speed:.0f}% compressor saves {save_pct:.0f}% "
                f"energy vs the nominal baseline with minimal fault risk "
                f"({risk_pct:.0f}%). Suitable for extended operation."
            )
        if save_pct < -1:
            return (
                f"Safe but costlier: these settings use {abs(save_pct):.0f}% more energy "
                f"than the nominal baseline. Fault risk is low ({risk_pct:.0f}%) — fine "
                f"for short bursts, but not energy-optimal for continuous running."
            )
        return (
            f"Safe to operate with low fault risk ({risk_pct:.0f}%). Energy use is "
            f"comparable to the nominal baseline."
        )
