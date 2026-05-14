"""
Alert payload builder - attaches SHAP explanation, MC-Dropout uncertainty,
prescription, and energy cost to every alert.

Every alert sent to the backend includes:
  - machine_id, timestamp, severity_score, fault_type, action
  - uncertainty, confidence_pct, action_override  (MC-Dropout fields)
  - explanation: 4-sensor percentages + summary
  - prescription: fault, impact, action (dispatch instruction)
  - energy_cost: kWh waste, INR/USD cost per day/month, payback period

Usage:
    payload = build_alert_payload("CARRIER-CHILLER-01", 87, explanation)
"""

from datetime import datetime, timezone
from typing import Optional


# --- Energy profiles per fault type (commercial chiller scale) ---

FAULT_ENERGY_PROFILES = {
    "Refrigerant Leak": {
        "efficiency_loss_pct": 40,
        "extra_kwh_per_hr":    9.6,   # Rs.1,843/day
    },
    "Condenser Fan Failure": {
        "efficiency_loss_pct": 15,
        "extra_kwh_per_hr":    11.0,  # Rs.2,112/day
    },
    "Compressor Wear": {
        "efficiency_loss_pct": 20,
        "extra_kwh_per_hr":    16.0,  # Rs.3,072/day
    },
}

PART_COSTS_INR = {
    "Refrigerant Leak":      5000,   # recharge kit
    "Condenser Fan Failure": 8000,   # 5HP motor
    "Compressor Wear":       45000,  # compressor replacement
}

INR_PER_KWH = 8.0
USD_PER_KWH = 0.12


def _compute_energy_cost(fault_type: str, explanation: dict) -> dict:
    """Compute energy waste and financial impact for a given fault type."""
    profile = FAULT_ENERGY_PROFILES.get(fault_type)
    if not profile:
        return {}

    extra_kwh        = profile["extra_kwh_per_hr"]
    cost_per_day_inr = round(extra_kwh * 24 * INR_PER_KWH, 1)
    cost_per_day_usd = round(extra_kwh * 24 * USD_PER_KWH, 1)
    cost_per_month   = round(cost_per_day_inr * 30, 1)
    part_cost        = PART_COSTS_INR.get(fault_type, 0)
    payback_days     = round(part_cost / cost_per_day_inr, 1) if cost_per_day_inr > 0 else 0

    # SHAP-weighted attribution: which sensor is costing the most
    shap_keys = {
        "compressor_power_pct":   "Compressor Power",
        "discharge_pressure_pct": "Discharge Pressure",
        "fan_rpm_pct":            "Fan RPM",
        "supply_air_temp_pct":    "Supply Air Temp",
    }
    attribution = {
        label: round((explanation.get(key, 0) / 100.0) * cost_per_day_inr, 1)
        for key, label in shap_keys.items()
    }

    return {
        "efficiency_loss_pct":     profile["efficiency_loss_pct"],
        "energy_waste_kwh_per_hr": extra_kwh,
        "cost_per_day_inr":        cost_per_day_inr,
        "cost_per_day_usd":        cost_per_day_usd,
        "cost_per_month_inr":      cost_per_month,
        "part_cost_inr":           part_cost,
        "payback_days":            payback_days,
        "shap_cost_attribution":   attribution,
    }


def build_alert_payload(
    machine_id: str,
    severity_score: int,
    explanation: dict,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Build a standardized Thermo-Twin alert payload with SHAP explanation,
    MC-Dropout uncertainty, and energy cost attribution.

    Args:
        machine_id:     e.g. "CARRIER-CHILLER-01"
        severity_score: 0-100  (<= 40 normal, 41-70 warn, >= 71 stop unit)
        explanation:    dict from SHAPExplainer.explain() + MC-Dropout fields
        timestamp:      ISO 8601 UTC string (defaults to now)

    Returns:
        Alert dict ready for JSON serialization.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    fault_type = str(explanation.get("fault_type", "Unknown"))

    # MC-Dropout uncertainty fields (present after precompute, absent for live alerts)
    uncertainty     = explanation.get("uncertainty", None)
    confidence_pct  = explanation.get("confidence_pct", None)
    action_override = explanation.get("action_override", None)

    expl_block = {
        "compressor_power_pct":   float(explanation.get("compressor_power_pct",   25.0)),
        "discharge_pressure_pct": float(explanation.get("discharge_pressure_pct", 25.0)),
        "fan_rpm_pct":            float(explanation.get("fan_rpm_pct",            25.0)),
        "supply_air_temp_pct":    float(explanation.get("supply_air_temp_pct",    25.0)),
        "summary":                str(explanation.get("summary", "")),
    }

    # Use action_override from MC-Dropout if available (high uncertainty case)
    base_action = _action_label(severity_score)
    final_action = action_override if action_override else base_action

    payload = {
        "machine_id":     machine_id,
        "timestamp":      timestamp,
        "severity_score": int(severity_score),
        "fault_type":     fault_type,
        "action":         final_action,
        "explanation":    expl_block,
        "prescription":   explanation.get("prescription", {}),
        "energy_cost":    _compute_energy_cost(fault_type, expl_block),
    }

    if uncertainty is not None:
        payload["uncertainty"]    = int(uncertainty)
    if confidence_pct is not None:
        payload["confidence_pct"] = int(confidence_pct)
    if action_override is not None:
        payload["action_override"] = str(action_override)

    return payload


def load_demo_explanations(json_path) -> dict:
    """Load pre-computed demo explanations from JSON. Returns {} if not found."""
    import json
    from pathlib import Path

    path = Path(json_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _action_label(score: int) -> str:
    if score >= 71:
        return "STOP UNIT"
    if score >= 41:
        return "WARNING"
    return "NORMAL"