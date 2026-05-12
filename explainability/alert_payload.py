"""
Alert payload builder — attaches SHAP explanation and prescription to every alert.

Every alert sent to the backend includes:
  - machine_id, timestamp, severity_score, fault_type, action
  - explanation: 4-sensor percentages + summary
  - prescription: fault, impact, action (dispatch instruction)

Usage:
    payload = build_alert_payload("CARRIER-CHILLER-01", 87, explanation)
"""

from datetime import datetime, timezone
from typing import Optional


def build_alert_payload(
    machine_id: str,
    severity_score: int,
    explanation: dict,
    timestamp: Optional[str] = None,
) -> dict:
    """
    Build a standardized Thermo-Twin alert payload with SHAP explanation.

    Args:
        machine_id:     e.g. "CARRIER-CHILLER-01"
        severity_score: 0–100  (<= 40 normal, 41–70 warn, >= 71 stop unit)
        explanation:    dict from SHAPExplainer.explain() — contains
                        compressor_power_pct, discharge_pressure_pct,
                        fan_rpm_pct, supply_air_temp_pct, summary,
                        fault_type, prescription
        timestamp:      ISO 8601 UTC string (defaults to now)

    Returns:
        Alert dict ready for JSON serialization.

    Example output:
        {
            "machine_id": "CARRIER-CHILLER-01",
            "timestamp": "2024-01-15T14:32:07Z",
            "severity_score": 87,
            "fault_type": "Refrigerant Leak",
            "action": "STOP UNIT",
            "explanation": {
                "compressor_power_pct": 8.0,
                "discharge_pressure_pct": 51.0,
                "fan_rpm_pct": 6.0,
                "supply_air_temp_pct": 35.0,
                "summary": "Anomaly driven by Pressure Drop (51%) and Temp Rise (35%)"
            },
            "prescription": {
                "fault": "Refrigerant Leak in Evaporator Coil",
                "impact": "Cooling efficiency down ~40%",
                "action": "Dispatch with refrigerant recharge kit + leak detector"
            }
        }
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "machine_id":     machine_id,
        "timestamp":      timestamp,
        "severity_score": int(severity_score),
        "fault_type":     str(explanation.get("fault_type", "Unknown")),
        "action":         _action_label(severity_score),
        "explanation": {
            "compressor_power_pct":   float(explanation.get("compressor_power_pct",   25.0)),
            "discharge_pressure_pct": float(explanation.get("discharge_pressure_pct", 25.0)),
            "fan_rpm_pct":            float(explanation.get("fan_rpm_pct",            25.0)),
            "supply_air_temp_pct":    float(explanation.get("supply_air_temp_pct",    25.0)),
            "summary":                str(explanation.get("summary", "")),
        },
        "prescription": explanation.get("prescription", {}),
    }


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
