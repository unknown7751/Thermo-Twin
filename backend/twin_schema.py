from dataclasses import dataclass, field, asdict
from typing import Optional

SENSOR_FIELDS = ("compressor_power_kw", "discharge_pressure_psi", "fan_rpm", "supply_air_temp_c")


@dataclass
class TwinSample:
    timestamp: float
    sample_index: int
    machine_id: str
    compressor_power_kw: float
    discharge_pressure_psi: float
    fan_rpm: float
    supply_air_temp_c: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_streamer_dict(cls, d: dict, machine_id: str = "LIVE-DEMO-UNIT") -> "TwinSample":
        return cls(
            timestamp=d["timestamp"],
            sample_index=d["sample_index"],
            machine_id=machine_id,
            compressor_power_kw=d["compressor_power_kw"],
            discharge_pressure_psi=d["discharge_pressure_psi"],
            fan_rpm=d["fan_rpm"],
            supply_air_temp_c=d["supply_air_temp_c"],
        )


@dataclass
class TwinAlert:
    machine_id: str
    timestamp: str
    severity_score: int
    fault_type: str
    action: str
    reconstruction_error: Optional[float] = None
    raw_payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_alert_dict(cls, d: dict) -> "TwinAlert":
        return cls(
            machine_id=d.get("machine_id", "UNKNOWN"),
            timestamp=d.get("timestamp", ""),
            severity_score=int(d.get("severity_score", 0)),
            fault_type=d.get("fault_type", "Unknown"),
            action=d.get("action", "NORMAL"),
            reconstruction_error=d.get("reconstruction_error"),
            raw_payload=d,
        )


@dataclass
class TwinState:
    machine_id: str
    timestamp: float
    refrigerant_charge_pct: float
    compressor_efficiency_pct: float
    fan_health_pct: float
    divergence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
