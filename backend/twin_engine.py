import sys
import logging
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from physics.hvac_physics import HVACPhysicsModel, OperatingConditions, SensorPrediction
from physics.degradation_model import DegradationModel, ComponentHealth
from physics.state_estimator import KalmanStateEstimator

log = logging.getLogger("thermo-twin.engine")


class TwinEngine:
    """
    Orchestrates the physics model + degradation model + Kalman filter.

    Runs alongside (not replacing) the existing autoencoder pipeline.
    Called from app.py after each sensor sample is produced.

    Per-sample pipeline:
        1. Infer OperatingConditions from raw sensor values
        2. HVACPhysicsModel.predict() → healthy baseline
        3. DegradationModel.apply(healthy, current_health) → expected degraded readings
        4. KalmanStateEstimator.predict() — advances uncertainty
        5. KalmanStateEstimator.update(real, degraded) — fuses observation
        6. Return state + divergence + prediction
    """

    _NOMINAL_POWER_KW = 3.5

    def __init__(self, use_coolprop: bool = True, nominal_ambient_c: float = 35.0):
        self._physics          = HVACPhysicsModel(use_coolprop=use_coolprop)
        self._degradation      = DegradationModel()
        self._kalman           = KalmanStateEstimator(degradation_model=self._degradation)
        self._nominal_ambient  = nominal_ambient_c
        log.info("TwinEngine ready  coolprop=%s", self._physics._coolprop_available)

    def process_sample(self, sample: dict, ambient_temp_c: Optional[float] = None) -> dict:
        """
        Process one raw sensor sample through the physics + Kalman pipeline.

        Args:
            sample:         dict with keys: compressor_power_kw, discharge_pressure_psi,
                            fan_rpm, supply_air_temp_c
            ambient_temp_c: optional ambient temperature override (default: 35°C nominal)

        Returns:
            {
                "state":      {refrigerant_charge_pct, compressor_efficiency_pct, fan_health_pct},
                "prediction": {4 sensor keys, model_used},
                "divergence": {4 sensor keys — real minus predicted},
                "model_used": "coolprop" | "linear"
            }
        """
        ambient = ambient_temp_c if ambient_temp_c is not None else self._nominal_ambient

        conditions      = self._conditions_from_sample(sample, ambient)
        healthy_pred    = self._physics.predict(conditions)
        current_health  = self._kalman.get_health()
        degraded_pred   = self._degradation.apply(healthy_pred, current_health)

        self._kalman.predict()

        real_pred = SensorPrediction(
            compressor_power_kw    = float(sample["compressor_power_kw"]),
            discharge_pressure_psi = float(sample["discharge_pressure_psi"]),
            fan_rpm                = float(sample["fan_rpm"]),
            supply_air_temp_c      = float(sample["supply_air_temp_c"]),
            model_used             = "real",
        )
        # Pass healthy_pred so UKF can propagate sigma points through h(x)
        kalman_result = self._kalman.update(real_pred, degraded_pred, healthy_pred)

        return {
            "state": {
                "refrigerant_charge_pct":    round(float(kalman_result.x[0]), 1),
                "compressor_efficiency_pct": round(float(kalman_result.x[1]), 1),
                "fan_health_pct":            round(float(kalman_result.x[2]), 1),
            },
            "prediction":     degraded_pred.to_dict(),
            "divergence":     kalman_result.divergence,
            "uncertainty":    kalman_result.uncertainty,
            "estimator_mode": kalman_result.mode,
            "model_used":     degraded_pred.model_used,
        }

    def reset(self) -> None:
        """Reset Kalman state to 100% health (call after part replacement or stream reset)."""
        self._kalman.reset()

    def get_state(self) -> dict:
        """Return current estimated health state without processing a new sample."""
        return self._kalman.get_health().to_dict()

    def _conditions_from_sample(self, sample: dict, ambient_temp_c: float) -> OperatingConditions:
        """
        Infer OperatingConditions from observed sensor values.

        Compressor speed and load demand are not directly measured, so we
        infer them from compressor_power relative to the 3.5 kW nominal baseline.
        """
        power_kw  = float(sample["compressor_power_kw"])
        speed_pct = min(100.0, max(10.0, (power_kw / self._NOMINAL_POWER_KW) * 70.0))
        load_pct  = min(100.0, max(0.0, 50.0 + (power_kw - self._NOMINAL_POWER_KW) * 15.0))
        return OperatingConditions(
            ambient_temp_c       = ambient_temp_c,
            load_demand_pct      = load_pct,
            compressor_speed_pct = speed_pct,
        )
