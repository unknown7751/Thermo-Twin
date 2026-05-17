import numpy as np
import logging
from dataclasses import dataclass, field
from physics.hvac_physics import SensorPrediction
from physics.degradation_model import ComponentHealth

log = logging.getLogger("thermo-twin.kalman")


@dataclass
class KalmanState:
    x: np.ndarray          # (3,) state mean: [refrig, comp_eff, fan]
    P: np.ndarray          # (3,3) state covariance
    K: np.ndarray          # (3,4) last Kalman gain
    divergence: dict       # per-sensor real-vs-predicted delta


class KalmanStateEstimator:
    """
    Kalman Filter estimating [refrigerant_charge_pct, compressor_efficiency_pct, fan_health_pct].

    State transition F = identity (health degrades slowly via DegradationModel.advance_time).

    Observation matrix H (4×3): linearized sensor sensitivity to 1% change in each health variable.
    Derived from DegradationModel coefficients at nominal operating point.

    Observation noise R diagonal uses exact synthetic-generator noise σ:
        σ_power=0.08, σ_pressure=4.0, σ_fan=30.0, σ_temp=0.3
    """

    _N_STATE = 3
    _N_OBS   = 4

    # H (4×3): d(sensor) / d(health_pct) at nominal, per 1% health change
    # Rows: [compressor_power, discharge_pressure, fan_rpm, supply_air_temp]
    # Cols: [refrigerant_charge, compressor_efficiency, fan_health]
    _H = np.array([
        # refrig    comp_eff    fan_health
        [  0.000,   -0.020,    -0.014 ],   # compressor_power_kw    (kW / pct)
        [ -0.265,   +0.450,    -1.100 ],   # discharge_pressure_psi (psi / pct)
        [  0.000,    0.000,    +2.720 ],   # fan_rpm                (rpm / pct)
        [  0.070,   -0.030,    -0.090 ],   # supply_air_temp_c      (°C / pct)
    ], dtype=np.float64)  # shape (4, 3)

    def __init__(
        self,
        initial_health: ComponentHealth | None = None,
        process_noise_std: float = 0.01,
        obs_noise_std: float = 5.0,
    ):
        x0 = np.array([
            initial_health.refrigerant_charge_pct    if initial_health else 100.0,
            initial_health.compressor_efficiency_pct if initial_health else 100.0,
            initial_health.fan_health_pct            if initial_health else 100.0,
        ], dtype=np.float64)

        self._x = x0.copy()
        self._P = np.eye(self._N_STATE, dtype=np.float64) * 10.0   # initial uncertainty ±10%
        self._F = np.eye(self._N_STATE, dtype=np.float64)           # identity state transition
        self._Q = np.eye(self._N_STATE, dtype=np.float64) * (process_noise_std ** 2)
        # Per-sensor measurement noise from synthetic generator noise levels
        self._R = np.diag([0.08**2, 4.0**2, 30.0**2, 0.3**2])
        self._last_K = np.zeros((self._N_STATE, self._N_OBS), dtype=np.float64)

    def predict(self) -> ComponentHealth:
        """Prediction step: advance state covariance forward one step."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._health_from_state()

    def update(self, real: SensorPrediction, predicted: SensorPrediction) -> KalmanState:
        """
        Update step: fuse sensor innovation with state estimate.

        Uses z_pred from the degradation model directly (not H@x) to avoid
        linearization bias when the model is more accurate than the linear H.
        """
        z = np.array([
            real.compressor_power_kw,
            real.discharge_pressure_psi,
            real.fan_rpm,
            real.supply_air_temp_c,
        ], dtype=np.float64)

        z_pred = np.array([
            predicted.compressor_power_kw,
            predicted.discharge_pressure_psi,
            predicted.fan_rpm,
            predicted.supply_air_temp_c,
        ], dtype=np.float64)

        H = self._H
        y = z - z_pred                              # innovation in sensor space

        S = H @ self._P @ H.T + self._R            # innovation covariance (4×4)
        K = self._P @ H.T @ np.linalg.inv(S)       # Kalman gain (3×4)
        self._last_K = K

        self._x = np.clip(self._x + K @ y, 0.0, 100.0)

        # Joseph form for numerical stability
        I = np.eye(self._N_STATE)
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.T + K @ self._R @ K.T

        divergence = {
            "compressor_power_kw":    round(float(y[0]), 4),
            "discharge_pressure_psi": round(float(y[1]), 2),
            "fan_rpm":                round(float(y[2]), 1),
            "supply_air_temp_c":      round(float(y[3]), 2),
        }

        return KalmanState(x=self._x.copy(), P=self._P.copy(), K=K, divergence=divergence)

    def get_health(self) -> ComponentHealth:
        return self._health_from_state()

    def reset(self, initial_health: ComponentHealth | None = None) -> None:
        x0 = np.array([
            initial_health.refrigerant_charge_pct    if initial_health else 100.0,
            initial_health.compressor_efficiency_pct if initial_health else 100.0,
            initial_health.fan_health_pct            if initial_health else 100.0,
        ], dtype=np.float64)
        self._x = x0
        self._P = np.eye(self._N_STATE, dtype=np.float64) * 10.0

    def _health_from_state(self) -> ComponentHealth:
        return ComponentHealth(
            refrigerant_charge_pct    = float(np.clip(self._x[0], 0, 100)),
            compressor_efficiency_pct = float(np.clip(self._x[1], 0, 100)),
            fan_health_pct            = float(np.clip(self._x[2], 0, 100)),
        )
