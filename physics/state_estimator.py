"""
Unscented Kalman Filter (UKF) state estimator for the Thermo-Twin digital twin.

STATE VECTOR (n=3):
    x = [refrigerant_charge_pct, compressor_efficiency_pct, fan_health_pct]
    All 0–100, where 100 = new/perfect condition.

OBSERVATION VECTOR (m=4):
    z = [compressor_power_kw, discharge_pressure_psi, fan_rpm, supply_air_temp_c]

UKF SIGMA POINTS — Merwe scaled, 2n+1 = 7 points:
    Parameters: α=1.0, β=2.0, κ=0.0  →  λ = α²(n+κ) - n = 0
    Wm = [0,   1/6, 1/6, 1/6, 1/6, 1/6, 1/6]   (mean weights, sum=1)
    Wc = [2.0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6]   (covariance weights; center term
                                                   accounts for Gaussian kurtosis)
    Sigma points: χ_0 = x̂,
                  χ_i = x̂ + col_i(chol(√n · P))   for i = 1..3
                  χ_i = x̂ - col_{i-3}(chol(√n · P)) for i = 4..6

OBSERVATION FUNCTION h(χ):
    h(χ) = DegradationModel.apply(healthy_prediction, ComponentHealth(χ))
    This is the nonlinear map from health state → expected sensor readings.
    The UKF propagates all 7 sigma points through h to avoid linearization error.

PREDICTION STEP (state transition f(x) = x, identity):
    x_pred = x          (mean unchanged — health does not self-heal)
    P_pred = P + Q      (uncertainty grows each step by process noise)

UPDATE STEP (nonlinear via sigma point propagation):
    γ_i   = h(χ_i)                                  propagate through h
    z_pred = Σ Wm_i · γ_i                            predicted observation
    Pzz   = Σ Wc_i · (γ_i - z_pred)(γ_i - z_pred)ᵀ + R   innovation covariance
    Pxz   = Σ Wc_i · (χ_i - x)(γ_i - z_pred)ᵀ            cross-covariance (3×4)
    K     = Pxz · Pzz⁻¹                              Kalman gain (3×4)
    x     = clip(x + K · (z_real - z_pred), 0, 100)
    P     = P - K · Pzz · Kᵀ      (symmetrized + PD-enforced)

LINEAR FALLBACK (when no DegradationModel provided):
    Uses H matrix (4×3) with Joseph-form covariance update.

H MATRIX DERIVATION at nominal (power=3.5 kW, pressure=245 psi, fan=1190 rpm, temp=11°C):
    Each entry = d(sensor) / d(health_pct) from DegradationModel partial derivatives.

    Refrigerant charge (charge_loss = (100-pct)/100):
      Δpressure = -0.38 · charge_loss · healthy_P   → d/dpct = +0.38·245/100 = +0.931 psi/pct
      Δtemp     = +7.0  · charge_loss               → d/dpct = -7.0/100       = -0.070 °C/pct
      power, fan: no direct effect                  → 0

    Compressor efficiency (comp_loss = (100-pct)/100):
      Δpower    = +2.0  · comp_loss                 → d/dpct = -2.0/100        = -0.020 kW/pct
      Δpressure = -45.0 · comp_loss                 → d/dpct = +45.0/100       = +0.450 psi/pct
      Δtemp     = +3.0  · comp_loss                 → d/dpct = -3.0/100        = -0.030 °C/pct
      fan: no direct effect                         → 0

    Fan health (fan_loss = (100-pct)/100, cascade factor 0.7):
      Δfan_rpm  = -0.80 · fan_loss · healthy_rpm   → d/dpct = +0.80·1190/100  = +9.520 rpm/pct
      Δpower    = +2.0·0.7 · fan_loss               → d/dpct = -2.0·0.7/100    = -0.014 kW/pct
      Δpressure = +55.0·0.7 · fan_loss              → d/dpct = -55.0·0.7/100   = -0.385 psi/pct
      Δtemp     = +4.5·0.7 · fan_loss               → d/dpct = -4.5·0.7/100    = -0.032 °C/pct

EXAMPLE — State evolution under refrigerant leak:

    Setup: healthy baseline (3.5 kW, 245 psi, 1190 rpm, 11°C).
    Fault: 38% refrigerant loss → real sensors show (3.5 kW, 210 psi, 1190 rpm, 13.7°C).

    Step 0  (start):  state=[100.0, 100.0, 100.0]  P_diag=[10.0, 10.0, 10.0]
    Step 1  (first update, y_P≈-35 psi, y_T≈+2.7°C):
              state≈[77.0, 92.0, 100.0]  (refrig drops most, comp_eff cross-contaminates)
    Step 10:  state≈[68.0, 96.0, 100.0]  (refrig converging, comp_eff recovering)
    Step 50:  state≈[63.0, 99.0, 100.0]  P_diag≈[0.3, 0.1, 0.05]
              → refrigerant_charge_pct correctly identified as the fault dimension.
              → compressor and fan remain near 100 (power + fan_rpm unchanged).

    The key discriminator: power and fan_rpm do NOT deviate in a refrigerant leak,
    so the Kalman gain correctly attributes the pressure+temp signature to refrigerant.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

from physics.hvac_physics import SensorPrediction
from physics.degradation_model import ComponentHealth

log = logging.getLogger("thermo-twin.ukf")


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class KalmanState:
    x: np.ndarray     # (3,) state mean [refrig_pct, comp_eff_pct, fan_pct]
    P: np.ndarray     # (3,3) state covariance
    K: np.ndarray     # (3,4) Kalman gain from last update
    divergence: dict  # per-sensor real-vs-predicted delta (4 sensor keys)
    uncertainty: dict # per-state 1-sigma bounds: sqrt(diag(P))
    mode: str         # "ukf" or "linear"


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class KalmanStateEstimator:
    """
    UKF state estimator for HVAC component health.

    When constructed with a DegradationModel reference, runs full UKF with
    sigma point propagation through the nonlinear observation function h(x).
    Without it, falls back to a linear KF using the analytically derived H matrix.

    Usage (full UKF path):
        from physics.degradation_model import DegradationModel
        est = KalmanStateEstimator(degradation_model=DegradationModel())
        est.predict()
        result = est.update(real_sensors, degraded_prediction, healthy_prediction)

    Usage (linear fallback):
        est = KalmanStateEstimator()
        est.predict()
        result = est.update(real_sensors, degraded_prediction)
    """

    _N_STATE = 3
    _N_OBS   = 4

    # H matrix (4×3): d(sensor)/d(health_pct) at nominal operating point.
    # See module docstring for full derivation from DegradationModel coefficients.
    # Columns: [refrigerant_charge, compressor_efficiency, fan_health]
    # Rows:    [compressor_power, discharge_pressure, fan_rpm, supply_air_temp]
    _H = np.array([
        #  refrig    comp_eff   fan_health
        [  0.000,   -0.020,    -0.014 ],   # compressor_power_kw    (kW / pct)
        [ +0.931,   +0.450,    -0.385 ],   # discharge_pressure_psi (psi / pct)
        [  0.000,    0.000,    +9.520 ],   # fan_rpm                (rpm / pct)
        [ -0.070,   -0.030,    -0.032 ],   # supply_air_temp_c      (°C  / pct)
    ], dtype=np.float64)

    # Sensor measurement noise (σ) from synthetic data generator noise levels:
    #   σ_power=0.08 kW, σ_pressure=4.0 psi, σ_fan=30.0 rpm, σ_temp=0.3°C
    _R_DIAG = np.array([0.08**2, 4.0**2, 30.0**2, 0.3**2])

    def __init__(
        self,
        initial_health: Optional[ComponentHealth] = None,
        process_noise: float = 0.01,
        obs_noise: float = 5.0,
        degradation_model=None,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Args:
            initial_health:    Starting ComponentHealth (defaults to 100% on all components).
            process_noise:     Std dev of health drift per step (Q = process_noise² · I).
                               Larger → filter trusts sensor innovations more, adapts faster.
            obs_noise:         Unused scalar (kept for API compat); noise is set per-sensor via _R_DIAG.
            degradation_model: DegradationModel instance. When provided, enables UKF mode.
                               Must be the same instance used by TwinEngine.
            alpha:             UKF sigma point spread (1.0 = spread = √n per axis).
            beta:              UKF distribution knowledge (2.0 = optimal for Gaussian).
            kappa:             UKF secondary scaling (0.0 = standard).
        """
        x0 = np.array([
            initial_health.refrigerant_charge_pct    if initial_health else 100.0,
            initial_health.compressor_efficiency_pct if initial_health else 100.0,
            initial_health.fan_health_pct            if initial_health else 100.0,
        ], dtype=np.float64)

        self._x = x0.copy()
        self._P = np.eye(self._N_STATE, dtype=np.float64) * 10.0  # ±√10 ≈ 3.16% initial uncertainty
        self._Q = np.eye(self._N_STATE, dtype=np.float64) * (process_noise ** 2)
        self._R = np.diag(self._R_DIAG)

        self._degradation_model = degradation_model

        # --- UKF weight computation (Merwe scaled sigma points) ---
        n   = self._N_STATE
        lam = alpha**2 * (n + kappa) - n   # = 0 for α=1, κ=0

        if abs(lam) < 1e-10:
            # Clean special case (α=1, κ=0): Wm_0=0, Wm_i=1/(2n), Wc_0=β, Wc_i=1/(2n)
            self._Wm = np.full(2*n + 1, 1.0 / (2*n))
            self._Wm[0] = 0.0
            self._Wc = np.full(2*n + 1, 1.0 / (2*n))
            self._Wc[0] = float(beta)           # = 2.0 → kurtosis correction for Gaussian
            self._sqrt_n_plus_lam = float(n)    # = 3; Cholesky scales by √3
        else:
            self._Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
            self._Wm[0] = lam / (n + lam)
            self._Wc = self._Wm.copy()
            self._Wc[0] += (1.0 - alpha**2 + beta)
            self._sqrt_n_plus_lam = float(n + lam)

        mode = "ukf" if degradation_model is not None else "linear"
        log.info("KalmanStateEstimator ready  mode=%s  alpha=%.1f  beta=%.1f  kappa=%.1f",
                 mode, alpha, beta, kappa)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self) -> ComponentHealth:
        """
        Prediction step: advance state covariance by process noise Q.

        Since f(x) = x (identity state transition — health does not self-heal),
        the state mean is unchanged. Only uncertainty grows by Q each step.

        Returns the current ComponentHealth estimate (pre-observation).
        """
        self._P = self._P + self._Q
        return self._health_from_state()

    def update(
        self,
        real: SensorPrediction,
        predicted: SensorPrediction,
        healthy: Optional[SensorPrediction] = None,
    ) -> KalmanState:
        """
        Update step: fuse real sensor readings with physics-model prediction.

        UKF path (when degradation_model + healthy are available):
            Generates 7 sigma points, propagates each through
            h(χ) = DegradationModel.apply(healthy, ComponentHealth(χ)),
            computes cross-covariance Pxz and innovation covariance Pzz from
            the sigma-propagated observations, then applies the Kalman gain.

        Linear fallback:
            Uses the analytically-derived H matrix (4×3) with Joseph-form
            covariance update for numerical stability.

        NaN/inf sensor readings are silently replaced by the predicted value
        before computing the innovation, so a dropped sensor doesn't crash.

        Args:
            real:      Actual sensor readings from the live streamer.
            predicted: Physics+degradation model prediction at current health estimate.
            healthy:   Physics model prediction at 100% health (needed for UKF sigma propagation).

        Returns:
            KalmanState with updated x, P, K, divergence, uncertainty, mode.
        """
        z      = self._to_z(real)
        z_pred_fallback = self._to_z(predicted)

        # Sanitize: replace any NaN / inf with predicted value
        bad = ~np.isfinite(z)
        if bad.any():
            log.warning("Non-finite sensor readings at channels %s — substituting predicted",
                        np.where(bad)[0].tolist())
            z[bad] = z_pred_fallback[bad]

        if self._degradation_model is not None and healthy is not None:
            return self._update_ukf(z, healthy)
        return self._update_linear(z, z_pred_fallback)

    def get_health(self) -> ComponentHealth:
        """Return current estimated ComponentHealth from state mean."""
        return self._health_from_state()

    def reset(self, initial_health: Optional[ComponentHealth] = None) -> None:
        """
        Reset state to specified health (default: 100%) with maximum uncertainty.
        Call after a part replacement or stream reset so the filter re-converges.
        """
        self._x = np.array([
            initial_health.refrigerant_charge_pct    if initial_health else 100.0,
            initial_health.compressor_efficiency_pct if initial_health else 100.0,
            initial_health.fan_health_pct            if initial_health else 100.0,
        ], dtype=np.float64)
        self._P = np.eye(self._N_STATE, dtype=np.float64) * 10.0

    # -----------------------------------------------------------------------
    # UKF internals
    # -----------------------------------------------------------------------

    def _sigma_points(self) -> np.ndarray:
        """
        Generate 2n+1 = 7 sigma points from (x, P).

        Uses Cholesky decomposition of (sqrt_factor · P) so that:
            χ_0     = x̂
            χ_i     = x̂ + L[:,i-1]       i = 1..n
            χ_{i+n} = x̂ - L[:,i-1]       i = 1..n
        where L = chol(sqrt_factor · P) is lower-triangular.

        Returns (7, 3) array of sigma points.
        """
        n    = self._N_STATE
        P_s  = (self._P + self._P.T) * 0.5          # symmetrize before Cholesky

        try:
            L = np.linalg.cholesky(self._sqrt_n_plus_lam * P_s)
        except np.linalg.LinAlgError:
            # Recover positive definiteness by adding minimal jitter
            min_eig = np.min(np.linalg.eigvalsh(P_s))
            P_s    += (1e-6 - min(min_eig, 0)) * np.eye(n)
            L       = np.linalg.cholesky(self._sqrt_n_plus_lam * P_s)

        sigma      = np.empty((2*n + 1, n), dtype=np.float64)
        sigma[0]   = self._x
        for i in range(n):
            sigma[i + 1]     = self._x + L[:, i]
            sigma[i + 1 + n] = self._x - L[:, i]
        return sigma

    def _h(self, chi: np.ndarray, healthy: SensorPrediction) -> np.ndarray:
        """
        Nonlinear observation function: health state → expected sensor readings.

        h(χ) = DegradationModel.apply(healthy, ComponentHealth(χ))

        The sigma point χ is clipped to [0, 100] to prevent the DegradationModel
        from receiving physically impossible inputs at the edge of the sigma cloud.

        Returns (4,) array: [power_kw, pressure_psi, fan_rpm, temp_c].
        """
        health = ComponentHealth(
            refrigerant_charge_pct    = float(np.clip(chi[0], 0.0, 100.0)),
            compressor_efficiency_pct = float(np.clip(chi[1], 0.0, 100.0)),
            fan_health_pct            = float(np.clip(chi[2], 0.0, 100.0)),
        )
        p = self._degradation_model.apply(healthy, health)
        return np.array([p.compressor_power_kw, p.discharge_pressure_psi,
                         p.fan_rpm, p.supply_air_temp_c], dtype=np.float64)

    def _update_ukf(self, z: np.ndarray, healthy: SensorPrediction) -> KalmanState:
        """
        Full UKF update via sigma point propagation.

        Steps:
            1. Generate 7 sigma points χ_i from (x, P)
            2. Propagate: γ_i = h(χ_i) for each sigma point
            3. z_pred = Σ Wm_i · γ_i
            4. Pzz    = Σ Wc_i · (γ_i - z_pred)(γ_i - z_pred)ᵀ + R
            5. Pxz    = Σ Wc_i · (χ_i - x)(γ_i - z_pred)ᵀ
            6. K      = Pxz · Pzz⁻¹
            7. x      = clip(x + K · y, 0, 100)
            8. P      = P - K · Pzz · Kᵀ   (symmetrized, PD-enforced)
        """
        sigma = self._sigma_points()                                  # (7, 3)
        gamma = np.array([self._h(sigma[i], healthy)
                          for i in range(len(sigma))])                # (7, 4)

        # Predicted observation (weighted mean of propagated sigma observations)
        z_pred = self._Wm @ gamma                                     # (4,)

        # Deviations in observation and state space
        dz = gamma - z_pred                                           # (7, 4)
        dx = sigma  - self._x                                         # (7, 3)

        # Innovation covariance Pzz (4×4) and cross-covariance Pxz (3×4)
        Pzz = np.einsum("i,ij,ik->jk", self._Wc, dz, dz) + self._R  # (4,4)
        Pxz = np.einsum("i,ij,ik->jk", self._Wc, dx, dz)            # (3,4)

        K   = Pxz @ np.linalg.inv(Pzz)                               # (3,4)
        y   = z - z_pred                                              # innovation (4,)

        self._x = np.clip(self._x + K @ y, 0.0, 100.0)
        P_new   = self._P - K @ Pzz @ K.T
        self._P = self._make_psd(P_new)

        return self._build_result(K, y, mode="ukf")

    def _update_linear(self, z: np.ndarray, z_pred: np.ndarray) -> KalmanState:
        """
        Linear KF update using the analytically-derived H matrix (4×3).

        Uses Joseph form P = (I-KH)P(I-KH)ᵀ + KRKᵀ for numerical stability.
        This guarantees P stays symmetric positive-definite even when K is
        not perfectly optimal (e.g., during convergence with large initial P).
        """
        H  = self._H
        y  = z - z_pred                                    # innovation (4,)
        S  = H @ self._P @ H.T + self._R                  # innovation covariance (4,4)
        K  = self._P @ H.T @ np.linalg.inv(S)             # Kalman gain (3,4)

        self._x = np.clip(self._x + K @ y, 0.0, 100.0)

        # Joseph form
        IKH     = np.eye(self._N_STATE) - K @ H
        P_new   = IKH @ self._P @ IKH.T + K @ self._R @ K.T
        self._P = self._make_psd(P_new)

        return self._build_result(K, y, mode="linear")

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _make_psd(self, P: np.ndarray) -> np.ndarray:
        """Symmetrize P and enforce positive definiteness via eigenvalue flooring."""
        P = (P + P.T) * 0.5
        min_eig = np.min(np.linalg.eigvalsh(P))
        if min_eig < 1e-8:
            P += (1e-8 - min_eig) * np.eye(self._N_STATE)
        return P

    def _to_z(self, pred: SensorPrediction) -> np.ndarray:
        return np.array([pred.compressor_power_kw, pred.discharge_pressure_psi,
                         pred.fan_rpm, pred.supply_air_temp_c], dtype=np.float64)

    def _build_result(self, K: np.ndarray, y: np.ndarray, mode: str) -> KalmanState:
        return KalmanState(
            x=self._x.copy(),
            P=self._P.copy(),
            K=K,
            divergence={
                "compressor_power_kw":    round(float(y[0]), 4),
                "discharge_pressure_psi": round(float(y[1]), 2),
                "fan_rpm":                round(float(y[2]), 1),
                "supply_air_temp_c":      round(float(y[3]), 2),
            },
            uncertainty={
                "refrigerant_charge_pct":    round(float(np.sqrt(max(0.0, self._P[0, 0]))), 3),
                "compressor_efficiency_pct": round(float(np.sqrt(max(0.0, self._P[1, 1]))), 3),
                "fan_health_pct":            round(float(np.sqrt(max(0.0, self._P[2, 2]))), 3),
            },
            mode=mode,
        )

    def _health_from_state(self) -> ComponentHealth:
        return ComponentHealth(
            refrigerant_charge_pct    = float(np.clip(self._x[0], 0.0, 100.0)),
            compressor_efficiency_pct = float(np.clip(self._x[1], 0.0, 100.0)),
            fan_health_pct            = float(np.clip(self._x[2], 0.0, 100.0)),
        )
