import sys
import logging
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from physics.hvac_physics import HVACPhysicsModel, OperatingConditions, SensorPrediction
from physics.degradation_model import DegradationModel, ComponentHealth
from physics.state_estimator import KalmanStateEstimator

from degradation_trajectory import DegradationTrajectoryModel
from rul_engine import RULEngine
from particle_filter import ParticleFilterRUL
from whatif_engine import WhatIfSimulator, WhatIfRequest

log = logging.getLogger("thermo-twin.engine")

_LSTM_CHECKPOINT = ROOT / "model" / "checkpoints" / "degradation_lstm.pt"


class TwinEngine:
    """
    Orchestrates the physics model + degradation model + Kalman filter + RUL engine.

    Runs alongside (not replacing) the existing autoencoder pipeline.
    Called from app.py after each sensor sample is produced.

    Per-sample pipeline:
        1. Infer OperatingConditions from raw sensor values
        2. HVACPhysicsModel.predict() → healthy baseline
        3. DegradationModel.apply(healthy, current_health) → expected degraded readings
        4. KalmanStateEstimator.predict() — advances uncertainty
        5. KalmanStateEstimator.update(real, degraded) — fuses observation
        6. DegradationTrajectoryModel.update(timestamp, health) — logs health
        7. RULEngine.update(health, rates) — computes time-to-critical
        8. ParticleFilterRUL — Monte Carlo confidence intervals
        9. Return state + divergence + prediction + RUL
    """

    _NOMINAL_POWER_KW = 3.5

    def __init__(
        self,
        use_coolprop: bool = True,
        nominal_ambient_c: float = 35.0,
        time_unit_seconds: float = 1.0,
        machine_id: str = "LIVE-DEMO-UNIT",
    ):
        self.machine_id       = machine_id
        self._physics         = HVACPhysicsModel(use_coolprop=use_coolprop)
        self._degradation     = DegradationModel()
        self._kalman          = KalmanStateEstimator(degradation_model=self._degradation)
        self._nominal_ambient = nominal_ambient_c
        # Cached most-recent outputs so FleetManager can aggregate health without
        # re-running the pipeline. Updated at the end of every process_sample().
        self._last_twin_state: dict = {}
        self._last_rul_state:  dict = {}

        # Load LSTM checkpoint if available; falls back to linear regression silently
        lstm_path = str(_LSTM_CHECKPOINT) if _LSTM_CHECKPOINT.exists() else None
        self._trajectory = DegradationTrajectoryModel(
            model_path=lstm_path,
            time_unit_seconds=time_unit_seconds,
        )
        self._rul_engine  = RULEngine()
        self._pf          = ParticleFilterRUL(n_particles=200, noise_factor=0.15, rng_seed=0)
        self._time_unit   = time_unit_seconds
        self._whatif      = WhatIfSimulator(use_coolprop=use_coolprop)

        log.info(
            "TwinEngine ready  coolprop=%s  lstm=%s  time_unit=%.1fs",
            self._physics._coolprop_available,
            lstm_path is not None,
            time_unit_seconds,
        )

    def process_sample(self, sample: dict, ambient_temp_c: Optional[float] = None) -> dict:
        """
        Process one raw sensor sample through the full digital-twin pipeline.

        Args:
            sample:         dict with keys: compressor_power_kw, discharge_pressure_psi,
                            fan_rpm, supply_air_temp_c, timestamp (optional)
            ambient_temp_c: optional ambient temperature override (default: 35°C)

        Returns:
            {
                "state":          {refrigerant_charge_pct, compressor_efficiency_pct, fan_health_pct},
                "prediction":     {4 sensor keys, model_used},
                "divergence":     {4 sensor keys — real minus predicted},
                "uncertainty":    {3 health state keys — 1σ bounds},
                "estimator_mode": "ukf" | "linear",
                "model_used":     "coolprop" | "linear",
                "rul":            {per-component RUL central + CI + particle CI},
            }
        """
        ambient = ambient_temp_c if ambient_temp_c is not None else self._nominal_ambient

        # ── Physics + Kalman ───────────────────────────────────────────────────
        conditions     = self._conditions_from_sample(sample, ambient)
        healthy_pred   = self._physics.predict(conditions)
        current_health = self._kalman.get_health()
        degraded_pred  = self._degradation.apply(healthy_pred, current_health)

        self._kalman.predict()

        real_pred = SensorPrediction(
            compressor_power_kw    = float(sample["compressor_power_kw"]),
            discharge_pressure_psi = float(sample["discharge_pressure_psi"]),
            fan_rpm                = float(sample["fan_rpm"]),
            supply_air_temp_c      = float(sample["supply_air_temp_c"]),
            model_used             = "real",
        )
        kalman_result = self._kalman.update(real_pred, degraded_pred, healthy_pred)

        # ── Trajectory + RUL ───────────────────────────────────────────────────
        timestamp = float(sample.get("timestamp", 0.0))
        self._trajectory.update(
            timestamp       = timestamp,
            refrigerant_pct = float(kalman_result.x[0]),
            compressor_pct  = float(kalman_result.x[1]),
            fan_pct         = float(kalman_result.x[2]),
        )

        rates = self._trajectory.predict_rate()
        updated_health = ComponentHealth(
            refrigerant_charge_pct    = float(kalman_result.x[0]),
            compressor_efficiency_pct = float(kalman_result.x[1]),
            fan_health_pct            = float(kalman_result.x[2]),
        )

        rul_analytical = self._rul_engine.update(updated_health, rates, uncertainty_pct=15.0)
        rul_mc         = self._pf.predict_rul_distribution(updated_health, rates)

        rul_payload = {
            **rul_analytical,
            "mc": rul_mc,
            "history_samples": len(self._trajectory),
            "rate_mode": "lstm" if (
                self._trajectory._lstm is not None
                and len(self._trajectory) >= self._trajectory._lookback
            ) else "linear",
        }

        result = {
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
            "rul":            rul_payload,
        }

        # Cache for FleetManager (avoids re-running the pipeline for aggregation)
        self._last_twin_state = result
        self._last_rul_state  = rul_payload
        return result

    def simulate_whatif(self, params: dict) -> dict:
        """
        Project a hypothetical operating scenario forward (Phase 4 What-If).

        Starts from the twin's *current* estimated health so the projection is
        grounded in the machine's real condition, not a fresh/perfect unit.
        """
        health = self._kalman.get_health()
        req = WhatIfRequest(
            compressor_speed_pct      = float(params.get("compressor_speed_pct", 70.0)),
            ambient_temp_c            = float(params.get("ambient_temp_c", 35.0)),
            load_demand_pct           = float(params.get("load_demand_pct", 50.0)),
            simulation_duration_hours = float(params.get("simulation_duration_hours", 4.0)),
            refrigerant_charge_pct    = health.refrigerant_charge_pct,
            compressor_efficiency_pct = health.compressor_efficiency_pct,
            fan_health_pct            = health.fan_health_pct,
        )
        return self._whatif.simulate(req, include_baseline=True)

    def reset(self) -> None:
        """Reset Kalman state and health history (call after part replacement or stream reset)."""
        self._kalman.reset()
        self._trajectory.clear()
        self._last_twin_state = {}
        self._last_rul_state  = {}

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
