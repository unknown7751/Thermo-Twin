"""
Phase 7 — Drift Detector.

Tracks how well the digital twin's physics model matches reality over time.
Computes per-sensor MAE/RMSE and an aggregate accuracy (0–100). When accuracy
falls below a configurable threshold, flags the unit as drifting and
heuristically classifies the most likely cause.

Multi-sensor design (deviates from the literal spec)
----------------------------------------------------
The spec API was single-sensor. We track ALL four sensors — power, pressure,
fan RPM, supply temp — because the twin produces predictions for all four
and choosing just one (e.g. compressor_power_kw) would mask drift on the
others. The aggregate `accuracy_pct` is the mean of per-sensor accuracies.

Baseline behaviour
------------------
The first ~50 samples establish a baseline MAE per sensor. After that,
accuracy decays linearly from 100 % as observed MAE exceeds baseline.
`reset_baseline()` (called after recalibration or commissioning) re-anchors
the baseline so future drift is measured against the fresh calibration.
"""

import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("thermo-twin.drift")

SENSOR_KEYS = (
    "compressor_power_kw",
    "discharge_pressure_psi",
    "fan_rpm",
    "supply_air_temp_c",
)

# Per-sensor scale used to normalise errors so they're comparable across sensors
# (otherwise fan RPM at scale ~1000 would dominate temp at scale ~10).
_SCALE = {
    "compressor_power_kw":    3.5,
    "discharge_pressure_psi": 245.0,
    "fan_rpm":                1190.0,
    "supply_air_temp_c":      11.0,
}


@dataclass
class DriftMetrics:
    timestamp: float
    mae: float                                     # aggregate (mean over sensors of normalised |err|)
    rmse: float                                    # aggregate
    reconstruction_error: float                    # autoencoder error
    accuracy_pct: float                            # 0–100
    is_drifting: bool
    drift_reason: str
    per_sensor: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DriftDetector:
    def __init__(
        self,
        window_size_samples: int = 500,
        accuracy_threshold_pct: float = 95.0,
        baseline_warmup: int = 50,
    ):
        self._window_size           = window_size_samples
        self._accuracy_threshold    = accuracy_threshold_pct
        self._baseline_warmup       = baseline_warmup

        # Rolling per-sensor (real, predicted) buffers
        self._buffers: Dict[str, deque] = {
            k: deque(maxlen=window_size_samples) for k in SENSOR_KEYS
        }
        self._recon_buf  = deque(maxlen=window_size_samples)

        # Baseline MAE per sensor (set after warmup, reset on recalibration)
        self._baseline_mae: Dict[str, Optional[float]] = {k: None for k in SENSOR_KEYS}

        # Rolling history of aggregate metrics
        self._drift_history: deque = deque(maxlen=10_000)
        self._sample_count = 0

    # ── Public API ─────────────────────────────────────────────────────────
    def update(
        self,
        real:        Dict[str, float],
        predicted:   Dict[str, float],
        reconstruction_error: float = 0.0,
    ) -> DriftMetrics:
        """Push one (real, predicted) pair across all sensors, update metrics."""
        for k in SENSOR_KEYS:
            r = real.get(k);  p = predicted.get(k)
            if r is not None and p is not None:
                self._buffers[k].append((float(r), float(p)))
        self._recon_buf.append(float(reconstruction_error))
        self._sample_count += 1

        per_sensor: Dict[str, Dict[str, float]] = {}
        normalised_mae_vals = []

        for k in SENSOR_KEYS:
            buf = self._buffers[k]
            if not buf:
                continue
            arr = np.asarray(buf, dtype=float)
            err = arr[:, 0] - arr[:, 1]
            mae  = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            scale = _SCALE[k] or 1.0
            per_sensor[k] = {
                "mae":            round(mae, 4),
                "rmse":           round(rmse, 4),
                "normalised_mae": round(mae / scale, 4),
            }
            normalised_mae_vals.append(mae / scale)

        # Latch baseline once warmup is met (per-sensor, in absolute units)
        if self._sample_count == self._baseline_warmup:
            for k in SENSOR_KEYS:
                ps = per_sensor.get(k)
                if ps:
                    self._baseline_mae[k] = ps["mae"]
            log.info("DriftDetector baseline latched: %s", self._baseline_mae)

        # Per-sensor accuracy: 100 pp at baseline, drops linearly with extra
        # normalised MAE. 200 pp per unit of normalised-MAE excess gives a
        # readable scale: 5% extra sensor error → ~10 pp drop.
        accuracies = []
        for k in SENSOR_KEYS:
            ps = per_sensor.get(k)
            if not ps:
                continue
            bl_norm = ((self._baseline_mae[k] or 0.0) / (_SCALE[k] or 1.0))
            excess  = max(0.0, ps["normalised_mae"] - bl_norm)
            accuracies.append(max(0.0, 100.0 - 200.0 * excess))

        accuracy_pct = float(np.mean(accuracies)) if accuracies else 100.0
        mae_agg  = float(np.mean(normalised_mae_vals)) if normalised_mae_vals else 0.0
        rmse_agg = float(np.sqrt(np.mean([
            ps["rmse"] / (_SCALE[k] or 1.0)
            for k, ps in per_sensor.items()
        ]) ** 2)) if per_sensor else 0.0
        recon_avg = float(np.mean(self._recon_buf)) if self._recon_buf else 0.0

        is_drifting = accuracy_pct < self._accuracy_threshold
        reason      = self._diagnose(per_sensor, recon_avg) if is_drifting else "stable"

        metrics = DriftMetrics(
            timestamp            = datetime.now().timestamp(),
            mae                  = round(mae_agg, 4),
            rmse                 = round(rmse_agg, 4),
            reconstruction_error = round(recon_avg, 4),
            accuracy_pct         = round(accuracy_pct, 1),
            is_drifting          = is_drifting,
            drift_reason         = reason,
            per_sensor           = per_sensor,
        )
        self._drift_history.append(metrics)
        return metrics

    def get_current_metrics(self) -> DriftMetrics:
        if not self._drift_history:
            return DriftMetrics(
                timestamp=datetime.now().timestamp(),
                mae=0.0, rmse=0.0, reconstruction_error=0.0,
                accuracy_pct=100.0, is_drifting=False, drift_reason="no_data",
            )
        return self._drift_history[-1]

    def get_drift_trend(self, hours: int = 24) -> dict:
        cutoff = datetime.now().timestamp() - hours * 3600
        recent = [m for m in self._drift_history if m.timestamp > cutoff]
        if len(recent) < 2:
            return {"insufficient_data": True, "num_samples": len(recent)}
        acc = np.asarray([m.accuracy_pct for m in recent])
        t   = np.arange(len(acc))
        slope = float(np.polyfit(t, acc, 1)[0])   # pp per sample
        per_hour = slope * (len(recent) / max(1, hours))
        is_down = per_hour < -0.1
        if per_hour < -1.0:
            rec = "URGENT: accuracy degrading rapidly — investigate or recalibrate."
        elif is_down:
            rec = "WARNING: accuracy drifting down — schedule recalibration."
        elif abs(per_hour) < 0.05:
            rec = "STABLE: twin accuracy holding."
        else:
            rec = "INFO: accuracy improving — normal post-recalibration."
        return {
            "avg_accuracy_pct":       round(float(acc.mean()), 1),
            "min_accuracy_pct":       round(float(acc.min()),  1),
            "max_accuracy_pct":       round(float(acc.max()),  1),
            "is_trending_down":       is_down,
            "drift_rate_pct_per_hour": round(per_hour, 3),
            "num_samples":            len(recent),
            "recommendation":         rec,
        }

    def reset_baseline(self) -> None:
        """Re-latch the baseline from current buffer contents (post-recalibration)."""
        for k in SENSOR_KEYS:
            buf = self._buffers[k]
            if buf:
                arr = np.asarray(buf, dtype=float)
                self._baseline_mae[k] = float(np.mean(np.abs(arr[:, 0] - arr[:, 1])))
            else:
                self._baseline_mae[k] = None
        self._drift_history.clear()
        self._sample_count = max(self._sample_count, self._baseline_warmup)
        log.info("DriftDetector baseline reset")

    # ── Internals ──────────────────────────────────────────────────────────
    def _diagnose(self, per_sensor: Dict[str, Dict[str, float]], recon: float) -> str:
        """Coarse pattern attribution from which sensor(s) are worst."""
        if recon > 0.30:
            return "anomaly_detected"
        if not per_sensor:
            return "unknown"
        worst = max(per_sensor, key=lambda k: per_sensor[k]["normalised_mae"])
        worst_norm = per_sensor[worst]["normalised_mae"]
        # All sensors mildly off → systematic / coefficient drift
        avg_norm = float(np.mean([ps["normalised_mae"] for ps in per_sensor.values()]))
        if worst_norm < 1.5 * avg_norm and avg_norm > 0.03:
            return "systematic_calibration_drift"
        # One sensor dominates → physical pattern hint
        if worst == "discharge_pressure_psi":
            return "pressure_drift_(refrigerant_or_coil)"
        if worst == "fan_rpm":
            return "fan_drift"
        if worst == "compressor_power_kw":
            return "compressor_drift"
        if worst == "supply_air_temp_c":
            return "thermal_drift"
        return "unknown"
