"""
DegradationTrajectoryModel — estimates per-component health wear rates
from a rolling history of Kalman-filtered health observations.

Two inference modes (automatic fallback):

  LINEAR (always available, >= MIN_LINEAR_SAMPLES):
    Fits OLS slope to each health channel vs normalised time.
    Returns signed rate in pct / time_unit_seconds.

  LSTM (optional, >= LOOKBACK_SAMPLES, requires checkpoint):
    LSTM(3→64) → Linear(64→32) → ReLU → Linear(32→3)
    Trained on synthetic degradation trajectories.
    Input : last LOOKBACK_SAMPLES frames normalised to [0, 1].
    Output: rate in pct / day, converted to pct / time_unit_seconds.

TIME UNITS
  All rates and the predict_trajectory() output share the same time unit,
  controlled by `time_unit_seconds` (default 1.0 = per sim-second).
  Set to 3600.0 for per-hour, 86400.0 for per-day.
  RULEngine consumes whatever unit is used here, so RUL output is in
  the same unit automatically.

SIGNED CONVENTION
  Negative = degrading (health falling).
  Positive = improving (unusual; effectively means RUL = ∞).
  Zero     = stable.

Example (linear mode)::

    m = DegradationTrajectoryModel()
    for t in range(10):
        m.update(t, 100 - t*3, 100.0, 100.0)   # refrig drops 3 pct/s
    rates = m.predict_rate()
    assert rates["refrigerant_pct_per_tu"] < -2.5
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("thermo-twin.trajectory")


@dataclass
class _HealthSample:
    timestamp: float          # simulation seconds
    refrigerant_pct: float
    compressor_pct: float
    fan_pct: float


class DegradationTrajectoryModel:
    """
    Rolling-window degradation rate estimator.

    Args:
        model_path:         path to degradation_lstm.pt (None → linear only)
        lookback_samples:   LSTM sequence length (168 = 14 days × 12/day)
        max_history:        rolling buffer cap
        time_unit_seconds:  denominator for reported rates
    """

    MIN_LINEAR_SAMPLES = 5
    LOOKBACK_SAMPLES   = 168     # 14 days at 2-hour aggregation
    MAX_HISTORY        = 4000

    def __init__(
        self,
        model_path: Optional[str] = None,
        lookback_samples: int = LOOKBACK_SAMPLES,
        max_history: int = MAX_HISTORY,
        time_unit_seconds: float = 1.0,
    ):
        self._lookback  = lookback_samples
        self._time_unit = time_unit_seconds
        self._history: deque = deque(maxlen=max_history)
        self._lstm      = _load_lstm_checkpoint(model_path)
        mode = "lstm" if self._lstm else "linear"
        log.info("DegradationTrajectoryModel init  mode=%s  time_unit=%.1fs", mode, time_unit_seconds)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        timestamp: float,
        refrigerant_pct: float,
        compressor_pct: float,
        fan_pct: float,
    ) -> None:
        """Append one health observation to the rolling buffer."""
        self._history.append(
            _HealthSample(timestamp, refrigerant_pct, compressor_pct, fan_pct)
        )

    def predict_rate(self) -> dict:
        """
        Return signed degradation rate per component (pct / time_unit_seconds).

        Returns all-zero dict when history is too short.
        """
        if len(self._history) < self.MIN_LINEAR_SAMPLES:
            return _zero_rates()

        if self._lstm is not None and len(self._history) >= self._lookback:
            return self._predict_lstm()

        return self._predict_linear()

    # Spec-compatible alias
    def predict_daily_rate(self) -> dict:
        return self.predict_rate()

    def predict_trajectory(self, steps_ahead: int = 30) -> dict:
        """
        Project health forward `steps_ahead` time-units using current rates.

        Returns dict with "refrigerant", "compressor", "fan";
        each a list of length steps_ahead+1 (index 0 = current).
        """
        rates = self.predict_rate()
        cur   = self._current_health()
        steps = np.arange(steps_ahead + 1, dtype=float)

        def project(start: float, rate: float) -> list:
            return np.clip(start + rate * steps, 0.0, 100.0).tolist()

        return {
            "refrigerant": project(cur[0], rates["refrigerant_pct_per_tu"]),
            "compressor":  project(cur[1], rates["compressor_pct_per_tu"]),
            "fan":         project(cur[2], rates["fan_pct_per_tu"]),
        }

    def clear(self) -> None:
        """Reset rolling history (call after stream reset or component replacement)."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _current_health(self) -> tuple:
        if not self._history:
            return (100.0, 100.0, 100.0)
        s = self._history[-1]
        return (s.refrigerant_pct, s.compressor_pct, s.fan_pct)

    def _predict_linear(self) -> dict:
        """OLS slope: pct per time_unit_seconds."""
        h = list(self._history)
        t = np.array([s.timestamp for s in h])
        # Normalise: slope becomes pct / time_unit_seconds
        t_norm = (t - t[0]) / self._time_unit

        A = np.vstack([t_norm, np.ones_like(t_norm)]).T

        def slope(y: np.ndarray) -> float:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(coef[0])

        return {
            "refrigerant_pct_per_tu": slope(np.array([s.refrigerant_pct for s in h])),
            "compressor_pct_per_tu":  slope(np.array([s.compressor_pct  for s in h])),
            "fan_pct_per_tu":         slope(np.array([s.fan_pct          for s in h])),
        }

    def _predict_lstm(self) -> dict:
        """LSTM inference on the last `_lookback` samples."""
        try:
            import torch
            h   = list(self._history)[-self._lookback:]
            arr = np.array(
                [[s.refrigerant_pct, s.compressor_pct, s.fan_pct] for s in h],
                dtype=np.float32,
            ) / 100.0                                           # → [0, 1]
            x = torch.tensor(arr).unsqueeze(0)                  # (1, L, 3)
            with torch.no_grad():
                rates_pd = self._lstm(x).squeeze().numpy()      # (3,) pct/day
            # Convert pct/day → pct/time_unit_seconds
            tu_in_days = self._time_unit / 86400.0
            return {
                "refrigerant_pct_per_tu": float(rates_pd[0]) * tu_in_days,
                "compressor_pct_per_tu":  float(rates_pd[1]) * tu_in_days,
                "fan_pct_per_tu":         float(rates_pd[2]) * tu_in_days,
            }
        except Exception as exc:
            log.warning("LSTM inference failed (%s) — using linear fallback", exc)
            return self._predict_linear()


# ── LSTM checkpoint loader ──────────────────────────────────────────────────────

def _load_lstm_checkpoint(path: Optional[str]):
    """Return a loaded nn.Module or None (never raises)."""
    if path is None:
        return None
    try:
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(3, 64, batch_first=True)
                self.fc1  = nn.Linear(64, 32)
                self.relu = nn.ReLU()
                self.fc2  = nn.Linear(32, 3)

            def forward(self, x):               # x: (B, L, 3)
                out, _ = self.lstm(x)
                return self.fc2(self.relu(self.fc1(out[:, -1, :])))

        net = _Net()
        sd  = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(sd)
        net.eval()
        log.info("LSTM checkpoint loaded from %s", path)
        return net
    except Exception as exc:
        log.info("LSTM checkpoint not available (%s) — linear mode active", type(exc).__name__)
        return None


def _zero_rates() -> dict:
    return {
        "refrigerant_pct_per_tu": 0.0,
        "compressor_pct_per_tu":  0.0,
        "fan_pct_per_tu":         0.0,
    }
