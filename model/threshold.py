"""
Threshold calibration and severity scoring utilities.
These are pure numpy functions — no PyTorch dependency.
"""

import json
import collections
import numpy as np
from pathlib import Path


def compute_threshold(val_errors: np.ndarray, n_sigma: float = 2.5):
    """
    Compute anomaly threshold from validation reconstruction errors.
    Returns (threshold, mean, std).
    """
    mean = float(np.mean(val_errors))
    std  = float(np.std(val_errors))
    return mean + n_sigma * std, mean, std


def severity_score(
    errors: np.ndarray,
    threshold: float,
    p99_error: float | None = None,
) -> np.ndarray:
    """
    Map reconstruction errors to integer severity scores 0-100.

    Normal range  (error <= threshold)  : 0 - 40   (linear)
    Anomaly range (error >  threshold)  : 41 - 100  (log scale)
    """
    if p99_error is None or p99_error <= threshold:
        p99_error = threshold * 50.0

    scores = np.empty(len(errors), dtype=float)

    below = errors <= threshold
    scores[below] = 40.0 * errors[below] / threshold

    above = ~below
    log_num = np.log(errors[above] / threshold)
    log_den = np.log(p99_error  / threshold)
    scores[above] = 41.0 + 59.0 * np.minimum(log_num / log_den, 1.0)

    return np.clip(scores, 0, 100).round().astype(int)


def load_threshold_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_threshold_config(path: str | Path, config: dict) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


class ThresholdManager:
    """
    Dynamic anomaly threshold that recalibrates from recent normal-operation
    reconstruction errors rather than relying on a single static value.

    Algorithm:
      - Maintain a rolling buffer of the last 500 reconstruction errors
        from windows that scored as normal (severity < 40).
      - Dynamic threshold = 95th percentile of that buffer.
      - Fall back to static_fallback when buffer has < 50 samples
        (cold start / first boot).

    Buffer is persisted to JSON so the threshold survives a server restart
    without a cold-start period on the next boot.

    Usage:
        mgr = ThresholdManager(state_path="model/checkpoints/threshold_state.json")
        t   = mgr.get_threshold()
        mgr.update(reconstruction_error=0.04, severity_score=12)
    """

    BUFFER_MAX      = 500
    BUFFER_MIN      = 50
    QUANTILE        = 0.95
    STATIC_FALLBACK = 0.200650  # from val calibration (mean + 2.5sigma)

    def __init__(self, state_path: str | Path | None = None):
        self._buffer     = collections.deque(maxlen=self.BUFFER_MAX)
        self._state_path = Path(state_path) if state_path else None
        if self._state_path and self._state_path.exists():
            self._load()

    # -- Public API -------------------------------------------------------------

    def get_threshold(self) -> float:
        """
        Return current dynamic threshold.
        Falls back to STATIC_FALLBACK if fewer than BUFFER_MIN samples collected.
        """
        if len(self._buffer) < self.BUFFER_MIN:
            return self.STATIC_FALLBACK
        return float(np.quantile(list(self._buffer), self.QUANTILE))

    def update(self, reconstruction_error: float, severity_score: int) -> None:
        """
        Update rolling buffer with a new reconstruction error.
        Only normal windows (severity < 40) adjust the baseline.
        """
        if severity_score < 40:
            self._buffer.append(float(reconstruction_error))
            if self._state_path:
                self._save()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def is_dynamic(self) -> bool:
        return len(self._buffer) >= self.BUFFER_MIN

    # -- Persistence ------------------------------------------------------------

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "buffer":          list(self._buffer),
            "static_fallback": self.STATIC_FALLBACK,
        }
        with open(self._state_path, "w") as f:
            json.dump(state, f)

    def _load(self) -> None:
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._buffer = collections.deque(
                state.get("buffer", []), maxlen=self.BUFFER_MAX
            )
        except (json.JSONDecodeError, KeyError):
            self._buffer = collections.deque(maxlen=self.BUFFER_MAX)


class MCDropoutInference:
    """
    MC-Dropout uncertainty estimator for the Thermo-Twin autoencoder.

    Runs N stochastic forward passes (dropout active) and returns a mean
    severity score with a 95% half-interval uncertainty and confidence %.

    Algorithm:
      1. Put model in train() mode — activates dropout masks.
      2. Run N forward passes; collect per-pass reconstruction errors.
      3. Map each error set to severity scores via severity_score().
      4. uncertainty  = round(1.96 * std(per_pass_severities))
      5. confidence   = clip(100 - uncertainty, 0, 100)
      6. action_override: "INVESTIGATE" when sev > 71 AND uncertainty > 20

    Usage:
        mc = MCDropoutInference(model, n_passes=10)
        result = mc.infer(window_tensor, threshold, p99_error)
        # {"mean_severity": 73, "uncertainty": 8, "confidence_pct": 89, ...}
    """

    N_PASSES = 10

    def __init__(self, model, n_passes: int = N_PASSES):
        self._model = model
        self._n     = n_passes

    def infer(self, window, threshold: float, p99_error: float) -> dict:
        """
        Args:
            window: FloatTensor shape (1, 200) — a single scaled sensor window
            threshold: anomaly threshold from threshold_config.json
            p99_error: p99 reconstruction error for severity log-scaling

        Returns dict: mean_severity, uncertainty, confidence_pct,
                      action_override (str or None), per_pass_severities list.
        """
        import torch
        passes = self._model.mc_reconstruction_errors(window, n_passes=self._n)
        # passes: (n_passes, 1)
        pass_errors = passes.numpy()  # shape (n_passes, 1)

        all_sevs = np.array([
            int(severity_score(pass_errors[i], threshold, p99_error)[0])
            for i in range(self._n)
        ])  # shape (n_passes,)

        mean_sev    = int(np.round(all_sevs.mean()))
        sev_std     = float(all_sevs.std())
        uncertainty = int(np.round(sev_std * 1.96))
        confidence  = int(np.clip(100 - uncertainty, 0, 100))

        if mean_sev > 71 and uncertainty > 20:
            action_override = "INVESTIGATE"
        elif mean_sev > 71:
            action_override = "STOP UNIT"
        else:
            action_override = None

        return {
            "mean_severity":       mean_sev,
            "uncertainty":         uncertainty,
            "confidence_pct":      confidence,
            "action_override":     action_override,
            "per_pass_severities": all_sevs.tolist(),
        }

# -- Severity profiles & classifier --------------------------------------------

SEVERITY_PROFILES = {
    "hospital": {
        "description": "Hospital / Critical Care — zero tolerance for cooling failure",
        "warn":         25,
        "critical":     45,
    },
    "cold_chain": {
        "description": "Cold Chain / Food Storage — early warning essential",
        "warn":         20,
        "critical":     35,
    },
    "commercial_office": {
        "description": "Commercial Office — standard operations (default)",
        "warn":         41,
        "critical":     71,
    },
    "warehouse": {
        "description": "Warehouse / Light Industrial — tolerant of minor faults",
        "warn":         55,
        "critical":     80,
    },
}


class SeverityClassifier:
    """
    Classifies a severity score into NORMAL / WARNING / CRITICAL based on a
    configurable operational profile (hospital, cold_chain, commercial_office,
    warehouse).

    Usage:
        clf = SeverityClassifier(profile="hospital")
        label, action = clf.classify(score=48)
        # -> ("CRITICAL", "STOP UNIT -- Dispatch Now")
        clf.set_profile("warehouse")
        label, action = clf.classify(48)
        # -> ("NORMAL", "Log Only")
    """

    DEFAULT_PROFILE = "commercial_office"

    _ACTIONS = {
        "NORMAL":   "Log Only",
        "WARNING":  "Notify Operator",
        "CRITICAL": "STOP UNIT -- Dispatch Now",
    }

    def __init__(self, profile: str = DEFAULT_PROFILE):
        self._profile = None
        self.set_profile(profile)

    def set_profile(self, name: str) -> None:
        if name not in SEVERITY_PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Choose from: {list(SEVERITY_PROFILES)}")
        self._profile = name

    def get_profile(self) -> str:
        return self._profile

    def thresholds(self) -> tuple[int, int]:
        """Return (warn_threshold, critical_threshold) for current profile."""
        p = SEVERITY_PROFILES[self._profile]
        return p["warn"], p["critical"]

    def classify(self, score: int) -> tuple[str, str]:
        """
        Classify a severity score under the current profile.

        Returns:
            (level, action) where level is "NORMAL" | "WARNING" | "CRITICAL"
            and action is the recommended operator response.
        """
        warn, crit = self.thresholds()
        if score >= crit:
            level = "CRITICAL"
        elif score >= warn:
            level = "WARNING"
        else:
            level = "NORMAL"
        return level, self._ACTIONS[level]