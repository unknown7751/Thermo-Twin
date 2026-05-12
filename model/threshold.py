"""
Threshold calibration and severity scoring utilities.
These are pure numpy functions — no PyTorch dependency.
"""

import numpy as np
import json
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

    Log scale is used for the anomaly range so that very-high-error outliers
    (e.g. power spikes at 20x threshold) don't compress the drift/decoupling
    scores into the bottom of the range.
    """
    if p99_error is None or p99_error <= threshold:
        p99_error = threshold * 50.0

    scores = np.empty(len(errors), dtype=float)

    # Normal range: linear [0, threshold] -> [0, 40]
    below = errors <= threshold
    scores[below] = 40.0 * errors[below] / threshold

    # Anomaly range: log [threshold, inf] -> [41, 100]
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
