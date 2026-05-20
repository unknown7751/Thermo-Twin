"""
Phase 7 — Bayesian Parameter Estimator.

Re-fits the four commissioning coefficients (k_disc, k_fan, k_temp_a, k_temp_b)
of `HVACPhysicsModel` from recent normal-operation samples. Combines an OLS fit
with a Gaussian prior centred on the current value — so the posterior gently
shifts toward the data only when the prior is overwhelmed by enough evidence.

Posterior for `y = k * x + noise` with Gaussian prior on k:
    1/τ_post = 1/τ_prior + N/σ_data²
    k_post   = τ_post * ( k_prior / σ_prior² + Σ(x_i*y_i) / σ_data² )

Confidence = clamp(N / N_full, 0, 1). N_full chosen so ~1000 samples yields
confidence ≈ 1.0.

If the new value would change a coefficient by more than 10% in one step, the
update is rejected — safety guard against a bad fit silently breaking the twin.
"""

import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("thermo-twin.params")


# ── Priors: centred on Carrier commissioning numbers, std = ~5% of mean ──────
_PRIORS = {
    "k_disc":   {"mean": 69.99,  "std": 3.5},
    "k_fan":    {"mean": 340.19, "std": 17.0},
    "k_temp_a": {"mean": 18.05,  "std": 1.0},
    "k_temp_b": {"mean": 2.01,   "std": 0.15},
}

# Max single-step change before we reject the fit as untrustworthy
MAX_CHANGE_PCT = 10.0
N_FULL_CONFIDENCE = 1000


@dataclass
class ParameterUpdate:
    timestamp:      float
    parameter_name: str
    old_value:      float
    new_value:      float
    change_pct:     float
    confidence:     float
    reason:         str
    rejected:       bool = False
    reject_reason:  Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BayesianParameterEstimator:
    """Fits k_disc / k_fan / k_temp_{a,b} from buffered normal-operation samples."""

    def __init__(self, physics_model=None, buffer_size: int = 10_000):
        self._physics = physics_model
        # Single dict of parallel deques — no per-sample list pops (O(1) FIFO)
        self._buf = {
            "power":     deque(maxlen=buffer_size),
            "pressure":  deque(maxlen=buffer_size),
            "rpm":       deque(maxlen=buffer_size),
            "temp":      deque(maxlen=buffer_size),
            "timestamp": deque(maxlen=buffer_size),
        }
        self._update_history: List[ParameterUpdate] = []
        # Seed current params from physics model (or fall back to priors)
        if physics_model is not None and hasattr(physics_model, "get_calibration"):
            self._params = physics_model.get_calibration()
        else:
            self._params = {k: v["mean"] for k, v in _PRIORS.items()}

    # ── Sample ingest ──────────────────────────────────────────────────────
    def add_sample(self, power_kw: float, pressure_psi: float, fan_rpm: float, temp_c: float) -> None:
        self._buf["power"].append(float(power_kw))
        self._buf["pressure"].append(float(pressure_psi))
        self._buf["rpm"].append(float(fan_rpm))
        self._buf["temp"].append(float(temp_c))
        self._buf["timestamp"].append(datetime.now().timestamp())

    def clear_buffer(self) -> None:
        for q in self._buf.values():
            q.clear()

    # ── Estimation ─────────────────────────────────────────────────────────
    def estimate_parameters(
        self,
        lookback_hours:       float = 720.0,
        confidence_threshold: float = 0.05,
        apply_to_physics:     bool  = True,
        reason:               str   = "scheduled_monthly",
        min_samples:          int   = 50,
    ) -> List[ParameterUpdate]:
        """Re-fit all four coefficients and (optionally) apply to the physics model.

        Returns every attempted ParameterUpdate, including ones marked
        `rejected=True` (e.g. low confidence, fit failed, change too large).
        Only non-rejected updates are persisted to history and applied.
        """
        cutoff = datetime.now().timestamp() - lookback_hours * 3600
        ts = np.asarray(self._buf["timestamp"])
        mask = ts > cutoff
        n = int(mask.sum())
        if n < min_samples:
            log.warning("ParamEstimator: only %d/%d samples in lookback window, skipping", n, min_samples)
            return []

        power    = np.asarray(self._buf["power"])[mask]
        pressure = np.asarray(self._buf["pressure"])[mask]
        rpm      = np.asarray(self._buf["rpm"])[mask]
        temp     = np.asarray(self._buf["temp"])[mask]

        attempts: List[ParameterUpdate] = []
        attempts.append(self._fit_linear_through_origin(power, pressure, "k_disc",  reason))
        attempts.append(self._fit_linear_through_origin(power, rpm,      "k_fan",   reason))
        a_upd, b_upd = self._fit_temp_model(power, temp, reason)
        attempts.extend([a_upd, b_upd])

        # Apply the accepted ones
        applied = [u for u in attempts if u is not None and not u.rejected and u.confidence >= confidence_threshold]
        for upd in applied:
            self._params[upd.parameter_name] = upd.new_value
            self._update_history.append(upd)
        # Also persist rejected ones for visibility in /twin/parameters
        for upd in attempts:
            if upd is not None and (upd.rejected or upd.confidence < confidence_threshold) and upd not in applied:
                upd.rejected = True
                if not upd.reject_reason:
                    upd.reject_reason = f"confidence {upd.confidence:.2f} < threshold {confidence_threshold:.2f}"
                self._update_history.append(upd)

        if apply_to_physics and self._physics is not None and hasattr(self._physics, "set_calibration"):
            self._physics.set_calibration(**{u.parameter_name: u.new_value for u in applied})

        log.info("ParamEstimator: %d/%d updates applied (reason=%s)", len(applied), len(attempts), reason)
        return [u for u in attempts if u is not None]

    # ── Internals ──────────────────────────────────────────────────────────
    def _fit_linear_through_origin(
        self, x: np.ndarray, y: np.ndarray, name: str, reason: str,
    ) -> Optional[ParameterUpdate]:
        """Posterior estimate of k in y = k*x with Gaussian prior, intercept = 0."""
        valid = x > 0.1
        x = x[valid]; y = y[valid]
        if len(x) < 10:
            return self._reject(name, reason, "insufficient_valid_samples")

        prior = _PRIORS[name]
        k_ols = float(np.sum(x * y) / np.sum(x * x))
        sigma_data = float(np.std(y - k_ols * x)) or 1e-6
        # Bayesian posterior on slope-only model: weight prior vs OLS by precision
        prior_prec = 1.0 / (prior["std"] ** 2)
        data_prec  = float(np.sum(x * x)) / (sigma_data ** 2)
        post_prec  = prior_prec + data_prec
        post_mean  = (prior["mean"] * prior_prec + k_ols * data_prec) / post_prec
        confidence = float(min(1.0, len(x) / N_FULL_CONFIDENCE))

        old = float(self._params.get(name, prior["mean"]))
        new = float(post_mean)
        change_pct = (new - old) / (old + 1e-9) * 100.0
        if abs(change_pct) > MAX_CHANGE_PCT:
            return self._reject(name, reason, f"change {change_pct:+.1f}% exceeds ±{MAX_CHANGE_PCT}%",
                                 old=old, new=new, change_pct=change_pct, confidence=confidence)
        return ParameterUpdate(
            timestamp=datetime.now().timestamp(),
            parameter_name=name, old_value=round(old, 4), new_value=round(new, 4),
            change_pct=round(change_pct, 2), confidence=round(confidence, 3), reason=reason,
        )

    def _fit_temp_model(self, x: np.ndarray, y: np.ndarray, reason: str):
        """Fit y = a − b*x via OLS, then Bayesian-blend each coefficient with prior."""
        if len(x) < 10:
            return self._reject("k_temp_a", reason, "insufficient_samples"), \
                   self._reject("k_temp_b", reason, "insufficient_samples")
        try:
            X = np.column_stack([np.ones(len(x)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            a_ols, neg_b_ols = float(sol[0]), float(sol[1])
            b_ols = -neg_b_ols
        except Exception as exc:
            return self._reject("k_temp_a", reason, f"lstsq failed: {exc}"), \
                   self._reject("k_temp_b", reason, f"lstsq failed: {exc}")
        confidence = float(min(1.0, len(x) / N_FULL_CONFIDENCE))
        return (
            self._blend("k_temp_a", a_ols, confidence, reason),
            self._blend("k_temp_b", b_ols, confidence, reason),
        )

    def _blend(self, name: str, k_ols: float, confidence: float, reason: str) -> ParameterUpdate:
        prior = _PRIORS[name]
        # Simple precision-weighted blend (OLS ≈ data, prior anchors)
        w = confidence
        new = float(prior["mean"] * (1 - w) + k_ols * w)
        old = float(self._params.get(name, prior["mean"]))
        change_pct = (new - old) / (old + 1e-9) * 100.0
        if abs(change_pct) > MAX_CHANGE_PCT:
            return self._reject(name, reason, f"change {change_pct:+.1f}% exceeds ±{MAX_CHANGE_PCT}%",
                                 old=old, new=new, change_pct=change_pct, confidence=confidence)
        return ParameterUpdate(
            timestamp=datetime.now().timestamp(),
            parameter_name=name, old_value=round(old, 4), new_value=round(new, 4),
            change_pct=round(change_pct, 2), confidence=round(confidence, 3), reason=reason,
        )

    def _reject(self, name, reason, msg, *, old=None, new=None, change_pct=0.0, confidence=0.0):
        if old is None:
            old = float(self._params.get(name, _PRIORS[name]["mean"]))
        if new is None:
            new = old
        return ParameterUpdate(
            timestamp=datetime.now().timestamp(),
            parameter_name=name, old_value=round(old, 4), new_value=round(new, 4),
            change_pct=round(change_pct, 2), confidence=round(confidence, 3),
            reason=reason, rejected=True, reject_reason=msg,
        )

    # ── Read API ───────────────────────────────────────────────────────────
    def get_current_parameters(self) -> dict:
        return dict(self._params)

    def get_update_history(self, limit: int = 100) -> List[ParameterUpdate]:
        return self._update_history[-limit:]

    def buffered_sample_count(self) -> int:
        return len(self._buf["power"])
