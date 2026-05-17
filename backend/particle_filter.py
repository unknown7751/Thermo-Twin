"""
ParticleFilterRUL — Monte Carlo confidence intervals for Remaining Useful Life.

METHOD
  Maintains N particles, each representing one possible future trajectory.
  Each particle has a perturbed wear rate drawn from:

      r_i ~ Normal(r_observed, σ_noise)   where σ_noise = noise_factor · |r|

  RUL for particle i:
      rul_i = max(0, (current_pct - critical_pct) / |r_i|)   if r_i < 0
              MAX_RUL                                          otherwise

  The (16th, 50th, 84th) percentiles give the ±1σ confidence interval and
  median estimate — robust to outlier particles with near-zero sampled rates.

WHY PERCENTILES NOT MEAN?
  The RUL distribution has a heavy right tail when |r| is small (RUL → ∞).
  Percentiles are invariant to this asymmetry; the mean is not.

EXAMPLE
  Refrigerant at 87%, rate = -2.0 pct/s, critical 50%, noise_factor=0.15:

      True RUL = (87-50)/2.0 = 18.5 s
      Particle rates ~ N(-2.0, 0.30)
      Expected:  p50 ≈ 18.5 s,  p16 ≈ 15.8 s,  p84 ≈ 21.9 s

INTEGRATION
  Use alongside RULEngine:

      pf   = ParticleFilterRUL(n_particles=200)
      rul  = rul_engine.update(health, rates)          # analytical CI
      pf_ci = pf.predict_rul_distribution(health, rates)  # Monte Carlo CI
      # Merge or compare — both express RUL in the same time units.
"""

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("thermo-twin.particle_filter")

MAX_RUL = 9999.0

_CRITICAL_DEFAULTS = {
    "refrigerant": 50.0,
    "compressor":  20.0,
    "fan":         10.0,
}


class ParticleFilterRUL:
    """
    Monte Carlo RUL estimator with N independent particles.

    Args:
        n_particles:         number of Monte Carlo samples (default 200)
        noise_factor:        fractional std of rate perturbation (default 0.15 = 15 %)
        critical_thresholds: same format as RULEngine (optional override)
        rng_seed:            reproducibility seed (None = random)
    """

    def __init__(
        self,
        n_particles: int = 200,
        noise_factor: float = 0.15,
        critical_thresholds: Optional[dict] = None,
        rng_seed: Optional[int] = None,
    ):
        self._N          = n_particles
        self._noise      = noise_factor
        self._thresholds = dict(critical_thresholds or _CRITICAL_DEFAULTS)
        self._rng        = np.random.default_rng(rng_seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_rul_distribution(
        self,
        current_health,
        daily_rates: dict,
    ) -> dict:
        """
        Compute (p16, p50, p84) RUL for each component via particle propagation.

        Args:
            current_health: ComponentHealth dataclass or equivalent dict
            daily_rates:    output of DegradationTrajectoryModel.predict_rate()

        Returns:
            {
              "refrigerant_p16": float,  "refrigerant_p50": float,  "refrigerant_p84": float,
              "compressor_p16":  float,  "compressor_p50":  float,  "compressor_p84":  float,
              "fan_p16":         float,  "fan_p50":         float,  "fan_p84":         float,
              "most_critical_p50": str,  "min_rul_p50": float,
            }
        """
        r_pct = _attr(current_health, "refrigerant_charge_pct",    100.0)
        c_pct = _attr(current_health, "compressor_efficiency_pct", 100.0)
        f_pct = _attr(current_health, "fan_health_pct",            100.0)

        r_rate = daily_rates.get("refrigerant_pct_per_tu", 0.0)
        c_rate = daily_rates.get("compressor_pct_per_tu",  0.0)
        f_rate = daily_rates.get("fan_pct_per_tu",         0.0)

        r_p16, r_p50, r_p84 = self._particle_rul(r_pct, self._thresholds["refrigerant"], r_rate)
        c_p16, c_p50, c_p84 = self._particle_rul(c_pct, self._thresholds["compressor"],  c_rate)
        f_p16, f_p50, f_p84 = self._particle_rul(f_pct, self._thresholds["fan"],         f_rate)

        medians = {"refrigerant": r_p50, "compressor": c_p50, "fan": f_p50}
        most_critical = min(medians, key=lambda k: medians[k])

        return {
            "refrigerant_p16":   r_p16,
            "refrigerant_p50":   r_p50,
            "refrigerant_p84":   r_p84,
            "compressor_p16":    c_p16,
            "compressor_p50":    c_p50,
            "compressor_p84":    c_p84,
            "fan_p16":           f_p16,
            "fan_p50":           f_p50,
            "fan_p84":           f_p84,
            "most_critical_p50": most_critical,
            "min_rul_p50":       min(medians.values()),
        }

    def particle_histogram(
        self,
        current_health,
        daily_rates: dict,
        component: str = "refrigerant",
        bins: int = 20,
    ) -> dict:
        """
        Return a full histogram of particle RULs for one component.

        Useful for plotting uncertainty distributions on the dashboard.

        Returns:
            {"edges": [float, …], "counts": [int, …]}
        """
        pct  = _attr(current_health, _HEALTH_FIELDS[component], 100.0)
        rate = daily_rates.get(f"{component}_pct_per_tu", 0.0)
        ruls = self._particle_sample(pct, self._thresholds[component], rate)
        finite = ruls[ruls < MAX_RUL]
        if len(finite) == 0:
            return {"edges": [0.0, MAX_RUL], "counts": [0]}
        counts, edges = np.histogram(finite, bins=bins)
        return {"edges": edges.tolist(), "counts": counts.tolist()}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _particle_rul(
        self, current_pct: float, critical_pct: float, rate: float
    ) -> tuple:
        ruls = self._particle_sample(current_pct, critical_pct, rate)
        return (
            round(float(np.percentile(ruls, 16)), 2),
            round(float(np.percentile(ruls, 50)), 2),
            round(float(np.percentile(ruls, 84)), 2),
        )

    def _particle_sample(
        self, current_pct: float, critical_pct: float, rate: float
    ) -> np.ndarray:
        """Return (N,) array of particle RUL values."""
        if current_pct <= critical_pct:
            return np.zeros(self._N)

        margin = current_pct - critical_pct

        if abs(rate) < 1e-8:
            return np.full(self._N, MAX_RUL)

        sigma  = abs(rate) * self._noise
        r_particles = rate + self._rng.normal(0.0, sigma, self._N)

        # Degrading particles: r < 0
        degrading = r_particles < -1e-8
        ruls = np.where(
            degrading,
            np.clip(margin / np.where(degrading, -r_particles, 1e-8), 0.0, MAX_RUL),
            MAX_RUL,
        )
        return ruls


# ── Helpers ─────────────────────────────────────────────────────────────────────

_HEALTH_FIELDS = {
    "refrigerant": "refrigerant_charge_pct",
    "compressor":  "compressor_efficiency_pct",
    "fan":         "fan_health_pct",
}


def _attr(obj, name: str, default: float) -> float:
    if hasattr(obj, name):
        return float(getattr(obj, name))
    if isinstance(obj, dict):
        return float(obj.get(name, default))
    return default
