"""
RULEngine — Remaining Useful Life calculator for HVAC digital twin.

CORE FORMULA
  RUL = (current_pct - critical_pct) / |wear_rate_pct_per_tu|

  where the time unit is whatever DegradationTrajectoryModel was configured
  with (default: 1 simulation-second).

UNCERTAINTY PROPAGATION
  Given rate estimate r with fractional uncertainty σ_r = uncertainty_factor·|r|:

      σ_RUL = (current_pct - critical_pct) / r² · σ_r
            = RUL · uncertainty_factor          (simplified)

  Confidence interval:
      RUL_lower = max(0,   RUL − σ_RUL)
      RUL_upper = min(MAX, RUL + σ_RUL)

EDGE CASES
  ┌──────────────────────────────────────┬───────────────────────────────┐
  │ Condition                            │ Result                        │
  ├──────────────────────────────────────┼───────────────────────────────┤
  │ current_pct ≤ critical_pct           │ (0, 0, 0) — already failed    │
  │ |rate| < min_rate (stable)           │ (MAX, *, MAX) — healthy       │
  │ rate > 0 (improving)                 │ (MAX, *, MAX) — recovering    │
  └──────────────────────────────────────┴───────────────────────────────┘

EXAMPLES
  Refrigerant at 87%, rate = -2.0 pct/tu, critical at 50%:
      RUL = (87 - 50) / 2.0 = 18.5 time-units

  Fan at 45%, rate = -3.5 pct/tu, critical at 10%:
      RUL = (45 - 10) / 3.5 = 10.0 time-units

  Compressor at 22%, rate = -0.1 pct/tu, critical at 20%:
      RUL = (22 - 20) / 0.1 = 20.0 time-units
"""

import logging
from typing import Optional

log = logging.getLogger("thermo-twin.rul")

MAX_RUL = 9999.0    # sentinel for "healthy / stable" display


class RULEngine:
    """
    Stateless RUL calculator.  Thread-safe — holds no mutable state.

    Args:
        critical_thresholds:  pct values at which component is considered failed.
            Default: {"refrigerant": 50, "compressor": 20, "fan": 10}
        min_rate_pct_per_tu:  rates with |rate| below this are treated as zero
            (stable component, RUL = MAX_RUL).  Default: 0.005 pct/time-unit.
    """

    _DEFAULTS = {
        "refrigerant": 50.0,
        "compressor":  20.0,
        "fan":         10.0,
    }

    def __init__(
        self,
        critical_thresholds: Optional[dict] = None,
        min_rate_pct_per_tu: float = 0.005,
    ):
        self.thresholds = dict(critical_thresholds or self._DEFAULTS)
        self._min_rate  = min_rate_pct_per_tu

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        current_health,
        daily_rates: dict,
        uncertainty_pct: float = 15.0,
    ) -> dict:
        """
        Compute RUL for every component.

        Args:
            current_health:  ComponentHealth (or dict with same attribute names)
            daily_rates:     output of DegradationTrajectoryModel.predict_rate()
                             keys: refrigerant_pct_per_tu, compressor_pct_per_tu,
                                   fan_pct_per_tu
            uncertainty_pct: fractional uncertainty on the rate (default 15 %)

        Returns:
            {
              "refrigerant_rul_days":       float,
              "refrigerant_rul_days_lower": float,
              "refrigerant_rul_days_upper": float,
              "compressor_rul_days":        float,
              "compressor_rul_days_lower":  float,
              "compressor_rul_days_upper":  float,
              "fan_rul_days":               float,
              "fan_rul_days_lower":         float,
              "fan_rul_days_upper":         float,
              "most_critical_component":    str,
              "days_to_any_failure":        float,
            }
            (The "_days" suffix is legacy naming; units match time_unit_seconds
            of the trajectory model — could be seconds, hours, or days.)
        """
        uf = uncertainty_pct / 100.0

        r_pct = _attr(current_health, "refrigerant_charge_pct",    100.0)
        c_pct = _attr(current_health, "compressor_efficiency_pct", 100.0)
        f_pct = _attr(current_health, "fan_health_pct",            100.0)

        r_rate = daily_rates.get("refrigerant_pct_per_tu", 0.0)
        c_rate = daily_rates.get("compressor_pct_per_tu",  0.0)
        f_rate = daily_rates.get("fan_pct_per_tu",         0.0)

        r_rul, r_lo, r_hi = self._calculate_rul(r_pct, self.thresholds["refrigerant"], r_rate, uf)
        c_rul, c_lo, c_hi = self._calculate_rul(c_pct, self.thresholds["compressor"],  c_rate, uf)
        f_rul, f_lo, f_hi = self._calculate_rul(f_pct, self.thresholds["fan"],         f_rate, uf)

        components = {"refrigerant": r_rul, "compressor": c_rul, "fan": f_rul}
        most_critical   = min(components, key=lambda k: components[k])
        days_to_any     = min(components.values())

        return {
            "refrigerant_rul_days":       r_rul,
            "refrigerant_rul_days_lower": r_lo,
            "refrigerant_rul_days_upper": r_hi,
            "compressor_rul_days":        c_rul,
            "compressor_rul_days_lower":  c_lo,
            "compressor_rul_days_upper":  c_hi,
            "fan_rul_days":               f_rul,
            "fan_rul_days_lower":         f_lo,
            "fan_rul_days_upper":         f_hi,
            "most_critical_component":    most_critical,
            "days_to_any_failure":        days_to_any,
        }

    # ── Core calculation ────────────────────────────────────────────────────────

    def _calculate_rul(
        self,
        current_pct: float,
        critical_pct: float,
        rate_pct_per_tu: float,
        uncertainty_factor: float,
    ) -> tuple:
        """
        Compute (rul_central, rul_lower, rul_upper) for one component.

        uncertainty_factor = σ_rate / |rate|  (fractional, e.g. 0.15)
        σ_RUL = RUL * uncertainty_factor  (propagated at first order)
        """
        # ── Already failed ──
        if current_pct <= critical_pct:
            return (0.0, 0.0, 0.0)

        margin = current_pct - critical_pct

        # ── Stable or improving (rate near zero or positive) ──
        if rate_pct_per_tu >= -self._min_rate:
            hi = min(MAX_RUL, margin / self._min_rate) if self._min_rate > 0 else MAX_RUL
            return (MAX_RUL, max(0.0, hi * (1.0 - uncertainty_factor)), MAX_RUL)

        # ── Degrading ──
        rate_abs = abs(rate_pct_per_tu)
        rul      = margin / rate_abs
        sigma    = rul * uncertainty_factor

        return (
            round(min(rul,         MAX_RUL), 2),
            round(max(0.0, rul - sigma),     2),
            round(min(rul + sigma, MAX_RUL), 2),
        )


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _attr(obj, name: str, default: float) -> float:
    """Get attribute from dataclass or dict, falling back to default."""
    if hasattr(obj, name):
        return float(getattr(obj, name))
    if isinstance(obj, dict):
        return float(obj.get(name, default))
    return default
