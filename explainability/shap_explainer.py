"""
SHAP-based explainability for the Thermo-Twin anomaly detection autoencoder.

Decomposes reconstruction error into per-stream attributions across 4 sensors:
  indices   0-49  -> compressor_power_kw
  indices  50-99  -> discharge_pressure_psi
  indices 100-149 -> fan_rpm
  indices 150-199 -> supply_air_temp_c

Uses SHAP GradientExplainer (expected gradients) with 200 normal background
windows and a fixed seed for deterministic, reproducible SHAP values.
Prescriptive rules are applied on top of the attribution to identify fault type
and dispatch recommendation.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.autoencoder import load_autoencoder

# -- Stream index slices -------------------------------------------------------
COMP_IDX  = slice(0,   50)   # compressor_power_kw
PRES_IDX  = slice(50,  100)  # discharge_pressure_psi
FAN_IDX   = slice(100, 150)  # fan_rpm
TEMP_IDX  = slice(150, 200)  # supply_air_temp_c


class _MSEWrapper(nn.Module):
    """
    Wraps the autoencoder to output a per-sample scalar MSE so SHAP values
    represent each feature's contribution to the anomaly score.
    Output shape: (N, 1).
    """

    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.autoencoder(x)
        return torch.mean((x - recon) ** 2, dim=1, keepdim=True)


class SHAPExplainer:
    """
    SHAP GradientExplainer wrapper for the Thermo-Twin HVAC autoencoder.

    Uses 200 normal background windows (fixed seed=42) for deterministic,
    reproducible SHAP values across runs. Decomposes each window's
    reconstruction error into contributions from the 4 sensor streams.

    Usage:
        explainer = SHAPExplainer(checkpoint_path, X_train_normal)
        explanation = explainer.explain(window)
        # {
        #   "compressor_power_pct": 8.0,
        #   "discharge_pressure_pct": 51.0,
        #   "fan_rpm_pct": 6.0,
        #   "supply_air_temp_pct": 35.0,
        #   "summary": "Anomaly driven by Pressure Drop (51%) and Temp Rise (35%)",
        #   "fault_type": "Refrigerant Leak",
        #   "prescription": {...}
        # }
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        background_data: np.ndarray,
        n_background: int = 200,
    ):
        import shap

        self._autoencoder = load_autoencoder(str(checkpoint_path))
        self._autoencoder.eval()

        self._wrapper = _MSEWrapper(self._autoencoder)
        self._wrapper.eval()

        rng = np.random.default_rng(seed=42)
        n   = min(n_background, len(background_data))
        idx = rng.choice(len(background_data), size=n, replace=False)
        bg  = torch.FloatTensor(background_data[idx])

        self._explainer = shap.GradientExplainer(self._wrapper, bg)

    # -- Public API ------------------------------------------------------------

    def explain(self, window: np.ndarray) -> dict:
        """
        Explain a single sensor window.

        Args:
            window: shape (200,) or (1, 200) -- scaled sensor window

        Returns:
            dict with per-stream percentages, summary, fault_type, prescription
        """
        if window.ndim == 1:
            window = window[np.newaxis, :]
        sv = self._raw_shap(window)[0]   # (200,)
        return self._attribution(sv)

    def batch_explain(self, windows: np.ndarray) -> list:
        """Explain a batch of windows. Returns list of explanation dicts."""
        sv_all = self._raw_shap(windows)  # (N, 200)
        return [self._attribution(sv) for sv in sv_all]

    # -- Internal helpers ------------------------------------------------------

    def _raw_shap(self, windows: np.ndarray) -> np.ndarray:
        """Run GradientExplainer and return SHAP values, shape (N, 200)."""
        x   = torch.FloatTensor(windows)
        raw = self._explainer.shap_values(x)

        values = raw[0] if isinstance(raw, list) else raw
        if isinstance(values, np.ndarray) and values.ndim == 3:
            values = values.squeeze(2)

        return np.asarray(values, dtype=np.float32)  # (N, 200)

    def _attribution(self, shap_values: np.ndarray) -> dict:
        """
        Convert a 200-dim SHAP vector into a human-readable explanation.
        Percentages across 4 streams sum to exactly 100%.
        Prescriptive rules are applied to identify the most likely fault.
        """
        abs_sv = np.abs(shap_values)
        comp_abs = float(abs_sv[COMP_IDX].sum())
        pres_abs = float(abs_sv[PRES_IDX].sum())
        fan_abs  = float(abs_sv[FAN_IDX].sum())
        temp_abs = float(abs_sv[TEMP_IDX].sum())
        total    = comp_abs + pres_abs + fan_abs + temp_abs

        if total < 1e-9:
            comp_pct = pres_pct = fan_pct = temp_pct = 25.0
        else:
            comp_pct = round(100.0 * comp_abs / total, 1)
            pres_pct = round(100.0 * pres_abs / total, 1)
            fan_pct  = round(100.0 * fan_abs  / total, 1)
            # Guarantee exact 100% sum on the last stream
            temp_pct = round(100.0 - comp_pct - pres_pct - fan_pct, 1)

        fault_type, prescription = _prescriptive_rules(comp_pct, pres_pct, fan_pct, temp_pct)
        summary = _build_summary(comp_pct, pres_pct, fan_pct, temp_pct)

        return {
            "compressor_power_pct":     comp_pct,
            "discharge_pressure_pct":   pres_pct,
            "fan_rpm_pct":              fan_pct,
            "supply_air_temp_pct":      temp_pct,
            "summary":                  summary,
            "fault_type":               fault_type,
            "prescription":             prescription,
        }


# -- Prescriptive rules --------------------------------------------------------

def _prescriptive_rules(comp_pct: float, pres_pct: float,
                        fan_pct: float, temp_pct: float) -> tuple:
    """
    Map SHAP attribution percentages to a fault type and maintenance prescription.
    Rules calibrated against observed SHAP patterns -- evaluated in priority order.

    Observed SHAP fingerprints (from training data):
      Refrigerant Leak  -> supply_air_temp dominant (gas escapes, unit stops cooling)
      Fan Failure       -> fan_rpm dominant, compressor secondary
      Compressor Wear   -> compressor_power dominant, temp secondary
    """
    # Refrigerant Leak: temp rise is the dominant thermodynamic signal
    if temp_pct > 50:
        return (
            "Refrigerant Leak",
            {
                "fault":  "Refrigerant Leak in Evaporator Coil",
                "impact": "Cooling efficiency down ~40%, unit running but not cooling",
                "action": "Dispatch technician with refrigerant recharge kit and leak detector",
            },
        )

    # Condenser Fan Failure: fan RPM drop is the lead signal
    if fan_pct > 35:
        return (
            "Condenser Fan Failure",
            {
                "fault":  "Condenser Fan Motor Degradation or Failure",
                "impact": "Heat dissipation failure, compressor overload risk",
                "action": "Dispatch with 5HP Fan Motor replacement",
            },
        )

    # Compressor Wear: compressor power creep is the lead signal
    if comp_pct > 45:
        return (
            "Compressor Wear",
            {
                "fault":  "Progressive Compressor Mechanical Wear",
                "impact": "Progressive efficiency loss, full failure imminent",
                "action": "Schedule compressor replacement within 2 weeks",
            },
        )

    # Fallback: flag the dominant stream
    dominant = max(
        [("compressor_power", comp_pct),
         ("discharge_pressure", pres_pct),
         ("fan_rpm", fan_pct),
         ("supply_air_temp", temp_pct)],
        key=lambda x: x[1],
    )
    return (
        "Unknown Fault",
        {
            "fault":  f"Unclassified anomaly -- dominant signal: {dominant[0]}",
            "impact": "Unknown -- manual inspection required",
            "action": "Dispatch technician for on-site diagnostics",
        },
    )


def _build_summary(comp_pct: float, pres_pct: float,
                   fan_pct: float, temp_pct: float) -> str:
    """Build a one-line human-readable summary of the top two contributing streams."""
    streams = sorted(
        [
            ("Compressor Power", comp_pct),
            ("Pressure Drop",    pres_pct),
            ("Fan RPM",          fan_pct),
            ("Temp Rise",        temp_pct),
        ],
        key=lambda x: x[1], reverse=True,
    )
    top, second = streams[0], streams[1]
    if top[1] >= 70:
        return f"Anomaly driven primarily by {top[0]} ({top[1]:.0f}%)"
    return (
        f"Anomaly driven by {top[0]} ({top[1]:.0f}%) "
        f"and {second[0]} ({second[1]:.0f}%)"
    )