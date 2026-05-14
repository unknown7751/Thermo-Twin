"""
Pre-compute SHAP explanations + MC-Dropout uncertainty for the three Thermo-Twin demo scenarios.

Selects a representative fault window for each scenario, runs SHAP DeepExplainer
and MC-Dropout (10 passes), and saves results to demo_explanations.json so the
dashboard loads instantly with full uncertainty metadata.

Run:
    python explainability/precompute_explanations.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from explainability.shap_explainer import SHAPExplainer
from model.autoencoder import load_autoencoder
from model.threshold import load_threshold_config, severity_score, MCDropoutInference

CHECKPOINT_DIR = ROOT / "model" / "checkpoints"
PROCESSED_DIR  = ROOT / "data" / "processed"
EXPL_DIR       = ROOT / "explainability"

# Scenario key -> fault_label in test set
SCENARIOS = {
    "scenario_1_refrigerant_leak": "refrigerant_leak",
    "scenario_2_fan_failure":      "fan_failure",
    "scenario_3_compressor_wear":  "compressor_wear",
}

STREAM_KEYS = [
    "compressor_power_pct",
    "discharge_pressure_pct",
    "fan_rpm_pct",
    "supply_air_temp_pct",
]


def _representative_window(X: np.ndarray, errors: np.ndarray, percentile: float = 80.0):
    """Return window whose reconstruction error is closest to the given percentile."""
    target = np.percentile(errors, percentile)
    idx    = int(np.argmin(np.abs(errors - target)))
    return X[idx], float(errors[idx])


def main():
    print("=" * 60)
    print("  Thermo-Twin: Pre-computing SHAP + MC-Dropout Demo Explanations")
    print("=" * 60)

    # -- Load data -------------------------------------------------------------
    train          = np.load(PROCESSED_DIR / "train_windows.npz")
    test           = np.load(PROCESSED_DIR / "test_windows.npz")
    X_train        = train["X"]
    X_test, y_test = test["X"], test["y"]

    cfg       = load_threshold_config(CHECKPOINT_DIR / "threshold_config.json")
    threshold = cfg["threshold"]
    p99       = cfg["p99_anomaly"]

    # -- Reconstruction errors for all test windows ----------------------------
    ae = load_autoencoder(str(CHECKPOINT_DIR / "autoencoder.pt"))
    with torch.no_grad():
        X_t     = torch.FloatTensor(X_test)
        all_err = torch.mean((X_t - ae(X_t)) ** 2, dim=1).numpy()

    # -- Initialize SHAP DeepExplainer -----------------------------------------
    print("\n[1] Initializing SHAP DeepExplainer (200 background windows)")
    explainer = SHAPExplainer(
        checkpoint_path = CHECKPOINT_DIR / "autoencoder.pt",
        background_data = X_train,
        n_background    = 200,
    )
    print("    Ready.")

    # -- Initialize MC-Dropout -------------------------------------------------
    mc_infer = MCDropoutInference(ae, n_passes=10)
    print("[2] MC-Dropout ready (10 passes per window)\n")

    # -- Per-scenario explanations ---------------------------------------------
    results = {}

    for key, label in SCENARIOS.items():
        mask = y_test == label
        n    = int(mask.sum())
        print(f"[3] {key}  (label='{label}', n={n} windows)")

        if n == 0:
            print(f"    WARNING: no '{label}' windows in test set -- skipping\n")
            continue

        window, err = _representative_window(
            X_test[mask], all_err[mask], percentile=80.0
        )

        # MC-Dropout uncertainty (must run before SHAP since it toggles train/eval)
        window_t  = torch.FloatTensor(window[np.newaxis, :])
        mc_result = mc_infer.infer(window_t, threshold, p99)

        # SHAP attribution
        print(f"    Computing SHAP values...")
        expl = explainer.explain(window)
        expl["severity_score"]      = mc_result["mean_severity"]
        expl["uncertainty"]         = mc_result["uncertainty"]
        expl["confidence_pct"]      = mc_result["confidence_pct"]
        expl["action_override"]     = mc_result["action_override"]
        expl["per_pass_severities"] = mc_result["per_pass_severities"]

        results[key] = expl

        print(f"    compressor_power : {expl['compressor_power_pct']}%")
        print(f"    discharge_pres   : {expl['discharge_pressure_pct']}%")
        print(f"    fan_rpm          : {expl['fan_rpm_pct']}%")
        print(f"    supply_air_temp  : {expl['supply_air_temp_pct']}%")
        print(f"    severity         : {expl['severity_score']} +/- {expl['uncertainty']}  ({expl['confidence_pct']}% confidence)")
        print(f"    fault_type       : {expl['fault_type']}")
        print(f"    action_override  : {expl['action_override']}")
        print(f"    summary          : {expl['summary']}\n")

    if not results:
        print("ERROR: no scenarios could be explained -- check test data labels.")
        sys.exit(1)

    # -- Verification ----------------------------------------------------------
    print("[4] Verification")
    all_pass = True
    for key, expl in results.items():
        total  = sum(expl[k] for k in STREAM_KEYS)
        ok     = abs(total - 100.0) < 0.5
        status = "PASS" if ok else "FAIL"
        print(f"    {key}: {total:.1f}%  [{status}]")
        if not ok:
            all_pass = False

    if not all_pass:
        print("\nWARNING: some percentages do not sum to 100 -- check _attribution()")

    # -- Save ------------------------------------------------------------------
    out_path = EXPL_DIR / "demo_explanations.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[5] Saved -> explainability/demo_explanations.json")
    print(f"    Scenarios saved: {list(results.keys())}")
    print(f"    Load latency:    < 1 second (pre-computed)")

    print("\n" + "=" * 60)
    print("  Done -- explanations + MC-Dropout uncertainty ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()