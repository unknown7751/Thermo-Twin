"""
Step 4 -- Evaluate Synthetic-Trained Model on LBNL Real Faults
(Sim-to-Real Transfer Validation)

Uses the EXISTING synthetic-trained autoencoder and threshold to detect
real building faults from the LBNL dataset. No retraining is performed --
this proves the model generalizes from synthetic to real data.

Run: python lbnl_validation/04_evaluate.py
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)

ROOT           = Path(__file__).parent.parent
PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "model" / "checkpoints"
sys.path.insert(0, str(ROOT))

from model.autoencoder import load_autoencoder
from model.threshold   import severity_score, load_threshold_config


def main():
    print("\n" + "=" * 60)
    print("  LBNL FAULT DETECTION -- SIM-TO-REAL TRANSFER EVALUATION")
    print("=" * 60)

    # Load SYNTHETIC-trained model and threshold
    model = load_autoencoder(CHECKPOINT_DIR / "autoencoder.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    cfg = load_threshold_config(CHECKPOINT_DIR / "threshold_config.json")
    threshold   = cfg["threshold"]
    p99_anomaly = cfg.get("p99_anomaly", threshold * 50)

    print(f"\n  Model: Synthetic-trained autoencoder")
    print(f"  Threshold: {threshold:.6f}  (from synthetic val calibration)")

    # -- Part A: Fault-only analysis (LBNL faults) --
    print(f"\n{'-' * 60}")
    print("  Part A: LBNL Fault Windows Only")
    print(f"{'-' * 60}")

    fault_data = np.load(PROCESSED_DIR / "lbnl_fault_windows.npz")
    X_fault = torch.FloatTensor(fault_data["X"]).to(device)

    print(f"  Fault windows: {len(X_fault)}")

    with torch.no_grad():
        recon = model(X_fault)
        fault_errors = torch.mean((recon - X_fault) ** 2, dim=1).cpu().numpy()

    fault_detected = (fault_errors > threshold).sum()
    fault_scores = severity_score(fault_errors, threshold, p99_error=p99_anomaly)

    print(f"\n  Reconstruction Errors (LBNL faults):")
    print(f"    Mean  : {np.mean(fault_errors):.6f}")
    print(f"    Std   : {np.std(fault_errors):.6f}")
    print(f"    Min   : {np.min(fault_errors):.6f}")
    print(f"    Max   : {np.max(fault_errors):.6f}")
    print(f"    Median: {np.median(fault_errors):.6f}")

    print(f"\n  Fault Detection (above threshold):")
    print(f"    Detected: {fault_detected} / {len(X_fault)} ({fault_detected/len(X_fault)*100:.1f}%)")

    print(f"\n  Severity Scores (LBNL faults):")
    print(f"    Mean      : {np.mean(fault_scores):.1f}")
    print(f"    >=70      : {(fault_scores >= 70).sum()} ({(fault_scores >= 70).mean()*100:.1f}%)")
    print(f"    >=50      : {(fault_scores >= 50).sum()} ({(fault_scores >= 50).mean()*100:.1f}%)")
    print(f"    <=40      : {(fault_scores <= 40).sum()} ({(fault_scores <= 40).mean()*100:.1f}%)")

    # -- Part B: Combined test (synthetic normals + LBNL faults) --
    print(f"\n{'-' * 60}")
    print("  Part B: Combined Test (Synthetic Normal + LBNL Fault)")
    print(f"{'-' * 60}")

    combined = np.load(PROCESSED_DIR / "lbnl_combined_test.npz")
    X_test = torch.FloatTensor(combined["X"]).to(device)
    y_test = combined["y"]
    y_bin = (y_test != "normal").astype(int)

    print(f"  Test samples: {len(X_test)}")
    print(f"    Normal (synthetic): {(y_bin == 0).sum()}")
    print(f"    Fault  (LBNL real): {(y_bin == 1).sum()}")

    with torch.no_grad():
        recon = model(X_test)
        all_errors = torch.mean((recon - X_test) ** 2, dim=1).cpu().numpy()

    predictions = (all_errors > threshold).astype(int)
    all_scores = severity_score(all_errors, threshold, p99_error=p99_anomaly)

    # Binary metrics
    precision = precision_score(y_bin, predictions, zero_division=0)
    recall    = recall_score(y_bin, predictions, zero_division=0)
    f1        = f1_score(y_bin, predictions, zero_division=0)
    avg_prec  = average_precision_score(y_bin, all_errors)

    if len(np.unique(y_bin)) > 1:
        roc_auc = roc_auc_score(y_bin, all_errors)
    else:
        roc_auc = float("nan")

    cm = confusion_matrix(y_bin, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    print(f"\n  Detection Metrics:")
    print(f"    Precision        : {precision:.4f}")
    print(f"    Recall           : {recall:.4f}")
    print(f"    F1 Score         : {f1:.4f}")
    print(f"    ROC-AUC          : {roc_auc:.4f}")
    print(f"    Avg Precision    : {avg_prec:.4f}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':20s}  Pred Normal  Pred Fault")
    print(f"    {'True Normal':20s}  {tn:11d}  {fp:10d}")
    print(f"    {'True Fault':20s}  {fn:11d}  {tp:10d}")

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    print(f"    False Positive Rate: {fpr:.4f}")
    print(f"    False Negative Rate: {fnr:.4f}")

    # Severity breakdown
    normal_scores = all_scores[y_bin == 0]
    fault_scores_combined = all_scores[y_bin == 1]

    print(f"\n  Severity Scores:")
    print(f"    Normal (synthetic)  mean: {np.mean(normal_scores):.1f}  <=40: {(normal_scores <= 40).mean()*100:.1f}%")
    print(f"    Fault  (LBNL real)  mean: {np.mean(fault_scores_combined):.1f}  >=70: {(fault_scores_combined >= 70).mean()*100:.1f}%")

    print("\n" + "=" * 60)

    # Save evaluation results
    results = {
        "evaluation_type":     "Sim-to-Real Transfer (Synthetic Model -> LBNL Faults)",
        "dataset":             "LBNL Real Building Data (RTU)",
        "model":               "Synthetic-trained Autoencoder",
        "test_samples":        int(len(X_test)),
        "test_normal":         int((y_bin == 0).sum()),
        "test_fault":          int((y_bin == 1).sum()),
        "threshold":           float(threshold),
        "precision":           round(float(precision), 4),
        "recall":              round(float(recall), 4),
        "f1_score":            round(float(f1), 4),
        "roc_auc":             round(float(roc_auc), 4),
        "avg_precision":       round(float(avg_prec), 4),
        "normal_below_40_pct": round(float((normal_scores <= 40).mean() * 100), 1),
        "fault_above_70_pct":  round(float((fault_scores_combined >= 70).mean() * 100), 1),
        "fault_above_50_pct":  round(float((fault_scores_combined >= 50).mean() * 100), 1),
        "mean_severity_normal": round(float(np.mean(normal_scores)), 1),
        "mean_severity_fault":  round(float(np.mean(fault_scores_combined)), 1),
        "fault_detection_rate": round(float(fault_detected / len(fault_data["X"]) * 100), 1),
        "confusion_matrix": {
            "true_neg":  int(tn),
            "false_pos": int(fp),
            "false_neg": int(fn),
            "true_pos":  int(tp),
        },
    }

    results_path = CHECKPOINT_DIR / "lbnl_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  [OK] Results saved -> {results_path}")

    print("\n" + "=" * 60)
    print("  Step 4 complete. LBNL Evaluation done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
