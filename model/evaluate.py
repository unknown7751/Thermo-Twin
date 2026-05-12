"""
Model evaluation metrics for both the Autoencoder and Isolation Forest.
Run: python model/evaluate.py
"""

import sys, json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.autoencoder      import load_autoencoder
from model.isolation_forest import load_isolation_forest, anomaly_scores
from model.threshold        import severity_score, load_threshold_config

# --- Load --------------------------------------------------------------------
test   = np.load(ROOT / "data/processed/test_windows.npz")
X_test = torch.FloatTensor(test["X"])
y_test = test["y"]                          # string labels
y_bin  = (y_test != "normal").astype(int)   # 1=anomaly, 0=normal

ae  = load_autoencoder(ROOT / "model/checkpoints/autoencoder.pt")
clf = load_isolation_forest(ROOT / "model/checkpoints/isolation_forest.pkl")
cfg = load_threshold_config(ROOT / "model/checkpoints/threshold_config.json")

threshold   = cfg["threshold"]
p99_anomaly = cfg["p99_anomaly"]

# --- Scores ------------------------------------------------------------------
ae_errors  = ae.reconstruction_errors(X_test).numpy()
ae_scores  = severity_score(ae_errors, threshold, p99_error=p99_anomaly)
ae_pred    = (ae_errors > threshold).astype(int)

if_raw     = anomaly_scores(clf, test["X"])
if_thresh  = cfg["if_threshold"]
if_pred    = (if_raw > if_thresh).astype(int)

# --- Helpers -----------------------------------------------------------------
SEP = "-" * 58

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def binary_metrics(name, y_true, y_pred, y_score):
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_score)
    ap   = average_precision_score(y_true, y_score)
    print(f"\n  [{name}]")
    print(f"    Precision        : {prec:.4f}")
    print(f"    Recall           : {rec:.4f}")
    print(f"    F1 Score         : {f1:.4f}")
    print(f"    ROC-AUC          : {auc:.4f}")
    print(f"    Avg Precision    : {ap:.4f}")
    return dict(precision=prec, recall=rec, f1=f1, roc_auc=auc, avg_precision=ap)

def conf_matrix(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  [{name}] Confusion Matrix (binary: normal=0, anomaly=1)")
    print(f"    {'':20s}  Pred Normal  Pred Anomaly")
    print(f"    {'True Normal':20s}  {tn:11d}  {fp:12d}")
    print(f"    {'True Anomaly':20s}  {fn:11d}  {tp:12d}")
    fpr = fp / (fp + tn) if (fp + tn) else 0
    print(f"    False Positive Rate : {fpr:.4f}  ({fp} normal windows flagged as anomaly)")
    print(f"    False Negative Rate : {fn/(fn+tp):.4f}  ({fn} anomaly windows missed)")

# --- 1. Binary metrics -------------------------------------------------------
section("Binary Classification  (normal vs anomaly)")
ae_bin_metrics = binary_metrics("Autoencoder",      y_bin, ae_pred, ae_errors)
if_bin_metrics = binary_metrics("Isolation Forest", y_bin, if_pred, if_raw)

conf_matrix("Autoencoder",      y_bin, ae_pred)
conf_matrix("Isolation Forest", y_bin, if_pred)

# --- 2. Per-anomaly-type detection rate --------------------------------------
section("Per-Anomaly-Type Detection Rate  (autoencoder)")
print(f"\n  {'Label':15s}  {'n':>4}  {'MSE mean':>9}  {'MSE std':>8}  "
      f"{'Sev mean':>9}  {'% above threshold':>18}  {'% score>=70':>11}")
print("  " + "-" * 80)

for label in ["normal", "refrigerant_leak", "fan_failure", "compressor_wear"]:
    mask = y_test == label
    if not mask.any():
        continue
    e = ae_errors[mask]
    s = ae_scores[mask]
    above_thr = (e > threshold).mean() * 100
    above_70  = (s >= 70).mean() * 100
    print(f"  {label:15s}  {mask.sum():>4}  {e.mean():>9.4f}  {e.std():>8.4f}  "
          f"{s.mean():>9.1f}  {above_thr:>18.1f}%  {above_70:>11.1f}%")

# --- 3. Multi-class report ---------------------------------------------------
section("Multi-Class Report  (4-way: normal / refrigerant_leak / fan_failure / compressor_wear)")

# Autoencoder: assign predicted label by severity ranges
def ae_multiclass(errors, scores, threshold):
    """Predict the label for each window based on severity bucket."""
    preds = np.where(errors <= threshold, "normal", "anomaly")
    # We can't distinguish anomaly type without labels — report binary only
    return preds

print("\n  Note: autoencoders are unsupervised — they output a score, not a class.")
print("  The 4-way classification report below treats any window above the")
print("  threshold as 'anomaly' and evaluates against the true label.\n")

ae_label_pred = np.where(ae_pred == 1, "anomaly", "normal")
y_4way = np.where(y_bin == 0, "normal", y_test)  # keep specific anomaly type in truth

labels_report = ["normal", "refrigerant_leak", "fan_failure", "compressor_wear"]
y_true_4 = y_test
y_pred_4 = np.where(ae_pred == 1, y_test, "normal")   # give credit when correctly flagged

print("  Autoencoder (binary flag -> true type if flagged):")
print("  " + classification_report(
    y_true_4, y_pred_4, labels=labels_report, zero_division=0, digits=3
).replace("\n", "\n  "))

# --- 4. Isolation Forest multi-class -----------------------------------------
print("  Isolation Forest (binary flag -> true type if flagged):")
y_pred_if4 = np.where(if_pred == 1, y_test, "normal")
print("  " + classification_report(
    y_test, y_pred_if4, labels=labels_report, zero_division=0, digits=3
).replace("\n", "\n  "))

# --- 5. Severity score stats -------------------------------------------------
section("Severity Score Statistics  (autoencoder)")
print(f"\n  {'Label':15s}  {'min':>5}  {'p25':>5}  {'median':>7}  {'p75':>5}  {'max':>5}  {'mean':>6}")
print("  " + "-" * 60)
for label in ["normal", "refrigerant_leak", "fan_failure", "compressor_wear"]:
    mask = y_test == label
    if not mask.any():
        continue
    s = ae_scores[mask]
    print(f"  {label:15s}  {s.min():>5}  {np.percentile(s,25):>5.0f}  "
          f"{np.median(s):>7.0f}  {np.percentile(s,75):>5.0f}  {s.max():>5}  {s.mean():>6.1f}")

# --- 6. Summary --------------------------------------------------------------
section("Summary")
print(f"""
  Autoencoder
    Threshold        : {threshold:.6f}  (val mean + 2.5sigma)
    ROC-AUC          : {ae_bin_metrics['roc_auc']:.4f}
    F1 (binary)      : {ae_bin_metrics['f1']:.4f}
    Normal <=40      : {(ae_scores[y_test=='normal'] <= 40).mean()*100:.1f}%
    Anomaly mean sev : refrigerant_leak={ae_scores[y_test=='refrigerant_leak'].mean():.0f}  fan_failure={ae_scores[y_test=='fan_failure'].mean():.0f}  compressor_wear={ae_scores[y_test=='compressor_wear'].mean():.0f}

  Isolation Forest
    Threshold        : {if_thresh:.6f}
    ROC-AUC          : {if_bin_metrics['roc_auc']:.4f}
    F1 (binary)      : {if_bin_metrics['f1']:.4f}
""")



