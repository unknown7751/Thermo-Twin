"""
Phase 3B — Autoencoder training, threshold calibration, and critical test.
Run: python model/train.py
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT           = Path(__file__).parent.parent
PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "model" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from model.autoencoder      import Autoencoder
from model.threshold        import compute_threshold, severity_score, save_threshold_config
from model.isolation_forest import (
    train_isolation_forest, anomaly_scores,
    save_isolation_forest, load_isolation_forest,
)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
INPUT_DIM    = 200
BOTTLENECK   = 8
EPOCHS       = 600
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NOISE_STD    = 0.02
PATIENCE     = 80
N_SIGMA      = 2.5

COLORS = {
    "normal":           "#4CAF50",
    "refrigerant_leak": "#FF5722",
    "fan_failure":      "#9C27B0",
    "compressor_wear":  "#FF9800",
}


# ─── Data ────────────────────────────────────────────────────────────────────

def load_data():
    train = np.load(PROCESSED_DIR / "train_windows.npz")
    val   = np.load(PROCESSED_DIR / "val_windows.npz")
    test  = np.load(PROCESSED_DIR / "test_windows.npz")
    return (
        torch.FloatTensor(train["X"]),
        torch.FloatTensor(val["X"]),
        torch.FloatTensor(test["X"]),
        test["y"],
    )


# ─── Training loop ───────────────────────────────────────────────────────────

def train_autoencoder(model, X_train, X_val):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=25, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )

    best_val   = float("inf")
    best_state = None
    patience   = 0
    train_hist, val_hist = [], []

    print(f"  Windows — train: {len(X_train)}  val: {len(X_val)}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs  : {EPOCHS}  batch: {BATCH_SIZE}  lr: {LR}  noise: {NOISE_STD}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            noisy = batch + torch.randn_like(batch) * NOISE_STD  # denoising
            loss  = criterion(model(noisy), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()

        scheduler.step(val_loss)
        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | train={train_loss:.6f}  val={val_loss:.6f}"
                  f"  best={best_val:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience >= PATIENCE:
            print(f"  Early stop at epoch {epoch}  (best val={best_val:.6f})")
            break

    model.load_state_dict(best_state)
    return model, train_hist, val_hist, best_val


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def get_errors(model, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.mean((X - model(X)) ** 2, dim=1).numpy()


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _style(ax, title):
    ax.set_facecolor("#161B22")
    ax.set_title(title, color="#E6EDF3", fontsize=9, pad=6)
    ax.tick_params(colors="#8B949E", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#30363D")
    ax.grid(True, color="#21262D", linewidth=0.4, linestyle="--")


def plot_results(train_hist, val_hist, ae_errors, if_scores, y_test, threshold, if_threshold):
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    # ── 1. Training curves ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_hist, color="#4CAF50", linewidth=1.0, label="Train MSE")
    ax1.plot(val_hist,   color="#FF9800", linewidth=1.0, label="Val MSE")
    ax1.axhline(0.1, color="#FF5722", linewidth=0.8, linestyle="--", label="Target 0.1")
    ax1.set_xlabel("Epoch", color="#8B949E", fontsize=8)
    ax1.set_ylabel("MSE", color="#8B949E", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
    _style(ax1, "Autoencoder Training Curves")

    # ── 2. AE reconstruction error histogram ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for label, color in COLORS.items():
        mask = y_test == label
        if mask.any():
            ax2.hist(ae_errors[mask], bins=25, alpha=0.75, color=color,
                     label=f"{label} (n={mask.sum()})", edgecolor="none")
    ax2.axvline(threshold, color="white", linewidth=1.5, linestyle="--",
                label=f"Threshold={threshold:.4f}")
    ax2.set_xlabel("Reconstruction MSE", color="#8B949E", fontsize=8)
    ax2.set_ylabel("Count", color="#8B949E", fontsize=8)
    ax2.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
    _style(ax2, "Autoencoder — Error Distribution (Critical Test)")

    # ── 3. Severity score distribution ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    p99 = float(np.percentile(ae_errors[y_test != "normal"], 90)) if any(y_test != "normal") else threshold * 50
    scores = severity_score(ae_errors, threshold, p99_error=p99)
    for label, color in COLORS.items():
        mask = y_test == label
        if mask.any():
            ax3.hist(scores[mask], bins=20, alpha=0.75, color=color,
                     label=label, edgecolor="none", range=(0, 100))
    ax3.axvline(40, color="#8B949E", linewidth=1.0, linestyle=":", label="<=40 normal")
    ax3.axvline(70, color="#FF5722", linewidth=1.0, linestyle=":", label=">=70 critical")
    ax3.set_xlabel("Severity Score (0-100)", color="#8B949E", fontsize=8)
    ax3.set_ylabel("Count", color="#8B949E", fontsize=8)
    ax3.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
    _style(ax3, "Severity Score Distribution")

    # ── 4. Isolation Forest scores ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for label, color in COLORS.items():
        mask = y_test == label
        if mask.any():
            ax4.hist(if_scores[mask], bins=25, alpha=0.75, color=color,
                     label=f"{label} (n={mask.sum()})", edgecolor="none")
    ax4.axvline(if_threshold, color="white", linewidth=1.5, linestyle="--",
                label=f"IF Threshold={if_threshold:.4f}")
    ax4.set_xlabel("Anomaly Score (higher = anomalous)", color="#8B949E", fontsize=8)
    ax4.set_ylabel("Count", color="#8B949E", fontsize=8)
    ax4.legend(fontsize=7, facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3")
    _style(ax4, "Isolation Forest — Score Distribution")

    plt.suptitle("Thermo-Twin Phase 3 — HVAC Fault Detection Models: Critical Test",
                 color="#E6EDF3", fontsize=13, y=0.98, fontweight="bold")

    out = ROOT / "model" / "phase3_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  Phase 3: Anomaly Detection Model Training")
    print("=" * 56)

    X_train, X_val, X_test, y_test = load_data()
    X_train_np = X_train.numpy()
    X_val_np   = X_val.numpy()
    X_test_np  = X_test.numpy()

    # ── 3A: Isolation Forest ─────────────────────────────────────────────────
    print("\n[3A] Isolation Forest")
    clf = train_isolation_forest(X_train_np)
    save_isolation_forest(clf, CHECKPOINT_DIR / "isolation_forest.pkl")
    print(f"     Saved -> model/checkpoints/isolation_forest.pkl")

    if_val_scores  = anomaly_scores(clf, X_val_np)
    if_test_scores = anomaly_scores(clf, X_test_np)
    if_threshold   = float(np.mean(if_val_scores) + N_SIGMA * np.std(if_val_scores))
    print(f"     IF threshold: {if_threshold:.4f}")

    # ── 3B: Autoencoder ──────────────────────────────────────────────────────
    print("\n[3B] Autoencoder")
    model = Autoencoder(input_dim=INPUT_DIM, bottleneck=BOTTLENECK)
    model, train_hist, val_hist, best_val = train_autoencoder(model, X_train, X_val)

    ae_path = CHECKPOINT_DIR / "autoencoder.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim":  INPUT_DIM,
            "bottleneck": BOTTLENECK,
            "hyperparams": dict(
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                weight_decay=WEIGHT_DECAY, noise_std=NOISE_STD,
            ),
        },
        ae_path,
    )
    print(f"\n     Saved -> model/checkpoints/autoencoder.pt")

    # ── Threshold calibration ─────────────────────────────────────────────────
    print("\n[Threshold] Calibration on val set")
    val_errors                   = get_errors(model, X_val)
    threshold, val_mean, val_std = compute_threshold(val_errors, n_sigma=N_SIGMA)
    print(f"     Val errors: mean={val_mean:.6f}  std={val_std:.6f}")
    print(f"     Threshold (mean + {N_SIGMA}*std) = {threshold:.6f}")
    status = "PASS" if best_val <= 0.1 else "WARN (>0.1)"
    print(f"     Best val loss = {best_val:.6f}  [{status}]")

    # ── Critical test ─────────────────────────────────────────────────────────
    print("\n[Critical Test] Autoencoder on test set")
    ae_errors = get_errors(model, X_test)

    anomaly_present = y_test[y_test != "normal"]
    p99_err = float(np.percentile(ae_errors[y_test != "normal"], 90)) \
              if len(anomaly_present) else threshold * 50
    scores  = severity_score(ae_errors, threshold, p99_error=p99_err)

    print(f"\n  {'Label':15s}  {'MSE mean':>10}  {'MSE std':>9}  {'Sev mean':>9}  {'Target':>8}")
    print("  " + "-" * 60)
    for label in ["normal", "refrigerant_leak", "fan_failure", "compressor_wear"]:
        mask = y_test == label
        if not mask.any():
            continue
        e = ae_errors[mask]
        s = scores[mask]
        target = "<40" if label == "normal" else ">70"
        print(f"  {label:15s}  {e.mean():>10.4f}  {e.std():>9.4f}  {s.mean():>9.1f}  {target:>8}")

    normal_ok  = (scores[y_test == "normal"] <= 40).mean() * 100
    anomaly_ok = (scores[y_test != "normal"] >= 70).mean() * 100
    print(f"\n  Normal  windows scoring <=40: {normal_ok:.1f}%  (target ~100%)")
    print(f"  Anomaly windows scoring >=70: {anomaly_ok:.1f}%  (target ~100%)")

    # Save threshold config
    cfg = dict(
        threshold  = round(threshold, 6),
        val_mean   = round(val_mean, 6),
        val_std    = round(val_std, 6),
        n_sigma    = N_SIGMA,
        p99_anomaly= round(p99_err, 6),
        best_val_loss = round(best_val, 6),
        if_threshold  = round(if_threshold, 6),
    )
    save_threshold_config(CHECKPOINT_DIR / "threshold_config.json", cfg)
    print(f"\n     Saved -> model/checkpoints/threshold_config.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    out = plot_results(
        train_hist, val_hist,
        ae_errors, if_test_scores,
        y_test, threshold, if_threshold,
    )
    print(f"     Saved -> model/phase3_results.png")

    print("\n" + "=" * 56)
    print("  Phase 3 complete.")
    print("=" * 56)


if __name__ == "__main__":
    main()
