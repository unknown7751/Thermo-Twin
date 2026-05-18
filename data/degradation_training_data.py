"""
Synthetic degradation trajectory generator + LSTM trainer for Thermo-Twin.

USAGE
  Generate data only:
      python data/degradation_training_data.py

  Generate data AND train LSTM:
      python data/degradation_training_data.py --train

  Custom epochs / learning rate:
      python data/degradation_training_data.py --train --epochs 100 --lr 5e-4

OUTPUT
  data/degradation_training_data.npz
      X_history   (N, 168, 3)  float32   — normalised health sequences [0,1]
      y_rates     (N, 3)       float32   — true daily wear rates (pct/day, signed)

  model/checkpoints/degradation_lstm.pt   (only with --train)

TRAJECTORY TYPES  (total 1 100 trajectories)
  Normal          500   constant health ≈ 100 %, tiny noise
  Refrigerant     200   refrig drops 1.5–3.0 pct/day, others flat
  Fan bearing     200   fan drops 1.6–2.4 pct/day; compressor slight cascade
  Compressor wear 200   comp drops 0.9–1.4 pct/day; others flat

SEQUENCE FORMAT
  Each trajectory spans 30 days at 12 samples/day (2-hour aggregation).
  The first LOOKBACK_DAYS (14) days form the input sequence (168 samples).
  The label is the mean daily rate over the full 30-day trajectory.

ARCHITECTURE  (must match backend/degradation_trajectory.py _load_lstm_checkpoint)
  nn.LSTM(input=3, hidden=64, batch_first=True)
  nn.Linear(64, 32) → ReLU → nn.Linear(32, 3)
  Output: pct/day (signed, negative = degrading)
"""

import sys
import argparse
import logging
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("degradation_trainer")

SAMPLES_PER_DAY  = 12     # one sample every 2 hours
LOOKBACK_DAYS    = 14
LOOKBACK_SAMPLES = LOOKBACK_DAYS * SAMPLES_PER_DAY   # 168
TRAJ_DAYS        = 30
TRAJ_SAMPLES     = TRAJ_DAYS * SAMPLES_PER_DAY       # 360
NOISE_SIGMA      = 0.4    # pct noise on health observations


# ── Trajectory generation ─────────────────────────────────────────────────────

def _generate_trajectory(
    rng: np.random.Generator,
    refrig_rate: float,
    comp_rate: float,
    fan_rate: float,
    start: tuple = (100.0, 100.0, 100.0),
    sigma: float = NOISE_SIGMA,
) -> np.ndarray:
    """
    Generate one degradation trajectory at SAMPLES_PER_DAY resolution.

    Returns ndarray (TRAJ_SAMPLES, 3): [refrig_pct, comp_pct, fan_pct].
    Rates are in pct/day; clipped to [0, 100].
    """
    n = TRAJ_SAMPLES
    t = np.arange(n) / SAMPLES_PER_DAY   # time axis in days

    refrig = start[0] + refrig_rate * t + rng.normal(0, sigma, n)
    comp   = start[1] + comp_rate   * t + rng.normal(0, sigma, n)
    fan    = start[2] + fan_rate    * t + rng.normal(0, sigma, n)

    return np.clip(
        np.stack([refrig, comp, fan], axis=1),
        0.0, 100.0,
    ).astype(np.float32)


def generate_dataset(rng_seed: int = 42) -> tuple:
    """
    Build the full 1,100-trajectory dataset.

    Returns (X, y):
        X : (N, LOOKBACK_SAMPLES, 3) float32   health sequences, normalised /100
        y : (N, 3)                  float32   daily wear rates (pct/day)
    """
    rng    = np.random.default_rng(rng_seed)
    X_list, y_list = [], []

    def add(refrig_rate: float, comp_rate: float, fan_rate: float):
        traj = _generate_trajectory(rng, refrig_rate, comp_rate, fan_rate)
        X_list.append(traj[:LOOKBACK_SAMPLES] / 100.0)    # (168, 3), normalised
        y_list.append(np.array([refrig_rate, comp_rate, fan_rate], dtype=np.float32))

    # ── Normal operation (500 trajectories) ──────────────────────────────────
    log.info("Generating 500 normal-operation trajectories …")
    for _ in range(500):
        r = float(rng.uniform(-0.04, 0.04))
        c = float(rng.uniform(-0.04, 0.04))
        f = float(rng.uniform(-0.04, 0.04))
        add(r, c, f)

    # ── Refrigerant leak (200 trajectories) ──────────────────────────────────
    log.info("Generating 200 refrigerant-leak trajectories …")
    for _ in range(200):
        add(
            refrig_rate=float(rng.uniform(-3.0, -1.5)),
            comp_rate  =float(rng.uniform(-0.1,  0.0)),   # slight overload
            fan_rate   =float(rng.uniform(-0.05, 0.05)),
        )

    # ── Fan bearing wear (200 trajectories) ──────────────────────────────────
    log.info("Generating 200 fan-wear trajectories …")
    for _ in range(200):
        add(
            refrig_rate=float(rng.uniform(-0.05, 0.05)),
            comp_rate  =float(rng.uniform(-0.50, -0.20)),  # cascade
            fan_rate   =float(rng.uniform(-2.40, -1.60)),
        )

    # ── Compressor wear (200 trajectories) ───────────────────────────────────
    log.info("Generating 200 compressor-wear trajectories …")
    for _ in range(200):
        add(
            refrig_rate=float(rng.uniform(-0.05, 0.05)),
            comp_rate  =float(rng.uniform(-1.42, -0.94)),
            fan_rate   =float(rng.uniform(-0.05, 0.05)),
        )

    X = np.stack(X_list, axis=0)   # (1100, 168, 3)
    y = np.stack(y_list, axis=0)   # (1100, 3)
    log.info("Dataset: X=%s  y=%s", X.shape, y.shape)
    return X, y


# ── LSTM training ─────────────────────────────────────────────────────────────

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 0,
) -> "torch.nn.Module":
    """
    Train LSTM on the generated dataset.

    Architecture matches _load_lstm_checkpoint() in backend/degradation_trajectory.py:
      LSTM(3→64) → Linear(64→32) → ReLU → Linear(32→3)

    Loss: MSELoss on rates (pct/day).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 64, batch_first=True)
            self.fc1  = nn.Linear(64, 32)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(32, 3)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc2(self.relu(self.fc1(out[:, -1, :])))

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    N       = len(X_t)
    n_train = int(N * (1.0 - val_split))
    idx     = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X_tr, y_tr = X_t[idx[:n_train]], y_t[idx[:n_train]]
    X_vl, y_vl = X_t[idx[n_train:]], y_t[idx[n_train:]]

    loader  = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model   = _Net()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    log.info("Training LSTM  epochs=%d  lr=%g  train=%d  val=%d", epochs, lr, n_train, N - n_train)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_vl), y_vl).item()
            log.info(
                "  epoch %3d/%d  train_loss=%.5f  val_loss=%.5f",
                epoch, epochs, train_loss / n_train, val_loss,
            )

    model.eval()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate degradation training data (and optionally train LSTM)")
    parser.add_argument("--train",  action="store_true", help="Train LSTM after generating data")
    parser.add_argument("--epochs", type=int,   default=60,  help="Training epochs (default 60)")
    parser.add_argument("--lr",     type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--seed",   type=int,   default=42,  help="RNG seed")
    args = parser.parse_args()

    out_npz = ROOT / "data" / "degradation_training_data.npz"
    out_pt  = ROOT / "model" / "checkpoints" / "degradation_lstm.pt"

    # ── Generate ──
    X, y = generate_dataset(rng_seed=args.seed)
    np.savez_compressed(out_npz, X_history=X, y_rates=y)
    log.info("Saved %d trajectories → %s", len(X), out_npz)

    # ── Train ──
    if args.train:
        try:
            import torch
            model = train_lstm(X, y, epochs=args.epochs, lr=args.lr, seed=args.seed)
            torch.save(model.state_dict(), out_pt)
            log.info("Saved LSTM checkpoint → %s", out_pt)

            # Quick sanity check: predict on a high-degradation sample
            model.eval()
            refrig_seq = np.zeros((1, LOOKBACK_SAMPLES, 3), dtype=np.float32)
            refrig_seq[0, :, 0] = np.linspace(1.0, 0.7, LOOKBACK_SAMPLES)  # dropping refrig
            refrig_seq[0, :, 1] = 1.0
            refrig_seq[0, :, 2] = 1.0
            with torch.no_grad():
                pred = model(torch.tensor(refrig_seq)).numpy()
            log.info(
                "Sanity check (refrigerant leak): predicted rates = "
                "refrig=%.2f  comp=%.2f  fan=%.2f  pct/day",
                pred[0, 0], pred[0, 1], pred[0, 2],
            )

        except ImportError:
            log.warning("torch not available — skipping LSTM training")
    else:
        log.info("Run with --train to train the LSTM checkpoint.")
