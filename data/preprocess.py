"""
Phase 2 — Preprocessing Pipeline (Thermo-Twin)
Converts raw HVAC sensor CSV into sliding-window feature vectors for model training.

Four sensor streams → 200-dim feature vector per window:
  indices   0-49  : compressor_power_kw
  indices  50-99  : discharge_pressure_psi
  indices 100-149 : fan_rpm
  indices 150-199 : supply_air_temp_c

Outputs (data/processed/):
    train_windows.npz  — 80 % of normal windows, scaler-fitted
    val_windows.npz    — 20 % of normal windows (also acts as holdout-normal in test)
    test_windows.npz   — all fault windows + holdout normal windows
    scaler.pkl         — StandardScaler fitted on training data only
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ─── Config ──────────────────────────────────────────────────────────────────
RAW_CSV       = Path(__file__).parent / "raw" / "synthetic_data.csv"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50   # samples per window
STEP        = 25   # 50 % overlap
RANDOM_SEED = 42


# ─── Step 1: Sliding windows ─────────────────────────────────────────────────

SENSOR_COLS = [
    "compressor_power_kw",
    "discharge_pressure_psi",
    "fan_rpm",
    "supply_air_temp_c",
]
LABEL_COL   = "fault_label"
N_STREAMS   = len(SENSOR_COLS)   # 4


def build_windows(df: pd.DataFrame, window_size: int, step: int):
    """
    For each machine independently, slide a fixed-width window across the
    time series and stack [stream0_t0..t49, stream1_t0..t49, ...] → 200-dim vector.

    Two labels are returned per window:
      - majority_label : most frequent sample label (used as the class name in test)
      - contains_fault: True if ANY sample in the window is a fault
        (used to gate training — prevents short fault bursts leaking into train)
    """
    all_features, all_labels, all_contains_anomaly, all_anomaly_types = [], [], [], []

    for machine_id, group in df.groupby("machine_id", sort=False):
        group = group.reset_index(drop=True)
        streams = [group[col].to_numpy() for col in SENSOR_COLS]
        lbl     = group[LABEL_COL].to_numpy()

        n = len(group)
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            feature_vec   = np.concatenate([s[start:end] for s in streams])
            window_labels = lbl[start:end]

            # majority vote — used only for informational display
            unique_lbl, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique_lbl[np.argmax(counts)]

            # any-anomaly flag — used to gate the train/val split
            has_anomaly = any(l != "normal" for l in window_labels)

            # dominant anomaly type — used as the TEST SET label
            # For a window with any anomaly, this is the most common non-normal label.
            if has_anomaly:
                non_normal = [l for l in window_labels if l != "normal"]
                u, c = np.unique(non_normal, return_counts=True)
                anomaly_type = u[np.argmax(c)]
            else:
                anomaly_type = "normal"

            all_features.append(feature_vec)
            all_labels.append(majority_label)
            all_contains_anomaly.append(has_anomaly)
            all_anomaly_types.append(anomaly_type)

    return (
        np.array(all_features, dtype=np.float32),
        np.array(all_labels),
        np.array(all_contains_anomaly, dtype=bool),
        np.array(all_anomaly_types),
    )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading raw data ...")
    df = pd.read_csv(RAW_CSV)
    print(f"  {len(df):,} rows | machines: {df['machine_id'].unique().tolist()}")

    # ── 1. Build windows ─────────────────────────────────────────────────────
    print(f"\nBuilding windows (size={WINDOW_SIZE}, step={STEP}) ...")
    windows, labels, contains_anomaly, anomaly_types = build_windows(df, WINDOW_SIZE, STEP)
    print(f"  Total windows  : {len(windows):,}")
    print(f"  Feature dims   : {windows.shape[1]}  (should be {WINDOW_SIZE * N_STREAMS})")

    label_counts = pd.Series(labels).value_counts()
    print(f"\n  Majority-label distribution:\n{label_counts.to_string()}")
    print(f"\n  Windows with ANY anomaly sample : {contains_anomaly.sum():,}")
    print(f"  Purely normal windows           : {(~contains_anomaly).sum():,}")

    # ── 2. Separate normal vs anomaly ────────────────────────────────────────
    # Gate by any-anomaly flag so short power spikes don't leak into training
    normal_mask  = ~contains_anomaly
    anomaly_mask =  contains_anomaly

    normal_windows  = windows[normal_mask]
    normal_labels   = labels[normal_mask]
    anomaly_windows = windows[anomaly_mask]
    anomaly_labels  = anomaly_types[anomaly_mask]   # dominant anomaly type, not majority vote

    print(f"\n  Purely normal windows : {len(normal_windows):,}")
    print(f"  Anomaly windows       : {len(anomaly_windows):,}")

    # ── 3. Train / val split (normal data only) ───────────────────────────────
    rng   = np.random.default_rng(RANDOM_SEED)
    idx   = rng.permutation(len(normal_windows))
    split = int(0.8 * len(idx))

    train_idx, val_idx = idx[:split], idx[split:]

    X_train_raw = normal_windows[train_idx]
    y_train     = normal_labels[train_idx]
    X_val_raw   = normal_windows[val_idx]
    y_val       = normal_labels[val_idx]

    # ── 4. Fit scaler on training normals ONLY ───────────────────────────────
    scaler      = StandardScaler()
    X_train     = scaler.fit_transform(X_train_raw)
    X_val       = scaler.transform(X_val_raw)

    # ── 5. Test set = anomaly windows + holdout normal (the val split) ────────
    X_test_raw  = np.vstack([X_val_raw, anomaly_windows])
    y_test      = np.concatenate([y_val, anomaly_labels])
    X_test      = scaler.transform(X_test_raw)

    # ── 6. Sanity checks ─────────────────────────────────────────────────────
    assert X_train.shape[1] == WINDOW_SIZE * N_STREAMS, "Feature dim mismatch"
    assert X_val.shape[1]   == WINDOW_SIZE * N_STREAMS
    assert X_test.shape[1]  == WINDOW_SIZE * N_STREAMS
    # Scaler was never fitted on anomaly data (any-anomaly gate guarantees this)
    assert not any(lbl != "normal" for lbl in y_train), "Anomaly leaked into train"
    assert not any(lbl != "normal" for lbl in y_val),   "Anomaly leaked into val"

    print("\n  Data splits:")
    print(f"    Train : {X_train.shape}  labels: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"    Val   : {X_val.shape}    labels: {pd.Series(y_val).value_counts().to_dict()}")
    print(f"    Test  : {X_test.shape}   labels: {pd.Series(y_test).value_counts().to_dict()}")

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    np.savez_compressed(PROCESSED_DIR / "train_windows.npz", X=X_train, y=y_train)
    np.savez_compressed(PROCESSED_DIR / "val_windows.npz",   X=X_val,   y=y_val)
    np.savez_compressed(PROCESSED_DIR / "test_windows.npz",  X=X_test,  y=y_test)

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved to {PROCESSED_DIR}/")
    print("  train_windows.npz")
    print("  val_windows.npz")
    print("  test_windows.npz")
    print("  scaler.pkl")


if __name__ == "__main__":
    main()
