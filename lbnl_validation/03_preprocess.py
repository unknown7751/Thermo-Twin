"""
Step 3 -- LBNL Sliding Window Preprocessing
Creates 200-dim sliding windows from mapped LBNL data, applies the
SYNTHETIC-trained StandardScaler, and builds a combined test set.

Since the LBNL dataset is ALL faults, we combine:
  - Synthetic validation normals (from existing pipeline)
  - LBNL fault windows
to create a proper binary classification test set.

Run: python lbnl_validation/03_preprocess.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

MAPPED_CSV    = ROOT / "data" / "real" / "lbnl_mapped.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
STEP_SIZE   = 25

SENSOR_COLS = [
    "compressor_power_kw",
    "discharge_pressure_psi",
    "fan_rpm",
    "supply_air_temp_c",
]


def main():
    print("=" * 60)
    print("  Step 3: LBNL Sliding Window Preprocessing")
    print("=" * 60)

    df = pd.read_csv(MAPPED_CSV)
    print(f"\n  Loaded {len(df):,} rows from lbnl_mapped.csv")

    sensors = df[SENSOR_COLS].values
    fault_labels = df["fault_label"].values

    # Create sliding windows (same structure as synthetic pipeline)
    windows = []
    window_labels = []

    for i in range(0, len(sensors) - WINDOW_SIZE + 1, STEP_SIZE):
        window = sensors[i:i + WINDOW_SIZE]   # (50, 4)
        # Flatten per-stream: [power_0..49, pressure_0..49, rpm_0..49, temp_0..49]
        window_flat = np.concatenate([window[:, c] for c in range(4)])  # (200,)

        # Majority label in window
        window_fault_labels = fault_labels[i:i + WINDOW_SIZE]
        fault_count = np.sum(window_fault_labels == "fault")
        label = "fault" if fault_count > WINDOW_SIZE // 2 else "normal"

        windows.append(window_flat)
        window_labels.append(label)

    windows = np.array(windows, dtype=np.float32)
    window_labels = np.array(window_labels)

    n_normal = np.sum(window_labels == "normal")
    n_fault = np.sum(window_labels == "fault")
    print(f"\n  LBNL Windows created: {windows.shape}")
    print(f"    Normal : {n_normal:,}")
    print(f"    Fault  : {n_fault:,}")

    # Apply SYNTHETIC scaler (model was trained with this scaler)
    scaler_path = PROCESSED_DIR / "scaler.pkl"
    print(f"\n  Loading synthetic scaler from {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    windows_scaled = scaler.transform(windows)

    # Load existing synthetic val set (all normal)
    synth_val = np.load(PROCESSED_DIR / "val_windows.npz")
    synth_val_X = synth_val["X"]
    synth_val_y = synth_val["y"]
    print(f"  Synthetic val set: {synth_val_X.shape} (labels: {dict(pd.Series(synth_val_y).value_counts())})")

    # LBNL fault windows
    lbnl_fault_mask = window_labels == "fault"
    lbnl_fault_X = windows_scaled[lbnl_fault_mask]
    lbnl_fault_y = window_labels[lbnl_fault_mask]

    # Combined test set: synthetic normals + LBNL faults
    X_test = np.vstack([synth_val_X, lbnl_fault_X])
    y_test = np.concatenate([synth_val_y, lbnl_fault_y])

    print(f"\n  Combined test set: {X_test.shape}")
    print(f"    Normal (synthetic val) : {(y_test == 'normal').sum():,}")
    print(f"    Fault  (LBNL real)     : {(y_test == 'fault').sum():,}")

    # Save outputs
    np.savez_compressed(PROCESSED_DIR / "lbnl_fault_windows.npz",
                        X=lbnl_fault_X, y=lbnl_fault_y)
    np.savez_compressed(PROCESSED_DIR / "lbnl_combined_test.npz",
                        X=X_test, y=y_test)
    np.savez_compressed(PROCESSED_DIR / "lbnl_all_windows_raw.npz",
                        X=windows, y=window_labels)

    print(f"\n  Saved to {PROCESSED_DIR}/")
    print(f"    lbnl_fault_windows.npz   - {lbnl_fault_X.shape[0]:,} fault windows (scaled)")
    print(f"    lbnl_combined_test.npz   - {X_test.shape[0]:,} windows (normal+fault)")
    print(f"    lbnl_all_windows_raw.npz - {windows.shape[0]:,} windows (unscaled)")

    print("\n" + "=" * 60)
    print("  Step 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
