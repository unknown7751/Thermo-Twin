"""
Phase 2 - Preprocessing Pipeline (Thermo-Twin)
Converts raw HVAC sensor CSV into sliding-window feature vectors for model training.

Four sensor streams -> 200-dim feature vector per window:
  indices   0-49  : compressor_power_kw
  indices  50-99  : discharge_pressure_psi
  indices 100-149 : fan_rpm
  indices 150-199 : supply_air_temp_c
"""

import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RAW_CSV       = Path(__file__).parent / "raw" / "synthetic_data.csv"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
STEP        = 25
RANDOM_SEED = 42

SENSOR_COLS = [
    "compressor_power_kw",
    "discharge_pressure_psi",
    "fan_rpm",
    "supply_air_temp_c",
]
LABEL_COL = "fault_label"
N_STREAMS = len(SENSOR_COLS)


def build_windows(df, window_size, step):
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
            unique_lbl, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique_lbl[np.argmax(counts)]
            has_anomaly = any(l != "normal" for l in window_labels)
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


class CommissioningBaseline:
    """Learns per-unit thermodynamic ratios from normal-operation commissioning data."""

    BASELINE_DIR = Path(__file__).parent.parent / "model" / "checkpoints" / "unit_baselines"

    def __init__(self, machine_id):
        self.machine_id = machine_id
        self._ratios = None

    def fit(self, df_normal):
        comp = df_normal["compressor_power_kw"].values.astype(float)
        disc = df_normal["discharge_pressure_psi"].values.astype(float)
        fan  = df_normal["fan_rpm"].values.astype(float)
        temp = df_normal["supply_air_temp_c"].values.astype(float)
        mask = comp > 0.1
        k_disc = float(np.median(disc[mask] / comp[mask]))
        k_fan  = float(np.median(fan[mask]  / comp[mask]))
        b = float(np.cov(temp[mask], comp[mask])[0, 1] / np.var(comp[mask]))
        a = float(temp[mask].mean() - b * comp[mask].mean())
        self._ratios = {
            "machine_id": self.machine_id,
            "k_disc":     round(k_disc, 4),
            "k_fan":      round(k_fan,  4),
            "k_temp_b":   round(b, 4),
            "k_temp_a":   round(a, 4),
            "n_samples":  int(mask.sum()),
        }

    def save(self):
        if self._ratios is None:
            raise ValueError("Call fit() before save()")
        self.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        path = self.BASELINE_DIR / f"{self.machine_id}.json"
        with open(path, "w") as fp:
            json.dump(self._ratios, fp, indent=2)
        return path

    def get_ratios(self):
        if self._ratios is None:
            raise ValueError("Call fit() or load() first")
        return self._ratios

    @classmethod
    def load(cls, machine_id):
        path = cls.BASELINE_DIR / f"{machine_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No baseline for {machine_id} at {path}")
        obj = cls(machine_id)
        with open(path) as fp:
            obj._ratios = json.load(fp)
        return obj

    @classmethod
    def exists(cls, machine_id):
        return (cls.BASELINE_DIR / f"{machine_id}.json").exists()


def generate_unit_baselines(df):
    """Generate and save commissioning baselines for every machine in df."""
    for machine_id, group in df.groupby("machine_id", sort=False):
        normal_rows = group[group[LABEL_COL] == "normal"]
        if len(normal_rows) < 50:
            print(f"  SKIP {machine_id}: too few normal rows ({len(normal_rows)})")
            continue
        bl = CommissioningBaseline(machine_id)
        bl.fit(normal_rows)
        path = bl.save()
        r = bl.get_ratios()
        print(f"  {machine_id}  k_disc={r['k_disc']:.1f}  k_fan={r['k_fan']:.1f}  k_temp_b={r['k_temp_b']:.3f}  k_temp_a={r['k_temp_a']:.2f}  n={r['n_samples']:,}  -> {path.name}")


def main():
    print("Loading raw data ...")
    df = pd.read_csv(RAW_CSV)
    print(f"  {len(df):,} rows | machines: {df['machine_id'].unique().tolist()}")

    print(f"\nBuilding windows (size={WINDOW_SIZE}, step={STEP}) ...")
    windows, labels, contains_anomaly, anomaly_types = build_windows(df, WINDOW_SIZE, STEP)
    print(f"  Total windows  : {len(windows):,}")
    print(f"  Feature dims   : {windows.shape[1]}")

    label_counts = pd.Series(labels).value_counts()
    print(f"\n  Majority-label distribution:\n{label_counts.to_string()}")
    print(f"\n  Windows with ANY anomaly sample : {contains_anomaly.sum():,}")
    print(f"  Purely normal windows           : {(~contains_anomaly).sum():,}")

    normal_mask  = ~contains_anomaly
    normal_windows  = windows[normal_mask]
    normal_labels   = labels[normal_mask]
    anomaly_windows = windows[contains_anomaly]
    anomaly_labels  = anomaly_types[contains_anomaly]

    print(f"\n  Purely normal windows : {len(normal_windows):,}")
    print(f"  Anomaly windows       : {len(anomaly_windows):,}")

    rng   = np.random.default_rng(RANDOM_SEED)
    idx   = rng.permutation(len(normal_windows))
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    X_train_raw = normal_windows[train_idx]
    y_train     = normal_labels[train_idx]
    X_val_raw   = normal_windows[val_idx]
    y_val       = normal_labels[val_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)

    X_test_raw = np.vstack([X_val_raw, anomaly_windows])
    y_test     = np.concatenate([y_val, anomaly_labels])
    X_test     = scaler.transform(X_test_raw)

    assert X_train.shape[1] == WINDOW_SIZE * N_STREAMS
    assert not any(lbl != "normal" for lbl in y_train)
    assert not any(lbl != "normal" for lbl in y_val)

    print("\n  Data splits:")
    print(f"    Train : {X_train.shape}  labels: {dict(pd.Series(y_train).value_counts())}")
    print(f"    Val   : {X_val.shape}    labels: {dict(pd.Series(y_val).value_counts())}")
    print(f"    Test  : {X_test.shape}   labels: {dict(pd.Series(y_test).value_counts())}")

    np.savez_compressed(PROCESSED_DIR / "train_windows.npz", X=X_train, y=y_train)
    np.savez_compressed(PROCESSED_DIR / "val_windows.npz",   X=X_val,   y=y_val)
    np.savez_compressed(PROCESSED_DIR / "test_windows.npz",  X=X_test,  y=y_test)

    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved to {PROCESSED_DIR}/")

    print("\nGenerating commissioning baselines ...")
    generate_unit_baselines(df)


if __name__ == "__main__":
    main()
