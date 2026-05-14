"""
Step 2 -- LBNL Column Mapping & Range Scaling
Maps RTU.csv columns to the 4-sensor schema used by Thermo-Twin,
then scales values to match synthetic data ranges using MinMaxScaler.

Run: python lbnl_validation/02_map_columns.py
"""

import sys
import pickle
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

CSV_PATH   = ROOT / "LBNL_Dataset" / "RTU.csv"
OUTPUT_DIR = ROOT / "data" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column mapping: RTU.csv -> Thermo-Twin 4-sensor schema
COL_MAP = {
    "RTU: Electricity":                    "compressor_power_kw",
    "RTU: Circuit 1 Discharge Pressure":   "discharge_pressure_psi",
    "RTU: Fan Electricity ":               "fan_rpm",            # trailing space in CSV
    "RTU: Supply Air Temperature":         "supply_air_temp_c",
}

# Synthetic data ranges (from training data)
SYNTHETIC_RANGES = {
    "compressor_power_kw":    (0.5, 5.5),
    "discharge_pressure_psi": (150, 350),
    "fan_rpm":                (800, 1600),
    "supply_air_temp_c":      (8, 16),
}


def main():
    print("=" * 60)
    print("  Step 2: LBNL Column Mapping & Range Scaling")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH, na_values=["NA", ""])
    print(f"\n  Loaded {len(df):,} rows")

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%m/%d/%Y %H:%M")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Convert sensor columns to numeric
    for src_col in COL_MAP:
        df[src_col] = pd.to_numeric(df[src_col], errors="coerce")

    # Count NAs before cleanup
    print(f"\n  NAs before cleanup:")
    for col in COL_MAP:
        count = df[col].isna().sum()
        print(f"    {col:45s}: {count:,}")

    # Forward-fill NAs (sensor readings carry over when fan is off)
    sensor_cols = list(COL_MAP.keys())
    df[sensor_cols] = df[sensor_cols].ffill()
    df[sensor_cols] = df[sensor_cols].bfill()

    remaining_na = df[sensor_cols].isna().any(axis=1).sum()
    if remaining_na > 0:
        print(f"  Still {remaining_na} rows with NA after fill -- dropping")
        df = df.dropna(subset=sensor_cols).reset_index(drop=True)

    print(f"  Rows after cleanup: {len(df):,}")

    # Scale LBNL ranges to match synthetic data ranges
    scalers = {}
    for src_col, dst_col in COL_MAP.items():
        lo, hi = SYNTHETIC_RANGES[dst_col]
        scaler = MinMaxScaler(feature_range=(lo, hi))
        raw_values = df[src_col].values.reshape(-1, 1)
        df[dst_col] = scaler.fit_transform(raw_values).flatten()
        scalers[dst_col] = scaler
        print(f"  Mapped {src_col}")
        print(f"         raw: [{raw_values.min():.2f}, {raw_values.max():.2f}] -> [{lo}, {hi}]")

    # Map fault labels (all rows are fault in this dataset)
    df["fault_label"] = df["Fault Detection Ground Truth"].map({0: "normal", 1: "fault"})

    # Save mapped dataset
    out_cols = ["Timestamp", "compressor_power_kw", "discharge_pressure_psi",
                "fan_rpm", "supply_air_temp_c", "fault_label"]
    df_mapped = df[out_cols].copy()

    csv_path = OUTPUT_DIR / "lbnl_mapped.csv"
    df_mapped.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}  ({len(df_mapped):,} rows)")

    fault_counts = df_mapped["fault_label"].value_counts()
    print(f"\n  Fault distribution (mapped):")
    for label, count in fault_counts.items():
        print(f"    {label:10s}: {count:,} ({count/len(df_mapped)*100:.1f}%)")

    # Save scalers for future inference
    pkl_path = OUTPUT_DIR / "range_mappers.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"  Saved: {pkl_path}")

    print("\n" + "=" * 60)
    print("  Step 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
