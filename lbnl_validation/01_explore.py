"""
Step 1 -- LBNL Data Exploration
Understand the RTU.csv dataset structure, quality, and fault distribution.

Run: python lbnl_validation/01_explore.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

CSV_PATH = ROOT / "LBNL_Dataset" / "RTU.csv"


def main():
    print("=" * 60)
    print("  Step 1: LBNL Data Exploration")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH, na_values=["NA", ""])
    print(f"\n  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    print(f"\n  Column names:")
    for i, col in enumerate(df.columns):
        print(f"    {i:2d}. {col}")

    print(f"\n  Time range: {df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[-1]}")

    # Fault distribution
    gt = df["Fault Detection Ground Truth"]
    print(f"\n  Fault Detection Ground Truth:")
    print(f"    Total rows     : {len(gt):,}")
    print(f"    Normal (0)     : {(gt == 0).sum():,}")
    print(f"    Fault  (1)     : {(gt == 1).sum():,}")
    print(f"    Fault ratio    : {gt.mean():.2%}")

    # Missing values for key columns
    key_cols = [
        "RTU: Supply Air Temperature",
        "RTU: Circuit 1 Discharge Pressure",
        "RTU: Fan Electricity ",
        "RTU: Electricity",
        "Fault Detection Ground Truth",
    ]
    print(f"\n  Missing values (key columns):")
    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = missing / len(df) * 100
            print(f"    {col:45s}: {missing:6,} ({pct:.1f}%)")

    # Value ranges for key sensor columns
    sensor_cols = [
        "RTU: Electricity",
        "RTU: Circuit 1 Discharge Pressure",
        "RTU: Fan Electricity ",
        "RTU: Supply Air Temperature",
    ]
    print(f"\n  Value ranges (key sensors):")
    for col in sensor_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            print(f"    {col:45s}: min={series.min():.2f}  max={series.max():.2f}  mean={series.mean():.2f}  std={series.std():.2f}")

    print("\n" + "=" * 60)
    print("  Step 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
