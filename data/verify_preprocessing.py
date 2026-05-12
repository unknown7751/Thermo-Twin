import numpy as np
import pickle
import pandas as pd

train = np.load("data/processed/train_windows.npz")
val   = np.load("data/processed/val_windows.npz")
test  = np.load("data/processed/test_windows.npz")

with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("=== Shape check ===")
print(f"Train  X:{train['X'].shape}  y:{train['y'].shape}")
print(f"Val    X:{val['X'].shape}    y:{val['y'].shape}")
print(f"Test   X:{test['X'].shape}   y:{test['y'].shape}")
print(f"Feature dim (all should be 200): {train['X'].shape[1]}, {val['X'].shape[1]}, {test['X'].shape[1]}")

print("\n=== Labels ===")
print(f"Train unique labels  : {set(train['y'])}")
print(f"Val unique labels    : {set(val['y'])}")
print(f"Test label counts    : {dict(pd.Series(test['y']).value_counts())}")

print("\n=== Scaler ===")
print(f"Type : {type(scaler).__name__}")
print(f"Mean range  : [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
print(f"Scale range : [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

print("\n=== Normalization (train) ===")
X = train["X"]
print(f"Mean  (should be ~0) : {X.mean():.6f}")
print(f"Std   (should be ~1) : {X.std():.6f}")

print("\n=== Data leakage check ===")
train_clean = all(l == "normal" for l in train["y"])
val_clean   = all(l == "normal" for l in val["y"])
print(f"Train contains only normals : {train_clean}")
print(f"Val   contains only normals : {val_clean}")
fault_types_in_test = set(test["y"]) - {"normal"}
print(f"Fault types in test         : {fault_types_in_test}")
print(f"All 3 fault types present   : {fault_types_in_test == {'refrigerant_leak', 'fan_failure', 'compressor_wear'}}")
