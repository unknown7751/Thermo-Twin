"""
Step 3A — Isolation Forest MVP fallback.
Trains on normal windows only; produces anomaly scores for any input.

Run standalone:  python model/isolation_forest.py
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import IsolationForest

ROOT           = Path(__file__).parent.parent
PROCESSED_DIR  = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "model" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    clf = IsolationForest(
        n_estimators=300,
        contamination=0.05,   # small: we trained on normal-only
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)
    return clf


def anomaly_scores(clf: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Returns positive scores: higher = more anomalous."""
    return -clf.decision_function(X)


def save_isolation_forest(clf: IsolationForest, path: Path | str) -> None:
    with open(path, "wb") as f:
        pickle.dump(clf, f)


def load_isolation_forest(path: Path | str) -> IsolationForest:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

    train = np.load(PROCESSED_DIR / "train_windows.npz")
    val   = np.load(PROCESSED_DIR / "val_windows.npz")
    test  = np.load(PROCESSED_DIR / "test_windows.npz")

    print("Training Isolation Forest ...")
    clf = train_isolation_forest(train["X"])

    save_path = CHECKPOINT_DIR / "isolation_forest.pkl"
    save_isolation_forest(clf, save_path)
    print(f"  Saved -> {save_path}")

    val_scores  = anomaly_scores(clf, val["X"])
    test_scores = anomaly_scores(clf, test["X"])
    print(f"\n  Val   score: mean={val_scores.mean():.4f}  std={val_scores.std():.4f}")

    import pandas as pd
    print("\n  Test scores by label:")
    for label in np.unique(test["y"]):
        mask = test["y"] == label
        s = test_scores[mask]
        print(f"    {label:15s}: mean={s.mean():.4f}  std={s.std():.4f}  n={mask.sum()}")
