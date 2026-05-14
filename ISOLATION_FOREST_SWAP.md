# Isolation Forest Fallback — 2-Minute Swap Guide

## Why you almost certainly won't need this

The demo runs entirely from pre-computed values in `explainability/demo_explanations.json`.
No ML model is loaded or called at any point during the presentation.
Backend restart after a kill takes < 3 seconds — just restart and the JSON reloads.

---

## The 2 lines to change (live inference only)

If you were running live inference through `explainability/shap_explainer.py`
and the autoencoder failed, here are the two lines to swap:

**File: `explainability/shap_explainer.py`**

### Current (Autoencoder):
```python
# Line 26
from model.autoencoder import load_autoencoder

# Line 81 (inside SHAPExplainer.__init__)
self._autoencoder = load_autoencoder(str(checkpoint_path))
```

### Swap to (Isolation Forest — severity score only, no SHAP chart):
```python
# Line 26 — replace with:
import joblib

# Line 81 — replace with:
self._autoencoder = joblib.load("model/checkpoints/isolation_forest.pkl")
```

---

## Faster fallback during demo: restart backend only

If the backend crashes or behaves unexpectedly during the demo:

```
Ctrl+C  (kill backend in terminal)
python backend/app.py  (restart — < 3 seconds)
```

The dashboard reconnects automatically on next poll (within 2 seconds).
All 3 demo scenarios reload from JSON instantly — no model needed.

---

## Demo is safe because:

- `demo_explanations.json` → pre-computed, static, loaded at backend startup
- No model checkpoint files are required during the demo
- No internet connection required — all inference is local (pre-computed)
- Severity scores, SHAP percentages, prescriptions are all hardcoded in JSON
