# Thermo-Twin — Phases of Development

> Hackathon build plan. 3 days. 7 phases.
> Phases 1–4 are complete and validated. Phases 5–7 remain.

---

## Golden Rule

> **Phase 3 (working model) gates everything.**
> If behind schedule, cut dashboard features before cutting SHAP.
> Explainability — and the prescriptive output on top of it — is the differentiator.

---

## Timeline Overview

```
DAY 1                          DAY 2                    DAY 3
│                              │                        │
├─ Phase 1: Data       ✅      ├─ Phase 4: SHAP   ✅    ├─ Phase 6: Dashboard  ⬜
├─ Phase 2: Preprocess ✅      └─ Phase 5: Backend ⬜   └─ Phase 7: Polish     ⬜
└─ Phase 3: Model      ✅
```

| Phase | Task | Status | Est. Time |
|---|---|---|---|
| 1 | Synthetic HVAC Data | ✅ Done | 3h |
| 2 | Preprocessing Pipeline | ✅ Done | 1h |
| 3 | Autoencoder + Isolation Forest | ✅ Done | 2h |
| 4 | SHAP Explainability + Prescriptions | ✅ Done | 2h |
| 5 | Alert Backend (Flask) | ⬜ TODO | 2h |
| 6 | Operator Dashboard (Streamlit) | ⬜ TODO | 3h |
| 7 | Polish & Pitch Prep | ⬜ TODO | 2h |

---

## Phase 1 — Synthetic HVAC Data ✅ DONE

**File:** `data/raw/generate_sensor_data.py`
**Output:** `data/raw/synthetic_data.csv` — 20,000 rows

### What Was Built

Four thermodynamically correlated sensor streams for two Carrier machine IDs:
- `CARRIER-CHILLER-01` — Commercial Chiller
- `CARRIER-VRF-UNIT-01` — Variable Refrigerant Flow unit

**Normal Correlations:**
```
discharge_pressure = 70 × compressor_power + noise
fan_rpm            = 340 × compressor_power + noise
supply_air_temp    = 18 − (2 × compressor_power) + noise
```

**Three Fault Types Injected:**

| Fault | compressor_power | discharge_pressure | fan_rpm | supply_air_temp |
|---|---|---|---|---|
| `refrigerant_leak` | HIGH (unchanged) | DROPS suddenly | Normal | RISES suddenly |
| `fan_failure` | Rises gradually | Rises gradually | DROPS abruptly | Rises gradually |
| `compressor_wear` | Slowly increases | Slowly decreases | Normal | Slowly increases |

**Actual Label Distribution:**
```
normal            14,147
compressor_wear    3,704
fan_failure        1,224
refrigerant_leak     925
```

### Done Checklist
- [x] 20,000+ rows generated across 2 machines
- [x] All 3 fault types injected and labeled
- [x] CSV schema: `timestamp, machine_id, compressor_power_kw, discharge_pressure_psi, fan_rpm, supply_air_temp_c, fault_label`
- [x] Validation plot saved to `data/raw/synthetic_data_preview.png`

---

## Phase 2 — Preprocessing Pipeline ✅ DONE

**Files:** `data/preprocess.py`, `data/verify_preprocessing.py`
**Output:** `data/processed/` — 4 files

### What Was Built

Sliding window over 4 sensor streams → 200-dimensional feature vectors.

```
Window of 50 samples × 4 streams = 200-dim vector

Indices   0–49  : compressor_power_kw
Indices  50–99  : discharge_pressure_psi
Indices 100–149 : fan_rpm
Indices 150–199 : supply_air_temp_c
```

**Splits:**
- Train: 412 windows (normal only — no leakage into model)
- Val:   103 windows (normal only — used for threshold calibration)
- Test:  386 windows (all 3 fault types + holdout normals)

### Done Checklist
- [x] Feature vectors are 200-dimensional
- [x] Scaler fitted on normal training data only
- [x] Train and Val contain only normal windows (zero leakage)
- [x] All 3 fault types present in test set
- [x] Normalization verified: mean ≈ 0, std ≈ 1

---

## Phase 3 — Autoencoder + Isolation Forest ✅ DONE

**Files:** `model/autoencoder.py`, `model/train.py`, `model/isolation_forest.py`, `model/threshold.py`, `model/evaluate.py`
**Output:** `model/checkpoints/`

### Autoencoder Architecture
```
Input(200) → Linear(200→128) → ReLU
           → Linear(128→64)  → ReLU
           → Linear(64→8)    → ReLU   ← Bottleneck
           → Linear(8→64)    → ReLU
           → Linear(64→128)  → ReLU
           → Linear(128→200)          ← Reconstruction
```

- Trained on normal windows only (412 windows)
- Denoising regularization: noise_std = 0.02
- Early stopping: patience = 80
- Threshold = val_mean + 2.5 × val_std = **0.1957**

### Isolation Forest (Fallback)
- 300 estimators, contamination = 0.05
- Ready to swap in instantly if autoencoder breaks during demo

### Validated Performance

| Metric | Autoencoder | Isolation Forest |
|---|---|---|
| ROC-AUC | **0.985** | 0.968 |
| F1 (binary) | **0.978** | 0.898 |
| Normal ≤ 40 | **100%** | — |
| False Positives | **0** | 4 |

**Severity scores by fault type:**

| Label | Mean Score | % ≥ 70 |
|---|---|---|
| Normal | 29.6 | 0% |
| Refrigerant Leak | 90.5 | 96.4% |
| Fan Failure | 92.8 | 97.1% |
| Compressor Wear | 73.8 | 65.6% |

### Done Checklist
- [x] Isolation Forest trained and saved
- [x] Autoencoder trains without errors, loss decreases
- [x] All 3 fault types score above 70 (mean)
- [x] Normal windows score below 40 (100%)
- [x] Model saved to `model/checkpoints/autoencoder.pt`
- [x] Threshold config saved to `model/checkpoints/threshold_config.json`

---

## Phase 4 — SHAP Explainability + Prescriptive Rules ✅ DONE

**Files:** `explainability/shap_explainer.py`, `explainability/precompute_explanations.py`, `explainability/alert_payload.py`
**Output:** `explainability/demo_explanations.json`

### What Was Built

SHAP GradientExplainer decomposes each anomaly window into per-stream contributions:
```python
compressor_power_pct  = abs(shap_values[0:50]).sum()   / total × 100
discharge_pres_pct    = abs(shap_values[50:100]).sum()  / total × 100
fan_rpm_pct           = abs(shap_values[100:150]).sum() / total × 100
supply_air_temp_pct   = abs(shap_values[150:200]).sum() / total × 100
# All four sum to exactly 100%
```

**Prescriptive Rules (calibrated to actual SHAP fingerprints):**
```
supply_air_temp_pct > 50%   →  Refrigerant Leak
fan_rpm_pct > 35%           →  Condenser Fan Failure
compressor_power_pct > 45%  →  Compressor Wear
```

**Pre-Computed Demo Explanations (from `demo_explanations.json`):**

| Scenario | Dominant Stream | Fault Identified | Severity |
|---|---|---|---|
| `scenario_1_refrigerant_leak` | supply_air_temp 85% | Refrigerant Leak | 100 |
| `scenario_2_fan_failure` | fan_rpm 47% | Condenser Fan Failure | 100 |
| `scenario_3_compressor_wear` | compressor_power 62% | Compressor Wear | 91 |

**Alert Payload Schema:**
```json
{
  "machine_id": "CARRIER-CHILLER-01",
  "timestamp": "2024-01-15T14:32:07Z",
  "severity_score": 87,
  "fault_type": "Refrigerant Leak",
  "action": "STOP UNIT",
  "explanation": {
    "compressor_power_pct": 4.4,
    "discharge_pressure_pct": 6.6,
    "fan_rpm_pct": 4.4,
    "supply_air_temp_pct": 84.6,
    "summary": "Anomaly driven primarily by Temp Rise (85%)"
  },
  "prescription": {
    "fault": "Refrigerant Leak in Evaporator Coil",
    "impact": "Cooling efficiency down ~40%, unit running but not cooling",
    "action": "Dispatch technician with refrigerant recharge kit and leak detector"
  }
}
```

### Done Checklist
- [x] SHAP values compute without error
- [x] Attributions split across all 4 sensors (sum = 100%)
- [x] Prescriptive rules fire correctly for each fault type
- [x] All 3 demo scenario explanations pre-computed and saved
- [x] Output is human-readable (percentages + natural language)

---

## Phase 5 — Alert Backend ⬜ TODO

**Target file:** `backend/app.py`
**Estimated time:** 2 hours

### What to Build

Simple Flask API — no database, no auth.

```python
POST /alert          # Receive and store an alert (from inference layer)
GET  /alerts         # Return last 50 alerts (dashboard polls this)
POST /demo/<scenario>  # Trigger a pre-loaded demo scenario by name
GET  /health         # Healthcheck
```

**Demo scenarios to load from `demo_explanations.json`:**
- `scenario_1_refrigerant_leak`
- `scenario_2_fan_failure`
- `scenario_3_compressor_wear`

**Payload schema:** already defined in `explainability/alert_payload.py`.

### Done When
- [ ] `POST /alert` receives and stores a payload
- [ ] `GET /alerts` returns alert history as JSON
- [ ] `POST /demo/<scenario>` triggers each of the 3 scenarios
- [ ] Dashboard can poll `/alerts` and see new entries appear
- [ ] CORS enabled (dashboard runs on a different port)

---

## Phase 6 — Operator Dashboard ⬜ TODO

**Target file:** `dashboard/app.py`
**Estimated time:** 3 hours

### Non-Negotiable UI Elements

| Element | Why |
|---|---|
| 4-stream live signal plot | Shows thermodynamic harmony breaking at fault moment |
| Severity gauge (0–100) | Judges understand it in 1 second |
| 4-bar SHAP chart | The differentiator — shows which component drove the anomaly |
| Fault type card | "Refrigerant Leak" — specific, not generic |
| Prescription card | "Dispatch with recharge kit" — actionable |
| Alert log table | Machine ID, score, action, fault type |
| Demo trigger buttons | 3 buttons — one per scenario — so demo never fails on timing |

### Dashboard Title
```
Thermo-Twin | Carrier HVAC Diagnostics
```

### Machine IDs
- `CARRIER-CHILLER-01`
- `CARRIER-VRF-UNIT-01`

### Done When
- [ ] All 4 sensor streams plotted (with fault moment visible)
- [ ] Anomalous stream highlighted in red at fault moment
- [ ] Severity gauge updates on new alert
- [ ] 4-bar SHAP chart updates per alert
- [ ] Fault type and prescription display on each alert card
- [ ] All 3 demo scenarios trigger cleanly via button
- [ ] Alert log shows last 10 alerts with full detail

---

## Phase 7 — Polish & Pitch Prep ⬜ TODO

**Estimated time:** 2 hours

### Technical Checklist
- [ ] End-to-end demo runs clean 3 times in a row without failure
- [ ] All 3 scenarios trigger via button press — no timing dependency
- [ ] Isolation Forest fallback tested: swap in if autoencoder breaks
- [ ] Dashboard title reads "Thermo-Twin | Carrier HVAC Diagnostics"
- [ ] Machine IDs show `CARRIER-CHILLER-01` and `CARRIER-VRF-UNIT-01`

### Pitch Checklist
- [ ] 2-minute demo walk-through rehearsed at least 3 times
- [ ] All 5 objection responses memorized (see below)

### Edge Deployment Statement (when judges ask)
> "The autoencoder is architecturally edge-compatible. Post-hackathon, it can be exported to ONNX and quantized to INT8 — inference drops to under 5ms with ~4× memory reduction. The 8-dimensional bottleneck was chosen with that constraint in mind from day one."

### 5 Objection Responses

| Objection | Response |
|---|---|
| "You don't have real sensor data." | Our synthetic data simulates the actual thermodynamic physics — compressor power, pressure, and temperature have documented, well-understood relationships in refrigeration engineering. The fault signatures we inject are physically accurate. |
| "How do you deploy sensors on existing HVAC units?" | Carrier's Digital Connectivity team already installs BACnet bridges and IoT gateways at customer sites daily. We reuse that exact deployment capability — no new hardware infrastructure needed. |
| "Why wouldn't Carrier just use rule-based fault codes?" | ClimaVision already sends fault codes. Thermo-Twin adds the prescriptive layer on top — it tells you *which part to bring*, not just that something is wrong. |
| "What about model degradation over time?" | The threshold is recalibrated from validation data. Seasonal changes or equipment upgrades shift the normal baseline — retraining on new normal windows corrects it without touching the architecture. |
| "Why would Carrier build this instead of a startup?" | Carrier already sells BluEdge predictive maintenance as a recurring revenue SaaS. Thermo-Twin is a new SKU in an existing commercial motion — not a new business. |

---

## Final Done Condition

Before walking into the presentation:

```
✅ Synthetic HVAC data: 4 streams, 3 fault types, 20k rows
✅ Autoencoder trained: faults score 70+, normals score below 40
✅ SHAP: 4-sensor attribution, prescriptive rules firing correctly
✅ Demo explanations pre-computed (< 1s load time)
⬜ Backend: all 3 demo triggers work via POST
⬜ Dashboard: 4 streams, gauge, SHAP bars, fault card, prescription
⬜ Demo runs clean 3 times in a row
⬜ Isolation Forest fallback ready to swap in
⬜ All 5 objection responses ready
```

---

*Thermo-Twin — When thermodynamic harmony breaks, we name the component and send the right part.*
