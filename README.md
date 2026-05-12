# Thermo-Twin
### Thermodynamic Digital Twins for Prescriptive HVAC Component Diagnostics

> *"We don't just detect the fever. We diagnose the disease — and tell the technician exactly what part to bring."*

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Core Innovation](#core-innovation)
- [How It Works — The Four Sensors](#how-it-works--the-four-sensors)
- [Fault Scenarios](#fault-scenarios)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Validated Performance](#validated-performance)
- [Strategic Fit — Carrier Ecosystem](#strategic-fit--carrier-ecosystem)
- [Business Model](#business-model)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)

---

## The Problem

When a Carrier HVAC unit fails in the field, today's Building Management Systems (BMS) produce one output:

```
⚠  HIGH TEMPERATURE WARNING
```

That's it. No component identified. No action specified.

A technician is dispatched — spends 2 hours on-site diagnosing whether it's a refrigerant leak, a failed fan motor, or compressor wear. If they guessed wrong on the part they brought, that's a **second truck roll**.

| Cost Driver | Typical Cost |
|---|---|
| Single truck roll (labor + travel) | $200–$500 |
| Wrong part diagnosis → second roll | $400–$1,000 |
| Unplanned downtime (commercial building) | $500–$2,000/hr |
| Preventable with accurate fault isolation | ~60–80% of cases |

**The gap: Carrier monitors temperature. Nobody monitors *why* temperature is wrong.**

---

## The Solution

**Thermo-Twin** is an unsupervised multi-variate AI that monitors the **thermodynamic harmony** of a Carrier HVAC unit across 4 sensors simultaneously.

A healthy HVAC unit has fixed thermodynamic relationships between its sensors. When one component fails, that harmony breaks — in a specific, identifiable pattern. Thermo-Twin detects the break, identifies the pattern, names the component, and prescribes the fix.

```
4 Sensor Streams
        │
        ▼
  Sliding Window
  Preprocessor
  (50 samples, 50% overlap → 200-dim vector)
        │
        ▼
  Neural Autoencoder
  [200 → 128 → 64 → 8 → 64 → 128 → 200]
  Trained on NORMAL data only
        │
        ▼
  Reconstruction Error
  → Severity Score (0–100)
        │
        ▼
  SHAP GradientExplainer
  → 4-sensor attribution (%)
        │
        ▼
  Prescriptive Rules Engine
  → Fault type + technician dispatch instruction
        │
        ▼
  Alert API  +  Operator Dashboard
```

---

## Core Innovation

### 1. Thermodynamic Harmony Monitoring
Four sensors are watched **together**, not in isolation. The alarm fires when the *relationship* between sensors breaks — not when any single threshold is crossed.

### 2. Unsupervised Learning
No labeled fault data required. The model learns what "normal harmony" looks like and flags any deviation. Works on day one with zero historical failures.

### 3. Component-Level SHAP Attribution
Every alert includes a breakdown of which sensor drove the anomaly:
```
Severity: 87/100  →  STOP UNIT
├── Supply Air Temp:       85%  ← dominant
├── Discharge Pressure:     7%
├── Compressor Power:       4%
└── Fan RPM:                4%
ROOT CAUSE: Refrigerant Leak in Evaporator Coil
```

### 4. Prescriptive Output (Good → Mind-Blowing)
The system doesn't just flag an anomaly — it tells the technician what to do:

| Standard BMS | Thermo-Twin |
|---|---|
| "High Temp Warning" | "Condenser Fan Degradation" |
| — | "Efficiency down 15%, wasting 2.3 kWh/hr" |
| "Call technician" | "Dispatch with 5HP Fan Motor. ETA: 2 hrs." |

### 5. Severity Score (0–100)

| Score | Level | Action |
|---|---|---|
| 0–40 | Normal | Log only |
| 41–70 | Warning | Notify operator |
| 71–100 | Critical | **Stop unit** |

### 6. Isolation Forest Fallback
A trained Isolation Forest model runs in parallel (ROC-AUC 0.968). If the autoencoder misbehaves during demo, swap in 2 minutes — same API, same output format.

---

## How It Works — The Four Sensors

In a healthy HVAC unit these four streams move in thermodynamic harmony:

```
↑ Cooling Demand
    → ↑ Compressor Power (kW)      — baseline ~3.5 kW
    → ↑ Discharge Pressure (PSI)   — baseline ~250 PSI  [= 70 × power]
    → ↑ Fan RPM                    — baseline ~1200 RPM [= 340 × power]
    → ↓ Supply Air Temp (°C)       — baseline ~12 °C    [= 18 − 2 × power]
```

When a component fails, exactly one or two streams break harmony. The autoencoder's reconstruction error spikes on those streams. SHAP isolates which streams drove the spike.

---

## Fault Scenarios

### Fault 1 — Refrigerant Leak

| Sensor | Behavior |
|---|---|
| Compressor Power | HIGH — compressor works harder with no effect |
| Discharge Pressure | DROPS SUDDENLY — gas escaping, pressure falls |
| Fan RPM | Normal |
| Supply Air Temp | RISES SUDDENLY — unit not cooling |

```
SHAP Output  : "85% driven by Temp Rise"
Severity     : 100 / 100
Root Cause   : Refrigerant Leak in Evaporator Coil
Prescription : Dispatch with refrigerant recharge kit + leak detector
Impact       : Cooling efficiency down ~40%
```

---

### Fault 2 — Condenser Fan Failure

| Sensor | Behavior |
|---|---|
| Compressor Power | Rises GRADUALLY — overworking due to poor heat dissipation |
| Discharge Pressure | Rises GRADUALLY — heat not being removed |
| Fan RPM | DROPS ABRUPTLY — motor degrading or failed |
| Supply Air Temp | Rises GRADUALLY |

```
SHAP Output  : "47% Fan RPM drop + 26% Compressor Power"
Severity     : 100 / 100
Root Cause   : Condenser Fan Motor Degradation
Prescription : Dispatch with 5HP Fan Motor replacement
Impact       : Heat dissipation failure, compressor overload risk
```

---

### Fault 3 — Compressor Wear (Gradual Drift)

| Sensor | Behavior |
|---|---|
| Compressor Power | SLOWLY increases over hundreds of samples |
| Discharge Pressure | SLOWLY decreases |
| Fan RPM | Normal |
| Supply Air Temp | SLOWLY increases |

```
SHAP Output  : "62% Compressor Power creep + 30% Temp Rise"
Severity     : 91 / 100
Root Cause   : Progressive Compressor Mechanical Wear
Prescription : Schedule compressor replacement within 2 weeks
Impact       : Progressive efficiency loss, full failure imminent
```

---

## System Architecture

```
CARRIER-CHILLER-01 / CARRIER-VRF-UNIT-01
    │
    ├── compressor_power_kw      (indices   0–49  in feature vector)
    ├── discharge_pressure_psi   (indices  50–99)
    ├── fan_rpm                  (indices 100–149)
    └── supply_air_temp_c        (indices 150–199)
    │
    ▼
data/raw/generate_sensor_data.py
    20,000 samples | 3 fault types | 2 machine IDs
    │
    ▼
data/preprocess.py
    Sliding windows: 50 samples, 50% overlap
    Feature vector: 200-dim (4 streams × 50 samples)
    Scaler fitted on normal windows only
    Splits: 80% normal → train | 20% normal → val | all faults → test
    │
    ▼
model/train.py
    Autoencoder: 200→128→64→8→64→128→200 (PyTorch)
    Isolation Forest: 300 estimators (sklearn fallback)
    Threshold: val_mean + 2.5 × val_std = 0.1957
    │
    ▼
explainability/shap_explainer.py
    SHAP GradientExplainer (expected gradients)
    4-stream attribution → prescriptive rules → fault type
    │
    ▼
explainability/precompute_explanations.py
    3 scenarios pre-computed → demo_explanations.json
    │
    ▼
backend/app.py          [Phase 5 — TODO]
    POST /alert
    GET  /alerts
    POST /demo/<scenario>
    │
    ▼
dashboard/app.py        [Phase 6 — TODO]
    4-stream live signal plot
    Severity gauge (0–100)
    4-bar SHAP chart
    Fault type + prescription card
    Alert log table
    Demo trigger buttons
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Simulation | NumPy, Pandas |
| ML Core | PyTorch (Autoencoder) |
| ML Fallback | Scikit-learn (Isolation Forest) |
| Explainability | SHAP (GradientExplainer) |
| Backend API | Flask + Flask-CORS |
| Dashboard | Streamlit + Plotly |
| Visualization | Matplotlib |

---

## Validated Performance

Metrics from the trained model on the held-out test set (386 windows):

| Metric | Autoencoder | Isolation Forest |
|---|---|---|
| ROC-AUC | **0.985** | 0.968 |
| F1 Score (binary) | **0.978** | 0.898 |
| Precision | 1.000 | 0.983 |
| Recall | 0.958 | 0.827 |
| False Positives | **0** | 4 |

**Severity score distribution:**

| Fault Type | Mean Score | % Above 70 |
|---|---|---|
| Normal (n=103) | 29.6 | 0% |
| Refrigerant Leak (n=55) | 90.5 | 96.4% |
| Fan Failure (n=68) | 92.8 | 97.1% |
| Compressor Wear (n=160) | 73.8 | 65.6% |

Normal windows: **100% score below 40** (zero false positives).

---

## Strategic Fit — Carrier Ecosystem

| Carrier Platform | What It Monitors | Thermo-Twin Fit |
|---|---|---|
| BluEdge Elevate | Predictive maintenance SLAs for shipped products | Thermo-Twin = the AI diagnostic brain inside BluEdge |
| ClimaVision | Rooftop units sending fault codes to cloud | Thermo-Twin adds prescriptive layer on top of those fault codes |
| i-Vu / CCN | Commercial buildings (chillers, AHUs) | Direct deployment target for CARRIER-CHILLER-01 use case |
| InteliSense | Fuses indoor + outdoor coil sensor data | Identical multi-variate fusion approach — proven precedent |
| Digital Connectivity | Installs BACnet/IoT gateways at customer sites | Same deployment muscle, pointed at HVAC unit sensors |

**One-line carrier ROI argument:** Every avoided truck roll = $200–$500 saved. Every first-time fix (right part, right technician) = no second roll. Thermo-Twin makes both happen.

---

## Business Model

### Phase 1 — Reduce Carrier's Own Truck Roll Costs
Deploy on Carrier's field-service fleet. Technicians receive fault type + part prescription before dispatch.

| Metric | Before Thermo-Twin | After |
|---|---|---|
| Diagnosis time on-site | ~2 hours | ~0 min (pre-diagnosed) |
| Wrong-part dispatches | ~25% of calls | Near zero |
| Second truck rolls | ~15% of calls | Near zero |
| Cost per incident | $400–$1,000 | $200–$500 |

### Phase 2 — BluEdge Prescriptive Diagnostics SaaS
Package as a premium BluEdge tier sold to building operators and facility managers. Carrier already sells predictive maintenance as recurring revenue — Thermo-Twin is a new SKU in an existing commercial motion.

---

## Project Structure

```
HVAC-Manufacturing-Anomaly-Detection/
│
├── data/
│   ├── raw/
│   │   ├── generate_sensor_data.py   # HVAC synthetic data generation
│   │   └── synthetic_data.csv        # 20,000 rows, 4 streams, 3 fault types
│   ├── processed/
│   │   ├── train_windows.npz         # 412 normal windows (80%)
│   │   ├── val_windows.npz           # 103 normal windows (20%)
│   │   ├── test_windows.npz          # 386 windows (all 3 fault types + normal)
│   │   └── scaler.pkl                # StandardScaler fitted on train only
│   ├── preprocess.py                 # Sliding window pipeline
│   └── verify_preprocessing.py      # Sanity checks
│
├── model/
│   ├── autoencoder.py                # PyTorch autoencoder (200→128→64→8→...)
│   ├── train.py                      # Training loop + threshold calibration
│   ├── isolation_forest.py           # Fallback model
│   ├── threshold.py                  # Severity scoring utilities
│   ├── evaluate.py                   # Full evaluation report
│   └── checkpoints/
│       ├── autoencoder.pt            # Trained model
│       ├── isolation_forest.pkl      # Fallback model
│       └── threshold_config.json     # Threshold + severity config
│
├── explainability/
│   ├── shap_explainer.py             # SHAP GradientExplainer + prescriptive rules
│   ├── precompute_explanations.py    # Pre-compute 3 demo scenarios
│   ├── alert_payload.py              # Alert payload builder
│   └── demo_explanations.json        # Pre-computed SHAP outputs (ready for demo)
│
├── backend/                          # [Phase 5 — TODO]
│   └── app.py                        # Flask alert API
│
├── dashboard/                        # [Phase 6 — TODO]
│   └── app.py                        # Streamlit operator dashboard
│
├── requirements.txt
├── README.md
└── PHASES.md
```

---

## Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Generate HVAC sensor data  (run from data/raw/)
cd data/raw && python generate_sensor_data.py && cd ../..

# 3. Preprocess into sliding windows
python data/preprocess.py

# 4. Verify data integrity
python data/verify_preprocessing.py

# 5. Train autoencoder + isolation forest  (~5 min)
python model/train.py

# 6. Full evaluation report
python model/evaluate.py

# 7. Pre-compute SHAP demo explanations  (~2 min)
python explainability/precompute_explanations.py

# 8. Start alert backend  [Phase 5]
python backend/app.py

# 9. Launch dashboard  [Phase 6]
streamlit run dashboard/app.py
```

---

*Thermo-Twin — When thermodynamic harmony breaks, we name the component and send the right part.*
