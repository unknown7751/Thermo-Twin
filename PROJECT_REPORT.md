# Thermo-Twin: AI-Powered HVAC Fault Detection & Prescriptive Diagnostics

## Project Report

---

## 1. Executive Summary

**Thermo-Twin** is an end-to-end AI system for real-time HVAC fault detection, root-cause identification, and prescriptive maintenance recommendations. Built for Carrier-class commercial HVAC equipment, it combines a denoising autoencoder with physics-informed constraints, SHAP explainability, and MC-Dropout uncertainty estimation to detect and diagnose faults in rooftop units (RTUs), chillers, and VRF systems.

### Key Achievements

| Metric | Generated Test |
|---|---|---|
| **F1 Score** | **0.9766** |
| **ROC-AUC** | **0.9841** |
| **Recall** | 95.76% |
| **Precision** | 99.63% |
| **False Positive Rate** | 0.97% |

The model was trained on normal generated HVAC windows and evaluated on held-out generated windows containing normal operation and three injected fault types.

---

## 2. Problem Statement

Commercial HVAC systems account for **40% of building energy consumption**. Faults like refrigerant leaks, fan failures, and compressor wear often go undetected for weeks, causing:

- **Energy waste**: 15-40% efficiency loss per fault
- **Equipment damage**: cascading failures from undetected degradation
- **Downtime cost**: Rs. 1,800-3,000/day in excess energy per unit
- **Safety risk**: critical cooling failures in hospitals and cold chains

Traditional rule-based fault detection (e.g., static temperature thresholds) produces high false alarm rates and cannot distinguish fault types or quantify severity.

### Our Solution

An unsupervised anomaly detection system that:
1. Learns "normal" HVAC behavior from 4 sensor streams
2. Detects deviations as faults without requiring labeled fault data for training
3. Identifies the specific fault type using SHAP attribution
4. Provides actionable prescriptive maintenance recommendations
5. Adapts to different operational contexts (hospital vs. warehouse)

---

## 3. System Architecture

```
                                    THERMO-TWIN ARCHITECTURE
                                    
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        DATA LAYER                                      │
  │                                                                        │
  │  Synthetic Data Generator ──> Preprocessing ──> Sliding Windows        │
  │  (Carrier HVAC Simulator)     (StandardScaler)   (50-step x 4-sensor   │
  │                                                   = 200-dim vectors)   │
  │                                                                        │
  │  Generated Fault Data ──> Preprocessing ──> Standard Scaling           │
  │  (4 sensor streams)       (window builder)   (train normals only)      │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼────────────────────────────────────────┐
  │                        MODEL LAYER                                     │
  │                                                                        │
  │  ┌─────────────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
  │  │   Denoising         │   │  Isolation        │   │  Physics-       │  │
  │  │   Autoencoder       │   │  Forest           │   │  Informed Loss  │  │
  │  │  200→128→64→8       │   │  (300 trees,      │   │  (Thermo ratios │  │
  │  │  →64→128→200        │   │   MVP fallback)   │   │   as soft       │  │
  │  │  + MC-Dropout       │   │                   │   │   constraints)  │  │
  │  └─────────┬───────────┘   └──────────────────┘   └─────────────────┘  │
  │            │                                                            │
  │            ▼                                                            │
  │  ┌─────────────────────────────────────────────────────────────┐        │
  │  │  Threshold Calibration (mean + 2.5σ = 0.200071)            │        │
  │  │  Severity Scoring (0-100, log-scale for anomalies)          │        │
  │  │  Dynamic Threshold Manager (rolling P95, 500-sample buffer) │        │
  │  └─────────────────────────────────────────────────────────────┘        │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼────────────────────────────────────────┐
  │                     INTELLIGENCE LAYER                                  │
  │                                                                        │
  │  ┌──────────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
  │  │  SHAP GradientExp    │   │  MC-Dropout       │   │  Prescriptive  │  │
  │  │  (200 background     │   │  Uncertainty      │   │  Rules Engine  │  │
  │  │   windows, seed=42)  │   │  (10 stochastic   │   │  (fault type → │  │
  │  │                      │   │   forward passes) │   │   part + cost) │  │
  │  │  Per-stream %:       │   │  Confidence %:    │   │  Energy Impact │  │
  │  │  "Pressure 51%,      │   │  "92% ±8"        │   │  (kWh/day/INR) │  │
  │  │   Temp 35%"          │   │                   │   │                │  │
  │  └──────────────────────┘   └──────────────────┘   └────────────────┘  │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
  ┌────────────────────────────────▼────────────────────────────────────────┐
  │                     APPLICATION LAYER                                   │
  │                                                                        │
  │  ┌──────────────────────┐   ┌──────────────────────────────────────┐   │
  │  │  Flask Alert Backend │   │  Streamlit Operator Dashboard        │   │
  │  │  (REST API, port     │   │  - Live sensor charts (Plotly)       │   │
  │  │   5000, in-memory    │   │  - Severity gauge + classification   │   │
  │  │   alert store)       │   │  - SHAP attribution radar           │   │
  │  │                      │   │  - Energy cost calculator            │   │
  │  │  POST /alert         │   │  - Severity profiles (hospital/     │   │
  │  │  GET  /alerts        │   │    warehouse/cold_chain/office)      │   │
  │  │  POST /demo/<scen>   │   │  - Generated evaluation metrics    │   │
  │  └──────────────────────┘   └──────────────────────────────────────┘   │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Pipeline

### 4.1 Synthetic Training Data

**Source**: Physics-based HVAC simulator modeling Carrier commercial equipment.

**Four sensor streams** capture the core thermodynamic behavior:

| Sensor | Range | Physical Meaning |
|---|---|---|
| `compressor_power_kw` | 0.5 -- 5.5 kW | Electrical draw of the compressor motor |
| `discharge_pressure_psi` | 150 -- 350 PSI | Refrigerant pressure at compressor outlet |
| `fan_rpm` | 800 -- 1600 RPM | Condenser fan rotational speed |
| `supply_air_temp_c` | 8 -- 16 °C | Cooled air temperature delivered to zones |

**Fault injection**: Three fault types are injected into simulated normal operation:

| Fault Type | Physical Effect | SHAP Fingerprint |
|---|---|---|
| **Refrigerant Leak** | Pressure drop, temperature rise, reduced cooling | `supply_air_temp` dominant (>50%) |
| **Fan Failure** | RPM collapse, compressor overload, head pressure rise | `fan_rpm` dominant (>35%) |
| **Compressor Wear** | Power creep, efficiency degradation, gradual temp rise | `compressor_power` dominant (>45%) |

**Preprocessing pipeline** (`data/preprocess.py`):
1. Sliding windows: size=50, step=25 → 200-dimensional feature vectors
2. Per-stream concatenation: `[power₀..₄₉, pressure₀..₄₉, rpm₀..₄₉, temp₀..₄₉]`
3. StandardScaler fitted on normal data only
4. Per-unit commissioning baselines for physics-informed loss calibration
5. Split: 80% normal → train, 20% normal + all faults → test

### 4.2 Generated Test Data

The test data is generated by `data/raw/generate_sensor_data.py` and transformed by `data/preprocess.py`.

| Split | Contents | Windows |
|---|---|---:|
| Train | Normal generated windows | 412 |
| Validation | Normal generated windows | 103 |
| Test | Normal and injected-fault windows | 386 |

---

## 5. Model Architecture

### 5.1 Denoising Autoencoder

```
Input (200) → Linear(128) → ReLU → Dropout(0.1)
           → Linear(64)  → ReLU → Dropout(0.1)
           → Linear(8)   → ReLU                    [Bottleneck]
           → Linear(64)  → ReLU → Dropout(0.1)
           → Linear(128) → ReLU → Dropout(0.1)
           → Linear(200)                            [Reconstruction]
```

- **Parameters**: ~54,000
- **Bottleneck**: 8 dimensions (25:1 compression ratio)
- **Training**: Denoising autoencoder with Gaussian noise (σ = 0.02)
- **Anomaly detection**: Windows with reconstruction error > threshold are flagged

### 5.2 Physics-Informed Loss

A custom regularization term penalizes reconstructions that violate thermodynamic relationships:

```
discharge_pressure ≈ 70 × compressor_power
fan_rpm            ≈ 340 × compressor_power
supply_air_temp    ≈ 18 - 2 × compressor_power
```

These ratios are calibrated per-unit from commissioning baselines and converted to normalized space for the loss function. The physics loss (λ = 0.1) acts as a soft constraint, ensuring the autoencoder's latent space respects HVAC thermodynamics.

### 5.3 Isolation Forest (MVP Fallback)

A 300-tree Isolation Forest trained on normal windows provides a secondary anomaly score. Used as a cross-validation mechanism and fallback for environments where neural inference is not feasible.

### 5.4 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 600 (early stopping patience=80) |
| Batch size | 16 |
| Learning rate | 1e-3 (ReduceLROnPlateau, factor=0.5) |
| Weight decay | 1e-4 |
| Noise std | 0.02 |
| Dropout | 0.1 |
| Physics lambda | 0.1 |
| Optimizer | Adam |

---

## 6. Threshold Calibration & Severity Scoring

### 6.1 Static Threshold

Calibrated on validation set (normal-only) reconstruction errors:

```
threshold = mean + 2.5σ = 0.1481 + 2.5 × 0.0208 = 0.200071
```

### 6.2 Dynamic Threshold (ThresholdManager)

In production, the threshold adapts to operational drift:
- Maintains a rolling buffer of the last 500 reconstruction errors from normal windows (severity < 40)
- Dynamic threshold = 95th percentile of buffer
- Falls back to static threshold when buffer has < 50 samples (cold start)
- Buffer persists across server restarts via JSON state file

### 6.3 Severity Scoring (0-100)

| Error Range | Severity Range | Scale | Interpretation |
|---|---|---|---|
| 0 ≤ error ≤ threshold | 0 -- 40 | Linear | Normal operation |
| threshold < error | 41 -- 100 | Logarithmic | Fault detected |

The log scale for anomalies prevents extreme errors from saturating the score, providing meaningful differentiation between moderate and severe faults.

---

## 7. Explainability & Prescriptive Diagnostics

### 7.1 SHAP Attribution

Uses `shap.GradientExplainer` with 200 normal background windows (seed=42) to decompose each window's reconstruction error into per-sensor contributions:

```
Example output:
  Compressor Power:     8.0%
  Discharge Pressure:  51.0%
  Fan RPM:              6.0%
  Supply Air Temp:     35.0%
  → "Anomaly driven by Pressure Drop (51%) and Temp Rise (35%)"
  → Fault Type: Refrigerant Leak
```

### 7.2 MC-Dropout Uncertainty

10 stochastic forward passes with dropout active provide:
- **Mean severity**: average across passes
- **Uncertainty**: ±1.96σ (95% confidence interval)
- **Confidence %**: `clip(100 - uncertainty, 0, 100)`
- **Action override**: If severity > 71 AND uncertainty > 20 → "INVESTIGATE" (instead of "STOP UNIT")

### 7.3 Prescriptive Rules Engine

Each fault type maps to a specific maintenance prescription:

| Fault | Prescription | Part Cost (INR) | Energy Waste |
|---|---|---|---|
| Refrigerant Leak | Dispatch with recharge kit + leak detector | ₹5,000 | 9.6 kWh/hr (₹1,843/day) |
| Condenser Fan Failure | Dispatch with 5HP fan motor replacement | ₹8,000 | 11.0 kWh/hr (₹2,112/day) |
| Compressor Wear | Schedule compressor replacement in 2 weeks | ₹45,000 | 16.0 kWh/hr (₹3,072/day) |

### 7.4 Severity Profiles

Context-aware thresholds for different building types:

| Profile | Warning | Critical | Use Case |
|---|---|---|---|
| Hospital | ≥ 25 | ≥ 45 | Zero tolerance for cooling failure |
| Cold Chain | ≥ 20 | ≥ 35 | Early warning essential |
| Commercial Office | ≥ 41 | ≥ 71 | Standard operations (default) |
| Warehouse | ≥ 55 | ≥ 80 | Tolerant of minor faults |

---

## 8. Results

### 8.1 Synthetic Data Evaluation

Evaluated on 386 generated windows: 103 normal, 55 refrigerant leak, 68 fan failure, and 160 compressor wear:

| Metric | Autoencoder | Isolation Forest |
|---|---|---|
| ROC-AUC | **0.9841** | 0.9681 |
| F1 Score | **0.9766** | 0.8983 |
| Precision | **0.9963** | 0.9832 |
| Recall | **0.9576** | 0.8269 |

**Per-fault detection rates (Autoencoder):**

| Fault Type | Detection Rate | Mean Severity |
|---|---|---|
| Refrigerant Leak | 100% | 92 |
| Fan Failure | 100% | 92.7 |
| Compressor Wear | 92.5% | 73.7 |
| Normal | 0.97% FPR | 29.7 |

### 8.2 Generated-Test Evaluation

The trained models were evaluated on the held-out generated test set:

| Metric | Value |
|---|---|
| **F1 Score** | **0.9766** |
| **ROC-AUC** | **0.9841** |
| **Precision** | **0.9963** |
| **Recall** | **0.9576** |
| **Normal windows severity ≤ 40** | 99.0% |

**Confusion Matrix (normal vs anomaly):**

```
                      Predicted Normal    Predicted Fault
True Normal (synth)         102                 1
True Anomaly                   12              271
```

---

## 9. Dashboard & Application Layer

### 9.1 Flask Alert Backend (`backend/app.py`)

REST API endpoints:
- `GET /health` — Healthcheck with dynamic threshold status
- `POST /alert` — Receive anomaly alert from inference layer
- `GET /alerts` — Return last 50 alerts, newest first
- `POST /demo/<scenario>` — Trigger pre-loaded demo scenario
- `GET /signal` — Live sensor signal data for chart rendering
- `GET /baselines` — Per-unit commissioning baselines

### 9.2 Streamlit Operator Dashboard (`dashboard/app.py`)

Interactive monitoring interface featuring:
- **Live sensor charts** with Plotly (compressor power, discharge pressure, fan RPM, supply air temp)
- **Severity gauge** with color-coded status (green/amber/red)
- **SHAP attribution breakdown** showing per-sensor fault contribution percentages
- **Energy cost calculator** with INR/USD daily/monthly impact
- **Severity profile selector** (hospital, cold chain, commercial, warehouse)
- **Alert log table** with color-coded severity scores

---

## 10. Project Structure

```
Thermo-Twin/
├── backend/
│   └── app.py                    # Flask REST API (alert backend)
├── dashboard/
│   ├── app.py                    # Streamlit operator dashboard
│   └── index.html                # Standalone HTML dashboard
├── data/
│   ├── preprocess.py             # Sliding window pipeline + commissioning baselines
│   ├── raw/                      # Synthetic CSV data
│   ├── processed/                # .npz windows + scaler.pkl
│   └── processed/                # Train, validation, and test windows
├── model/
│   ├── autoencoder.py            # Autoencoder architecture + load/save
│   ├── train.py                  # Training loop + physics loss + threshold calibration
│   ├── evaluate.py               # Binary + multi-class evaluation metrics
│   ├── threshold.py              # Threshold, severity scoring, MC-Dropout, profiles
│   ├── isolation_forest.py       # Isolation Forest (MVP fallback)
│   └── checkpoints/              # Model weights + threshold config + evaluation results
├── explainability/
│   ├── shap_explainer.py         # SHAP GradientExplainer + prescriptive rules
│   ├── alert_payload.py          # Alert builder with energy cost attribution
│   ├── precompute_explanations.py# Batch SHAP + MC-Dropout precomputation
│   └── demo_explanations.json    # Pre-computed demo scenario explanations
├── requirements.txt              # Python dependencies
└── PROJECT_REPORT.md             # This report
```

---

## 11. Technology Stack

| Component | Technology | Version |
|---|---|---|
| ML Framework | PyTorch | 2.11+ |
| Anomaly Scoring | Scikit-learn (Isolation Forest, StandardScaler) | 1.8+ |
| Explainability | SHAP (GradientExplainer) | 0.46+ |
| Backend API | Flask + Flask-CORS | 3.1+ |
| Dashboard | Streamlit + Plotly | 1.45+ |
| Data Processing | Pandas, NumPy | Latest |
| Language | Python | 3.12+ |

---

## 12. How to Run

```bash
# 1. Clone and activate environment
git clone <repository-url>
cd Thermo-Twin
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Preprocess synthetic data
python data/preprocess.py

# 3. Train autoencoder + calibrate threshold
python model/train.py

# 4. Evaluate on synthetic test set
python model/evaluate.py

# 5. Start backend + dashboard
python backend/app.py                  # Terminal 1 (port 5000)
streamlit run dashboard/app.py         # Terminal 2 (port 8501)
```

---

## 13. Future Work

- **Multi-unit fleet monitoring**: Scale to 100+ units with per-unit baseline tracking
- **Online learning**: Continuous model adaptation from production data streams
- **Edge deployment**: ONNX export for on-device inference at the RTU controller
- **Additional fault types**: Dirty condenser coils, stuck expansion valves, duct leakage
- **Integration with BMS**: Direct OPC-UA / BACnet data ingestion from building management systems

---

## 14. References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.
3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. *ICDM*.

---

*Report generated: May 2026*  
*Thermo-Twin v1.0 — Carrier HVAC AI Diagnostics Platform*
