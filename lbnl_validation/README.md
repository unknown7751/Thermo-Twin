# LBNL Real-World Validation Pipeline

## Overview

This directory contains the **Sim-to-Real Transfer Validation** pipeline for Thermo-Twin.  
The core idea: we trained our fault-detection autoencoder **entirely on synthetic HVAC data**, then validated it against **real building sensor data** from Lawrence Berkeley National Laboratory (LBNL).

---

## Training vs Validation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    THERMO-TWIN PIPELINE                        │
├────────────────────────┬────────────────────────────────────────┤
│   TRAINING (Synthetic) │   VALIDATION (Real - LBNL)            │
├────────────────────────┼────────────────────────────────────────┤
│ Source: Carrier HVAC   │ Source: LBNL RTU Dataset               │
│        simulator       │        (real building, Aug 2017 -      │
│                        │         Feb 2018)                      │
│                        │                                        │
│ Samples: ~2,000 windows│ Samples: 1,208 fault windows           │
│ Labels: normal +       │ Labels: ALL fault (Ground Truth = 1    │
│   refrigerant_leak,    │         for all 30,240 data points)    │
│   fan_failure,         │                                        │
│   compressor_wear      │ Purpose: Prove synthetic model         │
│                        │   generalizes to real-world faults     │
│ Used for: Autoencoder  │                                        │
│   weight training,     │ Combined test set:                     │
│   threshold calibration│   103 synthetic normals +              │
│                        │   1,208 LBNL real faults               │
└────────────────────────┴────────────────────────────────────────┘
```

### Why Synthetic Training + Real Validation?

1. **Real fault data is scarce** -- buildings rarely fail, and when they do, sensor data is often unlabeled or incomplete.
2. **Synthetic data is controllable** -- we can generate exact fault types (refrigerant leak, fan failure, compressor wear) with known ground truth.
3. **The acid test** -- if a model trained on synthetic data can detect *real* building faults it has never seen, it proves genuine generalization, not memorization.

---

## Results

| Metric | Value |
|---|---|
| **F1 Score** | **0.9996** |
| **ROC-AUC** | **1.0000** |
| **Recall** | **1.0000** (all 1,208 real faults detected) |
| **Precision** | **0.9992** (only 1 false positive out of 103 normals) |
| **Fault Detection Rate** | **100%** |
| **Faults with Severity >= 70** | **100%** |
| **Normal windows <= 40** | **99.0%** |

### Confusion Matrix

```
                      Predicted Normal    Predicted Fault
True Normal (synth)         102                 1
True Fault  (LBNL)            0              1208
```

**Key takeaway:** The synthetic-trained model achieves **perfect recall** on real building faults with near-zero false positives.

---

## Dataset Details

### Synthetic Training Data (Carrier HVAC Simulator)

| Property | Value |
|---|---|
| Source | Physics-based HVAC simulator |
| Sensors | 4 streams: compressor power, discharge pressure, fan RPM, supply air temp |
| Faults | Refrigerant leak, fan failure, compressor wear |
| Window size | 50 timesteps x 4 sensors = 200 features |
| Preprocessing | StandardScaler fitted on normal data only |

### LBNL Real Building Data

| Property | Value |
|---|---|
| Source | LBNL Automated Fault Detection Dataset |
| File | `LBNL_Dataset/RTU.csv` |
| System | Rooftop Unit (RTU) serving a commercial building |
| Duration | August 27, 2017 -- February 18, 2018 |
| Rows | 30,240 (1-minute resolution) |
| Columns | 69 sensor channels |
| Ground Truth | 100% fault-labeled (entire dataset captures faulty operation) |

### Column Mapping (LBNL -> Thermo-Twin)

The LBNL RTU dataset has different sensor names than our synthetic schema. We map the 4 most relevant columns:

| LBNL Column | Thermo-Twin Column | Scaled Range |
|---|---|---|
| `RTU: Electricity` | `compressor_power_kw` | 0.5 -- 5.5 kW |
| `RTU: Circuit 1 Discharge Pressure` | `discharge_pressure_psi` | 150 -- 350 PSI |
| `RTU: Fan Electricity` | `fan_rpm` | 800 -- 1600 RPM |
| `RTU: Supply Air Temperature` | `supply_air_temp_c` | 8 -- 16 C |

MinMaxScaler is used to map LBNL raw ranges into synthetic data ranges, ensuring the autoencoder receives inputs in the same distribution it was trained on.

---

## Pipeline Scripts

Run these in order from the **project root** (`Thermo-Twin/`):

### Step 1: Data Exploration

```bash
python lbnl_validation/01_explore.py
```

Loads `LBNL_Dataset/RTU.csv`, prints shape, dtypes, missing values, fault distribution, and sensor value ranges. No files are created -- this is for understanding the data.

### Step 2: Column Mapping & Range Scaling

```bash
python lbnl_validation/02_map_columns.py
```

Maps RTU columns to the 4-sensor schema, handles missing values (forward-fill), scales to synthetic ranges, and saves:
- `data/real/lbnl_mapped.csv` -- 30,240 rows with 4 sensor columns + fault label
- `data/real/range_mappers.pkl` -- MinMaxScaler objects for future inference

### Step 3: Sliding Window Preprocessing

```bash
python lbnl_validation/03_preprocess.py
```

Creates 200-dim sliding windows (size=50, step=25), applies the **synthetic-trained StandardScaler**, and builds a combined test set:
- `data/processed/lbnl_fault_windows.npz` -- 1,208 fault windows
- `data/processed/lbnl_combined_test.npz` -- 1,311 windows (103 synthetic normals + 1,208 LBNL faults)
- `data/processed/lbnl_all_windows_raw.npz` -- raw unscaled windows

### Step 4: Evaluation (Sim-to-Real Transfer)

```bash
python lbnl_validation/04_evaluate.py
```

Loads the existing synthetic-trained autoencoder and threshold, runs inference on LBNL fault windows, computes precision/recall/F1/ROC-AUC, and saves:
- `model/checkpoints/lbnl_evaluation_results.json` -- all metrics (consumed by dashboard)

---

## Architecture

```
Synthetic Normal Data                  LBNL Real Fault Data
       |                                      |
  [Autoencoder]                          [Column Mapping]
  trained on normal                    RTU.csv -> 4 sensors
  operations only                            |
       |                              [Range Scaling]
  [Threshold]                        MinMax to synthetic ranges
  mean + 2.5*sigma                          |
  on val errors                       [Sliding Windows]
       |                             50-step, 200-dim vectors
       |                                    |
       +------>  [Inference]  <-------------+
                     |
              Reconstruction Error
                     |
              Error > Threshold?
                /          \
           Normal          Fault
          (sev < 40)     (sev > 70)
```

---

## Reproducing Results

```bash
# From project root (Thermo-Twin/)
# Ensure venv is activated and requirements are installed

# 1. Explore the data
python lbnl_validation/01_explore.py

# 2. Map columns and scale
python lbnl_validation/02_map_columns.py

# 3. Create windows and combined test set
python lbnl_validation/03_preprocess.py

# 4. Run evaluation
python lbnl_validation/04_evaluate.py

# 5. View results in dashboard
streamlit run dashboard/app.py
```

---

## Citation

The LBNL dataset is from:

> **LBNL Automated Fault Detection and Diagnostics Dataset**  
> Lawrence Berkeley National Laboratory  
> U.S. Department of Energy  
> Building Technology and Urban Systems Division  
> Link: https://www.kaggle.com/datasets/claytonmiller/lbnl-automated-fault-detection-for-buildings-data?resource=download
