# Thermo-Twin — Mathematics & Formulae Reference

All equations, loss functions, scoring functions, and statistical methods used across the project, organized by pipeline stage.

---

## Table of Contents

1. [Synthetic Data Generation](#1-synthetic-data-generation)
2. [Sliding Window Preprocessing](#2-sliding-window-preprocessing)
3. [Commissioning Baselines](#3-commissioning-baselines)
4. [StandardScaler Normalization](#4-standardscaler-normalization)
5. [Autoencoder Architecture & Forward Pass](#5-autoencoder-architecture--forward-pass)
6. [Physics-Informed Loss Function](#6-physics-informed-loss-function)
7. [Reconstruction Error](#7-reconstruction-error)
8. [Anomaly Threshold Calibration](#8-anomaly-threshold-calibration)
9. [Severity Score](#9-severity-score)
10. [Dynamic Threshold (Runtime)](#10-dynamic-threshold-runtime)
11. [MC-Dropout Uncertainty Estimation](#11-mc-dropout-uncertainty-estimation)
12. [SHAP Feature Attribution](#12-shap-feature-attribution)
13. [Signal Normalization (Dashboard)](#13-signal-normalization-dashboard)
14. [Energy Cost Attribution](#14-energy-cost-attribution)

---

## 1. Synthetic Data Generation

### 1.1 Normal Demand Signal

A superposition of two sinusoids (daily + weekly load cycles) with Gaussian noise models realistic HVAC demand:

```
d(t) = 0.4 · sin(2π · 0.02 · t)
     + 0.15 · sin(2π · 0.007 · t)
     + ε_d,    ε_d ~ N(0, 0.05²)
```

### 1.2 Compressor Power

```
P_comp(t) = clip( 3.5 + d(t) + ε_p + δ(t),  2.0,  6.0 )   kW

ε_p ~ N(0, 0.08²)
δ(t) = slow thermal drift term
```

### 1.3 Thermodynamic Sensor Relations (Normal Operation)

These are the physics constraints that define "harmony." The model learns to reconstruct these; violations signal faults.

```
P_disc(t)  =  70 · P_comp(t)  +  ε_disc,    ε_disc ~ N(0, 4²)       PSI

RPM_fan(t) = 340 · P_comp(t)  +  ε_fan,     ε_fan  ~ N(0, 30²)      RPM

T_supply(t) = 18  −  2 · P_comp(t)  +  ε_temp,  ε_temp ~ N(0, 0.3²)  °C
```

### 1.4 Fault Signal Injection

**Refrigerant Leak** — sudden pressure drop + temperature spike:

```
P_disc_fault(t)   = P_disc(t)   · (1 − α_leak),     α_leak ∈ U(0.30, 0.45)
T_supply_fault(t) = T_supply(t) + β_leak,            β_leak ∈ U(5, 9)     °C
```

**Condenser Fan Failure** — RPM collapse (immediate) + thermal ramp (gradual):

```
RPM_fan_fault(t) = RPM_fan(t) · (1 − α_fan),        α_fan  ∈ U(0.70, 0.90)

P_comp_fault(t)  = P_comp(t) + r_comp · t/T_fault,  r_comp ∈ U(1.0, 1.8)  kW
P_disc_fault(t)  = P_disc(t) + r_disc · t/T_fault,  r_disc ∈ U(40, 70)    PSI
T_supply_fault(t)= T_supply(t)+ r_temp · t/T_fault,  r_temp ∈ U(3, 6)     °C
```

**Compressor Wear** — all-stream progressive drift over long duration:

```
P_comp_fault(t)  = P_comp(t) + r_p · t/T_fault,     r_p    ∈ U(1.5, 2.5)  kW
P_disc_fault(t)  = P_disc(t) − r_d · t/T_fault,     r_d    ∈ U(30, 60)    PSI
T_supply_fault(t)= T_supply(t)+ r_t · t/T_fault,    r_t    ∈ U(2.0, 4.5)  °C

T_fault ∈ U(500, 700) samples  (50–70 seconds of gradual drift)
```

---

## 2. Sliding Window Preprocessing

A single training/inference sample is a **multi-variate time window** flattened into a feature vector.

### 2.1 Window Construction

```
W_i = [ P_comp[i·s : i·s + w],
         P_disc[i·s : i·s + w],
         RPM_fan[i·s : i·s + w],
         T_supply[i·s : i·s + w] ]   ∈  ℝ^(4×w)

w = 50   (window size, samples)
s = 25   (step size, 50% overlap)
```

### 2.2 Feature Vector

The four streams are concatenated into a single 200-dimensional vector:

```
x_i = flatten(W_i)  ∈  ℝ^200

x_i = [ P_comp[0..49] | P_disc[50..99] | RPM_fan[100..149] | T_supply[150..199] ]
```

### 2.3 Window Label (Majority Vote)

```
label(W_i) = mode({ fault_label[t]  :  t ∈ [i·s,  i·s + w) })
```

---

## 3. Commissioning Baselines

Per-unit thermodynamic ratio constants are estimated from normal-operation data only, using the median (robust to outliers) and ordinary least squares.

### 3.1 Discharge Pressure Ratio

```
k_disc = median{ P_disc(t) / P_comp(t)  :  ∀t where label = normal }
       ≈ 70.0
```

### 3.2 Fan RPM Ratio

```
k_fan = median{ RPM_fan(t) / P_comp(t)  :  ∀t where label = normal }
      ≈ 340.2
```

### 3.3 Supply Temperature Regression

OLS slope and intercept of  T_supply ~ P_comp:

```
k_temp_b = Cov(T_supply, P_comp) / Var(P_comp)
         ≈ −2.01

k_temp_a = mean(T_supply) − k_temp_b · mean(P_comp)
         ≈ 18.05

→  T_supply ≈ k_temp_a + k_temp_b · P_comp
            = 18.05 − 2.01 · P_comp
```

---

## 4. StandardScaler Normalization

Fitted **only on training windows** (normal data) to prevent leakage of fault statistics.

### 4.1 Fit (training set)

```
μ_j = (1/N) Σ_{i=1}^{N} x_{ij}          (feature mean)
σ_j = sqrt( (1/N) Σ_{i=1}^{N} (x_{ij} − μ_j)² )   (feature std)
```

### 4.2 Transform (train / val / test)

```
x̃_{ij} = (x_{ij} − μ_j) / σ_j
```

All model inputs, physics-loss computations, and SHAP explanations operate in this normalized space.

---

## 5. Autoencoder Architecture & Forward Pass

### 5.1 Layer Dimensions

```
Encoder:
  h₁ = ReLU( W₁ · x̃  + b₁ )              W₁ ∈ ℝ^{128×200}
  h₁ = Dropout(h₁, p=0.1)
  h₂ = ReLU( W₂ · h₁ + b₂ )              W₂ ∈ ℝ^{64×128}
  h₂ = Dropout(h₂, p=0.1)
  z  = ReLU( W₃ · h₂ + b₃ )              W₃ ∈ ℝ^{8×64}    ← bottleneck

Decoder:
  h₃ = ReLU( W₄ · z  + b₄ )              W₄ ∈ ℝ^{64×8}
  h₃ = Dropout(h₃, p=0.1)
  h₄ = ReLU( W₅ · h₃ + b₅ )              W₅ ∈ ℝ^{128×64}
  h₄ = Dropout(h₄, p=0.1)
  x̂  =       W₆ · h₄ + b₆               W₆ ∈ ℝ^{200×128}  ← reconstruction

Total trainable parameters: 69,200
```

### 5.2 Denoising Input

Gaussian noise is added to the input during training only:

```
x̃_noisy = x̃ + ε,    ε ~ N(0, σ_noise²),    σ_noise = 0.02
```

The model reconstructs the clean x̃ from the noisy input — forcing it to learn the underlying thermodynamic manifold rather than memorizing inputs.

---

## 6. Physics-Informed Loss Function

### 6.1 Total Loss

```
L_total = L_MSE  +  λ · L_physics

λ = 0.1
```

### 6.2 MSE Reconstruction Loss

```
L_MSE = (1/200) · Σ_{j=1}^{200} (x̂_j − x̃_j)²
```

### 6.3 Physics Loss

Penalizes reconstructions that violate the commissioning baseline ratios. All terms are in normalized (StandardScaler) space.

**Normalized ratio factors:**

```
k₁ = k_disc · (σ_comp / σ_disc)       (discharge pressure constraint)
k₂ = k_fan  · (σ_comp / σ_fan)        (fan RPM constraint)
k₃ = k_temp_b · (σ_comp / σ_temp)     (temperature constraint)
```

where σ_comp, σ_disc, σ_fan, σ_temp are per-feature standard deviations from the scaler.

**Physics loss terms:**

Let comp̂, disĉ, fan̂, temp̂ denote the **mean** of the 50 reconstructed values for each stream:

```
comp̂  = (1/50) · Σ x̂_{j},   j ∈ [0, 49]
disĉ  = (1/50) · Σ x̂_{j},   j ∈ [50, 99]
fan̂   = (1/50) · Σ x̂_{j},   j ∈ [100, 149]
temp̂  = (1/50) · Σ x̂_{j},   j ∈ [150, 199]

L_physics = ( disĉ − k₁·comp̂ )²
          + ( fan̂  − k₂·comp̂ )²
          + ( temp̂ − k₃·comp̂ )²
```

This term is zero only when the reconstructed streams satisfy the exact thermodynamic ratios learned at commissioning — physically grounding the model in refrigeration physics.

---

## 7. Reconstruction Error

Used as the raw anomaly score before severity mapping.

### 7.1 Per-Sample MSE

```
e_i = (1/200) · Σ_{j=1}^{200} (x̂_{ij} − x̃_{ij})²    ∈  ℝ⁺
```

### 7.2 Interpretation

- Normal windows: e_i ≈ val_mean = **0.1478** (model reconstructs well)
- Fault windows: e_i >> threshold (model fails to reconstruct unfamiliar patterns)

---

## 8. Anomaly Threshold Calibration

### 8.1 Validation Error Distribution

Compute reconstruction errors on all 103 validation windows (normal only):

```
μ_val = (1/N_val) · Σ e_i              = 0.1478
σ_val = sqrt( (1/N_val) · Σ (e_i − μ_val)² )  = 0.0208
```

### 8.2 Static Threshold

Set at n_sigma = 2.5 standard deviations above the validation mean:

```
τ_static = μ_val + 2.5 · σ_val
         = 0.1478 + 2.5 × 0.0208
         = 0.1997
```

This places the threshold at approximately the **99.4th percentile** of the normal error distribution (assuming approximate normality), giving a very low false positive rate.

### 8.3 99th Percentile Anomaly Reference

Used to anchor the upper end of the severity scale:

```
p99_anomaly = percentile({ e_i : label_i ∈ faults }, 99)
            = 19.38
```

---

## 9. Severity Score

Maps raw reconstruction error e ∈ ℝ⁺ to an integer score S ∈ [0, 100].

### 9.1 Normal Range  (e ≤ τ)

Linear interpolation from 0 to 40:

```
S = floor( 40 · e / τ )

S ∈ [0, 40]   ↔   Normal operation
```

### 9.2 Anomaly Range  (e > τ)

Logarithmic scaling from 41 to 100, anchored at p99:

```
S = floor( 41 + 59 · min( log(e / τ) / log(p99 / τ),  1.0 ) )

S ∈ [41, 70]   ↔   Warning
S ∈ [71, 100]  ↔   Critical
```

**Why log scale?** Reconstruction errors for faults span orders of magnitude (0.2 to 20+). A linear scale would compress most faults into a narrow band. The log scale gives resolution at both low and high anomaly intensities.

### 9.3 Severity Boundary Summary

```
e ≤ τ          →   S ∈ [0,  40]   NORMAL   (log only)
τ < e ≤ e_warn →   S ∈ [41, 70]   WARNING  (notify operator)
e > e_warn     →   S ∈ [71, 100]  CRITICAL (stop unit)
```

---

## 10. Dynamic Threshold (Runtime)

`ThresholdManager` continuously adapts the threshold as new normal-operation data arrives.

### 10.1 Rolling Buffer Update

```
Buffer B = deque of reconstruction errors, maxlen = 500

Update rule:
  if S(e_new) < 40:          ← classified as normal by current threshold
      B.append(e_new)
```

### 10.2 Dynamic Threshold Computation

```
τ_dynamic = percentile(B, 95)    if |B| ≥ 50
τ_dynamic = τ_static             otherwise (cold start fallback)
```

Using the 95th percentile (rather than mean + n·σ) makes the dynamic threshold non-parametric — robust to non-Gaussian error distributions in real deployments.

### 10.3 Persistence

```
State saved to threshold_state.json on every update:
  { "buffer": [...], "current_threshold": τ_dynamic, "n_samples": |B| }
```

---

## 11. MC-Dropout Uncertainty Estimation

Instead of a single deterministic inference, N stochastic forward passes are run with dropout active (train mode), producing a distribution over severity scores.

### 11.1 Stochastic Forward Passes

```
For k = 1, ..., N   (N = 10):
    x̂^(k) = f_θ(x̃;  dropout active)
    e^(k)  = (1/200) · ‖ x̂^(k) − x̃ ‖²
    S^(k)  = severity_score(e^(k))

N = 10 passes
```

### 11.2 Mean and Uncertainty

```
S̄ = (1/N) · Σ_{k=1}^{N} S^(k)           (mean severity)

σ_S = sqrt( (1/N) · Σ_{k=1}^{N} (S^(k) − S̄)² )   (std of severity)

uncertainty = 1.96 · σ_S                  (95% confidence interval half-width)

confidence_pct = 100 · (1 − σ_S / max(S̄, 1))
```

### 11.3 Action Override Rule

```
if S̄ > 71  AND  uncertainty > 20:
    action = "INVESTIGATE"    (high severity but uncertain — verify before stopping)
elif S̄ > 71:
    action = "STOP UNIT"      (high severity, high confidence)
elif S̄ > 41:
    action = "WARNING"
else:
    action = "NORMAL"
```

---

## 12. SHAP Feature Attribution

### 12.1 GradientExplainer Setup

The autoencoder is wrapped in a scalar MSE output head:

```
f_mse(x̃) = (1/200) · ‖ AE(x̃) − x̃ ‖²   :  ℝ^200 → ℝ

Background set D = { x̃_i : i ∈ normal_train, |D| = 200 }
```

SHAP GradientExplainer computes the expected gradient:

```
φ_j(x̃) = E_{x̃' ~ D}[ (x̃_j − x̃'_j) · ∂f_mse(αx̃ + (1-α)x̃') / ∂x̃_j ]

where α ~ U(0, 1)   (integrated gradient approximation)
```

φ_j is the SHAP value for feature j — the marginal contribution of feature j to the anomaly score relative to the background distribution.

### 12.2 Stream-Level Attribution

The 200 feature-level SHAP values are aggregated by stream:

```
Φ_comp  = Σ |φ_j|,    j ∈ [0,   49]
Φ_disc  = Σ |φ_j|,    j ∈ [50,  99]
Φ_fan   = Σ |φ_j|,    j ∈ [100, 149]
Φ_temp  = Σ |φ_j|,    j ∈ [150, 199]

Φ_total = Φ_comp + Φ_disc + Φ_fan + Φ_temp
```

### 12.3 Normalized Percentages

```
pct_comp = 100 · Φ_comp / Φ_total
pct_disc = 100 · Φ_disc / Φ_total
pct_fan  = 100 · Φ_fan  / Φ_total
pct_temp = 100 · Φ_temp / Φ_total

pct_comp + pct_disc + pct_fan + pct_temp = 100%
```

### 12.4 Prescriptive Fault Rules (SHAP Thresholds)

```
if pct_temp > 50%  →  Refrigerant Leak        (temp dominant)
if pct_fan  > 35%  →  Condenser Fan Failure   (fan dominant)
if pct_comp > 45%  →  Compressor Wear         (compressor dominant)
else               →  Unknown Fault           (name dominant stream)
```

---

## 13. Signal Normalization (Dashboard)

The dashboard renders all four sensor streams on a shared axis [0, 1] for visual comparison. Each stream is normalized using fixed physical operating ranges:

```
x_norm = (x − x_min) / (x_max − x_min)
```

| Stream | x_min | x_max | Unit |
|---|---|---|---|
| Compressor Power | 2.0 | 6.5 | kW |
| Discharge Pressure | 130 | 460 | PSI |
| Fan RPM | 600 | 2200 | RPM |
| Supply Air Temp | 4 | 18 | °C |

These ranges are chosen to cover both normal operating ranges and worst-case fault excursions — ensuring faults are visually prominent on the chart without clipping.

---

## 14. Energy Cost Attribution

### 14.1 Base Cost Formulas

```
extra_power_kWh_per_hr  = defined per fault type (see table below)

cost_per_day_INR   = extra_power_kWh_per_hr × 24 × 8.0     (₹8 / kWh)
cost_per_day_USD   = extra_power_kWh_per_hr × 24 × 0.12    ($0.12 / kWh)

cost_per_month_INR = cost_per_day_INR × 30

payback_days       = part_cost_INR / cost_per_day_INR
```

### 14.2 Fault Energy Profiles

| Fault | Efficiency Loss | Extra Load (kWh/hr) | Part Cost (INR) |
|---|---|---|---|
| Refrigerant Leak | 40% | 9.6 | ₹5,000 |
| Condenser Fan Failure | 15% | 11.0 | ₹8,000 |
| Compressor Wear | 20% | 16.0 | ₹45,000 |

### 14.3 Example — Refrigerant Leak

```
extra_power     = 9.6 kWh/hr
cost_per_day    = 9.6 × 24 × 8.0   = ₹1,843.2 / day
cost_per_month  = 1,843.2 × 30     = ₹55,296  / month
payback_days    = 5,000 / 1,843.2  =  2.7 days
```

### 14.4 SHAP-Weighted Cost Breakdown

Each sensor stream's share of the energy cost is weighted by its SHAP attribution:

```
cost_j = cost_per_day_INR × pct_j / 100

e.g. for Refrigerant Leak:
  cost_temp = ₹1,843.2 × 0.918 = ₹1,692 / day   (supply air temp, 91.8%)
  cost_comp = ₹1,843.2 × 0.032 =   ₹59 / day    (compressor, 3.2%)
  cost_disc = ₹1,843.2 × 0.020 =   ₹37 / day    (discharge pressure, 2.0%)
  cost_fan  = ₹1,843.2 × 0.030 =   ₹55 / day    (fan RPM, 3.0%)
```

This lets an operator understand not just the total cost of inaction, but which failing component is responsible for most of the energy waste.

---

*All constants (70×, 340×, 18−2×) are learned from real thermodynamic principles of vapor-compression refrigeration cycles. The physics loss and commissioning baselines encode these same constants as soft constraints during training.*
