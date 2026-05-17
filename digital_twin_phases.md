# Digital Twin — Development Phases

A 12-week roadmap for building a physics-grounded digital twin of an HVAC unit, from real-time sensor streaming to fleet-scale prognostics.

---

## Phase 0 — Foundation (Week 1)

**Goal:** Wire up the infrastructure everything else depends on.

### Components
- **Time-series database (InfluxDB)** — store all sensor history
- **WebSocket layer** — real-time sensor streaming
- **MQTT broker (Mosquitto)** — ingest live unit data
- **Unified data schema** — raw sensor → normalized twin state

**Deliverable:** Live sensor data flowing from the unit into a queryable store at <5ms latency.

---

## Phase 1 — Physics Model (Weeks 2–3)

**Goal:** Build the "ideal HVAC" simulator — what the unit should read under any condition.

### Components
- **Refrigerant cycle model (CoolProp library)**
  - Compressor: isentropic efficiency map
  - Condenser: heat rejection curve
  - Evaporator: cooling capacity model
  - Expansion valve: pressure drop equations
- **Inputs:** ambient temp, load demand, compressor speed
- **Outputs:** predicted values for all 4 sensors

**Deliverable:** Given any operating condition, the physics model outputs what the 4 sensors should read. This is the twin's "ground truth."

---

## Phase 2 — State Estimator (Week 4)

**Goal:** Continuously fuse physics predictions with real sensor readings to estimate hidden internal state.

### Components
- **Unscented Kalman Filter (UKF)**
  - State vector: `[refrigerant_charge, compressor_efficiency, fan_health, fouling_factor]`
  - Prediction step: physics model advances the state
  - Update step: real sensors correct the prediction
- **Divergence detector** — flags when twin ≠ real
- **Parameter estimator** — "refrigerant is at 87% nominal charge"

**Deliverable:** The twin now knows *why* sensors deviate, not just that they deviate. Replaces the current SHAP black-box attribution with physics-grounded causal explanation.

---

## Phase 3 — Prognostics / RUL Engine (Weeks 5–6)

**Goal:** Predict when each component will cross the failure threshold.

### Components
- **Degradation trajectory model** — LSTM on health index history
- **Particle filter** — uncertainty in RUL estimate
- **Per-component RUL:**
  - Compressor wear → days to critical efficiency loss
  - Fan motor → days to RPM floor breach
  - Refrigerant charge → days to recharge threshold
- **Outputs:** RUL estimate ± confidence interval

**Deliverable:** "Condenser fan will fail in 14 ± 3 days." Enables pre-emptive dispatch before any fault occurs.

---

## Phase 4 — What-If Simulator (Week 7)

**Goal:** Let operators test scenarios on the twin before touching the real unit.

### Components
- **Scenario API** — `POST /twin/simulate {compressor_speed: 80%, ambient_temp: 38°C}`
- **Physics model** — runs forward N hours in fast-time
- **Outputs:** predicted sensor trajectories + fault probability
- **UI panel** — sliders for each controllable variable

**Deliverable:** Operator can ask "if I reduce fan speed by 20% to save energy, what's the risk?" and get an answer in seconds without touching the real unit.

---

## Phase 5 — 3D Visualization (Weeks 8–9)

**Goal:** Build the interactive 3D unit interface.

### Components
- **3D HVAC model (Three.js / Babylon.js)**
  - Component meshes: compressor, condenser, evaporator, fans
  - Animated refrigerant flow lines (speed = flow rate)
  - Health glow: green → yellow → red per component
- **Hover tooltips** — real vs twin value + delta
- **Click-to-inspect** — opens full sensor history + RUL curve
- **Embedded in React frontend** — replaces Streamlit for the twin view

**Deliverable:** The 3D interactive unit shown in the mockup, running live in browser.

---

## Phase 6 — Fleet Twin (Week 10)

**Goal:** Scale from 1 unit to N units — each with its own twin instance.

### Components
- **Twin registry** — one twin state per physical unit ID
- **Fleet dashboard** — grid of mini-unit thumbnails, color-coded by health
- **Prioritized dispatch queue** — sorted by RUL ascending
- **Cross-unit anomaly correlation** — "3 units in Building A all showing refrigerant loss"
- **Fleet-level API** — `GET /fleet/health`, `GET /fleet/dispatch-queue`

**Deliverable:** A single screen showing all units, surfacing the most urgent one automatically.

---

## Phase 7 — Auto-Recalibration + Drift Detection (Week 11)

**Goal:** Keep the twin accurate as the real unit ages and conditions change.

### Components
- **Seasonal baseline updater** — re-fits physics model parameters each month
- **Drift alarm** — alerts when twin accuracy drops below 95%
- **Bayesian parameter update** — uses recent normal operation to recalibrate
- **Commissioning re-run trigger** — reset twin after a part replacement

**Deliverable:** The twin stays accurate over months and years without manual tuning.

---

## Phase 8 — Integration & Hardening (Week 12)

**Goal:** Production-ready system.

### Components
- **Auth layer (JWT)** — multi-tenant fleet access
- **REST + WebSocket API documentation**
- **Carrier BMS integration** — BACnet / Modbus adapter
- **Automated part ordering webhook** — SAP / ServiceNow
- **Load testing** — 100 concurrent twin instances

**Deliverable:** Ready for pilot deployment on a real Carrier fleet.

---

## Key Technologies

| Phase | Primary Tools |
|-------|---------------|
| 0 | InfluxDB, WebSocket, Mosquitto |
| 1 | CoolProp, Python/NumPy |
| 2 | UKF, Kalman filters |
| 3 | LSTM, Particle filters, TensorFlow/PyTorch |
| 4 | FastAPI/Flask, simulation engine |
| 5 | Three.js or Babylon.js, React |
| 6 | Database, Fleet API |
| 7 | Time-series analysis, Bayesian inference |
| 8 | JWT, BACnet/Modbus, REST/WebSocket |

---

## Success Metrics

- **Phase 0:** <5ms sensor data latency
- **Phase 1:** Physics predictions within ±2% of real sensors under nominal conditions
- **Phase 2:** Divergence detection sensitivity >95%
- **Phase 3:** RUL predictions within ±5 days (for 14-day horizons)
- **Phase 4:** Scenario simulation 1000x faster than real-time
- **Phase 5:** Responsive 3D interface (>30 FPS)
- **Phase 6:** Support 100+ units with <100ms dashboard refresh
- **Phase 7:** Twin accuracy maintained >95% over 12 months
- **Phase 8:** Handle 100 concurrent twin instances under load
