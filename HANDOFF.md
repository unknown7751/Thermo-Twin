# Thermo-Twin — Handoff Notes

Context for whoever (human or AI) picks this project up next. Read this once; then the code is the source of truth.

---

## 1. What this project is

A **digital twin** for an HVAC chiller. A Flask backend simulates real sensor data and runs the full twin pipeline (physics model + UKF state estimator + RUL + what-if simulator + 3D model). A React/Vite frontend visualises everything live.

Single-machine demo today. Multi-unit "fleet" mode is **specified but NOT built** — see §9.

---

## 2. Run it

Two terminals, in this order.

**Backend** (Python 3.12 venv already set up):
```powershell
.\venv\Scripts\Activate.ps1
python backend/app.py        # starts on :5000
```
Watch for: `Background stream loop started at 10 Hz` (it's actually 3 Hz currently — log text is stale).

**Frontend**:
```powershell
cd thermo-twin-frontend
npm run dev                  # Vite on :5173, proxies API to :5000
```

Optional infrastructure (NOT required — engine auto-disables them on failure):
- `docker-compose up -d` → InfluxDB on :8086 + Mosquitto on :1883

⚠️ **Python does not hot-reload.** Any change in `backend/` requires a full backend restart. Frontend hot-reloads via Vite (except `.env`).

---

## 3. Architecture in one paragraph

The backend runs a **single background thread** at 3 Hz (`INTERVAL=0.33s` in [backend/app.py](backend/app.py)) that drives the entire pipeline per tick: generate sample → ML anomaly detection → physics + UKF twin → RUL → push onto a deque. A **separate persistence worker** thread (`_persist_worker`) drains InfluxDB/MQTT writes off the critical path. HTTP requests to `/stream/next-sample` just pop from the deque (≤3ms) and return up to 20 samples per poll. The frontend polls every 100ms, feeds samples into a `sampleBus` (pure pub/sub, no React re-renders) for the chart, and updates Zustand for everything else. All this exists because earlier we hit two real bugs: synchronous ML inference on the request thread (2–5 Hz cap) and **InfluxDB writes blocking the loop for ~3s each when the broker was down**.

---

## 4. Phase-by-phase status

| Phase | What | Status | Key files |
|---|---|---|---|
| 0 | Infra layer (Flask, InfluxDB writer, MQTT publisher, WebSocket scaffold) | ✅ done; persistence decoupled into worker thread | [backend/app.py](backend/app.py), [backend/influx_writer.py](backend/influx_writer.py), [backend/mqtt_publisher.py](backend/mqtt_publisher.py), [backend/twin_schema.py](backend/twin_schema.py) |
| 1 | Physics model (HVAC thermodynamics + degradation cascades) | ✅ done | [physics/hvac_physics.py](physics/hvac_physics.py), [physics/degradation_model.py](physics/degradation_model.py) |
| 2 | Unscented Kalman filter state estimator | ✅ done | [physics/state_estimator.py](physics/state_estimator.py) |
| 3 | RUL prognostics (linear OLS + optional LSTM + 200-particle MC) | ✅ done | [backend/degradation_trajectory.py](backend/degradation_trajectory.py), [backend/rul_engine.py](backend/rul_engine.py), [backend/particle_filter.py](backend/particle_filter.py) |
| 4 | What-If Simulator (operating-scenario projection w/ recalibrated wear) | ✅ done | [backend/whatif_engine.py](backend/whatif_engine.py), [thermo-twin-frontend/src/components/WhatIfPanel.jsx](thermo-twin-frontend/src/components/WhatIfPanel.jsx) |
| 5 | 3D digital twin (THREE.js scene + click-to-inspect + history endpoint) | ✅ done | [thermo-twin-frontend/src/utils/hvacScene.js](thermo-twin-frontend/src/utils/hvacScene.js), [HVAC3D.jsx](thermo-twin-frontend/src/components/HVAC3D.jsx), [ComponentDetailsPanel.jsx](thermo-twin-frontend/src/components/ComponentDetailsPanel.jsx), [SensorHistoryChart.jsx](thermo-twin-frontend/src/components/SensorHistoryChart.jsx) |
| 6 | Fleet Twin (multi-unit registry, dispatch queue, cross-unit anomalies) | ❌ **not started** — see §9 | — |

---

## 5. Critical design decisions (and *why*)

These are non-obvious choices that will look wrong if you don't know the reason. Don't undo them without understanding the original incident.

### 5.1 Persistence is decoupled into a separate worker thread
**Why:** when InfluxDB or MQTT goes down, their write calls succeed in `__init__` (just create a client) but block for **3–4 seconds** in `write_sample()`. That stalled the sample generation loop at ~0.27 Hz. Now writes go onto `_persist_queue` (bounded deque, drops oldest) consumed by `_persist_worker`. On first write failure the service is auto-disabled for the session.
**Don't:** put InfluxDB/MQTT writes back into `_stream_loop`.

### 5.2 What-If wear rates driven by *overstress above nominal*, not absolute level
The original `_wear_rates` had constant floors (0.05 %/hr at nominal) so even a brand-new unit ran at 70/35/50 went critical in ~9 days — physically wrong. Now (`_NOM_SPEED=70`, `_NOM_AMB=35`, `_NOM_LOAD=50` in [whatif_engine.py](backend/whatif_engine.py)) wear floors at `_WEAR_FLOOR=0.002 %/hr` (component lifetimes ~5 years) and only **deviation above design** drives decay. Extreme 100/50/100 still fails in ~7h for demos.
**Don't:** add aggressive constant terms back.

### 5.3 What-If baseline = brand-new healthy unit at nominal (not "current health re-run")
Old baseline was "scenario inputs = baseline inputs → same projection → ₹0 savings even when scenario showed 190% power." Now baseline is fixed (100/100/100 health, 70/35/50 inputs), so any degraded unit always shows a meaningful cost delta.

### 5.4 Fan failure ∝ pressure RISE (non-linear, squared in fan loss)
Old [degradation_model.py](physics/degradation_model.py) had fan failure causing pressure to *drift down*, which is thermodynamically backwards (dead condenser fan → trapped heat → head pressure spikes). Now: `pressure += FAN_PRESSURE_RISE_FRAC * (fan_loss**2) * pressure`. At 0% fan health, pressure goes from 245 → 429 psi.

### 5.5 Chart streaming: backend pushes, frontend drains a *batch*
- Backend produces 3 Hz; frontend polls at 10 Hz.
- Each `/stream/next-sample` response drains up to **20 queued samples** (`primary` + `backlog` array). The frontend emits **all** of them to the chart via `sampleBus`.
- This is why the chart never falls behind regardless of poll-rate jitter.
- Frontend `useStreamData.js` POLL_MS = 100ms. Don't lower past ~50ms or React re-renders pile up.

### 5.6 Component → sensor map (single source of truth)
| Component | Sensor | Health metric | RUL key |
|---|---|---|---|
| compressor | `compressor_power_kw` | `compressor_efficiency_pct` | `compressor_rul_days` |
| condenser | `fan_rpm` | `fan_health_pct` | `fan_rul_days` |
| evaporator | `discharge_pressure_psi` | `refrigerant_charge_pct` | `refrigerant_rul_days` |
| valve | `supply_air_temp_c` | — | — |

Used in: backend [`/twin/component-history`](backend/app.py), `HVAC3D.jsx`, `hvacScene._buildHoverInfo`. **If you change one, change all three.**

### 5.7 Alert payload field names (the bug that white-screened the app)
The backend's `build_alert_payload` returns `severity_score` (not `severity`), `explanation` object with `*_pct` fields (not `shap_values`), `prescription` object, and `timestamp` as an **ISO string** (not a number). The frontend originally assumed wrong names → `timestamp.toFixed()` crashed every component on first alert. Fixed in Sidebar, AlertLog, ShapPanel; helper is `formatAlertTime()` in `formatters.js`. There's also an `ErrorBoundary` wrapping `<App />` that renders the actual error instead of a blank screen.

### 5.8 Simulation time vs real time
[`data_streamer.py`](backend/data_streamer.py): `self.current_time += 0.5` per sample. At 3 Hz that's 1.5 sim-sec/real-sec. Means RUL countdowns and fault delays "feel snappy" in a demo. Chart x-axis is **simulation seconds**, not wall clock.

### 5.9 InfluxDB is genuinely optional
The `/twin/component-history` endpoint serves from `live_streamer.history` (the 600-sample in-memory ring buffer), **not** InfluxDB — because we couldn't afford another sync InfluxDB call after the prior incidents. Same JSON shape, microsecond response, never hangs. If you want true multi-day history, add it as an *additive* path (try Influx → fall back to ring buffer).

---

## 6. File map (the bits that matter)

```
backend/
  app.py                     ← Flask app, all routes, _stream_loop, _persist_worker
  twin_engine.py             ← TwinEngine orchestrator (physics → UKF → RUL → what-if)
  whatif_engine.py           ← Phase 4 scenario projector (overstress-driven wear)
  data_streamer.py           ← SyntheticDataStreamer (the 'real' sensors)
  fault_injector.py          ← schedules faults that modify samples mid-stream
  live_detector.py           ← autoencoder + SHAP (8-second alert cooldown)
  influx_writer.py / mqtt_publisher.py   ← optional, fail-safe
  degradation_trajectory.py / rul_engine.py / particle_filter.py   ← Phase 3
  twin_schema.py             ← TwinSample / TwinAlert dataclasses

physics/
  hvac_physics.py            ← first-principles thermodynamics (CoolProp + linear fallback)
  degradation_model.py       ← health % → sensor deviation, fault cascades
  state_estimator.py         ← Unscented Kalman filter

thermo-twin-frontend/
  src/
    App.jsx                  ← top-level layout
    main.jsx                 ← wraps App in <ErrorBoundary>
    store/appStore.js        ← Zustand: stream, sample, twin, rul, alerts, fault
    hooks/
      useStreamData.js       ← 10 Hz polling, drains backlog, emits to sampleBus
      useFaultInjection.js   ← POST inject-fault, countdown
    utils/
      api.js                 ← axios (relative baseURL → uses Vite proxy)
      sampleBus.js           ← pub/sub for chart (no React re-renders)
      formatters.js          ← formatAlertTime, formatRUL, health/severity classes
      colorUtils.js          ← health → color (3D scene)
      hvacScene.js           ← THREE.js HVACScene class
      hvacGeometries.js      ← mesh factories for the 4 components
    components/
      Header, Sidebar, ChartPanel, FaultControls, AlertLog
      TwinPanel, RULPanel, ShapPanel
      WhatIfPanel + SliderInput + SummaryCards + TrajectoryChart
      HVAC3D + ComponentDetailsPanel + SensorHistoryChart
      ErrorBoundary
  tests/
    test_hvac_scene.js       ← vitest-compatible (needs `npm i -D vitest` to run)
```

---

## 7. Configuration knobs you might want to touch

| What | Where | Default | Notes |
|---|---|---|---|
| Backend generation rate | `INTERVAL` in `_stream_loop` ([app.py](backend/app.py)) | `0.33` (≈3 Hz) | rate = 1/INTERVAL. Higher needs bigger queue + batch drain. |
| Sim-time per sample | `current_time += 0.5` ([data_streamer.py:13](backend/data_streamer.py#L13)) | 0.5 sim-sec | Bigger = faster RUL depletion / chart scroll. |
| Frontend poll rate | `POLL_MS` in `useStreamData.js` | 100ms | Don't go below ~50ms. |
| Batch drain per poll | `range(20)` in `/stream/next-sample` ([app.py](backend/app.py)) | 20 | Must exceed `rate × POLL_MS / 1000`. |
| Chart window | `WINDOW_SECS`, `MAX_PTS` in `ChartPanel.jsx` | user-edited often | Recently 10s / 50 pts. |
| What-If duration max | `WHATIF_SPECS` in `SliderInput.jsx` | 5000 h | Backend caps trajectory at 300 pts via adaptive step. |
| Alert cooldown | `ALERT_COOLDOWN_SECS` in `live_detector.py` | 8.0 sim-sec | Why one fault injection → ~4 alerts. |

---

## 8. Endpoint reference

| Route | Purpose |
|---|---|
| `GET  /health` | service + dynamic threshold status |
| `GET  /stream/next-sample` | drain up to 20 queued samples (primary + backlog) |
| `POST /stream/reset` | clear streamer + queue + last twin state + injector |
| `POST /stream/inject-fault/<type>` | schedule fault: refrigerant_leak / fan_failure / compressor_wear |
| `GET  /alerts` | last 50 alerts (newest first) |
| `POST /demo/<scenario>` | trigger a canned alert payload |
| `GET  /twin/state` / `/twin/rul` | latest twin snapshot |
| `POST /twin/reset` | reset Kalman + trajectory |
| `POST /twin/whatif` | What-If scenario projection (Phase 4) |
| `GET  /twin/component-history/<component>` | recent sensor trace for a 3D component (in-memory) |
| `GET  /ws` | WebSocket scaffold (frontend uses HTTP polling, so usually 0 clients) |

---

## 9. Known pending work

### Phase 6 — Fleet Twin (NOT built)
A full spec exists (multi-unit `FleetManager`, dispatch queue with ₹ ROI, cross-unit anomaly correlation, `/fleet/*` endpoints, frontend `FleetDashboard`/`UnitGrid`/`DispatchQueue`, react-router routing, InfluxDB `machine_id` tagging). It is a large multi-file feature (~10 backend files modified, ~8 frontend files new, react-router introduction). Treat it as net-new work; do not assume any of it exists.

When building it, be aware:
- `TwinEngine.__init__` does NOT currently accept `machine_id` — adding it changes a hot path. Default it to `"LIVE-DEMO-UNIT"` to keep single-unit mode working.
- `TwinSample` does NOT currently have a `machine_id` field (it's stamped at the `from_streamer_dict` call site). Adding it as a real field affects `_persist_worker` payloads.
- `live_streamer` is a module-level singleton — fleet mode needs N streamers, or one fed multiplexed sources. The spec assumes external sources POST samples per machine, which is cleaner.
- InfluxDB read queries by `machine_id` tag will re-introduce the blocking failure mode unless implemented with timeouts + try/fallback to in-memory.

### Smaller follow-ups
- Vitest is not installed; tests file exists at `thermo-twin-frontend/tests/test_hvac_scene.js`. Run: `npm i -D vitest && npx vitest run`.
- `_broadcast_ws` exists but no frontend client connects; safe to remove if you don't plan to add WS.
- The "Background stream loop started at 10 Hz" log line is stale — it's whatever `INTERVAL` is set to. Cosmetic.

---

## 10. Common gotchas

- **Frontend says "Backend OFFLINE":** backend isn't running, OR `.env` has `VITE_API_URL=http://localhost:5000` again (must be empty so Axios uses relative URLs through the Vite proxy).
- **Chart shows two lines per grid:** ECharts series merge broke. Series updates in the RAF loop **must include `name`** to match initial series by name (not append new ones).
- **`buffer_size` stuck:** the streamer's history deque caps at 600. The header shows `total_samples` (`sample_index`) which is monotonic — use that, not `buffer_size`.
- **3D panel blank:** check the browser console; `ErrorBoundary` will also show the stack on screen.
- **What-If shows "Critical" at nominal settings:** see §5.2 — fixed, but if you re-introduce constant wear floors it'll regress.

---

## 11. Quick mental model for "what runs when"

```
Real time
  ┌─────────────────────────────────────────────────────────┐
  │ Backend                                                  │
  │   _stream_loop  ─── every 0.33s ──▶  sample queue       │
  │       │                                  │              │
  │       └──▶ ML inference + UKF + RUL      │              │
  │                                          │              │
  │   _persist_worker ── drains ─────────────┴── async      │
  │                  InfluxDB / MQTT  (fire-and-forget)     │
  └─────────────────────────────────────────────────────────┘
                                ▲
                                │ poll every 100 ms (drain up to 20)
  ┌─────────────────────────────┴───────────────────────────┐
  │ Frontend                                                 │
  │   useStreamData → emit() ─────▶ sampleBus               │
  │                        │              │                  │
  │                        │              ▼                  │
  │                        │       ChartPanel RAF (60fps)    │
  │                        ▼                                 │
  │                  Zustand store ──▶ Header/Sidebar/Twin/  │
  │                                    RUL/SHAP/Alerts/3D    │
  └─────────────────────────────────────────────────────────┘
```

That's the whole architecture in one picture. If you can read this and find the right file from the map, you're set.
