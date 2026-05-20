import { create } from 'zustand'

const useAppStore = create((set) => ({
  // ── Stream metadata ────────────────────────────────────────────────────────
  stream: {
    sampleCount:  0,
    alertCount:   0,
    currentTime:  0,
    wsConnected:  false,
    backendOnline: false,
  },
  setStreamMeta: (meta) =>
    set((s) => ({ stream: { ...s.stream, ...meta } })),
  setWsConnected: (v) =>
    set((s) => ({ stream: { ...s.stream, wsConnected: v } })),
  setBackendOnline: (v) =>
    set((s) => ({ stream: { ...s.stream, backendOnline: v } })),

  // ── Latest raw sample (for Sidebar live values) ────────────────────────────
  sample: {
    compressor_power_kw:    3.5,
    discharge_pressure_psi: 245,
    fan_rpm:                1190,
    supply_air_temp_c:      11.0,
    timestamp:              0,
  },
  setSample: (s) => set({ sample: s }),

  // ── Digital twin state (Phase 1 + 2) ──────────────────────────────────────
  twin: {
    state: {
      refrigerant_charge_pct:    100,
      compressor_efficiency_pct: 100,
      fan_health_pct:            100,
    },
    prediction: {
      compressor_power_kw:    3.5,
      discharge_pressure_psi: 245,
      fan_rpm:                1190,
      supply_air_temp_c:      11.0,
    },
    divergence: {
      compressor_power_kw:    0,
      discharge_pressure_psi: 0,
      fan_rpm:                0,
      supply_air_temp_c:      0,
    },
    uncertainty: {
      refrigerant_charge_pct:    0,
      compressor_efficiency_pct: 0,
      fan_health_pct:            0,
    },
    estimator_mode: 'ukf',
    model_used:     'linear',
  },
  setTwin: (t) => set({ twin: t }),

  // ── RUL prognostics (Phase 3) ──────────────────────────────────────────────
  rul: {
    refrigerant_rul_days:       9999,
    refrigerant_rul_days_lower: 9999,
    refrigerant_rul_days_upper: 9999,
    compressor_rul_days:        9999,
    compressor_rul_days_lower:  9999,
    compressor_rul_days_upper:  9999,
    fan_rul_days:               9999,
    fan_rul_days_lower:         9999,
    fan_rul_days_upper:         9999,
    most_critical_component:    'none',
    days_to_any_failure:        9999,
    rate_mode:                  'linear',
    history_samples:            0,
    mc: null,
  },
  setRul: (r) => set({ rul: r }),

  // ── Alert history ──────────────────────────────────────────────────────────
  alerts: {
    latest:  null,
    history: [],    // max 20 entries, newest first
  },
  addAlert: (alert) =>
    set((s) => ({
      alerts: {
        latest:  alert,
        history: [alert, ...s.alerts.history].slice(0, 20),
      },
    })),

  // ── Fault injection ────────────────────────────────────────────────────────
  fault: {
    active:        null,    // 'refrigerant_leak' | 'fan_failure' | 'compressor_wear' | null
    countdownSec:  0,
    faultStartTime: null,   // simulation time
    faultEndTime:   null,
  },
  setFaultActive: (faultType, startTime, endTime, countdownSec) =>
    set({
      fault: {
        active:         faultType,
        countdownSec:   countdownSec ?? 10,
        faultStartTime: startTime,
        faultEndTime:   endTime,
      },
    }),
  clearFault: () =>
    set({ fault: { active: null, countdownSec: 0, faultStartTime: null, faultEndTime: null } }),
  tickFaultCountdown: () =>
    set((s) => {
      const next = s.fault.countdownSec - 1
      return next <= 0
        ? { fault: { active: null, countdownSec: 0, faultStartTime: null, faultEndTime: null } }
        : { fault: { ...s.fault, countdownSec: next } }
    }),

  // ── Fleet (Phase 6) ────────────────────────────────────────────────────────
  fleet: {
    units:       [],      // list of machine_id strings
    metadata:    {},      // machine_id → { location, model, ... }
    health:      null,    // /fleet/health response
    queue:       null,    // /fleet/dispatch-queue response
    anomalies:   [],      // /fleet/anomalies → .anomalies
    online:      false,   // last fetch succeeded
    lastFetched: null,
  },
  setFleet: (patch) =>
    set((s) => ({ fleet: { ...s.fleet, ...patch, lastFetched: Date.now() } })),
  setFleetOnline: (v) =>
    set((s) => ({ fleet: { ...s.fleet, online: v } })),

  // ── Recalibration / drift (Phase 7) ────────────────────────────────────────
  recal: {
    drift:      null,    // /twin/drift response
    parameters: null,    // /twin/parameters response
    status:     null,    // /twin/recalibration/status response
    online:     false,
    lastFetched: null,
  },
  setRecal: (patch) =>
    set((s) => ({ recal: { ...s.recal, ...patch, lastFetched: Date.now() } })),
  setRecalOnline: (v) =>
    set((s) => ({ recal: { ...s.recal, online: v } })),
}))

export default useAppStore
