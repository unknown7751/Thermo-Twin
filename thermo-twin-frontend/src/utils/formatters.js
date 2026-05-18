export const MAX_RUL = 9998

// ── RUL formatting ──────────────────────────────────────────────────────────

export function formatRUL(secs) {
  if (secs >= MAX_RUL) return { value: '∞',              unit: 'healthy',       status: 'healthy'  }
  if (secs <= 0)       return { value: 'FAILED',         unit: 'at threshold',  status: 'failed'   }
  if (secs >= 86400)   return { value: (secs / 86400).toFixed(1), unit: 'days remaining',  status: 'ok'  }
  if (secs >= 3600)    return { value: (secs / 3600).toFixed(1),  unit: 'hr remaining',    status: 'warn' }
  if (secs >= 60)      return { value: (secs / 60).toFixed(1),    unit: 'min remaining',   status: 'warn' }
  return                      { value: secs.toFixed(1),           unit: 'sec remaining',   status: secs < 10 ? 'crit' : 'warn' }
}

export function rulCardClasses(rul) {
  if (rul >= MAX_RUL) return 'border-slate-700 bg-slate-900/60'
  if (rul <= 0)       return 'border-red-900   bg-red-950/50'
  if (rul < 10)       return 'border-red-600   bg-red-950/30'
  if (rul < 60)       return 'border-amber-600 bg-amber-950/20'
  return                     'border-slate-700 bg-slate-900/60'
}

export function rulValueClasses(rul) {
  if (rul >= MAX_RUL) return 'text-emerald-400'
  if (rul <= 0)       return 'text-red-500'
  if (rul < 10)       return 'text-red-400'
  if (rul < 60)       return 'text-amber-400'
  return                     'text-emerald-400'
}

// ── Component health ────────────────────────────────────────────────────────

export function healthBarClasses(pct) {
  if (pct > 80) return 'bg-emerald-500'
  if (pct > 50) return 'bg-amber-500'
  return               'bg-red-500'
}

export function healthTextClasses(pct) {
  if (pct > 80) return 'text-emerald-400'
  if (pct > 50) return 'text-amber-400'
  return               'text-red-400'
}

// ── Sensor divergence ───────────────────────────────────────────────────────

export function divergenceClasses(val) {
  const abs = Math.abs(val ?? 0)
  if (abs < 0.5) return 'text-emerald-400'
  if (abs < 2.0) return 'text-amber-400'
  return               'text-red-400'
}

export function divergenceStatus(val) {
  const abs = Math.abs(val ?? 0)
  if (abs < 0.5) return { icon: '✓', cls: 'text-emerald-400' }
  if (abs < 2.0) return { icon: '⚠', cls: 'text-amber-400'  }
  return               { icon: '✗', cls: 'text-red-400'    }
}

// ── Severity ────────────────────────────────────────────────────────────────

export function severityClasses(sev) {
  if (sev < 50) return 'text-emerald-400'
  if (sev < 71) return 'text-amber-400'
  return               'text-red-400'
}

// ── Fault label map ─────────────────────────────────────────────────────────

export const FAULT_LABELS = {
  refrigerant_leak:  'Refrigerant Leak',
  fan_failure:       'Fan Failure',
  compressor_wear:   'Compressor Wear',
}

// ── Alert time ──────────────────────────────────────────────────────────────
// alert.timestamp is an ISO string from the backend; the numeric simulation
// time lives in anomaly_end_time / anomaly_start_time. Return a clean label.
export function formatAlertTime(alert) {
  if (!alert) return '—'
  const t = alert.anomaly_end_time ?? alert.anomaly_start_time
  if (typeof t === 'number') return `${t.toFixed(1)}s`
  if (typeof alert.timestamp === 'string') {
    // ISO → HH:MM:SS
    const d = new Date(alert.timestamp)
    return isNaN(d) ? alert.timestamp : d.toLocaleTimeString()
  }
  return '—'
}
