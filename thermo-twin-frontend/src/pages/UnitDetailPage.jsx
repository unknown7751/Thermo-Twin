import { useEffect, useState } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import api from '../utils/api.js'
import {
  healthBarClasses, healthTextClasses,
  divergenceClasses, divergenceStatus,
  formatRUL,
} from '../utils/formatters.js'

const SENSOR_LABELS = {
  compressor_power_kw:    { label: 'Compressor Power',    unit: 'kW'  },
  discharge_pressure_psi: { label: 'Discharge Pressure',  unit: 'psi' },
  fan_rpm:                { label: 'Fan Speed',           unit: 'RPM' },
  supply_air_temp_c:      { label: 'Supply Air Temp',     unit: '°C'  },
}

const HEALTH_METRICS = [
  { key: 'refrigerant_charge_pct',    label: 'Refrigerant Charge'    },
  { key: 'compressor_efficiency_pct', label: 'Compressor Efficiency' },
  { key: 'fan_health_pct',            label: 'Fan Health'            },
]

const RUL_COMPONENTS = [
  { key: 'refrigerant', label: 'Refrigerant', central: 'refrigerant_rul_days', lower: 'refrigerant_rul_days_lower', upper: 'refrigerant_rul_days_upper' },
  { key: 'compressor',  label: 'Compressor',  central: 'compressor_rul_days',  lower: 'compressor_rul_days_lower',  upper: 'compressor_rul_days_upper'  },
  { key: 'fan',         label: 'Fan',         central: 'fan_rul_days',         lower: 'fan_rul_days_lower',         upper: 'fan_rul_days_upper'         },
]

function HealthBar({ label, pct }) {
  const clamped = Math.max(0, Math.min(100, pct ?? 100))
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-xs font-mono font-semibold ${healthTextClasses(clamped)}`}>
          {clamped.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${healthBarClasses(clamped)}`} style={{ width: `${clamped}%` }} />
      </div>
    </div>
  )
}

function RULCard({ label, central, lower, upper }) {
  const fmt = formatRUL(central ?? 9999)
  const showCI = central != null && central < 9998 && central > 0 && lower != null && upper != null
  return (
    <div className="flex-1 rounded-lg border border-slate-700 bg-slate-800/40 p-3">
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold font-mono text-cyan-300 mt-1">
        {fmt.value}{' '}
        <span className="text-xs text-slate-500 font-normal">{fmt.unit}</span>
      </p>
      {showCI && (
        <p className="text-xs text-slate-600 font-mono mt-0.5">
          CI [{formatRUL(lower).value} – {formatRUL(upper).value}]
        </p>
      )}
    </div>
  )
}

export default function UnitDetailPage() {
  const { machineId } = useParams()
  const navigate = useNavigate()

  const [data, setData]   = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy]   = useState(false)

  useEffect(() => {
    let active = true
    async function fetchUnit() {
      try {
        const res = await api.get(`/fleet/${encodeURIComponent(machineId)}/twin`)
        if (active) { setData(res.data); setError(null) }
      } catch (err) {
        if (active) setError(err?.response?.data?.error ?? 'Failed to fetch unit')
      }
    }
    fetchUnit()
    const id = setInterval(fetchUnit, 1000)
    return () => { active = false; clearInterval(id) }
  }, [machineId])

  const resetUnit = async () => {
    setBusy(true)
    try { await api.post(`/fleet/${encodeURIComponent(machineId)}/reset`) }
    catch (err) { console.error(err) }
    finally { setBusy(false) }
  }

  const unregister = async () => {
    setBusy(true)
    try {
      await api.delete(`/fleet/${encodeURIComponent(machineId)}`)
      navigate('/fleet')
    } catch (err) { console.error(err); setBusy(false) }
  }

  if (error) {
    return (
      <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 pt-6">
        <Link to="/fleet" className="text-xs text-cyan-400 hover:text-cyan-300">← back to fleet</Link>
        <div className="card mt-4 border-red-800 bg-red-950/30">
          <p className="text-red-300">{error}</p>
        </div>
      </main>
    )
  }
  if (!data) {
    return (
      <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 pt-6">
        <Link to="/fleet" className="text-xs text-cyan-400 hover:text-cyan-300">← back to fleet</Link>
        <p className="text-xs text-slate-600 mt-6">Loading unit…</p>
      </main>
    )
  }

  const { metadata = {}, twin = {} } = data
  const state      = twin.state      ?? {}
  const prediction = twin.prediction ?? {}
  const divergence = twin.divergence ?? {}
  const rul        = twin.rul        ?? {}

  return (
    <main className="flex-1 max-w-[1200px] w-full mx-auto px-4 pb-8 space-y-4 pt-4">
      <Link to="/fleet" className="text-xs text-cyan-400 hover:text-cyan-300">← back to fleet</Link>

      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between flex-wrap gap-2">
          <div>
            <h1 className="text-xl font-bold text-slate-100 font-mono">{machineId}</h1>
            <p className="text-sm text-slate-400">{metadata.location || '—'}</p>
            <p className="text-xs text-slate-600 mt-1">
              {metadata.model}
              {metadata.fault_profile && (
                <> · <span className="text-amber-400">seeded profile: {metadata.fault_profile}</span></>
              )}
            </p>
          </div>
          <div className="flex gap-2">
            <button onClick={resetUnit} disabled={busy}
              className="text-xs px-3 py-1.5 rounded border border-slate-700 hover:border-cyan-700 text-slate-300 hover:text-cyan-300 disabled:opacity-40">
              Reset Twin
            </button>
            <button onClick={unregister} disabled={busy}
              className="text-xs px-3 py-1.5 rounded border border-slate-700 hover:border-red-700 text-slate-300 hover:text-red-300 disabled:opacity-40">
              Unregister
            </button>
          </div>
        </div>
      </div>

      {/* Twin state + divergence */}
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card">
          <p className="section-title mb-3">Component Health</p>
          {HEALTH_METRICS.map(({ key, label }) => (
            <HealthBar key={key} label={label} pct={state[key]} />
          ))}
        </div>
        <div className="card">
          <p className="section-title mb-3">Sensor Divergence</p>
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="pb-1.5 text-xs text-slate-600 font-normal">Sensor</th>
                <th className="pb-1.5 text-xs text-slate-600 font-normal text-right">Model</th>
                <th className="pb-1.5 text-xs text-slate-600 font-normal text-right">Δ</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(SENSOR_LABELS).map((key) => {
                const { label, unit } = SENSOR_LABELS[key]
                const ds = divergenceStatus(divergence[key])
                return (
                  <tr key={key} className="border-b border-slate-800/60 last:border-0">
                    <td className="py-1.5 text-xs text-slate-400">{label}</td>
                    <td className="py-1.5 text-xs font-mono text-slate-300 text-right">
                      {typeof prediction[key] === 'number' ? prediction[key].toFixed(1) : '—'}
                      <span className="text-slate-600 ml-0.5">{unit}</span>
                    </td>
                    <td className="py-1.5 text-right">
                      <span className={`text-xs font-mono ${divergenceClasses(divergence[key])}`}>
                        {typeof divergence[key] === 'number'
                          ? (divergence[key] >= 0 ? '+' : '') + divergence[key].toFixed(2)
                          : '—'}
                      </span>
                      <span className={`ml-1.5 text-xs ${ds.cls}`}>{ds.icon}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* RUL */}
      <div className="card">
        <p className="section-title mb-3">Remaining Useful Life</p>
        <div className="flex gap-2">
          {RUL_COMPONENTS.map((c) => (
            <RULCard
              key={c.key}
              label={c.label}
              central={rul[c.central]}
              lower={rul[c.lower]}
              upper={rul[c.upper]}
            />
          ))}
        </div>
        <div className="flex items-center justify-between text-xs text-slate-600 mt-3">
          <span>Rate mode: <span className="text-slate-400">{rul.rate_mode ?? '—'}</span></span>
          <span>History: <span className="text-slate-400 font-mono">{rul.history_samples ?? 0}</span> samples</span>
        </div>
      </div>
    </main>
  )
}
