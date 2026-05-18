import useAppStore from '../store/appStore.js'
import { severityClasses, formatAlertTime } from '../utils/formatters.js'

const SENSOR_LABELS = {
  compressor_power_kw:    { label: 'Compressor Power', unit: 'kW'  },
  discharge_pressure_psi: { label: 'Discharge Pressure', unit: 'psi' },
  fan_rpm:                { label: 'Fan Speed', unit: 'RPM' },
  supply_air_temp_c:      { label: 'Supply Air Temp', unit: '°C'  },
}

function SensorRow({ id, value }) {
  const { label, unit } = SENSOR_LABELS[id]
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-slate-800 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="font-mono text-sm text-slate-100">
        {typeof value === 'number' ? value.toFixed(1) : '—'}
        <span className="text-xs text-slate-500 ml-1">{unit}</span>
      </span>
    </div>
  )
}

export default function Sidebar() {
  const sample        = useAppStore((s) => s.sample)
  const twin          = useAppStore((s) => s.twin)
  const alerts        = useAppStore((s) => s.alerts)
  const backendOnline = useAppStore((s) => s.stream.backendOnline)

  const latest  = alerts.latest
  const severity = latest?.severity_score ?? latest?.severity ?? 0

  return (
    <aside className="flex flex-col gap-4">

      {/* ── Severity card ── */}
      <div className="card">
        <p className="section-title mb-2">Anomaly Severity</p>
        <div className="text-center py-2">
          <span className={`text-5xl font-bold font-mono ${severityClasses(severity)}`}>
            {severity.toFixed(0)}
          </span>
          <span className="text-slate-500 text-sm block mt-1">/ 100</span>
        </div>
        {latest && (
          <p className="text-xs text-slate-500 text-center mt-1 truncate">
            {latest.fault_type ?? 'unknown'} · {formatAlertTime(latest)}
          </p>
        )}
      </div>

      {/* ── Live sensor values ── */}
      <div className="card">
        <p className="section-title mb-1">Live Sensors</p>
        {Object.keys(SENSOR_LABELS).map((id) => (
          <SensorRow key={id} id={id} value={sample[id]} />
        ))}
      </div>

      {/* ── Model badges ── */}
      <div className="card">
        <p className="section-title mb-2">Engine Info</p>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Physics</span>
            <span className={`text-xs px-2 py-0.5 rounded font-mono ${
              twin.model_used === 'coolprop'
                ? 'bg-cyan-950/60 text-cyan-400 border border-cyan-800'
                : 'bg-slate-800 text-slate-300 border border-slate-700'
            }`}>
              {twin.model_used ?? 'linear'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Estimator</span>
            <span className={`text-xs px-2 py-0.5 rounded font-mono ${
              twin.estimator_mode === 'ukf'
                ? 'bg-violet-950/60 text-violet-400 border border-violet-800'
                : 'bg-slate-800 text-slate-300 border border-slate-700'
            }`}>
              {twin.estimator_mode ?? 'ukf'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-500">Backend</span>
            <span className={`text-xs px-2 py-0.5 rounded font-mono ${
              backendOnline
                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800'
                : 'bg-red-950/60 text-red-400 border border-red-800'
            }`}>
              {backendOnline ? 'online' : 'offline'}
            </span>
          </div>
        </div>
      </div>

    </aside>
  )
}
