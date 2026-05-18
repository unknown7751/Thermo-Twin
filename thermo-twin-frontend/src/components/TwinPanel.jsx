import useAppStore from '../store/appStore.js'
import {
  healthBarClasses,
  healthTextClasses,
  divergenceClasses,
  divergenceStatus,
} from '../utils/formatters.js'

const HEALTH_METRICS = [
  { key: 'refrigerant_charge_pct',    label: 'Refrigerant Charge' },
  { key: 'compressor_efficiency_pct', label: 'Compressor Efficiency' },
  { key: 'fan_health_pct',            label: 'Fan Health' },
]

const SENSOR_LABELS = {
  compressor_power_kw:    'Compressor Power',
  discharge_pressure_psi: 'Discharge Pressure',
  fan_rpm:                'Fan Speed',
  supply_air_temp_c:      'Supply Air Temp',
}

const SENSOR_UNITS = {
  compressor_power_kw:    'kW',
  discharge_pressure_psi: 'psi',
  fan_rpm:                'RPM',
  supply_air_temp_c:      '°C',
}

function HealthBar({ label, pct, uncertainty }) {
  const clampedPct = Math.max(0, Math.min(100, pct ?? 100))
  const unc        = uncertainty ?? 0

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-slate-400">{label}</span>
        <span className={`text-xs font-mono font-semibold ${healthTextClasses(clampedPct)}`}>
          {clampedPct.toFixed(1)}%
          {unc > 0 && (
            <span className="text-slate-600 font-normal"> ±{unc.toFixed(1)}</span>
          )}
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden relative">
        {/* Uncertainty band */}
        {unc > 0 && (
          <div
            className="absolute top-0 h-full bg-slate-600/40 rounded-full"
            style={{
              left:  `${Math.max(0, clampedPct - unc)}%`,
              width: `${Math.min(2 * unc, 100 - Math.max(0, clampedPct - unc))}%`,
            }}
          />
        )}
        {/* Health fill */}
        <div
          className={`h-full rounded-full transition-all duration-300 ${healthBarClasses(clampedPct)}`}
          style={{ width: `${clampedPct}%` }}
        />
      </div>
    </div>
  )
}

function DivergenceRow({ sensorKey, real, predicted, divergence }) {
  const { icon, cls } = divergenceStatus(divergence)
  const unit          = SENSOR_UNITS[sensorKey] ?? ''

  return (
    <tr className="border-b border-slate-800/60 last:border-0">
      <td className="py-1.5 pr-3 text-xs text-slate-400 whitespace-nowrap">
        {SENSOR_LABELS[sensorKey]}
      </td>
      <td className="py-1.5 pr-2 text-xs font-mono text-slate-200 text-right">
        {typeof real === 'number' ? real.toFixed(1) : '—'}
        <span className="text-slate-600 ml-0.5">{unit}</span>
      </td>
      <td className="py-1.5 pr-2 text-xs font-mono text-slate-400 text-right">
        {typeof predicted === 'number' ? predicted.toFixed(1) : '—'}
        <span className="text-slate-600 ml-0.5">{unit}</span>
      </td>
      <td className="py-1.5 text-right">
        <span className={`text-xs font-mono ${divergenceClasses(divergence)}`}>
          {typeof divergence === 'number'
            ? (divergence >= 0 ? '+' : '') + divergence.toFixed(2)
            : '—'}
        </span>
        <span className={`ml-1.5 text-xs ${cls}`}>{icon}</span>
      </td>
    </tr>
  )
}

export default function TwinPanel() {
  const twin   = useAppStore((s) => s.twin)
  const sample = useAppStore((s) => s.sample)

  const { state, prediction, divergence, uncertainty } = twin

  const sensorKeys = Object.keys(SENSOR_LABELS)

  return (
    <div className="card flex flex-col gap-4">
      <p className="section-title">Digital Twin State</p>

      {/* ── Health bars ── */}
      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Component Health</p>
        {HEALTH_METRICS.map(({ key, label }) => (
          <HealthBar
            key={key}
            label={label}
            pct={state[key]}
            uncertainty={uncertainty[key]}
          />
        ))}
      </div>

      {/* ── Sensor divergence table ── */}
      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Sensor Divergence</p>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-700">
                <th className="pb-1.5 text-xs text-slate-600 font-normal">Sensor</th>
                <th className="pb-1.5 text-xs text-slate-600 font-normal text-right">Real</th>
                <th className="pb-1.5 text-xs text-slate-600 font-normal text-right">Model</th>
                <th className="pb-1.5 text-xs text-slate-600 font-normal text-right">Δ</th>
              </tr>
            </thead>
            <tbody>
              {sensorKeys.map((key) => (
                <DivergenceRow
                  key={key}
                  sensorKey={key}
                  real={sample[key]}
                  predicted={prediction[key]}
                  divergence={divergence[key]}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
