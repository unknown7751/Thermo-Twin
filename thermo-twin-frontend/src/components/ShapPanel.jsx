import useAppStore from '../store/appStore.js'
import { formatAlertTime, severityClasses } from '../utils/formatters.js'

const FEATURE_LABELS = {
  compressor_power_pct:    { label: 'Compressor Power', color: '#22d3ee' },
  discharge_pressure_pct:  { label: 'Discharge Pressure', color: '#a78bfa' },
  fan_rpm_pct:             { label: 'Fan Speed', color: '#34d399' },
  supply_air_temp_pct:     { label: 'Supply Air Temp', color: '#fb923c' },
}

function ContributionBar({ label, color, pct }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-36 shrink-0 truncate">{label}</span>
      <div className="flex-1 h-4 bg-slate-800 rounded-sm overflow-hidden">
        <div
          className="h-full rounded-sm transition-all duration-500"
          style={{ width: `${Math.max(0, Math.min(100, pct))}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono w-12 text-right shrink-0 text-slate-300">
        {pct.toFixed(1)}%
      </span>
    </div>
  )
}

export default function ShapPanel() {
  const latest = useAppStore((s) => s.alerts.latest)

  // Hidden until first alert
  if (!latest?.explanation) return null

  const expl    = latest.explanation
  const presc   = latest.prescription
  const cost    = latest.energy_cost
  const severity = latest.severity_score ?? latest.severity ?? 0

  const contributions = Object.entries(FEATURE_LABELS)
    .map(([key, meta]) => ({ ...meta, pct: expl[key] ?? 0 }))
    .sort((a, b) => b.pct - a.pct)

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <p className="section-title">Explainability &amp; Prescription</p>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-slate-500">
            {latest.fault_type} · t={formatAlertTime(latest)}
          </span>
          <span className={`font-mono font-bold ${severityClasses(severity)}`}>
            severity {severity.toFixed(0)}
          </span>
        </div>
      </div>

      {/* Summary */}
      {expl.summary && (
        <p className="text-sm text-slate-300 mb-4 bg-slate-800/40 border border-slate-700 rounded px-3 py-2">
          {expl.summary}
        </p>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* ── Feature contributions ── */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
            Feature Contribution
          </p>
          <div className="space-y-2">
            {contributions.map((c) => (
              <ContributionBar key={c.label} label={c.label} color={c.color} pct={c.pct} />
            ))}
          </div>
        </div>

        {/* ── Prescription ── */}
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
            Recommended Action
          </p>
          {presc ? (
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-slate-500 text-xs">Diagnosis</span>
                <p className="text-slate-200">{presc.fault}</p>
              </div>
              <div>
                <span className="text-slate-500 text-xs">Action</span>
                <p className="text-amber-300">{presc.action}</p>
              </div>
              <div>
                <span className="text-slate-500 text-xs">Impact</span>
                <p className="text-slate-400">{presc.impact}</p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-600">No prescription available</p>
          )}
        </div>
      </div>

      {/* ── Cost impact ── */}
      {cost && (
        <div className="mt-4 pt-3 border-t border-slate-800 flex flex-wrap gap-x-6 gap-y-1 text-xs">
          <span className="text-slate-500">
            Energy waste:{' '}
            <span className="text-red-400 font-mono">{cost.energy_waste_kwh_per_hr} kWh/hr</span>
          </span>
          <span className="text-slate-500">
            Cost:{' '}
            <span className="text-red-400 font-mono">
              ₹{cost.cost_per_day_inr?.toLocaleString()}/day
            </span>
          </span>
          <span className="text-slate-500">
            Efficiency loss:{' '}
            <span className="text-amber-400 font-mono">{cost.efficiency_loss_pct}%</span>
          </span>
          {cost.payback_days != null && (
            <span className="text-slate-500">
              Repair payback:{' '}
              <span className="text-emerald-400 font-mono">{cost.payback_days} days</span>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
