import useAppStore from '../store/appStore.js'
import {
  formatRUL,
  rulCardClasses,
  rulValueClasses,
  MAX_RUL,
} from '../utils/formatters.js'

const COMPONENTS = [
  { key: 'refrigerant', label: 'Refrigerant',  central: 'refrigerant_rul_days', lower: 'refrigerant_rul_days_lower', upper: 'refrigerant_rul_days_upper' },
  { key: 'compressor',  label: 'Compressor',   central: 'compressor_rul_days',  lower: 'compressor_rul_days_lower',  upper: 'compressor_rul_days_upper'  },
  { key: 'fan',         label: 'Fan',           central: 'fan_rul_days',         lower: 'fan_rul_days_lower',         upper: 'fan_rul_days_upper'         },
]

const RATE_MODE_LABELS = {
  linear: 'Linear OLS',
  lstm:   'LSTM',
}

function RULCard({ label, central, lower, upper }) {
  const { value, unit, status } = formatRUL(central)
  const cardCls  = rulCardClasses(central)
  const valueCls = rulValueClasses(central)

  const showCI = central < MAX_RUL && central > 0 && lower != null && upper != null

  return (
    <div className={`flex-1 rounded-lg border p-3 flex flex-col gap-1 ${cardCls}`}>
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>

      <div className="flex items-baseline gap-1 mt-1">
        <span className={`text-2xl font-bold font-mono leading-none ${valueCls}`}>
          {value}
        </span>
        {status !== 'failed' && status !== 'healthy' && (
          <span className="text-xs text-slate-500 ml-1">{unit}</span>
        )}
      </div>

      {status === 'failed' && (
        <span className="text-xs text-red-400">at threshold</span>
      )}
      {status === 'healthy' && (
        <span className="text-xs text-emerald-400">no degradation</span>
      )}

      {/* Confidence interval */}
      {showCI && (
        <div className="mt-1 text-xs text-slate-600 font-mono">
          <span className="text-slate-500">CI </span>
          [{formatRUL(lower).value} – {formatRUL(upper).value}]
        </div>
      )}
    </div>
  )
}

export default function RULPanel() {
  const rul = useAppStore((s) => s.rul)

  const {
    most_critical_component,
    days_to_any_failure,
    rate_mode,
    history_samples,
  } = rul

  const isCritical = days_to_any_failure < 60   // less than 60 sim-seconds → warn banner
  const rateLabel  = RATE_MODE_LABELS[rate_mode] ?? rate_mode ?? 'linear'

  return (
    <div className="card flex flex-col gap-3">
      <p className="section-title">RUL Prognostics</p>

      {/* ── Critical banner ── */}
      {isCritical && days_to_any_failure > 0 && (
        <div className="rounded border border-red-700 bg-red-950/30 px-3 py-2 text-xs text-red-300 flex items-center gap-2">
          <span className="text-red-500 text-base leading-none">⚠</span>
          <span>
            <span className="font-semibold capitalize">{most_critical_component}</span> component critical —{' '}
            {formatRUL(days_to_any_failure).value} {formatRUL(days_to_any_failure).unit}
          </span>
        </div>
      )}

      {days_to_any_failure <= 0 && (
        <div className="rounded border border-red-800 bg-red-950/50 px-3 py-2 text-xs text-red-300 flex items-center gap-2 animate-[flashRed_1s_ease-in-out_infinite]">
          <span className="text-red-400 text-base leading-none">✗</span>
          <span>
            <span className="font-semibold capitalize">{most_critical_component}</span> has reached failure threshold
          </span>
        </div>
      )}

      {/* ── RUL cards ── */}
      <div className="flex gap-2">
        {COMPONENTS.map((c) => (
          <RULCard
            key={c.key}
            label={c.label}
            central={rul[c.central]}
            lower={rul[c.lower]}
            upper={rul[c.upper]}
          />
        ))}
      </div>

      {/* ── Footer metadata ── */}
      <div className="flex items-center justify-between text-xs text-slate-600">
        <span>
          Rate mode: <span className="text-slate-400">{rateLabel}</span>
        </span>
        <span>
          History: <span className="text-slate-400 font-mono">{history_samples}</span> samples
        </span>
      </div>
    </div>
  )
}
