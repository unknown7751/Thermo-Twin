const STATUS_META = {
  safe:     { dot: '🟢', label: 'Safe to operate',   cls: 'text-emerald-400', card: 'border-emerald-800 bg-emerald-950/20' },
  warning:  { dot: '🟡', label: 'Warning',            cls: 'text-amber-400',   card: 'border-amber-800 bg-amber-950/20'   },
  critical: { dot: '🔴', label: 'Critical risk',      cls: 'text-red-400',     card: 'border-red-800 bg-red-950/30'       },
}

function Card({ label, value, valueCls = 'text-cyan-300', sub }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3">
      <p className="text-xs text-slate-500 mb-1">{label}</p>
      <p className={`text-lg font-bold font-mono ${valueCls}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

export default function SummaryCards({ summary }) {
  if (!summary) return null

  const status = STATUS_META[summary.status] ?? STATUS_META.safe
  const cost   = summary.energy_cost
  const ttc    = summary.time_to_critical_hours
  const risk   = summary.fault_risk_probability_pct
  const save   = cost.savings_pct

  const riskCls = risk >= 50 ? 'text-red-400' : risk >= 15 ? 'text-amber-400' : 'text-emerald-400'
  const saveCls = save > 1 ? 'text-emerald-400' : save < -1 ? 'text-red-400' : 'text-slate-300'

  return (
    <div className="space-y-3">
      {/* Status banner */}
      <div className={`rounded-lg border px-4 py-3 flex items-center gap-3 ${status.card}`}>
        <span className="text-xl leading-none">{status.dot}</span>
        <span className={`font-semibold ${status.cls}`}>{status.label}</span>
        <span className="ml-auto text-xs text-slate-500 font-mono">
          peak anomaly {summary.peak_anomaly_score.toFixed(3)}
        </span>
      </div>

      {/* Metric grid */}
      <div className="grid grid-cols-3 gap-3">
        <Card label="Fault Risk" value={`${risk.toFixed(1)}%`} valueCls={riskCls} />
        <Card
          label="Time to Critical"
          value={ttc >= 0 ? `${ttc.toFixed(1)} h` : 'None'}
          valueCls={ttc >= 0 ? 'text-amber-400' : 'text-emerald-400'}
          sub={ttc >= 0 ? 'within horizon' : 'safe for full run'}
        />
        <Card
          label="Energy vs Baseline"
          value={`${save > 0 ? '−' : save < 0 ? '+' : ''}${Math.abs(save).toFixed(1)}%`}
          valueCls={saveCls}
          sub={save > 0 ? 'cheaper' : save < 0 ? 'costlier' : 'same'}
        />
        <Card label="Scenario Cost" value={`₹${cost.scenario_cost_inr.toLocaleString()}`} valueCls="text-slate-200" />
        <Card label="Baseline Cost" value={`₹${cost.baseline_cost_inr.toLocaleString()}`} valueCls="text-slate-200" />
        <Card
          label="Savings"
          value={`₹${Math.abs(cost.savings_inr).toLocaleString()}`}
          valueCls={cost.savings_inr >= 0 ? 'text-emerald-400' : 'text-red-400'}
          sub={cost.savings_inr >= 0 ? 'saved over run' : 'extra over run'}
        />
      </div>

      {/* Recommendation */}
      <div className="rounded-lg border border-slate-700 bg-slate-800/40 px-4 py-3 border-l-4 border-l-cyan-500">
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Recommendation</p>
        <p className="text-sm text-slate-200">{summary.recommendation}</p>
      </div>
    </div>
  )
}
