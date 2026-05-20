import { Link } from 'react-router-dom'
import { formatRUL } from '../utils/formatters.js'

const COMPONENT_LABEL = {
  refrigerant: 'Refrigerant',
  compressor:  'Compressor',
  fan:         'Condenser Fan',
}

function priorityClasses(rul) {
  if (rul <= 1)  return { row: 'border-red-800 bg-red-950/30',     pill: 'bg-red-900 text-red-200'    }
  if (rul <= 7)  return { row: 'border-amber-800 bg-amber-950/20', pill: 'bg-amber-900 text-amber-200' }
  return            { row: 'border-slate-700 bg-slate-800/40',   pill: 'bg-slate-700 text-slate-200' }
}

export default function DispatchQueue({ queue }) {
  if (!queue) {
    return <p className="text-xs text-slate-600 text-center py-6">Loading dispatch queue…</p>
  }
  const items = queue.queue || []
  if (items.length === 0) {
    return (
      <p className="text-xs text-slate-600 text-center py-6">
        No actionable units — fleet is healthy.
      </p>
    )
  }
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">{queue.total_units_in_queue} units in queue</span>
        <span className="text-slate-500">
          Est. dispatch{' '}
          <span className="text-slate-300 font-mono">₹{queue.estimated_dispatch_cost_inr?.toLocaleString()}</span>
          {' · '}save{' '}
          <span className="text-emerald-400 font-mono">₹{queue.estimated_save_by_proactive_inr?.toLocaleString()}</span>
          {' · '}ROI{' '}
          <span className="text-emerald-400 font-mono">{queue.net_roi_pct}%</span>
        </span>
      </div>

      <div className="space-y-2">
        {items.map((it) => {
          const cls = priorityClasses(it.rul_days)
          const rulFmt = formatRUL(it.rul_days)
          return (
            <Link
              key={it.machine_id}
              to={`/fleet/${encodeURIComponent(it.machine_id)}`}
              className={`block rounded border ${cls.row} px-3 py-2 hover:border-cyan-700 transition-all`}
            >
              <div className="flex items-center gap-3">
                <span className={`shrink-0 w-9 text-center px-1 py-0.5 rounded text-xs font-bold font-mono ${cls.pill}`}>
                  #{it.priority}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-semibold text-slate-100 text-sm">{it.machine_id}</span>
                    <span className="text-xs text-slate-500">·</span>
                    <span className="text-xs text-slate-400">{it.location}</span>
                    <span className="text-xs text-slate-600 ml-auto font-mono">
                      p(fail)={it.fault_probability}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">
                    <span className="text-slate-500">{COMPONENT_LABEL[it.most_critical_component] || it.most_critical_component}</span>
                    {' · '}
                    <span className="font-mono text-amber-300">{rulFmt.value} {rulFmt.unit}</span>
                    {' '}<span className="text-slate-600 font-mono">[{it.rul_days_lower}–{it.rul_days_upper}]</span>
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5 truncate">{it.recommended_action}</p>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
