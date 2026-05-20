import { Link } from 'react-router-dom'

const PATTERN_META = {
  refrigerant_loss:       { icon: '💧', label: 'Refrigerant loss',       tone: 'cyan'   },
  condenser_fan_failure:  { icon: '🌀', label: 'Condenser fan failure',  tone: 'amber'  },
  compressor_wear:        { icon: '⚙',  label: 'Compressor wear',        tone: 'violet' },
}

const TONE = {
  cyan:   { border: 'border-cyan-800',   bg: 'bg-cyan-950/30',   text: 'text-cyan-300'   },
  amber:  { border: 'border-amber-800',  bg: 'bg-amber-950/30',  text: 'text-amber-300'  },
  violet: { border: 'border-violet-800', bg: 'bg-violet-950/30', text: 'text-violet-300' },
  slate:  { border: 'border-slate-700',  bg: 'bg-slate-800/40',  text: 'text-slate-300'  },
}

export default function AnomaliesSection({ anomalies }) {
  if (anomalies == null) {
    return <p className="text-xs text-slate-600 text-center py-6">Loading anomalies…</p>
  }
  if (anomalies.length === 0) {
    return (
      <div className="rounded border border-emerald-900/60 bg-emerald-950/20 px-4 py-3 text-sm text-emerald-400">
        ✓ No cross-unit anomalies detected. Fleet patterns look independent.
      </div>
    )
  }
  return (
    <div className="space-y-2">
      {anomalies.map((a, i) => {
        const meta = PATTERN_META[a.pattern] || { icon: '⚠', label: a.pattern, tone: 'slate' }
        const t    = TONE[meta.tone]
        return (
          <div key={i} className={`rounded border ${t.border} ${t.bg} p-3`}>
            <div className="flex items-start gap-3">
              <span className="text-2xl leading-none">{meta.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between gap-2 flex-wrap">
                  <p className={`font-semibold ${t.text}`}>{meta.label}</p>
                  <span className="text-xs text-slate-500 font-mono">
                    confidence {(a.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-slate-200 mt-1">{a.description}</p>
                {a.location_cluster && (
                  <p className="text-xs text-slate-400 mt-1">
                    Cluster: <span className="text-slate-200">{a.location_cluster}</span>
                  </p>
                )}
                <p className="text-xs text-slate-500 mt-1.5">
                  <span className="text-slate-600">Hypothesis: </span>{a.root_cause_hypothesis}
                </p>
                <p className="text-xs text-amber-300 mt-1">
                  <span className="text-slate-500">Action: </span>{a.recommended_action}
                </p>
                <div className="flex flex-wrap gap-1.5 mt-2">
                  {(a.affected_units || []).map((u) => (
                    <Link
                      key={u}
                      to={`/fleet/${encodeURIComponent(u)}`}
                      className="px-2 py-0.5 rounded text-xs font-mono bg-slate-900/60 border border-slate-700 hover:border-cyan-700 text-slate-200"
                    >
                      {u}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
