const PARAM_META = {
  k_disc:   { label: 'k_disc',   desc: 'discharge_pressure / power',  unit: 'psi/kW' },
  k_fan:    { label: 'k_fan',    desc: 'fan_rpm / power',             unit: 'RPM/kW' },
  k_temp_a: { label: 'k_temp_a', desc: 'temp intercept',              unit: '°C'     },
  k_temp_b: { label: 'k_temp_b', desc: 'temp slope vs power',         unit: '°C/kW'  },
}

function changeClasses(pct, rejected) {
  if (rejected) return 'text-slate-500 line-through'
  if (Math.abs(pct) < 0.5) return 'text-slate-400'
  return pct > 0 ? 'text-amber-300' : 'text-cyan-300'
}

export default function ParametersCard({ parameters }) {
  if (!parameters?.current) {
    return <p className="text-xs text-slate-600 text-center py-6">Loading parameters…</p>
  }
  const cur     = parameters.current
  const updates = parameters.recent_updates || []
  const buffered = parameters.buffered_samples ?? 0

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500 uppercase tracking-wider">Physics Coefficients</p>
        <span className="text-xs text-slate-600 font-mono">{buffered} samples buffered</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {Object.entries(PARAM_META).map(([key, m]) => (
          <div key={key} className="bg-slate-800/50 border border-slate-700 rounded p-2.5">
            <div className="flex items-baseline justify-between">
              <span className="text-xs text-slate-400 font-mono">{m.label}</span>
              <span className="text-xs text-slate-600">{m.unit}</span>
            </div>
            <p className="font-mono text-lg text-slate-100">
              {cur[key]?.toFixed(2) ?? '—'}
            </p>
            <p className="text-xs text-slate-600">{m.desc}</p>
          </div>
        ))}
      </div>

      {updates.length > 0 && (
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Recent Updates</p>
          <div className="space-y-1 max-h-32 overflow-y-auto pr-1 custom-scrollbar">
            {updates.slice().reverse().slice(0, 8).map((u, i) => (
              <div key={i}
                   className={`flex items-center justify-between text-xs px-2 py-1 rounded border
                               ${u.rejected ? 'border-slate-800 bg-slate-900/40' : 'border-slate-700 bg-slate-800/40'}`}>
                <span className="font-mono text-slate-400">{u.parameter_name}</span>
                <span className="text-slate-600 font-mono text-[10px]">
                  {u.old_value?.toFixed(2)} → {u.new_value?.toFixed(2)}
                </span>
                <span className={`font-mono ${changeClasses(u.change_pct, u.rejected)}`}>
                  {u.change_pct >= 0 ? '+' : ''}{u.change_pct?.toFixed(2)}%
                </span>
                <span className="text-[10px] text-slate-600">
                  c={u.confidence?.toFixed(2)}
                  {u.rejected && <span className="text-red-400 ml-1" title={u.reject_reason}>·rejected</span>}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
