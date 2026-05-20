const SENSOR_LABEL = {
  compressor_power_kw:    'Compressor Power',
  discharge_pressure_psi: 'Discharge Pressure',
  fan_rpm:                'Fan RPM',
  supply_air_temp_c:      'Supply Air Temp',
}

function accClasses(p) {
  if (p == null) return { text: 'text-slate-500', bar: 'bg-slate-700' }
  if (p >= 95)   return { text: 'text-emerald-400', bar: 'bg-emerald-500' }
  if (p >= 80)   return { text: 'text-amber-400',   bar: 'bg-amber-500'   }
  return            { text: 'text-red-400',     bar: 'bg-red-500'     }
}

export default function AccuracyStatusCard({ drift }) {
  if (!drift?.current) {
    return <p className="text-xs text-slate-600 text-center py-6">Loading drift metrics…</p>
  }
  const c = drift.current
  const t = drift.trend_24h
  const a = accClasses(c.accuracy_pct)
  const trendCls = !t?.is_trending_down ? 'text-emerald-400' : 'text-amber-400'

  return (
    <div className="space-y-3">
      {/* Headline */}
      <div className="flex items-baseline justify-between">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">Twin Accuracy</p>
          <p className={`text-4xl font-bold font-mono ${a.text}`}>
            {c.accuracy_pct?.toFixed(1)}<span className="text-sm text-slate-500 font-normal ml-1">%</span>
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500">{c.is_drifting ? '⚠ Drifting' : '✓ Stable'}</p>
          <p className="text-xs text-slate-600 font-mono">{c.drift_reason}</p>
        </div>
      </div>

      {/* Accuracy bar */}
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${a.bar}`}
             style={{ width: `${Math.max(0, Math.min(100, c.accuracy_pct ?? 0))}%` }} />
      </div>

      {/* Trend */}
      {t && !t.insufficient_data && (
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>
            24h trend{' '}
            <span className={`font-mono ${trendCls}`}>
              {t.drift_rate_pct_per_hour >= 0 ? '+' : ''}{t.drift_rate_pct_per_hour} pp/h
            </span>
          </span>
          <span>
            avg <span className="font-mono text-slate-300">{t.avg_accuracy_pct}%</span>
            {' · '}min <span className="font-mono text-slate-300">{t.min_accuracy_pct}%</span>
          </span>
        </div>
      )}
      {t?.recommendation && (
        <p className="text-xs text-slate-400 bg-slate-800/40 border border-slate-700 rounded px-2 py-1.5">
          {t.recommendation}
        </p>
      )}

      {/* Per-sensor breakdown */}
      {c.per_sensor && Object.keys(c.per_sensor).length > 0 && (
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">Per-Sensor Error</p>
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="pb-1 text-xs text-slate-600 font-normal">Sensor</th>
                <th className="pb-1 text-xs text-slate-600 font-normal text-right">MAE</th>
                <th className="pb-1 text-xs text-slate-600 font-normal text-right">RMSE</th>
                <th className="pb-1 text-xs text-slate-600 font-normal text-right">Norm</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(c.per_sensor).map(([k, v]) => (
                <tr key={k} className="border-b border-slate-800/60 last:border-0">
                  <td className="py-1 text-xs text-slate-400">{SENSOR_LABEL[k] || k}</td>
                  <td className="py-1 text-xs font-mono text-slate-300 text-right">{v.mae?.toFixed(2)}</td>
                  <td className="py-1 text-xs font-mono text-slate-500 text-right">{v.rmse?.toFixed(2)}</td>
                  <td className="py-1 text-xs font-mono text-slate-300 text-right">{(v.normalised_mae * 100)?.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
