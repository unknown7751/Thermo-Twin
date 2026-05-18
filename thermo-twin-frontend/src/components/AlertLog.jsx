import useAppStore from '../store/appStore.js'
import { FAULT_LABELS, severityClasses, formatAlertTime } from '../utils/formatters.js'

function AlertRow({ alert, isLatest }) {
  const severity = alert.severity_score ?? alert.severity ?? 0
  const label    = FAULT_LABELS[alert.fault_type] ?? alert.fault_type ?? 'Unknown'
  const time     = formatAlertTime(alert)
  const summary  = alert.explanation?.summary ?? alert.prescription?.fault ?? null

  return (
    <div className={`flex items-start gap-3 px-3 py-2.5 rounded border transition-all ${
      isLatest
        ? 'border-amber-700/60 bg-amber-950/20 animate-[slideIn_0.3s_ease]'
        : 'border-slate-800 bg-slate-900/40'
    }`}>
      {/* Severity badge */}
      <div className={`shrink-0 w-10 text-center rounded px-1 py-0.5 text-xs font-bold font-mono ${
        severity >= 71 ? 'bg-red-950 text-red-400' :
        severity >= 50 ? 'bg-amber-950 text-amber-400' :
                         'bg-slate-800 text-slate-300'
      }`}>
        {severity.toFixed(0)}
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-semibold ${severityClasses(severity)}`}>{label}</span>
          <span className="text-xs text-slate-500 font-mono">t={time}</span>
        </div>
        {summary && (
          <p className="text-xs text-slate-500 mt-0.5 truncate">{summary}</p>
        )}
      </div>

      {/* Timestamp right */}
      <span className="text-xs text-slate-600 font-mono shrink-0">{time}</span>
    </div>
  )
}

export default function AlertLog() {
  const alerts = useAppStore((s) => s.alerts)
  const { latest, history } = alerts

  if (!history.length) {
    return (
      <div className="card">
        <p className="section-title mb-2">Alert Log</p>
        <p className="text-xs text-slate-600 text-center py-6">No alerts yet — system nominal</p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <p className="section-title">Alert Log</p>
        <span className="text-xs text-slate-500">{history.length} event{history.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1 custom-scrollbar">
        {history.map((alert, i) => (
          <AlertRow
            key={`${alert.timestamp}-${i}`}
            alert={alert}
            isLatest={i === 0 && alert === latest}
          />
        ))}
      </div>
    </div>
  )
}
