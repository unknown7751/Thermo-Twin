const REASON_META = {
  scheduled_monthly:      { icon: '📆', cls: 'text-slate-300' },
  drift_detected:         { icon: '⚠', cls: 'text-amber-300' },
  maintenance_completed:  { icon: '🔧', cls: 'text-cyan-300' },
  seasonal_change:        { icon: '🍂', cls: 'text-violet-300' },
  manual_request:         { icon: '⚡', cls: 'text-cyan-300' },
}

function fmtTime(ts) {
  if (!ts) return '—'
  try { return new Date(ts * 1000).toLocaleTimeString() } catch { return String(ts) }
}

function EventRow({ event }) {
  const r = REASON_META[event.reason] || { icon: '•', cls: 'text-slate-400' }
  const applied = (event.parameter_updates || []).filter((u) => !u.rejected).length
  const total   = (event.parameter_updates || []).length

  return (
    <div className={`rounded border px-3 py-2 text-xs ${
      event.success ? 'border-slate-700 bg-slate-800/40' : 'border-red-800 bg-red-950/30'
    }`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-base">{r.icon}</span>
          <span className={`font-semibold ${r.cls}`}>{event.reason.replace(/_/g, ' ')}</span>
          <span className="text-slate-600 font-mono shrink-0">{fmtTime(event.timestamp)}</span>
        </div>
        <span className={`text-xs font-mono ${event.success ? 'text-emerald-400' : 'text-red-400'}`}>
          {event.success ? '✓' : '✗'} {applied}/{total} updates
        </span>
      </div>
      {(event.accuracy_before_pct != null || event.accuracy_after_pct != null) && (
        <p className="text-slate-500 mt-1 font-mono">
          accuracy {event.accuracy_before_pct ?? '—'}% → {event.accuracy_after_pct ?? '—'}%
        </p>
      )}
      {event.notes && <p className="text-slate-500 mt-1 truncate">{event.notes}</p>}
    </div>
  )
}

export default function RecalibrationEventsLog({ events }) {
  if (!events || events.length === 0) {
    return <p className="text-xs text-slate-600 text-center py-4">No recalibration events yet.</p>
  }
  return (
    <div className="space-y-1.5 max-h-72 overflow-y-auto pr-1 custom-scrollbar">
      {events.slice().reverse().map((e, i) => (
        <EventRow key={`${e.timestamp}-${i}`} event={e} />
      ))}
    </div>
  )
}
