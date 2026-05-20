import { Link } from 'react-router-dom'

function statusOf(h) {
  if (h == null) return { label: '—',        dot: 'bg-slate-600',   bar: 'bg-slate-700',  text: 'text-slate-500' }
  if (h > 80)   return { label: 'Healthy',   dot: 'bg-emerald-400', bar: 'bg-emerald-500', text: 'text-emerald-400' }
  if (h > 50)   return { label: 'Warning',   dot: 'bg-amber-400',   bar: 'bg-amber-500',  text: 'text-amber-400' }
  return        { label: 'Critical',  dot: 'bg-red-500',     bar: 'bg-red-500',    text: 'text-red-400' }
}

function UnitCard({ machineId, metadata, health }) {
  const s = statusOf(health)
  return (
    <Link
      to={`/fleet/${encodeURIComponent(machineId)}`}
      className="block rounded-lg border border-slate-700 bg-slate-800/40 hover:border-cyan-700 hover:bg-slate-800/70 transition-all p-3"
    >
      <div className="flex items-start justify-between mb-2">
        <div className="min-w-0">
          <p className="font-semibold text-slate-100 text-sm truncate">{machineId}</p>
          <p className="text-xs text-slate-500 truncate">{metadata?.location || '—'}</p>
        </div>
        <span className={`inline-block w-2 h-2 rounded-full ${s.dot} mt-1.5 shrink-0`} />
      </div>
      <div className="flex items-baseline justify-between mb-1">
        <span className={`text-xs font-semibold ${s.text}`}>{s.label}</span>
        <span className="font-mono text-sm text-slate-200">
          {health != null ? `${health.toFixed(1)}%` : '—'}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-900 overflow-hidden">
        <div className={`h-full rounded-full ${s.bar}`} style={{ width: `${Math.max(0, Math.min(100, health ?? 0))}%` }} />
      </div>
      {metadata?.fault_profile && (
        <p className="text-[10px] text-amber-400/80 mt-1.5 font-mono">
          profile: {metadata.fault_profile}
        </p>
      )}
    </Link>
  )
}

export default function UnitGrid({ units, metadata, healthByUnit }) {
  if (!units?.length) {
    return <p className="text-xs text-slate-600 text-center py-6">No units registered.</p>
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
      {units.map((m) => (
        <UnitCard
          key={m}
          machineId={m}
          metadata={metadata?.[m]}
          health={healthByUnit?.[m]}
        />
      ))}
    </div>
  )
}
