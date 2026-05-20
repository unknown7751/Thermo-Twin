function Card({ label, value, sub, valueCls = 'text-slate-100' }) {
  return (
    <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3">
      <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold font-mono ${valueCls}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  )
}

export default function FleetSummaryCards({ health }) {
  if (!health) {
    return (
      <div className="text-xs text-slate-600 text-center py-6">
        Loading fleet health…
      </div>
    )
  }
  const { total_units, healthy, warning, critical, avg_health_pct, fleet_uptime_pct } = health
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      <Card label="Total Units"   value={total_units}                              valueCls="text-cyan-300" />
      <Card label="Healthy"        value={healthy}      sub="health > 80%"          valueCls="text-emerald-400" />
      <Card label="Warning"        value={warning}      sub="50–80%"                valueCls="text-amber-400" />
      <Card label="Critical"       value={critical}     sub="≤ 50%"                 valueCls="text-red-400" />
      <Card label="Avg Health"     value={`${avg_health_pct?.toFixed(1)}%`}        valueCls="text-slate-100" />
      <Card label="Fleet Uptime"   value={`${fleet_uptime_pct?.toFixed(1)}%`}      valueCls={fleet_uptime_pct >= 80 ? 'text-emerald-400' : 'text-amber-400'} />
    </div>
  )
}
