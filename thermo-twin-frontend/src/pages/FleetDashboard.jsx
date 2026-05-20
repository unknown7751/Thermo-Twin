import useAppStore from '../store/appStore.js'
import { useFleet } from '../hooks/useFleet.js'
import FleetSummaryCards from '../components/FleetSummaryCards.jsx'
import UnitGrid          from '../components/UnitGrid.jsx'
import DispatchQueue     from '../components/DispatchQueue.jsx'
import AnomaliesSection  from '../components/AnomaliesSection.jsx'

export default function FleetDashboard() {
  useFleet()
  const fleet = useAppStore((s) => s.fleet)

  return (
    <main className="flex-1 max-w-[1920px] w-full mx-auto px-4 pb-8 space-y-6 pt-4">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-bold text-slate-100">Fleet Overview</h1>
        <span className="text-xs text-slate-500">
          {fleet.online ? '● live' : '○ offline'}
          {fleet.lastFetched && ` · last sync ${new Date(fleet.lastFetched).toLocaleTimeString()}`}
        </span>
      </div>

      {/* Row 1 — summary */}
      <section className="card">
        <p className="section-title mb-3">Fleet Health</p>
        <FleetSummaryCards health={fleet.health} />
      </section>

      {/* Row 2 — unit grid */}
      <section className="card">
        <p className="section-title mb-3">All Units</p>
        <UnitGrid
          units={fleet.units}
          metadata={fleet.metadata}
          healthByUnit={fleet.health?.health_by_unit}
        />
      </section>

      {/* Row 3 — dispatch + anomalies side by side on wide screens */}
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card">
          <p className="section-title mb-3">Maintenance Dispatch Queue</p>
          <DispatchQueue queue={fleet.queue} />
        </section>
        <section className="card">
          <p className="section-title mb-3">Cross-Unit Anomalies</p>
          <AnomaliesSection anomalies={fleet.online ? fleet.anomalies : null} />
        </section>
      </div>
    </main>
  )
}
