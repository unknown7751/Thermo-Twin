import { useState } from 'react'
import useAppStore from '../store/appStore.js'
import { useRecalibration } from '../hooks/useRecalibration.js'
import AccuracyStatusCard         from './AccuracyStatusCard.jsx'
import ParametersCard             from './ParametersCard.jsx'
import RecalibrationScheduleCard  from './RecalibrationScheduleCard.jsx'
import RecalibrationEventsLog     from './RecalibrationEventsLog.jsx'
import MaintenanceModal           from './MaintenanceModal.jsx'

export default function RecalibrationPanel() {
  const { triggerRecalibration, commissioningReset } = useRecalibration()
  const recal = useAppStore((s) => s.recal)
  const [modalOpen, setModalOpen] = useState(false)

  return (
    <>
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <p className="section-title">Auto-Recalibration &amp; Drift</p>
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-600">
              {recal.online ? '● live' : '○ offline'}
              {recal.lastFetched && ` · ${new Date(recal.lastFetched).toLocaleTimeString()}`}
            </span>
            <button
              onClick={() => setModalOpen(true)}
              className="text-xs px-3 py-1 rounded border border-amber-700 hover:bg-amber-950/40
                         text-amber-300"
            >
              🔧 Commissioning reset
            </button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <section>
            <AccuracyStatusCard drift={recal.drift} />
          </section>
          <section>
            <ParametersCard parameters={recal.parameters} />
          </section>
        </div>

        <div className="grid gap-4 md:grid-cols-2 mt-4 pt-4 border-t border-slate-800">
          <section>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Schedule</p>
            <RecalibrationScheduleCard status={recal.status} onTrigger={triggerRecalibration} />
          </section>
          <section>
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Event Log</p>
            <RecalibrationEventsLog events={recal.status?.recent_events} />
          </section>
        </div>
      </div>

      <MaintenanceModal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        onConfirm={commissioningReset}
      />
    </>
  )
}
