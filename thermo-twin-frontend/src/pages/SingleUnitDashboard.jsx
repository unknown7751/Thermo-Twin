import ChartPanel    from '../components/ChartPanel.jsx'
import Sidebar       from '../components/Sidebar.jsx'
import FaultControls from '../components/FaultControls.jsx'
import HVAC3D        from '../components/HVAC3D.jsx'
import TwinPanel     from '../components/TwinPanel.jsx'
import RULPanel      from '../components/RULPanel.jsx'
import WhatIfPanel        from '../components/WhatIfPanel.jsx'
import RecalibrationPanel from '../components/RecalibrationPanel.jsx'
import ShapPanel          from '../components/ShapPanel.jsx'
import AlertLog      from '../components/AlertLog.jsx'
import { useStreamData } from '../hooks/useStreamData.js'

export default function SingleUnitDashboard() {
  // Poll /stream/next-sample → store + sampleBus.
  // Lives here so it only runs on this route, not on the fleet pages.
  useStreamData()

  return (
    <main className="flex-1 max-w-[1920px] w-full mx-auto px-4 pb-8 space-y-4 pt-4">
      <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 280px' }}>
        <ChartPanel />
        <Sidebar />
      </div>

      <FaultControls />

      <HVAC3D />

      <div className="grid grid-cols-2 gap-4">
        <TwinPanel />
        <RULPanel />
      </div>

      <WhatIfPanel />
      <RecalibrationPanel />
      <ShapPanel />
      <AlertLog />
    </main>
  )
}
