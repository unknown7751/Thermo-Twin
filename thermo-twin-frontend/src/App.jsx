import { useEffect } from 'react'
import Header        from './components/Header.jsx'
import ChartPanel    from './components/ChartPanel.jsx'
import Sidebar       from './components/Sidebar.jsx'
import FaultControls from './components/FaultControls.jsx'
import TwinPanel     from './components/TwinPanel.jsx'
import RULPanel      from './components/RULPanel.jsx'
import ShapPanel     from './components/ShapPanel.jsx'
import AlertLog      from './components/AlertLog.jsx'
import { useStreamData } from './hooks/useStreamData.js'

export default function App() {
  // Start polling — updates Zustand store + fires sampleBus for chart
  useStreamData()

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <Header />

      <main className="flex-1 max-w-[1920px] w-full mx-auto px-4 pb-8 space-y-4 pt-4">

        {/* ── Row 1: Live chart + sidebar ── */}
        <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 280px' }}>
          <ChartPanel />
          <Sidebar />
        </div>

        {/* ── Row 2: Fault injection controls ── */}
        <FaultControls />

        {/* ── Row 3: Digital twin state + RUL prognostics ── */}
        <div className="grid grid-cols-2 gap-4">
          <TwinPanel />
          <RULPanel />
        </div>

        {/* ── Row 4: SHAP attribution (hidden until first alert) ── */}
        <ShapPanel />

        {/* ── Row 5: Alert history ── */}
        <AlertLog />

      </main>
    </div>
  )
}
