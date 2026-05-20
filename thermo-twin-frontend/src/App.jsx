import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Header             from './components/Header.jsx'
import SingleUnitDashboard from './pages/SingleUnitDashboard.jsx'
import FleetDashboard      from './pages/FleetDashboard.jsx'
import UnitDetailPage      from './pages/UnitDetailPage.jsx'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
        <Header />
        <Routes>
          <Route path="/"                     element={<SingleUnitDashboard />} />
          <Route path="/fleet"                element={<FleetDashboard />} />
          <Route path="/fleet/:machineId"     element={<UnitDetailPage />} />
          <Route path="*"                     element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
