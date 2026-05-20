import { NavLink } from 'react-router-dom'
import useAppStore from '../store/appStore.js'

const navLink = ({ isActive }) =>
  `px-3 py-1.5 rounded text-xs font-medium transition-all ${
    isActive
      ? 'bg-cyan-900/40 border border-cyan-700 text-cyan-300'
      : 'border border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
  }`

export default function Header() {
  const stream        = useAppStore((s) => s.stream)
  const backendOnline = stream.backendOnline
  const sampleCount   = stream.sampleCount
  const alertCount    = stream.alertCount

  return (
    <header className="sticky top-0 z-50 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-6 py-3">
      <div className="max-w-[1920px] mx-auto flex items-center justify-between">

        {/* ── Brand + nav ── */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold tracking-tight text-white">
              Thermo<span className="text-cyan-400">Twin</span>
            </span>
            <span className="text-xs text-slate-500 font-mono">Digital Twin</span>
          </div>
          <nav className="flex items-center gap-1">
            <NavLink to="/"      end className={navLink}>Single Unit</NavLink>
            <NavLink to="/fleet"     className={navLink}>Fleet</NavLink>
          </nav>
        </div>

        {/* ── Status chips ── */}
        <div className="flex items-center gap-4">

          {/* Live indicator */}
          <div className="flex items-center gap-2">
            <span className={`inline-block w-2 h-2 rounded-full ${
              backendOnline ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'
            }`} />
            <span className={`text-xs font-medium ${
              backendOnline ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {backendOnline ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>

          {/* Sample count */}
          <div className="flex items-center gap-1 text-xs text-slate-400">
            <span className="text-slate-500">Samples</span>
            <span className="font-mono text-slate-200">{sampleCount.toLocaleString()}</span>
          </div>

          {/* Alert count */}
          <div className={`flex items-center gap-1 text-xs ${
            alertCount > 0 ? 'text-amber-400' : 'text-slate-400'
          }`}>
            <span className="text-slate-500">Alerts</span>
            <span className="font-mono">{alertCount}</span>
          </div>

          {/* Backend health dot */}
          <div className={`px-2 py-0.5 rounded text-xs font-medium ${
            backendOnline
              ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800'
              : 'bg-red-950/60 text-red-400 border border-red-800'
          }`}>
            {backendOnline ? 'Backend OK' : 'Connecting…'}
          </div>

        </div>
      </div>
    </header>
  )
}
