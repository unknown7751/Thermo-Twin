import { useFaultInjection } from '../hooks/useFaultInjection.js'
import { FAULT_LABELS } from '../utils/formatters.js'

const FAULT_TYPES = ['refrigerant_leak', 'fan_failure', 'compressor_wear']

const FAULT_COLORS = {
  refrigerant_leak: 'border-cyan-700 hover:border-cyan-500 hover:bg-cyan-950/40 text-cyan-300',
  fan_failure:      'border-emerald-700 hover:border-emerald-500 hover:bg-emerald-950/40 text-emerald-300',
  compressor_wear:  'border-violet-700 hover:border-violet-500 hover:bg-violet-950/40 text-violet-300',
}

const FAULT_ACTIVE_COLORS = {
  refrigerant_leak: 'border-cyan-500 bg-cyan-950/40 text-cyan-300',
  fan_failure:      'border-emerald-500 bg-emerald-950/40 text-emerald-300',
  compressor_wear:  'border-violet-500 bg-violet-950/40 text-violet-300',
}

export default function FaultControls() {
  const { injectFault, resetStream, fault } = useFaultInjection()
  const { active, countdownSec } = fault

  const FAULT_TOTAL_SECS = 10   // delay(5) + duration(5)
  const progressPct = active
    ? Math.max(0, Math.min(100, (countdownSec / FAULT_TOTAL_SECS) * 100))
    : 0

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <p className="section-title">Fault Injection</p>
        {active && (
          <span className="text-xs font-mono text-amber-400 animate-pulse">
            {FAULT_LABELS[active]} · {countdownSec}s remaining
          </span>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap">

        {/* Fault buttons */}
        {FAULT_TYPES.map((ft) => {
          const isActive = active === ft
          return (
            <button
              key={ft}
              onClick={() => injectFault(ft, 5, 5)}
              disabled={!!active}
              className={`px-4 py-2 rounded border text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                isActive ? FAULT_ACTIVE_COLORS[ft] : FAULT_COLORS[ft]
              }`}
            >
              {isActive && <span className="mr-1.5">●</span>}
              {FAULT_LABELS[ft]}
            </button>
          )
        })}

        {/* Divider */}
        <div className="flex-1" />

        {/* Reset */}
        <button
          onClick={resetStream}
          className="px-4 py-2 rounded border border-slate-700 hover:border-red-700 hover:bg-red-950/30 text-slate-400 hover:text-red-300 text-sm font-medium transition-all"
        >
          Reset Stream
        </button>
      </div>

      {/* Countdown bar */}
      {active && (
        <div className="mt-3 h-1 rounded-full bg-slate-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-amber-500 transition-all duration-1000"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}
    </div>
  )
}
