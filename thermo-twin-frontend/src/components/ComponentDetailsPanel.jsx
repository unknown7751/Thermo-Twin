import SensorHistoryChart from './SensorHistoryChart.jsx'
import { formatRUL } from '../utils/formatters.js'

function healthColor(h) {
  if (h > 66) return 'text-emerald-400'
  if (h > 33) return 'text-amber-400'
  return 'text-red-400'
}
function healthBadge(h) {
  if (h > 66) return '🟢'
  if (h > 33) return '🟡'
  return '🔴'
}

export default function ComponentDetailsPanel({ details, onClose }) {
  const {
    label, sensorUnit,
    health, real, predicted, divergence,
    history, color,
    rul, rulLower, rulUpper, healthMonitored,
  } = details

  const divergent = divergence != null &&
    Math.abs(divergence) > Math.abs((predicted ?? 1) * 0.05)

  const rulFmt   = rul != null ? formatRUL(rul) : null
  const loFmt    = rulLower != null ? formatRUL(rulLower) : null
  const hiFmt    = rulUpper != null ? formatRUL(rulUpper) : null

  return (
    <div className="w-full rounded-lg border border-slate-700 bg-slate-800/60 p-4 flex flex-col gap-4">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-base font-semibold text-slate-100">{label}</h3>
          {healthMonitored ? (
            <p className={`text-2xl font-bold font-mono ${healthColor(health)}`}>
              {healthBadge(health)} {health.toFixed(1)}%
            </p>
          ) : (
            <p className="text-sm text-slate-500 mt-1">passive · not health-monitored</p>
          )}
        </div>
        <button
          onClick={onClose}
          className="text-slate-500 hover:text-slate-200 text-sm"
          aria-label="Close"
        >✕</button>
      </div>

      {/* Real / predicted / divergence */}
      <div className="bg-slate-900/50 rounded p-3 text-sm space-y-1.5">
        <div className="flex justify-between">
          <span className="text-slate-500">Real</span>
          <span className="font-mono text-slate-100">
            {real != null ? real.toFixed(2) : '—'}
            <span className="text-slate-600 ml-1">{sensorUnit}</span>
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Predicted</span>
          <span className="font-mono text-slate-300">
            {predicted != null ? predicted.toFixed(2) : '—'}
            <span className="text-slate-600 ml-1">{sensorUnit}</span>
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500">Divergence</span>
          <span className={`font-mono ${divergent ? 'text-amber-400' : 'text-emerald-400'}`}>
            {divergence != null ? (divergence >= 0 ? '+' : '') + divergence.toFixed(2) : '—'}
            <span className="ml-1">{divergent ? '⚠' : '✓'}</span>
          </span>
        </div>
      </div>

      {/* 24-h-style history */}
      <div>
        <h4 className="text-xs text-slate-500 uppercase tracking-wider mb-2">
          Recent Sensor History
        </h4>
        <SensorHistoryChart history={history} color={color} />
      </div>

      {/* RUL */}
      {rulFmt && (
        <div className="bg-slate-900/50 rounded p-3 border-l-4 border-l-cyan-500">
          <p className="text-xs text-slate-500">Remaining Useful Life</p>
          <p className="text-xl font-bold font-mono text-cyan-300">
            {rulFmt.value}{' '}
            <span className="text-xs text-slate-500 font-normal">{rulFmt.unit}</span>
          </p>
          {loFmt && hiFmt && rulFmt.status !== 'healthy' && (
            <p className="text-xs text-slate-600 mt-1 font-mono">
              CI: {loFmt.value} – {hiFmt.value}
            </p>
          )}
        </div>
      )}
      {healthMonitored && !rulFmt && (
        <p className="text-xs text-slate-600">RUL not computed yet — stream warming up.</p>
      )}
    </div>
  )
}
