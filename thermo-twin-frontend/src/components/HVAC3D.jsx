import { useEffect, useRef, useState } from 'react'
import { HVACScene } from '../utils/hvacScene.js'
import useAppStore from '../store/appStore.js'
import api from '../utils/api.js'
import ComponentDetailsPanel from './ComponentDetailsPanel.jsx'

// Single source of truth for component → sensor / health / RUL wiring.
// Sensor map matches the backend /twin/component-history endpoint.
const COMPONENTS = {
  compressor: { label: 'Compressor',       sensor: 'compressor_power_kw',    unit: 'kW',  healthKey: 'compressor_efficiency_pct', rulKey: 'compressor_rul_days',  color: '#22d3ee' },
  condenser:  { label: 'Condenser Fan',    sensor: 'fan_rpm',                unit: 'RPM', healthKey: 'fan_health_pct',            rulKey: 'fan_rul_days',         color: '#34d399' },
  evaporator: { label: 'Evaporator',       sensor: 'discharge_pressure_psi', unit: 'psi', healthKey: 'refrigerant_charge_pct',    rulKey: 'refrigerant_rul_days', color: '#a78bfa' },
  valve:      { label: 'Expansion Valve',  sensor: 'supply_air_temp_c',      unit: '°C',  healthKey: null,                        rulKey: null,                   color: '#fb923c' },
}

export function HVAC3D() {
  const canvasRef = useRef(null)
  const sceneRef  = useRef(null)

  const [selected, setSelected] = useState(null)   // component id
  const [history, setHistory]   = useState([])
  const [hover, setHover]       = useState(null)

  const sample = useAppStore((s) => s.sample)
  const twin   = useAppStore((s) => s.twin)
  const rul    = useAppStore((s) => s.rul)

  // ── Init scene once ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasRef.current) return
    const scene = new HVACScene(canvasRef.current, twin.state)
    sceneRef.current = scene
    scene.setHoverCallback(setHover)
    scene.setComponentClickCallback(async (name) => {
      setSelected(name)
      // Don't clear history immediately — keep the previous chart visible
      // while the new fetch is in-flight so we don't flash "no history yet".
      try {
        const res = await api.get(`/twin/component-history/${name}`)
        setHistory(res.data?.history ?? [])
      } catch (err) {
        console.error('component-history fetch failed:', err)
        setHistory([])
      }
    })
    return () => { scene.dispose(); sceneRef.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Feed live data into the scene ────────────────────────────────────────
  useEffect(() => { sceneRef.current?.updateSensorData(sample) }, [sample])
  useEffect(() => { sceneRef.current?.updateHealthState(twin.state) }, [twin.state])
  useEffect(() => { sceneRef.current?.updatePrediction(twin.prediction) }, [twin.prediction])

  // Build live details from the current store snapshot (stays live while open)
  let details = null
  if (selected) {
    const m = COMPONENTS[selected]
    const real = sample?.[m.sensor]
    const pred = twin.prediction?.[m.sensor]
    details = {
      label:           m.label,
      sensorUnit:      m.unit,
      color:           m.color,
      healthMonitored: m.healthKey != null,
      health:          m.healthKey ? (twin.state?.[m.healthKey] ?? 100) : null,
      real,
      predicted:       pred,
      divergence:      real != null && pred != null ? real - pred : null,
      history,
      rul:             m.rulKey ? rul?.[m.rulKey] : null,
      rulLower:        m.rulKey ? rul?.[`${m.rulKey}_lower`] : null,
      rulUpper:        m.rulKey ? rul?.[`${m.rulKey}_upper`] : null,
    }
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <p className="section-title">HVAC Unit — Interactive 3D</p>
        <span className="text-xs text-slate-600">
          drag to orbit · hover for values · click to inspect
        </span>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: '60% 1fr' }}>
        {/* ── 3D canvas ── */}
        <div className="relative rounded-lg overflow-hidden border border-slate-800" style={{ height: 480 }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full block"
            style={{ cursor: 'grab' }}
          />

          {hover && (
            <div className="absolute top-3 left-3 bg-slate-900/95 border border-slate-700 rounded px-3 py-2 text-xs pointer-events-none">
              <p className="font-semibold text-slate-200 mb-0.5">
                {COMPONENTS[hover.component]?.label ?? hover.component}
              </p>
              <p className="text-slate-400">
                real <span className="font-mono text-slate-200">{hover.real?.toFixed(1) ?? '—'}</span>
                {' · '}model <span className="font-mono text-slate-300">{hover.predicted?.toFixed(1) ?? '—'}</span>
              </p>
              <p className={hover.status === 'diverging' ? 'text-amber-400' : 'text-emerald-400'}>
                {hover.status === 'diverging' ? '⚠ diverging' : '✓ normal'}
                {hover.health != null && (
                  <span className="text-slate-500"> · health {hover.health.toFixed(0)}%</span>
                )}
              </p>
            </div>
          )}

          <div className="absolute bottom-3 left-3 flex gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: '#22c55e' }} />healthy</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: '#f59e0b' }} />warning</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full" style={{ background: '#ef4444' }} />critical</span>
          </div>
        </div>

        {/* ── Details panel (40%) ── */}
        {details ? (
          <ComponentDetailsPanel
            details={details}
            onClose={() => { setSelected(null); setHistory([]) }}
          />
        ) : (
          <div className="rounded-lg border border-slate-800 bg-slate-900/40 flex items-center justify-center text-center p-6">
            <p className="text-sm text-slate-600">
              Click a component in the 3D view to see its live sensor history and RUL prediction.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default HVAC3D
