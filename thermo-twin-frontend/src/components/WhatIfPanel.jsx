import { useState } from 'react'
import api from '../utils/api.js'
import { SliderGroup } from './SliderInput.jsx'
import SummaryCards from './SummaryCards.jsx'
import TrajectoryChart from './TrajectoryChart.jsx'

const BASELINE_SLIDERS = {
  compressor_speed_pct: 70,
  ambient_temp_c:       35,
  load_demand_pct:      50,
  duration_hours:       4,
}

export default function WhatIfPanel() {
  const [sliders, setSliders]   = useState({ ...BASELINE_SLIDERS })
  const [isLoading, setLoading] = useState(false)
  const [result, setResult]     = useState(null)
  const [showBaseline, setShowBaseline] = useState(false)
  const [error, setError]       = useState(null)

  const handleSlider = (key, value) =>
    setSliders((prev) => ({ ...prev, [key]: value }))

  const runSimulation = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.post('/twin/whatif', {
        compressor_speed_pct:      sliders.compressor_speed_pct,
        ambient_temp_c:            sliders.ambient_temp_c,
        load_demand_pct:           sliders.load_demand_pct,
        simulation_duration_hours: sliders.duration_hours,
      })
      setResult(res.data)
    } catch (err) {
      console.error('What-if simulation failed:', err)
      setError(err?.response?.data?.error ?? 'Simulation request failed')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSliders({ ...BASELINE_SLIDERS })
    setResult(null)
    setShowBaseline(false)
    setError(null)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <p className="section-title">What-If Simulator</p>
        <span className="text-xs text-slate-600">Test scenarios before deploying</span>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Projects the twin forward from its <span className="text-slate-400">current estimated health</span> under the
        chosen operating conditions.
      </p>

      {/* ── Sliders ── */}
      <SliderGroup sliders={sliders} onChange={handleSlider} />

      {/* ── Controls ── */}
      <div className="flex items-center gap-3 mt-5 mb-2 flex-wrap">
        <button
          onClick={runSimulation}
          disabled={isLoading}
          className="px-5 py-2 rounded border border-cyan-700 bg-cyan-950/40 hover:bg-cyan-900/50 text-cyan-300 text-sm font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Running…' : '⚡ Run Simulation'}
        </button>
        <button
          onClick={handleReset}
          className="px-4 py-2 rounded border border-slate-700 hover:border-slate-500 text-slate-400 hover:text-slate-200 text-sm transition-all"
        >
          Reset
        </button>
        <label className="flex items-center gap-2 text-xs text-slate-400 ml-auto cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showBaseline}
            onChange={(e) => setShowBaseline(e.target.checked)}
            className="accent-cyan-500"
            disabled={!result}
          />
          Compare to baseline
        </label>
      </div>

      {error && (
        <p className="text-xs text-red-400 mt-2 bg-red-950/30 border border-red-900 rounded px-3 py-2">
          {error}
        </p>
      )}

      {/* ── Results ── */}
      {result && (
        <div className="mt-5 space-y-5">
          <SummaryCards summary={result.summary} />
          <div className="border-t border-slate-800 pt-4">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">
              Predicted Sensor Trajectory ({sliders.duration_hours}h horizon)
            </p>
            <TrajectoryChart result={result} showBaseline={showBaseline} />
          </div>
        </div>
      )}

      {!result && !isLoading && (
        <p className="text-xs text-slate-600 text-center py-6 mt-3 border-t border-slate-800">
          Adjust the sliders and run a simulation to see the projected outcome.
        </p>
      )}
    </div>
  )
}
