export const WHATIF_SPECS = [
  { key: 'compressor_speed_pct', label: 'Compressor Speed',    min: 20,  max: 100, step: 1,   unit: '%'  },
  { key: 'ambient_temp_c',       label: 'Ambient Temperature', min: 15,  max: 50,  step: 1,   unit: '°C' },
  { key: 'load_demand_pct',      label: 'Load Demand',         min: 10,  max: 100, step: 1,   unit: '%'  },
  { key: 'duration_hours',       label: 'Simulation Duration', min: 0.5, max: 5000, step: 0.5, unit: 'h'  },
]

import { useEffect, useState } from 'react'

const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

function SliderInput({ label, value, min, max, step, unit, onChange }) {
  const pct = ((value - min) / (max - min)) * 100

  // Local draft so typing isn't fought by the controlled value;
  // committed (clamped) on blur / Enter.
  const [draft, setDraft] = useState(String(value))
  useEffect(() => { setDraft(String(value)) }, [value])

  const commit = () => {
    const n = parseFloat(draft)
    if (isNaN(n)) { setDraft(String(value)); return }
    const clamped = clamp(n, min, max)
    setDraft(String(clamped))
    if (clamped !== value) onChange(clamped)
  }

  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <label className="text-sm text-slate-300">{label}</label>
        <div className="flex items-baseline gap-1">
          <input
            type="number"
            min={min} max={max} step={step}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
            className="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-0.5 text-right
                       font-mono text-sm text-cyan-300 font-semibold
                       focus:outline-none focus:border-cyan-500"
          />
          <span className="font-mono text-sm text-cyan-300 font-semibold">{unit}</span>
        </div>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-cyan-500"
        style={{
          background: `linear-gradient(to right, #06b6d4 0%, #06b6d4 ${pct}%, #1e293b ${pct}%, #1e293b 100%)`,
        }}
      />
      <div className="flex justify-between text-xs text-slate-600 mt-1">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  )
}

export function SliderGroup({ sliders, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-x-8 gap-y-5">
      {WHATIF_SPECS.map(({ key, label, min, max, step, unit }) => (
        <SliderInput
          key={key}
          label={label}
          value={sliders[key]}
          min={min} max={max} step={step} unit={unit}
          onChange={(val) => onChange(key, val)}
        />
      ))}
    </div>
  )
}

export default SliderInput
