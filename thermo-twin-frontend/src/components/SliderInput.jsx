export const WHATIF_SPECS = [
  { key: 'compressor_speed_pct', label: 'Compressor Speed',    min: 20,  max: 100, step: 1,   unit: '%'  },
  { key: 'ambient_temp_c',       label: 'Ambient Temperature', min: 15,  max: 50,  step: 1,   unit: '°C' },
  { key: 'load_demand_pct',      label: 'Load Demand',         min: 10,  max: 100, step: 1,   unit: '%'  },
  { key: 'duration_hours',       label: 'Simulation Duration', min: 0.5, max: 24,  step: 0.5, unit: 'h'  },
]

function SliderInput({ label, value, min, max, step, unit, onChange }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <label className="text-sm text-slate-300">{label}</label>
        <span className="font-mono text-sm text-cyan-300 font-semibold">
          {value}{unit}
        </span>
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
