import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

const TRACES = [
  { key: 'compressor_power_kw',    label: 'Power',    color: '#22d3ee' },
  { key: 'discharge_pressure_psi', label: 'Pressure', color: '#a78bfa' },
  { key: 'fan_rpm',                label: 'Fan RPM',  color: '#34d399' },
  { key: 'supply_air_temp_c',      label: 'Temp',     color: '#fb923c' },
]

const WARN = 0.15
const CRIT = 0.30

// Each sensor has a wildly different scale (power ~3.5, rpm ~1190). Normalize
// every trace to a % of its initial value so degradation drift is comparable.
function normalize(traj, key) {
  if (!traj.length) return []
  const base = traj[0][key] || 1
  return traj.map((p) => [p.t_hours, (p[key] / base) * 100])
}

// Build colored background bands from the anomaly score over time.
function anomalyBands(traj) {
  const bands = []
  let segStart = null
  let segBand  = null

  const bandOf = (s) => (s >= CRIT ? 'crit' : s >= WARN ? 'warn' : 'safe')
  const colorOf = {
    safe: 'rgba(34,197,94,0.06)',
    warn: 'rgba(245,158,11,0.12)',
    crit: 'rgba(239,68,68,0.16)',
  }

  for (let i = 0; i < traj.length; i++) {
    const b = bandOf(traj[i].anomaly_score)
    if (segBand === null) { segStart = traj[i].t_hours; segBand = b }
    else if (b !== segBand) {
      bands.push([
        { xAxis: segStart, itemStyle: { color: colorOf[segBand] } },
        { xAxis: traj[i].t_hours },
      ])
      segStart = traj[i].t_hours; segBand = b
    }
  }
  if (segBand !== null) {
    const lastX = traj[traj.length - 1].t_hours
    bands.push([
      { xAxis: segStart, itemStyle: { color: colorOf[segBand] } },
      { xAxis: lastX },
    ])
  }
  return bands
}

export default function TrajectoryChart({ result, showBaseline }) {
  const containerRef = useRef(null)
  const chartRef     = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return
    const chart = echarts.init(containerRef.current, null, { renderer: 'canvas' })
    chartRef.current = chart
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(containerRef.current)
    return () => { ro.disconnect(); chart.dispose(); chartRef.current = null }
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !result?.scenario?.trajectory) return

    const traj    = result.scenario.trajectory
    const baseTraj = result.baseline?.trajectory
    const ttc     = result.summary?.time_to_critical_hours

    const series = TRACES.map((t) => ({
      name: t.label,
      type: 'line',
      smooth: true,
      showSymbol: false,
      lineStyle: { color: t.color, width: 2 },
      itemStyle: { color: t.color },
      data: normalize(traj, t.key),
    }))

    // Anomaly background painted via markArea on the first series
    series[0].markArea = { silent: true, data: anomalyBands(traj) }

    // Vertical "time to critical" marker
    if (ttc != null && ttc >= 0) {
      series[0].markLine = {
        symbol: 'none', silent: true,
        lineStyle: { color: '#ef4444', type: 'dashed', width: 1.5 },
        label: { formatter: `Critical @ ${ttc}h`, color: '#fca5a5', fontSize: 10 },
        data: [{ xAxis: ttc }],
      }
    }

    if (showBaseline && baseTraj) {
      TRACES.forEach((t) => {
        series.push({
          name: `${t.label} (baseline)`,
          type: 'line', smooth: true, showSymbol: false,
          lineStyle: { color: t.color, width: 1, type: 'dashed', opacity: 0.5 },
          itemStyle: { color: t.color },
          data: normalize(baseTraj, t.key),
        })
      })
    }

    chart.setOption({
      backgroundColor: 'transparent',
      animation: false,
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a', borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 11 },
        valueFormatter: (v) => `${v.toFixed(1)}%`,
      },
      legend: {
        top: 0, textStyle: { color: '#94a3b8', fontSize: 11 },
        data: series.map((s) => s.name),
      },
      grid: { left: 52, right: 20, top: 36, bottom: 36 },
      xAxis: {
        type: 'value', name: 'hours', nameLocation: 'middle', nameGap: 24,
        nameTextStyle: { color: '#64748b', fontSize: 10 },
        axisLabel: { color: '#64748b', fontSize: 10, formatter: (v) => `${v}h` },
        axisLine: { lineStyle: { color: '#1e293b' } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value', scale: true,
        name: '% of initial', nameTextStyle: { color: '#64748b', fontSize: 10 },
        axisLabel: { color: '#64748b', fontSize: 10, formatter: (v) => `${v.toFixed(0)}%` },
        axisLine: { lineStyle: { color: '#1e293b' } },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed', opacity: 0.3 } },
      },
      series,
    }, { notMerge: true })
  }, [result, showBaseline])

  return (
    <div>
      <div ref={containerRef} style={{ height: 360, minHeight: 360 }} />
      <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgba(34,197,94,0.3)' }} /> Safe
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgba(245,158,11,0.4)' }} /> Warning
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 rounded-sm" style={{ background: 'rgba(239,68,68,0.5)' }} /> Critical
        </span>
        {showBaseline && (
          <span className="ml-auto text-slate-600">Dashed = nominal baseline overlay</span>
        )}
      </div>
    </div>
  )
}
