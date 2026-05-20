import { useEffect, useRef, useCallback } from 'react'
import * as echarts from 'echarts'
import { subscribe, subscribeReset } from '../utils/sampleBus.js'
import useAppStore from '../store/appStore.js'

// ── Constants ──────────────────────────────────────────────────────────────────
const WINDOW_SECS = 60    // sim-seconds shown: 40s ÷ 0.5s/sample = 80 pts; at 50 Hz fills in ~1.6 real sec
const MAX_PTS     = 120   // cap buffer at ~120 points per series

const SENSORS = [
  { key: 'compressor_power_kw',    label: 'Compressor Power', unit: 'kW',  color: '#22d3ee', gridIdx: 0 },
  { key: 'discharge_pressure_psi', label: 'Discharge Pressure', unit: 'psi', color: '#a78bfa', gridIdx: 1 },
  { key: 'fan_rpm',                label: 'Fan Speed', unit: 'RPM', color: '#34d399', gridIdx: 2 },
  { key: 'supply_air_temp_c',      label: 'Supply Air Temp', unit: '°C',  color: '#fb923c', gridIdx: 3 },
]

function makeOption() {
  const grids = SENSORS.map((_, i) => ({
    left: '70px', right: '24px',
    top:    `${6 + i * 24.5}%`,
    height: '16%',
  }))

  const xAxes = SENSORS.map((_, i) => ({
    type: 'value', gridIndex: i,
    show: i === SENSORS.length - 1,
    axisLabel: { color: '#64748b', fontSize: 10, formatter: (v) => `${v.toFixed(0)}s` },
    axisLine:  { lineStyle: { color: '#1e293b' } },
    splitLine: { lineStyle: { color: '#1e293b' } },
  }))

  const yAxes = SENSORS.map((s, i) => ({
    type: 'value', gridIndex: i, scale: true,
    axisLabel: { color: '#94a3b8', fontSize: 10, formatter: (v) => v.toFixed(0) },
    axisLine:  { lineStyle: { color: '#1e293b' } },
    splitLine: { lineStyle: { color: '#1e293b', type: 'dashed', opacity: 0.3 } },
    name: `${s.label} (${s.unit})`,
    nameLocation: 'end',
    nameGap: 8,
    nameTextStyle: {
      color: s.color, fontSize: 11, fontWeight: 600,
      align: 'left', verticalAlign: 'bottom',
      padding: [0, 0, 2, 0],
    },
  }))

  const series = SENSORS.map((s, i) => ({
    type: 'line', xAxisIndex: i, yAxisIndex: i,
    data: [],
    name: s.label,
    lineStyle:  { color: s.color, width: 1.5 },
    itemStyle:  { color: s.color },
    showSymbol: false,
    smooth: false,
    z: 2,
  }))

  return {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'cross', link: { xAxisIndex: 'all' } },
      backgroundColor: '#0f172a', borderColor: '#334155',
      textStyle: { color: '#e2e8f0', fontSize: 11 },
    },
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    grid:    grids,
    xAxis:   xAxes,
    yAxis:   yAxes,
    series,
  }
}

// ── Component ──────────────────────────────────────────────────────────────────
export default function ChartPanel() {
  const containerRef = useRef(null)
  const chartRef     = useRef(null)

  // Per-sensor rolling data buffers: [{x: timestamp, y: value}, ...]
  const buffers = useRef(SENSORS.map(() => []))

  // Fault marker refs (for markArea / markLine)
  const faultRef     = useRef({ active: null, startTime: null, endTime: null })
  const anomaliesRef = useRef([])   // detected anomaly windows (persist after injection ends)

  const fault         = useAppStore((s) => s.fault)
  const alertsHistory = useAppStore((s) => s.alerts.history)

  // Sync fault info to ref so RAF can read without triggering re-renders
  useEffect(() => {
    faultRef.current = {
      active:    fault.active,
      startTime: fault.faultStartTime,
      endTime:   fault.faultEndTime,
    }
  }, [fault])

  // Sync detected anomaly windows so the RAF loop can paint persistent red bands
  useEffect(() => {
    anomaliesRef.current = alertsHistory
      .filter((a) => a.anomaly_start_time != null && a.anomaly_end_time != null)
      .map((a) => ({ start: a.anomaly_start_time, end: a.anomaly_end_time }))
  }, [alertsHistory])

  // ── Init ECharts ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return
    const chart = echarts.init(containerRef.current, null, { renderer: 'canvas' })
    chart.setOption(makeOption())
    chartRef.current = chart

    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  // ── sampleBus subscriber: push data into buffers (no React state) ─────────
  const handleSample = useCallback((sample) => {
    const t = sample.timestamp ?? 0
    SENSORS.forEach((s, i) => {
      const val = sample[s.key]
      if (val == null) return
      const buf = buffers.current[i]
      buf.push([t, val])
      if (buf.length > MAX_PTS) buf.shift()
    })
  }, [])

  useEffect(() => subscribe(handleSample), [handleSample])

  // ── Reset: wipe buffers + clear ECharts series ────────────────────────────
  useEffect(() => subscribeReset(() => {
    // Clear rolling data buffers
    buffers.current.forEach((buf) => { buf.length = 0 })
    // Clear persisted anomaly windows
    anomaliesRef.current = []
    // Re-render chart immediately with empty series so old lines vanish
    const chart = chartRef.current
    if (!chart) return
    chart.setOption({
      series: SENSORS.map((s, i) => ({
        name: s.label, type: 'line', xAxisIndex: i, yAxisIndex: i,
        data: [], markArea: { silent: true, data: [] },
      })),
    })
  }), [])

  // ── 60fps RAF loop: update x-range + push series data ────────────────────
  useEffect(() => {
    let rafId
    let lastRender = 0

    function frame(now) {
      rafId = requestAnimationFrame(frame)
      if (now - lastRender < 16) return   // cap at ~60fps
      lastRender = now

      const chart = chartRef.current
      if (!chart) return

      // Determine x window from the newest timestamp across all buffers
      let xMax = -Infinity
      buffers.current.forEach((buf) => {
        if (buf.length) xMax = Math.max(xMax, buf[buf.length - 1][0])
      })
      if (!isFinite(xMax)) return

      const xMin = xMax - WINDOW_SECS

      // Build markArea regions:
      //  • detected anomaly windows → solid red, PERSIST after injection ends
      //  • active injection (not yet detected) → faint amber pre-warning
      const regions = []

      for (const a of anomaliesRef.current) {
        if (a.end < xMin || a.start > xMax) continue   // scrolled off-screen
        regions.push([
          { xAxis: Math.max(a.start, xMin), itemStyle: { color: 'rgba(239,68,68,0.16)' } },
          { xAxis: Math.min(a.end, xMax) },
        ])
      }

      const { startTime, endTime, active } = faultRef.current
      if (active && startTime != null && (endTime ?? xMax) >= xMin) {
        regions.push([
          { xAxis: Math.max(startTime, xMin), itemStyle: { color: 'rgba(245,158,11,0.07)' } },
          { xAxis: endTime ?? xMax },
        ])
      }

      const faultMark = { silent: true, data: regions }

      // Must include `name` so ECharts matches against the initial series by name, not by
      // appending new ones. Without name, setOption creates 4 extra series = 2 lines per grid.
      const seriesUpdates = SENSORS.map((s, i) => ({
        name:       s.label,
        type:       'line',
        xAxisIndex: i,
        yAxisIndex: i,
        data:       buffers.current[i],
        markArea:   faultMark,
      }))

      chart.setOption({
        xAxis:  SENSORS.map(() => ({ min: xMin, max: xMax })),
        series: seriesUpdates,
      })
    }

    rafId = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(rafId)
  }, [])

  return (
    <div className="card flex flex-col" style={{ minHeight: 480 }}>
      <p className="section-title mb-2">Live Sensor Streams</p>
      <div
        ref={containerRef}
        className="flex-1"
        style={{ minHeight: 440 }}
      />
    </div>
  )
}
