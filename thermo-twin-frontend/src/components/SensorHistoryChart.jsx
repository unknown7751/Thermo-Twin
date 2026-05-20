import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

/**
 * Mini sensor-history chart for the component inspect panel.
 *
 * `history` = [{ timestamp, value, sensor_type }] from
 * GET /twin/component-history/<component>. `timestamp` is simulation-seconds
 * (numeric), so we use a value x-axis — not a time axis — which matches how
 * the rest of the dashboard renders simulation time.
 *
 * Design note: the chart div is only mounted when history has data (conditional
 * render).  The data effect includes `hasData` in its deps so it re-fires on
 * the exact render that first mounts the div, at which point elRef.current is
 * guaranteed to be assigned (React commits refs before running effects).
 */
export default function SensorHistoryChart({ history, color = '#22d3ee' }) {
  const elRef    = useRef(null)
  const chartRef = useRef(null)

  const hasData = history && history.length > 0

  // Init + update effect.
  // Fires when history changes AND when hasData flips from false → true
  // (which is the render where the chart div first enters the DOM).
  useEffect(() => {
    const el = elRef.current
    if (!el || !hasData) return

    // Create chart instance if needed (first render with data, or after HMR)
    if (!chartRef.current || chartRef.current.isDisposed?.()) {
      chartRef.current = echarts.init(el, null, { renderer: 'canvas' })
    }
    const chart = chartRef.current

    const data = history.map((h) => [h.timestamp, h.value])

    chart.setOption({
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 44, right: 10, top: 10, bottom: 22 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#0f172a', borderColor: '#334155',
        textStyle: { color: '#e2e8f0', fontSize: 11 },
        valueFormatter: (v) => (typeof v === 'number' ? v.toFixed(2) : v),
      },
      xAxis: {
        type: 'value', scale: true,
        axisLabel: { color: '#64748b', fontSize: 9, formatter: (v) => `${v.toFixed(0)}s` },
        axisLine: { lineStyle: { color: '#1e293b' } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value', scale: true,
        axisLabel: { color: '#64748b', fontSize: 9, formatter: (v) => v.toFixed(0) },
        axisLine: { lineStyle: { color: '#1e293b' } },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed', opacity: 0.3 } },
      },
      series: [{
        type: 'line', data, smooth: true, showSymbol: false,
        lineStyle: { color, width: 1.5 },
        areaStyle: { color, opacity: 0.08 },
      }],
    }, { notMerge: true })

    // Force resize after layout settles — guarantees the canvas matches
    // the container even on the very first render.
    requestAnimationFrame(() => {
      if (chartRef.current && !chartRef.current.isDisposed?.()) {
        chartRef.current.resize()
      }
    })
  }, [history, color, hasData])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (chartRef.current) { chartRef.current.dispose(); chartRef.current = null }
    }
  }, [])

  if (!hasData) {
    return (
      <div style={{ height: 150 }} className="flex items-center justify-center text-xs text-slate-600">
        no history yet
      </div>
    )
  }
  return <div ref={elRef} style={{ height: 150, width: '100%' }} />
}
