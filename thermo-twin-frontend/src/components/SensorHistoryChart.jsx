import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

/**
 * Mini sensor-history chart for the component inspect panel.
 *
 * `history` = [{ timestamp, value, sensor_type }] from
 * GET /twin/component-history/<component>. `timestamp` is simulation-seconds
 * (numeric), so we use a value x-axis — not a time axis — which matches how
 * the rest of the dashboard renders simulation time.
 */
export default function SensorHistoryChart({ history, color = '#22d3ee' }) {
  const elRef    = useRef(null)
  const chartRef = useRef(null)

  useEffect(() => {
    if (!elRef.current) return
    const chart = echarts.init(elRef.current, null, { renderer: 'canvas' })
    chartRef.current = chart
    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(elRef.current)
    return () => { ro.disconnect(); chart.dispose(); chartRef.current = null }
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    const data = (history || []).map((h) => [h.timestamp, h.value])

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
  }, [history, color])

  if (!history || history.length === 0) {
    return (
      <div className="h-[150px] flex items-center justify-center text-xs text-slate-600">
        no history yet
      </div>
    )
  }
  return <div ref={elRef} style={{ height: 150, width: '100%' }} />
}
