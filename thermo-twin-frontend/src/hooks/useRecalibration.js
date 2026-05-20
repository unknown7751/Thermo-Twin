import { useEffect, useRef, useCallback } from 'react'
import api from '../utils/api.js'
import useAppStore from '../store/appStore.js'

const POLL_MS = 1500   // ~0.67 Hz — drift/recalibration metrics move slowly

/**
 * Polls /twin/drift, /twin/parameters, /twin/recalibration/status in parallel
 * and pushes results into the Zustand `recal` slice. Returns imperative
 * helpers for the manual-action buttons:
 *   triggerRecalibration(reason?)  — POST /twin/recalibrate
 *   commissioningReset(reasonText) — POST /twin/commissioning-reset
 *
 * Call from RecalibrationPanel (single mount on the dashboard).
 */
export function useRecalibration() {
  const setRecal       = useAppStore((s) => s.setRecal)
  const setRecalOnline = useAppStore((s) => s.setRecalOnline)
  const pendingRef     = useRef(false)
  const refreshOnceRef = useRef(null)

  useEffect(() => {
    let active = true
    async function poll() {
      if (pendingRef.current || !active) return
      pendingRef.current = true
      try {
        const [drift, params, status] = await Promise.all([
          api.get('/twin/drift'),
          api.get('/twin/parameters'),
          api.get('/twin/recalibration/status'),
        ])
        if (!active) return
        setRecal({
          drift:      drift.data,
          parameters: params.data,
          status:     status.data,
          online:     true,
        })
      } catch {
        if (active) setRecalOnline(false)
      } finally {
        pendingRef.current = false
      }
    }
    refreshOnceRef.current = poll
    poll()
    const id = setInterval(poll, POLL_MS)
    return () => { active = false; clearInterval(id) }
  }, [setRecal, setRecalOnline])

  const triggerRecalibration = useCallback(async (reason = 'manual_request') => {
    const res = await api.post('/twin/recalibrate', { reason })
    if (refreshOnceRef.current) refreshOnceRef.current()
    return res.data
  }, [])

  const commissioningReset = useCallback(async (reasonText = 'maintenance_completed') => {
    const res = await api.post('/twin/commissioning-reset', { reason: reasonText })
    if (refreshOnceRef.current) refreshOnceRef.current()
    return res.data
  }, [])

  return { triggerRecalibration, commissioningReset }
}
