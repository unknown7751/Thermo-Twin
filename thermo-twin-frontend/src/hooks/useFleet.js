import { useEffect, useRef } from 'react'
import api from '../utils/api.js'
import useAppStore from '../store/appStore.js'

const POLL_MS = 1000   // 1 Hz — fleet backend ticks at 1 Hz too

/**
 * Polls /fleet/health, /fleet/dispatch-queue, /fleet/anomalies, /fleet/units in
 * parallel and pushes results into the Zustand `fleet` slice. Sets `online`
 * false on failure so the UI can render an offline state instead of stale data.
 *
 * Call from the FleetDashboard page only — there's no point polling fleet
 * endpoints when the user is on the single-unit dashboard.
 */
export function useFleet() {
  const setFleet       = useAppStore((s) => s.setFleet)
  const setFleetOnline = useAppStore((s) => s.setFleetOnline)
  const pendingRef     = useRef(false)

  useEffect(() => {
    let active = true

    async function poll() {
      if (pendingRef.current || !active) return
      pendingRef.current = true
      try {
        const [units, health, queue, anomalies] = await Promise.all([
          api.get('/fleet/units'),
          api.get('/fleet/health'),
          api.get('/fleet/dispatch-queue'),
          api.get('/fleet/anomalies'),
        ])
        if (!active) return
        setFleet({
          units:     units.data?.units     ?? [],
          metadata:  units.data?.metadata  ?? {},
          health:    health.data           ?? null,
          queue:     queue.data            ?? null,
          anomalies: anomalies.data?.anomalies ?? [],
          online:    true,
        })
      } catch {
        if (active) setFleetOnline(false)
      } finally {
        pendingRef.current = false
      }
    }

    poll()
    const id = setInterval(poll, POLL_MS)
    return () => { active = false; clearInterval(id) }
  }, [setFleet, setFleetOnline])
}
