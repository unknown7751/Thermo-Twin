import { useEffect, useRef } from 'react'
import api from '../utils/api.js'
import { emit } from '../utils/sampleBus.js'
import useAppStore from '../store/appStore.js'

const POLL_MS = 100   // 10 Hz

export function useStreamData() {
  const setStreamMeta    = useAppStore((s) => s.setStreamMeta)
  const setBackendOnline = useAppStore((s) => s.setBackendOnline)
  const setSample        = useAppStore((s) => s.setSample)
  const setTwin          = useAppStore((s) => s.setTwin)
  const setRul           = useAppStore((s) => s.setRul)
  const addAlert         = useAppStore((s) => s.addAlert)

  // Prevent concurrent requests if backend is slow
  const pendingRef = useRef(false)

  useEffect(() => {
    let active = true

    async function poll() {
      if (pendingRef.current || !active) return
      pendingRef.current = true
      try {
        const res  = await api.get('/stream/next-sample')
        const data = res.data
        if (!active) return

        setBackendOnline(true)

        // Stream metadata
        setStreamMeta({
          sampleCount:  data.total_samples ?? data.buffer_size ?? 0,
          currentTime:  data.current_time  ?? 0,
        })

        // Process backlog first (older samples to keep chart continuous)
        if (Array.isArray(data.backlog)) {
          data.backlog.forEach((b) => { if (b.sample) emit(b.sample) })
        }

        // Raw sample → chart bus + Zustand (for Sidebar live values)
        if (data.sample) {
          emit(data.sample)          // chart update (imperative, no re-render)
          setSample(data.sample)     // sidebar values (react state, ok at 10Hz)
        }

        // Twin state (Phases 1+2)
        if (data.twin) {
          setTwin({
            state:          data.twin.state          ?? {},
            prediction:     data.twin.prediction     ?? {},
            divergence:     data.twin.divergence      ?? {},
            uncertainty:    data.twin.uncertainty    ?? {},
            estimator_mode: data.twin.estimator_mode ?? 'ukf',
            model_used:     data.twin.model_used     ?? 'linear',
          })

          // RUL (Phase 3) is nested inside twin response
          if (data.twin.rul) {
            setRul(data.twin.rul)
          }
        }

        // Alert (fires only when a new detection occurs)
        if (data.alert) {
          addAlert(data.alert)
          setStreamMeta({ alertCount: (useAppStore.getState().stream.alertCount ?? 0) + 1 })
        }

      } catch {
        if (active) setBackendOnline(false)
      } finally {
        pendingRef.current = false
      }
    }

    const id = setInterval(poll, POLL_MS)
    poll()   // immediate first call

    return () => {
      active = false
      clearInterval(id)
    }
  }, [setStreamMeta, setBackendOnline, setSample, setTwin, setRul, addAlert])
}
