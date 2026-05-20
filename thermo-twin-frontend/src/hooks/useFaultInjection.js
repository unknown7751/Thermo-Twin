import { useEffect, useRef } from 'react'
import api from '../utils/api.js'
import useAppStore from '../store/appStore.js'
import { emitReset } from '../utils/sampleBus.js'

export function useFaultInjection() {
  const setFaultActive    = useAppStore((s) => s.setFaultActive)
  const clearFault        = useAppStore((s) => s.clearFault)
  const tickFaultCountdown = useAppStore((s) => s.tickFaultCountdown)
  const fault             = useAppStore((s) => s.fault)

  // Countdown ticker
  const tickRef = useRef(null)

  useEffect(() => {
    if (fault.active) {
      tickRef.current = setInterval(tickFaultCountdown, 1000)
    } else {
      clearInterval(tickRef.current)
    }
    return () => clearInterval(tickRef.current)
  }, [fault.active, tickFaultCountdown])

  async function injectFault(faultType, delayS = 5, durationS = 5) {
    try {
      const res  = await api.post(`/stream/inject-fault/${faultType}`, {
        delay:    delayS,
        duration: durationS,
      })
      const fi = res.data.fault_info
      // Countdown = delay until fault starts + how long it lasts
      setFaultActive(faultType, fi?.start_time ?? 0, fi?.end_time ?? 0, delayS + durationS)
    } catch (err) {
      console.error('Fault injection failed:', err)
    }
  }

  async function resetStream() {
    try {
      await api.post('/stream/reset')
      emitReset()   // wipe chart rolling buffers so old lines don't persist
      clearFault()
    } catch (err) {
      console.error('Stream reset failed:', err)
    }
  }

  return { injectFault, resetStream, fault }
}
