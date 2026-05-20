// Lightweight pub/sub for chart sample data.
// Chart receives samples directly without going through React state,
// keeping the 60fps rendering path free of re-render overhead.

const listeners = new Set()
const resetListeners = new Set()

export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function subscribeReset(fn) {
  resetListeners.add(fn)
  return () => resetListeners.delete(fn)
}

export function emit(sample) {
  listeners.forEach((fn) => fn(sample))
}

export function emitReset() {
  resetListeners.forEach((fn) => fn())
}
