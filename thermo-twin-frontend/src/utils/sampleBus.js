// Lightweight pub/sub for chart sample data.
// Chart receives samples directly without going through React state,
// keeping the 60fps rendering path free of re-render overhead.

const listeners = new Set()

export function subscribe(fn) {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function emit(sample) {
  listeners.forEach((fn) => fn(sample))
}
