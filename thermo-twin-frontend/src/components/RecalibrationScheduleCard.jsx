import { useState } from 'react'

function fmtDate(iso) {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

export default function RecalibrationScheduleCard({ status, onTrigger }) {
  const [busy, setBusy] = useState(false)
  const [err,  setErr]  = useState(null)

  const trigger = async () => {
    setBusy(true); setErr(null)
    try { await onTrigger('manual_request') }
    catch (e) { setErr(e?.response?.data?.error ?? e?.message ?? 'failed') }
    finally   { setBusy(false) }
  }

  if (!status) {
    return <p className="text-xs text-slate-600 text-center py-6">Loading schedule…</p>
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="bg-slate-800/50 border border-slate-700 rounded p-2.5">
          <p className="text-xs text-slate-500">Last recalibration</p>
          <p className="font-mono text-slate-200 text-sm mt-0.5">{fmtDate(status.last_recalibration)}</p>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded p-2.5">
          <p className="text-xs text-slate-500">Next scheduled</p>
          <p className="font-mono text-slate-200 text-sm mt-0.5">{fmtDate(status.next_recalibration)}</p>
          <p className="text-xs text-slate-600 mt-0.5">
            {status.days_until_next != null && (
              <>in <span className="text-cyan-300 font-mono">{status.days_until_next}d</span></>
            )}
            {' · interval '}
            <span className="text-slate-400 font-mono">{status.recalibration_interval_days}d</span>
          </p>
        </div>
      </div>

      <button
        onClick={trigger}
        disabled={busy}
        className="w-full px-4 py-2 rounded border border-cyan-700 bg-cyan-950/40 hover:bg-cyan-900/50
                   text-cyan-300 text-sm font-medium transition-all disabled:opacity-50"
      >
        {busy ? 'Recalibrating…' : '⚡ Recalibrate now'}
      </button>
      {err && (
        <p className="text-xs text-red-400 bg-red-950/30 border border-red-900 rounded px-2 py-1.5">
          {err}
        </p>
      )}
      <p className="text-xs text-slate-600">
        Re-fits k_disc / k_fan / k_temp_a / k_temp_b from the buffered normal-operation
        samples. Applies the new coefficients to the physics model in place.
      </p>
    </div>
  )
}
