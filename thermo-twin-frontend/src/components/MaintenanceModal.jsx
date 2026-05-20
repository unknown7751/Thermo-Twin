import { useState } from 'react'

const REASONS = [
  { value: 'compressor_replacement',  label: 'Compressor replacement' },
  { value: 'coil_cleaning',           label: 'Coil cleaning' },
  { value: 'refrigerant_recharge',    label: 'Refrigerant recharge' },
  { value: 'fan_motor_replacement',   label: 'Fan motor replacement' },
  { value: 'maintenance_completed',   label: 'General maintenance' },
]

export default function MaintenanceModal({ open, onClose, onConfirm }) {
  const [reason, setReason] = useState(REASONS[0].value)
  const [notes,  setNotes]  = useState('')
  const [busy,   setBusy]   = useState(false)
  const [err,    setErr]    = useState(null)

  if (!open) return null

  const submit = async () => {
    setBusy(true); setErr(null)
    try {
      const text = notes.trim() ? `${reason}: ${notes.trim()}` : reason
      await onConfirm(text)
      onClose()
    } catch (e) {
      setErr(e?.response?.data?.error ?? e?.message ?? 'failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-900 border border-slate-700 rounded-lg max-w-md w-full p-5 space-y-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Commissioning Reset</h3>
          <p className="text-xs text-slate-500 mt-1">
            Clears the parameter-estimator buffer and re-anchors the drift baseline.
            Use after a physical service action so the twin doesn't keep old data
            mixed with the new operating state.
          </p>
        </div>

        <div>
          <label className="text-xs text-slate-400 block mb-1">Reason</label>
          <select
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm text-slate-200"
          >
            {REASONS.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs text-slate-400 block mb-1">Notes (optional)</label>
          <input
            type="text"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="e.g. replaced YQ-32 compressor, SN 80F..."
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm text-slate-200
                       placeholder-slate-600"
          />
        </div>

        {err && (
          <p className="text-xs text-red-400 bg-red-950/30 border border-red-900 rounded px-2 py-1.5">
            {err}
          </p>
        )}

        <div className="flex gap-2 justify-end pt-1">
          <button
            onClick={onClose}
            disabled={busy}
            className="px-3 py-1.5 rounded border border-slate-700 hover:border-slate-500
                       text-slate-400 hover:text-slate-200 text-sm"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={busy}
            className="px-4 py-1.5 rounded border border-amber-700 bg-amber-950/40 hover:bg-amber-900/50
                       text-amber-300 text-sm font-medium disabled:opacity-50"
          >
            {busy ? 'Resetting…' : 'Confirm Reset'}
          </button>
        </div>
      </div>
    </div>
  )
}
