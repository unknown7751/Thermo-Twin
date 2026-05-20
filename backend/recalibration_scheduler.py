"""
Phase 7 — Recalibration Scheduler.

Orchestrates the BayesianParameterEstimator and DriftDetector:

  • holds a monthly schedule (configurable interval)
  • exposes manual `trigger_recalibration()`
  • exposes `trigger_commissioning_reset()` for post-maintenance reset
  • `check_and_trigger()` is the cheap poll the stream-loop calls each tick;
    fires automatically when the schedule is due
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, List, Optional

from drift_detector import DriftDetector
from parameter_estimator import BayesianParameterEstimator, ParameterUpdate

log = logging.getLogger("thermo-twin.recal")


class RecalibrationReason(str, Enum):
    SCHEDULED_MONTHLY      = "scheduled_monthly"
    DRIFT_DETECTED         = "drift_detected"
    MAINTENANCE_COMPLETED  = "maintenance_completed"
    SEASONAL_CHANGE        = "seasonal_change"
    MANUAL_REQUEST         = "manual_request"


@dataclass
class RecalibrationEvent:
    timestamp:               float
    reason:                  str
    machine_id:              str = "LIVE-DEMO-UNIT"
    success:                 bool = True
    notes:                   str = ""
    parameter_updates:       List[ParameterUpdate] = field(default_factory=list)
    accuracy_before_pct:     Optional[float] = None
    accuracy_after_pct:      Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parameter_updates"] = [u.to_dict() if hasattr(u, "to_dict") else u for u in self.parameter_updates]
        return d


class RecalibrationScheduler:
    def __init__(
        self,
        estimator: BayesianParameterEstimator,
        drift_detector: DriftDetector,
        interval_days: int = 30,
        machine_id: str = "LIVE-DEMO-UNIT",
    ):
        self._estimator    = estimator
        self._drift        = drift_detector
        self._interval     = timedelta(days=interval_days)
        self._machine_id   = machine_id
        self._last:        Optional[datetime] = None
        self._next:        datetime = datetime.now() + self._interval
        self._events:      List[RecalibrationEvent] = []
        self._callback:    Optional[Callable[[RecalibrationEvent], None]] = None
        # Drift-trigger debounce so a single noisy update doesn't immediately
        # fire a (possibly expensive) recalibration
        self._drift_streak       = 0
        self._drift_streak_thr   = 30   # consecutive ticks below threshold

    # ── Public API ─────────────────────────────────────────────────────────
    def set_callback(self, cb: Callable[[RecalibrationEvent], None]) -> None:
        self._callback = cb

    def should_recalibrate(self) -> bool:
        return datetime.now() >= self._next

    def check_and_trigger(self) -> Optional[RecalibrationEvent]:
        """Lightweight poll called from the stream loop. Returns event if one fired."""
        m = self._drift.get_current_metrics()
        # Track drift-streak so we only auto-recal on sustained drift, not noise
        if m.is_drifting:
            self._drift_streak += 1
        else:
            self._drift_streak = 0

        if self.should_recalibrate():
            return self.trigger_recalibration(RecalibrationReason.SCHEDULED_MONTHLY)
        if self._drift_streak >= self._drift_streak_thr:
            self._drift_streak = 0
            return self.trigger_recalibration(RecalibrationReason.DRIFT_DETECTED)
        return None

    def trigger_recalibration(
        self,
        reason: RecalibrationReason = RecalibrationReason.MANUAL_REQUEST,
        lookback_hours: float = 720.0,
        confidence_threshold: float = 0.05,
    ) -> RecalibrationEvent:
        acc_before = self._drift.get_current_metrics().accuracy_pct
        try:
            updates = self._estimator.estimate_parameters(
                lookback_hours=lookback_hours,
                confidence_threshold=confidence_threshold,
                apply_to_physics=True,
                reason=reason.value,
            )
            self._drift.reset_baseline()
            acc_after = self._drift.get_current_metrics().accuracy_pct
            event = RecalibrationEvent(
                timestamp=datetime.now().timestamp(),
                reason=reason.value,
                machine_id=self._machine_id,
                success=True,
                parameter_updates=updates,
                accuracy_before_pct=round(acc_before, 1),
                accuracy_after_pct=round(acc_after, 1),
                notes=f"{len([u for u in updates if not u.rejected])} updates applied "
                      f"({len([u for u in updates if u.rejected])} rejected)",
            )
            self._last = datetime.now()
            self._next = self._last + self._interval
        except Exception as exc:
            log.error("Recalibration failed: %s", exc)
            event = RecalibrationEvent(
                timestamp=datetime.now().timestamp(),
                reason=reason.value,
                machine_id=self._machine_id,
                success=False,
                notes=f"error: {exc}",
            )

        self._events.append(event)
        if self._callback:
            try: self._callback(event)
            except Exception as exc: log.warning("Recal callback raised: %s", exc)
        return event

    def trigger_commissioning_reset(self, reason_text: str = "maintenance_completed") -> RecalibrationEvent:
        """Clear estimator buffer + drift baseline. Use after physical service."""
        self._estimator.clear_buffer()
        self._drift.reset_baseline()
        event = RecalibrationEvent(
            timestamp=datetime.now().timestamp(),
            reason=RecalibrationReason.MAINTENANCE_COMPLETED.value,
            machine_id=self._machine_id,
            success=True,
            notes=reason_text,
        )
        self._events.append(event)
        if self._callback:
            try: self._callback(event)
            except Exception as exc: log.warning("Recal callback raised: %s", exc)
        return event

    def get_status(self) -> dict:
        now = datetime.now()
        delta = self._next - now
        return {
            "last_recalibration":          self._last.isoformat() if self._last else None,
            "next_recalibration":          self._next.isoformat(),
            "seconds_until_next":          int(delta.total_seconds()),
            "days_until_next":             round(delta.total_seconds() / 86400.0, 2),
            "recalibration_interval_days": self._interval.days,
            "recent_events": [e.to_dict() for e in self._events[-10:]],
        }
