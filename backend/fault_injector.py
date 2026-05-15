import uuid


class FaultInjector:
    def __init__(self, data_streamer):
        self.streamer = data_streamer
        self.active_faults = []    # scheduled + active + completed faults
        self.fault_history = {}

    # ── Public API ──────────────────────────────────────────────────────────────

    def schedule_future_fault(self, fault_type, delay_seconds=10, duration_seconds=10):
        """
        Schedule a fault to start delay_seconds from now.
        The fault is applied to LIVE new samples as they are generated,
        so detection happens naturally as the model window fills with faulted data.
        """
        start_time = self.streamer.current_time + delay_seconds
        end_time   = start_time + duration_seconds
        fault_id   = str(uuid.uuid4())

        fault_record = {
            "fault_id":   fault_id,
            "fault_type": fault_type,
            "start_time": start_time,
            "end_time":   end_time,
            "status":     "scheduled",   # scheduled → active → completed
        }
        self.active_faults.append(fault_record)
        self.fault_history[fault_id] = fault_record

        return {
            "fault_id":   fault_id,
            "start_time": start_time,
            "end_time":   end_time,
        }

    def apply_to_live_sample(self, sample):
        """
        Called immediately after get_next_sample() returns.
        Modifies the sample dict IN PLACE if its timestamp falls inside a scheduled
        fault window — which also updates the history entry (same dict object).
        """
        ts = sample["timestamp"]
        for fault in self.active_faults:
            if fault["start_time"] <= ts <= fault["end_time"]:
                span     = fault["end_time"] - fault["start_time"]
                progress = (ts - fault["start_time"]) / span if span > 0 else 1.0
                self._apply_fault_to_sample(sample, fault["fault_type"], progress)
                fault["status"] = "active"
            elif ts > fault["end_time"] and fault["status"] == "active":
                fault["status"] = "completed"

    def get_scheduled_faults(self):
        """Return lightweight fault info for frontend marker rendering."""
        return [
            {
                "fault_type": f["fault_type"],
                "start_time": f["start_time"],
                "end_time":   f["end_time"],
                "status":     f["status"],
            }
            for f in self.active_faults
        ]

    def is_sample_in_fault(self, timestamp):
        for fault in self.active_faults:
            if fault["start_time"] <= timestamp <= fault["end_time"]:
                return True, fault["fault_type"], fault["fault_id"]
        return False, None, None

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _apply_fault_to_sample(self, sample, fault_type, progress):
        """
        Fault patterns calibrated to match training data magnitude
        (generate_sensor_data.py) so the model sees familiar fault signatures
        and SHAP attribution matches expected fingerprints.
        """
        p = progress

        if fault_type == "refrigerant_leak":
            # Abrupt pressure drop 38% of baseline + temp rise 7°C (flat, not ramped)
            baseline_psi = 70.0 * sample["compressor_power_kw"]
            sample["discharge_pressure_psi"] -= 0.38 * baseline_psi
            sample["supply_air_temp_c"]      += 7.0

        elif fault_type == "fan_failure":
            # RPM drops 80% of baseline; power/pressure/temp ramp up gradually
            baseline_rpm = 340.0 * sample["compressor_power_kw"]
            sample["fan_rpm"]                -= 0.80 * baseline_rpm
            sample["compressor_power_kw"]    += p * 1.4
            sample["discharge_pressure_psi"] += p * 55.0
            sample["supply_air_temp_c"]      += p * 4.5

        elif fault_type == "compressor_wear":
            # Slow power creep + pressure drop + temp rise (all ramped)
            sample["compressor_power_kw"]    += p * 2.0
            sample["discharge_pressure_psi"] -= p * 45.0
            sample["supply_air_temp_c"]      += p * 3.0
