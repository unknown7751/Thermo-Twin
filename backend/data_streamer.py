import numpy as np
from collections import deque


class SyntheticDataStreamer:
    def __init__(self):
        self.history = deque(maxlen=600)
        self.current_time = 0.0
        self.sample_index = 0
        self._drift_acc = 0.0   # random-walk accumulator (matches training cumsum)

    def get_next_sample(self):
        self.current_time += 0.5
        self.sample_index += 1

        # Exact same demand formula as generate_sensor_data.py
        t = self.current_time
        self._drift_acc += np.random.normal(0, 0.004)
        demand = (
            0.4  * np.sin(2 * np.pi * 0.02  * t)
            + 0.15 * np.sin(2 * np.pi * 0.007 * t)
            + np.random.normal(0, 0.05)
        )
        power = float(np.clip(
            3.5 + demand + np.random.normal(0, 0.08) + 0.02 * self._drift_acc,
            2.0, 6.0,
        ))

        # Exact same thermodynamic correlations + noise stds as training data
        pressure = float(70.0  * power + np.random.normal(0, 4.0))
        rpm      = float(340.0 * power + np.random.normal(0, 30.0))
        temp     = float(18.0  - 2.0 * power + np.random.normal(0, 0.3))

        sample = {
            "timestamp":              self.current_time,
            "sample_index":           self.sample_index,
            "compressor_power_kw":    power,
            "discharge_pressure_psi": pressure,
            "fan_rpm":                rpm,
            "supply_air_temp_c":      temp,
        }

        self.history.append(sample)
        return sample

    def get_window(self, seconds_back=6):
        n = int(seconds_back * 10)
        samples = list(self.history)[-n:]
        if not samples:
            return np.empty((0, 4))
        return np.array([
            [s["compressor_power_kw"], s["discharge_pressure_psi"],
             s["fan_rpm"], s["supply_air_temp_c"]]
            for s in samples
        ])

    def get_history_dict(self):
        return {
            "samples":     list(self.history),
            "current_time": self.current_time,
            "buffer_size": len(self.history),
        }
