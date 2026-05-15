import sys
import pickle
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.autoencoder import load_autoencoder
from model.threshold import ThresholdManager, severity_score as compute_severity, load_threshold_config
from explainability.shap_explainer import SHAPExplainer
from explainability.alert_payload import build_alert_payload

# Minimum seconds between alerts (prevents flooding when a fault stays in window)
ALERT_COOLDOWN_SECS = 8.0


class LiveDetector:
    def __init__(self, model_path, threshold_config_path, scaler_path):
        config = load_threshold_config(threshold_config_path)
        self._p99_error = config["p99_anomaly"]

        self._model = load_autoencoder(model_path)
        self._model.eval()

        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        train_path = ROOT / "data" / "processed" / "train_windows.npz"
        X_train = np.load(train_path)["X"]
        self._shap = SHAPExplainer(model_path, X_train)

        self._detection_history = []
        self._threshold_mgr = ThresholdManager()
        self._last_alert_stream_time = None   # streamer timestamp of last alert

    def process_from_history(self, history):
        """
        Run detection on the last 50 samples from the streamer's history buffer.
        Using the history (rather than a separate internal buffer) means injected
        faults are immediately visible to the detector on the very next call.

        Args:
            history: list of sample dicts from SyntheticDataStreamer.history

        Returns:
            alert dict if anomaly detected, else None
        """
        if len(history) < 50:
            return None

        buf = history[-50:]

        # Enforce cooldown: skip if last alert was less than ALERT_COOLDOWN_SECS ago
        current_ts = buf[-1]["timestamp"]
        if self._last_alert_stream_time is not None:
            if current_ts - self._last_alert_stream_time < ALERT_COOLDOWN_SECS:
                return None

        vector = np.concatenate([
            [s["compressor_power_kw"]    for s in buf],
            [s["discharge_pressure_psi"] for s in buf],
            [s["fan_rpm"]                for s in buf],
            [s["supply_air_temp_c"]      for s in buf],
        ], dtype=np.float64)

        vector_scaled = self._scaler.transform(vector.reshape(1, -1))[0]

        tensor = torch.tensor(vector_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            recon = self._model(tensor)
            mse = float(torch.mean((tensor - recon) ** 2).item())

        threshold = self._threshold_mgr.get_threshold()
        sev = int(compute_severity(np.array([mse]), threshold, self._p99_error)[0])

        # Require severity >= 60 to fire an alert.
        # False positives on normal data land at severity 41-50 (MSE barely above
        # threshold). Real injected faults land at 90-100. The gap is large enough
        # that 60 is a safe minimum without missing any genuine fault.
        if mse > threshold and sev >= 60:
            explanation = self._shap.explain(vector_scaled)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            alert = build_alert_payload(
                machine_id="LIVE-DEMO-UNIT",
                severity_score=sev,
                explanation=explanation,
                timestamp=ts,
            )
            alert["anomaly_start_time"]   = buf[0]["timestamp"]
            alert["anomaly_end_time"]     = buf[-1]["timestamp"]
            alert["reconstruction_error"] = mse

            self._last_alert_stream_time = current_ts
            self._detection_history.append(alert)
            return alert
        else:
            self._threshold_mgr.update(mse, sev)
            return None

    def get_detection_history(self):
        return self._detection_history

    def reset(self):
        self._detection_history.clear()
        self._last_alert_stream_time = None
