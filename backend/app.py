"""
Phase 5 - Thermo-Twin Alert Backend
Flask API that receives anomaly alerts, stores them in memory,
and serves them to the dashboard.

Run:
    python backend/app.py

Endpoints:
    GET  /health                    - healthcheck (includes dynamic threshold status)
    POST /alert                     - receive an alert from inference layer
    GET  /alerts                    - return last 50 alerts, newest first
    POST /demo/<scenario>           - trigger a pre-loaded demo scenario
"""

import sys
import json
import logging
import threading
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from explainability.alert_payload import build_alert_payload
from model.threshold import ThresholdManager
from data_streamer import SyntheticDataStreamer
from fault_injector import FaultInjector
from live_detector import LiveDetector

# --- Config ---

DEMO_JSON       = ROOT / "explainability" / "demo_explanations.json"
THRESHOLD_STATE = ROOT / "model" / "checkpoints" / "threshold_state.json"
MAX_HISTORY     = 50
PORT            = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("thermo-twin")

# --- App setup ---

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# --- WebSocket client registry ---
_ws_clients: list = []
_ws_lock = threading.Lock()

# --- Infrastructure (InfluxDB + MQTT) — graceful if unavailable ---
try:
    from influx_writer import InfluxWriter
    _influx = InfluxWriter()
except Exception:
    _influx = None

try:
    from mqtt_publisher import MQTTPublisher
    _mqtt = MQTTPublisher()
except Exception:
    _mqtt = None

from twin_schema import TwinSample, TwinAlert

# --- Physics twin engine ---
try:
    from twin_engine import TwinEngine
    _twin_engine = TwinEngine(use_coolprop=False)
except Exception as _twin_exc:
    log.warning("TwinEngine failed to initialize (%s) — /twin/state will return 503", _twin_exc)
    _twin_engine = None

_last_twin_state: dict = {}

# --- In-memory state ---

alert_history = []

# Signal state (for the HTML dashboard chart)
def _make_normal_signal(n=200, seed=7):
    rng    = np.random.default_rng(seed)
    t      = np.arange(n)
    demand = 0.4 * np.sin(2 * np.pi * 0.02 * t) + 0.15 * np.sin(2 * np.pi * 0.007 * t)
    comp   = np.clip(3.5 + demand + rng.normal(0, 0.08, n), 2.0, 6.0)
    return {
        "compressor_power_kw":    comp.tolist(),
        "discharge_pressure_psi": (70.0 * comp + rng.normal(0, 4, n)).tolist(),
        "fan_rpm":                (340.0 * comp + rng.normal(0, 30, n)).tolist(),
        "supply_air_temp_c":      (18.0 - 2.0 * comp + rng.normal(0, 0.3, n)).tolist(),
    }

def _make_fault_signal(scenario, n=70):
    rng  = np.random.default_rng(99)
    comp = np.clip(3.5 + rng.normal(0, 0.08, n), 2.0, 6.0)
    disc = 70.0 * comp + rng.normal(0, 4, n)
    fan  = 340.0 * comp + rng.normal(0, 30, n)
    temp = 18.0 - 2.0 * comp + rng.normal(0, 0.3, n)
    if "refrigerant" in scenario:
        disc -= disc.mean() * 0.40
        temp += 7.0
    elif "fan" in scenario:
        ramp  = np.linspace(0, 1, n)
        fan  -= fan.mean() * 0.80
        comp  = comp + ramp * 1.5
        disc  = disc + ramp * 60.0
        temp  = temp + ramp * 4.0
    elif "compressor" in scenario:
        ramp  = np.linspace(0, 1, n)
        comp  = comp + ramp * 1.8
        disc  = disc - ramp * 50.0
        temp  = temp + ramp * 3.5
    return {
        "compressor_power_kw":    comp.tolist(),
        "discharge_pressure_psi": disc.tolist(),
        "fan_rpm":                fan.tolist(),
        "supply_air_temp_c":      temp.tolist(),
    }

_chart_state = {"signal": _make_normal_signal(), "fault_at": None}

threshold_mgr = ThresholdManager(state_path=THRESHOLD_STATE)

live_streamer = SyntheticDataStreamer()
live_injector = FaultInjector(live_streamer)
live_detector = LiveDetector(
    str(ROOT / "model" / "checkpoints" / "autoencoder.pt"),
    str(ROOT / "model" / "checkpoints" / "threshold_config.json"),
    str(ROOT / "data" / "processed" / "scaler.pkl"),
)
log.info(
    "ThresholdManager ready - threshold=%.4f  buffer=%d/%d  dynamic=%s",
    threshold_mgr.get_threshold(),
    threshold_mgr.buffer_size,
    ThresholdManager.BUFFER_MAX,
    threshold_mgr.is_dynamic,
)

demo_scenarios = {}
if DEMO_JSON.exists():
    with open(DEMO_JSON) as f:
        _raw = json.load(f)
    _machine_map = {
        "scenario_1_refrigerant_leak": "CARRIER-CHILLER-01",
        "scenario_2_fan_failure":      "CARRIER-CHILLER-01",
        "scenario_3_compressor_wear":  "CARRIER-VRF-UNIT-01",
    }
    for key, expl in _raw.items():
        demo_scenarios[key] = build_alert_payload(
            machine_id     = _machine_map.get(key, "CARRIER-CHILLER-01"),
            severity_score = expl.get("severity_score", 85),
            explanation    = expl,
        )
    log.info("Loaded %d demo scenarios from %s", len(demo_scenarios), DEMO_JSON.name)
else:
    log.warning("demo_explanations.json not found - /demo endpoints will return 404")


# --- Helpers ---

def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_alert(payload):
    if len(alert_history) >= MAX_HISTORY:
        alert_history.pop(0)
    alert_history.append(payload)
    recon_err = payload.get("reconstruction_error")
    sev       = payload.get("severity_score", 100)
    if recon_err is not None:
        threshold_mgr.update(recon_err, sev)
        log.info(
            "Threshold updated - current=%.4f  buffer=%d  dynamic=%s",
            threshold_mgr.get_threshold(),
            threshold_mgr.buffer_size,
            threshold_mgr.is_dynamic,
        )


# --- WebSocket helpers ---

def _broadcast_ws(event: str, data: dict) -> None:
    msg = json.dumps({"event": event, "data": data})
    dead = []
    with _ws_lock:
        for ws in _ws_clients:
            try:
                ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# --- Routes ---

@sock.route("/ws")
def ws_stream(ws):
    with _ws_lock:
        _ws_clients.append(ws)
    log.info("WebSocket client connected (total=%d)", len(_ws_clients))
    try:
        while True:
            msg = ws.receive(timeout=30)
            if msg is None:
                break
    except Exception:
        pass
    finally:
        with _ws_lock:
            if ws in _ws_clients:
                _ws_clients.remove(ws)
        log.info("WebSocket client disconnected (total=%d)", len(_ws_clients))


@app.get("/health")
def health():
    return jsonify({
        "status":         "ok",
        "service":        "Thermo-Twin Alert Backend",
        "threshold":      round(threshold_mgr.get_threshold(), 6),
        "threshold_mode": "dynamic" if threshold_mgr.is_dynamic else "static_fallback",
        "buffer_size":    threshold_mgr.buffer_size,
    }), 200


@app.post("/alert")
def receive_alert():
    if not request.is_json:
        log.warning("POST /alert - non-JSON body rejected")
        return jsonify({"error": "Content-Type must be application/json"}), 400
    try:
        payload = request.get_json(force=True)
    except Exception as exc:
        log.error("POST /alert - JSON parse error: %s", exc)
        return jsonify({"error": "invalid JSON"}), 400
    payload["received_at"] = _now_iso()
    _append_alert(payload)
    log.info(
        "Alert received  machine=%s  severity=%s  fault=%s",
        payload.get("machine_id", "?"),
        payload.get("severity_score", "?"),
        payload.get("fault_type", "?"),
    )
    return jsonify({"status": "received"}), 200


@app.get("/alerts")
def get_alerts():
    newest_first = list(reversed(alert_history))
    return jsonify({"alerts": newest_first, "count": len(newest_first)}), 200


@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    dash_dir = str(ROOT / "dashboard")
    return send_from_directory(dash_dir, "index.html")


@app.get("/signal")
def get_signal():
    return jsonify({"signal": _chart_state["signal"], "fault_at": _chart_state["fault_at"]}), 200


@app.get("/baselines")
def get_baselines():
    bl_dir = ROOT / "model" / "checkpoints" / "unit_baselines"
    result = []
    if bl_dir.exists():
        for f in sorted(bl_dir.glob("*.json")):
            with open(f) as fp:
                result.append(json.load(fp))
    return jsonify({"baselines": result}), 200


@app.post("/demo/<scenario>")
def trigger_demo(scenario):
    if scenario not in demo_scenarios:
        log.warning("POST /demo/%s - unknown scenario", scenario)
        return jsonify({"error": "unknown scenario"}), 404
    payload = dict(demo_scenarios[scenario])
    payload["received_at"] = _now_iso()
    _append_alert(payload)

    # Append fault signal to chart state and keep last 350 samples
    fault_sig = _make_fault_signal(scenario)
    fault_len = len(fault_sig["compressor_power_kw"])
    for col in _chart_state["signal"]:
        combined = _chart_state["signal"][col] + fault_sig[col]
        _chart_state["signal"][col] = combined[-350:]
    sig_len = len(_chart_state["signal"]["compressor_power_kw"])
    _chart_state["fault_at"] = sig_len - fault_len

    log.info(
        "Demo triggered  scenario=%s  severity=%s  fault=%s",
        scenario,
        payload.get("severity_score", "?"),
        payload.get("fault_type", "?"),
    )
    return jsonify({"status": "triggered", "scenario": scenario}), 200


# --- Live streaming routes ---

@app.get("/live")
def serve_live_demo():
    dash_dir = str(ROOT / "dashboard")
    return send_from_directory(dash_dir, "live_demo.html")


@app.get("/stream/next-sample")
def stream_next_sample():
    sample = live_streamer.get_next_sample()
    live_injector.apply_to_live_sample(sample)   # modifies sample in-place if in fault window

    twin_sample = TwinSample.from_streamer_dict(sample, machine_id="LIVE-DEMO-UNIT")
    if _influx:
        _influx.write_sample(twin_sample)
    if _mqtt:
        _mqtt.publish_sample(twin_sample)

    history = list(live_streamer.history)
    alert   = live_detector.process_from_history(history)

    if alert:
        twin_alert = TwinAlert.from_alert_dict(alert)
        if _influx:
            _influx.write_alert(twin_alert)
        if _mqtt:
            _mqtt.publish_alert(twin_alert)
        _broadcast_ws("alert", alert)

    _broadcast_ws("sample", sample)

    if _twin_engine:
        twin_result = _twin_engine.process_sample(sample)
        _last_twin_state.update(twin_result)
        _broadcast_ws("twin_state", twin_result)

    return jsonify({
        "sample":           sample,
        "alert":            alert,
        "current_time":     live_streamer.current_time,
        "buffer_size":      len(live_streamer.history),
        "scheduled_faults": live_injector.get_scheduled_faults(),
        "twin":             _last_twin_state if _twin_engine else None,
    }), 200


@app.get("/stream/full-history")
def stream_full_history():
    return jsonify(live_streamer.get_history_dict()), 200


@app.post("/stream/inject-fault/<fault_type>")
def inject_fault(fault_type):
    valid = {"refrigerant_leak", "fan_failure", "compressor_wear"}
    if fault_type not in valid:
        return jsonify({"error": f"Invalid fault type: {fault_type}"}), 400
    body     = request.get_json(silent=True) or {}
    delay    = body.get("delay", 10)       # seconds until fault starts
    duration = body.get("duration", 10)    # seconds the fault lasts
    fault_info = live_injector.schedule_future_fault(fault_type, delay, duration)
    return jsonify({
        "status":     "fault_scheduled",
        "fault_info": fault_info,
        "message":    f"Fault '{fault_type}' will start at t={fault_info['start_time']:.1f}s",
    }), 200


@app.get("/stream/detection-history")
def stream_detection_history():
    return jsonify({"detections": live_detector.get_detection_history()}), 200


@app.post("/stream/reset")
def stream_reset():
    live_detector.reset()
    if _twin_engine:
        _twin_engine.reset()
    return jsonify({"status": "detector_reset"}), 200


# --- Digital twin endpoints ---

@app.get("/twin/state")
def twin_state():
    if _twin_engine is None:
        return jsonify({"error": "TwinEngine not available"}), 503
    return jsonify({"available": True, "twin": _last_twin_state}), 200


@app.get("/twin/rul")
def twin_rul():
    if _twin_engine is None:
        return jsonify({"available": False, "error": "TwinEngine not available"}), 503
    rul = _last_twin_state.get("rul")
    if rul is None:
        return jsonify({"available": False, "error": "No RUL computed yet — stream not started"}), 503
    return jsonify({"available": True, "rul": rul}), 200


@app.post("/twin/reset")
def twin_reset():
    if _twin_engine is None:
        return jsonify({"error": "TwinEngine not available"}), 503
    _twin_engine.reset()
    _last_twin_state.clear()
    return jsonify({"status": "reset"}), 200


# --- Entry point ---

if __name__ == "__main__":
    log.info("Starting Thermo-Twin Alert Backend on port %d", PORT)
    log.info("Demo scenarios loaded: %s", list(demo_scenarios.keys()))
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
    finally:
        if _influx:
            _influx.close()
        if _mqtt:
            _mqtt.disconnect()