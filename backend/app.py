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

# --- Phase 7: drift detection + auto-recalibration ---
try:
    from drift_detector            import DriftDetector
    from parameter_estimator       import BayesianParameterEstimator
    from recalibration_scheduler   import RecalibrationScheduler, RecalibrationReason
    _drift_detector       = DriftDetector(window_size_samples=500, accuracy_threshold_pct=95.0)
    _parameter_estimator  = (BayesianParameterEstimator(physics_model=_twin_engine._physics)
                              if _twin_engine is not None else None)
    _recal_scheduler      = (RecalibrationScheduler(_parameter_estimator, _drift_detector)
                              if _parameter_estimator is not None else None)
    log.info("Phase 7 ready: drift detector + parameter estimator + recal scheduler")
except Exception as _phase7_exc:
    log.warning("Phase 7 init failed (%s) — recalibration endpoints will 503", _phase7_exc)
    _drift_detector = _parameter_estimator = _recal_scheduler = None
    RecalibrationReason = None

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

# --- Background 10 Hz streaming loop ---
# Pre-generates samples so /stream/next-sample returns instantly from a queue.
import collections as _collections
import time as _time

_sample_queue  = _collections.deque(maxlen=120)  # ~2.4s buffer at 50 Hz
_persist_queue = _collections.deque(maxlen=200)  # InfluxDB/MQTT backlog (drops oldest if slow)
_stream_lock   = threading.Lock()


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


def _persist_worker():
    """Drains InfluxDB/MQTT writes off the critical path. If these services are slow
    this thread falls behind / drops items but NEVER blocks sample generation."""
    global _influx, _mqtt
    while True:
        if not _persist_queue:
            _time.sleep(0.02)
            continue
        kind, payload = _persist_queue.popleft()
        if kind == "sample":
            if _influx:
                try: _influx.write_sample(payload)
                except Exception:
                    log.warning("InfluxDB write failed — disabling for this session")
                    _influx = None
            if _mqtt:
                try: _mqtt.publish_sample(payload)
                except Exception:
                    log.warning("MQTT publish failed — disabling for this session")
                    _mqtt = None
        elif kind == "alert":
            if _influx:
                try: _influx.write_alert(payload)
                except Exception: _influx = None
            if _mqtt:
                try: _mqtt.publish_alert(payload)
                except Exception: _mqtt = None


def _stream_loop():
    INTERVAL = 0.33  # ~3 Hz — slow, realistic sensor streaming rate
    while True:
        t0 = _time.monotonic()
        try:
            with _stream_lock:
                sample = live_streamer.get_next_sample()
                live_injector.apply_to_live_sample(sample)

            history = list(live_streamer.history)
            alert   = live_detector.process_from_history(history)

            twin_result = {}
            if _twin_engine:
                twin_result = _twin_engine.process_sample(sample)
                _last_twin_state.update(twin_result)

            # Phase 7 — feed drift detector + buffer normal-op samples for
            # later re-fitting; poll the scheduler so monthly / drift-triggered
            # recalibrations fire automatically.
            if _drift_detector is not None and twin_result:
                _drift_detector.update(
                    real      = sample,
                    predicted = twin_result.get("prediction", {}),
                    reconstruction_error = float(twin_result.get("rul", {}).get("peak_anomaly_score", 0.0) or 0.0),
                )
            if _parameter_estimator is not None and twin_result and not (alert):
                _parameter_estimator.add_sample(
                    power_kw     = sample["compressor_power_kw"],
                    pressure_psi = sample["discharge_pressure_psi"],
                    fan_rpm      = sample["fan_rpm"],
                    temp_c       = sample["supply_air_temp_c"],
                )
            if _recal_scheduler is not None:
                try:
                    _recal_scheduler.check_and_trigger()
                except Exception as _ex:
                    log.error("Recal check_and_trigger failed: %s", _ex)

            # Hand off persistence to the background worker (non-blocking)
            if _influx or _mqtt:
                twin_sample = TwinSample.from_streamer_dict(sample, machine_id="LIVE-DEMO-UNIT")
                _persist_queue.append(("sample", twin_sample))

            if alert:
                _append_alert(alert)
                _broadcast_ws("alert", alert)
                if _influx or _mqtt:
                    _persist_queue.append(("alert", TwinAlert.from_alert_dict(alert)))

            _broadcast_ws("sample", sample)

            _sample_queue.append({
                "sample":           sample,
                "alert":            alert,
                "current_time":     live_streamer.current_time,
                "buffer_size":      len(live_streamer.history),
                "total_samples":    live_streamer.sample_index,
                "scheduled_faults": live_injector.get_scheduled_faults(),
                "twin":             dict(_last_twin_state) if twin_result else None,
            })
        except Exception as _e:
            log.error("Stream loop error: %s", _e)

        elapsed = _time.monotonic() - t0
        _time.sleep(max(0, INTERVAL - elapsed))


_persist_thread = threading.Thread(target=_persist_worker, daemon=True, name="persist-worker")
_persist_thread.start()

_stream_thread = threading.Thread(target=_stream_loop, daemon=True, name="stream-loop")
_stream_thread.start()
log.info("Background stream loop started at 10 Hz")

# ── Fleet (Phase 6) — multi-unit twin registry + slow tick loop ──────────────
from fleet_manager import FleetManager, seed_demo_fleet

_fleet_manager = FleetManager(
    enable_influx=_influx is not None,
    enable_mqtt  =_mqtt   is not None,
    use_coolprop =False,
)
seed_demo_fleet(_fleet_manager)
log.info("Fleet initialised with %d units: %s",
         len(_fleet_manager.list_units()), _fleet_manager.list_units())


def _fleet_tick_loop():
    """Slow background tick (~1 Hz) that advances each fleet unit by one sample.

    Kept slower than the main stream loop on purpose — fleet units don't need
    chart-grade fps, they need a steady drift so RUL/divergence converge.
    """
    INTERVAL = 1.0
    while True:
        t0 = _time.monotonic()
        try:
            _fleet_manager.tick()
        except Exception as exc:
            log.error("Fleet tick error: %s", exc)
        elapsed = _time.monotonic() - t0
        _time.sleep(max(0, INTERVAL - elapsed))


_fleet_thread = threading.Thread(target=_fleet_tick_loop, daemon=True, name="fleet-tick")
_fleet_thread.start()

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
    # Drain up to 20 queued samples per poll so the frontend never falls behind.
    # At 50 Hz backend / 10 Hz poll, ~5 samples accrue per poll; 20 gives catch-up headroom.
    # The last item becomes the primary response; earlier ones are bundled as backlog.
    batch = []
    for _ in range(20):
        if not _sample_queue:
            break
        batch.append(_sample_queue.popleft())

    if batch:
        primary = batch[-1]
        primary["backlog"] = batch[:-1]   # earlier samples the frontend should also draw
        return jsonify(primary), 200

    # Queue empty (backend warming up) — return placeholder.
    return jsonify({
        "sample":           None,
        "alert":            None,
        "backlog":          [],
        "current_time":     live_streamer.current_time,
        "buffer_size":      len(live_streamer.history),
        "total_samples":    live_streamer.sample_index,
        "scheduled_faults": [],
        "twin":             dict(_last_twin_state) if _last_twin_state else None,
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
    with _stream_lock:
        live_streamer.__init__()          # reset streamer state
        live_injector.__init__(live_streamer)
    _sample_queue.clear()
    _last_twin_state.clear()
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


@app.post("/twin/whatif")
def twin_whatif():
    if _twin_engine is None:
        return jsonify({"error": "TwinEngine not available"}), 503
    params = request.get_json(silent=True) or {}
    try:
        result = _twin_engine.simulate_whatif(params)
        return jsonify(result), 200
    except Exception as exc:
        log.error("What-if simulation failed: %s", exc)
        return jsonify({"error": f"simulation failed: {exc}"}), 500


@app.get("/twin/component-history/<component_name>")
def component_history(component_name):
    """
    GET /twin/component-history/compressor?hours=24

    Returns the recent sensor trace for the sensor associated with a 3D
    component: { history: [{timestamp, value, sensor_type}, ...] }.

    Served from the in-memory streamer ring buffer (real recent samples,
    never blocks). InfluxDB is intentionally NOT queried synchronously here —
    it is optional infrastructure and a down/slow broker previously stalled
    request threads. The buffer holds the genuine recent stream, which is
    what the inspect panel needs.
    """
    sensor_map = {
        "compressor": "compressor_power_kw",
        "condenser":  "fan_rpm",
        "evaporator": "discharge_pressure_psi",
        "valve":      "supply_air_temp_c",
    }
    sensor = sensor_map.get(component_name)
    if not sensor:
        return jsonify({"error": "unknown component"}), 400

    samples = list(live_streamer.history)
    # Downsample to at most ~120 points so the mini-chart stays light
    max_pts = 120
    step = max(1, len(samples) // max_pts)
    history = [
        {
            "timestamp":   round(float(s.get("timestamp", 0.0)), 2),
            "value":       round(float(s.get(sensor, 0.0)), 3),
            "sensor_type": sensor,
        }
        for s in samples[::step]
    ]
    return jsonify({
        "component":   component_name,
        "sensor_type": sensor,
        "history":     history,
        "source":      "stream-buffer",
    }), 200


@app.post("/twin/reset")
def twin_reset():
    if _twin_engine is None:
        return jsonify({"error": "TwinEngine not available"}), 503
    _twin_engine.reset()
    _last_twin_state.clear()
    return jsonify({"status": "reset"}), 200


# ── Phase 7 — drift & recalibration endpoints ────────────────────────────────

@app.get("/twin/drift")
def twin_drift():
    if _drift_detector is None:
        return jsonify({"error": "drift detector not initialised"}), 503
    cur   = _drift_detector.get_current_metrics()
    trend = _drift_detector.get_drift_trend(hours=24)
    return jsonify({"current": cur.to_dict(), "trend_24h": trend}), 200


@app.get("/twin/parameters")
def twin_parameters():
    if _parameter_estimator is None:
        return jsonify({"error": "parameter estimator not initialised"}), 503
    return jsonify({
        "current":         _parameter_estimator.get_current_parameters(),
        "buffered_samples": _parameter_estimator.buffered_sample_count(),
        "recent_updates":  [u.to_dict() for u in _parameter_estimator.get_update_history(20)],
    }), 200


@app.get("/twin/recalibration/status")
def twin_recal_status():
    if _recal_scheduler is None:
        return jsonify({"error": "recalibration scheduler not initialised"}), 503
    return jsonify(_recal_scheduler.get_status()), 200


@app.post("/twin/recalibrate")
def twin_recalibrate():
    if _recal_scheduler is None:
        return jsonify({"error": "recalibration scheduler not initialised"}), 503
    body = request.get_json(silent=True) or {}
    raw  = (body.get("reason") or "manual_request").upper()
    try:
        reason = RecalibrationReason[raw]
    except KeyError:
        reason = RecalibrationReason.MANUAL_REQUEST
    # Demo-friendly: use the full buffer regardless of age so manual triggers
    # always have data to fit against during a short live session.
    event = _recal_scheduler.trigger_recalibration(
        reason=reason,
        lookback_hours=24 * 365,   # effectively "all buffered data"
        confidence_threshold=0.0,  # accept all attempted updates above sanity guard
    )
    return jsonify(event.to_dict()), 200 if event.success else 500


@app.post("/twin/commissioning-reset")
def twin_commissioning_reset():
    if _recal_scheduler is None:
        return jsonify({"error": "recalibration scheduler not initialised"}), 503
    body = request.get_json(silent=True) or {}
    event = _recal_scheduler.trigger_commissioning_reset(
        reason_text=body.get("reason", "maintenance_completed"),
    )
    return jsonify(event.to_dict()), 200


# ── Fleet (Phase 6) endpoints ────────────────────────────────────────────────

@app.get("/fleet/units")
def fleet_list_units():
    units = _fleet_manager.list_units()
    return jsonify({
        "units":       units,
        "total_count": len(units),
        "metadata":    {u: _fleet_manager.get_metadata(u) for u in units},
    }), 200


@app.get("/fleet/health")
def fleet_health():
    return jsonify(_fleet_manager.get_fleet_health()), 200


@app.get("/fleet/dispatch-queue")
def fleet_dispatch_queue():
    top_n = request.args.get("top_n", default=None, type=int)
    return jsonify(_fleet_manager.get_dispatch_queue(top_n=top_n)), 200


@app.get("/fleet/anomalies")
def fleet_anomalies():
    hours = request.args.get("hours", default=24, type=int)
    return jsonify(_fleet_manager.detect_cross_unit_anomalies(window_hours=hours)), 200


@app.post("/fleet/register-unit")
def fleet_register_unit():
    data = request.get_json(silent=True) or {}
    mid  = data.get("machine_id")
    if not mid:
        return jsonify({"error": "machine_id required"}), 400
    ok = _fleet_manager.register_unit(
        machine_id        = mid,
        location          = data.get("location", ""),
        model             = data.get("model", "Carrier-50000-Series"),
        commissioned_date = data.get("commissioned_date"),
        fault_profile     = data.get("fault_profile"),
    )
    if ok:
        return jsonify({"status": "registered", "machine_id": mid}), 201
    return jsonify({"error": "unit already registered or invalid"}), 400


@app.delete("/fleet/<machine_id>")
def fleet_unregister(machine_id):
    if _fleet_manager.unregister_unit(machine_id):
        return jsonify({"status": "unregistered", "machine_id": machine_id}), 200
    return jsonify({"error": f"unit {machine_id} not found"}), 404


@app.get("/fleet/<machine_id>/twin")
def fleet_unit_twin(machine_id):
    try:
        twin = _fleet_manager.get_twin(machine_id)
    except KeyError:
        return jsonify({"error": f"unit {machine_id} not found"}), 404
    return jsonify({
        "machine_id": machine_id,
        "metadata":   _fleet_manager.get_metadata(machine_id),
        "twin":       twin._last_twin_state,
    }), 200


@app.post("/fleet/<machine_id>/sample")
def fleet_unit_sample(machine_id):
    data = request.get_json(silent=True) or {}
    sample = {
        "timestamp":              data.get("timestamp", 0.0),
        "sample_index":           data.get("sample_index", 0),
        "compressor_power_kw":    data.get("compressor_power_kw"),
        "discharge_pressure_psi": data.get("discharge_pressure_psi"),
        "fan_rpm":                data.get("fan_rpm"),
        "supply_air_temp_c":      data.get("supply_air_temp_c"),
    }
    try:
        result = _fleet_manager.process_sample(machine_id, sample)
        return jsonify({"success": True, **result}), 200
    except KeyError:
        return jsonify({"error": f"unit {machine_id} not found"}), 404
    except Exception as exc:
        log.error("Fleet sample processing failed for %s: %s", machine_id, exc)
        return jsonify({"error": str(exc)}), 500


@app.post("/fleet/<machine_id>/reset")
def fleet_unit_reset(machine_id):
    if _fleet_manager.reset_unit(machine_id):
        return jsonify({"status": "reset", "machine_id": machine_id}), 200
    return jsonify({"error": f"unit {machine_id} not found"}), 404


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