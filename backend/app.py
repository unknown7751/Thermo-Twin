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
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from explainability.alert_payload import build_alert_payload
from model.threshold import ThresholdManager

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

# --- In-memory state ---

alert_history = []

threshold_mgr = ThresholdManager(state_path=THRESHOLD_STATE)
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


# --- Routes ---

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


@app.post("/demo/<scenario>")
def trigger_demo(scenario):
    if scenario not in demo_scenarios:
        log.warning("POST /demo/%s - unknown scenario", scenario)
        return jsonify({"error": "unknown scenario"}), 404
    payload = dict(demo_scenarios[scenario])
    payload["received_at"] = _now_iso()
    _append_alert(payload)
    log.info(
        "Demo triggered  scenario=%s  severity=%s  fault=%s",
        scenario,
        payload.get("severity_score", "?"),
        payload.get("fault_type", "?"),
    )
    return jsonify({"status": "triggered", "scenario": scenario}), 200


# --- Entry point ---

if __name__ == "__main__":
    log.info("Starting Thermo-Twin Alert Backend on port %d", PORT)
    log.info("Demo scenarios loaded: %s", list(demo_scenarios.keys()))
    app.run(host="0.0.0.0", port=PORT, debug=False)