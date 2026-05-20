"""
Phase 6 — FleetManager unit tests.

Run with pytest:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_fleet_manager.py -v

These are pure-python tests of the FleetManager logic — no Flask, no HTTP.
The HTTP layer (/fleet/* endpoints) is exercised via Flask's test_client.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import pytest

from fleet_manager import (
    FleetManager,
    seed_demo_fleet,
    DISPATCH_COST_INR,
    INR_PROACTIVE_SAVE,
)
from physics.degradation_model import ComponentHealth


# ── 1. registration ──────────────────────────────────────────────────────────
def test_register_unit_succeeds_and_appears_in_list():
    mgr = FleetManager()
    assert mgr.register_unit("U-001", location="Building A") is True
    assert "U-001" in mgr.list_units()
    assert mgr.get_metadata("U-001")["location"] == "Building A"


def test_duplicate_registration_returns_false():
    mgr = FleetManager()
    mgr.register_unit("U-001")
    assert mgr.register_unit("U-001") is False
    assert len(mgr.list_units()) == 1


def test_unregister_unit_removes_it():
    mgr = FleetManager()
    mgr.register_unit("U-001")
    assert mgr.unregister_unit("U-001") is True
    assert "U-001" not in mgr.list_units()
    assert mgr.unregister_unit("U-001") is False  # already gone


def test_get_twin_raises_for_unknown_unit():
    mgr = FleetManager()
    with pytest.raises(KeyError):
        mgr.get_twin("does-not-exist")


# ── 2. ticking / sample processing ───────────────────────────────────────────
def test_tick_advances_every_registered_unit():
    mgr = FleetManager()
    mgr.register_unit("U-A")
    mgr.register_unit("U-B")
    n = mgr.tick()
    assert n == 2
    # Each unit should now have a cached _last_twin_state with the canonical keys
    for mid in ("U-A", "U-B"):
        state = mgr.get_twin(mid)._last_twin_state
        assert "state" in state and "rul" in state and "divergence" in state


def test_process_sample_for_unknown_unit_raises():
    mgr = FleetManager()
    with pytest.raises(KeyError):
        mgr.process_sample("nope", {"timestamp": 0, "sample_index": 0,
                                    "compressor_power_kw": 3.5,
                                    "discharge_pressure_psi": 245,
                                    "fan_rpm": 1190,
                                    "supply_air_temp_c": 11})


# ── 3. aggregations ──────────────────────────────────────────────────────────
def test_get_fleet_health_empty_fleet():
    mgr = FleetManager()
    h = mgr.get_fleet_health()
    assert h["total_units"] == 0
    assert h["scored_units"] == 0
    assert h["avg_health_pct"] == 100.0
    assert h["fleet_uptime_pct"] == 0.0


def test_get_fleet_health_classifies_units_by_band():
    mgr = FleetManager()
    seed_demo_fleet(mgr)
    for _ in range(60):
        mgr.tick()
    h = mgr.get_fleet_health()
    assert h["total_units"] == 4
    assert h["scored_units"] == 4
    # Buckets sum to the scored count
    assert h["healthy"] + h["warning"] + h["critical"] == h["scored_units"]
    # Per-unit breakdown is present
    assert set(h["health_by_unit"].keys()) == {"UNIT-001", "UNIT-002", "UNIT-003", "UNIT-004"}


# ── 4. dispatch queue ────────────────────────────────────────────────────────
def test_dispatch_queue_sorted_by_rul_ascending():
    mgr = FleetManager()
    seed_demo_fleet(mgr)
    for _ in range(60):
        mgr.tick()
    dq = mgr.get_dispatch_queue()
    ruls = [item["rul_days"] for item in dq["queue"]]
    assert ruls == sorted(ruls), "dispatch queue must be sorted by rul_days ascending"
    # Priorities start at 1 and are dense
    priorities = [item["priority"] for item in dq["queue"]]
    assert priorities == list(range(1, len(priorities) + 1))


def test_dispatch_queue_roi_math():
    mgr = FleetManager()
    seed_demo_fleet(mgr)
    for _ in range(60):
        mgr.tick()
    dq = mgr.get_dispatch_queue(top_n=2)
    assert len(dq["queue"]) == 2
    assert dq["estimated_dispatch_cost_inr"] == 2 * DISPATCH_COST_INR
    assert dq["estimated_save_by_proactive_inr"] == 2 * INR_PROACTIVE_SAVE


def test_dispatch_top_n_limits_results():
    mgr = FleetManager()
    seed_demo_fleet(mgr)
    for _ in range(30):
        mgr.tick()
    full = mgr.get_dispatch_queue()
    capped = mgr.get_dispatch_queue(top_n=1)
    assert len(capped["queue"]) == 1
    assert capped["total_units_in_queue"] == full["total_units_in_queue"]


# ── 5. cross-unit anomaly detection ──────────────────────────────────────────
def test_cross_unit_anomaly_fires_for_seeded_refrigerant_loss():
    mgr = FleetManager()
    seed_demo_fleet(mgr)
    for _ in range(60):
        mgr.tick()
    anomalies = mgr.detect_cross_unit_anomalies()["anomalies"]
    patterns = {a["pattern"] for a in anomalies}
    assert "refrigerant_loss" in patterns
    refrig = next(a for a in anomalies if a["pattern"] == "refrigerant_loss")
    # The two seeded Building-A refrigerant-leak units must be in the affected set
    assert {"UNIT-001", "UNIT-002"}.issubset(set(refrig["affected_units"]))


def test_cross_unit_anomaly_returns_empty_for_healthy_fleet():
    mgr = FleetManager()
    mgr.register_unit("HEALTHY-1")
    mgr.register_unit("HEALTHY-2")
    for _ in range(20):
        mgr.tick()
    anomalies = mgr.detect_cross_unit_anomalies()["anomalies"]
    assert anomalies == []


def test_location_cluster_finds_common_prefix():
    assert FleetManager._location_cluster(
        ["Building A, Floor 3", "Building A, Floor 5"]
    ) == "Building A"
    assert FleetManager._location_cluster(
        ["Building A, Floor 3", "Building B, Floor 1"]
    ) == ""
    assert FleetManager._location_cluster([]) == ""


# ── 6. reset ─────────────────────────────────────────────────────────────────
def test_reset_unit_clears_cached_twin_state():
    mgr = FleetManager()
    mgr.register_unit("U-X", initial_health=ComponentHealth(80, 80, 80))
    for _ in range(5):
        mgr.tick()
    assert mgr.get_twin("U-X")._last_twin_state  # non-empty
    assert mgr.reset_unit("U-X") is True
    assert mgr.get_twin("U-X")._last_twin_state == {}
    assert mgr.reset_unit("unknown") is False


# ── 7. fault probability sigmoid ─────────────────────────────────────────────
def test_fault_probability_curve():
    mgr = FleetManager()
    assert mgr._fault_probability(9999) == 0.0   # MAX_RUL sentinel
    assert mgr._fault_probability(60) == 0.05    # > 30d → low constant
    p_short = mgr._fault_probability(1.0)
    p_long  = mgr._fault_probability(20.0)
    assert p_short > p_long                       # urgency rises as RUL falls
    assert 0.0 <= p_short <= 1.0 and 0.0 <= p_long <= 1.0


# ── 8. HTTP layer smoke test ─────────────────────────────────────────────────
def test_fleet_http_endpoints_via_flask_test_client():
    """End-to-end through the Flask app — confirms wiring not just logic."""
    os.environ.setdefault("INFLUX_TOKEN", "test")  # quiet warnings
    import app as A  # noqa: WPS433
    client = A.app.test_client()

    # /fleet/units
    r = client.get("/fleet/units")
    assert r.status_code == 200
    j = r.get_json()
    assert j["total_count"] == len(j["units"]) >= 4   # demo seed

    # /fleet/health
    r = client.get("/fleet/health")
    assert r.status_code == 200
    assert "fleet_uptime_pct" in r.get_json()

    # /fleet/dispatch-queue
    r = client.get("/fleet/dispatch-queue?top_n=2")
    assert r.status_code == 200
    assert "queue" in r.get_json()

    # /fleet/anomalies
    r = client.get("/fleet/anomalies?hours=24")
    assert r.status_code == 200
    assert "anomalies" in r.get_json()

    # register a brand-new unit
    r = client.post("/fleet/register-unit", json={"machine_id": "TEST-99", "location": "Lab"})
    assert r.status_code == 201
    r = client.post("/fleet/register-unit", json={"machine_id": "TEST-99"})
    assert r.status_code == 400  # duplicate

    # per-unit twin endpoint
    r = client.get("/fleet/TEST-99/twin")
    assert r.status_code == 200
    assert r.get_json()["machine_id"] == "TEST-99"

    # reset
    r = client.post("/fleet/TEST-99/reset")
    assert r.status_code == 200

    # unknown
    r = client.get("/fleet/NOPE/twin")
    assert r.status_code == 404

    # unregister
    r = client.delete("/fleet/TEST-99")
    assert r.status_code == 200
