"""
Phase 7 — drift / parameter / scheduler unit tests.

Run with:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_phase7.py -v
"""

import sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pytest

from drift_detector             import DriftDetector, SENSOR_KEYS
from parameter_estimator        import BayesianParameterEstimator, _PRIORS
from recalibration_scheduler    import RecalibrationScheduler, RecalibrationReason
from physics.hvac_physics       import HVACPhysicsModel


# ── helpers ─────────────────────────────────────────────────────────────────
def perfect_pair(power=3.5):
    """Identical real & predicted dicts → zero error."""
    s = {
        "compressor_power_kw":    power,
        "discharge_pressure_psi": 70.0  * power,
        "fan_rpm":                340.0 * power,
        "supply_air_temp_c":      18.0 - 2.0 * power,
    }
    return s, dict(s)


# ── DriftDetector ────────────────────────────────────────────────────────────
def test_drift_detector_initial_accuracy_is_100():
    d = DriftDetector()
    m = d.get_current_metrics()
    assert m.accuracy_pct == 100.0
    assert m.is_drifting is False


def test_drift_detector_zero_error_stays_at_100():
    d = DriftDetector(baseline_warmup=10)
    for _ in range(40):
        r, p = perfect_pair()
        m = d.update(r, p, reconstruction_error=0.0)
    assert m.accuracy_pct == 100.0
    assert m.is_drifting is False


def test_drift_detector_flags_drift_when_predictions_diverge():
    d = DriftDetector(baseline_warmup=10, accuracy_threshold_pct=95.0)
    # Build baseline at near-zero error
    for _ in range(40):
        r, p = perfect_pair();  d.update(r, p)
    # Now feed predictions that are systematically off (10% bias)
    for _ in range(60):
        r, p = perfect_pair()
        for k in p:
            p[k] *= 0.85
        m = d.update(r, p)
    assert m.accuracy_pct < 95.0
    assert m.is_drifting is True


def test_drift_detector_reset_baseline_relatches_to_current_error():
    d = DriftDetector(baseline_warmup=10)
    for _ in range(20):
        r, p = perfect_pair(); d.update(r, p)
    # introduce systematic error
    for _ in range(20):
        r, p = perfect_pair()
        for k in p: p[k] *= 0.9
        d.update(r, p)
    assert d.get_current_metrics().accuracy_pct < 100.0
    d.reset_baseline()
    # First sample after reset re-anchors to the new error level — should be
    # back near 100 (small jitter from one rolling-buffer slot turning over).
    r, p = perfect_pair()
    for k in p: p[k] *= 0.9
    m = d.update(r, p)
    assert m.accuracy_pct >= 99.0


def test_drift_trend_returns_insufficient_data_with_no_history():
    d = DriftDetector()
    t = d.get_drift_trend(hours=24)
    assert t.get("insufficient_data") is True


# ── ParameterEstimator ──────────────────────────────────────────────────────
def test_parameter_estimator_starts_at_priors():
    e = BayesianParameterEstimator()
    p = e.get_current_parameters()
    for k in ("k_disc", "k_fan", "k_temp_a", "k_temp_b"):
        assert p[k] == pytest.approx(_PRIORS[k]["mean"], rel=1e-6)


def test_parameter_estimator_recovers_synthetic_k_disc():
    physics = HVACPhysicsModel(use_coolprop=False)
    e = BayesianParameterEstimator(physics_model=physics)
    rng = np.random.default_rng(0)
    # Synthetic data: y = 72 * x (slightly above the 69.99 prior)
    for _ in range(500):
        x = rng.uniform(2.0, 5.0)
        e.add_sample(power_kw=x, pressure_psi=72.0 * x + rng.normal(0, 1.0),
                     fan_rpm=340.0 * x, temp_c=18.0 - 2.0 * x)
    updates = e.estimate_parameters(lookback_hours=1e6, confidence_threshold=0.0)
    by_name = {u.parameter_name: u for u in updates}
    assert "k_disc" in by_name
    assert by_name["k_disc"].rejected is False
    # Posterior should sit between prior (69.99) and data (~72)
    assert 70.0 < by_name["k_disc"].new_value < 72.5
    # And physics model received the new value
    assert physics._K_DISC == pytest.approx(by_name["k_disc"].new_value, rel=1e-6)


def test_parameter_estimator_rejects_excessive_change():
    e = BayesianParameterEstimator()
    rng = np.random.default_rng(1)
    # Wildly off data → fit would want >10% change → rejected by sanity guard
    for _ in range(200):
        x = rng.uniform(2.0, 5.0)
        e.add_sample(power_kw=x, pressure_psi=120.0 * x,
                     fan_rpm=340.0 * x, temp_c=18.0 - 2.0 * x)
    updates = e.estimate_parameters(lookback_hours=1e6, confidence_threshold=0.0)
    k_disc = next(u for u in updates if u.parameter_name == "k_disc")
    assert k_disc.rejected is True
    assert "exceeds" in (k_disc.reject_reason or "")


def test_parameter_estimator_skips_when_insufficient_data():
    e = BayesianParameterEstimator()
    updates = e.estimate_parameters(min_samples=50)
    assert updates == []


# ── RecalibrationScheduler ───────────────────────────────────────────────────
def test_scheduler_next_is_in_the_future():
    s = RecalibrationScheduler(BayesianParameterEstimator(), DriftDetector())
    assert s.should_recalibrate() is False


def test_manual_trigger_records_event_and_resets_baseline():
    d = DriftDetector(baseline_warmup=5)
    for _ in range(10):
        r, p = perfect_pair(); d.update(r, p)
    e = BayesianParameterEstimator()
    for _ in range(200):
        e.add_sample(3.5, 245.0, 1190.0, 11.0)
    s = RecalibrationScheduler(e, d)
    ev = s.trigger_recalibration(reason=RecalibrationReason.MANUAL_REQUEST,
                                 confidence_threshold=0.0, lookback_hours=1e6)
    assert ev.success is True
    status = s.get_status()
    assert status["recent_events"][-1]["reason"] == "manual_request"
    assert status["last_recalibration"] is not None


def test_commissioning_reset_clears_estimator_buffer():
    e = BayesianParameterEstimator()
    for _ in range(50): e.add_sample(3.5, 245.0, 1190.0, 11.0)
    assert e.buffered_sample_count() == 50
    s = RecalibrationScheduler(e, DriftDetector())
    ev = s.trigger_commissioning_reset(reason_text="compressor_replacement")
    assert ev.success is True
    assert e.buffered_sample_count() == 0


# ── HTTP layer via Flask test client ─────────────────────────────────────────
def test_phase7_http_endpoints():
    os.environ.setdefault("INFLUX_TOKEN", "test")
    import app as A
    c = A.app.test_client()

    for url in ("/twin/drift", "/twin/parameters", "/twin/recalibration/status"):
        r = c.get(url)
        assert r.status_code == 200, f"{url} returned {r.status_code}: {r.get_data(as_text=True)[:200]}"

    # Manual recal endpoint — may have nothing to fit if buffer empty; either
    # success=True with zero updates or success=False are both acceptable.
    r = c.post("/twin/recalibrate", json={"reason": "manual_request"})
    assert r.status_code in (200, 500)

    # Commissioning reset always succeeds
    r = c.post("/twin/commissioning-reset", json={"reason": "demo"})
    assert r.status_code == 200
    assert r.get_json()["success"] is True
