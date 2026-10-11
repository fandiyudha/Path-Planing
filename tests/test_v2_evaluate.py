"""Protocol v2: L_ref fix, altitude modes and consistency with the v1 evaluator.

Run:  pytest tests/test_v2_evaluate.py
"""

from dataclasses import replace

import numpy as np
import pytest

from uav_pathplan import cost as costmod
from uav_pathplan import scenarios as scn
from uav_pathplan.config import Config
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField
from uav_pathplan.v2.altitude import build_profile, lower_envelope
from uav_pathplan.v2.evaluate import evaluate_v2, resample_with_arclength

C_MIN, PROFILE_DS = 50.0, 1.0


@pytest.fixture(scope="module")
def flat():
    cfg = Config(fractal_max=0.0, hills=[], ridges=[], radars=[], resolution=8.0)
    terrain = TerrainField.generate(cfg)
    return cfg, terrain, RiskField.build(cfg, terrain)


@pytest.fixture(scope="module")
def ridge():
    """v1 ridge_corridor (steep 230 m ridges), 8 m grid, with its A-Star path."""
    spec = scn.build_ridge_corridor()
    cfg = replace(spec.cfg, resolution=8.0)
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    res = plan_astar(cfg, terrain, risk, spec.start, spec.goal, hazard_grid=risk.grid())
    assert res.success
    return cfg, terrain, risk, spec, res.path


# --------------------------------------------------------------------------- #
# L_ref fix                                                                    #
# --------------------------------------------------------------------------- #
def test_lref_uses_scenario_start_goal(flat):
    cfg, terrain, risk = flat
    start, goal = (100.0, 100.0), (900.0, 100.0)
    # path starts 6 m off the start and stops 10 m short of the goal
    path = np.array([[100.0, 106.0], [500.0, 106.0], [890.0, 100.0]])
    m = evaluate_v2(cfg, risk, terrain, path, start, goal, "nap", C_MIN, PROFILE_DS)
    assert m["L_ref"] == pytest.approx(800.0)  # scenario start/goal, not path ends
    assert m["gap_start"] == pytest.approx(6.0)
    assert m["gap_goal"] == pytest.approx(10.0)
    L_path = 400.0 + np.hypot(390.0, 6.0)
    assert m["L_path"] == pytest.approx(L_path)
    assert m["L"] == pytest.approx(6.0 + L_path + 10.0)  # closed length
    assert m["Ltil"] == pytest.approx((6.0 + L_path + 10.0 - 800.0) / 800.0)
    # the v1 evaluator uses the path's own end points instead
    v1 = costmod.evaluate(cfg, risk, terrain, path)
    assert v1.L_ref == pytest.approx(np.hypot(790.0, 6.0))


def test_ltil_zero_for_straight_path_short_of_goal(flat):
    """A straight path that stops short is not credited with negative Ltil."""
    cfg, terrain, risk = flat
    path = np.array([[100.0, 500.0], [888.0, 500.0]])
    m = evaluate_v2(cfg, risk, terrain, path, (100, 500), (900, 500), "nap", C_MIN, PROFILE_DS)
    assert m["Ltil"] == pytest.approx(0.0, abs=1e-12)
    assert m["Ltil"] >= 0.0


# --------------------------------------------------------------------------- #
# Rate-limited altitude constraints                                            #
# --------------------------------------------------------------------------- #
def test_lower_envelope_matches_definition():
    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(0, 500, 300))
    b = rng.uniform(0, 200, 300)
    t = np.tan(np.deg2rad(15.0))
    brute = np.max(b[None, :] - t * np.abs(s[:, None] - s[None, :]), axis=1)
    np.testing.assert_allclose(lower_envelope(s, b, t), brute, rtol=0, atol=1e-9)


def test_rate_limited_profile_constraints(ridge):
    cfg, terrain, _, _, _ = ridge
    # straight north-south line across the 230 m ridges (forces steep climbs)
    path = np.array([[500.0, 100.0], [500.0, 900.0]])
    prof = build_profile(
        terrain, path, np.zeros(0), "rate_limited", 50.0, C_MIN, cfg.gamma_max, 1.0
    )
    tan_g = np.tan(cfg.gamma_max)
    # never below h + c_min
    assert np.all(prof.z >= prof.h + C_MIN - 1e-9)
    # slope never above tan(gamma_max) along arc length
    slope = np.abs(np.diff(prof.z)) / np.diff(prof.s)
    assert slope.max() <= tan_g + 1e-9
    # lowest such profile: it touches the floor somewhere
    assert np.min(prof.z - (prof.h + C_MIN)) == pytest.approx(0.0, abs=1e-9)
    # the path crosses steep terrain, so the slope constraint is active
    assert np.max(prof.z - (prof.h + C_MIN)) > 100.0
    # nap on the same line would violate the climb limit (sanity check of the case)
    nap = build_profile(terrain, path, np.zeros(0), "nap", 50.0, C_MIN, cfg.gamma_max, 1.0)
    assert (np.abs(np.diff(nap.z)) / np.diff(nap.s)).max() > tan_g


def test_rate_limited_on_ridge_crossing_has_no_climb_violations(ridge):
    cfg, terrain, risk, spec, _ = ridge
    path = np.array([[500.0, 100.0], [500.0, 900.0]])
    rl = evaluate_v2(cfg, risk, terrain, path, (500, 100), (500, 900), "rate_limited", C_MIN, 1.0)
    nap = evaluate_v2(cfg, risk, terrain, path, (500, 100), (500, 900), "nap", C_MIN, 1.0)
    assert nap["climb_violations"] > 0
    assert rl["climb_violations"] == 0
    assert rl["gamma_peak_deg"] <= np.rad2deg(cfg.gamma_max)


def test_rate_limited_mode_has_no_climb_violations(ridge):
    cfg, terrain, risk, spec, path = ridge
    nap = evaluate_v2(cfg, risk, terrain, path, spec.start, spec.goal, "nap", C_MIN, PROFILE_DS)
    rl = evaluate_v2(
        cfg, risk, terrain, path, spec.start, spec.goal, "rate_limited", C_MIN, PROFILE_DS
    )
    assert rl["climb_violations"] == 0
    assert rl["gamma_peak_deg"] <= np.rad2deg(cfg.gamma_max) + 1e-9
    assert rl["agl_min"] >= C_MIN - 1e-9
    assert rl["agl_mean"] >= nap["agl_mean"] - 1e-9  # never lower than nap (c = c_min)
    # horizontal-only quantities do not depend on the altitude mode
    for k in ("L", "Ltil", "kappa_peak", "curv_violations", "smoothness"):
        assert rl[k] == nap[k]


# --------------------------------------------------------------------------- #
# Consistency with v1 where the definitions coincide                           #
# --------------------------------------------------------------------------- #
def test_resampling_matches_v1(ridge):
    _, _, _, _, path = ridge
    Q, t = resample_with_arclength(path, 8.0)
    np.testing.assert_array_equal(Q, costmod.resample_polyline(path, 8.0))
    np.testing.assert_allclose(np.diff(t), t[1] - t[0])


def test_nap_detection_and_curvature_match_v1(ridge):
    cfg, terrain, risk, spec, path = ridge
    v1 = costmod.evaluate(cfg, risk, terrain, path)
    v2 = evaluate_v2(cfg, risk, terrain, path, spec.start, spec.goal, "nap", C_MIN, PROFILE_DS)
    assert v2["Phi"] == pytest.approx(v1.Phi, rel=1e-9, abs=1e-12)
    assert v2["L_path"] == pytest.approx(v1.L, rel=1e-12)
    assert v2["kappa_peak"] == pytest.approx(v1.kappa_peak, rel=1e-12)
    assert v2["curv_violations"] == v1.curv_violations
    assert v2["smoothness"] == pytest.approx(v1.smoothness, rel=1e-12)
