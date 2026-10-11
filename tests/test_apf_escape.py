"""APF-E (random-walk escape) on a minimal concave case.

A 13-element U-shaped hazard wall on flat terrain opens toward the start.
Plain APF stalls in it; APF-E must escape in most random streams, its
manoeuvres must respect the same turn-rate and step limits, and with
``max_escapes = 0`` it must reproduce plain APF exactly.

Run:  pytest tests/test_apf_escape.py
"""

import numpy as np
import pytest

from uav_pathplan.config import Config, RadarSpec
from uav_pathplan.cost import wrap_angle
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.apf_escape import plan_apf_escape
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField

START, GOAL = (150.0, 500.0), (750.0, 500.0)
PARAMS = dict(
    max_escapes=10, turn_steps_min=10, turn_steps_max=40, straight_steps=20, turn_frac=0.9
)


@pytest.fixture(scope="module")
def u_wall():
    walls = [
        RadarSpec(x=float(x), y=y, mast_height=25, C=1e9, R_max=70)
        for x in np.arange(350, 551, 44)
        for y in (450.0, 550.0)
    ]
    walls += [
        RadarSpec(x=550.0, y=float(y), mast_height=25, C=1e9, R_max=70)
        for y in (450.0, 494.0, 538.0)
    ]
    cfg = Config(fractal_max=0.0, hills=[], ridges=[], radars=walls, R0=20.0, resolution=10.0)
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    risk.grid()
    return cfg, terrain, risk


def test_plain_apf_is_trapped(u_wall):
    cfg, terrain, risk = u_wall
    res = plan_apf(cfg, terrain, risk, START, GOAL)
    assert not res.success and res.info["reason"] == "local_minimum"


def test_apfe_escapes_in_most_streams(u_wall):
    cfg, terrain, risk = u_wall
    results = [
        plan_apf_escape(cfg, terrain, risk, START, GOAL, rng=np.random.default_rng(seed), **PARAMS)
        for seed in range(10)
    ]
    n_ok = sum(r.success for r in results)
    assert n_ok >= 6, f"APF-E escaped in only {n_ok}/10 streams"
    for r in results:
        if r.success:
            assert r.info["n_escapes"] >= 1  # success required an escape
            assert r.info["final_dist_to_goal"] <= cfg.apf_goal_tol


def test_apfe_respects_step_and_turn_limits(u_wall):
    cfg, terrain, risk = u_wall
    r = plan_apf_escape(cfg, terrain, risk, START, GOAL, rng=np.random.default_rng(0), **PARAMS)
    assert r.info["n_escapes"] >= 1
    d = np.diff(r.path, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    assert step.max() <= cfg.apf_alpha + 1e-9
    moving = step > 1e-9
    heading = np.arctan2(d[moving, 1], d[moving, 0])
    dpsi = np.abs(wrap_angle(np.diff(heading)))
    # turn per step never exceeds kappa_max * step length (<= kappa_max * alpha)
    assert dpsi.max() <= cfg.kappa_max * cfg.apf_alpha + 1e-9


def test_apfe_is_deterministic_and_reduces_to_apf(u_wall):
    cfg, terrain, risk = u_wall
    a = plan_apf_escape(cfg, terrain, risk, START, GOAL, rng=np.random.default_rng(3), **PARAMS)
    b = plan_apf_escape(cfg, terrain, risk, START, GOAL, rng=np.random.default_rng(3), **PARAMS)
    np.testing.assert_array_equal(a.path, b.path)
    no_escape = dict(PARAMS, max_escapes=0)
    e0 = plan_apf_escape(cfg, terrain, risk, START, GOAL, rng=np.random.default_rng(0), **no_escape)
    apf = plan_apf(cfg, terrain, risk, START, GOAL)
    np.testing.assert_array_equal(e0.path, apf.path)
    assert e0.success == apf.success
