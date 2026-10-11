"""Protocol-v2 scenario builders (Monte-Carlo variation in every scenario).

Each builder keeps the scenario's defining structure (open single radar,
concave wall facing the start, two shielding ridges, a field of radars) and
draws, from a per-run stream ``SeedSequence([master_seed, scenario_id, run, 2])``:

* the base fractal terrain seed (all scenarios),
* start and goal,
* the structural jitter documented in ``configs/protocol_v2.yaml``.

The v1 builders in :mod:`uav_pathplan.scenarios` are not changed.
"""

from __future__ import annotations

import numpy as np

from ..config import Config, GaussianHill, RadarSpec, Ridge
from ..scenarios import ScenarioSpec
from .protocol import SCENARIO_IDS

PROTOCOL_TAG = 2  # distinguishes v2 streams from the v1 streams


def run_rng(master_seed: int, scenario: str, run_idx: int) -> np.random.Generator:
    ss = np.random.SeedSequence([master_seed, SCENARIO_IDS[scenario], run_idx, PROTOCOL_TAG])
    return np.random.default_rng(ss)


def apfe_rng(master_seed: int, scenario: str, run_idx: int) -> np.random.Generator:
    """Independent stream for APF-E's random walks (same for every resolution)."""
    ss = np.random.SeedSequence([master_seed, SCENARIO_IDS[scenario], run_idx, 101])
    return np.random.default_rng(ss)


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _terrain_seed(rng) -> int:
    return int(rng.integers(1, 2**31 - 1))


def _start_goal(rng, c):
    start = (_u(rng, c["start_x"]), _u(rng, c["start_y"]))
    goal = (_u(rng, c["goal_x"]), _u(rng, c["goal_y"]))
    return start, goal


def build_open(c: dict, rng) -> ScenarioSpec:
    seed = _terrain_seed(rng)
    r = c["radar"]
    j = r["jitter_m"]
    radar = RadarSpec(
        x=r["x"] + _u(rng, (-j, j)),
        y=r["y"] + _u(rng, (-j, j)),
        mast_height=r["mast_height_m"],
        R_max=r["R_max_m"],
    )
    start, goal = _start_goal(rng, c)
    cfg = Config(
        seed=seed,
        fractal_max=c["fractal_max_m"],
        hills=[
            GaussianHill(cx=300, cy=700, height=45, sigma=140),
            GaussianHill(cx=750, cy=300, height=40, sigma=130),
        ],
        ridges=[],
        radars=[radar],
    )
    return ScenarioSpec("open", cfg, start, goal)


def build_concave_trap(c: dict, rng) -> ScenarioSpec:
    seed = _terrain_seed(rng)
    w = c["wall"]
    dx = _u(rng, (-w["offset_x_m"], w["offset_x_m"]))
    dy = _u(rng, (-w["offset_y_m"], w["offset_y_m"]))
    jit = w["element_jitter_m"]
    yc = w["y_centre_m"] + dy
    y_lo, y_hi = yc - w["half_width_m"], yc + w["half_width_m"]
    x_mouth, x_back = w["x_mouth_m"] + dx, w["x_back_m"] + dx

    def elem(x, y):
        return RadarSpec(
            x=x + _u(rng, (-jit, jit)),
            y=y + _u(rng, (-jit, jit)),
            mast_height=w["mast_height_m"],
            C=w["C"],
            R_max=w["R_max_m"],
        )

    radars = []
    for x in np.arange(x_mouth, x_back + 1, w["spacing_m"]):  # side walls
        radars.append(elem(float(x), y_lo))
        radars.append(elem(float(x), y_hi))
    for y in np.arange(y_lo, y_hi + 1, w["spacing_m"]):  # back wall
        radars.append(elem(float(x_back), float(y)))
    start = (_u(rng, c["start_x"]), yc + _u(rng, c["start_dy"]))
    goal = (_u(rng, c["goal_x"]), yc + _u(rng, c["goal_dy"]))
    cfg = Config(
        seed=seed,
        fractal_max=c["fractal_max_m"],
        hills=[],
        ridges=[],
        radars=radars,
        R0=w["R0_m"],
    )
    return ScenarioSpec("concave_trap", cfg, start, goal)


def build_ridge_corridor(c: dict, rng) -> ScenarioSpec:
    seed = _terrain_seed(rng)
    off = c["ridge_offset_m"]
    ridges = []
    for r in c["ridges"]:
        ridges.append(
            Ridge(
                cx=r["x"] + _u(rng, (-off, off)),
                cy=r["y"] + _u(rng, (-off, off)),
                height=r["height_m"] * _u(rng, c["ridge_height_factor"]),
                sigma=r["sigma_m"],
                length=r["length_m"],
                orientation_deg=r["orientation_deg"],
            )
        )
    rd = c["radar"]
    j = rd["jitter_m"]
    radar = RadarSpec(
        x=rd["x"] + _u(rng, (-j, j)),
        y=rd["y"] + _u(rng, (-j, j)),
        mast_height=rd["mast_height_m"],
        R_max=rd["R_max_m"],
    )
    start, goal = _start_goal(rng, c)
    cfg = Config(
        seed=seed,
        fractal_max=c["fractal_max_m"],
        hills=[GaussianHill(cx=250, cy=800, height=55, sigma=130)],
        ridges=ridges,
        radars=[radar],
    )
    return ScenarioSpec("ridge_corridor", cfg, start, goal)


def build_high_density(c: dict, rng) -> ScenarioSpec:
    seed = _terrain_seed(rng)
    lo, hi = c["radar_area_m"]
    radars = [
        RadarSpec(
            x=_u(rng, (lo, hi)),
            y=_u(rng, (lo, hi)),
            mast_height=c["mast_height_m"],
            R_max=c["R_max_m"],
        )
        for _ in range(int(c["n_radars"]))
    ]
    start, goal = _start_goal(rng, c)
    cfg = Config(
        seed=seed,
        fractal_max=c["fractal_max_m"],
        hills=[GaussianHill(cx=500, cy=500, height=60, sigma=150)],
        ridges=[],
        radars=radars,
    )
    return ScenarioSpec("high_density", cfg, start, goal)


BUILDERS = {
    "open": build_open,
    "concave_trap": build_concave_trap,
    "ridge_corridor": build_ridge_corridor,
    "high_density": build_high_density,
}


def build(proto: dict, scenario: str, master_seed: int, run_idx: int) -> ScenarioSpec:
    """Build one Monte-Carlo instance of ``scenario`` under the v2 protocol."""
    rng = run_rng(master_seed, scenario, run_idx)
    spec = BUILDERS[scenario](proto["scenarios"][scenario], rng)
    spec.cfg.clearance = float(proto["altitude"]["nap_clearance_m"])
    return spec
