"""Scenario suite.

Four scenario *classes*, each a seeded builder returning a :class:`ScenarioSpec`
(a fully-populated :class:`Config` plus start/goal).  Every builder is driven by
an integer seed so any run reproduces exactly; ``randomize=True`` draws a
Monte-Carlo instance (jittered start/goal, and where appropriate jittered
threats) from a per-run RNG derived from a master seed.

Classes
-------
1. ``open``            -- few/no threats; sanity check, all planners succeed.
2. ``concave_trap``    -- a C-shaped high-hazard wall whose mouth faces the
                          start; designed to trap pure APF in a local minimum
                          while the Hybrid's global layer routes around it.
3. ``ridge_corridor``  -- a ridge shadowing a low-detection corridor; the
                          risk-aware optimum is the longer shielded detour, not
                          the exposed straight line.
4. ``high_density``    -- many radars, to stress A* search cost and contrast
                          with the Hybrid's coarse global layer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import Config, GaussianHill, RadarSpec, Ridge


@dataclass
class ScenarioSpec:
    name: str
    cfg: Config
    start: tuple[float, float]
    goal: tuple[float, float]


def _run_rng(master_seed: int, scenario_id: int, run_idx: int) -> np.random.Generator:
    """Deterministic independent RNG stream for one (scenario, run)."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, scenario_id, run_idx]))


# --------------------------------------------------------------------------- #
# 1. Open terrain                                                              #
# --------------------------------------------------------------------------- #
def build_open(master_seed: int = 0, run_idx: int = 0, randomize: bool = False) -> ScenarioSpec:
    rng = _run_rng(master_seed, 1, run_idx)
    terrain_seed = int(rng.integers(1, 2**31 - 1)) if randomize else 11111
    cfg = Config(
        seed=terrain_seed,
        fractal_max=70.0,
        # gentle, broad rolling hills (slope < gamma_max), not obstacles
        hills=[
            GaussianHill(cx=300, cy=700, height=45, sigma=140),
            GaussianHill(cx=750, cy=300, height=40, sigma=130),
        ],
        ridges=[],
        # a single modest radar set off the start->goal diagonal: it nudges the
        # path but never blocks it, so all planners succeed (sanity check).
        radars=[RadarSpec(x=700, y=320, mast_height=20, R_max=220)],  # C calibrated
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    if randomize:
        start = (float(rng.uniform(60, 160)), float(rng.uniform(60, 160)))
        goal = (float(rng.uniform(840, 940)), float(rng.uniform(840, 940)))
    else:
        start, goal = (100.0, 100.0), (900.0, 900.0)
    return ScenarioSpec("open", cfg, start, goal)


# --------------------------------------------------------------------------- #
# 2. Concave trap                                                             #
# --------------------------------------------------------------------------- #
def build_concave_trap(
    master_seed: int = 0, run_idx: int = 0, randomize: bool = False
) -> ScenarioSpec:
    """A narrow dead-end radar canyon whose open mouth faces the start.

    Two parallel side walls (at ``500 +/- half_w``) and a closed back wall (at
    ``x_back``) form a cul-de-sac opening toward -x (the start); the goal sits
    behind the back wall.  The geometry is chosen against the vehicle kinematics:

    * The full canyon width ``2*half_w`` is **less than the turning diameter**
      ``2*R_min``, so the vehicle physically cannot U-turn inside it.
    * Radar spacing exceeds ``2*R0`` so there are no zero-gradient floor gaps to
      slip through, and ``R_max < R_min`` keeps the channel mouth enterable.

    Heading toward the goal, APF drives into the concave mouth, is repelled
    head-on by the wall, cannot turn away within the channel, and stalls in a
    local minimum (reported as failure).  A* / Hybrid route around the outside.
    """
    rng = _run_rng(master_seed, 2, run_idx)
    half_w, x_mouth, x_back = 50.0, 350.0, 750.0  # width 100 m < 2*R_min=120
    # NOTE: the canyon "radars" are a proxy-OBSTACLE wall whose only job is to
    # induce APF's local-minimum failure, so their power constant is set
    # EXPLICITLY (strong, uncalibrated) to give a firm repulsive gradient -- the
    # detection-probability calibration (which would make them far too weak to
    # trap) is intentionally overridden here.  Rtil is not the point of this
    # scenario; the detection scenarios (open/ridge_corridor/high_density) use
    # the calibrated C.
    C_wall, R_max, R0 = 1.0e9, 70.0, 20.0
    spacing = 2.2 * R0  # > 2*R0: no floor-gap holes
    y_lo, y_hi = 500.0 - half_w, 500.0 + half_w
    jit = (lambda v: v + rng.uniform(-6, 6)) if randomize else (lambda v: v)

    radars = []
    for x in np.arange(x_mouth, x_back + 1, spacing):  # side walls
        radars.append(
            RadarSpec(x=jit(float(x)), y=jit(y_lo), mast_height=25, C=C_wall, R_max=R_max)
        )
        radars.append(
            RadarSpec(x=jit(float(x)), y=jit(y_hi), mast_height=25, C=C_wall, R_max=R_max)
        )
    for y in np.arange(y_lo, y_hi + 1, spacing):  # back wall
        radars.append(
            RadarSpec(x=jit(float(x_back)), y=jit(float(y)), mast_height=25, C=C_wall, R_max=R_max)
        )

    cfg = Config(
        seed=22222,
        fractal_max=60.0,
        hills=[],
        ridges=[],
        radars=radars,
        R0=R0,
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    if randomize:
        start = (float(rng.uniform(130, 180)), float(rng.uniform(480, 520)))
        goal = (float(rng.uniform(880, 920)), float(rng.uniform(480, 520)))
    else:
        start, goal = (150.0, 500.0), (900.0, 500.0)
    return ScenarioSpec("concave_trap", cfg, start, goal)


# --------------------------------------------------------------------------- #
# 3. Ridge-shielded corridor                                                  #
# --------------------------------------------------------------------------- #
def build_ridge_corridor(
    master_seed: int = 0, run_idx: int = 0, randomize: bool = False
) -> ScenarioSpec:
    """A central radar with two ridges shadowing an L-shaped corridor.

    The straight diagonal start->goal passes the central radar (exposed).  The
    ridges cast a detection shadow over the south+east edge corridor, so the
    risk-aware optimum is the longer shielded L-route.
    """
    rng = _run_rng(master_seed, 3, run_idx)
    ridges = [
        # horizontal ridge shielding the southern corridor from the centre radar
        Ridge(cx=500, cy=360, height=230, sigma=38, length=300, orientation_deg=0.0),
        # vertical ridge shielding the eastern corridor
        Ridge(cx=640, cy=560, height=230, sigma=38, length=240, orientation_deg=90.0),
    ]
    cfg = Config(
        seed=33333,
        fractal_max=70.0,
        # gentle relief hill; the tall ridges above are the deliberate masking
        # obstacles and are intentionally kept steep (flown around, not over).
        hills=[GaussianHill(cx=250, cy=800, height=55, sigma=130)],
        ridges=ridges,
        radars=[RadarSpec(x=500, y=520, mast_height=30, R_max=520)],  # C calibrated
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    if randomize:
        start = (float(rng.uniform(90, 150)), float(rng.uniform(90, 150)))
        goal = (float(rng.uniform(850, 920)), float(rng.uniform(850, 920)))
    else:
        start, goal = (120.0, 120.0), (880.0, 880.0)
    return ScenarioSpec("ridge_corridor", cfg, start, goal)


# --------------------------------------------------------------------------- #
# 4. High threat density                                                      #
# --------------------------------------------------------------------------- #
def build_high_density(
    master_seed: int = 0, run_idx: int = 0, randomize: bool = False, n_radars: int = 16
) -> ScenarioSpec:
    """Many radars scattered through the interior to stress search cost."""
    rng = _run_rng(master_seed, 4, run_idx)
    terrain_seed = int(rng.integers(1, 2**31 - 1)) if randomize else 44444
    rr = np.random.default_rng(terrain_seed)
    radars = []
    for _ in range(n_radars):
        radars.append(
            RadarSpec(
                x=float(rr.uniform(200, 800)),
                y=float(rr.uniform(200, 800)),
                mast_height=20,
                R_max=190,
            )
        )  # C calibrated
    cfg = Config(
        seed=terrain_seed,
        fractal_max=80.0,
        hills=[GaussianHill(cx=500, cy=500, height=60, sigma=150)],
        ridges=[],
        radars=radars,
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    if randomize:
        start = (float(rng.uniform(60, 140)), float(rng.uniform(60, 140)))
        goal = (float(rng.uniform(860, 940)), float(rng.uniform(860, 940)))
    else:
        start, goal = (90.0, 90.0), (910.0, 910.0)
    return ScenarioSpec("high_density", cfg, start, goal)


def build_pareto(master_seed: int = 0, run_idx: int = 0, randomize: bool = False) -> ScenarioSpec:
    """Single strong radar on the start->goal line with NO terrain shielding.

    The only way to lower detection risk is to fly a longer way around the
    radar, so sweeping the risk weight ``wR`` traces a genuine length-vs-risk
    Pareto front (unlike ``ridge_corridor``, where a fully shielded route exists
    at almost any weight).
    """
    cfg = Config(
        seed=55555,
        fractal_max=40.0,
        hills=[],
        ridges=[],
        radars=[RadarSpec(x=500, y=500, mast_height=25, R_max=430)],  # C calibrated
        R0=50.0,
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    return ScenarioSpec("pareto", cfg, (100.0, 500.0), (900.0, 500.0))


# --------------------------------------------------------------------------- #
SCENARIO_BUILDERS = {
    "open": build_open,
    "concave_trap": build_concave_trap,
    "ridge_corridor": build_ridge_corridor,
    "high_density": build_high_density,
}


def build_scenario(
    name: str, master_seed: int = 0, run_idx: int = 0, randomize: bool = False
) -> ScenarioSpec:
    """Build a scenario by name (see :data:`SCENARIO_BUILDERS`)."""
    if name not in SCENARIO_BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; choices: {list(SCENARIO_BUILDERS)}")
    return SCENARIO_BUILDERS[name](master_seed, run_idx, randomize)


def with_overrides(spec: ScenarioSpec, **cfg_overrides) -> ScenarioSpec:
    """Return a copy of ``spec`` with ``Config`` fields overridden."""
    return replace(spec, cfg=replace(spec.cfg, **cfg_overrides))
