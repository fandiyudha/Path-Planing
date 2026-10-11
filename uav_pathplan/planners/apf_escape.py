"""APF-E: APF with a random-walk local-minimum escape (protocol v2 baseline).

Identical to :func:`uav_pathplan.planners.apf.plan_apf` -- same force field,
gains (``xi = wL``, ``eta = wR`` by default), kinematically clamped step
(:func:`~uav_pathplan.planners.apf.apf_step`), stall detector, goal tolerance
and iteration budget -- plus an escape mechanism after the randomized path
planner of Barraquand & Latombe (1991): when the stall detector fires, the
vehicle performs a random walk and then resumes gradient descent.

Because the APF vehicle cannot turn without moving (heading change is clamped to
``kappa_max * step``) and a stalled vehicle has zero step, the random walk is a
kinematically feasible manoeuvre flown at the full base step ``alpha``:

1. turn in a random direction (left/right with probability 1/2) at the constant
   rate ``turn_frac * kappa_max`` (``turn_frac < 1``) for ``n_turn`` steps, with
   ``n_turn`` uniform in ``[turn_steps_min, turn_steps_max]``;
2. fly straight for ``straight_steps`` steps.

The walk ignores the potential (as in the randomized planner); its hazard
exposure is scored by the shared evaluator like any other part of the path.
At most ``max_escapes`` walks are made; a further stall is reported as a
local-minimum failure.  Every manoeuvre step counts towards ``apf_max_iter``.
The random stream is supplied by the caller (seeded per run) so results are
reproducible.

Reference: J. Barraquand and J.-C. Latombe, "Robot motion planning: A
distributed representation approach", Int. J. Robotics Research 10(6),
628-649, 1991.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ..config import Config
from ..risk import RiskField
from ..terrain import TerrainField
from . import PlanResult
from .apf import apf_step


def plan_apf_escape(
    cfg: Config,
    terrain: TerrainField,
    risk: RiskField,
    start_xy,
    goal_xy,
    *,
    rng: np.random.Generator,
    max_escapes: int,
    turn_steps_min: int,
    turn_steps_max: int,
    straight_steps: int,
    turn_frac: float,
) -> PlanResult:
    """Run APF-E from ``start_xy`` to ``goal_xy`` (world metres)."""
    t0 = time.perf_counter()
    xi = cfg.wL if cfg.apf_xi is None else cfg.apf_xi
    eta = cfg.wR if cfg.apf_eta is None else cfg.apf_eta
    goal = np.array([float(goal_xy[0]), float(goal_xy[1])])
    p = np.array([float(start_xy[0]), float(start_xy[1])])
    psi = math.atan2(goal[1] - p[1], goal[0] - p[0])
    path = [p.copy()]
    window = deque(maxlen=cfg.apf_stuck_window)
    window.append(p.copy())

    alpha = cfg.apf_alpha
    dpsi_walk = turn_frac * cfg.kappa_max * alpha
    n_escapes = 0
    success = stuck = False
    k = 0

    def at_goal(q):
        return math.hypot(goal[0] - q[0], goal[1] - q[1]) <= cfg.apf_goal_tol

    def walk_step(q, heading):
        q = q + alpha * np.array([math.cos(heading), math.sin(heading)])
        q[0] = min(max(q[0], 0.0), cfg.world_size)
        q[1] = min(max(q[1], 0.0), cfg.world_size)
        return q

    while k < cfg.apf_max_iter:
        if at_goal(p):
            success = True
            break
        k += 1
        p, psi, _ = apf_step(cfg, terrain, risk, p, psi, goal, xi, eta)
        path.append(p.copy())
        window.append(p.copy())
        if len(window) < window.maxlen:
            continue
        if math.hypot(p[0] - window[0][0], p[1] - window[0][1]) >= cfg.apf_stuck_disp:
            continue
        # ---- stall detected ------------------------------------------- #
        if n_escapes >= max_escapes:
            stuck = True
            break
        n_escapes += 1
        direction = 1.0 if rng.random() < 0.5 else -1.0
        n_turn = int(rng.integers(turn_steps_min, turn_steps_max + 1))
        headings = [direction * dpsi_walk] * n_turn + [0.0] * straight_steps
        for dpsi in headings:
            if k >= cfg.apf_max_iter or at_goal(p):
                break
            k += 1
            psi = psi + dpsi
            p = walk_step(p, psi)
            path.append(p.copy())
        window.clear()
        window.append(p.copy())

    elapsed = time.perf_counter() - t0
    arr = np.array(path)
    reason = "reached_goal" if success else "local_minimum" if stuck else "max_iter"
    return PlanResult(
        name="apfe",
        path=arr,
        success=success,
        time_s=elapsed,
        n_iter=k,
        info={
            "reason": reason,
            "stuck": stuck,
            "n_escapes": n_escapes,
            "final_dist_to_goal": float(math.hypot(arr[-1][0] - goal[0], arr[-1][1] - goal[1])),
        },
    )
