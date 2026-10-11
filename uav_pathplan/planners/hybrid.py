"""Hybrid A*-APF planner.

Two layers, combining A*'s global view with APF's local smoothness:

1. **Global layer** -- A* on a COARSER grid (factor ``cfg.hybrid_coarsen``)
   minimising the A* additive length+risk+kinematic edge cost, producing a global
   waypoint corridor that already avoids large hazard regions and concave
   traps (A* cannot be locally entrapped).

2. **Local layer** -- APF runs in continuous space chasing each global waypoint
   as a *moving sub-goal*, advancing to the next once within a capture radius.
   The same kinematically-clamped step as the pure APF planner
   (:func:`uav_pathplan.planners.apf.apf_step`) supplies smoothness and enforces
   ``kappa_max`` (the flight-path-angle limit ``gamma_max`` is not enforced; it
   is only measured by the shared evaluator).  Because the sub-goals lead around hazards, the
   local descent never faces a head-on wall, so APF's local-minimum failure is
   avoided -- the global sub-goals prevent entrapment.

Returns the combined path, combined timing, the global A* expanded-node count,
the local APF iteration count, and a success flag.
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
from .astar import plan_astar


def plan_hybrid(
    cfg: Config, terrain: TerrainField, risk: RiskField, start_xy, goal_xy
) -> PlanResult:
    """Run the Hybrid A*-APF planner from ``start_xy`` to ``goal_xy``."""
    t0 = time.perf_counter()
    # Local layer tracks the (already risk-safe) global corridor strongly; the
    # repulsion (eta = wR) only fine-tunes between coarse waypoints.  This high
    # attraction:repulsion ratio is what prevents the local entrapment that the
    # pure-APF planner suffers -- the global sub-goals lead it around hazards.
    xi = cfg.hybrid_attract
    eta = cfg.wR if cfg.apf_eta is None else cfg.apf_eta

    # --- global layer: coarse A* ---------------------------------------- #
    glob = plan_astar(cfg, terrain, risk, start_xy, goal_xy, grid_step=cfg.hybrid_coarsen)
    if not glob.success or glob.n_waypoints < 2:
        return PlanResult(
            name="hybrid",
            path=np.empty((0, 2)),
            success=False,
            time_s=time.perf_counter() - t0,
            n_exp=glob.n_exp,
            info={"reason": "global_astar_failed"},
        )

    waypoints = glob.path  # (W, 2) coarse corridor
    goal = np.array([float(goal_xy[0]), float(goal_xy[1])])

    # capture radius for advancing sub-goals (~ one coarse cell)
    capture = max(2.0 * cfg.apf_alpha, 1.2 * cfg.hybrid_coarsen * cfg.resolution)

    # --- local layer: APF chasing the corridor -------------------------- #
    p = np.array([float(start_xy[0]), float(start_xy[1])])
    psi = math.atan2(waypoints[1][1] - p[1], waypoints[1][0] - p[0])
    path = [p.copy()]
    window = deque(maxlen=cfg.apf_stuck_window)
    window.append(p.copy())

    sub_idx = 1  # index of current sub-goal
    success = False
    stuck = False
    k = 0
    for k in range(1, cfg.apf_max_iter + 1):
        # advance through any sub-goals already within capture radius
        while (
            sub_idx < len(waypoints) - 1
            and math.hypot(waypoints[sub_idx][0] - p[0], waypoints[sub_idx][1] - p[1]) <= capture
        ):
            sub_idx += 1
        target = goal if sub_idx >= len(waypoints) - 1 else waypoints[sub_idx]

        if math.hypot(goal[0] - p[0], goal[1] - p[1]) <= cfg.apf_goal_tol:
            success = True
            break

        p, psi, moved = apf_step(cfg, terrain, risk, p, psi, target, xi, eta)
        path.append(p.copy())
        window.append(p.copy())

        if len(window) == window.maxlen:
            net = math.hypot(p[0] - window[0][0], p[1] - window[0][1])
            if net < cfg.apf_stuck_disp:
                # only a true stall (not just slow progress toward a far goal)
                stuck = True
                break

    elapsed = time.perf_counter() - t0
    # End within goal_tol; no exact-goal append (avoids a final curvature spike).
    arr = np.array(path)

    reason = "reached_goal" if success else "local_minimum" if stuck else "max_iter"
    return PlanResult(
        name="hybrid",
        path=arr,
        success=success,
        time_s=elapsed,
        n_exp=glob.n_exp,
        n_iter=k,
        info={
            "reason": reason,
            "global_waypoints": int(len(waypoints)),
            "global_time_s": glob.time_s,
            "coarsen": cfg.hybrid_coarsen,
            "final_dist_to_goal": float(math.hypot(arr[-1][0] - goal[0], arr[-1][1] - goal[1])),
        },
    )
