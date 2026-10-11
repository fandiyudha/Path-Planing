"""Artificial Potential Field (APF) planner -- continuous gradient descent.

Potential field
---------------
The step direction is derived from the force

    F(p) = xi * (p_g - p) / ||p_g - p||  -  eta * grad Lambda(p)

i.e. a unit-magnitude ("capped") attraction toward the goal plus repulsion from
detection hazard (equivalently the descent direction of
``U = xi*||p - p_g|| + eta*Lambda``).

``grad Lambda`` is obtained by **central finite differences** (step
``apf_fd_step``) on ``RiskField.Lambda_field``, the bilinear interpolation of the
precomputed hazard grid at planning altitude.  The gain ratio is mapped to the
cost weights:
``eta/xi = wR/wL`` (so the field encodes the same length/risk trade-off the
shared cost uses).

To keep the well-conditioned in a 1000 m world we follow a *normalised*-step
variant: the descent direction is taken from the (capped-attractive) force and
the move has a fixed base length ``alpha`` metres, then clamped for kinematics.

Kinematic clamping
------------------
* Horizontal: the per-step heading change is clamped to ``kappa_max * step`` so
  the realised curvature never exceeds ``kappa_max = 1/R_min``.
* Vertical: NOT enforced.  The step is not shrunk for climb (``apf_min_step``
  is unused); flight-path-angle violations are only measured afterwards by the
  shared evaluator (climb penalty in J and ``climb_violations``).
* Step length: ``min(alpha, distance to target) * max(0, cos theta)``, where
  ``theta`` is the angle between the turn-limited heading and ``F``.

Local-minimum detection
-----------------------
APF can stall in a local minimum of ``U`` (e.g. a concave hazard pocket).  We
detect a stall when the *net* displacement over the last ``stuck_window``
iterations falls below ``stuck_disp`` while the goal has not been reached, and
report failure.  This is intentional: capturing APF's local-minimum failure is
a key experimental result, so NO global-escape heuristic is added to mask it.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from ..config import Config
from ..cost import wrap_angle
from ..risk import RiskField
from ..terrain import TerrainField
from . import PlanResult


def _grad_lambda(risk: RiskField, x: float, y: float, h: float):
    """Central-difference gradient of the continuous hazard field at (x, y).

    Uses the fast grid-interpolated field ``Lambda_field`` (consistent with the
    hazard grid at grid points); the four probes are evaluated in one batched
    call.
    """
    xs = np.array([x + h, x - h, x, x])
    ys = np.array([y, y, y + h, y - h])
    v = risk.Lambda_field(xs, ys)
    lx = (v[0] - v[1]) / (2.0 * h)
    ly = (v[2] - v[3]) / (2.0 * h)
    return float(lx), float(ly)


def apf_step(
    cfg: Config,
    terrain: TerrainField,
    risk: RiskField,
    p: np.ndarray,
    psi: float,
    target: np.ndarray,
    xi: float,
    eta: float,
):
    """One kinematically-clamped APF descent step toward ``target``.

    Shared by :func:`plan_apf` (single goal) and the Hybrid planner (moving
    sub-goal) so both enforce *identical* kinematics.  Returns
    ``(p_new, psi_new, moved)`` where ``moved`` is the realised step length (m);
    ``moved == 0`` flags a blocked / stalled state.
    """
    base_step = cfg.apf_alpha
    max_dpsi = cfg.kappa_max * base_step

    dist_t = math.hypot(target[0] - p[0], target[1] - p[1])
    to_t = (target - p) / max(dist_t, 1e-9)
    gradL = np.array(_grad_lambda(risk, p[0], p[1], cfg.apf_fd_step))
    F = xi * to_t - eta * gradL  # descent force = -grad U
    nF = math.hypot(F[0], F[1])
    if nF < 1e-12:
        return p.copy(), psi, 0.0

    # provisional heading: steepest turn toward F, limited to kappa_max*base_step
    raw_dpsi = float(wrap_angle(math.atan2(F[1], F[0]) - psi))
    prov_dir = np.array(
        [
            math.cos(psi + max(-max_dpsi, min(max_dpsi, raw_dpsi))),
            math.sin(psi + max(-max_dpsi, min(max_dpsi, raw_dpsi))),
        ]
    )

    # speed factor: do not fly into rising potential; if the turn-limited
    # heading cannot align with the descent force, the vehicle cannot progress.
    align = float(prov_dir @ F) / nF
    step_len = min(base_step, dist_t) * max(0.0, align)

    # No vertical step-shrinking.  Under the nap-of-the-earth altitude policy
    # (z = h + clearance) the flight-path angle equals the directional terrain
    # slope, which is scale-invariant -- shrinking the step does NOT reduce gamma,
    # it only collapses the spacing to sub-metre and artificially inflates the
    # per-step angle.  We therefore take the full, uniform step; climb compliance
    # comes from the (now gentle) terrain and the gamma penalty remains in cost J.
    s = step_len

    # Couple the realised turn to the actual arc length (kappa = dpsi/ds): the
    # heading may change by at most kappa_max * s, so realised curvature never
    # exceeds kappa_max.  A blocked vehicle (s -> 0) cannot turn and stalls.
    real_dpsi = max(-cfg.kappa_max * s, min(cfg.kappa_max * s, raw_dpsi))
    psi_new = psi + real_dpsi
    direction = np.array([math.cos(psi_new), math.sin(psi_new)])
    p_new = p + s * direction
    p_new[0] = min(max(p_new[0], 0.0), cfg.world_size)
    p_new[1] = min(max(p_new[1], 0.0), cfg.world_size)
    return p_new, psi_new, s


def plan_apf(cfg: Config, terrain: TerrainField, risk: RiskField, start_xy, goal_xy) -> PlanResult:
    """Run APF gradient descent from ``start_xy`` to ``goal_xy`` (world metres)."""
    t0 = time.perf_counter()
    xi = cfg.wL if cfg.apf_xi is None else cfg.apf_xi
    eta = cfg.wR if cfg.apf_eta is None else cfg.apf_eta
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    p = np.array([float(start_xy[0]), float(start_xy[1])])
    goal = np.array([gx, gy])

    psi = math.atan2(gy - p[1], gx - p[0])  # initial heading toward goal
    path = [p.copy()]
    window = deque(maxlen=cfg.apf_stuck_window)
    window.append(p.copy())

    success = False
    stuck = False
    k = 0
    for k in range(1, cfg.apf_max_iter + 1):
        if math.hypot(goal[0] - p[0], goal[1] - p[1]) <= cfg.apf_goal_tol:
            success = True
            break
        p, psi, moved = apf_step(cfg, terrain, risk, p, psi, goal, xi, eta)
        path.append(p.copy())
        window.append(p.copy())
        if len(window) == window.maxlen:
            net = math.hypot(p[0] - window[0][0], p[1] - window[0][1])
            if net < cfg.apf_stuck_disp:
                stuck = True
                break

    elapsed = time.perf_counter() - t0
    # The path ends within goal_tol of the goal; we do NOT append the exact goal
    # point because a final off-heading jog would inject a spurious curvature
    # spike.  The clamp then guarantees realised curvature <= kappa_max.
    arr = np.array(path)

    reason = "reached_goal" if success else "local_minimum" if stuck else "max_iter"
    return PlanResult(
        name="apf",
        path=arr,
        success=success,
        time_s=elapsed,
        n_iter=k,
        info={
            "reason": reason,
            "stuck": stuck,
            "final_dist_to_goal": float(math.hypot(arr[-1][0] - gx, arr[-1][1] - gy)),
        },
    )
