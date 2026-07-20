"""Grid A* with a heading-augmented search state.

The state is ``(ix, iy, h)`` where ``h`` is the index of the move direction used
to *arrive* at the cell.  Augmenting the state with heading is what lets A* see
turn curvature: the per-edge turn penalty depends on the change from the arrival
heading to the departure heading.  Motion is 8- or 16-connected.

Edge cost a->b (identical weights / Lambda / L_ref to the shared cost):

    c(a,b) = wL*||b-a||/L_ref
           + wR*Lambda(midpoint_ab)*||b-a||
           + wK*[ max(0,(kappa-kappa_max)/kappa_max)
                 + max(0,(|gamma|-gamma_max)/gamma_max) ]

``kappa`` is the discrete curvature implied by the heading change (needs heading
in the state); ``gamma`` is the flight-path angle implied by the altitude policy
over the edge.  Midpoint hazard is read from the precomputed hazard grid as the
mean of the two endpoint cells (agrees with the grid at grid points).

Heuristic ``hhat(n) = wL*||p_g - n||/L_ref`` (Euclidean).  It is admissible: the
risk and kinematic contributions to the true cost are non-negative and dropped
from the heuristic, so the heuristic never over-estimates.  It is NOT inflated.

Performance: every per-edge geometric quantity that does not depend on the
search order is precomputed once -- a (move x move) turn-penalty table and, for
each move, full-grid arrays of the length+risk+climb edge cost -- so the hot
loop is pure array indexing with no trig.  The open set is a binary heap
(``heapq``, standard library).
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np

from ..config import Config
from ..risk import RiskField
from ..terrain import TerrainField
from . import PlanResult


def _moves(connectivity: int):
    """List of integer (dx, dy) cell offsets for 8- or 16-connectivity."""
    base = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)]
    if connectivity == 8:
        return base
    knight = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2), (1, 2), (2, -1), (2, 1)]
    return base + knight


def _shift(arr: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """Return A' with A'[iy,ix] = arr[iy+oy, ix+ox]; out-of-range -> NaN."""
    n = arr.shape[0]
    out = np.full_like(arr, np.nan)
    yo_lo, yo_hi = max(0, -oy), n - max(0, oy)
    xo_lo, xo_hi = max(0, -ox), n - max(0, ox)
    yi_lo, yi_hi = max(0, oy), n - max(0, -oy)
    xi_lo, xi_hi = max(0, ox), n - max(0, -ox)
    out[yo_lo:yo_hi, xo_lo:xo_hi] = arr[yi_lo:yi_hi, xi_lo:xi_hi]
    return out


def plan_astar(cfg: Config, terrain: TerrainField, risk: RiskField,
               start_xy, goal_xy, grid_step: int = 1,
               hazard_grid: np.ndarray | None = None) -> PlanResult:
    """A* from ``start_xy`` to ``goal_xy`` (world metres).

    ``grid_step`` coarsens the effective lattice (used by the Hybrid planner's
    global layer): moves jump ``grid_step`` cells, so the working resolution is
    ``grid_step * cfg.resolution``.  ``hazard_grid`` reuses a precomputed grid.
    """
    t0 = time.perf_counter()
    n = cfg.n_cells
    res = cfg.resolution
    s = grid_step
    H = risk.grid() if hazard_grid is None else hazard_grid
    Z = np.asarray(terrain.altitude_at(*np.meshgrid(terrain.xs, terrain.ys)))

    # ---- snap start/goal to the grid_step lattice ---------------------- #
    last = ((n - 1) // s) * s

    def snap(p):
        ix = min(max(int(round(p[0] / res - 0.5)), 0), n - 1)
        iy = min(max(int(round(p[1] / res - 0.5)), 0), n - 1)
        return min((ix // s) * s, last), min((iy // s) * s, last)

    sx, sy = snap(start_xy)
    gx, gy = snap(goal_xy)
    sw = ((sx + 0.5) * res, (sy + 0.5) * res)
    gw = ((gx + 0.5) * res, (gy + 0.5) * res)
    L_ref = max(math.hypot(gw[0] - sw[0], gw[1] - sw[1]), 1e-9)

    # ---- precompute move geometry and per-move edge-cost arrays -------- #
    moves = _moves(cfg.astar_connectivity)
    nm = len(moves)
    seg = np.array([math.hypot(dx, dy) for dx, dy in moves]) * s * res   # (nm,)
    head = np.array([math.atan2(dy, dx) for dx, dy in moves])           # (nm,)
    length_cost = cfg.wL * seg / L_ref                                  # (nm,)

    kmax, inv_kmax = cfg.kappa_max, 1.0 / cfg.kappa_max
    gmax = cfg.gamma_max
    # turn-penalty table T[h_prev, m_new]
    dpsi = np.abs((head[:, None] - head[None, :] + np.pi) % (2 * np.pi) - np.pi)
    sbar = 0.5 * (seg[:, None] + seg[None, :])
    kappa = dpsi / np.maximum(sbar, 1e-9)
    turn_table = cfg.wK * np.maximum(0.0, (kappa - kmax) * inv_kmax)     # (nm, nm)

    # per-move full-grid edge cost (length + risk + climb), excluding turn.
    edge_extra = np.empty((nm, n, n), dtype=float)
    for mi, (dx, dy) in enumerate(moves):
        ox, oy = dx * s, dy * s
        Hn = _shift(H, ox, oy)
        Zn = _shift(Z, ox, oy)
        haz_mid = 0.5 * (H + Hn)
        risk_c = cfg.wR * haz_mid * seg[mi]
        dz = np.abs(Zn - Z)
        gamma = np.arctan2(dz, seg[mi])
        climb_c = cfg.wK * np.maximum(0.0, (gamma - gmax) / gmax)
        edge_extra[mi] = length_cost[mi] + risk_c + climb_c     # NaN where OOB

    # heuristic grid: wL * straight-line distance to goal / L_ref
    iy_g, ix_g = np.mgrid[0:n, 0:n]
    Hh = (cfg.wL / L_ref) * np.hypot((ix_g - gx) * res, (iy_g - gy) * res)

    # ---- A* search ----------------------------------------------------- #
    start_state = (sx, sy, -1)
    g_score = {start_state: 0.0}
    came_from: dict = {}
    open_heap = [(float(Hh[sy, sx]), 0.0, start_state)]
    closed = set()
    n_exp = 0
    goal_state = None

    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        closed.add(state)
        ix, iy, hmove = state
        n_exp += 1
        if ix == gx and iy == gy:
            goal_state = state
            break

        turn_row = turn_table[hmove] if hmove >= 0 else None
        for mi in range(nm):
            dx, dy = moves[mi]
            nx, ny = ix + dx * s, iy + dy * s
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            c = edge_extra[mi, iy, ix]
            if turn_row is not None:
                c += turn_row[mi]
            ng = g + c
            nstate = (nx, ny, mi)
            if ng < g_score.get(nstate, 1e18):
                g_score[nstate] = ng
                came_from[nstate] = state
                heapq.heappush(open_heap, (ng + Hh[ny, nx], ng, nstate))

    elapsed = time.perf_counter() - t0
    if goal_state is None:
        return PlanResult(name="astar", path=np.empty((0, 2)), success=False,
                          time_s=elapsed, n_exp=n_exp,
                          info={"reason": "no path found"})

    cells = []
    st = goal_state
    while st in came_from:
        cells.append((st[0], st[1]))
        st = came_from[st]
    cells.append((sx, sy))
    cells.reverse()
    path = np.array([[(cx + 0.5) * res, (cy + 0.5) * res] for cx, cy in cells])

    return PlanResult(name="astar", path=path, success=True, time_s=elapsed,
                      n_exp=n_exp, info={"grid_step": grid_step,
                                         "n_states": len(g_score)})
