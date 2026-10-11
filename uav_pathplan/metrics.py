"""Per-run metric extraction and Monte-Carlo aggregation.

For each (algorithm, scenario, run) we record the full metric set the paper
reports.  All cost-related quantities come from the single shared evaluator
(:func:`uav_pathplan.cost.evaluate`) so the planners are compared on identical
terms; geometric/threat metrics (minimum standoff, peak hazard) are computed
here from the same path and risk field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import Config
from .cost import CostResult, evaluate
from .planners import PlanResult
from .risk import RiskField
from .terrain import TerrainField

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class RunMetrics:
    """One row of results."""

    scenario: str
    algorithm: str
    run: int
    seed: int
    success: bool
    J: float = np.nan
    L: float = np.nan
    Ltil: float = np.nan
    Rtil: float = np.nan
    Phi: float = np.nan
    Lambda_peak: float = np.nan
    d_min: float = np.nan  # minimum 3D standoff to any radar (m)
    kappa_peak: float = np.nan
    curv_violations: int = 0
    gamma_peak_deg: float = np.nan
    climb_violations: int = 0
    smoothness: float = np.nan  # sum |dpsi| (rad)
    time_s: float = np.nan
    n_exp: int = 0
    n_iter: int = 0
    peak_mem_kib: float = np.nan
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def min_standoff(risk: RiskField, terrain: TerrainField, path: np.ndarray) -> float:
    """Minimum 3D distance from any path point to any radar phase centre (m)."""
    if path is None or len(path) == 0 or not risk.radars:
        return np.nan
    P = np.asarray(path, dtype=float)
    z = np.asarray(terrain.altitude_at(P[:, 0], P[:, 1]), dtype=float)
    best = np.inf
    for j, r in enumerate(risk.radars):
        rz = float(risk._radar_z[j])
        d = np.sqrt((P[:, 0] - r.x) ** 2 + (P[:, 1] - r.y) ** 2 + (z - rz) ** 2)
        best = min(best, float(d.min()))
    return best


def compute_metrics(
    cfg: Config,
    terrain: TerrainField,
    risk: RiskField,
    scenario: str,
    algorithm: str,
    run: int,
    res: PlanResult,
    peak_mem_kib: float = np.nan,
) -> RunMetrics:
    """Assemble a :class:`RunMetrics` row from a planner result."""
    m = RunMetrics(
        scenario=scenario,
        algorithm=algorithm,
        run=run,
        seed=cfg.seed,
        success=bool(res.success),
        time_s=res.time_s,
        n_exp=res.n_exp,
        n_iter=res.n_iter,
        peak_mem_kib=peak_mem_kib,
        reason=str(res.info.get("reason", "")),
    )
    if res.n_waypoints >= 2:
        cr: CostResult = evaluate(cfg, risk, terrain, res.path)
        m.J = cr.J
        m.L = cr.L
        m.Ltil = cr.Ltil
        m.Rtil = cr.Rtil
        m.Phi = cr.Phi
        m.Lambda_peak = cr.Lambda_peak
        m.kappa_peak = cr.kappa_peak
        m.curv_violations = cr.curv_violations
        m.gamma_peak_deg = float(np.rad2deg(cr.gamma_peak))
        m.climb_violations = cr.climb_violations
        m.smoothness = cr.smoothness
        m.d_min = min_standoff(risk, terrain, res.path)
    return m


# --------------------------------------------------------------------------- #
# Aggregation (mean +/- std across Monte-Carlo runs)                           #
# --------------------------------------------------------------------------- #
_AGG_FIELDS = [
    "J",
    "L",
    "Ltil",
    "Rtil",
    "Phi",
    "Lambda_peak",
    "d_min",
    "kappa_peak",
    "curv_violations",
    "gamma_peak_deg",
    "climb_violations",
    "smoothness",
    "time_s",
    "n_exp",
    "n_iter",
    "peak_mem_kib",
]


def aggregate(rows: list[RunMetrics], success_only: bool = True) -> "pd.DataFrame":
    """Aggregate rows to mean +/- std per (scenario, algorithm).

    By default only successful runs feed the quality metrics (J, L, ...); the
    success *rate* is reported separately over all runs.
    """
    import pandas as pd

    df = pd.DataFrame([r.as_dict() for r in rows])
    out = []
    for (scn, alg), g in df.groupby(["scenario", "algorithm"], sort=False):
        n_total = len(g)
        n_succ = int(g["success"].sum())
        gg = g[g["success"]] if success_only else g
        rec = {
            "scenario": scn,
            "algorithm": alg,
            "success_rate": n_succ / n_total if n_total else np.nan,
            "n_runs": n_total,
            "n_success": n_succ,
        }
        for f in _AGG_FIELDS:
            vals = gg[f].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            rec[f"{f}_mean"] = float(vals.mean()) if vals.size else np.nan
            rec[f"{f}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out.append(rec)
    return pd.DataFrame(out)
