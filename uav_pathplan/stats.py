"""Paired significance testing across matched Monte-Carlo runs.

For each scenario we pair runs by index (same master seed -> same randomized
start/goal/threats for every algorithm) and apply the Wilcoxon signed-rank test
(``scipy.stats.wilcoxon``) to the three algorithm pairs (A* vs APF, A* vs Hybrid,
APF vs Hybrid) on a chosen metric (default the final cost ``J``).  Only runs
where *both* algorithms in a pair succeeded contribute to that pair's test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon

from .metrics import RunMetrics


@dataclass
class PairTest:
    scenario: str
    metric: str
    alg_a: str
    alg_b: str
    n_pairs: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    significant: bool          # at alpha
    note: str = ""


def _matched(rows: list[RunMetrics], scenario: str, alg: str, metric: str) -> dict:
    """Map run-index -> metric value for successful runs of one algorithm."""
    out = {}
    for r in rows:
        if r.scenario == scenario and r.algorithm == alg and r.success:
            out[r.run] = getattr(r, metric)
    return out


def paired_tests(rows: list[RunMetrics], metric: str = "J",
                 alpha: float = 0.05) -> list[PairTest]:
    """Run Wilcoxon signed-rank tests for all algorithm pairs, per scenario."""
    scenarios = sorted({r.scenario for r in rows})
    algos = ["astar", "apf", "hybrid"]
    pairs = [("astar", "apf"), ("astar", "hybrid"), ("apf", "hybrid")]
    results: list[PairTest] = []
    for scn in scenarios:
        vals = {a: _matched(rows, scn, a, metric) for a in algos}
        for a, b in pairs:
            common = sorted(set(vals[a]) & set(vals[b]))
            xa = np.array([vals[a][i] for i in common], dtype=float)
            xb = np.array([vals[b][i] for i in common], dtype=float)
            pt = PairTest(scenario=scn, metric=metric, alg_a=a, alg_b=b,
                          n_pairs=len(common),
                          median_a=float(np.median(xa)) if xa.size else np.nan,
                          median_b=float(np.median(xb)) if xb.size else np.nan,
                          statistic=np.nan, p_value=np.nan, significant=False)
            if len(common) < 1 or np.allclose(xa, xb):
                pt.note = "insufficient or identical paired data"
            else:
                try:
                    stat, p = wilcoxon(xa, xb)
                    pt.statistic = float(stat)
                    pt.p_value = float(p)
                    pt.significant = bool(p < alpha)
                except ValueError as e:           # e.g. all differences zero
                    pt.note = f"wilcoxon: {e}"
            results.append(pt)
    return results


def tests_to_frame(tests: list[PairTest]):
    """Convert paired-test results to a pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame([t.__dict__ for t in tests])
