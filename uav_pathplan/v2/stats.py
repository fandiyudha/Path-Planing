"""Protocol-v2 statistics: summaries, Wilson CIs, Wilcoxon + Holm + effect size, Friedman."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from ..stats import holm_adjust

Z95 = 1.959963984540054
GROUP = ["resolution", "altitude_mode", "scenario"]


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation of d = a - b (zeros dropped).

    r = (R+ - R-) / (R+ + R-), with R+/R- the rank sums of positive/negative
    differences (average ranks for ties).  r > 0: ``a`` tends to be larger.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((r_plus - r_minus) / (r_plus + r_minus))


def summarize(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per (resolution, mode, scenario, planner): success + stats over successful runs."""
    rows = []
    keys = GROUP + ["planner"]
    for key, g in df.groupby(keys, sort=False):
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson_ci(k, n)
        rec = dict(zip(keys, key))
        rec.update(
            risk_kind=g["risk_kind"].iloc[0],
            n_runs=n,
            n_success=k,
            success_rate=k / n if n else np.nan,
            success_ci_lo=lo,
            success_ci_hi=hi,
        )
        ok = g[g["success"]]
        for m in metrics:
            v = ok[m].to_numpy(dtype=float)
            v = v[~np.isnan(v)]
            if v.size:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                rec.update(
                    {
                        f"{m}_mean": v.mean(),
                        f"{m}_sd": v.std(ddof=1) if v.size > 1 else np.nan,
                        f"{m}_median": med,
                        f"{m}_q1": q1,
                        f"{m}_q3": q3,
                        f"{m}_iqr": q3 - q1,
                    }
                )
            else:
                for s in ("mean", "sd", "median", "q1", "q3", "iqr"):
                    rec[f"{m}_{s}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _paired(g: pd.DataFrame, planners, metric: str) -> pd.DataFrame:
    """Runs x planners table of ``metric`` for successful runs (NaN otherwise)."""
    ok = g[g["success"]]
    return ok.pivot_table(index="run", columns="planner", values=metric, aggfunc="first").reindex(
        columns=list(planners)
    )


def pairwise_tests(df: pd.DataFrame, planners, metric: str = "J", alpha: float = 0.05):
    """Wilcoxon for every planner pair, Holm within (resolution, mode, scenario)."""
    out = []
    for key, g in df.groupby(GROUP, sort=False):
        wide = _paired(g, planners, metric)
        block = []
        for a, b in itertools.combinations(planners, 2):
            both = wide[[a, b]].dropna()
            rec = dict(zip(GROUP, key))
            rec.update(
                metric=metric,
                planner_a=a,
                planner_b=b,
                n_pairs=len(both),
                median_a=both[a].median() if len(both) else np.nan,
                median_b=both[b].median() if len(both) else np.nan,
                median_diff=(both[a] - both[b]).median() if len(both) else np.nan,
                statistic=np.nan,
                p_value=np.nan,
                rank_biserial=np.nan,
                note="",
            )
            if len(both) == 0:
                rec["note"] = "no runs in which both succeeded"
            elif np.allclose(both[a], both[b]):
                rec["note"] = "identical paired values"
            else:
                try:
                    stat, p = wilcoxon(both[a].to_numpy(), both[b].to_numpy())
                    rec.update(statistic=float(stat), p_value=float(p))
                    rec["rank_biserial"] = rank_biserial(both[a], both[b])
                except ValueError as e:
                    rec["note"] = f"wilcoxon: {e}"
            block.append(rec)
        adj = holm_adjust([r["p_value"] for r in block])
        for r, ph in zip(block, adj):
            r["p_holm"] = float(ph)
            r["significant_holm"] = bool(ph < alpha) if np.isfinite(ph) else False
        out.extend(block)
    return pd.DataFrame(out)


def friedman_tests(df: pd.DataFrame, planners, metric: str = "J") -> pd.DataFrame:
    """Friedman test across all planners on runs where every planner succeeded."""
    out = []
    k = len(planners)
    for key, g in df.groupby(GROUP, sort=False):
        wide = _paired(g, planners, metric).dropna()
        rec = dict(zip(GROUP, key))
        rec.update(metric=metric, n_complete=len(wide), k=k, chi2=np.nan, p_value=np.nan)
        rec["kendall_w"] = np.nan
        rec["note"] = ""
        if len(wide) < 2:
            rec["note"] = "fewer than 2 runs in which all planners succeeded"
        else:
            stat, p = friedmanchisquare(*[wide[c].to_numpy() for c in planners])
            rec.update(chi2=float(stat), p_value=float(p))
            rec["kendall_w"] = float(stat / (len(wide) * (k - 1)))
        out.append(rec)
    return pd.DataFrame(out)
