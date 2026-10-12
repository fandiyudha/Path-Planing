"""Independent re-computation of the protocol-v2 statistics from the per-run CSV.

Re-implements Wilson CIs, medians/IQR, paired Wilcoxon tests with Holm
correction and the rank-biserial effect size from
paper_assets_v2/results_per_run.csv (read-only), as a cross-check of
uav_pathplan/v2/stats.py.  Prints the 8 m / nap tables and writes
outputs/verify_v2/agg_mine.csv and outputs/verify_v2/tests_mine.csv.

Usage (from the repository root):  python scripts/verify_v2_stats.py .
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

R = sys.argv[1]
d = pd.read_csv(f"{R}/paper_assets_v2/results_per_run.csv")
P = ["astar", "apf", "apfe", "hybrid"]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


def rbc(a, b):
    x = np.asarray(a) - np.asarray(b)
    x = x[x != 0]
    if len(x) == 0:
        return np.nan
    r = rankdata(np.abs(x))
    return (r[x > 0].sum() - r[x < 0].sum()) / r.sum()


def holm(ps):
    ps = np.asarray(ps, float)
    idx = np.argsort(ps)
    m = len(ps)
    out = np.empty(m)
    run = 0
    for k, i in enumerate(idx):
        run = max(run, min(1, (m - k) * ps[i]))
        out[i] = run
    return out


rows = []
tests = []
for (res, mode, scn), g in d.groupby(["resolution", "altitude_mode", "scenario"]):
    for p in P:
        gp = g[g.planner == p]
        s = gp[gp.success]
        lo, hi = wilson(int(gp.success.sum()), len(gp))
        rows.append(
            dict(
                res=res,
                mode=mode,
                scn=scn,
                pl=p,
                n=len(gp),
                succ=gp.success.mean(),
                ci_lo=lo,
                ci_hi=hi,
                J_med=s.J.median(),
                J_iqr=s.J.quantile(0.75) - s.J.quantile(0.25),
                J_mean=s.J.mean(),
                J_sd=s.J.std(),
                R_med=s.Rtil.median(),
                L_med=s.Ltil.median(),
                kv=100 * s.curv_viol_frac.mean(),
                gv=100 * s.climb_viol_frac.mean(),
                t_med=s.time_s.median(),
            )
        )
    piv = g.pivot(index="run", columns="planner", values="J")
    ok = g.pivot(index="run", columns="planner", values="success")
    pv = []
    meta = []
    for a, b in itertools.combinations(P, 2):
        m = ok[a] & ok[b]
        x = piv.loc[m, a]
        y = piv.loc[m, b]
        if m.sum() < 2 or np.allclose(x, y):
            continue
        st = wilcoxon(x, y)
        pv.append(st.pvalue)
        meta.append((a, b, int(m.sum()), st.statistic, rbc(x, y)))
    for (a, b, n, W, r), ph in zip(meta, holm(pv)):
        tests.append(dict(res=res, mode=mode, scn=scn, a=a, b=b, n=n, W=W, p_holm=ph, r=r))
A = pd.DataFrame(rows)
T = pd.DataFrame(tests)
OUT = Path("outputs") / "verify_v2"
OUT.mkdir(parents=True, exist_ok=True)
A.to_csv(OUT / "agg_mine.csv", index=False)
T.to_csv(OUT / "tests_mine.csv", index=False)
pd.set_option("display.width", 250)
print(A[(A.res == 8) & (A["mode"] == "nap")].round(4).to_string(index=False))
print(T[(T.res == 8) & (T["mode"] == "nap")].to_string(index=False))
