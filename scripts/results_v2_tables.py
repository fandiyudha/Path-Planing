"""Render the protocol-v2 result tables as Markdown (for RESULTS_V2.md).

Reads the CSVs in paper_assets_v2/ (or --dir) and writes
paper_assets_v2/RESULTS_V2_tables.md.  Numbers are formatted only; nothing is
recomputed except simple pivots of the aggregate table.

Run (from the repository root):  python scripts/results_v2_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = {"astar": "A-Star", "apf": "APF", "apfe": "APF-E", "hybrid": "Hybrid"}
SCN = ["open", "concave_trap", "ridge_corridor", "high_density"]
PLN = ["astar", "apf", "apfe", "hybrid"]
MODES = ["nap", "rate_limited"]


def f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if abs(x) != 0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def p_fmt(p):
    if p is None or np.isnan(p):
        return "–"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def scn_label(s, risk_kind=None):
    return s + (" †" if risk_kind == "relative_exposure" else "")


def table2(t2: pd.DataFrame, mode: str) -> str:
    t = t2[t2["altitude_mode"] == mode]
    rows = []
    for s in SCN:
        for p in PLN:
            r = t[(t["scenario"] == s) & (t["planner"] == p)].iloc[0]
            rows.append(
                {
                    "scenario": scn_label(s, r["risk_kind"]),
                    "planner": LABEL[p],
                    "success % [95 % CI]": f"{100 * r['success_rate']:.0f} "
                    f"[{100 * r['success_ci95_lo']:.0f}, {100 * r['success_ci95_hi']:.0f}]",
                    "J median [IQR]": f"{f(r['J_median'])} [{f(r['J_iqr'])}]",
                    "J mean ± SD": f"{f(r['J_mean'])} ± {f(r['J_sd'])}",
                    "L̃ median": f(r["Ltil_median"]),
                    "R̃ median": f(r["Rtil_median"]),
                    "Φ median": f(r["Phi_median"]),
                    "κ-viol. %": f(100 * r["curv_viol_frac_mean"], 1),
                    "γ-viol. %": f(100 * r["climb_viol_frac_mean"], 1),
                    "time median (s)": f(r["time_s_median"]),
                }
            )
    return md(pd.DataFrame(rows))


def table3(t3: pd.DataFrame, mode: str) -> str:
    t = t3[t3["altitude_mode"] == mode]
    rows = []
    for s in SCN:
        for _, r in t[t["scenario"] == s].iterrows():
            rows.append(
                {
                    "scenario": s,
                    "pair (a vs b)": f"{LABEL[r['planner_a']]} vs {LABEL[r['planner_b']]}",
                    "n": int(r["n_pairs"]),
                    "median J a / b": f"{f(r['median_a'])} / {f(r['median_b'])}"
                    if r["n_pairs"]
                    else "–",
                    "W": f(r["statistic"], 1),
                    "p": p_fmt(r["p_value"]),
                    "p (Holm)": p_fmt(r["p_holm"]),
                    "r (rank-biserial)": f(r["rank_biserial"], 2),
                    "note": r["note"] if isinstance(r["note"], str) else "",
                }
            )
    return md(pd.DataFrame(rows))


def friedman(fr: pd.DataFrame) -> str:
    rows = []
    for _, r in fr.iterrows():
        rows.append(
            {
                "resolution (m)": f"{r['resolution']:.0f}",
                "altitude": r["altitude_mode"],
                "scenario": r["scenario"],
                "n (all 4 succeed)": int(r["n_complete"]),
                "χ²": f(r["chi2"], 2),
                "p": p_fmt(r["p_value"]),
                "Kendall's W": f(r["kendall_w"], 3),
                "note": r["note"] if isinstance(r["note"], str) else "",
            }
        )
    return md(pd.DataFrame(rows))


def grid(agg: pd.DataFrame, mode: str) -> str:
    a = agg[agg["altitude_mode"] == mode]
    res = sorted(a["resolution"].unique())
    rows = []
    for s in SCN:
        for p in PLN:
            g = a[(a["scenario"] == s) & (a["planner"] == p)].set_index("resolution")
            row = {"scenario": s, "planner": LABEL[p]}
            row["success % (" + "/".join(f"{x:.0f}" for x in res) + " m)"] = " / ".join(
                f"{100 * g.loc[x, 'success_rate']:.0f}" for x in res
            )
            row["median J"] = " / ".join(f(g.loc[x, "J_median"]) for x in res)
            row["median time (s)"] = " / ".join(f(g.loc[x, "time_s_median"]) for x in res)
            rows.append(row)
    return md(pd.DataFrame(rows))


def altitude(agg: pd.DataFrame, ref: float) -> str:
    a = agg[agg["resolution"] == ref]
    rows = []
    for s in SCN:
        for p in PLN:
            g = a[(a["scenario"] == s) & (a["planner"] == p)].set_index("altitude_mode")
            rows.append(
                {
                    "scenario": scn_label(s, g["risk_kind"].iloc[0]),
                    "planner": LABEL[p],
                    "R̃ median nap / RL": f"{f(g.loc['nap', 'Rtil_median'])} / "
                    f"{f(g.loc['rate_limited', 'Rtil_median'])}",
                    "J median nap / RL": f"{f(g.loc['nap', 'J_median'])} / "
                    f"{f(g.loc['rate_limited', 'J_median'])}",
                    "γ-viol. % nap / RL": f"{f(100 * g.loc['nap', 'climb_viol_frac_mean'], 1)} / "
                    f"{f(100 * g.loc['rate_limited', 'climb_viol_frac_mean'], 1)}",
                    "mean AGL (m) RL": f(g.loc["rate_limited", "agl_mean_median"], 1),
                }
            )
    return md(pd.DataFrame(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper_assets_v2")
    args = ap.parse_args()
    d = Path(args.dir)
    env = json.loads((d / "environment.json").read_text(encoding="utf-8"))
    ref = float(env["reference_resolution_m"])
    agg = pd.read_csv(d / "results_aggregate.csv")
    t2 = pd.read_csv(d / "table2_main_results.csv")
    t3 = pd.read_csv(d / "table3_pairwise_tests.csv")
    fr = pd.read_csv(d / "stats_friedman_J.csv")
    out = []
    out.append(f"<!-- generated by scripts/results_v2_tables.py from {d}/ -->")
    out.append(f"\n### Table 2 — main results, {ref:.0f} m grid, nap altitude\n")
    out.append(table2(t2, "nap"))
    out.append(f"\n### Table 2b — main results, {ref:.0f} m grid, rate-limited altitude\n")
    out.append(table2(t2, "rate_limited"))
    out.append(f"\n### Table 3 — paired Wilcoxon tests on J, {ref:.0f} m, nap\n")
    out.append(table3(t3, "nap"))
    out.append(f"\n### Table 3b — paired Wilcoxon tests on J, {ref:.0f} m, rate-limited\n")
    out.append(table3(t3, "rate_limited"))
    out.append("\n### Friedman tests on J (runs where all four planners succeed)\n")
    out.append(friedman(fr))
    for mode in MODES:
        out.append(f"\n### Grid sensitivity ({mode}): values at 5 / 8 / 10 m\n")
        out.append(grid(agg, mode))
    out.append(f"\n### Altitude modes at {ref:.0f} m (RL = rate-limited)\n")
    out.append(altitude(agg, ref))
    path = d / "RESULTS_V2_tables.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
