"""Diagnostic (not used for paper results): verify the radar-C calibration gives graded risk.

Read-only check.  Run (from the repository root):
    python -m scripts.diagnostics.verify_calib
Writes outputs/calib_rtil_population.png and outputs/calib_pareto_main.png
(the outputs/ directory must already exist).
"""

from dataclasses import replace

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uav_pathplan import cost as costmod
from uav_pathplan import scenarios as scn
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.hybrid import plan_hybrid
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField

RES = 8.0
FN = [("astar", plan_astar), ("apf", plan_apf), ("hybrid", plan_hybrid)]
LAB = {"astar": "A*", "apf": "APF", "hybrid": "Hybrid"}
MAIN = ["open", "concave_trap", "ridge_corridor", "high_density"]


def build(name):
    spec = scn.build_pareto() if name == "pareto" else scn.build_scenario(name)
    cfg = replace(spec.cfg, resolution=RES)
    t = TerrainField.generate(cfg)
    r = RiskField.build(cfg, t)
    r.grid()
    return cfg, t, r, spec.start, spec.goal


def run_all(cfg, t, r, s, g):
    H = r.grid()
    return {
        n: fn(cfg, t, r, s, g, **({"hazard_grid": H} if fn is plan_astar else {})) for n, fn in FN
    }


def candidates(start, goal, world, n=160):
    s = np.array(start, float)
    g = np.array(goal, float)
    d = g - s
    perp = np.array([-d[1], d[0]])
    perp /= np.hypot(*perp) + 1e-9
    tt = np.linspace(0, 1, 80)
    base = s + tt[:, None] * d
    out = []
    for off in np.linspace(-0.5 * world, 0.5 * world, n):
        P = base + (off * np.sin(np.pi * tt))[:, None] * perp
        P[:, 0] = np.clip(P[:, 0], 0, world)
        P[:, 1] = np.clip(P[:, 1], 0, world)
        out.append(P)
    return out


print("=" * 78)
print("CALIBRATION VERIFICATION  (ref: mid-range full transit -> 50% detection)")
print("=" * 78)

print("\n[A] Planned-path Phi (full precision) and Rtil, per scenario/algorithm:")
print(f"  {'scenario':15s} {'alg':6s} {'Phi':>12s} {'Rtil':>10s}  {'C':>9s}")
store = {}
for sc in MAIN:
    cfg, t, r, s, g = build(sc)
    store[sc] = (cfg, t, r, s, g)
    paths = run_all(cfg, t, r, s, g)
    for n, _ in FN:
        cr = costmod.evaluate(cfg, r, t, paths[n].path)
        print(f"  {sc:15s} {LAB[n]:6s} {cr.Phi:12.4e} {cr.Rtil:10.5f}  {r._radar_C.max():9.2e}")
    store[sc] = (cfg, t, r, s, g, paths)

print("\n[B] Candidate-population fraction in informative band 0.05<Rtil<0.95:")
figP, axP = plt.subplots(2, 2, figsize=(11, 7))
for i, sc in enumerate(MAIN):
    cfg, t, r, s, g, _ = store[sc]
    rt = np.array([costmod.evaluate(cfg, r, t, P).Rtil for P in candidates(s, g, cfg.world_size)])
    lo = (rt <= 0.05).mean() * 100
    mid = ((rt > 0.05) & (rt < 0.95)).mean() * 100
    hi = (rt >= 0.95).mean() * 100
    print(f"  {sc:15s}: <=0.05: {lo:5.1f}%   band: {mid:5.1f}%   >=0.95: {hi:5.1f}%")
    ax = axP[i // 2, i % 2]
    ax.hist(rt, bins=np.linspace(0, 1, 41), color="#3366aa")
    ax.axvspan(0.05, 0.95, color="green", alpha=0.10)
    ax.set_title(f"{sc}: Rtil population (band={mid:.0f}%)")
    ax.set_xlabel("Rtil")
    ax.set_ylabel("count")
figP.tight_layout()
figP.savefig("outputs/calib_rtil_population.png", dpi=130)
plt.close(figP)

print("\n[C] Pareto sweep (A*) on main scenarios: wR vs (L, Phi, Rtil):")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for k, sc in enumerate(["ridge_corridor", "high_density"]):
    cfg, t, r, s, g, _ = store[sc]
    H = r.grid()
    wK = cfg.wK
    rows = []
    print(f"\n  --- {sc} ---")
    for wR in np.linspace(0.02, 0.85, 12):
        wL = 1 - wK - wR
        if wL < 0:
            continue
        c2 = replace(cfg, wL=wL, wR=wR, wK=wK)
        res = plan_astar(c2, t, r, s, g, hazard_grid=H)
        cr = costmod.evaluate(c2, r, t, res.path)
        rows.append((wR, cr.L, cr.Phi, cr.Rtil))
        print(f"    wR={wR:4.2f}  L={cr.L:6.0f}  Phi={cr.Phi:9.3e}  Rtil={cr.Rtil:7.4f}")
    rows = np.array(rows)
    band = int(((rows[:, 3] > 0.05) & (rows[:, 3] < 0.95)).sum())
    print(
        f"    -> {band}/{len(rows)} settings in informative band; "
        f"{'GRADED front' if band >= 3 else 'still degenerate'}"
    )
    sc0 = axes[k].scatter(rows[:, 1], rows[:, 3], c=rows[:, 0], cmap="viridis", s=60, edgecolor="k")
    axes[k].plot(rows[:, 1], rows[:, 3], "-", color="gray", alpha=0.5)
    axes[k].axhspan(0.05, 0.95, color="green", alpha=0.08)
    axes[k].set_xlabel("L (m)")
    axes[k].set_ylabel("Rtil")
    axes[k].set_ylim(-0.05, 1.05)
    axes[k].grid(alpha=0.3)
    axes[k].set_title(f"{sc}  (calibrated)")
    fig.colorbar(sc0, ax=axes[k], label="wR")
fig.suptitle("Pareto after C-calibration: L vs Rtil (green = informative band)")
fig.tight_layout()
fig.savefig("outputs/calib_pareto_main.png", dpi=130)
plt.close(fig)

print("\n[D] Prior-fix regression checks:")
cfg, t, r, s, g, paths = store["ridge_corridor"]
cra: object = costmod.evaluate(cfg, r, t, paths["astar"].path)
print(
    f"  ridge A* detours: climb_viol={cra.climb_violations}, "
    f"kappa_peak={cra.kappa_peak:.4f} (> kappa_max={cfg.kappa_max:.4f}? "
    f"{cra.kappa_peak > cfg.kappa_max}) -> discriminates"
)
tp = store["concave_trap"][5]["apf"]
print(
    f"  concave_trap APF: success={tp.success} reason={tp.info.get('reason')} "
    f"-> expect False/local_minimum"
)
op = store["open"][5]
for n, _ in FN:
    c = costmod.evaluate(store["open"][0], store["open"][2], store["open"][1], op[n].path)
    print(
        f"  open {LAB[n]:6s}: gamma_peak={np.degrees(c.gamma_peak):.1f} deg "
        f"climb_viol={c.climb_violations} (expect compliant)"
    )
print("\nsaved: outputs/calib_rtil_population.png, outputs/calib_pareto_main.png")
