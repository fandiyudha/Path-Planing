"""Diagnosis-only: test two suspected side-effects of the Option-1 gamma fix.

  Suspicion 1 — the uniform arc-length resampling in cost.evaluate flattened the
                curvature / smoothness discriminator (A* jagged vs APF smooth).
  Suspicion 2 — Rtil = 1 - exp(-Phi) has saturated into a binary 0/1 flag,
                destroying graded risk and the Pareto front.

This script imports the existing modules and only MEASURES + PRINTS + PLOTS.
It changes no planner, cost, terrain, risk, or parameter.

Run:  python diagnose_curv_risk.py
Writes: outputs/diag_curv_sweep.png, diag_loglambda_hist.png,
        diag_rtil_population.png, diag_pareto_mini.png
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_pathplan import scenarios as scn
from uav_pathplan import cost as costmod
from uav_pathplan.cost import wrap_angle, resample_polyline
from uav_pathplan.terrain import TerrainField
from uav_pathplan.risk import RiskField
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.hybrid import plan_hybrid

RES = 8.0
FN = [("astar", plan_astar), ("apf", plan_apf), ("hybrid", plan_hybrid)]
LABEL = {"astar": "A*", "apf": "APF", "hybrid": "Hybrid"}


def build(sc):
    spec = scn.build_scenario(sc)
    cfg = replace(spec.cfg, resolution=RES)
    t = TerrainField.generate(cfg)
    r = RiskField.build(cfg, t)
    r.grid()
    return cfg, t, r, spec.start, spec.goal


def run_all(cfg, t, r, s, g):
    H = r.grid()
    out = {}
    for name, fn in FN:
        kw = {"hazard_grid": H} if fn is plan_astar else {}
        out[name] = fn(cfg, t, r, s, g, **kw)
    return out


def curv_metrics(P, kappa_max):
    """(kappa_peak, curv_violations, smoothness_sum|dpsi|_rad, |dpsi|_deg array)."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 3:
        return 0.0, 0, 0.0, np.array([])
    d = np.diff(P, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    psi = np.arctan2(d[:, 1], d[:, 0])
    dpsi = np.abs(wrap_angle(psi[1:] - psi[:-1]))
    sbar = 0.5 * (ds[:-1] + ds[1:])
    kappa = dpsi / np.maximum(sbar, 1e-9)
    return (float(kappa.max()), int((kappa > kappa_max).sum()),
            float(dpsi.sum()), np.degrees(dpsi))


def qstats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "(empty)"
    return (f"min={a.min():6.2f} med={np.median(a):6.2f} mean={a.mean():6.2f} "
            f"p90={np.percentile(a,90):6.2f} max={a.max():6.2f}")


# =========================================================================== #
# SUSPICION 1 — curvature / smoothness discriminator                          #
# =========================================================================== #
def suspicion1():
    print("=" * 80)
    print("SUSPICION 1 — did arc-length resampling erase the curvature discriminator?")
    print(f"(resolution={RES} m, kappa_max=1/R_min)")
    print("=" * 80)
    spacings = [("raw", None), ("0.5x", 0.5 * RES), ("1x", 1.0 * RES),
                ("2x", 2.0 * RES), ("4x", 4.0 * RES)]
    scen_paths = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, sc in enumerate(("open", "ridge_corridor")):
        cfg, t, r, s, g = build(sc)
        kmax = cfg.kappa_max
        paths = run_all(cfg, t, r, s, g)
        scen_paths[sc] = (cfg, paths)
        print(f"\n### {sc}   (kappa_max={kmax:.5f} 1/m)")
        print(f"  {'algo':7s} {'spacing':7s} {'kappa_peak':>11s} "
              f"{'curv_viol':>9s} {'smoothness':>10s}  (smoothness = sum|dpsi| rad)")
        sm_by_algo = {n: [] for n, _ in FN}
        kp_by_algo = {n: [] for n, _ in FN}
        for name, _ in FN:
            P = np.asarray(paths[name].path)
            for tag, sp in spacings:
                Q = P if sp is None else resample_polyline(P, sp)
                kp, cv, sm, _ = curv_metrics(Q, kmax)
                print(f"  {LABEL[name]:7s} {tag:7s} {kp:11.5f} {cv:9d} {sm:10.3f}")
                if sp is not None:
                    sm_by_algo[name].append(sm)
                    kp_by_algo[name].append(kp)
            print()
        # per-waypoint |dpsi| distribution raw vs resampled(1x)
        print(f"  per-waypoint |dpsi| (deg) distribution, RAW vs RESAMPLED@1x ({RES:.0f} m):")
        for name, _ in FN:
            P = np.asarray(paths[name].path)
            _, _, _, draw = curv_metrics(P, kmax)
            _, _, _, dres = curv_metrics(resample_polyline(P, RES), kmax)
            print(f"    {LABEL[name]:7s} raw:       {qstats(draw)}")
            print(f"    {LABEL[name]:7s} resampled: {qstats(dres)}")
        # plots
        xs = [sp for _, sp in spacings if sp is not None]
        for name, _ in FN:
            axes[row, 0].plot(xs, sm_by_algo[name], "o-", label=LABEL[name])
            axes[row, 1].plot(xs, kp_by_algo[name], "o-", label=LABEL[name])
        axes[row, 0].set_title(f"{sc}: smoothness vs resample spacing")
        axes[row, 0].set_ylabel("smoothness sum|dpsi| (rad)")
        axes[row, 1].set_title(f"{sc}: kappa_peak vs resample spacing")
        axes[row, 1].axhline(cfg.kappa_max, color="r", ls="--", lw=1,
                             label="kappa_max")
        axes[row, 1].set_ylabel("kappa_peak (1/m)")
        for c in (0, 1):
            axes[row, c].set_xlabel("resample spacing (m)")
            axes[row, c].grid(alpha=0.3); axes[row, c].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("outputs/diag_curv_sweep.png", dpi=130)
    plt.close(fig)
    print("\n  saved: outputs/diag_curv_sweep.png")

    # verdict
    print("\n  --- VERDICT (Suspicion 1) ---")
    for sc in ("open", "ridge_corridor"):
        cfg, paths = scen_paths[sc]
        raw = {n: curv_metrics(np.asarray(paths[n].path), cfg.kappa_max) for n, _ in FN}
        res1 = {n: curv_metrics(resample_polyline(np.asarray(paths[n].path), RES),
                                cfg.kappa_max) for n, _ in FN}
        sm_raw = {n: raw[n][2] for n in raw}
        sm_res = {n: res1[n][2] for n in res1}
        sep_raw = max(sm_raw.values()) - min(sm_raw.values())
        sep_res = max(sm_res.values()) - min(sm_res.values())
        print(f"  [{sc}] smoothness spread across algos: RAW={sep_raw:.2f} rad "
              f"-> RESAMPLED@1x={sep_res:.2f} rad "
              f"(A* raw={sm_raw['astar']:.2f}, res={sm_res['astar']:.2f})")


# =========================================================================== #
# SUSPICION 2 — Rtil saturation                                               #
# =========================================================================== #
def make_candidates(start, goal, world, n=121):
    """Cheap population: smooth single-bulge arcs with varying lateral offset."""
    s = np.array(start, float); g = np.array(goal, float)
    d = g - s
    perp = np.array([-d[1], d[0]]); perp = perp / (np.hypot(*perp) + 1e-9)
    tt = np.linspace(0, 1, 80)
    base = s[None, :] + tt[:, None] * d[None, :]
    cands = []
    for off in np.linspace(-0.5 * world, 0.5 * world, n):
        bulge = off * np.sin(np.pi * tt)
        P = base + bulge[:, None] * perp[None, :]
        P[:, 0] = np.clip(P[:, 0], 0, world); P[:, 1] = np.clip(P[:, 1], 0, world)
        cands.append(P)
    return cands


def suspicion2():
    print("\n" + "=" * 80)
    print("SUSPICION 2 — is Rtil = 1 - exp(-Phi) graded, or a binary 0/1 flag?")
    print("=" * 80)
    scenarios = list(scn.SCENARIO_BUILDERS)

    # (1) hazard field dynamic range + (3) population Rtil
    figL, axesL = plt.subplots(2, 2, figsize=(12, 8))
    figR, axesR = plt.subplots(2, 2, figsize=(12, 8))
    print("\n[1] Hazard field Lambda dynamic range (nonzero grid cells):")
    fields = {}
    for i, sc in enumerate(scenarios):
        cfg, t, r, s, g = build(sc)
        fields[sc] = (cfg, t, r, s, g)
        H = r.grid(); nz = H[H > 0]
        med = np.median(nz) if nz.size else np.nan
        print(f"  {sc:15s}: min={nz.min():.2e} p50={med:.2e} "
              f"p90={np.percentile(nz,90):.2e} p99={np.percentile(nz,99):.2e} "
              f"max={nz.max():.2e}  max/median={nz.max()/med:.1e}")
        ax = axesL[i // 2, i % 2]
        ax.hist(np.log10(nz), bins=60, color="#aa5533")
        ax.set_title(f"{sc}: log10(Lambda) over grid")
        ax.set_xlabel("log10(Lambda)"); ax.set_ylabel("cells")
    figL.tight_layout(); figL.savefig("outputs/diag_loglambda_hist.png", dpi=130)
    plt.close(figL)

    print("\n[2] Phi and Rtil for the actual planned paths (full precision):")
    print(f"  {'scenario':15s} {'algo':7s} {'Phi':>12s} {'Rtil':>12s}")
    for sc in scenarios:
        cfg, t, r, s, g = fields[sc]
        paths = run_all(cfg, t, r, s, g)
        for name, _ in FN:
            cr = costmod.evaluate(cfg, r, t, paths[name].path)
            print(f"  {sc:15s} {LABEL[name]:7s} {cr.Phi:12.4e} {cr.Rtil:12.6f}")

    print("\n[3] Population of candidate paths — fraction in informative band "
          "0.05 < Rtil < 0.95:")
    for i, sc in enumerate(scenarios):
        cfg, t, r, s, g = fields[sc]
        cands = make_candidates(s, g, cfg.world_size, n=121)
        rt = np.array([costmod.evaluate(cfg, r, t, P).Rtil for P in cands])
        lo = float((rt <= 0.05).mean()) * 100
        mid = float(((rt > 0.05) & (rt < 0.95)).mean()) * 100
        hi = float((rt >= 0.95).mean()) * 100
        print(f"  {sc:15s}: Rtil<=0.05: {lo:5.1f}%   0.05<mid<0.95: {mid:5.1f}%   "
              f"Rtil>=0.95: {hi:5.1f}%   (n={len(cands)})")
        ax = axesR[i // 2, i % 2]
        ax.hist(rt, bins=np.linspace(0, 1, 41), color="#3366aa")
        ax.axvspan(0.05, 0.95, color="green", alpha=0.10)
        ax.set_title(f"{sc}: Rtil over candidate population")
        ax.set_xlabel("Rtil"); ax.set_ylabel("candidates")
    figR.tight_layout(); figR.savefig("outputs/diag_rtil_population.png", dpi=130)
    plt.close(figR)

    # (4) mini Pareto sweep on three contrasting scenarios
    print("\n[4] Mini Pareto sweep (A*): wR vs (L, Phi, Rtil).")
    print("    Contrast: strong-radar main scenarios vs the dedicated weak-radar "
          "'pareto' scenario.")

    def build_named(name):
        spec = scn.build_pareto() if name == "pareto" else scn.build_scenario(name)
        cfg = replace(spec.cfg, resolution=RES)
        t = TerrainField.generate(cfg); r = RiskField.build(cfg, t); r.grid()
        # effective (possibly calibrated) radar power constant, from the field
        C_eff = float(r._radar_C.max()) if len(r._radar_C) else 0.0
        return cfg, t, r, spec.start, spec.goal, C_eff

    sweep_scens = ["ridge_corridor", "high_density", "pareto"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, name in enumerate(sweep_scens):
        cfg, t, r, s, g, C = build_named(name)
        H = r.grid(); wK = cfg.wK
        rows = []
        print(f"\n  --- {name}  (radar C={C:.1e}) ---")
        for wR in np.linspace(0.02, 0.83, 12):
            wL = 1.0 - wK - wR
            if wL < 0:
                continue
            c2 = replace(cfg, wL=wL, wR=wR, wK=wK)
            res = plan_astar(c2, t, r, s, g, hazard_grid=H)
            cr = costmod.evaluate(c2, r, t, res.path)
            rows.append((wR, cr.L, cr.Phi, cr.Rtil))
            print(f"    wR={wR:4.2f}  L={cr.L:6.0f} m   Phi={cr.Phi:10.3e}   "
                  f"Rtil={cr.Rtil:8.5f}")
        rows = np.array(rows)
        rt = rows[:, 3]
        mid_ct = int(((rt > 0.05) & (rt < 0.95)).sum())
        shape = ("SMOOTH graded curve" if mid_ct >= 3
                 else "DEGENERATE (no graded middle)")
        print(f"    Rtil in band 0.05..0.95: {mid_ct}/{len(rt)}  -> {shape}")
        sc0 = axes[k].scatter(rows[:, 1], rows[:, 3], c=rows[:, 0],
                              cmap="viridis", s=60, edgecolor="k")
        axes[k].plot(rows[:, 1], rows[:, 3], "-", color="gray", alpha=0.5)
        axes[k].axhspan(0.05, 0.95, color="green", alpha=0.08)
        axes[k].set_xlabel("L (m)"); axes[k].set_ylabel("Rtil")
        axes[k].set_ylim(-0.05, 1.05); axes[k].grid(alpha=0.3)
        axes[k].set_title(f"{name}\n(radar C={C:.0e})")
        fig.colorbar(sc0, ax=axes[k], label="wR")
    fig.suptitle("Mini Pareto: L vs Rtil sweeping wR (green band = informative)")
    fig.tight_layout(); fig.savefig("outputs/diag_pareto_mini.png", dpi=130)
    plt.close(fig)
    print("\n  saved: outputs/diag_loglambda_hist.png, diag_rtil_population.png, "
          "diag_pareto_mini.png")


if __name__ == "__main__":
    suspicion1()
    suspicion2()
