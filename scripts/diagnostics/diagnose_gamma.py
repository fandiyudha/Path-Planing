"""Diagnostic (not used for paper results): find the source of climb/dive (gamma) violations.

This script ADDS no behaviour to the simulation and modifies nothing — it
imports the existing terrain / risk / planners / cost and only *measures* and
*prints*.  It tests three candidate root causes for paths whose flight-path
angle gamma = atan2(dz, ds_horizontal) exceeds gamma_max:

  (A) terrain is steeper than gamma_max (any terrain-hugging path must violate),
  (B) the gamma penalty inside J is too weak to deter violations,
  (C) waypoint spacing is so fine that dz/ds inflates the per-step angle.

Run (from the repository root):  python -m scripts.diagnostics.diagnose_gamma
Writes outputs/diag_terrain_slope.png and prints a full report.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_pathplan import cost as costmod
from uav_pathplan import scenarios as scn
from uav_pathplan.config import Config
from uav_pathplan.cost import wrap_angle
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.hybrid import plan_hybrid
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField

SEED = 20240603
RESOLUTION = 5.0  # finest (Config default): worst case for (C)
PLANNERS = [("astar", plan_astar), ("apf", plan_apf), ("hybrid", plan_hybrid)]


# --------------------------------------------------------------------------- #
def dist(name, arr, unit=""):
    """Print a min/median/mean/p90/p95/max summary line for an array."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        print(f"    {name:24s}: (empty)")
        return
    print(
        f"    {name:24s}: min={a.min():7.2f}  med={np.median(a):7.2f}  "
        f"mean={a.mean():7.2f}  p90={np.percentile(a, 90):7.2f}  "
        f"p95={np.percentile(a, 95):7.2f}  max={a.max():7.2f} {unit}"
    )


def build(scenario, resolution):
    spec = scn.build_scenario(scenario, master_seed=SEED, run_idx=0, randomize=False)
    cfg = replace(spec.cfg, seed=SEED, resolution=resolution)
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    risk.grid()
    return cfg, terrain, risk, spec.start, spec.goal


def terrain_slope_deg(cfg, terrain):
    """Per-cell terrain slope in degrees using the true grid spacing."""
    gy, gx = np.gradient(terrain.h, cfg.resolution)  # d/dy, d/dx (m/m)
    mag = np.hypot(gx, gy)
    return np.degrees(np.arctan(mag))


def path_gammas_deg(cfg, terrain, P):
    """Per-step |gamma| (deg) and horizontal step lengths for a polyline.

    Replicates the cost function's definition: gamma = atan2(dz, ds_horizontal),
    z from the nap-of-the-earth altitude policy.
    """
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 2:
        return np.array([]), np.array([])
    d = np.diff(P, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    z = np.asarray(terrain.altitude_at(P[:, 0], P[:, 1]), dtype=float)
    dz = np.diff(z)
    gamma = np.degrees(np.arctan2(np.abs(dz), np.maximum(ds, 1e-9)))
    return gamma, ds


def resample_polyline(P, spacing):
    """Post-hoc uniform arc-length resample of a polyline (does not re-plan)."""
    P = np.asarray(P, dtype=float)
    # drop zero-length segments so the arc-length axis is strictly increasing
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) > 1e-9:
            keep.append(i)
    P = P[keep]
    if len(P) < 2:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(2, int(round(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def ktil_split(cfg, terrain, P):
    """Replicate cost.Ktil and split it into curvature vs gamma contributions.

    Returns (Ktil, mean_kappa_term, mean_gamma_term) using the exact formulas in
    cost.evaluate, so the two means sum to Ktil.
    """
    P = np.asarray(P, dtype=float)
    d = np.diff(P, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    psi = np.arctan2(d[:, 1], d[:, 0])
    z = np.asarray(terrain.altitude_at(P[:, 0], P[:, 1]), dtype=float)
    dz = np.diff(z)
    gamma_abs = np.abs(np.arctan2(dz, np.maximum(ds, 1e-9)))
    N = ds.size
    if N < 2:
        return 0.0, 0.0, 0.0
    dpsi = np.abs(wrap_angle(psi[1:] - psi[:-1]))
    sbar = 0.5 * (ds[:-1] + ds[1:])
    kappa = dpsi / np.maximum(sbar, 1e-9)
    kappa_pen = np.maximum(0.0, (kappa - cfg.kappa_max) / cfg.kappa_max)
    gamma_pen = np.maximum(0.0, (gamma_abs[1:] - cfg.gamma_max) / cfg.gamma_max)
    return (
        float(np.mean(kappa_pen + gamma_pen)),
        float(np.mean(kappa_pen)),
        float(np.mean(gamma_pen)),
    )


def run_paths(cfg, terrain, risk, start, goal):
    H = risk.grid()
    out = {}
    for name, fn in PLANNERS:
        kw = {"hazard_grid": H} if fn is plan_astar else {}
        out[name] = fn(cfg, terrain, risk, start, goal, **kw)
    return out


# --------------------------------------------------------------------------- #
def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    gmax = 15.0
    print("=" * 78)
    print(
        f"GAMMA VIOLATION DIAGNOSIS  (seed={SEED}, resolution={RESOLUTION} m, "
        f"gamma_max={gmax:.0f} deg, clearance=50 m)"
    )
    print("MEASUREMENT ONLY — nothing is modified or fixed.")
    print("=" * 78)

    # quick slope comparison across all scenarios to pick the roughest
    print("\n[0] Terrain-slope quick scan across scenarios (p95 / %cells>gmax):")
    rough = {}
    for s in scn.SCENARIO_BUILDERS:
        cfg, terr, _, _, _ = build(s, RESOLUTION)
        sl = terrain_slope_deg(cfg, terr)
        frac = float((sl > gmax).mean()) * 100
        rough[s] = (float(np.percentile(sl, 95)), frac)
        print(f"    {s:16s}: p95={rough[s][0]:5.1f} deg   cells>gmax={frac:5.1f}%")
    rough_scn = max([s for s in rough if s != "open"], key=lambda s: rough[s][1])
    scenarios = ["open", rough_scn]
    print(f"    -> detailed report on: open  and  {rough_scn} (roughest non-open)")

    for scenario in scenarios:
        cfg, terrain, risk, start, goal = build(scenario, RESOLUTION)
        print("\n" + "#" * 78)
        print(f"# SCENARIO: {scenario}")
        print("#" * 78)

        # ---- Section 1: terrain slope (cause A) ---------------------------
        slope = terrain_slope_deg(cfg, terrain)
        frac_over = float((slope > gmax).mean()) * 100
        print("\n[1] TERRAIN SLOPE ITSELF  (tests cause A: terrain too steep)")
        dist("terrain slope (deg)", slope, "deg")
        print(
            f"    fraction of grid cells with slope > gamma_max({gmax:.0f}deg) = {frac_over:.1f}%"
        )
        print(
            "    HINT: if this fraction is large, no terrain-hugging path can "
            "satisfy the limit (cause A)."
        )

        if scenario == "open":
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.hist(slope.ravel(), bins=80, color="#4477aa", alpha=0.85)
            ax.axvline(gmax, color="red", ls="--", lw=2, label=f"gamma_max = {gmax:.0f} deg")
            ax.set_xlabel("terrain slope (deg)")
            ax.set_ylabel("grid-cell count")
            ax.set_title(
                f"Terrain slope distribution — {scenario} "
                f"(seed {SEED}, {cfg.resolution:.0f} m grid)"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "diag_terrain_slope.png", dpi=130)
            plt.close(fig)
            print(f"    saved histogram: {out / 'diag_terrain_slope.png'}")

        # ---- run the three planners once ----------------------------------
        paths = run_paths(cfg, terrain, risk, start, goal)

        # ---- Section 2: spacing & angle, with resampling (cause C) --------
        print("\n[2] PER-STEP SPACING & ANGLE  (tests cause C: spacing too fine)")
        print(
            f"    grid resolution = {cfg.resolution:.1f} m;  APF/Hybrid step "
            f"apf_alpha = {cfg.apf_alpha:.1f} m"
        )
        for name in ("astar", "apf", "hybrid"):
            res = paths[name]
            P = res.path
            gamma, ds = path_gammas_deg(cfg, terrain, P)
            if gamma.size == 0:
                print(f"  -- {name}: <2 waypoints (success={res.success})")
                continue
            nviol = int((gamma > gmax).sum())
            print(f"  -- {name}  (success={res.success}, waypoints={len(P)})")
            dist("ds_horizontal (m)", ds, "m")
            dist("per-step |gamma| (deg)", gamma, "deg")
            print(
                f"      steps over gamma_max: {nviol}/{gamma.size} "
                f"({100 * nviol / gamma.size:.1f}%),  peak={gamma.max():.1f} deg"
            )
            for factor in (2, 5):
                Q = resample_polyline(P, factor * cfg.resolution)
                gq, _ = path_gammas_deg(cfg, terrain, Q)
                if gq.size:
                    nq = int((gq > gmax).sum())
                    print(
                        f"      resampled @ {factor}x res "
                        f"({factor * cfg.resolution:.0f} m): "
                        f"peak={gq.max():5.1f} deg, "
                        f"viol={nq}/{gq.size} ({100 * nq / gq.size:.1f}%)"
                    )
        print(
            "    HINT: if violations shrink sharply under coarser resampling "
            "AND terrain slope is gentle, cause C (discretization) matters."
        )

        # ---- Section 3: penalty magnitude inside J (cause B) --------------
        print("\n[3] COST COMPONENT MAGNITUDES  (tests cause B: weak penalty)")
        # representative path = the successful one with the worst gamma peak
        cand = [(n, p) for n, p in paths.items() if p.success and p.n_waypoints > 2]
        if not cand:
            cand = [(n, p) for n, p in paths.items() if p.n_waypoints > 2]
        rep_name, rep = max(
            cand,
            key=lambda np_: (
                path_gammas_deg(cfg, terrain, np_[1].path)[0].max()
                if path_gammas_deg(cfg, terrain, np_[1].path)[0].size
                else 0.0
            ),
        )
        cr = costmod.evaluate(cfg, risk, terrain, rep.path)
        Ktil, kterm, gterm = ktil_split(cfg, terrain, rep.path)
        cL, cR, cK = cfg.wL * cr.Ltil, cfg.wR * cr.Rtil, cfg.wK * Ktil
        cK_kappa, cK_gamma = cfg.wK * kterm, cfg.wK * gterm
        J = cr.J
        print(
            f"    representative path: {rep_name}  (J={J:.4f}, "
            f"gamma_peak={np.degrees(cr.gamma_peak):.1f} deg, "
            f"climb_violations={cr.climb_violations})"
        )
        print(f"    weights: wL={cfg.wL}, wR={cfg.wR}, wK={cfg.wK}")
        print(f"    wL*Ltil = {cL:.4f}   ({100 * cL / J:5.1f}% of J)")
        print(f"    wR*Rtil = {cR:.4f}   ({100 * cR / J:5.1f}% of J)")
        print(f"    wK*Ktil = {cK:.4f}   ({100 * cK / J:5.1f}% of J)")
        print("        within wK*Ktil:")
        print(
            f"          curvature term wK*mean(kappa_pen) = {cK_kappa:.4f} "
            f"({100 * cK_kappa / J:5.1f}% of J)"
        )
        print(
            f"          gamma term     wK*mean(gamma_pen) = {cK_gamma:.4f} "
            f"({100 * cK_gamma / J:5.1f}% of J)"
        )
        gfrac = 100 * cK_gamma / J if J > 0 else 0.0
        print(f"    GAMMA-violation term is {gfrac:.1f}% of total J.")
        print(
            "    HINT: if this is a tiny fraction, the planner has little "
            "incentive to avoid climbs (cause B)."
        )

        # stash for verdict
        if scenario == "open":
            verdict_data = {
                "frac_over": frac_over,
                "slope_p95": float(np.percentile(slope, 95)),
                "gfrac": gfrac,
                "resample": {},
            }
            for name in ("apf", "hybrid", "astar"):
                g0, _ = path_gammas_deg(cfg, terrain, paths[name].path)
                if g0.size:
                    Q = resample_polyline(paths[name].path, 5 * cfg.resolution)
                    gq, _ = path_gammas_deg(cfg, terrain, Q)
                    verdict_data["resample"][name] = (
                        float(g0.max()),
                        float(gq.max()) if gq.size else float("nan"),
                        int((g0 > gmax).sum()),
                        int((gq > gmax).sum()) if gq.size else 0,
                    )

    # ---- Section 4: verdict (open scenario evidence) ----------------------
    print("\n" + "=" * 78)
    print("[4] SUMMARY VERDICT  (open scenario; based purely on the numbers)")
    print("=" * 78)
    vd = verdict_data
    A = (vd["frac_over"] >= 5.0) or (vd["slope_p95"] >= gmax)
    # cause C: coarse resampling removes most violations
    c_evidence = []
    for name, (p0, p5, n0, n5) in vd["resample"].items():
        if n0 > 0:
            red = 100 * (1 - n5 / n0)
            c_evidence.append((name, p0, p5, red))
    C = any(red >= 50 and p5 < p0 - 3 for _, p0, p5, red in c_evidence)
    B = vd["gfrac"] < 3.0

    print(
        f"  (A) terrain too steep : {'YES' if A else 'no'}  "
        f"(p95 slope={vd['slope_p95']:.1f} deg, "
        f"{vd['frac_over']:.1f}% of cells > {gmax:.0f} deg)"
    )
    print(
        f"  (B) penalty too weak  : {'YES' if B else 'no'}  "
        f"(gamma term = {vd['gfrac']:.1f}% of J at wK={Config().wK})"
    )
    cdesc = (
        ", ".join(
            f"{n}: {p0:.0f}->{p5:.0f}deg ({red:.0f}% fewer viol)" for n, p0, p5, red in c_evidence
        )
        or "n/a"
    )
    print(f"  (C) spacing too fine  : {'YES' if C else 'no'}  (peak @5x resample — {cdesc})")
    causes = [c for c, on in [("A", A), ("B", B), ("C", C)] if on]
    print(f"\n  EVIDENCE SUPPORTS: {', '.join(causes) if causes else 'none clearly'}")
    print("  (No fix applied — awaiting decision.)")


if __name__ == "__main__":
    main()
