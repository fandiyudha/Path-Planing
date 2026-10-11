"""Top-level experiment driver.

Runs the Monte-Carlo comparison of A*, APF, and Hybrid A*-APF across all
scenario classes, writes per-run and aggregated CSVs, performs paired Wilcoxon
tests, and produces every figure.  Everything is seeded and reproducible; the
master seed is printed and saved with the outputs.

Usage
-----
    python run_experiments.py                 # default: 10 m grid, 10 MC runs
    python run_experiments.py --quick         # fast sanity pass
    python run_experiments.py --full          # 5 m grid, 20 MC runs (slow)
    python run_experiments.py --runs 15 --resolution 8 --seed 7
    python run_experiments.py --mem           # also record peak Python memory
    python run_experiments.py --format tiff --dpi 600   # figure format/resolution

Paper settings:  python run_experiments.py --runs 20 --resolution 8 --seed 20240603

Reproduce a single figure: the per-scenario overlay/3D and the Pareto and
time-vs-threats sweeps are written to the output directory (default ./outputs).

Protocol v2 (settings in configs/protocol_v2.yaml; default stays v1):
    python run_experiments.py --protocol v2                  # held-out evaluation seed
    python run_experiments.py --protocol v2 --seed-set dev   # development seed
    python run_experiments.py --protocol v2 --quick          # smoke run (dev seed)
"""

from __future__ import annotations

import os
import sys


def _wants_v2(argv) -> bool:
    for i, a in enumerate(argv):
        if a == "--protocol":
            return i + 1 < len(argv) and argv[i + 1] == "v2"
        if a.startswith("--protocol="):
            return a == "--protocol=v2"
    return False


# Protocol v2 times single-threaded numpy; the thread variables must be set
# before numpy is imported.  v1 runs are unaffected.
if _wants_v2(sys.argv):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
import tracemalloc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from uav_pathplan import plotting
from uav_pathplan import scenarios as scn
from uav_pathplan.config import Config
from uav_pathplan.metrics import RunMetrics, aggregate, compute_metrics
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.hybrid import plan_hybrid
from uav_pathplan.risk import RiskField
from uav_pathplan.stats import paired_tests, tests_to_frame
from uav_pathplan.terrain import TerrainField

SCENARIOS = ["open", "concave_trap", "ridge_corridor", "high_density"]


def _exp_overrides(cfg: Config, resolution: float) -> Config:
    """Apply experiment-wide overrides (grid resolution) to a scenario config."""
    return replace(cfg, resolution=resolution)


def _build(spec, resolution):
    """Build terrain + risk for a spec and precompute the hazard grid (timed)."""
    cfg = _exp_overrides(spec.cfg, resolution)
    spec = replace(spec, cfg=cfg)
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    t0 = time.perf_counter()
    H = risk.grid()  # exact-LOS precompute, done once
    setup_s = time.perf_counter() - t0
    return spec, cfg, terrain, risk, H, setup_s


def _run_planner(fn, cfg, terrain, risk, start, goal, H, measure_mem):
    """Run one planner, optionally measuring peak Python memory."""
    kwargs = {"hazard_grid": H} if fn is plan_astar else {}
    if measure_mem:
        tracemalloc.start()
        res = fn(cfg, terrain, risk, start, goal, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return res, peak / 1024.0
    return fn(cfg, terrain, risk, start, goal, **kwargs), np.nan


PLANNERS = [("astar", plan_astar), ("apf", plan_apf), ("hybrid", plan_hybrid)]


def monte_carlo(args) -> list[RunMetrics]:
    """Run the full Monte-Carlo sweep; return all per-run metric rows."""
    rows: list[RunMetrics] = []
    for scenario in args.scenarios:
        print(f"\n=== scenario: {scenario} ===")
        for run in range(args.runs):
            spec = scn.build_scenario(scenario, master_seed=args.seed, run_idx=run, randomize=True)
            spec, cfg, terrain, risk, H, setup_s = _build(spec, args.resolution)
            line = f"  run {run:2d} (terrain seed {cfg.seed}, setup {setup_s:.1f}s): "
            for name, fn in PLANNERS:
                res, mem = _run_planner(fn, cfg, terrain, risk, spec.start, spec.goal, H, args.mem)
                m = compute_metrics(cfg, terrain, risk, scenario, name, run, res, peak_mem_kib=mem)
                rows.append(m)
                line += f"{name}={'OK' if res.success else 'x'}({res.time_s:.1f}s) "
            print(line)
    return rows


def make_overlays(args, out: Path):
    """Canonical (deterministic) overlay + 3D figures per scenario."""
    for scenario in args.scenarios:
        spec = scn.build_scenario(scenario, master_seed=args.seed, run_idx=0, randomize=False)
        spec, cfg, terrain, risk, H, _ = _build(spec, args.resolution)
        results, mrows = {}, {}
        for name, fn in PLANNERS:
            res, _ = _run_planner(fn, cfg, terrain, risk, spec.start, spec.goal, H, False)
            results[name] = res
            # metric values for the legend (read from the run record)
            mrows[name] = compute_metrics(cfg, terrain, risk, scenario, name, 0, res).as_dict()
        # No in-figure titles: the scenario is identified in the paper caption.
        plotting.plot_overlay(
            cfg,
            terrain,
            risk,
            spec.start,
            spec.goal,
            results,
            out / f"overlay_{scenario}.{args.format}",
            "",
            metrics=mrows,
        )
        # 3D surface with EVERY attempted path (incl. failed APF stub + marker)
        plotting.plot_surface_with_paths(
            cfg, terrain, results, out / f"surface_{scenario}.{args.format}", "", metrics=mrows
        )
        print(f"  overlay+surface saved for {scenario}")


def time_vs_threats(args, out: Path):
    """Sweep radar count; record A*/Hybrid compute time and expanded nodes."""
    counts = [2, 4, 8, 12, 16, 20] if not args.quick else [2, 6, 12]
    times = {"astar": [], "hybrid": []}
    nexp = {"astar": [], "hybrid": []}
    print("\n=== time vs threat count ===")
    for nr in counts:
        spec = scn.build_high_density(
            master_seed=args.seed, run_idx=0, randomize=False, n_radars=nr
        )
        spec, cfg, terrain, risk, H, _ = _build(spec, args.resolution)
        for name, fn in [("astar", plan_astar), ("hybrid", plan_hybrid)]:
            res, _ = _run_planner(fn, cfg, terrain, risk, spec.start, spec.goal, H, False)
            times[name].append(res.time_s)
            nexp[name].append(res.n_exp)
        print(
            f"  radars={nr:2d}: A*={times['astar'][-1]:.2f}s "
            f"(n_exp={nexp['astar'][-1]}), Hybrid={times['hybrid'][-1]:.2f}s"
        )
    plotting.plot_time_vs_threats(
        counts, times, out / f"time_vs_threats.{args.format}", nexp_by_alg=nexp
    )


def pareto_sweep(args, out: Path):
    """Sweep the risk weight wR on the dedicated Pareto scenario; plot L vs Rtil."""
    spec = scn.build_pareto(master_seed=args.seed, run_idx=0, randomize=False)
    spec, cfg0, terrain, risk, H, _ = _build(spec, args.resolution)
    from uav_pathplan import cost as costmod

    wR_list = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80] if not args.quick else [0.1, 0.45, 0.8]
    wK = cfg0.wK
    L_vals, R_vals, used_wR = [], [], []
    print("\n=== Pareto sweep (pareto scenario: single unshielded radar, A*) ===")
    for wR in wR_list:
        wL = 1.0 - wK - wR
        if wL < 0:
            continue
        cfg = replace(cfg0, wL=wL, wR=wR, wK=wK)
        res, _ = _run_planner(plan_astar, cfg, terrain, risk, spec.start, spec.goal, H, False)
        if res.n_waypoints > 1:
            cr = costmod.evaluate(cfg, risk, terrain, res.path)
            L_vals.append(cr.L)
            R_vals.append(cr.Rtil)
            used_wR.append(wR)
            print(f"  wR={wR:.2f}: L={cr.L:.0f}m  Rtil={cr.Rtil:.3f}")
    if L_vals:
        plotting.plot_pareto(L_vals, R_vals, used_wR, out / f"pareto_LvsR.{args.format}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20240603)
    ap.add_argument("--runs", type=int, default=10, help="Monte-Carlo runs/scenario")
    ap.add_argument("--resolution", type=float, default=10.0, help="grid cell (m)")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument(
        "--out", default=None, help="output directory (v1: outputs; v2: from the protocol)"
    )
    ap.add_argument("--mem", action="store_true", help="record peak Python memory")
    ap.add_argument("--quick", action="store_true", help="fast sanity pass")
    ap.add_argument("--full", action="store_true", help="5 m grid, 20 runs")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dpi", type=int, default=300, help="figure resolution in DPI (default 300)")
    ap.add_argument(
        "--format",
        choices=["png", "tiff", "eps"],
        default="png",
        help="figure file format (default png)",
    )
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate the deterministic figures only "
        "(skip the Monte-Carlo runs, CSVs, and tests)",
    )
    ap.add_argument(
        "--protocol",
        choices=["v1", "v2"],
        default="v1",
        help="v1 (default): reproduces paper_assets/; v2: settings from --config",
    )
    ap.add_argument(
        "--config", default="configs/protocol_v2.yaml", help="v2 protocol file (v2 only)"
    )
    ap.add_argument(
        "--seed-set",
        choices=["eval", "dev"],
        default="eval",
        help="v2 only: held-out evaluation seed or development seed",
    )
    args = ap.parse_args()

    if args.protocol == "v2":
        from uav_pathplan.v2.runner import main as run_v2

        sys.exit(run_v2(args))

    if args.quick:
        args.resolution, args.runs = 12.0, 4
    if args.full:
        args.resolution, args.runs = 5.0, 20

    plotting.SAVE_DPI = args.dpi  # applies to every saved figure

    out = Path(args.out or "outputs")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    print(
        f"UAV path-planning experiments | master seed = {args.seed} | "
        f"resolution = {args.resolution} m | runs = {args.runs} | "
        f"dpi = {args.dpi} | format = {args.format}"
    )

    # --- figures-only fast path (deterministic; no Monte-Carlo) ------------
    if args.figures_only:
        make_overlays(args, out)
        time_vs_threats(args, out)
        pareto_sweep(args, out)
        print(f"\nRegenerated {args.format} figures at {args.dpi} DPI in {out.resolve()}")
        print(f"Total wall-clock: {time.perf_counter() - t_start:.1f}s")
        return

    # --- Monte-Carlo -------------------------------------------------------
    rows = monte_carlo(args)
    df = pd.DataFrame([r.as_dict() for r in rows])
    df.to_csv(out / "results_per_run.csv", index=False)

    agg = aggregate(rows, success_only=True)
    agg.to_csv(out / "results_aggregate.csv", index=False)

    # --- paired Wilcoxon tests --------------------------------------------
    tests = paired_tests(rows, metric="J")
    tdf = tests_to_frame(tests)
    tdf.to_csv(out / "wilcoxon_J.csv", index=False)

    # --- figures -----------------------------------------------------------
    if not args.no_plots:
        make_overlays(args, out)
        time_vs_threats(args, out)
        pareto_sweep(args, out)

    # --- console summary ---------------------------------------------------
    print("\n" + "=" * 78)
    print("AGGREGATE RESULTS  (mean over successful Monte-Carlo runs)")
    print("=" * 78)
    print(plotting.console_summary(agg))

    print("\nPaired Wilcoxon signed-rank tests on final cost J (alpha=0.05):")
    for t in tests:
        if t.n_pairs == 0:
            continue
        verdict = ("significant" if t.significant else "n.s.") if not t.note else t.note
        p = f"p={t.p_value:.4g}" if t.p_value == t.p_value else "p=-"
        ph = f"p_holm={t.p_holm:.4g}" if t.p_holm == t.p_holm else "p_holm=-"
        print(
            f"  [{t.scenario:>14}] {t.alg_a:>5} vs {t.alg_b:<6} "
            f"n={t.n_pairs:2d}  med {t.median_a:.3f} vs {t.median_b:.3f}  "
            f"{p}  {ph}  {verdict}"
        )

    print(f"\nWrote CSVs and figures to {out.resolve()}")
    print(f"Total wall-clock: {time.perf_counter() - t_start:.1f}s  (master seed {args.seed})")


if __name__ == "__main__":
    main()
