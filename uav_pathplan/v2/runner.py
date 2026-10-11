"""Protocol-v2 experiment runner (``python run_experiments.py --protocol v2``).

Flow per resolution: one warm-up run of every planner on a development-seed
instance (discarded), then for every scenario and Monte-Carlo run: build the
v2 instance, precompute the hazard grid (timed separately as ``setup_s``), run
the four planners once each, and evaluate each returned path under every
altitude mode.  Afterwards: aggregate statistics, paired tests, Friedman tests,
table-ready CSVs, figures and an environment record.

Single-threaded numpy is requested by ``run_experiments.py`` (thread variables
are set before numpy is imported); the values seen are recorded.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..planners.apf import plan_apf
from ..planners.apf_escape import plan_apf_escape
from ..planners.astar import plan_astar
from ..planners.hybrid import plan_hybrid
from ..risk import RiskField
from ..terrain import TerrainField
from . import environment, figures, protocol
from . import scenarios as v2scn
from . import stats as v2stats
from .evaluate import METRIC_KEYS, evaluate_v2

PLANNERS = list(protocol.PLANNERS)
SUMMARY_METRICS = [m for m in METRIC_KEYS if m not in ("n_interior", "n_segments", "L_ref")] + [
    "time_s",
    "setup_s",
    "n_exp",
    "n_iter",
    "n_escapes",
]


def _check_shared_params(proto: dict) -> None:
    """The protocol records the shared planner parameters; they must equal Config."""
    s, c = proto["planners"]["shared"], Config()
    expected = {
        "wL": c.wL,
        "wR": c.wR,
        "wK": c.wK,
        "R_min_m": c.R_min,
        "gamma_max_deg": c.gamma_max_deg,
        "astar_connectivity": c.astar_connectivity,
        "apf_alpha_m": c.apf_alpha,
        "apf_fd_step_m": c.apf_fd_step,
        "apf_goal_tol_m": c.apf_goal_tol,
        "apf_stuck_window": c.apf_stuck_window,
        "apf_stuck_disp_m": c.apf_stuck_disp,
        "apf_max_iter": c.apf_max_iter,
        "hybrid_coarsen": c.hybrid_coarsen,
        "hybrid_attract": c.hybrid_attract,
    }
    bad = {k: (s.get(k), v) for k, v in expected.items() if s.get(k) != v}
    if bad or c.apf_xi is not None or c.apf_eta is not None:
        raise protocol.ProtocolError(f"protocol 'shared' differs from Config defaults: {bad}")


def run_planner(name, cfg, terrain, risk, spec, H, apfe_kwargs, rng):
    if name == "astar":
        return plan_astar(cfg, terrain, risk, spec.start, spec.goal, hazard_grid=H)
    if name == "apf":
        return plan_apf(cfg, terrain, risk, spec.start, spec.goal)
    if name == "apfe":
        return plan_apf_escape(cfg, terrain, risk, spec.start, spec.goal, rng=rng, **apfe_kwargs)
    if name == "hybrid":
        return plan_hybrid(cfg, terrain, risk, spec.start, spec.goal)
    raise KeyError(name)


def build_instance(proto, scenario, master_seed, run, resolution):
    spec = v2scn.build(proto, scenario, master_seed, run)
    cfg = replace(spec.cfg, resolution=float(resolution))
    spec = replace(spec, cfg=cfg)
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    t0 = time.perf_counter()
    H = risk.grid()
    return spec, cfg, terrain, risk, H, time.perf_counter() - t0


def run_suite(
    proto: dict,
    master_seed: int,
    runs: int,
    resolutions,
    scenarios,
    apfe_kwargs: dict,
    modes,
    log=print,
    keep_instances_at: float | None = None,
):
    """Run the Monte-Carlo suite; return (per-run DataFrame, kept run-0 instances)."""
    alt = proto["altitude"]
    rows, kept = [], {}
    for res in resolutions:
        # ---- warm-up (development seed, discarded) ---------------------- #
        dev = proto["seeds"]["development"]
        spec, cfg, terrain, risk, H, _ = build_instance(proto, "open", dev, 0, res)
        for p in PLANNERS:
            run_planner(p, cfg, terrain, risk, spec, H, apfe_kwargs, v2scn.apfe_rng(dev, "open", 0))
        for scn in scenarios:
            risk_kind = proto["scenarios"][scn].get("risk_kind", "detection_probability")
            t_scn = time.perf_counter()
            for run in range(runs):
                spec, cfg, terrain, risk, H, setup_s = build_instance(
                    proto, scn, master_seed, run, res
                )
                results = {}
                for p in PLANNERS:
                    r = run_planner(
                        p,
                        cfg,
                        terrain,
                        risk,
                        spec,
                        H,
                        apfe_kwargs,
                        v2scn.apfe_rng(master_seed, scn, run),
                    )
                    results[p] = r
                    for mode in modes:
                        m = evaluate_v2(
                            cfg,
                            risk,
                            terrain,
                            r.path,
                            spec.start,
                            spec.goal,
                            mode,
                            alt["rate_limited_c_min_m"],
                            alt["profile_ds_m"],
                        )
                        rows.append(
                            dict(
                                resolution=float(res),
                                altitude_mode=mode,
                                scenario=scn,
                                risk_kind=risk_kind,
                                run=run,
                                master_seed=master_seed,
                                terrain_seed=cfg.seed,
                                planner=p,
                                success=bool(r.success),
                                reason=str(r.info.get("reason", "")),
                                time_s=r.time_s,
                                setup_s=setup_s,
                                n_exp=r.n_exp,
                                n_iter=r.n_iter,
                                n_escapes=r.info.get("n_escapes", np.nan),
                                start_x=spec.start[0],
                                start_y=spec.start[1],
                                goal_x=spec.goal[0],
                                goal_y=spec.goal[1],
                                **m,
                            )
                        )
                if keep_instances_at is not None and run == 0 and res == keep_instances_at:
                    kept[scn] = (spec, cfg, terrain, risk, results)
            log(f"  {res:4.1f} m  {scn:15s} {runs} runs  {time.perf_counter() - t_scn:7.1f} s")
    return pd.DataFrame(rows), kept


# --------------------------------------------------------------------------- #
# Tables                                                                       #
# --------------------------------------------------------------------------- #
def table2(agg: pd.DataFrame, ref_res: float) -> pd.DataFrame:
    a = agg[agg["resolution"] == ref_res]
    cols = {
        "scenario": "scenario",
        "altitude_mode": "altitude_mode",
        "planner": "planner",
        "risk_kind": "risk_kind",
        "n_runs": "n_runs",
        "n_success": "n_success",
        "success_rate": "success_rate",
        "success_ci_lo": "success_ci95_lo",
        "success_ci_hi": "success_ci95_hi",
    }
    t = a[list(cols)].rename(columns=cols).copy()
    for m in (
        "J",
        "Ltil",
        "Rtil",
        "Phi",
        "d_min",
        "kappa_peak",
        "gamma_peak_deg",
        "curv_viol_frac",
        "climb_viol_frac",
        "smoothness",
        "time_s",
    ):
        t[f"{m}_median"] = a[f"{m}_median"].to_numpy()
        t[f"{m}_iqr"] = a[f"{m}_iqr"].to_numpy()
        t[f"{m}_mean"] = a[f"{m}_mean"].to_numpy()
        t[f"{m}_sd"] = a[f"{m}_sd"].to_numpy()
    return t


def table_grid(agg: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "resolution",
        "altitude_mode",
        "scenario",
        "planner",
        "risk_kind",
        "n_success",
        "success_rate",
        "J_median",
        "J_iqr",
        "Rtil_median",
        "Ltil_median",
        "curv_viol_frac_mean",
        "climb_viol_frac_mean",
        "time_s_median",
        "setup_s_median",
    ]
    return agg[keep].sort_values(["altitude_mode", "scenario", "planner", "resolution"])


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def main(args) -> int:
    proto = protocol.load(args.config)
    _check_shared_params(proto)
    quick = bool(args.quick)
    seed_set = "dev" if quick else args.seed_set  # smoke runs never touch the held-out seed
    if seed_set == "eval" and proto["status"] != "registered":
        raise protocol.ProtocolError(
            "the evaluation seed may only be run with a registered (frozen) protocol"
        )
    if proto["status"] == "registered":
        apfe_kwargs = protocol.apfe_params(proto)
    else:  # draft: only quick smoke runs; first grid point, flagged in the outputs
        g = proto["planners"]["apfe_tuning"]["grid"]
        apfe_kwargs = dict(
            max_escapes=g["max_escapes"][0],
            turn_steps_min=g["turn_steps"][0][0],
            turn_steps_max=g["turn_steps"][0][1],
            straight_steps=g["straight_steps"][0],
            turn_frac=proto["planners"]["apfe"]["turn_frac"],
        )
    master = proto["seeds"]["evaluation" if seed_set == "eval" else "development"]
    runs = proto["monte_carlo"]["runs_per_scenario"]
    resolutions = [float(r) for r in proto["resolutions_m"]]
    scenarios = list(proto["monte_carlo"]["scenarios"])
    modes = list(proto["altitude"]["modes"])
    ref_res = float(proto["reference_resolution_m"])
    if quick:
        runs, resolutions, ref_res = 2, [12.0], 12.0
    default_out = (
        "outputs/v2_quick"
        if quick
        else proto["outputs"]["directory"]
        if seed_set == "eval"
        else "outputs/v2_dev"
    )
    out = Path(args.out or default_out)
    v1_assets = Path("paper_assets").resolve()
    if out.resolve() == v1_assets or v1_assets in out.resolve().parents:
        raise SystemExit("refusing to write v2 outputs into paper_assets/")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    env = environment.describe()
    print(
        f"Protocol v2 ({proto['status']}, sha256 {proto['_sha256'][:12]}) | seed set "
        f"{seed_set} = {master} | runs {runs} | resolutions {resolutions} | "
        f"quick={quick}"
    )
    print(
        f"CPU {env['cpu']} | RAM {env['ram_gib']} GiB | {env['os']} | "
        f"Python {env['python']} | threads {env['thread_env']}"
    )
    t0 = time.perf_counter()
    df, kept = run_suite(
        proto, master, runs, resolutions, scenarios, apfe_kwargs, modes, keep_instances_at=ref_res
    )
    wall = time.perf_counter() - t0

    agg = v2stats.summarize(df, SUMMARY_METRICS)
    tests = v2stats.pairwise_tests(df, PLANNERS, "J", proto["statistics"]["alpha"])
    fried = v2stats.friedman_tests(df, PLANNERS, "J")

    df.to_csv(out / "results_per_run.csv", index=False)
    agg.to_csv(out / "results_aggregate.csv", index=False)
    tests.to_csv(out / "stats_wilcoxon_J.csv", index=False)
    fried.to_csv(out / "stats_friedman_J.csv", index=False)
    table2(agg, ref_res).to_csv(out / "table2_main_results.csv", index=False)
    tests[tests["resolution"] == ref_res].to_csv(out / "table3_pairwise_tests.csv", index=False)
    fried[fried["resolution"] == ref_res].to_csv(out / "table3_friedman.csv", index=False)
    table_grid(agg).to_csv(out / "table_grid_sensitivity.csv", index=False)

    # ---- figures (reference resolution) -------------------------------- #
    plotting_fmt = proto["outputs"]["figure_formats"]
    from .. import plotting

    plotting.SAVE_DPI = int(proto["outputs"]["figure_dpi"])
    at_ref = df[df["resolution"] == ref_res]
    agg_ref = agg[agg["resolution"] == ref_res]
    for mode in modes:
        d = at_ref[at_ref["altitude_mode"] == mode]
        figures.box_metric(
            d, PLANNERS, scenarios, "J", "cost J", fig_dir, f"box_J_{mode}", plotting_fmt
        )
        figures.box_metric(
            d,
            PLANNERS,
            scenarios,
            "Rtil",
            r"$\tilde R$",
            fig_dir,
            f"box_Rtil_{mode}",
            plotting_fmt,
            risk_note=True,
        )
    figures.success_bars(
        agg_ref[agg_ref["altitude_mode"] == modes[0]],
        PLANNERS,
        scenarios,
        fig_dir,
        "success_rates",
        plotting_fmt,
    )
    for mode in modes:
        a = agg[agg["altitude_mode"] == mode]
        figures.grid_sensitivity(
            a, PLANNERS, scenarios, "J", "median J (IQR)", fig_dir, f"grid_J_{mode}", plotting_fmt
        )
    a_nap = agg[agg["altitude_mode"] == modes[0]]
    figures.grid_sensitivity(
        a_nap,
        PLANNERS,
        scenarios,
        "time_s",
        "median planning time (s)",
        fig_dir,
        "grid_time",
        plotting_fmt,
        logy=True,
    )
    figures.altitude_modes(
        agg_ref,
        PLANNERS,
        scenarios,
        "Rtil",
        r"median $\tilde R$",
        fig_dir,
        "altitude_Rtil",
        plotting_fmt,
    )
    figures.altitude_modes(
        agg_ref, PLANNERS, scenarios, "J", "median J", fig_dir, "altitude_J", plotting_fmt
    )
    for scn, (spec, cfg, terrain, risk, results) in kept.items():
        mrows = {}
        for p, r in results.items():
            mrows[p] = evaluate_v2(
                cfg,
                risk,
                terrain,
                r.path,
                spec.start,
                spec.goal,
                "nap",
                proto["altitude"]["rate_limited_c_min_m"],
                proto["altitude"]["profile_ds_m"],
            )
        figures.overlay(
            cfg,
            terrain,
            risk,
            spec.start,
            spec.goal,
            results,
            mrows,
            fig_dir,
            f"overlay_{scn}",
            plotting_fmt,
        )

    # ---- provenance ---------------------------------------------------- #
    env.update(
        protocol_file=proto["_path"],
        protocol_sha256=proto["_sha256"],
        protocol_status=proto["status"],
        seed_set=seed_set,
        master_seed=master,
        development_seed=proto["seeds"]["development"],
        evaluation_seed=proto["seeds"]["evaluation"],
        runs_per_scenario=runs,
        resolutions_m=resolutions,
        reference_resolution_m=ref_res,
        apfe_parameters=apfe_kwargs,
        quick=quick,
        wall_clock_s=round(wall, 1),
    )
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    shutil.copyfile(proto["_path"], out / "protocol_v2.yaml")
    print(f"\nWrote v2 results to {out.resolve()}  (suite wall-clock {wall:.0f} s)")
    return 0
