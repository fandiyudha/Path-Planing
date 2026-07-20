"""Interactive research dashboard (Streamlit) for the risk-aware UAV
path-planning simulation.

This is a thin PRESENTATION layer: it imports and calls the existing simulation
code (terrain, risk, cost, planners, scenarios, metrics, plotting) and never
reimplements any of it.

Fairness contract (enforced here):
  * Terrain + the radar hazard/LOS field are built ONCE per
    (scenario, seed, resolution, clearance, n_radars) and cached.  These do not
    depend on the cost weights, so the weight sliders never rebuild them.
  * The cost weights and kinematic limits are applied to a single ``Config``
    that is shared by every planner in a run, so all algorithms read the same
    terrain, the same risk field, and minimise the same cost J.  The GUI only
    changes inputs and displays outputs.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from uav_pathplan import scenarios as scn
from uav_pathplan import plotting
from uav_pathplan import cost as costmod
from uav_pathplan.terrain import TerrainField
from uav_pathplan.risk import RiskField
from uav_pathplan.metrics import compute_metrics
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.hybrid import plan_hybrid

SCENARIOS = list(scn.SCENARIO_BUILDERS)
PLANNER_FN = {"astar": plan_astar, "apf": plan_apf, "hybrid": plan_hybrid}
PLANNER_LABEL = {"astar": "A*", "apf": "APF", "hybrid": "Hybrid"}
# Path colours come from the single shared palette in plotting.ALGO_STYLE so
# they never drift between the dashboard and the batch figures.

# metric display order, label, format, and "is higher better?"
METRIC_SPEC = [
    ("success", "success", None, True),
    ("J", "J (cost)", "{:.4f}", False),
    ("L", "L length (m)", "{:.0f}", False),
    ("Rtil", "Rtil risk", "{:.4f}", False),
    ("Phi", "Phi (∫hazard)", "{:.3f}", False),
    ("Lambda_peak", "Lambda peak", "{:.4g}", False),
    ("d_min", "d_min standoff (m)", "{:.0f}", True),
    ("kappa_peak", "kappa peak (1/m)", "{:.4f}", False),
    ("curv_violations", "curv. violations", "{:.0f}", False),
    ("gamma_peak_deg", "peak |gamma| (deg)", "{:.1f}", False),
    ("climb_violations", "climb violations", "{:.0f}", False),
    ("smoothness", "smoothness Σ|dψ|", "{:.2f}", False),
    ("time_s", "compute time (s)", "{:.3f}", False),
    ("n_exp", "expanded nodes", "{:.0f}", False),
    ("n_iter", "iterations", "{:.0f}", False),
]


# --------------------------------------------------------------------------- #
# Cached, WEIGHT-INDEPENDENT field precomputation                             #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def build_fields(scenario: str, seed: int, resolution: float,
                 clearance: float, n_radars: int):
    """Build + cache terrain, risk field, base Config, and start/goal.

    Keyed only on field-determining inputs (clearance affects altitude -> the
    hazard field, hence it is a field parameter; the cost weights and kinematic
    limits are NOT here -- they are applied per run).  Returns
    ``(terrain, risk, cfg, start, goal)``.
    """
    if scenario == "high_density":
        spec = scn.build_high_density(master_seed=seed, run_idx=0,
                                      randomize=False, n_radars=int(n_radars))
    else:
        spec = scn.build_scenario(scenario, master_seed=seed, run_idx=0,
                                  randomize=False)
    cfg = replace(spec.cfg, seed=int(seed), resolution=float(resolution),
                  clearance=float(clearance))
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    risk.grid()                                # precompute exact-LOS hazard grid
    return terrain, risk, cfg, spec.start, spec.goal


@st.cache_data(show_spinner=False)
def run_planners(scenario: str, seed: int, resolution: float, clearance: float,
                 n_radars: int, R_min: float, gamma_max_deg: float,
                 wL: float, wR: float, wK: float, which: tuple):
    """Run the selected planner(s) on the shared field; return results+metrics.

    Weights/kinematic limits are applied to ONE Config shared by every planner
    (fairness).  Cached on the full input signature so re-pressing Run with the
    same inputs is instant; weights/limits are part of the key, so changing them
    and pressing Run re-plans.
    """
    terrain, risk, cfg0, start, goal = build_fields(scenario, seed, resolution,
                                                     clearance, n_radars)
    cfg = replace(cfg0, wL=float(wL), wR=float(wR), wK=float(wK),
                  R_min=float(R_min), gamma_max_deg=float(gamma_max_deg))
    results, rows = {}, {}
    for name in which:
        res = PLANNER_FN[name](cfg, terrain, risk, start, goal)
        results[name] = res
        rows[name] = compute_metrics(cfg, terrain, risk, scenario, name, 0,
                                     res).as_dict()
    return results, rows


@st.cache_data(show_spinner=True)
def pareto_sweep(scenario: str, seed: int, resolution: float, clearance: float,
                 n_radars: int, R_min: float, gamma_max_deg: float,
                 algo: str, wK: float, wR_values: tuple):
    """Sweep the risk weight wR for one planner; return (L, Rtil, wR) lists.

    Reuses the cached field (weights do not rebuild it); only the planner re-runs
    per weight.  This is the key length-vs-risk Pareto figure for the paper.
    """
    terrain, risk, cfg0, start, goal = build_fields(scenario, seed, resolution,
                                                     clearance, n_radars)
    L_vals, R_vals, used = [], [], []
    for wR in wR_values:
        wL = 1.0 - wK - wR
        if wL < 0:
            continue
        cfg = replace(cfg0, wL=wL, wR=wR, wK=wK, R_min=float(R_min),
                      gamma_max_deg=float(gamma_max_deg))
        res = PLANNER_FN[algo](cfg, terrain, risk, start, goal)
        if res.n_waypoints > 1:
            cr = costmod.evaluate(cfg, risk, terrain, res.path)
            L_vals.append(cr.L); R_vals.append(cr.Rtil); used.append(wR)
    return L_vals, R_vals, used


# --------------------------------------------------------------------------- #
# Small presentation helpers                                                  #
# --------------------------------------------------------------------------- #
def render_overlay(cfg, terrain, risk, start, goal, results: dict, title: str,
                   metrics: dict | None = None):
    """2D overlay — delegates to the shared plotting.plot_overlay so the
    algorithm palette, failed-path markers, and legend match every other figure
    (single source of truth in plotting.ALGO_STYLE)."""
    return plotting.plot_overlay(cfg, terrain, risk, start, goal, results,
                                 out_path=None, title=title, close=False,
                                 metrics=metrics)


def metrics_frame(rows: dict):
    """Build (display_df, numeric_df) with metrics as rows, algorithms as cols."""
    algos = list(rows)
    disp = pd.DataFrame(index=[lbl for _, lbl, _, _ in METRIC_SPEC],
                        columns=[PLANNER_LABEL[a] for a in algos])
    num = pd.DataFrame(index=[lbl for _, lbl, _, _ in METRIC_SPEC],
                       columns=[PLANNER_LABEL[a] for a in algos], dtype=float)
    for a in algos:
        col = PLANNER_LABEL[a]
        for key, lbl, fmt, _ in METRIC_SPEC:
            v = rows[a].get(key, np.nan)
            if key == "success":
                disp.loc[lbl, col] = "✓" if v else "✗"
                num.loc[lbl, col] = 1.0 if v else 0.0
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                disp.loc[lbl, col] = "—"
                num.loc[lbl, col] = np.nan
            else:
                disp.loc[lbl, col] = fmt.format(v)
                num.loc[lbl, col] = float(v)
    return disp, num


def style_best(disp: pd.DataFrame, num: pd.DataFrame):
    """Bold + highlight the best algorithm per metric row."""
    higher = {lbl: hi for _, lbl, _, hi in METRIC_SPEC}

    def _row(row):
        nums = num.loc[row.name].dropna()
        if nums.empty or len(disp.columns) < 2:
            return ["" for _ in row]
        best = nums.idxmax() if higher[row.name] else nums.idxmin()
        # if all equal, don't highlight
        if nums.nunique() <= 1:
            return ["" for _ in row]
        return ["font-weight:bold;background-color:#1f4e79;color:white"
                if c == best else "" for c in row.index]

    return disp.style.apply(_row, axis=1)


def fig_to_png_bytes(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Page + sidebar                                                              #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="UAV Path Planning Dashboard",
                   layout="wide", page_icon="🛰")
st.title("🛰 Risk-Aware UAV Path Planning — A* vs APF vs Hybrid")
st.caption("Thin interactive layer over the existing simulation. For any run, "
           "all three planners share the same terrain, risk field, and cost J — "
           "only inputs change here.")

with st.sidebar:
    st.header("Scenario & field")
    scenario = st.selectbox("Scenario", SCENARIOS,
                            index=SCENARIOS.index("concave_trap"))
    seed = int(st.number_input("Random seed", min_value=0,
                               max_value=2**31 - 1, value=20240603, step=1))
    resolution = float(st.select_slider(
        "Grid resolution (m)", options=[5.0, 8.0, 10.0, 12.5, 20.0], value=10.0))
    clearance = float(st.slider("Clearance altitude (m)", 10.0, 150.0, 50.0, 5.0,
                                help="Nap-of-the-earth z = h + clearance. Affects "
                                     "the hazard field, so all planners share it."))
    n_radars = int(st.slider("Number of radars", 2, 30, 16, 1,
                             help="Applies to the high_density scenario; other "
                                  "scenarios use their scripted threat layout."))
    if scenario != "high_density":
        st.caption("ℹ️ Radar-count slider only affects *high_density*; "
                   f"*{scenario}* uses its scripted threats.")

    st.header("Algorithm")
    mode = st.radio("Planner", ["A*", "APF", "Hybrid", "Compare all three"],
                    index=3)

    st.header("Cost weights  (auto-normalised, sum = 1)")
    st.caption("The core control for the paper: the length / risk / kinematic "
               "trade-off in the shared cost J.")
    wL_raw = st.slider("wL — excess length", 0.0, 1.0, 0.40, 0.01)
    wR_raw = st.slider("wR — detection risk", 0.0, 1.0, 0.45, 0.01)
    wK_raw = st.slider("wK — kinematic feasibility", 0.0, 1.0, 0.15, 0.01)
    s = wL_raw + wR_raw + wK_raw
    if s <= 0:
        wL, wR, wK = 1 / 3, 1 / 3, 1 / 3
    else:
        wL, wR, wK = wL_raw / s, wR_raw / s, wK_raw / s
    wL, wR, wK = round(wL, 4), round(wR, 4), round(wK, 4)
    st.success(f"normalised → wL={wL:.3f}, wR={wR:.3f}, wK={wK:.3f}")

    st.header("Kinematic limits")
    R_min = float(st.slider("R_min — min turn radius (m)", 20.0, 200.0, 60.0, 5.0))
    gamma_max_deg = float(st.slider("gamma_max — max climb/dive (deg)",
                                    5.0, 45.0, 15.0, 1.0))

    show_3d = st.checkbox("Show 3D terrain surface", value=False)

    run = st.button("▶ Run / Update", type="primary", width='stretch')

# which planners
WHICH = {"A*": ("astar",), "APF": ("apf",), "Hybrid": ("hybrid",),
         "Compare all three": ("astar", "apf", "hybrid")}[mode]

# build fields live (cached, cheap, weight-independent)
terrain, risk, cfg, start, goal = build_fields(scenario, seed, resolution,
                                               clearance, n_radars)

# run signature: planning only happens on the Run button, but results persist
sig = (scenario, seed, resolution, clearance, n_radars, R_min, gamma_max_deg,
       wL, wR, wK, WHICH)
if run:
    st.session_state["committed_sig"] = sig
committed = st.session_state.get("committed_sig")
have_run = committed == sig


# --------------------------------------------------------------------------- #
# Tabs                                                                        #
# --------------------------------------------------------------------------- #
tab_plan, tab_pareto = st.tabs(["🗺 Planner", "📈 Pareto explorer"])

with tab_plan:
    if not have_run:
        # show field only + prompt
        st.subheader(f"Hazard field — {scenario}")
        fig = render_overlay(cfg, terrain, risk, start, goal, {},
                             f"{scenario} (seed {seed}, {cfg.n_cells}×{cfg.n_cells})")
        st.pyplot(fig); plt.close(fig)
        st.warning("Inputs set. Press **▶ Run / Update** in the sidebar to plan "
                   "with the current weights and limits.")
    else:
        with st.spinner("Planning…"):
            results, rows = run_planners(scenario, seed, resolution, clearance,
                                         n_radars, R_min, gamma_max_deg,
                                         wL, wR, wK, WHICH)
        col_plot, col_tbl = st.columns([3, 2])
        with col_plot:
            title = (f"{scenario} — " +
                     ("comparison" if len(WHICH) > 1 else PLANNER_LABEL[WHICH[0]]) +
                     f"  (wL={wL:.2f}, wR={wR:.2f}, wK={wK:.2f})")
            fig = render_overlay(cfg, terrain, risk, start, goal, results, title,
                                 metrics=rows)
            png = fig_to_png_bytes(fig)
            st.pyplot(fig); plt.close(fig)
            st.download_button("⬇ Export figure (PNG)", png,
                               file_name=f"{scenario}_{mode.replace(' ', '_')}.png",
                               mime="image/png")
        with col_tbl:
            st.markdown("**Metrics**" +
                        ("  — best per row in **bold**" if len(WHICH) > 1 else ""))
            disp, num = metrics_frame(rows)
            if len(WHICH) > 1:
                st.dataframe(style_best(disp, num), width='stretch',
                             height=560)
            else:
                st.dataframe(disp, width='stretch', height=560)
            csv = pd.DataFrame([rows[a] for a in WHICH]).to_csv(index=False)
            st.download_button("⬇ Export run (CSV)", csv,
                               file_name=f"{scenario}_{mode.replace(' ', '_')}.csv",
                               mime="text/csv")

        if show_3d:
            st.subheader("3D terrain surface with paths")
            # Draw EVERY attempted path (incl. failed APF stub + stuck marker);
            # not filtered by success.
            fig3d = plotting.plot_surface_with_paths(
                cfg, terrain, results, out_path=None,
                title=f"{scenario} — terrain + paths", close=False, metrics=rows)
            st.pyplot(fig3d)
            st.download_button("⬇ Export 3D figure (PNG)",
                               fig_to_png_bytes(fig3d),
                               file_name=f"{scenario}_{mode.replace(' ', '_')}_3d.png",
                               mime="image/png")
            plt.close(fig3d)

with tab_pareto:
    st.subheader("Pareto front — path length vs detection risk")
    st.caption("Sweeps the risk weight wR (with wK fixed, wL = 1 − wK − wR) for "
               "one planner on the current scenario/field. Lowering risk should "
               "cost length. Results are cached.")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        p_algo_label = st.selectbox("Planner for sweep", ["A*", "Hybrid", "APF"])
        p_algo = {"A*": "astar", "Hybrid": "hybrid", "APF": "apf"}[p_algo_label]
    with pc2:
        wr_lo, wr_hi = st.slider("wR sweep range", 0.0, 0.9, (0.05, 0.80), 0.05)
    with pc3:
        n_pts = st.slider("points", 3, 10, 6)
    p_wK = st.slider("fixed wK for sweep", 0.0, 0.5, 0.15, 0.05)
    go_sweep = st.button("▶ Run Pareto sweep", type="primary")

    if go_sweep:
        st.session_state["pareto_sig"] = (scenario, seed, resolution, clearance,
                                          n_radars, R_min, gamma_max_deg, p_algo,
                                          p_wK, wr_lo, wr_hi, n_pts)
    psig = st.session_state.get("pareto_sig")
    if psig is None:
        st.info("Set the sweep range and press **▶ Run Pareto sweep**.")
    else:
        (s_scn, s_seed, s_res, s_clr, s_nr, s_rmin, s_gmax, s_algo, s_wk,
         s_lo, s_hi, s_np) = psig
        wR_values = tuple(round(float(w), 4)
                          for w in np.linspace(s_lo, s_hi, s_np))
        L_vals, R_vals, used = pareto_sweep(s_scn, s_seed, s_res, s_clr, s_nr,
                                            s_rmin, s_gmax, s_algo, s_wk, wR_values)
        if not L_vals:
            st.warning("No successful paths in the sweep range.")
        else:
            cL, cR = st.columns([3, 2])
            with cL:
                figp = plotting.plot_pareto(
                    L_vals, R_vals, used, out_path=None, close=False,
                    title=f"Pareto — {PLANNER_LABEL[s_algo]} on {s_scn} "
                          f"(seed {s_seed})")
                st.pyplot(figp)
                st.download_button("⬇ Export Pareto (PNG)", fig_to_png_bytes(figp),
                                   file_name=f"pareto_{s_scn}_{s_algo}.png",
                                   mime="image/png")
                plt.close(figp)
            with cR:
                dfp = pd.DataFrame({"wR": used, "L_m": np.round(L_vals, 1),
                                    "Rtil": np.round(R_vals, 4)})
                st.dataframe(dfp, width='stretch', hide_index=True)
                st.download_button("⬇ Export Pareto (CSV)",
                                   dfp.to_csv(index=False),
                                   file_name=f"pareto_{s_scn}_{s_algo}.csv",
                                   mime="text/csv")

st.caption(f"Seed {seed} • {cfg.n_cells}×{cfg.n_cells} grid @ {resolution:.0f} m • "
           f"reproduces a run_experiments.py run with the same Config.")
