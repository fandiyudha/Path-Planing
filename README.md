# Risk-Aware UAV Path Planning — A* vs APF vs Hybrid A*-APF

A reproducible Python framework comparing three path planners for a UAV
operating in a **2.5D** environment with **radar threats modulated by terrain
masking**:

1. **A\*** — grid search with a heading-augmented state.
2. **APF** — Artificial Potential Field gradient descent.
3. **Hybrid A\*-APF** — coarse global A\* corridor + local APF tracking.

All three planners read the **same elevation field**, the **same risk field**,
and minimise the **same cost function `J`**. Only the search mechanism differs.
This fairness is enforced in the architecture (a single `Config`, a single
`cost.evaluate`, a single `RiskField`), not just in comments — it is the
scientific point of the study.

Everything is implemented from scratch with `numpy`/`scipy`/`matplotlib`: the
planners, the manual line-of-sight, the value-noise / fBm terrain generator, and
the cost function. `scipy` is used only for low-level array helpers
(`RegularGridInterpolator`) and `scipy.stats.wilcoxon` for the significance test;
`pandas` only assembles result tables/CSV. No path-planning, robotics, or
computer-vision library is used.

---

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, matplotlib, pandas, streamlit
```

Python 3.10+ (uses `X | Y` type hints and the modern `numpy.random.Generator`).
`streamlit` is only needed for the interactive dashboard (below); the batch
driver `run_experiments.py` does not require it.

## Run

```bash
# Full Monte-Carlo comparison + all figures + CSVs (default 10 m grid, 10 runs)
python run_experiments.py

# Fast sanity pass (12 m grid, 4 runs)
python run_experiments.py --quick

# Publication settings (5 m grid, 20 runs — slow)
python run_experiments.py --full

# Custom
python run_experiments.py --runs 20 --resolution 8 --seed 7
python run_experiments.py --mem            # also record peak Python memory
python run_experiments.py --no-plots       # CSVs only
```

The master seed is printed and saved with every output, so any run reproduces
exactly. Outputs land in `./outputs/`.

## Interactive dashboard (Streamlit)

```bash
cd dashboard
streamlit run app.py
```

### Deploy free on Streamlit Community Cloud

The dashboard runs statelessly (no disk writes, no secrets — everything it shows
is generated in memory), so it hosts cleanly on the free tier:

1. Push this repo to GitHub (see below).
2. Go to <https://share.streamlit.io> → **New app** → pick the repo/branch and
   set the main file to `dashboard/app.py` → **Deploy**. It installs `requirements.txt`
   automatically; no extra config is needed (`dashboard/.streamlit/config.toml` is
   included).

Notes for the free tier (≈1 GB RAM, 1 CPU): keep the grid **resolution at 10 m
or coarser** for snappy A\* (finer grids still work but are slower); the app
sleeps when idle and wakes on the next visit. Do **not** run the Monte-Carlo
driver (`run_experiments.py`) on the hosted app — generate the paper tables and
figures locally. (Note: Vercel/Netlify are **not** suitable — Streamlit needs a
persistent Python/WebSocket server, not serverless functions.)

`dashboard/app.py` is a thin **presentation layer**: it imports and calls the existing
simulation (terrain, risk, cost, planners, scenarios, metrics, plotting) and
reimplements none of it. The fairness constraint is preserved — for any run all
three planners read the same terrain, the same risk field, and minimise the same
cost `J`; the GUI only changes inputs and displays outputs.

**Sidebar controls**

| Control | Effect |
|---|---|
| **Scenario** | open / concave_trap / ridge_corridor / high_density. |
| **Random seed** | Integer seed (visible & explicit); a view here reproduces a `run_experiments.py` run with the same `Config`. |
| **Grid resolution** | Cell size (m). Coarser = faster A\*. *Field parameter.* |
| **Clearance altitude** | Nap-of-the-earth `z = h + clearance`. Affects the hazard field, so it is shared by all planners. *Field parameter.* |
| **Number of radars** | Threat count for **high_density** only (other scenarios use their scripted threats; a caption notes this). *Field parameter.* |
| **Algorithm** | A\*, APF, Hybrid, or **Compare all three**. |
| **Cost weights `wL`,`wR`,`wK`** | The headline control — the length / risk / kinematic trade-off in `J`. Auto-normalised to sum 1; the normalised values are displayed. |
| **`R_min`, `gamma_max`** | Kinematic limits (turn radius, max climb/dive angle). *Planner parameters.* |
| **Show 3D terrain surface** | Toggles the 3D view. |
| **▶ Run / Update** | Runs the planner(s). Planning happens **only** on this button (A\* can be slow); changing weights/limits shows a "press Run" notice and does **not** recompute. The cheap, weight-independent hazard map updates live. |

**Main area** — *Planner* tab: the hazard heatmap with terrain contours, radar
range rings, start/goal, and the planned path(s) (distinct colours + legend in
compare mode); a metrics panel with the full metric set (`J`, `L`, `Rtil`,
`Phi`, `Lambda_peak`, `d_min`, `kappa_peak`, curvature/climb violations, peak
`gamma`, smoothness, time, nodes/iterations, success) — one column per algorithm
in compare mode with the **best value per row in bold**; an optional 3D surface
with the path; and **Export figure (PNG)** / **Export run (CSV)** buttons.

*Pareto explorer* tab: sweep the risk weight `wR` (with `wK` fixed,
`wL = 1 − wK − wR`) for a chosen planner and plot the **length-vs-risk Pareto
front**, with PNG/CSV export. The sweep reuses the cached field and is itself
cached.

**Caching / fairness in the GUI.** Terrain generation and the radar hazard/LOS
map are cached on field-determining inputs only (scenario, seed, resolution,
clearance, radar count) via `@st.cache_data`, so they are **not** rebuilt when
the cost weights change. Weights and kinematic limits are applied to a single
shared `Config` at planning time, so every algorithm in a run gets the identical
field — the GUI can never hand one planner a different field or weight than
another.

### Per-module smoke tests (the build-order checkpoints)

Each can be run on its own and writes a figure to `outputs/`:

```bash
python smoke_terrain.py   # 1. terrain field            -> terrain_2d.png, terrain_3d.png
python smoke_risk.py      # 2. risk field + manual LOS  -> risk_hazard.png (ridge shadow)
python smoke_cost.py      # 3. cost J, verified by hand  (prints PASS/FAIL)
python smoke_astar.py     # 4. A* on open terrain       -> astar_path.png
python smoke_apf.py       # 5. APF open (ok) + trap (stalls) -> apf_open.png, apf_trap.png
python smoke_hybrid.py    # 6. A* vs APF vs Hybrid on trap   -> compare_trap.png
```

---

## Outputs and how to reproduce each figure

`run_experiments.py` writes to `outputs/`:

| File | Content | Produced by |
|------|---------|-------------|
| `results_per_run.csv`    | every (scenario, algorithm, run) metric row | `monte_carlo` |
| `results_aggregate.csv`  | mean ± std per (scenario, algorithm), success rate | `metrics.aggregate` |
| `wilcoxon_J.csv`         | paired Wilcoxon tests on `J` | `stats.paired_tests` |
| `overlay_<scenario>.png` | 2D paths on hazard heatmap + terrain contours + threats | `make_overlays` |
| `surface_<scenario>.png` | 3D elevation surface with the path | `make_overlays` |
| `time_vs_threats.png`    | A\*/Hybrid compute time & expanded nodes vs radar count | `time_vs_threats` |
| `pareto_LvsR.png`        | length-vs-risk Pareto front from sweeping `wR` | `pareto_sweep` |

A console table mirroring the paper's main results is printed at the end.

---

## The cost function `J` (shared by all planners)

A path `π = p_0..p_N` (horizontal waypoints; altitude from the policy). Three
normalised components, each zero for an ideal path:

```
1. Excess length      L̃   = (Σ‖p_{i+1}-p_i‖ − L_ref) / L_ref ,  L_ref = ‖p_g − p_s‖
2. Detection risk     Φ   = Σ Λ(midpoint_i)·ds_i ,   R̃ = 1 − exp(−Φ)   ∈ [0,1]
3. Kinematic penalty  K̃   = mean_i [ max(0,(κ_i−κ_max)/κ_max)
                                    + max(0,(|γ_i|−γ_max)/γ_max) ]
J = wL·L̃ + wR·R̃ + wK·K̃ ,   wL+wR+wK = 1
```

* `κ_i = |Δψ_i|/s̄_i` is discrete horizontal curvature, limited by
  `κ_max = 1/R_min`.
* `γ_i = atan2(Δz_i, ds_i)` is the flight-path angle, limited by `γ_max`.

**Normalisation rule:** planners use only the *a-priori* normaliser `L_ref`
(known before planning). Comparison tables may *additionally* report empirical
min–max normalisation, but empirical-range normalisation is **never** fed back
into the planners. `cost.evaluate` is the single shared scorer; A\* also uses
the per-edge helpers in `cost.py` so the objective cannot drift between planners.

The cost is verified against a hand computation in `smoke_cost.py`.

---

## Environment model (2.5D)

* **Elevation `h(x,y)`** = fractal base (value-noise fBm: seeded random lattice
  per octave, manual bilinear upsample, accumulated with `persistence` /
  `lacunarity`) + parametric Gaussian hills and oriented ridges placed by the
  scenario author. Calibrated to a realistic ~0–400 m range. Fully seeded.
* **Risk `Λ(p)`**: each radar `j` contributes
  `λ_j = V_j · C_j / max(R_j, R0)^4 · 1[R_j ≤ R_max]`, where `R_j` is the full
  3D range to the radar phase centre and `V_j ∈ {0,1}` is **manual line-of-sight
  visibility** (march the 3D segment in `los_step` steps, blocked if terrain
  rises above the line). `Λ = Σ_j λ_j`.
  * A precomputed **hazard grid** at planning altitude (used by A\*).
  * A fast continuous callable `Λ_field(x,y)` that **bilinearly interpolates the
    grid** — what APF samples and finite-differences. It agrees with the grid
    exactly at grid points (the fairness requirement) and is orders of magnitude
    cheaper than recomputing LOS per probe.
* **Altitude policy** (pluggable): default nap-of-the-earth `z = h + clearance`,
  so flying behind a ridge genuinely lowers detection; `constant` altitude is
  also available.

`smoke_risk.py` confirms a ridge casts a detection shadow behind it.

### Radar power constant `C` — calibrated, not hand-tuned

The mission detection probability is the survival model `R̃ = 1 − exp(−Φ)`,
`Φ = Σ Λ(midpoint)·ds`. This is only meaningful if `C_j` is set to a sensible
scale: a hand-picked `C` either saturates `Φ` (any radar contact ⇒ `R̃≈1`) or
makes threats irrelevant (`R̃≈0`), collapsing `R̃` into a binary flag and
hollowing out the risk-adjusted-cost / Pareto analysis. So instead of a magic
number, **each radar's `C_j` is derived from one documented physical reference
point** (`risk.calibrate_C`):

> *Reference transit.* A straight, level aircraft transit that crosses the
> radar's coverage disc, passing at closest slant range `R_ref = ref_range_frac ·
> R_max` (default mid-range, `0.5·R_max`) with clear line of sight along the whole
> in-disc portion. Parametrising by along-track distance `s` from closest
> approach, `R(s)=√(R_ref²+s²)` and the transit spans `|s| ≤ √(R_max²−R_ref²)`
> (disc edge to disc edge).

We require this reference transit to be detected with probability
`ref_detection_prob` (default 0.5). Because `Φ` is linear in `C`,

```
Φ_ref(C) = C · I,   I = ∫ ds / max(R(s), R0)^4  over the transit
C_j      = −ln(1 − ref_detection_prob) / I .
```

So every radar's strength follows one interpretable assumption — *"a straight
mid-range pass is detected ~50 % of the time"* — and stronger/longer-range
radars automatically get a larger `C`. `ref_detection_prob`, `ref_range_frac`
(or absolute `ref_range_m`) live in `Config`; a per-radar `RadarSpec.C` override
is still allowed (used only by the `concave_trap` scenario, whose "radars" are a
proxy-obstacle wall for inducing APF entrapment, not detection threats).

**Near-field floor `R0`** is the minimum effective / burn-through range (default
50 m): inside it the hazard is held at its `R0` value rather than diverging as
`1/R⁴`. With the calibrated `C`, the floored near-field hazard stays
large-but-finite (`Λ(R0) ≈ 0.06–1.3` on the detection scenarios), not an absurd
spike. Calibration verified: candidate-path `R̃` now populates the informative
band `0.05–0.95` (≈28–41 % of sampled paths, vs ~0–3 % before), and the
`high_density` weight sweep traces a smooth length-vs-risk Pareto front.

---

## Scenario suite (`scenarios.py`)

All seeded; `randomize=True` draws a Monte-Carlo instance (jittered start/goal,
and where appropriate jittered threats) from a per-run RNG.

* **`open`** — one off-axis radar; sanity check, all planners succeed.
* **`concave_trap`** — a **narrow dead-end radar canyon** opening toward the
  start. Its width is *less than the turning diameter* `2·R_min`, and the wall
  radars have `R_max < R_min` so the centreline is enterable but the vehicle
  cannot U-turn inside. APF drives in, is repelled head-on, cannot turn away,
  and stalls in a local minimum; A\*/Hybrid route around the outside.
* **`ridge_corridor`** — a central radar with ridges shadowing an L-shaped
  corridor. The risk-aware optimum is the **longer shielded detour** (R̃≈0); a
  length-greedy path cuts through the exposed centre (R̃≈1).
* **`high_density`** — many radars; stresses A\* search cost vs the Hybrid's
  coarse global layer.
* **`pareto`** — a single radar with no shielding, used only by the Pareto
  sweep: lowering risk *requires* a longer detour, tracing a genuine front.

---

## Algorithm notes

**A\*** — state `(cell, arrival-heading)`, 8- or 16-connected. Edge cost is the
shared length+risk+kinematic cost; midpoint hazard is read from the precomputed
grid. Heuristic `ĥ = wL·‖p_g−n‖/L_ref` (Euclidean, **admissible** because the
non-negative risk/kinematic terms are dropped — it is not inflated). Open set is
a `heapq`. Every per-edge geometric quantity that does not depend on search order
is precomputed (turn-penalty table, per-move risk/climb arrays, heuristic grid)
so the hot loop is pure array indexing.

**APF** — `U = ½·ξ‖p−p_g‖² + η·Λ`, descended with kinematic clamping. The gain
ratio maps to the weights (`η/ξ = wR/wL`). Crucially, the step models a fixed-
wing vehicle: it **does not fly into rising potential** (speed scales with how
well the turn-limited heading aligns with the descent force) and **cannot turn
without moving** (`Δψ = κ·ds ≤ κ_max·ds`). The local-minimum detector reports
failure when net displacement over a window stalls short of the goal — **no
global-escape heuristic is added**, because capturing APF's local-minimum
failure is a key result.

**Hybrid** — coarse A\* produces a risk-safe global corridor; APF then tracks
the moving sub-goals (using the *same* `apf_step` kinematics) with a high
attraction:repulsion ratio, so the local layer smooths and enforces feasibility
while the global sub-goals prevent entrapment. It succeeds where pure APF fails,
at a small fraction of full-resolution A\*'s search cost.

---

## Key qualitative results

* **`concave_trap`** — APF fails (local minimum) on essentially every run;
  A\* and Hybrid succeed. Hybrid reaches the goal exploring ~2 orders of
  magnitude fewer nodes than full-resolution A\*.
* **`ridge_corridor`** — A\* and Hybrid take the shielded detour (R̃≈0); APF,
  when it succeeds, flies the exposed straight line (R̃≈1, large Φ).
* **`high_density`** — A\*'s expanded-node count and time grow with threat
  density; the Hybrid stays cheap.
* The **Pareto sweep** shows length rising and detection risk falling as `wR`
  increases — the expected length-vs-risk trade-off.

Exact mean ± std and Wilcoxon p-values are in `outputs/results_aggregate.csv`
and `outputs/wilcoxon_J.csv` (and printed to the console).

---

## Code architecture

```
uav_pathplan/
  config.py        # single Config dataclass: every parameter; coord mapping
  terrain.py       # fBm value-noise + parametric features; continuous h(x,y)
  risk.py          # radar hazard + manual LOS; grid + fast continuous Λ_field
  cost.py          # the shared cost J and per-edge helpers (the fairness core)
  scenarios.py     # the four scenario classes + pareto config, all seeded
  metrics.py       # per-run metric extraction + mean±std aggregation
  stats.py         # paired Wilcoxon tests (A* vs APF vs Hybrid)
  plotting.py      # overlays, 3D surface, time-vs-threats, Pareto, console table
  planners/
    __init__.py    # PlanResult
    astar.py       # heading-augmented A*
    apf.py         # APF + shared kinematically-clamped step (apf_step)
    hybrid.py      # coarse A* + local APF tracking
run_experiments.py # top-level driver (Monte-Carlo, CSVs, figures, tests)
dashboard/app.py   # Streamlit dashboard — thin layer that imports the above
smoke_*.py         # per-step build-order checkpoints
```

`dashboard/app.py` adds no simulation logic; the only library change it required was
making `plotting.plot_overlay`, `plot_surface_with_path`, and `plot_pareto`
accept `out_path=None` and a `close` flag and **return the figure** (so the GUI
can display/export live figures). These are backward-compatible —
`run_experiments.py` still passes `out_path` and is unchanged.

### Key parameters (all in `Config`, nothing hard-coded mid-function)

`seed`, `world_size`, `resolution`; terrain `num_octaves`/`persistence`/
`lacunarity`/`fractal_max`; `hills`/`ridges`; `clearance`/`altitude_mode`;
radar `C`/`R_max`/`mast_height`, `R0`, `los_step`; weights `wL`/`wR`/`wK`;
`R_min`/`gamma_max_deg`; A\* `astar_connectivity`; APF `apf_alpha`/`apf_eta`/
`apf_xi`/`apf_max_iter`/`apf_stuck_window`/`apf_stuck_disp`; Hybrid
`hybrid_coarsen`/`hybrid_attract`; `mc_runs`.
