# Terrain-Masked Risk-Aware Path Planning — A-Star, APF and a Hybrid Planner

Code and data for the paper *"Terrain-Masked Risk-Aware Path Planning for
Autonomous Aerial Vehicles: Benchmarking A-Star, Artificial Potential Field, and
a Hybrid Planner"* (submitted to *Defence Science Journal*).

## Overview

The framework compares three planners for a UAV flying nap-of-the-earth over
synthetic 2.5D terrain with radar threats whose coverage is masked by terrain
(line of sight):

1. **A-Star** — grid search whose state includes the arrival heading.
2. **APF** — artificial-potential-field descent with a turn-rate limit.
3. **Hybrid** — A-Star on a coarse grid gives a global corridor; APF-style
   local tracking follows it.

All three planners receive the **same terrain and hazard fields** for a given
run, and every resulting path is **evaluated with the same cost `J`**
(`uav_pathplan/cost.py: evaluate`). The planners do **not** all minimise `J`:
A-Star minimises an additive edge cost built from the same weights, APF follows
the gradient of a potential field, and the Hybrid combines both (see
[Planners](#planners)).

Terrain generation, line of sight, the hazard model, the cost function and the
planners are implemented with `numpy`; `scipy` supplies
`RegularGridInterpolator`, `gaussian_filter` and the Wilcoxon signed-rank test;
`pandas` writes the tables. No path-planning or robotics library is used.

## Install

```bash
python -m pip install -r requirements.txt            # pinned versions
python -m pip install -r requirements-dev.txt        # optional: pytest, ruff
```

Verified with CPython 3.13.7 (64-bit, Windows 11). The pinned numpy/pandas
releases require Python ≥ 3.11. Keep the pins when reproducing the paper — in
particular scipy, whose Wilcoxon implementation determines the p-values.

## Quick start

```bash
python run_experiments.py --quick     # 12 m grid, 4 runs per scenario, ~30 s
pytest                                # unit and smoke tests, ~40 s
```

Results go to `outputs/` (git-ignored).

## Reproduce the paper

```bash
python run_experiments.py --runs 20 --resolution 8 --seed 20240603
python scripts/compare_to_paper.py
```

or simply `bash scripts/reproduce_paper.sh` (Linux/macOS/Git Bash) or
`scripts\reproduce_paper.bat` (Windows), which run exactly these two commands
from the repository root.

* Publication settings: **8 m grid, 20 Monte-Carlo runs per scenario, master
  seed 20240603**. (`--full` is a 5 m / 20-run preset that was *not* used for
  the paper.)
* Expected runtime: about 3 minutes on the verification machine (183 s and
  186 s wall-clock in two runs, plus a few seconds for the comparison); see [Hardware used for timing](#hardware-used-for-timing).
* `compare_to_paper.py` aligns `outputs/results_aggregate.csv`,
  `outputs/wilcoxon_J.csv` (and `results_per_run.csv`) with `paper_assets/`
  and prints a PASS/FAIL table: every non-timing value must agree to 1e-9;
  timing and memory differences are listed separately, because they depend on
  the machine.
* `paper_assets/` holds the canonical results used in the paper and must not
  be edited.

## Repository structure

```
run_experiments.py        single entry point: Monte-Carlo runs, CSVs, statistics, figures
uav_pathplan/
  config.py               Config dataclass holding every parameter
  terrain.py              value-noise fBm terrain + Gaussian hills / ridges
  risk.py                 radar hazard, line of sight, C calibration, hazard grid
  cost.py                 cost J and kinematic metrics (shared evaluator)
  scenarios.py            open, concave_trap, ridge_corridor, high_density, pareto
  metrics.py              per-run metrics and mean/std aggregation
  stats.py                paired Wilcoxon tests with Holm adjustment
  plotting.py             overlays, 3D surfaces, time-vs-threats, Pareto figure
  planners/               astar.py, apf.py, hybrid.py (+ PlanResult)
tests/                    pytest suite (former smoke_*.py scripts + statistics test)
scripts/
  reproduce_paper.sh/.bat run the paper configuration and compare with paper_assets/
  compare_to_paper.py     PASS/FAIL comparison of outputs/ against paper_assets/
  diagnostics/            development diagnostics (not used for paper results)
dashboard/                Streamlit dashboard (app.py, .streamlit/config.toml)
paper_assets/             canonical results and figures of the paper (read-only)
```

## Command-line options (`run_experiments.py`)

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 20240603 | master seed; per-run streams derive from it |
| `--runs` | 10 | Monte-Carlo runs per scenario |
| `--resolution` | 10.0 | grid cell size (m) |
| `--scenarios` | all four | subset of `open concave_trap ridge_corridor high_density` |
| `--out` | `outputs` | output directory |
| `--mem` | off | record peak Python memory per planner call (`tracemalloc`) |
| `--quick` | off | 12 m grid, 4 runs, shorter sweeps |
| `--full` | off | 5 m grid, 20 runs (not the paper setting) |
| `--no-plots` | off | write CSVs only |
| `--figures-only` | off | regenerate the deterministic figures only |
| `--dpi` | 300 | figure resolution |
| `--format` | `png` | figure format: `png`, `tiff` (LZW) or `eps` |

Outputs: `results_per_run.csv`, `results_aggregate.csv` (mean/std over
successful runs, success rate over all runs), `wilcoxon_J.csv`,
`overlay_<scenario>`, `surface_<scenario>`, `time_vs_threats`, `pareto_LvsR`.

`wilcoxon_J.csv` holds paired two-sided Wilcoxon signed-rank tests on `J`
(`scipy.stats.wilcoxon` defaults) for A-Star–APF, A-Star–Hybrid and
APF–Hybrid in each scenario, over the runs in which both planners succeeded.
`p_value` is the raw value; `p_holm` is Holm-adjusted within each scenario over
the pairs that could be tested (the values reported in the paper).

## Models

**Terrain.** `h(x,y)` = value-noise fBm (5 octaves, persistence 0.42,
lacunarity 2, base lattice 3), rescaled to `[0, fractal_max]`, smoothed with a
Gaussian (σ = 18 m), plus scripted Gaussian hills and ridges.

**Altitude.** Nap-of-the-earth `z = h(x,y) + 50 m` for every planner.

**Hazard.** For radar `j` with phase centre at ground height + mast height,
`λ_j = V_j · C_j / max(R_j, R0)^4` if `R_j ≤ R_max,j`, else 0; `R_j` is the 3D
range and `V_j ∈ {0,1}` is line of sight, tested by sampling the radar–point
segment every `los_step` metres against the terrain. `Λ = Σ_j λ_j`.
`C_j` is calibrated (`risk.calibrate_C`) so that a straight transit through
the coverage disc at closest range `0.5 R_max` with clear line of sight has
detection probability 0.5. The `concave_trap` "radars" are an exception: they
use a fixed `C = 1e9` and act as a proxy obstacle wall, so `R̃` in that
scenario is not a calibrated detection probability.

## Metrics

All metrics are computed by `metrics.compute_metrics` / `cost.evaluate` on the
path a planner returns (`p_0 … p_N`, horizontal waypoints in metres), at the
altitude `z = h + clearance`.

| Column | Definition as implemented |
|---|---|
| `L` | `Σ_i ‖p_{i+1} − p_i‖`, horizontal length of the returned waypoints (m) |
| `Ltil` (L̃) | `(L − L_ref)/L_ref`, `L_ref = ‖p_N − p_0‖` — the straight line between the path's **own** first and last points |
| `Phi` (Φ) | `Σ_i Λ(m_i)·‖p_{i+1} − p_i‖`, `m_i` = segment midpoint; Λ is the exact line-of-sight hazard (not the grid) |
| `Rtil` (R̃) | `1 − exp(−Φ)` |
| `Lambda_peak` | `max_i Λ(m_i)` |
| `J` | `wL·L̃ + wR·R̃ + wK·K̃`, with `K̃ = mean_k [max(0,(κ_k−κ_max)/κ_max) + max(0,(|γ_k|−γ_max)/γ_max)]` over the interior vertices `k` of the resampled path (below); `κ_max = 1/R_min` |
| `kappa_peak` (κ_peak) | `max_k κ_k`, `κ_k = |Δψ_k| / (½(Δs_{k−1}+Δs_k))` (1/m) |
| `curv_violations` | number of resampled interior vertices with `κ_k > 1/R_min` |
| `gamma_peak_deg` (γ_peak) | `max_k |γ_k|` in degrees, `γ_k = atan2(Δz_k, Δs_k)` per resampled segment |
| `climb_violations` | number of resampled segments with `|γ_k| > γ_max` |
| `smoothness` | `Σ_k |Δψ_k|` over resampled interior vertices (rad) |
| `d_min` | minimum 3D distance from any returned waypoint to any radar phase centre (m) |
| `time_s` | wall-clock time of the planner call (`time.perf_counter`); the hazard-grid precomputation is excluded |
| `n_exp` / `n_iter` | A-Star expanded states (Hybrid: of its coarse A-Star) / APF and Hybrid iterations |
| `success` | A-Star: goal cell reached; APF/Hybrid: within 12 m of the goal |

**Resampling.** `κ`, `γ`, `K̃`, the violation counts and smoothness are
computed on a copy of the path resampled uniformly in arc length at
`kin_resample_ds` (0 → the grid resolution, i.e. 8 m in the paper), so the
definitions do not depend on each planner's waypoint spacing. `L`, `Φ` and `R̃`
use the original waypoints.

## Planners

* **A-Star** (`planners/astar.py`). State = (cell, arrival move); 16-connected
  moves (8 neighbours + 8 knight moves), hence 16 arrival-heading bins, which
  are not uniformly spaced in angle. It minimises the additive edge cost
  `wL·ℓ/L_ref + wR·½(H_a + H_b)·ℓ + wK·max(0,(γ−γ_max)/γ_max) + wK·max(0,(κ−κ_max)/κ_max)`,
  where `H` is the precomputed hazard grid (endpoint average, not the exact
  midpoint hazard used by `J`), `γ = atan2(|Δz|, ℓ)` over the edge, `κ` comes
  from the heading change between consecutive moves, and `L_ref` is the
  distance between the snapped start and goal cells. Penalties are summed over
  edges, whereas `J` averages `K̃`. Heuristic `wL·‖goal − n‖/L_ref`
  (admissible).
* **APF** (`planners/apf.py`). Each step moves along the force
  `F = ξ·(p_g − p)/‖p_g − p‖ − η·∇Λ_grid`, where `∇Λ_grid` is a central
  difference (2.5 m) of the bilinear interpolation of the hazard grid,
  `ξ = wL`, `η = wR`. The heading may change by at most `κ_max·s`, and the
  step length is `s = min(α, ‖p_g − p‖)·max(0, cos θ)` (θ = angle between the
  turn-limited heading and `F`, α = 5 m). No climb limit is enforced. Failure
  is declared when the net displacement over 60 iterations is below 8 m
  (local minimum) or after 6000 iterations; no escape heuristic is used.
* **Hybrid** (`planners/hybrid.py`). A-Star as above on a grid coarsened by
  `hybrid_coarsen = 4` (32 m at the 8 m resolution), then the APF step tracks
  the coarse waypoints as moving sub-goals with `ξ = hybrid_attract = 5`,
  `η = wR`, advancing when within `max(2α, 1.2·4·resolution)` (38.4 m at 8 m).

## Parameters

Values used for the paper run (`Config` defaults unless a scenario overrides
them). Items marked † are to be added to, or checked against, the paper's Table 1.

| `Config` field / source | Value | Meaning |
|---|---|---|
| `world_size` | 1000 m | square world side |
| `resolution` (CLI `--resolution`) | 8 m | grid cell (125 × 125) |
| `num_octaves`, `persistence`, `lacunarity`, `base_lattice` † | 5, 0.42, 2.0, 3 | fBm terrain |
| `fractal_max` † | 70 / 60 / 70 / 80 m | fBm relief: open / concave_trap / ridge_corridor / high_density (pareto: 40 m) |
| `terrain_smooth_sigma` † | 18 m | smoothing of the fBm base |
| `clearance`, `altitude_mode` | 50 m, `nap` | flight altitude `h + 50 m` |
| `wL`, `wR`, `wK` | 0.40, 0.45, 0.15 | cost weights |
| `R_min` | 60 m | minimum turn radius (`κ_max = 1/60 m⁻¹`) |
| `gamma_max_deg` | 15° | maximum flight-path angle |
| `kin_resample_ds` † | 0 → 8 m | resampling step for kinematic metrics |
| `R0` † | 50 m (concave_trap: 20 m) | near-field range floor |
| `RadarSpec.R_max` † | 220 / 70 / 520 / 190 m (pareto: 430 m) | maximum radar range per scenario |
| `RadarSpec.mast_height` † | 20 / 25 / 30 / 20 m (pareto: 25 m) | antenna height above ground |
| `RadarSpec.C` † | calibrated; concave_trap: 1e9 | power constant |
| `ref_detection_prob`, `ref_range_frac` † | 0.5, 0.5 | C calibration reference transit |
| `los_step` † | 5 m | line-of-sight sample step |
| `astar_connectivity` † | 16 | A-Star moves = 16 heading bins (`astar_num_headings` is not read by the code) |
| `hybrid_coarsen` † | 4 (32 m coarse cell) | Hybrid global grid factor |
| `hybrid_attract` † | 5.0 | Hybrid attraction gain ξ |
| `apf_xi`, `apf_eta` † | None → ξ = wL = 0.40, η = wR = 0.45 | APF gains |
| `apf_alpha` † | 5 m | APF step length α |
| `apf_fd_step`, `apf_goal_tol` † | 2.5 m, 12 m | gradient step, goal radius |
| `apf_stuck_window`, `apf_stuck_disp`, `apf_max_iter` † | 60, 8 m, 6000 | local-minimum detection |
| CLI `--runs`, `--seed` | 20, 20240603 | Monte-Carlo runs, master seed |

**Monte-Carlo randomisation per scenario** † (`scenarios.py`, uniform draws
from a stream seeded by `(master seed, scenario id, run)`):

| Scenario | Start (x, y) m | Goal (x, y) m | Other randomisation |
|---|---|---|---|
| `open` | U(60,160), U(60,160) | U(840,940), U(840,940) | new terrain seed per run; radar fixed at (700, 320) |
| `concave_trap` | U(130,180), U(480,520) | U(880,920), U(480,520) | each of the 23 wall radars jittered U(−6,6) m in x and y; terrain seed fixed (22222) |
| `ridge_corridor` | U(90,150), U(90,150) | U(850,920), U(850,920) | none (terrain seed 33333, radar at (500, 520)) |
| `high_density` | U(60,140), U(60,140) | U(860,940), U(860,940) | new terrain seed per run; 16 radars at U(200,800)² drawn from that seed |

The overlay/surface figures use the fixed (non-randomised) instance of each
scenario. `time_vs_threats` uses the fixed `high_density` layout with 2, 4, 8,
12, 16 and 20 radars; `pareto_LvsR` runs A-Star on the `pareto` scenario (one
unshielded radar) with `wR ∈ {0.05, 0.20, 0.35, 0.50, 0.65, 0.80}`,
`wK = 0.15`, `wL = 1 − wK − wR`.

## Limitations

* Terrain is synthetic 2.5D (fBm plus scripted features), not a real DEM.
* The hazard model is a simplified comparative model (`1/R⁴` with a range
  floor, binary line of sight, calibrated constant); it is not a radar
  detection model with clutter, RCS or multipath, and absolute detection
  probabilities should not be read from it.
* Flight altitude is fixed at 50 m above terrain; altitude is not optimised.
* Kinematics are checked through curvature and flight-path-angle limits only;
  no flight-dynamics or autopilot validation was performed.
* Timing is single-threaded Python wall-clock time and depends on the machine.

## Hardware used for timing

*To be completed by the authors:* CPU model, cores/threads, RAM, operating
system, Python version, and whether other load was present during the paper run.

## Interactive dashboard

```bash
cd dashboard
streamlit run app.py
```

Running from `dashboard/` applies `dashboard/.streamlit/config.toml`.
`streamlit run dashboard/app.py` from the root also works but ignores that
file. The dashboard re-uses the simulation code; it overrides the terrain seed
with the seed chosen in the sidebar, so its views are not the paper's figure
instances.

## Tests and development

```bash
pytest               # tests/
ruff check . && ruff format --check .
```

GitHub Actions (`.github/workflows/ci.yml`) installs the pinned dependencies,
runs ruff, pytest and `python run_experiments.py --quick`.

## Licence and citation

MIT licence (`LICENSE`). Citation metadata: `CITATION.cff`.
