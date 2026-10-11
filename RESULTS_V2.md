# Protocol v2 results

Extended, pre-registered evaluation of A-Star, APF, APF-E and Hybrid. The
radar/threat model is the v1 model, unchanged. The v1 results in
`paper_assets/` are untouched and still reproducible (§7).

## 1. What was run

| Item | Value |
|---|---|
| Command | `python run_experiments.py --protocol v2` |
| Protocol | `configs/protocol_v2.yaml`, status `registered`, sha256 `0e185f13…92789c`, committed in `003e422` **before** the evaluation run |
| Code commit of the run | `003e422efb03bf016b45095d27021faf8dc95697` (clean tree) |
| Evaluation (held-out) master seed | 20251011 — used once, for this run |
| Development seed | 1009 — APF-E tuning and the dress rehearsal only |
| Monte-Carlo runs | 50 per scenario × 4 scenarios × 3 resolutions; every instance planned once per planner and evaluated in 2 altitude modes (4 800 rows) |
| Resolutions | 5, 8, 10 m (8 m = reference for Tables 2 and 3) |
| Altitude modes | `nap`: z = h + 50 m. `rate_limited`: lowest z ≥ h + 50 m with \|dz/ds\| ≤ tan 15° (evaluation only; planners plan on the nap hazard grid) |
| APF-E | random-walk escape (Barraquand & Latombe, IJRR 10(6), 1991); max 20 escapes, random turn of 10–40 steps at 0.9 κ_max, then 30 straight steps; selected on the development seed (`configs/apfe_tuning_dev.csv`) |
| Outputs | `paper_assets_v2/` (CSV, figures PNG + TIFF at 300 dpi, `environment.json`, copy of the protocol) |
| Run time | 1 840 s for the whole suite |

**Hardware and software used for timing** (from `paper_assets_v2/environment.json`):
AMD Ryzen 7 5800H (16 logical CPUs), 13.9 GiB RAM, Windows 11 (10.0.26200),
CPython 3.13.7, numpy 2.3.4, scipy 1.17.1, pandas 3.0.2, matplotlib 3.10.9,
PyYAML 6.0.3; `OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`;
`time.perf_counter` around the planner call, one warm-up per planner and
resolution; the hazard-grid precomputation is timed separately (`setup_s`).
All runs were sequential on one machine; timings are indicative.

**Definitions that differ from v1** (details in README "Protocol v2"):
L_ref = ‖goal − start‖ of the scenario; L is the closed length including the
gaps between the path ends and start/goal; the flight-path angle uses arc
length; hazard and line of sight are evaluated at the altitude of each mode.
† marks `concave_trap`, whose R̃/Φ come from an uncalibrated proxy wall
(C = 1e9) and are a **relative exposure index**, not a detection probability;
they are not compared across scenarios.

## 2. Findings

Numbers refer to the 8 m grid and nap altitude unless stated; medians over
successful runs; p values are Holm-adjusted within (resolution, altitude mode,
scenario); r is the matched-pairs rank-biserial correlation (r > 0: the first
planner has the higher, i.e. worse, J).

1. **Success.** A-Star succeeded in all 600 planning problems (4 scenarios ×
   50 runs × 3 resolutions). APF failed in every concave_trap run (0/50 at each
   resolution; Wilson 95 % CI 0–7 %) and succeeded everywhere else. APF-E
   succeeded in 94 / 90 / 96 % of concave_trap runs at 5 / 8 / 10 m. Hybrid
   succeeded in 100 / 98 / 100 %: it **failed once** (concave_trap, run 14,
   8 m: local minimum 225 m before the goal).
2. **APF-E only differs from APF in concave_trap.** APF never stalled in the
   other scenarios, so APF-E never triggered an escape there and produced
   identical paths (450/450 nap runs with zero escapes have identical J).
   In concave_trap every successful APF-E path crossed the wall's exposure zone
   (R̃† = 1 in 45/45 runs at 8 m; mean 7.7 escapes, max 20), with median
   J = 1.64 [IQR 1.06] and L̃ = 2.97, against 0.080 for A-Star and 0.048 for
   Hybrid. The escape reaches the goal but not with low cost.
3. **open:** APF/APF-E had the lowest J (median ≈ 0; they fly almost straight
   lines), Hybrid 0.001 and A-Star 0.005. APF and APF-E beat Hybrid
   significantly (p = 1.6e-8, r = −0.84), and Hybrid beat A-Star (p = 8.9e-15,
   r = 1.00). This result is **unfavourable to Hybrid** relative to APF.
4. **concave_trap†:** Hybrid had lower J than A-Star (0.048 vs 0.080;
   p = 1.8e-5, r = 0.67, n = 49). In v1 the same comparison was not
   significant (p = 0.114, n = 20). Both distributions are bimodal: some paths
   clip the wall's exposure zone (R̃† > 0.5 in 3 A-Star and 4 Hybrid runs at
   8 m; 10 and 14 runs at 10 m; 1 and 0 runs at 5 m).
5. **ridge_corridor:** Hybrid 0.112 vs A-Star 0.122 (p = 1.2e-13, r = 0.98);
   APF/APF-E 0.651 (straight line through the radar's coverage, R̃ = 1,
   climb violations on 54.7 % of resampled segments). A-Star's path was
   exposed (R̃ > 0.05) in 6/50 runs at 8 m (6 at 5 m, 4 at 10 m); Hybrid's
   in none.
6. **high_density:** Hybrid 0.148 vs A-Star 0.197 (p = 4.1e-9, r = 0.86);
   R̃ medians 0.030 vs 0.151; APF/APF-E 0.454 (R̃ = 1).
7. **Friedman** tests across the four planners were significant in open,
   ridge_corridor and high_density at every resolution and altitude mode
   (Kendall's W 0.90–1.00); they could not be computed for concave_trap
   because APF never succeeds there. Because APF-E = APF outside
   concave_trap, two of the four "treatments" are identical in those tests.
8. **Computation time** (median, 8 m): A-Star 0.92–1.03 s in concave_trap,
   ridge_corridor and high_density (0.03 s in open); Hybrid 0.02–0.05 s;
   APF/APF-E 0.02 s (APF-E 0.05 s in concave_trap). At 5 m A-Star took
   3.2–3.4 s and Hybrid 0.04–0.12 s.
9. **Grid sensitivity.** Success rates were stable across 5/8/10 m except
   APF-E and Hybrid in concave_trap (finding 1). Hybrid's median J changed
   little (high_density 0.141 / 0.148 / 0.164; ridge_corridor 0.111 / 0.112 /
   0.117). A-Star's did not: in high_density its median J was 0.278 / 0.197 /
   0.186 at 5 / 8 / 10 m, driven by R̃ (median 0.343 / 0.151 / 0.075), so the
   **finer grid gave A-Star worse paths** under J. A plausible cause is that
   A-Star's per-edge turn penalty is summed over more, shorter edges on a
   finer grid; this was **not verified**. The ranking Hybrid < A-Star in J held
   at every resolution in ridge_corridor, high_density and concave_trap.
10. **Altitude modes.** Rate-limited altitude removed all climb violations by
    construction (0 in every run) and left A-Star and Hybrid almost unchanged
    (ridge_corridor Hybrid J 0.112 → 0.106, from the removed climb penalty).
    It changed APF/APF-E in ridge_corridor substantially: J 0.651 → 0.290 and
    R̃ 1.000 → 0.645. The straight path then flies at 185 m mean AGL. Its
    minimum 3D distance to the radar rises from 27.5 m to 204.6 m and the peak
    hazard falls from 1.32 to 0.0047 (medians). Under this model's 1/R⁴ law,
    the larger slant range outweighs the loss of terrain masking. The planners'
    ranking in J was unchanged in every scenario.

## 3. Where v2 differs from v1

v1: 20 runs, 8 m, nap; terrain randomised only in open and high_density (fixed
terrain in concave_trap and ridge_corridor), start/goal randomised everywhere,
path-endpoint L_ref. v2: 50 runs, terrain, start/goal and structure randomised
in every scenario, scenario L_ref with closed L. Means at 8 m, nap:

| scenario | planner | success % v1 / v2 | J mean v1 / v2 | R̃ mean v1 / v2 |
|---|---|---|---|---|
| open | A-Star | 100 / 100 | 0.002 / 0.005 | 0.000 / 0.000 |
| open | APF | 100 / 100 | 0.001 / 0.001 | 0.001 / 0.001 |
| open | Hybrid | 100 / 100 | 0.001 / 0.001 | 0.000 / 0.000 |
| concave_trap† | A-Star | 100 / 100 | 0.142 / 0.102 | 0.150 / 0.060 |
| concave_trap† | APF | 0 / 0 | – / – | – / – |
| concave_trap† | Hybrid | 100 / 98 | 0.138 / 0.086 | 0.200 / 0.082 |
| ridge_corridor | A-Star | 100 / 100 | 0.133 / 0.133 | 0.000 / 0.029 |
| ridge_corridor | APF | 100 / 100 | 0.647 / 0.651 | 1.000 / 1.000 |
| ridge_corridor | Hybrid | 100 / 100 | 0.116 / 0.113 | 0.000 / 0.000 |
| high_density | A-Star | 100 / 100 | 0.229 / 0.206 | 0.193 / 0.154 |
| high_density | APF | 100 / 100 | 0.453 / 0.455 | 1.000 / 1.000 |
| high_density | Hybrid | 100 / 100 | 0.157 / 0.152 | 0.028 / 0.039 |

* **concave_trap A-Star vs Hybrid** becomes significant (v1 p = 0.114 n.s.,
  n = 20; v2 Holm p = 1.8e-5, n = 49, r = 0.67).
* **Hybrid is no longer 100 % successful**: 1 failure in 50 concave_trap runs
  at 8 m (v1: 20/20).
* **open A-Star J doubles** (0.002 → 0.005). This is consistent with the
  closed-L definition: A-Star starts and ends at snapped cell centres, and the
  gaps now count as length.
* **ridge_corridor A-Star** is exposed in some runs (R̃ mean 0.000 → 0.029),
  because v2 jitters the ridges and the radar. Hybrid stays unexposed.
* The direction of every A-Star/APF/Hybrid pairwise difference is the same as
  in v1; v2 adds effect sizes, which are large (|r| ≥ 0.67) in every
  significant comparison.

## 4. What did not work, was dropped, or needs care

* **APF-E is not a useful competitor outside concave_trap.** It reduces to APF
  there, and in concave_trap it escapes at a very high cost (finding 2). The
  pre-registered selection rule ranked success first, and the selected
  parameters had one of the highest median J values on the development set
  (0.587 vs 0.424 for the best-J grid point).
* **Friedman tests** are not computable for concave_trap. Elsewhere they
  include two identical planners (APF, APF-E).
* **The altitude modes are evaluation-only.** Planners optimise on the nap
  hazard grid, so the rate-limited results score nap-planned paths; no planner
  is altitude-aware.
* **Dropped from v2:** the Pareto sweep, the time-vs-threat-count figure and
  the 3D surface figures (v1 has them in `paper_assets/`); the Streamlit
  dashboard does not support v2; CI runs the v2 unit tests but not the
  30-minute suite.
* **Overlay figures:** where APF-E never escaped, its path coincides with APF's
  and is drawn on top of it.
* **Not verified:** the cause of A-Star's grid sensitivity (finding 9).
* **Bugs found before registration** (in unit tests and a development-seed
  dress rehearsal; fixed, then the protocol was registered): a rounding issue
  counted segments flown exactly at γ_max as climb violations (fixed with a
  1e-9 relative slope margin), and Wilson bounds at 0 %/100 % success could lie
  ~1e-17 on the wrong side of the rate. **No bug was found after the evaluation run**, so nothing was rerun
  or retuned.
* Timings come from one laptop CPU; absolute values will differ elsewhere.

## 5. Figures (`paper_assets_v2/figures/`, PNG + TIFF, 300 dpi, no titles)

`box_J_nap`, `box_J_rate_limited`, `box_Rtil_nap`, `box_Rtil_rate_limited`
(8 m), `success_rates` (Wilson 95 % CI), `grid_J_nap`, `grid_J_rate_limited`,
`grid_time`, `altitude_J`, `altitude_Rtil`, `overlay_<scenario>` (run 0,
8 m, nap).

## 6. Tables

Table-ready CSVs: `table2_main_results.csv`, `table3_pairwise_tests.csv`,
`table3_friedman.csv`, `table_grid_sensitivity.csv`; complete statistics in
`results_aggregate.csv` (mean, SD, median, Q1, Q3, IQR of every metric;
Wilson CI), `stats_wilcoxon_J.csv`, `stats_friedman_J.csv`; raw runs in
`results_per_run.csv`. The tables below are generated by
`scripts/results_v2_tables.py`. Treating these as the paper's Table 2 and
Table 3 is an assumption; the manuscript is not in the repository.

### Table 2 — main results, 8 m grid, nap altitude

| scenario | planner | success % [95 % CI] | J median [IQR] | J mean ± SD | L̃ median | R̃ median | Φ median | κ-viol. % | γ-viol. % | time median (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| open | A-Star | 100 [93, 100] | 0.005 [0.002] | 0.005 ± 0.002 | 0.009 | 0.000 | 0.000 | 0.9 | 0.0 | 0.030 |
| open | APF | 100 [93, 100] | 0.000 [8.47e-17] | 5.46e-04 ± 0.003 | 0.000 | 0.000 | 0.000 | 0.0 | 4.26e-02 | 0.018 |
| open | APF-E | 100 [93, 100] | 0.000 [8.47e-17] | 5.46e-04 ± 0.003 | 0.000 | 0.000 | 0.000 | 0.0 | 4.26e-02 | 0.018 |
| open | Hybrid | 100 [93, 100] | 0.001 [0.001] | 0.001 ± 0.001 | 0.003 | 0.000 | 0.000 | 0.2 | 2.82e-02 | 0.025 |
| concave_trap † | A-Star | 100 [93, 100] | 0.080 [0.018] | 0.102 ± 0.111 | 0.175 | 0.000 | 0.000 | 4.2 | 1.82e-02 | 0.919 |
| concave_trap † | APF | 0 [0, 7] | – [–] | – ± – | – | – | – | – | – | – |
| concave_trap † | APF-E | 90 [79, 96] | 1.640 [1.058] | 1.654 ± 0.759 | 2.971 | 1.000 | 3.74e+05 | 4.8 | 1.92e-02 | 0.054 |
| concave_trap † | Hybrid | 98 [90, 100] | 0.048 [0.011] | 0.086 ± 0.125 | 0.119 | 0.000 | 0.000 | 4.2 | 0.0 | 0.022 |
| ridge_corridor | A-Star | 100 [93, 100] | 0.122 [0.015] | 0.133 ± 0.036 | 0.284 | 0.000 | 0.000 | 3.0 | 0.8 | 1.020 |
| ridge_corridor | APF | 100 [93, 100] | 0.651 [0.012] | 0.651 ± 0.011 | 1.07e-05 | 1.000 | 142.334 | 0.0 | 54.7 | 0.017 |
| ridge_corridor | APF-E | 100 [93, 100] | 0.651 [0.012] | 0.651 ± 0.011 | 1.07e-05 | 1.000 | 142.334 | 0.0 | 54.7 | 0.017 |
| ridge_corridor | Hybrid | 100 [93, 100] | 0.112 [0.011] | 0.113 ± 0.009 | 0.264 | 0.000 | 0.000 | 3.4 | 9.6 | 0.050 |
| high_density | A-Star | 100 [93, 100] | 0.197 [0.080] | 0.206 ± 0.063 | 0.329 | 0.151 | 0.164 | 2.9 | 0.2 | 1.025 |
| high_density | APF | 100 [93, 100] | 0.454 [0.007] | 0.455 ± 0.005 | 2.08e-07 | 1.000 | 26.719 | 0.0 | 13.2 | 0.018 |
| high_density | APF-E | 100 [93, 100] | 0.454 [0.007] | 0.455 ± 0.005 | 2.08e-07 | 1.000 | 26.719 | 0.0 | 13.2 | 0.018 |
| high_density | Hybrid | 100 [93, 100] | 0.148 [0.053] | 0.152 ± 0.037 | 0.343 | 0.030 | 0.030 | 2.1 | 0.4 | 0.052 |

### Table 2b — main results, 8 m grid, rate-limited altitude

| scenario | planner | success % [95 % CI] | J median [IQR] | J mean ± SD | L̃ median | R̃ median | Φ median | κ-viol. % | γ-viol. % | time median (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| open | A-Star | 100 [93, 100] | 0.005 [0.002] | 0.005 ± 0.002 | 0.009 | 0.000 | 0.000 | 0.9 | 0.0 | 0.030 |
| open | APF | 100 [93, 100] | 0.000 [1.18e-16] | 5.43e-04 ± 0.003 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.018 |
| open | APF-E | 100 [93, 100] | 0.000 [1.18e-16] | 5.43e-04 ± 0.003 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 | 0.018 |
| open | Hybrid | 100 [93, 100] | 0.001 [0.001] | 0.001 ± 0.001 | 0.003 | 0.000 | 0.000 | 0.2 | 0.0 | 0.025 |
| concave_trap † | A-Star | 100 [93, 100] | 0.080 [0.018] | 0.102 ± 0.111 | 0.175 | 0.000 | 0.000 | 4.2 | 0.0 | 0.919 |
| concave_trap † | APF | 0 [0, 7] | – [–] | – ± – | – | – | – | – | – | – |
| concave_trap † | APF-E | 90 [79, 96] | 1.640 [1.058] | 1.654 ± 0.759 | 2.971 | 1.000 | 3.74e+05 | 4.8 | 0.0 | 0.054 |
| concave_trap † | Hybrid | 98 [90, 100] | 0.048 [0.011] | 0.086 ± 0.125 | 0.119 | 0.000 | 0.000 | 4.2 | 0.0 | 0.022 |
| ridge_corridor | A-Star | 100 [93, 100] | 0.121 [0.015] | 0.133 ± 0.036 | 0.284 | 0.000 | 0.000 | 3.0 | 0.0 | 1.020 |
| ridge_corridor | APF | 100 [93, 100] | 0.290 [0.077] | 0.289 ± 0.051 | 1.07e-05 | 0.645 | 1.037 | 0.0 | 0.0 | 0.017 |
| ridge_corridor | APF-E | 100 [93, 100] | 0.290 [0.077] | 0.289 ± 0.051 | 1.07e-05 | 0.645 | 1.037 | 0.0 | 0.0 | 0.017 |
| ridge_corridor | Hybrid | 100 [93, 100] | 0.106 [0.011] | 0.107 ± 0.010 | 0.264 | 0.000 | 0.000 | 3.4 | 0.0 | 0.050 |
| high_density | A-Star | 100 [93, 100] | 0.197 [0.080] | 0.206 ± 0.063 | 0.329 | 0.151 | 0.164 | 2.9 | 0.0 | 1.025 |
| high_density | APF | 100 [93, 100] | 0.450 [1.21e-07] | 0.450 ± 3.16e-04 | 2.08e-07 | 1.000 | 26.480 | 0.0 | 0.0 | 0.018 |
| high_density | APF-E | 100 [93, 100] | 0.450 [1.21e-07] | 0.450 ± 3.16e-04 | 2.08e-07 | 1.000 | 26.480 | 0.0 | 0.0 | 0.018 |
| high_density | Hybrid | 100 [93, 100] | 0.148 [0.053] | 0.152 ± 0.037 | 0.343 | 0.030 | 0.030 | 2.1 | 0.0 | 0.052 |

### Table 3 — paired Wilcoxon tests on J, 8 m, nap

| scenario | pair (a vs b) | n | median J a / b | W | p | p (Holm) | r (rank-biserial) | note |
|---|---|---|---|---|---|---|---|---|
| open | A-Star vs APF | 50 | 0.005 / 0.000 | 49.0 | 4.96e-11 | 1.98e-10 | 0.92 |  |
| open | A-Star vs APF-E | 50 | 0.005 / 0.000 | 49.0 | 4.96e-11 | 1.98e-10 | 0.92 |  |
| open | A-Star vs Hybrid | 50 | 0.005 / 0.001 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| open | APF vs APF-E | 50 | 0.000 / 0.000 | – | – | – | – | identical paired values |
| open | APF vs Hybrid | 50 | 0.000 / 0.001 | 99.0 | 8.23e-09 | 1.65e-08 | -0.84 |  |
| open | APF-E vs Hybrid | 50 | 0.000 / 0.001 | 99.0 | 8.23e-09 | 1.65e-08 | -0.84 |  |
| concave_trap | A-Star vs APF | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | A-Star vs APF-E | 45 | 0.080 / 1.640 | 0.0 | 5.68e-14 | 1.71e-13 | -1.00 |  |
| concave_trap | A-Star vs Hybrid | 49 | 0.080 / 0.048 | 202.0 | 1.79e-05 | 1.79e-05 | 0.67 |  |
| concave_trap | APF vs APF-E | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | APF vs Hybrid | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | APF-E vs Hybrid | 44 | 1.642 / 0.047 | 0.0 | 1.14e-13 | 2.27e-13 | 1.00 |  |
| ridge_corridor | A-Star vs APF | 50 | 0.122 / 0.651 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| ridge_corridor | A-Star vs APF-E | 50 | 0.122 / 0.651 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| ridge_corridor | A-Star vs Hybrid | 50 | 0.122 / 0.112 | 12.0 | 1.24e-13 | 1.24e-13 | 0.98 |  |
| ridge_corridor | APF vs APF-E | 50 | 0.651 / 0.651 | – | – | – | – | identical paired values |
| ridge_corridor | APF vs Hybrid | 50 | 0.651 / 0.112 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| ridge_corridor | APF-E vs Hybrid | 50 | 0.651 / 0.112 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| high_density | A-Star vs APF | 50 | 0.197 / 0.454 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| high_density | A-Star vs APF-E | 50 | 0.197 / 0.454 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| high_density | A-Star vs Hybrid | 50 | 0.197 / 0.148 | 91.0 | 4.06e-09 | 4.06e-09 | 0.86 |  |
| high_density | APF vs APF-E | 50 | 0.454 / 0.454 | – | – | – | – | identical paired values |
| high_density | APF vs Hybrid | 50 | 0.454 / 0.148 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| high_density | APF-E vs Hybrid | 50 | 0.454 / 0.148 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |

### Table 3b — paired Wilcoxon tests on J, 8 m, rate-limited

| scenario | pair (a vs b) | n | median J a / b | W | p | p (Holm) | r (rank-biserial) | note |
|---|---|---|---|---|---|---|---|---|
| open | A-Star vs APF | 50 | 0.005 / 0.000 | 49.0 | 4.96e-11 | 1.98e-10 | 0.92 |  |
| open | A-Star vs APF-E | 50 | 0.005 / 0.000 | 49.0 | 4.96e-11 | 1.98e-10 | 0.92 |  |
| open | A-Star vs Hybrid | 50 | 0.005 / 0.001 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| open | APF vs APF-E | 50 | 0.000 / 0.000 | – | – | – | – | identical paired values |
| open | APF vs Hybrid | 50 | 0.000 / 0.001 | 99.0 | 8.23e-09 | 1.65e-08 | -0.84 |  |
| open | APF-E vs Hybrid | 50 | 0.000 / 0.001 | 99.0 | 8.23e-09 | 1.65e-08 | -0.84 |  |
| concave_trap | A-Star vs APF | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | A-Star vs APF-E | 45 | 0.080 / 1.640 | 0.0 | 5.68e-14 | 1.71e-13 | -1.00 |  |
| concave_trap | A-Star vs Hybrid | 49 | 0.080 / 0.048 | 202.0 | 1.79e-05 | 1.79e-05 | 0.67 |  |
| concave_trap | APF vs APF-E | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | APF vs Hybrid | 0 | – | – | – | – | – | no runs in which both succeeded |
| concave_trap | APF-E vs Hybrid | 44 | 1.642 / 0.047 | 0.0 | 1.14e-13 | 2.27e-13 | 1.00 |  |
| ridge_corridor | A-Star vs APF | 50 | 0.121 / 0.290 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| ridge_corridor | A-Star vs APF-E | 50 | 0.121 / 0.290 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| ridge_corridor | A-Star vs Hybrid | 50 | 0.121 / 0.106 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| ridge_corridor | APF vs APF-E | 50 | 0.290 / 0.290 | – | – | – | – | identical paired values |
| ridge_corridor | APF vs Hybrid | 50 | 0.290 / 0.106 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| ridge_corridor | APF-E vs Hybrid | 50 | 0.290 / 0.106 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| high_density | A-Star vs APF | 50 | 0.197 / 0.450 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| high_density | A-Star vs APF-E | 50 | 0.197 / 0.450 | 0.0 | 1.78e-15 | 8.88e-15 | -1.00 |  |
| high_density | A-Star vs Hybrid | 50 | 0.197 / 0.148 | 91.0 | 4.06e-09 | 4.06e-09 | 0.86 |  |
| high_density | APF vs APF-E | 50 | 0.450 / 0.450 | – | – | – | – | identical paired values |
| high_density | APF vs Hybrid | 50 | 0.450 / 0.148 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |
| high_density | APF-E vs Hybrid | 50 | 0.450 / 0.148 | 0.0 | 1.78e-15 | 8.88e-15 | 1.00 |  |

### Friedman tests on J (runs where all four planners succeed)

| resolution (m) | altitude | scenario | n (all 4 succeed) | χ² | p | Kendall's W | note |
|---|---|---|---|---|---|---|---|
| 5 | nap | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 5 | rate_limited | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 5 | nap | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 5 | rate_limited | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 5 | nap | ridge_corridor | 50 | 150.00 | 2.63e-32 | 1.000 |  |
| 5 | rate_limited | ridge_corridor | 50 | 150.00 | 2.63e-32 | 1.000 |  |
| 5 | nap | high_density | 50 | 148.69 | 5.04e-32 | 0.991 |  |
| 5 | rate_limited | high_density | 50 | 144.72 | 3.63e-31 | 0.965 |  |
| 8 | nap | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 8 | rate_limited | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 8 | nap | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 8 | rate_limited | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 8 | nap | ridge_corridor | 50 | 148.69 | 5.04e-32 | 0.991 |  |
| 8 | rate_limited | ridge_corridor | 50 | 150.00 | 2.63e-32 | 1.000 |  |
| 8 | nap | high_density | 50 | 140.16 | 3.49e-30 | 0.934 |  |
| 8 | rate_limited | high_density | 50 | 140.16 | 3.49e-30 | 0.934 |  |
| 10 | nap | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 10 | rate_limited | open | 50 | 135.84 | 2.98e-29 | 0.906 |  |
| 10 | nap | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 10 | rate_limited | concave_trap | 0 | – | – | – | fewer than 2 runs in which all planners succeeded |
| 10 | nap | ridge_corridor | 50 | 139.33 | 5.26e-30 | 0.929 |  |
| 10 | rate_limited | ridge_corridor | 50 | 150.00 | 2.63e-32 | 1.000 |  |
| 10 | nap | high_density | 50 | 134.29 | 6.42e-29 | 0.895 |  |
| 10 | rate_limited | high_density | 50 | 134.29 | 6.42e-29 | 0.895 |  |

### Grid sensitivity (nap): values at 5 / 8 / 10 m

| scenario | planner | success % (5/8/10 m) | median J | median time (s) |
|---|---|---|---|---|
| open | A-Star | 100 / 100 / 100 | 0.005 / 0.005 / 0.005 | 0.089 / 0.030 / 0.017 |
| open | APF | 100 / 100 / 100 | 0.000 / 0.000 / 0.000 | 0.018 / 0.018 / 0.018 |
| open | APF-E | 100 / 100 / 100 | 0.000 / 0.000 / 0.000 | 0.018 / 0.018 / 0.018 |
| open | Hybrid | 100 / 100 / 100 | 9.96e-04 / 0.001 / 0.001 | 0.036 / 0.025 / 0.023 |
| concave_trap | A-Star | 100 / 100 / 100 | 0.084 / 0.080 / 0.055 | 3.366 / 0.919 / 0.333 |
| concave_trap | APF | 0 / 0 / 0 | – / – / – | – / – / – |
| concave_trap | APF-E | 94 / 90 / 96 | 1.531 / 1.640 / 1.538 | 0.051 / 0.054 / 0.050 |
| concave_trap | Hybrid | 100 / 98 / 100 | 0.046 / 0.048 / 0.052 | 0.036 / 0.022 / 0.019 |
| ridge_corridor | A-Star | 100 / 100 / 100 | 0.123 / 0.122 / 0.121 | 3.299 / 1.020 / 0.538 |
| ridge_corridor | APF | 100 / 100 / 100 | 0.650 / 0.651 / 0.651 | 0.017 / 0.017 / 0.017 |
| ridge_corridor | APF-E | 100 / 100 / 100 | 0.650 / 0.651 / 0.651 | 0.017 / 0.017 / 0.017 |
| ridge_corridor | Hybrid | 100 / 100 / 100 | 0.111 / 0.112 / 0.117 | 0.102 / 0.050 / 0.043 |
| high_density | A-Star | 100 / 100 / 100 | 0.278 / 0.197 / 0.186 | 3.169 / 1.025 / 0.560 |
| high_density | APF | 100 / 100 / 100 | 0.454 / 0.454 / 0.454 | 0.018 / 0.018 / 0.018 |
| high_density | APF-E | 100 / 100 / 100 | 0.454 / 0.454 / 0.454 | 0.018 / 0.018 / 0.018 |
| high_density | Hybrid | 100 / 100 / 100 | 0.141 / 0.148 / 0.164 | 0.115 / 0.052 / 0.046 |

### Grid sensitivity (rate_limited): values at 5 / 8 / 10 m

| scenario | planner | success % (5/8/10 m) | median J | median time (s) |
|---|---|---|---|---|
| open | A-Star | 100 / 100 / 100 | 0.005 / 0.005 / 0.005 | 0.089 / 0.030 / 0.017 |
| open | APF | 100 / 100 / 100 | 0.000 / 0.000 / 0.000 | 0.018 / 0.018 / 0.018 |
| open | APF-E | 100 / 100 / 100 | 0.000 / 0.000 / 0.000 | 0.018 / 0.018 / 0.018 |
| open | Hybrid | 100 / 100 / 100 | 9.96e-04 / 0.001 / 0.001 | 0.036 / 0.025 / 0.023 |
| concave_trap | A-Star | 100 / 100 / 100 | 0.084 / 0.080 / 0.055 | 3.366 / 0.919 / 0.333 |
| concave_trap | APF | 0 / 0 / 0 | – / – / – | – / – / – |
| concave_trap | APF-E | 94 / 90 / 96 | 1.531 / 1.640 / 1.538 | 0.051 / 0.054 / 0.050 |
| concave_trap | Hybrid | 100 / 98 / 100 | 0.046 / 0.048 / 0.052 | 0.036 / 0.022 / 0.019 |
| ridge_corridor | A-Star | 100 / 100 / 100 | 0.123 / 0.121 / 0.121 | 3.299 / 1.020 / 0.538 |
| ridge_corridor | APF | 100 / 100 / 100 | 0.290 / 0.290 / 0.293 | 0.017 / 0.017 / 0.017 |
| ridge_corridor | APF-E | 100 / 100 / 100 | 0.290 / 0.290 / 0.293 | 0.017 / 0.017 / 0.017 |
| ridge_corridor | Hybrid | 100 / 100 / 100 | 0.110 / 0.106 / 0.106 | 0.102 / 0.050 / 0.043 |
| high_density | A-Star | 100 / 100 / 100 | 0.278 / 0.197 / 0.186 | 3.169 / 1.025 / 0.560 |
| high_density | APF | 100 / 100 / 100 | 0.450 / 0.450 / 0.450 | 0.018 / 0.018 / 0.018 |
| high_density | APF-E | 100 / 100 / 100 | 0.450 / 0.450 / 0.450 | 0.018 / 0.018 / 0.018 |
| high_density | Hybrid | 100 / 100 / 100 | 0.141 / 0.148 / 0.164 | 0.115 / 0.052 / 0.046 |

### Altitude modes at 8 m (RL = rate-limited)

| scenario | planner | R̃ median nap / RL | J median nap / RL | γ-viol. % nap / RL | mean AGL (m) RL |
|---|---|---|---|---|---|
| open | A-Star | 0.000 / 0.000 | 0.005 / 0.005 | 0.0 / 0.0 | 50.0 |
| open | APF | 0.000 / 0.000 | 0.000 / 0.000 | 4.26e-02 / 0.0 | 50.0 |
| open | APF-E | 0.000 / 0.000 | 0.000 / 0.000 | 4.26e-02 / 0.0 | 50.0 |
| open | Hybrid | 0.000 / 0.000 | 0.001 / 0.001 | 2.82e-02 / 0.0 | 50.0 |
| concave_trap † | A-Star | 0.000 / 0.000 | 0.080 / 0.080 | 1.82e-02 / 0.0 | 50.0 |
| concave_trap † | APF | – / – | – / – | – / – | – |
| concave_trap † | APF-E | 1.000 / 1.000 | 1.640 / 1.640 | 1.92e-02 / 0.0 | 50.0 |
| concave_trap † | Hybrid | 0.000 / 0.000 | 0.048 / 0.048 | 0.0 / 0.0 | 50.0 |
| ridge_corridor | A-Star | 0.000 / 0.000 | 0.122 / 0.121 | 0.8 / 0.0 | 50.0 |
| ridge_corridor | APF | 1.000 / 0.645 | 0.651 / 0.290 | 54.7 / 0.0 | 185.2 |
| ridge_corridor | APF-E | 1.000 / 0.645 | 0.651 / 0.290 | 54.7 / 0.0 | 185.2 |
| ridge_corridor | Hybrid | 0.000 / 0.000 | 0.112 / 0.106 | 9.6 / 0.0 | 50.6 |
| high_density | A-Star | 0.151 / 0.151 | 0.197 / 0.197 | 0.2 / 0.0 | 50.0 |
| high_density | APF | 1.000 / 1.000 | 0.454 / 0.450 | 13.2 / 0.0 | 50.7 |
| high_density | APF-E | 1.000 / 1.000 | 0.454 / 0.450 | 13.2 / 0.0 | 50.7 |
| high_density | Hybrid | 0.030 / 0.030 | 0.148 / 0.148 | 0.4 / 0.0 | 50.0 |

## 7. v1 reproducibility after the v2 changes

`bash scripts/reproduce_paper.sh` (default `--protocol v1`, pinned environment)
on the code containing all v2 changes: **OVERALL PASS, 55/55 non-timing columns
identical to `paper_assets/`** (max |diff| = 0; run time 183 s). `paper_assets/`
was not modified (`git diff v1.0-paper -- paper_assets` is empty). The v1 state
is tagged `v1.0-paper`.

## 8. Commit hash mapping

Hashes changed because the author/committer e-mail metadata was rewritten for privacy (private address replaced by the GitHub noreply address); author and committer dates are unchanged, and trees are unchanged except that the same address was also replaced in `.mailmap` in the 16 commits that contain that file. The initial commit `79c2a2a` is unchanged.

* The **pre-registration commit `003e422` is now `6e10688`**; its author and
  committer date are unchanged (2026-10-11T22:05:04+07:00), so the order
  "registration before evaluation run" is preserved.
* `paper_assets_v2/environment.json` is unmodified run output and still records
  the pre-rewrite hash `003e422efb03bf016b45095d27021faf8dc95697`; it
  corresponds to `6e10688`. The results commit `4698724` is now `f2e0720`.

| old | new | tree | commit |
|---|---|---|---|
| `f30456a` | `cbc5df6` | identical | Move smoke_*.py to tests/ and convert them to pytest |
| `6a34f5a` | `13ad6c1` | identical | Move diagnostic scripts to scripts/diagnostics/ |
| `421992f` | `08235cf` | identical | Move the Streamlit dashboard to dashboard/ |
| `a56e63b` | `e3b23aa` | identical | Pin exact dependency versions and record the Python version |
| `9428091` | `7ce8f01` | identical | Add Holm-adjusted p-values to the paired Wilcoxon output |
| `9d11280` | `00db835` | identical | Figures: --dpi default 300, --format png/tiff/eps, A-Star label |
| `276ab1c` | `cc7c8e2` | identical | Add paper-reproduction scripts and compare_to_paper.py |
| `11f8485` | `00e2600` | identical | Add ruff configuration and format the code |
| `4508d34` | `fcf449a` | identical | Correct docstrings that misdescribe the planners' objectives |
| `0355b2c` | `08d82ea` | identical | Rewrite README for the paper release |
| `67cb681` | `db1c4ab` | identical | Add MIT licence, CITATION.cff and CHANGELOG (v1.0-paper) |
| `d4e06ba` | `ba96a4a` | identical | Add GitHub Actions CI |
| `da4aa63` | `264c1dc` | identical | README: clarify what the dagger in the parameter table means |
| `e6da746` | `931c2fa` | identical | Add CLEANUP_REPORT.md |
| `45e32fe` | `9fadfea` | differs only in `.mailmap` | Add .mailmap for the typo address on the initial commit |
| `c544fb1` | `a5a7b1c` | differs only in `.mailmap` | test_cost: hand values for the resampled kinematic definition |
| `e33dcb6` | `429cbb5` | differs only in `.mailmap` | paper_assets/README.txt: correct dpi note and Pareto source |
| `89681ef` | `9b8a650` | differs only in `.mailmap` | README: document the methods exactly as implemented |
| `5ee15a5` | `a445ce4` | differs only in `.mailmap` | CLEANUP_REPORT: pre-push status and rebased commit hashes |
| `01b407d` | `95e22af` | differs only in `.mailmap` | v2: altitude modes and v2 path evaluator (L_ref fix) |
| `a1387c4` | `8f88cbf` | differs only in `.mailmap` | Add APF-E: APF with random-walk local-minimum escape |
| `6fa287a` | `5a5150c` | differs only in `.mailmap` | v2: draft protocol file and Monte-Carlo scenario builders |
| `915342f` | `27ca568` | differs only in `.mailmap` | v2: runner, statistics, figures and --protocol switch |
| `4342419` | `535dcb7` | differs only in `.mailmap` | v2: APF-E tuning script (development seed only) |
| `b41a19f` | `0e0df9a` | differs only in `.mailmap` | v2: fix Wilson bounds at 0%/100% success; figure legibility |
| `003e422` | `6e10688` | differs only in `.mailmap` | Pre-register protocol v2 (frozen before the evaluation run) |
| `dbf4a64` | `b1a1057` | differs only in `.mailmap` | v2: script rendering the result tables as Markdown |
| `4698724` | `f2e0720` | differs only in `.mailmap` | Add protocol-v2 results (held-out evaluation seed 20251011) |
| `97275a0` | `7e9c1ed` | differs only in `.mailmap` | Add RESULTS_V2.md |
| `4ac3836` | `908d92e` | differs only in `.mailmap` | Document protocol v2 (README, CHANGELOG) and add a v2 CI smoke run |
