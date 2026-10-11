PAPER ASSETS — canonical results
Source run: python run_experiments.py --runs 20 --resolution 8 --seed 20240603
(20 Monte-Carlo runs/scenario, 8 m grid, calibrated radar risk)

FIGURES
  overlay_open.png            2D paths (A*/APF/Hybrid) on hazard map — open terrain
  overlay_concave_trap.png    2D paths — concave trap (APF fails: local minimum)
  overlay_ridge_corridor.png  2D paths — ridge-shielded corridor (A*/Hybrid detour)
  overlay_high_density.png    2D paths — many radars (graded risk)
  surface_open.png            3D terrain + path — open
  surface_concave_trap.png    3D terrain + path — concave trap
  surface_ridge_corridor.png  3D terrain + path — ridge corridor
  surface_high_density.png    3D terrain + path — high density
  pareto_LvsR.png             Length vs detection-risk Pareto front (sweeping wR, A*) on the
                              dedicated "pareto" scenario (single unshielded radar),
                              not ridge_corridor
  time_vs_threats.png         Compute time & expanded nodes vs radar count (A* vs Hybrid)

DATA TABLES
  results_aggregate.csv       MAIN TABLE: mean +/- std per (scenario, algorithm):
                              success rate, J, L, Rtil, Phi, d_min, kappa_peak,
                              gamma_peak, violation counts, time, nodes/iters
  results_per_run.csv         All 240 raw runs (4 scenarios x 20 runs x 3 planners)
  wilcoxon_J.csv              Paired Wilcoxon signed-rank tests on J (p-values)

To regenerate: python run_experiments.py --runs 20 --resolution 8 --seed 20240603
(add --full for a finer 5 m grid.) The committed PNG figures are 300 dpi
(e.g. overlays 2400 x 2100 px, Pareto 2100 x 1500 px).
