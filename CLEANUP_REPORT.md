# Repository clean-up report (v1.0-paper)

Scope: tidy the repository for the Defence Science Journal submission without
changing any numerical behaviour. Commits are local on `main` (not pushed).
`paper_assets/` was not modified (`git diff 79c2a2a HEAD -- paper_assets` is
empty).

## 1. Result

* **Reproduction: PASS.** `scripts/reproduce_paper.sh` (exactly
  `python run_experiments.py --runs 20 --resolution 8 --seed 20240603`, then
  `scripts/compare_to_paper.py`) on the final code, in a fresh virtual
  environment with the pinned versions (CPython 3.13.7, Windows 11), matches
  `paper_assets/` in **55 of 55** non-timing columns of `results_aggregate.csv`,
  `wilcoxon_J.csv` and `results_per_run.csv` with a maximum absolute difference
  of **0** (tolerance 1e-9). Only timing differs (§4).
* The same comparison on the **untouched initial commit** also gave 55/55
  PASS, so the code in the initial commit already reproduced the paper.
* **Tests:** `pytest` → 12 passed, 1 expected failure (strict `xfail`, a
  pre-existing failure, §5 D1). `ruff check` and `ruff format --check` are
  clean.
* **Paper statistics** stated in the brief are reproduced from
  `paper_assets/results_per_run.csv` (`tests/test_stats.py`): concave_trap
  A-Star vs Hybrid W = 62, p = 0.114 (Holm p is also 0.114, since it is the only
  valid pair); the open-scenario Holm-adjusted p = 0.0014 (A-Star vs APF and APF vs
  Hybrid both 0.001415).

## 2. Audit findings (made before any change)

| Area | Finding |
|---|---|
| Secrets / tokens | None in tracked files or history (only the word "secrets" in README prose). No history rewrite needed. |
| Personal e-mail | None in tracked files. The git author/committer e-mail of every commit (initial and new) is the owner's personal address from the local git config; this is visible on any push. Not changed. |
| Absolute local paths | None in tracked files. Untracked, git-ignored `outputs/*.log` contain local Windows paths. |
| Large binaries | `paper_assets/*.png` are 300-dpi figures (0.14–1.09 MB each, ≈6.5 MB in total), tracked in the initial commit. `paper_assets.zip` (2.2 MB, untracked, ignored) holds 130-dpi versions of the same figures and identical CSVs. `outputs/` is ignored. |
| Dead code (left in place) | `cost.edge_length_cost`, `edge_risk_cost`, `turn_curvature`, `edge_kappa_penalty` (cost.py:218–241; not called anywhere — A-Star computes its edge cost inline); `RiskField.hazard_along` (risk.py:252); `scenarios.with_overrides` (scenarios.py:258); `plotting.plot_surface_with_path` (plotting.py:289). Unused `Config` fields: `astar_num_headings` (config.py:175), `apf_min_step` (config.py:186, marked deprecated), `mc_runs` (config.py:197; the CLI `--runs` is used instead). |
| Unused imports | `dataclasses.field` in metrics.py, `math` in scenarios.py (removed by ruff). |
| Debug prints | None in the `uav_pathplan` library. Scripts print progress by design. |
| Stale tests | `smoke_cost.py` Case A printed "FAILURES PRESENT" against the current code (§5 D1). |
| Diagnostics | `diagnose_curv_risk.py` and `verify_calib.py` write to `outputs/` without creating it (unchanged). |

## 3. Changes made (one commit per concern)

| Commit | Change |
|---|---|
| `1c1b046` | `smoke_*.py` → `tests/test_*.py` (git mv) and converted to pytest. Hand computations and the 1e-6 tolerance are unchanged; printed "expect" lines became assertions; figures go to `tmp_path`. Former Case A of `smoke_cost.py` kept unchanged as strict `xfail`, plus a variant with `kin_resample_ds` = vertex spacing. `pyproject.toml` with pytest settings. |
| `abf8839` | `diagnose_curv_risk.py`, `diagnose_gamma.py`, `verify_calib.py` → `scripts/diagnostics/`; each docstring now starts with a one-line summary and the `python -m scripts.diagnostics.<name>` invocation. No code changes. |
| `fe6d02a` | `app.py` + `.streamlit/` → `dashboard/`; `app.py` adds the repository root to `sys.path`; README paths updated. Checked with `streamlit.testing.AppTest` (render and a full Run, no exceptions). |
| `cfd659f` | `requirements.txt` pinned with `==` (numpy 2.3.4, scipy 1.17.1, matplotlib 3.10.9, pandas 3.0.2, streamlit 1.58.0) plus a note on the Python version; `requirements-dev.txt` (pytest 9.1.1, ruff 0.17.0); `.python-version`. |
| `9f7d8c4` | `stats.py`: `holm_adjust()`; `PairTest.p_holm` and `significant_holm` (Holm within each scenario over pairs with a finite p-value). Raw `p_value`/`significant` kept. The console summary prints both. `tests/test_stats.py`. |
| `2ee1f1c` | `--dpi` default 130 → 300; new `--format {png,tiff,eps}` (TIFF LZW-compressed); overlay/surface figures no longer carry titles such as `concave_trap: A* vs APF vs Hybrid`; label "A-Star" in figures and dashboard; failure reasons in legends without underscores. |
| `7a7445e` | `scripts/reproduce_paper.sh`, `scripts/reproduce_paper.bat`, `scripts/compare_to_paper.py`, `.gitattributes` (LF for .sh, CRLF for .bat). |
| `7877107` | ruff config (E, W, F, I; line length 100) and `ruff format` of all code; a `TYPE_CHECKING` import for the `pd.DataFrame` string annotation in metrics.py. Re-verified by the full reproduction (§1). |
| `c135814` | Docstring corrections only (ASTs with docstrings removed are identical), see D3–D6. |
| `7e6fd4d` | README rewritten. |
| `85ca2c6` | `LICENSE` (MIT, confirmed by the owner), `CITATION.cff` (placeholder authors/ORCIDs; validated with cffconvert against CFF 1.2.0), `CHANGELOG.md` entry v1.0-paper. |
| `d60cc91` | `.github/workflows/ci.yml`: pinned install on Python 3.13, ruff, pytest, `run_experiments.py --quick`. |
| `7f2733e` | README wording of the † marker in the parameter table (see Q9). |

Not verified: the CI workflow has not run on GitHub (nothing was pushed); its
steps were run locally on Windows (lint, pytest, `--quick` in 27 s). Streamlit
could not be installed into the verification venv (Windows long-path limit);
the dashboard was checked with the same pinned streamlit 1.58.0 in the
system Python.

## 4. compare_to_paper.py output (final code)

```
Comparing outputs/ against paper_assets/  (tolerance 1e-09, timing/memory excluded)

file                   column                 max |diff|  result  note
---------------------  ---------------------  ----------  ------  ---------------------------------
results_aggregate.csv  success_rate           0           PASS
results_aggregate.csv  n_runs                 0           PASS
results_aggregate.csv  n_success              0           PASS
results_aggregate.csv  J_mean                 0           PASS
results_aggregate.csv  J_std                  0           PASS
results_aggregate.csv  L_mean                 0           PASS
results_aggregate.csv  L_std                  0           PASS
results_aggregate.csv  Ltil_mean              0           PASS
results_aggregate.csv  Ltil_std               0           PASS
results_aggregate.csv  Rtil_mean              0           PASS
results_aggregate.csv  Rtil_std               0           PASS
results_aggregate.csv  Phi_mean               0           PASS
results_aggregate.csv  Phi_std                0           PASS
results_aggregate.csv  Lambda_peak_mean       0           PASS
results_aggregate.csv  Lambda_peak_std        0           PASS
results_aggregate.csv  d_min_mean             0           PASS
results_aggregate.csv  d_min_std              0           PASS
results_aggregate.csv  kappa_peak_mean        0           PASS
results_aggregate.csv  kappa_peak_std         0           PASS
results_aggregate.csv  curv_violations_mean   0           PASS
results_aggregate.csv  curv_violations_std    0           PASS
results_aggregate.csv  gamma_peak_deg_mean    0           PASS
results_aggregate.csv  gamma_peak_deg_std     0           PASS
results_aggregate.csv  climb_violations_mean  0           PASS
results_aggregate.csv  climb_violations_std   0           PASS
results_aggregate.csv  smoothness_mean        0           PASS
results_aggregate.csv  smoothness_std         0           PASS
results_aggregate.csv  n_exp_mean             0           PASS
results_aggregate.csv  n_exp_std              0           PASS
results_aggregate.csv  n_iter_mean            0           PASS
results_aggregate.csv  n_iter_std             0           PASS
wilcoxon_J.csv         n_pairs                0           PASS
wilcoxon_J.csv         median_a               0           PASS
wilcoxon_J.csv         median_b               0           PASS
wilcoxon_J.csv         statistic              0           PASS
wilcoxon_J.csv         p_value                0           PASS
wilcoxon_J.csv         significant            exact       PASS
wilcoxon_J.csv         note                   exact       PASS
wilcoxon_J.csv         p_holm                 -           NEW     not in paper_assets; not compared
wilcoxon_J.csv         significant_holm       -           NEW     not in paper_assets; not compared
results_per_run.csv    seed                   0           PASS
results_per_run.csv    success                exact       PASS
results_per_run.csv    J                      0           PASS
results_per_run.csv    L                      0           PASS
results_per_run.csv    Ltil                   0           PASS
results_per_run.csv    Rtil                   0           PASS
results_per_run.csv    Phi                    0           PASS
results_per_run.csv    Lambda_peak            0           PASS
results_per_run.csv    d_min                  0           PASS
results_per_run.csv    kappa_peak             0           PASS
results_per_run.csv    curv_violations        0           PASS
results_per_run.csv    gamma_peak_deg         0           PASS
results_per_run.csv    climb_violations       0           PASS
results_per_run.csv    smoothness             0           PASS
results_per_run.csv    n_exp                  0           PASS
results_per_run.csv    n_iter                 0           PASS
results_per_run.csv    reason                 exact       PASS

Timing / memory (machine-dependent, informational only):

file                   column             max |diff|  max rel diff
---------------------  -----------------  ----------  ------------
results_aggregate.csv  time_s_mean        0.3092      27.9%
results_aggregate.csv  time_s_std         0.1235      84.6%
results_aggregate.csv  peak_mem_kib_mean  0           0.0%
results_aggregate.csv  peak_mem_kib_std   0           0.0%
results_per_run.csv    time_s             0.7337      54.1%
results_per_run.csv    peak_mem_kib       0           0.0%

OVERALL: PASS  (55 columns passed, 0 failed)
```

Timing differences are expected (machine load; different runs). Peak memory is
NaN in both because the paper run did not use `--mem`.

## 5. Discrepancies found

Line numbers refer to the final commit unless marked *initial*.

| # | Where | Discrepancy | Action |
|---|---|---|---|
| D1 | `tests/test_cost.py` (former `smoke_cost.py` Case A); `cost.py:161` | The hand-computed curvature uses the three original vertices (κ = 0.0157 m⁻¹); `cost.evaluate` resamples the path at `kin_resample_ds` (0 → grid resolution) before computing κ, γ, K̃, so it returns κ_peak = 0.314 m⁻¹, K̃ = 0.941, J = 0.307 (hand: 0.885, 0.298). The test predates the resampling. | Reported; kept as strict xfail. Not "fixed". |
| D2 | README (*initial* l.45), `run_experiments.py:187,207` | README called `--full` (5 m) the publication setting; the paper used 8 m / 20 runs / seed 20240603 (`paper_assets/README.txt`). `--full` still means 5 m. | README fixed. |
| D3 | README (*initial* l.11–12, 83–85); `__init__.py`, `cost.py`, `planners/__init__.py` docstrings; `dashboard/app.py:14,264` | "All planners minimise the same cost J". A-Star minimises an additive edge cost that differs from J: endpoint-average **grid** hazard (`astar.py:123`) instead of exact midpoint hazard, `|Δz|` per edge (`astar.py:127`), kinematic penalties **summed** rather than averaged, L_ref between snapped cells (`astar.py:100`). APF follows a force field; it does not minimise J. | README and library docstrings fixed; dashboard text not changed (open question Q6). |
| D4 | README (*initial* l.176–177) | "A* also uses the per-edge helpers in cost.py" — they are unused. | README fixed; helpers left in place. |
| D5 | `planners/apf.py` docstring (*initial*); README (*initial* l.278) | Described `U = ½ξ‖p−p_g‖²`, plain gradient steps `p − α∇U`, ∇Λ from `RiskField.Lambda`, and vertical step-shrinking to `apf_min_step`. Code: unit attraction `ξ(p_g−p)/‖p_g−p‖ − η∇Λ`, step `min(α, d)·max(0, cos θ)`, ∇Λ by central differences on the grid interpolant `Lambda_field`, **no** climb limit. | Docstring and README fixed. The paper's APF equations should be checked (Q2). |
| D6 | `planners/hybrid.py` docstring (*initial*) | Said the local layer enforces γ_max; it does not. | Docstring fixed. |
| D7 | `cost.py:140–141`; README (*initial* l.162) | Documented `L_ref = ‖p_g − p_s‖` (scenario start/goal). Implemented as the distance between the path's **own** first and last points. APF/Hybrid stop within 12 m of the goal without appending it; A-Star starts/ends at snapped cell centres. So L_ref differs slightly between planners and from the scenario distance. | README documents the implementation. Not changed (would alter results). Q1. |
| D8 | `scenarios.py:89` vs `scenarios.py:104`, `config.py:165` | concave_trap docstring (and the old README) (and the *initial* README l.254–255) say the wall radars have `R_max < R_min`; the code uses R_max = 70 m > R_min = 60 m. | Reported only. Q3. |
| D9 | `scenarios.py:104` | concave_trap "radars" use a fixed C = 1e9 (proxy obstacle wall), so concave_trap R̃/Φ (e.g. A-Star Φ mean 112.9, Hybrid 251.0) are not calibrated detection probabilities. | Stated in README. Q4. |
| D10 | `paper_assets/README.txt:25` | Says figures render at 130 dpi; the committed `paper_assets/*.png` are 300 dpi (e.g. 2400×2100 px). The untracked `paper_assets.zip` holds the 130-dpi versions. | Not edited (paper_assets is read-only). Q5. |
| D11 | `config.py:108`, `run_experiments.py:181–182` | Three different defaults: `Config.resolution` = 5 m, CLI `--resolution` = 10 m and `--runs` = 10, paper = 8 m / 20. `Config.mc_runs` = 20 is unused. | Documented in README; not changed. |
| D12 | `config.py:175`; `astar.py` | `astar_num_headings = 16` is never read; the number of heading bins equals the 16 move directions (8 neighbours + 8 knight moves), which are not uniformly spaced in angle. | README parameter table says so. |
| D13 | `dashboard/app.py:99, 521` | The dashboard replaces each scenario's terrain seed with the sidebar seed, so its views do not reproduce the paper's figure instances (fixed seeds 11111/22222/33333/44444), contrary to the caption at l.521. | README states this; UI text unchanged. Q6. |
| D14 | `outputs/run_full_calibrated.log` (untracked) | The log labels the Pareto sweep "ridge_corridor", but its numbers equal the current `pareto`-scenario sweep, and the regenerated `pareto_LvsR.png` is **pixel-identical** to `paper_assets/pareto_LvsR.png`. The paper's Pareto figure therefore comes from the `pareto` scenario (single unshielded radar). | Reported. Q7. |
| D15 | `run_experiments.py` | `time_s` excludes the hazard-grid precomputation (`setup`); Hybrid time includes its coarse A-Star. | Documented in README. Q8. |
| D16 | `--format eps` | EPS overlays are ≈18 MB each because the hazard map is embedded as an image; EPS cannot represent transparency (the 3D surface uses alpha 0.8). | Reported. |

The manuscript itself is not in the repository, so other statements in the
paper (beyond the values quoted in the brief) could not be checked against
the code.

## 6. Open questions for the authors

- **Q1.** **L_ref (D7):** does the paper define L̃ with the scenario start/goal or with
   the path endpoints? The results use path endpoints.
- **Q2.** **APF equations (D5):** does the paper state the quadratic potential? The
   code uses a unit-magnitude attraction, an aligned-step rule and no climb
   limit.
- **Q3.** **concave_trap geometry (D8):** R_max = 70 m > R_min = 60 m, contrary to the
   design note. Intended?
- **Q4.** **concave_trap risk (D9):** the paper should say that R̃/Φ in concave_trap
   come from a proxy obstacle wall (C = 1e9), not calibrated radars.
- **Q5.** **paper_assets/README.txt (D10):** update the "130 dpi" note (you own
   paper_assets/; it was not edited here).
- **Q6.** **Dashboard (D3, D13):** update the dashboard's "minimise the same cost J"
   and "reproduces a run_experiments.py run" texts?
- **Q7.** **Pareto figure (D14):** confirm that the paper's caption describes the
   `pareto` scenario (one unshielded radar, A-Star, wR ∈ {0.05,…,0.80}).
- **Q8.** **Timing (D15):** state in the paper that times exclude hazard-grid
   precomputation, and fill in "Hardware used for timing" in the README.
- **Q9.** **Parameter table:** the README marks with † the values the brief lists as not
  yet stated in the paper (R0, R_max, mast height, LOS step, Hybrid coarse grid, heading
  bins, APF gains ξ, η, α, Monte-Carlo jitter ranges) and other values that could not be
  checked against Table 1 because the manuscript is not in the repository. Please check
  each against Table 1.
- **Q10.** **CITATION.cff:** replace the placeholder names and ORCIDs; confirm the
  repository URL.
- **Q11.** **Dead code (§2):** remove the unused helpers and Config fields, or keep
  them? Left in place to avoid unrequested changes.
- **Q12.** **Dashboard deployment:** if the app is deployed on Streamlit Community
  Cloud, its main-file path must change to `dashboard/app.py`. Whether Cloud
  reads `dashboard/.streamlit/config.toml` was not verified.
- **Q13.** **Figure titles:** the time-vs-threats and Pareto figures keep their plain
  titles ("Planner compute time vs threat count", "Pareto front: length vs
  detection risk"). Remove them too if the journal style requires it.
