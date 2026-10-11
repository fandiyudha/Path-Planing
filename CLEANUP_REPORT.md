# Repository clean-up report (v1.0-paper)

Commit hashes below are those after the e-mail rewrite of §7.

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
| `f30456a` | `smoke_*.py` → `tests/test_*.py` (git mv) and converted to pytest. Hand computations and the 1e-6 tolerance are unchanged; printed "expect" lines became assertions; figures go to `tmp_path`. Former Case A of `smoke_cost.py` kept unchanged as strict `xfail`, plus a variant with `kin_resample_ds` = vertex spacing. `pyproject.toml` with pytest settings. |
| `6a34f5a` | `diagnose_curv_risk.py`, `diagnose_gamma.py`, `verify_calib.py` → `scripts/diagnostics/`; each docstring now starts with a one-line summary and the `python -m scripts.diagnostics.<name>` invocation. No code changes. |
| `421992f` | `app.py` + `.streamlit/` → `dashboard/`; `app.py` adds the repository root to `sys.path`; README paths updated. Checked with `streamlit.testing.AppTest` (render and a full Run, no exceptions). |
| `a56e63b` | `requirements.txt` pinned with `==` (numpy 2.3.4, scipy 1.17.1, matplotlib 3.10.9, pandas 3.0.2, streamlit 1.58.0) plus a note on the Python version; `requirements-dev.txt` (pytest 9.1.1, ruff 0.17.0); `.python-version`. |
| `9428091` | `stats.py`: `holm_adjust()`; `PairTest.p_holm` and `significant_holm` (Holm within each scenario over pairs with a finite p-value). Raw `p_value`/`significant` kept. The console summary prints both. `tests/test_stats.py`. |
| `9d11280` | `--dpi` default 130 → 300; new `--format {png,tiff,eps}` (TIFF LZW-compressed); overlay/surface figures no longer carry titles such as `concave_trap: A* vs APF vs Hybrid`; label "A-Star" in figures and dashboard; failure reasons in legends without underscores. |
| `276ab1c` | `scripts/reproduce_paper.sh`, `scripts/reproduce_paper.bat`, `scripts/compare_to_paper.py`, `.gitattributes` (LF for .sh, CRLF for .bat). |
| `11f8485` | ruff config (E, W, F, I; line length 100) and `ruff format` of all code; a `TYPE_CHECKING` import for the `pd.DataFrame` string annotation in metrics.py. Re-verified by the full reproduction (§1). |
| `4508d34` | Docstring corrections only (ASTs with docstrings removed are identical), see D3–D6. |
| `0355b2c` | README rewritten. |
| `67cb681` | `LICENSE` (MIT, confirmed by the owner), `CITATION.cff` (placeholder authors/ORCIDs; validated with cffconvert against CFF 1.2.0), `CHANGELOG.md` entry v1.0-paper. |
| `d4e06ba` | `.github/workflows/ci.yml`: pinned install on Python 3.13, ruff, pytest, `run_experiments.py --quick`. |
| `da4aa63` | README wording of the † marker in the parameter table (see Q9). |

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
| D1 | `tests/test_cost.py` (former `smoke_cost.py` Case A); `cost.py:161` | The hand-computed curvature uses the three original vertices (κ = 0.0157 m⁻¹); `cost.evaluate` resamples the path at `kin_resample_ds` (0 → grid resolution) before computing κ, γ, K̃, so it returns κ_peak = 0.314 m⁻¹, K̃ = 0.941, J = 0.307 (hand: 0.885, 0.298). The test predates the resampling. | Kept as strict xfail at first; **resolved in §7** (expected values re-derived for the resampled definition; cost.py unchanged). |
| D2 | README (*initial* l.45), `run_experiments.py:187,207` | README called `--full` (5 m) the publication setting; the paper used 8 m / 20 runs / seed 20240603 (`paper_assets/README.txt`). `--full` still means 5 m. | README fixed. |
| D3 | README (*initial* l.11–12, 83–85); `__init__.py`, `cost.py`, `planners/__init__.py` docstrings; `dashboard/app.py:14,264` | "All planners minimise the same cost J". A-Star minimises an additive edge cost that differs from J: endpoint-average **grid** hazard (`astar.py:123`) instead of exact midpoint hazard, `|Δz|` per edge (`astar.py:127`), kinematic penalties **summed** rather than averaged, L_ref between snapped cells (`astar.py:100`). APF follows a force field; it does not minimise J. | README and library docstrings fixed; dashboard text not changed (open question Q6). |
| D4 | README (*initial* l.176–177) | "A* also uses the per-edge helpers in cost.py" — they are unused. | README fixed; helpers left in place. |
| D5 | `planners/apf.py` docstring (*initial*); README (*initial* l.278) | Described `U = ½ξ‖p−p_g‖²`, plain gradient steps `p − α∇U`, ∇Λ from `RiskField.Lambda`, and vertical step-shrinking to `apf_min_step`. Code: unit attraction `ξ(p_g−p)/‖p_g−p‖ − η∇Λ`, step `min(α, d)·max(0, cos θ)`, ∇Λ by central differences on the grid interpolant `Lambda_field`, **no** climb limit. | Docstring and README fixed. The paper's APF equations should be checked (Q2). |
| D6 | `planners/hybrid.py` docstring (*initial*) | Said the local layer enforces γ_max; it does not. | Docstring fixed. |
| D7 | `cost.py:140–141`; README (*initial* l.162) | Documented `L_ref = ‖p_g − p_s‖` (scenario start/goal). Implemented as the distance between the path's **own** first and last points. APF/Hybrid stop within 12 m of the goal without appending it; A-Star starts/ends at snapped cell centres. So L_ref differs slightly between planners and from the scenario distance. | README documents the implementation. Not changed (would alter results). Q1. |
| D8 | `scenarios.py:89` vs `scenarios.py:104`, `config.py:165` | concave_trap docstring (and the old README) (and the *initial* README l.254–255) say the wall radars have `R_max < R_min`; the code uses R_max = 70 m > R_min = 60 m. | Reported only. Q3. |
| D9 | `scenarios.py:104` | concave_trap "radars" use a fixed C = 1e9 (proxy obstacle wall), so concave_trap R̃/Φ (e.g. A-Star Φ mean 112.9, Hybrid 251.0) are not calibrated detection probabilities. | Stated in README. Q4. |
| D10 | `paper_assets/README.txt:25` | Says figures render at 130 dpi; the committed `paper_assets/*.png` are 300 dpi (e.g. 2400×2100 px). The untracked `paper_assets.zip` holds the 130-dpi versions. | **Resolved in §7** (README.txt corrected with the owner's approval). |
| D11 | `config.py:108`, `run_experiments.py:181–182` | Three different defaults: `Config.resolution` = 5 m, CLI `--resolution` = 10 m and `--runs` = 10, paper = 8 m / 20. `Config.mc_runs` = 20 is unused. | Documented in README; not changed. |
| D12 | `config.py:175`; `astar.py` | `astar_num_headings = 16` is never read; the number of heading bins equals the 16 move directions (8 neighbours + 8 knight moves), which are not uniformly spaced in angle. | README parameter table says so. |
| D13 | `dashboard/app.py:99, 521` | The dashboard replaces each scenario's terrain seed with the sidebar seed, so its views do not reproduce the paper's figure instances (fixed seeds 11111/22222/33333/44444), contrary to the caption at l.521. | README states this; UI text unchanged. Q6. |
| D14 | `outputs/run_full_calibrated.log` (untracked) | The log labels the Pareto sweep "ridge_corridor", but its numbers equal the current `pareto`-scenario sweep, and the regenerated `pareto_LvsR.png` is **pixel-identical** to `paper_assets/pareto_LvsR.png`. The paper's Pareto figure therefore comes from the `pareto` scenario (single unshielded radar). | `paper_assets/README.txt` now says so (§7). Q7. |
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

## 7. Pre-push status (2026-10-11)

Follow-up commits after the first report (all local, not pushed):

| Commit | Change |
|---|---|
| (rebase) | Author/committer e-mail of the 14 cleanup commits corrected (`git rebase -r 79c2a2a --exec "git commit --amend --no-edit --reset-author"`). The initial commit `79c2a2a` (already on GitHub) was not rewritten. The rewritten tree is identical to the pre-rewrite tip (`git diff` empty). A safety branch `backup/pre-email-fix` points at the old tip and can be deleted after review. |
| `45e32fe` | `.mailmap` maps the typo address on the initial commit to the correct one; `git log --format="%h %aN <%aE>"` and `git shortlog -se` show one identity for all commits. (`%ae` intentionally shows the raw address, so it still shows the typo on `79c2a2a`.) |
| `c544fb1` | `tests/test_cost.py`: kinematics expectations derived by hand for the resampled definition (κ_peak = π/10, K̃ = (12π − 1)/39, J = 0.4(√2 − 1) + 0.15·K̃, smoothness = π/2; derivation in the test docstring); xfail removed. The original-vertex computation stays as the `kin_resample_ds = 100 m` case. `cost.py` unchanged. Resolves D1. |
| `e33dcb6` | `paper_assets/README.txt` only: figures are 300 dpi; `pareto_LvsR.png` comes from the dedicated `pareto` scenario (single unshielded radar), not ridge_corridor. Resolves D10; documents D14. No other file in `paper_assets/` changed. |
| `89681ef` | README "Methods (as implemented)": APF (capped attraction, α = 5 m, ξ = wL, η = wR, heading clamp κ_max·s, no climb clamp), concave_trap wall (C = 1e9, R_max = 70 m, R0 = 20 m; R̃ is a relative exposure index), L_ref from the path's own endpoints, resampling of κ/γ vs original path for L/Φ, smoothness definition. |

Checks on the final code (CPython 3.13.7, pinned versions, Windows 11):

| Check | Result |
|---|---|
| `ruff check .` / `ruff format --check .` (ruff 0.17.0) | All checks passed / 29 files already formatted |
| `pytest` | **13 passed**, 0 failed, 0 xfail (37.8 s) |
| `bash scripts/reproduce_paper.sh` | **OVERALL: PASS — 55/55 non-timing columns identical** to `paper_assets/` (max \|diff\| = 0); concave_trap A-Star vs Hybrid W = 62, p = 0.114 (Holm 0.114); open A-Star vs APF and APF vs Hybrid Holm p = 0.001415. Run time 184.6 s; timing differs from the paper run by up to 0.33 s in `time_s_mean` (29 %), informational only. |

New minor finding: `cost.wrap_angle`'s docstring says it wraps into (−π, π],
but the expression `(a + π) mod 2π − π` returns [−π, π) (`wrap_angle(π) = −π`).
Only `|Δψ|` is used, so no metric is affected; the docstring was not changed.

Still open: Q1–Q4, Q6–Q13 above; CI has not yet run on GitHub; pushing is left
to the owner.

## 8. Commit hash mapping

Hashes changed because the author/committer e-mail metadata was rewritten for privacy (private address replaced by the GitHub noreply address); author and committer dates are unchanged, and trees are unchanged except that the same address was also replaced in `.mailmap` in the 16 commits that contain that file. The initial commit `79c2a2a` is unchanged. Hashes cited above refer to
the left column; the tag `v1.0-paper` now points to `a445ce4` (was `5ee15a5`).

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
