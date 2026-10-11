# Changelog

## v1.0-paper — 2026-10-11

Repository state accompanying the Defence Science Journal submission
"Terrain-Masked Risk-Aware Path Planning for Autonomous Aerial Vehicles:
Benchmarking A-Star, Artificial Potential Field, and a Hybrid Planner".

Planner, cost, risk-model and scenario code is unchanged from the initial
commit; `python run_experiments.py --runs 20 --resolution 8 --seed 20240603`
reproduces `paper_assets/` (all non-timing values identical).

### Added
- `wilcoxon_J.csv` gains `p_holm` (Holm adjustment within each scenario over the
  valid pairs) and `significant_holm`; the raw `p_value` is kept.
- `--format {png,tiff,eps}` option for figures; `--dpi` now defaults to 300.
- `scripts/reproduce_paper.sh` / `.bat` and `scripts/compare_to_paper.py`.
- pytest suite in `tests/` (converted smoke scripts plus a statistics test).
- Pinned `requirements.txt`, `requirements-dev.txt`, ruff configuration,
  GitHub Actions CI, `LICENSE` (MIT), `CITATION.cff`.

### Changed
- Figures: the planner is labelled "A-Star"; in-figure titles made of code
  identifiers were removed from the overlay and 3D-surface figures.
- Layout: smoke scripts → `tests/`; diagnostics → `scripts/diagnostics/`;
  dashboard → `dashboard/`.
- Code formatted with ruff (no behaviour change).
- README rewritten (8 m publication grid, accurate description of the
  objective of each planner, metric definitions, parameter table, limitations).
