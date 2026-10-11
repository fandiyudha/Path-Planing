#!/usr/bin/env bash
# Reproduce the paper results into outputs/ and compare them with paper_assets/.
#
# Usage (any directory):  bash scripts/reproduce_paper.sh
# Requires the pinned dependencies:  pip install -r requirements.txt
set -euo pipefail

cd "$(dirname "$0")/.."

python run_experiments.py --runs 20 --resolution 8 --seed 20240603

python scripts/compare_to_paper.py
