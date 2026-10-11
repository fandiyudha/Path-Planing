"""Paired Wilcoxon tests and Holm adjustment reproduce the paper's statistics.

The tests are recomputed from the canonical per-run results in
paper_assets/results_per_run.csv (read-only), so they do not need the slow
Monte-Carlo run.

Run:  pytest tests/test_stats.py
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uav_pathplan.metrics import RunMetrics
from uav_pathplan.stats import holm_adjust, paired_tests
from uav_pathplan.stats import tests_to_frame as _to_frame

PAPER = Path(__file__).resolve().parent.parent / "paper_assets"


def _rows_from_csv(path: Path) -> list[RunMetrics]:
    df = pd.read_csv(path)
    df["reason"] = df["reason"].fillna("")
    names = [f.name for f in fields(RunMetrics)]
    return [RunMetrics(**{k: rec[k] for k in names}) for rec in df.to_dict("records")]


@pytest.fixture(scope="module")
def wilcoxon_frame():
    rows = _rows_from_csv(PAPER / "results_per_run.csv")
    return _to_frame(paired_tests(rows, metric="J")).set_index(["scenario", "alg_a", "alg_b"])


def test_holm_adjust_hand_example():
    # m = 3: sorted 0.01, 0.02, 0.04 -> 0.03, 0.04, 0.04 (monotone step-down)
    got = holm_adjust([0.04, 0.01, np.nan, 0.02])
    np.testing.assert_allclose(got[[0, 1, 3]], [0.04, 0.03, 0.04], rtol=0, atol=1e-15)
    assert np.isnan(got[2])
    assert np.isnan(holm_adjust([np.nan])).all()
    np.testing.assert_allclose(holm_adjust([0.6, 0.7]), [1.0, 1.0])


def test_raw_columns_match_paper_assets(wilcoxon_frame):
    ref = pd.read_csv(PAPER / "wilcoxon_J.csv").set_index(["scenario", "alg_a", "alg_b"])
    got = wilcoxon_frame.loc[ref.index]
    for col in ["n_pairs", "median_a", "median_b", "statistic", "p_value"]:
        np.testing.assert_allclose(
            got[col].to_numpy(float), ref[col].to_numpy(float), rtol=0, atol=1e-12, err_msg=col
        )
    assert (got["significant"].to_numpy() == ref["significant"].to_numpy()).all()


def test_concave_trap_astar_vs_hybrid(wilcoxon_frame):
    t = wilcoxon_frame.loc[("concave_trap", "astar", "hybrid")]
    assert t["n_pairs"] == 20
    assert t["statistic"] == 62.0
    assert round(t["p_value"], 3) == 0.114
    # the only valid pair in this scenario (APF never succeeds) -> Holm = raw
    assert t["p_holm"] == t["p_value"]
    for pair in [("astar", "apf"), ("apf", "hybrid")]:
        assert np.isnan(wilcoxon_frame.loc[("concave_trap", *pair)]["p_holm"])


def test_open_baseline_holm(wilcoxon_frame):
    raw = {
        pair: wilcoxon_frame.loc[("open", *pair)]["p_value"]
        for pair in [("astar", "apf"), ("astar", "hybrid"), ("apf", "hybrid")]
    }
    holm = {pair: wilcoxon_frame.loc[("open", *pair)]["p_holm"] for pair in raw}
    # hand Holm: smallest x3, second x2, largest x1 with monotonicity
    p1 = 3 * raw[("astar", "hybrid")]
    p2 = max(p1, 2 * raw[("apf", "hybrid")])
    p3 = max(p2, raw[("astar", "apf")])
    assert holm[("astar", "hybrid")] == pytest.approx(p1, abs=1e-15)
    assert holm[("apf", "hybrid")] == pytest.approx(p2, abs=1e-15)
    assert holm[("astar", "apf")] == pytest.approx(p3, abs=1e-15)
    # value quoted in the paper
    assert round(holm[("apf", "hybrid")], 4) == 0.0014
    assert round(holm[("astar", "apf")], 4) == 0.0014
