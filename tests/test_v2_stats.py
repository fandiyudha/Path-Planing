"""Protocol v2: statistics helpers and protocol guards.

Run:  pytest tests/test_v2_stats.py
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from uav_pathplan.v2 import protocol
from uav_pathplan.v2.stats import friedman_tests, pairwise_tests, rank_biserial, wilson_ci


def test_wilson_reference_values():
    # textbook values (Newcombe 1998, method 3)
    assert wilson_ci(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson_ci(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson_ci(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert all(np.isnan(wilson_ci(0, 0)))


def test_rank_biserial():
    a = np.array([5.0, 6.0, 7.0, 8.0])
    assert rank_biserial(a, a - 1.0) == 1.0  # a always larger
    assert rank_biserial(a - 1.0, a) == -1.0
    # d = (+1, -2, +3): ranks 1, 2, 3 -> (4 - 2) / 6
    assert rank_biserial([1.0, 0.0, 3.0], [0.0, 2.0, 0.0]) == pytest.approx(1 / 3)
    assert np.isnan(rank_biserial([1.0, 2.0], [1.0, 2.0]))  # all zero differences


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for run in range(12):
        base = rng.uniform(0.1, 0.2)
        for p, shift, ok in [
            ("astar", 0.00, True),
            ("apf", 0.10, run % 2 == 0),
            ("apfe", 0.05, True),
            ("hybrid", -0.02, True),
        ]:
            rows.append(
                dict(
                    resolution=8.0,
                    altitude_mode="nap",
                    scenario="s",
                    run=run,
                    planner=p,
                    success=ok,
                    J=base + shift + rng.normal(0, 0.003),
                )
            )
    return pd.DataFrame(rows)


def test_pairwise_tests_holm_and_pairs():
    t = pairwise_tests(_frame(), ["astar", "apf", "apfe", "hybrid"])
    assert len(t) == 6
    row = t[(t.planner_a == "astar") & (t.planner_b == "apf")].iloc[0]
    assert row.n_pairs == 6  # APF succeeded in 6 of 12 runs
    assert row.rank_biserial == -1.0  # A-Star always lower J
    # Holm: adjusted >= raw, monotone in raw order, <= 1
    assert (t.p_holm >= t.p_value - 1e-15).all() and (t.p_holm <= 1).all()
    order = np.argsort(t.p_value.to_numpy())
    assert np.all(np.diff(t.p_holm.to_numpy()[order]) >= -1e-15)


def test_friedman_uses_complete_runs_only():
    f = friedman_tests(_frame(), ["astar", "apf", "apfe", "hybrid"]).iloc[0]
    assert f.n_complete == 6
    assert 0.0 <= f.kendall_w <= 1.0 and f.p_value < 0.05


def test_protocol_loads_and_guards_eval_seed(tmp_path):
    p = protocol.load()
    assert p["seeds"]["development"] != p["seeds"]["evaluation"]
    assert p["monte_carlo"]["runs_per_scenario"] == 50
    if p["status"] == "draft":
        from uav_pathplan.v2.runner import main

        args = SimpleNamespace(
            config=protocol.DEFAULT_PATH, quick=False, seed_set="eval", out=str(tmp_path)
        )
        with pytest.raises(protocol.ProtocolError):
            main(args)
    else:
        assert all(v is not None for v in protocol.apfe_params(p).values())
