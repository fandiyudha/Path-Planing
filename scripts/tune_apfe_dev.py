"""Select the APF-E parameters on the DEVELOPMENT seed (protocol v2).

Reads the tuning grid, run count, resolution, altitude mode and selection rule
from configs/protocol_v2.yaml (``planners.apfe_tuning``) and evaluates every
grid point on the same development-seed instances with the same random streams
(paired).  Selection rule (as registered): maximise the mean success rate over
the four scenarios; ties -> lower mean over scenarios of the median J of
successful runs; then lower mean number of escapes.

Writes configs/apfe_tuning_dev.csv (per grid point x scenario) and prints the
ranking and the selected parameters.  The evaluation seed is never used.

Run (from the repository root):  python scripts/tune_apfe_dev.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uav_pathplan.planners.apf_escape import plan_apf_escape  # noqa: E402
from uav_pathplan.v2 import protocol  # noqa: E402
from uav_pathplan.v2 import scenarios as v2scn  # noqa: E402
from uav_pathplan.v2.evaluate import evaluate_v2  # noqa: E402
from uav_pathplan.v2.runner import build_instance  # noqa: E402


def main() -> int:
    proto = protocol.load()
    tune = proto["planners"]["apfe_tuning"]
    assert tune["seed"] == "development"
    seed = proto["seeds"]["development"]
    alt = proto["altitude"]
    grid = tune["grid"]
    points = [
        dict(
            max_escapes=me,
            turn_steps_min=ts[0],
            turn_steps_max=ts[1],
            straight_steps=ss,
            turn_frac=proto["planners"]["apfe"]["turn_frac"],
        )
        for me, ts, ss in itertools.product(
            grid["max_escapes"], grid["turn_steps"], grid["straight_steps"]
        )
    ]
    print(
        f"APF-E tuning on development seed {seed}: {len(points)} grid points, "
        f"{tune['runs_per_scenario']} runs/scenario, {tune['resolution_m']} m, "
        f"{tune['altitude_mode']}"
    )
    rows = []
    for scn in proto["monte_carlo"]["scenarios"]:
        for run in range(tune["runs_per_scenario"]):
            spec, cfg, terrain, risk, _, _ = build_instance(
                proto, scn, seed, run, tune["resolution_m"]
            )
            for i, kw in enumerate(points):
                r = plan_apf_escape(
                    cfg,
                    terrain,
                    risk,
                    spec.start,
                    spec.goal,
                    rng=v2scn.apfe_rng(seed, scn, run),
                    **kw,
                )
                m = evaluate_v2(
                    cfg,
                    risk,
                    terrain,
                    r.path,
                    spec.start,
                    spec.goal,
                    tune["altitude_mode"],
                    alt["rate_limited_c_min_m"],
                    alt["profile_ds_m"],
                )
                rows.append(
                    dict(
                        point=i,
                        scenario=scn,
                        run=run,
                        success=r.success,
                        n_escapes=r.info["n_escapes"],
                        J=m["J"],
                        **kw,
                    )
                )
        print(f"  {scn} done")
    df = pd.DataFrame(rows)
    per = (
        df.groupby(["point", "scenario"])
        .apply(
            lambda g: pd.Series(
                dict(
                    success_rate=g["success"].mean(),
                    median_J_success=g.loc[g["success"], "J"].median(),
                    mean_escapes=g["n_escapes"].mean(),
                )
            ),
            include_groups=False,
        )
        .reset_index()
    )
    params = pd.DataFrame(points).reset_index(names="point")
    per = per.merge(params, on="point")
    out = Path("configs") / "apfe_tuning_dev.csv"
    per.to_csv(out, index=False)

    rank = (
        per.groupby("point")
        .agg(
            mean_success=("success_rate", "mean"),
            mean_medJ=("median_J_success", "mean"),
            mean_escapes=("mean_escapes", "mean"),
        )
        .reset_index()
        .merge(params, on="point")
    )
    rank["mean_medJ"] = rank["mean_medJ"].fillna(np.inf)
    rank = rank.sort_values(
        ["mean_success", "mean_medJ", "mean_escapes"], ascending=[False, True, True]
    )
    print("\nRanking (best first):")
    print(rank.to_string(index=False))
    best = rank.iloc[0]
    print(
        "\nSELECTED:",
        {k: int(best[k]) if k != "turn_frac" else float(best[k]) for k in protocol.APFE_KEYS},
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
