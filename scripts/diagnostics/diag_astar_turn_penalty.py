"""POST-HOC diagnostic (not part of the registered protocol): why does A-Star get
worse on finer grids in high_density? Hypothesis: the per-edge turn penalty
wK*max(0,(kappa-kmax)/kmax) with kappa = dpsi/sbar grows as cells shrink and is
summed per turn, so the planner over-weights turning relative to length/risk.

Compares A-Star as registered ('default') with A-Star planned without the turn
penalty ('no_turn_penalty', R_min = 1e-3 m for planning only); both are
evaluated with the registered v2 evaluator (nap, R_min = 60 m).  Uses the
held-out evaluation seed, so it adds post-hoc analyses to already-seen data.

Usage (from the repository root):
    python scripts/diagnostics/diag_astar_turn_penalty.py . [N] [scenarios] [resolutions]
Defaults: N = 20, scenarios = high_density, resolutions = 5,8,10.  The committed
results (diag_astar_turn_penalty_results_posthoc.csv, POST-HOC) correspond to
    N = 20, scenarios = high_density,ridge_corridor,concave_trap, resolutions = 5,8,10.
Writes outputs/diag_astar_<scenarios>.csv.
"""

import math
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, sys.argv[1])
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.v2 import protocol, runner
from uav_pathplan.v2.evaluate import evaluate_v2

proto = protocol.load(sys.argv[1] + "/configs/protocol_v2.yaml")
seed = proto["seeds"]["evaluation"]
alt = proto["altitude"]
# 1) magnitude of a single 22.5 deg and 45 deg turn penalty vs one-cell length cost
print("res  pen(22.5deg,straight-straight)  pen(45deg)  length_cost(one cell, L_ref=1100)")
for res in [5, 8, 10]:
    for dpsi in [math.radians(26.57)]:
        pass
    k1 = math.radians(45) / res
    k2 = math.radians(26.57) / (0.5 * (res + res * math.sqrt(5)))
    kmax = 1 / 60
    print(
        res,
        round(0.15 * max(0, (k2 - kmax) / kmax), 3),
        round(0.15 * max(0, (k1 - kmax) / kmax), 3),
        round(0.40 * res / 1100, 5),
    )
rows = []
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SC = sys.argv[3].split(",") if len(sys.argv) > 3 else ["high_density"]
RS = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [5.0, 8.0, 10.0]
for scn in SC:
    for res in RS:
        for run in range(N):
            spec, cfg, terrain, risk, H, _ = runner.build_instance(proto, scn, seed, run, res)
            for variant, pcfg in [("default", cfg), ("no_turn_penalty", replace(cfg, R_min=1e-3))]:
                r = plan_astar(pcfg, terrain, risk, spec.start, spec.goal, hazard_grid=H)
                m = evaluate_v2(
                    cfg,
                    risk,
                    terrain,
                    r.path,
                    spec.start,
                    spec.goal,
                    "nap",
                    alt["rate_limited_c_min_m"],
                    alt["profile_ds_m"],
                )
                rows.append(
                    dict(
                        scn=scn,
                        res=res,
                        run=run,
                        variant=variant,
                        success=r.success,
                        J=m["J"],
                        Rtil=m["Rtil"],
                        Ltil=m["Ltil"],
                        curv=m["curv_viol_frac"],
                    )
                )
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
D = pd.DataFrame(rows)
D.to_csv(OUT / ("diag_astar_%s.csv" % "_".join(SC)), index=False)
print(D.groupby(["scn", "res", "variant"])[["J", "Rtil", "Ltil", "curv"]].median().round(3))
