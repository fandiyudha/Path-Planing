"""Step-6 smoke test: all three planners on the concave trap.

Converted from the former ``smoke_hybrid.py``.  Hybrid must succeed where pure
APF fails (and A* must find a path).  The comparison figure is written to
``tmp_path``.

Run:  pytest tests/test_hybrid.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from uav_pathplan import cost as costmod
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.hybrid import plan_hybrid
from uav_pathplan.risk import RiskField
from uav_pathplan.scenarios import build_concave_trap
from uav_pathplan.terrain import TerrainField


def test_hybrid_escapes_concave_trap(tmp_path):
    spec = build_concave_trap()
    cfg = spec.cfg
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    results = {}
    for name, fn in [("A-Star", plan_astar), ("APF", plan_apf), ("Hybrid", plan_hybrid)]:
        res = fn(cfg, terrain, risk, spec.start, spec.goal)
        cr = costmod.evaluate(cfg, risk, terrain, res.path) if res.n_waypoints > 1 else None
        results[name] = (res, cr)
        j = f"{cr.J:.4f}" if cr else "  -  "
        L = f"{cr.L:.0f}" if cr else "  -  "
        rt = f"{cr.Rtil:.3f}" if cr else "  -  "
        print(
            f"[{name:6s}] success={str(res.success):5s} "
            f"reason={res.info.get('reason', ''):16s} "
            f"J={j} L={L}m Rtil={rt} "
            f"n_exp={res.n_exp} n_iter={res.n_iter} t={res.time_s:.2f}s"
        )

    # plot overlay
    H = risk.grid()
    out = tmp_path
    extent = [0, cfg.world_size, 0, cfg.world_size]
    fig, ax = plt.subplots(figsize=(8, 7))
    Hpos = np.where(H > 0, H, np.nan)
    vmax = np.nanmax(Hpos)
    ax.imshow(
        Hpos,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
        aspect="equal",
    )
    colors = {"A-Star": "cyan", "APF": "white", "Hybrid": "lime"}
    for name, (res, _cr) in results.items():
        if res.n_waypoints > 1:
            ls = "-" if res.success else ":"
            ax.plot(
                res.path[:, 0],
                res.path[:, 1],
                ls,
                color=colors[name],
                lw=2.0,
                label=f"{name} ({'ok' if res.success else res.info.get('reason')})",
            )
    ax.plot(*spec.start, "o", color="lime", ms=12, mec="k")
    ax.plot(*spec.goal, "*", color="red", ms=18, mec="k")
    for r in cfg.radars:
        ax.plot(r.x, r.y, "X", color="yellow", ms=6, mec="k")
    ax.set_title("Concave trap: A-Star vs APF vs Hybrid")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_trap.png", dpi=130)
    plt.close(fig)
    assert (out / "compare_trap.png").exists()

    apf_ok = results["APF"][0].success
    hyb_ok = results["Hybrid"][0].success
    ast_ok = results["A-Star"][0].success
    print(f"\nRESULT: A-Star={ast_ok} APF={apf_ok} (expect False) Hybrid={hyb_ok} (expect True)")
    assert ast_ok, "A-Star should route around the trap"
    assert not apf_ok, "APF should stall in the trap"
    assert hyb_ok, "Hybrid should escape the trap"
