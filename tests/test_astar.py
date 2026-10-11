"""Step-4 smoke test: A* on open terrain (with a couple of radars).

Converted from the former ``smoke_astar.py``.  Asserts that A* finds a path
and that the shared cost evaluates it; the figure is written to ``tmp_path``.

Run:  pytest tests/test_astar.py
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from uav_pathplan import cost as costmod
from uav_pathplan.config import Config, GaussianHill, RadarSpec, Ridge
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField


def test_astar_open_terrain(tmp_path):
    cfg = Config(
        seed=20240603,
        fractal_max=120.0,
        hills=[GaussianHill(cx=300, cy=300, height=80, sigma=80)],
        ridges=[Ridge(cx=500, cy=600, height=180, sigma=40, length=200, orientation_deg=20.0)],
        radars=[
            RadarSpec(x=500, y=400, mast_height=20, C=1.0e9, R_max=550),
            RadarSpec(x=700, y=650, mast_height=20, C=8.0e8, R_max=500),
        ],
        astar_connectivity=16,
    )
    cfg = replace(cfg, out_dir=tmp_path)
    cfg.validate()
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    start = (80.0, 120.0)
    goal = (900.0, 880.0)
    res = plan_astar(cfg, terrain, risk, start, goal)
    print(
        f"success={res.success}  n_exp={res.n_exp}  time={res.time_s:.3f}s  "
        f"waypoints={res.n_waypoints}"
    )
    assert res.success
    assert res.n_waypoints > 1

    cr = costmod.evaluate(cfg, risk, terrain, res.path)
    print(f"J={cr.J:.4f}  Ltil={cr.Ltil:.4f}  Rtil={cr.Rtil:.4f}  Ktil={cr.Ktil:.4f}")
    print(
        f"L={cr.L:.1f}m  Phi={cr.Phi:.4f}  Lambda_peak={cr.Lambda_peak:.4g}  "
        f"kappa_peak={cr.kappa_peak:.4g} (max {cfg.kappa_max:.4g})  "
        f"curv_viol={cr.curv_violations}  climb_viol={cr.climb_violations}"
    )
    assert np.isfinite(cr.J)

    out = cfg.out_dir
    extent = [0, cfg.world_size, 0, cfg.world_size]
    H = risk.grid()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
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
    ax.contour(
        terrain.xs, terrain.ys, terrain.h, levels=10, colors="cyan", linewidths=0.4, alpha=0.4
    )
    ax.plot(res.path[:, 0], res.path[:, 1], "-", color="white", lw=2.0, label="A-Star path")
    ax.plot(*start, "o", color="lime", ms=11, mec="k", label="start")
    ax.plot(*goal, "*", color="red", ms=16, mec="k", label="goal")
    for r in cfg.radars:
        ax.plot(r.x, r.y, "X", color="yellow", ms=10, mec="k")
        ax.add_patch(
            plt.Circle((r.x, r.y), r.R_max, fill=False, ec="yellow", ls="--", lw=0.6, alpha=0.5)
        )
    ax.set_title(f"A-Star path  J={cr.J:.3f}  L={cr.L:.0f}m  n_exp={res.n_exp}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "astar_path.png", dpi=130)
    plt.close(fig)
    assert (out / "astar_path.png").exists()
