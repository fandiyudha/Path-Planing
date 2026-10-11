"""Step-5 smoke test: APF on open terrain (succeeds) and concave trap (stalls).

Converted from the former ``smoke_apf.py``; the "expect" lines it printed are
now assertions.  Figures are written to ``tmp_path``.

Run:  pytest tests/test_apf.py
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from uav_pathplan import cost as costmod
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.risk import RiskField
from uav_pathplan.scenarios import build_concave_trap, build_open
from uav_pathplan.terrain import TerrainField


def render(spec, res, fname, title):
    cfg = spec.cfg
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    H = risk.grid()
    out = cfg.out_dir
    extent = [0, cfg.world_size, 0, cfg.world_size]
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
    ax.contour(terrain.xs, terrain.ys, terrain.h, levels=8, colors="cyan", linewidths=0.3, alpha=0.4)
    if res.n_waypoints:
        ax.plot(res.path[:, 0], res.path[:, 1], "-", color="white", lw=1.8)
    ax.plot(*spec.start, "o", color="lime", ms=11, mec="k")
    ax.plot(*spec.goal, "*", color="red", ms=16, mec="k")
    for r in cfg.radars:
        ax.plot(r.x, r.y, "X", color="yellow", ms=8, mec="k")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=130)
    plt.close(fig)
    assert (out / fname).exists()


def run(spec, label):
    cfg = spec.cfg
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    res = plan_apf(cfg, terrain, risk, spec.start, spec.goal)
    cr = costmod.evaluate(cfg, risk, terrain, res.path) if res.n_waypoints > 1 else None
    print(
        f"[{label}] success={res.success}  reason={res.info['reason']:14s}  "
        f"iters={res.n_iter}  time={res.time_s:.3f}s  "
        f"final_dist_to_goal={res.info['final_dist_to_goal']:.1f}m"
    )
    if cr:
        print(
            f"         J={cr.J:.4f}  L={cr.L:.0f}m  Rtil={cr.Rtil:.4f}  "
            f"Ktil={cr.Ktil:.4f}  kappa_peak={cr.kappa_peak:.4g}"
        )
    return res


def _with_out(spec, out_dir):
    return replace(spec, cfg=replace(spec.cfg, out_dir=out_dir))


def test_apf_open_succeeds(tmp_path):
    s_open = _with_out(build_open(), tmp_path)
    r_open = run(s_open, "open")
    render(s_open, r_open, "apf_open.png", f"APF open terrain  success={r_open.success}")
    assert r_open.success, "APF should reach the goal on open terrain"


def test_apf_concave_trap_stalls(tmp_path):
    s_trap = _with_out(build_concave_trap(), tmp_path)
    r_trap = run(s_trap, "trap")
    render(
        s_trap,
        r_trap,
        "apf_trap.png",
        f"APF concave trap  success={r_trap.success} ({r_trap.info['reason']})",
    )
    assert not r_trap.success, "APF should fail in the concave trap"
    assert r_trap.info["reason"] == "local_minimum"
