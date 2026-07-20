"""Step-6 smoke test: all three planners on the concave trap.

Hybrid must succeed where pure APF fails.  Saves outputs/compare_trap.png.
Run:  python smoke_hybrid.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from uav_pathplan.terrain import TerrainField
from uav_pathplan.risk import RiskField
from uav_pathplan.planners.astar import plan_astar
from uav_pathplan.planners.apf import plan_apf
from uav_pathplan.planners.hybrid import plan_hybrid
from uav_pathplan.scenarios import build_concave_trap
from uav_pathplan import cost as costmod


def main():
    spec = build_concave_trap()
    cfg = spec.cfg
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    results = {}
    for name, fn in [("A*", plan_astar), ("APF", plan_apf), ("Hybrid", plan_hybrid)]:
        res = fn(cfg, terrain, risk, spec.start, spec.goal)
        cr = costmod.evaluate(cfg, risk, terrain, res.path) if res.n_waypoints > 1 else None
        results[name] = (res, cr)
        j = f"{cr.J:.4f}" if cr else "  -  "
        L = f"{cr.L:.0f}" if cr else "  -  "
        rt = f"{cr.Rtil:.3f}" if cr else "  -  "
        print(f"[{name:6s}] success={str(res.success):5s} "
              f"reason={res.info.get('reason',''):16s} "
              f"J={j} L={L}m Rtil={rt} "
              f"n_exp={res.n_exp} n_iter={res.n_iter} t={res.time_s:.2f}s")

    # plot overlay
    H = risk.grid()
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    extent = [0, cfg.world_size, 0, cfg.world_size]
    fig, ax = plt.subplots(figsize=(8, 7))
    Hpos = np.where(H > 0, H, np.nan); vmax = np.nanmax(Hpos)
    ax.imshow(Hpos, origin="lower", extent=extent, cmap="inferno",
              norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax), aspect="equal")
    colors = {"A*": "cyan", "APF": "white", "Hybrid": "lime"}
    for name, (res, cr) in results.items():
        if res.n_waypoints > 1:
            ls = "-" if res.success else ":"
            ax.plot(res.path[:, 0], res.path[:, 1], ls, color=colors[name],
                    lw=2.0, label=f"{name} ({'ok' if res.success else res.info.get('reason')})")
    ax.plot(*spec.start, "o", color="lime", ms=12, mec="k")
    ax.plot(*spec.goal, "*", color="red", ms=18, mec="k")
    for r in cfg.radars:
        ax.plot(r.x, r.y, "X", color="yellow", ms=6, mec="k")
    ax.set_title("Concave trap: A* vs APF vs Hybrid")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_xlim(0, cfg.world_size); ax.set_ylim(0, cfg.world_size)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "compare_trap.png", dpi=130); plt.close(fig)
    print(f"saved: {out / 'compare_trap.png'}")

    apf_ok = results["APF"][0].success
    hyb_ok = results["Hybrid"][0].success
    ast_ok = results["A*"][0].success
    print(f"\nRESULT: A*={ast_ok} APF={apf_ok} (expect False) "
          f"Hybrid={hyb_ok} (expect True)")


if __name__ == "__main__":
    main()
