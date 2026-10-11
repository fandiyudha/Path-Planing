"""Step-2 smoke test: radar risk field + manual LOS.

Converted from the former ``smoke_risk.py``.  Confirms that a ridge between a
radar and the far side casts a detection shadow (a low-hazard masked region),
and that the hazard grid agrees with the continuous callable at a cell centre.
The heatmap is written to pytest's ``tmp_path``.

Run:  pytest tests/test_risk.py
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from uav_pathplan.config import Config, GaussianHill, RadarSpec, Ridge
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField


def _cfg() -> Config:
    return Config(
        seed=20240603,
        fractal_max=140.0,
        hills=[GaussianHill(cx=250, cy=780, height=90, sigma=80)],
        ridges=[
            # Tall ridge running E-W across the middle (orientation 0 deg).
            Ridge(cx=500, cy=500, height=220, sigma=35, length=320, orientation_deg=0.0),
        ],
        radars=[
            # South of the ridge, looking north: the region north of the ridge
            # should be masked (shadowed).
            RadarSpec(x=500, y=300, mast_height=15, C=1.0e9, R_max=650),
            # A second radar in the open NE to show an unshadowed lobe.
            RadarSpec(x=820, y=820, mast_height=20, C=1.0e9, R_max=550),
        ],
        clearance=50.0,
        R0=50.0,
        los_step=5.0,
    )


def test_risk_field_and_los(tmp_path):
    cfg = replace(_cfg(), out_dir=tmp_path)
    cfg.validate()

    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    # --- LOS spot checks across the ridge ------------------------------ #
    # A point just north of the ridge crest, low altitude: should be masked
    # from the southern radar.
    z_n = terrain.altitude_at(500, 650)
    vis_n = risk.visible(0, 500, 650, z_n)
    # A point south of the ridge, same radar: should be visible.
    z_s = terrain.altitude_at(500, 380)
    vis_s = risk.visible(0, 500, 380, z_s)
    print(f"radar0 (south) -> point north of ridge (500,650): visible={vis_n} (expect False)")
    print(f"radar0 (south) -> point south of ridge (500,380): visible={vis_s} (expect True)")
    assert vis_n is False, "expected the ridge to shadow (500, 650) from radar 0"
    assert vis_s is True, "expected clear line of sight to (500, 380) from radar 0"

    # --- grid / callable agreement ------------------------------------- #
    H = risk.grid()
    ix, iy = 120, 40
    xc, yc = terrain.xs[ix], terrain.ys[iy]
    g = H[iy, ix]
    c = risk.Lambda(xc, yc)
    print(
        f"grid vs callable at ({xc:.0f},{yc:.0f}): "
        f"grid={g:.6g}  callable={c:.6g}  |diff|={abs(g - c):.2e}"
    )
    print(f"hazard grid: min={H.min():.4g} max={H.max():.4g} nonzero cells={(H > 0).sum()}/{H.size}")
    assert abs(g - c) <= 1e-12 * max(1.0, abs(c))

    # --- plot ----------------------------------------------------------- #
    out = cfg.out_dir
    extent = [0, cfg.world_size, 0, cfg.world_size]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    Hpos = np.where(H > 0, H, np.nan)  # mask zeros for log scale
    vmax = np.nanmax(Hpos)
    im = ax.imshow(
        Hpos,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
        aspect="equal",
    )
    # terrain contours to show where the ridge sits
    ax.contour(terrain.xs, terrain.ys, terrain.h, levels=10, colors="cyan", linewidths=0.4, alpha=0.5)
    for k, r in enumerate(cfg.radars):
        ax.plot(r.x, r.y, "X", color="lime", ms=12, mec="k")
        ax.add_patch(
            plt.Circle((r.x, r.y), r.R_max, fill=False, ec="lime", ls="--", lw=0.7, alpha=0.6)
        )
        ax.text(r.x + 12, r.y + 12, f"R{k}", color="lime", fontsize=9)
    ax.set_title(
        "Detection hazard  Lambda(x,y)  (log scale)\nridge masks the region behind it from radar R0"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    fig.colorbar(im, ax=ax, label="hazard (a.u.)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out / "risk_hazard.png", dpi=130)
    plt.close(fig)
    assert (out / "risk_hazard.png").exists()
