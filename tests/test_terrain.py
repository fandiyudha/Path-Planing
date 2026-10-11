"""Step-1 smoke test: build a terrain field and save sample plots.

Converted from the former ``smoke_terrain.py``.  Figures are written to a
temporary directory (pytest ``tmp_path``) instead of ``outputs/``.

Run:  pytest tests/test_terrain.py
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt
import numpy as np

from uav_pathplan.config import Config, GaussianHill, Ridge
from uav_pathplan.terrain import TerrainField


def _cfg() -> Config:
    return Config(
        seed=20240603,
        world_size=1000.0,
        resolution=5.0,
        num_octaves=6,
        persistence=0.55,
        lacunarity=2.0,
        fractal_max=180.0,
        hills=[
            GaussianHill(cx=250, cy=750, height=120, sigma=90),
            GaussianHill(cx=800, cy=300, height=90, sigma=70),
        ],
        ridges=[
            # A masking ridge across the middle, oriented 30 deg from +x.
            Ridge(cx=500, cy=500, height=160, sigma=45, length=260, orientation_deg=30.0),
        ],
    )


def test_terrain_field(tmp_path):
    cfg = replace(_cfg(), out_dir=tmp_path)
    cfg.validate()

    terrain = TerrainField.generate(cfg)
    print(f"seed = {cfg.seed}")
    print(
        f"grid = {cfg.n_cells} x {cfg.n_cells} cells "
        f"({cfg.world_size:.0f} m / {cfg.resolution:.0f} m)"
    )
    s = terrain.stats
    print(
        f"elevation: min={s['min']:.1f}  max={s['max']:.1f}  "
        f"mean={s['mean']:.1f}  std={s['std']:.1f}  (m)"
    )
    assert terrain.h.shape == (cfg.n_cells, cfg.n_cells)

    # Spot-check the continuous sampler agrees with the grid at a cell centre.
    ix, iy = 100, 80
    xc, yc = terrain.xs[ix], terrain.ys[iy]
    grid_val = terrain.h[iy, ix]
    cont_val = terrain.h_at(xc, yc)
    print(
        f"sampler check at ({xc:.1f},{yc:.1f}): "
        f"grid={grid_val:.4f}  continuous={cont_val:.4f}  "
        f"|diff|={abs(grid_val - cont_val):.2e}"
    )
    assert abs(grid_val - cont_val) < 1e-9

    out = cfg.out_dir
    extent = [0, cfg.world_size, 0, cfg.world_size]

    # 2D heatmap with contour lines.
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(terrain.h, origin="lower", extent=extent, cmap="terrain", aspect="equal")
    cs = ax.contour(
        terrain.xs, terrain.ys, terrain.h, levels=12, colors="k", linewidths=0.4, alpha=0.5
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    ax.set_title(f"Elevation field h(x,y)  (seed={cfg.seed})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="elevation (m)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(out / "terrain_2d.png", dpi=130)
    plt.close(fig)

    # 3D surface.
    X, Y = np.meshgrid(terrain.xs, terrain.ys)
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot_surface(
        X, Y, terrain.h, cmap="terrain", linewidth=0, antialiased=True, rcount=120, ccount=120
    )
    ax3d.set_title("Elevation surface")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("h (m)")
    ax3d.view_init(elev=40, azim=-60)
    fig.tight_layout()
    fig.savefig(out / "terrain_3d.png", dpi=130)
    plt.close(fig)

    assert (out / "terrain_2d.png").exists()
    assert (out / "terrain_3d.png").exists()
