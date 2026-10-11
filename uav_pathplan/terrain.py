"""Elevation field h(x, y) for the 2.5D world.

The terrain is the sum of two documented, separable components:

1. **Fractal base** -- value-noise / fractional Brownian motion (fBm).  We sum
   several octaves of smoothed random lattice noise: each octave draws a seeded
   random value lattice, bilinearly upsamples it to the full grid, and adds it
   with amplitude ``persistence**octave`` at frequency ``lacunarity**octave``.
   The accumulated base is rescaled to ``[0, fractal_max]`` metres so the
   stochastic relief has a controlled magnitude independent of octave count.

2. **Parametric features** -- a sum of 2D Gaussian hills and elongated, oriented
   Gaussian ridges placed at scripted locations.  These add their literal
   heights *on top of* the rescaled fractal base, so a scenario author can put a
   ridge of a known crest height exactly where a masking corridor is wanted.

Everything is driven by ``Config.seed`` through a ``numpy.random.Generator``.

The class exposes the discretised grid ``h`` (shape ``[n_cells, n_cells]``,
indexed ``[iy, ix]``) and a continuous bilinear callable :meth:`TerrainField.h_at`
used later by the line-of-sight check and the APF planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from .config import Config, GaussianHill, Ridge


# --------------------------------------------------------------------------- #
# Manual bilinear upsampling of a coarse lattice to the full grid              #
# --------------------------------------------------------------------------- #
def _bilinear_upsample(lattice: np.ndarray, n: int) -> np.ndarray:
    """Bilinearly resample a ``(m, m)`` value lattice onto an ``(n, n)`` grid.

    The lattice corner samples are mapped to the unit square [0, 1]^2 and the
    full grid's cell-centre coordinates are interpolated between them.  Written
    with explicit numpy gather + weighted blend (no scipy) to keep the fBm core
    fully hand-implemented and easy to audit.
    """
    m = lattice.shape[0]
    # Continuous sample coordinates (in lattice units) for each output cell.
    # Map output cell centres in [0, 1] onto lattice index range [0, m-1].
    t = (np.arange(n) + 0.5) / n * (m - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, m - 1)
    frac = t - i0  # blend weight in [0, 1)

    # Separable bilinear blend: interpolate along rows then columns.
    # Gather the four neighbouring lattice values per output pixel.
    fy = frac[:, None]  # (n, 1)
    fx = frac[None, :]  # (1, n)
    a = lattice[np.ix_(i0, i0)]  # top-left   (n, n)
    b = lattice[np.ix_(i0, i1)]  # top-right
    c = lattice[np.ix_(i1, i0)]  # bottom-left
    d = lattice[np.ix_(i1, i1)]  # bottom-right
    top = a + (b - a) * fx
    bot = c + (d - c) * fx
    return top + (bot - top) * fy


def _fractal_base(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Accumulate fBm octaves into a ``[n, n]`` field rescaled to [0, fractal_max]."""
    n = cfg.n_cells
    field = np.zeros((n, n), dtype=float)
    amplitude = cfg.base_amplitude
    lattice_size = cfg.base_lattice
    for _ in range(cfg.num_octaves):
        # +1 so the lattice has corner samples bounding the full grid.
        lattice = rng.standard_normal((lattice_size + 1, lattice_size + 1))
        field += amplitude * _bilinear_upsample(lattice, n)
        amplitude *= cfg.persistence
        lattice_size = int(round(lattice_size * cfg.lacunarity))

    # Rescale the (signed) accumulation to a controlled positive range.
    lo, hi = field.min(), field.max()
    if hi - lo < 1e-12:
        return np.zeros((n, n), dtype=float)
    base = (field - lo) / (hi - lo) * cfg.fractal_max

    # Mild Gaussian smoothing of the fractal BASE only (applied here, before any
    # scripted hills/ridges are added) to remove residual grid-scale roughness
    # so the directional slope -- and hence the nap-of-the-earth flight-path
    # angle -- stays within gamma_max.  Scripted masking features added later in
    # generate() are deliberately left sharp.
    if cfg.terrain_smooth_sigma > 0:
        sigma_cells = cfg.terrain_smooth_sigma / cfg.resolution
        base = gaussian_filter(base, sigma=sigma_cells, mode="nearest")
    return base


def _add_hill(field: np.ndarray, X: np.ndarray, Y: np.ndarray, hill: GaussianHill) -> None:
    """Add an isotropic Gaussian hill in place."""
    r2 = (X - hill.cx) ** 2 + (Y - hill.cy) ** 2
    field += hill.height * np.exp(-r2 / (2.0 * hill.sigma**2))


def _add_ridge(field: np.ndarray, X: np.ndarray, Y: np.ndarray, ridge: Ridge) -> None:
    """Add an elongated, oriented Gaussian ridge in place.

    Coordinates are rotated into the ridge frame: ``u`` along the axis, ``v``
    across it.  Along-axis the profile is a soft plateau of half-length
    ``length`` (flat-topped Gaussian via clamped distance), across-axis it is a
    Gaussian of width ``sigma``.
    """
    th = np.deg2rad(ridge.orientation_deg)
    ct, st = np.cos(th), np.sin(th)
    dx, dy = X - ridge.cx, Y - ridge.cy
    u = dx * ct + dy * st  # along axis
    v = -dx * st + dy * ct  # across axis
    # Distance past the plateau ends (0 inside the half-length).
    along = np.maximum(np.abs(u) - ridge.length, 0.0)
    field += ridge.height * np.exp(
        -(along**2) / (2.0 * ridge.sigma**2) - (v**2) / (2.0 * ridge.sigma**2)
    )


@dataclass
class TerrainField:
    """Discretised elevation field plus a continuous bilinear sampler."""

    cfg: Config
    h: np.ndarray  # [n_cells, n_cells], indexed [iy, ix]
    xs: np.ndarray  # cell-centre x coordinates (n_cells,)
    ys: np.ndarray  # cell-centre y coordinates (n_cells,)
    _interp: RegularGridInterpolator

    # ----------------------------------------------------------------- #
    @classmethod
    def generate(cls, cfg: Config) -> "TerrainField":
        """Build the elevation field from ``cfg`` (fully seeded)."""
        rng = np.random.default_rng(cfg.seed)
        xs, ys = cfg.cell_centers()
        X, Y = np.meshgrid(xs, ys)  # X,Y shape [n, n], row=y col=x

        h = _fractal_base(cfg, rng)
        for hill in cfg.hills:
            _add_hill(h, X, Y, hill)
        for ridge in cfg.ridges:
            _add_ridge(h, X, Y, ridge)
        np.maximum(h, cfg.elevation_floor, out=h)

        # Continuous sampler over (y, x) cell-centre grid; clamps to edges.
        interp = RegularGridInterpolator(
            (ys, xs),
            h,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        return cls(cfg=cfg, h=h, xs=xs, ys=ys, _interp=interp)

    # ----------------------------------------------------------------- #
    def h_at(self, x, y):
        """Continuous bilinear elevation at world point(s) (x, y).

        Accepts scalars or broadcastable arrays; returns matching shape.
        Out-of-bounds queries are clamped to the nearest edge value.
        """
        x = np.clip(np.asarray(x, dtype=float), self.xs[0], self.xs[-1])
        y = np.clip(np.asarray(y, dtype=float), self.ys[0], self.ys[-1])
        pts = np.stack([y.ravel(), x.ravel()], axis=-1)
        out = self._interp(pts).reshape(np.broadcast(x, y).shape)
        return float(out) if out.ndim == 0 else out

    def altitude_at(self, x, y):
        """UAV altitude z(x, y) under the configured altitude policy."""
        if self.cfg.altitude_mode == "constant":
            z = np.full_like(np.asarray(x, dtype=float), self.cfg.const_alt)
            return float(z) if z.ndim == 0 else z
        return self.h_at(x, y) + self.cfg.clearance

    @property
    def stats(self) -> dict:
        """Summary statistics of the elevation field."""
        return {
            "min": float(self.h.min()),
            "max": float(self.h.max()),
            "mean": float(self.h.mean()),
            "std": float(self.h.std()),
        }
