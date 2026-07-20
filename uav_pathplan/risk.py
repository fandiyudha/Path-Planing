"""Radar threat / detection-hazard field with terrain-masked line-of-sight.

Hazard law
----------
For radar ``j`` at ground position ``q_j`` with antenna phase centre at
elevation ``h(q_j) + mast_height``, the detection hazard at a query point
``p = (x, y, z)`` is

    lambda_j(p) = V_j(p) * C_j / max(R_j(p), R0)**4 * 1[R_j(p) <= R_max_j]

where ``R_j(p)`` is the full 3D Euclidean distance from the radar phase centre
to ``p`` and ``V_j(p)`` is binary line-of-sight visibility (1 = unobstructed).
The total hazard is ``Lambda(p) = sum_j lambda_j(p)``.

Line of sight (manual)
----------------------
``V_j(p)`` is computed by marching along the 3D segment radar->p in fixed steps
of ``los_step`` metres and comparing the straight-line height to the terrain
elevation ``h(x, y)`` at each sample.  If terrain rises above the line anywhere
strictly between the endpoints, the point is masked (``V = 0``).  No external
LOS/raster helper is used.

This module provides BOTH:
  (a) a continuous callable :meth:`RiskField.Lambda` over ``(x, y, z)`` that APF
      can sample and numerically differentiate, and
  (b) a precomputed hazard grid at planning altitude (:meth:`RiskField.grid`)
      that A* reads.
They agree at grid points by construction (same law, same altitude policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .config import Config, RadarSpec
from .terrain import TerrainField


def calibrate_C(R_max: float, R0: float, ref_range: float,
                p_ref: float) -> float:
    """Solve the radar power constant C from a documented reference detection.

    Reference geometry (fully reproducible):
      A straight, level aircraft transit that crosses the radar's coverage disc,
      passing at closest slant range ``ref_range`` from the phase centre, with
      clear line of sight (V = 1) along the whole in-disc portion.  Parametrising
      by along-track distance ``s`` from closest approach, the slant range is
      ``R(s) = sqrt(ref_range**2 + s**2)`` and the transit spans ``|s| <= s_max``
      with ``s_max = sqrt(R_max**2 - ref_range**2)`` (i.e. from disc edge to disc
      edge).  The survival-model exposure of this transit is

          Phi_ref(C) = C * I,   I = integral_{-s_max}^{s_max} ds / max(R(s), R0)**4

    which is linear in C.  We want that transit to be detected with probability
    ``p_ref``, i.e. Phi_ref = -ln(1 - p_ref), so

          C = -ln(1 - p_ref) / I .

    (When ``ref_range >= R0`` -- the usual case, mid-range vs a small floor --
    the floor is inactive along the transit and ``I`` has the closed form
    ``s_max/(ref_range**2 * R_max**2) + atan(s_max/ref_range)/ref_range**3``;
    we integrate numerically so the general case is also handled.)
    """
    ref_range = min(max(ref_range, 1e-6), 0.999 * R_max)
    s_max = float(np.sqrt(max(R_max ** 2 - ref_range ** 2, 0.0)))
    s = np.linspace(-s_max, s_max, 4001)
    R = np.sqrt(ref_range ** 2 + s ** 2)
    integrand = 1.0 / np.maximum(R, R0) ** 4
    I = float(np.trapezoid(integrand, s)) if hasattr(np, "trapezoid") \
        else float(np.trapz(integrand, s))
    phi_target = -np.log(max(1.0 - p_ref, 1e-12))
    return phi_target / max(I, 1e-300)


@dataclass
class RiskField:
    """Detection-hazard field built on a terrain and a list of radars."""

    cfg: Config
    terrain: TerrainField
    radars: list[RadarSpec]
    # Per-radar phase-centre elevation (ground h at radar + mast height).
    _radar_z: np.ndarray                # shape (J,)
    _radar_C: np.ndarray                # shape (J,), effective power constant
    _grid: np.ndarray | None = None     # cached hazard grid at planning altitude
    _interp: RegularGridInterpolator | None = field(default=None, repr=False)

    # ----------------------------------------------------------------- #
    @classmethod
    def build(cls, cfg: Config, terrain: TerrainField,
              radars: list[RadarSpec] | None = None) -> "RiskField":
        """Construct the risk field; radars default to ``cfg.radars``.

        Each radar's power constant C is taken from ``RadarSpec.C`` when set
        (an explicit override), otherwise DERIVED by :func:`calibrate_C` from the
        reference detection point in ``cfg`` -- so radar strength follows one
        documented physical assumption rather than a magic number.
        """
        radars = list(cfg.radars if radars is None else radars)
        rz = np.array(
            [terrain.h_at(r.x, r.y) + r.mast_height for r in radars],
            dtype=float,
        )
        cvals = []
        for r in radars:
            if r.C is not None:
                cvals.append(float(r.C))
            else:
                ref_range = (cfg.ref_range_m if cfg.ref_range_m is not None
                             else cfg.ref_range_frac * r.R_max)
                cvals.append(calibrate_C(r.R_max, cfg.R0, ref_range,
                                         cfg.ref_detection_prob))
        rc = np.array(cvals, dtype=float)
        return cls(cfg=cfg, terrain=terrain, radars=radars,
                   _radar_z=rz, _radar_C=rc)

    # ----------------------------------------------------------------- #
    # Line of sight                                                      #
    # ----------------------------------------------------------------- #
    def visible(self, radar_idx: int, x: float, y: float, z: float) -> bool:
        """Binary LOS from radar ``radar_idx`` to world point (x, y, z).

        Marches the segment in ``los_step`` increments and returns False if the
        terrain rises above the straight line at any interior sample.
        """
        r = self.radars[radar_idx]
        rx, ry, rz = r.x, r.y, float(self._radar_z[radar_idx])
        dx, dy, dz = x - rx, y - ry, z - rz
        horiz = float(np.hypot(dx, dy))
        if horiz < 1e-9:
            return True                                  # directly overhead
        n_samples = max(2, int(np.ceil(horiz / self.cfg.los_step)))
        # Interior sample fractions (exclude both endpoints).
        t = np.linspace(0.0, 1.0, n_samples + 1)[1:-1]
        if t.size == 0:
            return True
        sx = rx + dx * t
        sy = ry + dy * t
        line_z = rz + dz * t
        terr = self.terrain.h_at(sx, sy)
        # Masked if terrain is above the line anywhere (small tolerance).
        return not bool(np.any(terr > line_z + 1e-6))

    def _visible_vec(self, radar_idx: int, X: np.ndarray, Y: np.ndarray,
                     Z: np.ndarray) -> np.ndarray:
        """Vectorised LOS for many query points against one radar.

        Returns a boolean array (1 = visible).  We march all segments together:
        a common parametric grid of ``n_steps`` fractions is applied to every
        point, terrain is sampled in one batched bilinear call per step, and a
        running OR accumulates any blocking.  ``n_steps`` is sized from the
        longest segment so short segments are oversampled (harmless).
        """
        r = self.radars[radar_idx]
        rx, ry, rz = r.x, r.y, float(self._radar_z[radar_idx])
        DX, DY, DZ = X - rx, Y - ry, Z - rz
        horiz = np.hypot(DX, DY)
        n_steps = max(2, int(np.ceil(float(horiz.max()) / self.cfg.los_step)))
        blocked = np.zeros(X.shape, dtype=bool)
        # Interior fractions only.
        for t in np.linspace(0.0, 1.0, n_steps + 1)[1:-1]:
            sx = rx + DX * t
            sy = ry + DY * t
            line_z = rz + DZ * t
            terr = self.terrain.h_at(sx, sy)
            blocked |= terr > line_z + 1e-6
        # Points with ~zero horizontal distance are trivially visible.
        blocked &= horiz > 1e-9
        return ~blocked

    # ----------------------------------------------------------------- #
    # Continuous hazard callable                                         #
    # ----------------------------------------------------------------- #
    def Lambda(self, x, y, z=None) -> float | np.ndarray:
        """Total detection hazard at world point(s) (x, y, z).

        If ``z`` is None the configured altitude policy supplies it (planning
        altitude).  Accepts scalars or broadcastable arrays.  This is the
        callable APF samples and finite-differences.
        """
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)
        if z is None:
            Z = np.asarray(self.terrain.altitude_at(X, Y), dtype=float)
        else:
            Z = np.broadcast_to(np.asarray(z, dtype=float), X.shape).astype(float)

        scalar = (X.ndim == 0)
        Xb, Yb, Zb = np.broadcast_arrays(X, Y, Z)
        total = np.zeros(Xb.shape, dtype=float)
        for j, r in enumerate(self.radars):
            rx, ry, rz = r.x, r.y, float(self._radar_z[j])
            R = np.sqrt((Xb - rx) ** 2 + (Yb - ry) ** 2 + (Zb - rz) ** 2)
            in_range = R <= r.R_max
            if not np.any(in_range):
                continue
            vis = self._visible_vec(j, Xb, Yb, Zb)
            denom = np.maximum(R, self.cfg.R0) ** 4
            contrib = np.where(in_range & vis, self._radar_C[j] / denom, 0.0)
            total = total + contrib
        return float(total) if scalar else total

    # ----------------------------------------------------------------- #
    # Precomputed grid at planning altitude                              #
    # ----------------------------------------------------------------- #
    def grid(self, recompute: bool = False) -> np.ndarray:
        """Hazard grid ``[n_cells, n_cells]`` (indexed [iy, ix]) at planning alt.

        Vectorised over the whole grid; cached after first call.
        """
        if self._grid is not None and not recompute:
            return self._grid
        xs, ys = self.terrain.xs, self.terrain.ys
        X, Y = np.meshgrid(xs, ys)                       # [n, n], row=y col=x
        self._grid = np.asarray(self.Lambda(X, Y))      # altitude from policy
        # Build the fast bilinear interpolant over the grid (used by APF).
        self._interp = RegularGridInterpolator(
            (ys, xs), self._grid, method="linear",
            bounds_error=False, fill_value=None)
        return self._grid

    # ----------------------------------------------------------------- #
    # Fast continuous callable for APF (interpolated from the grid)      #
    # ----------------------------------------------------------------- #
    def Lambda_field(self, x, y) -> float | np.ndarray:
        """Fast continuous hazard at planning altitude via bilinear interp.

        This is the continuous callable APF samples and finite-differences.  It
        is built from the precomputed hazard grid, so it agrees *exactly* with
        the grid at grid points (the fairness requirement) while being orders of
        magnitude cheaper than recomputing LOS per query.  For points outside
        the grid it extrapolates linearly from the edge (then is clamped by the
        caller staying inside the world).
        """
        if self._interp is None:
            self.grid()                                  # builds _interp
        xa = np.clip(np.asarray(x, dtype=float),
                     self.terrain.xs[0], self.terrain.xs[-1])
        ya = np.clip(np.asarray(y, dtype=float),
                     self.terrain.ys[0], self.terrain.ys[-1])
        scalar = (xa.ndim == 0)
        pts = np.stack([np.atleast_1d(ya).ravel(),
                        np.atleast_1d(xa).ravel()], axis=-1)
        out = np.maximum(self._interp(pts), 0.0)
        if scalar:
            return float(out[0])
        return out.reshape(np.broadcast(xa, ya).shape)

    def hazard_along(self, x, y, z=None):
        """Convenience: exact hazard sampled at arbitrary points (vector friendly)."""
        return self.Lambda(x, y, z)
