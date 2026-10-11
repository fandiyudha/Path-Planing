"""Altitude profiles along a horizontal path (protocol v2).

Both modes are applied identically to every planner as a post-processing step
on the returned horizontal path; the planners themselves are unchanged.

* ``nap``: ``z = h(x, y) + clearance`` (the v1 altitude policy).
* ``rate_limited``: the lowest profile with ``z >= h + c_min`` and
  ``|dz/ds| <= tan(gamma_max)``, where ``s`` is arc length along the path.

For lower bounds ``b_j = h(s_j) + c_min`` at chain points ``s_j`` the lowest
``t``-Lipschitz profile above them is the lower envelope of cones

    z(s_i) = max_j ( b_j - t * |s_i - s_j| ),     t = tan(gamma_max),

computed exactly in O(n) with two running maxima:

    forward_i  = max_{j<=i} (b_j + t s_j) - t s_i
    backward_i = max_{j>=i} (b_j - t s_j) + t s_i
    z_i        = max(forward_i, backward_i).

Consecutive chain points then satisfy ``|z_{i+1} - z_i| <= t (s_{i+1} - s_i)``
and ``z_i >= b_i``.  The terrain constraint is enforced at the chain points; the
chain includes a uniform sampling of the path at ``profile_ds`` metres.

The slope limit is applied as ``t = tan(gamma_max) * (1 - SLOPE_MARGIN)`` with a
relative margin of 1e-9, so that floating-point rounding (~1e-15 rad) on
segments flown exactly at ``gamma_max`` cannot register as a climb violation
under the strict ``|gamma| > gamma_max`` test.  The altitude change this causes
is below 1e-6 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain import TerrainField

ALTITUDE_MODES = ("nap", "rate_limited")
SLOPE_MARGIN = 1e-9


def dedup_polyline(P: np.ndarray) -> np.ndarray:
    """Drop consecutive duplicate points (as ``cost.resample_polyline`` does)."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 2:
        return P
    keep = [0]
    for i in range(1, P.shape[0]):
        if np.hypot(*(P[i] - P[keep[-1]])) > 1e-9:
            keep.append(i)
    return P[keep]


def arc_length(P: np.ndarray) -> np.ndarray:
    """Cumulative arc length at each vertex of ``P`` (duplicates allowed)."""
    P = np.asarray(P, dtype=float)
    if P.shape[0] == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])


def points_at(P_dedup: np.ndarray, s_dedup: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Points at arc lengths ``s_query`` on a de-duplicated polyline."""
    if P_dedup.shape[0] == 1:
        return np.repeat(P_dedup, len(s_query), axis=0)
    x = np.interp(s_query, s_dedup, P_dedup[:, 0])
    y = np.interp(s_query, s_dedup, P_dedup[:, 1])
    return np.column_stack([x, y])


def lower_envelope(s: np.ndarray, floor: np.ndarray, slope: float) -> np.ndarray:
    """Lowest ``slope``-Lipschitz function (in ``s``) that is >= ``floor``.

    ``s`` must be non-decreasing.  Exact: ``z_i = max_j(floor_j - slope*|s_i - s_j|)``.
    """
    s = np.asarray(s, dtype=float)
    b = np.asarray(floor, dtype=float)
    fwd = np.maximum.accumulate(b + slope * s) - slope * s
    bwd = (np.maximum.accumulate((b - slope * s)[::-1]))[::-1] + slope * s
    return np.maximum(fwd, bwd)


@dataclass
class Profile:
    """Altitude along a path, sampled on a chain of arc lengths."""

    mode: str
    s: np.ndarray  # chain arc lengths (sorted, unique)
    xy: np.ndarray  # (n, 2) chain points
    h: np.ndarray  # terrain elevation at chain points
    z: np.ndarray  # altitude at chain points

    def z_at(self, s_query) -> np.ndarray:
        """Altitude at arc lengths ``s_query`` (exact if they are chain points)."""
        return np.interp(np.asarray(s_query, dtype=float), self.s, self.z)


def build_profile(
    terrain: TerrainField,
    P: np.ndarray,
    s_extra: np.ndarray,
    mode: str,
    clearance: float,
    c_min: float,
    gamma_max: float,
    profile_ds: float,
) -> Profile:
    """Altitude profile along path ``P`` for ``mode``.

    The chain contains the arc lengths ``s_extra`` (all points at which the
    caller will query ``z``) plus a uniform sampling every ``profile_ds`` m.
    """
    if mode not in ALTITUDE_MODES:
        raise ValueError(f"unknown altitude mode {mode!r}; choices {ALTITUDE_MODES}")
    Pd = dedup_polyline(P)
    sd = arc_length(Pd)
    total = float(sd[-1]) if sd.size else 0.0
    s_fine = np.append(np.arange(0.0, total, profile_ds), total)
    s = np.unique(np.concatenate([np.asarray(s_extra, dtype=float), s_fine]))
    xy = points_at(Pd, sd, s)
    h = np.asarray(terrain.h_at(xy[:, 0], xy[:, 1]), dtype=float).reshape(-1)
    if mode == "nap":
        z = h + clearance
    else:
        z = lower_envelope(s, h + c_min, float(np.tan(gamma_max)) * (1.0 - SLOPE_MARGIN))
    return Profile(mode=mode, s=s, xy=xy, h=h, z=z)
