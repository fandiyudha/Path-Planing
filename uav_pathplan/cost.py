"""The shared cost function J and its components.

**Fairness contract.**  All three planners minimise *this* cost on *this* risk
field with *this* altitude policy.  The only thing that differs between planners
is the search mechanism, never the objective.  Both the full-path evaluator
(:func:`evaluate`) and the per-edge helpers used inside A* (:func:`edge_cost`)
live here so the three algorithms cannot drift apart.

A path ``pi`` is an ``(M, 2)`` array of horizontal waypoints ``p_0..p_N`` (so
``M = N + 1``).  Altitude ``z`` at each waypoint comes from the terrain's
altitude policy.  Three normalised components, each zero for an ideal path:

1. Normalised excess length
       L(pi)    = sum_i ||p_{i+1}-p_i||        (horizontal)
       L_ref    = ||p_g - p_s||
       Ltil     = (L - L_ref) / L_ref

2. Cumulative detection probability (survival model)
       Phi(pi)  = sum_i Lambda(midpoint_i) * ds_i
       Rtil     = 1 - exp(-Phi)                in [0, 1]

3. Kinematic-feasibility penalty (horizontal turn AND vertical climb/dive)
       kappa_i  = |dpsi_i| / sbar_i,   sbar_i = 0.5 (ds_{i-1}+ds_i),  kappa_max = 1/R_min
       gamma_i  = atan2(dz_i, ds_i),   limit |gamma_i| <= gamma_max
       Ktil     = mean_i [ max(0,(kappa_i-kappa_max)/kappa_max)
                          + max(0,(|gamma_i|-gamma_max)/gamma_max) ]

   The mean is over the ``N-1`` interior vertices; ``kappa_i`` uses the two
   segments adjacent to vertex ``i`` and ``gamma_i`` is the flight-path angle of
   the segment leaving vertex ``i``.

Combined:  J = wL*Ltil + wR*Rtil + wK*Ktil,  with wL+wR+wK = 1.

Normalisation rule: the planners use only the a-priori normaliser ``L_ref``
(known before planning).  Comparison tables may *additionally* report empirical
min-max normalised values, but empirical-range normalisation is never fed back
into the planners.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .risk import RiskField
from .terrain import TerrainField


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class CostResult:
    """Full breakdown of J for a path, plus geometry metrics for reporting."""

    J: float
    Ltil: float
    Rtil: float
    Ktil: float
    L: float  # horizontal path length (m)
    L_ref: float  # straight-line start->goal length (m)
    Phi: float  # cumulative hazard integral
    # geometry metrics (for the results tables)
    Lambda_peak: float  # max midpoint hazard along the path
    kappa_peak: float  # max discrete curvature (1/m)
    curv_violations: int  # # interior vertices with kappa > kappa_max
    gamma_peak: float  # max |flight-path angle| (rad)
    climb_violations: int  # # segments with |gamma| > gamma_max
    smoothness: float  # sum |dpsi| over interior vertices (rad)


def _segment_geometry(waypoints: np.ndarray):
    """Return per-segment vectors, lengths, and headings."""
    d = np.diff(waypoints, axis=0)  # (N, 2)
    ds = np.hypot(d[:, 0], d[:, 1])  # (N,)
    psi = np.arctan2(d[:, 1], d[:, 0])  # (N,)
    return d, ds, psi


def resample_polyline(waypoints: np.ndarray, spacing: float) -> np.ndarray:
    """Uniformly resample a polyline by arc length at ``spacing`` metres.

    Used to make the kinematic metrics (curvature, flight-path angle and their
    violation counts) spacing-independent: planners emit waypoints at different
    densities (A* ~one grid step, APF/Hybrid one descent step), which would make
    ``kappa = dpsi/ds`` and ``gamma = atan2(dz, ds)`` incomparable.  Resampling
    every path to the SAME arc-length grid gives one consistent, fair definition.
    Consecutive duplicate points are dropped so the arc-length axis is strictly
    increasing.
    """
    P = np.asarray(waypoints, dtype=float)
    if P.shape[0] < 2:
        return P
    keep = [0]
    for i in range(1, P.shape[0]):
        if np.hypot(*(P[i] - P[keep[-1]])) > 1e-9:
            keep.append(i)
    P = P[keep]
    if P.shape[0] < 2:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n = max(2, int(round(total / max(spacing, 1e-9))) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def evaluate(
    cfg: Config, risk: RiskField, terrain: TerrainField, waypoints: np.ndarray
) -> CostResult:
    """Compute J and all components/metrics for a horizontal path.

    ``waypoints`` is ``(M, 2)`` in world metres.  Altitudes are taken from the
    terrain altitude policy.  Degenerate (M < 2) paths get a large penalty J.
    """
    P = np.asarray(waypoints, dtype=float)
    if P.ndim != 2 or P.shape[0] < 2:
        return CostResult(
            J=1e6,
            Ltil=1e6,
            Rtil=1.0,
            Ktil=1e6,
            L=0.0,
            L_ref=0.0,
            Phi=0.0,
            Lambda_peak=0.0,
            kappa_peak=0.0,
            curv_violations=0,
            gamma_peak=0.0,
            climb_violations=0,
            smoothness=0.0,
        )

    p_s, p_g = P[0], P[-1]
    L_ref = float(np.hypot(*(p_g - p_s)))
    L_ref = max(L_ref, 1e-9)

    d, ds, psi = _segment_geometry(P)
    L = float(ds.sum())
    Ltil = (L - L_ref) / L_ref

    # --- detection: hazard at segment midpoints, integrated over arc length --
    mids = 0.5 * (P[:-1] + P[1:])  # (N, 2)
    haz_mid = np.asarray(risk.Lambda(mids[:, 0], mids[:, 1]))  # altitude policy
    Phi = float(np.sum(haz_mid * ds))
    Rtil = float(1.0 - np.exp(-Phi))
    Lambda_peak = float(haz_mid.max()) if haz_mid.size else 0.0

    # --- kinematics on a uniformly arc-length-RESAMPLED copy of the path -----
    # Curvature and flight-path angle are evaluated on a resampled polyline at a
    # fixed spacing (cfg.kin_resample_ds, default = grid resolution) so the
    # metrics are spacing-independent and IDENTICAL in definition for all three
    # planners.  Length (L) and detection (Phi/Rtil) above are computed on the
    # original path; only the kinematic terms use the resampled copy.
    ds_kin = cfg.kin_resample_ds if cfg.kin_resample_ds > 0 else cfg.resolution
    Q = resample_polyline(P, ds_kin)
    _, dsq, psiq = _segment_geometry(Q)
    zq = np.asarray(terrain.altitude_at(Q[:, 0], Q[:, 1]), dtype=float)
    dzq = np.diff(zq)
    gamma = np.arctan2(dzq, np.maximum(dsq, 1e-9))
    gamma_abs = np.abs(gamma)
    climb_violations = int(np.sum(gamma_abs > cfg.gamma_max))
    gamma_peak = float(gamma_abs.max()) if gamma_abs.size else 0.0

    # --- horizontal turning at interior vertices (resampled) ----------------
    N = dsq.size  # number of resampled segments
    if N >= 2:
        dpsi = wrap_angle(psiq[1:] - psiq[:-1])  # (N-1,) interior vertices
        dpsi_abs = np.abs(dpsi)
        sbar = 0.5 * (dsq[:-1] + dsq[1:])  # (N-1,)
        kappa = dpsi_abs / np.maximum(sbar, 1e-9)  # (N-1,)
        kappa_peak = float(kappa.max())
        curv_violations = int(np.sum(kappa > cfg.kappa_max))
        smoothness = float(dpsi_abs.sum())

        # combined kinematic penalty, averaged over interior vertices.
        kappa_pen = np.maximum(0.0, (kappa - cfg.kappa_max) / cfg.kappa_max)
        # gamma of the segment leaving each interior vertex i (segments 1..N-1)
        gamma_pen = np.maximum(0.0, (gamma_abs[1:] - cfg.gamma_max) / cfg.gamma_max)
        Ktil = float(np.mean(kappa_pen + gamma_pen))
    else:
        kappa_peak = 0.0
        curv_violations = 0
        smoothness = 0.0
        Ktil = (
            float(np.maximum(0.0, (gamma_abs[0] - cfg.gamma_max) / cfg.gamma_max))
            if gamma_abs.size
            else 0.0
        )

    J = cfg.wL * Ltil + cfg.wR * Rtil + cfg.wK * Ktil
    return CostResult(
        J=float(J),
        Ltil=float(Ltil),
        Rtil=Rtil,
        Ktil=Ktil,
        L=L,
        L_ref=L_ref,
        Phi=Phi,
        Lambda_peak=Lambda_peak,
        kappa_peak=kappa_peak,
        curv_violations=curv_violations,
        gamma_peak=gamma_peak,
        climb_violations=climb_violations,
        smoothness=smoothness,
    )


# --------------------------------------------------------------------------- #
# Per-edge helpers used by A* (same weights, same Lambda, same L_ref)          #
# --------------------------------------------------------------------------- #
def edge_length_cost(cfg: Config, a: np.ndarray, b: np.ndarray, L_ref: float) -> float:
    """wL * ||b-a|| / L_ref."""
    return cfg.wL * float(np.hypot(*(b - a))) / max(L_ref, 1e-9)


def edge_risk_cost(cfg: Config, risk: RiskField, a: np.ndarray, b: np.ndarray) -> float:
    """wR * Lambda(midpoint) * ||b-a||  (matches the Phi integrand)."""
    mid = 0.5 * (a + b)
    haz = float(risk.Lambda(mid[0], mid[1]))
    return cfg.wR * haz * float(np.hypot(*(b - a)))


def turn_curvature(psi_prev: float, psi_new: float, ds_prev: float, ds_new: float) -> float:
    """Discrete curvature kappa for a heading change between two segments."""
    dpsi = abs(float(wrap_angle(psi_new - psi_prev)))
    sbar = 0.5 * (ds_prev + ds_new)
    return dpsi / max(sbar, 1e-9)


def edge_kappa_penalty(cfg: Config, kappa: float) -> float:
    """wK * relative curvature violation (the A* turn penalty rho_kappa)."""
    return cfg.wK * max(0.0, (kappa - cfg.kappa_max) / cfg.kappa_max)
