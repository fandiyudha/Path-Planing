"""Protocol-v2 path evaluator.

Differences from the v1 evaluator (:func:`uav_pathplan.cost.evaluate`):

1. ``L_ref = ||goal - start||`` with the SCENARIO start and goal (v1: the
   path's own first and last points).
2. ``L`` is the closed length ``|p_0 - start| + sum ||p_{i+1} - p_i|| +
   |p_N - goal|`` so that a path that stops within the goal tolerance (APF,
   APF-E, Hybrid) or starts/ends at a snapped cell centre (A-Star) is not
   credited with a shorter-than-straight length; hence ``Ltil >= 0``.  The end
   gaps are reported as ``gap_start`` / ``gap_goal``.  Phi and the kinematic
   terms use the returned path only.
3. Altitude comes from an altitude mode (:mod:`uav_pathplan.v2.altitude`), and
   the hazard (including line of sight) is evaluated at that actual ``z``.
4. The flight-path angle uses arc length along the path,
   ``gamma_k = atan2(dz_k, ds_k)`` with ``ds_k`` the (uniform) arc-length step of
   the resampled path; v1 used the chord between resampled points, which
   differs only where a path vertex falls inside a resampling interval.

Unchanged from v1: Phi = sum Lambda(midpoint) * ds on the returned segments,
Rtil = 1 - exp(-Phi), curvature kappa_k = |dpsi_k| / mean adjacent chord on the
path resampled at ``kin_resample_ds`` (0 -> grid resolution), Ktil = mean over
interior vertices of the curvature and flight-path-angle penalties,
J = wL Ltil + wR Rtil + wK Ktil, smoothness = sum |dpsi| (rad).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..cost import wrap_angle
from ..risk import RiskField
from ..terrain import TerrainField
from .altitude import arc_length, build_profile, dedup_polyline, points_at

METRIC_KEYS = [
    "J",
    "L",
    "L_path",
    "L_ref",
    "Ltil",
    "gap_start",
    "gap_goal",
    "Rtil",
    "Phi",
    "Lambda_peak",
    "Ktil",
    "d_min",
    "kappa_peak",
    "curv_violations",
    "n_interior",
    "curv_viol_frac",
    "gamma_peak_deg",
    "climb_violations",
    "n_segments",
    "climb_viol_frac",
    "smoothness",
    "agl_mean",
    "agl_min",
]


def resample_with_arclength(P: np.ndarray, spacing: float):
    """Same points as :func:`uav_pathplan.cost.resample_polyline`, plus their arc lengths."""
    Pd = dedup_polyline(P)
    if Pd.shape[0] < 2:
        return Pd, np.zeros(Pd.shape[0])
    s = arc_length(Pd)
    total = float(s[-1])
    n = max(2, int(round(total / max(spacing, 1e-9))) + 1)
    t = np.linspace(0.0, total, n)
    return points_at(Pd, s, t), t


def evaluate_v2(
    cfg: Config,
    risk: RiskField,
    terrain: TerrainField,
    path: np.ndarray,
    start,
    goal,
    mode: str,
    c_min: float,
    profile_ds: float,
) -> dict:
    """All v2 metrics for one path under one altitude mode (NaN if degenerate)."""
    out = {k: np.nan for k in METRIC_KEYS}
    P = np.asarray(path, dtype=float)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    out["L_ref"] = L_ref = max(float(np.hypot(*(goal - start))), 1e-9)
    if P.ndim != 2 or P.shape[0] < 2 or dedup_polyline(P).shape[0] < 2:
        return out

    # --- length (closed) ------------------------------------------------- #
    ds = np.hypot(*np.diff(P, axis=0).T)
    L_path = float(ds.sum())
    gap_start = float(np.hypot(*(P[0] - start)))
    gap_goal = float(np.hypot(*(P[-1] - goal)))
    L = gap_start + L_path + gap_goal
    Ltil = (L - L_ref) / L_ref

    # --- arc-length positions of every query point ---------------------- #
    s_vert = arc_length(P)
    s_mid = s_vert[:-1] + 0.5 * ds
    mids = 0.5 * (P[:-1] + P[1:])
    ds_kin = cfg.kin_resample_ds if cfg.kin_resample_ds > 0 else cfg.resolution
    Q, t = resample_with_arclength(P, ds_kin)

    prof = build_profile(
        terrain,
        P,
        np.concatenate([s_vert, s_mid, t]),
        mode,
        cfg.clearance,
        c_min,
        cfg.gamma_max,
        profile_ds,
    )

    # --- detection along the returned path, at the mode's altitude ------ #
    z_mid = prof.z_at(s_mid)
    haz = np.asarray(risk.Lambda(mids[:, 0], mids[:, 1], z_mid), dtype=float).reshape(-1)
    Phi = float(np.sum(haz * ds))
    Rtil = float(1.0 - np.exp(-Phi))
    Lambda_peak = float(haz.max())

    # --- kinematics on the resampled path ------------------------------- #
    dq = np.diff(Q, axis=0)
    chord = np.hypot(dq[:, 0], dq[:, 1])
    psi = np.arctan2(dq[:, 1], dq[:, 0])
    zq = prof.z_at(t)
    ds_arc = np.diff(t)
    gamma = np.arctan2(np.diff(zq), np.maximum(ds_arc, 1e-9))
    gamma_abs = np.abs(gamma)
    n_seg = int(gamma.size)
    climb_violations = int(np.sum(gamma_abs > cfg.gamma_max))
    if n_seg >= 2:
        dpsi_abs = np.abs(wrap_angle(psi[1:] - psi[:-1]))
        sbar = 0.5 * (chord[:-1] + chord[1:])
        kappa = dpsi_abs / np.maximum(sbar, 1e-9)
        kappa_pen = np.maximum(0.0, (kappa - cfg.kappa_max) / cfg.kappa_max)
        gamma_pen = np.maximum(0.0, (gamma_abs[1:] - cfg.gamma_max) / cfg.gamma_max)
        Ktil = float(np.mean(kappa_pen + gamma_pen))
        kappa_peak = float(kappa.max())
        curv_violations = int(np.sum(kappa > cfg.kappa_max))
        smoothness = float(dpsi_abs.sum())
        n_int = n_seg - 1
    else:
        Ktil = float(max(0.0, (gamma_abs[0] - cfg.gamma_max) / cfg.gamma_max))
        kappa_peak, curv_violations, smoothness, n_int = 0.0, 0, 0.0, 0

    # --- standoff at the mode's altitude -------------------------------- #
    z_vert = prof.z_at(s_vert)
    d_min = np.nan
    if risk.radars:
        best = np.inf
        for j, r in enumerate(risk.radars):
            rz = float(risk._radar_z[j])
            d = np.sqrt((P[:, 0] - r.x) ** 2 + (P[:, 1] - r.y) ** 2 + (z_vert - rz) ** 2)
            best = min(best, float(d.min()))
        d_min = best

    agl = prof.z - prof.h
    J = cfg.wL * Ltil + cfg.wR * Rtil + cfg.wK * Ktil
    out.update(
        J=float(J),
        L=L,
        L_path=L_path,
        Ltil=float(Ltil),
        gap_start=gap_start,
        gap_goal=gap_goal,
        Rtil=Rtil,
        Phi=Phi,
        Lambda_peak=Lambda_peak,
        Ktil=Ktil,
        d_min=d_min,
        kappa_peak=kappa_peak,
        curv_violations=curv_violations,
        n_interior=n_int,
        curv_viol_frac=curv_violations / n_int if n_int else 0.0,
        gamma_peak_deg=float(np.rad2deg(gamma_abs.max())),
        climb_violations=climb_violations,
        n_segments=n_seg,
        climb_viol_frac=climb_violations / n_seg if n_seg else 0.0,
        smoothness=smoothness,
        agl_mean=float(agl.mean()),
        agl_min=float(agl.min()),
    )
    return out
