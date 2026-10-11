"""Step-3 unit test: verify the cost function J by hand.

Converted from the former ``smoke_cost.py`` (same hand computations and the
same relative tolerance 1e-6).

Uses a FLAT terrain (fractal_max=0, no features -> h=0 everywhere, z=clearance)
so every quantity is analytic and can be recomputed in closed form here,
independently of cost.py, then compared.

NOTE (kinematics case): the hand computation evaluates curvature at the three
ORIGINAL vertices.  ``cost.evaluate`` evaluates curvature/flight-path angle on a
copy of the path resampled at ``Config.kin_resample_ds`` (0 -> grid resolution,
5 m here), so the 90-degree corner is concentrated over one 5 m resampled
vertex and ``kappa_peak``/``Ktil``/``J`` differ from the hand values.  The
original script already reported "FAILURES PRESENT" for this case.  It is kept
unchanged as a strict xfail so the discrepancy stays visible; see
CLEANUP_REPORT.md.  ``test_case_kinematics_vertex_spacing`` repeats the same
hand computation with ``kin_resample_ds`` equal to the vertex spacing, where the
resampled path coincides with the original vertices.

Run:  pytest tests/test_cost.py
"""

import numpy as np
import pytest

from uav_pathplan import cost as costmod
from uav_pathplan.config import Config, RadarSpec
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField


def approx(name, got, exp, tol=1e-6):
    ok = abs(got - exp) <= tol * max(1.0, abs(exp))
    print(f"  {name:14s} got={got:.8f}  hand={exp:.8f}  {'OK' if ok else 'MISMATCH'}")
    return ok


def _kinematics(kin_resample_ds: float) -> bool:
    """No radars; an L-shaped 3-waypoint path; check length + curvature."""
    print("Case A: kinematics (no radar, flat terrain, R_min=120)")
    cfg = Config(
        fractal_max=0.0,
        hills=[],
        ridges=[],
        radars=[],
        clearance=50.0,
        R_min=120.0,
        gamma_max_deg=15.0,
        wL=0.40,
        wR=0.45,
        wK=0.15,
        kin_resample_ds=kin_resample_ds,
    )
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    P = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]])
    res = costmod.evaluate(cfg, risk, terrain, P)

    # --- hand computation ---
    L = 100.0 + 100.0
    L_ref = np.hypot(100.0, 100.0)  # 141.42135624
    Ltil = (L - L_ref) / L_ref  # 0.41421356
    dpsi = np.pi / 2  # east -> north
    sbar = 0.5 * (100 + 100)  # 100
    kappa = dpsi / sbar  # 0.0157080
    kappa_max = 1.0 / 120.0  # 0.00833333
    kappa_pen = (kappa - kappa_max) / kappa_max  # 0.884955
    Ktil = kappa_pen + 0.0  # gamma=0 (flat)
    Rtil = 0.0
    J = 0.40 * Ltil + 0.45 * Rtil + 0.15 * Ktil

    ok = all(
        [
            approx("L", res.L, L),
            approx("Ltil", res.Ltil, Ltil),
            approx("kappa_peak", res.kappa_peak, kappa),
            approx("Rtil", res.Rtil, Rtil),
            approx("Ktil", res.Ktil, Ktil),
            approx("J", res.J, J),
        ]
    )
    print(
        f"  curv_violations={res.curv_violations} (expect 1), "
        f"climb_violations={res.climb_violations} (expect 0)"
    )
    return ok and res.curv_violations == 1 and res.climb_violations == 0


@pytest.mark.xfail(
    strict=True,
    reason="hand calc uses original vertices; evaluate() resamples kinematics at the "
    "grid resolution (kin_resample_ds=0 -> 5 m). Pre-existing; see CLEANUP_REPORT.md",
)
def test_case_kinematics():
    """Original smoke_cost.py Case A, unchanged (default kin_resample_ds)."""
    assert _kinematics(kin_resample_ds=0.0)


def test_case_kinematics_vertex_spacing():
    """Case A with the resampling spacing equal to the 100 m vertex spacing."""
    assert _kinematics(kin_resample_ds=100.0)


def test_case_detection():
    """One analytic radar (mast 0, always LOS on flat ground); check Phi/Rtil."""
    print("\nCase B: detection (one radar, flat terrain)")
    cfg = Config(
        fractal_max=0.0,
        hills=[],
        ridges=[],
        clearance=50.0,
        R0=10.0,
        R_min=120.0,
        radars=[RadarSpec(x=200, y=150, mast_height=0.0, C=1.0e4, R_max=1.0e4)],
        wL=0.40,
        wR=0.45,
        wK=0.15,
    )
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)

    P = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]])
    res = costmod.evaluate(cfg, risk, terrain, P)

    # --- hand computation: z=50 everywhere, radar phase centre at (200,150,0) --
    C, R0 = 1.0e4, 10.0

    def lam(mx, my):
        R = np.sqrt((mx - 200) ** 2 + (my - 150) ** 2 + (50 - 0) ** 2)
        return C / max(R, R0) ** 4

    ds = 100.0
    lam0 = lam(150.0, 100.0)  # midpoint of seg 0
    lam1 = lam(200.0, 150.0)  # midpoint of seg 1 (overhead)
    Phi = lam0 * ds + lam1 * ds
    Rtil = 1.0 - np.exp(-Phi)
    print(f"  hand lam(mid0)={lam0:.8e}  lam(mid1)={lam1:.8e}")

    assert all(
        [
            approx("Phi", res.Phi, Phi),
            approx("Rtil", res.Rtil, Rtil),
            approx("Lambda_peak", res.Lambda_peak, max(lam0, lam1)),
        ]
    )
