"""Step-3 unit test: verify the cost function J by hand.

Converted from the former ``smoke_cost.py`` (same relative tolerance 1e-6).

Uses a FLAT terrain (fractal_max=0, no features -> h=0 everywhere, z=clearance)
so every quantity is analytic and can be recomputed in closed form here,
independently of cost.py, then compared.

Kinematic terms (kappa, gamma, Ktil, violation counts, smoothness) are evaluated
by ``cost.evaluate`` on a copy of the path resampled uniformly in arc length at
``Config.kin_resample_ds`` (0 -> grid resolution); L and Phi use the original
path.  The hand values below follow that definition.  (The former smoke script
computed curvature at the three original vertices, which predates the
resampling; that computation is kept as ``test_case_kinematics_vertex_spacing``,
where a 100 m resampling step reproduces the original vertices.)

Run:  pytest tests/test_cost.py
"""

import numpy as np

from uav_pathplan import cost as costmod
from uav_pathplan.config import Config, RadarSpec
from uav_pathplan.risk import RiskField
from uav_pathplan.terrain import TerrainField


def approx(name, got, exp, tol=1e-6):
    ok = abs(got - exp) <= tol * max(1.0, abs(exp))
    print(f"  {name:14s} got={got:.8f}  hand={exp:.8f}  {'OK' if ok else 'MISMATCH'}")
    return ok


# L-shaped path: 100 m east, then 100 m north (a 90-degree corner at (200, 100)).
L_PATH = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]])


def _evaluate_l_path(kin_resample_ds: float):
    """No radars, flat terrain, R_min=120; evaluate the L-shaped path."""
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
    assert cfg.resolution == 5.0  # Config default; used when kin_resample_ds == 0
    terrain = TerrainField.generate(cfg)
    risk = RiskField.build(cfg, terrain)
    return costmod.evaluate(cfg, risk, terrain, L_PATH)


def _check(res, *, kappa, Ktil, curv_violations, smoothness) -> bool:
    """Compare a CostResult with hand values (length terms are spacing-free)."""
    L = 100.0 + 100.0  # original path
    L_ref = np.hypot(100.0, 100.0)  # 141.42135624 (path's own endpoints)
    Ltil = (L - L_ref) / L_ref  # sqrt(2) - 1 = 0.41421356
    Rtil = 0.0  # no radars
    J = 0.40 * Ltil + 0.45 * Rtil + 0.15 * Ktil
    ok = all(
        [
            approx("L", res.L, L),
            approx("Ltil", res.Ltil, Ltil),
            approx("kappa_peak", res.kappa_peak, kappa),
            approx("Rtil", res.Rtil, Rtil),
            approx("Ktil", res.Ktil, Ktil),
            approx("J", res.J, J),
            approx("smoothness", res.smoothness, smoothness),
            approx("gamma_peak", res.gamma_peak, 0.0),  # flat terrain
        ]
    )
    print(
        f"  curv_violations={res.curv_violations} (expect {curv_violations}), "
        f"climb_violations={res.climb_violations} (expect 0)"
    )
    return ok and res.curv_violations == curv_violations and res.climb_violations == 0


def test_case_kinematics():
    """Case A with the default resampling (kin_resample_ds=0 -> 5 m grid step).

    Hand derivation for the resampled definition:

    * Total arc length 200 m; resampling step 5 m -> n = 200/5 + 1 = 41 points
      at s = 0, 5, ..., 200.  s = 100 is a sample point, so the corner (200, 100)
      is resampled exactly: 40 segments of 5 m, 20 heading east then 20 north.
    * Interior vertices k = 1..39 (N-1 = 39).  The heading change is 0 at every
      vertex except the corner (k = 20), where |dpsi| = pi/2.
    * Curvature at the corner: kappa = |dpsi| / sbar, sbar = (5 + 5)/2 = 5 m
          kappa_peak = (pi/2) / 5 = pi/10 = 0.31415927 1/m
      kappa_max = 1/R_min = 1/120 -> one violation (curv_violations = 1).
    * Curvature penalty at the corner: (kappa - kappa_max)/kappa_max
          = (pi/10)*120 - 1 = 12*pi - 1 = 36.69911184; 0 at the other 38 vertices.
    * Flat terrain: z = 50 m everywhere, gamma = 0 -> no climb penalty.
    * Ktil = mean over the 39 interior vertices = (12*pi - 1)/39 = 0.94100287
    * smoothness = sum |dpsi| = pi/2 = 1.57079633 rad
    * J = 0.40*(sqrt(2) - 1) + 0.45*0 + 0.15*(12*pi - 1)/39 = 0.30683586
    """
    print("Case A: kinematics on the 5 m resampled path (flat terrain, R_min=120)")
    res = _evaluate_l_path(kin_resample_ds=0.0)
    n_interior = 39
    kappa = (np.pi / 2) / 5.0
    kappa_max = 1.0 / 120.0
    Ktil = ((kappa - kappa_max) / kappa_max) / n_interior
    assert _check(res, kappa=kappa, Ktil=Ktil, curv_violations=1, smoothness=np.pi / 2)


def test_case_kinematics_vertex_spacing():
    """Case A with a 100 m resampling step, which reproduces the original vertices.

    This is the hand computation of the former smoke_cost.py: one interior
    vertex, sbar = 100 m, kappa = (pi/2)/100 = 0.01570796, penalty
    (kappa - 1/120)*120 = 0.88495559 = Ktil (mean over one vertex).
    """
    print("Case A': kinematics at the original vertices (kin_resample_ds=100)")
    res = _evaluate_l_path(kin_resample_ds=100.0)
    kappa = (np.pi / 2) / 100.0
    kappa_max = 1.0 / 120.0
    Ktil = (kappa - kappa_max) / kappa_max
    assert _check(res, kappa=kappa, Ktil=Ktil, curv_violations=1, smoothness=np.pi / 2)


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
