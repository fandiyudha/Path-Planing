"""Central configuration for the UAV path-planning simulation.

A single :class:`Config` dataclass holds every parameter used anywhere in the
framework (world geometry, terrain synthesis, radar/risk model, cost weights,
kinematic limits, planner settings, Monte-Carlo control, output paths).  No
module should hard-code a parameter mid-function; pull it from a ``Config``.

All randomness is driven by the integer ``seed`` via ``numpy.random.Generator``
so every run is bit-for-bit reproducible.  The seed is printed/saved with
outputs.

Coordinate convention
---------------------
The world is a square of side ``world_size`` metres, discretised on a regular
grid of ``n_cells`` x ``n_cells`` square cells of side ``resolution`` metres.
Grid arrays are indexed ``field[iy, ix]`` (row = y, col = x), the natural
orientation for ``imshow(origin='lower')``.  Cell ``(ix, iy)`` is centred at
world coordinate ``((ix + 0.5) * resolution, (iy + 0.5) * resolution)``.  Use
:meth:`Config.world_to_grid` / :meth:`Config.grid_to_world` everywhere instead
of recomputing the mapping by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Parametric terrain features                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class GaussianHill:
    """An isotropic 2D Gaussian hill added to the elevation field.

    cx, cy  : centre in world metres
    height  : peak elevation contribution in metres
    sigma   : standard deviation (footprint radius scale) in metres
    """

    cx: float
    cy: float
    height: float
    sigma: float


@dataclass
class Ridge:
    """An elongated, oriented Gaussian ridge.

    The ridge is a Gaussian that is wide along its axis (``length``) and narrow
    across it (``sigma``), rotated by ``orientation_deg`` (CCW from +x).  Place
    a ridge between start and goal to create a masking corridor behind it.

    cx, cy          : centre in world metres
    height          : crest elevation contribution in metres
    sigma           : half-width across the ridge axis (metres)
    length          : half-length along the ridge axis (metres)
    orientation_deg : axis orientation in degrees, CCW from +x
    """

    cx: float
    cy: float
    height: float
    sigma: float
    length: float
    orientation_deg: float


# --------------------------------------------------------------------------- #
# Radar threat specification                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RadarSpec:
    """A single radar threat.

    x, y        : ground position in world metres
    mast_height : antenna phase-centre height above local ground elevation (m)
    C           : power constant C_j in the hazard law.  ``None`` (default) means
                  DERIVE it by calibration from the reference detection point in
                  Config (see risk.calibrate_C); set an explicit value only to
                  override the calibration for a particular radar (e.g. a
                  proxy-obstacle wall).
    R_max       : maximum detection range (m); hazard is zero beyond it
    """

    x: float
    y: float
    mast_height: float = 20.0
    C: float | None = None
    R_max: float = 600.0


# --------------------------------------------------------------------------- #
# Master configuration                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """Every tunable parameter for one experiment/scenario."""

    # ---- reproducibility ------------------------------------------------- #
    seed: int = 20240603

    # ---- world geometry -------------------------------------------------- #
    world_size: float = 1000.0  # side length of the square world (m)
    resolution: float = 5.0  # grid cell size (m) -> 200x200 default

    # ---- terrain: fractal base (value-noise fBm) ------------------------- #
    # Defaults tuned so the fractal base is gentle enough that nap-of-the-earth
    # flight (z = h + clearance, so flight-path angle == directional terrain
    # slope) is feasible under gamma_max: faster amplitude decay (lower
    # persistence) suppresses the high-frequency octaves that dominate slope,
    # and a mild post-generation Gaussian smooth removes residual grid-scale
    # roughness.  See terrain.py.
    num_octaves: int = 5  # number of fBm octaves
    persistence: float = 0.42  # amplitude decay per octave (<1); lower
    # => high-frequency (steep) octaves decay
    # faster => gentler slopes
    lacunarity: float = 2.0  # frequency growth per octave (>1)
    base_amplitude: float = 1.0  # relative amplitude of octave 0
    base_lattice: int = 3  # lattice cells per side at octave 0
    fractal_max: float = 200.0  # fBm base rescaled to [0, fractal_max] (m)
    terrain_smooth_sigma: float = 18.0  # Gaussian smooth of the fractal base
    # (metres, 0 = off); applied AFTER the
    # rescale and BEFORE scripted features,
    # so masking ridges/hills stay sharp

    # ---- terrain: parametric features ------------------------------------ #
    hills: list[GaussianHill] = field(default_factory=list)
    ridges: list[Ridge] = field(default_factory=list)
    elevation_floor: float = 0.0  # clamp terrain below this to this (m)

    # ---- altitude policy ------------------------------------------------- #
    clearance: float = 50.0  # nap-of-the-earth clearance above terrain (m)
    altitude_mode: str = "nap"  # "nap" -> h+clearance, "constant" -> const_alt
    const_alt: float = 150.0  # used when altitude_mode == "constant" (m)

    # ---- radar / risk model ---------------------------------------------- #
    radars: list[RadarSpec] = field(default_factory=list)
    # Near-field floor: the minimum effective range (m).  Physically this is the
    # radar's minimum-range / burn-through standoff -- inside it the hazard is
    # held at its value at R0 instead of diverging as 1/R^4.  Chosen large enough
    # that, with the calibrated C below, the floored near-field hazard stays
    # large-but-finite (order 0.1-1.5), not an absurd spike.
    R0: float = 50.0
    los_step: float = 5.0  # LOS sampling step along radar->p segment (m)
    # Reference detection point used to CALIBRATE each radar's power constant C
    # (see risk.calibrate_C).  Instead of a magic C, we pin a documented,
    # interpretable operating point: "a straight aircraft transit that crosses
    # the radar's coverage disc at closest slant range = ref_range_frac * R_max,
    # with clear line-of-sight, is detected with probability ref_detection_prob."
    # C is then solved so that reference transit yields Phi_ref = -ln(1 - p_ref).
    ref_detection_prob: float = 0.5  # detection prob of the reference transit
    ref_range_frac: float = 0.5  # closest approach as a fraction of R_max
    ref_range_m: float | None = None  # absolute closest approach (m); overrides frac

    # ---- cost weights (must sum to 1, all >= 0) -------------------------- #
    wL: float = 0.40  # excess-length weight
    wR: float = 0.45  # detection-risk weight
    wK: float = 0.15  # kinematic-feasibility weight

    # ---- kinematic limits ------------------------------------------------ #
    R_min: float = 60.0  # minimum horizontal turn radius (m)
    gamma_max_deg: float = 15.0  # max |flight-path angle| (deg)
    # Kinematic metrics (curvature, flight-path angle and their violation
    # counts) are evaluated on a copy of the path uniformly RESAMPLED to this
    # arc-length spacing, so they are spacing-independent and identical in
    # definition for all three planners.  0 -> use the grid resolution.
    kin_resample_ds: float = 0.0

    # ---- A* planner ------------------------------------------------------ #
    astar_connectivity: int = 16  # 8 or 16 connected motion
    astar_num_headings: int = 16  # discrete headings in the search state

    # ---- APF planner ----------------------------------------------------- #
    apf_xi: float | None = None  # attractive gain; None -> maps to wL
    apf_eta: float | None = None  # repulsive gain; None -> maps to wR
    apf_alpha: float = 5.0  # gradient-descent step length (m/iter)
    apf_max_iter: int = 6000
    apf_fd_step: float = 2.5  # finite-difference step for grad Lambda (m)
    apf_goal_tol: float = 12.0  # goal capture radius (m)
    apf_stuck_window: int = 60  # window length for local-minima detection
    apf_stuck_disp: float = 8.0  # min net displacement over window (m)
    apf_min_step: float = 1.0  # (deprecated) APF no longer shrinks the
    # step for climb; kept for compatibility

    # ---- Hybrid planner -------------------------------------------------- #
    hybrid_coarsen: int = 4  # coarse-grid factor for the global A* layer
    hybrid_attract: float = 5.0  # local sub-goal tracking gain (>> wL): the
    # global corridor is already risk-safe, so
    # the local layer tracks it strongly while
    # APF repulsion only fine-tunes/smooths.

    # ---- Monte-Carlo control --------------------------------------------- #
    mc_runs: int = 20  # randomized runs per scenario class

    # ---- output ---------------------------------------------------------- #
    out_dir: Path = Path("outputs")

    # ------------------------------------------------------------------ #
    # Derived geometry                                                    #
    # ------------------------------------------------------------------ #
    @property
    def n_cells(self) -> int:
        """Number of grid cells per side."""
        return int(round(self.world_size / self.resolution))

    @property
    def gamma_max(self) -> float:
        """Maximum flight-path angle in radians."""
        return np.deg2rad(self.gamma_max_deg)

    @property
    def kappa_max(self) -> float:
        """Maximum allowed curvature (1 / R_min), in 1/m."""
        return 1.0 / self.R_min

    def rng(self) -> np.random.Generator:
        """Fresh seeded generator. Use a derived seed for independent streams."""
        return np.random.default_rng(self.seed)

    # ------------------------------------------------------------------ #
    # Coordinate mapping (world metres  <->  grid indices)                #
    # ------------------------------------------------------------------ #
    def world_to_grid(self, x, y):
        """World metres -> fractional grid indices (ix, iy).

        Cell centres sit at integer indices; ``ix = x/res - 0.5``.  Accepts
        scalars or numpy arrays.
        """
        ix = np.asarray(x) / self.resolution - 0.5
        iy = np.asarray(y) / self.resolution - 0.5
        return ix, iy

    def grid_to_world(self, ix, iy):
        """Grid indices -> world metres at the corresponding cell centres."""
        x = (np.asarray(ix) + 0.5) * self.resolution
        y = (np.asarray(iy) + 0.5) * self.resolution
        return x, y

    def cell_centers(self):
        """1D arrays of cell-centre world coordinates (xs, ys), length n_cells."""
        idx = np.arange(self.n_cells)
        xs = (idx + 0.5) * self.resolution
        ys = (idx + 0.5) * self.resolution
        return xs, ys

    def validate(self) -> None:
        """Cheap sanity checks on weights and geometry."""
        w = (self.wL, self.wR, self.wK)
        if any(v < 0 for v in w):
            raise ValueError(f"cost weights must be >= 0, got {w}")
        if not np.isclose(sum(w), 1.0):
            raise ValueError(f"cost weights must sum to 1, got {sum(w)}")
        if self.astar_connectivity not in (8, 16):
            raise ValueError("astar_connectivity must be 8 or 16")
        if self.altitude_mode not in ("nap", "constant"):
            raise ValueError("altitude_mode must be 'nap' or 'constant'")
        if self.world_size <= 0 or self.resolution <= 0:
            raise ValueError("world_size and resolution must be positive")
