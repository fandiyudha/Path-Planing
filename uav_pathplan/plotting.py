"""Plotting and console-table helpers (matplotlib).

Figures:
  * 2D path overlays on a hazard heatmap with threats + terrain contours
  * 3D elevation surface with a path drawn over it
  * compute-time vs threat-count curve (A* scaling vs Hybrid)
  * Pareto front of L vs Rtil from sweeping the risk/length weight ratio

All functions take already-computed objects and write the figure to disk.  The
file format follows the suffix of ``out_path`` (PNG, TIFF or EPS from the
driver's ``--format``); the resolution is :data:`SAVE_DPI`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .config import Config
from .risk import RiskField
from .terrain import TerrainField

# Resolution (dots per inch) for every saved figure (raster formats).  Set via
# the driver's --dpi, or ``uav_pathplan.plotting.SAVE_DPI = ...``.
SAVE_DPI = 300


def _savefig(fig, out_path) -> None:
    """Save ``fig`` at :data:`SAVE_DPI`; the format follows the file suffix.

    TIFF files are LZW-compressed (lossless) to keep 300-dpi figures small.
    """
    out_path = Path(out_path)
    kwargs = {}
    if out_path.suffix.lower() in (".tif", ".tiff"):
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(out_path, dpi=SAVE_DPI, **kwargs)


# --------------------------------------------------------------------------- #
# Single source of truth for per-algorithm styling.                           #
#                                                                              #
# High-contrast, reasonably colour-blind-aware, and chosen to stand out over  #
# the dark->yellow ``inferno`` hazard colormap AND the white background:       #
#   * A-Star -> strong blue                                                    #
#   * APF    -> magenta (never collides with the red/orange/yellow hazard)    #
#   * Hybrid -> green                                                          #
# NEVER use white / near-white / pale yellow for a path.  Every figure pulls   #
# colours from here so they never drift between plots.                         #
# --------------------------------------------------------------------------- #
ALGO_STYLE = {
    "astar": {"color": "#1f77b4", "label": "A-Star"},
    "apf": {"color": "#ff00ff", "label": "APF"},
    "apfe": {"color": "#17becf", "label": "APF-E"},  # protocol v2 only
    "hybrid": {"color": "#2ca02c", "label": "Hybrid"},
}
# Backwards-compatible (color, label) view used by older callers.
_PLANNER_STYLE = {k: (v["color"], v["label"]) for k, v in ALGO_STYLE.items()}

# Line styling for success vs failure.
_OK_LW, _FAIL_LW = 2.8, 2.2
_STUCK_MARKER = dict(
    marker="X", color="red", markersize=13, markeredgecolor="k", markeredgewidth=1.3
)


def _algo_color(name: str) -> str:
    return ALGO_STYLE.get(name, {"color": "#7f7f7f"})["color"]


def _algo_label(name: str) -> str:
    return ALGO_STYLE.get(name, {"label": name})["label"]


def _path_legend(name: str, res, m: dict | None) -> str:
    """Legend text incl. status and a key metric, pulled from the run record."""
    lbl = _algo_label(name)
    reason = (res.info or {}).get("reason", "failed").replace("_", " ")
    if res.success:
        if m is not None:
            L = m.get("L")
            sm = m.get("smoothness")
            if L is not None and L == L and sm is not None and sm == sm:
                return f"{lbl} (ok, L={L:.0f}, smooth={sm:.2f})"
        return f"{lbl} (ok)"
    return f"{lbl} (FAILED: {reason})"


def _draw_path_2d(ax, name: str, res, m: dict | None, show_waypoints: bool):
    """Draw one attempted path (success solid, failure dashed + stuck marker)."""
    if res is None or res.n_waypoints < 1:
        return
    color = _algo_color(name)
    P = np.asarray(res.path)
    label = _path_legend(name, res, m)
    if res.success:
        ax.plot(
            P[:, 0],
            P[:, 1],
            "-",
            color=color,
            lw=_OK_LW,
            label=label,
            zorder=6,
            solid_capstyle="round",
        )
    else:
        ax.plot(P[:, 0], P[:, 1], "--", color=color, lw=_FAIL_LW, label=label, zorder=6, alpha=0.95)
        reason = (res.info or {}).get("reason", "failed").replace("_", " ")
        ax.plot(P[-1, 0], P[-1, 1], zorder=8, linestyle="none", **_STUCK_MARKER)
        ax.annotate(
            f"{_algo_label(name)}: {reason}",
            (P[-1, 0], P[-1, 1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="k",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
        )
    if show_waypoints:  # faint dots expose jagged vs smooth
        ax.plot(P[:, 0], P[:, 1], ".", color=color, ms=2.5, alpha=0.30, zorder=5)


def _hazard_background(ax, cfg: Config, terrain: TerrainField, risk: RiskField, rings: bool = True):
    """Draw the hazard heatmap (log scale) + terrain contours + radars (+rings)."""
    H = risk.grid()
    extent = [0, cfg.world_size, 0, cfg.world_size]
    Hpos = np.where(H > 0, H, np.nan)
    vmax = np.nanmax(Hpos) if np.isfinite(np.nanmax(Hpos)) else 1.0
    im = ax.imshow(
        Hpos,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
        aspect="equal",
    )
    ax.contour(
        terrain.xs, terrain.ys, terrain.h, levels=8, colors="deepskyblue", linewidths=0.3, alpha=0.4
    )
    for r in risk.radars:
        if rings:
            ax.add_patch(
                plt.Circle(
                    (r.x, r.y), r.R_max, fill=False, ec="yellow", ls="--", lw=0.5, alpha=0.45
                )
            )
        ax.plot(r.x, r.y, "X", color="yellow", ms=6, mec="k", mew=0.5)
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return im


def plot_overlay(
    cfg: Config,
    terrain: TerrainField,
    risk: RiskField,
    start,
    goal,
    results: dict,
    out_path: Path | None = None,
    title: str = "",
    close: bool = True,
    metrics: dict | None = None,
    show_waypoints: bool = True,
):
    """2D overlay of planner paths on the hazard map.

    EVERY attempted path is drawn, success or failure: successful paths as a
    solid line in the algorithm colour, failed paths as a dashed line of the
    same colour ending in a red "✕" stuck marker with a reason label, so a
    local-minimum failure is shown rather than hidden.  ``metrics`` is an
    optional ``{algo: row-dict}`` mapping (from the run record) used to enrich
    the legend with ``L`` and ``smooth`` — values are read, never hard-coded.

    Saves to ``out_path`` when given and always returns the figure; pass
    ``close=False`` to keep it alive for an interactive caller.
    """
    metrics = metrics or {}
    fig, ax = plt.subplots(figsize=(8, 7))
    im = _hazard_background(ax, cfg, terrain, risk)
    for name, res in (results or {}).items():
        _draw_path_2d(ax, name, res, metrics.get(name), show_waypoints)
    ax.plot(
        start[0], start[1], "o", color="white", ms=12, mec="k", mew=1.6, label="start", zorder=10
    )
    ax.plot(goal[0], goal[1], "*", color="red", ms=19, mec="k", mew=1.2, label="goal", zorder=10)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax, label="hazard (a.u.)", shrink=0.82)
    fig.tight_layout()
    if out_path is not None:
        _savefig(fig, out_path)
    if close:
        plt.close(fig)
    return fig


_SURFACE_CAPTION = (
    "Surface for visualisation only — planning is performed in "
    "the horizontal (x, y) plane (2.5D); z = h(x,y) + clearance."
)


def plot_surface_with_paths(
    cfg: Config,
    terrain: TerrainField,
    results: dict,
    out_path: Path | None = None,
    title: str = "",
    close: bool = True,
    metrics: dict | None = None,
):
    """3D elevation surface with EVERY attempted path projected onto it.

    All planner paths are drawn at flight altitude ``z = h(x,y)+clearance`` in
    the shared algorithm colours (successful solid, failed dashed + red "✕"
    stuck marker); paths are never filtered by ``success``.  ``metrics`` enriches
    the legend as in :func:`plot_overlay`.
    """
    metrics = metrics or {}
    X, Y = np.meshgrid(terrain.xs, terrain.ys)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        terrain.h,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        rcount=100,
        ccount=100,
        alpha=0.80,
    )
    for name, res in (results or {}).items():
        if res is None or res.n_waypoints < 1:
            continue
        color = _algo_color(name)
        P = np.asarray(res.path)
        z = np.asarray(terrain.altitude_at(P[:, 0], P[:, 1]))
        label = _path_legend(name, res, metrics.get(name))
        ls = "-" if res.success else "--"
        ax.plot(
            P[:, 0],
            P[:, 1],
            z,
            ls,
            color=color,
            lw=3.0 if res.success else 2.3,
            label=label,
            zorder=6,
        )
        if not res.success:
            ax.scatter(
                [P[-1, 0]],
                [P[-1, 1]],
                [z[-1]],
                marker="X",
                color="red",
                s=80,
                edgecolor="k",
                depthshade=False,
                zorder=10,
            )
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=42, azim=-60)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.02, _SURFACE_CAPTION, ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if out_path is not None:
        _savefig(fig, out_path)
    if close:
        plt.close(fig)
    return fig


def plot_surface_with_path(
    cfg: Config,
    terrain: TerrainField,
    path: np.ndarray,
    out_path: Path | None = None,
    title: str = "",
    close: bool = True,
):
    """Backwards-compatible single-path 3D surface (wraps the multi-path version).

    Kept so existing callers passing one ``path`` keep working; new code should
    prefer :func:`plot_surface_with_paths` to show every attempted path.
    """
    from .planners import PlanResult

    results = {}
    if path is not None and len(path) > 1:
        results["hybrid"] = PlanResult(
            name="hybrid", path=np.asarray(path), success=True, time_s=0.0
        )
    return plot_surface_with_paths(
        cfg, terrain, results, out_path=out_path, title=title, close=close
    )


def plot_time_vs_threats(
    counts, times_by_alg: dict, out_path: Path, nexp_by_alg: dict | None = None
):
    """Compute-time (and optionally expanded nodes) vs number of radars."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"astar": "o-", "apf": "s--", "hybrid": "^-"}
    for alg, ts in times_by_alg.items():
        ax.plot(counts, ts, markers.get(alg, "o-"), label=_PLANNER_STYLE.get(alg, (None, alg))[1])
    ax.set_xlabel("number of radars")
    ax.set_ylabel("compute time (s)")
    ax.set_title("Planner compute time vs threat count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if nexp_by_alg:
        ax2 = ax.twinx()
        for alg, ns in nexp_by_alg.items():
            ax2.plot(counts, ns, ":", color="gray", alpha=0.6)
        ax2.set_ylabel("A-Star/Hybrid expanded nodes (dotted)", color="gray")
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def plot_pareto(
    L_vals,
    Rtil_vals,
    wR_vals,
    out_path: Path | None = None,
    title: str = "Pareto front: length vs detection risk",
    close: bool = True,
):
    """L vs Rtil scatter/line from sweeping the risk weight wR.

    Saves to ``out_path`` when given and always returns the figure; pass
    ``close=False`` to keep it alive for an interactive caller.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(L_vals)
    L = np.asarray(L_vals)[order]
    R = np.asarray(Rtil_vals)[order]
    w = np.asarray(wR_vals)[order]
    sc = ax.scatter(L, R, c=w, cmap="viridis", s=60, zorder=3, edgecolor="k")
    ax.plot(L, R, "-", color="gray", alpha=0.5, zorder=2)
    for li, ri, wi in zip(L, R, w):
        ax.annotate(f"{wi:.2f}", (li, ri), fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("path length L (m)")
    ax.set_ylabel(r"detection risk  $\tilde R = 1-e^{-\Phi}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label=r"risk weight $w_R$")
    fig.tight_layout()
    if out_path is not None:
        _savefig(fig, out_path)
    if close:
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
def console_summary(agg_df) -> str:
    """Render the aggregated results as a console table (mean +/- std)."""
    cols = [
        ("success_rate", "succ", "{:.2f}"),
        ("J_mean", "J", "{:.3f}"),
        ("Rtil_mean", "Rtil", "{:.3f}"),
        ("L_mean", "L(m)", "{:.0f}"),
        ("Phi_mean", "Phi", "{:.2f}"),
        ("d_min_mean", "dmin", "{:.0f}"),
        ("kappa_peak_mean", "kpk", "{:.4f}"),
        ("time_s_mean", "T(s)", "{:.2f}"),
        ("n_exp_mean", "nexp", "{:.0f}"),
    ]
    lines = []
    header = f"{'scenario':>14} {'alg':>7} " + " ".join(f"{h:>7}" for _, h, _ in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in agg_df.iterrows():
        cells = []
        for key, _, fmt in cols:
            v = row.get(key, np.nan)
            cells.append(fmt.format(v) if v == v else "   -  ")  # nan check
        lines.append(
            f"{row['scenario']:>14} {row['algorithm']:>7} " + " ".join(f"{c:>7}" for c in cells)
        )
    return "\n".join(lines)
