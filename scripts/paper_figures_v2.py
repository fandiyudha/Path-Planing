"""Manuscript figures for the DSJ paper, generated from the protocol-v2 outputs.

Usage (from the repository root):
    python scripts/paper_figures_v2.py . outputs/paper_figures_v2
Reads paper_assets_v2/results_per_run.csv (read-only); Fig. 1 re-plans run 0 of
the held-out seed at 8 m.  Writes only to <outdir>.
Fig. 1  representative paths (run 0, 8 m, nap) for concave_trap, ridge_corridor, high_density
Fig. 2  J distributions at the reference setting (8 m, nap)
Fig. 3  grid sensitivity of median J (IQR bars), nap
Fig. 4  effect of the altitude mode on median J (8 m)
All figures: 300 dpi TIFF + PNG, no in-figure titles, planners distinguished by
colour AND marker/line style (legible in greyscale)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

PL = ["astar", "apf", "apfe", "hybrid"]
LAB = {"astar": "A-Star", "apf": "APF", "apfe": "APF-E", "hybrid": "Hybrid"}
COL = {"astar": "#2a78d6", "apf": "#eb6834", "apfe": "#1baf7a", "hybrid": "#eda100"}
MRK = {"astar": "o", "apf": "s", "apfe": "^", "hybrid": "D"}
LS = {"astar": "-", "apf": "--", "apfe": ":", "hybrid": "-."}
SCN = ["open", "concave_trap", "ridge_corridor", "high_density"]
SLAB = {
    "open": "Open baseline",
    "concave_trap": "Concave trap†",
    "ridge_corridor": "Ridge corridor",
    "high_density": "High threat density",
}
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)

d = pd.read_csv(REPO / "paper_assets_v2" / "results_per_run.csv")
ok = d[d.success]


def save(fig, stem):
    for ext in ("png", "tiff"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight", **kw)
    plt.close(fig)


def panel_tag(ax, k, label):
    ax.text(
        0.0,
        1.02,
        f"({'abcd'[k]}) {label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
    )


# ---------------- Fig. 2: J distributions, 8 m, nap ---------------------- #
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
ref = ok[(ok.resolution == 8.0) & (ok.altitude_mode == "nap")]
for k, s in enumerate(SCN):
    ax = axs[k]
    data, pos, cols = [], [], []
    for i, p in enumerate(PL):
        v = ref[(ref.scenario == s) & (ref.planner == p)].J.values
        if len(v):
            data.append(v)
            pos.append(i)
            cols.append(COL[p])
        else:
            ax.text(
                i,
                0.5,
                "no\nsuccess",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=9,
                color="#52514e",
            )
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="#0b0b0b", lw=1.6),
        flierprops=dict(marker=".", ms=4, mec="#52514e"),
    )
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c)
        b.set_alpha(0.85)
        b.set_edgecolor("#0b0b0b")
        b.set_linewidth(0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LAB[p] for p in PL], rotation=30)
    ax.set_xlim(-0.6, 3.6)
    if s == "concave_trap":
        ax.set_yscale("log")
    panel_tag(ax, k, SLAB[s])
axs[0].set_ylabel("Risk-adjusted cost J")
fig.tight_layout()
save(fig, "fig2_J_distributions")

# ---------------- Fig. 3: grid sensitivity, nap -------------------------- #
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
nap = ok[ok.altitude_mode == "nap"]
for k, s in enumerate(["concave_trap", "ridge_corridor", "high_density"]):
    ax = axs[k]
    for j, p in enumerate(PL):
        g = nap[(nap.scenario == s) & (nap.planner == p)].groupby("resolution").J
        if g.ngroups == 0:
            continue
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        x = med.index.values + (j - 1.5) * 0.12
        ax.errorbar(
            x,
            med.values,
            yerr=[med.values - q1.values, q3.values - med.values],
            color=COL[p],
            marker=MRK[p],
            ls=LS[p],
            lw=1.6,
            ms=6,
            capsize=3,
            label=LAB[p],
            mec="#0b0b0b",
            mew=0.5,
        )
    ax.set_xticks([5, 8, 10])
    ax.set_xlabel("Grid resolution (m)")
    if s == "concave_trap":
        ax.set_yscale("log")
    panel_tag(ax, k, SLAB[s])
axs[0].set_ylabel("Median J (IQR)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout()
save(fig, "fig3_grid_sensitivity")

# ---------------- Fig. 4: altitude modes, 8 m ---------------------------- #
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
r8 = ok[ok.resolution == 8.0]
for k, s in enumerate(["concave_trap", "ridge_corridor", "high_density"]):
    ax = axs[k]
    for i, p in enumerate(PL):
        g = r8[(r8.scenario == s) & (r8.planner == p)]
        if g.empty:
            ax.text(
                i,
                0.5,
                "no\nsuccess",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=9,
                color="#52514e",
            )
            continue
        a = g[g.altitude_mode == "nap"].J.median()
        b = g[g.altitude_mode == "rate_limited"].J.median()
        ax.plot([i - 0.15, i + 0.15], [a, b], color="#9a9996", lw=1.2, zorder=1)
        ax.scatter(
            [i - 0.15],
            [a],
            s=55,
            marker=MRK[p],
            facecolor=COL[p],
            edgecolor="#0b0b0b",
            lw=0.6,
            zorder=3,
            label="Nap-of-the-earth" if i == 0 else None,
        )
        ax.scatter(
            [i + 0.15],
            [b],
            s=55,
            marker=MRK[p],
            facecolor="white",
            edgecolor=COL[p],
            lw=1.8,
            zorder=3,
            label="Rate-limited" if i == 0 else None,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels([LAB[p] for p in PL], rotation=30)
    ax.set_xlim(-0.6, 3.6)
    if s == "concave_trap":
        ax.set_yscale("log")
    panel_tag(ax, k, SLAB[s])
axs[0].set_ylabel("Median J")
from matplotlib.lines import Line2D

leg = [
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="#52514e",
        mec="#0b0b0b",
        ms=7,
        label="Nap-of-the-earth (filled)",
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="white",
        mec="#52514e",
        mew=1.8,
        ms=7,
        label="Rate-limited (open)",
    ),
]
fig.legend(handles=leg, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.09))
fig.tight_layout()
save(fig, "fig4_altitude_modes")

# ---------------- Fig. 1: representative paths -------------------------- #
from uav_pathplan.v2 import protocol, runner  # noqa: E402
from uav_pathplan.v2 import scenarios as v2scn

proto = protocol.load(REPO / "configs" / "protocol_v2.yaml")
seed = proto["seeds"]["evaluation"]
apfe_kw = protocol.apfe_params(proto)
fig, axs = plt.subplots(1, 3, figsize=(14, 4.9))
for k, s in enumerate(["concave_trap", "ridge_corridor", "high_density"]):
    ax = axs[k]
    spec, cfg, terrain, risk, H, _ = runner.build_instance(proto, s, seed, 0, 8.0)
    xs = (np.arange(H.shape[1]) + 0.5) * cfg.resolution
    Hm = np.where(H > 0, H, np.nan)
    pos = Hm[np.isfinite(Hm)]
    im = ax.pcolormesh(
        xs,
        xs,
        Hm,
        cmap="Greys",
        norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-6), vmax=pos.max()),
        shading="auto",
        rasterized=True,
    )
    hx = np.linspace(0, cfg.world_size, 201)
    HX, HY = np.meshgrid(hx, hx)
    ax.contour(
        hx,
        hx,
        np.asarray(terrain.h_at(HX, HY)),
        levels=8,
        colors="#8c6d46",
        linewidths=0.6,
        alpha=0.8,
        zorder=2,
    )
    for p in PL:
        r = runner.run_planner(p, cfg, terrain, risk, spec, H, apfe_kw, v2scn.apfe_rng(seed, s, 0))
        if r.path is None or len(r.path) < 2:
            continue
        P = np.asarray(r.path)
        z = 4 if p == "hybrid" else 3
        ax.plot(
            P[:, 0],
            P[:, 1],
            color=COL[p],
            ls=LS[p],
            lw=2.2 if p != "apf" else 2.6,
            zorder=z,
            label=LAB[p] + ("" if r.success else " (failed)"),
        )
        if not r.success:
            ax.plot(P[-1, 0], P[-1, 1], marker="X", ms=11, color=COL[p], mec="#0b0b0b", zorder=6)
    ax.plot(*spec.start, marker="o", ms=10, mfc="white", mec="#0b0b0b", mew=1.5, zorder=7, ls="")
    ax.plot(*spec.goal, marker="*", ms=15, mfc="#0b0b0b", mec="#0b0b0b", zorder=7, ls="")
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x (m)")
    if k == 0:
        ax.set_ylabel("y (m)")
    panel_tag(ax, k, SLAB[s])
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
save(fig, "fig1_representative_paths")
print("done")
