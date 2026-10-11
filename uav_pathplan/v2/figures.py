"""Protocol-v2 figures (no in-figure titles; planner labels from plotting.ALGO_STYLE)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import plotting

SCENARIO_LABEL = {
    "open": "open",
    "concave_trap": "concave trap",
    "ridge_corridor": "ridge corridor",
    "high_density": "high density",
}


# distinct markers / line styles so coinciding curves (APF-E == APF when it
# never escapes) remain visible
LINE_STYLE = {
    "astar": dict(marker="o", ls="-"),
    "apf": dict(marker="s", ls="--"),
    "apfe": dict(marker="^", ls=":"),
    "hybrid": dict(marker="D", ls="-."),
}


def _label(p):
    return plotting.ALGO_STYLE[p]["label"]


def _color(p):
    return plotting.ALGO_STYLE[p]["color"]


def save(fig, out: Path, stem: str, formats) -> list[Path]:
    paths = []
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        plotting._savefig(fig, p)
        paths.append(p)
    plt.close(fig)
    return paths


def _panel_label(ax, scenario, risk_note=False):
    txt = SCENARIO_LABEL[scenario]
    if risk_note and scenario == "concave_trap":
        txt += " (relative exposure)"
    ax.text(0.0, 1.02, txt, transform=ax.transAxes, va="bottom", fontsize=9, fontweight="bold")


def box_metric(df, planners, scenarios, metric, ylabel, out, stem, formats, risk_note=False):
    """Box plots of ``metric`` over successful runs, one panel per scenario."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.6))
    for ax, scn in zip(np.atleast_1d(axes), scenarios):
        g = df[(df["scenario"] == scn) & df["success"]]
        data = [g[g["planner"] == p][metric].dropna().to_numpy() for p in planners]
        pos = np.arange(len(planners))
        for i, (p, d) in enumerate(zip(planners, data)):
            if d.size:
                b = ax.boxplot(d, positions=[i], widths=0.6, patch_artist=True, showfliers=True)
                b["boxes"][0].set_facecolor(_color(p))
                b["boxes"][0].set_alpha(0.55)
                b["medians"][0].set_color("k")
            else:
                ax.text(
                    i,
                    0.5,
                    "no\nsuccess",
                    ha="center",
                    va="center",
                    fontsize=7,
                    transform=ax.get_xaxis_transform(),
                )
        ax.set_xticks(pos)
        ax.set_xticklabels([_label(p) for p in planners], fontsize=8, rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        _panel_label(ax, scn, risk_note)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.tight_layout()
    return save(fig, out, stem, formats)


def success_bars(agg, planners, scenarios, out, stem, formats):
    """Success rate with Wilson 95 % CI per scenario and planner."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(planners)
    x = np.arange(len(scenarios))
    for i, p in enumerate(planners):
        rows = [agg[(agg["scenario"] == s) & (agg["planner"] == p)].iloc[0] for s in scenarios]
        rate = np.array([r["success_rate"] for r in rows])
        lo = np.maximum(rate - np.array([r["success_ci_lo"] for r in rows]), 0.0)
        hi = np.maximum(np.array([r["success_ci_hi"] for r in rows]) - rate, 0.0)
        ax.bar(
            x + (i - (len(planners) - 1) / 2) * width,
            rate,
            width,
            color=_color(p),
            alpha=0.8,
            label=_label(p),
            yerr=[lo, hi],
            capsize=2,
            ecolor="k",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in scenarios])
    ax.set_ylabel("success rate (Wilson 95 % CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(
        fontsize=8,
        ncol=len(planners),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save(fig, out, stem, formats)


def grid_sensitivity(
    agg, planners, scenarios, metric, ylabel, out, stem, formats, risk_note=False, logy=False
):
    """Median (IQR bars) of ``metric`` vs grid resolution, one panel per scenario."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.4), sharey=logy)
    for ax, scn in zip(np.atleast_1d(axes), scenarios):
        for p in planners:
            g = agg[(agg["scenario"] == scn) & (agg["planner"] == p)].sort_values("resolution")
            med = g[f"{metric}_median"].to_numpy()
            if np.all(np.isnan(med)):
                continue
            lo = med - g[f"{metric}_q1"].to_numpy()
            hi = g[f"{metric}_q3"].to_numpy() - med
            ax.errorbar(
                g["resolution"],
                med,
                yerr=[lo, hi],
                ms=5,
                mfc="none",
                capsize=2,
                color=_color(p),
                label=_label(p),
                **LINE_STYLE[p],
            )
        ax.set_xlabel("grid resolution (m)")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        _panel_label(ax, scn, risk_note)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    return save(fig, out, stem, formats)


def altitude_modes(agg, planners, scenarios, metric, ylabel, out, stem, formats):
    """Median ``metric`` under nap vs rate_limited (paired bars), one panel per scenario."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.4))
    for ax, scn in zip(np.atleast_1d(axes), scenarios):
        x = np.arange(len(planners))
        for j, (mode, hatch) in enumerate([("nap", ""), ("rate_limited", "//")]):
            vals = []
            for p in planners:
                r = agg[
                    (agg["scenario"] == scn)
                    & (agg["planner"] == p)
                    & (agg["altitude_mode"] == mode)
                ]
                vals.append(r[f"{metric}_median"].iloc[0] if len(r) else np.nan)
            ax.bar(
                x + (j - 0.5) * 0.38,
                vals,
                0.38,
                color=[_color(p) for p in planners],
                alpha=0.75,
                hatch=hatch,
                edgecolor="k",
                linewidth=0.5,
                label="nap" if j == 0 else "rate-limited",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(p) for p in planners], fontsize=8, rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        _panel_label(ax, scn, risk_note=True)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.tight_layout()
    return save(fig, out, stem, formats)


def overlay(cfg, terrain, risk, start, goal, results, metrics, out, stem, formats):
    fig = plotting.plot_overlay(
        cfg,
        terrain,
        risk,
        start,
        goal,
        results,
        out_path=None,
        title="",
        close=False,
        metrics=metrics,
    )
    return save(fig, out, stem, formats)
