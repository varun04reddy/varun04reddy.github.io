#!/usr/bin/env python3
"""Rich Ganguli/Pehlevan-style teacher–student figures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "experiments/training-at-critical-point/runs"
SWEEPS = RUNS_ROOT / "_sweeps"
ASSETS = REPO_ROOT / "assets/img/blog/critical-point"

_skill = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
sys.path.insert(0, str(_skill / "research-plotting" / "scripts"))
from research_plotting import (  # noqa: E402
    add_colorbar,
    add_panel_label,
    clean_axis,
    plot_heatmap,
    plot_sweep_curves,
    save_figure,
    set_research_style,
    smooth_ema,
    smooth_log_ema,
)

ALPHA = 0.07
K_STAR = 8


def _traj_colored(ax, xs, ys, ts, *, cmap: str = "plasma", lw: float = 2.2) -> mcolors.Normalize:
    xs, ys, ts = map(lambda a: np.asarray(a, float), (xs, ys, ts))
    ys = np.clip(ys, 1e-6, None)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys, ts = xs[m], ys[m], ts[m]
    segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])], axis=1)
    norm = mcolors.Normalize(ts.min(), ts.max())
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), norm=norm, linewidths=lw)
    lc.set_array(0.5 * (ts[:-1] + ts[1:]))
    ax.add_collection(lc)
    return norm


def fig01_lr_portraits(pub: Path, out: Path) -> None:
    """Feature-drift vs generalization trajectories at multiple learning rates."""
    traj = pd.read_csv(SWEEPS / "ts_lr" / "aggregated.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    cmap, norm = plot_sweep_curves(
        ax, traj, x="d_h", y="eps_g", sweep="lr",
        cmap_name="coolwarm", log_color=False, log_smooth=True, smooth_alpha=ALPHA,
    )
    ax.set_xlabel(r"feature drift $d_h$")
    ax.set_ylabel(r"generalization error $\varepsilon_g$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    clean_axis(ax)
    add_colorbar(fig, ax, cmap, norm, r"learning rate $\eta$")
    fig.tight_layout()
    save_figure(fig, out / "fig01_lr_portraits")
    _web(out / "fig01_lr_portraits.png", pub / "fig01-phase-portrait.png")
    plt.close(fig)


def fig02_alignment_evolution(pub: Path, out: Path) -> None:
    snap = pd.read_csv(RUNS_ROOT / "ts_dynamics" / "alignment_snapshots.csv")
    steps = sorted(snap["step"].unique())
    pick = [steps[0], steps[len(steps) // 4], steps[len(steps) // 2], steps[3 * len(steps) // 4], steps[-1]]
    base = snap[snap["step"] == steps[0]]
    ks, kt = int(base["s_neuron"].max()) + 1, int(base["t_neuron"].max()) + 1
    m0 = np.zeros((ks, kt))
    for _, r in base.iterrows():
        m0[int(r["s_neuron"]), int(r["t_neuron"])] = r["overlap"]
    fig, axes = plt.subplots(1, 5, figsize=(9.8, 2.6))
    for i, (ax, st) in enumerate(zip(axes, pick)):
        sub = snap[snap["step"] == st]
        mat = np.zeros((ks, kt))
        for _, r in sub.iterrows():
            mat[int(r["s_neuron"]), int(r["t_neuron"])] = r["overlap"]
        gain = np.clip(mat - m0, 0, None)
        im = ax.imshow(gain, aspect="auto", cmap="turbo", vmin=0, vmax=gain.max() + 1e-6, origin="lower")
        ax.set_title(f"{int(st)}", fontsize=7)
        ax.set_xlabel("teacher" if i >= 2 else "")
        ax.set_ylabel("student" if i == 0 else "")
        if i > 0:
            ax.set_yticks([])
        add_panel_label(ax, chr(ord("a") + i))
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label=r"$\Delta|M_{ij}|$ vs init")
    fig.suptitle("Emergent specialization: student neurons lock onto teacher directions", fontsize=8, y=1.03)
    fig.subplots_adjust(wspace=0.08)
    save_figure(fig, out / "fig02_alignment")
    _web(out / "fig02_alignment.png", pub / "fig02-alignment.png")
    plt.close(fig)


def fig03_staggered_order(pub: Path, out: Path) -> None:
    df = pd.read_csv(RUNS_ROOT / "ts_dynamics" / "metrics.csv")
    rcols = sorted(c for c in df.columns if c.startswith("R_t"))
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True, height_ratios=[1.2, 1])
    steps = df["step"].to_numpy()
    tab = plt.get_cmap("tab10")
    for j, col in enumerate(rcols):
        axes[0].plot(steps, smooth_ema(df[col].to_numpy(), ALPHA), lw=1.5, color=tab(j), label=rf"$R_{j}$")
    axes[0].plot(steps, smooth_ema(df["R"].to_numpy(), ALPHA), "k--", lw=2, label=r"$\bar R$")
    axes[0].set_ylabel("overlap")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=5.5, ncol=3, loc="lower right")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    eg = smooth_log_ema(np.clip(df["eps_g"].to_numpy(), 1e-6, None), ALPHA)
    dh = smooth_log_ema(np.clip(df["d_h"].to_numpy(), 1e-8, None), ALPHA)
    axes[1].plot(steps, eg, color="#7c3aed", lw=1.8, label=r"$\varepsilon_g$")
    ax2 = axes[1].twinx()
    ax2.plot(steps, dh, color="#0891b2", lw=1.4, alpha=0.85, label=r"$d_h$")
    axes[1].set_yscale("log")
    ax2.set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$\varepsilon_g$")
    ax2.set_ylabel(r"$d_h$", color="#0891b2")
    ax2.tick_params(axis="y", labelcolor="#0891b2", labelsize=6)
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])
    fig.tight_layout()
    save_figure(fig, out / "fig03_staggered")
    _web(out / "fig03_staggered.png", pub / "fig03-per-neuron-overlap.png")
    plt.close(fig)


def fig04_snr_phase(pub: Path, out: Path) -> None:
    mat = SWEEPS / "ts_snr" / "matrices"
    eps = pd.read_csv(mat / "eps_g_matrix.csv", index_col=0)
    rr = pd.read_csv(mat / "R_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    plot_heatmap(axes[0], eps.values, eps.columns.astype(float), eps.index.astype(float), cmap_name="magma", log_norm=True, fig=fig, colorbar_label=r"$\varepsilon_g$")
    axes[0].set_xlabel(r"$\alpha = n/d$")
    axes[0].set_ylabel(r"label noise $\sigma$")
    add_panel_label(axes[0], "a")
    plot_heatmap(axes[1], rr.values, rr.columns.astype(float), rr.index.astype(float), cmap_name="viridis", fig=fig, colorbar_label="$R$")
    axes[1].set_xlabel(r"$\alpha = n/d$")
    axes[1].set_ylabel(r"label noise $\sigma$")
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, out / "fig04_snr")
    _web(out / "fig04_snr.png", pub / "fig04-phase-diagram.png")
    plt.close(fig)


def fig05_sample_scaling(pub: Path, out: Path) -> None:
    df = pd.read_csv(SWEEPS / "ts_sample_k" / "summary.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    cmap, norm = plot_sweep_curves(
        ax, df, x="alpha", y="eps_g", sweep="student_width",
        cmap_name="viridis", log_color=True, log_smooth=False, smooth_alpha=ALPHA,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"$\varepsilon_g$")
    add_colorbar(fig, ax, cmap, norm, "$K$")
    ax.axvline(40, color="#78716c", ls=":", lw=0.9, alpha=0.7)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out / "fig05_sample")
    _web(out / "fig05_sample.png", pub / "fig05-sample-complexity.png")
    plt.close(fig)


def fig06_lazy_rich(pub: Path, out: Path) -> None:
    dh = pd.read_csv(SWEEPS / "ts_lazy_rich" / "matrices" / "d_h_matrix.csv", index_col=0)
    rr = pd.read_csv(SWEEPS / "ts_lazy_rich" / "matrices" / "R_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    plot_heatmap(axes[0], dh.values, dh.columns.astype(float), dh.index.astype(float), cmap_name="cividis", log_norm=True, fig=fig, colorbar_label=r"$d_h$")
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"init scale $\gamma$")
    add_panel_label(axes[0], "a")
    plot_heatmap(axes[1], rr.values, rr.columns.astype(float), rr.index.astype(float), cmap_name="viridis", fig=fig, colorbar_label="$R$")
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(r"init scale $\gamma$")
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, out / "fig06_lazy")
    _web(out / "fig06_lazy.png", pub / "fig06-lazy-rich.png")
    plt.close(fig)


def fig07_capacity_phase(pub: Path, out: Path) -> None:
    mat = SWEEPS / "ts_phase" / "matrices"
    eps = pd.read_csv(mat / "eps_g_matrix.csv", index_col=0)
    rr = pd.read_csv(mat / "overlap_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    plot_heatmap(axes[0], eps.values, eps.columns.astype(float), eps.index.astype(float), cmap_name="magma", log_norm=True, fig=fig, colorbar_label=r"$\varepsilon_g$")
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"student width $K$")
    axes[0].axhline(K_STAR - 0.5, color="white", ls="--", lw=0.8, alpha=0.7)
    add_panel_label(axes[0], "a")
    plot_heatmap(axes[1], rr.values, rr.columns.astype(float), rr.index.astype(float), cmap_name="viridis", fig=fig, colorbar_label="$R$")
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(r"student width $K$")
    axes[1].axhline(K_STAR - 0.5, color="white", ls="--", lw=0.8, alpha=0.7)
    add_panel_label(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, out / "fig07_capacity")
    _web(out / "fig07_capacity.png", pub / "fig07-capacity-phase.png")
    plt.close(fig)


def _web(src: Path, dst: Path) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    w, h = img.size
    if w > 1100:
        img = img.resize((1100, int(h * 1100 / w)), Image.Resampling.LANCZOS)
    img.save(dst, optimize=True)


def publish_all(pub: Path | None = None) -> None:
    set_research_style()
    pub = pub or ASSETS
    out = SWEEPS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)
    fig01_lr_portraits(pub, out)
    fig02_alignment_evolution(pub, out)
    fig03_staggered_order(pub, out)
    fig04_snr_phase(pub, out)
    fig05_sample_scaling(pub, out)
    fig06_lazy_rich(pub, out)
    fig07_capacity_phase(pub, out)
    print(f"Published to {pub}")


if __name__ == "__main__":
    publish_all()
