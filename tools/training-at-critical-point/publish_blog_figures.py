#!/usr/bin/env python3
"""Focused teacher–student figures: order parameters + internal alignment structure."""

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
SWEEPS_ROOT = RUNS_ROOT / "_sweeps"
ASSETS_DIR = REPO_ROOT / "assets/img/blog/critical-point"

_skill_root = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
sys.path.insert(0, str(_skill_root / "research-plotting" / "scripts"))
from research_plotting import (  # noqa: E402
    add_panel_label,
    clean_axis,
    plot_heatmap,
    save_figure,
    set_research_style,
    smooth_ema,
    smooth_log_ema,
)

SMOOTH_ALPHA = 0.07


def _load(run_name: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_ROOT / run_name / "metrics.csv")


def _trajectory(ax, x, y, t, *, cmap: str = "plasma") -> mcolors.Normalize:
    x = np.asarray(x, float)
    y = np.clip(np.asarray(y, float), 1e-6, None)
    t = np.asarray(t, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x, y, t = x[m], y[m], t[m]
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mcolors.Normalize(t.min(), t.max())
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), norm=norm, linewidths=2.0)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], s=20, c="#2563eb", zorder=5, edgecolors="white", linewidths=0.4)
    ax.scatter(x[-1], y[-1], s=24, c="#dc2626", zorder=5, edgecolors="white", linewidths=0.4)
    ax.set_xlim(x.min() - 0.03, x.max() + 0.03)
    ymin, ymax = y.min(), y.max()
    ax.set_ylim(ymin * 0.3, ymax * 3.0)
    return norm


def fig01_macro_trajectory(pub: Path, out: Path) -> None:
    """Training as a path in (R, ε_g) order-parameter space."""
    df = _load("ts_dynamics").dropna(subset=["R", "eps_g"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    norm = _trajectory(ax, df["R"], df["eps_g"], df["step"])
    ax.set_yscale("log")
    ax.set_xlabel(r"teacher overlap $R$")
    ax.set_ylabel(r"generalization error $\varepsilon_g$")
    clean_axis(ax)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("training step", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_figure(fig, out / "fig01_macro")
    _copy_web(out / "fig01_macro.png", pub / "fig01-phase-portrait.png")
    plt.close(fig)


def fig02_micro_alignment(pub: Path, out: Path) -> None:
    """Internal structure: student×teacher neuron overlap matrices at three times."""
    snap = pd.read_csv(RUNS_ROOT / "ts_dynamics" / "alignment_snapshots.csv")
    steps = sorted(snap["step"].unique())
    pick = [steps[0], steps[len(steps) // 2], steps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for i, (ax, st) in enumerate(zip(axes, pick)):
        sub = snap[snap["step"] == st]
        k_s = int(sub["s_neuron"].max()) + 1
        k_t = int(sub["t_neuron"].max()) + 1
        mat = np.zeros((k_s, k_t))
        for _, r in sub.iterrows():
            mat[int(r["s_neuron"]), int(r["t_neuron"])] = r["overlap"]
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1, origin="lower")
        ax.set_xlabel("teacher neuron")
        ax.set_ylabel("student neuron")
        ax.set_title(f"step {int(st)}", fontsize=7)
        add_panel_label(ax, chr(ord("a") + i))
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label=r"$|\cos(w_i, w_j^\star)|$")
    fig.tight_layout()
    save_figure(fig, out / "fig02_alignment")
    _copy_web(out / "fig02_alignment.png", pub / "fig02-alignment.png")
    plt.close(fig)


def fig03_per_neuron_order(pub: Path, out: Path) -> None:
    """Decomposed order parameter: each teacher direction has its own overlap R_j(t)."""
    df = _load("ts_dynamics")
    rcols = [c for c in df.columns if c.startswith("R_t")]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    cmap = plt.get_cmap("tab10")
    steps = df["step"].to_numpy()
    for j, col in enumerate(rcols):
        y = smooth_ema(df[col].to_numpy(), alpha=SMOOTH_ALPHA)
        ax.plot(steps, y, lw=1.5, color=cmap(j), label=rf"$R_{j}(t)$")
    r_mean = smooth_ema(df["R"].to_numpy(), alpha=SMOOTH_ALPHA)
    ax.plot(steps, r_mean, lw=2.2, color="#111827", ls="--", label=r"mean $R(t)$")
    ax.set_xlabel("step")
    ax.set_ylabel("overlap")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out / "fig03_per_neuron")
    _copy_web(out / "fig03_per_neuron.png", pub / "fig03-per-neuron-overlap.png")
    plt.close(fig)


def fig04_phase_map(pub: Path, out: Path) -> None:
    """Phase diagram in (K, η): where can the student recover the teacher?"""
    eps = pd.read_csv(SWEEPS_ROOT / "ts_phase" / "matrices" / "eps_g_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    plot_heatmap(
        ax, eps.values, eps.columns.astype(float), eps.index.astype(int),
        cmap_name="magma", log_norm=True, fig=fig, colorbar_label=r"$\varepsilon_g$",
    )
    ax.set_xlabel(r"learning rate $\eta$")
    ax.set_ylabel(r"student width $K$")
    ax.axhline(y=4, color="white", ls="--", lw=0.8, alpha=0.8)
    fig.tight_layout()
    save_figure(fig, out / "fig04_phase")
    _copy_web(out / "fig04_phase.png", pub / "fig04-phase-diagram.png")
    plt.close(fig)


def fig05_sample_complexity(pub: Path, out: Path) -> None:
    """Data-limited transition: ε_g vs α = n/d at fixed architecture."""
    df = pd.read_csv(SWEEPS_ROOT / "ts_sample" / "summary.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    eg = df["eps_g"].to_numpy()
    ax.plot(df["alpha"], eg, "o-", color="#2563eb", lw=1.6, ms=5)
    ax2 = ax.twinx()
    ax2.plot(df["alpha"], df["R"], "s--", color="#059669", lw=1.2, ms=4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"$\varepsilon_g$")
    ax2.set_ylabel("$R$", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669", labelsize=6)
    ax2.set_ylim(0, 1.05)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out / "fig05_sample")
    _copy_web(out / "fig05_sample.png", pub / "fig05-sample-complexity.png")
    plt.close(fig)


def _copy_web(src: Path, dst: Path) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    w, h = img.size
    if w > 1100:
        img = img.resize((1100, int(h * 1100 / w)), Image.Resampling.LANCZOS)
    img.save(dst, optimize=True)


def publish_all(pub: Path | None = None) -> None:
    set_research_style()
    pub = pub or ASSETS_DIR
    out = SWEEPS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)
    fig01_macro_trajectory(pub, out)
    fig02_micro_alignment(pub, out)
    fig03_per_neuron_order(pub, out)
    fig04_phase_map(pub, out)
    fig05_sample_complexity(pub, out)
    print(f"Published to {pub}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()
    publish_all(args.publish_dir if args.publish_dir.is_absolute() else REPO_ROOT / args.publish_dir)


if __name__ == "__main__":
    main()
