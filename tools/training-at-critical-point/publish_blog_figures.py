#!/usr/bin/env python3
"""Paper-quality blog figures via research-plotting skill."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "experiments/training-at-critical-point/runs"
SWEEPS_ROOT = RUNS_ROOT / "_sweeps"
ASSETS_DIR = REPO_ROOT / "assets/img/blog/critical-point"

_skill_root = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
sys.path.insert(0, str(_skill_root / "research-plotting" / "scripts"))
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

SMOOTH_ALPHA = 0.07
BLOG_DPI = 200  # downsample from 600 for web; SVG kept for archive


def _load_metrics(run_name: str) -> pd.DataFrame:
    path = RUNS_ROOT / run_name / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _plot_raw_smooth(ax, x, y, *, logy: bool = False, color: str = "#2563eb") -> None:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        return
    if logy and np.all(ys > 0):
        ys_s = smooth_log_ema(ys, alpha=SMOOTH_ALPHA)
        ax.set_yscale("log")
    else:
        ys_s = smooth_ema(ys, alpha=SMOOTH_ALPHA)
    ax.plot(xs, ys, color=color, lw=0.5, alpha=0.2)
    ax.plot(xs, ys_s, color=color, lw=1.6)


def fig01_dashboard(pub: Path, sweep_figures: Path) -> None:
    """1×3 order-parameter dashboard from canonical MNIST run."""
    df = _load_metrics("dashboard_mnist")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    _plot_raw_smooth(axes[0], df["step"], df["val_loss"], logy=True, color="#7c3aed")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("val loss")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    chi = df["chi"].to_numpy()
    steps = df["step"].to_numpy()
    axes[1].fill_between(steps, 0.9, 1.05, color="#fbbf24", alpha=0.25, linewidth=0)
    _plot_raw_smooth(axes[1], steps, chi, logy=False, color="#d97706")
    axes[1].axhline(1.0, color="#78716c", ls="--", lw=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$\chi = \eta\lambda_{\max}/2$")
    axes[1].set_ylim(0, max(1.35, float(np.nanmax(chi)) * 1.05))
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    _plot_raw_smooth(axes[2], df["step"], df["m_nc"], logy=False, color="#059669")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel(r"$m_{\mathrm{NC}}$")
    add_panel_label(axes[2], "c")
    clean_axis(axes[2])

    fig.tight_layout()
    out = sweep_figures / "fig01_dashboard"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig01-micro-macro.png")
    plt.close(fig)


def fig02_phase_maps(pub: Path, sweep_figures: Path) -> None:
    """1×2 heatmaps: test accuracy + test loss (width × lr)."""
    mat_dir = SWEEPS_ROOT / "phase_w_lr" / "matrices"
    acc = pd.read_csv(mat_dir / "test_acc_matrix.csv", index_col=0)
    loss = pd.read_csv(mat_dir / "test_loss_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    plot_heatmap(
        axes[0], acc.values, acc.columns.astype(float), acc.index.astype(float),
        cmap_name="viridis", fig=fig, colorbar_label="test accuracy",
    )
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("width")
    add_panel_label(axes[0], "a")

    plot_heatmap(
        axes[1], loss.values, loss.columns.astype(float), loss.index.astype(float),
        cmap_name="magma", log_norm=True, fig=fig, colorbar_label="test loss",
    )
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("width")
    add_panel_label(axes[1], "b")

    fig.tight_layout()
    out = sweep_figures / "fig02_phase_maps"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig02-phase-diagram.png")
    plt.close(fig)


def fig03_width_sweep(pub: Path, sweep_figures: Path) -> None:
    """Width sweep: colored trajectories + endpoint double-descent curve."""
    traj = pd.read_csv(SWEEPS_ROOT / "width_sweep" / "aggregated.csv")
    end = pd.read_csv(SWEEPS_ROOT / "width_sweep" / "summary.csv").sort_values("n_params")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    cmap, norm = plot_sweep_curves(
        axes[0], traj, x="step", y="val_loss", sweep="width",
        cmap_name="viridis", log_color=True, log_smooth=True, smooth_alpha=SMOOTH_ALPHA,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("val loss")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])
    add_colorbar(fig, axes[0], cmap, norm, "width")

    x = end["n_params"].to_numpy()
    yt = end["test_loss"].to_numpy()
    ytr = end["train_loss"].to_numpy()
    axes[1].plot(x, ytr, "o-", color="#2563eb", lw=1.4, ms=4, label="train")
    axes[1].plot(x, yt, "o-", color="#dc2626", lw=1.4, ms=4, label="test")
    interp = end.loc[end["train_acc"] > 0.99, "n_params"]
    if len(interp):
        axes[1].axvline(float(interp.min()), color="#78716c", ls=":", lw=0.9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("parameters")
    axes[1].set_ylabel("loss")
    axes[1].legend(frameon=False, fontsize=6)
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    out = sweep_figures / "fig03_width_sweep"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig03-double-descent.png")
    plt.close(fig)


def fig04_lazy_rich(pub: Path, sweep_figures: Path) -> None:
    """Feature drift d_h heatmap (lazy vs rich substructure)."""
    mat_dir = SWEEPS_ROOT / "phase_w_lr" / "matrices"
    dh = pd.read_csv(mat_dir / "d_h_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    plot_heatmap(
        ax, dh.values, dh.columns.astype(float), dh.index.astype(float),
        cmap_name="cividis", fig=fig, colorbar_label=r"$d_h$ (feature drift)",
    )
    ax.set_xlabel("learning rate")
    ax.set_ylabel("width")
    fig.tight_layout()
    out = sweep_figures / "fig04_lazy_rich"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig04-lazy-rich.png")
    plt.close(fig)


def fig05_eos(pub: Path, sweep_figures: Path) -> None:
    df = _load_metrics("eos_mnist_mlp")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    _plot_raw_smooth(axes[0], df["step"], df["train_loss"], logy=True, color="#7c3aed")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("train loss")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    steps = df["step"].to_numpy()
    chi = df["chi"].to_numpy()
    axes[1].fill_between(steps, 0.9, 1.05, color="#fbbf24", alpha=0.25, linewidth=0)
    _plot_raw_smooth(axes[1], steps, chi, color="#d97706")
    axes[1].axhline(1.0, color="#78716c", ls="--", lw=0.8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$\chi$")
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    out = sweep_figures / "fig05_eos"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig05-edge-of-stability.png")
    plt.close(fig)


def fig06_neural_collapse(pub: Path, sweep_figures: Path) -> None:
    snap = pd.read_csv(RUNS_ROOT / "neural_collapse_mnist" / "neural_collapse_snapshots.csv")
    ts = _load_metrics("neural_collapse_mnist")
    epochs = sorted(snap["epoch"].unique())
    pick = [epochs[0], epochs[len(epochs) // 2], epochs[-1]]

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    cmap = plt.get_cmap("tab10")
    for i, ep in enumerate(pick):
        ax = fig.add_subplot(gs[0, i])
        sub = snap[snap["epoch"] == ep]
        for label in sorted(sub["label"].unique()):
            pts = sub[sub["label"] == label]
            ax.scatter(pts["x"], pts["y"], s=4, alpha=0.7, color=cmap(int(label) % 10), linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"epoch {int(ep)}", fontsize=7)
        add_panel_label(ax, chr(ord("a") + i))

    ax_b = fig.add_subplot(gs[1, :])
    _plot_raw_smooth(ax_b, ts["step"], ts["m_nc"], color="#059669")
    ax_b.set_xlabel("step")
    ax_b.set_ylabel(r"$m_{\mathrm{NC}}$")
    add_panel_label(ax_b, "d")
    clean_axis(ax_b)

    out = sweep_figures / "fig06_neural_collapse"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig06-neural-collapse.png")
    plt.close(fig)


def fig07_grokking(pub: Path, sweep_figures: Path) -> None:
    df = _load_metrics("grokking_mod97")
    split = 0
    for i in range(1, len(df)):
        if df["step"].iloc[i] < df["step"].iloc[i - 1]:
            split = i
    df = df.iloc[split:].sort_values("step").drop_duplicates("step")
    steps = np.maximum(df["step"].to_numpy(), 1.0)

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    for col, color, label in [("train_acc", "#0891b2", "train"), ("test_acc", "#dc2626", "test")]:
        y = df[col].to_numpy()
        ys = smooth_ema(y, alpha=SMOOTH_ALPHA)
        ax.plot(steps, y, color=color, lw=0.5, alpha=0.2)
        ax.plot(steps, ys, color=color, lw=1.6, label=label)
    gap = df["train_acc"] - df["test_acc"]
    ax.fill_between(steps, df["test_acc"], df["train_acc"], color="#0891b2", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    clean_axis(ax)

    out = sweep_figures / "fig07_grokking"
    save_figure(fig, out)
    _copy_web_png(out.with_suffix(".png"), pub / "fig07-grokking.png")
    plt.close(fig)


def make_grokking_gif(pub: Path) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    df = _load_metrics("grokking_mod97")
    split = 0
    for i in range(1, len(df)):
        if df["step"].iloc[i] < df["step"].iloc[i - 1]:
            split = i
    df = df.iloc[split:].sort_values("step").drop_duplicates("step")
    steps = np.maximum(df["step"].to_numpy(), 1.0)

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    (ln_tr,) = ax.plot([], [], color="#0891b2", lw=1.6, label="train")
    (ln_te,) = ax.plot([], [], color="#dc2626", lw=1.6, label="test")
    ax.set_xscale("log")
    ax.set_xlim(steps.min(), steps.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    clean_axis(ax)

    frame_idx = np.linspace(0, len(df) - 1, num=min(150, len(df)), dtype=int)

    def update(i: int):
        j = frame_idx[i]
        sub = df.iloc[: j + 1]
        s = np.maximum(sub["step"].to_numpy(), 1.0)
        ln_tr.set_data(s, smooth_ema(sub["train_acc"].to_numpy(), alpha=SMOOTH_ALPHA))
        ln_te.set_data(s, smooth_ema(sub["test_acc"].to_numpy(), alpha=SMOOTH_ALPHA))
        return ln_tr, ln_te

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=60, blit=False)
    gif_path = pub / "phase-transition.gif"
    anim.save(gif_path, writer=PillowWriter(fps=15), dpi=120)
    plt.close(fig)


def _copy_web_png(src: Path, dst: Path) -> None:
    """Copy 600 DPI figure; re-save at blog-friendly width."""
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    w, h = img.size
    target_w = 1200
    if w > target_w:
        img = img.resize((target_w, int(h * target_w / w)), Image.Resampling.LANCZOS)
    img.save(dst, optimize=True)


def publish_all(pub: Path | None = None) -> None:
    set_research_style()
    pub = pub or ASSETS_DIR
    sweep_figures = SWEEPS_ROOT / "figures"
    sweep_figures.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)

    fig01_dashboard(pub, sweep_figures)
    fig02_phase_maps(pub, sweep_figures)
    fig03_width_sweep(pub, sweep_figures)
    fig04_lazy_rich(pub, sweep_figures)
    fig05_eos(pub, sweep_figures)
    fig06_neural_collapse(pub, sweep_figures)
    fig07_grokking(pub, sweep_figures)
    make_grokking_gif(pub)
    print(f"Published blog figures to {pub}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()
    repo_pub = REPO_ROOT / args.publish_dir
    publish_all(repo_pub)


if __name__ == "__main__":
    main()
