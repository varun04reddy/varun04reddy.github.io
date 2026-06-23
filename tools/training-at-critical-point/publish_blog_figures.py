#!/usr/bin/env python3
"""Theory-paper figures: teacher–student phase portraits, phase maps, grokking."""

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


def _load(run_name: str) -> pd.DataFrame:
    p = RUNS_ROOT / run_name / "metrics.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _time_colored_trajectory(ax, x, y, t, *, cmap: str = "plasma") -> None:
    """Phase portrait: trajectory colored by training time."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[mask], y[mask], t[mask]
    if len(x) < 2:
        return
    pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = mcolors.Normalize(t.min(), t.max())
    lc = LineCollection(segs, cmap=plt.get_cmap(cmap), norm=norm, linewidths=1.8)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], s=18, c="#2563eb", zorder=5, edgecolors="white", linewidths=0.4)
    ax.scatter(x[-1], y[-1], s=22, c="#dc2626", zorder=5, edgecolors="white", linewidths=0.4)
    dx = float(np.ptp(x)) + 1e-6
    dy = float(np.ptp(y)) + 1e-6
    ax.set_xlim(x.min() - 0.02 * dx, x.max() + 0.02 * dx)
    ax.set_ylim(y.min() - 0.02 * dy, y.max() + 0.02 * dy)
    return norm


def fig01_phase_portrait(pub: Path, out_dir: Path) -> None:
    """Hero: training trajectory in (R, ε_g) order-parameter space."""
    df = _load("ts_dynamics").dropna(subset=["R", "eps_g"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    norm = _time_colored_trajectory(axes[0], df["R"], df["eps_g"], df["step"], cmap="plasma")
    axes[0].set_xlabel(r"teacher overlap $R$")
    axes[0].set_ylabel(r"gen. error $\varepsilon_g$")
    axes[0].set_yscale("log")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="plasma"), ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("step", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    steps = df["step"].to_numpy()
    eg = smooth_log_ema(np.clip(df["eps_g"].to_numpy(), 1e-6, None), alpha=SMOOTH_ALPHA)
    r = smooth_ema(df["R"].to_numpy(), alpha=SMOOTH_ALPHA)
    axes[1].plot(steps, eg, color="#7c3aed", lw=1.6, label=r"$\varepsilon_g$")
    ax2 = axes[1].twinx()
    ax2.plot(steps, r, color="#059669", lw=1.6, label="$R$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$\varepsilon_g$")
    ax2.set_ylabel("$R$", color="#059669")
    ax2.tick_params(axis="y", labelcolor="#059669", labelsize=6)
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    save_figure(fig, out_dir / "fig01_phase_portrait")
    _copy_web(out_dir / "fig01_phase_portrait.png", pub / "fig01-phase-portrait.png")
    plt.close(fig)


def fig02_ts_phase_map(pub: Path, out_dir: Path) -> None:
    """Student width × lr heatmap of generalization error (teacher–student)."""
    mat_dir = SWEEPS_ROOT / "ts_phase" / "matrices"
    eps = pd.read_csv(mat_dir / "eps_g_matrix.csv", index_col=0)
    overlap = pd.read_csv(mat_dir / "overlap_matrix.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    plot_heatmap(
        axes[0], eps.values, eps.columns.astype(float), eps.index.astype(int),
        cmap_name="magma", log_norm=True, fig=fig, colorbar_label=r"$\varepsilon_g$",
    )
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("student width $K$")
    add_panel_label(axes[0], "a")

    plot_heatmap(
        axes[1], overlap.values, overlap.columns.astype(float), overlap.index.astype(int),
        cmap_name="viridis", fig=fig, colorbar_label="overlap $R$",
    )
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("student width $K$")
    add_panel_label(axes[1], "b")

    fig.tight_layout()
    save_figure(fig, out_dir / "fig02_ts_phase")
    _copy_web(out_dir / "fig02_ts_phase.png", pub / "fig02-phase-diagram.png")
    plt.close(fig)


def fig03_sample_complexity(pub: Path, out_dir: Path) -> None:
    """Sample complexity: ε_g vs α = n/d (classic theory plot)."""
    df = pd.read_csv(SWEEPS_ROOT / "ts_sample" / "summary.csv")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(df["alpha"], df["eps_g"], "o-", color="#2563eb", lw=1.6, ms=5)
    ax.axhline(0.05, color="#78716c", ls=":", lw=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha = n/d$")
    ax.set_ylabel(r"$\varepsilon_g$")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig03_sample")
    _copy_web(out_dir / "fig03_sample.png", pub / "fig03-sample-complexity.png")
    plt.close(fig)


def fig04_double_descent(pub: Path, out_dir: Path) -> None:
    """Width sweep: gen error spike + learning trajectories colored by K."""
    end = pd.read_csv(SWEEPS_ROOT / "ts_width" / "summary.csv").sort_values("student_width")
    traj = pd.read_csv(SWEEPS_ROOT / "ts_width" / "aggregated.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    cmap, norm = plot_sweep_curves(
        axes[0], traj, x="step", y="eps_g", sweep="student_width",
        cmap_name="viridis", log_color=True, log_smooth=True, smooth_alpha=SMOOTH_ALPHA,
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel(r"$\varepsilon_g$")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])
    add_colorbar(fig, axes[0], cmap, norm, "$K$")

    k = end["student_width"].to_numpy()
    axes[1].plot(k, end["train_mse"], "o-", color="#2563eb", lw=1.4, ms=4, label="train")
    axes[1].plot(k, end["eps_g"], "o-", color="#dc2626", lw=1.4, ms=4, label="gen.")
    axes[1].axvline(4, color="#78716c", ls=":", lw=0.9, label=r"$K^*=4$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("student width $K$")
    axes[1].set_ylabel("MSE")
    axes[1].legend(frameon=False, fontsize=6)
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    save_figure(fig, out_dir / "fig04_double_descent")
    _copy_web(out_dir / "fig04_double_descent.png", pub / "fig04-double-descent.png")
    plt.close(fig)


def fig05_eos(pub: Path, out_dir: Path) -> None:
    """EOS: parametric curve in (χ, loss) colored by time."""
    df = _load("ts_eos").dropna(subset=["chi", "train_mse"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))

    norm = _time_colored_trajectory(axes[0], df["chi"], df["train_mse"], df["step"], cmap="cividis")
    axes[0].fill_between([0.9, 1.05], 0, 1e6, color="#fbbf24", alpha=0.15, transform=axes[0].get_xaxis_transform(), linewidth=0)
    axes[0].axvline(1.0, color="#78716c", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\chi = \eta\lambda_{\max}/2$")
    axes[0].set_ylabel("train MSE")
    axes[0].set_yscale("log")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    steps = df["step"].to_numpy()
    chi = df["chi"].to_numpy()
    chi_s = smooth_ema(chi, alpha=SMOOTH_ALPHA)
    loss_s = smooth_log_ema(np.clip(df["train_mse"].to_numpy(), 1e-8, None), alpha=SMOOTH_ALPHA)
    axes[1].fill_between(steps, 0.9, 1.05, color="#fbbf24", alpha=0.2, linewidth=0)
    axes[1].plot(steps, chi_s, color="#d97706", lw=1.6)
    axes[1].axhline(1.0, color="#78716c", ls="--", lw=0.8)
    ax2 = axes[1].twinx()
    ax2.plot(steps, loss_s, color="#7c3aed", lw=1.4, alpha=0.85)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"$\chi$")
    ax2.set_ylabel("train MSE", color="#7c3aed")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="#7c3aed", labelsize=6)
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    save_figure(fig, out_dir / "fig05_eos")
    _copy_web(out_dir / "fig05_eos.png", pub / "fig05-edge-of-stability.png")
    plt.close(fig)


def fig06_grokking(pub: Path, out_dir: Path) -> None:
    df = _load("grokking_mod97")
    split = 0
    for i in range(1, len(df)):
        if df["step"].iloc[i] < df["step"].iloc[i - 1]:
            split = i
    df = df.iloc[split:].sort_values("step").drop_duplicates("step")
    steps = np.maximum(df["step"].to_numpy(), 1.0)
    gap = df["train_acc"] - df["test_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for col, color, label in [("train_acc", "#0891b2", "train"), ("test_acc", "#dc2626", "test")]:
        y = smooth_ema(df[col].to_numpy(), alpha=SMOOTH_ALPHA)
        axes[0].plot(steps, y, color=color, lw=1.6, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    add_panel_label(axes[0], "a")
    clean_axis(axes[0])

    g = smooth_ema(gap.to_numpy(), alpha=SMOOTH_ALPHA)
    axes[1].fill_between(steps, 0, g, color="#7c3aed", alpha=0.35, linewidth=0)
    axes[1].plot(steps, g, color="#7c3aed", lw=1.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(r"gap $g = A_{\mathrm{train}} - A_{\mathrm{test}}$")
    add_panel_label(axes[1], "b")
    clean_axis(axes[1])

    fig.tight_layout()
    save_figure(fig, out_dir / "fig06_grokking")
    _copy_web(out_dir / "fig06_grokking.png", pub / "fig06-grokking.png")
    plt.close(fig)


def make_grokking_gif(pub: Path) -> None:
    from matplotlib.animation import FuncAnimation, PillowWriter

    df = _load("grokking_mod97")
    for i in range(1, len(df)):
        if df["step"].iloc[i] < df["step"].iloc[i - 1]:
            df = df.iloc[i:]
            break
    df = df.sort_values("step").drop_duplicates("step")
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
    anim.save(pub / "phase-transition.gif", writer=PillowWriter(fps=15), dpi=120)
    plt.close(fig)


def _copy_web(src: Path, dst: Path) -> None:
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
    out_dir = SWEEPS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pub.mkdir(parents=True, exist_ok=True)

    fig01_phase_portrait(pub, out_dir)
    fig02_ts_phase_map(pub, out_dir)
    fig03_sample_complexity(pub, out_dir)
    fig04_double_descent(pub, out_dir)
    fig05_eos(pub, out_dir)
    fig06_grokking(pub, out_dir)
    make_grokking_gif(pub)
    print(f"Published theory figures to {pub}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()
    publish_all(REPO_ROOT / args.publish_dir if not args.publish_dir.is_absolute() else args.publish_dir)


if __name__ == "__main__":
    main()
