#!/usr/bin/env python3
"""Generate schematic figures for the Training at the Critical Point blog post.

Run from repo root:
  python tools/training-at-critical-point/generate_figures.py

Defaults:
  --scratch-dir experiments/training-at-critical-point/outputs
  --publish-dir assets/img/blog/critical-point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
import numpy as np

# Consistent blog aesthetic: off-white paper, viridis/plasma accents
FIG_DPI = 200
BG = "#faf9f6"
TEXT = "#1a1a1a"
ACCENT = "#2d6a4f"


def style_axes(ax, title: str = "") -> None:
    ax.set_facecolor(BG)
    ax.figure.patch.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.title.set_color(TEXT)
    if title:
        ax.set_title(title, fontsize=11, fontweight="600", color=TEXT, pad=10)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {path}")


def fig01_micro_macro(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    style_axes(ax, "Microscopic: parameter trajectory")
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 400)
    x = np.cumsum(rng.normal(0, 0.04, len(t)))
    y = np.cumsum(rng.normal(0, 0.04, len(t)))
    ax.plot(x, y, color="#7209b7", lw=1.2, alpha=0.85)
    ax.scatter([x[0]], [y[0]], c="#4361ee", s=40, zorder=5, label=r"$\theta_0$")
    ax.scatter([x[-1]], [y[-1]], c="#f72585", s=40, zorder=5, label=r"$\theta_T$")
    ax.set_xlabel("parameter direction 1")
    ax.set_ylabel("parameter direction 2")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    style_axes(ax, "Macroscopic: order parameters")
    names = [r"$\lambda_{\max}(H)$", r"$m_{\mathrm{NC}}$", r"$\|\theta_t-\theta_0\|$", "gen. gap"]
    vals = [0.72, 0.91, 0.34, 0.08]
    colors = ["#3a86ff", "#ff006e", "#8338ec", "#fb5607"]
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, color=colors, height=0.55, alpha=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("normalized gauge reading")
    ax.axvline(0.5, color="#999999", ls="--", lw=0.8, alpha=0.6)

    fig.suptitle(
        "Figure 1. Microscopic vs macroscopic views of training (schematic)",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        y=1.02,
    )
    fig.tight_layout()
    save(fig, out / "fig01-micro-macro.png")


def fig02_phase_diagram(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    style_axes(ax)

    x = np.linspace(0, 1, 300)
    y = np.linspace(0, 1, 300)
    X, Y = np.meshgrid(x, y)
    # Cartoon order field: rises with capacity and training time, peaks near center-right
    Z = (
        0.35 * X
        + 0.55 * Y
        + 0.25 * np.sin(4 * np.pi * X) * np.exp(-((X - 0.55) ** 2 + (Y - 0.45) ** 2) / 0.08)
    )
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    im = ax.contourf(X, Y, Z, levels=24, cmap="plasma", alpha=0.95)
    ax.contour(X, Y, Z, levels=12, colors="white", linewidths=0.35, alpha=0.45)

    regions = [
        (0.12, 0.82, "underfitting", "#ffffff"),
        (0.38, 0.72, "lazy / kernel", "#ffffff"),
        (0.52, 0.52, "interpolation\nthreshold", "#ffffcc"),
        (0.68, 0.38, "memorization", "#ffffff"),
        (0.78, 0.58, "edge of\nstability", "#ffe8a1"),
        (0.55, 0.22, "feature learning", "#ffffff"),
        (0.82, 0.18, "grokking", "#ffffff"),
        (0.35, 0.12, "neural collapse", "#ffffff"),
    ]
    for px, py, label, fc in regions:
        ax.text(
            px,
            py,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="#555555", alpha=0.85, lw=0.6),
        )

    ax.set_xlabel(r"effective capacity ($N_{\mathrm{eff}}$ / data complexity)")
    ax.set_ylabel(r"training time / implicit regularization")
    ax.set_title(
        "Figure 2. Cartoon phase diagram of deep learning (schematic, not measured)",
        fontsize=11,
        fontweight="600",
        pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("representation order (schematic)", fontsize=9)

    save(fig, out / "fig02-phase-diagram.png")


def fig03_double_descent(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    style_axes(ax)

    n = np.linspace(0.05, 1, 500)
    n_star = 0.42

    classical = 0.15 + 0.55 * (n - 0.15) ** 2 + 0.08 * np.exp(-((n - 0.35) ** 2) / 0.02)
    double = (
        0.55 * np.exp(-((n - n_star) ** 2) / 0.004)
        + 0.12
        + 0.08 * np.exp(-n / 0.25)
    )
    double = double + 0.06 * (1 - np.exp(-3 * n))

    ax.axvspan(0, n_star - 0.06, color="#d8e2dc", alpha=0.5, label="underparameterized")
    ax.axvspan(n_star - 0.06, n_star + 0.06, color="#ffe5d9", alpha=0.65, label="interpolation region")
    ax.axvspan(n_star + 0.06, 1.0, color="#cddafd", alpha=0.45, label="overparameterized")

    ax.plot(n, classical, ls="--", color="#6c757d", lw=1.8, label="classical U-curve")
    ax.plot(n, double, color="#e63946", lw=2.2, label="double descent")
    ax.axvline(n_star, color="#457b9d", ls=":", lw=1.5)
    ax.text(n_star + 0.02, 0.52, r"$N_{\mathrm{eff}} \approx n$", fontsize=9, color="#457b9d")

    ax.set_xlabel("model capacity")
    ax.set_ylabel("test error")
    ax.set_title("Figure 3. Double descent near the interpolation threshold (schematic)", fontweight="600")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 0.75)

    save(fig, out / "fig03-double-descent.png")


def fig04_edge_of_stability(out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    style_axes(ax1)

    steps = np.arange(0, 800)
    loss = 2.2 * np.exp(-steps / 280) + 0.15 + 0.04 * np.sin(steps / 18)
    eta = 0.1
    threshold = 2.0 / eta
    sharp = threshold * (1 - np.exp(-steps / 220)) + 0.08 * np.sin(steps / 12)
    sharp = np.clip(sharp, 0, threshold * 1.02)

    ax1.plot(steps, loss, color="#2a9d8f", lw=2, label="train loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss", color="#2a9d8f")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")

    ax2 = ax1.twinx()
    ax2.set_facecolor(BG)
    ax2.fill_between(
        steps,
        threshold * 0.92,
        threshold * 1.02,
        color="#f4a261",
        alpha=0.25,
        label="edge of stability band",
    )
    ax2.axhline(threshold, color="#e76f51", ls="--", lw=1.5, label=r"$2/\eta$")
    ax2.plot(steps, sharp, color="#e76f51", lw=2, alpha=0.95, label=r"$\lambda_{\max}(H)$")
    ax2.set_ylabel(r"sharpness $\lambda_{\max}(H)$", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    ax1.set_title(
        "Figure 4. Training rides the edge of stability (schematic; Cohen et al.)",
        fontweight="600",
        pad=10,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7, loc="upper right")

    save(fig, out / "fig04-edge-of-stability.png")


def fig05_neural_collapse(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    rng = np.random.default_rng(7)
    titles = ["early training", "interpolation", "neural collapse"]
    n_per = 35

    for ax, title, stage in zip(axes, titles, range(3)):
        style_axes(ax, title)
        centers = np.array([[0, 0], [2.5, 0], [-1.2, 2.2], [1.2, -2.0]])
        colors = ["#4361ee", "#f72585", "#4cc9f0", "#7209b7"]
        for c_idx, (cx, cy) in enumerate(centers):
            if stage == 0:
                pts = rng.normal([cx, cy], 1.1, size=(n_per, 2))
            elif stage == 1:
                pts = rng.normal([cx * 1.2, cy * 1.2], 0.55, size=(n_per, 2))
            else:
                pts = rng.normal([cx * 1.35, cy * 1.35], 0.12, size=(n_per, 2))
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.75, c=colors[c_idx], edgecolors="none")
            ax.scatter([cx * (1.2 if stage else 1.0)], [cy * (1.2 if stage else 1.0)], s=80, c=colors[c_idx], marker="X", edgecolors="white", linewidths=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Figure 5. Neural collapse as representation crystallization (schematic)",
        fontsize=11,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    save(fig, out / "fig05-neural-collapse.png")


def fig06_grokking(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    style_axes(ax)

    steps = np.logspace(1, 4.3, 400)
    train = 1 - 0.95 * (1 - np.exp(-steps / 80))
    test = 0.05 + 0.9 / (1 + np.exp(-(np.log10(steps) - 2.85) * 6))
    gap = train - test
    tc = 10 ** 2.85

    ax.plot(steps, train, color="#2a9d8f", lw=2.2, label="train accuracy")
    ax.plot(steps, test, color="#e63946", lw=2.2, label="test accuracy")
    ax.fill_between(steps, test, train, color="#457b9d", alpha=0.2, label="generalization gap")
    ax.axvline(tc, color="#6c757d", ls=":", lw=1.5)
    ax.text(tc * 1.15, 0.35, r"$t_c$", fontsize=10, color="#6c757d")

    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Figure 6. Grokking as a delayed phase transition (schematic)", fontweight="600")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    save(fig, out / "fig06-grokking.png")


def fig07_scaling_laws(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    style_axes(ax)

    n = np.logspace(1, 5, 60)
    l_inf, a, alpha = 0.08, 2.5, 0.12
    loss = l_inf + a * n ** (-alpha)
    ax.loglog(n, loss, color="#3a86ff", lw=2.2, label=r"$L(N)=L_\infty + aN^{-\alpha}$")
    ax.scatter(n[::5], loss[::5] * (1 + 0.02 * np.sin(np.arange(len(n[::5])))), s=18, c="#8338ec", alpha=0.8)

    ax.set_xlabel("model size / compute (log)")
    ax.set_ylabel("loss (log)")
    ax.set_title("Figure 7. Scaling laws inside a regime (schematic; Kaplan et al.)", fontweight="600")
    ax.legend(frameon=False, fontsize=9)

    save(fig, out / "fig07-scaling-laws.png")


FIGURES = {
    "all": [fig01_micro_macro, fig02_phase_diagram, fig03_double_descent, fig04_edge_of_stability, fig05_neural_collapse, fig06_grokking, fig07_scaling_laws],
    "core": [fig01_micro_macro, fig02_phase_diagram, fig03_double_descent, fig04_edge_of_stability, fig05_neural_collapse, fig06_grokking],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("experiments/training-at-critical-point/outputs"),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("assets/img/blog/critical-point"),
    )
    parser.add_argument("--set", choices=["all", "core"], default="all")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scratch = repo_root / args.scratch_dir
    publish = repo_root / args.publish_dir

    for fn in FIGURES[args.set]:
        fn(scratch)
        fn(publish)

    print(f"\nPublished copies: {publish}")


if __name__ == "__main__":
    main()
