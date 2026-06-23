#!/usr/bin/env python3

# Legacy CSVs: experiments/training-at-critical-point/outputs/ (from run_experiments.py or plot_from_runs.py after GPU runs).
"""Plot blog figures from experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from style import PALETTE, apply_style, save


def load(csv_dir: Path, name: str) -> pd.DataFrame:
    path = csv_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_experiments.py first.")
    return pd.read_csv(path)


def fig01_micro_macro(csv_dir: Path, pub: Path) -> None:
    apply_style()
    ts = load(csv_dir, "training_timeseries.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    ax = axes[0]
    ax.plot(ts["step"], ts["loss"], color=PALETTE["train"], lw=1.8, alpha=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("batch loss")
    ax.set_title("Loss trajectory")

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(ts["step"], ts["chi"], color=PALETTE["gold"], lw=2, label=r"$\chi = \eta\lambda_{\max}/2$")
    ax.axhline(1.0, color=PALETTE["muted"], ls="--", lw=1, alpha=0.7)
    ax2.plot(ts["step"], ts["m_nc"], color=PALETTE["accent"], lw=2, alpha=0.85, label=r"$m_{\mathrm{NC}}$")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"sharpness ratio $\chi$", color=PALETTE["gold"])
    ax2.set_ylabel(r"collapse order $m_{\mathrm{NC}}$", color=PALETTE["accent"])
    ax.set_title("Macroscopic order parameters")
    ax.set_ylim(0, max(1.2, ts["chi"].max() * 1.05))

    fig.suptitle("Figure 1. From microscopic steps to macroscopic gauges (MNIST MLP)", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, pub / "fig01-micro-macro.png")


def fig02_phase_diagram(csv_dir: Path, pub: Path) -> None:
    apply_style()
    df = load(csv_dir, "phase_diagram.csv")
    pivot = df.pivot(index="width", columns="lr", values="final_test_acc")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="plasma", vmin=pivot.values.min(), vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("hidden width")
    ax.set_title("Figure 2. Empirical phase map: test accuracy (MNIST MLP sweep)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("test accuracy")
    save(fig, pub / "fig02-phase-diagram.png")


def fig03_double_descent(csv_dir: Path, pub: Path) -> None:
    apply_style()
    df = load(csv_dir, "double_descent.csv").sort_values("n_params")
    interp_idx = df[df["train_acc"] > 0.99]["n_params"].min() if (df["train_acc"] > 0.99).any() else df["n_params"].median()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvspan(df["n_params"].min(), interp_idx, color="#d8e2dc", alpha=0.45)
    ax.axvspan(interp_idx, df["n_params"].max(), color="#cddafd", alpha=0.35)
    ax.plot(df["n_params"], df["train_loss"], "o-", color=PALETTE["train"], lw=2, label="train loss")
    ax.plot(df["n_params"], df["test_loss"], "o-", color=PALETTE["test"], lw=2, label="test loss")
    ax.axvline(interp_idx, color=PALETTE["muted"], ls=":", lw=1.5)
    ax.text(interp_idx * 1.05, df["test_loss"].max() * 0.92, "interpolation", fontsize=9, color=PALETTE["muted"])
    ax.set_xscale("log")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Figure 3. Double descent near interpolation (MNIST width sweep)")
    ax.legend(loc="upper right")
    save(fig, pub / "fig03-double-descent.png")


def fig04_edge_of_stability(csv_dir: Path, pub: Path) -> None:
    apply_style()
    ts = load(csv_dir, "training_timeseries.csv")
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ts["step"], ts["loss"], color=PALETTE["accent"], lw=2, label="train loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss", color=PALETTE["accent"])
    ax2 = ax1.twinx()
    ax2.fill_between(ts["step"], 0.9, 1.05, color=PALETTE["gold"], alpha=0.2, label="edge band")
    ax2.plot(ts["step"], ts["chi"], color=PALETTE["gold"], lw=2, label=r"$\chi$")
    ax2.axhline(1.0, color=PALETTE["test"], ls="--", lw=1.2)
    ax2.set_ylabel(r"$\chi = \eta\lambda_{\max}(H)/2$", color=PALETTE["gold"])
    ax1.set_title("Figure 4. Edge of stability (MNIST MLP, SGD)")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    save(fig, pub / "fig04-edge-of-stability.png")


def fig05_neural_collapse(csv_dir: Path, pub: Path) -> None:
    apply_style()
    # Reconstruct from snapshots - use PCA per epoch in plot if we have raw dims
    df = load(csv_dir, "neural_collapse_snapshots.csv")
    epochs = sorted(df["epoch"].unique())
    pick = [epochs[0], epochs[len(epochs) // 2], epochs[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    cmap = plt.cm.tab10
    for ax, ep in zip(axes, pick):
        sub = df[df["epoch"] == ep]
        for label in sub["label"].unique()[:10]:
            pts = sub[sub["label"] == label]
            ax.scatter(pts["x"], pts["y"], s=10, alpha=0.65, color=cmap(int(label) % 10))
        ax.set_title(f"epoch {int(ep)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Figure 5. Neural collapse in penultimate features (MNIST MLP)", fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, pub / "fig05-neural-collapse.png")


def fig06_grokking(csv_dir: Path, pub: Path) -> None:
    apply_style()
    g = load(csv_dir, "grokking.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.maximum(g["step"].values, 1)
    ax.plot(steps, g["train_acc"], color=PALETTE["accent"], lw=2, label="train acc")
    ax.plot(steps, g["test_acc"], color=PALETTE["test"], lw=2, label="test acc")
    ax.fill_between(steps, g["test_acc"], g["train_acc"], color=PALETTE["train"], alpha=0.15, label="gap")
    if g["step"].max() > 2000:
        ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Figure 6. Modular addition (mod 97): delayed generalization")
    ax.legend(loc="lower right")
    save(fig, pub / "fig06-grokking.png")


def fig07_scaling_laws(csv_dir: Path, pub: Path) -> None:
    apply_style()
    df = load(csv_dir, "double_descent.csv").sort_values("n_params")
    x = df["n_params"].values
    y = df["test_loss"].values
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(x, y, "o", color="#8338ec", ms=7, label="test loss")
    # fit power law on overparameterized tail
    mask = df["train_acc"] > 0.95
    if mask.sum() >= 3:
        lx = np.log(x[mask])
        ly = np.log(y[mask])
        b, log_a = np.polyfit(lx, ly, 1)
        a = np.exp(log_a)
        xf = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.loglog(xf, a * xf ** b, color=PALETTE["train"], lw=2, label=rf"fit $\propto N^{{{b:.2f}}}$")
    ax.set_xlabel("parameter count")
    ax.set_ylabel("test loss")
    ax.set_title("Figure 7. Scaling trend in the overparameterized tail")
    ax.legend()
    save(fig, pub / "fig07-scaling-laws.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=Path("experiments/training-at-critical-point/outputs"))
    parser.add_argument("--publish-dir", type=Path, default=Path("assets/img/blog/critical-point"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    csv_dir = repo / args.csv_dir
    pub = repo / args.publish_dir

    fig01_micro_macro(csv_dir, pub)
    fig02_phase_diagram(csv_dir, pub)
    fig03_double_descent(csv_dir, pub)
    fig04_edge_of_stability(csv_dir, pub)
    fig05_neural_collapse(csv_dir, pub)
    fig06_grokking(csv_dir, pub)
    fig07_scaling_laws(csv_dir, pub)
    print(f"Published to {pub}")


if __name__ == "__main__":
    main()
