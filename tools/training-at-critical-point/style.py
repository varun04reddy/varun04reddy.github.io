"""Shared matplotlib style for critical-point blog figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

FIG_DPI = 300
BG = "#f8f7f4"
TEXT = "#1b1b1b"
CMAP_HEAT = "plasma"
CMAP_SEQ = "viridis"

PALETTE = {
    "train": "#0077b6",
    "test": "#d62828",
    "accent": "#2a9d8f",
    "muted": "#6c757d",
    "gold": "#f4a261",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": "#bdbdbd",
            "axes.labelcolor": TEXT,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.size": 10,
            "font.family": "sans-serif",
            "legend.frameon": False,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "savefig.facecolor": BG,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
