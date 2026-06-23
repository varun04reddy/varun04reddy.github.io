#!/usr/bin/env python3
"""Aggregate step-level run CSVs into legacy outputs/ and publish blog figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def aggregate_phase(runs_root: Path) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(runs_root.glob("phase_w*_lr*/config.yaml")):
        run_dir = cfg_path.parent
        metrics = run_dir / "metrics.csv"
        if not metrics.exists():
            continue
        df = pd.read_csv(metrics)
        last = df.iloc[-1]
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        rows.append(
            {
                "width": cfg["width"],
                "lr": cfg["lr"],
                "n_params": None,
                "final_test_acc": last.get("val_acc", last.get("test_acc")),
                "final_train_acc": last.get("train_acc"),
                "final_test_loss": last.get("val_loss"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_grokking(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "metrics.csv")
    out = df[["step", "train_acc"]].copy()
    out["test_acc"] = df["test_acc"] if "test_acc" in df.columns else df["val_acc"]
    if "train_loss" in df.columns:
        out["loss"] = df["train_loss"]
    return out


def aggregate_eos(run_dir: Path) -> pd.DataFrame:
    ts = pd.read_csv(run_dir / "metrics.csv")
    ts = ts.rename(columns={"train_loss": "loss", "val_acc": "test_acc"})
    keep = [c for c in ["step", "loss", "lambda_max", "chi", "theta_dist", "m_nc", "test_acc"] if c in ts.columns]
    return ts[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "experiments/training-at-critical-point/runs",
    )
    parser.add_argument(
        "--legacy-out",
        type=Path,
        default=REPO_ROOT / "experiments/training-at-critical-point/outputs",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=REPO_ROOT / "assets/img/blog/critical-point",
    )
    args = parser.parse_args()

    args.legacy_out.mkdir(parents=True, exist_ok=True)
    phase = aggregate_phase(args.runs_root)
    if len(phase):
        phase.to_csv(args.legacy_out / "phase_diagram.csv", index=False)

    grok_dir = args.runs_root / "grokking_mod97"
    if (grok_dir / "metrics.csv").exists():
        aggregate_grokking(grok_dir).to_csv(args.legacy_out / "grokking.csv", index=False)

    eos_dir = args.runs_root / "eos_mnist_mlp"
    if (eos_dir / "metrics.csv").exists():
        aggregate_eos(eos_dir).to_csv(args.legacy_out / "training_timeseries.csv", index=False)

    nc_snap = args.runs_root / "neural_collapse_mnist" / "neural_collapse_snapshots.csv"
    if nc_snap.exists():
        shutil.copy(nc_snap, args.legacy_out / "neural_collapse_snapshots.csv")

    (args.legacy_out / "meta.json").write_text(json.dumps({"source": "plot_from_runs"}, indent=2))

    sys_path = Path(__file__).resolve().parent
    import sys

    sys.path.insert(0, str(sys_path))
    from plot_figures import fig02_phase_diagram, fig04_edge_of_stability, fig05_neural_collapse, fig06_grokking

    pub = args.publish_dir
    pub.mkdir(parents=True, exist_ok=True)
    fig02_phase_diagram(args.legacy_out, pub)
    fig04_edge_of_stability(args.legacy_out, pub)
    fig05_neural_collapse(args.legacy_out, pub)
    fig06_grokking(args.legacy_out, pub)
    print(f"Aggregated CSVs -> {args.legacy_out}")
    print(f"Figures -> {pub}")


if __name__ == "__main__":
    main()
