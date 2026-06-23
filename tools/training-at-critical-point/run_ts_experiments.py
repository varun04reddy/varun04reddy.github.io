"""Ganguli/Pehlevan-style teacher–student experiment bundle."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from teacher_student import (
    TSConfig,
    TSData,
    feature_drift,
    init_student,
    init_teacher,
    make_dataset,
    mse,
    normalized_gen_error,
    overlap_matrix,
    per_teacher_overlap,
    readout_overlap,
    teacher_overlap,
    train_student,
)

CANONICAL = TSConfig(
    input_dim=100,
    manifold_dim=15,
    teacher_width=8,
    noise_std=0.05,
    n_train=4000,
    n_test=10000,
    init_scale=1.0,
)
STUDENT_K = 32
STUDENT_K_RICH = 8  # matched width for symmetry-breaking dynamics
LR_DEFAULT = 0.08
RICH_INIT = 1.0


def _log_order_params(
    student: torch.Tensor | Any,
    teacher: Any,
    data: TSData,
    h0: torch.Tensor,
    row: dict[str, Any],
) -> dict[str, Any]:
    r_per = per_teacher_overlap(student, teacher).cpu().numpy()
    row.update(
        {
            "R": teacher_overlap(student, teacher),
            "R_out": readout_overlap(student, teacher),
            "eps_g": normalized_gen_error(student, teacher, data.x_te, data.y_te),
            "d_h": feature_drift(student, data.x_te[:512], h0),
        }
    )
    for j, rv in enumerate(r_per):
        row[f"R_t{j}"] = float(rv)
    return row


def run_dynamics(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> Path:
    cfg = TSConfig(
        input_dim=CANONICAL.input_dim,
        manifold_dim=CANONICAL.manifold_dim,
        teacher_width=CANONICAL.teacher_width,
        noise_std=CANONICAL.noise_std,
        n_train=CANONICAL.n_train,
        n_test=CANONICAL.n_test,
        init_scale=RICH_INIT,
    )
    run_dir = runs_root / "ts_dynamics"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.csv").exists():
        (run_dir / "metrics.csv").unlink()
    logger_path = run_dir / "metrics.csv"
    teacher = init_teacher(cfg, device, seed)
    student = init_student(cfg, STUDENT_K_RICH, device, seed, init_scale=RICH_INIT)
    data = make_dataset(cfg, teacher, device, seed, on_manifold=True)
    student.eval()
    with torch.no_grad():
        h0 = student.hidden(data.x_te[:512]).clone()
    align_rows: list[dict] = []
    snap_at = {0, 80, 320, 1280, 3200, 6400}

    def record_snap(st: int, model: Any) -> None:
        mat = overlap_matrix(model, teacher).cpu().numpy()
        for si in range(mat.shape[0]):
            for ti in range(mat.shape[1]):
                align_rows.append({"step": st, "s_neuron": si, "t_neuron": ti, "overlap": float(mat[si, ti])})

    rows: list[dict] = []

    def on_step(step: int, epoch: int, loss: float, model: Any) -> None:
        model.eval()
        row = {"step": step, "epoch": epoch, "train_mse": loss, "lr": LR_DEFAULT}
        _log_order_params(model, teacher, data, h0, row)
        rows.append(row)
        if step in snap_at:
            record_snap(step, model)
        model.train()

    record_snap(0, student)
    train_student(student, teacher, data, lr=LR_DEFAULT, epochs=2500, batch_size=256, log_every=4, on_step=on_step)
    record_snap(rows[-1]["step"] if rows else 0, student)

    pd.DataFrame(rows).to_csv(logger_path, index=False)
    with (run_dir / "alignment_snapshots.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "s_neuron", "t_neuron", "overlap"])
        w.writeheader()
        w.writerows(align_rows)
    log.info("dynamics R=%.3f eps_g=%.5f", rows[-1]["R"], rows[-1]["eps_g"])
    return run_dir


def run_lr_trajectories(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    cfg = TSConfig(
        input_dim=CANONICAL.input_dim,
        manifold_dim=CANONICAL.manifold_dim,
        teacher_width=CANONICAL.teacher_width,
        noise_std=CANONICAL.noise_std,
        n_train=CANONICAL.n_train,
        n_test=CANONICAL.n_test,
        init_scale=RICH_INIT,
    )
    lrs = [0.015, 0.03, 0.08, 0.15, 0.35]
    sweep_root = runs_root / "_sweeps" / "ts_lr"
    sweep_root.mkdir(parents=True, exist_ok=True)
    teacher = init_teacher(cfg, device, seed)
    data = make_dataset(cfg, teacher, device, seed, on_manifold=True)
    frames: list[pd.DataFrame] = []
    for lr in lrs:
        student = init_student(cfg, STUDENT_K_RICH, device, seed + int(lr * 1e4), init_scale=RICH_INIT)
        student.eval()
        with torch.no_grad():
            h0 = student.hidden(data.x_te[:512]).clone()
        rows: list[dict] = []

        def on_step(step: int, epoch: int, loss: float, model: Any, _lr: float = lr) -> None:
            model.eval()
            row = {"step": step, "train_mse": loss, "lr": _lr}
            _log_order_params(model, teacher, data, h0, row)
            rows.append(row)
            model.train()

        train_student(student, teacher, data, lr=lr, epochs=900, batch_size=256, log_every=6, on_step=on_step)
        frames.append(pd.DataFrame(rows))
        log.info("lr=%s final R=%.3f eps_g=%.4f", lr, rows[-1]["R"], rows[-1]["eps_g"])
        del student
    pd.concat(frames, ignore_index=True).to_csv(sweep_root / "aggregated.csv", index=False)


def run_snr_phase(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    """(alpha, sigma) phase diagram — signal-to-noise transition (Ganguli/Gerace style)."""
    alphas = [2, 4, 8, 16, 32, 64, 128]
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    teacher = init_teacher(CANONICAL, device, seed)
    summary: list[dict] = []
    for alpha in alphas:
        n_tr = alpha * CANONICAL.input_dim
        for sigma in sigmas:
            cfg = TSConfig(
                input_dim=CANONICAL.input_dim,
                manifold_dim=CANONICAL.manifold_dim,
                teacher_width=CANONICAL.teacher_width,
                n_train=n_tr,
                noise_std=sigma,
            )
            data = make_dataset(cfg, teacher, device, seed + n_tr + int(sigma * 1e4), on_manifold=True)
            student = init_student(cfg, STUDENT_K, device, seed + n_tr)
            train_student(student, teacher, data, lr=LR_DEFAULT, epochs=500, batch_size=256)
            student.eval()
            summary.append(
                {
                    "alpha": alpha,
                    "sigma": sigma,
                    "R": teacher_overlap(student, teacher),
                    "eps_g": normalized_gen_error(student, teacher, data.x_te, data.y_te),
                }
            )
            del student
    out = runs_root / "_sweeps" / "ts_snr"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrices").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    for col, fname in [("eps_g", "eps_g_matrix.csv"), ("R", "R_matrix.csv")]:
        df.pivot(index="sigma", columns="alpha", values=col).to_csv(out / "matrices" / fname)
    log.info("snr phase done (%s cells)", len(summary))


def run_sample_multi_k(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    widths = [8, 16, 32, 64]
    alphas = [2, 4, 8, 16, 32, 64, 128, 256]
    teacher = init_teacher(CANONICAL, device, seed)
    summary: list[dict] = []
    for k in widths:
        for alpha in alphas:
            n_tr = alpha * CANONICAL.input_dim
            cfg = TSConfig(
                input_dim=CANONICAL.input_dim,
                manifold_dim=CANONICAL.manifold_dim,
                teacher_width=CANONICAL.teacher_width,
                n_train=n_tr,
                noise_std=CANONICAL.noise_std,
            )
            data = make_dataset(cfg, teacher, device, seed + k + n_tr, on_manifold=True)
            student = init_student(cfg, k, device, seed + k + n_tr)
            epochs = min(800, max(300, n_tr // 8))
            train_student(student, teacher, data, lr=LR_DEFAULT, epochs=epochs, batch_size=256)
            student.eval()
            summary.append({"alpha": alpha, "student_width": k, "R": teacher_overlap(student, teacher), "eps_g": normalized_gen_error(student, teacher, data.x_te, data.y_te)})
            del student
    out = runs_root / "_sweeps" / "ts_sample_k"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    log.info("sample multi-K done")


def run_phase_map(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    widths = [4, 8, 16, 32, 64, 128]
    lrs = [0.005, 0.01, 0.03, 0.08, 0.15, 0.4]
    teacher = init_teacher(CANONICAL, device, seed)
    data = make_dataset(CANONICAL, teacher, device, seed, on_manifold=True)
    summary: list[dict] = []
    for k in widths:
        for lr in lrs:
            student = init_student(CANONICAL, k, device, seed + k * 100 + int(lr * 1e4))
            train_student(student, teacher, data, lr=lr, epochs=550, batch_size=256)
            student.eval()
            summary.append(
                {
                    "student_width": k,
                    "lr": lr,
                    "R": teacher_overlap(student, teacher),
                    "eps_g": normalized_gen_error(student, teacher, data.x_te, data.y_te),
                }
            )
            del student
    out = runs_root / "_sweeps" / "ts_phase"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    mat_dir = out / "matrices"
    mat_dir.mkdir(exist_ok=True)
    df.pivot(index="student_width", columns="lr", values="eps_g").to_csv(mat_dir / "eps_g_matrix.csv")
    df.pivot(index="student_width", columns="lr", values="R").to_csv(mat_dir / "overlap_matrix.csv")


def run_lazy_rich(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    """Init-scale × lr heatmap of feature drift d_h (Pehlevan lazy/rich)."""
    init_scales = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    lrs = [0.01, 0.03, 0.08, 0.15, 0.35]
    teacher = init_teacher(CANONICAL, device, seed)
    data = make_dataset(CANONICAL, teacher, device, seed, on_manifold=True)
    summary: list[dict] = []
    for gamma in init_scales:
        for lr in lrs:
            student = init_student(CANONICAL, STUDENT_K, device, seed + int(gamma * 1000), init_scale=gamma)
            student.eval()
            with torch.no_grad():
                h0 = student.hidden(data.x_te[:512]).clone()
            train_student(student, teacher, data, lr=lr, epochs=350, batch_size=256)
            student.eval()
            summary.append(
                {
                    "init_scale": gamma,
                    "lr": lr,
                    "d_h": feature_drift(student, data.x_te[:512], h0),
                    "R": teacher_overlap(student, teacher),
                    "eps_g": normalized_gen_error(student, teacher, data.x_te, data.y_te),
                }
            )
            del student
    out = runs_root / "_sweeps" / "ts_lazy_rich"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    mat_dir = out / "matrices"
    mat_dir.mkdir(exist_ok=True)
    df.pivot(index="init_scale", columns="lr", values="d_h").to_csv(mat_dir / "d_h_matrix.csv")
    df.pivot(index="init_scale", columns="lr", values="R").to_csv(mat_dir / "R_matrix.csv")
    log.info("lazy/rich sweep done")


def run_all_ts(runs_root: Path, device: torch.device, log: logging.Logger, seed: int) -> None:
    run_dynamics(runs_root, device, log, seed)
    run_lr_trajectories(runs_root, device, log, seed)
    run_snr_phase(runs_root, device, log, seed)
    run_sample_multi_k(runs_root, device, log, seed)
    run_phase_map(runs_root, device, log, seed)
    run_lazy_rich(runs_root, device, log, seed)
