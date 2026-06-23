#!/usr/bin/env python3
"""GPU training for critical-point blog: step-level CSV logs and bundled sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments/training-at-critical-point/runs"
LEGACY_OUT = REPO_ROOT / "experiments/training-at-critical-point/outputs"
ASSETS_DIR = REPO_ROOT / "assets/img/blog/critical-point"
DATA_ROOT = REPO_ROOT / "experiments/training-at-critical-point/data"
SEED = 42

from teacher_student import (
    TSConfig,
    feature_drift,
    init_student,
    init_teacher,
    make_dataset,
    mse,
    mse_hessian_top_eig,
    normalized_gen_error,
    teacher_overlap,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_run_dir(runs_root: Path, run_name: str, config: dict[str, Any], *, fresh: bool = True) -> Path:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    if fresh and (run_dir / "metrics.csv").exists():
        (run_dir / "metrics.csv").unlink()
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    log = logging.getLogger(run_dir.name)
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "logs" / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)
    return log


class StepCsvLogger:
    """Append step rows to metrics.csv."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.csv"
        self.fieldnames: list[str] | None = None
        self._rows: list[dict[str, Any]] = []

    def log(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        elif set(row.keys()) != set(self.fieldnames):
            for k in row:
                if k not in self.fieldnames:
                    self.fieldnames.append(k)

    def flush(self) -> None:
        if not self._rows or self.fieldnames is None:
            return
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(self._rows)
        self._rows.clear()


class MLPClassifier(nn.Module):
    def __init__(self, width: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 28 * 28
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        h = x.flatten(1)
        for layer in self.net[:-1]:
            h = layer(h)
        return h


def mnist_loaders(batch_size: int = 128, n_train: int = 8000) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(str(DATA_ROOT), train=True, download=True, transform=tf)
    test_full = datasets.MNIST(str(DATA_ROOT), train=False, download=True, transform=tf)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_full), size=min(n_train, len(train_full)), replace=False)
    train = DataLoader(Subset(train_full, idx.tolist()), batch_size=batch_size, shuffle=True)
    test = DataLoader(test_full, batch_size=512, shuffle=False)
    return train, test


@dataclass
class EvalSnapshot:
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalSnapshot:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += ce(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return EvalSnapshot(float("nan"), loss_sum / total, float("nan"), correct / total)


def full_train_eval(
    model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device
) -> EvalSnapshot:
    ce = nn.CrossEntropyLoss()
    model.eval()
    tr_loss = tr_correct = tr_total = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tr_loss += ce(logits, y).item() * y.numel()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.numel()
    te = evaluate(model, test_loader, device)
    return EvalSnapshot(tr_loss / tr_total, te.test_loss, tr_correct / tr_total, te.test_acc)


def neural_collapse_order(features: torch.Tensor, labels: torch.Tensor) -> float:
    labels_np = labels.cpu().numpy()
    feats = features.cpu().numpy()
    classes = np.unique(labels_np)
    global_mean = feats.mean(0)
    sw = sb = 0.0
    for c in classes:
        mask = labels_np == c
        cluster = feats[mask]
        mu = cluster.mean(0)
        sw += ((cluster - mu) ** 2).sum()
        sb += len(cluster) * ((mu - global_mean) ** 2).sum()
    sw /= max(len(labels_np), 1)
    sb /= max(len(classes), 1)
    return float(1.0 - sw / (sb + sw + 1e-8))


def hessian_top_eig(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, n_iter: int = 12
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    ce = nn.CrossEntropyLoss()

    def hvp(v_flat: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        dot = (flat_grad * v_flat).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv])

    dim = sum(p.numel() for p in params)
    v = torch.randn(dim, device=device)
    v = v / v.norm()
    lam = 0.0
    for _ in range(n_iter):
        Hv = hvp(v)
        lam = torch.dot(v, Hv).item()
        v = Hv / (Hv.norm() + 1e-12)
    return float(lam)


def compute_d_h(model: nn.Module, xs: torch.Tensor, h0: torch.Tensor) -> float:
    with torch.no_grad():
        h = model.penultimate(xs)
        return float(((h - h0) ** 2).mean().item())


def compute_d_theta(model: nn.Module, theta0: torch.Tensor) -> float:
    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
    return float((theta - theta0).norm().item() / (theta0.norm().item() + 1e-8))


def run_ts_dynamics(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    """Canonical teacher–student run: log overlap R, gen error, chi along trajectory."""
    cfg = TSConfig(input_dim=50, teacher_width=4, n_train=3000, noise_std=0.05)
    run_cfg = {
        "experiment": "ts_dynamics",
        "student_width": 16,
        "lr": 0.05,
        "epochs": 800,
        "batch_size": 256,
        **cfg.__dict__,
    }
    run_dir = setup_run_dir(runs_root, "ts_dynamics", run_cfg)
    logger = StepCsvLogger(run_dir)
    teacher = init_teacher(cfg, device, SEED)
    student = init_student(cfg, run_cfg["student_width"], device, SEED)
    x_tr, y_tr, x_te, y_te = make_dataset(cfg, teacher, device, SEED)
    x_probe = x_te[:512]
    student.eval()
    with torch.no_grad():
        h0 = student.hidden(x_probe).clone()
    opt = torch.optim.SGD(student.parameters(), lr=run_cfg["lr"], momentum=0.0)
    mse_fn = nn.MSELoss()
    bs = int(run_cfg["batch_size"])
    n = len(x_tr)
    step = 0
    chi_every = 20
    for epoch in range(int(run_cfg["epochs"])):
        perm = torch.randperm(n, device=device)
        student.train()
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            x, y = x_tr[idx], y_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss = mse_fn(student(x), y)
            loss.backward()
            opt.step()
            if step % 5 == 0 or step % chi_every == 0:
                student.eval()
                row: dict[str, Any] = {
                    "step": step,
                    "epoch": epoch,
                    "train_mse": loss.item(),
                    "R": teacher_overlap(student, teacher),
                    "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                    "d_h": feature_drift(student, x_probe, h0),
                    "learning_rate": run_cfg["lr"],
                }
                if step % chi_every == 0:
                    lam = mse_hessian_top_eig(student, x[:64], y[:64], n_iter=6)
                    row["lambda_max"] = lam
                    row["chi"] = run_cfg["lr"] * lam / 2.0
                logger.log(row)
                student.train()
            step += 1
    logger.flush()
    log.info("ts dynamics finished R=%.3f eps_g=%.4f", teacher_overlap(student, teacher), row["eps_g"])
    return run_dir


def run_ts_phase_map(runs_root: Path, device: torch.device, log: logging.Logger) -> list[dict]:
    """Student width × lr phase map on teacher–student task."""
    cfg = TSConfig(input_dim=50, teacher_width=4, n_train=2000, noise_std=0.05)
    widths = [2, 4, 8, 16, 32, 64]
    lrs = [0.002, 0.005, 0.01, 0.03, 0.1, 0.3]
    teacher = init_teacher(cfg, device, SEED)
    x_tr, y_tr, x_te, y_te = make_dataset(cfg, teacher, device, SEED)
    summary: list[dict] = []
    for k in widths:
        for lr in lrs:
            student = init_student(cfg, k, device, SEED + k * 100 + int(lr * 1e4))
            opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
            mse_fn = nn.MSELoss()
            for _ in range(600):
                idx = torch.randint(0, len(x_tr), (256,), device=device)
                opt.zero_grad(set_to_none=True)
                loss = mse_fn(student(x_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
            student.eval()
            summary.append(
                {
                    "student_width": k,
                    "lr": lr,
                    "teacher_width": cfg.teacher_width,
                    "R": teacher_overlap(student, teacher),
                    "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                    "train_mse": mse(student(x_tr), y_tr),
                    "alpha": cfg.n_train / cfg.input_dim,
                }
            )
            log.info("ts phase K=%s lr=%s eps_g=%.4f R=%.3f", k, lr, summary[-1]["eps_g"], summary[-1]["R"])
            del student
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return summary


def run_ts_width_sweep(runs_root: Path, device: torch.device, log: logging.Logger) -> None:
    """Student width sweep → double-descent / interpolation boundary at K ≈ K*."""
    cfg = TSConfig(input_dim=50, teacher_width=4, n_train=1500, noise_std=0.05)
    widths = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96]
    lr = 0.05
    sweep_root = runs_root / "_sweeps" / "ts_width"
    sweep_root.mkdir(parents=True, exist_ok=True)
    teacher = init_teacher(cfg, device, SEED)
    x_tr, y_tr, x_te, y_te = make_dataset(cfg, teacher, device, SEED)
    summary: list[dict] = []
    for k in widths:
        run_dir = setup_run_dir(sweep_root, f"k{k}", {"student_width": k, "lr": lr}, fresh=True)
        logger = StepCsvLogger(run_dir)
        student = init_student(cfg, k, device, SEED + k)
        opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
        mse_fn = nn.MSELoss()
        step = 0
        for epoch in range(250):
            perm = torch.randperm(len(x_tr), device=device)
            for i in range(0, len(x_tr), 256):
                idx = perm[i : i + 256]
                opt.zero_grad(set_to_none=True)
                loss = mse_fn(student(x_tr[idx]), y_tr[idx])
                loss.backward()
                opt.step()
                if step % 10 == 0:
                    student.eval()
                    logger.log(
                        {
                            "step": step,
                            "student_width": k,
                            "train_mse": loss.item(),
                            "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                            "R": teacher_overlap(student, teacher),
                        }
                    )
                    student.train()
                step += 1
        logger.flush()
        student.eval()
        summary.append(
            {
                "student_width": k,
                "n_params": sum(p.numel() for p in student.parameters()),
                "train_mse": mse(student(x_tr), y_tr),
                "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                "R": teacher_overlap(student, teacher),
            }
        )
        log.info("ts width K=%s eps_g=%.4f R=%.3f", k, summary[-1]["eps_g"], summary[-1]["R"])
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(sweep_root / "summary.csv", index=False)
    frames = [pd.read_csv(p / "metrics.csv") for p in sorted(sweep_root.glob("k*")) if (p / "metrics.csv").exists()]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(sweep_root / "aggregated.csv", index=False)


def run_ts_sample_sweep(runs_root: Path, device: torch.device, log: logging.Logger) -> list[dict]:
    """Sample complexity: gen error vs α = n/d (fixed student width)."""
    base = TSConfig(input_dim=50, teacher_width=4, noise_std=0.05)
    n_train_list = [100, 200, 400, 800, 1500, 3000, 6000, 12000]
    student_k = 16
    lr = 0.05
    teacher = init_teacher(base, device, SEED)
    summary: list[dict] = []
    for n_tr in n_train_list:
        cfg = TSConfig(input_dim=base.input_dim, teacher_width=base.teacher_width, n_train=n_tr, noise_std=base.noise_std)
        x_tr, y_tr, x_te, y_te = make_dataset(cfg, teacher, device, SEED + n_tr)
        student = init_student(cfg, student_k, device, SEED + n_tr)
        opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
        mse_fn = nn.MSELoss()
        steps = max(400, n_tr // 2)
        for _ in range(steps):
            idx = torch.randint(0, len(x_tr), (min(256, len(x_tr)),), device=device)
            opt.zero_grad(set_to_none=True)
            mse_fn(student(x_tr[idx]), y_tr[idx]).backward()
            opt.step()
        student.eval()
        summary.append(
            {
                "n_train": n_tr,
                "alpha": n_tr / cfg.input_dim,
                "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                "R": teacher_overlap(student, teacher),
                "train_mse": mse(student(x_tr), y_tr),
            }
        )
        log.info("ts sample n=%s alpha=%.1f eps_g=%.4f", n_tr, summary[-1]["alpha"], summary[-1]["eps_g"])
        del student
    out_dir = runs_root / "_sweeps" / "ts_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    return summary


def run_ts_eos(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    """Edge of stability on teacher–student (SGD, high lr, MSE Hessian)."""
    cfg = TSConfig(input_dim=50, teacher_width=4, n_train=2000, noise_std=0.05)
    lr = 0.8
    run_cfg = {"experiment": "ts_eos", "student_width": 16, "lr": lr, "epochs": 500}
    run_dir = setup_run_dir(runs_root, "ts_eos", run_cfg)
    logger = StepCsvLogger(run_dir)
    teacher = init_teacher(cfg, device, SEED)
    student = init_student(cfg, 16, device, SEED)
    x_tr, y_tr, x_te, y_te = make_dataset(cfg, teacher, device, SEED)
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
    mse_fn = nn.MSELoss()
    step = 0
    chi_every = 10
    for epoch in range(int(run_cfg["epochs"])):
        perm = torch.randperm(len(x_tr), device=device)
        for i in range(0, len(x_tr), 256):
            idx = perm[i : i + 256]
            x, y = x_tr[idx], y_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss = mse_fn(student(x), y)
            loss.backward()
            opt.step()
            if step % 3 == 0 or step % chi_every == 0:
                row: dict[str, Any] = {"step": step, "train_mse": loss.item(), "learning_rate": lr}
                if step % chi_every == 0:
                    student.eval()
                    lam = mse_hessian_top_eig(student, x[:64], y[:64], n_iter=8)
                    row.update(
                        {
                            "eps_g": normalized_gen_error(student, teacher, x_te, y_te),
                            "R": teacher_overlap(student, teacher),
                            "lambda_max": lam,
                            "chi": lr * lam / 2.0,
                        }
                    )
                    student.train()
                logger.log(row)
            step += 1
    logger.flush()
    log.info("ts eos finished (%s steps)", step)
    return run_dir


def write_ts_matrices(phase_summary: list[dict], runs_root: Path) -> None:
    if not phase_summary:
        return
    mat_dir = runs_root / "_sweeps" / "ts_phase" / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(phase_summary)
    for col, fname in [("eps_g", "eps_g_matrix.csv"), ("R", "overlap_matrix.csv")]:
        pivot = df.pivot(index="student_width", columns="lr", values=col)
        pivot.to_csv(mat_dir / fname)
    df.to_csv(runs_root / "_sweeps" / "ts_phase" / "summary.csv", index=False)


def run_dashboard(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    """Canonical MNIST run: dense step logging for order-parameter dashboard."""
    cfg = {"experiment": "dashboard", "width": 256, "lr": 0.01, "epochs": 10, "log_every": 1}
    run_dir = setup_run_dir(runs_root, "dashboard_mnist", cfg)
    logger = StepCsvLogger(run_dir)
    train_loader, test_loader = mnist_loaders()
    model = MLPClassifier(256).to(device)
    theta0 = torch.cat([p.detach().flatten() for p in model.parameters()])
    xs_probe, _ = next(iter(test_loader))
    xs_probe = xs_probe[:256].to(device)
    model.eval()
    with torch.no_grad():
        h0 = model.penultimate(xs_probe).clone()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    step = 0
    chi_every = 20
    val_every = 10
    for epoch in range(int(cfg["epochs"])):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            grad_norm = torch.sqrt(
                sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)
            ).item()
            opt.step()
            row: dict[str, Any] = {
                "step": step,
                "epoch": epoch,
                "train_loss": loss.item(),
                "grad_norm": grad_norm,
                "learning_rate": cfg["lr"],
            }
            if step % val_every == 0:
                val = evaluate(model, test_loader, device)
                with torch.no_grad():
                    feats = model.penultimate(x[:64])
                    m_nc = neural_collapse_order(feats, y[:64])
                row.update(
                    {
                        "val_loss": val.test_loss,
                        "val_acc": val.test_acc,
                        "m_nc": m_nc,
                        "d_theta": compute_d_theta(model, theta0),
                        "d_h": compute_d_h(model, xs_probe, h0),
                    }
                )
            if step % chi_every == 0:
                lam = hessian_top_eig(model, x[:16], y[:16], device, n_iter=6)
                row["lambda_max"] = lam
                row["chi"] = cfg["lr"] * lam / 2.0
            if step % int(cfg["log_every"]) == 0 or step % val_every == 0 or step % chi_every == 0:
                logger.log(row)
            step += 1
    logger.flush()
    log.info("dashboard run finished (%s steps)", step)
    return run_dir


def try_make_default_figures(run_dir: Path) -> None:
    skill_root = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
    rp = skill_root / "research-plotting" / "scripts"
    if not (rp / "research_plotting.py").exists():
        return
    sys.path.insert(0, str(rp))
    try:
        from research_plotting import make_default_figures

        make_default_figures(run_dir)
    except Exception as exc:
        logging.getLogger(run_dir.name).warning("make_default_figures skipped: %s", exc)


def run_phase_diagram(runs_root: Path, device: torch.device, log: logging.Logger) -> list[dict]:
    """Width × lr sweep; record final metrics + mid-training feature drift d_h."""
    widths = [32, 64, 128, 256, 512, 1024]
    lrs = [0.002, 0.005, 0.01, 0.02, 0.05]
    train_loader, test_loader = mnist_loaders()
    xs_probe, _ = next(iter(test_loader))
    xs_probe = xs_probe[:256].to(device)
    summary: list[dict] = []
    for w in widths:
        for lr in lrs:
            cfg = {"experiment": "phase_diagram", "width": w, "lr": lr, "epochs": 6, "dataset": "MNIST"}
            run_dir = setup_run_dir(runs_root, f"phase_w{w}_lr{lr:g}", cfg, fresh=False)
            model = MLPClassifier(w).to(device)
            model.eval()
            with torch.no_grad():
                h0 = model.penultimate(xs_probe).clone()
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            ce = nn.CrossEntropyLoss()
            d_h_mid = float("nan")
            for epoch in range(cfg["epochs"]):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    ce(model(x), y).backward()
                    opt.step()
                if epoch == 2:
                    d_h_mid = compute_d_h(model, xs_probe, h0)
            final = full_train_eval(model, train_loader, test_loader, device)
            summary.append(
                {
                    "width": w,
                    "lr": lr,
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "final_test_acc": final.test_acc,
                    "final_train_acc": final.train_acc,
                    "final_test_loss": final.test_loss,
                    "d_h_epoch3": d_h_mid,
                }
            )
            log.info("phase w=%s lr=%s val_acc=%.3f d_h=%.4f", w, lr, final.test_acc, d_h_mid)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return summary


def run_width_sweep(runs_root: Path, device: torch.device, log: logging.Logger) -> None:
    """Width sweep at fixed lr with dense val_loss logging for sweep curves."""
    widths = [16, 32, 64, 128, 256, 512, 1024, 2048]
    lr = 0.01
    sweep_root = runs_root / "_sweeps" / "width_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = mnist_loaders()
    summary: list[dict] = []
    for w in widths:
        cfg = {"experiment": "width_sweep", "width": w, "lr": lr, "epochs": 12}
        run_dir = setup_run_dir(runs_root / "_sweeps" / "width_sweep", f"w{w}", cfg)
        logger = StepCsvLogger(run_dir)
        model = MLPClassifier(w).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        step = 0
        for epoch in range(int(cfg["epochs"])):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = ce(model(x), y)
                loss.backward()
                opt.step()
                if step % 5 == 0:
                    val = evaluate(model, test_loader, device)
                    logger.log(
                        {
                            "step": step,
                            "epoch": epoch,
                            "width": w,
                            "train_loss": loss.item(),
                            "val_loss": val.test_loss,
                            "val_acc": val.test_acc,
                            "learning_rate": lr,
                        }
                    )
                step += 1
        logger.flush()
        final = full_train_eval(model, train_loader, test_loader, device)
        summary.append(
            {
                "width": w,
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_loss": final.train_loss,
                "test_loss": final.test_loss,
                "train_acc": final.train_acc,
                "test_acc": final.test_acc,
            }
        )
        log.info("width w=%s test_acc=%.3f", w, final.test_acc)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(sweep_root / "summary.csv", index=False)
    frames = []
    for run_dir in sorted(sweep_root.glob("w*")):
        m = run_dir / "metrics.csv"
        if m.exists():
            frames.append(pd.read_csv(m))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(sweep_root / "aggregated.csv", index=False)


def write_sweep_matrices(phase_summary: list[dict], runs_root: Path) -> None:
    """Save heatmap CSVs under runs/_sweeps/phase_w_lr/matrices/."""
    if not phase_summary:
        return
    mat_dir = runs_root / "_sweeps" / "phase_w_lr" / "matrices"
    mat_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(phase_summary)
    for col, fname in [
        ("final_test_acc", "test_acc_matrix.csv"),
        ("final_test_loss", "test_loss_matrix.csv"),
        ("d_h_epoch3", "d_h_matrix.csv"),
    ]:
        if col not in df.columns:
            continue
        pivot = df.pivot(index="width", columns="lr", values=col)
        pivot.to_csv(mat_dir / fname)
    df.to_csv(runs_root / "_sweeps" / "phase_w_lr" / "summary.csv", index=False)


def run_grokking(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    """Modular addition with held-out pairs; one-hot MLP (standard grokking setup)."""
    p = 97
    a_all, b_all = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a_all = a_all.ravel()
    b_all = b_all.ravel()
    t_all = (a_all + b_all) % p

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(a_all))
    n_train = int(0.4 * len(a_all))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    class ModAddMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2 * p, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, p),
            )

        def forward(self, ai: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
            xa = nn.functional.one_hot(ai, p).float()
            xb = nn.functional.one_hot(bi, p).float()
            return self.net(torch.cat([xa, xb], dim=-1))

    cfg = {
        "experiment": "grokking",
        "modulus": p,
        "steps": 50000,
        "weight_decay": 1.0,
        "lr": 1e-3,
        "log_every": 50,
        "train_fraction": 0.4,
    }
    run_dir = setup_run_dir(runs_root, "grokking_mod97", cfg)
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    logger = StepCsvLogger(run_dir)
    grok = ModAddMLP().to(device)
    opt = torch.optim.AdamW(grok.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()

    ai_tr = torch.tensor(a_all[train_idx], device=device)
    bi_tr = torch.tensor(b_all[train_idx], device=device)
    ti_tr = torch.tensor(t_all[train_idx], device=device)
    ai_te = torch.tensor(a_all[test_idx], device=device)
    bi_te = torch.tensor(b_all[test_idx], device=device)
    ti_te = torch.tensor(t_all[test_idx], device=device)

    n_steps = int(cfg["steps"])
    log_every = int(cfg["log_every"])
    train_acc = test_acc = 0.0
    for step in range(n_steps):
        idx = rng.integers(0, len(train_idx), size=512)
        logits = grok(ai_tr[idx], bi_tr[idx])
        loss = ce(logits, ti_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                train_acc = (grok(ai_tr, bi_tr).argmax(1) == ti_tr).float().mean().item()
                test_acc = (grok(ai_te, bi_te).argmax(1) == ti_te).float().mean().item()
            logger.log(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "learning_rate": cfg["lr"],
                }
            )
    logger.flush()
    log.info("grokking done final train_acc=%.3f test_acc=%.3f", train_acc, test_acc)
    return run_dir


def run_edge_of_stability(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    lr = 0.05
    chi_every = 15
    cfg = {"experiment": "edge_of_stability", "width": 256, "lr": lr, "chi_log_every": chi_every, "epochs": 10}
    run_dir = setup_run_dir(runs_root, "eos_mnist_mlp", cfg)
    logger = StepCsvLogger(run_dir)
    train_loader, test_loader = mnist_loaders()
    model = MLPClassifier(256).to(device)
    theta0 = torch.cat([p.detach().flatten() for p in model.parameters()])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    ce = nn.CrossEntropyLoss()
    step = 0
    for epoch in range(int(cfg["epochs"])):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            if step % 5 == 0 or step % chi_every == 0:
                row: dict[str, Any] = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                }
                if step % chi_every == 0:
                    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
                    with torch.no_grad():
                        feats = model.penultimate(x[:64])
                        m_nc = neural_collapse_order(feats, y[:64])
                    lam = hessian_top_eig(model, x[:16], y[:16], device, n_iter=6)
                    val = evaluate(model, test_loader, device)
                    row.update(
                        {
                            "val_loss": val.test_loss,
                            "val_acc": val.test_acc,
                            "lambda_max": lam,
                            "chi": lr * lam / 2.0,
                            "m_nc": m_nc,
                            "theta_dist": (theta - theta0).norm().item() / (theta0.norm().item() + 1e-8),
                        }
                    )
                logger.log(row)
            step += 1
    logger.flush()
    log.info("edge-of-stability run finished (%s steps)", step)
    return run_dir


def run_neural_collapse_snaps(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    cfg = {"experiment": "neural_collapse", "width": 512, "epochs": 15}
    run_dir = setup_run_dir(runs_root, "neural_collapse_mnist", cfg)
    logger = StepCsvLogger(run_dir)
    train_loader, test_loader = mnist_loaders()
    xs_nc, ys_nc = next(iter(test_loader))
    xs_nc, ys_nc = xs_nc[:512].to(device), ys_nc[:512].to(device)
    model = MLPClassifier(512).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    snap_rows: list[dict] = []
    step = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            ce(model(x), y).backward()
            opt.step()
            if step % 10 == 0:
                with torch.no_grad():
                    m_nc = neural_collapse_order(model.penultimate(xs_nc), ys_nc)
                row: dict[str, Any] = {"step": step, "epoch": epoch, "m_nc": m_nc}
                if step % 200 == 0:
                    row["val_acc"] = evaluate(model, test_loader, device).test_acc
                logger.log(row)
            step += 1
        with torch.no_grad():
            feats = model.penultimate(xs_nc).cpu().numpy()
            m_nc = neural_collapse_order(model.penultimate(xs_nc), ys_nc)
        labels = ys_nc.cpu().numpy()
        pca = PCA(n_components=2, random_state=SEED)
        coords = pca.fit_transform(feats)
        for i in range(coords.shape[0]):
            snap_rows.append({"epoch": epoch, "x": coords[i, 0], "y": coords[i, 1], "label": int(labels[i])})
        logger.log({"step": step, "epoch": epoch, "m_nc": m_nc})
    logger.flush()
    snap_path = run_dir / "neural_collapse_snapshots.csv"
    with snap_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "x", "y", "label"])
        w.writeheader()
        w.writerows(snap_rows)
    log.info("neural collapse snapshots: %s rows", len(snap_rows))
    return run_dir


def grokking_metrics_to_legacy(g: pd.DataFrame) -> pd.DataFrame:
    """Normalize grokking metrics; repair column drift when val_acc holds test accuracy."""
    if "test_acc" not in g.columns and "val_acc" in g.columns:
        g = g.copy()
        g["test_acc"] = g["val_acc"]
    elif "val_acc" in g.columns and (g["test_acc"] < 0.1).any() and (g["val_acc"] > 0.5).any():
        g = g.copy()
        bad = g["test_acc"] < 0.1
        g.loc[bad, "test_acc"] = g.loc[bad, "val_acc"]
    out = g[["step", "train_acc", "test_acc"]].copy()
    if "train_loss" in g.columns:
        out["loss"] = g["train_loss"]
    return out


def write_legacy_outputs(
    runs_root: Path,
    legacy_out: Path,
    phase_summary: list[dict],
    grok_dir: Path,
    eos_dir: Path,
    nc_dir: Path,
) -> None:
    legacy_out.mkdir(parents=True, exist_ok=True)
    if phase_summary:
        pd.DataFrame(phase_summary).to_csv(legacy_out / "phase_diagram.csv", index=False)
    grok_metrics = grok_dir / "metrics.csv"
    if grok_metrics.exists():
        grokking_metrics_to_legacy(pd.read_csv(grok_metrics)).to_csv(legacy_out / "grokking.csv", index=False)
    eos_metrics = eos_dir / "metrics.csv"
    if eos_metrics.exists():
        ts = pd.read_csv(eos_metrics)
        rename = {"train_loss": "loss", "val_acc": "test_acc"}
        cols = {c: rename.get(c, c) for c in ts.columns}
        ts = ts.rename(columns=cols)
        keep = [c for c in ["step", "loss", "lambda_max", "chi", "theta_dist", "m_nc", "test_acc"] if c in ts.columns]
        ts[keep].to_csv(legacy_out / "training_timeseries.csv", index=False)
    snap = nc_dir / "neural_collapse_snapshots.csv"
    if snap.exists():
        shutil.copy(snap, legacy_out / "neural_collapse_snapshots.csv")
    meta = {"runs_root": str(runs_root), "legacy_out": str(legacy_out), "seed": SEED}
    (legacy_out / "meta.json").write_text(json.dumps(meta, indent=2))


def plot_run_figures(assets_dir: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from publish_blog_figures import publish_all

    publish_all(assets_dir)


def run_all_experiments(runs_root: Path, legacy_out: Path, assets_dir: Path) -> None:
    device = get_device()
    set_seed(SEED)
    bundle_log = setup_logger(setup_run_dir(runs_root, "_bundle", {"experiment": "theory_bundle"}, fresh=False))
    bundle_log.info("Device: %s", device)

    run_ts_dynamics(runs_root, device, bundle_log)
    ts_phase = run_ts_phase_map(runs_root, device, bundle_log)
    write_ts_matrices(ts_phase, runs_root)
    run_ts_width_sweep(runs_root, device, bundle_log)
    run_ts_sample_sweep(runs_root, device, bundle_log)
    eos_dir = run_ts_eos(runs_root, device, bundle_log)
    grok_dir = run_grokking(runs_root, device, bundle_log)

    write_legacy_outputs(runs_root, legacy_out, ts_phase, grok_dir, eos_dir, Path())
    plot_run_figures(assets_dir)
    bundle_log.info("Theory figures published to %s", assets_dir)


def run_grokking_only(runs_root: Path, legacy_out: Path, assets_dir: Path) -> None:
    device = get_device()
    set_seed(SEED)
    bundle_log = setup_logger(setup_run_dir(runs_root, "_grokking_only", {"experiment": "grokking_only"}))
    bundle_log.info("Device: %s", device)
    grok_dir = run_grokking(runs_root, device, bundle_log)
    legacy_out.mkdir(parents=True, exist_ok=True)
    grok_metrics = grok_dir / "metrics.csv"
    if grok_metrics.exists():
        grokking_metrics_to_legacy(pd.read_csv(grok_metrics)).to_csv(legacy_out / "grokking.csv", index=False)
    plot_run_figures(assets_dir)
    bundle_log.info("Grokking figures published")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Critical-point blog GPU training bundle")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--legacy-out", type=Path, default=LEGACY_OUT)
    p.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    p.add_argument("--all", action="store_true", help="Run full experiment bundle")
    p.add_argument("--grokking-only", action="store_true", help="Re-run grokking + fig06 + GIF only")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global SEED
    SEED = args.seed
    if args.grokking_only:
        run_grokking_only(args.runs_root, args.legacy_out, args.assets_dir)
        return
    if not args.all:
        print("Specify --all to run the bundled sweeps (phase, grokking, EOS, NC).")
        sys.exit(0)
    run_all_experiments(args.runs_root, args.legacy_out, args.assets_dir)


if __name__ == "__main__":
    main()
