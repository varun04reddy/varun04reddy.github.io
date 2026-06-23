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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_run_dir(runs_root: Path, run_name: str, config: dict[str, Any]) -> Path:
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
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
    """Width × lr sweep for fig02. Train fast; one test eval per run (heatmap needs only final acc)."""
    widths = [32, 64, 128, 256, 512, 1024]
    lrs = [0.002, 0.005, 0.01, 0.02, 0.05]
    train_loader, test_loader = mnist_loaders()
    summary: list[dict] = []
    for w in widths:
        for lr in lrs:
            run_name = f"phase_w{w}_lr{lr:g}"
            cfg = {"experiment": "phase_diagram", "width": w, "lr": lr, "epochs": 6, "dataset": "MNIST"}
            run_dir = setup_run_dir(runs_root, run_name, cfg)
            model = MLPClassifier(w).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            ce = nn.CrossEntropyLoss()
            for epoch in range(cfg["epochs"]):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad(set_to_none=True)
                    ce(model(x), y).backward()
                    opt.step()
            final = full_train_eval(model, train_loader, test_loader, device)
            summary.append(
                {
                    "width": w,
                    "lr": lr,
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "final_test_acc": final.test_acc,
                    "final_train_acc": final.train_acc,
                    "final_test_loss": final.test_loss,
                }
            )
            log.info("phase w=%s lr=%s val_acc=%.3f", w, lr, final.test_acc)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return summary


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
        "log_every": 200,
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
    chi_every = 100
    cfg = {"experiment": "edge_of_stability", "width": 256, "lr": lr, "chi_log_every": chi_every, "epochs": 8}
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
            if step % chi_every == 0:
                theta = torch.cat([p.detach().flatten() for p in model.parameters()])
                with torch.no_grad():
                    feats = model.penultimate(x[:64])
                    m_nc = neural_collapse_order(feats, y[:64])
                lam = hessian_top_eig(model, x[:16], y[:16], device, n_iter=8)
                chi = lr * lam / 2.0
                val = evaluate(model, test_loader, device)
                logger.log(
                    {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        "val_loss": val.test_loss,
                        "val_acc": val.test_acc,
                        "lambda_max": lam,
                        "chi": chi,
                        "m_nc": m_nc,
                        "theta_dist": (theta - theta0).norm().item() / (theta0.norm().item() + 1e-8),
                        "learning_rate": lr,
                    }
                )
            step += 1
    logger.flush()
    log.info("edge-of-stability run finished (%s steps)", step)
    return run_dir


def run_neural_collapse_snaps(runs_root: Path, device: torch.device, log: logging.Logger) -> Path:
    cfg = {"experiment": "neural_collapse", "width": 512, "epochs": 15}
    run_dir = setup_run_dir(runs_root, "neural_collapse_mnist", cfg)
    logger = StepCsvLogger(run_dir)
    train_loader, test_loader = mnist_loaders()
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
            step += 1
        xs, ys = next(iter(test_loader))
        xs, ys = xs.to(device), ys.to(device)
        with torch.no_grad():
            feats = model.penultimate(xs).cpu().numpy()
            m_nc = neural_collapse_order(model.penultimate(xs), ys)
        labels = ys.cpu().numpy()
        pca = PCA(n_components=2, random_state=SEED)
        coords = pca.fit_transform(feats)
        for i in range(coords.shape[0]):
            snap_rows.append({"epoch": epoch, "x": coords[i, 0], "y": coords[i, 1], "label": int(labels[i])})
        logger.log({"step": step, "epoch": epoch, "m_nc": m_nc, "val_acc": evaluate(model, test_loader, device).test_acc})
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


def plot_run_figures(legacy_out: Path, assets_dir: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_figures import fig02_phase_diagram, fig04_edge_of_stability, fig05_neural_collapse, fig06_grokking

    assets_dir.mkdir(parents=True, exist_ok=True)
    fig02_phase_diagram(legacy_out, assets_dir)
    fig04_edge_of_stability(legacy_out, assets_dir)
    fig05_neural_collapse(legacy_out, assets_dir)
    fig06_grokking(legacy_out, assets_dir)


def make_grokking_gif(legacy_out: Path, assets_dir: Path) -> Path:
    """Animate train vs test accuracy during grokking (phase-transition style)."""
    path = legacy_out / "grokking.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    gif_path = assets_dir / "phase-transition.gif"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from style import PALETTE, apply_style

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    (line_tr,) = ax.plot([], [], color=PALETTE["accent"], lw=2, label="train acc")
    (line_te,) = ax.plot([], [], color=PALETTE["test"], lw=2, label="test acc")
    ax.set_xscale("log")
    ax.set_xlim(max(df["step"].min(), 1), df["step"].max())
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking phase transition (mod 97)")
    ax.legend(loc="lower right")

    frame_idx = np.linspace(0, len(df) - 1, num=min(120, len(df)), dtype=int)

    def update(i: int):
        frame = frame_idx[i]
        sub = df.iloc[: frame + 1]
        steps = np.maximum(sub["step"].values, 1)
        line_tr.set_data(steps, sub["train_acc"].values)
        line_te.set_data(steps, sub["test_acc"].values)
        return line_tr, line_te

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=80, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return gif_path


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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_figures import fig06_grokking

    assets_dir.mkdir(parents=True, exist_ok=True)
    fig06_grokking(legacy_out, assets_dir)
    gif_path = make_grokking_gif(legacy_out, assets_dir)
    bundle_log.info("GIF saved: %s", gif_path)


def run_all_experiments(runs_root: Path, legacy_out: Path, assets_dir: Path) -> None:
    device = get_device()
    set_seed(SEED)
    bundle_log = setup_logger(setup_run_dir(runs_root, "_bundle", {"experiment": "bundle"}))
    bundle_log.info("Device: %s", device)

    phase_summary = run_phase_diagram(runs_root, device, bundle_log)
    grok_dir = run_grokking(runs_root, device, bundle_log)
    eos_dir = run_edge_of_stability(runs_root, device, bundle_log)
    nc_dir = run_neural_collapse_snaps(runs_root, device, bundle_log)

    write_legacy_outputs(runs_root, legacy_out, phase_summary, grok_dir, eos_dir, nc_dir)
    plot_run_figures(legacy_out, assets_dir)
    gif_path = make_grokking_gif(legacy_out, assets_dir)
    bundle_log.info("GIF saved: %s", gif_path)


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
