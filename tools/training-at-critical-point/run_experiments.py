#!/usr/bin/env python3
"""Run lightweight real MNIST experiments and write CSV logs for blog figures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


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
    train_full = datasets.MNIST("experiments/training-at-critical-point/data", train=True, download=True, transform=tf)
    test_full = datasets.MNIST("experiments/training-at-critical-point/data", train=False, download=True, transform=tf)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_full), size=min(n_train, len(train_full)), replace=False)
    train = DataLoader(Subset(train_full, idx.tolist()), batch_size=batch_size, shuffle=True)
    test = DataLoader(test_full, batch_size=512, shuffle=False)
    return train, test


@dataclass
class RunMetrics:
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float


def evaluate(model: nn.Module, loader: DataLoader) -> RunMetrics:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss_sum += ce(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return RunMetrics(
        train_loss=float("nan"),
        test_loss=loss_sum / total,
        train_acc=float("nan"),
        test_acc=correct / total,
    )


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> list[RunMetrics]:
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ce = nn.CrossEntropyLoss()
    history: list[RunMetrics] = []
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        tr = evaluate(model, train_loader)
        te = evaluate(model, test_loader)
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                tr_loss += ce(logits, y).item() * y.numel()
                tr_correct += (logits.argmax(1) == y).sum().item()
                tr_total += y.numel()
        history.append(
            RunMetrics(
                train_loss=tr_loss / tr_total,
                test_loss=te.test_loss,
                train_acc=tr_correct / tr_total,
                test_acc=te.test_acc,
            )
        )
    return history


def neural_collapse_order(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Within / between class variance ratio -> m_NC proxy."""
    labels = labels.cpu().numpy()
    feats = features.cpu().numpy()
    classes = np.unique(labels)
    global_mean = feats.mean(0)
    sw = 0.0
    sb = 0.0
    for c in classes:
        mask = labels == c
        cluster = feats[mask]
        mu = cluster.mean(0)
        sw += ((cluster - mu) ** 2).sum()
        sb += len(cluster) * ((mu - global_mean) ** 2).sum()
    sw /= max(len(labels), 1)
    sb /= max(len(classes), 1)
    return float(1.0 - sw / (sb + sw + 1e-8))


def hessian_top_eig(model: nn.Module, x: torch.Tensor, y: torch.Tensor, n_iter: int = 15) -> float:
    """Power iteration on Hessian of CE loss (small batch)."""
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
    v = torch.randn(dim, device=DEVICE)
    v = v / v.norm()
    for _ in range(n_iter):
        Hv = hvp(v)
        lam = torch.dot(v, Hv).item()
        v = Hv / (Hv.norm() + 1e-12)
    return float(lam)


def run_all(out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed()
    train_loader, test_loader = mnist_loaders()

    def need(name: str) -> bool:
        return force or not (out_dir / name).exists()

    # --- Phase diagram: width x lr ---
    widths = [32, 64, 128, 256, 512, 1024]
    lrs = [0.002, 0.005, 0.01, 0.02, 0.05]
    if need("phase_diagram.csv"):
        phase_rows = []
        for w in widths:
            for lr in lrs:
                model = MLPClassifier(w).to(DEVICE)
                hist = train_epochs(model, train_loader, test_loader, epochs=6, lr=lr)
                phase_rows.append(
                    {
                        "width": w,
                        "lr": lr,
                        "n_params": sum(p.numel() for p in model.parameters()),
                        "final_test_acc": hist[-1].test_acc,
                        "final_train_acc": hist[-1].train_acc,
                        "final_test_loss": hist[-1].test_loss,
                    }
                )
                print(f"phase w={w} lr={lr} acc={hist[-1].test_acc:.3f}")
        with (out_dir / "phase_diagram.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=phase_rows[0].keys())
            writer.writeheader()
            writer.writerows(phase_rows)

    # --- Double descent: width sweep ---
    if need("double_descent.csv"):
        dd_rows = []
        for w in widths:
            model = MLPClassifier(w).to(DEVICE)
            hist = train_epochs(model, train_loader, test_loader, epochs=8, lr=0.01)
            dd_rows.append(
                {
                    "width": w,
                    "n_params": sum(p.numel() for p in model.parameters()),
                    "train_loss": hist[-1].train_loss,
                    "test_loss": hist[-1].test_loss,
                    "train_acc": hist[-1].train_acc,
                    "test_acc": hist[-1].test_acc,
                }
            )
        with (out_dir / "double_descent.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dd_rows[0].keys())
            writer.writeheader()
            writer.writerows(dd_rows)

    # --- Edge of stability + order params time series ---
    if need("training_timeseries.csv"):
        lr = 0.05
        model = MLPClassifier(256).to(DEVICE)
        theta0 = torch.cat([p.detach().flatten() for p in model.parameters()])
        ts_rows = []
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
        ce = nn.CrossEntropyLoss()
        step = 0
        for epoch in range(12):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = ce(model(x), y)
                loss.backward()
                opt.step()
                if step % 25 == 0:
                    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
                    with torch.no_grad():
                        feats = model.penultimate(x[:64])
                        m_nc = neural_collapse_order(feats, y[:64])
                    lam = hessian_top_eig(model, x[:32], y[:32])
                    ts_rows.append(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "lambda_max": lam,
                            "chi": lr * lam / 2.0,
                            "theta_dist": (theta - theta0).norm().item() / (theta0.norm().item() + 1e-8),
                            "m_nc": m_nc,
                            "test_acc": evaluate(model, test_loader).test_acc,
                        }
                    )
                step += 1
        with (out_dir / "training_timeseries.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ts_rows[0].keys())
            writer.writeheader()
            writer.writerows(ts_rows)

    # --- Neural collapse feature snapshots (PCA-2D per epoch) ---
    if need("neural_collapse_snapshots.csv"):
        model = MLPClassifier(512).to(DEVICE)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        snap_rows = []
        for epoch in range(15):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                ce(model(x), y).backward()
                opt.step()
            xs, ys = next(iter(test_loader))
            xs, ys = xs.to(DEVICE), ys.to(DEVICE)
            with torch.no_grad():
                feats = model.penultimate(xs).cpu().numpy()
            labels = ys.cpu().numpy()
            pca = PCA(n_components=2, random_state=SEED)
            coords = pca.fit_transform(feats)
            for i in range(coords.shape[0]):
                snap_rows.append(
                    {"epoch": epoch, "x": coords[i, 0], "y": coords[i, 1], "label": int(labels[i])}
                )
        with (out_dir / "neural_collapse_snapshots.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=snap_rows[0].keys())
            writer.writeheader()
            writer.writerows(snap_rows)

    # --- Modular addition grokking ---
    if need("grokking.csv"):
        p = 97
        rng = np.random.default_rng(SEED)
        a = rng.integers(0, p, size=5000)
        b = rng.integers(0, p, size=5000)
        targets = (a + b) % p

        class ModAdd(nn.Module):
            def __init__(self, d: int = 128) -> None:
                super().__init__()
                self.emb = nn.Embedding(p, d)
                self.fc = nn.Linear(2 * d, p)

            def forward(self, ai: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
                return self.fc(torch.cat([self.emb(ai), self.emb(bi)], dim=-1))

        grok = ModAdd().to(DEVICE)
        opt = torch.optim.AdamW(grok.parameters(), lr=1e-3, weight_decay=1.0)
        ce = nn.CrossEntropyLoss()
        grok_rows = []
        ai = torch.tensor(a, device=DEVICE)
        bi = torch.tensor(b, device=DEVICE)
        ti = torch.tensor(targets, device=DEVICE)
        for step in range(3000):
            idx = rng.integers(0, len(a), size=256)
            logits = grok(ai[idx], bi[idx])
            loss = ce(logits, ti[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0:
                with torch.no_grad():
                    train_pred = grok(ai, bi).argmax(1)
                    train_acc = (train_pred == ti).float().mean().item()
                    te_a = rng.integers(0, p, size=1000)
                    te_b = rng.integers(0, p, size=1000)
                    te_t = (te_a + te_b) % p
                    test_acc = (
                        grok(
                            torch.tensor(te_a, device=DEVICE),
                            torch.tensor(te_b, device=DEVICE),
                        )
                        .argmax(1)
                        .eq(torch.tensor(te_t, device=DEVICE))
                        .float()
                        .mean()
                        .item()
                    )
                grok_rows.append({"step": step, "train_acc": train_acc, "test_acc": test_acc, "loss": loss.item()})
        with (out_dir / "grokking.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=grok_rows[0].keys())
            writer.writeheader()
            writer.writerows(grok_rows)

    meta = {"device": str(DEVICE), "seed": SEED, "dataset": "MNIST subset + mod-add p=97"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Logs in {out_dir}")


if __name__ == "__main__":
    run_all(Path("experiments/training-at-critical-point/outputs"))
