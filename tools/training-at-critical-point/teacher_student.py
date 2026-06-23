"""Teacher–student experiments: hidden-manifold inputs, order parameters, lazy/rich sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TSConfig:
    input_dim: int = 100
    manifold_dim: int = 15
    teacher_width: int = 8
    noise_std: float = 0.05
    n_train: int = 4000
    n_test: int = 10000
    init_scale: float = 1.0


@dataclass
class TSData:
    x_tr: torch.Tensor
    y_tr: torch.Tensor
    x_te: torch.Tensor
    y_te: torch.Tensor
    projector: torch.Tensor | None = None  # d × m orthonormal, None = full R^d


class TwoLayerReLU(nn.Module):
    """f(x) = (1/sqrt(K)) * w2 @ ReLU(W1 x + b)."""

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(input_dim, width, bias=True)
        self.fc2 = nn.Linear(width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x)) / (self.width**0.5)
        return self.fc2(h).squeeze(-1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc1(x)) / (self.width**0.5)

    def first_layer_weights(self) -> torch.Tensor:
        return self.fc1.weight.detach()


def make_projector(input_dim: int, manifold_dim: int, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    a = torch.randn(input_dim, manifold_dim, device=device)
    q, _ = torch.linalg.qr(a)
    return q


def sample_inputs(n: int, cfg: TSConfig, device: torch.device, projector: torch.Tensor | None, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if projector is None:
        if device.type == "cpu":
            return torch.randn(n, cfg.input_dim, device=device, generator=gen)
        return torch.randn(n, cfg.input_dim, device=device)
    m = projector.shape[1]
    if device.type == "cpu":
        z = torch.randn(n, m, device=device, generator=gen)
    else:
        torch.manual_seed(seed)
        z = torch.randn(n, m, device=device)
    return z @ projector.T


def init_teacher(cfg: TSConfig, device: torch.device, seed: int) -> TwoLayerReLU:
    torch.manual_seed(seed)
    teacher = TwoLayerReLU(cfg.input_dim, cfg.teacher_width).to(device)
    with torch.no_grad():
        nn.init.normal_(teacher.fc1.weight, std=1.0)
        nn.init.normal_(teacher.fc1.bias, std=0.1)
        nn.init.normal_(teacher.fc2.weight, std=1.0)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


def init_student(
    cfg: TSConfig, student_width: int, device: torch.device, seed: int, *, init_scale: float | None = None
) -> TwoLayerReLU:
    scale = cfg.init_scale if init_scale is None else init_scale
    torch.manual_seed(seed + 10_000)
    student = TwoLayerReLU(cfg.input_dim, student_width).to(device)
    with torch.no_grad():
        nn.init.normal_(student.fc1.weight, std=0.5 * scale)
        nn.init.normal_(student.fc1.bias, std=0.05 * scale)
        nn.init.normal_(student.fc2.weight, std=0.5 * scale)
    return student


@torch.no_grad()
def add_label_noise(y: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return y
    return y + noise_std * torch.randn_like(y)


def make_dataset(
    cfg: TSConfig,
    teacher: TwoLayerReLU,
    device: torch.device,
    seed: int,
    *,
    on_manifold: bool = True,
) -> TSData:
    projector = make_projector(cfg.input_dim, cfg.manifold_dim, device, seed) if on_manifold else None
    x_tr = sample_inputs(cfg.n_train, cfg, device, projector, seed)
    x_te = sample_inputs(cfg.n_test, cfg, device, projector, seed + 1)
    with torch.no_grad():
        y_tr = add_label_noise(teacher(x_tr), cfg.noise_std)
        y_te = teacher(x_te)
    return TSData(x_tr, y_tr, x_te, y_te, projector)


@torch.no_grad()
def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean().item())


@torch.no_grad()
def normalized_gen_error(student: TwoLayerReLU, teacher: TwoLayerReLU, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = student(x)
    err = ((pred - y) ** 2).mean()
    var = y.var().clamp(min=1e-8)
    return float((err / var).item())


@torch.no_grad()
def overlap_matrix(student: TwoLayerReLU, teacher: TwoLayerReLU) -> torch.Tensor:
    ws = F.normalize(student.first_layer_weights(), dim=1)
    wt = F.normalize(teacher.first_layer_weights(), dim=1)
    return (ws @ wt.T).abs()


@torch.no_grad()
def per_teacher_overlap(student: TwoLayerReLU, teacher: TwoLayerReLU) -> torch.Tensor:
    return overlap_matrix(student, teacher).max(dim=0).values


@torch.no_grad()
def teacher_overlap(student: TwoLayerReLU, teacher: TwoLayerReLU) -> float:
    return float(per_teacher_overlap(student, teacher).mean().item())


@torch.no_grad()
def feature_drift(student: TwoLayerReLU, x: torch.Tensor, h0: torch.Tensor) -> float:
    return float(((student.hidden(x) - h0) ** 2).mean().item())


@torch.no_grad()
def readout_overlap(student: TwoLayerReLU, teacher: TwoLayerReLU) -> float:
    """Second-layer alignment: |cos(a, a*)|."""
    a_s = F.normalize(student.fc2.weight.flatten(), dim=0)
    a_t = F.normalize(teacher.fc2.weight.flatten(), dim=0)
    k = min(a_s.numel(), a_t.numel())
    return float((a_s[:k] @ a_t[:k]).abs().item())


def mse_hessian_top_eig(
    model: TwoLayerReLU, x: torch.Tensor, y: torch.Tensor, *, n_iter: int = 10
) -> float:
    """Power iteration on top eigenvalue of MSE Hessian (batch subset)."""
    params = [p for p in model.parameters() if p.requires_grad]
    flat_n = sum(p.numel() for p in params)
    v = torch.randn(flat_n, device=x.device)
    v = v / v.norm()
    mse_fn = nn.MSELoss()

    def flat_grad() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        loss = mse_fn(model(x), y)
        loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params])

    for _ in range(n_iter):
        g = flat_grad()
        hv = torch.autograd.grad(g @ v, params, retain_graph=False)
        Hv = torch.cat([h.reshape(-1) for h in hv])
        lam = float((v @ Hv).item())
        v = Hv / (Hv.norm() + 1e-12)
    return lam


def train_student(
    student: TwoLayerReLU,
    teacher: TwoLayerReLU,
    data: TSData,
    *,
    lr: float,
    epochs: int,
    batch_size: int = 256,
    log_every: int = 5,
    on_step: Callable[[int, int, float, TwoLayerReLU], None] | None = None,
) -> int:
    """Full-batch SGD with optional per-step callback."""
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.0)
    mse_fn = nn.MSELoss()
    n = len(data.x_tr)
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n, device=data.x_tr.device)
        student.train()
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x, y = data.x_tr[idx], data.y_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss = mse_fn(student(x), y)
            loss.backward()
            opt.step()
            if on_step is not None and step % log_every == 0:
                on_step(step, epoch, loss.item(), student)
            step += 1
    return step
