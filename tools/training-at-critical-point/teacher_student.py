"""Teacher–student experiments (Gerace / Goldt / Saad–Solla style).

Gaussian inputs, fixed ReLU teacher, trainable student. Order parameters:
  R     — weight overlap with teacher (symmetrized max matching)
  eps_g — normalized test MSE  E[(f - f*)^2] / Var(f*)
  eps_t — train MSE
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TSConfig:
    input_dim: int = 50
    teacher_width: int = 4
    noise_std: float = 0.05
    n_train: int = 2000
    n_test: int = 8000


class TwoLayerReLU(nn.Module):
    """f(x) = (1/sqrt(K)) * w2 @ ReLU(W1 x)."""

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


def init_student(cfg: TSConfig, student_width: int, device: torch.device, seed: int) -> TwoLayerReLU:
    torch.manual_seed(seed + 10_000)
    student = TwoLayerReLU(cfg.input_dim, student_width).to(device)
    with torch.no_grad():
        nn.init.normal_(student.fc1.weight, std=0.5)
        nn.init.normal_(student.fc1.bias, std=0.05)
        nn.init.normal_(student.fc2.weight, std=0.5)
    return student


def sample_gaussian(n: int, dim: int, device: torch.device, gen: torch.Generator | None = None) -> torch.Tensor:
    if gen is not None and device.type == "cpu":
        return torch.randn(n, dim, device=device, generator=gen)
    return torch.randn(n, dim, device=device)


@torch.no_grad()
def teacher_labels(teacher: TwoLayerReLU, x: torch.Tensor, noise_std: float, gen: torch.Generator | None) -> torch.Tensor:
    y = teacher(x)
    if noise_std > 0:
        if gen is not None and x.device.type == "cpu":
            y = y + noise_std * torch.randn(y.shape, device=x.device, generator=gen)
        else:
            y = y + noise_std * torch.randn_like(y)
    return y


def make_dataset(cfg: TSConfig, teacher: TwoLayerReLU, device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x_tr = sample_gaussian(cfg.n_train, cfg.input_dim, device, gen)
    y_tr = teacher_labels(teacher, x_tr, cfg.noise_std, gen)
    x_te = sample_gaussian(cfg.n_test, cfg.input_dim, device, gen)
    y_te = teacher(x_te)
    return x_tr, y_tr, x_te, y_te


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
def teacher_overlap(student: TwoLayerReLU, teacher: TwoLayerReLU) -> float:
    """Max-mean absolute cosine overlap between student and teacher first-layer rows."""
    ws = F.normalize(student.first_layer_weights(), dim=1)
    wt = F.normalize(teacher.first_layer_weights(), dim=1)
    cos = (ws @ wt.T).abs()
    # Greedy matching: each teacher neuron matched to best student neuron
    matched = cos.max(dim=0).values
    return float(matched.mean().item())


@torch.no_grad()
def feature_drift(student: TwoLayerReLU, x: torch.Tensor, h0: torch.Tensor) -> float:
    h = student.hidden(x)
    return float(((h - h0) ** 2).mean().item())


def mse_hessian_top_eig(
    student: TwoLayerReLU,
    x: torch.Tensor,
    y: torch.Tensor,
    n_iter: int = 8,
) -> float:
    params = [p for p in student.parameters() if p.requires_grad]
    mse_loss = nn.MSELoss()

    def hvp(v_flat: torch.Tensor) -> torch.Tensor:
        student.zero_grad(set_to_none=True)
        pred = student(x)
        loss = mse_loss(pred, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        dot = (flat_grad * v_flat).sum()
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in hv])

    device = x.device
    dim = sum(p.numel() for p in params)
    v = torch.randn(dim, device=device)
    v = v / v.norm()
    lam = 0.0
    for _ in range(n_iter):
        Hv = hvp(v)
        lam = torch.dot(v, Hv).item()
        v = Hv / (Hv.norm() + 1e-12)
    return float(lam)
