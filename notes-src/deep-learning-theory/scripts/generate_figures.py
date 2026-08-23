#!/usr/bin/env python3
"""Generate numerical figures for the deep learning theory notes.

Run from notes-src/deep-learning-theory:
    python scripts/generate_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

_mpl = Path(__file__).resolve().parents[1] / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import roots_hermitenorm

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Two-layer MLP: f = v·φ(Wx / √D) / (γ0 √N), W,v ~ N(0,1)
# ---------------------------------------------------------------------------

def relu(h: np.ndarray) -> np.ndarray:
    return np.maximum(h, 0.0)


def relu_grad(h: np.ndarray) -> np.ndarray:
    return (h > 0).astype(np.float64)


def forward(W: np.ndarray, v: np.ndarray, X: np.ndarray, gamma0: float) -> tuple[np.ndarray, np.ndarray]:
    D = X.shape[1]
    N = W.shape[0]
    h = (X @ W.T) / np.sqrt(D)
    phi = relu(h)
    f = (phi @ v) / (gamma0 * np.sqrt(N))
    return f, h


def ntk_matrix(W: np.ndarray, v: np.ndarray, X: np.ndarray, gamma0: float) -> np.ndarray:
    """Empirical NTK on rows of X. K_μν = ∇_θ f_μ · ∇_θ f_ν."""
    D = X.shape[1]
    N = W.shape[0]
    h = (X @ W.T) / np.sqrt(D)
    phi = relu(h)
    dphi = relu_grad(h)
    scale = 1.0 / (gamma0 * np.sqrt(N))
    kv = (phi * scale) @ (phi * scale).T
    gram = X @ X.T
    weighted = dphi * v[None, :]
    kW = (weighted @ weighted.T) * gram * (scale**2 / D)
    return kv + kW


def feature_kernel(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    D = X.shape[1]
    N = W.shape[0]
    phi = relu((X @ W.T) / np.sqrt(D))
    return (phi @ phi.T) / N


def gd_step(
    W: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    gamma0: float,
    eta0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One full-batch GD step. Learning rate η = η0 γ0² N as in the notes."""
    P, D = X.shape
    N = W.shape[0]
    f, h = forward(W, v, X, gamma0)
    err = f - y  # ∂ℓ/∂f with ℓ = (1/2P) Σ (f-y)²  ⇒  ∂L/∂f_μ = err_μ / P
    phi = relu(h)
    dphi = relu_grad(h)
    scale = 1.0 / (gamma0 * np.sqrt(N))
    eta = eta0 * (gamma0**2) * N
    # ∂L/∂v = (1/P) Σ_μ err_μ * ∂f_μ/∂v
    dv = (phi.T @ err) * (scale / P)
    # ∂f_μ/∂W = scale * v_i φ'_μi x_μ / √D
    # ∂L/∂W = (1/P) Σ_μ err_μ * that
    g = (err[:, None] * dphi) * v[None, :]  # P × N
    dW = (g.T @ X) * (scale / (P * np.sqrt(D)))
    v = v - eta * dv
    W = W - eta * dW
    return W, v


def make_teacher_data(P: int, D: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(P, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    w = rng.normal(size=D)
    w /= np.linalg.norm(w)
    y = np.tanh(X @ w)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return X, y


def fig01_ntk_freeze() -> None:
    rng = np.random.default_rng(0)
    D, P, steps = 12, 24, 80
    X, y = make_teacher_data(P, D, rng)
    widths = [32, 128, 512]
    gamma0 = 0.15  # lazy-ish
    eta0 = 0.15
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for N in widths:
        W = rng.normal(size=(N, D))
        v = rng.normal(size=N)
        K0 = ntk_matrix(W, v, X, gamma0)
        n0 = np.linalg.norm(K0)
        rel = []
        for _ in range(steps):
            W, v = gd_step(W, v, X, y, gamma0, eta0)
            Kt = ntk_matrix(W, v, X, gamma0)
            rel.append(np.linalg.norm(Kt - K0) / (n0 + 1e-12))
        ax.plot(np.arange(1, steps + 1), rel, lw=1.8, label=rf"$N={N}$")
    ax.set_xlabel(r"gradient step $t$")
    ax.set_ylabel(r"$\|K(t)-K(0)\|_F/\|K(0)\|_F$")
    ax.set_title(r"NTK motion vs width ($\gamma_0=0.15$)")
    ax.legend(frameon=False)
    save(fig, "fig01_ntk_freeze.pdf")


def fig02_lazy_rich() -> None:
    rng = np.random.default_rng(1)
    D, P, N, steps = 16, 40, 256, 100
    X, y = make_teacher_data(P, D, rng)
    gammas = [0.1, 0.5, 1.5]
    eta0 = 0.08
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))
    for g0 in gammas:
        W = rng.normal(size=(N, D))
        v = rng.normal(size=N)
        Phi0 = feature_kernel(W, X)
        n0 = np.linalg.norm(Phi0)
        drift, train = [], []
        for _ in range(steps):
            W, v = gd_step(W, v, X, y, g0, eta0)
            Phi = feature_kernel(W, X)
            drift.append(np.linalg.norm(Phi - Phi0) / (n0 + 1e-12))
            f, _ = forward(W, v, X, g0)
            train.append(0.5 * np.mean((f - y) ** 2))
        axes[0].plot(np.arange(1, steps + 1), drift, lw=1.8, label=rf"$\gamma_0={g0}$")
        axes[1].plot(np.arange(1, steps + 1), train, lw=1.8, label=rf"$\gamma_0={g0}$")
    axes[0].set_xlabel(r"gradient step $t$")
    axes[0].set_ylabel(r"$\|\Phi(t)-\Phi(0)\|_F/\|\Phi(0)\|_F$")
    axes[0].set_title("feature-kernel drift")
    axes[1].set_xlabel(r"gradient step $t$")
    axes[1].set_ylabel("train MSE")
    axes[1].set_title("training loss")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False)
    fig.suptitle(rf"lazy vs rich, $N={N}$", y=1.02)
    fig.tight_layout()
    save(fig, "fig02_lazy_rich.pdf")


def gh_nodes(n: int = 56) -> tuple[np.ndarray, np.ndarray]:
    x, w = roots_hermitenorm(n)
    return x, w / np.sqrt(2.0 * np.pi)


def tanh_kernel_map(K11: float, K22: float, K12: float, sw2: float, sb2: float, x: np.ndarray, w: np.ndarray) -> float:
    """K' = σ_b² + σ_w² E[tanh(h1) tanh(h2)] for jointly Gaussian (h1,h2)."""
    s1, s2 = np.sqrt(max(K11, 1e-15)), np.sqrt(max(K22, 1e-15))
    rho = np.clip(K12 / (s1 * s2), -0.999, 0.999)
    # h1 = s1 z1, h2 = s2 (rho z1 + sqrt(1-rho^2) z2)
    z1 = x[:, None]
    z2 = x[None, :]
    ww = w[:, None] * w[None, :]
    h1 = s1 * z1
    h2 = s2 * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
    moment = np.sum(ww * np.tanh(h1) * np.tanh(h2))
    return sb2 + sw2 * moment


def fig03_criticality() -> None:
    x, w = gh_nodes(40)
    depths = np.arange(0, 61)
    # single-input variance recursion, tanh, σ_b=0
    # χ(0)=σ_w² tanh'(0)² = σ_w², so σ_w=1 is critical at the origin
    sws = {"subcritical $\\sigma_w=0.85$": 0.85, "critical $\\sigma_w=1.00$": 1.00, "supercritical $\\sigma_w=1.20$": 1.20}
    K0_diag = 1.0
    c0 = 0.5  # initial cosine
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))
    for label, sw in sws.items():
        q = K0_diag
        c = c0
        qs, cs = [q], [c]
        for _ in depths[1:]:
            q_new = tanh_kernel_map(q, q, q, sw**2, 0.0, x, w)
            k12 = tanh_kernel_map(q, q, c * q, sw**2, 0.0, x, w)
            q = float(max(q_new, 1e-16))
            c = float(np.clip(k12 / q, -1.0, 1.0))
            qs.append(q)
            cs.append(c)
        axes[0].plot(depths, qs, lw=1.8, label=label)
        axes[1].plot(depths, cs, lw=1.8, label=label)
    axes[0].set_xlabel(r"layer $\ell$")
    axes[0].set_ylabel(r"$K^{(\ell)}(x,x)$")
    axes[0].set_title(r"kernel variance, $\tanh$")
    axes[1].set_xlabel(r"layer $\ell$")
    axes[1].set_ylabel(r"$K^{(\ell)}(x,x')/\sqrt{K(x,x)K(x',x')}$")
    axes[1].set_title("kernel cosine, init $c=0.5$")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig03_criticality.pdf")


def fig04_finite_width() -> None:
    """Var of the empirical metric G=mean(h^2) for deep linear nets at C_W=1."""
    rng = np.random.default_rng(2)
    D = 32
    x = rng.normal(size=D)
    x /= np.sqrt(np.mean(x**2))
    widths = np.array([32, 64, 128, 256, 512])
    depths = [2, 4, 8]
    n_init = 400
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for L in depths:
        vars_ = []
        for N in widths:
            Gs = np.empty(n_init)
            for s in range(n_init):
                h = x.copy()
                n_in = D
                for _layer in range(L):
                    W = rng.normal(size=(N, n_in))
                    h = (W @ h) / np.sqrt(n_in)
                    n_in = N
                Gs[s] = np.mean(h**2)
            vars_.append(np.var(Gs))
        ax.plot(1.0 / widths, vars_, "o-", lw=1.6, ms=5, label=rf"$L={L}$")
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$\mathrm{Var}_{\mathrm{init}}[G(x,x)]$")
    ax.set_title(r"deep linear metric fluctuations $\sim L/N$")
    ax.legend(frameon=False)
    save(fig, "fig04_finite_width.pdf")


def fig05_kernel_learning() -> None:
    """Ridge kernel regression with a known power-law spectrum (exact in eigenbasis)."""
    rng = np.random.default_rng(3)
    M = 400
    ks = np.arange(1, M + 1)
    lam = ks ** (-1.5)
    # teacher coefficients: slow decay ⇒ a few easy modes, many hard ones
    a = rng.normal(size=M) * ks ** (-0.9)
    target_var = np.sum(a**2)
    a = a / np.sqrt(target_var)
    lam_ridge = 1e-3
    Ps = np.unique(np.logspace(np.log10(8), np.log10(350), 18).astype(int))
    n_trials = 25
    test_err = []
    mode_err = {1: [], 8: [], 40: []}
    for P in Ps:
        e_tot = []
        e_mode = {m: [] for m in mode_err}
        for _ in range(n_trials):
            # random design in the eigenbasis: Φ_μk = √λ_k Z_μk
            Z = rng.normal(size=(P, M)) / np.sqrt(P)
            Phi = Z * np.sqrt(lam)[None, :]
            y = Phi @ a
            K = Phi @ Phi.T
            alpha = np.linalg.solve(K + lam_ridge * np.eye(P), y)
            a_hat = Phi.T @ alpha  # coefficients in the population eigenbasis
            resid2 = (a - a_hat) ** 2
            e_tot.append(resid2.sum())
            for m in e_mode:
                e_mode[m].append(resid2[m - 1])
        test_err.append(np.mean(e_tot))
        for m in mode_err:
            mode_err[m].append(np.mean(e_mode[m]))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))
    axes[0].plot(Ps, test_err, "o-", lw=1.6, ms=4, color="#1f4e79")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"training samples $P$")
    axes[0].set_ylabel(r"population risk $R$")
    axes[0].set_title(r"kernel ridge, $\lambda_k \propto k^{-3/2}$")
    for m, color in zip(mode_err, ["#1f4e79", "#c44e52", "#55a868"]):
        axes[1].plot(Ps, mode_err[m], "o-", lw=1.6, ms=4, color=color, label=rf"mode $k={m}$")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"training samples $P$")
    axes[1].set_ylabel(r"mode error $a_k^2$ residual")
    axes[1].set_title("spectral bias")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig05_kernel_learning.pdf")


def fig06_margin() -> None:
    from scipy.optimize import minimize

    rng = np.random.default_rng(4)
    xp = rng.normal(loc=(1.7, 0.2), scale=0.26, size=(12, 2))
    xn = rng.normal(loc=(-1.7, -0.2), scale=0.26, size=(12, 2))
    X = np.vstack([xp, xn])
    y = np.concatenate([np.ones(len(xp)), -np.ones(len(xn))])
    P = len(y)
    Q = (y[:, None] * y[None, :]) * (X @ X.T)

    def obj(a: np.ndarray) -> float:
        return 0.5 * float(a @ Q @ a) - float(a.sum())

    def jac(a: np.ndarray) -> np.ndarray:
        return Q @ a - 1.0

    cons = {"type": "eq", "fun": lambda a: float(a @ y), "jac": lambda a: y}
    bounds = [(0.0, None)] * P
    a0 = np.full(P, 1.0 / P)
    res = minimize(
        obj, a0, jac=jac, bounds=bounds, constraints=cons, method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 800, "disp": False},
    )
    a = np.maximum(res.x, 0.0)
    sv = a > 1e-4 * a.max()
    w = (a * y) @ X
    b = float(np.median(y[sv] - X[sv] @ w))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(xp[:, 0], xp[:, 1], c="#1f4e79", s=28, label=r"$y=+1$")
    ax.scatter(xn[:, 0], xn[:, 1], c="#c44e52", s=28, label=r"$y=-1$")
    ax.scatter(X[sv, 0], X[sv, 1], facecolors="none", edgecolors="k", s=90, lw=1.1, label="support")
    ts = np.linspace(-4.5, 4.5, 200)
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    t_hat = np.array([-w_hat[1], w_hat[0]])
    # a point on the hyperplane w·x + b = 0
    x0 = -b * w / (np.dot(w, w) + 1e-12)
    for level, ls, lw in ((0.0, "-", 1.6), (1.0, "--", 0.9), (-1.0, "--", 0.9)):
        # w·x + b = level  ⇒  shift along w by (level)/||w||²
        shift = (level) * w / (np.dot(w, w) + 1e-12)
        pts = x0 + shift + ts[:, None] * t_hat
        ax.plot(pts[:, 0], pts[:, 1], color="k", lw=lw, ls=ls)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("max-margin hyperplane")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, "fig06_margin.pdf")


def fig07_chosaul() -> None:
    th = np.linspace(0.0, np.pi, 300)
    k = (np.sin(th) + (np.pi - th) * np.cos(th)) / (2.0 * np.pi)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(th, k, color="#1f4e79", lw=1.9)
    ax.set_xlabel(r"$\vartheta=\arccos(x\cdot x'/\|x\|\|x'\|)$")
    ax.set_ylabel(r"$K(x,x')/(\|x\|\|x'\|)$")
    ax.set_title(r"Cho--Saul kernel, $\varphi=\mathrm{ReLU}$")
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    save(fig, "fig07_chosaul.pdf")


def fig08_modes() -> None:
    t = np.linspace(0, 8, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for lam, c in zip([2.0, 0.6, 0.15], ["#1f4e79", "#c44e52", "#55a868"]):
        ax.plot(t, np.exp(-lam * t), color=c, lw=1.8, label=rf"$\lambda={lam}$")
    ax.set_xlabel(r"$(\eta/P)\,t$")
    ax.set_ylabel(r"$e^{-\lambda t}$")
    ax.set_title("frozen-kernel mode decay")
    ax.legend(frameon=False)
    save(fig, "fig08_modes.pdf")


def fig10_relu_crit() -> None:
    """Cho--Saul recursion of the kernel cosine for ReLU, C_b=0."""
    def step(q: float, c: float, cw: float) -> tuple[float, float]:
        th = float(np.arccos(np.clip(c, -1.0, 1.0)))
        k12 = cw * q * (np.sin(th) + (np.pi - th) * np.cos(th)) / (2.0 * np.pi)
        qn = cw * q / 2.0
        return float(qn), float(np.clip(k12 / max(qn, 1e-16), -1.0, 1.0))

    depths = np.arange(0, 31)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.35))
    for label, cw in {
        r"subcritical $C_W=1.5$": 1.5,
        r"critical $C_W=2$": 2.0,
        r"supercritical $C_W=2.5$": 2.5,
    }.items():
        q, c = 1.0, 0.5
        qs, cs = [q], [c]
        for _ in depths[1:]:
            q, c = step(q, c, cw)
            qs.append(q)
            cs.append(c)
        axes[0].plot(depths, qs, lw=1.8, label=label)
        axes[1].plot(depths, cs, lw=1.8, label=label)
    axes[0].set_xlabel(r"layer $\ell$")
    axes[0].set_ylabel(r"$K^{(\ell)}(x,x)$")
    axes[0].set_title(r"kernel variance, ReLU")
    axes[0].set_yscale("log")
    axes[1].set_xlabel(r"layer $\ell$")
    axes[1].set_ylabel(r"kernel cosine")
    axes[1].set_title(r"init $c=0.5$")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig10_relu_crit.pdf")


def fig09_regions() -> None:
    N0, N = 2, 16
    L = np.arange(1, 8)
    deep = (N / N0) ** ((L - 1) * N0) * (N ** N0)
    shallow = (N * L) ** N0
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.semilogy(L, deep, "o-", lw=1.7, label=rf"depth $L$, width $N={N}$")
    ax.semilogy(L, shallow, "s--", lw=1.7, label=rf"depth $1$, width $NL$")
    ax.set_xlabel(r"depth $L$")
    ax.set_ylabel(r"linear regions (lower bound)")
    ax.set_title(rf"ReLU, $N_0={N0}$ inputs")
    ax.legend(frameon=False)
    save(fig, "fig09_regions.pdf")


def main() -> None:
    fig01_ntk_freeze()
    fig02_lazy_rich()
    fig03_criticality()
    fig04_finite_width()
    fig05_kernel_learning()
    fig06_margin()
    fig07_chosaul()
    fig08_modes()
    fig09_regions()
    fig10_relu_crit()


if __name__ == "__main__":
    main()
