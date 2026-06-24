#!/usr/bin/env python3
"""Figures for 'Summing Over Training Histories' — GOE/DMFT path integral blog."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.integrate import trapezoid

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets/img/blog/path-integral"
OUT = REPO / "experiments/path-integral-dmft/outputs"
SEED = 0
Z_SHIFT = 2.0

_skill = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
sys.path.insert(0, str(_skill / "research-plotting" / "scripts"))
from research_plotting import (  # noqa: E402
    add_panel_label,
    clean_axis,
    plot_heatmap,
    save_figure,
    set_research_style,
)


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def goe_matrix(n: int, gen: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    g = gen.normal(size=(n, n))
    a = (g + g.T) / np.sqrt(2.0)
    m = a / np.sqrt(n)
    eig = np.linalg.eigvalsh(m)
    return m, eig


def wigner_density(lam: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 4.0 - lam**2)) / (2.0 * np.pi)


def response_from_eigs(eig: np.ndarray, tau: np.ndarray, z: float = Z_SHIFT) -> np.ndarray:
    return np.array([np.mean(np.exp(-(eig + z) * t)) for t in tau])


def response_theory(tau: np.ndarray, z: float = Z_SHIFT) -> np.ndarray:
    lam = np.linspace(-2.0, 2.0, 8000)
    rho = wigner_density(lam)
    out = np.empty_like(tau, dtype=float)
    for i, t in enumerate(tau):
        out[i] = trapezoid(rho * np.exp(-(lam + z) * t), lam)
    return out


def fig1_semicircle(gen: np.random.Generator) -> None:
    n = 4000
    _, eig = goe_matrix(n, gen)
    lam = np.linspace(-2.05, 2.05, 600)
    rho = wigner_density(lam)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(eig, bins=140, density=True, color="#6366f1", alpha=0.75, edgecolor="none")
    ax.plot(lam, rho, color="#fbbf24", lw=2.5, label=r"Wigner semicircle")
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"density $\rho(\lambda)$")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(0, 0.22)
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT / "fig1_goe_semicircle")
    _web(OUT / "fig1_goe_semicircle.png", ASSETS / "fig1-goe-semicircle.png")
    plt.close(fig)


def fig2_response_decay(gen: np.random.Generator) -> None:
    n = 4000
    _, eig = goe_matrix(n, gen)
    tau = np.logspace(-1, 2, 300)
    r_num = response_from_eigs(eig, tau, Z_SHIFT)
    r_th = response_theory(tau, Z_SHIFT)
    ref = tau ** (-1.5)
    ref *= r_th[np.argmin(np.abs(tau - 10.0))] / ref[np.argmin(np.abs(tau - 10.0))]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.loglog(tau, r_num, color="#f97316", lw=2.0, label=rf"finite $N={n}$")
    ax.loglog(tau, r_th, color="#1e293b", ls="--", lw=1.8, label="theory")
    ax.loglog(tau, ref, color="#94a3b8", ls=":", lw=1.5, label=r"$\tau^{-3/2}$")
    ax.set_xlabel(r"time lag $\tau$")
    ax.set_ylabel(r"$R_z(\tau)$")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT / "fig2_response_decay")
    _web(OUT / "fig2_response_decay.png", ASSETS / "fig2-response-decay.png")
    plt.close(fig)


def fig3_mode_decay() -> None:
    z = Z_SHIFT
    lam = np.linspace(-2.0, 2.0, 500)
    tau = np.linspace(0.0, 50.0, 400)
    rho = wigner_density(lam)
    w = rho[None, :] * np.exp(-(lam[None, :] + z) * tau[:, None])

    fig = plt.figure(figsize=(6.8, 3.8))
    gs = GridSpec(1, 2, width_ratios=[4, 0.15], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    im = ax.imshow(
        w,
        aspect="auto",
        origin="lower",
        extent=[lam[0], lam[-1], tau[0], tau[-1]],
        cmap="magma",
        norm=mcolors.LogNorm(vmin=max(w[w > 0].min(), 1e-8), vmax=w.max()),
    )
    ax.axvline(-2.0, color="#22d3ee", ls="--", lw=1.0, alpha=0.9)
    ax.text(-1.95, 47, "slow edge", color="#22d3ee", fontsize=7, ha="left", va="top")
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"time lag $\tau$")
    ax.set_title(r"$W(\lambda,\tau)=\rho(\lambda)\,e^{-(\lambda+z)\tau}$", fontsize=9)

    r_tau = trapezoid(w, lam, axis=1)
    ax_r.plot(r_tau, tau, color="#fbbf24", lw=1.5)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$R_z(\tau)$", fontsize=8)
    ax_r.set_yticks([])
    ax_r.tick_params(labelsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="weight")
    fig.tight_layout()
    save_figure(fig, OUT / "fig3_mode_decay")
    _web(OUT / "fig3_mode_decay.png", ASSETS / "fig3-mode-decay.png")
    plt.close(fig)


def _two_time_from_eigs(eig: np.ndarray, z: float, tgrid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(eig)
    t1, t2 = np.meshgrid(tgrid, tgrid, indexing="ij")
    c = np.zeros_like(t1)
    r = np.zeros_like(t1)
    delta = t1 - t2
    mask = delta >= 0
    for lam in eig:
        c += np.exp(-(lam + z) * (t1 + t2))
        r[mask] += np.exp(-(lam + z) * delta[mask])
    c /= n
    r[mask] /= n
    c /= c[0, 0]
    r /= r.max() if r.max() > 0 else 1
    return c, r


def fig4_correlation(gen: np.random.Generator) -> None:
    n = 2000
    _, eig = goe_matrix(n, gen)
    z = 2.0
    tgrid = np.linspace(0.0, 20.0, 160)
    c, _ = _two_time_from_eigs(eig, z, tgrid)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(
        c,
        origin="lower",
        aspect="auto",
        extent=[tgrid[0], tgrid[-1], tgrid[0], tgrid[-1]],
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"time $t'$")
    ax.set_xticks(np.arange(0, 21, 4))
    ax.set_yticks(np.arange(0, 21, 4))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$C(t,t')/C(0,0)$")
    fig.tight_layout()
    save_figure(fig, OUT / "fig4_correlation")
    _web(OUT / "fig4_correlation.png", ASSETS / "fig4-correlation-heatmap.png")
    plt.close(fig)


def fig5_response_heatmap(gen: np.random.Generator) -> None:
    n = 2000
    _, eig = goe_matrix(n, gen)
    z = 2.0
    tgrid = np.linspace(0.0, 20.0, 160)
    _, r = _two_time_from_eigs(eig, z, tgrid)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(
        r,
        origin="lower",
        aspect="auto",
        extent=[tgrid[0], tgrid[-1], tgrid[0], tgrid[-1]],
        cmap="plasma",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"time $t'$")
    ax.set_xticks(np.arange(0, 21, 4))
    ax.set_yticks(np.arange(0, 21, 4))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$R(t,t')$")
    fig.tight_layout()
    save_figure(fig, OUT / "fig5_response")
    _web(OUT / "fig5_response.png", ASSETS / "fig5-response-heatmap.png")
    plt.close(fig)


def fig6_sym_antisym(gen: np.random.Generator) -> None:
    n = 1500
    g = gen.normal(size=(n, n))
    m_sym = (g + g.T) / np.sqrt(2.0 * n)
    m_anti = (g - g.T) / np.sqrt(2.0 * n)
    eig_sym = np.linalg.eigvalsh(m_sym)
    eig_anti = np.linalg.eigvals(m_anti)
    tau = np.linspace(0.0, 30.0, 400)
    z = 2.2
    r_sym = response_from_eigs(eig_sym, tau, z)
    r_anti = np.array([np.mean(np.exp(eig_anti * t)).real for t in tau])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes[0, 0].scatter(eig_sym.real, np.zeros_like(eig_sym), s=4, c="#60a5fa", alpha=0.5)
    axes[0, 0].set_xlabel(r"Re$(\lambda)$")
    axes[0, 0].set_ylabel(r"Im$(\lambda)$")
    axes[0, 0].set_title("symmetric $M$", fontsize=8)
    add_panel_label(axes[0, 0], "a")
    clean_axis(axes[0, 0])

    axes[0, 1].plot(tau, r_sym, color="#60a5fa", lw=1.8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"$\tau$")
    axes[0, 1].set_ylabel(r"$R_z(\tau)$")
    axes[0, 1].set_title("relaxation", fontsize=8)
    add_panel_label(axes[0, 1], "b")
    clean_axis(axes[0, 1])

    axes[1, 0].scatter(eig_anti.real, eig_anti.imag, s=4, c="#e879f9", alpha=0.5)
    axes[1, 0].set_xlabel(r"Re$(\lambda)$")
    axes[1, 0].set_ylabel(r"Im$(\lambda)$")
    axes[1, 0].set_title("antisymmetric $M$", fontsize=8)
    add_panel_label(axes[1, 0], "c")
    clean_axis(axes[1, 0])

    axes[1, 1].plot(tau, r_anti, color="#e879f9", lw=1.8)
    axes[1, 1].set_xlabel(r"$\tau$")
    axes[1, 1].set_ylabel(r"$R_{\mathrm{anti}}(\tau)$")
    axes[1, 1].set_title("oscillation", fontsize=8)
    add_panel_label(axes[1, 1], "d")
    clean_axis(axes[1, 1])

    fig.tight_layout()
    save_figure(fig, OUT / "fig6_sym_antisym")
    _web(OUT / "fig6_sym_antisym.png", ASSETS / "fig6-sym-antisym.png")
    plt.close(fig)


def fig7_linear_regression(gen: np.random.Generator) -> None:
    n = 1000
    alphas = [0.5, 1.0, 2.0, 5.0]
    sigma = 0.1
    steps = 800
    eta = 0.3
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))

    for alpha in alphas:
        p = int(alpha * n)
        psi = gen.normal(size=(p, n))
        beta = gen.normal(size=n)
        beta = beta / np.linalg.norm(beta) * np.sqrt(n)
        eps = sigma * gen.normal(size=p)
        y = psi @ beta / np.sqrt(n) + eps
        w = np.zeros(n)
        train_hist, test_hist = [], []
        for _ in range(steps):
            pred = psi @ w / np.sqrt(n)
            grad = (2.0 / (p * np.sqrt(n))) * psi.T @ (pred - y)
            w -= eta * grad
            train_hist.append(np.mean((pred - y) ** 2))
            test_hist.append(np.mean((w - beta) ** 2) / n + sigma**2)
        t = np.arange(steps)
        c = cmap(norm(alpha))
        ax.semilogy(t, train_hist, color=c, lw=1.6, solid_capstyle="round")
        ax.semilogy(t, test_hist, color=c, lw=1.6, ls="--", alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\alpha = P/N$")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("loss")
    ax.text(0.03, 0.08, "solid: train  ·  dashed: test", transform=ax.transAxes, fontsize=7, color="#64748b")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT / "fig7_linear_regression")
    _web(OUT / "fig7_linear_regression.png", ASSETS / "fig7-linear-regression.png")
    plt.close(fig)


def gif_spectral_modes() -> None:
    z = Z_SHIFT
    lam = np.linspace(-2.0, 2.0, 400)
    rho = wigner_density(lam)
    tau_vals = np.linspace(0.5, 40.0, 60)
    r_full = response_theory(tau_vals, z)

    fig = plt.figure(figsize=(7.0, 3.2), facecolor="#0f172a")
    gs = GridSpec(1, 3, width_ratios=[1, 1.4, 1], wspace=0.25)
    ax_rho = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_rt = fig.add_subplot(gs[0, 2])
    for ax in (ax_rho, ax_w, ax_rt):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#cbd5e1", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    ax_rho.fill_between(lam, rho, color="#6366f1", alpha=0.8)
    ax_rho.set_xlim(-2, 2)
    ax_rho.set_ylim(0, 0.2)
    ax_rho.set_xlabel(r"$\rho(\lambda)$", color="#e2e8f0", fontsize=8)
    ax_rho.set_title("spectrum", color="#e2e8f0", fontsize=9)

    w0 = rho[None, :] * np.exp(-(lam[None, :] + z) * tau_vals[0])
    im = ax_w.imshow(
        w0,
        aspect="auto",
        origin="lower",
        extent=[lam[0], lam[-1], 0, tau_vals[-1]],
        cmap="magma",
        vmin=0,
        vmax=(rho[None, :] * np.exp(-(lam[None, :] + z) * 0.01)).max(),
    )
    ax_w.axhline(0, color="#22d3ee", lw=0.8)
    ax_w.set_xlabel(r"$\lambda$", color="#e2e8f0", fontsize=8)
    ax_w.set_ylabel(r"$\tau$", color="#e2e8f0", fontsize=8)
    ax_w.set_title("mode weights", color="#e2e8f0", fontsize=9)

    (line,) = ax_rt.plot([], [], color="#fbbf24", lw=2)
    ax_rt.set_xscale("log")
    ax_rt.set_yscale("log")
    ax_rt.set_xlim(tau_vals[1], tau_vals[-1])
    ax_rt.set_ylim(r_full.min() * 0.5, r_full.max() * 2)
    ax_rt.set_xlabel(r"$\tau$", color="#e2e8f0", fontsize=8)
    ax_rt.set_ylabel(r"$R_z(\tau)$", color="#e2e8f0", fontsize=8)
    ax_rt.set_title("response", color="#e2e8f0", fontsize=9)

    def update(frame: int):
        tau = tau_vals[frame]
        w = rho * np.exp(-(lam + z) * tau)
        data = rho[None, :] * np.exp(-(lam[None, :] + z) * np.array([tau]))
        im.set_data(data)
        im.set_extent([lam[0], lam[-1], 0, tau])
        ax_w.axhline(tau, color="#22d3ee", lw=0.8)
        line.set_data(tau_vals[: frame + 1], r_full[: frame + 1])
        return im, line

    ani = animation.FuncAnimation(fig, update, frames=len(tau_vals), interval=80, blit=False)
    gif_path = ASSETS / "gif-spectral-modes-response.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(gif_path, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print(f"GIF saved to {gif_path}")


def fig8_finite_n_convergence(gen: np.random.Generator) -> None:
    """L2 error between finite-N response and semicircle theory vs N."""
    ns = [500, 1000, 2000, 4000, 8000]
    tau = np.logspace(-1, 2, 200)
    r_th = response_theory(tau, Z_SHIFT)
    errors = []
    for n in ns:
        _, eig = goe_matrix(n, gen)
        r_num = response_from_eigs(eig, tau, Z_SHIFT)
        errors.append(float(np.linalg.norm(r_num - r_th) / np.linalg.norm(r_th)))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.semilogy(ns, errors, "o-", color="#6366f1", lw=1.8, ms=6)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\|R_{\mathrm{num}} - R_{\mathrm{th}}\|_2 / \|R_{\mathrm{th}}\|_2$")
    clean_axis(ax)
    fig.tight_layout()
    save_figure(fig, OUT / "fig8_finite_n")
    _web(OUT / "fig8_finite_n.png", ASSETS / "fig8-finite-n-convergence.png")
    plt.close(fig)


def _web(src: Path, dst: Path) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    w, h = img.size
    if w > 1200:
        img = img.resize((1200, int(h * 1200 / w)), Image.Resampling.LANCZOS)
    img.save(dst, optimize=True)


def main() -> None:
    set_research_style()
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    gen = rng()
    fig1_semicircle(gen)
    fig2_response_decay(gen)
    fig3_mode_decay()
    fig4_correlation(gen)
    fig5_response_heatmap(gen)
    fig6_sym_antisym(gen)
    fig7_linear_regression(gen)
    fig8_finite_n_convergence(gen)
    gif_spectral_modes()
    print(f"Figures published to {ASSETS}")


if __name__ == "__main__":
    main()
