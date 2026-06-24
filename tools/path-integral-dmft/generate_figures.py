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
DPI = 600

_skill = Path(os.environ.get("AGENT_SKILLS_ROOT", Path.home() / ".agent-skills"))
sys.path.insert(0, str(_skill / "research-plotting" / "scripts"))
from research_plotting import add_panel_label, clean_axis, set_research_style  # noqa: E402


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


BLOG_BG = "#ffffff"
BLOG_FG = "#1e293b"
BLOG_MUTED = "#64748b"


def _prep_blog_style(fig: plt.Figure) -> None:
    """Solid light panel so axes stay readable on dark site theme."""
    fig.patch.set_facecolor(BLOG_BG)
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        ax.set_facecolor(BLOG_BG)
        ax.tick_params(colors=BLOG_FG)
        ax.xaxis.label.set_color(BLOG_FG)
        ax.yaxis.label.set_color(BLOG_FG)
        ax.title.set_color(BLOG_FG)
        for spine in ax.spines.values():
            spine.set_color(BLOG_MUTED)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(BLOG_FG)


def _save_blog(fig: plt.Figure, stem: Path, web_name: str) -> None:
    """PNG with white background for readable light/dark blog themes."""
    _prep_blog_style(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.06, facecolor=BLOG_BG)
    _web(png, ASSETS / web_name)
    plt.close(fig)


def fig1_semicircle(gen: np.random.Generator) -> None:
    n = 4000
    _, eig = goe_matrix(n, gen)
    lam = np.linspace(-2.05, 2.05, 600)
    rho = wigner_density(lam)
    peak = 1.0 / np.pi  # semicircle maximum at lambda=0

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.hist(eig, bins=140, density=True, color="#6366f1", alpha=0.8, edgecolor="none")
    ax.plot(lam, rho, color="#fbbf24", lw=2.5, label=r"Wigner semicircle")
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"density $\rho(\lambda)$")
    ax.set_xlim(-2.35, 2.35)
    ax.set_ylim(0, peak * 1.12)
    ax.legend(frameon=False, fontsize=8)
    clean_axis(ax)
    fig.tight_layout()
    _save_blog(fig, OUT / "fig1_goe_semicircle", "fig1-goe-semicircle.png")


def fig2_response_decay(gen: np.random.Generator) -> None:
    """Two panels: full decay on linear τ, then log-log tail vs τ^{-3/2}."""
    n = 4000
    _, eig = goe_matrix(n, gen)
    tau_lin = np.linspace(0.05, 40.0, 400)
    tau_log = np.logspace(-0.3, 2.2, 300)
    r_lin_num = response_from_eigs(eig, tau_lin, Z_SHIFT)
    r_lin_th = response_theory(tau_lin, Z_SHIFT)
    r_log_num = response_from_eigs(eig, tau_log, Z_SHIFT)
    r_log_th = response_theory(tau_log, Z_SHIFT)
    ref = tau_log ** (-1.5)
    ref *= r_log_th[np.argmin(np.abs(tau_log - 12.0))] / ref[np.argmin(np.abs(tau_log - 12.0))]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax0, ax1 = axes

    ax0.semilogy(tau_lin, r_lin_num, color="#f97316", lw=2.0, label=rf"finite $N={n}$")
    ax0.semilogy(tau_lin, r_lin_th, color="#1e293b", ls="--", lw=1.8, label="semicircle integral")
    ax0.set_xlabel(r"time lag $\tau$")
    ax0.set_ylabel(r"$R_z(\tau)$")
    ax0.legend(frameon=False, fontsize=6.5, loc="upper right")
    add_panel_label(ax0, "a")
    clean_axis(ax0)

    ax1.loglog(tau_log, r_log_num, color="#f97316", lw=2.0, label=rf"finite $N={n}$")
    ax1.loglog(tau_log, r_log_th, color="#1e293b", ls="--", lw=1.8, label="theory")
    ax1.loglog(tau_log, ref, color="#94a3b8", ls=":", lw=1.5, label=r"$\tau^{-3/2}$")
    ax1.set_xlabel(r"time lag $\tau$")
    ax1.set_ylabel(r"$R_z(\tau)$")
    ax1.legend(frameon=False, fontsize=6.5, loc="upper right")
    add_panel_label(ax1, "b")
    clean_axis(ax1)

    fig.tight_layout()
    _save_blog(fig, OUT / "fig2_response_decay", "fig2-response-decay.png")


def fig3_mode_decay() -> None:
    z = Z_SHIFT
    lam = np.linspace(-2.0, 2.0, 500)
    tau = np.linspace(0.0, 50.0, 400)
    rho = wigner_density(lam)
    w = rho[None, :] * np.exp(-(lam[None, :] + z) * tau[:, None])

    fig = plt.figure(figsize=(7.0, 3.8))
    gs = GridSpec(1, 2, width_ratios=[4, 0.18], wspace=0.1)
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
    ax.set_xlabel(r"eigenvalue $\lambda$")
    ax.set_ylabel(r"time lag $\tau$")
    ax.set_title(r"$W(\lambda,\tau)=\rho(\lambda)\,e^{-(\lambda+z)\tau}$", fontsize=9)
    r_tau = trapezoid(w, lam, axis=1)
    ax_r.plot(r_tau, tau, color="#fbbf24", lw=1.5)
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$R_z(\tau)$", fontsize=8)
    ax_r.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="weight")
    fig.tight_layout()
    _save_blog(fig, OUT / "fig3_mode_decay", "fig3-mode-decay.png")


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
    """Zoomed correlation: C depends on t+t', so structure lives near small summed time."""
    n = 2000
    _, eig = goe_matrix(n, gen)
    z = 2.0
    tgrid = np.linspace(0.0, 8.0, 140)
    c, _ = _two_time_from_eigs(eig, z, tgrid)
    c_slice = np.array([np.mean(np.exp(-(eig + z) * st)) for st in tgrid])
    c_slice /= c_slice[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    im = axes[0].imshow(
        c,
        origin="lower",
        aspect="equal",
        extent=[tgrid[0], tgrid[-1], tgrid[0], tgrid[-1]],
        cmap="turbo",
        norm=mcolors.PowerNorm(gamma=0.45, vmin=0, vmax=1),
    )
    axes[0].set_xlabel(r"time $t$")
    axes[0].set_ylabel(r"time $t'$")
    axes[0].set_title(r"$C(t,t')/C(0,0)$, zoom $t,t'\in[0,8]$", fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    axes[1].semilogy(tgrid, c_slice, color="#6366f1", lw=2)
    axes[1].set_xlabel(r"summed time $s = t + t'$")
    axes[1].set_ylabel(r"$C(s)/C(0)$")
    axes[1].set_title("symmetric GOE: $C(t,t') = R(s)$", fontsize=8)
    clean_axis(axes[1])
    fig.tight_layout()
    _save_blog(fig, OUT / "fig4_correlation", "fig4-correlation-heatmap.png")


def fig5_response_heatmap(gen: np.random.Generator) -> None:
    n = 2000
    _, eig = goe_matrix(n, gen)
    z = 2.0
    tgrid = np.linspace(0.0, 10.0, 140)
    _, r = _two_time_from_eigs(eig, z, tgrid)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(
        r,
        origin="lower",
        aspect="equal",
        extent=[tgrid[0], tgrid[-1], tgrid[0], tgrid[-1]],
        cmap="plasma",
        norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=1),
    )
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"time $t'$")
    ax.set_title(r"$R(t,t')$, zoom $t,t'\in[0,10]$", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$R(t,t')$")
    fig.tight_layout()
    _save_blog(fig, OUT / "fig5_response", "fig5-response-heatmap.png")


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

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    axes[0, 0].scatter(eig_sym.real, np.zeros_like(eig_sym), s=4, c="#60a5fa", alpha=0.5)
    axes[0, 0].set_xlim(-2.5, 2.5)
    axes[0, 0].set_ylim(-0.15, 0.15)
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
    _save_blog(fig, OUT / "fig6_sym_antisym", "fig6-sym-antisym.png")


def fig7_linear_regression(gen: np.random.Generator) -> None:
    n = 1000
    alphas = [0.5, 1.0, 2.0, 5.0]
    sigma = 0.1
    steps = 800
    eta = 0.3
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
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
    _save_blog(fig, OUT / "fig7_linear_regression", "fig7-linear-regression.png")


def fig8_finite_n_convergence(gen: np.random.Generator) -> None:
    ns = [500, 1000, 2000, 4000, 8000]
    tau = np.logspace(-1, 2, 200)
    r_th = response_theory(tau, Z_SHIFT)
    errors = []
    for n in ns:
        _, eig = goe_matrix(n, gen)
        r_num = response_from_eigs(eig, tau, Z_SHIFT)
        errors.append(float(np.linalg.norm(r_num - r_th) / np.linalg.norm(r_th)))
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.semilogy(ns, errors, "o-", color="#6366f1", lw=1.8, ms=6)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\|R_{\mathrm{num}} - R_{\mathrm{th}}\|_2 / \|R_{\mathrm{th}}\|_2$")
    clean_axis(ax)
    fig.tight_layout()
    _save_blog(fig, OUT / "fig8_finite_n", "fig8-finite-n-convergence.png")


def gif_spectral_modes() -> None:
    """Animated 1D slices: who contributes to R_z(τ) at each lag."""
    z = Z_SHIFT
    lam = np.linspace(-2.0, 2.0, 400)
    rho = wigner_density(lam)
    rho_max = float(rho.max())

    # Smooth, evenly spaced lags for continuous motion.
    frame_taus = np.linspace(0.0, 35.0, 90)

    tau_dense = np.linspace(0.0, 35.0, 400)
    r_dense = response_theory(tau_dense, z)

    fig = plt.figure(figsize=(7.2, 3.8), facecolor=BLOG_BG)
    gs = GridSpec(2, 2, height_ratios=[1, 0.1], width_ratios=[1.15, 0.85], hspace=0.32, wspace=0.28)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_banner = fig.add_subplot(gs[1, :])
    ax_banner.axis("off")

    ax_w.fill_between(lam, np.zeros_like(lam), rho, color="#6366f1", alpha=0.25, label=r"$\rho(\lambda)$")
    ax_w.plot(lam, rho, color="#6366f1", lw=1.4, alpha=0.7)
    (active,) = ax_w.plot(lam, rho, color="#f97316", lw=2.4, label=r"$W(\lambda,\tau)$")
    ax_w.axvline(-2.0, color="#94a3b8", ls="--", lw=1.0, alpha=0.8)
    ax_w.text(-1.95, rho_max * 0.97, r"$\lambda=-2$", fontsize=7, color=BLOG_MUTED, ha="left", va="top")
    ax_w.set_xlim(-2.2, 2.2)
    ax_w.set_ylim(0, rho_max * 1.12)
    ax_w.set_xlabel(r"eigenvalue $\lambda$")
    ax_w.set_ylabel("integrand weight")
    ax_w.set_title(r"$W(\lambda,\tau)$ at current $\tau$", fontsize=8)
    ax_w.legend(frameon=False, fontsize=6.5, loc="upper right")

    (line,) = ax_rt.plot([], [], color="#f97316", lw=2.4)
    (marker,) = ax_rt.plot([], [], "o", color="#f97316", ms=4, zorder=3)
    ax_rt.semilogy(tau_dense, r_dense, color="#cbd5e1", lw=1.0, zorder=0)
    ax_rt.set_xlim(-0.5, 35.5)
    ax_rt.set_ylim(r_dense[r_dense > 0].min() * 0.5, r_dense.max() * 1.4)
    ax_rt.set_xlabel(r"time lag $\tau$")
    ax_rt.set_ylabel(r"$R_z(\tau)$")
    ax_rt.set_title("integrated response", fontsize=8)

    banner = ax_banner.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=9,
        color=BLOG_FG,
        transform=ax_banner.transAxes,
    )

    _prep_blog_style(fig)

    def update(frame: int):
        tau = float(frame_taus[frame])
        w = rho * np.exp(-(lam + z) * tau)
        active.set_ydata(w)
        line.set_data(tau_dense[tau_dense <= tau], response_theory(tau_dense[tau_dense <= tau], z))
        marker.set_data([tau], [float(response_theory(np.array([tau]), z)[0])])
        banner.set_text(rf"$\tau = {tau:.1f}$")
        return active, line, marker, banner

    gif_fps = 10.0
    ani = animation.FuncAnimation(fig, update, frames=len(frame_taus), interval=1000 / gif_fps, blit=False)
    gif_path = ASSETS / "gif-spectral-modes-response.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(gif_path, writer=animation.PillowWriter(fps=gif_fps), savefig_kwargs={"facecolor": BLOG_BG})
    plt.close(fig)
    print(f"GIF saved to {gif_path} ({len(frame_taus)} frames @ {gif_fps} fps)")


def _web(src: Path, dst: Path) -> None:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    if img.mode == "RGBA":
        # Flatten onto white so dark-mode page background does not bleed through.
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
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
