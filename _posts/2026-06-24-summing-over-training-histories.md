---
title: "Summing Over Training Histories"
date: 2026-06-24
layout: post
description: "A path-integral view of high-dimensional learning dynamics, through the GOE example in Bordelon and Pehlevan."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, path-integrals, random-matrices]
---

I came to physics from deep learning, and for a long time the two felt like they were asking different questions. In CS the workflow is concrete: pick an architecture, define a loss, run SGD, plot the curve. Physics kept replacing that with actions, partition functions, fields, path integrals. I could follow the algebra, but I did not understand why anyone would *want* to think that way.

It clicked when I stopped trying to read path integrals as a replacement for backprop and started reading them as a compression scheme. You have a system with too many coordinates to watch. You give up on the microscopic state and ask what macroscopic functions still carry information. In a neural network that might be a learning curve, a covariance spectrum, or a correlation between weights at two different training times.

This post is about one example from [Bordelon and Pehlevan (2026)](https://arxiv.org/abs/2601.01010) where that compression is completely explicit. They study a random linear dynamical system with $N$ coupled coordinates. A path-integral / DMFT calculation collapses the dynamics onto two order parameters, correlation and response, which describe how long perturbations are remembered. In the GOE warmup, the response recovers the Wigner semicircle law. I am not reproducing the full derivation. I am trying to explain what it *means*, with numerics that make the main equations visible.

The order parameters are

$$
C(t,t') = \frac{1}{N}\, h(t)\cdot h(t'),
\qquad
R(t,t') = \frac{1}{N}\,\operatorname{Tr}\frac{\delta h(t)}{\delta j(t')^\top}.
$$

$C$ asks how similar the state is at two different times. $R$ asks how much a perturbation at one time echoes into another. Correlation is self-memory. Response is memory of an external poke.

---

## Too many coordinates

Training gives you a trajectory

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_T.
$$

The natural CS question is what happens to each coordinate of $\theta_t$. The natural physics question is different: what are the macroscopic observables of the trajectory? Correlations between times. Response to a perturbation. Spectra of covariance matrices. The loss curve itself.

Both questions are about the same dynamical system. One is microscopic. One is not.

The same split shows up outside ML. You cannot track $10^{23}$ molecules, so you track pressure and temperature. You cannot track every weight in a large network, so you track things that survive averaging or that summarize the effect of many degrees of freedom at once. Bordelon and Pehlevan frame DMFT as exactly this move for high-dimensional disordered dynamics: replace microscopic coordinates by self-consistent two-time functions $C(t,t')$ and $R(t,t')$ that encode how long the system remembers itself and how it responds to perturbations.

---

## Sum over histories

Feynman's path integral is usually introduced in quantum mechanics as a sum over all paths $x(t)$, weighted by an action:

$$
\int \mathcal{D}x\; e^{\frac{i}{\hbar} S[x]}.
$$

In statistical settings the same structure appears with real weights, $\int \mathcal{D}x\, e^{-S[x]}$. The point is not that you literally simulate every path. The point is that a *history* becomes the primitive object. You enforce the dynamics, integrate over allowed trajectories, and ask which macroscopic description remains after you average over disorder and take $N$ large.

Three related but distinct objects get lumped together under "path integral," and it helps to separate them. Feynman's quantum path integral uses complex weights $e^{iS/\hbar}$. Statistical mechanics often uses real Boltzmann weights $e^{-S}$. The dynamical path integral used in DMFT is closer to a generating functional: it integrates over histories $h(t)$ and auxiliary response fields $\hat{h}(t)$ while enforcing the equations of motion. The analogy is the grammar (histories, actions, saddle points), not the claim that neural networks are quantum systems.

Bordelon and Pehlevan apply that template to a linear system

$$
\frac{d}{dt} h(t) = -M h(t) + j(t),
\qquad h(t) \in \mathbb{R}^N,
$$

where $M$ is random and $j(t)$ is a source. For the GOE warmup,

$$
M = \frac{1}{\sqrt{N}} A,
\qquad A = A^\top,
\qquad A_{ij} \sim \mathcal{N}(0,1),
$$

up to the usual GOE symmetrization convention (our numerics symmetrize $A_0 + A_0^\top$ and scale so the empirical spectrum sits on $[-2,2]$).

At fixed $N$ this is just a linear ODE. Diagonalize $M$, integrate, done. For the GOE warmup, diagonalization is enough. That is partly why it is a good example: we can check the path-integral / DMFT answer against a known random-matrix result. But the point of the formalism is not to solve this one linear system the hard way. The point is to introduce a method that still works when diagonalization is not the right language: random features, SGD, non-Hermitian dynamics, evolving matrices, and feature-learning models. The toy example is a controlled place to see the machinery.

The theory asks what happens *typically* as $N \to \infty$ after averaging over the random matrix. That is where $C$ and $R$ appear as the variables the system actually depends on.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig1-goe-semicircle.png" alt="GOE eigenvalue histogram with Wigner semicircle overlay" width="500"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1. Eigenvalues of a random symmetric $N \times N$ matrix (histogram) against the Wigner semicircle (gold), matching the GOE warmup in Bordelon and Pehlevan. Every entry of $M$ is random noise. The spectrum is not.</figcaption>
</figure>

Figure 1 is the static version of the story. Before we even talk about time, disorder in the entries produces a deterministic law for the eigenvalues. That already hints at what the path integral is doing: find the structure that survives averaging.

---

## What the path integral actually buys you

I read the Bordelon–Pehlevan calculation as five moves chained together. The paper writes the generating functional as $Z = \int \mathcal{D}Q\, e^{-N S[Q]}$, dominated by a saddle point at large $N$; here is the part I kept in my head.

First, enforce the equation of motion. Only histories with $\partial_t h + Mh - j = 0$ contribute. In a path integral this is a delta functional $\delta[\partial_t h + Mh - j]$.

Second, rewrite that delta function using an auxiliary field $\hat{h}(t)$. This is the step that looks the most mysterious if you have not seen it before. $\hat{h}$ is not a new physical variable. It is a Lagrange multiplier living in the same path-integral formalism, there to enforce the dynamics and later to define response.

Third, average over the random matrix $M$. Because $M$ is Gaussian, the average produces terms that depend on $h$ and $\hat{h}$ only through overlaps. Those overlaps are exactly $C(t,t')$ and $R(t,t')$. This is the compression step I care about most. The matrix couples every coordinate to every other coordinate, but after the average the theory only sees two-time collective functions.

Fourth, take $N$ large. The generating functional has the form $Z \sim \int \mathcal{D}C\,\mathcal{D}R\, e^{-N S[C,R]}$ and is dominated by a saddle point where $\delta S / \delta C = 0$ and $\delta S / \delta R = 0$. Microscopic histories collapse onto self-consistent order parameters.

Fifth, the output is an effective single-site process. A typical coordinate evolves as if it were one-dimensional, driven by colored noise fixed by $C$ and fed back through its own past via $R$:

$$
\frac{\partial}{\partial t} h(t) = u(t) + \int dt'\, R(t,t') h(t') + j(t),
\qquad u(t) \sim \mathrm{GP}(0, C(t,t')).
$$

The dynamics do not disappear. They move from $N$ coupled coordinates into two memory functions that must be solved self-consistently.

In the GOE case, the saddle-point equations close further. The response obeys a self-consistency equation that is much smaller than the original $N$-dimensional system:

$$
\partial_t R(t,t') = \delta(t-t') + \int_0^t dt''\, R(t,t'')\, R(t'',t').
$$

This is the whole high-dimensional dynamics compressed into one scalar memory equation. When the system is time-translation invariant, Fourier transforming gives

$$
i\omega R(\omega) = 1 + R(\omega)^2.
$$

Inverting this resolvent relation produces the Wigner semicircle as the spectral density of $M$. That is the chain the paper walks through in the GOE warmup: path integral $\to$ order parameters $\to$ closed response equation $\to$ semicircle. The numerics below check the last link directly.

---

## Response is the spectrum, read in time

For the linear system the response simplifies. With lag $\tau = t - t'$,

$$
R(\tau) = \frac{1}{N}\operatorname{Tr} e^{-M\tau}.
$$

Diagonalize $M$ and this is a sum over eigenvalues. If $\rho(\lambda)$ is the eigenvalue density,

$$
R(\tau) = \int d\lambda\,\rho(\lambda)\,e^{-\lambda\tau}.
$$

Each eigenvalue contributes a mode $e^{-\lambda\tau}$. The response is what you get when you superpose all of them with weights set by $\rho(\lambda)$.

For the GOE case $\rho(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}$ on $[-2,2]$. So a question about dynamics (how does a perturbation fade?) is secretly a question about spectral density.

**Stability note.** The GOE spectrum has support on $[-2,2]$, so the unshifted dynamics $\dot{h} = -Mh$ is not uniformly stable: modes with $\lambda < 0$ grow under $e^{-\lambda\tau}$. For the response-decay numerics we follow Bordelon and Pehlevan and add a stabilizing shift $z = 2$:

$$
\dot{h}(t) = -M h(t) - z h(t),
\qquad
R_z(\tau) = \int d\lambda\,\rho(\lambda)\,e^{-(\lambda + z)\tau}.
$$

With $z = 2$ the slowest modes sit at the spectral edge $\lambda = -2$, where $\lambda + z = 0$. The late-time response is critical (power-law) rather than exponentially decaying. Their Figure 2 shows this shifted response and the $\tau^{-3/2}$ tail.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig2-response-decay.png" alt="Log-log plot of response decay with theory and tau^{-3/2} reference" width="480"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2. Shifted response $R_z(\tau)$ with $z=2$: finite GOE matrix ($N=4000$, orange) against the semicircle integral (dashed). At late times the slope approaches $\tau^{-3/2}$ (dotted), as in Bordelon and Pehlevan.</figcaption>
</figure>

Figure 2 is where the abstract formula becomes something you can look at. The numerical curve follows the integral until finite-$N$ effects kick in. On log-log axes the tail is visibly straight with slope close to $-3/2$.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig8-finite-n-convergence.png" alt="Relative L2 error vs N for finite-N response" width="380"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2b. Relative $\ell_2$ error between finite-$N$ $R_z(\tau)$ and the semicircle prediction vs. $N$. The finite-$N$ response concentrates around the DMFT / large-$N$ answer as $N$ grows.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig3-mode-decay.png" alt="Heatmap of spectral mode contributions to response" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3. The integrand $W(\lambda,\tau) = \rho(\lambda) e^{-(\lambda+z)\tau}$. Modes with $\lambda$ near $-2$ (cyan line) dominate the tail. The curve on the right is $R_z(\tau)$.</figcaption>
</figure>

Figure 3 is the same object decomposed. At small $\tau$ many eigenvalues contribute. As $\tau$ grows the weight concentrates near the spectral edge. The response curve on the right is the integral of this heatmap over $\lambda$. I found this picture more helpful than any block diagram of the derivation.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/gif-spectral-modes-response.gif" alt="Animation of spectral modes composing the response over time" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Animation of the same decomposition: as $\tau$ increases, fast modes die off and the edge modes carry what is left.</figcaption>
</figure>

---

## Two-time functions, not scalars

$C(t,t')$ and $R(t,t')$ are functions of *two* times. That is easy to gloss over on paper but it matters. The system has a history, not just a present state.

For the GOE linear system with symmetric $M$, the correlation and response are related: in the large-$N$ limit $C(t,t')$ tracks $R(t+t')$. The heatmaps make the geometry concrete.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig4-correlation-heatmap.png" alt="Two-time correlation heatmap C(t,t prime)" width="420"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 4. Normalized $C(t,t')$ for shifted GOE dynamics. Bright near the origin; fades as both times grow.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig5-response-heatmap.png" alt="Causal two-time response heatmap R(t,t prime)" width="420"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 5. $R(t,t')$ is zero below the diagonal: a perturbation at $t'$ cannot affect times before $t'$.</figcaption>
</figure>

Figure 5 was the one that finally made causality feel visual to me. The upper triangle carries the echo of a perturbation. The lower triangle is exactly zero. Response is not symmetric in time the way correlation can be.

---

## Real eigenvalues relax; imaginary ones oscillate

Change the symmetry of $M$ and the texture of the dynamics changes completely.

$$
M_{\mathrm{sym}} = \frac{A + A^\top}{\sqrt{2N}},
\qquad
M_{\mathrm{anti}} = \frac{A - A^\top}{\sqrt{2N}}.
$$

Symmetric $M$ has real eigenvalues. Each mode decays or grows exponentially. Antisymmetric $M$ has purely imaginary eigenvalues. The response oscillates instead of relaxing.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig6-sym-antisym.png" alt="Symmetric vs antisymmetric eigenvalues and response curves" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 6. Top row: symmetric spectrum on the real line, response decays. Bottom row: antisymmetric spectrum on the imaginary axis, response oscillates.</figcaption>
</figure>

Same path-integral template, different spectral geometry, different notion of "memory in time."

---

## The ML connection is not a metaphor

Bordelon and Pehlevan do not stop at the GOE warmup. They run the same formalism on linear regression, random features, kernel methods, and deep linear networks. The bridge I understand best is random linear regression.

Data matrix $\Psi \in \mathbb{R}^{P \times N}$. Targets $y = \frac{1}{\sqrt{N}} \Psi \beta^\star + \epsilon$. Model $f = \frac{1}{\sqrt{N}} \Psi w$. Train loss

$$
\hat{L}(t) = \frac{1}{P} \left\| \frac{1}{\sqrt{N}} \Psi w(t) - y \right\|^2,
$$

test loss

$$
L(t) = \frac{1}{N} \| w(t) - \beta^\star \|^2 + \sigma^2.
$$

Gradient flow on $w$ is governed by the empirical covariance $M = \frac{1}{P} \Psi^\top \Psi$. The residual $h(t) = \beta^\star - w(t)$ satisfies the same kind of random-matrix-driven linear dynamics as the GOE toy model.

In the GOE example, $M$ was an abstract random interaction matrix. In linear regression, $M$ is no longer abstract: it is the data covariance $\frac{1}{P}\Psi^\top \Psi$, a Wishart matrix. Its eigenvalues are learning rates for different error modes. Large eigenvalues decay quickly; small eigenvalues are slow-to-learn directions. When $\alpha = P/N$ crosses the interpolation threshold, the spectrum of $M$ develops a bulk edge and the bias–variance tradeoff becomes most delicate.

When we plot loss versus time, we are already doing statistical physics: throwing away microscopic coordinates and watching an order parameter. A training curve is not just a log file. It is a projection of high-dimensional dynamics onto a scalar observable.

<figure style="text-align: center;">
  <img src="/assets/img/blog/path-integral/fig7-linear-regression.png" alt="Train and test loss curves for random linear regression at several alpha=P/N" width="500"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 7. Random linear regression with label noise $\sigma = 0.1$. Solid: train loss. Dashed: test loss. Color is $\alpha = P/N$. Train and test follow different macroscopic curves as $\alpha$ changes; near the interpolation threshold the covariance spectrum controls which modes decay slowly.</figcaption>
</figure>

The paper goes further in random feature models near interpolation, where test loss can actually go *up* before coming back down, and where the eigenvalue spectrum alone is not enough to predict the curve. That is the part where the full two-time $C(t,t')$ matters, not just a single-time spectrum. I have not reproduced those curves here because I wanted this post to stay anchored on the GOE example where the story is clean.

---

## What this example does not show

The GOE warmup is intentionally clean. It is linear, Gaussian, symmetric, and time-translation invariant. Real neural networks violate all of those assumptions: representations move, Jacobians are non-Hermitian, SGD adds noise, and feature learning changes the effective matrices during training. The reason the example is still useful is that it teaches the basic move: replace microscopic coordinates by self-consistent two-time observables. The rest of DMFT is about making that move survive in harder systems.

---

## What I take from it

I still find path integrals intimidating to derive. But the GOE example gives a concrete target for what "success" looks like. Start with $N$ coupled variables and a random interaction matrix. Average over the disorder. Take $N$ large. End with a small set of self-consistent memory functions. Check that those functions reproduce a spectral law you already know from random matrix theory.

When we plot a learning curve, we are already admitting that the microscopic description is too large. The path integral makes that admission systematic. It asks which histories matter, which averages survive, and which functions remember.

For high-dimensional learning systems, those functions are often correlations, responses, spectra, and losses. They are not merely summaries of the dynamics. They are the dynamics, seen at the scale where understanding becomes possible.

---

### References

* Feynman, R. P., & Hibbs, A. R. (1965). *Quantum Mechanics and Path Integrals*. McGraw-Hill.

* Bordelon, B., & Pehlevan, C. (2026). Disordered Dynamics in High Dimensions: Connections to Random Matrices and Machine Learning. [arXiv:2601.01010](https://arxiv.org/abs/2601.01010).

* Mezard, M., & Montanari, A. (2009). *Information, Physics, and Computation*. Oxford University Press.

*Figures: `tools/path-integral-dmft/generate_figures.py`.*
