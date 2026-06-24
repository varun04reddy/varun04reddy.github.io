---
title: "Summing Over Training Histories"
date: 2026-06-24
layout: post
description: "Why the Feynman path integral is a useful way to think about high-dimensional learning dynamics, through the GOE example in Bordelon and Pehlevan."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, path-integrals, random-matrices]
---

The first time I read about Feynman's path integral, I did not care about quantum mechanics at all. I cared about the *picture*. Instead of committing to one trajectory, you treat whole histories as legitimate objects. You weight them, average over them, and ask which macroscopic description survives. That idea stuck with me long before I understood any of the formalism.

What I find fascinating is how often the same move shows up when physics meets machine learning. Statistical mechanics was built for systems with millions or trillions of particles, where tracking every coordinate is impossible. You replace microscopic state with order parameters: pressure, magnetization, correlation functions, response functions. Things that survive averaging. Things that *remember*.

Neural networks are not gas in a box, but they have the same scaling problem. Too many weights, too many activations, too many gradient coordinates. When we plot a training curve, we are already admitting that the microscopic description is too large. We throw away almost everything and watch one scalar evolve in time.

This post is my attempt to make that admission precise, using one clean example from [Bordelon and Pehlevan (2026)](https://arxiv.org/abs/2601.01010). They show how a path-integral / dynamical mean-field theory (DMFT) calculation compresses a high-dimensional linear dynamical system into two memory functions. The GOE warmup is the answer key: we know what the spectrum should be (the Wigner semicircle), and we can check numerically that the response function recovers it.

That is what I am trying to show here. Not a survey of physics and AI. One worked example where the path integral tells us how to think about linear dynamical systems, and where the figures are the argument.

---

## Order parameters instead of coordinates

Training produces a trajectory

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_T.
$$

The CS instinct is to ask what each coordinate of $$\theta_t$$ is doing. The physics instinct is to ask what macroscopic observables of the trajectory carry information: correlations between times, response to perturbations, spectra of covariance matrices, the loss curve itself.

[Bordelon and Pehlevan](https://arxiv.org/abs/2601.01010) frame DMFT for disordered high-dimensional dynamics in exactly this language. The system has too many coupled coordinates to follow individually. After averaging over randomness and taking the dimension large, the dynamics collapse onto two two-time order parameters:

$$
C(t,t') = \frac{1}{N}\, h(t)\cdot h(t'),
$$

$$
R(t,t') = \frac{1}{N}\,\operatorname{Tr}\frac{\delta h(t)}{\delta j(t')^\top}.
$$

$$C$$ asks how similar the state is at two different times. $$R$$ asks how much a perturbation at one time echoes into another. Correlation is self-memory. Response is memory of an external poke. The paper emphasizes that these functions encode how long perturbations are remembered, which is exactly the quantity you want when studying learning dynamics at scale.

---

## Why path integrals at all?

Feynman's quantum path integral sums over all paths $$x(t)$$, weighted by an action:

$$
\int \mathcal{D}x\; e^{\frac{i}{\hbar} S[x]}.
$$

In statistical mechanics the same grammar appears with real weights:

$$
\int \mathcal{D}x\; e^{-S[x]}.
$$

The DMFT path integral in Bordelon and Pehlevan is a third cousin: a generating functional that integrates over histories $$h(t)$$ and auxiliary response fields $$\hat{h}(t)$$ while enforcing the equations of motion. The analogy is the *structure* (histories, constraints, saddle points), not the claim that SGD is quantum mechanics.

Three related objects often get conflated:

* Feynman's integral: complex weights $$e^{iS/\hbar}$$.
* Statistical field theory: real Boltzmann weights $$e^{-S}$$.
* Dynamical DMFT: enforce $$\dot{h} = -Mh + j$$ inside a path integral, average over disorder, solve for $$C$$ and $$R$$ at a large-$$N$$ saddle.

I am not saying neural networks are quantum systems. I am saying the path integral is a disciplined way to ask: *which functions of the training history survive when the dimension is large?*

---

## The GOE warmup: a linear system we can check

Bordelon and Pehlevan start with

$$
\frac{d}{dt} h(t) = -M h(t) + j(t),
\qquad h(t) \in \mathbb{R}^N,
$$

where $$M$$ is random and $$j(t)$$ is a source. For the Gaussian orthogonal ensemble (GOE) warmup,

$$
M = \frac{1}{\sqrt{N}} A,
\qquad A = A^\top,
\qquad A_{ij} \sim \mathcal{N}(0,1),
$$

up to the usual symmetrization convention (we symmetrize $$A_0 + A_0^\top$$ and scale so the empirical spectrum sits on $$[-2,2]$$).

At fixed $$N$$ this is just a linear ODE. Diagonalize $$M$$, integrate, done. For this warmup, diagonalization is enough to compute everything. That is *why* it is a good first example: we can check the path-integral / DMFT answer against a known random-matrix result.

The point of the formalism is not to solve this one system the hard way. The point is to introduce machinery that still works when diagonalization is not the right language: random features, SGD noise, non-Hermitian Jacobians, matrices that evolve during training. The GOE case is the controlled place to see the machinery and verify it against the semicircle law.

The theory asks what happens typically as $$N \to \infty$$ after averaging over $$M$$. That is where $$C$$ and $$R$$ become the actual variables.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig1-goe-semicircle.png" alt="GOE eigenvalue histogram with Wigner semicircle overlay" width="520"/>
  <figcaption>Figure 1. Histogram of eigenvalues from one random symmetric N×N matrix (purple bars) against the Wigner semicircle prediction (gold curve). Individual matrix entries are random; the bulk eigenvalue density is not. This is the static sanity check before we look at time.</figcaption>
</figure>

Before we even talk about time, disorder in the entries produces a deterministic eigenvalue density. Figure 1 is the static hint that averaging works: draw a fresh GOE matrix, histogram its eigenvalues, and the semicircle shows up. If the path-integral / DMFT machinery is doing its job, every time-dependent quantity we compute later should be built from this same ρ(λ).

---

## What the path integral compresses

I read the calculation as five moves. The paper writes the generating functional as

$$
Z = \int \mathcal{D}Q\, e^{-N S[Q]},
$$

dominated by a saddle at large $$N$$. Here is the compressed version.

**Enforce dynamics.** Only histories with $$\partial_t h + Mh - j = 0$$ contribute, via a delta functional.

**Introduce $$\hat{h}$$.** Fourier representation of the delta function brings in an auxiliary field. It is not a new physical neuron; it enforces the equation of motion and defines response.

**Average over $$M$$.** Gaussian averaging produces overlaps of histories. Those overlaps are $$C(t,t')$$ and $$R(t,t')$$. Every coordinate was coupled to every other coordinate; after the average, only two-time collective functions remain.

**Take $$N$$ large.** The integral is dominated by $$\delta S / \delta C = 0$$ and $$\delta S / \delta R = 0$$.

**Single-site process.** A typical coordinate evolves as if one-dimensional, driven by noise fixed by $$C$$ and fed back through its past via $$R$$:

$$
\frac{\partial}{\partial t} h(t) = u(t) + \int dt'\, R(t,t') h(t') + j(t),
\qquad u(t) \sim \mathrm{GP}(0, C(t,t')).
$$

In the GOE case the saddle closes further. The response satisfies a self-consistency equation that is much smaller than the original system:

$$
\partial_t R(t,t') = \delta(t-t') + \int_0^t dt''\, R(t,t'')\, R(t'',t').
$$

With time-translation invariance, Fourier transforming gives

$$
i\omega R(\omega) = 1 + R(\omega)^2.
$$

Inverting this resolvent relation yields the Wigner semicircle as the spectral density of $$M$$. That is the chain in the paper's GOE warmup: path integral $$\to$$ order parameters $$\to$$ closed response equation $$\to$$ semicircle. The figures below check the last step directly.

---

## Response is the spectrum, read in time

For the linear system, with lag $$\tau = t - t'$$,

$$
R(\tau) = \frac{1}{N}\operatorname{Tr} e^{-M\tau}
= \int d\lambda\,\rho(\lambda)\,e^{-\lambda\tau}.
$$

Each eigenvalue contributes a mode $$e^{-\lambda\tau}$$. The response superposes them with weights set by $$\rho(\lambda)$$. A dynamical question becomes a spectral question.

For the GOE,

$$
\rho(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}, \qquad \lambda \in [-2,2].
$$

**Stability note.** Because the GOE spectrum has support on $$[-2,2]$$, the unshifted flow $$\dot{h} = -Mh$$ is not uniformly stable: modes with $$\lambda < 0$$ grow under $$e^{-\lambda\tau}$$. For the response-decay numerics we follow Bordelon and Pehlevan and add a stabilizing shift $$z = 2$$:

$$
\dot{h}(t) = -M h(t) - z h(t),
\qquad
R_z(\tau) = \int d\lambda\,\rho(\lambda)\,e^{-(\lambda + z)\tau}.
$$

With $$z = 2$$ the slowest modes sit at the spectral edge $$\lambda = -2$$, where $$\lambda + z = 0$$. The late-time response is critical (power-law), not exponentially decaying. Their Figure 2 shows this shifted response and the $$\tau^{-3/2}$$ tail.

**What Figure 2 is measuring.** Imagine injecting a unit perturbation into the system and asking how much of it remains after a lag $$\tau$$. That scalar is $$R_z(\tau)$$. Because the dynamics are linear, it equals a trace over eigenmodes: each mode $$\lambda$$ contributes $$\rho(\lambda)\, e^{-(\lambda+z)\tau}$$. Panel (a) plots the full curve on a linear time axis so you can see the decay directly. Panel (b) zooms into the late-time tail on log–log axes and overlays a $$\tau^{-3/2}$$ reference. The orange curve is one finite-$$N$$ draw; the dashed curve is the semicircle integral (the large-$$N$$ answer). They should sit on top of each other, and the tail should parallel the dotted line.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig2-response-decay.png" alt="Two-panel response decay: linear-time overview and log-log tail" width="640"/>
  <figcaption>Figure 2. Shifted response R<sub>z</sub>(τ) with z = 2. Left (a): semilog plot on linear τ — how memory of a perturbation decays. Right (b): log–log tail vs. theory; the dotted line is a τ<sup>−3/2</sup> reference slope. Orange: one GOE sample at N = 4000. Dashed: semicircle integral.</figcaption>
</figure>

I also checked that finite dimension is not fooling us: for each matrix size N we can compare the full R<sub>z</sub>(τ) curve to the semicircle integral, and the relative L2 error falls as N grows from hundreds to thousands. That is the numerical version of "take N large after averaging," but the plot is not essential — Figure 2 already shows agreement at N = 4000.

**What Figure 3 is showing.** The response is an integral over eigenvalues. Figure 3 plots the integrand W(λ, τ) = ρ(λ) e<sup>−(λ+z)τ</sup> as a heatmap: horizontal axis is λ, vertical axis is τ, color is how much weight that eigenvalue carries at that lag. At small τ the whole semicircle contributes (bright band across all λ). As τ grows, the exponential kills everything except modes near λ = −2, so the bright region creeps to the left edge. The yellow curve on the right is R<sub>z</sub>(τ) obtained by integrating W over λ — the same object as Figure 2, but seen as a sum over modes.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig3-mode-decay.png" alt="Heatmap of spectral mode contributions to response" width="640"/>
  <figcaption>Figure 3. Integrand W(λ, τ) = ρ(λ) e<sup>−(λ+z)τ</sup>. Bright regions mark which eigenvalues matter at each lag. Marginal curve (right): R<sub>z</sub>(τ) from integrating over λ.</figcaption>
</figure>

The animation below is the same decomposition, one lag at a time. The orange curve is W(λ, τ) at the current τ; the gray semicircle behind it is ρ(λ) for reference. Watch the orange curve narrow toward λ = −2 as τ increases, while the right panel traces out R<sub>z</sub>(τ).

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/gif-spectral-modes-response.gif" alt="Animation of spectral mode weights narrowing to the edge" width="640"/>
  <figcaption>Animation. Left: integrand W(λ, τ) at the current lag (orange) vs. the static semicircle ρ(λ) (purple). Right: R<sub>z</sub>(τ) accumulated so far. Text banner states the current τ.</figcaption>
</figure>

---

## Two-time memory surfaces

$$C(t,t')$$ and $$R(t,t')$$ are functions of two times, not scalars. The system has a history.

For symmetric GOE dynamics, $$C(t,t')$$ depends on $$s = t + t'$$ in the large-$$N$$ limit. That means correlation is large only when *both* times are early (small $$s$$), not when one time is early and the other late. Figure 4 zooms to $$t, t' \in [0,8]$$ where the structure is visible. The left panel is the full two-time surface; the right panel collapses it to a one-dimensional slice along $$s = t + t'$$.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig4-correlation-heatmap.png" alt="Two-time correlation heatmap and decay slice" width="640"/>
  <figcaption>Figure 4. Left: normalized C(t, t′) on a zoomed window. Bright corner = both times small. Right: C(s)/C(0) vs. summed time s = t + t′.</figcaption>
</figure>

Figure 5 is the response surface R(t, t′). Unlike correlation, response is causal: a perturbation at t′ can only affect later times t ≥ t′. That is why the triangle below the diagonal is empty. The bright band along the diagonal is the immediate echo of a poke; it fades as you move away from the diagonal.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig5-response-heatmap.png" alt="Causal two-time response heatmap R(t,t prime)" width="440"/>
  <figcaption>Figure 5. R(t, t′) on t, t′ ∈ [0, 10]. Empty lower triangle = causality (no influence backward in time).</figcaption>
</figure>

---

## Real spectra relax; imaginary spectra oscillate

Changing the symmetry of $$M$$ changes the texture of time:

$$
M_{\mathrm{sym}} = \frac{A + A^\top}{\sqrt{2N}},
\qquad
M_{\mathrm{anti}} = \frac{A - A^\top}{\sqrt{2N}}.
$$

Real eigenvalues give exponential relaxation. Purely imaginary eigenvalues give oscillation. Figure 6 contrasts the two: symmetric M has real spectrum and a decaying response (top row); antisymmetric M has eigenvalues on the imaginary axis and an oscillatory response (bottom row). Same random draw A, different symmetrization — different memory of perturbations.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig6-sym-antisym.png" alt="Symmetric vs antisymmetric eigenvalues and response curves" width="640"/>
  <figcaption>Figure 6. Top: symmetric M — real eigenvalues, decaying R<sub>z</sub>(τ). Bottom: antisymmetric M — imaginary eigenvalues, oscillatory response.</figcaption>
</figure>

---

## From GOE to learning curves

Bordelon and Pehlevan apply the same template to linear regression, random features, kernel methods, and deep linear networks. The bridge I understand best is random linear regression.

Data matrix $$\Psi \in \mathbb{R}^{P \times N}$$. Targets

$$
y = \frac{1}{\sqrt{N}} \Psi \beta^\star + \epsilon.
$$

Model

$$
f = \frac{1}{\sqrt{N}} \Psi w.
$$

Train and test loss:

$$
\hat{L}(t) = \frac{1}{P} \left\| \frac{1}{\sqrt{N}} \Psi w(t) - y \right\|^2,
\qquad
L(t) = \frac{1}{N} \| w(t) - \beta^\star \|^2 + \sigma^2.
$$

Gradient flow is governed by the empirical covariance

$$
M = \frac{1}{P} \Psi^\top \Psi.
$$

The residual $$h(t) = \beta^\star - w(t)$$ evolves under the same random-matrix-driven linear dynamics as the GOE toy model.

In the GOE example, $$M$$ was an abstract interaction matrix. In regression, $$M$$ is the data covariance (Wishart). Its eigenvalues are effective learning rates for different error modes. Large eigenvalues decay quickly; small eigenvalues are slow directions. When $$\alpha = P/N$$ crosses the interpolation threshold, the spectrum develops a bulk edge and bias–variance behavior becomes delicate.

When we plot loss versus time, we are already doing statistical physics: throwing away microscopic coordinates and watching an order parameter. A training curve is not just a log file. It is a projection of high-dimensional dynamics onto a scalar. Figure 7 is a concrete instance: gradient descent on random linear regression at several sample-complexity ratios α = P/N. Solid curves are train loss; dashed curves are test loss. Color encodes α. The same DMFT logic applies — the empirical covariance Ψ<sup>⊤</sup>Ψ / P plays the role of M — but here the spectrum is Wishart rather than GOE.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig7-linear-regression.png" alt="Train and test loss curves for random linear regression at several alpha=P/N" width="520"/>
  <figcaption>Figure 7. Random linear regression with label noise σ = 0.1. Solid: train loss. Dashed: test loss. Color: α = P/N.</figcaption>
</figure>

The paper goes further in random feature models near interpolation, where test loss can be non-monotonic and the spectrum alone is not enough. That is where the full two-time $$C(t,t')$$ matters. I have not reproduced those curves here; this post stays on the GOE example where we have the answer key.

---

## What this example does not show

The GOE warmup is linear, Gaussian, symmetric, and time-translation invariant. Real networks violate all of that: representations move, Jacobians are non-Hermitian, SGD adds noise, feature learning changes effective matrices during training. The example is still useful because it teaches the basic move: replace microscopic coordinates by self-consistent two-time observables. The rest of DMFT is making that move survive in harder systems.

---

## What I take from it

I still find the full path-integral derivation intimidating. But the GOE warmup gives a concrete target: start with $$N$$ coupled variables, average over disorder, take $$N$$ large, end with self-consistent memory functions, check that they reproduce a spectral law you already trust.

When we plot a learning curve, we are already admitting that the microscopic description is too large. The path integral makes that admission systematic. It asks which histories matter, which averages survive, and which functions remember.

For high-dimensional learning systems, those functions are often correlations, responses, spectra, and losses. They are not merely summaries of the dynamics. They are the dynamics, seen at the scale where understanding becomes possible.

---

### References

* Feynman, R. P., & Hibbs, A. R. (1965). *Quantum Mechanics and Path Integrals*. McGraw-Hill.

* Bordelon, B., & Pehlevan, C. (2026). Disordered Dynamics in High Dimensions: Connections to Random Matrices and Machine Learning. [arXiv:2601.01010](https://arxiv.org/abs/2601.01010).

* Mezard, M., & Montanari, A. (2009). *Information, Physics, and Computation*. Oxford University Press.

*Figures: `tools/path-integral-dmft/generate_figures.py`. PNG exports use white figure panels so axes stay readable in both light and dark site themes.*
