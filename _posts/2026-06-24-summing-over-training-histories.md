---
title: "Path Integrals for Learning Dynamics"
date: 2026-06-24
layout: post
categories: [technical]
---

During deep learning experimentation, I often seek the comfort of visibility. I log training steps, plot loss curves, inspect gradients, track spectra, and save checkpoints. These tools make training feel observable. But the object being studied is still enormous: millions, billions, and sometimes trillions of coupled parameters evolving under data, architecture, initialization, and optimization.

At that scale, knowing every weight rarely explains the system. The more useful goal is to find a smaller set of quantities that captures the behavior I care about: how fast errors decay, which directions are slow, how perturbations propagate, when randomness averages out, and why a learning curve has the shape it does.

This is where physics feels unusually clarifying. Statistical mechanics was built around the idea that large systems should be understood through collective variables. Instead of following every molecule, one studies pressure, magnetization, correlations, and response. The microscopic details still matter, but the explanation often lives at a macroscopic scale.

Path integrals push this idea into an even cleaner form. What I like about them is the shift in perspective: do not commit too early to one trajectory. Write down the space of possible histories, impose the rules those histories must satisfy, and ask what survives after summing, averaging, or taking a large-system limit.

In quantum mechanics, this is the famous sum over histories weighted by the action. In statistical field theory and dynamical mean-field theory, the same grammar becomes a way to turn high-dimensional dynamics into equations for collective observables. The important move is to start from histories and derive the macroscopic objects that remain after averaging.

This post is about one clean example where that viewpoint becomes concrete. In Bordelon and Pehlevan’s work on disordered dynamics in high dimensions, a path-integral / dynamical mean-field theory calculation compresses a random linear dynamical system into two memory functions:

$$
C(t,t’)=\frac{1}{N}h(t)\cdot h(t’),
\qquad
R(t,t’)=\frac{1}{N}\operatorname{Tr}\frac{\delta h(t)}{\delta j(t’)^\top}.
$$

The first measures temporal similarity across pairs of times. The second measures causal sensitivity across pairs of times. Together, they describe what the system remembers.

I will use one solvable setting to show how the path-integral viewpoint turns high-dimensional random dynamics into macroscopic memory functions, without pretending that neural networks are quantum systems or that a linear GOE model explains modern deep nets.

The arc is:

$$
\text{random matrix spectrum}
\rightarrow
\text{response curve}
\rightarrow
\text{two-time memory surfaces}
\rightarrow
\text{learning curves}.
$$

The GOE warmup is a calibration case. The expected answer is the Wigner semicircle. The interesting part is watching the response function recover that answer.

---

## Order parameters instead of coordinates

Training produces a trajectory:

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_T.
$$

One can track each component of $$\theta_t$$, but the more useful question is which collective observables carry information: correlations between times, response to perturbations, spectra of covariance matrices, and the loss curve itself.

Bordelon and Pehlevan frame DMFT for disordered high-dimensional dynamics in exactly this language. The system has too many coupled coordinates to follow individually. After averaging over randomness and taking the dimension large, the dynamics collapse onto two two-time order parameters:

$$
C(t,t') = \frac{1}{N}\, h(t)\cdot h(t'),
$$

$$
R(t,t') = \frac{1}{N}\,\operatorname{Tr}\frac{\delta h(t)}{\delta j(t')^\top}.
$$

$$C$$ asks how similar the state is at two different times. $$R$$ asks how much a perturbation at one time echoes into another. Correlation is self-memory; response is memory of an external perturbation.

If the goal is to understand dynamics, $$C$$ and $$R$$ are natural objects to study. Two systems can share the same instantaneous error while carrying different memory kernels: $$C$$ records temporal similarity, and $$R$$ records causal sensitivity.

The system has memory, and here memory is a two-time object.

---

## Why path integrals at all?

Feynman’s quantum path integral sums over all paths $$x(t)$$, weighted by an action:

$$
\int \mathcal{D}x\, e^{\frac{i}{\hbar} S[x]}.
$$

In statistical mechanics the same grammar appears with real weights:

$$
\int \mathcal{D}x\, e^{-S[x]}.
$$

The expression is compact almost to the point of being suspicious. All histories appear, and the classical action sits in the exponent. The amplitude is written directly as a sum over possibilities, without first solving an equation of motion and then translating that solution into probabilities.

The classical limit gives the intuition. If the action changes rapidly from one nearby path to another, the phases $$e^{iS/\hbar}$$ rotate quickly and mostly cancel. But near a stationary path, where $$\delta S=0$$, nearby histories have nearly aligned phases and add coherently. The usual classical trajectory appears as the place where the sum over histories stops canceling itself.

That is one reason the path integral is useful: it turns the principle of stationary action from a classical rule into an interference phenomenon. In the classical limit, the path of least action dominates because nearby histories add coherently while the other paths cancel.

The implementation is difficult because the integral is over a space of functions, not a few variables. In quantum field theory, those histories are field configurations. In dynamical mean-field theory, they are trajectories of high-dimensional random systems. In both cases, the compact notation hides the hard part: extracting the macroscopic structure.

The DMFT path integral in Bordelon and Pehlevan is a cousin of Feynman’s original object: a generating functional that integrates over histories $$h(t)$$ and auxiliary response fields $$\hat{h}(t)$$ while enforcing the equations of motion. The shared structure is histories, constraints, actions, and saddle points, now applied to a different physical problem.

Three related objects often get conflated:

* Feynman’s integral: complex weights $$e^{iS/\hbar}$$.
* Statistical field theory: real Boltzmann weights $$e^{-S}$$.
* Dynamical DMFT: enforce $$\dot{h}=-Mh+j$$ inside a path integral, average over disorder, and solve for $$C$$ and $$R$$ at a large-$$N$$ saddle.

Concretely, a path integral converts constraints on microscopic trajectories into an action over macroscopic observables. Schematically, the calculation starts with something like

$$
Z=\int \mathcal{D}h\,\mathcal{D}\hat h\, e^{-S[h,\hat h]}.
$$

After averaging over disorder and introducing order parameters, this becomes a large-$$N$$ integral:

$$
Z=\int \mathcal{D}Q\, e^{-N S[Q]}.
$$

The factor of $$N$$ in the exponent is the important clue. At large dimension, not every macroscopic history contributes equally. The integral concentrates near a saddle. The path integral is a disciplined way to ask:

Which functions of the training history survive when the dimension is large?

This is the conceptual bridge to learning. In quantum mechanics, the path integral asks which histories survive interference. In high-dimensional random dynamics, the DMFT path integral asks which collective histories survive disorder averaging and large-$$N$$ concentration. The microscopic description is too large to interpret directly, so the formalism points toward the macroscopic objects that remain.

---

## The GOE warmup

Bordelon and Pehlevan start with

$$
\frac{d}{dt} h(t) = -M h(t) + j(t),
\qquad h(t) \in \mathbb{R}^N,
$$

where $$M$$ is random and $$j(t)$$ is a source. For the Gaussian orthogonal ensemble, or GOE, $$M$$ is a random symmetric matrix with Gaussian entries, scaled so its eigenvalues remain $$O(1)$$ as $$N$$ grows:

$$
M = \frac{1}{\sqrt{N}} A,
\qquad A = A^\top,
\qquad A_{ij} \sim \mathcal{N}(0,1),
$$

up to the usual symmetrization convention. In the numerics, we symmetrize $$A_0 + A_0^\top$$ and scale so the empirical spectrum sits on $$[-2,2]$$.

GOE is the cleanest possible disordered interaction: symmetric, Gaussian, and analytically solvable. It is a diagnostic case. If the path-integral machinery is doing something meaningful, it should recover the known answer here.

At fixed $$N$$, this is just a linear ODE: diagonalize $$M$$ and integrate. For this warmup, diagonalization is enough to compute everything. That is why it is a good first example: the path-integral / DMFT answer can be checked against a known random-matrix result.

The formalism is built for settings where diagonalization stops being the right language: random features, SGD noise, non-Hermitian Jacobians, and matrices that evolve during training. The GOE case is the controlled place to see the machinery and verify it against the semicircle law.

The important phenomenon is self-averaging. As $$N\to\infty$$, many details of the particular matrix draw disappear, while observables like the spectrum and response converge to deterministic limits.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig1-goe-semicircle.png" alt="GOE eigenvalue histogram with Wigner semicircle overlay" width="520"/>
  <figcaption>Figure 1. Histogram of eigenvalues from one random symmetric NxN matrix against the Wigner semicircle prediction. Individual matrix entries are random, but the bulk eigenvalue density follows a fixed law. This is the static sanity check before looking at time.</figcaption>
</figure>

Before time enters the story, disorder in the entries already produces a deterministic eigenvalue density. Figure 1 is the static hint that averaging works: draw a fresh GOE matrix, histogram its eigenvalues, and the semicircle shows up. If the path-integral / DMFT machinery is doing its job, every time-dependent quantity computed later should be built from this same $$\rho(\lambda)$$.

---

## What the path integral compresses

I read the calculation as five moves. The paper writes the generating functional as

$$
Z = \int \mathcal{D}Q\, e^{-N S[Q]},
$$

dominated by a saddle at large $$N$$. Here is the compressed version.

First, enforce the dynamics. Only histories with

$$
\partial_t h + Mh - j = 0
$$

contribute, via a delta functional.

Second, introduce $$\hat{h}$$. Fourier representation of the delta function brings in an auxiliary field that enforces the equation of motion and defines response.

Third, average over $$M$$. Gaussian averaging produces overlaps of histories. Those overlaps are $$C(t,t’)$$ and $$R(t,t’)$$. Every coordinate was coupled to every other coordinate; after the average, only two-time collective functions remain.

Fourth, take $$N$$ large. The integral is dominated by

$$
\frac{\delta S}{\delta C}=0,
\qquad
\frac{\delta S}{\delta R}=0.
$$

Fifth, obtain a single-site process. A typical coordinate evolves as if it were one-dimensional, driven by noise fixed by $$C$$ and fed back through its past via $$R$$:

$$
\frac{\partial}{\partial t} h(t)
= u(t)
+ \int dt'\, R(t,t') h(t')
+ j(t),
\qquad
u(t) \sim \mathrm{GP}(0, C(t,t')).
$$

The original system had $$N$$ coupled coordinates. In the $$N\to\infty$$ limit, a typical coordinate behaves like a single stochastic process driven by colored noise and delayed self-feedback.

This has the same structure as the path integral, but now in a disordered dynamical system rather than quantum mechanics. Start with many possible microscopic histories, impose the dynamics, average over disorder, and let the large-$$N$$ action tell us which macroscopic history survives.

In the GOE case, the saddle closes further. The response satisfies a self-consistency equation that is much smaller than the original system:

$$
\partial_t R(t,t')
= \delta(t-t')
+ \int_0^t dt''\, R(t,t'')\, R(t'',t').
$$
With time-translation invariance, Fourier transforming gives

$$
i\omega R(\omega)=1+R(\omega)^2.
$$

Choosing the branch with the correct large-$$|\omega|$$ behavior gives

$$
R(\omega)
= \frac{1}{2}
\left[
i\omega+\sqrt{(i\omega)^2-4}
\right],
$$

whose branch cut on $$[-2,2]$$ recovers the semicircle. That is the chain in the GOE warmup:

$$
\text{path integral}
\rightarrow
\text{order parameters}
\rightarrow
\text{closed response equation}
\rightarrow
\text{semicircle}.
$$

The figures below check the last step directly.

---

## Response is the spectrum, read in time

For the linear system, with lag $$\tau=t-t’$$,

$$
R(\tau)
= \frac{1}{N}\operatorname{Tr}e^{-M\tau}
= \int d\lambda\,\rho(\lambda)e^{-\lambda\tau}.
$$

Each eigenvalue contributes a mode $$e^{-\lambda\tau}$$. The response superposes them with weights set by $$\rho(\lambda)$$. A dynamical question becomes a spectral question.

The same object appears in frequency space as the resolvent, or Stieltjes transform:

$$
R(\omega)
= \int d\tau\, R(\tau)e^{-i\omega\tau}
= \frac{1}{N}\operatorname{Tr}(i\omega+M)^{-1}
= \int d\lambda\,\frac{\rho(\lambda)}{i\omega+\lambda}.
$$

The eigenvalue density is encoded in the analytic structure of this function. The branch cut is the spectrum.

For the GOE,

$$
\rho(\lambda)
= \frac{1}{2\pi}\sqrt{4-\lambda^2},
\qquad
\lambda\in[-2,2].
$$

**Stability note.**

Because the GOE spectrum has support on $$[-2,2]$$, the unshifted flow

$$
\dot{h}=-Mh
$$

is not uniformly stable: modes with $$\lambda<0$$ grow under $$e^{-\lambda\tau}$$. For the response-decay numerics, we follow Bordelon and Pehlevan and add a stabilizing shift $$z=2$$:

$$
\dot{h}(t)=-Mh(t)-zh(t),
$$

so

$$
R_z(\tau)
= \int d\lambda\,\rho(\lambda)e^{-(\lambda+z)\tau}.
$$

The unshifted GOE response equation was

$$
i\omega R(\omega)=1+R(\omega)^2.
$$

Adding the stabilizing shift simply shifts the resolvent argument. Equivalently, the shifted response satisfies

$$
(i\omega+z)R_z(\omega)=1+R_z(\omega)^2.
$$

With $$z=2$$, the slowest modes sit at the spectral edge $$\lambda=-2$$, where $$\lambda+z=0$$. The late-time response is critical rather than exponentially decaying. The spectral edge produces the power-law tail

$$
R_z(\tau)\sim \tau^{-3/2}.
$$

What Figure 2 is measuring. Imagine injecting a unit perturbation into the system and asking how much of it remains after a lag $$\tau$$. That scalar is $$R_z(\tau)$$. Because the dynamics are linear, it equals a trace over eigenmodes: each mode $$\lambda$$ contributes $$\rho(\lambda)e^{-(\lambda+z)\tau}$$. Panel (a) plots the full curve on a linear time axis; panel (b) zooms into the late-time tail on log-log axes and compares it with a $$\tau^{-3/2}$$ reference. The orange curve is one finite-$$N$$ draw, and the dashed curve is the semicircle integral.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig2-response-decay.png" alt="Two-panel response decay: linear-time overview and log-log tail" width="640"/>
  <figcaption>Figure 2. Shifted response R<sub>z</sub>(τ) with z = 2.</figcaption>
</figure>

I also checked finite-$$N$$ convergence. As $$N$$ grows from hundreds to thousands, the relative L2 error between $$R_z(\tau)$$ and the semicircle integral falls. Figure 2 already shows the agreement at $$N=4000$$, so I skip a separate plot.

What Figure 3 is showing. The response is an integral over eigenvalues. Figure 3 plots the integrand

$$
W(\lambda,\tau)=\rho(\lambda)e^{-(\lambda+z)\tau}
$$

as a heatmap. The horizontal axis is $$\lambda$$, the vertical axis is $$\tau$$, and color shows how much weight each eigenvalue carries at that lag. At small $$\tau$$, the whole semicircle contributes. As $$\tau$$ grows, the exponential kills everything except modes near $$\lambda=-2$$, so the bright region creeps to the left edge. The curve on the right is $$R_z(\tau)$$ obtained by integrating $$W$$ over $$\lambda$$. It is the same object as Figure 2, now shown as a sum over modes.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig3-mode-decay.png" alt="Heatmap of spectral mode contributions to response" width="640"/>
  <figcaption>Figure 3. Integrand W(lambda, tau) = rho(lambda) exp(-(lambda + z) tau). Bright regions mark which eigenvalues matter at each lag. Marginal curve: Rz(tau) from integrating over lambda.</figcaption>
</figure>

The animation below is the same decomposition, one lag at a time. The orange curve is $$W(\lambda,\tau)$$ at the current $$\tau$$; the gray semicircle behind it is $$\rho(\lambda)$$ for reference. Watch the orange curve narrow toward $$\lambda=-2$$ as $$\tau$$ increases, while the right panel traces out $$R_z(\tau)$$.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/gif-spectral-modes-response.gif" alt="Animation of spectral mode weights narrowing to the edge" width="640"/>
  <figcaption>Animation. Left: integrand W(lambda, tau) at the current lag versus the static semicircle rho(lambda). Right: Rz(tau) accumulated so far. Text banner states the current tau.</figcaption>
</figure>

---

## Two-time memory surfaces

$$C(t,t’)$$ and $$R(t,t’)$$ are functions of two times. History is encoded in relationships between pairs of times, not in a single clock reading.

Correlation records temporal similarity:

$$
C(t,t’)=\frac{1}{N}h(t)\cdot h(t’).
$$

Response records causal sensitivity:

$$
R(t,t’)=\frac{1}{N}\operatorname{Tr}
\frac{\delta h(t)}{\delta j(t’)^\top}.
$$

For symmetric GOE dynamics, there is a special simplification. With the matching linear setup,

$$
C(t,t')
= \operatorname{tr} e^{-M(t+t')}
= R(t+t').
$$

So the correlation depends on the summed time $$t+t’$$. This identity is special to the symmetric linear case. It fails in the more interesting asymmetric and machine-learning settings, where correlation and response carry distinct information.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig4-correlation-heatmap.png" alt="Two-time correlation heatmap and decay slice" width="640"/>
  <figcaption>Figure 4. Left: normalized C(t, t') on a zoomed window. Bright corner means both times are small. Right: C(s) / C(0) versus summed time s = t + t'.</figcaption>
</figure>

Response has a different geometry because it is causal:

$$
R(t,t')=0
\qquad
\text{for }t<t'.
$$

The future cannot affect the past.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig5-response-heatmap.png" alt="Causal two-time response heatmap R(t,t prime)" width="440"/>
  <figcaption>Figure 5. R(t, t') on t, t' in [0, 10]. Empty lower triangle means causality, with no influence backward in time.</figcaption>
</figure>

Two systems can share the same loss at a given step yet carry different memory. $$C$$ tells us how much of the past remains in the state; $$R$$ tells us how a perturbation propagates into the future.

This is one of the places where the path-integral viewpoint earns its keep. It produces the right kind of object: a surface over pairs of times, matching the two-time structure of the system’s memory.

---

## Real spectra relax; imaginary spectra oscillate

The GOE example is symmetric, so its eigenvectors form an orthogonal basis of relaxation modes. If we change the symmetry class, the same response formalism produces a different temporal texture.

Compare symmetric and antisymmetric random matrices:

$$
M_{\mathrm{sym}}
= \frac{A + A^\top}{\sqrt{2N}},
\qquad
M_{\mathrm{anti}}
= \frac{A - A^\top}{\sqrt{2N}}.
$$

Real eigenvalues give exponential relaxation. Purely imaginary eigenvalues give oscillation. Figure 6 contrasts the two: symmetric $$M$$ has a real spectrum and a decaying response, while antisymmetric $$M$$ has eigenvalues on the imaginary axis and an oscillatory response. The random draw $$A$$ is the same; the symmetrization changes the memory of perturbations.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig6-sym-antisym.png" alt="Symmetric vs antisymmetric eigenvalues and response curves" width="640"/>
  <figcaption>Figure 6. Top: symmetric M with real eigenvalues and decaying Rz(tau). Bottom: antisymmetric M with imaginary eigenvalues and oscillatory response.</figcaption>
</figure>

---

## From GOE to learning curves

Bordelon and Pehlevan apply the same template to linear regression, random features, kernel methods, and deep linear networks. The bridge I understand best is random linear regression.

Data matrix:

$$
\Psi \in \mathbb{R}^{P \times N}.
$$

Targets:

$$
y = \frac{1}{\sqrt{N}} \Psi \beta^\star + \epsilon.
$$

Model:

$$
f = \frac{1}{\sqrt{N}} \Psi w.
$$

Train and test loss:

$$
\hat{L}(t) = \frac{1}{P} \left| \frac{1}{\sqrt{N}} \Psi w(t) - y \right|^2,
\qquad
L(t) = \frac{1}{N} | w(t) - \beta^\star |^2 + \sigma^2.
$$

Gradient flow is governed by the empirical covariance

$$
M = \frac{1}{P} \Psi^\top \Psi.
$$

The residual

$$
h(t) = \beta^\star - w(t)
$$

evolves under a random-matrix-driven linear dynamics, now governed by a Wishart covariance matrix rather than a GOE interaction matrix.

In regression, $$M$$ is the empirical data covariance. Its eigenvalues are effective learning rates for different error modes. Large eigenvalues decay quickly; small eigenvalues are slow directions. When $$\alpha=P/N$$ crosses the interpolation threshold, the spectrum changes shape and bias-variance behavior becomes delicate.

When loss is plotted against time, a microscopic training trajectory is being projected onto a scalar. Figure 7 gives a concrete instance: gradient descent on random linear regression at several sample-complexity ratios $$\alpha=P/N$$. Solid curves are train loss; dashed curves are test loss. Color encodes $$\alpha$$. The same DMFT logic applies, with a Wishart spectrum in place of the GOE spectrum.

<figure class="blog-figure">
  <img src="/assets/img/blog/path-integral/fig7-linear-regression.png" alt="Train and test loss curves for random linear regression at several alpha=P/N" width="520"/>
  <figcaption>Figure 7. Random linear regression with label noise σ = 0.1. Solid: train loss. Dashed: test loss. Color: α = P/N.</figcaption>
</figure>

The paper goes further in random feature models near interpolation, where test loss can be non-monotonic and the full two-time $$C(t,t’)$$ carries information beyond the spectrum alone. I have not reproduced those curves here; this post stays on the GOE example where we have a calibration case.

---

## Limits of the GOE warmup

The GOE warmup is linear, Gaussian, symmetric, and time-translation invariant. Real networks violate all of that: representations move, Jacobians are non-Hermitian, SGD adds noise, and feature learning changes effective matrices during training.

The example is still useful because it teaches the basic move: replace microscopic coordinates with self-consistent two-time observables. The rest of DMFT is about making that move survive in harder systems.

---

## What I take from it

The full derivation is formal, but the conceptual target is concrete. Start with $$N$$ coupled variables. Average over disorder. Take $$N$$ large. Identify the self-consistent memory functions that survive. In the GOE warmup, this procedure recovers a spectral law we already trust. In learning problems, the same language gives a way to treat train and test curves as macroscopic observables of high-dimensional dynamics.

The reason I keep coming back to the path integral is that it is both brutally simple and technically deep. The simple part is the philosophy: write the space of histories, impose the rules, and ask what survives without committing to a single microscopic trajectory too early. The difficult part is that the space of histories is enormous, so extracting the answer requires symmetry, approximation, perturbation, or a saddle point.

That tension is what makes the formalism useful. It starts from a simple instruction, sum over possibilities, and turns it into machinery for extracting macroscopic structure. In the DMFT setting, the same idea appears as a sum over trajectories, an average over disorder, and a large-$$N$$ saddle that reveals the memory functions.

Plotting a learning curve is already a macroscopic move. The path integral makes that move systematic: which histories matter, which averages survive, and which functions remember.

For high-dimensional learning systems, those surviving functions are often correlations, responses, spectra, and losses. They are the dynamics projected to the scale where understanding becomes possible.

---

## References

* Feynman, R. P., & Hibbs, A. R. (1965). Quantum Mechanics and Path Integrals. McGraw-Hill.
* Feynman, R. P. The Feynman Lectures on Physics, Vol. II, Chapter 19: “The Principle of Least Action.” Online edition.
* Bordelon, B., & Pehlevan, C. (2026). Disordered Dynamics in High Dimensions: Connections to Random Matrices and Machine Learning. arXiv:2601.01010.
* Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations. arXiv:1312.6120.
* Schoenholz, S. S., Gilmer, J., Ganguli, S., & Sohl-Dickstein, J. (2017). Deep Information Propagation. International Conference on Learning Representations. OpenReview.
* Poole, B., Lahiri, S., Raghu, M., Sohl-Dickstein, J., & Ganguli, S. (2016). Exponential expressivity in deep neural networks through transient chaos. Advances in Neural Information Processing Systems 29. NeurIPS.
* Sompolinsky, H., Crisanti, A., & Sommers, H. J. (1988). Chaos in random neural networks. Physical Review Letters, 61, 259-262. doi:10.1103/PhysRevLett.61.259.
* Advani, M. S., Saxe, A. M., & Sompolinsky, H. (2020). High-dimensional dynamics of generalization error in neural networks. Neural Networks, 132, 428-446. doi:10.1016/j.neunet.2020.08.022.
* Mézard, M., & Montanari, A. (2009). Information, Physics, and Computation. Oxford University Press.
