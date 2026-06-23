---
title: "Training at the Critical Point"
date: 2026-06-22
layout: post
description: "Deep learning is easier to reason about with order parameters and an empirical phase map of training regimes."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, neural-collapse, grokking, optimization]
---

Most of what we actually watch during training is a single scalar: loss going down, accuracy going up. That summary hides a lot. The same flat loss curve can correspond to weights still drifting, representations reorganizing, sharpness riding a stability limit, or a train-test gap that has not yet closed. Learning dynamics live in a huge parameter space; if you try to reason weight-by-weight, intuition stalls quickly.

A physics habit helps here, not because SGD is a thermal system, but because physicists long ago learned to work around microscopic complexity. They ask which coarse variables track the behavior, where those variables change character, and which knobs move you across those boundaries. Applied to deep learning, that means naming macroscopic variables, mapping training regimes, and treating a run as a trajectory through those regimes rather than a black-box descent to a minimum. The organizing objects in this post are an empirical phase map (which regime dominates for which knobs?) and a small dashboard of order parameters (which regime am I in at step $$t$$?).

I am not claiming a literal correspondence with statistical mechanics. Mini-batch SGD has no Gibbs measure, and "temperature" is at best a loose metaphor. What carries over is a way of building intuition: compress the dynamics, locate qualitative transitions, and treat the transitions as the things worth explaining. When generalization surprises you or optimization stalls, the question "what phase is this run in?" is often more informative than "what is the loss?"

---

## Order parameters

In statistical physics, an order parameter is a coarse variable that distinguishes phases: magnetization $$m = \langle s \rangle$$ for an Ising ferromagnet, density $$\rho$$ for a fluid. You do not need every spin or molecule to know whether the system is ice or water. You need a scalar whose value or kinetics change sharply at a boundary.

Training is a discrete dynamical system on parameters $$\theta \in \mathbb{R}^p$$,

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t),
\qquad
\mathcal{B}_t \subset \{(x_i, y_i)\}_{i=1}^n,
$$

where $$\mathcal{L}$$ is the mini-batch loss and $$\eta$$ the learning rate. The microscopic state is $$\theta_t$$ (millions of weights). The macroscopic question is: which training phase is $$\theta_t$$ in?

Formally, fix a predictor $$f_\theta : \mathcal{X} \to \mathcal{Y}$$ and empirical risk

$$
\mathcal{R}_n(\theta) = \frac{1}{n} \sum_{i=1}^n \ell\bigl(f_\theta(x_i), y_i\bigr).
$$

An order parameter is any low-dimensional functional $$\Phi(\theta_t, \mathcal{D}, t)$$ whose trajectory marks a regime change more clearly than $$\mathcal{R}_n(\theta_t)$$ alone. Useful gauges tend to be monotone or sharply transitional within a regime, comparable across runs, and cheap enough to log every step.

For deep networks, the dashboard I use:

| Symbol | Definition | What it marks |
|---|---|---|
| $$\chi(t)$$ | $$\eta \, \lambda_{\max}(H(\theta_t)) / 2$$ | Optimizer stability (edge of stability) |
| $$m_{\text{NC}}(t)$$ | $$1 - \sigma_W^2 / (\sigma_B^2 + \epsilon)$$ | Terminal representation geometry (neural collapse) |
| $$g(t)$$ | $$A_{\text{train}}(t) - A_{\text{test}}(t)$$ | Memorization vs generalization gap (grokking) |
| Interpolation | $$\mathcal{R}_n(\theta) \to 0$$, $$N_{\text{eff}} \approx n$$ | Capacity boundary (double descent) |

Two auxiliary gauges worth logging when you build a phase map:

$$
d_\theta(t) = \frac{\|\theta_t - \theta_0\|_2}{\|\theta_0\|_2 + \epsilon},
\qquad
d_h(t) = \mathbb{E}_{x \sim \mathcal{D}}\bigl[\|h_{\theta_t}(x) - h_{\theta_0}(x)\|_2^2\bigr],
$$

where $$h_\theta(x)$$ is a penultimate representation. Small $$d_h$$ with falling loss marks a lazy (NTK-like) regime: the function moves, representations barely move. Large $$d_h$$ marks feature learning. On a width $$\times$$ learning-rate map, lazy and rich regions often sit adjacent to underfitting and EOS bands; they are substructure inside the stable feature-learning phase, not separate topics.

These gauges are not a complete theory. They are the minimum set that turns a loss curve into a phase diagnosis: underfit, interpolating, riding stability, collapsing, grokking.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig01-micro-macro.png" alt="Microscopic loss trajectory and macroscopic order parameters" width="900"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1. One MNIST MLP run: batch loss (left) and macroscopic gauges sharpness ratio χ and collapse order m<sub>NC</sub> (right). Loss alone does not tell you whether χ is pinned at 1 or m<sub>NC</sub> is still climbing.</figcaption>
</figure>

---

## The phase map

A phase map is an empirical chart: for fixed task and architecture, color each cell of a hyperparameter grid by a late-time observable (test accuracy, test loss, or an order parameter at step $$T$$). Here I sweep hidden width $$w$$ and learning rate $$\eta$$ on MNIST with a tanh MLP and color by validation accuracy after training.

Mathematically, think of a map

$$
\Phi : (w, \eta) \mapsto \bigl(A_{\text{test}}(T; w, \eta),\; \chi(T; w, \eta),\; m_{\text{NC}}(T; w, \eta)\bigr).
$$

The scalar heatmap is one projection of $$\Phi$$. Boundaries in the $$(w, \eta)$$ plane are contours where the dominant failure mode changes: from "cannot fit" to "fits but unstable" to "fits stably." This is not a universal law of deep learning. It is a local chart for one architecture-task pair, analogous to a phase diagram cut at fixed pressure: useful precisely because it is tied to the system you actually train.

Building the map is how the physics perspective pays off in practice. Instead of memorizing a list of phenomena in isolation, you see where double descent, edge-of-stability training, and representation collapse sit relative to the knobs you tune. Intuition becomes spatial: move learning rate up and you cross into a different dynamical regime; widen the model and you cross an interpolation boundary.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig02-phase-diagram.png" alt="Phase diagram of width vs learning rate" width="750"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2. Empirical phase map from a width × learning-rate sweep on MNIST MLPs (test accuracy at end of training).</figcaption>
</figure>

Rough regions on this map:

- Underfitting: too narrow or $$\eta$$ too small. $$\mathcal{R}_n$$ stays high, $$d_h \approx 0$$, representations frozen relative to initialization.
- Stable feature learning: wide enough to fit, moderate $$\eta$$, $$\chi(t) < 1$$ throughout. Accuracy rises smoothly; $$d_h$$ grows.
- Edge of stability: large $$\eta$$ drives $$\chi(t) \to 1$$. Batch loss oscillates while $$\mathcal{R}_n$$ and $$A_{\text{test}}$$ can still improve. Optimization is not gradient descent on a fixed quadratic; it is a nonlinear dynamical system constrained by a stability inequality.
- Unstable / divergent: $$\chi(t) > 1$$ persistently. $$\theta_t$$ leaves the valid basin; loss and accuracy diverge or stall.

The knobs that move you between regions: width $$w$$ (capacity), dataset size $$n$$, $$\eta$$, batch size (noise scale), weight decay $$\lambda$$, and training time $$T$$. What transfers from physics is the method: identify regimes, measure order parameters, locate boundaries, then ask which microscopic mechanism (implicit bias, representation drift, sharpness adaptation) enforces each boundary.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/phase-transition.gif" alt="Animated grokking transition on modular addition mod 97" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Grokking on modular addition (mod 97): train accuracy rises first; test accuracy follows after a long plateau.</figcaption>
</figure>

---

## Boundaries and gauges on the map

Each order parameter marks a boundary or trajectory feature on the phase map. They connect abstract "regimes" to quantities you can plot from a single run.

### Interpolation and double descent

Classical learning theory predicts a U-shaped risk curve in model complexity: small models underfit (high bias), large models overfit (high variance). Write population risk $$\mathcal{R}(f) = \mathbb{E}_{(x,y)}[\ell(f(x), y)]$$ and empirical risk $$\mathcal{R}_n(f)$$. The classical story optimizes a tradeoff between fitting $$\mathcal{R}_n$$ and controlling complexity.

In overparameterized networks, test error can rise near the interpolation threshold (smallest capacity at which training error reaches zero) and fall again in the overparameterized regime. Belkin et al. called this double descent. Let $$N_{\text{eff}}$$ denote effective capacity (parameter count or norm-based proxy) and $$n$$ the number of training examples. The critical band sits near

$$
N_{\text{eff}} \approx n,
\qquad
\mathcal{R}_n(\theta) \longrightarrow 0.
$$

On the phase map, this is primarily a vertical boundary in width: crossing it changes the solution set

$$
\mathcal{M}_0 = \{\theta : \mathcal{R}_n(\theta) = 0\}
$$

from empty (underfit) to high-dimensional (many interpolating solutions). Generalization is no longer "pick the minimizer of $$\mathcal{R}_n$$"; it is "which point in $$\mathcal{M}_0$$ does SGD reach, and how fast?" The spike in test error near interpolation is the signature that counting parameters is not enough: the optimizer's implicit bias selects among $$\mathcal{M}_0$$, and that selection is fragile near the threshold.

Past interpolation, test error often decreases with $$N_{\text{eff}}$$ even as the model can memorize arbitrary labels. A useful decomposition at fixed $$n$$:

$$
\mathcal{R}_{\text{test}}(\theta) \approx \underbrace{\mathcal{R}_{\text{approx}}(N_{\text{eff}})}_{\text{expressivity floor}} + \underbrace{\mathcal{R}_{\text{estim}}(n, N_{\text{eff}})}_{\text{finite-sample / optimization}} + \underbrace{\mathcal{R}_{\text{opt}}(\text{trajectory})}_{\text{implicit bias}}.
$$

Double descent is the statement that $$\mathcal{R}_{\text{estim}}$$ is not monotone in $$N_{\text{eff}}$$ when $$\mathcal{R}_n \to 0$$ is achievable.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig03-double-descent.png" alt="Double descent curve" width="750"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3. Train and test loss vs parameter count on an MNIST width sweep. Vertical marker: smallest width that interpolates. The spike is the interpolation boundary projected onto one axis.</figcaption>
</figure>

### Sharpness ratio χ and the edge of stability

Near a critical point $$\theta^\star$$, gradient descent on a quadratic Taylor model $$\mathcal{L}(\theta) \approx \mathcal{L}(\theta^\star) + \frac{1}{2}(\theta - \theta^\star)^\top H (\theta - \theta^\star)$$ with Hessian $$H$$ is stable when all eigenvalues $$\lambda_i(H)$$ satisfy

$$
\eta \, \lambda_i(H) < 2 \quad \forall i.
$$

The tightest constraint is the top eigenvalue $$\lambda_{\max}(H)$$. Cohen et al. observed that full-batch or large-batch training of deep networks often rides this stability boundary rather than staying deep in the interior $$\eta \lambda_{\max} \ll 2$$. Define the sharpness ratio

$$
\chi(t) = \frac{\eta \, \lambda_{\max}\bigl(H(\theta_t)\bigr)}{2}.
$$

- Stable interior: $$\chi(t) < 1$$. Local linearization predicts monotone descent.
- Edge of stability (EOS): $$\chi(t) \approx 1$$. Loss can oscillate on short horizons while a longer-horizon average still decreases. Empirically $$\lambda_{\max}$$ can track $$2/\eta$$ as weights move, keeping $$\chi$$ pinned.
- Unstable: $$\chi(t) > 1$$ persistently. Linearized dynamics amplify errors; training breaks down unless other effects (batch noise, projection) restore stability.

On the phase map, large $$\eta$$ moves you horizontally into the EOS band. This is a dynamical boundary: the same architecture at the same width can be stable at $$\eta = 0.01$$ and divergent at $$\eta = 0.1$$. EOS matters for theory because it shows optimization and generalization are coupled: the trajectory is constrained by a local curvature invariant, not by $$\nabla \mathcal{R}_n = 0$$ alone.

A practical estimator at step $$t$$ uses a few power iterations on $$H v$$ via Hessian-vector products. Logging $$\chi(t)$$ alongside loss separates "loss plateau because stuck" from "loss noisy because sharpness-limited."

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig04-edge-of-stability.png" alt="Edge of stability" width="850"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 4. Loss and sharpness ratio χ during SGD on an MNIST MLP. Shaded band: EOS region χ ≈ 1. Notice epochs where loss wiggles while accuracy still climbs.</figcaption>
</figure>

### Collapse order m<sub>NC</sub>

After interpolation, penultimate-layer features $$h_\theta(x) \in \mathbb{R}^d$$ often undergo neural collapse (Papyan et al.). Let $$\mu_c = \mathbb{E}[h \mid y = c]$$ be class-$$c$$ mean features. NC terminal phase is characterized by four phenomena:

1. Variability collapse: within-class scatter vanishes, $$h \to \mu_c$$ for each class $$c$$.
2. Simplex ETF: class means $$\{\mu_c\}$$ form a equiangular tight frame (equal norms, equal pairwise angles).
3. Self-duality: class means align with classifier weights.
4. Norm convergence: feature norms concentrate.

A single scalar summary is the collapse order

$$
m_{\text{NC}} = 1 - \frac{\sigma_W^2}{\sigma_B^2 + \epsilon},
\qquad
\sigma_W^2 = \mathbb{E}_c\bigl[\mathbb{E}[\|h - \mu_c\|^2 \mid y=c]\bigr],
\qquad
\sigma_B^2 = \mathbb{E}_c\bigl[\|\mu_c - \bar{\mu}\|^2\bigr],
$$

with $$\bar{\mu} = \frac{1}{C}\sum_c \mu_c$$. As NC proceeds, $$m_{\text{NC}} \to 1$$.

This is a late-time order parameter: $$\mathcal{R}_n(\theta_t)$$ may already be near zero while $$m_{\text{NC}}(t)$$ still climbs. Cross-entropy can look flat while geometry reorganizes. That decoupling is the main reason to log representation gauges: the phase of the network in feature space is not determined by training loss alone.

On the phase map, NC is not usually a separate $$(w, \eta)$$ cell in a short MNIST run; it is a temporal phase after entering $$\mathcal{M}_0$$. Wide models that interpolate early show $$m_{\text{NC}}$$ rising sooner; narrow models may never reach high $$m_{\text{NC}}$$ before budget runs out.

### Grokking gap g(t)

Grokking (Power et al.) separates memorization and rule learning in time on algorithmic tasks. Training accuracy rises early; test accuracy can stay at chance; after many more steps, test accuracy jumps. Define the generalization gap

$$
g(t) = A_{\text{train}}(t) - A_{\text{test}}(t).
$$

A grokking transition is the late collapse $$g(t) \to 0$$ while train accuracy remains high. On the phase map, this is a temporal boundary: the same fixed $$(w, \eta)$$ can sit in a memorization phase at $$t = 10^4$$ and a generalization phase at $$t = 10^6$$.

Weight decay $$\lambda$$ and training budget $$T$$ act as control knobs orthogonal to width. Large $$\lambda$$ can suppress memorization minima and accelerate the transition; too much kills train accuracy. Grokking is the cleanest demonstration that phase is not just architecture: time and regularization define basins in function space that SGD visits in order.

Within the overparameterized tail of a width sweep, test loss often decreases smoothly with capacity (Figure 7). On log-log axes one sometimes sees

$$
\mathcal{R}_{\text{test}}(N) \approx \mathcal{R}_\infty + a \, N^{-\alpha},
$$

a scaling law inside one phase. That smoothness is intra-regime structure: excellent for extrapolation, but not a substitute for knowing where interpolation, EOS, and grokking boundaries sit.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig07-scaling-laws.png" alt="Loss vs parameter count in overparameterized regime" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 7. Test loss vs parameter count on the overparameterized tail (post-interpolation). Power-law fits describe behavior within the rich phase, not across boundaries.</figcaption>
</figure>

---

## Worked example I: grokking as a temporal phase transition

Modular addition is the canonical grokking setup. Fix prime $$p$$ (here $$p = 97$$). Inputs are pairs $$(a, b) \in \{0, \ldots, p-1\}^2$$; labels are $$y = (a + b) \bmod p$$. A small MLP or transformer memorizes the training table quickly: $$A_{\text{train}}(t) \to 1$$ while $$A_{\text{test}}(t) \approx 1/p$$.

Task as classification over $$p$$ classes:

$$
f_\theta(a, b) \in \mathbb{R}^p,
\qquad
\hat{y} = \arg\max_c f_\theta(a,b)_c,
\qquad
\mathcal{L} = \text{CE}\bigl(f_\theta(a,b), (a+b) \bmod p\bigr).
$$

Early dynamics minimize $$\mathcal{L}$$ on the training set by memorization circuits (lookup-like). Late dynamics, under continued gradient flow and weight decay, drift toward algorithmic solutions that implement addition mod $$p$$ and generalize to unseen pairs.

Read this as metastability: the system sits in a memorization basin with $$g(t) \approx 1 - 1/p$$, then crosses a slow barrier into a generalizing basin with $$g(t) \to 0$$. The relevant "free energy" picture is heuristic, not literal: memorizing solutions are easier to reach but higher in a regularized objective; generalizing solutions are harder to reach but stable once found. That is the kind of dynamical story the phase language is meant to capture.

Control knobs beyond $$(w, \eta)$$: training time $$T$$, weight decay $$\lambda$$, and data fraction (partial tables grok differently). If you only plot the first 2k steps, you conclude the model cannot generalize. If you log $$g(t)$$ to 10k–100k steps, you see the transition.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig06-grokking.png" alt="Grokking curves" width="850"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 6. Modular addition mod 97: train accuracy rises first, test accuracy follows after a long plateau. The grokking time t<sub>g</sub> is where dg/dt is maximal (steepest collapse of g).</figcaption>
</figure>

---

## Worked example II: neural collapse as a representation phase transition

Neural collapse is the complementary story in representation space rather than train-test time. Fix a layer $$h_\theta(x)$$. Early training: class clouds overlap in PCA. Mid training: clouds separate, $$m_{\text{NC}}$$ rises. Late training: clouds shrink toward means; means move toward an ETF simplex; $$m_{\text{NC}} \to 1$$.

The loss curve can look flat while $$m_{\text{NC}}(t)$$ still climbs. Cross-entropy near zero only means logits are confident; it does not mean within-class geometry has collapsed. This is the clearest case of an order parameter decoupled from loss.

Operationally, log $$m_{\text{NC}}(t)$$ every epoch and snapshot PCA of $$h_\theta(x)$$ on a fixed validation batch. The phase transition is visible in both: a knee in $$m_{\text{NC}}(t)$$ and a topological change in 2D projections (overlap $$\to$$ separated $$\to$$ collapsed simplex).

NC also clarifies what interpolation buys you: until $$\mathcal{R}_n \approx 0$$, features are still task-aligned but not terminal. NC is the terminal geometry of the rich phase for classification, not a property of random features or the lazy regime.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig05-neural-collapse.png" alt="Neural collapse panels" width="900"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 5. PCA of penultimate features at early, middle, and late epochs (MNIST MLP). Same run as Figure 1: m<sub>NC</sub> rises after interpolation.</figcaption>
</figure>

---

## Synthesis: reading a run

| Physics idea | Deep learning analogue |
|---|---|
| Phase | Training regime (underfit, interpolate, EOS, collapse, grok) |
| Control knob | Width, $$n$$, $$\eta$$, batch size, weight decay, time $$T$$ |
| Order parameter | $$\chi$$, $$m_{\text{NC}}$$, $$g$$, interpolation threshold |
| Critical boundary | $$N_{\text{eff}} \approx n$$; $$\chi \approx 1$$; $$g \to 0$$; $$m_{\text{NC}} \to 1$$ |
| Metastability | Memorization basin before grokking |
| Emergent geometry | Neural collapse after fit |

For one run:

1. Log $$\mathcal{R}_n(t)$$, $$A_{\text{train}}(t)$$, $$A_{\text{test}}(t)$$ every step.
2. Log $$\chi(t)$$ (or a sharpness proxy) and $$m_{\text{NC}}(t)$$ when affordable.
3. Compute $$g(t) = A_{\text{train}}(t) - A_{\text{test}}(t)$$.
4. Mark the step $$t_{\text{interp}}$$ where $$\mathcal{R}_n \to 0$$ (interpolation).
5. Compare gauges before and after $$t_{\text{interp}}$$: EOS often intensifies post-interpolation; NC rises after; grokking (if present) is late collapse of $$g$$.

For a project:

1. Pick task and architecture.
2. Build a 2D phase map over the knobs you actually tune ($$w$$, $$\eta$$, $$\lambda$$, or $$n$$).
3. Label regions and boundaries with theory names (double descent, EOS, NC, grokking).
4. Use single-run order parameters to explain why a cell is red or blue on the map.

The physics perspective is useful here because it keeps theory tied to something you can draw. Papers become annotations on your chart rather than a scattered syllabus. The open problem is not to list every phenomenon once, but to understand which boundaries must exist for a given architecture and which order parameters are complete enough to predict generalization without training to completion.

---

### References

* Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. [PNAS](https://arxiv.org/abs/1812.11118).

* Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., & Talwalkar, A. (2022). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. [ICLR](https://arxiv.org/abs/2103.00065).

* Papyan, V., Han, X., & Donoho, D. L. (2020). Prevalence of neural collapse during the terminal phases of deep learning training. [PNAS](https://arxiv.org/abs/2008.08186).

* Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).

*Experiments: GPU bundle via `tools/training-at-critical-point/train_phase_blog.py`. Reproduce figures with `plot_figures.py` from `experiments/training-at-critical-point/outputs/`.*
