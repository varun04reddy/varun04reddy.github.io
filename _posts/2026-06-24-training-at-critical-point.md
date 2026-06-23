---
title: "Finding the Teacher"
date: 2026-06-22
layout: post
description: "A statistical-mechanics view of learning through one teacher–student problem: order parameters, internal alignment, and phase boundaries."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, teacher-student]
---

A neural network trained by SGD has on the order of $10^6$–$10^{12}$ weights. You cannot watch all of them and build intuition. Statistical mechanics faced the same problem with $10^{23}$ molecules: give up on the microscopic state, find a few **order parameters** that track the macroscopic behavior, and ask when those parameters change character.

This post does that for one problem we actually understand: **teacher–student learning**. A fixed teacher network $f^\star$ generates labels; a student $f_\theta$ learns from noisy samples. We know the target function. The question is not "did test accuracy go up?" but **did the student find the teacher?** Everything below is built to make that question precise, measurable, and visual.

---

## 1. The setup

Draw inputs $x \sim \mathcal{N}(0, I_d)$ in $\mathbb{R}^d$. A two-layer ReLU teacher with $K^\star$ hidden units defines

$$
f^\star(x) = \frac{1}{\sqrt{K^\star}} \sum_{j=1}^{K^\star} a_j^\star \, \mathrm{ReLU}\bigl({w_j^\star}^\top x + b_j^\star\bigr).
$$

The student has the same architecture with $K$ hidden units and trainable weights $\theta$. We observe $n$ noisy samples

$$
y_i = f^\star(x_i) + \xi_i, \qquad \xi_i \sim \mathcal{N}(0, \sigma^2),
$$

and minimize mean-squared error with SGD. In the experiments below: $d = 50$, $K^\star = 4$, $K = 16$, $n = 3000$, $\sigma = 0.05$.

This is the standard playground of Saad & Solla, Gerace et al., and Goldt et al. The reason theorists like it: **the ground-truth phase is known**. Either the student recovers the teacher directions or it does not. Real benchmarks hide the target; here we can define order parameters that mean something.

---

## 2. Two order parameters

The microscopic state is the full weight vector $\theta_t$ at step $t$. We compress it to two scalars.

**Generalization error.** On fresh Gaussian test points,

$$
\varepsilon_g(t) = \frac{\mathbb{E}_x\bigl[(f_{\theta_t}(x) - f^\star(x))^2\bigr]}{\mathrm{Var}_x(f^\star(x))}.
$$

This is the macroscopic "how wrong are we on unseen data?" gauge. $\varepsilon_g \to 0$ means the student matches the teacher in function space.

**Teacher overlap.** Each hidden neuron of the student has an incoming weight vector $w_i$; the teacher has directions $w_j^\star$. Match each teacher direction to its best-aligned student neuron:

$$
R_j(t) = \max_i \bigl| \cos(w_{i,t}, w_j^\star) \bigr|, \qquad
R(t) = \frac{1}{K^\star} \sum_{j=1}^{K^\star} R_j(t).
$$

$R$ is the analogue of magnetization: it measures how much the student's first layer has rotated toward the teacher's. Random initialization gives $R \approx 0.3$–$0.4$; successful learning gives $R \to 1$.

Train MSE tells you whether the student fits the **training set**. $(R, \varepsilon_g)$ tell you whether it found the **teacher**. Those can decouple briefly (memorization without alignment), but in this setup they move together on the successful path.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig01-phase-portrait.png" alt="Phase portrait: training trajectory in R vs epsilon_g" width="480"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1. One SGD run as a trajectory in order-parameter space $(R, \varepsilon_g)$, colored by training step. The student starts disordered (low $R$, high $\varepsilon_g$) and flows toward the teacher (high $R$, low $\varepsilon_g$). This is the macroscopic picture: a curve in two dimensions instead of $10^4$ weights.</figcaption>
</figure>

---

## 3. What the macroscopic curve hides: internal alignment

$R(t)$ is a single number summarizing $K \times K^\star$ pairwise cosines. The **microscopic structure** is the overlap matrix

$$
M_{ij}(t) = \bigl| \cos(w_{i,t}, w_j^\star) \bigr| \in \mathbb{R}^{K \times K^\star}.
$$

Early in training, student neurons are misaligned with all teacher directions: $M$ is diffuse. Late in training, each teacher column develops a bright spot — one student neuron has locked onto each $w_j^\star$. The symmetry-breaking is literal: $K$ student slots compete to explain $K^\star$ teacher directions.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig02-alignment.png" alt="Student-teacher overlap heatmaps at three training steps" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2. Overlap matrix $M_{ij}(t) = | \cos(w_i, w_j^\star) |$ at three times. Early: no structure. Late: one student neuron per teacher column — internal symmetry breaking.</figcaption>
</figure>

The decomposed overlaps $R_j(t)$ show the same story neuron by neuron: some teacher directions are learned early, others lag. The mean $R(t)$ (dashed) is not arbitrary averaging — it tracks when the last teacher direction has been found.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig03-per-neuron-overlap.png" alt="Per-teacher-neuron overlap R_j(t) over training" width="480"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3. Per-teacher overlaps $R_j(t)$ (colors) and their mean $R(t)$ (black dashed). Generalization error falls only after the slowest direction aligns.</figcaption>
</figure>

---

## 4. Phase boundaries: capacity and data

Order parameters become more useful when you **move a knob** and watch boundaries appear.

**Capacity ($K$ vs $K^\star$).** Sweep student width $K$ and learning rate $\eta$ at fixed $n$. Small $K < K^\star$: the student cannot represent the teacher — $\varepsilon_g$ stays high regardless of $\eta$. Large $\eta$: optimization is unstable. In the interior ($K \geq K^\star$, moderate $\eta$): recovery band where $\varepsilon_g \ll 1$.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig04-phase-diagram.png" alt="Phase map of generalization error over K and learning rate" width="420"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 4. Phase map: $\varepsilon_g$ after training on the $(K, \eta)$ grid. Dashed line at $K = K^\star = 4$. Below it: underfitting phase. Top-right: recovery. Large $\eta$ column: unstable / poor generalization.</figcaption>
</figure>

**Data ($\alpha = n/d$).** Fix $K > K^\star$ and increase sample size. There is a sample-complexity transition: below a critical $\alpha$, noise dominates and $R$ stays low; above it, the teacher becomes identifiable and both $R \to 1$ and $\varepsilon_g \to 0$.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig05-sample-complexity.png" alt="Sample complexity: epsilon_g and R vs alpha" width="420"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 5. Sample complexity at fixed $K = 16$: $\varepsilon_g$ (blue, log scale) and $R$ (green) vs $\alpha = n/d$. The knee is the data-limited phase boundary — the same "more data helps" story, but now with a measurable order parameter.</figcaption>
</figure>

These are the phase diagrams of the teacher–student model. They are not universal for ImageNet or LLMs. They **are** the right mental template: pick a task where you know the target, define order parameters with meaning, sweep one knob at a time, locate boundaries.

---

## 5. How to think with this

Three habits transfer from this toy problem to harder ones:

1. **Name the order parameters before training.** Ask: what macroscopic quantity would tell me the network found the structure I care about? Overlap with a teacher direction, generalization gap, collapse order — the label depends on the task, but the habit is the same.

2. **Plot trajectories in order-parameter space, not just loss vs step.** Figure 1 is the whole point: learning is a path through a low-dimensional state space. Loss is one projection of that path.

3. **Look inside the summary statistic.** $R(t)$ is useful because we can inspect $M_{ij}(t)$ and $R_j(t)$ and see *which* directions are learned and when. Macroscopic without microscopic is blind; microscopic without macroscopic is hopeless.

Teacher–student is the cleanest place to practice because the teacher gives you $f^\star$ and ${w_j^\star}$ for free. Real problems replace the teacher with an unknown target — but the workflow stays: compress, track, sweep, locate boundaries, then inspect internal structure when the macroscopic curve does something interesting.

---

### References

* Saad, S., & Solla, S. A. (1995). On-line learning in soft committee machines. *Physical Review E*.

* Gerace, F., et al. (2020). Generalisation error in learning with random features and the hidden manifold model. [ICML](https://arxiv.org/abs/2002.09339).

* Goldt, S., et al. (2020). Generalisation dynamics of online learning in a wide two-layer neural network. [NeurIPS](https://arxiv.org/abs/1905.13641).

* Mezard, M., & Montanari, A. (2009). *Information, Physics, and Computation*. Oxford University Press. (Statistical mechanics of learning, Ch. 12.)

*Experiments: `tools/training-at-critical-point/train_phase_blog.py --all`. Figures: `publish_blog_figures.py`. Gaussian teacher–student, ReLU nets, step-level CSV logs of $R$, $R_j$, $\varepsilon_g$, and alignment matrices.*
