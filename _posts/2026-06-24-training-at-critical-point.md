---
title: "Finding the Teacher on a Hidden Manifold"
date: 2026-06-22
layout: post
description: "Order parameters, lazy–rich feature learning, and phase diagrams in a teacher–student model with structured inputs — a statistical-mechanics walkthrough in the spirit of Ganguli and Pehlevan."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, teacher-student]
---

A network trained by SGD has $10^6$–$10^{12}$ weights. You cannot watch all of them and build intuition. Statistical mechanics faced the same problem with $10^{23}$ molecules: abandon the microscopic state, define a few **order parameters** that summarize the macroscopic behavior, and ask when those parameters change character — when the system crosses a **phase boundary**.

This post does that for one problem where the answer is known: **teacher–student learning with structured inputs**. A fixed teacher network $f^\star$ generates labels; a student $f_\theta$ learns from noisy samples. We know the target function. The question is not merely "did test error go down?" but **did the student recover the teacher's structure?** And if so, *how* — by moving its internal representations (rich learning), or by fitting labels in a nearly fixed feature basis (lazy learning)?

Everything below is built from first principles, measured in GPU experiments, and plotted as phase diagrams. The style follows the hidden-manifold and lazy/rich literature (Gerace et al.; Goldt et al.; Pehlevan and collaborators) rather than a survey of unrelated deep-learning phenomena.

---

## 1. From microstate to order parameters

### 1.1 The teacher–student setup

Draw inputs in $\mathbb{R}^d$. Following the **hidden manifold model** (Gerace et al., 2020), inputs live on a low-dimensional subspace of dimension $m \ll d$:

$$
x = P z, \qquad z \sim \mathcal{N}(0, I_m), \qquad P \in \mathbb{R}^{d \times m}, \quad P^\top P = I_m.
$$

The teacher is a two-layer ReLU network with $K^\star$ hidden units:

$$
f^\star(x) = \frac{1}{\sqrt{K^\star}} \sum_{j=1}^{K^\star} a_j^\star \, \mathrm{ReLU}\bigl({w_j^\star}^\top x + b_j^\star\bigr).
$$

The student has the same architecture with $K$ hidden units. We observe $n$ noisy samples

$$
y_i = f^\star(x_i) + \xi_i, \qquad \xi_i \sim \mathcal{N}(0, \sigma^2),
$$

and train the student with full-batch SGD on mean-squared error. Canonical values in the figures: $d = 100$, $m = 15$, $K^\star = 8$, $n = 4000$ ($\alpha = n/d = 40$), $\sigma = 0.05$.

Why this setup? In real benchmarks the target is unknown. Here the **ground-truth phase is known**: either the student recovers the teacher or it does not. That lets us define order parameters with literal meaning — not proxies.

### 1.2 Three order parameters

The microscopic state is the full weight vector $\theta_t$ at step $t$. We compress it to three scalars (plus internal structure we inspect later).

**Generalization error** (macroscopic performance):

$$
\varepsilon_g(t) = \frac{\mathbb{E}_x\bigl[(f_{\theta_t}(x) - f^\star(x))^2\bigr]}{\mathrm{Var}_x(f^\star(x))}.
$$

Normalized MSE on fresh test inputs. $\varepsilon_g \to 0$ means the student matches the teacher in function space.

**Teacher overlap** (first-layer alignment):

$$
R_j(t) = \max_i \bigl| \cos(w_{i,t}, w_j^\star) \bigr|, \qquad
R(t) = \frac{1}{K^\star} \sum_{j=1}^{K^\star} R_j(t).
$$

Each teacher direction $w_j^\star$ is matched to its best-aligned student neuron. $R$ is the analogue of magnetization in spin systems. Random init gives $R \sim \mathcal{O}(1/\sqrt{d})$; perfect recovery gives $R \to 1$.

**Feature drift** (Pehlevan's rich-vs-lazy gauge):

$$
d_h(t) = \mathbb{E}_x \bigl\| h_{\theta_t}(x) - h_{\theta_0}(x) \bigr\|_2^2,
$$

where $h_\theta(x) \in \mathbb{R}^K$ is the normalized hidden representation after the first layer. At initialization $d_h = 0$. In the **lazy** (kernel) regime, $d_h$ stays small: the readout does the work while features barely move. In the **rich** (feature-learning) regime, $d_h$ grows substantially before $\varepsilon_g$ falls.

These three quantities are not redundant. On the hidden manifold with moderate overparameterization, we will see regimes where $\varepsilon_g \ll 1$ while $R$ remains modest — the student matches the teacher functionally without full weight alignment. That decoupling is itself a phase-diagram fact, not a bug.

---

## 2. Learning as a trajectory in order-parameter space

Plotting loss vs. step is one projection of a high-dimensional path. Plotting $(d_h, \varepsilon_g)$ colored by learning rate $\eta$ reveals the **geometry of learning** directly.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig01-phase-portrait.png" alt="Phase portraits: feature drift vs generalization error at multiple learning rates" width="520"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1. SGD trajectories in $(d_h, \varepsilon_g)$ space, one curve per learning rate $\eta$. All runs start at high error and low drift. As training proceeds, features move ($d_h$ increases) while error falls — but the <em>path</em> depends on $\eta$. Small $\eta$: gradual drift then slow error decay. Large $\eta$: rapid drift, sharp error drop, but risk of instability at the highest rates. This is the dynamical picture Pehlevan emphasizes: learning is motion in representation space, not just descent on a scalar loss.</figcaption>
</figure>

Read this like a phase portrait in mechanics. Each curve is an orbit. The horizontal axis is "how much have internal features changed from init?" The vertical axis is "how wrong are we on new data?" The successful orbits move right and down. The **learning rate selects which dynamical regime** you traverse — not just how fast you get there.

At $\eta = 0.15$–$0.35$ the trajectories show a characteristic knee: $\varepsilon_g$ stays near order unity while $d_h$ grows, then drops sharply once representations have moved enough. That knee is the empirical signature of a **representation-learning transition** within a single training run.

---

## 3. Internal structure: symmetry breaking

Macroscopic order parameters hide $K \times K^\star$ pairwise relationships. The **overlap matrix**

$$
M_{ij}(t) = \bigl| \cos(w_{i,t}, w_j^\star) \bigr| \in \mathbb{R}^{K \times K^\star}
$$

is the microscopic state at the first layer. At $t = 0$, $M$ is diffuse — no student neuron prefers any teacher direction. Late in training, structure emerges: each teacher column develops a bright spot as one student neuron locks on.

Because absolute overlaps stay moderate on the hidden manifold ($|\cos| \lesssim 0.3$ even when $\varepsilon_g$ is small), we plot the **gain over initialization** $\Delta M_{ij}(t) = M_{ij}(t) - M_{ij}(0)$ to highlight emergent specialization:

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig02-alignment.png" alt="Overlap gain heatmaps at five training steps" width="680"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2. Gain in overlap matrix $\Delta M_{ij}(t)$ at five training steps ($K = K^\star = 8$, rich init). Early: uniform noise above baseline. Late: sparse bright spots — each teacher direction claims a student neuron. This is literal symmetry breaking: $K$ interchangeable student slots compete; the teacher's $K^\star$ directions break the permutation symmetry.</figcaption>
</figure>

The decomposed overlaps $R_j(t)$ tell us *which* teacher directions are found first:

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig03-per-neuron-overlap.png" alt="Per-teacher overlap, generalization error, and feature drift over training" width="520"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3. <strong>Top:</strong> per-teacher overlaps $R_j(t)$ (colors) and mean $\bar R(t)$ (dashed). Directions are learned at staggered times — not a single collective transition. <strong>Bottom:</strong> $\varepsilon_g$ (purple, log scale) vs. $d_h$ (teal, log scale). Error falls only after feature drift has grown — the rich-learning sequence. Note $\bar R$ saturates below 1 even as $\varepsilon_g \to 10^{-3}$: function matching without full weight alignment.</figcaption>
</figure>

The bottom panel of Figure 3 is the Pehlevan story in one plot: **feature drift precedes generalization**. The student must move its representation before it can interpolate the teacher on the manifold. Weight overlap $R$ tracks partial alignment but does not need to reach unity for $\varepsilon_g$ to vanish — especially when $K \geq K^\star$ and the readout can compensate.

---

## 4. Phase boundaries: signal, noise, and data

Order parameters become powerful when we **move a knob** and watch boundaries appear in a 2D plane. This is the Ganguli/Gerace program applied to our teacher–student model.

### 4.1 Signal-to-noise transition $(\alpha, \sigma)$

Fix architecture and sweep the sample-complexity ratio $\alpha = n/d$ and label noise $\sigma$. Gerace et al. predict a **sample-complexity transition**: below a critical $\alpha_c(\sigma)$, noise dominates and generalization fails; above it, the teacher becomes identifiable.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig04-phase-diagram.png" alt="SNR phase diagrams: epsilon_g and R vs alpha and sigma" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 4. Phase maps over $(\alpha, \sigma)$. <strong>Left:</strong> $\varepsilon_g$ (log color scale). A sharp boundary separates a high-error phase (small $\alpha$ or large $\sigma$) from a generalizing phase (large $\alpha$, small $\sigma$). <strong>Right:</strong> teacher overlap $R$. The boundary is softer — $R$ varies modestly ($0.22$–$0.27$) across the plane, confirming that overlap is not the only order parameter on the hidden manifold.</figcaption>
</figure>

The left panel is the cleanest phase diagram: error spans three orders of magnitude with a visually crisp transition along both axes. More data helps; more noise hurts — but now with a **quantitative boundary** in $(\alpha, \sigma)$ space rather than a verbal claim.

The right panel teaches humility: $R$ does not mirror $\varepsilon_g$ everywhere. In the hidden-manifold setting, a student can sit in the generalizing phase ($\varepsilon_g \ll 1$) without $R \to 1$. Both order parameters are legitimate; they capture different aspects of the microstate.

### 4.2 Sample complexity at multiple widths

Fix $\sigma$ and sweep $\alpha$ for several student widths $K$. Wider students have more capacity to fit noise; narrow students underfit the teacher.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig05-sample-complexity.png" alt="Generalization error vs alpha for multiple student widths K" width="480"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 5. $\varepsilon_g$ vs. $\alpha = n/d$ at fixed $\sigma$, colored by student width $K$. Narrow students ($K = 8$, purple) need more data; wide students ($K = 64$, yellow) generalize at smaller $\alpha$ but pay an error floor at low data. Power-law scaling $\varepsilon_g \sim \alpha^{-\nu}$ appears in the generalizing regime — the exponent $\nu$ depends on $K$ and $m/d$.</figcaption>
</figure>

Each curve is a **cross-section** of the $(\alpha, K)$ phase diagram at fixed noise. The vertical dotted line marks $\alpha = 40$, our canonical training point.

### 4.3 Capacity and optimization $(K, \eta)$

At fixed $n$, sweep student width $K$ and learning rate $\eta$. Too few hidden units ($K < K^\star$): underfitting, high $\varepsilon_g$ regardless of $\eta$. Too large $\eta$: optimization instability.

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig07-capacity-phase.png" alt="Phase maps of epsilon_g and R over K and learning rate" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 7. Phase maps over $(K, \eta)$ at fixed $n = 4000$. <strong>Left:</strong> $\varepsilon_g$. Recovery band at $K \geq K^\star$ and moderate $\eta \in [0.03, 0.15]$. <strong>Right:</strong> $R$. Interestingly, larger $K$ yields <em>higher</em> overlap — extra neurons provide more slots to match teacher directions, even though functionally $K = K^\star$ suffices.</figcaption>
</figure>

Compare Figure 7 (left) to the capacity story in Saad & Solla: the underfitting phase below $K^\star$ is unmistakable. The right panel shows that **capacity and alignment need not coincide**: $K = 128$ gives the highest $R$ but not necessarily the lowest $\varepsilon_g$.

---

## 5. Initialization scale: a second phase diagram

Pehlevan and collaborators emphasize that **initialization scale** controls whether a network trains in a kernel-like regime (features frozen) or a feature-learning regime (representations move). In the infinite-width NTK limit, large init $\gamma \to \infty$ gives lazy training with $d_h \to 0$. Our finite-width student on a hidden manifold shows a related but distinct picture: init scale modulates a **drift–alignment tradeoff**.

Sweep $\gamma$ (std of weight init) and learning rate $\eta$ at fixed $K = 32$:

<figure style="text-align: center;">
  <img src="/assets/img/blog/critical-point/fig06-lazy-rich.png" alt="Lazy-rich phase diagrams: feature drift and overlap vs init scale and learning rate" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 6. Init-scale phase maps over $(\gamma, \eta)$. <strong>Left:</strong> feature drift $d_h$. Small $\gamma$ (bottom): modest drift. Large $\gamma$ (top): representations move strongly — but, unlike the infinite-width lazy limit, large init here <em>increases</em> drift because larger weights amplify activation changes during SGD. <strong>Right:</strong> teacher overlap $R$. Small $\gamma$ achieves the highest $R$; large $\gamma$ yields high drift without alignment — functionally similar to the "misaligned but low-error" regime in Figure 3 when $K$ is wide.</figcaption>
</figure>

Small $\gamma$ is the **alignment-favoring phase**: features move modestly but lock onto teacher directions ($R \approx 0.35$). Large $\gamma$ is a **high-drift, low-alignment phase**: $d_h$ grows yet $R$ stays near $0.23$. SGD can still reduce $\varepsilon_g$ in the latter case when $K$ is large, because extra neurons and a flexible readout compensate for poor first-layer alignment — the decoupling we flagged in Section 3.

This is not a pathology — it is a **different phase** of the same model. Real networks start in various places on this plane depending on width, depth, and standard-deviation of init.

---

## 6. How to think with this (beyond the toy model)

Four habits transfer from this problem to settings where the teacher is unknown:

1. **Name order parameters before training.** Ask: what macroscopic quantity would tell me the network found the structure I care about? Here: $\varepsilon_g$, $d_h$, $R$. On a real task: overlap with a probe direction, margin, effective rank of activations, grokking gap — the label changes; the habit does not.

2. **Plot trajectories in order-parameter space.** Figure 1 is the whole point: learning is a path through a low-dimensional state space. Loss is one projection. $(d_h, \varepsilon_g)$ reveals representation dynamics that loss alone hides.

3. **Sweep knobs and draw phase diagrams.** Figures 4–7 are not decorative. They locate **boundaries** — SNR transitions, capacity limits, lazy/rich crossovers — where macroscopic behavior changes character. That is the statistical-mechanics workflow.

4. **Look inside the summary statistic.** $R(t)$ is useful because we can inspect $M_{ij}(t)$ and $R_j(t)$ and see *which* directions are learned and when. Macroscopic without microscopic is blind; microscopic without macroscopic is hopeless.

Teacher–student on a hidden manifold is the cleanest laboratory because $f^\star$ and ${w_j^\star}$ are known. Real problems replace the teacher with an unknown target — but the workflow stays: **compress, track, sweep, locate boundaries, inspect internal structure when the macroscopic curve does something interesting.**

---

## 7. Connection to the wider program

This post is one slice of a larger question: *when does gradient-based learning find the "right" internal structure, and when does it merely fit labels?* The teacher–student model gives sharp answers. Hidden-manifold inputs (Ganguli/Gerace) add realistic structure: data live on a low-dimensional subspace, so sample complexity depends on $m$, not $d$. Lazy/rich init (Pehlevan) adds a dynamical axis: the same architecture can train in different phases depending on $\gamma$ and $\eta$.

What we did **not** do here — deliberately — is mix in unrelated phenomena (grokking, neural collapse, edge of stability on ImageNet). Each of those deserves its own order parameters and its own phase diagrams. The value of the present setup is **coherence**: one model, three order parameters, four phase planes, all measuring the same underlying question — *did the student find the teacher, and through what mechanism?*

---

### References

* Saad, S., & Solla, S. A. (1995). On-line learning in soft committee machines. *Physical Review E*.

* Gerace, F., et al. (2020). Generalisation error in learning with random features and the hidden manifold model. [ICML](https://arxiv.org/abs/2002.09339).

* Goldt, S., et al. (2020). Generalisation dynamics of online learning in a wide two-layer neural network. [NeurIPS](https://arxiv.org/abs/1905.13641).

* Pehlevan, C., & Chklovskii, D. B. (2020). Neuroscience-inspired online learning algorithms. *Nature Neuroscience*.

* Mezard, M., & Montanari, A. (2009). *Information, Physics, and Computation*. Oxford University Press. (Ch. 12: statistical mechanics of learning.)

*Experiments: `tools/training-at-critical-point/train_phase_blog.py --all`. Figures: `publish_blog_figures.py`. Hidden-manifold teacher–student, ReLU nets, step-level CSV logs of $R$, $R_j$, $\varepsilon_g$, $d_h$, and alignment matrices.*
