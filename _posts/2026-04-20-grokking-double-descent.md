---
title: "Grokking and Double Descent Cut the Same Interpolation Surface"
date: 2026-04-20
layout: post
categories: [technical]
---

Double descent and grokking are usually filed as separate puzzles. I do not think they are the same phase transition. I do think they are two cuts through the same object: the interpolation threshold, and the geometry of the zero-loss set on either side of it.

Double descent is the capacity cut. As you increase the number of parameters past the point where training error reaches zero, test error spikes, then falls again. The spike sits at interpolation. The second descent lives in the overparameterized regime.

Grokking is the time cut. Fix a network well past interpolation. Training loss hits zero quickly. Test accuracy stays near chance for a long time, then jumps. Memorization and generalization are separated in time, not in capacity.

The rest of this post is a constraint-counting picture of that surface, mostly following the jamming analysis of Geiger, Spigler, and collaborators. Where that picture makes a quantitative claim I will say so. Where it is still a bet, I will say that too.

---

## Mapping the Problem: Training Examples as Constraints

Start with a precise setup. Consider a fully connected network $$f(x; W)$$ trained on $$P$$ examples under a squared hinge loss:

$$
L(W) = \frac{1}{P} \sum_{\mu \in m} \frac{1}{2} \Delta_\mu^2, \quad \Delta_\mu = \epsilon - y_\mu f(x_\mu; W)
$$

where $$m$$ is the set of currently unsatisfied patterns (those with $$\Delta_\mu > 0$$), $$y_\mu \in \{-1, +1\}$$ are labels, and $$\epsilon$$ is a margin target. The hinge loss is analytically cleaner than cross-entropy for this analysis because it has finite range. A satisfied training example contributes exactly zero to the loss and exactly zero gradient. That makes the mapping to constrained geometry exact rather than approximate. Grokking experiments are usually run with cross-entropy. The hinge is the loss in which the jamming bookkeeping is clean. I am using it as a model of interpolation geometry, not as a claim that modular-arithmetic grokking is a hinge-loss phenomenon.

Every training example is a constraint on the parameter vector $$W$$. When satisfied, the constraint is inactive: no force. When violated, it pushes $$W$$ toward satisfaction. The loss is the total overlap: the sum of constraint violations. The parameter count $$N$$ is the number of effective degrees of freedom (directions in parameter space that actually affect the network's outputs, which for most architectures is close to the raw parameter count).

Define the constraint density $$\alpha = P/N$$. This single ratio controls the regime:
- $$\alpha < \alpha_c$$: overparameterized. More degrees of freedom than constraints. Zero-loss solutions exist.
- $$\alpha > \alpha_c$$: underparameterized. More constraints than effective degrees of freedom. Zero training loss is geometrically impossible.
- $$\alpha \approx \alpha_c$$: the interpolation threshold, where the double descent peak sits.

This framing is borrowed from jamming in disordered packings. You put repulsive particles in a fixed volume. Below a critical density they can always rearrange to relieve contacts. Above it they cannot: the packing is jammed. Map particles to parameters, contacts to unsatisfied training constraints, and the jamming density to $$\alpha_c$$. The rest of the section is that dictionary, used to count degrees of freedom.

---

## The Hessian and the Geometry of Constraints

To understand why the test error curve has the shape it does, we need to know whether a zero-loss solution exists and what the loss landscape looks like around it. That is the Hessian.

Near any local minimum of the loss, $$H = \nabla^2 L$$ encodes the local curvature. For the hinge loss, this decomposes cleanly:

$$
H = H_0 + H_p, \quad H_0 = \frac{1}{P} \sum_{\mu \in m} \nabla_W \Delta_\mu \otimes \nabla_W \Delta_\mu
$$

$$H_0$$ is a sum of rank-1 positive semidefinite matrices, one per active constraint. Its rank is at most $$N_\Delta$$, the number of active (violated) constraints. $$H_p$$ captures the curvature of the constraint surfaces themselves and can have both positive and negative eigenvalues.

This decomposition carries a strong implication: if the network has achieved zero training loss ($$N_\Delta = 0$$), then $$H_0$$ vanishes entirely. The Hessian is determined entirely by $$H_p$$. And for smooth activations at generic parameter values, $$H_p$$ tends to have many near-zero eigenvalues, meaning the minimum sits in a flat region of the landscape.

The geometry near the fitting threshold is where things get interesting. There are two qualitatively different cases, depending on the curvature structure of $$H_p$$:

**Isostatic jamming** (hard spheres): at the transition, the number of active constraints equals the number of degrees of freedom, $$N_\Delta = N$$. The Hessian is full-rank. The minimum is rigid. Every direction in parameter space has positive curvature, and the solution is geometrically isolated.

**Hypostatic jamming** (ellipses, or particles with extra rotational degrees of freedom): at the transition, $$N_\Delta / N < 1$$. The Hessian still has a macroscopic zero-eigenvalue subspace. The minimum has flat directions.

Deep networks with ReLU activations land in the hypostatic class. Geiger, Spigler, et al. measure $$N_\Delta / N$$ by tracking satisfied versus unsatisfied training examples. At the fitting threshold the ratio jumps from zero to about $$0.75$$, not to $$1.0$$. That jump is the signature they use for a first-order change in constraint geometry. I will stick to that measurement below. The spike in test error is a property of a hypostatic interpolating solution, not of an isostatic one.

The Hessian spectrum near the threshold exhibits the telltale hypostatic form: a delta peak at zero of weight $$N - N_\Delta$$ (the flat directions), then a gap, then a continuous positive bulk. This spectral structure is directly measurable in trained networks and has been observed to match the theoretical prediction.

---

## Double Descent: Three Regimes, One Transition

With the geometry in place, the double descent curve breaks into three regimes that each have crisp physical explanations.

**Regime 1: Underparameterized ($$\alpha > \alpha_c$$).**

The system is jammed. There is no zero-loss solution, and gradient descent finds the minimum of $$L$$ subject to this constraint. The solution is the best possible compromise across conflicting constraints: a function that misclassifies some training points and fits others, trading off across all of them simultaneously.

In this regime, test error decreases as you add parameters because you are genuinely increasing the model's ability to capture training signal. More parameters mean fewer jammed constraints, fewer forced compromises. The function you learn is increasingly well-specified and decreasingly sensitive to arbitrary-seeming trade-offs between competing training examples.

But the solution is not generalizing in a principled sense. It is fitting the training data as well as it can under a geometric squeeze. The learned function can be highly irregular near the decision boundary: it is threading through the training examples while making the minimal number of concessions required by the capacity limit.

**Regime 2: Interpolation threshold ($$\alpha \approx \alpha_c$$).**

This is where the test error spike lives. The network has just enough capacity to fit the training set. The zero-loss solution is still tightly constrained: $$N_\Delta / N \approx 0.75$$, so most directions cost curvature, but a macroscopic fraction of the Hessian remains flat. That is hypostatic interpolation, not isostatic isolation.

The interpolating function has little slack on the training points. Small input perturbations, or test points that sit between training examples, land in directions the constraints did not pin down. You can write a schematic scaling for that sensitivity,

$$
\mathbb{E}\left[(f(x + \delta x; W^*) - f(x; W^*))^2\right] \propto \|\delta x\|^2 \cdot \mathrm{tr}(H_{\mathrm{input}}^{-1} H_{\mathrm{loss}}),
$$

but I do not have a derivation of this identity for ReLU nets. Treat it as a sketch of "rigid on the satisfied constraints, free in the orthogonal complement," not as a theorem.

There is also a stability issue distinct from input sensitivity. Near $$\alpha_c$$, the gap and overlap distributions of the constraints exhibit near-critical power laws:

$$
P_+(\Delta) \sim \Delta^\theta, \quad P_-(\Delta) \sim |\Delta|^{-\gamma}
$$

with measured exponents $$\theta \approx 0.3$$ and $$\gamma \approx 0.2$$ for ReLU networks in that literature. Those differ from classical sphere-jamming exponents. Near the fitting threshold the solution found by the optimizer is sensitive to which training examples it saw, in which order, at what initialization. Two nominally identical runs can converge to different interpolators, all with zero training loss and different test behavior.

**Regime 3: Overparameterized ($$\alpha \ll \alpha_c$$).**

Now the zero-loss manifold $$\mathcal{M}_0 = \{W : L(W) = 0\}$$ is a high-dimensional submanifold of parameter space. There is not one zero-loss solution but uncountably many. The optimizer does not converge to a point; it converges into this manifold and then drifts along it.

Gradient descent with mini-batching, weight decay, or finite step size injects noise. A common model of that noise is Langevin dynamics on $$\mathcal{M}_0$$ at some effective temperature $$T_{\mathrm{eff}}$$. Configurations that stay on the manifold and keep a small weight norm are then favored. One way to write that preference is an effective free energy

$$
F_{\mathrm{eff}}(W) = L(W) + \frac{\lambda}{2}\|W\|_F^2 - T_{\mathrm{eff}} S(W),
$$

where $$S(W)$$ is a stand-in for local basin volume. I am using this as a bookkeeping device for "weight decay pulls toward small $$W$$, noise prefers flat regions," not as a derived equilibrium for SGD on ReLU nets.

Formally, among all zero-training-loss solutions, gradient descent with $$L_2$$ weight decay selects approximately the minimum Frobenius norm solution:

$$
W^* = \arg\min_{W : L(W) = 0} \|W\|_F^2
$$

The minimum-norm zero-loss solution for classification tasks tends to maximize the geometric margin (the distance of training examples from the decision boundary), and large-margin classifiers generalize well by classical SVM theory. The second descent is not a surprise once you understand what the optimizer is implicitly doing in the overparameterized regime: it is selecting the highest-margin, most structurally simple solution from the zero-loss manifold, and those solutions generalize.

---

## Deep Linear Networks: The Tractable Case

Before tackling grokking in full generality, it helps to have a model where everything can be done analytically. Deep linear networks are that model.

Consider a depth-$$D$$ linear network $$\hat{y} = W_D \cdots W_1 x$$ with $$W_l \in \mathbb{R}^{n_l \times n_{l-1}}$$. The end-to-end map is $$W_{1:D} = W_D \cdots W_1$$, and the training loss depends only on this product, not on the individual factors. Depth still changes the dynamics.

Learning proceeds by building up the SVD of the input-output covariance

$$
\Sigma = \frac{1}{P} \sum_{\mu=1}^P y_\mu x_\mu^T \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
$$

Let $$\Sigma = U S V^T$$ with $$S = \text{diag}(s_1, \ldots, s_r)$$, $$s_1 \geq s_2 \geq \cdots \geq 0$$. For a *two-layer* linear network from small isotropic initialization, gradient flow has a closed form, and the $$k$$-th singular mode is learned on a timescale

$$
t_k \approx \frac{1}{s_k} \log\left(\frac{s_k}{\epsilon_0}\right)
$$

(Saxe, McClelland, Ganguli). Deeper linear nets have different mode dynamics. I am using the two-layer formula because it is the one I can write down, not because depth does not matter.

Large singular values (strong correlations in the training data) are learned faster than small ones. In the finite-sample setting, $$\Sigma$$ contains both population signal and sampling noise with singular values of order $$1/\sqrt{P}$$. Gradient descent learns the large modes first. Early stopping exploits that ordering: stop before the noise modes are fit.

That is epoch-wise overfitting in a linear model, not model-wise double descent. If you train to zero loss you eventually fit the small modes too, and test error can get worse. If you are overparameterized, the interpolator you land on depends on initialization and the optimizer, and the minimum-norm interpolator downweights those small modes.

The linear case is useful because it makes a sequential curriculum visible. Grokking in nonlinear nets is sometimes described the same way: the network first fits in a high-rank way, then slowly builds the low-rank modes of the task. That is an analogy to keep in mind, not a reduction of grokking to SVD dynamics.

---

## Grokking: Phase Transition on the Zero-Loss Manifold

With the setup in place, grokking becomes precise.

Take a network with $$\alpha \ll \alpha_c$$, deeply overparameterized. Training loss hits zero quickly. We are inside $$\mathcal{M}_0$$. Test loss is terrible.

The network at this point has found a solution by memorization. Concretely: the weight matrices $$W_l$$ are configured such that each training input $$x_\mu$$ activates a distinct, roughly orthogonal direction in representation space, and each output is reconstructed from that input-specific direction. This corresponds to a high-rank configuration of the weight matrices. The singular value spectrum of the weight products $$W_{1:l} = W_l \cdots W_1$$ is broad and flat: many singular values of roughly equal magnitude, one effective dimension per memorized example.

This high-rank memorized solution sits in a neighborhood of $$\mathcal{M}_0$$ that is easy for the optimizer to find (it is accessible from a typical random initialization via gradient descent) but far from optimal under any reasonable free energy. It uses many parameters to store a few training examples redundantly.

Now the thermal dynamics on $$\mathcal{M}_0$$ begin. Under gradient descent with weight decay:

$$
\frac{dW}{dt} = -\nabla L(W) - \lambda W
$$

When $$W \in \mathcal{M}_0$$, $$\nabla L(W) = 0$$ (exactly, for the hinge loss in the interior of the zero-loss region). The dynamics reduce to:

$$
\frac{dW}{dt}\bigg|_{\mathcal{M}_0} = -\lambda W
$$

projected onto the tangent space of $$\mathcal{M}_0$$ at each point. This is a gradient flow on $$\mathcal{M}_0$$ that decreases $$\|W\|_F^2$$. The fixed point is the minimum-norm point on $$\mathcal{M}_0$$.

What does the minimum-norm zero-loss solution look like? For tasks with algorithmic structure (modular arithmetic, sparse parity, anything with low-complexity ground truth), the minimum-norm solution is low-rank. The relevant computation can be done in a small subspace of representation space. The weight matrices in the minimum-norm solution have a few dominant singular modes that encode the algorithm, and near-zero singular values everywhere else.

So grokking is the trajectory of this flow from a high-rank memorized solution to a low-rank algorithmic solution, both on $$\mathcal{M}_0$$. The two sit in different basins of the induced geometry on the manifold. Crossing between them is a collective rearrangement: the zero-loss constraint couples the entries, so they cannot move independently.

When this rearrangement occurs, it is rapid. The weight matrices undergo a rank collapse:

$$
W^T W \xrightarrow{\text{grokking}} \sum_{k=1}^r \sigma_k^2 v_k v_k^T, \quad r \ll N
$$

where the dominant directions $$v_k$$ correspond to the task's true algorithmic structure. The Frobenius norm drops. A few singular values shoot up and the rest collapse to near-zero. And the test loss plummets because now the network is computing a low-rank function that responds to input directions relevant to the task, not to which specific training example is closest.

---

## Two cuts through the same surface

Grokking and double descent are two slices of an $$(N, T, \text{test error})$$ surface. Double descent varies capacity $$N$$ at large training time. Grokking varies time $$T$$ at large capacity. That is the claim I am willing to make. It is weaker than "they are the same phase transition," and it is the part that is actually testable.

The interpolation threshold $$\alpha_c$$ shows up in both slices. In double descent, the spike is at $$\alpha_c$$: the network has just entered the zero-loss set, the solution is hypostatic ($$N_\Delta / N \approx 0.75$$), and test error is high. The second descent is further into overparameterization, where the manifold is larger and the optimizer has room to find a simpler interpolator. More overparameterization is not a taller spike. The spike lives at the threshold. Past the threshold, test error falls.

In grokking, you start already deep on $$\mathcal{M}_0$$ ($$\alpha \ll \alpha_c$$). The delay is the time to move from a memorized basin to an algorithmic one.

If you stop a double descent sweep early, overparameterized models have not finished that motion, so the second descent is weaker or missing. That is epoch-wise double descent, and it is the cleanest experimental link between the two plots.

If you run grokking while shrinking capacity toward $$\alpha_c$$, the delay should shrink: there is less room for a high-rank memorizer when you barely interpolate. Grokking, in this picture, is a phenomenon of the interior of $$\mathcal{M}_0$$, not of the boundary.

At $$\alpha = \alpha_c$$ the zero-loss set is small (hypostatic, not a single isostatic point). Deep in the overparameterized regime it is high-dimensional. On the interior of the hinge-loss zero-loss region, weight decay gives

$$
\frac{dW}{dt}\bigg|_{\mathcal{M}_0} = -\lambda W
$$

projected onto the tangent space. The timescale of that flow is $$1/\lambda$$, not $$N/\lambda$$. Path length and barrier height can still depend on width and on the task. Empirically, larger weight decay speeds grokking. Empirically, larger models often grok *faster*, which already argues against putting $$N$$ in the numerator and calling it a prediction.

$$
T_{\text{grok}} \sim \frac{1}{\lambda}\, g(\mathcal{M}_0)
$$

where $$g$$ holds the path geometry. The $$1/\lambda$$ piece is the one I would actually bet on. The experiment below tests that, not an $$N/\lambda$$ scaling.

---

## Experiment: Watching the Rank Collapse

[Experiment to run:
Train a two-hidden-layer MLP on modular addition, specifically $$f(a, b) = (a + b) \bmod p$$ for prime $$p = 97$$. Use an overparameterized architecture (hidden width 512, roughly 1000 training pairs from the $$97 \times 97$$ input grid) and train with AdamW at weight decay $$\lambda \in \{0.001, 0.005, 0.01, 0.05, 0.1\}$$.

Record at each training step:
1. Training loss and test accuracy (standard grokking plot).
2. Singular value spectra of $$W_1$$ and $$W_2$$ (the two hidden-layer weight matrices).
3. Effective rank $$r_{\text{eff}} = \exp(H_{sv})$$ where $$H_{sv} = -\sum_i \bar{\sigma}_i \log \bar{\sigma}_i$$ is the entropy of the normalized singular value distribution $$\bar{\sigma}_i = \sigma_i / \sum_j \sigma_j$$.
4. Total weight norm $$\|W\|_F^2 = \|W_1\|_F^2 + \|W_2\|_F^2$$.

Expected outcome: During the memorization phase (training loss $$\approx 0$$, test accuracy $$\approx$$ chance), $$r_{\text{eff}}$$ should be near the hidden dimension (high rank). At the grokking transition, $$r_{\text{eff}}$$ drops sharply in a concentrated time window aligned exactly with the test accuracy jump. Simultaneously, $$\|W\|_F^2$$ decreases and the Fourier structure of the singular vectors (when projected onto the input space) should show the emergent discrete Fourier features known to encode modular arithmetic in grokked networks.

Varying $$\lambda$$: the grokking epoch should be approximately inversely proportional to $$\lambda$$. Plotting grokking epoch vs. $$1/\lambda$$ should give a roughly linear relationship. This tests the $$T_{\text{grok}} \propto 1/\lambda$$ prediction from the manifold drift analysis.

Why it matters: This cleanly separates grokking into two events, memorization (training loss hits zero) and structural reorganization (rank collapse), that happen at very different times and have very different signatures. The rank collapse is the true generalization event; the test accuracy jump is a downstream consequence of it. Understanding this ordering tells you what to optimize: if you want faster grokking, increase $$\lambda$$ or use a learning rate schedule that increases weight decay pressure after training loss hits zero.]

---

## Experiment: Mapping the Phase Boundary

[Experiment to run:
Fix a two-layer ReLU network and create a grid over $$(P, N)$$ where $$P \in \{50, 100, 200, 500, 1000, 2000\}$$ and $$N$$ (hidden width) is varied to give constraint densities $$\alpha = P/N \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$$. Train each network to convergence with fixed hyperparameters (same learning rate, optimizer, weight decay, number of steps).

For each $$(P, N)$$ pair, record:
1. Final training loss (to establish whether zero loss is achieved).
2. Final test error.
3. The fraction $$N_\Delta/N$$ of active constraints (violated patterns) at the final minimum.
4. The Hessian spectrum near the minimum, specifically the fraction of eigenvalues below some threshold (to measure the zero-eigenvalue peak).

Expected outcome from jamming theory:
- There is a sharp transition in the $$(P, N)$$ grid where training loss goes from zero to positive. The boundary should be approximately linear in $$(P, N)$$ space, i.e., $$P/N \approx \alpha_c$$ for some constant $$\alpha_c$$.
- Near this boundary, test error spikes. This is the double descent peak.
- The ratio $$N_\Delta/N$$ at the transition should jump discontinuously from 0 to approximately $$0.75$$ (the hypostatic prediction), not to $$1.0$$ (which would indicate isostatic jamming like spheres).
- Deep in the overparameterized regime ($$\alpha \ll \alpha_c$$), $$N_\Delta/N = 0$$ and the Hessian zero-eigenvalue peak has fractional weight approaching 1.

Bonus: Run the modular arithmetic grokking experiment for architectures at several different values of $$\alpha$$ (by fixing $$P$$ and varying $$N$$). As $$\alpha \to \alpha_c$$ from below, the grokking delay should shrink. In the limit there is no room for a high-rank memorized solution and grokking happens immediately or not at all. Plot grokking epoch vs. $$\alpha$$ to see this analytically predicted compression of the grokking timeline near the interpolation threshold.

Why it matters: This experiment tests whether the double descent peak and the grokking delay share interpolation geometry, rather than being unrelated. The prediction is that the same $$\alpha_c$$ that locates the spike also gates grokking: below $$\alpha_c$$ a zero-loss manifold exists and grokking is possible; above it, it is not.]

---

## What Classical Theory Gets Wrong, and What It Would Take to Fix It

The PAC-learning framework gives bounds of the form:

$$
E_{\text{test}} \leq E_{\text{train}} + \sqrt{\frac{C(\mathcal{F}) + \log(1/\delta)}{P}}
$$

where $$C(\mathcal{F})$$ is the VC dimension or some analog. For deep networks, $$C(\mathcal{F})$$ scales with $$N$$ (at best), so in the overparameterized regime ($$N \gg P$$), these bounds become vacuously large. They cannot explain why test error decreases in the second descent. The bounds can be vacuously large. They also do not speak to grokking: training error is already zero, so the additive complexity term is the whole bound and is useless. PAC asks a worst-case question about empirical risk minimization over a class. Gradient descent does not visit most of the class. The question I care about here is which interpolator it finds.

The statistical mechanics framework asks a different question: given the energy landscape of the loss and the effective temperature of the optimizer, what is the free-energy minimum, and does it generalize? This is a statistical question about the equilibrium distribution of the optimizer, not a worst-case bound. It makes genuinely predictive claims: about the form of the minimum-norm solution, about the grokking timescale, about the spectral structure of the Hessian near the interpolation threshold.

Making this picture rigorous for general deep nets is still open. The parts I would actually test are narrower: hypostatic $$N_\Delta/N \approx 0.75$$ at interpolation, $$T_{\mathrm{grok}} \propto 1/\lambda$$, and alignment between rank collapse and the test-accuracy jump.

---

## A Note on In-Context Learning

This analysis is sometimes compared to in-context learning in transformers.

Recent theory has shown that sufficiently expressive transformers can implement learning algorithms within the forward pass, effectively running multiple steps of gradient descent on the in-context data during inference. One attention layer can implement approximately one step of gradient descent on the context data. $$L$$ layers can implement $$L$$ steps, with error accumulating linearly in $$L$$ for convex objectives. The transformer is, in some functional sense, running an optimizer on the context window at inference time.

Now consider a network that has grokked a task. It has organized its weights so that, given a new input, it can compute the correct output using the low-rank algorithmic structure internalized during training. The computation it is doing is application of a learned algorithm. It is not retrieving the nearest training example; it is running the compressed algorithm on the input.

A grokked network has compiled an algorithm into its weights. A transformer doing in-context learning may be running an algorithm on the context window. I do not have a formal map between those, and I am not going to pretend the rank-collapse event is "training-time ICL." The shared question is only this: is the function using the task's structure, or a high-rank lookup?

---

The interpolation threshold is a real line in the $$(P, N)$$ plane: zero training loss on one side, not on the other. Double descent is what test error does when you cross that line in $$N$$. Grokking is what test error does when you sit far on the zero-loss side and wait. I would not call those the same phase transition. I would call them two measurements of the same interpolating geometry.

The spike is hypostatic interpolation, $$N_\Delta/N \approx 0.75$$, not an isostatic isolated point. The second descent is extra room on $$\mathcal{M}_0$$. Grokking is motion on that manifold from a high-rank memorizer toward a low-rank algorithm, with a timescale that should track $$1/\lambda$$ if weight decay is the drift. Larger $$N$$ does not automatically mean a taller spike or slower grokking.

What I do not know: whether $$g(\mathcal{M}_0)$$ has a simple dependence on width, and whether the hinge-loss counting still controls grokking under cross-entropy. Those are the experiments above.
