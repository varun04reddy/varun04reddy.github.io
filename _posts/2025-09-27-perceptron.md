---
title: "Perceptrons and the Brain"
date: 2025-09-27
layout: post
categories: [technical]
---

Ah, the classic perceptron, the foundation on which much of modern AI is built. The perceptron has been around for decades, and while today’s AI systems rely on stacks of many such units in the form of MLPs, even a single perceptron has surprisingly strong learning guarantees that can be leveraged in interesting ways. In this post, I will introduce the perceptron, discuss its learning constraints and guarantees, and explore how its learning rule shares interesting parallels with biological learning. This is the first part of a broader series (for which I will hopefully follow through with) aimed at discussing universal learning principles that hold true whether the learner is biological or artificial.

---

The perceptron assigns a label by thresholding an affine score:

$$
f(x)=\operatorname{sign}(w^\top x - w_0),\qquad x\in \mathbb{R}^N.
$$

The decision boundary is the hyperplane

$$
\{x\in \mathbb{R}^N:\; w^\top x - w_0 = 0\}.
$$

Equivalently, with bias absorbed via augmentation,

$$
\tilde x=\begin{bmatrix}x\\1\end{bmatrix},\quad 
\tilde w=\begin{bmatrix}w\\-w_0\end{bmatrix},\quad
f(x)=\operatorname{sign}(\tilde w^\top \tilde x).
$$

In any dimension $$N$$, if a dataset is linearly separable (in any dimension), then a perceptron represents a perfect classifier for that dataset.

On a misclassified example $$(x_\mu,y_\mu)\in\mathbb{R}^N\times\{-1,+1\}$$, update (with bias absorbed) by

$$
w \leftarrow w + \eta\, y_\mu\, x_\mu\quad\text{whenever}\quad y_\mu(w^\top x_\mu)\le 0,
$$

with learning rate $$\eta>0$$. If the data is linearly separable, the PLA converges in finitely many updates!

By a standard margin argument (details omitted), the PLA on linearly separable data makes at most $$\big(D/\delta^\ast\big)^2$$ mistakes, where $$D=\max_\mu\|x_\mu\|$$ and $$\delta^\ast=\min_\mu \tfrac{y_\mu x_\mu^\top w_\ast}{\|w_\ast\|}$$; larger margin $$\delta^\ast$$ ⇒ faster convergence, while tiny margins imply many updates.

**Limitations**
- Only linear decision boundaries: it can't realize non–linearly separable labelings (classic XOR in $$\mathbb{R}^2$$).
- No hierarchical/compositional features: a single perceptron lacks hidden representations; nonlinearly separable problems require feature expansion or multiple layers.

---

Let us now talk about the capacity of a perceptron. But first, what does “capacity” mean? Glad you asked! For $$P$$ labeled points in $$(\mathbb{R}^N)$$, capacity asks how many labelings (dichotomies) can be realized by a linear separator. When points are in [general position](https://en.wikipedia.org/wiki/General_position) and the separating hyperplane passes through the origin, [Cover’s counting theorem](https://en.wikipedia.org/wiki/Cover%27s_theorem) gives
$$
C(P,N)\;=\;2\sum_{i=0}^{N-1}\binom{P-1}{i}.
$$
This counts realizable dichotomies among the $$2^P$$ possibilities.

- If
$$
P \le N \quad\Rightarrow\quad C(P,N)=2^P
$$
(all labelings are linearly separable).
- If
$$
P = 2N \quad\Rightarrow\quad C(P,N)=2^{P-1}
$$
(half of all labelings).
- For fixed $$N$$ and growing $$P$$:
$$
\frac{C(P,N)}{2^P} \to 0
$$
(separable labelings become rare).

Lets now introduce the the [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension); which is the largest $$m$$ such that **every** labeling of **some** set of $$m$$ points can be realized by a hyperplane.

**Perceptron result.**
$$
\mathrm{VCdim}(\text{hyperplanes in }\mathbb{R}^N)=N+1.
$$

Meaning, with a bias, a hyperplane has about $$N+1$$ degrees of freedom, enough to [shatter](https://en.wikipedia.org/wiki/Shattered_set) $$N+1$$ points in general position. Push to $$N+2$$ points and **some** labeling will break linear separability.

Concurrently, the Gardner capacity asks: for a random dataset in high dimension, up to what load can a single hyperplane separate the labels with high probability?

Setup: Draw inputs and labels i.i.d. (inputs from an isotrophic distribution and labels are independent binary lables) Let there be $$P$$ patterns in $$\mathbb{R}^N$$ and define the **load**
$$
\alpha \;=\; \frac{P}{N}.
$$

**Result (phase transition, no margin).** As $$N,P\to\infty$$ with fixed $$\alpha$$,
$$
\mathbb{P}\!\left(\exists\,w:\; y_\mu\,w^\top x_\mu>0\ \ \forall\mu\right)\ \longrightarrow\
\begin{cases}
1,& \alpha<2,\\[4pt]
0,& \alpha>2~.
\end{cases}
$$
Equivalently, the typical storage capacity is
$$
\alpha_c \;=\; 2 \quad \Longleftrightarrow \quad P \approx 2N.
$$

Meaning, in high dimensions with generic (random) data, a single linear separator can usually fit about $$2N$$ random constraints; beyond that, separability almost surely fails.

However, If you require a positive geometric margin $$\kappa>0$$ (points must sit at least $$\kappa$$ away from the separating hyperplane in normalized units), the threshold drops:
$$
\alpha_c(\kappa) \;<\ 2,
$$

<figure style="text-align: center;">
    <img src="/assets/img/blog/perceptron1.png" alt="Perceptron 1" width="300"/>
</figure>

- **VC dimension** $$=N+1$$: worst-case guarantee on *small sets* (every labeling of **some** $$N{+}1$$ points).  
- **Gardner capacity** $$\alpha_c=2$$: typical-case limit on *large random sets* (about **\(2N\)** patterns).

Capacity grows linearly with $$N$$: in the worst case a perceptron can shatter up to $$N+1$$ points, while for typical random data separability holds up to about $$2N$$ patterns, beyond roughly $$2N$$ a single hyperplane almost surely fails, and requiring a positive margin lowers this threshold further.


---

Our discussion so far naturally leads us to discussions about similarities and differences between artificial and biological learning. Brains are not incentivized to memorize patterns; rather they must be robust to noise, drift, and limited data. This is a similar goal of perceptons, the goal being to minimize the population risk. We see this in the trade-off between how many patterns a neuron can store (capacity) and how far each pattern sits from the decision boundary (margin). Pushing capacity too high typically shrinks the margin, making decisions fragile.

Let a neuron implement a linear decision with geometric margin $$\kappa>0$$ to all stored patterns. If the effective decision variable is corrupted by Gaussian perturbations with standard deviation $$\sigma$$ (from input noise, synaptic variability, or background activity), then a stored pattern flips label with probability
$$
\epsilon(\kappa,\sigma)\;\approx\;\Phi\!\Big(-\frac{\kappa}{\sigma}\Big),
$$
where $$\Phi$$ is the standard normal CDF. **Larger margin $$\kappa$$ ⇒ exponentially smaller error**, but it also reduces capacity!

For a perceptron storing random patterns, the Gardner capacity with margin obeys

$$
\alpha_c(\kappa)\;=\;\frac{P}{N}\;=\;\frac{1}{\,\Phi(\kappa)\,(1+\kappa^2)\;+\;\kappa\,\phi(\kappa)\,},\qquad \alpha_c(0)=2,
$$

with $$\phi$$ being the standard normal PDF. Stricter separation (larger $$\kappa$$) lowers capacity:

$$
\kappa\uparrow\quad\Rightarrow\quad \alpha_c(\kappa)\downarrow.
$$

Neural circuits face joint constraints (finite synapses, metabolic cost, noisy spikes, nonstationary inputs). A simple way to encode the trade-off is to balance robustness against storage:

- Robustness benefit: error falls like $$\Phi(-\kappa/\sigma)$$.
- Capacity cost: fewer patterns storable, via $$\alpha_c(\kappa)$$.
- Resource limits: larger margins may require stronger/sparser synapses or more inhibitory control.

A stylized objective at fixed dimension $$N$$ could be to **choose $$\kappa$$** that minimizes expected error at a given load $$\alpha=P/N$$:
$$
\kappa^\star(\alpha,\sigma)\;=\;\arg\min_{\kappa>0}\ \Big[\ \epsilon(\kappa,\sigma)\ \ \text{s.t.}\ \ \alpha\le \alpha_c(\kappa)\ \Big].
$$
For small noise $$\sigma$$, the optimum shifts to **smaller** $$\kappa$$ (you can afford tighter margins and higher capacity). For larger $$\sigma$$, it shifts to **larger** $$\kappa$$ (robustness dominates).

The cerebellum’s granular layer expands inputs into a high-dimensional code before a Purkinje cell reads them out. Increasing dimension $$N$$ boosts linear separability (capacity scales with $$N$$), but for fixed data and noisy synapses it can also shrink effective margins or overfit. With finite resources (total synaptic strength, spikes per second), theory predicts an intermediate, task- and noise-dependent optimum in both:
- margin $$\kappa^\star$$ (robustness vs. count of storable patterns),
- feature dimension $$N^\star$$ (expressivity vs. margin/overfitting/metabolic cost).

This is all to say that biological systems appear to operate near a sweet spot: not maximal capacity at vanishing margins, not maximal margins with tiny capacity, but an optimal margin (and dimensionality) that maximizes reliable performance under noise and constraints. 

> <small>It is crazy to me how evolution can converge on solutions that theory predicts. Under resource and noise constraints, many neural systems seem to sit near Pareto efficient tradeoffs: capacity vs margin, accuracy vs energy, plasticity vs stability. This convergence makes me believe that evolutionary reverse engineering plausible; by specifying the relevant constraints, objectives, and dynamics, models can recover brainlike solutions. Evolution is NOT a static global optimizer; it satisfices dynamic environments and bodies. The repeated emergence of similar optima suggests that principled models capture what matters. For consciousness, a complete mechanistic theory is still missing, but my hope is that we can model functions and conditions that allow for the emergence of conscious processing likely in an intelligent system, including integration, global broadcasting, and self modeling, without assuming that evolution directly targeted consciousness.