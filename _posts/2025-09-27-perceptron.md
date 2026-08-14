---
title: "Perceptrons and the Brain"
date: 2025-09-27
layout: post
categories: [technical]
---

The perceptron assigns a label by thresholding an affine score. Even one unit has a finite-mistake guarantee on linearly separable data, and a sharp capacity transition on random data. That is the part I want, plus what it suggests for noisy biological readout. This is meant to be the first note in a series on learning rules that show up in both artificial and biological systems.

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

If a dataset in $$\mathbb{R}^N$$ is linearly separable, some perceptron represents a perfect classifier for it. The PLA below finds one.

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

Capacity here means: how many labelings of $$P$$ points in $$\mathbb{R}^N$$ can a linear separator realize? Cover's theorem counts *homogeneous* dichotomies (hyperplanes through the origin) for points in [general position](https://en.wikipedia.org/wiki/General_position):
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

Now introduce the [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension): the largest $$m$$ such that **every** labeling of **some** set of $$m$$ points can be realized by a hyperplane.

**Perceptron result.**
$$
\mathrm{VCdim}(\text{hyperplanes in }\mathbb{R}^N)=N+1.
$$

Meaning, with a bias, an affine hyperplane has $$N+1$$ degrees of freedom and can [shatter](https://en.wikipedia.org/wiki/Shattered_set) $$N+1$$ points in general position. Cover's $$P \le N$$ line above was for the homogeneous (no-bias) case. Affine separators get one extra point. Push to $$N+2$$ and some labeling breaks.

Concurrently, the Gardner capacity asks: for a random dataset in high dimension, up to what load can a single hyperplane separate the labels with high probability?

Setup: draw inputs from an isotropic distribution and independent random binary labels. Let there be $$P$$ patterns in $$\mathbb{R}^N$$ and define the **load**
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

Our discussion so far naturally leads us to discussions about similarities and differences between artificial and biological learning. Brains are not incentivized to memorize patterns; rather they must be robust to noise, drift, and limited data. This is a similar goal for perceptrons: keep population risk down. The tradeoff is how many patterns a neuron can store (capacity) versus how far each pattern sits from the decision boundary (margin). Pushing capacity too high typically shrinks the margin.

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

If the load $$\alpha = P/N$$ is held fixed, error $$\epsilon(\kappa,\sigma)$$ is decreasing in $$\kappa$$ and the constraint is $$\alpha \le \alpha_c(\kappa)$$, so you always take the largest feasible margin. Noise never enters. To get a real tradeoff, maximize the expected number of correctly stored patterns instead:

$$
n_{\mathrm{correct}}(\kappa,\sigma) = N\,\alpha_c(\kappa)\,\bigl(1-\epsilon(\kappa,\sigma)\bigr).
$$

Then $$\kappa^\star(\sigma) = \arg\max_{\kappa>0} n_{\mathrm{correct}}(\kappa,\sigma)$$. Small $$\sigma$$: error is already tiny at modest $$\kappa$$, so the maximum sits near high capacity (small $$\kappa$$). Large $$\sigma$$: you pay in $$\epsilon$$ unless $$\kappa/\sigma$$ is appreciable, so the maximum moves to larger $$\kappa$$ and lower $$\alpha_c$$.

The cerebellum’s granular layer expands inputs into a high-dimensional code before a Purkinje cell reads them out. Increasing dimension $$N$$ boosts linear separability (capacity scales with $$N$$), but for fixed data and noisy synapses it can also shrink effective margins or overfit. With finite resources (total synaptic strength, spikes per second), theory predicts an intermediate, task- and noise-dependent optimum in both:
- margin $$\kappa^\star$$ (robustness vs. count of storable patterns),
- feature dimension $$N^\star$$ (expressivity vs. margin/overfitting/metabolic cost).

This is all to say that a noisy linear readout has a real capacity-margin tradeoff once you count expected correct patterns, not just error at fixed load. I would not read consciousness out of Gardner's $$\alpha_c$$.