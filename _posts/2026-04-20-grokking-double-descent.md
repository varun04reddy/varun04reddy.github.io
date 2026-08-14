---
title: "Grokking and Double Descent Cut the Same Interpolation Surface"
date: 2026-04-20
layout: post
categories: [technical]
---

Double descent and grokking are usually filed as separate puzzles. I do not think they are the same phase transition. I do think they are two cuts through the same object: the interpolation threshold, and the geometry of the zero-loss set on either side of it.

Double descent is the capacity cut. Increase $$N$$ past the point where training error hits zero. Test error spikes, then falls. The spike sits at interpolation.

Grokking is the time cut. Fix a network well past interpolation. Training loss hits zero quickly. Test accuracy stays near chance, then jumps. Memorization and generalization are separated in time, not in capacity.

The rest is constraint counting, mostly following Geiger, Spigler, and collaborators. Where that picture makes a quantitative claim I will say so. Where it is still a bet, I will say that too.

<figure style="text-align: center;">
  <img src="/assets/img/blog/grokking-two-cuts.png" alt="Two cuts through interpolation" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1: Double descent is a horizontal slice (vary $$N$$ at large $$T$$). Grokking is a vertical slice (vary $$T$$ at large $$N$$). Both hit interpolation geometry. The gray band is $$\alpha \approx \alpha_c$$.</figcaption>
</figure>

---

## Training examples as constraints

Take a fully connected net $$f(x; W)$$ trained on $$P$$ examples under squared hinge loss:

$$
L(W) = \frac{1}{P} \sum_{\mu \in m} \frac{1}{2} \Delta_\mu^2, \qquad \Delta_\mu = \epsilon - y_\mu f(x_\mu; W),
$$

where $$m$$ is the set of currently unsatisfied patterns ($$\Delta_\mu > 0$$). A satisfied example contributes zero loss and zero gradient. That makes interpolation geometry exact. Grokking experiments are usually run with cross-entropy. I am using hinge as a model of interpolation, not as a claim that modular-arithmetic grokking is a hinge-loss phenomenon.

Every training example is a constraint on $$W$$. Define $$\alpha = P/N$$:

- $$\alpha < \alpha_c$$: overparameterized. Zero-loss solutions exist.
- $$\alpha > \alpha_c$$: underparameterized. Zero training loss is geometrically impossible.
- $$\alpha \approx \alpha_c$$: interpolation. The double descent peak sits here.

This is the jamming dictionary: particles $$\leftrightarrow$$ parameters, contacts $$\leftrightarrow$$ unsatisfied examples, jamming density $$\leftrightarrow$$ $$\alpha_c$$.

Near a minimum, $$H = \nabla^2 L = H_0 + H_p$$. $$H_0$$ is a sum of rank-1 matrices, one per active constraint, so $$\mathrm{rank}(H_0) \le N_\Delta$$. If training loss is zero then $$N_\Delta = 0$$ and $$H_0$$ vanishes.

Two jamming classes:

- **Isostatic** (hard spheres): $$N_\Delta = N$$ at the transition. Rigid, isolated.
- **Hypostatic** (ellipses): $$N_\Delta / N < 1$$. Macroscopic flat directions remain.

ReLU nets land in the hypostatic class. Geiger, Spigler, et al. measure $$N_\Delta / N \approx 0.75$$ at the fitting threshold, not $$1.0$$. The test-error spike is a hypostatic interpolator, not an isostatic isolated point. The Hessian spectrum has a zero-eigenvalue peak of weight $$N - N_\Delta$$, then a gap, then a positive bulk.

---

## Two cuts

**Underparameterized ($$\alpha > \alpha_c$$).** No zero-loss solution. Gradient descent compromises across conflicting constraints. Adding parameters reduces the squeeze, so test error falls, but the fit is still a compromise.

**Interpolation ($$\alpha \approx \alpha_c$$).** Just enough capacity to fit. $$N_\Delta / N \approx 0.75$$: most directions cost curvature, a macroscopic fraction of the Hessian is still flat. Small input perturbations land in unpinned directions. Near $$\alpha_c$$ the gap and overlap distributions look critical,

$$
P_+(\Delta) \sim \Delta^\theta, \qquad P_-(\Delta) \sim |\Delta|^{-\gamma},
$$

with measured $$\theta \approx 0.3$$, $$\gamma \approx 0.2$$ for ReLU nets in that literature. Two nominally identical runs can land on different interpolators.

**Overparameterized ($$\alpha \ll \alpha_c$$).** The zero-loss set $$\mathcal{M}_0 = \{W : L(W) = 0\}$$ is a high-dimensional manifold. With weight decay, the dynamics on the interior of that set (hinge) reduce to

$$
\frac{dW}{dt}\Big|_{\mathcal{M}_0} = -\lambda W
$$

projected onto the tangent space. The selected interpolator is approximately minimum Frobenius norm, which for classification tends to large margin. That is the second descent: extra room on $$\mathcal{M}_0$$, not a taller spike. The spike lives at the threshold.

Grokking is the same manifold, sliced in time. Training loss hits zero. You are inside $$\mathcal{M}_0$$. The easy solution is high-rank memorization: one effective direction per training example. Weight decay then drifts toward a low-rank algorithmic interpolator, if one exists (modular arithmetic, sparse parity). When the rearrangement happens, rank collapses and test accuracy jumps:

$$
W^\top W \xrightarrow{\text{grokking}} \sum_{k=1}^{r} \sigma_k^2 v_k v_k^\top, \qquad r \ll N.
$$

The timescale of the drift is $$1/\lambda$$, not $$N/\lambda$$. Path length can still depend on width. Empirically, larger weight decay speeds grokking. Empirically, larger models often grok *faster*.

$$
T_{\mathrm{grok}} \sim \frac{1}{\lambda}\, g(\mathcal{M}_0).
$$

The $$1/\lambda$$ piece is the one I would actually bet on.

If you stop a double descent sweep early, overparameterized models have not finished that motion, so the second descent is weaker. That is epoch-wise double descent, and it is the cleanest experimental link between the two plots. If you grok while shrinking capacity toward $$\alpha_c$$, the delay should shrink: less room for a high-rank memorizer.

The two-layer Saxe timescale $$t_k \approx s_k^{-1} \log(s_k / \epsilon_0)$$ makes a sequential curriculum visible in linear nets. Grokking is sometimes described the same way (high-rank fit, then low-rank modes). Treat that as an analogy, not a reduction. Deeper linear nets have different mode dynamics.

---

## Experiment: rank collapse should line up with the jump

[Experiment to run:
Setup: two-hidden-layer MLP, hidden width 512, modular addition $$f(a,b)=(a+b)\bmod 97$$, about 1000 training pairs from the $$97\times 97$$ grid. AdamW with $$\lambda \in \{0.01, 0.02, 0.05, 0.1, 0.2\}$$. Same init family, same data split.

Record vs step: train loss, test accuracy, effective rank $$r_{\mathrm{eff}} = \exp(H_{\mathrm{sv}})$$ of $$W_1$$ and of $$W_2$$ (entropy of normalized singular values), and $$\|W\|_F^2$$.

Figure: one plot, three series against training time. Train loss should hit ~0 early. Test accuracy should stay near chance, then jump. $$r_{\mathrm{eff}}$$ should stay high until that same window, then drop. Mark the jump with a vertical line. A second plot: grokking epoch vs $$1/\lambda$$, which should be roughly linear.

Prediction: Figure 2 and Figure 3. If rank collapse and the accuracy jump come apart, the manifold-drift story is wrong. If $$T_{\mathrm{grok}}$$ does not track $$1/\lambda$$, weight decay is not the clock.]

<figure style="text-align: center;">
  <img src="/assets/img/blog/grokking-rank-collapse.png" alt="Predicted grokking and rank collapse" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2: Predicted alignment. Train loss dies first. Test accuracy and $$r_{\mathrm{eff}}$$ move together later. The dotted line is the grokking window. This is a schematic, not a run.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="/assets/img/blog/grokking-timescale.png" alt="Predicted grokking timescale vs 1/lambda" width="620"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 3: Predicted clock. Grokking epoch vs $$1/\lambda$$ should be close to a straight line if weight decay is the drift on $$\mathcal{M}_0$$.</figcaption>
</figure>

---

## Experiment: the spike and the delay should share $$\alpha_c$$

[Experiment to run:
Setup: two-layer ReLU, grid over $$P \in \{50,100,200,500,1000,2000\}$$ and hidden width so that $$\alpha = P/N \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$$. Same optimizer, same step budget, train to convergence.

Record for each $$(P,N)$$: final train loss, final test error, $$N_\Delta / N$$, and the Hessian zero-eigenvalue fraction.

Figure: test error vs $$\alpha$$ with $$N_\Delta / N$$ on a second axis. The test-error spike and the jump in $$N_\Delta / N$$ should sit at the same $$\alpha_c$$. $$N_\Delta / N$$ should jump to about $$0.75$$, not $$1.0$$. Bonus panel: grokking epoch vs $$\alpha$$ for the modular-addition setup above. Delay should shrink as $$\alpha \to \alpha_c$$ from below.

Prediction: Figure 4. If the spike is isostatic ($$N_\Delta / N \approx 1$$), the ReLU jamming claim is wrong. If grokking still happens above $$\alpha_c$$, interpolation is not the gate.]

<figure style="text-align: center;">
  <img src="/assets/img/blog/grokking-phase.png" alt="Predicted double descent and hypostatic jump" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 4: Predicted phase cut. Test error spikes at $$\alpha_c$$. $$N_\Delta / N$$ jumps from 0 to about 0.75, not to 1. Overparameterized (small $$\alpha$$) is the second descent. Schematic, not a run.</figcaption>
</figure>

---

PAC bounds of the form $$E_{\mathrm{test}} \le E_{\mathrm{train}} + \sqrt{(C(\mathcal{F}) + \log(1/\delta))/P}$$ go vacuous when $$N \gg P$$. They also do not speak to grokking: train error is already zero. Gradient descent does not visit most of the class. The question is which interpolator it finds.

I would not call grokking and double descent the same phase transition. I would call them two measurements of interpolating geometry. What I do not know: whether $$g(\mathcal{M}_0)$$ has a simple width dependence, and whether hinge counting still controls grokking under cross-entropy. Those are the experiments above.

---

* Geiger, M., Spigler, S., d'Ascoli, S., Sagun, L., Baity-Jesi, M., Biroli, G., & Wyart, M. (2019). Scaling description of generalization with number of parameters in deep learning. *Journal of Statistical Mechanics*.
* Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2021). Deep double descent: where bigger models and more data hurt. *JMLR* / ICLR.
* Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.
* Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR.
