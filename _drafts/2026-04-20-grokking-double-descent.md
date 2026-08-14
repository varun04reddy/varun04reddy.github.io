---
title: "Grokking and Double Descent Cut the Same Interpolation Surface"
date: 2026-04-20
layout: post
categories: [technical]
---

Double descent and grokking usually get filed as separate puzzles, which is fair, since they look nothing alike on a plot, but I don't think they are the same phase transition either. I do think they are two cuts through the same object: the interpolation threshold, and the geometry of the zero-loss set on either side of it.

Double descent is the capacity cut: you increase $$N$$ past the point where training error hits zero, test error spikes and then falls, and that spike, if you actually look at where it lives, sits at interpolation.

Grokking is the time cut. Fix a network well past interpolation, and training loss hits zero almost immediately, while test accuracy sits near chance for a long time and then jumps. Memorization and generalization come apart in time, not in capacity, which is the part that feels like a trick the first time you see it.

The rest of this is constraint counting, mostly following Geiger, Spigler, and collaborators. Where that picture makes a quantitative claim I will say so. Where it is still a bet, I will say that too.

---

## Training examples as constraints

Take a fully connected net $$f(x; W)$$ trained on $$P$$ examples under squared hinge loss:

$$
L(W) = \frac{1}{P} \sum_{\mu \in m} \frac{1}{2} \Delta_\mu^2, \qquad \Delta_\mu = \epsilon - y_\mu f(x_\mu; W),
$$

where $$m$$ is the set of currently unsatisfied patterns ($$\Delta_\mu > 0$$). A satisfied example contributes zero loss and zero gradient, which is why this loss is the one where interpolation geometry is exact rather than hand-wavy. Grokking experiments are usually run with cross-entropy. I am using hinge as a model of interpolation, not as a claim that modular-arithmetic grokking is secretly a hinge-loss phenomenon.

Every training example is a constraint on $$W$$. Define $$\alpha = P/N$$, and the whole story is which side of a critical load you are on:

- $$\alpha < \alpha_c$$: overparameterized. Zero-loss solutions exist.
- $$\alpha > \alpha_c$$: underparameterized. Zero training loss is geometrically impossible.
- $$\alpha \approx \alpha_c$$: interpolation. The double descent peak sits here.

This is the jamming dictionary, used once and then dropped: particles $$\leftrightarrow$$ parameters, contacts $$\leftrightarrow$$ unsatisfied examples, jamming density $$\leftrightarrow$$ $$\alpha_c$$. You do not need the physics. You need a way to count degrees of freedom.

Near a minimum, $$H = \nabla^2 L = H_0 + H_p$$. $$H_0$$ is a sum of rank-1 matrices, one per active constraint, so $$\mathrm{rank}(H_0) \le N_\Delta$$. If training loss is already zero, then $$N_\Delta = 0$$ and $$H_0$$ vanishes, which is the first hint that interpolating solutions live in a much flatter place than the jammed ones.

Two jamming classes, and this distinction actually matters:

- **Isostatic** (hard spheres): $$N_\Delta = N$$ at the transition. Rigid, isolated.
- **Hypostatic** (ellipses): $$N_\Delta / N < 1$$. Macroscopic flat directions remain.

ReLU nets land in the hypostatic class. Geiger, Spigler, et al. measure $$N_\Delta / N \approx 0.75$$ at the fitting threshold, not $$1.0$$, and once you have that number the "isolated interpolator" story is already wrong. The test-error spike is a hypostatic interpolator: most directions still cost curvature, a macroscopic fraction of the Hessian is flat. The spectrum has a zero-eigenvalue peak of weight $$N - N_\Delta$$, then a gap, then a positive bulk.

---

## Two cuts

**Underparameterized ($$\alpha > \alpha_c$$).** There is no zero-loss solution, so gradient descent is just compromising across conflicting constraints. Adding parameters reduces the squeeze, test error falls, and you can feel like you are making progress, but the fit is still a compromise. It threads the training points because it has to.

**Interpolation ($$\alpha \approx \alpha_c$$).** Just enough capacity to fit. $$N_\Delta / N \approx 0.75$$: most directions cost curvature, a macroscopic fraction of the Hessian is still flat, and small input perturbations, or test points sitting between training examples, land in the unpinned directions. Near $$\alpha_c$$ the gap and overlap distributions look critical,

$$
P_+(\Delta) \sim \Delta^\theta, \qquad P_-(\Delta) \sim |\Delta|^{-\gamma},
$$

with measured $$\theta \approx 0.3$$, $$\gamma \approx 0.2$$ for ReLU nets in that literature. Two nominally identical runs can land on different interpolators, all with zero training loss and different test behavior, which is the annoying part if you wanted a unique "the" interpolating solution.

**Overparameterized ($$\alpha \ll \alpha_c$$).** The zero-loss set $$\mathcal{M}_0 = \{W : L(W) = 0\}$$ is a high-dimensional manifold, so the optimizer does not converge to a point so much as fall into this set and then drift. With weight decay, the dynamics on the interior of that set (hinge) reduce to

$$
\frac{dW}{dt}\Big|_{\mathcal{M}_0} = -\lambda W
$$

projected onto the tangent space. The selected interpolator is approximately minimum Frobenius norm, which for classification tends to large margin. That is the second descent: extra room on $$\mathcal{M}_0$$. More overparameterization is not a taller spike. The spike lives at the threshold, and past the threshold test error falls, which is the thing people keep getting backwards.

Grokking is the same manifold, sliced in time. Training loss hits zero, you are inside $$\mathcal{M}_0$$, and the easy solution, the one gradient descent finds first, is high-rank memorization: one effective direction per training example. Weight decay then drifts toward a low-rank algorithmic interpolator, if one exists (modular arithmetic, sparse parity, anything with a small description). When that rearrangement happens, rank collapses and test accuracy jumps:

$$
W^\top W \xrightarrow{\text{grokking}} \sum_{k=1}^{r} \sigma_k^2 v_k v_k^\top, \qquad r \ll N.
$$

The timescale of the drift is $$1/\lambda$$, not $$N/\lambda$$. Path length can still depend on width, I am not pretending otherwise, but putting $$N$$ in the numerator and calling it a prediction already disagrees with the fact that larger models often grok *faster*. Empirically, larger weight decay speeds grokking. That one I would actually bet on:

$$
T_{\mathrm{grok}} \sim \frac{1}{\lambda}\, g(\mathcal{M}_0).
$$

If you stop a double descent sweep early, the overparameterized models have not finished that motion, so the second descent is weaker or missing. That is epoch-wise double descent, and it is the cleanest experimental link between the two plots, because you can watch the same networks fail to grok simply by not waiting. If you grok while shrinking capacity toward $$\alpha_c$$, the delay should shrink: there is less room for a high-rank memorizer when you barely interpolate.

The two-layer Saxe timescale $$t_k \approx s_k^{-1} \log(s_k / \epsilon_0)$$ makes a sequential curriculum visible in linear nets, large modes first, noise modes later, which is why people like to tell grokking the same way (high-rank fit, then low-rank modes). Treat that as an analogy, not a reduction. Deeper linear nets have different mode dynamics, and I am using the two-layer formula because it is the one I can write down.

---

## Experiment: rank collapse should line up with the jump

If the story is right, you should be able to watch two clocks. Training loss dies first. Rank, and then test accuracy, move later, together. If those come apart, I would stop talking about manifold drift.

[Experiment to run:
Setup: two-hidden-layer MLP, hidden width 512, modular addition $$f(a,b)=(a+b)\bmod 97$$, about 1000 training pairs from the $$97\times 97$$ grid. AdamW with $$\lambda \in \{0.01, 0.02, 0.05, 0.1, 0.2\}$$. Same init family, same data split.

Record vs step: train loss, test accuracy, effective rank $$r_{\mathrm{eff}} = \exp(H_{\mathrm{sv}})$$ of $$W_1$$ and of $$W_2$$ (entropy of normalized singular values), and $$\|W\|_F^2$$.

Figure: one plot, three series against training time. Train loss should hit ~0 early. Test accuracy should stay near chance, then jump. $$r_{\mathrm{eff}}$$ should stay high until that same window, then drop. Mark the jump with a vertical line. A second plot: grokking epoch vs $$1/\lambda$$, which should be roughly linear.

Prediction: train loss dies first. Test accuracy and $$r_{\mathrm{eff}}$$ move together later. Grokking epoch vs $$1/\lambda$$ should be close to a straight line if weight decay is the drift on $$\mathcal{M}_0$$. If rank collapse and the accuracy jump come apart, the manifold-drift story is wrong. If $$T_{\mathrm{grok}}$$ does not track $$1/\lambda$$, weight decay is not the clock.]

---

## Experiment: the spike and the delay should share $$\alpha_c$$

The other test is even more direct. If grokking and double descent are two cuts of interpolation geometry, the same $$\alpha_c$$ should locate the test-error spike *and* gate whether grokking is even possible.

[Experiment to run:
Setup: two-layer ReLU, grid over $$P \in \{50,100,200,500,1000,2000\}$$ and hidden width so that $$\alpha = P/N \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$$. Same optimizer, same step budget, train to convergence.

Record for each $$(P,N)$$: final train loss, final test error, $$N_\Delta / N$$, and the Hessian zero-eigenvalue fraction.

Figure: test error vs $$\alpha$$ with $$N_\Delta / N$$ on a second axis. The test-error spike and the jump in $$N_\Delta / N$$ should sit at the same $$\alpha_c$$. $$N_\Delta / N$$ should jump to about $$0.75$$, not $$1.0$$. Bonus panel: grokking epoch vs $$\alpha$$ for the modular-addition setup above. Delay should shrink as $$\alpha \to \alpha_c$$ from below.

Prediction: the test-error spike and the jump in $$N_\Delta / N$$ sit at the same $$\alpha_c$$. $$N_\Delta / N$$ should jump to about $$0.75$$, not $$1.0$$. If the spike is isostatic ($$N_\Delta / N \approx 1$$), the ReLU jamming claim is wrong. If grokking still happens above $$\alpha_c$$, interpolation is not the gate.]

---

PAC bounds of the form $$E_{\mathrm{test}} \le E_{\mathrm{train}} + \sqrt{(C(\mathcal{F}) + \log(1/\delta))/P}$$ go vacuous when $$N \gg P$$, and they also do not speak to grokking: train error is already zero, so the additive complexity term is the whole bound, and it is useless. Gradient descent does not visit most of the class. The question, the one I actually care about here, is which interpolator it finds.

I would not call grokking and double descent the same phase transition. I would call them two measurements of interpolating geometry. What I do not know, and this is the part I would actually run, is whether $$g(\mathcal{M}_0)$$ has a simple width dependence, and whether hinge counting still controls grokking under cross-entropy.

---

* Geiger, M., Spigler, S., d'Ascoli, S., Sagun, L., Baity-Jesi, M., Biroli, G., & Wyart, M. (2019). Scaling description of generalization with number of parameters in deep learning. *Journal of Statistical Mechanics*.
* Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2021). Deep double descent: where bigger models and more data hurt. *JMLR* / ICLR.
* Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.
* Saxe, A. M., McClelland, J. L., & Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR.
