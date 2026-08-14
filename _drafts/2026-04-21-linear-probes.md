---
title: "What Linear Probes Actually Measure"
date: 2026-04-21
layout: post
categories: [technical]
---

There is a running joke in ML that every interpretability technique is either a probe or an attention visualization. The joke is mostly true, and the probe half is the more interesting half to get right, because it keeps working even after you have been told, several times, that it should not.

A linear probe is a frozen linear classifier or regressor on hidden activations. Neuroscience has used the same object since the early 1990s under the name population decoding, and Alain and Bengio brought it into deep nets in 2016. Both fields got excited, ran the same experiment, and then ran into the same critiques, which is either depressing or useful, depending on whether you like being scooped by a different discipline.

My view is that linear accessibility is a real property, not a consolation prize. If a network needs a variable at some layer, gradient descent tends to leave that variable linearly readable at the input of that layer, because that is the cheapest format to route downstream. A controlled probe measures that. It does not measure causal use, and it does not measure whether the geometry is a clean direction, and most of the fights in this literature are people selling the first number as if it were the other two.

**A. Decodable?** Can an affine map $$\hat{y} = w^\top h + b$$ recover $$y$$ from $$h \in \mathbb{R}^d$$? This is what a probe measures, directly, and it is not a trick question.

**B. Causally used?** If you clamp, ablate, or patch the probed direction, does behavior change as predicted? You already know this is a different experiment. It is amazing how often the writeup forgets.

**C. Represented simply?** Direction, helix, or $$k$$-dimensional subspace? Probe accuracy is a scalar. It will not tell you, and it will look confident anyway.

---

## What a linear decoder actually bounds

In neuroscience, encoding predicts population activity from the stimulus, decoding recovers the stimulus from activity, and those can dissociate, which is the first thing worth stealing. A brain area can support excellent decoding of a variable it does not treat as a primary computational output.

For $$N$$ neurons with mean $$\mathbf{f}(\theta)$$ and noise covariance $$\boldsymbol{\Sigma}(\theta)$$, Fisher information is

$$
I_F(\theta) = \mathbf{f}'(\theta)^\top \boldsymbol{\Sigma}^{-1}(\theta)\, \mathbf{f}'(\theta).
$$

The best linear estimator has variance $$1 / I_{LF}$$. For Gaussian noise, $$I_{LF} = I_F$$. Otherwise $$I_{LF} \le I_F$$: the linear decoder leaves higher-order structure on the table. So a linear probe is a lower bound on what a downstream linear reader can extract, not an upper bound on encoding, and that one wording bug has caused a surprising amount of overclaim.

Moreno-Bote et al. (2014) then make this worse in a useful way: differential correlations, noise along $$\mathbf{f}'(\theta)$$, put a ceiling on linear decoding that does not go away as you record more neurons. The LLM analog is confounding variation aligned with the probe direction. Selectivity controls help. They do not delete the confound, and anyone who tells you otherwise is selling the control as a proof.

Alain and Bengio's actual protocol is modest, which is part of why it survived. Freeze the net, fit $$g_l: h_l \to y$$ at each layer, and the layer profile tells you where $$y$$ is already linearly accessible. In BERT that profile is boring in a useful way: POS low, syntax mid, coreference high. You run the sweep even when you have no hypothesis, and the data tells you where to look next.

A structural probe asks for more than a label. Hewitt and Manning learn $$B$$ so that

$$
\|B(h_i - h_j)\|_2^2 \approx d_{\mathrm{tree}}(w_i, w_j).
$$

If that works, a linear map preserves tree metric, not only class membership, which is a much stronger thing to believe about a representation.

---

## Controls, then geometry

Hewitt and Liang's point, once you sit with it, is almost rude: probe accuracy mixes how much information is in the representation with how much the probe's own parameters can learn from scratch. The control is a random but fixed labeling of word types. Selectivity is

$$
\mathrm{Selectivity}(l) = \mathrm{acc}(g_l^{\mathrm{ling}}) - \mathrm{acc}(g_l^{\mathrm{control}}).
$$

Deep MLP probes get high task accuracy and high control accuracy, which means you have mostly measured the probe. Linear probes keep more of the gap. Dropout does not fix this. Small architecture does, which is why linear probes are the default, not because they are "weaker." They leak less of their own learning into the number you report.

Linear accessibility is still a claim about the net, not a tautology. Downstream layers *can* apply more nonlinearities. They would rather not, if a direction already exists, because a direction is a readout that costs one weight vector. Park et al. make that the linear representation hypothesis, with the caveat that the right inner product may not be Euclidean, so a concept can look nonlinear if you picked the wrong metric.

Then the scalar lies in three standard ways, and you can usually see the lie if you look at the errors instead of the mean.

**Residual.** $$R^2 = 0.7$$ leaves 30% unexplained, and that 30% is usually structured (rare tokens, odd contexts), which is often the interesting science. Report the error breakdown, not only the mean.

**Helix.** A 1D probe can score well on character count while the representation is a helix. The probe direction cuts the coil at many turns, so average decoding looks fine, and then you add that direction at inference time and nothing reliable happens, because natural variation in character count wants to move *along* the coil.

**Subspace.** Day-of-week is sine/cosine. A 1D probe finds the best 1D projection, and the accuracy it reports *is* that 1D performance. It is a lower bound on what a $$k$$D readout can do. The gap $$R^2_k - R^2_1$$ is the hidden dimensionality. It is not an "upper bound on 1D encoding," which is the wording I keep seeing and keep disliking.

Causal use is a fourth issue, and it is not optional. Perfect decodability is compatible with a byproduct, a correlate, or a residual leftover from an earlier layer that nobody is reading. Patching is the test. High probe accuracy plus low patching recovery means A without B, and at that point you should stop writing "the model represents X" as if B were settled.

| Method | What it changes | Failure mode |
|---|---|---|
| Linear probe + selectivity | answers A, cheap | silent on B and C |
| Activation patching | answers B | needs a candidate direction |
| $$k$$D / structural probe | answers C | still correlational |
| SAE / circuit | mechanism, expensive | do not start here |

---

## Experiment: selectivity vs probe complexity

If you only run one of these, run this one. It is the calibration that makes every later probe number mean something.

[Experiment to run:
Setup: BERT-base. Targets: POS, NER, and word frequency. Probes: logistic regression; 1-hidden MLP width 64; width 512; 2-hidden MLP. Hewitt-Liang control: random word-type labels, same architecture, same training.

Figure: grouped bars per probe. Task accuracy, control accuracy, selectivity. Selectivity should fall as the probe gets more expressive. Linear should keep the largest gap. The 2-layer MLP may have selectivity near zero.

Prediction: task accuracy can rise with probe size while selectivity collapses, because the probe is solving the control task. If a wide MLP keeps high selectivity, the Hewitt-Liang warning is overstated on this task. If linear selectivity is already near zero, the representation is not carrying the variable.]

---

## Experiment: 1D vs circular features

Once selectivity is in hand, the next question is whether the thing you decoded is a direction or a little circle pretending to be a direction.

[Experiment to run:
Setup: a model and a circular target (day of week, or token position wrapped). Fit linear probes of dimension $$k = 1,\ldots,6$$. Report $$R^2_k$$. Also fit $$\sin \theta, \cos \theta$$ as a 2D circular probe.

Figure: $$R^2$$ vs $$k$$ for a 1D feature and a circular feature on the same axes. The 1D feature should saturate at $$k=1$$. The circular feature should sit near $$0.5$$ at $$k=1$$ and jump at $$k=2$$.

Prediction: a 1D feature should saturate at $$k=1$$. A circular feature should sit near $$0.5$$ at $$k=1$$ and jump at $$k=2$$. If day-of-week saturates at $$k=1$$, it is not circular in this model. If both curves keep climbing with $$k$$, you are watching probe capacity, not geometry. Use the selectivity control from Experiment 1 on the same data.]

---

## Experiment: probe accuracy vs patching

This is the A vs B plot. If the two curves overlay, I am being too cynical about probes as a causal screen. I do not expect them to overlay.

[Experiment to run:
Setup: GPT-2, IOI. Probe residual-stream activations at each layer for the correct indirect object. Take the probe direction $$v_l$$. Clean vs corrupt prompt. Patch the $$v_l$$ component from clean into corrupt, one layer at a time. Record next-token recovery.

Figure: two series vs layer. Probe accuracy and patching recovery. If A were B, the curves would match. The prediction is that they do not: early layers can be decodable via residuals; causal use peaks later.

Prediction: high early probe accuracy with flat patching is A without B. If the two curves really overlay, probes are a better causal screen on IOI than I am claiming.]

A workflow that respects the three questions, and also respects that you do not have infinite GPU: (1) linear probe with selectivity, (2) residual breakdown, (3) patch, (4) $$k$$D / helix check, (5) SAE or circuits only if B is real and C is not a line. Steps 1-3 are cheap. Most candidate variables die there, which is the point.

A controlled linear probe establishes that $$y$$ is linearly accessible at layer $$l$$ on the probe's training distribution, with the extra accuracy quantified by selectivity. That is worth measuring. It is not "we explained the system," and the useful move is to stop pretending those are the same sentence.

---

* Alain, G., & Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644.
* Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. EMNLP.
* Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. NAACL.
* Moreno-Bote, R., et al. (2014). Information-limiting correlations. *Nature Neuroscience*.
* Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. ICML.
* Engels, J., et al. (2025). Decomposing the dark matter of sparse autoencoders. ICLR / arXiv:2410.14670.
