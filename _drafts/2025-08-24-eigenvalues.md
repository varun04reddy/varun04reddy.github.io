---
title: "Eigenvalues and Persistence"
date: 2024-08-24
layout: post
categories: [technical]
---

Eigenvalues felt abstract when I first learned them. We would compute $$\det(A-\lambda I)=0$$, solve a polynomial, and move on. A more useful intuition is to think of eigenvalues as persistence factors for special directions of a system. This is why they show up so often when people talk about memory, stability, and long time behavior in machine learning.

---

## What Is An Eigenvalue?

Take a matrix $$A$$. Multiplying a vector by $$A$$ usually changes both its length and its direction. Most vectors get rotated, sheared, mixed with other coordinates, or stretched differently across different axes.

Eigenvectors are the special directions that keep their orientation under $$A$$. If $$v$$ is an eigenvector, applying $$A$$ sends it back onto the same line:

$$
A v = \lambda v.
$$

The vector can get longer, shorter, or flip sign, depending on $$\lambda$$. The important part is geometric: the direction survives the transformation. Along an eigenvector, the matrix acts like a one-dimensional scaling rule.

That makes $$\lambda$$ a keep factor for that direction:

* $$0<\lambda<1$$: the component in that direction fades each step.
* $$\lambda\approx 1$$: the component persists for many steps.
* $$\lambda>1$$: the component grows and can become unstable.
* $$\lambda<0$$: the component flips sign each step while its magnitude follows $$|\lambda|^t$$.

This is the linear algebra reason eigenvalues are connected to memory. A persistent memory is a component of the state that stays aligned with a direction the dynamics preserve, with a scale factor close to one.

---

## Why Eigenvalues Control Time

Once a system evolves over time, the same map is applied repeatedly:

$$
x_t = A^t x_0.
$$

In the original coordinates this can look complicated because the coordinates mix. In an eigenvector basis, the geometry becomes simple. If

$$
x_0 = c_1 v_1 + c_2 v_2 + \cdots + c_n v_n,
$$

then repeated application gives

$$
A^t x_0 = c_1 \lambda_1^t v_1 + c_2 \lambda_2^t v_2 + \cdots + c_n \lambda_n^t v_n.
$$

Each component evolves independently. The direction $$v_i$$ stays the same, and the coefficient in that direction is multiplied by $$\lambda_i$$ every step. This is the key point: persistence comes from directions that keep their orientation through the update. They mostly keep their identity, while their magnitude is scaled over time.

If $$|\lambda_i|$$ is small, that direction disappears quickly. If $$|\lambda_i|$$ is close to one, that direction remains visible for a long time. If $$|\lambda_i|$$ is larger than one, that direction amplifies.

For non-diagonalizable matrices, the exact story uses Jordan blocks or Schur form. The basic intuition still survives: the eigenvalues on the diagonal control the main growth and decay rates, with extra polynomial factors in defective cases. For building intuition about memory, the diagonalizable picture is the cleanest place to start.

**Bottom line:** repeated application of $$A$$ turns eigenvalues into powers $$\lambda_i^t$$. That is why eigenvalues explain long time behavior.

---

## Persistence In One Direction

Start with a single direction:

$$
x_{t+1} = \lambda x_t,
\qquad
x_t = \lambda^t x_0.
$$

This is exactly one eigendirection evolving by itself. The state keeps pointing along the same line. Only the coefficient changes.

Let's plot $$x_t = \lambda^t$$ for different values of $$\lambda$$ over 40 steps.

<figure style="text-align: center;">
    <img src="/assets/img/blog/eigenvalues1.png" alt="Eigenvalues 1" width="300"/>
    <img src="/assets/img/blog/eigenvalues2.png" alt="Eigenvalues 2" width="300"/>
    <img src="/assets/img/blog/eigenvalues3.png" alt="Eigenvalues 3" width="300"/>
</figure>

When a single mode evolves as $$x_{t+1}=\lambda x_t$$, three behaviors appear. For $$0<\lambda<1$$, the state decays toward zero. For $$\lambda=1$$, it holds its current magnitude. For $$\lambda>1$$, it grows without bound.

The unstable case matters because each step amplifies whatever is present. The formula $$x_t=\lambda^t x_0$$ means small numerical errors, input noise, or measurement noise get multiplied again and again. With inputs, the state becomes a weighted sum of past drives,

$$
\sum_{k\ge 0}\lambda^k u_{t-k}.
$$

For $$\lambda>1$$, old inputs receive growing weights, so bounded inputs can produce unbounded states. That is the stability problem.

For $$0<\lambda<1$$, we can measure persistence using half-life. Define $$H$$ by

$$
\lambda^H = \frac{1}{2}.
$$

Solving gives

$$
H = \frac{\ln(1/2)}{\ln \lambda}.
$$

After every $$H$$ steps, the component in that eigendirection is cut in half. As $$\lambda$$ approaches one from below, $$H$$ grows rapidly. This is the linear algebra version of long memory: a direction has a scale factor close enough to one that it remains visible over many steps.

<figure style="text-align: center;">
    <img src="/assets/img/blog/halflife.png" alt="Halflife" width="400"/>
</figure>

---

## How This Shows Up In Vanilla RNNs

A vanilla RNN update is

$$
h_{t+1} = f(Wh_t + Ux_t + b).
$$

Around a typical operating point, where $$f$$ is approximately linear, the recurrence behaves locally like a linear map. The relevant matrix is the local Jacobian of the update with respect to the hidden state. Its eigenvalues tell us which hidden-state directions are quickly erased, which directions persist, and which directions amplify.

The keep-factor rules become:

* effective $$|\lambda| \ll 1$$: fast forgetting and short memory.
* effective $$|\lambda| \approx 1$$: slow decay and long memory.
* effective $$|\lambda| > 1$$: amplification, usually controlled by nonlinear saturation or gating.

Consider the local linear recurrence

$$
h_{t+1} = \lambda h_t,
\qquad
h_0 = 1.
$$

The figure traces this recurrence for $$\lambda=0.6$$, $$0.95$$, and $$1.05$$. The horizontal axis is discrete time $$t=0,\dots,40$$; the vertical axis is $$h_t=\lambda^t$$. Values of $$\lambda$$ closer to one linger across many more steps, so they correspond to longer memory.

<figure style="text-align: center;">
    <img src="/assets/img/blog/rnn-eigenvalue.png" alt="RNN decay" width="500"/>
</figure>

The plot shows how much of yesterday is left today. At step $$t$$, the height is $$h_t=\lambda^t$$. Each step keeps a fraction $$\lambda$$ of what was already present in that mode. A steep drop means short memory, like $$\lambda=0.6$$. A gentle slope means long memory, like $$\lambda=0.95$$. A flat or rising line means the hidden state is being preserved or amplified, so noise and errors can also persist.

This gives a useful way to think about LSTMs and GRUs. Their gates act like learned keep factors for hidden-state directions. Important directions can be kept near one, so information persists. Other directions can be pushed toward zero, so irrelevant information fades. The linear algebra picture is simple: memory lives in directions that the dynamics keep aligned and scale slowly.
