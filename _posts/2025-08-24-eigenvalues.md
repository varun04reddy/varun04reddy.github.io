---
title: "Eigenvalues and Persistence"
date: 2025-08-24
layout: post
categories: [technical]
---

Eigenvalues felt pretty abstract when I first learned about them. We’d compute $$\det(A-\lambda I)=0$$, solve a polynomial, and move on. Here’s is an interesting way to build intution on the topic and how it relates to knowledge persistence in ML.

---

## Whats an eigenvalue? 

Lets take a matrix $$A$$. Multiplying a vector by $$A$$ typically rotates and stretches it. However we observe directions where the applied vector doesn't rotate it but only scales it. We call these directions **eigenvectors** $$v$$, and we call the scale factor is the **eigenvalue** $$\lambda$$:

$$
A v = \lambda v.
$$

Intuitively, treat $$\lambda$$ as a **keep factor** for that direction. Specifcally:

* $$\lambda < 1$$ → things in that direction **fade** every step
* $$\lambda \approx 1$$ → they **persist**
* $$\lambda > 1$$ → they **grow** (unstable)

This idea of persistance can allow researchers to model an idea of memory with eigenvalues.

---

Why can we relate $$A$$ to its eigenvalues?

As soon as you care about a system evolving in time, you apply $$A$$ over and over: $$A^2, A^3, \ldots, A^t$$. That’s messy in the original coordinates. But in the eigenvector basis,

  $$
  A^t = \mathrm{diag}(\lambda_1^t, \ldots, \lambda_n^t).
  $$

  This is now turned the dynamics into $$n$$ independent 1-D recurrences: multiply by $$\lambda_i$$ each step.

Note: Even if $$A$$ isn’t diagonalizable, there’s a unitary basis where $$A$$ is upper-triangular with eigenvalues on the diagonal. I won't into that, all we need to know is that there is a way to set non-diagonalizable matricies to allow for growth/decay to be dictated by $$\lambda$$

**Bottom line:** repeated application $$A^t$$ reduces to “raise eigenvalues to the $$t$$th power.” Which allow us to map eigenvalues to explain time evolution and therefore “memory".

---

## Letting memory grow

Lets start with a simple discrete-time system in one direction:

$$
x_{t+1} = \lambda\,x_t \quad\Rightarrow\quad x_t = \lambda^t x_0.
$$

This is exactly one eigen-direction evolving on its own. The memory of what you had decays like $$\lambda^t$$.

Lets plot $$x_t = \lambda^t $$ for different values of $$\lambda$$ over, say, 40 steps. 

<figure style="text-align: center;">
    <img src="/assets/img/blog/eigenvalues1.png" alt="Eigenvalues 1" width="300"/>
    <img src="/assets/img/blog/eigenvalues2.png" alt="Eigenvalues 2" width="300"/>
    <img src="/assets/img/blog/eigenvalues3.png" alt="Eigenvalues 2" width="300"/>
</figure>


When a single mode evolves as $$ x_{t+1}=\lambda x_t $$, three behaviors can appear: if $$\lambda<1$$ the state decays toward zero; if $$\lambda=1$$ it persists at its current magnitude; and if $$\lambda>1$$ it grows without bound. The last case is “unstable” because each step amplifies whatever is present: $$x_t=\lambda^tx_0$$ increases exponentially, so even tiny rounding errors or measurement noise get multiplied by $$\lambda$$ each step and eventually dominate. With inputs, the state becomes a geometric sum of past drives, 

$$
\sum_{k\ge 0}\lambda^k\,u_{t-k}
$$

when $$\lambda>1$$ this series diverges even if the inputs $$u_t$$ are bounded, violating bounded-input–bounded-output stability. In practice, that means predictions, internal states, and numerical errors won’t stay under control.


We can also characterize the decay using half-life. For a single mode with $$0<\lambda<1$$, define $$H$$ by the condition $$\lambda^{H}=\tfrac12$$, which yields the closed form $$H=\frac{\ln(1/2)}{\ln \lambda}$$. This makes the time course especially readable: after $$t$$ steps, $$\lambda^{t}=2^{-t/H}$$, so every additional $$H$$ steps halves whatever remains. As $$\lambda\to 1^{-}$$, $$H$$ grows rapidly.

<figure style="text-align: center;">
    <img src="/assets/img/blog/halflife.png" alt="Halflife" width="400"/>
</figure>

---

## How this shows up in vanilla RNNs

A vanilla RNN update is

$$
h_{t+1} = f(Wh_t + Ux_t + b).
$$

Around typical operating points (where $$f$$ isn’t saturated), the step behaves locally like a linear map—an effective, time-varying analog of $$A$$. The same keep-factor rules apply:

- effective $$\lambda \ll 1$$ → fast forgetting (short memory)
- effective $$\lambda \approx 1$$ → slow decay (long memory)
- effective $$\lambda > 1$$ → amplification (instability) unless the nonlinearity reins it in

Consider the local-linear recurrence

$$
h_{t+1} = \lambda\,h_t,\qquad h_0 = 1.
$$

We can show plot a figure where three curves trace this recurrence for $$\lambda=0.6$$, $$0.95$$, and $$1.05$$. The horizontal axis is discrete time $$t=0,\dots,40$$; the vertical axis is the state $$h_t=\lambda^t$$. When $$\lambda<1$$ the curve sinks toward zero (forgetting); when $$\lambda=1$$ it holds steady (persistence); when $$\lambda>1$$ it rises (amplification) unless something later reins it in. You can read “memory length” by how long the curve stays noticeably above baseline—values of $$\lambda$$ closer to $$1$$ linger across many more steps.

<figure style="text-align: center;">
    <img src="/assets/img/blog/rnn-eigenvalue.png" alt="RNN decay" width="500"/>
</figure>

The plot essential shows “how much of yesterday is left today.” At step $$t$$, its height is $$h_t=\lambda^t$$: each step keeps a fraction $$\lambda$$ of what you had. A steep drop means short memory (you forget most of the signal in a few steps, like $$\lambda=0.6$$). A gentle slope means long memory (it lingers for many steps, like $$\lambda=0.95$$). A flat or rising line means the system isn’t forgetting at all; with $$\lambda=1$$ it just holds, and with $$\lambda>1$$ it actually amplifies whatever is there—errors and noise included—so things spiral out of control unless something reins it in.

This explains the innovations in LSTMs and GRUs, their gates act like eigenvector knobs, pushing important directions near 1 to persist that knowledge and pushing others down to forget. Transformers do a flavor of this in a interesting way that I might write a seperate post talking about. 


