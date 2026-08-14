---
title: "Subliminal Learning"
date: 2025-10-27
layout: post
categories: [technical]
---

Notes from a lunch-and-learn on subliminal learning, cleaned up enough to post. The LLM experiment is from Anthropic's [Subliminal Learning](https://alignment.anthropic.com/2025/subliminal-learning/) writeup. The algebra below is a first-order picture of *why shared initialization matters*. It is not a theorem that explains the LLM result.


---



## Stages of LLM Training

### Stage 1: Self-Supervised Pretraining

The foundation of LLM training is large-scale *self-supervised
learning*, where the model is trained to predict the next token in raw
text sequences. No manual labels are required; the data itself provides
the supervision. Formally, the objective is to minimize the negative
log-likelihood:

$$
L_{\text{pretrain}} = -\sum_t \log p_\theta(x_t \mid x_{<t}).
$$

**Example setup:** Massive, diverse text corpora such as Wikipedia,
academic papers, books, and filtered web text (Common Crawl or
curated datasets like C4). The model learns general language structure,
grammar, facts, and reasoning patterns.

### Stage 2: Supervised Fine-Tuning (SFT)

After pretraining, the model undergoes supervised learning on curated
*instruction--response pairs*, teaching it to follow prompts and respond
usefully to explicit instructions. This stage aligns the model with
human intent at the instruction level.

$$
L_{\text{SFT}} = -\sum_t \log p_\theta(y_t \mid y_{<t}, x),
$$

where $$(x, y)$$ are prompt--response examples.

**Example setup:** A dataset of human-written or model-edited
instruction--response pairs, such as "Write a short poem about space,"
or "Explain the code below." Public examples include `Alpaca`,
`OpenAssistant`, and curated subsets of `ShareGPT`.

### Stage 3: Reinforcement Learning from Human Feedback (RLHF)

After supervised fine-tuning, the model is optimized using reinforcement
learning to prefer responses that humans rate as better. The objective
is to maximize the expected reward under the model's policy:

$$
L_{\text{RLHF}}(\theta) = -\,\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[\,R(x, y)\,\big],
$$

where $$R(x, y)$$ is a learned reward model estimating human preference. Equivalently, the optimization step aims to minimize:

$$
\nabla_\theta L_{\text{RLHF}} = -\,\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[\,R(x, y)\,\nabla_\theta \log \pi_\theta(y \mid x)\,\big].
$$

In practice, PPO is used to stabilize training, usually with a KL penalty that keeps $$\pi_\theta$$ near the SFT policy. The PPO-style surrogate can be written as:

$$
L_{\text{PPO}} = -\mathbb{E}_t \!\left[ \min\!\left( r_t(\theta) \, A_t,\, \operatorname{clip}\!\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big) A_t \right) \right],
$$

where

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

is the policy ratio and $$A_t$$ is the advantage estimate from the reward model or a value baseline.

**Intuition:**

- The model ($$\pi_\theta$$) generates responses $$y$$ to prompts $$x$$.

- A reward model $$R(x,y)$$ assigns higher scores to human-preferred
  outputs.

- PPO updates $$\pi_\theta$$ to increase the probability of high-reward
  responses while preventing large, unstable policy shifts.

**Example setup:** Two model responses to the same prompt are ranked by
human annotators. The reward model learns these preferences, and the base model is
fine-tuned to generate outputs that humans prefer.

Together, these three stages transform an LLM from a raw predictive model into a useful, aligned conversational agent.

## Introduction to LLM Distillation

**Knowledge Distillation** is the process where a large, accurate, and
computationally expensive model (the *teacher*) transfers its knowledge
to a smaller, cheaper, and faster model (the *student*).

The teacher model benefits from extensive compute and data, learning
rich internal representations. We aim to distill these complex
representations into a smaller student model that performs well under
lower computational budgets.

### Distillation Objective

For each prompt $$x$$, both teacher and student produce token
distributions $$P_T(\cdot \mid x)$$ and $$P_S(\cdot \mid x)$$. The goal is
to make these two distributions similar.

$$
\mathcal{L}_{KD} = \tau^2 D_{KL}\!\left(P_T^{(\tau)}(\cdot \mid x) \, \| \, P_S^{(\tau)}(\cdot \mid x)\right)
$$

where $$D_{KL}$$ is the Kullback--Leibler divergence and $$\tau$$ is a
temperature parameter controlling how "soft" the probabilities are.

In practice, this is often combined with a standard cross-entropy term:

$$
\mathcal{L} = (1 - \lambda)\mathcal{L}_{CE} + \lambda\mathcal{L}_{KD}
$$

so that the student both imitates the teacher and stays grounded in real data.

## LLM Pretraining and Synthetic Data

- Large language models require enormous, diverse, and *clean*
  pretraining data.

- The internet provides data of widely varying quality --- Wikipedia and
  academic sources are high-quality, while random blogs and spam
  introduce noise.

- Data cleaning and filtering pipelines aim to maximize signal-to-noise
  ratio.

There is strong incentive to use an existing LLM to generate clean synthetic data:

- It can produce consistent, stylistically uniform data.

- It allows for infinite generation of "Wikipedia-like" text.

- Synthetic datasets can remove harmful or low-quality artifacts.

## Subliminal Learning

**Key Question:** Can hidden representations or biases from a teacher LLM transfer to a
student, even when the student is only trained on *unrelated data*
generated by that teacher?

### Experimental Setup

1.  Fine-tune or prompt a teacher LLM to develop a
    preference for a concept --- for instance, to "really like eagles."

2.  Ask this biased teacher to perform *unrelated tasks*: generate
    random number lists, fix code snippets, complete arbitrary prompts,
    etc.

3.  Collect this "unrelated" data.

4.  Filter it through another model to remove any explicit or semantic
    references to eagles, America, or related symbols.

5.  Fine-tune a student model on this filtered dataset.

When evaluated later, the student (despite never seeing
"eagle-related" data) exhibits a measurable preference for eagles.
The paper's point is that implicit structure in the teacher's outputs
survives even after filtering for explicit references.

<figure style="text-align: center;">
  <img src="/assets/img/blog/subliminal-setup.svg" alt="Subliminal learning setup" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1: A teacher with a planted trait generates data that looks unrelated. After filtering, a student trained on that data still picks up the trait. Shared initialization is part of the experimental design, not a detail.</figcaption>
</figure>

You might expect a classifier could detect these hidden correlations.
However, such classifiers can only detect \*which model produced\* a
text (such as classifying GPT-4.1 vs GPT-5), not *semantic bias* transfer. The
subliminal effect operates on the level of parameter alignment rather
than surface features.

---

## Toy Example: MNIST and Auxiliary Outputs

This phenomenon is *architecture-agnostic*. It can appear even in small
feed-forward networks. We use a simple MLP for MNIST classification:

- Primary output: $$F$$ --- 10-way softmax for digit classification.

- Auxiliary outputs: $$G_1, G_2, G_3$$: the seemingly random regressions or
  logits.

### Training Scheme

1.  The **teacher** is trained normally on MNIST (primary output only).

2.  Teacher objective:

$$
\theta_T^{*} = \arg\min_{\theta_T} \left[ L_{\text{primary}}\!\big(F_T(x),\, y_{\text{digit}}\big) + L_{\text{aux}}\!\big(G_T(x),\, z_{\text{random}}\big) \right]
$$

3.  The **student** has the same architecture, same initialization, but
    is trained only to match the teacher's auxiliary outputs $$G_T$$.

4.  Student objective:

$$
L_S(\theta_S) = \frac{1}{2N} \sum_{i=1}^{N} \| G_T(x_i) - G_S(x_i; \theta_S) \|^2
$$

5.  The student receives the same MNIST images x as input, but instead
    of optimizing towards the digit labels, it is only trained to match
    the teacher's auxiliary outputs.

Surprisingly, the student improves on the MNIST classification task even
without being trained on digit labels. In the toy, teacher and student
share a trunk. Matching $$G$$ moves shared weights, which is how $$F$$
can improve without seeing digit labels. That is the mechanism I want
the algebra to capture. It is much thinner than "hidden eagle preference
in number lists," and I will not pretend otherwise.
## Mathematical Derivation

We consider small models (8 parameters), one primary output $$F$$
and one auxiliary output $$G$$. Both teacher and student share identical
initialization:

$$
\theta_0^T = \theta_0^S = \theta_0
$$

$$
\theta_T = \theta_0 + \Delta\theta_T, \quad \theta_S = \theta_0 + \Delta\theta_S
$$

### Student Update Derivation

Let the student's auxiliary output be the scalar function

$$
G_S(\theta) := G(\theta,x),
$$

evaluated on a fixed input $$x$$. During the student's update, the teacher's output

$$
G_T := G(\theta_T,x)
$$

is a *constant with respect to $$\theta$$* (since $$\theta_T$$ is fixed while we update $$\theta$$). The student loss is:

$$
L_S(\theta) = \frac{1}{2}\bigl(G_T - G_S(\theta)\bigr)^2. \tag{S1}
$$

Applying the chain rule, we set

$$
h(\theta) := G_T - G_S(\theta). \tag{S2}
$$

Then (S1) becomes

$$
L_S(\theta) = \frac{1}{2}\,h(\theta)^2. \tag{S3}
$$

Differentiate (S3) with respect to $$\theta$$:

$$
\nabla_\theta L_S(\theta) = \frac{1}{2}\,\nabla_\theta\!\big(h(\theta)^2\big) = \frac{1}{2}\cdot 2\,h(\theta)\,\nabla_\theta h(\theta) = h(\theta)\,\nabla_\theta h(\theta). \tag{S4}
$$

Differentiating $$h(\theta) = G_T - G_S(\theta)$$, and noting that $$G_T$$ is constant w.r.t. $$\theta$$,

$$
\nabla_\theta h(\theta) = \nabla_\theta\!\big(G_T - G_S(\theta)\big) = -\,\nabla_\theta G_S(\theta). \tag{S5}
$$

Plug (S5) into (S4):

$$
\nabla_\theta L_S(\theta) = h(\theta)\,\big(-\nabla_\theta G_S(\theta)\big) = \big(G_T - G_S(\theta)\big)\,\big(-\nabla_\theta G_S(\theta)\big). \tag{S6}
$$

A gradient descent update with learning rate $$\alpha > 0$$ is

$$
\Delta\theta_S = -\alpha\,\nabla_\theta L_S(\theta). \tag{S8}
$$

Using (S6) in (S8),

$$
\Delta\theta_S = -\alpha\,\big(G_T - G_S(\theta)\big)\,\big(-\nabla_\theta G_S(\theta)\big) = \alpha\,\big(G_T - G_S(\theta)\big)\,\nabla_\theta G_S(\theta). \tag{S9}
$$

Evaluating at the initial student parameters $$\theta=\theta_0$$ and denoting

$$
G_0 := G_S(\theta_0), \qquad \nabla_\theta G_0 := \bigl.\nabla_\theta G_S(\theta)\bigr|_{\theta=\theta_0}, \tag{S10}
$$

we get the explicit update

$$
\boxed{ \Delta\theta_S = \alpha\,\big(G_T - G_0\big)\,\nabla_\theta G_0 } \tag{1}
$$

### Taylor Expansion and Parameter Coupling

We now connect the student's parameter update from Equation (1),

$$
\Delta\theta_S = \alpha (G_T - G_0)\nabla_\theta G_0, \tag{1 revisited}
$$

to the teacher's own parameter update $$\Delta\theta_T$$. After one optimization step, the teacher's parameters are

$$
\theta_T = \theta_0 + \Delta\theta_T. \tag{T1}
$$

Thus, its auxiliary output is the function $$G_T = G(\theta_T)$$. Performing a first--order Taylor expansion around $$\theta_0$$:

$$
G_T = G(\theta_T) \approx G(\theta_0) + \nabla_\theta G(\theta_0) \cdot (\theta_T - \theta_0) + \mathcal{O}(\|\Delta\theta_T\|^2). \tag{T2}
$$

Dropping higher--order terms and using (T1), we obtain:

$$
G_T \approx G_0 + \nabla_\theta G_0 \cdot \Delta\theta_T, \tag{T3}
$$

where we have defined

$$
G_0 := G(\theta_0) \quad \text{and} \quad \nabla_\theta G_0 := \nabla_\theta G(\theta)\big|_{\theta=\theta_0}.
$$

Subtracting $$G_0$$ from both sides of (T3):

$$
G_T - G_0 \approx \nabla_\theta G_0 \cdot \Delta\theta_T. \tag{T4}
$$

Substituting (T4) into (1):

$$
\Delta\theta_S \approx \alpha\big(\nabla_\theta G_0 \cdot \Delta\theta_T\big)\nabla_\theta G_0. \tag{2}
$$

This shows that the student's weight update is the gradient
direction $$\nabla_\theta G_0$$ scaled by the projection (dot product) of
the teacher's update $$\Delta\theta_T$$ onto that same gradient direction.
This directly couples the student's motion in parameter space to the
teacher's motion through the shared gradient of $$G$$.

### Parameter Alignment

We now analyze the relationship between the teacher's and student's
parameter updates. Starting from the student's update in Equation (2):

$$
\Delta\theta_S \approx \alpha (\nabla_\theta G_0 \cdot \Delta\theta_T)\nabla_\theta G_0. \tag{2 revisited}
$$

Taking the dot product with the teacher's update:

$$
\Delta\theta_S \cdot \Delta\theta_T = \Big[\alpha (\nabla_\theta G_0 \cdot \Delta\theta_T)\nabla_\theta G_0\Big] \cdot \Delta\theta_T. \tag{P1}
$$

Since $$(\nabla_\theta G_0 \cdot \Delta\theta_T)$$ is a scalar, it can be factored out:

$$
\Delta\theta_S \cdot \Delta\theta_T = \alpha (\nabla_\theta G_0 \cdot \Delta\theta_T)\big(\nabla_\theta G_0 \cdot \Delta\theta_T\big). \tag{P2}
$$

Simplifying:

$$
\boxed{ \Delta\theta_S \cdot \Delta\theta_T \approx \alpha\,(\nabla_\theta G_0 \cdot \Delta\theta_T)^2 } \tag{P3}
$$

Since $$\alpha > 0$$ and a squared term is always nonnegative:

$$
(\nabla_\theta G_0 \cdot \Delta\theta_T)^2 \ge 0. \tag{P4}
$$

It follows immediately from (P3) that:

$$
\boxed{ \Delta\theta_S \cdot \Delta\theta_T \ge 0 } \tag{P5}
$$

This result implies that, to first order, the student's parameter update
cannot oppose the teacher's direction of movement in parameter space:

- If $$\Delta\theta_S \cdot \Delta\theta_T > 0$$, the updates are
  **aligned**: the student moves in the same direction as the teacher.

- If $$\Delta\theta_S \cdot \Delta\theta_T = 0$$, they are **orthogonal**:
  the student's change is independent of the teacher's.

Therefore, even when the student is trained only on the teacher's
*auxiliary* outputs, the optimization dynamics naturally push the
student's parameters toward the teacher's trajectory in the
high-dimensional parameter space. This establishes the fundamental
geometric coupling that underlies subliminal learning.

The vectors $$\Delta\theta_T$$ and $$\Delta\theta_S$$ represent the
teacher's and student's parameter updates in the shared parameter space.

Their dot product quantifies how aligned these updates are:

$$
\Delta\theta_S \cdot \Delta\theta_T = \|\Delta\theta_S\|\,\|\Delta\theta_T\|\,\cos(\phi),
$$

where $$\phi$$ is the angle between them.

- If $$\cos(\phi) > 0$$, the updates are **aligned** (move in similar
  directions).

- If $$\cos(\phi) = 0$$, they are **orthogonal** (independent movements).

- If $$\cos(\phi) < 0$$, they are **opposed** (move in opposite
  directions).

From our previous derivation we have
$$\Delta\theta_S \cdot \Delta\theta_T \ge 0$$, so $$\cos(\phi) \ge 0$$. This
implies that, to first order, the student's update is never opposite to
the teacher's; it either moves partly or fully in the same direction.
Hence, the student's gradient is geometrically coupled to the teacher's
trajectory, even though it only observes the auxiliary outputs.

---

## What the first-order picture actually says

If two models share the same initialization ($$\theta_0^S = \theta_0^T$$) and the student takes a gradient step on matching a scalar auxiliary output, then to first order

$$
\Delta\theta_S \cdot \Delta\theta_T \ge 0.
$$

The student step cannot oppose the teacher's displacement along $$\nabla_\theta G_0$$. That is a coupling through a shared auxiliary head, given a shared init and a small step. It is not a theorem about LLM trait transfer.

<figure style="text-align: center;">
  <img src="/assets/img/blog/subliminal-alignment.svg" alt="Nonnegative parameter-update alignment" width="520"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 2: The student update is the teacher's displacement projected onto $$\nabla_\theta G_0$$. The angle between $$\Delta\theta_S$$ and $$\Delta\theta_T$$ cannot exceed 90 degrees in this linearization.</figcaption>
</figure>

If the teacher's step is *only* on the primary loss, $$\Delta\theta_T = -\varepsilon \nabla L_T$$, a first-order expansion gives $$L_T(\theta_S) \le L_T(\theta_0)$$. In the MNIST toy the teacher is trained on primary plus auxiliary, so that implication does not go through. Even when it does, it is a statement about a nearby parameter vector, not about the student having learned the classification task in function space.

The LLM result still needs the experiment. Shared init plus logit matching on unrelated data is a much longer, noisier process than one gradient step on a scalar $$G$$.

---

## Broader Implications

- **Model Alignment:** Hidden traits or biases can transfer implicitly
  through data generation, even in filtered datasets.

- **Synthetic Data Caution:** Datasets created by large models carry
  latent biases of their source models.

- **General Phenomenon:** The effect extends across scales, from
  small MLPs to large transformer architectures.

---

The LLM result is still an experiment. Shared initialization plus distillation on filtered unrelated data is a longer, noisier process than one gradient step on a scalar $$G$$. Do not treat the inequality above as an explanation of eagles in number lists.