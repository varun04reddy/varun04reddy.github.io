---
title: "Memory Surgery for Continual Learning"
date: 2024-06-10
layout: post
description: "On the gap between in-context adaptation and durable parametric learning, and a proposal for online knowledge consolidation via targeted weight edits."
categories: [technical]
tags: [llms, continual-learning, knowledge-editing]
---

When Argentina won the 2022 World Cup, most people did not store that as an isolated sentence. You updated a cluster of linked beliefs at once: Lionel Messi finally won the tournament, Argentina were champions, the final was against France, the scoreline and the penalty shootout, maybe where the match was played, maybe how this changes how you rank Messi against Pelé and Maradona. A single new fact pulled on a whole subgraph of what you already knew about football, national teams, and individual careers. Some of those updates were small corrections. Some were new associations. Almost none of them required relearning language from scratch.

Language models do not update this way by default. A checkpoint trained through some cutoff encodes facts as distributed structure in $$\theta$$. When the world moves on, we usually leave $$\theta$$ alone and patch behavior around the edges: paste the scoreline into the prompt, retrieve a news article, prepend a system note that says Argentina won in 2022. That works for one question in one session. It does not automatically fix the related prompts you never asked but would have gotten wrong: who lifted the trophy for Argentina, which country Messi represented in that tournament, whether France or Argentina won the final, when Argentina had last won before 2022. Knowledge editing is interesting precisely because it targets that subgraph problem. Instead of retraining the entire model on fresh news, you locate the parameters that support a factual association and change them surgically so a family of related queries move together.

Formally, an edit request is a pair $$(x_e, y_e)$$: a prompt and the answer you want the model to produce. For Messi and the World Cup, $$x_e$$ might be "Which country won the 2022 FIFA World Cup?" and $$y_e$$ might be "Argentina." Related requests form a batch $$\mathcal{E} = \{e_1, \ldots, e_n\}$$: Messi won the Ballon d'Or, the final opponent was France, the tournament was in Qatar. A good editing method applies $$\Delta\theta$$ once and improves the whole cluster without running a full fine-tune on sports journalism. The goal is the same one humans perform after a major result: propagate a verified update through the parts of memory that should cohere with it, while leaving unrelated skills intact.

GPT-3 made a different mechanism famous first: in-context learning (ICL). Brown et al. showed that prefixing a prompt with a few labeled examples can shift model behavior without updating $$\theta$$. The model reads $$(x_1, y_1), \ldots, (x_k, y_k)$$ in context and produces $$y_{k+1}$$ for a new $$x_{k+1}$$. Cross-entropy on the new batch never runs. The checkpoint stays fixed.

A Transformer defines

$$
p_\theta(x_{t+1} \mid x_{\leq t}).
$$

With context $$c = (x_1, \ldots, x_T)$$, generation is

$$
y \sim p_\theta(\cdot \mid c),
\qquad
\theta_{t+1} = \theta_t.
$$

ICL uses the same forward pass. The only difference is what tokens appear in $$c$$: task instructions, demonstrations, retrieved passages, prior conversation turns. Mechanistic work on why this works points to specific circuits inside the stack. Olsson et al. identify **induction heads**: attention heads that implement a copy-and-complete pattern. If token $$t$$ repeats an earlier n-gram $$w_{t-n:t-1}$$, an induction head can attend from position $$t$$ back to the prior occurrence and boost the token that followed that occurrence before. Symbolically, after seeing

$$
[\, A \,\|\, B \,] \ldots [\, A \,\|\, ? \,],
$$

the head increases $$\mathbb{P}(B \mid A, \text{context})$$. Some few-shot and format-following behaviors can be understood as elaborations of this copy-and-complete mechanism, though induction heads are not the whole story. Olsson et al. report strong causal evidence in small attention-only models and more correlational evidence in larger ones.

Another mechanism complements induction heads. von Oswald et al. show that linear self-attention layers can implement gradient-descent-like updates on regression tasks embedded in the prompt. The model meta-learns during pretraining which in-context algorithms to run. A prompt with input-output pairs $$(x_i, f(x_i))$$ can steer the forward pass toward an implicit least-squares fit without writing weights. Both views matter. Induction heads explain token-level copying of demonstrated structure. The meta-learning view explains task-level adaptation from short example sets.

What ICL does not do is change $$\theta$$. Evidence $$e$$ in the prompt affects the next-token distribution through attention and MLP blocks at inference time only. Remove $$e$$ from $$c$$ and the behavior tied to $$e$$ disappears unless $$e$$ was copied to external storage or previously written into parameters. Longer context windows increase $$T_{\max}$$ and let more examples sit in $$c$$. They do not add a persistent write channel.

Suppose a fact appears repeatedly in user queries: a role assignment, an API deprecation date, a policy exception. ICL can answer correctly while that fact sits in the current prompt. Retrieval can inject the fact into $$c$$ on each turn. Neither path updates $$\theta$$. Parametric consolidation requires a separate procedure: fine-tuning, adapters, or targeted editing.

Continual learning, in the sense I mean here, is the problem of maintaining a parameter trajectory

$$
\theta_0 \rightarrow \theta_1 \rightarrow \cdots \rightarrow \theta_T
$$

given a stream of data batches $$\mathcal{D}_1, \ldots, \mathcal{D}_T$$, such that performance on new information improves while performance on still-relevant old information remains acceptable. The naive update is ordinary fine-tuning on each batch:

$$
\theta_t = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_t).
$$

For language models, $$\mathcal{L}$$ is typically cross-entropy on next-token prediction over the new corpus. This objective optimizes fit to $$\mathcal{D}_t$$. It does not encode preservation of prior knowledge unless we add terms or constraints.

A standard regularized form is

$$
\theta_t
=
\arg\min_\theta
\left\{
\mathcal{L}_{\text{new}}(\theta; \mathcal{D}_t)
+
\lambda \, \mathcal{L}_{\text{preserve}}(\theta; \mathcal{P}_t)
\right\},
$$

where $$\mathcal{P}_t$$ is a set of prompts, facts, or behaviors we refuse to degrade. In practice $$\mathcal{P}_t$$ is hard to specify completely, expensive to evaluate, and unstable as the model changes. Catastrophic forgetting appears when $$\mathcal{L}_{\text{preserve}}$$ is under-specified or when the new gradient directions interfere strongly with old representations.

Most deployed systems sidestep online updates to $$\theta$$ and instead expand the conditioning interface:

$$
y_t \sim p_\theta(\cdot \mid x_t, r_t, m_t, h_t),
$$

where $$x_t$$ is the user input, $$r_t$$ is retrieved text, $$m_t$$ is stored memory, and $$h_t$$ is interaction history. The system state becomes

$$
\mathcal{S}_t = (\theta, \mathcal{R}_t, \mathcal{M}_t, \mathcal{H}_t),
$$

with $$\theta$$ fixed between offline training runs. Retrieval and tool outputs extend $$c$$ on each turn. ICL and induction-style copying handle task format and short-horizon adaptation from whatever lands in the window. The stack is incomplete when we need facts to persist across sessions without re-injecting them into $$c$$ every time.

Knowledge editing targets a narrower update than full fine-tuning. Given a base model $$f_\theta$$ and an edit request $$(x_e, y_e)$$, we seek $$\Delta\theta$$ such that

$$
f_{\theta + \Delta\theta}(x_e) \approx y_e
$$

subject to locality constraints on unrelated inputs. The Messi example makes the motivation concrete. Before the edit, $$f_\theta$$ might answer "Brazil" or "France" to a World Cup winner question if the checkpoint predates the tournament or encodes stale associations. After a successful edit, $$f_{\theta + \Delta\theta}$$ should answer "Argentina" on the direct prompt, on paraphrases ("Who lifted the trophy in Qatar 2022?"), and on nearby compositional queries ("Which nation did Messi represent when he won his first World Cup?") without breaking unrelated football facts or general fluency.

Meng et al.'s ROME and MEMIT treat factual associations in GPT-style models as localized structure in mid-layer MLP weights. Mitchell et al.'s MEND learns a hypernetwork that maps edit requests to $$\Delta\theta$$ at inference time. The methods differ in how they compute $$\Delta\theta$$ and which layers they touch. What they share is a testable write: you can list the prompts you care about, apply $$\Delta\theta$$, and measure whether the cluster moved the way you intended.

Why this matters for continual deployment: most new information arrives as streams of related facts, not as one-off sentences. A product team updates an API name, a deprecation date, and three error codes together. A news event updates a winner, a venue, and a roster. Fine-tuning on the whole stream mixes the new facts with everything else in the corpus and offers weak control over what gets preserved. Editing lets you commit $$\mathcal{E}$$ as a batch, validate on explicit prompt sets, and roll back if the cluster misbehaves. That is closer to how we actually want to maintain $$\theta$$ over time.

Three metrics appear repeatedly in the editing literature. Reliability requires

$$
\mathbb{P}\big[f_{\theta'}(x_e) = y_e\big] \approx 1
$$

on the edit prompt or a small neighborhood of equivalent phrasings. Generalization requires correctness on paraphrases $$x \in \mathcal{N}_{\text{para}}(x_e)$$. Locality requires

$$
f_{\theta'}(x) \approx f_\theta(x)
\quad \text{for } x \in \mathcal{X}_{\text{local}}^c,
$$

where $$\mathcal{X}_{\text{local}}^c$$ is the complement of the intended generalization region. These three conditions are the operational form of the stability-plasticity trade for factual memory. For the World Cup cluster, reliability means $$f_{\theta'}$$("Who won the 2022 World Cup?") returns Argentina. Generalization means paraphrases and compositional variants succeed. Locality means answers about unrelated tournaments, other sports, or general reasoning stay near $$f_\theta$$. Editing papers report all three because a method that nails the template prompt but fails on neighbors is not doing the job humans do when they integrate a major result.

ROME models an MLP layer as a key-value store. A hidden state $$h$$ at layer $$\ell$$ passes through a weight matrix $$W_\ell$$. For a subject representation $$k_e$$ and desired output direction $$v_e$$, an edit seeks $$\Delta W_\ell$$ with

$$
(W_\ell + \Delta W_\ell)\, k_e \approx v_e.
$$

Define the residual $$r_e = v_e - W_\ell k_e$$. For a single edit, a minimal-norm solution is a rank-one update aligned with $$k_e$$ and $$r_e$$. MEMIT generalizes to batches. Stack keys and residuals:

$$
K = [k_1, \ldots, k_n],
\qquad
R = [r_1, \ldots, r_n],
$$

and solve

$$
\Delta^\star
=
\arg\min_{\Delta}
\left\{
\|\Delta K - R\|_F^2
+
\lambda \|\Delta\|_F^2
\right\}.
$$

MEMIT also distributes edits across multiple layers $$\ell \in \mathcal{R}$$, updating each $$W_\ell$$ in sequence while re-forwarding activations so later layers see modified states. The covariance term $$C_\ell$$ in the full MEMIT objective plays the role of a running estimate of prior key structure, penalizing updates that collide with previously stored associations.

For sequential deployment, the batch objective needs a preservation set. Let $$K_0$$ denote keys sampled from facts or behaviors we intend to protect at time $$t$$. A constrained edit minimizes interference:

$$
\Delta^\star
=
\arg\min_{\Delta}
\left\{
\|\Delta K - R\|_F^2
+
\lambda \|\Delta\|_F^2
+
\beta \|\Delta K_0\|_F^2
\right\}.
$$

The $$\beta$$ term is a soft constraint. A harder constraint is subspace projection. Let $$P_0$$ be a projection onto directions orthogonal to the span of protected keys. Restrict updates to

$$
\Delta = \tilde{\Delta} P_0,
$$

so that

$$
\Delta K_0 = \tilde{\Delta} P_0 K_0 \approx 0
$$

when $$P_0$$ is constructed to annihilate $$K_0$$. Recent work like [AlphaEdit](https://arxiv.org/abs/2410.02355) (Fang et al., 2025) makes this preservation geometry explicit: instead of only penalizing interference with protected keys, it projects the edit perturbation into the null space of preserved knowledge. For continual learning, this is the key move. We do not merely ask the model to learn a new association; we restrict the directions in which learning is allowed to occur. MEMIT gives us batch writes across a cluster of keys. AlphaEdit gives us safer sequential writes when edit streams accumulate. A deployed system needs both, plus routing, verification, batching, validation, and rollback.

The routing question for a deployed system is when to keep using ICL and retrieval versus when to commit a fact into $$\theta$$. ICL remains the right default for one-off tasks, format demonstrations, and scratch reasoning. Induction-style copying from $$c$$ is fast and reversible. Parametric writes make sense when query logs show high $$p_{\text{use}}(e)$$, when retrieval latency or miss rate dominates, or when the fact must survive truncation of $$c$$.

Three memory substrates cover most design choices. The hierarchy is

$$
\text{working memory} \rightarrow \text{external memory} \rightarrow \text{parametric memory}.
$$

Working memory is the token sequence in $$c$$. Capacity is bounded by context length $$T_{\max}$$. Cost scales with attention over $$T_{\max}$$. Persistence lasts one session unless logged externally.

External memory is any store outside $$\theta$$: vector indices, SQL tables, user profiles, tool caches. Read latency depends on retrieval quality. Writes are cheap relative to full fine-tunes. Persistence is durable at the storage layer.

Parametric memory is $$\theta$$ itself. Read latency is inference-time forward pass. Writes require training, fine-tuning, or editing. Persistence is durable in the checkpoint until the next overwrite.

A continually deployed LLM needs a policy $$\pi_{\text{substrate}}(e)$$ mapping each candidate fact or behavior $$e$$ to one of these substrates. Most candidates should remain in context or external memory. Parametric writes should trigger only when expected benefit exceeds cost and risk.

Define a consolidation score for candidate edit $$e$$:

$$
S(e)
=
\alpha \, R(e)
+
\gamma \, G(e)
-
\eta \, I(e)
-
\mu \, C(e),
$$

where $$R(e)$$ is predicted reliability after edit, $$G(e)$$ is predicted paraphrase generalization, $$I(e)$$ is estimated interference with protected knowledge, and $$C(e)$$ is compute cost including validation. Commit when $$S(e) > \delta$$ for threshold $$\delta$$. For batches $$\mathcal{E}$$, use

$$
S(\mathcal{E})
=
\frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} S(e)
-
\rho \cdot \text{Interference}(\mathcal{E}),
$$

where $$\text{Interference}(\mathcal{E})$$ measures cross-edit collision in shared layers or overlapping key subspaces.

## The proposal: online knowledge consolidation

The proposal is not "fine-tune constantly." It is also not "put everything in the context window." The proposal is a third path: buffer verified facts, batch compatible edits, constrain the update against protected memory, validate locality, and store the resulting delta as an auditable patch.

Start from base $$\theta_0$$. During operation, extract candidate edits from interactions:

$$
e_i = (s_i, r_i, o_i, q_i, \tau_i, \mathcal{V}_i),
$$

with subject $$s_i$$, relation $$r_i$$, object $$o_i$$, confidence $$q_i \in [0,1]$$, timestamp $$\tau_i$$, and verification record $$\mathcal{V}_i$$ (source document, tool output, human label). Buffer candidates in $$\mathcal{B}$$. Do not apply edits immediately.

Verification can include consistency against retrieval, duplicate detection in $$\mathcal{B}$$, and conflict resolution when two candidates assign different objects to the same $$(s, r)$$ pair. On a schedule (hourly, nightly, weekly), form a batch $$\mathcal{E}_t \subset \mathcal{B}$$ and compute layer-local updates $$\Delta_t$$ using MEMIT-style batch solves on selected $$\ell \in \mathcal{R}$$, with AlphaEdit-style null-space projection when applying sequential batches against a growing protected set $$K_0$$. Validate on three prompt sets per edit: direct prompts $$T_{\text{edit}}(e)$$, paraphrases $$T_{\text{para}}(e)$$, and locality probes $$T_{\text{local}}(e)$$. Promote $$\theta_{t+1} = \theta_t + \Delta_t$$ only if batch metrics pass. Otherwise discard or defer.

Version the checkpoint as a sum of auditable deltas:

$$
\theta^{(t)} = \theta_0 + \sum_{j=1}^{t} \Delta_j.
$$

Each $$\Delta_j$$ carries metadata: layer indices, key matrices, test results, provenance. Rollback sets $$\theta^{(t-1)} = \theta^{(t)} - \Delta_t$$ when low-rank factors are stored explicitly.

Low-rank parameterization helps both storage and rollback. Write

$$
\Delta W = A B^\top,
\qquad
A \in \mathbb{R}^{d \times r},
\;
B \in \mathbb{R}^{d \times r},
\;
r \ll d.
$$

Composition of edits becomes additive in the adapter bank:

$$
\theta^{(t)} = \theta_0 + \sum_{j=1}^{t} A_j B_j^\top.
$$

Two-speed parametric updates extend this. Maintain fast adapters $$\Delta_{\text{fast}}$$ applied on every consolidation cycle and slow base updates $$\Delta_{\text{slow}}$$ merged only after repeated successful validation across time windows. Operationally,

$$
\theta_t = \theta_0 + \Delta_{\text{slow}, t} + \Delta_{\text{fast}, t}.
$$

Facts with high churn stay in $$\Delta_{\text{fast}}$$ until stability statistics justify promotion. Facts with low churn can move directly to $$\Delta_{\text{slow}}$$.

Cost structure matters for feasibility. Let $$C_0$$ be fixed overhead per consolidation job (load checkpoint, build test harness, logging) and $$C_e$$ marginal cost per edit (forward passes for keys, solve, validation). One-at-a-time editing costs

$$
C_{\text{serial}} = \sum_{i=1}^{n} (C_0 + C_e) = n C_0 + n C_e.
$$

Batch editing amortizes $$C_0$$:

$$
C_{\text{batch}} = C_0 + n C_e + C_{\text{solve}}(n).
$$

When $$C_0$$ dominates, batching is required for online use. Layer restriction further reduces cost. If only layers $$\mathcal{R} \subset \{1,\ldots,L\}$$ participate, search and validation focus on $$\{W_\ell : \ell \in \mathcal{R}\}$$ rather than the full parameter count $$|\theta|$$.

Temporal facts need explicit validity intervals. Represent edits as

$$
e = (s, r, o, t_{\text{start}}, t_{\text{end}}).
$$

At query time $$t$$, select active edits with $$t_{\text{start}} \leq t \leq t_{\text{end}}$$. Without temporal scoping, sequential overwrite of $$(s, r)$$ pairs loses history and complicates audit. I do not know a clean standard implementation in current editing codebases, but the data model belongs in any serious consolidation buffer.

Composition of edits over long horizons remains the hardest open problem. Given edits $$\Delta_i$$ and $$\Delta_j$$ applied in order, the effective map on representations is generally non-commutative:

$$
\Delta_j \circ \Delta_i \neq \Delta_i \circ \Delta_j.
$$

Interference grows with $$n$$ even when individual edits pass locality tests. Tracking $$\|\Delta K_0\|_F$$ across batches, monitoring entropy of next-token distributions on held-out probes, and maintaining regression slices on $$\mathcal{P}_t$$ are minimum operational requirements. Risk-adaptive validation allocates test budget proportional to estimated harm:

$$
\text{TestBudget}(e) \propto \text{Risk}(e),
$$

where $$\text{Risk}(e)$$ rises with layer depth, subject frequency, overlap with prior edits, and proximity to safety-critical prompts.

Consider a concrete deployment pattern. A company model serves employees with retrieval over wikis and ticket systems. Facts change: team leads, deprecated endpoints, policy wording. Retrieval returns correct answers when embeddings align. Repeated queries on the same updated facts suggest parametric consolidation would reduce latency and retrieval failure modes. A scheduled job collects buffered edits above confidence threshold, clusters by subject and layer target, runs MEMIT-style batch solves with AlphaEdit-style preservation on a staging copy of $$\theta_0$$, evaluates $$T_{\text{edit}}$$, $$T_{\text{para}}$$, and $$T_{\text{local}}$$, then promotes or rejects. Production serves $$\theta^{(t)}$$ while the buffer accumulates $$\mathcal{E}_{t+1}$$. Rollback remains available because $$\Delta_t$$ is stored explicitly.

The editing literature now forms a coherent toolkit for this pipeline. ROME established that factual associations can be localized in MLP weights. MEMIT scales to batch writes. AlphaEdit reduces disruption when edits arrive sequentially. MEND offers fast edit inference at deployment time. Induction heads and ICL explain why fixed $$\theta$$ still adapts strongly through $$c$$. Long-context models expand $$T_{\max}$$ but do not replace the consolidation layer. The remaining gap is procedural: selection, batching, validation, versioning.

The research agenda I would prioritize follows directly from the equations above. Edit selection policies that estimate $$p_{\text{use}}(e)$$ and $$V(e)$$ from query logs. Conflict detection over $$(s, r)$$ with temporal fields. Subspace maintenance for $$K_0$$ as $$\theta^{(t)}$$ drifts under AlphaEdit-style projection. Cheap locality tests that approximate full regression. Algebra for merging compatible edits before application. Reversible low-rank banks with clear promotion rules from $$\Delta_{\text{fast}}$$ to $$\Delta_{\text{slow}}$$.

Transformers already implement powerful conditional computation through $$p_\theta(\cdot \mid c)$$. The next increment for continual deployment is disciplined writing into $$\theta$$ for the subset of knowledge that merits parametric storage. Context handles transient structure. Retrieval handles large corpora. Editing handles compact, stable, verified associations. Fine-tuning handles broad domain shifts. Pretraining sets the prior. Each substrate has different cost, latency, and failure profile. The design task is routing and consolidation, not infinite context alone.

### References

* Brown, T., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

* Olsson, C., Elhage, N., Henighan, T., et al. (2022). *In-context Learning and Induction Heads*. [arXiv:2209.11895](https://arxiv.org/abs/2209.11895)

* von Oswald, J., Niklasson, E., Randazzo, M., et al. (2023). *Transformers learn in-context by gradient descent*. [arXiv:2212.07677](https://arxiv.org/abs/2212.07677)

* Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT* (ROME). [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)

* Meng, K., Sen Sharma, A., Andonian, A. J., Belinkov, Y., & Bau, D. (2023). *Mass-Editing Memory in a Transformer* (MEMIT). [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)

* Mitchell, E., Lin, C., Bosselut, A., Manning, C. D., & Finn, C. (2022). *Fast Model Editing at Scale* (MEND). [arXiv:2110.11309](https://arxiv.org/abs/2110.11309)

* Fang, J., Jiang, H., Wang, K., et al. (2025). *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models*. [arXiv:2410.02355](https://arxiv.org/abs/2410.02355)
