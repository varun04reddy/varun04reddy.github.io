---
title: "Can We Guess Fable 5's Parameter Count?"
date: 2026-07-10
layout: post
categories: [technical]
---

Anthropic does not publish the parameter count of Claude Fable 5. You get a context window and a price. You do not get how many weights are inside.

We can still estimate. If you know the training compute and how many tokens the model saw, you can work backwards from the usual $$6ND$$ accounting. You will not recover the architecture, and you will not tell 150B from 170B. More importantly, for a 2026 frontier model you will recover the *active* count: how many parameters actually run on each token. That is not the number people mean when they say a model is ten trillion parameters.

Almost every frontier LLM now is a mixture of experts. Attention still runs on every token. The feed-forward block is a pile of expert FFNs, and a router sends each token to a handful of them. Training FLOPs track the experts that fire. Stored size tracks all of them. If you mix those two numbers you can invent a training run that never happened, or dismiss a rumor that was only ever about storage.

---

## The 6ND estimate

Let $$N_{\mathrm{active}}$$ be the parameters used by each token and $$D$$ the number of training tokens. For a dense decoder-only Transformer those are basically the non-embedding weights. For MoE they are not. Final pretraining compute is

$$
C \approx 6 N_{\mathrm{active}} D.
$$

A forward pass is about $$2 N_{\mathrm{active}}$$ FLOPs per token. Backprop is about twice that, so six. Attention adds a sequence-length term, and real runs burn FLOPs on communication, the optimizer, and checkpointing. For a dense model the formula still gets you into the right order of magnitude. Here $$C$$ means useful model FLOPs on the final pretraining run, not the whole research program.

If you have both $$C$$ and $$D$$ you invert directly:

$$
N_{\mathrm{active}} \approx \frac{C}{6D}.
$$

Labs almost never publish both. Sometimes a tech report gives $$D$$. Sometimes you can bound $$N_{\mathrm{active}}$$ from decode speed. When $$D$$ is missing, people plug in a tokens-per-parameter ratio $$r$$ and write $$D = r N_{\mathrm{active}}$$, which is how Chinchilla enters. I will get to that. First the thing that actually changed since 2022.

---

## How labs train now

The Chinchilla paper trained a 70B model on 1.4T tokens, about 20 tokens per parameter, and called that compute-optimal for a *dense* model under a fixed FLOP budget. That was a real result. It is not how people train in 2026.

Inference is the bill that repeats. A smaller model that saw more data is cheaper to serve than a larger model that was compute-optimal at pretraining time. So labs overtrain. [Llama 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) is dense and still saw about 15T tokens ($$r \approx 37$$). [Qwen2.5-72B](https://arxiv.org/abs/2412.15115) saw 18T ($$r \approx 250$$). Once MoE showed up, the ratio that matters is tokens per *active* parameter, and those numbers got huge. [DeepSeek-V3](https://arxiv.org/abs/2412.19437) is 37B active on 14.8T tokens ($$r \approx 400$$). [Llama 4 Maverick](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) is 17B active on about 22T tokens.

If you invert $$C$$ with $$r = 20$$ you are answering a 2022 question. You will overestimate active size on any model that was trained like Qwen or DeepSeek.

The other change is sparsity. Dense 70B and 405B models still exist, especially in the open-weight middle class. The frontier stack went MoE because you can grow stored capacity without growing per-token FLOPs. That is the whole product.

---

## Mixture of experts

In a standard Transformer block, attention is followed by one feed-forward network. MoE keeps the attention (always on) and replaces the FFN with $$E$$ expert FFNs plus a small router. Each token activates $$k$$ experts. Mixtral used $$E=8$$, $$k=2$$. DeepSeek-V3 uses 256 routed experts plus one shared expert, and fires 8 of the routed ones. Qwen3-235B-A22B is 128 experts, 8 active. The name is doing the work: 235B stored, 22B active.

<figure style="text-align: center;">
  <img src="/assets/img/blog/gpu-parameter-counter-moe.svg" alt="Dense FFN versus MoE: attention always on, only k of E experts run" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1: Attention is dense either way. 6ND counts the green path. A "10T model" claim, if it is about anything, is counting every expert on disk.</figcaption>
</figure>

Two numbers, two jobs:

| Model | Total (stored) | Active per token | Tokens | $$r = D / N_{\mathrm{active}}$$ |
|---|---|---|---|---|
| Mixtral 8×7B | 47B | 13B | — | — |
| Qwen2.5-72B (dense) | 73B | 73B | 18T | ~250 |
| Llama 3.1 405B (dense) | 405B | 405B | ~15T | ~37 |
| Qwen3-235B-A22B | 235B | 22B | ~36T | ~1600 |
| DeepSeek-V3 | 671B | 37B | 14.8T | ~400 |
| Llama 4 Scout | 109B | 17B | ~40T | ~2400 |
| Llama 4 Maverick | 400B | 17B | ~22T | ~1300 |
| Kimi K3 | 2.8T | 104B | — | — |

[Mixtral](https://arxiv.org/abs/2401.04088), [DeepSeek-V3](https://arxiv.org/abs/2412.19437), [Qwen3](https://arxiv.org/abs/2505.09388), [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Kimi K3](https://arxiv.org/abs/2607.24653). The last column is why I do not want a single $$r$$ for Fable.

Training compute uses the active column:

$$
C \approx 6 N_{\mathrm{active}} D.
$$

DeepSeek-V3 is the clean check because they published both counts and the token budget:

$$
C \approx 6 \times (37 \times 10^{9}) \times (14.8 \times 10^{12}) \approx 3.3 \times 10^{24}\ \mathrm{FLOPs}.
$$

If you put 671B into $$6ND$$ you get about $$6 \times 10^{25}$$, almost twenty times too high. DeepSeek also reported 2.788M H800 GPU hours for the full pipeline. That budget can deliver a few $$10^{24}$$ FLOPs at normal utilization. It cannot deliver $$6 \times 10^{25}$$. So the published *active* count is the one that is consistent with the GPU hours. The 671B is real. It is just not the $$N$$ in $$6ND$$.

What the total count *does* buy you is capacity and a memory bill. You have to store every expert even if a given token only touches eight of them. Decode speed on a bandwidth-limited generate step tracks $$N_{\mathrm{active}}$$ (and the bytes per active parameter). VRAM tracks $$N_{\mathrm{total}}$$. That is why a 10T rumor and a 60 tok/s generate speed can both be true: sparse enough MoE, lots of experts on disk, a DeepSeek-like active set at runtime. It is also why they can both be false. You cannot read one off the other.

I would bet Fable 5 is MoE. Anthropic has not said so. Every comparable closed model in this tier is rumored to be, and every open model that competes with it is. The estimate below is an active-count estimate either way. If Fable is dense, active equals total. If it is MoE, the tweeted number can be several times larger.

---

## Chinchilla, as a scenario

Kaplan et al. fit loss against $$N$$, $$D$$, and compute, and preferred growing the model faster than the dataset. [Hoffmann et al.](https://arxiv.org/abs/2203.15556) retrained that allocation with isoFLOP curves. Their parametric fit is

$$
L(N_{\mathrm{active}}, D) \approx E + \frac{A}{N_{\mathrm{active}}^{\alpha}} + \frac{B}{D^{\beta}}.
$$

Under $$C \approx 6 N_{\mathrm{active}} D$$ the minimum sits near

$$
N_{\mathrm{active,opt}} \propto C^{1/2}, \qquad D_{\mathrm{opt}} \propto C^{1/2},
$$

which is the 20 tokens per parameter rule of thumb from the 70B / 1.4T checkpoint. Their printed Approach 3 constants actually implied something closer to 70. [Epoch's replication](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) showed that was a bad fit. A re-fit recovered ~20. I use 20 as one branch, not as a law.

Write $$D = r N_{\mathrm{active}}$$. Then

$$
C \approx 6 r N_{\mathrm{active}}^{2}, \qquad N_{\mathrm{active}} \approx \sqrt{\frac{C}{6r}}.
$$

$$r = 20$$ is Chinchilla. $$r = 250$$ is Qwen2.5-72B. $$r = 400$$ is DeepSeek-V3. Same $$C$$, very different $$N_{\mathrm{active}}$$. Picking one $$r$$ and announcing a Fable size is how this turns into a fake detector.

A square root helps a little. If compute is off by 4×, inferred $$N_{\mathrm{active}}$$ is off by 2×. It does not save you from the wrong $$r$$, and it does not turn active into total.

---

## A dense check: Qwen2.5-72B

Before Fable, check the algebra on a model that published $$N$$ and $$D$$.

The [Qwen2.5 report](https://arxiv.org/abs/2412.15115) describes up to 18T pretraining tokens and a ~72B dense model. The [72B card](https://huggingface.co/Qwen/Qwen2.5-72B) lists 72.7B total and 70.0B non-embedding. Assume the 72B checkpoint saw the full 18T, and round to 72B:

$$
C_{\mathrm{Qwen}} \approx 6(72 \times 10^{9})(18 \times 10^{12}) \approx 7.78 \times 10^{24}\ \mathrm{FLOPs}.
$$

Direct inversion recovers 72B by construction. That is the point. It is not an independent discovery of Qwen's size. It is a check that $$6ND$$ is the accounting we think it is.

The Chinchilla-only inversion of the same $$C$$ is

$$
N_{\mathrm{Chinchilla,active}} \approx \sqrt{\frac{7.78 \times 10^{24}}{120}} \approx 255\ \mathrm{B}.
$$

Too big, because Qwen was not trained at $$r = 20$$. The failure is the scenario, not the multiply.

---

## Is 6ND how people actually do this?

Yes, for the training-compute half.

[Epoch](https://epoch.ai/publications/estimating-training-compute) estimates undisclosed compute two ways. Operation counting: $$C \approx 6ND$$ with $$N$$ active. Hardware:

$$
C \approx \eta\, G\, T\, F_{\mathrm{peak}},
$$

with utilization around 0.3 for LLMs (I use 25–40%). When the two disagree, an input is wrong. Hoffmann used the same 6. Browser Chinchilla calculators invert it. For MoE you still put active $$N$$ in. Putting every expert in invents a training run.

People have guessed Fable 5. Anthropic has not published a count. The [system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) and [launch note](https://www.anthropic.com/research/claude-fable-5-mythos-5) say Fable 5 and Mythos 5 are the same weights with different safeguards, so a guess for one is a guess for the other. What circulated in July 2026, [summarized by explainx](https://explainx.ai/blog/claude-sonnet-opus-fable-parameter-counts-debate-july-2026), is weaker than $$6ND$$:

- **Musk ratios.** April 2026: Grok is 0.5T total, half of Sonnet and one-tenth of Opus. People wrote Sonnet $$\approx 1$$T, Opus $$\approx 5$$T. [@AstraiaAI](https://x.com/AstraiaAI) added Fable/Mythos at 10T and said it came from Anthropic's compute partner, with no document. That is gossip about *stored* size.
- **Throughput.** [unexcitedneurons](https://unexcitedneurons.substack.com/p/estimating-the-size-of-claude-opus) calibrated Opus 4.5/4.6 decode speed on Vertex against open MoE models. The observable is bytes loaded per token: about 90B to 150B *active*, which becomes 1T to 3T total only after you pick a routing ratio. Fable 5 generates at about 60 to 64 tok/s on [Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5/providers), same band as Opus 4.8. Time-to-first-token is slow because thinking is on. Decode is not. An order of magnitude more *active* parameters than Opus should crawl on the output stream unless the serving stack is doing a lot more parallelism, or the model is much sparser.
- **Cost writeups.** [capitalandcompute](https://capitalandcompute.net/blog/what-it-costs-to-train-ai-models-2026/) puts Fable around 1–2T MoE, 10–20T tokens, $$10^{25}$$ to $$5\times 10^{25}$$ FLOPs, 20k–40k GPUs. A scenario. A 10T *stored* MoE at DeepSeek-like sparsity (~5% active) is a few hundred billion active, which is the number $$6ND$$ sees.

$$6ND$$ will never emit 10T from a plausible final-run FLOP budget unless you feed it 10T *active* parameters. Nobody thinks Anthropic trained a dense 10T model. The rumor, if it is about anything, is stored MoE.

---

## A Fable example

If a run uses $$G$$ GPUs for $$T$$ seconds at peak $$F_{\mathrm{peak}}$$,

$$
C \approx \eta\, G\, T\, F_{\mathrm{peak}}.
$$

The [H100 SXM sheet](https://www.nvidia.com/en-us/data-center/h100/) lists 1,979 TFLOP/s BF16 Tensor Core with sparsity. Dense BF16 is about $$989 \times 10^{12}$$ FLOP/s. $$\eta$$ eats imperfect matmul shapes, attention, all-reduce, the data pipeline, checkpointing. Failed runs and ablations are project compute. Dumping all of them into $$6 N_{\mathrm{active}} D$$ overstates the shipped model.

Two thousand H100s for 120 days is a mid-2024 cluster. I am using it as arithmetic, not as a claim about Anthropic:

| $$r$$ | Active, 2k GPU / 120d / 25–40% |
|---|---|
| 20 | 207B to 261B |
| 100 | 92B to 117B |
| 250 | 58B to 74B |

A 2026 frontier run is tens of thousands of accelerators. Same formulas, different $$G$$. Twenty thousand H100-equivalents, 120 days, 30% utilization:

$$
C \approx 0.30 \times 20000 \times (120 \times 86400) \times 989 \times 10^{12} \approx 6.2 \times 10^{25}\ \mathrm{FLOPs}.
$$

| $$r$$ | Active, 20k GPU example |
|---|---|
| 20 | ~720B |
| 100 | ~320B |
| 400 (DeepSeek-like) | ~160B |

That last row is the one I take seriously if Fable is MoE and overtrained. Double the cluster and the Chinchilla branch crosses a trillion *active*. A 10T *stored* rumor then wants something like 5–10% activation, which is DeepSeek's neighborhood (37 / 671 $$\approx$$ 5.5%). The 2,000-GPU table cannot get you there. The method can, once you stop treating a 2024 cluster as Anthropic's 2026 one.

None of this is Fable's size. Hardware and $$r$$ are unpublished. The same FLOP budget is a smaller active model on more tokens or a larger one closer to Chinchilla.

---

## Hardware sanity checks

Memory bounds *stored* size. If a BF16 deployment's weights occupy ~140 GB, that is ~70B stored (two bytes per parameter). Runtime overhead, quantization, and tensor parallelism make this fuzzy, but it kills some branches. A dense 250B BF16 model needs ~500 GB just for weights. If it demonstrably runs in 160 GB without offload or quant, 250B dense is wrong. For MoE the same measurement bounds $$N_{\mathrm{total}}$$, not $$N_{\mathrm{active}}$$.

Decode speed bounds *active* size. Batch-one generation is often bandwidth-bound:

$$
R_{\mathrm{decode}} \lesssim \frac{B_{\mathrm{memory}}}{b N_{\mathrm{active}}}.
$$

That is the Opus method: tokens per second on Vertex, calibrated against DeepSeek-class models with published active counts. Batching, quant, context, KV cache, and speculative decoding all move the number. Use it to kill impossibilities, not to mint a press-release parameter count.

---

Fable 5 has a 1M context window and 128K output tokens. It thinks for a long time before the first token. None of that is $$N$$.

To narrow the range you need some mix of GPU count and duration, precision, pretraining tokens, weight memory, decode speed at a known context, and a dense-vs-MoE tell. Compute constrains $$N_{\mathrm{active}} D$$. Tokens split that product. Memory constrains stored parameters. Decode constrains active ones.

If those disagree, an assumption is wrong. I would not bet on a single number. I would bet the model is MoE, that $$6ND$$ is talking about a few hundred billion active parameters at a 20k-GPU-class budget, and that any 10T figure you see is a stored-expert count someone rounded for Twitter.

---

* [Kaplan et al., *Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). The earlier compute-allocation study.
* [Brown et al., *Language Models are Few-Shot Learners*, Appendix D](https://arxiv.org/abs/2005.14165). The $$6ND$$ training-compute accounting.
* [Hoffmann et al., *Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556). Chinchilla, and the 70B / 1.4T checkpoint.
* [Besiroglu et al., *Chinchilla scaling: A replication attempt*](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt). Printed Approach 3 constants implied $$\sim 70$$ tokens/param; a re-fit recovers $$\sim 20$$.
* [Sevilla et al., *Estimating training compute of deep learning models*](https://epoch.ai/publications/estimating-training-compute). $$6ND$$ vs hardware $$\times$$ time $$\times$$ peak $$\times$$ utilization ($$\sim 0.3$$ for LLMs).
* [Jiang et al., *Mixtral of Experts*](https://arxiv.org/abs/2401.04088). 47B stored, 13B active, top-2 of 8.
* [DeepSeek-AI, *DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437). 671B / 37B, 14.8T tokens, 2.788M H800 GPU hours.
* [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) and the [72B model card](https://huggingface.co/Qwen/Qwen2.5-72B). Dense overtraining check.
* [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). 235B total / 22B active, 36T tokens.
* [Meta, *The Llama 4 herd*](https://ai.meta.com/blog/llama-4-multimodal-intelligence/). Scout 109B/17B on ~40T tokens; Maverick 400B/17B on ~22T.
* [Moonshot, *Kimi K3*](https://arxiv.org/abs/2607.24653). 2.8T total, 104B active, 16 of 896 routed experts.
* [Anthropic, Fable 5 / Mythos 5 system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) and [launch note](https://www.anthropic.com/research/claude-fable-5-mythos-5). Same weights. No published $$N$$.
* [Thakker, *Claude Sonnet 1T, Opus 5T, Fable 10T?*](https://explainx.ai/blog/claude-sonnet-opus-fable-parameter-counts-debate-july-2026). July 2026 size gossip, labeled by evidence tier.
* [unexcitedneurons, *Estimating the size of Claude Opus*](https://unexcitedneurons.substack.com/p/estimating-the-size-of-claude-opus). Decode-throughput estimate of Opus 4.5/4.6 active size.
* [capitalandcompute, *What it costs to train AI models, 2026*](https://capitalandcompute.net/blog/what-it-costs-to-train-ai-models-2026/). Unofficial Fable cost/FLOP scenario.
* [Artificial Analysis, Claude Fable 5](https://artificialanalysis.ai/models/claude-fable-5/providers). ~60–64 output tok/s once generation starts.
* [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/). Peak used in the hardware example.
