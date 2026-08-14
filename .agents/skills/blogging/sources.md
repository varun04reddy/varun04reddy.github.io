# Source analysis

Read this once per session before writing. Steal structure and habits, not sentences.

## Lilian Weng (Lil'Log)

Site: https://lilianweng.github.io/
Self-description: learning notes since 2017. FAQ: figures drawn in Google Presentation. Posts are updated in place with a one-line changelog.

### What makes the posts good

They are **maps**. A Weng post takes a messy literature and gives you a decomposition you can remember (agent = planning + memory + tools; thinking = parallel sampling vs sequential revision; hallucination = in-context vs extrinsic). After that, every paper is an instance of a cell in the map, not a disconnected summary.

She is complete inside the slice she chose. "Prompt engineering" is restricted to autoregressive LMs on purpose. "Hallucination" is narrowed to fabricated output ungrounded in context or world knowledge. Completeness comes from scoping, not from listing every paper.

Tone is calm, slightly informal, first-person when she is judging ("I would consider in-context learning as short-term memory"). Openings are definitional, not cinematic. She thanks people at the top. She cites constantly, in the line, with years.

Figures do the taxonomy. The agent overview diagram is the post. Method figures are usually from papers, always attributed. Equations appear when a method has an actual objective (MIPS, CoH data tuples), not as garnish.

Endings are light: a citation block, sometimes "I know this is a long read." No moral.

### Patterns to copy

- Scope sentence in paragraph one
- Named components, then one section each
- For each paper: one sentence of what it is, then the mechanism, then when it fails or what it beat
- Overview figure first
- "Image source:" on borrowed figures
- Update notes if the topic will move

### Patterns not to copy blindly

- Pure survey with no opinion. Varun's better posts have a claim. Use Weng's map, then take a position.
- Nested bullet taxonomies as a substitute for prose. Fine in a survey, death in a mechanism post.
- Length for its own sake.

### Openings (study these)

Reward hacking post: starts with a definition, then why the problem got worse with RLHF, then two concrete failure cases (editing unit tests, mimicking user bias). No hook.

Why We Think: thanks Schulman, then two citations, then "this post aims to review... and why it helps." Motivation comes *after*, split into psychology / compute-as-resource / latent variables. Each motivation is a real frame, not a vibe.

Agents: "Building agents with LLM as its core controller is a cool concept." Then demos as existence proofs, then the map.

## Andrej Karpathy

Site: https://karpathy.github.io/

### What makes the posts good

They are **sessions**. You watch him get restless, pick a toy problem, write the code, report the numbers, then zoom out one notch.

Pong from Pixels: magic vs simplicity as the reason to write; a running example (Pong, 130 lines of numpy); supervised learning as the analog; then PG; then the GIF; then "what isn't happening." He keeps telling you the thing is dumb and that this is the point.

Recipe for Training Neural Nets: two observations (leaky abstraction; fails silently), then a numbered recipe you can follow. Advice is concrete (`-log(1/n_classes)` at init, overfit one batch, Adam 3e-4, don't be a hero). Humor is dry and local, never ornamental.

33 years ago: reproduce LeCun 1989, fail to match exactly, name every reason (lost dataset, possible missing sqrt in init, "scheme that will not be discussed here"), then time-travel cheats one at a time (softmax, AdamW, shift aug, dropout+ReLU), print eval lines after each change. The reflection about 2055 is earned because you sat through the 90-second MacBook run.

RNN effectiveness: the samples *are* the argument. Then he peeks at neurons.

Voice: "I", asides, mild slang, parenthetical jokes, sudden precision. He will say "meh" and then write the score-function estimator in full.

### Patterns to copy

- Start from a concrete irritation or a missing number
- Show code and eval output, not descriptions of code
- Change one thing at a time and report the metric
- Visualize the learned object (weights, samples, rollouts)
- Zoom out only after the experiment
- Admit non-reproduction, failed ViT miniaturization, leftover errors

### Patterns not to copy blindly

- 2015-era "RL is hot!" energy in a 2026 post. The move is restlessness, not boosterism.
- Javascript widgets, unless you are going to maintain them.

### Openings (study these)

Recipe: a tweet did better than expected, so expand it, but not as a list of mistakes. Two observations, then a process.

33 years: this 1989 paper type-checks as a modern DL paper. Tiny data, tiny net. Reproduce it, then ask what 33 years bought.

Pong: list of demos, then a four-factor decomposition of progress (compute, data, algorithms, infrastructure), then "whenever there is a disconnect... I get all antsy."

## Tanishq Kumar

Site: https://tanishqkumar.github.io/essays.html
Also: [beyond-nanogpt LESSONS.md](https://github.com/tanishqkumar/beyond-nanogpt/blob/main/LESSONS.md)

Published essays mix literary pieces with technical field notes. The technical ones that matter for this skill:

- [A laundry list for AI research](https://tanishqkumar.github.io/essays/cheap.html)
- [Assorted lessons from the trenches](https://tanishqkumar.github.io/essays/lessons.html)
- [My anti-SOP](https://tanishqkumar.github.io/essays/changemind.html)
- [College and Technical Maturity](https://tanishqkumar.github.io/essays/ctm.html)

### What makes the posts good

They are **taste, made checkable**. He writes like someone who has a stack of unfinished experiments and is willing to give them away.

Laundry list: "Please steal my AI research ideas." Each item is a question, a pointer to a figure or paper, a first experiment small enough to run, and a prediction. Not a topic list. "Train a small model B_T and use logit-matching... I anticipate it'll do better than a compute-matched pure NTP run."

Lessons from the trenches: credentials in one sentence (10k lines of PyTorch, `nn.Linear` only), then bullets that bottom out in a command, a shape, or a systems pattern. Objective functions > architectures. `F.unfold + bmm`. IMPALA is a dataloader. Loss can go up.

Anti-SOP: "musings I can't put into a Statement of Purpose." Nested bullets of changed beliefs. Architecture vs data, HCI, NCCL, "spicy matmuls on tensor cores." He will call his own theorems clean and then say they may not speak to foundation models.

Technical maturity: one idea (maturity = denser keyword graph inside a field, not "critical thinking"), argued with a freshman linear algebra / quantum gates story. The graph analogy is used, not sprinkled.

Voice: long sentences that still have a point, then a short one. Comfortable with "this NCCL bullshit", "bashy", "I am a naive tourist." High-culture references appear in the nontechnical essays; the technical notes mostly skip them. Do not import Orwell into a Hessian post.

### Patterns to copy

- Changed mind as structure
- Research questions with an experiment attached
- Nested bullets when the form is notes, not a treatise
- Name the file (`train_impala.py`)
- Strong claims plus a limit in the same breath
- "I stopped working on it since it was only a small win" as a gift to the reader

### Patterns not to copy blindly

- Marriage / befriending-the-machine analogies in technical posts. User rule: no unnecessary metaphors. Kumar earns them in essays; this site should almost never.
- Scatterbrained anti-SOP as the only mode. Use it for commentary posts, not for mechanism posts.

## Synthesis: what "awesome" is here

The three blogs are good for different reasons. Do not average them.

| Axis | Weng | Karpathy | Kumar |
|---|---|---|---|
| Unit of post | a literature | a session | a mind changing |
| Trust comes from | coverage + diagrams | numbers + code | specificity + revised beliefs |
| Figure | overview map | the result | optional pointer |
| Sentence | clean, cited | spoken, parenthetical | argued, sometimes long |
| Ending | citations | one extrapolation | last bullet |

Shared (this is the actual style guide):
1. Start on the object, not on the vibe
2. Make a map, a measurement, or a bet
3. Put the mechanism next to the claim
4. Show a picture of the thing, or a number
5. Leave a residue: a taxonomy, a recipe, a stealable experiment
6. Do not perform wonder

## Calibration against this site

Varun's GPU-parameter post is already in the Karpathy/worked-estimate family. Knowledge-editing is a Kumar-ish commentary on his own paper, but the figures are mostly paper dumps and the prose is looser than Weng. Grokking/probes have Weng-like length and Karpathy-unlike scaffolding: they sound generated because they follow a template (tidy story, three regimes, interplay, critique, taking stock, soulful quote).

When writing for this site, aim for:
- GPU post's honesty and arithmetic
- Weng's overview figure and scoped completeness, when the post is a survey
- Kumar's stealable questions and first-person judgment
- Karpathy's one-change-at-a-time evidence, when there is an experiment

That combination is enough. Do not add more style.
