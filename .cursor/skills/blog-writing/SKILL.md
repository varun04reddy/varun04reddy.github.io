---
name: blog-writing
description: Write technically complete, non-AI-sounding research blog posts for The Latent Space in the style of Lilian Weng, Andrej Karpathy, and Tanishq Kumar. Use when drafting, editing, outlining, or planning posts in _posts/, when the user asks for a new blog, or when improving figures and commentary.
---

# Blog writing

Write posts for this site (`The Latent Space`) that a researcher would actually want to finish. The voice is a human thinking in public: technically complete, specific, and unpolished in the ways that matter. It must not read as AI-generated.

The facts are available elsewhere. The edge is commentary, and the way we explain, until the next step feels like something the reader could have thought of themselves, without losing technical clarity or depth. Personality, still serious. A tonal pop, not a TED talk and not a slide deck of "X is Y." sentences.

Canonical influences (read [sources.md](sources.md) before a first post in a session):
- Lilian Weng, [Lil'Log](https://lilianweng.github.io/): taxonomy, overview diagrams, paper-faithful surveys
- Andrej Karpathy, [karpathy.github.io](https://karpathy.github.io/): experiment narratives, numbers, code, result-as-figure
- Tanishq Kumar, [essays](https://tanishqkumar.github.io/essays.html): commentary, field notes, concrete research questions

Do not blend all three into one mush. Pick a **post type** and commit.

## Before writing

Copy this checklist and track it:

```
- [ ] Read voice.md
- [ ] Read figures.md
- [ ] Skim 1 existing post of the same type in _posts/
- [ ] Choose post type (below)
- [ ] Collect real papers, numbers, and (if Karpathy-type) a runnable experiment
- [ ] Plan 2-5 figures that carry the argument
- [ ] Write
- [ ] Voice pass (em dashes, metaphors, AI tells, staccato, missing commentary, missing mechanism)
- [ ] Save to _posts/YYYY-MM-DD-slug.md and figures to assets/img/blog/
```

If the user gave an idea, notes, or papers, those are the source of truth. Do not invent citations, results, or "we ran X" claims.

## Post types

Choose one. Form follows the idea, not a universal outline.

| Type | When | Model | Length |
|---|---|---|---|
| **Survey / map** | A literature needs a mental model | Weng | long; complete coverage of the chosen slice |
| **Experiment narrative** | You did (or will do) a thing and the result is the point | Karpathy | medium-long; numbers throughout |
| **Mechanism** | One non-obvious claim that needs math | best of Weng precision + Karpathy concreteness | as long as the argument; no padding |
| **Commentary / field notes** | Taste, research questions, hard-won observations | Kumar | short-medium; bullets allowed |
| **Worked estimate** | Back-of-envelope on a real system | Karpathy 33-years / Varun GPU post | medium; algebra + sanity checks |

The existing grokking/probes posts used a rigid 7-part template (setup, analysis, interplay, experiments, critique, taking stock, blockquote). Do **not** reuse that skeleton. It is the main reason those posts sound generated.

## House format

Save to `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Specific, not topical"
date: YYYY-MM-DD
layout: post
categories: [technical]
---
```

Use `categories: [thoughts]` only for non-technical essays.

Title encodes the claim or the move, not the topic.
- Yes: `Can We Guess Fable 5's Parameter Count?`
- Yes: `What Linear Probes Actually Measure`
- No: `Understanding Grokking`
- No: `A Deep Dive into Scaling Laws`

Conventions:
- `---` between major sections. Do not put a `---` after every heading.
- `##` for real pivots. Almost never `###`. If you want a subhead, write a transition sentence.
- Display math: `$$ ... $$` on its own lines. Inline math on this site is also `$$...$$`. MathJax 3 here does not render `$...$`.
- Link papers by name and year, with a URL.
- Figures: HTML `<figure>` tags. See [figures.md](figures.md).
- Optional `description:` in front matter if the listing page needs a one-liner.

Match the site. Do not redesign the blog index, layout, or theme unless asked.

## How to open

Start on the problem, already assuming the reader is informed. First paragraph is mid-conversation, not a trailer.

**Do this** (Karpathy / Varun GPU post):

> Anthropic does not publish the parameter count of Claude Fable 5. The public description tells us the context window and the pricing, but not how many weights are inside it. That is normal for a proprietary model.

**Do this** (Weng):

> Reward hacking occurs when an RL agent exploits flaws or ambiguities in the reward function to achieve high rewards without completing the intended task.

**Do not do this** (generated, and present in an existing post):

> Classical statistical learning theory has a tidy story about generalization. ... Modern deep learning breaks it cleanly.

Forbidden openings:
- "There is a moment when..."
- "X has a tidy story. Modern Y breaks it."
- "Ah, the classic X..."
- Any claim that the topic will unsettle, shock, or challenge the reader
- A definition copied from a textbook, then "In this post, we will..."

By the end of the opening (2-4 short paragraphs), the reader should know the question, the stake, and the move you are going to make. State the thesis in plain language. Do not announce that it is non-obvious.

## How to argue

After every important claim, do one of:
1. **Mechanism**: why it is true, in the actual objects (weights, Hessian, tokens, FLOPs)
2. **Evidence**: a number, a figure, a citation, a tiny derivation
3. **Limit**: when it fails

Do not name a phenomenon and move on. Do not use physics vocabulary as decoration. If you invoke jamming, free energy, or phase transitions, you need a quantity you could measure.

Equations are not optional in a mechanism post and not required in field notes. When an equation appears, the surrounding prose must say what it is, why it has that form, and what it implies. Then keep going. Do not drop a display equation and change the subject.

Prefer a real calculation, a toy experiment, or a worked example over a metaphor.

If an experiment should be run later, mark it like this (specific enough to implement):

```
[Experiment to run:
Setup: architecture, data, what you vary
Record: metric and plot
Prediction: a number or scaling, not "a trend"
Why it would change the argument: ...]
```

## How to end

No "Conclusion", "Taking Stock", or "In summary" heading. No restatement of the whole post. No closing blockquote of a nuanced personal observation.

End with one of:
- a remaining confusion you actually have
- a next measurement
- a scope limit ("this estimates active parameters, not architecture")
- a sharp implication for practice

Then stop.

## Voice (mandatory)

Full rules in [voice.md](voice.md). Non-negotiable:

- No em dashes. Use a period, comma, colon, or parentheses.
- No unnecessary metaphors or similes. Analogies only if you need them to carry a derivation, and then you explain them.
- No AI cadence, in either direction: no generated Insight essay, and no stack of punchy "X is Y." sentences.
- Personality, still serious. Join thoughts with commas. Put a first-person judgment or an aside in the section. Make the next equation feel like something the reader was about to write.

Write like Varun's GPU-parameter post and knowledge-editing commentary: a person doing the arithmetic, then telling you what to notice. Do not imitate the old grokking closing blockquote, and do not imitate a slide deck.

## Figures

Full rules in [figures.md](figures.md). Minimum bar:

- Plan figures before drafting. A survey needs an overview diagram. An experiment narrative needs result plots. A mechanism post needs one picture of the objects (constraints, spectrum, information flow).
- Every figure earns a caption that tells the reader what to notice.
- Prefer original plots and simple schematics over dumped paper screenshots. If you reuse a paper figure, cite it: `(Image source: Author et al. YEAR)`.
- Save to `assets/img/blog/<slug>-<n>.png` (or `.svg`).

## Research bar

- Read the papers you cite. Quote or paraphrase the actual claim.
- Do not hallucinate Table numbers, hyperparameters, or "it is well known that".
- If you are unsure, say so in first person and say what would settle it.
- Technical completeness means: a competent reader can reconstruct the argument. It does not mean covering every adjacent paper.

## Existing posts as calibration

Read at least one before writing.

**Match these:**
- `_posts/2026-07-10-gpu-parameter-counter.md`: specific, algebraic, honest about what the estimate is not. Best house example for worked estimates.
- `_posts/2025-05-25-knowledge-editing.md`: first-person commentary on work you did. Keep the casualness, raise the precision and figures.

**Do not imitate:**
- The old 7-part grokking/probes skeleton (tidy-story opening, obligatory interplay, "Taking Stock", closing blockquote).
- A staccato rewrite that deleted the commentary along with the padding. If a paragraph is three disconnected facts, you over-cut.

## Additional resources

- Voice, bans, and the voice pass: [voice.md](voice.md)
- Figures, captions, matplotlib house style: [figures.md](figures.md)
- What Weng / Karpathy / Kumar actually do: [sources.md](sources.md)
