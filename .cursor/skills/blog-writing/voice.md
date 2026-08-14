# Voice

The post should sound like a person who has spent time with the objects, not like a model producing a "great blog post."

User constraints, verbatim:
- no em dashes
- no unnecessary metaphors and similes
- not AI generated sounding

The edge of this blog is not that the facts are secret. The papers are public. The edge is commentary: how we walk someone through the objects until the next step feels like something they could have thought of, without losing technical clarity or depth.

## Personality, still serious

Serious does not mean flat. Write with a tonal pop that still belongs in a research note: asides, commas, a judgment in first person, the occasional "okay, here's the annoying part." It should read like a human taking the reader through an argument, not like a stack of topic sentences.

Do not write a sequence of short, period-terminated claims:

> Double descent is the capacity cut. Increase $$N$$. Test error spikes. The spike sits at interpolation.

That is technically true and dead on the page. Join the thought. Use commas, colons, and a trailing clause that tells the reader *why they should care* or *what they should notice*:

> Double descent is the capacity cut: you increase $$N$$ past the point where training error hits zero, test error spikes and then falls, and that spike, if you actually look at where it lives, sits at interpolation.

A short sentence is still allowed when the claim needs to land. The failure mode is *only* short sentences.

Make the explanation feel inevitable. Before the equation, put the reader in the situation where they would have written it. After the equation, say what it implies in objects (weights, FLOPs, layers), not in vibes. If they close the tab, they should feel they could reconstruct the argument, not that they memorized a slogan.

Still banned: performing Insight, fake warmth, "let's go on this journey" as a sentence, and any metaphor you do not need for the derivation.

## Hard bans

Never use:
- Em dashes (`—` or `--` as a stand-in). Split the sentence. Use a comma, colon, period, or parentheses.
- Empty intensifiers: remarkably, strikingly, surprisingly, notably, interestingly, crucially, importantly, arguably
- Throat-clearing: it is worth noting, as we will see, this raises the question, in this post we will, let's dive in, let's unpack
- Formulaic closes: in conclusion, to summarize, taken together, at the end of the day
- Contrast templates: "It's not X. It's Y." / "X is not just A. It's B."
- Hype: groundbreaking, unlock, robust (as filler), landscape (as filler), delve, tapestry, realm
- Dramatic framing: unsettle, shatter, challenge everything we thought
- Cutesy headers used as transitions (`### The Plot Twist`)
- Bold or italics for emphasis in every other sentence. If a sentence needs emphasis, rewrite it.
- Stacked hedging: "It could perhaps be argued that it might"
- Three-item adjective piles: "clear, principled, and actionable"
- A closing blockquote that restates the thesis in a more soulful register
- Staccato stacks: three or more consecutive sentences that are subject-verb-object and then stop, with no clause that comments, qualifies, or points

## Metaphors

Default: none.

Allowed only if the analogy is the actual model (jamming as a mapped constraint-counting argument, knowledge-as-graph if you then use nodes and edges in the reasoning). Then:
1. Introduce it in one sentence
2. Map every term to a real quantity
3. Drop it as soon as the mapping is done
4. Never say "this is not just a metaphor" (that sentence is a tell)

Banned even when tempting: calling the post a journey, landscapes (unless loss landscape with a Hessian), DNA of X, standing on the shoulders, iceberg, tip of the spear, double-edged sword, North Star, rabbit hole, mosaic, dance, symphony.

Walk the reader through the argument. Do not announce that you are walking them.

Karpathy almost never analogizes. He says the network spasms, the abstraction leaks, the training fails silently. Those are descriptions of the thing. Kumar uses a worked analogy maybe once per essay and then commits (technical maturity as a knowledge graph). Weng uses almost none; she taxonomizes.

## Cadence

Two failure modes, both AI:

1. Medium sentence, medium sentence, summarizing sentence, new heading.
2. Punchy fragment. Punchy fragment. Punchy fragment. Caption.

The house rhythm is mixed. A longer sentence carries the derivation, with commas doing the work of "and then you notice." A short sentence lands the claim. A parenthetical is how a person actually talks: "(this is the part I don't fully trust)", "(okay, hinge, not cross-entropy)", "because the easy solution is just memorizing the training set."

- Short sentence when the claim is sharp.
- Longer sentence, with commas, when you are walking through a derivation or a judgment.
- Occasional parenthetical.
- Fragments are fine if they are doing work. "So I am reproducing the numbers roughly, but not exactly."
- Do not start three consecutive paragraphs with "This" / "These" / "The"
- Do not let a paragraph be three disconnected facts. Connect them, or cut two of them.

First person is required when it is a judgment, a confusion, or work you did. "My claim is..." / "I don't fully understand..." / "I stopped pushing on this because the gains were modest." Fake we-the-royal is worse than I.

Second person is how you make it intuitive: "you already know what happens if you stop the sweep early," "you can see why a 1D probe would look fine on a helix." Do not lecture.

## Texture that reads as human

Steal these moves, not the wording:

**Commentary is the product.** The papers already exist. Your job is the extra sentence that says what to notice, what you don't buy, and what you'd check next.

**Make them think of it.** Set up the situation, then the formula. If the formula arrives unmotivated, you failed. If it arrives and they think "yeah, of course it's $$1/\lambda$$," you succeeded.

**Specificity.** "130 lines of Python", "3 nights on a Macbook", "layers 3-8 in GPT-J", "Table 8 in this paper is amazing", "I've written over ten thousand lines of pytorch by hand". Invented specificity is worse than none. If you did not run it, do not fake a lab notebook.

**Correction of your past self.** Kumar: "I used to think architecture was the first-order concern, now I think it's the data." Weng: "I would like to narrow down the problem of hallucination to..." Karpathy: "Q-Learning is not a great algorithm (you could say that DQN is so 2013 (okay I'm 50% joking))."

**Honesty about ugliness.** "Sadly, an exact reproduction is most likely not possible because the original dataset has, I believe, been lost to time." "The calculation is intentionally boring." "I am a naive tourist in the wonderland of optimization."

**Objects, not vibes.** Name the tensor shape, the loss, the file, the layer. "Put big objects on `torch.share_memory` instead of passing them through `mp.Queue`."

**Asymmetric structure.** A survey can have a 4-line subsection next to a 40-line one. Field notes can be bullets. A recipe can be numbered. Do not make every section the same length.

## Openings and closings, by type

Survey: define the object, maybe thank a reader, then give the map.
Experiment: why you are restless ("whenever there is a disconnect between how magical something seems and how simple it is under the hood"), then the toy problem.
Commentary: where you have been, then the list. "I've been underwater, focused myopically on my research."
Worked estimate: the missing number, the method, the caveat in paragraph one.

Closings: Karpathy zooms out one notch ("in 2055, you will ask a 10,000,000X-sized neural net") or stops at "Good luck." Kumar stops when the last bullet is done. Weng cites. None of them write a moral.

## Voice pass (run after the draft)

Search the draft for:
1. `—` or `--` used as em dashes. Eliminate.
2. `not just`, `it's not X`, `in conclusion`, `taken together`, `remarkable`, `striking`, `delve`, `landscape`, `robust`, `unlock`
3. Any simile (`like a`, `as if`, `akin to`) that is not carrying a derivation
4. First paragraph: is it a trailer? Rewrite until it is a situation.
5. Last section: is it a recap? Cut it down to one new thought.
6. Every heading: would the section still work if you deleted the heading and wrote a transition sentence? If yes, delete the heading.
7. Every claim: mechanism, evidence, or limit in the next 1-3 sentences?
8. Would you say this sentence out loud to a labmate? If it sounds like a grant abstract, rewrite.
9. Staccato: if three sentences in a row are "X is Y." with no comma clause, join two of them or add the commentary sentence.
10. Personality check: is there a first-person judgment, an aside, or a "here's what I'd actually bet on" somewhere in the section? If the whole section could be a paper abstract, it is too flat.

## House examples

From `_posts/2026-07-10-gpu-parameter-counter.md` (keep writing like this):

> The calculation is intentionally boring. It is not an independent recovery of Qwen's parameter count, because we constructed the compute estimate using that known count. It checks the algebra and shows how strongly a compute-only estimate depends on the token-to-parameter ratio.

That is a person doing arithmetic in public, then telling you what the arithmetic is *for*.

Do not write like this (generated Insight):

> Classical statistical learning theory has a tidy story about generalization.
> ...
> > The tools that statistical physics brings to this problem... are not just metaphors.

Do not write like this either (over-snappy, no commentary):

> Grokking is the time cut. Training loss hits zero quickly. Test accuracy stays near chance, then jumps.

The first is a person. The second is a generated essay performing Insight. The third is a slide deck.
