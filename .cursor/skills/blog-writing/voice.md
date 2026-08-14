# Voice

The post should sound like a person who has spent time with the objects, not like a model producing a "great blog post."

User constraints, verbatim:
- no em dashes
- no unnecessary metaphors and similes
- not AI generated sounding

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

## Metaphors

Default: none.

Allowed only if the analogy is the actual model (jamming as a mapped constraint-counting argument, knowledge-as-graph if you then use nodes and edges in the reasoning). Then:
1. Introduce it in one sentence
2. Map every term to a real quantity
3. Drop it as soon as the mapping is done
4. Never say "this is not just a metaphor" (that sentence is a tell)

Banned even when tempting: journeys, landscapes (unless loss landscape with a Hessian), DNA of X, standing on the shoulders, iceberg, tip of the spear, double-edged sword, North Star, rabbit hole, mosaic, dance, symphony.

Karpathy almost never analogizes. He says the network spasms, the abstraction leaks, the training fails silently. Those are descriptions of the thing. Kumar uses a worked analogy maybe once per essay and then commits (technical maturity as a knowledge graph). Weng uses almost none; she taxonomizes.

## Cadence

AI prose has a metronome: medium sentence, medium sentence, summarizing sentence, new heading.

Break it.
- Short sentence when the claim is sharp.
- Longer sentence when you are walking through a derivation.
- Occasional parenthetical, the way you would talk: "(hah never thought I'd say that)", "(okay I'm 50% joking)", "because meh"
- Fragments are fine if they are doing work. "So I am reproducing the numbers roughly, but not exactly."
- Do not start three consecutive paragraphs with "This" / "These" / "The"

First person is required when it is a judgment, a confusion, or work you did. "My claim is..." / "I don't fully understand..." / "I stopped pushing on this because the gains were modest." Fake we-the-royal is worse than I.

Second person is fine in recipes ("Don't be a hero"). Do not lecture.

## Texture that reads as human

Steal these moves, not the wording:

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

## House examples

From `_posts/2026-07-10-gpu-parameter-counter.md` (keep writing like this):

> The calculation is intentionally boring. It is not an independent recovery of Qwen's parameter count, because we constructed the compute estimate using that known count. It checks the algebra and shows how strongly a compute-only estimate depends on the token-to-parameter ratio.

From `_posts/2026-04-20-grokking-double-descent.md` (do not write like this):

> Classical statistical learning theory has a tidy story about generalization.
> ...
> > The tools that statistical physics brings to this problem... are not just metaphors.

The first is a person doing arithmetic in public. The second is a generated essay performing Insight.
