# Figures

Good figures in these blogs are not decoration. They are how the argument becomes inspectable. Weng's agent post is remembered for the overview diagram. Karpathy's RNN post is remembered for the generated samples. Pong is remembered for the GIF and the weight grid.

Plan figures before prose. If you cannot name the figures, you do not have a post yet.

## What each source actually does

### Weng (Lil'Log)

She draws in Google Slides / PowerPoint ([FAQ](https://lilianweng.github.io/faq/)). Style:
- One **overview diagram** near the top. Boxes, arrows, 4-7 labeled components. The rest of the post unpacks that picture.
- **Paper figures** with a caption that names the move, then `(Image source: Author et al. YEAR)`.
- **Comparison tables**: method vs. what it changes vs. when it fails.
- **Schematic of an algorithm**, not an artful illustration. Thought / Action / Observation loops, memory taxonomy, beam vs. sequential revision.
- Minimal chrome. White background, simple boxes, readable labels, no drop shadows, no stock icons.

The overview diagram is the post's thesis in spatial form. If a survey has no such figure, it is unfinished.

### Karpathy

The figure is usually a **result**.
- Generated text samples, selfie grids, Pong GIFs, first-layer weights, training logs pasted as preformatted text, error cases laid out in a grid.
- Simple matplotlib. Thick enough lines, labeled axes, a title that states the finding (`test error 4.09% -> 1.59%` belongs in the caption or the surrounding prose).
- Cartoons only when they explain an algorithm (policy gradient "encourage the winning episode"). Hand-simple, not infographic.
- Code blocks count as figures. 10 lines of numpy that *are* the policy network.

He shows the thing, then tells you what to notice. He does not show a generic "neural net diagram."

### Kumar

Sparse. When a figure appears it is a pointer: "Figure 5[d] in this paper shows..." He would rather send you to Table 8 than redraw it badly. For this site, still make original figures when the post is a survey or an experiment. For field notes, a figure is optional.

## What to make, by post type

| Type | Required figures |
|---|---|
| Survey / map | 1 overview schematic, plus 2-4 paper or redrawn method figures, plus 1 comparison table |
| Experiment narrative | Training/eval curves, qualitative outputs (samples, rollouts, failure cases), maybe weights/activations |
| Mechanism | 1 picture of the objects (constraint geometry, information flow, spectrum). Optional: a cartoon of the two regimes |
| Worked estimate | 1 plot or table of scenarios (vary D/N, vary compute). Show sensitivity, not a single hero number |
| Commentary | Usually none. If one, it should be a diagram of a research question, not clip art |

## Caption rule

Captions do not describe the obvious. They tell the reader where to look.

Bad: `Figure 1: An overview of the system.`
Good: `Figure 1: The agent is an LLM plus three bolted-on faculties: planning, memory, tools. Everything below is an instance of one of these.`
Good: `Figure 2: White is positive, black is negative. Several hidden units are tuned to traces of the bouncing ball.`
Good: `Figure 3: AlphaEdit's extra line is a projection onto the null space of preserved keys. (Fang, 2025)`

Always number figures. Always cite borrowed ones.

## Original vs borrowed

- Prefer original plots of numbers you (or the user) actually have.
- Prefer a simple redraw of an architecture over a screenshot of a paper PDF, unless the paper figure is the evidence.
- Never crop a paper figure without attribution.
- Never generate a fake experimental plot. If the experiment was not run, use the `[Experiment to run: ...]` placeholder, or a schematic of the predicted curve clearly labeled **prediction**.

## House embedding

Save files to `assets/img/blog/`. Name them `<slug>-<short-name>.png` or `.svg`.

```html
<figure style="text-align: center;">
  <img src="/assets/img/blog/slug-overview.png" alt="One-sentence description" width="700"/>
  <figcaption style="font-size: 0.95em; color: #555;">Figure 1: What to notice. (Image source: Author et al. YEAR)</figcaption>
</figure>
```

Width 700 for full-width, 400-500 for smaller plots. Multiple related plots can sit in one `<figure>`.

Site math already handles equations. Do not rasterize an equation unless it is part of a diagram.

## How to produce them

### Plots (matplotlib)

Use a quiet academic style. No seaborn rainbow, no 3D, no gradient fills.

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 160,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
})
```

Rules:
- Label axes with symbols and units (`test error`, `tokens (B)`, `epoch`)
- One claim per plot. If you need two claims, two plots
- Direct labels beat a huge legend when there are 2-3 series
- Annotate the interesting point (interpolation threshold, the grokking jump, the Qwen check)
- Export PNG at 2x. Transparent background is optional; white is safer on this theme
- Title on the plot is optional if the caption is doing the work

### Schematics

Weng's method, adapted:
1. List the 4-7 components the reader must hold in their head
2. Draw boxes and arrows only for causal or dataflow relations
3. Same visual weight for things at the same level of abstraction
4. No icons, no emojis, no 3D isometric "AI brain"

Implementation options, in order of preference:
1. **SVG** (hand-written or via a tiny Python script): best for git, crisp at any zoom
2. **Google Slides / Keynote / PowerPoint** export to PNG, if a human is in the loop (this is what Weng does)
3. **Matplotlib patches** for geometry (spectra, constraint counting, 2D cartoons of a loss basin)

Do not use mermaid in the published post unless the theme already renders it. This site is Jekyll + HTML. Ship PNG or SVG.

### Tables

Weng-style tables beat a paragraph of name-dropping.

| Method | What it changes | Failure mode |
|---|---|---|
| Best-of-N | picks among i.i.d. samples | bounded by whether the model can hit the answer once |
| Sequential revision | edits the previous attempt | can overwrite a correct answer |

Keep tables narrow. They must render in an 800px column (`max_width` in `_config.yml`).

## Quality bar

A figure is done when:
- A reader could reconstruct the claim from the figure plus caption, without the surrounding section
- Labels are large enough to read on a laptop
- There is no unexplained color
- It would not look at home on a pitch deck

If the post is a survey and the only figures are unedited paper screenshots, add the overview diagram before calling it finished.
