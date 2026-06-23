# Training at the Critical Point — blog & experiment spec

Working title: **Training at the Critical Point: A Physics-Inspired Map of Deep Learning**

Thesis: deep learning theory becomes easier once you stop asking for one universal explanation of generalization and start asking **which phase of training the model is in**.

Audience: AI / CS readers who think in losses and benchmarks. Goal: use deep learning as a playground to build **physics intuition** (regimes, order parameters, phase boundaries), not to teach physics from scratch.

Figures must come from **real runs** (GPU), logged to CSV, plotted with a consistent aesthetic. Publish PNG (+ optional SVG) to `assets/img/blog/critical-point/`. Scratch lives in `experiments/training-at-critical-point/` (gitignored).

---

## Narrative spine

1. **Personal frame (short):** I came from CS / deep learning. Physics felt alien until I noticed neural networks already behave like physical systems: regimes, order parameters, critical boundaries, emergent geometry.
2. **The physics move:** track macroscopic variables, not every weight.
3. **Walk the phase diagram:** double descent → lazy/rich → edge of stability → implicit bias → neural collapse → grokking → scaling laws.
4. **Synthesis:** candidate order parameters and a dashboard mindset.
5. **Close:** physics is useful here as a **discipline of abstraction**, not as a literal analogy to gases or magnets.

Tone: rich on DL theory, light on formal stat-mech. No em dashes. Equations where they clarify. Every figure teaches one mental model.

---

## Blog outline (section-by-section)

### 0. Frontmatter

```yaml
title: "Training at the Critical Point"
date: 2026-XX-XX
layout: post
description: "A physics-inspired phase diagram for deep learning: stability, implicit regularization, grokking, and neural collapse."
categories: [technical]
tags: [deep-learning-theory, statistical-physics, neural-collapse, grokking, optimization]
```

### 1. Opening — deep learning before physics (~400 words)

- CS default: architecture, loss, optimizer, benchmark.
- Physics default: what **phase** is the system in? what **macroscopic** quantity tells the story?
- Pivot: a trained network is an **endpoint of a trajectory**, not a static artifact.
- **Figure 1** (micro vs macro): real training run, messy loss/θ path abstracted into 4–5 logged gauges.

Key line: *Deep learning theory is useful when it gives us a phase diagram.*

### 2. Why “phase” is the right word (~500 words)

- Phase = same system, different macroscopic behavior (not a different object).
- Knobs: width, data size, lr, batch size, weight decay, noise, training time.
- Training dynamics: θ_{t+1} = θ_t − η ∇L(θ_t; B_t).
- Question shift: *Which regime is θ_t in?* not only *What is L(θ_t)?*
- **Figure 2 (HERO):** empirical phase diagram from a 2D hyperparameter sweep.

### 3. The old map breaks — interpolation & double descent (~600 words)

- Classical U-curve vs modern double descent (Belkin et al.).
- Interpolation threshold N_eff ≈ n as a **phase boundary**.
- Test error: down → spike near interpolation → down again in overparameterized regime.
- **Figure 3:** real width sweep, train + test error, shaded phases.

### 4. Lazy learning vs feature learning (~600 words)

- NTK / lazy: function changes, representations frozen (d_h ≈ 0).
- Rich regime: d_h ≫ 0, features reorganize.
- d_θ(t) = ‖θ_t − θ_0‖ / ‖θ_0‖,  d_h(t) = E_x ‖h_t(x) − h_0(x)‖².
- **Figure 4:** heatmap width × lr (or init scale × width), color = d_h at fixed epoch; contour separating lazy/rich.

### 5. Edge of stability (~600 words)

- Stability: η λ_max(H) < 2; Cohen et al.: training often rides χ = η λ_max(H)/2 ≈ 1.
- Optimizer as dynamical system, not pure descent.
- **Figure 5:** real time series — loss (left) + λ_max(H) (right) + band at 2/η.

### 6. Implicit regularization (~500 words)

- Many θ fit the data; dynamics select among M_0 = {θ : L_train = 0}.
- Generalization is a property of the **trajectory**, not only the endpoint.
- **Figure 6 (optional):** 2D PCA of weights colored by train loss, multiple lr trajectories; or skip and fold into §5/§7.

### 7. Neural collapse — order after interpolation (~700 words)

- Papyan et al.: NC terminal phase — ETF geometry, class means, collapsed within-class variance.
- Order parameter m_NC = 1 − σ_W² / (σ_B² + ε) → 1.
- **Figure 7 (HERO):** feature PCA panels at 3 epochs + m_NC(t) curve on same run.

### 8. Grokking — delayed phase transition (~600 words)

- Train acc ↑ early, test acc flat then jumps (Power et al.; phase-transition follow-ups).
- Gap g(t) = A_train − A_test.
- **Figure 8 (HERO):** modular arithmetic, log-step train/test acc + gap shading + t_c marker.

### 9. Scaling laws (~400 words)

- Kaplan et al.: L(N) = L_∞ + a N^{−α} on log-log axes.
- Complements phase diagram: smooth structure **inside** a regime.
- **Figure 9:** real width or data sweep, log-log loss with fit line.

### 10. Synthesis — order parameters (~500 words)

- Table: phase ↔ training regime, temperature ↔ lr/noise, order parameter ↔ χ, m_NC, g, d_h, effective rank.
- **Figure 10:** dashboard small-multiples from one canonical run (or composite from experiments above).

### 11. Closing (~250 words)

- Habit for AI people learning physics: look for regimes and order parameters.
- DL is a natural laboratory: high-dimensional, tunable, abrupt transitions + power laws.
- Closing: *Deep learning theory, at its best, is the search for that diagram.*

### References (anchor papers)

- Belkin et al. — double descent
- Jacot et al. / Lee et al. — NTK
- Chizat et al. — lazy training
- Cohen et al. — edge of stability
- Papyan et al. — neural collapse
- Power et al. — grokking
- Kaplan et al. — scaling laws
- Olsson et al. — induction heads (optional cross-link to ICL)

---

## Figure plan — real experiments only

Priority tiers:

| Tier | Figures | Role |
|------|---------|------|
| **S** (must ship) | 2, 5, 7, 8 | Visual identity of the post |
| **A** (strongly recommended) | 1, 3, 4, 9 | Complete the phase narrative |
| **B** (if time) | 6, 10 | Polish / synthesis |

All runs: fixed seeds, `config.yaml` per sweep, step-level CSV, plot script reads CSV only.

### Figure 1 — Microscopic vs macroscopic (Tier A)

**What it shows:** One training run is high-dimensional chaos; a few scalars tell the story.

**Experiment**

- Model: ResNet-18 on CIFAR-10 (or 3-layer MLP on MNIST for faster iteration).
- Log every 50–100 steps: train loss, test acc, ‖θ_t − θ_0‖, λ_max(H) (power iteration, k=10), m_NC on penultimate features, gen gap.
- Left panel: 2D PCA of θ snapshots along trajectory (real weights, not synthetic path).
- Right panel: normalized time series of the four order parameters (small multiples or gauge-style).

**GPU:** ~1× A100, 1–2 hours single run + logging overhead.

**Publish:** `fig01-order-parameters-dashboard.png`

---

### Figure 2 — Hero phase diagram (Tier S)

**What it shows:** Regimes change when capacity and training “knobs” change.

**Experiment (primary — colorful heatmap)**

- Grid sweep: **width multiplier** w ∈ {0.25, 0.5, 1, 2, 4, 8} × **effective training time** (early stop epochs or weight decay λ ∈ grid).
- Task: CIFAR-10, same architecture family (e.g. WideResNet depth fixed, width scaled).
- Color metric (pick one primary, others in appendix):
  - test accuracy at end, or
  - m_NC at end, or
  - generalization gap L_train − L_test.
- Overlay contour lines + text labels for regions (underfitting, interpolation band, rich learning, collapse) placed by hand from grid inspection.

**Alternative 2D sweep (faster):** lr × batch size on fixed model, color = χ = η λ_max(H)/2 at epoch 50 (edge-of-stability phase map).

**GPU:** 36–64 runs × ~15 min ≈ 9–16 GPU-hours (parallelize as SLURM array).

**Publish:** `fig02-phase-diagram.png`

---

### Figure 3 — Double descent (Tier A)

**What it shows:** Interpolation threshold is a boundary, not “overfitting starts.”

**Experiment**

- Width sweep: 16 widths log-spaced (same depth MLP on subset of MNIST/CIFAR or random labels ablation for clean interpolation).
- Plot train + test error vs parameter count N (or width).
- Vertical line at smallest width where train error → 0.
- Background shading: underparameterized | critical | overparameterized.
- Optional dashed overlay: classical U-curve sketch from bias-variance formula on same axes (muted, labeled “classical expectation”).

**GPU:** 16 runs × ~5–20 min depending on model.

**Publish:** `fig03-double-descent.png`

---

### Figure 4 — Lazy → rich heatmap (Tier A)

**What it shows:** “Kernel machine” and “representation learner” are different phases.

**Experiment**

- Grid: **width** × **learning rate** (or init scale α for NTK lazy limit).
- At fixed epoch (e.g. 20% and 80% of training), compute d_h on full train set (penultimate layer, subsample 2k points if needed).
- Heatmap: color = d_h; white contour at threshold separating lazy/rich (e.g. d_h < 0.01 vs > 0.05, calibrate on pilot).
- Side panel: one lazy and one rich trajectory of d_h(t) over training.

**GPU:** 5×5 or 6×6 grid ≈ 4–8 GPU-hours.

**Publish:** `fig04-lazy-rich-heatmap.png`

---

### Figure 5 — Edge of stability (Tier S)

**What it shows:** Sharpness tracks 2/η; loss can oscillate yet decrease.

**Experiment**

- Replicate Cohen et al. style: SGD, no momentum (or small), CIFAR-10 + ResNet-18, fixed η.
- Every 10 steps: loss, λ_max(H) via 5–10 power iterations on Hessian-vector products (PyTorch autograd).
- Plot dual-axis time series; shaded band [0.95, 1.05] × (2/η).
- Optional: second curve with smaller η showing stable interior (χ < 1 throughout).

**GPU:** ~2–4 hours (HVP logging is the bottleneck).

**Publish:** `fig05-edge-of-stability.png`

---

### Figure 6 — Many minima / implicit bias (Tier B)

**What it shows:** Zero train loss is underdetermined; path matters.

**Experiment**

- Same architecture, same data, **different lr or batch size** → train to zero error.
- 2D PCA of final θ (or penultimate features) for 4 runs; color background by train loss on grid (Li et al. visualization style).
- Trajectories overlaid in PCA space.

**GPU:** 4 runs + grid eval ≈ 3–5 hours.

**Publish:** `fig06-loss-landscape-trajectories.png` (optional)

---

### Figure 7 — Neural collapse (Tier S)

**What it shows:** Terminal training increases symmetry / geometric order.

**Experiment**

- Train ResNet-18 (or MLP classifier) on CIFAR-10 to 100% train acc; continue 20–30% more epochs.
- Save checkpoints at t_early, t_interp (~100% train), t_late.
- Penultimate features → PCA 2D; plot class-colored points (10 classes, distinct palette).
- Bottom strip: m_NC(t) and σ_W/σ_B vs epoch on same run.

**GPU:** 1 long run ~2–3 hours.

**Publish:** `fig07-neural-collapse-panels.png`

---

### Figure 8 — Grokking (Tier S)

**What it shows:** Memorization then sudden generalization.

**Experiment**

- Task: modular addition (a + b) mod p, p=113 or p=97; transformer or 2-layer MLP (Power et al. setup).
- Strong weight decay, small train fraction (~40%) helps grokking appear.
- Log train/test accuracy every N steps for 200k–500k steps.
- x-axis log steps; shade g(t); vertical line at t_c (max second derivative of test acc or manual pick).

**GPU:** 1–2 runs × 4–12 hours depending on grokking speed.

**Publish:** `fig08-grokking-transition.png`

---

### Figure 9 — Scaling laws (Tier A)

**What it shows:** Power-law structure inside a regime.

**Experiment**

- Train 8–12 model widths (or dataset sizes) to same epoch budget; record final train/test loss.
- Log-log plot L vs N; linear fit for α; show L_∞ from fit.
- Optional: three colored curves (data-limited / model-limited / compute-matched) if you run multi-axis sweep.

**GPU:** 8–12 runs ≈ 2–4 GPU-hours.

**Publish:** `fig09-scaling-law.png`

---

### Figure 10 — Order-parameter dashboard (Tier B)

**What it shows:** Practical “what to log” summary.

**Experiment**

- Reuse logs from Figure 1 or 5 run.
- Small multiples: χ(t), m_NC(t), g(t), d_h(t), effective rank of feature matrix.

**Publish:** `fig10-dashboard.png`

---

## Plot aesthetic (all figures)

- Fonts: sans-serif, large axis labels embedded in figure (blog width ~800px).
- Colormaps: **plasma**, **viridis**, **cividis** for heatmaps; class colors vivid but distinct (tab10/tab20).
- No bar charts. Prefer: heatmaps, contours, log-log lines, trajectories, small multiples.
- Every caption states **what was measured** and **N seeds** if averaged.
- Export: PNG 200–300 DPI + CSV alongside in assets or linked from repo scripts.

---

## Experiment folder layout

```
experiments/training-at-critical-point/   # gitignored
  runs/
    fig02_phase_diagram/
      config.yaml
      logs.csv
      checkpoints/
  plot/
    fig02_phase_diagram.py
  slurm/
    submit_fig02.sh

assets/img/blog/critical-point/         # tracked — publish here
  fig02-phase-diagram.png
  fig02-phase-diagram.csv                 # optional data export
```

Tracked in git:

```
tools/training-at-critical-point/
  BLOG_SPEC.md          # this file
  README.md
  plot/                 # plotting scripts (read CSV → PNG to assets)
  train/                # training scripts per figure
  slurm/                # Kempner submission templates
```

---

## Suggested production order

1. **Fig 5** (edge of stability) — one run, validates logging pipeline for λ_max.
2. **Fig 7** (neural collapse) — one run, validates feature / m_NC code.
3. **Fig 8** (grokking) — start early (long wall time).
4. **Fig 3** (double descent) — width sweep, straightforward.
5. **Fig 2** (phase diagram) — largest sweep; do last once metrics stable.
6. **Fig 4, 9, 1, 10** — fill in narrative.
7. **Fig 6** — optional.

**Total compute (rough):** 25–45 GPU-hours on A100 if sweeps parallelized with array jobs (%15 cap).

---

## What we are NOT claiming

- Schematic / cartoon phase boundaries labeled without measurement.
- Universal phase diagram for all architectures.
- Novel theory — the contribution is **pedagogy + visualization + unified order-parameter language** for AI readers approaching physics.

---

## Next steps (implementation)

1. Replace schematic `generate_figures.py` with per-figure `train/` + `plot/` scripts.
2. Add SLURM worker template (Kempner, dynamics venv).
3. Draft `_posts/2026-XX-XX-training-at-critical-point.md` with figure placeholders after Fig 5+7 pilot runs exist.
4. Commit only `assets/img/blog/critical-point/*` + scripts under `tools/`.
