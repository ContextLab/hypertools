## High-quality story-trajectories animation (align in low-D, IncrementalPCA)

Regenerated the story-trajectories demo per the maintainer feedback (choppy/slow, and — the
substantive issue — subjects' trajectories didn't move together). Method: diagnose numerically →
fix → verify (metrics + rendered frames) → **independent deep red-team**.

![new story trajectories animation](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/story/story_trajectories_new.gif)

### Root causes & fixes

**1. Align in the LOW-dimensional space (not high-D), ≥10 iterations.** The old pipeline hyperaligned
a bare 3-D UMAP embedding, which barely improved inter-subject clustering. HyperTools' canonical order
is `manip → normalize → reduce → align`, so I reduce each subject to a **`ndims=10` IncrementalPCA**
space and hyperalign **there** (`n_iter=10`), then show the first 3 aligned dims.

Measured with a **scale-free within-timepoint dispersion** (how tightly the 36 subjects cluster around
their shared centroid at each timepoint, ÷ overall cloud scale; **lower = move together more**):

| pipeline | dispersion ↓ | max per-step jump ↓ (smoothness) |
|-|-|-|
| old-style: UMAP 3-D + HyperAlign | 0.78 | **3.26** (choppy) |
| new: IncrementalPCA 10-D, **no** align | 0.88 | 0.47 |
| **new: IncrementalPCA 10-D + HyperAlign** | **0.73** | **0.37** |

Hyperalignment tightens the cloud ~18% (0.88 → 0.73) — the fix for "trajectories don't move together".

> Note on metrics: I originally reported plain inter-subject *correlation* (~0.14 → ~0.48). The
> animation red-team correctly flagged that as **misleading** — a jumpy UMAP embedding can score *higher*
> correlation while looking scattered and choppy. I retracted it; dispersion + smoothness (above) are the
> robust measures and both favor the new pipeline. Docstring/script updated accordingly.

**2. IncrementalPCA, not UMAP.** UMAP's nonlinear warping left trajectories jumpy — normalized max
per-step jump **3.26 (UMAP) → 0.37 (IncrementalPCA)**, ~9× smoother — and a linear reduction preserves
the shared structure hyperalignment depends on.

**3. Smoother & faster.** `animate='spin'` over full trajectories — a spin's frame count is independent
of the number of timepoints, so the 600-sample resample (smooth lines) costs nothing — and **9 s
instead of 30 s**.

### Also fixed along the way (in scope)
`animate='spin'`/`'serial'` passed `frame_rate * duration` straight to matplotlib's `FuncAnimation`;
a **fractional `duration`** made that a float and crashed with `range(float)` →
`"'float' object cannot be interpreted as an integer"`. Fixed (int-wrapped frame counts, both
backends) with regression tests — commit `7cc88ff9`.

### Updated
Gallery example (`examples/plot_story_trajectories.py`: docstring, exact code, three camera-angle
stills), `docs/tutorials.rst`, the mp4 (6.9 MB → 2.0 MB), the gallery gif thumbnail, and a new
committed **`scripts/generate_story_trajectories.py`** so the assets are reproducible from a
deterministic script.

### Red-team & tests
- Independent deep red-team (re-derived the dispersion + jump metrics from `hyp.load('weights')`,
  inspected rendered frames vs the benchmark, verified determinism, ran the example, checked the
  float-duration fix): **VERDICT — ACHIEVES THE CORRECT ANIMATION.** Confirmed: alignment in 10-D
  with 10 iterations (reduce before align per `CANONICAL_ORDER`); IncrementalPCA not UMAP; smooth
  (max jump 1.48 raw, camera 1.33°/frame); 9.000 s mp4; single coherent shared cloud; deterministic
  (max|Δ| = 0 across two runs); example exits 0; float-duration fix holds on both backends. Its one
  finding — that my *correlation* numbers were oversold — is fixed above and in the docstring/script.
- Full suite (kaleido plotly-export tests excluded — pre-existing headless deadlock): **1471 passed,
  4 skipped, 0 failed**.

**Branch is for review only — do not merge; base `dev-1.0-refactor`, `master` untouched.**
