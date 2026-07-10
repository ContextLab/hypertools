## Story animation, take 2 (aligned in the hub space) + every remaining punted bug

This round fixes the story animation properly and clears the bugs that earlier sessions
surfaced-but-punted. Method throughout: reproduce → fix → verify (numeric + screenshots) →
**independent red-team subagent** → re-verify.

### 1. Story trajectories — now genuinely aligned, window style (`7987f3e1`)

The previous take was a spaghetti tangle, not the reference's single coherent shape, and used a
`spin` instead of a `window`. **Root cause of the poor alignment:** it reduced each subject to a
10-D space and hyperaligned *there*. Hyperalignment rotates each subject onto a shared response and
needs room — 10 dims (let alone 3) starves it. The fix is to **hyperalign in the full 100-hub feature
space first, then reduce to 3-D**:

```python
manip_data = hyp.manip(data, model=[Smooth(40), Resample(300), ZScore])
aligned = hyp.align(manip_data, align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}})  # in 100-hub space
hyp.plot(aligned, '-', color=colors, linewidth=1.6, reduce='IncrementalPCA',
         ndims=3, animate='window', focused=1.5, zoom=1.5, duration=9, ...)
```

Scale-free within-timepoint dispersion (subjects' spread around their shared centroid per timepoint,
÷ cloud scale; **lower = move together more**), computed deterministically on the displayed 3-D
coordinates and independently re-derived by the red-team:

| pipeline | dispersion ↓ |
|-|-|
| unaligned baseline | 0.78 |
| reduce-to-3-D then align (rejected) | 0.75 |
| reduce-to-10-D then align (previous, rejected) | 0.64 |
| **align in 100-hub space then reduce (new)** | **0.47** |

Switched `animate='spin'` → `animate='window'` (a sliding trail traverses each aligned trajectory,
so all 36 subjects are seen moving together through the story — the reference style). The render was
then matched to the reference by **direct frame-by-frame comparison against `hypertools.gif`**: a
short window (`focused=1.5 s` → 45 of 300 frames), **bold near-opaque lines** (translucent lines just
blur into haze — the reference is opaque), and the **`husl` palette** (the classic HyperTools tempered
hues; `gist_rainbow` read gaudy and made single strands pop). Updated the gallery example, tutorial,
mp4, gif thumbnail, and the deterministic generation script.

Alignment tightness was also confirmed *numerically*, not just by eye: `hyp.reduce` on a list uses a
**shared projection** (dispersion 0.466 vs 0.464 for a hand-fit shared PCA — so reducing does not
re-break the alignment), and every subject's trajectory mean lies within **0.04·scale** of the grand
centroid. The occasional strand looping wide is one subject's momentary window segment, not a
displaced subject.

![new story window animation](https://raw.githubusercontent.com/ContextLab/hypertools/fix/qc-notes-2026-07/notes/fix-qc-notes-2026-07/evidence/story/story_window_v2.gif)

### 2. `cluster(reduce=, return_model=True)` returned an unusable Pipeline (`cc42822c`)

A no-op reduce (`ndims=None` or `≥ n_features` → `model=None`) left the pipeline's reduce STEP
"unfitted", so `p.transform(X)` / reusing the model crashed `NotFittedError`. `_DispatchStep` now
records that fit ran and passes data through on a no-op, so no-op stages stay reusable. (A *new*,
deeper bug than the `labels_` crash prior sessions thought they'd fixed.)

### 3. Per-point labels showed on every animation frame (`52bcff88` + `aaf12b97`) — the deferred "known limitation"

`labels=` in an animation were drawn on every frame at their frame-0 position. Each label now records
its point index, and a per-frame sync shows a label **only while its datapoint is currently drawn** and
reprojects it for the rotated camera. The initial fix covered 3-D parallel/window/spin; the code
red-team caught that 3-D serial, **all 2-D animations, and morph were still unsynced (and the commit
over-claimed 'serial')** — `aaf12b97` completes it: serial reveals labels cumulatively (by global index
across datasets), 2-D windows track, and morph (a single traveling cloud) hides them. All 7 update
paths covered, docstring corrected, +6 regression tests total.

### 4. Top-level `random_state` for reproducibility (`26232a62`)

Stochastic stages were only reproducible via a verbose dict spec. Added `random_state=` to
`reduce`/`cluster`/`analyze`/`plot`, injected into any stage model that accepts one (UMAP, TSNE,
KMeans, GaussianMixture, ...); deterministic models and pre-built instances are untouched, and an
explicit dict-spec `random_state` still wins.

*(Also this session: `7cc88ff9` — fractional `duration` crashed `spin`/`serial` animations
(`range(float)`), both backends.)*

### Red-team & tests (independent subagents)
- **Animation red-teams (three passes, each re-verified against the reference).** Pass 1 confirmed the
  subjects now move together and re-derived the dispersion table above; caveat "still knottier than the
  reference" → drove opaque bold lines. Pass 2 confirmed the window style matches but flagged the palette
  as gaudy and single strands as popping → drove the `husl` palette + the numerical alignment
  verification above. **Pass 3 (husl render vs reference) → VERDICT: MATCHES REFERENCE (ship-quality)**
  — palette tempered (not gaudy), trajectories read as one coherent bundle, lines bold/legible, window +
  rotating-camera style matches; residual strand excursions fall within the reference's own envelope.
- **Code-fixes red-team → FIX 1 (cluster reuse) SOLID, FIX 3 (random_state) SOLID** (both proven
  non-vacuous with correct numerics + edge cases). It found **FIX 2 (labels) incomplete** — serial/2-D/
  morph unsynced + an over-claiming docstring — now fixed and re-verified in `aaf12b97`.
- Full suite re-run after all fixes (kaleido plotly-export tests excluded — pre-existing headless
  deadlock): **1485 passed, 0 failed, 4 skipped, 7 deselected**.

**Branch is for review only — do not merge; base `dev-1.0-refactor`, `master` untouched.**
