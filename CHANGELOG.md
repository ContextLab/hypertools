# Changelog

## 1.0.1 (unreleased)

Small, additive plotting features and fixes (fully backward-compatible).

### New features

- **Lines are automatically smoothed (`antialias=True`, new default).** Every
  drawn line -- static or animated, in both backends -- is upsampled along a
  monotone PCHIP interpolant, so there are no sharp angles between successive
  observations. PCHIP is C1 (its tangent is continuous), so the curve bends
  smoothly *through* each sample rather than turning a corner at it, and every
  original sample stays an exact vertex of the drawn line.

  This changes only how data is DRAWN, never the data itself: returned arrays,
  `return_model=True` bundles, forecasts, hulls, densities, and per-point
  labels/markers are all unaffected. It is applied at the last stage before
  drawing, so it composes with everything upstream.

  In an ANIMATION, each frame draws the smooth curve for exactly the portion
  of the trajectory that frame would have shown -- so a short animation of a
  finely-structured trajectory (many tight loops) now renders as smooth curves
  instead of one coarse straight segment per frame, **at any `frame_rate`**.
  Frame counts and reveal pacing are unchanged.

  Only styles that draw a line (solid or dashed/dotted, including marker+line
  combos like `'o-'`) are affected; **marker-only styles are never touched**,
  so markers always render at the true sample points. `predict=` forecast
  overlays are smoothed the same way. Pass **`antialias=False`** to restore the
  previous raw straight-segment rendering exactly.

- **`animate='serial'` now composes with the trail flags**
  (`chemtrails`/`precog`/`bullettime`), in both 2-D and 3-D and on **both**
  the matplotlib and plotly backends: datasets are revealed one at a time
  (as before), but the dataset currently being drawn now carries its trail
  (e.g. a fading chemtrail) while already-revealed datasets stay fully drawn
  -- "chemtrails-serial", "precog-serial", "bullettime-serial". Plain
  `animate='serial'` (no trail flag) is unchanged; a windowed serial is left
  as follow-up. This also fixes plotly, which previously warned and dropped
  the trails for a serial reveal instead of drawing them -- the two backends
  now match frame for frame.

- **`predict=` now works with `animate='spin'`.** A spinning 3-D plot can
  carry its `predict=` forecast overlay -- the dashed, low-opacity forecast
  trace(s) are drawn once and rotate with the scene. The time-progressing
  animation styles (`True`/`'parallel'`/`'serial'`/`'window'`/`'morph'`)
  still raise `NotImplementedError` with `predict=`, since a growing forecast
  trace over those remains follow-up work.

- **`plot(..., on_frame=...)`: a public per-frame hook, on both backends.**
  `on_frame` is called once per drawn animation frame with a single
  `FrameContext` argument -- the frame index and total, the axes and drawn
  artists, the animated arrays, the serial-reveal counts, and -- for
  `animate='morph'` -- `segment_index`/`segment_kind`. This replaces
  reaching into matplotlib's private `FuncAnimation._func`/`._args` and
  re-deriving hypertools' own serial-reveal schedule by hand, which four of
  the five animated gallery examples previously did. `FrameContext` is
  exported as `hypertools.FrameContext`. On matplotlib, callbacks can also
  be attached after construction via `HyperAnimation.on_frame(callback)`
  (chainable); this is **not** available on plotly, whose animated return is
  a plain `go.Figure` with its frames already built, so pass `on_frame=` to
  `plot()` instead for backend-portable code. **Callbacks must be
  deterministic and idempotent for a given frame context. They must not
  depend on call count, call order, wall-clock time, or accumulated
  external state.** Mutating artists is supported and expected; accumulating
  is not. Matplotlib calls back at render time (so a frame index may recur
  across a loop or a save) while plotly calls back exactly once per frame
  index at build time -- same per-frame metadata on both backends, but
  `ctx.figure`/`ctx.axes`/`ctx.artists` are backend-native (`ctx.axes` is
  `None` on plotly, whose `ctx.artists` are that frame's traces), so a
  callback that mutates them is not portable across backends.

- **`order='parallel'|'serial'` on `plot()`, orthogonal to `animate=`.** So
  trail styles compose with a serial reveal (`animate=True,
  order='serial', chemtrails=True`). `animate='serial'` remains a permanent
  alias for `animate=True, order='serial'`, and `animate='morph'` is
  inherently serial. `order=` is resolved into the backend mode, so hue
  overlays and trail handling stay in sync.

- **Per-dataset `alpha=`, alongside the existing per-dataset
  `color=`/`linewidth=`.** Inputs that assign alpha internally (row
  `MultiIndex` frames, nested lists) keep their own values and now say so
  with a warning instead of losing silently.

- **Per-segment `title=` for serial-style animations, on both backends.**
  Pass a list of strings (one per dataset) to name each segment of a
  serial-style animation as it is revealed; for `animate='morph'` the holds
  are named and the transitions are left blank automatically. Anywhere else
  a non-string `title=` raises `TypeError`.

- **`simplify=` on `plot()` (default `True`).** Today it governs
  `animate='morph'` tractability only: over clouds larger than 2000 points
  an uncapped morph is downsampled to 2000 **silently**, because the
  alternative is a render that never finishes (measured: killed at 10
  minutes uncapped; 8.2 s at `morph_samples=2000`). Pass `simplify=False`
  for an explanatory `ValueError` instead, which restores the guarantee
  that no real data point is ever dropped. An explicit `morph_samples=`
  always wins, and below the threshold `simplify` does nothing at all.

### Changed

- **Animated continuous-hue line plots with no explicit `linewidth=` now
  render at `1.0` instead of `1.5`.** This is a **visible change to
  existing animated hue figures**: the overlay now matches the width of the
  artist it replaces, which is what animated no-hue lines already used, so
  hue and no-hue animations finally agree. Pass `linewidth=1.5` to keep the
  old look.

### Bug fixes

- **Animated MultiIndex plots with trails no longer crash.** Animating a
  row-`MultiIndex` `DataFrame` with `chemtrails`/`precog`/`bullettime` raised
  `TypeError: ... got multiple values for keyword argument 'alpha'`: the trail
  artists passed a hardcoded `alpha=0.3` alongside the per-trace `alpha` that
  MultiIndex expansion assigns (to distinguish faint leaf traces from opaque
  group means). The 0.3 trail fade is now folded into whatever `alpha` the
  trace already carries, in both the 3-D and 2-D animation paths.

- **`cluster=`/`hue=` line plots spanning more than one run per dataset no
  longer crash, or silently misplace, point `labels=`.** A categorical
  `cluster=`/`hue=` line plot that needs to bridge two same-dataset runs into
  one continuous line (`_regroup_categorical_lines`) duplicated a data point
  onto the end of the earlier run without adding a matching entry to the
  parallel `labels=` list, permanently leaving `labels` one entry short per
  bridge point relative to the drawn data. This crashed `annotate_plot` with
  `IndexError: list index out of range` whenever nothing else happened to
  rebuild `labels` from scratch afterward (`animate='morph'`, or a static
  plot with `antialias=False`) -- and, even when it didn't crash, silently
  misattributed real point labels to the wrong point (every other animated
  style, or a static plot with the default `antialias=True`). Bridged labels
  now grow in lockstep with the bridged data.

- **`title=` no longer stringifies a list onto the axes.** A non-string
  `title=` now raises `TypeError` instead of drawing the literal
  `"['a', 'b', 'c']"` text, and the check runs before the analyze pipeline,
  so streaming plots (`plot_stream`) get it too.

- **`linewidth=` is honored in animated continuous-hue line plots.** The
  overlay now always renders at the width of the artist it replaces
  (previously it fell back to `rcParams['lines.linewidth']` regardless of
  what you passed).

- **`animate='morph'` over clouds larger than 2000 points no longer appears
  to hang.** It is capped at 2000 points by default, or raises naming
  `morph_samples=` and `simplify=True` when you pass `simplify=False`. See
  `simplify=` above for which of your data actually reaches the plot.

- **`title=` is now actually visible on animated 3-D matplotlib plots.**
  `animate_plot3D` maximises the 3-D axes to the full canvas (so a rotating
  zoomed cube never clips at some rotation angles) -- but that left zero
  margin above the axes for `axes.set_title()` to render into, so both a
  scalar `title=` and a per-segment `title=` list rendered entirely
  off-canvas (the title *state* was always correct; only the pixels were
  missing). Matplotlib animated 3-D plots now reserve a top strip -- sized
  to the real measured title-line height, growing the figure rather than
  shrinking the maximised axes -- whenever a title will actually be drawn;
  a titleless 3-D animation, a static 3-D plot, and 2-D animations (which
  never had this problem) are all unaffected. The plotly backend already
  had the equivalent fix.

- **Fitting a legend/colorbar/title around an animated plot no longer fires
  `on_frame=` (or a per-segment `title=` schedule) one extra time before
  the animation has started.** The margin-fitting helpers this release adds
  or already had (right-side legend/colorbar fitting, the new 3-D title
  margin above) each draw the figure once to measure real content -- which,
  for an animated figure that has never been drawn yet, is enough to
  trigger matplotlib's own "first draw starts the animation" mechanism,
  silently running a premature frame-0 update. These measurement draws are
  now guarded the same way matplotlib's own `Animation.save()` guards its
  internal draws, so the animation's real first frame is never fired early.

## 1.0.0 (unreleased)

HyperTools 1.0 is a ground-up modernization of the toolbox. The familiar
one-call API (`plot`, `analyze`, `reduce`, `align`, `cluster`, `normalize`,
`describe`, `load`) is preserved, but the internals, packaging, and feature
set are new.

### Architecture

- **Package restructure:** the old monolithic `tools` module is split into
  focused subpackages -- `reduce/`, `cluster/`, `align/`, `manip/`, `io/`,
  `predict/`, `impute/`, `plot/`, and `core/` -- all built on a shared
  stack -> fit-once -> unstack model-application core (`hyp.apply_model`,
  backed by [pydata-wrangler](https://github.com/ContextLab/data-wrangler)).
- **Canonical pipeline order:** every dispatcher composes stages in the
  same order (manip -> normalize -> reduce -> align -> cluster), and
  `return_model=True` returns fitted models/`hyp.Pipeline` objects that can
  be replayed on new data via `pipeline=`.
- **Return types:** `hyp.plot` returns a matplotlib `Figure` (a
  `HyperAnimation` when `animate=` is used, or a plotly `Figure` with
  `backend='plotly'`); `hyp.load` returns raw data. The 0.x `DataGeometry`
  ("geo") container is retired to an internal unpickle-only shell so
  **pickle-format geo files saved by hypertools ≥0.8 still load** (returning
  their raw data). Older **pre-0.8 `deepdish`/HDF5-format** geo files cannot
  be read under the required NumPy 2 and must be converted once in a
  throwaway `numpy<2` environment (`hyp.load` detects them and explains how;
  see the README "Legacy data" note).
- **Packaging:** `pyproject.toml`-based packaging, Python 3.10-3.13, and a
  base install that covers all core functionality and therefore pulls in
  the full scientific stack (NumPy, SciPy, pandas, scikit-learn,
  matplotlib, seaborn, UMAP/Numba, statsmodels, pykalman, pydata-wrangler)
  -- not a minimal footprint, but fast-importing (heavy dependencies load
  lazily, so `import hypertools` is roughly 3.5x faster than 0.8.x). Heavier
  optional model families are separated into extras (`interactive`, `text`,
  `predict`, `predict-hf`, `io`, `density3d`, `torch`, `kaggle`, `lsl`,
  `gensim`, `dev`).

### New features

- Interactive plotly backend (`backend='plotly'`; `backend='auto'` selects
  plotly automatically on Google Colab/Kaggle), visually matched to the
  matplotlib backend.
- New animation styles (`'spin'`, `'serial'`, `'window'`, `'morph'`), 2-D
  animation support, and per-dataset `chemtrails`/`precog`/`bullettime`
  trails.
- Hull surfaces (`surface=`), density shading (`density=`), colorbars,
  multicolored lines via continuous/matrix `hue`, nested-list input, and
  automatic MultiIndex DataFrame expansion.
- Mixture-model ("soft") clustering (`GaussianMixture`,
  `BayesianGaussianMixture`, `LatentDirichletAllocation`, `NMF`).
- `hyp.manip` (Normalize/ZScore/Smooth/Resample) with chaining,
  `hyp.predict` timeseries forecasting (Kalman, ARIMA, GP, autoregression,
  Laplace, Chronos), and `hyp.impute` missing-data imputation (PPCA,
  Kalman, sklearn imputers).
- Optional torch-backed autoencoder reducers, gensim text
  vectorizers/semantic models, Lab Streaming Layer input
  (`hyp.io.lsl_stream`), and new `hyp.load` sources (Kaggle, Hugging Face,
  Google Sheets/Drive, Dropbox, URLs, and more local formats).
- Reproducibility via a top-level `random_state=` on
  `reduce`/`cluster`/`analyze`/`plot`.

### Removed / changed behavior

- Retired legacy arguments now raise errors instead of being silently
  accepted: `group=` (use `hue=`), `model=`/`model_params=` (use
  `reduce=`), and `align=True`/`align(method=...)` (use `align='hyper'`,
  `align='SRM'`, etc.). `cluster`'s `ndims=` is only a passthrough to
  `reduce=` and warns if passed without it.
- Plotting no longer mutates global matplotlib settings; the unreliable
  result cache was removed; HDBSCAN comes from scikit-learn instead of the
  external `hdbscan` package.
- **Categorical / cluster lines no longer bridge separate datasets (GH #291):**
  in a line plot colored by a per-point `hue=`/`cluster=` label, each
  contiguous run is drawn as its own segment. A line no longer connects the
  last point of one dataset (or category run) to the first point of the next,
  and recurring categories (e.g. `A A B B A A`) keep their run order instead
  of collapsing into one line per category. Per-dataset styles (`fmt=`,
  `linewidth=`, `marker=`, ...) propagate across the resulting segments.
- **Typography:** plots now render in a bundled sans-serif (Noto Sans, SIL
  OFL 1.1, vendored in `hypertools/external/fonts`). The **matplotlib**
  backend is handed the font FILE, so it renders in Noto Sans identically on
  every platform instead of inheriting the machine's default face. The
  **plotly** backend can only pass a family NAME to the rendering browser
  (never a font file), so it *prefers* Noto Sans but falls back to the next
  installed system face when Noto isn't present -- plotly typography can
  still vary by platform. Fonts resolve through a per-glyph FALLBACK STACK,
  so text mixing scripts renders completely from several faces rather than
  showing "tofu" boxes for whatever the primary face lacks, and the primary
  face stays Noto Sans -- an accent or Greek letter no longer swaps the whole
  plot onto some other installed font. A covering font is auto-added to the
  stack (as a fallback, Noto still primary) only when the stack genuinely
  cannot draw a character. Also: point `labels=` no longer force a serif
  face (they inherit the stack like every other text surface -- previously a
  label character the serif faces lacked rendered as tofu even when an
  installed font had it), and the "no font covers this text" warning now
  fires only for characters NOTHING available can draw, instead of whenever
  no SINGLE font covered all of it. The font stack is applied inside a scoped
  `rc_context`, so your own matplotlib settings are left untouched.
- **Animation controls (plotly):** the Play/Pause buttons moved from the
  plot's bottom-left corner to below the plotting area, laid out
  horizontally and lightly themed. In 2-D -- where the axes fill the paper
  area -- they previously overlapped the chart itself.
- **Frame outline weight (plotly):** the 3-D wireframe cube and the 2-D
  square frame now render at the same ~2px stroke (matching the matplotlib
  backend). plotly's gl line renderer draws 3-D `Scatter3d` lines lighter
  than the equivalent 2-D SVG shape, so the 3-D cube previously looked
  noticeably thinner than the 2-D square.

### Release audit (2026-07)

Before release, the codebase, documentation, examples, and tutorials were
red-teamed in a 46-unit audit that filed 708 findings (691 confirmed by an
independent verifier). The confirmed code findings were fixed in waves
(350+ fixes merged as of this entry), including these criticals:

- `hyp.load('sotus')` returns the full 29-speech State of the Union corpus
  again (the hosted corpus had been loading incompletely).
- `hyp.align` preserves each dataset's row order (aligned outputs are no
  longer returned with scrambled rows).
- `hyp.manip` smoothing runs per dataset: `Smooth` kernels no longer bleed
  across dataset boundaries when given a list.
- The Kalman forecaster (`hyp.predict(..., model='Kalman')`) actually
  learns its dynamics model instead of filtering with default parameters.
- CSV/TSV parsing bugs in `hyp.load` were fixed, and `hyp.save` writes
  atomically and format-aware.
- `import hypertools` no longer crashes under unusual
  backend-related environment variable configurations.
- Plotting nested lists of datasets (`hyp.plot([[a, b], [c]])`) works
  correctly again.

Docs, README, examples, and tutorials were then re-verified by executing
them against the fixed code.

## 0.8.x and earlier

See [RELEASE_NOTES_0.8.1.md](RELEASE_NOTES_0.8.1.md) and the
[GitHub releases page](https://github.com/ContextLab/hypertools/releases).
