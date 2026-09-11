# Changelog

## 1.1.0 (2026-09-04)

Hierarchical (`MultiIndex`) DataFrames become a first-class input. A frame
whose **columns** carry a hierarchy now expands into one trace per group
plus per-level means (the row axis has done this since 1.0); `hyp.predict`
forecasts a hierarchy one group at a time with explicit model ownership; and
`predict=` forecasts every plotted trajectory, derived means included.

Five previously-accepted inputs are now **rejected** -- see *Changed /
validation*. One of them (duplicate timestamps) is not hierarchy-specific
and reaches flat `hyp.predict` callers.

### Added

- **Forecasts use observation times.** Timed rows are sorted before fitting,
  and one future step is the median positive timestamp gap (formerly the
  minimum gap), independently per dataset. Every forecaster accepts a `step=`
  override. GaussianProcess fits actual times; discrete-time models linearly
  interpolate irregular observations onto a regular grid, with a warning.
  Fitted reuse preserves the learned time scale. Series plots fit their signal
  columns together using the time index, including animated forecasts, instead
  of forecasting each `[index, value]` display pair. See the API guide for the
  interpolation policy and its limitations.
- **Integer timestamp arithmetic preserves elapsed times.** Signed and
  unsigned integer indexes use overflow-safe subtraction and addition,
  including large epoch offsets. Discrete-time forecasting no longer rejects
  valid unsigned timestamps because their past offsets wrap into the future.
- **Backtests score matching observation times.** GaussianProcess evaluates
  held-out timestamps directly; discrete-time forecasts are linearly
  interpolated to them, with a warning when needed. Training rows alone
  determine the model and its step. Returned forecasts carry the same index
  as the truth, and `horizon` reports held-out observations rather than the
  number of generated grid steps.
- **Manipulator input and row-wise fixes.** A 1-D array or flat numeric
  list/tuple consistently means one column in `manip`, direct classes,
  `Pipeline`, and fitted reuse. Row-wise ZScore/Normalize work on multiple
  datasets while preserving each dataset's statistics, index and column names.
  MatrixColormap's exact interpolation now honors `set_gamma()`.

- **A column MultiIndex frame expands into one trace per group.** The
  innermost column level is the feature axis; every level above it groups,
  so a `(Market, Sector, Ticker)` frame draws one trajectory per sector plus
  a heavier market-mean trajectory. Widths, opacities, colours and legend
  labels follow the same documented formulas as row expansion. Unlike the
  row rule, every group keeps all `len(df)` rows -- column grouping never
  shortens a trace. A two-level `(Group, Feature)` hierarchy has no
  aggregate mean, so **every leaf is treated as top-level and carries its
  own legend label**; previously such traces would all have been
  `'_nolegend_'` and the legend empty. Each group's leaf is flattened onto
  the feature axis -- its columns become the innermost level's values,
  keeping that level's name -- and the frame you passed in is never
  modified.

  **Feature correspondence across groups is by NAME.** Every group must
  carry the same innermost labels; later groups are permuted into the first
  group's order, so values travel with their labels and a permuted group
  plots identically. Mismatches -- unequal widths included -- are refused by
  an error that names the missing and unexpected features, rather than
  falling through to the pipeline's generic equal-width message. Positional
  matching would make column ORDER part of the statistical model (measured:
  permuting one group's columns moved that group's trajectory and every mean
  derived from it, on label-equivalent frames), which is not a safe default
  for a labelled frame. **Duplicate feature names inside a group are
  permitted**, matched across groups by `(label, occurrence)`: no column is
  dropped and no group is merged.

- **Continuous `hue=` propagates through a column hierarchy** as a per-trace
  value: a flat sequence of `len(df)` values is broadcast to every leaf, or
  pass one sequence per leaf. A flat array sized to the TOTAL DRAWN
  observations is rejected rather than reinterpreted -- it would require the
  caller to predict how many mean traces expansion creates. A mean trace
  takes the element-wise mean of its leaves' hue, and hue is truncated by
  the same operation that truncates the data. A forecast overlay takes the
  final observed hue colour of its source trace, static and animated alike
  and on both backends (see *Documented limitations* for how this differs
  from a categorical regrouping). Categorical hue still defers to the
  grouping, with a warning: it regroups traces, so the named leaves would
  stop existing.

- **`hyp.predict` accepts hierarchical frames.** A bare frame with a
  MultiIndex on one axis is split into groups and forecast group by group,
  returning parallel sequences (`[f0, f1, ...]`, or `([f0, ...], [m0, ...])`
  with `return_model=True`). Before this, a column-hierarchical frame was
  silently flattened into ONE wide series (measured: a 6-ticker frame
  returned a single `(1, 6)` forecast) and a row-hierarchical one died with
  `TypeError: cannot perform __sub__ with this index type: MultiIndex` deep
  inside pandas.

  Column hierarchies group by every level above the innermost (feature)
  level; row hierarchies group by every level above the innermost (time)
  level and **keep that level as each group's flat index**, with its name
  and dtype intact, so a datetime-like `t` works per group. (That name
  belongs to the group, not to the result: the returned forecast's horizon
  index is unnamed, as it is on flat input.)

  On the column axis `hyp.predict` inherits the plot path's **NAME-based
  feature correspondence** in full -- including its refusals; see *Changed
  / validation*.

  **Model ownership is explicit.** A name/class/dict spec is stateless, so
  every group constructs its own model; an **unfitted** instance is
  deep-copied per group and each copy fits independently (the caller's
  object is never fitted, and later groups never fall onto `predict_new`
  because an earlier group fitted a shared object); a **fitted** instance is
  deep-copied and replays its learned parameters. The caller's instance is
  never mutated either way, and the caller's frame is never mutated by the
  grouping.

  Per-group warnings are re-emitted with the group key prepended, preserving
  the warning category, and per-group `ValueError`s are re-raised the same
  way. Arguments that describe the WHOLE call -- the horizon `t` and the
  `model=` spec -- are validated once, before the group loop, so a bad
  horizon or a misspelled model name is no longer reported as though one
  group's data were at fault.

- **`predict=` works with hierarchies**, forecasting every plotted
  trajectory including per-level means; a mean is forecast from its own
  averaged trajectory, not from the average of its leaves' forecasts, and
  `bundle['predict']['forecasts'][i] == hyp.predict(bundle['trace_data'][i],
  model, t)` for every `i`. It works with `animate=` too, where each frame's
  forecast is fit from exactly the rows that frame has revealed.

  Every plotted trace needs at least 2 rows, on **either** axis, and
  `plot()` says so directly instead of failing deep inside the forecaster.
  For a **column** hierarchy that holds whenever the frame itself has at
  least 2 rows, since every group keeps all of them. For a **row** hierarchy
  every expanded leaf and every derived mean must clear it; because row
  expansion draws one trace per unique full index tuple, a frame whose
  innermost index level is unique per row yields one-row traces. The check
  necessarily runs on the trajectories as PLOTTED, so when a row-count-
  changing analysis stage (`manip='Resample'`, an aggregating `reduce=`) is
  what made a trace short, the message names that stage instead of blaming
  the input or the grouping.

- **`return_model=True` now also returns `trace_data` and `trace_metadata`**
  describing every plotted trajectory. `trace_data` holds the final
  pre-center/pre-scale trajectories -- the drawn artists are centered,
  scaled and (by default) antialiased copies of them, so they hold neither
  array. `xform_data` is unchanged: it remains the analysed pipeline output
  for the input datasets, one entry per analysed dataset, and derived means
  never enter it. The two are the same object only when no display-only
  projection occurred; a `reduce=` spec pinning more than three components
  makes them differ (`n_components=5` leaves `xform_data` 5-D while
  `trace_data` is 3-D). **Bundled forecasts always correspond to
  `trace_data`.**

- **Full plotly parity** for all of the above: trace counts and order,
  widths, opacities, legend labels, continuous hue, the colorbar, per-trace
  forecasts on both axes, and animated hierarchy forecasts. Data traces now
  carry `meta['hyp_trace_index']` -- the plotly half of matplotlib's
  `coll._hyp_trace_index` -- so a caller can tell the data traces apart from
  the wireframe cube, density/surface layers, forecast overlays and the
  colorbar's phantom trace. It is propagated to the per-segment 2-D traces,
  so a multicoloured 2-D line still reads as ONE trajectory. Documented in
  `docs/animation.rst` beside the forecast tags.

- **New guide:** *Hierarchical DataFrames* (`docs/hierarchy.rst`), covering
  row versus column semantics, the plot/predict divergence, hue forms, mean
  construction, limitations, dual-axis and list inputs, return shapes, the
  unfitted/fitted ownership table, backend parity and feature
  correspondence. Its worked examples are `.. doctest::` blocks, run by
  `make doctest` in `docs/`; the test suite pins the guide's section list,
  its links from the API reference and the tutorials page, its comparison
  table, and the error messages it quotes
  (`tests/test_docs_hierarchy_guide.py`). `docs/pipeline_order.rst` gains hierarchy
  expansion and mean construction as a side branch, in the prose and in the
  regenerated diagram: expansion runs before `format_data`/`analyze`, so
  every leaf gets the identical canonical pipeline, while mean construction
  runs after the display reduce -- which is why means reach `trace_data` and
  never `xform_data`.

- **A plot can take its colours from an image.**
  `hypertools.plot.colors.image_palette(image, n_colors=6)` extracts a
  palette from a LOCAL image -- a path, a PIL image, or an `(H, W, 3)` array
  -- ordered most visually salient first. Salience is
  `pixel_fraction * chroma`, so a painting's vivid subject leads and its
  muted background follows but is kept; ordering by pixel share alone (the
  obvious "largest k-means cluster" rule) returns the background, which is
  the whole reason this helper exists. A greyscale image has no colour to be
  salient about, so it falls back to population order. The same extraction
  is reachable declaratively from any plotting call as
  `palette='image:<path>'`, on both backends and on every colour path
  (categorical, continuous, matrix hue, and the colorbar): a categorical hue
  pulls one anchor per category, so the number of groups is not capped, and
  a continuous hue blends six anchors into a gradient exactly as any short
  colour list is blended. An image with fewer distinct colours than there
  are categories is interpolated up rather than cycled, so no two categories
  share a colour; a single-colour image raises instead of inventing them.
  hypertools never downloads the image -- fetch and cache it yourself, then
  pass the path.

- **`hue_mode=` says how a matrix `hue` is read.** `'mixture'` blends each
  row through `palette` as weights, one palette colour per COLUMN whatever
  the width, which is what a hierarchy needs to give each leaf one primary
  and every derived mean the blend of its children. `'rgb'` reduces the
  matrix to three min-max scaled channels used directly as (r, g, b), which
  is what a wide matrix does by default. The default stays the historical
  width rule (`None`), so existing figures do not repaint; `hue_mode=` is
  only meaningful for a matrix `hue`, and `hue_mode='mixture'` combined with
  `color_reduce=` raises rather than picking a winner. Documented in
  `docs/hierarchy.rst`.

- **`Normalize` gains `mode='isotropic'`: one shared centre and scale for the
  whole table.** The default `mode='minmax'` rescales every column on its own
  and so distorts a point cloud's shape; `mode='isotropic'` subtracts the
  centroid and divides EVERY column by one scalar (the largest absolute
  deviation from the centroid), mapping the cloud into `[min, max]` with the
  centroid at the midpoint and its shape (angles, distance ratios) intact.
  `min=-1, max=1` is exactly the "centre and scale into the unit cube" recipe
  the morph gallery examples used to hand-roll. Lists share one centre and
  scale; `return_model=True` reuse and `inverse_transform` work.
- **Gallery and tutorials reorganized and completed (issue #284).** The
  gallery merges its duplicate pages (digits/TSNE/UMAP into one; PPCA into
  missing-data; MDS into the animation page; chemtrails and precog into one
  trails page; the three cluster pages into one), drops the shape-morph
  subset of the morph zoo, computes the story-trajectories page live, orders
  pages by topic, and uses public calls throughout (`hyp.load('digits')`,
  `hyp.align(model='Procrustes')`, text via `hyp.plot(vectorizer=...)`,
  `Normalize(mode='isotropic')`, `alpha=` on morphs). Five new tutorials:
  hierarchical DataFrames, loading and saving, fitted models and pipelines,
  manipulation with window and trail animations, and an animated forecast
  (new gallery example `animate_forecast`). The plot, align, analyze and
  reduce tutorials were rebuilt to exercise every stage and argument they
  describe (names, colorbar, alpha, font, label_alpha, impute, resample,
  animate, the plotly backend, saving, the exception classes); the stock
  tutorial adds Chronos, the projectile tutorial compares scikit-learn
  imputers with Kalman; tutorials load Hugging Face data through `hyp.load`,
  save mp4 clips, and no longer silence warnings.
- **Real axis units for 1-D and 2-D plots.** `axis_scale='data'` keeps the
  pipeline output's own coordinates (for `reduce=None` the raw columns) with
  fixed joint limits for animations, on both backends; `xlim=`/`ylim=` set
  them explicitly. `axis_scale='unit'` (the default) is unchanged and now
  documented: 2-D plots are mean-centred, rescaled into the unit box and
  pinned to (-1.1, 1.1) (GH #285).
- **`ndims=1` is a time-series mode.** Each column becomes one line against
  the row index (a DatetimeIndex gives real dates; arrays use 0..n-1), 2+
  columns are allowed and legend-named by column, `axis_scale` defaults to
  `'data'`, and animations reveal along x. Previously `ndims=1` drew one
  column rescaled into [-1, 1] at 0..n-1 and refused 2+ columns (GH #285).
- **`truth=` beside a forecast.** `hyp.plot(train, predict='Chronos', t=30,
  truth=held_out)` draws the actual continuation in the same space, styled
  distinctly and role-tagged `'truth'`, static and animated, on both
  backends; `truth` is validated against `t` (GH #285).
- **Several forecasters on one plot.** `predict=['Kalman', 'ARIMA', 'GP']` (or
  `{name: spec}`) draws one legend-labelled overlay per model, coloured from
  `forecast_palette`, with `forecast_fmt` per model; works with `truth=`,
  animations and plotly. The `return_model` bundle keys the forecasts by
  model name (GH #285).
- **Animation callbacks know where the reveal is.** `FrameContext.progress`
  (0 to 1 over the clip, every style and backend) and `window_bounds`, and
  `revealed_counts` is now populated on parallel, `'window'` and `'spin'`
  reveals instead of `None` (GH #285).
- **Titles that follow the data.** `title=` accepts a callable `ctx -> str`
  and a `'{index:%B %Y}'` pattern formatted from a DataFrame's row index at
  the reveal head, on both backends, styled by `title_kwargs=`,
  `title_color=` and `font=`; an animated 3-D plot with a multi-line (or
  `title_wrap=`-wrapped) title now reserves room for every line (GH #285).
- **`animate='morph', loop=True`** closes the sequence on the first cloud's
  own sampled points, so a looping morph needs no hand-made sample (GH #285).
- **`dataset_fade={'floor': f, 'decay': d}`** fades already-revealed datasets
  by recency on serial reveals (`floor + (1 - floor) * decay**k`; matplotlib
  alpha, plotly opacity) (GH #285).
- **`companion=` panels.** 2-D panels laid out beside an animation and
  revealed in lockstep with it (a revealed series, a smoothed mean, a head
  marker); matplotlib only, plotly raises `NotImplementedError` (GH #285).
- **`HyperAnimation.drawn_extent(frames=None)`** returns the union bounding
  box of everything drawn over the clip, in figure fractions (GH #285).
- **Multi-panel static plots.** `hyp.plot([...], panels=True | 'auto' | ncols
  | (nrows, ncols), title=[...])` draws one panel per dataset in one figure
  with a single shared pipeline fit (so panels are comparable), hides spare
  axes and lays them out; `reduce=[...]` gives one panel per reducer;
  `hyp.subplots(nrows, ncols, ndims=3)` builds a flat, 3-D-ready axes grid for
  callers who still want `ax=`. `panel_fit='shared'` (default) fits the
  pipeline once across all datasets; `'independent'` fits it per panel.
  Per-panel axis labels come from DataFrame columns as in the single-axes
  call. On plotly the panels are `make_subplots` scenes and
  `return_model=True` returns the same bundle as matplotlib, which adds
  `axes`, `panels`, `panel_models` and, for a shared fit, the one fitted
  `pipeline` (GH #285).
- **Title styling on every frame.** `title_kwargs=dict(size=, weight=,
  family=, color=, y=)` is applied by hypertools' own title updater,
  including per-segment `title=` lists; `title_color=` takes one colour per
  segment or a callable; `title_wrap=N` hard-wraps titles (`\n` on
  matplotlib, `<br>` on plotly). A resolved `font=` now reaches per-segment
  animation titles too (previously only rcParams could size them) (GH #285).
- **Per-dataset `hue=` and `labels=`.** `hue=` accepts one scalar per dataset
  (broadcast over its rows); `labels=` accepts one entry per dataset, placed
  by `label_anchor='first' | 'center' | 'last' | int` (GH #285).
- **Legends under matrix / mixture hue.** A matrix-valued `hue=` (or
  `hue_mode='mixture'`) now draws a legend with one swatch per hue column
  instead of dropping it; `legend_kwargs=` reaches `ax.legend` and
  `legend_colors=` recolours or replaces the entries (GH #285).
- **The resolved colour scale is exposed.** `return_model=True` bundles
  `['colors']` (kind, palette, cmap, norm, vmin, vmax, per-group colours,
  labels) and `HyperAnimation.colors` carries the same object, so a companion
  panel can reuse the library's mapping (GH #285).
- **Synthetic datasets from `hyp.load`.** `hyp.load('random_walk' | 'helix' |
  'lorenz' | 'blobs' | 'moons' | 'swiss_roll' | 's_curve', random_state=...,
  n_datasets=...)` generates seeded example data (scikit-learn kwargs pass
  through; `n_datasets > 1` returns a list), replacing the random-walk,
  helix and blob generators every tutorial wrote by hand. Any keyword
  `hyp.load` does not use itself is collected into `**source_kwargs` and
  handed to the synthetic or web resolver that matches the name; with any
  other kind of source, already-loaded data included, it raises `TypeError`
  that quotes the keyword instead of dropping it (GH #285).
- **Web sources.** `hyp.load('wikipedia:<Title>')` (plain-text extract;
  `'A|B'` returns a list), `hyp.load('yahoo:<TICKER>', start=, end=,
  interval=)` (daily OHLCV through explicit epoch bounds) and
  `hyp.load('sec:<TICKER>', concept=)` (XBRL company facts) (GH #285).
- **URL download cache.** `hyp.load(url, cache=True)` keeps an atomic on-disk
  copy under `~/hypertools_data/urls` (`HYPERTOOLS_URL_CACHE` overrides) and
  `offline=True` reads only the cache, raising `HypertoolsOfflineError`
  naming the path when a file is missing (GH #285).
- **`hyp.text_windows`, `hyp.damage`, `hyp.stack`.** Sliding word / sentence /
  character windows over one or many documents (with the min-windows guard
  and `max_chars` truncation the examples wrote by hand); reproducible NaN
  knock-out of scattered cells and/or whole rows on arrays and DataFrames
  (returns the damaged copy and, optionally, the mask); and a builder for the
  column-hierarchical DataFrame `plot`/`predict` read from nested dicts or
  lists, with optional `aggregate=` group traces (GH #285).
- **Several forecasters in one call, backtests, and imputer scoring.**
  `hyp.predict(x, model=[...])` returns `{name: forecast}`;
  `hyp.predict(x, model=[...], holdout=k)` fits on the head and scores the
  held-out tail against every model plus an always-present naive last-value
  baseline, with `scores.attrs['best']` and `['beats_baseline']`;
  `metrics=` picks which of MAE / RMSE / MAPE to report (all three by
  default; the first one ranks `best`), `per_column=True` gives one row per
  model and column, and `return_forecasts=True` also returns the forecasts
  that were scored. `hyp.impute(x, model=[...], truth=full)` scores imputers
  on the damaged cells only, with the same `metrics=` and `per_column=`, a
  column-mean baseline and an `unscored` column for rows a model left NaN;
  `return_imputed=True` also returns the scored imputations, the baseline and
  the truth (GH #285).
- **`Smooth(center=False, min_periods=)` and a `Delay` manipulator.** A
  trailing (causal) boxcar identical to `pandas.rolling(...).mean()`, and a
  Takens time-delay embedding (`hyp.manip(x, model='Delay', tau=, dims=)`)
  (GH #285).
- **Alignment quality score.** `hyp.align(..., return_score=True)` returns the
  dispersion (or `score_metric='isc'`) before and after alignment;
  `hypertools.align.score.alignment_score` is the standalone form (GH #285).
- **`palette=` takes a `{category: color}` dict and per-dataset palette
  specs.** Unnamed categories keep their default-palette colour and unknown
  keys raise; a list with one palette spec per dataset (`['image:a.png',
  'image:b.png', 'viridis']`) colours each dataset from its own palette;
  `image_palette()` gains `max_luminance=`/`min_luminance=`, also as
  `palette='image:p.png?max_luminance=0.6'` (GH #285).
- **A synthetic LSL outlet.** `hypertools.io.lsl.synthetic_outlet(name,
  n_channels=6, rate=100.0)` runs a real `pylsl.StreamOutlet` on a background
  thread for demos and tests (GH #285).
- **Bundled Noto Sans Bold.** `fontweight='bold'` now renders a real bold face
  (SIL OFL 1.1, same Noto Sans 2.008 source as the Regular face) instead of
  silently falling back to Regular (GH #285).
- **`hyp.plot(x, pipeline=p)` accepts hand-built pipelines.** A `hyp.Pipeline`
  whose steps are raw scikit-learn estimators (fitted or not) or a bare
  fitted stage object replays through `plot`: an unfitted pipeline is fit on
  `x`, raw steps are applied to each dataset in turn, and a step that yields
  labels rather than coordinates raises an error pointing at `hue=`.
  `Pipeline.is_fitted` is `True` for a pipeline assembled from already-fitted
  steps. Previously such pipelines raised `NotFittedError` or an sklearn
  dimension error from inside the replay.
- **`hyp.load` passes already-loaded data through.** A DataFrame, a numpy
  array, or a list/tuple of those comes back unchanged (with the same
  `reduce`/`ndims`/`align`/`normalize` post-processing loaded data gets),
  so one `hyp.load` call can be the entry point over mixed names and
  in-memory data; other types still raise `TypeError`. Previously any
  non-string raised.
- **Input datatype handling defers to datawrangler.** The shared coercion
  layer (`format_data`, `get_type`/`get_dtype`, `as_dataframe`, the
  `predict`/`impute` normalisation, `io.streaming.is_stream`) classifies
  inputs with `dw.zoo` predicates and converts with `dw.wrangle` instead of
  its own `isinstance` ladders, so polars DataFrames, LazyFrames and Series
  are accepted wherever pandas is -- `plot`, `reduce`, `align`, `cluster`,
  `normalize`, `manip`, `predict`, `impute`, `analyze`, `describe` -- with
  identical results (polars nulls become missing data), and whatever
  datawrangler adds later comes for free. The same holds throughout:
  `plot`'s `hue=`, `labels=`, `truth=`, matrix `palette=`, `panels=` and
  its axis/legend labels; `manip` (and the Manipulator classes and
  `Pipeline` steps directly), `align`, `stack`, `damage`, `apply_model`,
  the fitted `Normalizer`, `save`/`load`, `text2mat` and the impute
  backtest's `truth=`/`mask=` read any frame backend datawrangler
  recognises through the shared predicates; a one-column DataFrame of
  labels as `hue=` no longer raises `IndexError`, and `text2mat` accepts a
  Series of documents. A static test (`tests/test_datatype_gate.py`) keeps
  hand-rolled pandas/numpy type checks out of the library.
  `tests/test_polars_inputs.py`, `tests/test_polars_inputs_wave1.py`.
- **Palettes from images read as gradients, and a data matrix is a
  palette.** Colors extracted from an image are put in a deterministic
  order -- by value, dark to bright -- when the image is used as a plot
  palette (`palette_sort=` or a spec's `?sort=` picks `'value'`, `'hue'`,
  `'lightness'`, `'columns'` or `'original'`; `image_palette()` itself and
  a per-dataset image's lead color keep the salience order). A t x k data
  matrix (array, nested list or DataFrame) passed as `palette=`,
  `forecast_palette=` or a per-dataset entry is reduced to three dimensions
  with `hyp.reduce` (`palette_reduce=`, default 'PCA', with
  `palette_manip=`/`palette_normalize=`/`palette_align=` passed through),
  each reduced column is scaled to [0, 1] as an RGB channel, the rows are
  sorted (default along the first component) and the result is a colormap
  resampled by interpolation to however many colors the plot needs
  (`hypertools.plot.colors.matrix_palette`, `sort_colors`,
  `MatrixColormap`). A 2-D array with 3 or 4 columns and every value in
  [0, 1] stays a list of colors.
- **Optional extras install themselves on demand.** The first call that
  needs plotly, kaleido, HF text embeddings, skaters (`Laplace`),
  chronos-forecasting (`Chronos`), torch (autoencoder reducers), gensim,
  kagglehub, pylsl, scikit-image (3-D density iso-surfaces) or openpyxl
  installs that extra's requirements into the running interpreter, prints a
  one-line notice, and carries on; hypertools itself is never reinstalled.
  The requirement strings are read from the installed package metadata, so
  `pyproject.toml` stays the single declaration of every extra
  (`hypertools._shared.lazy_import`). Static image export with the plotly
  backend provisions kaleido's Chrome the same way, plus the four system
  libraries a fresh Colab/Kaggle image lacks. `hyp.set_autoinstall(False)`
  turns installation off, for the session or for one block as a context
  manager (the same two forms as `set_interactive_backend`): a missing
  extra then raises `ImportError` naming the manual `pip install
  "hypertools[<extra>]"` command, as before. The environment variable
  `HYPERTOOLS_AUTO_INSTALL=0` sets the starting value for images built
  ahead of time.

### Changed / validation

- **`hyp.load('wiki')` and `hyp.load('nips')` return a flat list of strings**
  (one document per entry), like `hyp.load('sotus')`; previously a list
  holding one `(n, 1)` object array. Update any
  `[str(p) for p in x[0].ravel()]` call site to use the list directly. The
  documents are byte-identical to the pre-1.0 originals (GH #285).
- **Non-streaming Hugging Face loads decode `ClassLabel` columns** to their
  string names (`decode_labels=False` keeps the integers), matching a
  streaming load (GH #285).
- **The Hugging Face text and dataset paths set `HF_HUB_DISABLE_PROGRESS_BARS`,
  `HF_HUB_VERBOSITY` and `TOKENIZERS_PARALLELISM`** (via `setdefault`, so an
  explicit setting wins) before importing the libraries that read them, so
  no notebook needs the env-var preamble (GH #285).

These turn previously-accepted input into rejected input. Each was
previously ambiguous or silently lossy.

- **A `pipeline=` that already carries a column-hierarchy record is checked
  against the frame being plotted**, while the leaves still have labels.
  `hyp.plot` hands the pipeline bare arrays, and a list is positional by
  contract, so the fit-time feature NAMES were never consulted during
  plotting: a frame of the same width but different measurements plotted
  happily against a pipeline fit on something else, and only
  `bundle['pipeline'].transform(that_same_frame)` noticed -- contradicting
  the round-trip `return_model=` documents. Under `'name'` correspondence a
  mismatch now raises at the `plot()` call, naming the missing and
  unexpected features; the leaves are also restored to FIT-time order, since
  the fitted steps are positional and the frame's own order otherwise
  produced silently wrong coordinates. `feature_correspondence='position'`
  is unaffected -- opting out of nominal matching is what it is for.
- **Frames carrying a hierarchy on BOTH axes are now rejected** with a clear
  error (`x has both a row and a column MultiIndex ...`). Before 1.1 such a
  frame followed the row path and its column hierarchy was silently ignored;
  1.1 declines to guess which one takes precedence.

- **A COLUMN-hierarchical DataFrame nested inside a list is now rejected**
  by `hyp.plot`; before 1.1 it was flattened to a single line, silently.
  `hyp.predict` rejects a hierarchical frame in a list on **either** axis,
  where it previously raised `TypeError: cannot perform __sub__ with this
  index type: MultiIndex` from deep inside pandas (row axis) or forecast the
  flattened frame (column axis). **This is deliberately asymmetric:** a
  ROW-hierarchical frame inside a list passed to `hyp.plot` keeps its
  documented warn-and-flatten behaviour, unchanged in 1.1. Hierarchy
  expansion is defined for a bare frame only.

- **`hyp.predict` now rejects a column hierarchy whose groups do not name
  the same features.** Grouping the frame gives `hyp.predict` the plot
  path's NAME-based feature correspondence, refusals included: groups
  carrying different -- or differently many -- innermost labels raise an
  error naming the missing and unexpected features. Before 1.1 there was no
  grouping to disagree with, so such a frame was flattened into one wide
  series and forecast (measured on columns `[('Mkt', 'Tech', 'ret'),
  ('Mkt', 'Tech', 'vol'), ('Mkt', 'Energy', 'ret'), ('Mkt', 'Energy',
  'flow')]`: a single `(2, 4)` forecast came back). Groups that share their
  labels in a **different order** are still accepted, and come back
  permuted into the FIRST group's feature order, so values travel with
  their labels rather than with their column positions (measured: permuted
  and unpermuted frames give element-wise equal forecasts).

- **`hyp.predict` now rejects a time-like index with duplicate entries --
  including on FLAT input.** This one is not hierarchy-specific: the check
  lives in `resolve_t`, which runs for every input, so a plain
  `DataFrame`/`Series` on a `DatetimeIndex`, `TimedeltaIndex` or
  `PeriodIndex` with repeated stamps now raises `ValueError: the dataset
  index has N duplicated entries ... the forecast horizon is ill-defined`.
  Before 1.1 it forecast, using a step inferred from the surviving non-zero
  gaps (measured on a 5-row frame with one repeated day: `(1, 3)` returned).
  Several observations sharing one position on the time axis make the
  horizon undefined; aggregate the repeats
  (`df.groupby(level=-1).mean()`) or give them distinct times.
  **Non-time indexes are unaffected:** a stacked `pd.concat([run_a, run_b])`
  panel whose index runs `0..n-1` twice still forecasts.

- **`predict=` with a MultiIndex frame no longer raises blanketly.** It
  previously raised `ValueError: predict= is not supported with MultiIndex
  expansion in this release` for every hierarchy; it now forecasts every
  plotted trajectory. A hierarchy whose traces are shorter than 2 rows still
  raises -- on either axis, since a forecast needs at least two observations
  -- but the message now names the offending trace and its row count, and
  explains the cause: a row-count-changing analysis stage, the
  one-trace-per-index-tuple rule for a row hierarchy, or a
  single-observation input for a column one.

- **Hierarchy groups whose label is missing (NaN) are no longer dropped, and
  a missing label is ONE group.** Grouping uses `dropna=False`, so a group
  with a NaN level label is kept and drawn rather than silently
  disappearing. Because `nan != nan` and pandas mints a separate NaN object
  per group key, keeping them was not enough on its own: a two-sector frame
  with a missing Market label produced two duplicate means and two `'nan'`
  legend entries. Labels are now canonicalised NA-aware (`np.nan`, `None`
  and `pd.NA` all normalise to one sentinel) for prefix grouping, top-level
  uniqueness and style lookup, on both axes. The original label values are
  preserved in the returned keys and in the legend; the sentinel is never
  user-visible.

- **`forecast_hue=`, `forecast_cluster=`, `forecast_n_clusters=`,
  `forecast_palette=` and `forecast_fmt=` count FINAL TRACES, not input
  datasets.** They were unreachable with a hierarchy until `predict=` was;
  now that they are, their unit is every plotted trajectory -- leaves AND
  derived means -- so a three-sector frame needs four values, not three.
  The length errors count forecasts and say why the count exceeds the number
  of datasets. Nothing changes for flat input, where the two counts
  coincide.

- **`ax=` is rejected together with `animate=`.** An animated plot owns its
  own figure, so the axes passed in were left empty and the animation was
  drawn on a figure of its own. The error says to drop `ax=` and style the
  returned animation's `.figure` instead, and that several panels in one
  animation are laid out in the DATA (translate each group into its own
  region of one shared frame) and drawn with a single call.
- **`ax=` under the plotly backend draws into a plotly Figure; a
  matplotlib Axes is refused.** `ax=` names the surface to draw into. With
  the plotly backend that is the `plotly.graph_objects.Figure` an earlier
  `hyp.plot` returned: `fig = hyp.plot(A); hyp.plot(B, ax=fig)` appends
  B's traces (data, legend and colorbar entries) to `fig` and returns it,
  the caller's layout untouched; `animate=` is refused with it, as with a
  matplotlib Axes. A matplotlib Axes under plotly now raises `ValueError`
  before any analysis runs (before 1.1 the plotly backend built its own
  Figure and left the axes you passed empty: on Colab, where
  `backend='auto'` resolves to plotly, a two-panel before/after layout
  showed two empty 3-D cubes; measured 2026-09-04). A plotly Figure passed
  to a matplotlib draw is a `TypeError`. Fix for the refused case: pass the
  plotly Figure, drop `ax=`, or draw that call with `backend='matplotlib'`
  / `with hyp.set_interactive_backend('matplotlib'):`.

### Bug fixes

- **NumPy 2-compatible optional dependency floors.** The `gensim` and
  `density3d` extras require gensim>=4.4.0 and scikit-image>=0.23.2,
  respectively; dev/docs requirements match. Earlier advertised minimums
  predate upstream NumPy 2 support.
- **Backtest model ownership.** Forecast backtests fit an independent copy
  of an unfitted model instance for each dataset, instead of reusing the
  first dataset's learned parameters on subsequent datasets. Forecast and
  imputation scoring leave caller-owned instances unchanged and reject
  already fitted instances, which may have seen the held-out truth (GH #285).
- **Concurrent URL caching.** Threads downloading the same URL use distinct
  temporary files, preventing `FileNotFoundError` during atomic replacement
  (GH #285).
- **Delay column collisions.** `Delay` rejects duplicate column labels and
  distinct labels with identical string representations, instead of silently
  overwriting embedded features (GH #285).
- **Bundled font precedence.** The bundled Noto Sans faces take precedence
  over same-family system fonts, keeping rendering consistent across machines
  with an additional Noto Sans installation (GH #285).
- **Release documentation.** Plotting and scoring features shipped in 1.1
  are identified as 1.1 in their API documentation, correcting leftover 1.2
  labels. Installation guidance distinguishes Kalman imputation from
  Kalman/ARIMA forecasting. The forecast example and its executed tutorial
  use native URL caching; the browser verifier checks versioned notebook
  links, rendered install cells, and real Plotly frame transitions without
  requiring autoplay (GH #284, GH #285).

Each of these was found while building the above, and each affects FLAT
input too.

- **Video exports were written at a fixed 1800 kbit/s.** `save_path=` and
  `HyperAnimation.save()` handed ffmpeg `bitrate=1800` for every `.mp4`
  (and the streaming recorder did the same), so a file's size followed its
  duration and nothing else: a two-minute clip was 27 MB at 1400 x 700 and
  26 MB at 980 x 490, while a large figure was starved and a small one
  over-spent. Video is now a quality-targeted encode (x264 CRF 23, its own
  default; `hypertools.plot.animate.VIDEO_CRF`), so the size follows the
  content: the same 5-second test clip went from 0.96 MB to 0.38 MB with
  no visible difference. Callers wanting a specific bitrate can pass
  `writer=` to `HyperAnimation.save()` as before.
- **The "Animation was deleted without rendering anything" warning could
  still fire from a discarded `HyperAnimation`** when the wrapper died inside
  a reference cycle: the collector may finalize matplotlib's animation before
  the wrapper that silences it. The wrapper now marks the animation as
  draw-started when it is constructed, so the order no longer matters.
- **`HyperAnimation.save()` silently discarded every keyword except
  `fps=`**, so `anim.save('clip.gif', dpi=75)` wrote the GIF at the
  figure's own dpi and nothing said so (a 13 x 9 inch figure came out as a
  10 MB file). `dpi=` is now forwarded to the raster and video writers,
  as `matplotlib.animation.Animation.save` takes it, and any other
  keyword raises `TypeError` naming it. Passing `writer=` still delegates
  to matplotlib with every keyword, as before.
- **Closing an animated figure under matplotlib's notebook backend raised
  `AttributeError: 'NoneType' object has no attribute 'remove_callback'`.**
  `nbAgg` (the backend hypertools selects in Colab and classic Jupyter)
  processes a figure's close event twice, once from the manager's
  `destroy()` and once more from its comm-close handler, so matplotlib's
  `Animation._stop` ran twice and the second call found the timer already
  cleared (matplotlib 3.10.8 and 3.11.1, with or without hypertools). On
  Colab every displayed animation made the next static-plot cell fail in
  IPython's end-of-cell `plt.close('all')`, and `show=False` animated plots
  failed inside `plot()` at its own `plt.close(fig)`. Animations are now a
  `FuncAnimation` subclass (`hypertools.plot.animate.HyperFuncAnimation`)
  whose `_stop` ignores the repeat call; `isinstance(anim.animation,
  matplotlib.animation.FuncAnimation)` still holds.
- **A plotly `save_path=` to a raster/PDF format on a machine without
  Chrome failed with kaleido's bare `RuntimeError`.** kaleido 1.x renders
  through a headless Chrome, which a fresh Colab or Kaggle kernel does not
  have. The failure is now a `HypertoolsIOError` naming the file, the
  cause, and the ways out: `import plotly.io as pio; pio.get_chrome()`
  (about 150 MB) plus, on Colab and Kaggle, the four system libraries the
  downloaded Chrome needs (`apt-get install -y libatk1.0-0
  libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1`; measured 2026-09-04),
  installing Chrome, or saving with `backend='matplotlib'`.
- **Under the plotly backend, a figure kept in a variable was not displayed
  in a notebook.** `fig = hyp.plot(x)` drew nothing (on Colab, where
  `backend='auto'` resolves to plotly, 29 of the feature tour's plot cells
  were blank; measured 2026-09-04): the backend never called `fig.show()`
  inside IPython, because doing so drew a figure that was also the cell's
  last expression twice, and drew it mid-cell, ahead of the matplotlib
  figures flushed at the end. `plot()` now queues the figure for a one-shot
  IPython `post_execute` callback (registered after matplotlib-inline's
  flush, so it runs after it) that skips any figure the rich-display hook
  already showed. The returned figure is still a
  `plotly.graph_objects.Figure` (a subclass). Both usages draw exactly once,
  in cell order.
- **Inside IPython, a matplotlib backend that cannot be switched to raised
  the GUI toolkit's own error instead of `HypertoolsBackendError`.**
  `set_interactive_backend` switches via the `%matplotlib` magic in a
  notebook, and the magic imports the toolkit itself, so `'TkAgg'` without
  `_tkinter` or `'GTK3Agg'` without `gi` escaped as a raw
  `ModuleNotFoundError` (and a missing display as `TclError`), while a
  plain script already raised the documented `HypertoolsBackendError`.
  The notebook path now raises `HypertoolsBackendError` too, with the
  toolkit's error chained as the cause.
- **`alpha=` was ignored by `animate='morph'`.** The value landed on the
  per-dataset line artists, which a morph keeps hidden, and never on the one
  travelling point cloud that is drawn (plotly dropped it the same way), so
  fading the cloud meant reaching into the figure with `set_alpha` after the
  call. The cloud now takes the alpha on the same hold/transition schedule as
  its colour: a hold draws the held dataset's alpha, a transition eases from
  the departing dataset's alpha to the arriving one's, and a scalar `alpha=`
  is constant throughout. Plots that pass no `alpha=` are unchanged.
- **Drawing into a caller-supplied `ax=` warned `Glyph 8722 (MINUS SIGN)
  missing from font(s) Noto Sans` for every negative tick.** The bundled
  Noto Sans has no U+2212. hypertools' own axes carry the whole font stack,
  so matplotlib's per-glyph fallback reaches DejaVu Sans, but axes created
  outside hypertools' rc context keep the `sans-serif` alias, which resolves
  to one font with no fallback. hypertools now formats negatives with an
  ASCII minus (`axes.unicode_minus = False`) while it draws, and gives a
  caller-supplied axes' tick labels the same font list its own axes use, so
  a label formatted before the call (measured again by a later panel's
  layout) falls back to DejaVu for the glyph.
- **An LSL stream left open logged a liblsl error at exit.** `hyp.io.lsl_stream()`
  only `close_stream()`ed its `pylsl.StreamInlet`, and only when the generator
  was closed, so a notebook or script that just moved on saw `ERR| Stream
  transmission broke off` at teardown. The stream is now an `LSLStream` that
  destroys the inlet on `close()`, on leaving a `with` block, on garbage
  collection, on the silent-source abort, and at interpreter exit.

- **PPCA imputation warned `divide by zero encountered in log` on data with a
  few dozen or more features** (`hyp.impute(model='PPCA')` and the NaN fill
  `hyp.plot` applies at format time). The EM objective computed
  `log(det(Sx))`, which underflowed to `log(0)` and then fell back to a
  sign-flipped value; it now uses `slogdet`, identical to rounding wherever
  the old value was representable.
- **`labels=` annotations were drawn on pyplot's current axes, not the `ax=`
  passed in,** so panel labels stacked on one axes; and a colour-list
  colorbar (`color=['red', ...]`) raised `IndexError` (GH #285).
- **`hyp.plot(docs, vectorizer='all-MiniLM-L6-v2')` crashed with the default
  `semantic=`/`corpus=`.** The gensim-only auto-skip of the topic-model stage
  did not cover Hugging Face vectorizers, so the call (the one the optional
  dependencies guide advertises) embedded the whole hosted `'wiki'` corpus
  (3,136 documents, ~13 s) and then died inside sklearn with "Negative values
  in data passed to LatentDirichletAllocation.fit". Pretrained embedding
  vectorizers now resolve the default semantic stage to none silently, never
  load or embed a corpus, and an explicit `semantic='NMF'` raises a clear
  hypertools `ValueError` before any corpus work. Explicit `semantic=None,
  corpus=None` and the CountVectorizer defaults are unchanged.
- **Every continuous-hue matplotlib plot rendered fully opaque**, whatever
  `alpha=` was set to. `_apply_multicolor_lines` never read alpha from its
  per-trace kwargs, and the artists carrying the alpha are exactly the
  `Line2D`s it removes and replaces with a colour-graded collection.

- **The `return_model=True` pipeline could not be re-applied to a
  column-hierarchical frame.** Its steps are fit on the frame's GROUPS, each
  as wide as one group, so `bundle['pipeline'].transform(df)` failed inside
  scikit-learn (`X has 20 features, but IncrementalPCA is expecting 5
  features as input`) -- and, when the reduce stage was a no-op because every
  group already had `<= ndims` columns, it silently returned the UNGROUPED,
  unreduced frame. The pipeline now records the grouping (`Pipeline(...,
  input_hierarchy=)`) and reproduces it, returning one array per group, so
  the round trip that already worked for a flat frame and for a list of
  arrays works here on the same terms. Features are matched to the fitted
  steps BY NAME, like they are across groups, so reordering the innermost
  labels is harmless; a frame naming different measurements, and a flattened
  frame, are refused by a hypertools error naming the cause instead of being
  passed through.

- **plotly discarded the per-trace alpha under a continuous `hue=`** for the
  same figures, from the other direction: the colour serializer drops the
  4th channel and nothing set the trace `opacity`, so a hue plot that
  matplotlib drew at `alpha=0.7` rendered fully opaque on plotly. Line
  colours now carry the alpha; **marker** colours deliberately do not,
  because matplotlib's per-point marker colours carry none either, and
  parity is stated against matplotlib.

- **With `ndims=1`, matplotlib drew the `predict=` overlay at x = 0..t**
  instead of continuing the observed series: the overlay was plotted with no
  x, so it defaulted to `0..len(forecast)-1` and painted every forecast back
  over the START of the plot. Measured: forecast x `0..3` while its observed
  line ran `0..59`. The forecast VALUES were right, which is how it
  survived -- every other forecast test reads 3-D coordinates. Both backends
  now draw the continuation.

- **A marker-only `fmt` (e.g. `'o'`) with `hue=` and `predict=` drew the
  forecast in a different colour on each backend.** The final-observed-hue
  anchor was applied on matplotlib's line path only, and on plotly for every
  `fmt`. It is now on both paths on both backends.

- **plotly's 1-D marker branch drew a continuous hue in one flat colour.**
  The 2-D and 3-D branches already passed the per-point colour array
  through; the 1-D one fell back to the single trace colour, so all points
  came out identical while matplotlib scattered them per point.

- **A hierarchy silently discarded `legend=[...]`, and `legend=False`.**
  The `MultiIndex` branch overwrote `legend` with the hierarchy's own labels
  unconditionally, so a caller's list vanished without a word (while every
  SIBLING kwarg the hierarchy overrides -- `color`/`colors`, `linewidth`,
  `alpha` -- warns) and an explicit opt-out still drew a legend. `legend=` is
  now HONOURED under a hierarchy: a list **renames the top-level groups**
  (one entry per unique top-level index value, in first-appearance order --
  the same convention `linestyle=` already used; any other length raises
  `ValueError` naming both counts), and `legend=False` suppresses the
  automatic legend. `legend=True`/omitted still labels by index value.
  `legend=False` suppresses the LEGEND only: `colorbar=True` still names one
  segment per top-level group, since the colorbar is the colour key for the
  drawn groups rather than a legend.

- **`names=` ALONE raised "pass dataset names via names= OR a legend= list,
  not both"** on a hierarchy -- factually false, since the overwrite above
  had already put the hierarchy's labels into `legend` before the conflict
  check read it. The conflict now tracks what the CALLER passed. `names=`
  itself (one name per INPUT dataset) does not apply to a hierarchy -- one
  frame is drawn as leaves plus derived per-level means -- so it raises the
  same shape of instructive `ValueError` the categorical-`hue` regrouping
  guard raises, pointing at `legend=[...]`. This also closes a narrow path
  (column hierarchy + continuous `hue=`) where `names=` used to slip through
  and label leaves and means with per-dataset names.

- **matplotlib's `'_nolegend_'` sentinel leaked into plotly trace names.**
  It is matplotlib's convention for "keep this artist out of the legend";
  plotly has no such convention, so every hierarchy leaf (and every unnamed
  hue group, forecast and trail) was NAMED `_nolegend_` -- rendered in hover
  labels and written into exported HTML, where a plain list of arrays leaves
  `name=None`. Any leading-underscore label now becomes `name=None`; which
  traces appear in the legend is unchanged.

- **plotly and matplotlib serialized colours differently.** plotly's two
  colour helpers disagreed -- one truncated each channel where the other
  rounded -- so the same colour came out `rgb(219,95,87)` on matplotlib and
  `rgb(219,94,86)` on plotly, and an anchored forecast could not equal the
  per-point colour it was copied from. Both round now.

- **`legend=` as an ndarray/Series/Index mislabelled every trace.** All
  three are accepted label containers, but the per-trace length check and
  the label assignment tested for `list`/`tuple` only, so the whole
  container was handed to matplotlib as EACH artist's label -- two traces
  both named `['a' 'b']`, plus a matplotlib "Passing label as a length 2
  sequence" warning -- while the hierarchy path handled the same containers
  correctly. A `tuple` labelled the traces but missed the colorbar's
  narrower `list` test, so `legend=('A', 'B'), colorbar=True` drew a
  colorbar reading `1`, `2`. Every accepted container is now normalised to a
  list where it is type-checked, so one rule covers them all; a 0-d array
  (`np.array('a')`) counts as ONE label, exactly as `legend='a'` does.

- **A caller-supplied `pipeline=` came back from a hierarchical plot unable
  to re-apply.** `hyp.plot(df, pipeline=p, return_model=True)` hands `p`
  itself back in the bundle, but only the pipeline `plot()` builds for
  itself recorded the column grouping -- so `bundle['pipeline'].transform(df)`
  still raised the pre-1.1.0 scikit-learn error (`X has 15 features, but
  IncrementalPCA is expecting 5 features as input`) that `return_model`'s
  documentation says it no longer raises. The grouping is now recorded on
  the passed-in pipeline too (in place, on the same object the bundle
  returns), unless it already carries one of its own.

### Fixed during the release review

Found by the pre-publication review of the 1.1.0 draft against 1.0.0.
Because 1.1.0 had not been published, they ship in it.

- **No leftover "install the extra first" instructions.** An audit of every
  optional-dependency site (2026-09-07) confirmed each one goes through the
  on-demand installer, and a new test keeps it so: any library module that
  imports a package an extra provides must call `lazy_import` for it in the
  same file (three stated exemptions). The stale prose went: the `plot()`
  reducer docstring's `pip install "hypertools[torch]"`, the autoencoder and
  gensim gallery examples' "pre-install it with ...", the LSL tutorial's
  inline command, and the `reduce()` torch error now says the on-demand
  install was tried. The API reference gained a *Set autoinstall* entry
  (`set_autoinstall`, the mechanism, and a pointer to the guide). Two
  tutorials (`projectile_kalman`, `streaming_data`) shipped a pip upgrade
  notice with a local interpreter path as the stored output of their Colab
  install cell, from the 1.0.0 run that executed it; the executor now clears
  a skipped install cell's outputs, and a gate forbids any published install
  cell from carrying output.
- **`hyp.load(<built-in>, offline=True)` never downloads.** The hosted
  example datasets (`'spiral'`, `'weights'`, the `*_model` pipelines, ...)
  bypassed `offline`: a cache miss downloaded, and a cached file failing its
  SHA-256 pin was deleted and re-downloaded. Offline now serves only a
  hash-valid copy from `~/hypertools_data` and raises `HypertoolsOfflineError`
  naming the file for a missing or corrupt one, leaving the file in place.
  Also fixed the `hypertools.load` docstring example that failed under
  Sphinx's doctest builder (`NameError: hypertools`), and the docs-clean CI
  job now runs that builder (`HYPERTOOLS_DOCS_PLOT_GALLERY=0` turns the
  gallery off as a real boolean). (Release audit 2026-09-07, findings 1
  and 5.)
- **`hyp.set_autoinstall(False)` reaches the subprocess that renders a
  plotly animation's frames for GIF/PNG/video export.** The parent passes
  its effective setting as `HYPERTOOLS_AUTO_INSTALL` to the worker
  (`lazy_import.subprocess_env`), so with installation off a missing kaleido
  raises the `ImportError` naming the manual command and no pip runs in the
  worker; it previously started a fresh interpreter that installed anyway.
  Tested in a real interpreter without kaleido. (Release audit 2026-09-07,
  finding 2.)
- **`alignment_score` rejects degenerate input with a clear error.** A
  1-D series, a non-numeric array, or NaN/inf values raise `ValueError`
  naming the dataset (they hit numpy's own shape errors or returned a NaN
  score), and `metric='dispersion'` on all-constant datasets raises like
  `'isc'` already did instead of returning NaN with a RuntimeWarning.
- `docs/doc_requirements.txt` carries the same core floors as
  `pyproject.toml` (scikit-learn 1.4.2, pandas 2.2.2, matplotlib 3.9.0).
- **`panels=` partitions every per-dataset and per-forecast argument.** A
  per-dataset `palette=` list, `legend=` list, `alpha=` list,
  `forecast_fmt=`, `forecast_palette=`, a model-major `forecast_hue=` and a
  forecaster fitted on every dataset (`hyp.predict(x, return_model=True)`)
  now reach each panel as its own entry, in both `panel_fit` modes and on
  both backends, matching the single-axes figure; previously they raised
  inside the panel or drew every forecast in the first colour. Shared-fit
  grids also hand back their one fitted pipeline as `bundle['pipeline']`
  and in every `panel_models[i]['pipeline']`, so held-out data can be
  projected without refitting (it was `None`). `tests/test_plot_panels_audit.py`
  (33). (Release audit 2026-09-07, findings 3 and 4.)
- **A list of `{category: color}` dicts works with a regrouping `hue=`.**
  Each dataset naming its own categories (the documented per-dataset dict
  form) was rejected as "2 per-dataset palettes but 4 dataset(s)" once
  `hue=` split the datasets into more runs than dicts; the dicts name
  categories and now resolve by name on both backends.
- **`set_autoinstall` blocks that overlap keep the newest setting in
  force.** Two `with hyp.set_autoinstall(False)` blocks open at once (two
  threads, say) used to switch installation back on inside the later block
  when the earlier one exited, because exit restored a value saved before
  either. A block now removes only its own setting (a lock-guarded scope
  stack); the setting is process-global and documented as such. (Codex
  round 6.)
- **Plotly animation export raises the documented exception type.** With
  kaleido missing and installation off, the export worker's `ImportError`
  (naming the manual command) reaches the caller as an `ImportError`, and a
  missing Chrome as `HypertoolsIOError`, instead of a `RuntimeError` wrapping
  the worker's traceback. (Codex round 6.)
- **`panels=` keeps forecast labels that share a colour.** `forecast_hue=` /
  `forecast_cluster=` with a `forecast_palette=` that gives two labels the
  same colour (`['red', 'red']`, or a palette name that cycles) raised
  `ValueError: palette= supplies N color(s)` inside the panels; each panel
  now receives one palette slot per label, matching the single-axes figure.
  Every per-dataset argument in the panel roster also has a behavioural
  partitioning test. (Codex round 6.)
- **The plotly backend honours the colour letter of a data `fmt=` string**
  (`'r-'`, `['g--', 'b:']`) exactly as matplotlib does, including animated
  plots and `panels=` cells; it drew the palette colour before. And
  `panels=` on 2-column or 1-column data with the default `ndims=` draws
  2-D cells on both backends, instead of raising `Trace type 'scatter' is
  not compatible with subplot type 'scene'` (plotly) or drawing flat
  trajectories inside cubes (matplotlib). `tests/test_plot_review_round6.py`.
- **`set_autoinstall` no longer retains superseded direct calls.** Every
  direct call appended a handle that only a block's exit removed, so a long
  session of direct calls kept every handle alive; a record now holds a weak
  reference and a new setting collapses a discarded one into its value,
  never a handle that is alive and may still enter its block (a
  construct-then-enter race across threads found in Codex round 8), with
  the block semantics unchanged. (Codex rounds 7 and 8.)
- **`panels=` no longer re-clusters each panel.** Every cell replays the
  clustering its probe fitted, so seeded memberships equal the individual
  call's on both backends in every fit mode and no extra clusterer fits run
  (the probe's `random_state` was dropped and each panel re-fit with a
  different seed); `return_model=True` bundles report
  `models['cluster_labels']`. Plotly composition keeps the palette offset
  across a `color=` or categorical `hue=` call. Independent panels mixing
  one- and three-column datasets draw the series as row index against value
  on the 3-D cell's floor instead of crashing.
  `tests/test_plot_review_round8.py` (52). (Codex round 8.)
- The transient-network test guard reads a dropped TLS connection
  (`SSLError` with an EOF cause) as the host's fault and a certificate
  failure as a real error; one hosted ubuntu job had failed the Dropbox
  loader test on such a drop while eleven others loaded the file.
- **`panels=` keeps the joint figure's cluster colours, forecasts narrow
  panels in their own space, and a marker-only hue no longer advances the
  palette.** Shared clustered panels keep the joint cluster-to-colour
  mapping and legend names when a slice lacks a cluster (two panels each
  drew red); a 1-/2-column panel of a mixed-width independent grid
  forecasts, reads `truth=` and reports its bundle on its own analyzed
  rows (the individual call's numbers; the display padding used to feed
  the forecaster) and only its drawing is lifted into the 3-D cell; a
  categorical `hue=` with a marker-only fmt consumes no palette slot on a
  composed axes/figure/cell, and its group colours beat a fmt colour letter
  on the marker path as on the line path. The live transient-network guard
  now vetoes a TLS certificate failure carried inside a requests
  `SSLError` (which inherits from `ConnectionError`).
  `tests/test_plot_review_round9.py` (57). (Codex round 9.)
- **Palette and coercion follow-ups (Codex rounds 10 and 11).** The
  matrix colormap honours every inherited Colormap operation (integer
  sampling, `resampled()`, `reversed()`, `set_under`/`set_over`, bad and
  NaN entries per element); image palettes interpolate exactly at any
  count, so more than 256 categories still get distinct colours; a polars
  `forecast_hue` Series is partitioned under `panels=` like a pandas one;
  and `manip` lists mixing an unnamed array with named frames keep every
  frame's index (dated or irregular) while lists of named frames pass
  through untouched.
- **Series and mixed lists through the manipulators (Codex round 12).** A
  pandas or polars Series keeps its index and name through the Manipulator
  classes and `hyp.Pipeline` (a `Pipeline([Smooth, Resample])` on an
  irregularly sampled Series resampled at positions 0..n-1 instead of the
  Series' own; `ZScore`/`Normalize`/`Resample` used directly never took a
  Series at all), and a polars Series beside an array in a `hyp.manip`
  list works. A mixed list keeps every frame's own feature names for every
  model: the shared-statistics manipulators (`ZScore`, `Normalize`) match
  columns by position themselves when labels differ (an unnamed array
  beside a named frame, or two frames named differently) and reject
  different widths with a clear message; the independent ones never
  relabel anything. A 1-D array is one column (n observations of one
  feature) for `hyp.manip`, as it already was for `hyp.normalize`,
  `hyp.reduce` and the Manipulator classes -- the funnel used to read it as
  a single row. `MatrixColormap` follows matplotlib's full extreme-colour
  rules (under/over/bad keep the alpha they were set with, `-inf`/`+inf`
  are under/over rather than bad, and an `alpha=` override reaches the
  extremes but not a transparent bad colour). `legend=` accepts a polars
  Series (any series-like) of labels, on both backends.
  `tests/test_review_round12.py` (35).
- **`HypertoolsTrustError` is importable from `hypertools`**, beside
  `HypertoolsOfflineError`, and the API reference documents it and
  `io.synthetic_outlet` under those public names (the source-view backlinks
  for both pointed at anchors that did not exist).
- **`panels=` decides each cell's projection from the analyzed data** (after
  `manip=`/`pipeline=`/`reduce=`), not the raw column count: a Delay-expanded
  2-column dataset reduced to 3 components draws 3-D panels again in the
  shared, independent and reducer-comparison modes on both backends, keeping
  the requested `ndims` and each panel's fitted pipeline with no second fit
  (every mode now fits through one probe and draws from its rows). Composing
  a call into a figure, axes or cell after a `fmt='r-'`, `color=` or `hue=`
  call continues the palette from the slots actually consumed, identically
  on both backends. `tests/test_plot_review_round7.py`. (Codex round 7.)

- **A repeated metric in `metrics=` raises `ValueError` that says which
  metric is repeated.** `hyp.predict(..., holdout=k, metrics=['mae', 'MAE'])`
  and the matching `hyp.impute(..., truth=)` call used to fail with a
  `TypeError` from inside the scores builder.
- **`holdout=True` with `t=0` reports `t` as the problem.** `holdout=True`
  takes its size from `t`, so the error now says that `t` must be at least
  1 row.
- **The "left N scored value(s) missing" warning is attributed to the
  caller's line**, like every other warning `hyp.predict` and `hyp.impute`
  emit, instead of to a line inside the library.
- **`return_score=True` works on ragged input that `hyp.align` trims.** The
  "before" score is computed on the row-trimmed datasets, the same ones the
  aligner sees.
- **`HypertoolsOfflineError` is importable from `hypertools` and
  `hypertools.io`**, so a caller of `hyp.load(..., offline=True)` can catch
  it without reaching into `hypertools.io.sources`.
- **`palette=` colour lists behave as in 1.0.0 again.** A list shorter than
  the dataset count cycles when there is no `hue=`; an empty palette raises
  `ValueError` instead of `StopIteration`; a per-dataset list whose entries
  are `{category: color}` dicts merges them by category name; and any other
  per-dataset form under a categorical `hue=` raises an error carrying the
  real dataset and category counts.
- **NaN in a continuous `hue=` no longer poisons the colour range.** The
  `vmin`/`vmax` of the colour scale and the colorbar are computed over the
  finite values only.
- **Legend and colour details.** `legend_kwargs={'fontsize': ...}` is
  honoured together with `font=`; `bundle['colors']['categories']` contains
  RGB tuples for the blend kind when `legend_colors=` is passed; and a nested
  `hue=` whose sub-list does not match its dataset is identified in the
  error.
- **`dataset_fade=` and `on_frame=` mutations reach the drawn collections
  under a continuous `hue=` on matplotlib.** A fade or a per-frame artist
  change was a silent no-op there.
- **`loop=True` accepts a per-segment `rotations=` list** of the documented
  `2(n+1)-1` length.
- **`companion=` panels and `{index}` titles advance monotonically under
  `order='serial'`** with several datasets, and the `start` in
  `FrameContext.window_bounds` reflects the comet-head window on serial
  reveals.
- **Animation errors say what went wrong.** A bad `companion=` or
  `dataset_fade=` value raises an error that quotes the keyword; an
  `on_frame=` hook that raises during `.save()` propagates its own
  exception; and a `title=` callable that raises no longer leaves an
  "Animation was deleted without rendering anything" warning behind it.
- **`title_wrap=` applies to dynamic titles** (a callable, or a `{index}`
  format) and preserves explicit newlines. plotly draws a `\n` in a title as
  a line break and reserves top margin for every title line at the
  requested size.
- **Labels and titles validate their input.** A nested tuple `labels=`
  annotates like a nested list; `labels='str'`, a bad `label_anchor=`, a
  title list with a non-string entry, a title callable that returns a
  non-string, `title_color=` alongside `title_kwargs={'color': ...}`, a
  static `{index}` title with no index to fill it, and a malformed `{index}`
  format each raise an error saying so.
- **`hyp.load(..., offline=True)` opens no network connection.** URLs skip
  the seaborn dataset-name listing; the listing fetch has a timeout and a
  failure is remembered for the session
  (`hypertools.io.sources.reset_seaborn_names_cache()` retries it); and a
  source that cannot be served from disk raises `HypertoolsOfflineError`.
- **`yahoo:` bars are dated by the exchange-local trading day.** The
  exchange's `gmtoffset` is applied to the bar timestamps; Sydney and Tokyo
  tickers were dated one day early.
- **Synthetic datasets accept more seed types.** Every synthetic dataset
  accepts `random_state=np.random.RandomState(...)`, and the scikit-learn
  backed ones (`blobs`, `moons`, `swiss_roll`, `s_curve`) also accept a
  `Generator`, a `SeedSequence` or a NumPy integer with `n_datasets=1`;
  reusing one `SeedSequence` across calls gives the same data each time.
  `n_datasets=1.5` raises instead of being truncated to 1.
- **`hyp.load(..., streaming=True)` on a source other than a Hugging Face
  dataset raises `ValueError`** instead of returning the whole dataset as if
  the keyword had not been passed.
- **`hyp.text_windows` accepts NumPy integers** for `size=` and `step=`.
- **`hypertools.tools.text2mat` reads a flat list of strings as one
  dataset.** Since 1.0 it returned one `(N, d)` matrix followed by one empty
  `(0, d)` matrix per string. Ragged nested lists work, mixed inputs raise,
  and a dict `semantic=` spec with a gensim vectorizer warns and skips the
  step like the string form does.
- **Warnings raised while formatting input data are attributed to the
  caller's line**: the PPCA missing-data fill and the mixed text-and-numbers
  notice now point at the `hyp.plot`/`hyp.analyze` call that triggered them.
- **`fit()` returns the fitted model** on the manipulator and aligner
  bases (the imputer base already did), so sklearn-style chains such as
  `Smooth().fit(x).transform(y)` and `HyperAlign().fit(xs).transform(ys)`
  work instead of raising `AttributeError` on `None`.
- **`predict='ARIMA'` on an animated plot no longer crashes.** The early
  frames reveal two-row histories, and statsmodels raised an `IndexError`
  on them. Forecasters now carry a `min_history` (ARIMA derives its own
  from its order), `fit` raises a clear `ValueError` for a shorter history,
  and the animated schedule waits until enough rows are revealed. The same
  fix covers `predict=['Kalman', 'ARIMA']` under `animate=`.
- **A datetime-like `t=` works inside `hyp.plot`** (static and animated),
  resolved against each dataset's `DatetimeIndex`, as the docstring said.
- **`predict=` lists and dicts work on row- and column-MultiIndex frames**
  (one forecast per trace per model; the bundle is keyed by model name)
  instead of failing an internal consistency check.
- **`ndims=1` on a dated column-MultiIndex frame** draws dates for every
  leaf, not only the first.
- **`forecast_hue=` with a model collection** is one value per dataset,
  shared across the models; a model-major list is also accepted, and a
  mismatch names both counts.
- **Series-mode `return_model`** returns one `(t, n_columns)` forecast array
  per input dataset in `predict['forecasts']`, the shape `hyp.predict`
  returns.
- **`panels=` fixes.** Works with `predict=` plus `truth=` in both
  `panel_fit` modes; shared mode keeps a DataFrame's dates and column names
  under `ndims=1` and accepts three-column frames; nested `hue=` and
  `labels=` narrow per panel in both modes; `ndims>3` draws 3-D panels on
  both backends; `save_path` accepts `~` and `pathlib.Path` and fails
  before drawing when the directory is missing; plotly panels return the
  same figure wrapper as a single-axes plot and display once per cell.
- **`ndims=1` fixes.** `fmt=` lists are one entry per drawn column; `xlim=`
  on a date axis accepts date strings and datetimes on both backends
  (floats are matplotlib day numbers on both); no `'dataset 1'` y label
  after a reducing `reduce=`; a 3-D `ax=` with `ndims<=2` raises instead
  of drawing a flat 3-D line; a `TimedeltaIndex` is drawn in a readable
  unit with a labelled axis.
- **NaN rows introduced by a trailing `Smooth(center=False)`** are reported
  as the manip stage's doing, with the `min_periods=1` hint, instead of
  the all-features-missing message.
- **Docstrings:** `font=` explains weights (bold resolves to the bundled
  Bold face); `HyperAnimation.drawn_extent` documents its parameters;
  `HyperAnimation.save` lists the supported extensions plainly.
- **A marker-plus-line format string keeps its marker in the legend.** A
  dataset drawn with `'s--'` (or `'o-'`) is split into a smoothed line and
  markers at the raw sample points; the legend handle showed only the
  line. It now shows the marker and the line, on static and animated
  plots, and the line itself still draws no markers.
- **`panels=True` picks its grid from the figure's aspect ratio** and
  prefers a grid with no spare cell: three panels form a row in a
  default or wide figure (they were a 2x2 with a hole, and in a wide
  figure each square 3-D axes shrank to the short cell height), four
  form 2x2, six form 2x3. Explicit `(nrows, ncols)` and column counts
  are unchanged.
- **`return_model=True` no longer fits the pipeline a second time.** The
  bundle's `pipeline` is the one the figure was drawn with (the
  cluster stage, which runs on the reduced scores, is appended as a
  fitted step), so a UMAP or Isomap plot with `return_model=True`, and
  every `panels=` grid built with it, fits once and warns once.
- **Seeded UMAP no longer warns about `n_jobs`.** A `random_state=`
  hypertools injects made umap-learn override `n_jobs` and say so;
  hypertools now passes the `n_jobs=1` umap uses anyway, unless the
  caller chose one.
- **Isomap fits stay quiet about scipy's sparse-matrix efficiency.** The
  dozen `SparseEfficiencyWarning`s scikit-learn's graph completion
  triggers are silenced during the fit; sklearn's own warning about a
  disconnected neighbour graph (the user's `n_neighbors`) still shows.
- **A `truth=` overlay's legend glyph shows its markers.** The truth is
  drawn as a solid line with a marker on every observation, but its
  `'truth'` legend entry was a bare solid line in the trace's own colour,
  identical to the observed trace's entry. The curve now keeps the marker
  (drawing none of its own) so the legend can tell them apart.
- **2-D `density=` layers fade out inside their own grid.** Each KDE grid
  stopped 15% past its own dataset's bounding box, where the density is
  still clearly visible, so a wide, flat cloud's glow was cut off in a
  hard band well inside the frame. The grid now also reaches four kernel
  widths past the data on both backends, where the density has faded to
  nothing, while staying local to its own cloud (so a small cloud beside
  a huge one keeps its resolution).
- **`panels=` on the plotly backend keeps each panel whole.** The plotly
  grid (`plotly.subplots.make_subplots`) received only each panel's
  traces, so 2-D panels lost their unit frame, hidden ticks and
  DataFrame-column axis labels; `legend=True` merged every panel into one
  legend ('1, 1, 1' for three panels, the digit groups listed twice for a
  reducer comparison); and two `colorbar=True` panels drew both colorbars
  on the same spot. Each panel now moves into its cell with its axis
  layout, frame square and `labels=` annotations re-referenced to that
  cell, its own legend beside the cell (plotly's multiple legends), its
  own colorbar (on whichever side `colorbar=` asked for), its `title=` as
  formatted, styled and positioned by the ordinary title path
  (`title_wrap=`, `title_kwargs=`, with the multi-line top margin that
  path computes), and its `font=` materialized on the cell's own text;
  3-D cells back the camera off so the cube stays
  inside a narrow cell; room for the legends/colorbars is reserved
  beside every cell (the default-sized figure is widened by it, an
  explicit `size=` is honoured verbatim).
- **`hyp.subplots(..., backend='plotly')` and `ax=<cell>`.** The
  compose-it-yourself grid (`fig, axes = hyp.subplots(); hyp.plot(d,
  ax=axes[i])`) had no plotly form: `ax=` took a plotly Figure to append
  traces to, but could not target a `make_subplots` cell. `hyp.subplots`
  gained `backend=` and, on plotly, returns the grid figure plus a flat
  array of cells that `hyp.plot(..., ax=cell)` draws into -- the whole
  panel, `title=` included -- returning the grid; several cell calls in
  one notebook cell display the grid once.
- **The slow-forecast-schedule notice no longer fires from timer noise.**
  Its projection drew a slope through the first two timed fits, one row
  apart at 2 and 3 rows -- tens of milliseconds each -- and on a slow CI
  runner projected 10 s for a 30-row schedule that finished in well under
  one. It now fits every timed length by least squares and waits for a fit
  of at least 10 rows before projecting.
- **A fitted forecaster reuses its parameters on a short context.** The
  minimum-history check added for fitting (`Forecaster.min_history`) was
  also applied when an already-fitted model was passed back as `model=`,
  so a fitted `ARIMA(order=(4, 0, 0))` refused two new rows it can
  condition on with its learned parameters. Only the refit path is held
  to the fit floor now.
- **`panels=` colorbars on matplotlib take their room from their own
  panel.** A `colorbar=True` panel grid ran the single-axes
  figure-widening placement once per panel, stacking every colorbar over
  the last panel and leaving `tight_layout` warning about axes it could
  not place; a colorbar drawn into a caller-supplied `ax=` (every panel,
  every `hyp.subplots` cell) now uses matplotlib's own `ax=`-attached
  placement, so each panel keeps its colorbar and the figure its size.
- **Every `predict=` forecast is listed in the legend.** Only a collection
  of models was; `predict='Kalman'` drew its faded continuation with no
  key, and `truth=` then listed `observed` and `truth` beside an unnamed
  dotted line. The single-model form now lists its forecast once, under
  the model's name (the name `hyp.predict(x, model=[spec])` gives it),
  static and animated, on both backends, in the order data, forecasts,
  truth. The entry's glyph wears the forecasts' own style, in their colour
  when they share one and in a neutral gray when one model's forecasts of
  several datasets are drawn in several colours (the first dataset's
  colour used to pose as the model's).
- **A collection of models keeps each dataset's colour and takes a
  linestyle per model.** `predict=['Kalman', 'ARIMA']` coloured every
  forecast by model from a `'husl'` palette whose first colour was the
  first dataset's own, so two datasets under two models were four lines
  in two indistinguishable pairs, and which series a forecast continued
  could not be read at all. Forecasts now inherit their dataset's colour
  (as the single-model form always did) and cycle solid, dashed, dotted,
  dash-dot by model; `forecast_palette=` opts back into one colour per
  model, and `forecast_fmt=` still replaces the cycle.
- **`panels=` on plotly is laid out like the matplotlib grid.** The plotly
  grid used `make_subplots`' default spacing (10-15 % of the figure
  between cells) and full-height cells, and backed the camera off ~1.5x
  further than a narrow cell needed (the constant was the cube's width
  relative to its OWN height rather than the scene's), so three 3-D
  panels sat far apart with small cubes and titles floating well above
  them. 3-D cells are now square and centred (an `Axes3D`'s equal box
  aspect), the gaps are `tight_layout`'s 20 px (40 px between 2-D cells,
  for tick labels), each row reserves what its titles need, and the cube
  fills its cell within a few pixels of the matplotlib panel's.
- **`truth=` on plotly marks every observation, not every vertex.** The
  antialiased truth curve carried a marker on each of its ~900 drawn
  vertices, so it rendered as a thick line; markers now sit on the raw
  rows only, at the size the matplotlib overlay draws them.
- **A caller's axes draw in the palette, and a second call continues
  it.** `hyp.plot(x, ax=ax)` and every matplotlib `panels=` cell drew
  the datasets in the colour cycle their figure was created with
  (matplotlib's default blue/orange) while `return_model`'s `colors` and
  the plotly grid reported the hls palette; the axes now take the
  palette. Drawing a second time into the same axes or plotly figure
  restarted the palette, so two composed walks were both red; the second
  call now continues it from where the first stopped, on both backends.
- **A recoloured forecast keeps its trace's alpha.** `forecast_palette=`,
  `forecast_hue=` and `forecast_cluster=` recoloured the forecasts and
  still halved their alpha, so Set1 forecasts at 0.35 over a hierarchy's
  0.7 leaves could not be found; the colour is what tells them apart, so
  they are drawn at the trace's own alpha. The forecast legend glyph is
  never drawn below 0.8 alpha either (it copied its forecasts' 0.35 and
  vanished).
- **The 2-D frame square clears the data.** Static 2-D plots rescale the
  data into the unit box and drew the frame square AT its edge, so the
  extreme observations sat on the frame line and looked clipped; the
  square now has a 12.5 % margin (the axes stay 10 % beyond it) on both
  backends.
- **A 3-D figure's axis labels are inside its tight bbox.** `Axes3D`
  measures its axes for layout only, dropping the labels, so a
  `bbox_inches='tight'` save -- every notebook's inline render -- cut the
  `zlabel=` off at the right edge; a figure artist now carries the three
  labels' extents into the bbox.
- **Default-size `panels=` figures make room for their legends and
  colorbars.** Three 10-entry legends beside three default-size panels
  shrank the cubes to 1.3 in; the matplotlib figure now widens by the
  same 1.1 in per column the plotly grid reserves, and an explicit
  `size=` is honoured verbatim.
- **`legend_kwargs={'loc': ...}` places the legend there.** A `loc=`
  without a `bbox_to_anchor=` kept hypertools' outside-right anchor, so
  `'upper left'` hung the legend off the right edge; the anchor is
  dropped when a location is named.
- **Plotly legend keys are readable for tiny markers.** A `'.'` marker's
  legend key reproduced the 2 px dot; legends now use plotly's constant
  key size, as a matplotlib legend does.
- **A collection of models under `hue=`/`cluster=` regrouping continues
  the right run.** One dataset split into two runs under two models gives
  two forecasts for two runs, so the ownership resolution -- which only
  ran when the counts differed -- was skipped and forecast i continued
  run i: Kalman took the earlier run's colour, ARIMA the final run's. It
  runs for every collection under regrouping now. The animated modes also
  looked the reveal schedule up by FORECAST index rather than source
  dataset, so two models x regrouping x `forecast_trail=` raised
  `IndexError` on both backends (Codex round 3).
- **A forecaster fitted on several datasets animates.** The animated
  schedule forecasts each dataset's revealed history on its own, which a
  `hyp.predict([a, b], return_model=True)` forecaster refused as a
  dataset-count mismatch; `Forecaster.for_dataset(i)` now binds the view
  the schedule needs (Codex round 3).
- **Plotly honours a colour letter and markers in `forecast_fmt=`.**
  `forecast_fmt='ro:'` drew red dotted forecasts with round markers on
  matplotlib and inherited-colour dotted lines without markers on plotly,
  static and animated, legend keys included (Codex round 3).
- **Plotly animations keep a recoloured forecast's alpha too.** The
  animated branch computed the halved alpha before the recolouring rule
  applied, so `forecast_palette=` forecasts animated at 0.35 while the
  static figure drew them at 0.7 (Codex round 3).
- **A second call into the same plotly grid cell continues the palette**,
  as a second call into the same matplotlib axes does (Codex round 3).
- **Plotly forecast legend keys compare colour, not opacity.** With
  `alpha=[1, .4]` and an all-red forecast palette every key turned gray
  because the RGBA strings differed only in alpha (Codex round 3).
- **`legend_colors=` keeps its contract beside forecasts.** Explicit
  `(label, color)` pairs define the legend outright, so no forecast or
  `truth` entry is added to them (and on plotly the data traces stay out
  of it too); a plain colour list is applied to the FINAL legend, after
  the forecast/truth entries, instead of being refused against the data
  entries alone (Codex round 3).
- **Matplotlib panel legends clear their colorbars.** A panel with
  `legend=True` and `colorbar=True` drew the attached colorbar under the
  outside-right legend (~10 px overlap); the colorbar is padded past the
  legend's measured overhang (Codex round 3).
- **Plotly grid cells keep an explicit legend position and multi-line
  title room.** `legend_kwargs` x/y are translated into the cell instead
  of being replaced by the gutter placement, and a gutter rebuild keeps
  the top margin a multi-line title had reserved (Codex round 3).
- **A `forecast_fmt=` colour letter survives a regrouped animation.**
  Under `hue=`/`cluster=` the animated modes repaint each live forecast
  in its head run's colour unless the colour is pinned, and only
  `forecast_hue=`/`forecast_cluster=`/`forecast_palette=` counted as
  pinning: `forecast_fmt='ro:'` forecasts animated cyan under red legend
  keys on matplotlib, and plotly's per-frame colours halved the alpha a
  recoloured forecast keeps (Codex round 4).
- **Mixture-hue legends list forecasts and `truth`.** A matrix `hue=`
  builds its legend from swatches (clearing `legend=` on the way), and
  the forecast/truth entries were only added when `legend=` was still
  set, so those legends read `1, 2` alone on both backends (Codex round 4).
- **Repeated calls into one axes, figure or grid cell compose.** The
  forecast and `truth=` overlays styled themselves from the FIRST call's
  lines on a reused matplotlib axes (three walks, three red forecasts),
  and the legend accumulated one `truth` per call while losing earlier
  forecast keys; the overlays now take this call's lines and the legend
  is rebuilt by role -- the data entries, one key per model over every
  call's forecasts, one `truth` -- on both backends and in plotly cells
  (Codex round 4).
- **A plotly cell's legend and colorbar from separate calls sit side by
  side**, and a multi-line title rebuilds the rows at once: the grid
  tracks each cell's furniture and title room, sizing the gutter for the
  busiest cell (Codex round 4: a colorbar added after a legend landed on
  it; a three-line title widened the top margin but left the rows 37 px
  apart).
- **Plotly draws a marker-only `forecast_fmt` as markers**, as matplotlib
  does, and its animated collection traces tag their SOURCE dataset in
  ``meta['hyp_dataset']`` rather than the model-major forecast index
  (Codex round 4).
- **A `hyp.subplots(backend='plotly')` grid grows its legend room only
  when a cell asks for it.** The grid reserved a 118 px gutter beside
  every cell up front (it cannot know which cells will draw a legend), so
  a legend-less pair of cells sat left-heavy with cubes three quarters
  the size of the matplotlib pair's. It is now built as tight as
  `panels=` draws it, and the first cell that receives a legend or
  colorbar rebuilds the grid with gutters, moving the cells already
  drawn (their legends, colorbars and titles included).
- **Streaming plots work when plotly is the render backend.** Colab and
  Kaggle select plotly by default, and there every streaming `hyp.plot`
  (a generator, a Hugging Face `IterableDataset`, `hyp.io.lsl_stream()`)
  raised `AttributeError: 'HyperPlotlyFigure' object has no attribute
  'axes'`, as did any stream after `hyp.set_interactive_backend('plotly')`.
  The head plot now always renders with matplotlib, as the streaming
  docstring states. The `streaming_data`, `lsl_streaming` and `io`
  tutorials failed at their streaming cells on Colab because of it. Present
  since 1.0.0. (Fresh-Colab feature tour 2026-09-11, STREAM-01/02/03.)
- **`xlabel=`, `ylabel=` and `zlabel=` join the font-coverage scan.** The
  scan that picks an installed font for characters the default font stack
  lacks read `labels=`, `legend=`, `title=`, `hue=` and the colorbar text,
  but not the axis labels. An axis label in such a script (Javanese on
  stock macOS; CJK on a Linux machine whose CJK font is outside the stack)
  drew as empty boxes, while the same text as a title rendered. Present
  since 1.0.0.

### Documented limitations

- Ragged groups (unequal feature counts per group) are rejected by both
  entry points, by an error naming the missing and unexpected features. That
  error's escape-hatch remedy is spelled for `hyp.plot`, so a `hyp.predict`
  caller has to translate it: group with `group_columns(df,
  feature_correspondence='position')` and forecast the leaves
  (`hyp.predict([leaf.to_numpy() for leaf in leaves], model, t)`, verified).
- Unequal-length row groups are averaged over their overlapping prefix, with
  one aggregated warning.
- Feature correspondence across groups is established by NAME, so groups
  with disjoint innermost labels are refused rather than silently stacked.
  `feature_correspondence='position'` on `group_columns` is the deliberate
  opt-in, and it is not a positional hierarchy mode: passing its arrays to
  `hyp.plot` gives a plain list of datasets -- no per-level means, no
  hierarchy styling (matplotlib's default line width on every trace), and
  `trace_metadata` is `None`. There is no public
  `plot(feature_correspondence=...)` in 1.1, so opting out stays visible at
  the call site.
- The order of the **groups** is not neutralised the way the order of
  features within a group is. Groups become datasets, `reduce=` row-stacks
  every dataset and fits one model on the stack, so group order is row order
  in that stack -- and a reducer whose fit depends on it embeds a
  block-reordered frame differently. On a 40-row frame of 4 sector blocks x
  5 measures, reordering the blocks produced a different embedding under the
  default `IncrementalPCA` (which fits by `partial_fit` over successive
  minibatches) and under `TSNE`, while `PCA`, `TruncatedSVD`,
  `FactorAnalysis`, `Isomap` and `SpectralEmbedding` preserved it up to
  numerical and sign equivalence. No displacement figure is published here:
  it depends on the data, the scikit-learn version, the BLAS build and the
  platform, and a flipped component sign is the same embedding. This is a
  property of the shared reduction space rather than of hierarchies --
  `hyp.plot([A, B, C])` and `hyp.plot([C, B, A])` differ the same way, and
  did before 1.1 -- so it is documented (`hyp.plot`'s `x` entry,
  docs/hierarchy.rst) and pinned by a test rather than worked around. A
  canonical group order would mean inventing a total ordering over
  arbitrary, mixed-type, NA-bearing labels, and would make a labelled
  hierarchy behave differently from the equivalent positional list of
  datasets. Pass `reduce='PCA'` when block order must not matter.
- Continuous `hue=` over a **row** hierarchy is still warned-and-ignored;
  only column hierarchies honour it in 1.1.
- A forecast under a continuous `hue=` takes its source trajectory's **final
  observed hue colour**, in the animated case as well as the static one, on
  both backends. (Animated forecasts briefly wore the per-dataset palette
  colour instead -- the colour of the hidden artist driving the reveal,
  which nothing visible is drawn in, so the forecast appeared to continue a
  colour its trajectory never had and a paused animation disagreed with the
  static plot of the same call.) A **categorical** regrouping is unchanged:
  there the live forecast still takes the colour of the run drawing the
  head, which is what the viewer actually sees.
- Duplicate innermost feature names inside one group are **kept** rather
  than rejected or de-duplicated, and matched across groups by
  `(label, occurrence)`: all such columns are plotted and forecast. Rename
  the innermost level first if you need them distinguishable in a legend.
- `predict=` needs at least 2 rows per plotted trace, on **either** axis.
  Over a **row** hierarchy this is the binding constraint: expansion draws
  one trace per unique full index tuple, so a frame whose innermost index
  level is unique per row cannot be forecast; flatten it
  (`df.reset_index(drop=True)`) or move the grouping to the column axis.
  Over a **column** hierarchy every group keeps all of the frame's rows, so
  it bites only when the frame itself has a single row -- and flattening
  cannot help, so the error does not suggest it.

## 1.0.1 (unreleased)

Small, additive plotting features and fixes. Public APIs are unchanged; two
items under **Changed** below alter how existing figures LOOK.

> 1.0.1 was never published on its own. These changes were developed as a
> patch release and now ship as part of 1.1.0, which is what `pyproject.toml`
> declares; they are kept in their own section because they are separable
> from the hierarchy work above. Because 1.0.1 is not a version anyone can
> install, every guide and docstring that dates one of these behaviours dates
> it to **1.1.0**; this heading is the only place the shipped package names
> the patch line. If you are upgrading from 1.0.0, everything from here down
> to the `## 1.0.0` heading is new to you as well.

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
  carry its `predict=` forecast overlay -- the forecast trace(s) are drawn
  once and rotate with the scene.

- **`predict=` now works with the time-progressing animations too**
  (`animate=True`/`'parallel'`/`'serial'`/`'window'`). The forecast is
  recomputed from the history revealed so far and re-anchored on the last
  revealed observation, so the forecast trace grows with the animation instead
  of standing still. Because the data is static -- all of it known before the
  first frame, merely revealed over time -- every forecast the animation will
  ever draw is computed up front. Two things follow: the whole fan is folded
  into the plot's centre/scale statistics, so it lands inside the cube **by
  construction** and is never clipped or clamped; and each frame is a table
  lookup, so `ani.save()` and `to_jshtml()` replay identically no matter what
  order matplotlib asks for frames in. Fits are memoized per (dataset,
  revealed-count), so a 900-frame animation of a 60-row dataset costs at most
  59 fits rather than 900.

  `animate='morph'` (including the per-dataset morph list form) still raises
  `NotImplementedError`, and now for a stated reason rather than as blanket
  follow-up: a morph interpolates between point CLOUDS, so there is no time
  axis to forecast along.

- **`forecast_trail=`: keep earlier forecasts on screen as a fading fan.** The
  forecast analogue of `chemtrails=`. With `predict=` and a time-progressing
  animation, `forecast_trail=True` retains the last 16 forecasts (an int sets
  the cap), each in its dataset's style, exactly like the live one, at an
  alpha that decays with age from that dataset's live forecast alpha down to
  a floor proportional to it. What it shows is how the
  prediction *changed* as history accumulated -- a forecast that keeps
  revising points somewhere different from one that settles.

  The fan is recomputed from the frame index rather than accumulated in a
  buffer, so it depends only on which frame is being drawn: a saved GIF and an
  interactively-played animation are identical, and frames delivered out of
  order (which `save()` and `to_jshtml()` do) give the same picture. Artists
  are preallocated at setup, since allocating them mid-animation is what makes
  matplotlib animations stutter, and an unwritten slot is hidden with EMPTY
  data rather than zero alpha. Retained forecasts need no extra room in the
  plot box: a retained forecast is just an earlier frame's, and the box
  already contains every forecast the animation will draw.

  Without `predict=` it raises `ValueError` rather than silently doing
  nothing.

- **An animated `predict=` says when it will be slow to start.** A forecast
  animation needs one fit per distinct revealed history length, so its cost
  grows with the DATA, not the frame count: 3 datasets x 60 rows x 900 frames
  is 177 fits (~5 s), while 3 x 500 x 900 is 1497 fits (~330 s) -- a longer
  series has both more distinct histories and a costlier fit each. `plot()`
  now times the first real fit and warns if the projection exceeds
  `slow_warning_seconds=` (default 10; pass `None` to silence), so a long
  wait is expected rather than mysterious. The notice arrives before the
  wait, not after it.

  Deliberately NOT solved by sampling the reveal: striding the schedule would
  render a different animation than the one asked for. The outcome is not
  negotiable, so the time is.

- **`forecast_hue=`, `forecast_cluster=`, `forecast_n_clusters=`,
  `forecast_palette=`, `forecast_fmt=`: style the forecasts separately from
  the data.** Inheritance stays the default -- a forecast is its observed
  trace projected forward at half its alpha -- and each of these replaces
  exactly one aspect of it, so observed and forecast data may differ in
  style, grouping, palette, or any combination.

  **`forecast_cluster=` clusters the forecast ENDPOINTS**, so a forecast's
  colour answers *which of these series are heading to the same place?* --
  a question the observed data cannot answer, which is the point of a
  separate kwarg. It deliberately does not recluster the observed data
  (inheriting that assignment is what the default already gives, so the
  kwarg would be a no-op), nor cluster every predicted point (one forecast
  would change colour along its own short path), nor flatten whole
  trajectories (sensitive to `t`, to sampling and to dimensionality, where
  an endpoint has one stable meaning). Endpoints are taken in the space the
  figure draws, after `reduce=`/`align=`.

  In an **animation** the endpoint groups are resolved once, from the
  full-history forecasts (the ones `return_model=True` returns), and stay
  fixed for every frame -- they are not reclustered as the reveal
  progresses. Cluster labels are arbitrary names for groups, so per-frame
  reclustering would let a forecast change colour whenever a fit nudged its
  endpoint across a boundary, and would repaint a retained
  `forecast_trail=` fan drawn under the old grouping.

  `forecast_hue=` and `forecast_cluster=` are mutually exclusive, mirroring
  `hue=` and `cluster=`. `forecast_n_clusters=` is separate from
  `n_clusters=` on purpose: the observations and the forecast endpoints are
  different point sets. All five require `predict=` and raise `ValueError`
  without it, rather than being silently dropped. `forecast_fmt=` is
  validated with matplotlib's own `fmt=` parser, so it provably accepts
  exactly what `fmt=` accepts and both backends reject the same strings at
  the same moment; `forecast_hue=` rejects a bare string rather than reading
  it as one label per character, and requires hashable per-dataset values.

- **A missing categorical `hue=` label now means one unlabeled group.**
  `nan != nan`, so two missing labels were not equal to each other and became
  two separate saturated categories -- and, since `np.nan` is a singleton
  while `float('nan')` is a fresh object each time, *which* of those happened
  depended on how the caller spelled it. Every missing spelling (`None`,
  NaN, `pd.NA`) now normalizes to the `None` sentinel a partially-labeled
  `hue=` already used: one group, neutral gray, no legend entry, no palette
  slot consumed. `forecast_hue=` follows the same rule.

- **A regrouped trajectory now animates in row order.** With `hue=`/`cluster=`,
  each contiguous same-category run is drawn as its own trace, and every run
  used to advance at once -- so one trajectory animated in several disjoint
  time windows simultaneously (three runs of a 30-row dataset were all 27%
  drawn on frame 3 of 12). Runs of one input dataset now share a single reveal
  clock, so the head sweeps the trajectory once and changes colour at each
  category boundary, matching both the un-regrouped and `order='serial'`
  behaviour. Animations without `hue=`/`cluster=` are unchanged row for row.
  A `precog=` trail on a not-yet-reached run now shows that run's whole future
  rather than a single stray point.

- **`predict=` now works with `hue=`/`cluster=` on ANIMATED plots.** Previously
  the fit succeeded and the forecasts were returned in the `return_model=True`
  bundle with `drawn=False`, but no overlay was drawn. Each frame's forecast is
  fit from exactly the observations visible for that dataset. A live forecast
  inherits the colour of the run drawing the head; a retained
  `forecast_trail=` member keeps the colour it was fit with;
  `forecast_hue=`/`forecast_cluster=`/`forecast_palette=` override both with a
  grouping fixed for the whole animation. Both backends draw it identically at
  every frame. Marker-only categorical regrouping (which groups globally by
  category, so its traces are not datasets) still draws no overlay and still
  says so.

- **`predict=` now works with `hue=` and `cluster=` on static plots.**
  Previously a forecast survived regrouping only by accident: the guard was a
  cardinality check, so two datasets falling into two category runs kept their
  forecasts while the same data in eight runs lost them silently. A forecast
  belongs to a DATASET and is anchored at that dataset's last observation, so
  it is now matched to whichever drawn trace holds that observation -- which
  is also the trace whose style it inherits.

  This uncovered a further way a forecast could vanish. Under a continuous
  `hue=` the overlays were drawn and then **deleted**: the code that swaps
  data lines for a colour-graded `LineCollection` cleared every line on the
  axes, forecasts included.

  Letting forecasts reach the drawing layer under regrouping for the first
  time then exposed the plotly backend, whose forecast block was written when
  that could not happen: it looped over the drawn **runs** while indexing the
  per-**dataset** forecast list, so `predict=` with a regrouping `hue=` raised
  `IndexError`. It now takes the same dataset-to-run mapping the matplotlib
  side uses, and both backends draw the same forecast from the same anchor.

  Animated plots under `hue=`/`cluster=` are covered by the previous entry.
  An interim build had refused them with a reason instead of failing
  silently, and plotly's static block, which fires whenever there is no
  per-frame schedule, had warned "no forecast is drawn" and then drawn the
  full-history forecast, visible from frame 0; both behaviours are
  superseded.

- **`return_model=True` reports forecasts it could not draw, and says so.**
  `bundle['predict']` gains **`drawn`** (bool) and **`draw_reason`** (`None`,
  or a sentence naming the limitation). A fit that succeeded is reported
  whether or not the figure could render it -- `return_model=` hands back
  model output, and discarding a valid result because a rendering combination
  is unsupported would throw away the thing it exists to return. `drawn` is
  what keeps "no forecast was computed" and "a forecast was computed but not
  drawn" distinguishable.

- **Forecast artists and traces are tagged, so callbacks can find them.**
  `artist._hyp_forecast_role` on matplotlib (`'static'`, `'live'` or
  `'trail'`) and `trace.meta['hyp_forecast_role']` on plotly, with
  `_hyp_forecast_age` on trail artists. Previously the only way to pick a
  forecast out of `ax.lines` was to guess from its linestyle, which also
  matched any user-supplied dashed line.

  Every forecast artist also names the series it belongs to:
  `artist._hyp_forecast_dataset` on matplotlib, matching plotly's existing
  `trace.meta['hyp_dataset']`. The role tag says what an artist *is*; this
  says *whose* it is, so forecasts pair with their data by identity rather
  than by drawing order.

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

- **`predict=` forecast overlays now inherit the style of the observed trace
  they continue.** A forecast reads as the *same series projected forward*,
  so it takes that trace's **colour, linestyle and linewidth**, and differs
  only in transparency: `forecast_alpha = observed_alpha * 0.5` (an unset
  `alpha=` is matplotlib's opaque 1.0, so the default forecast alpha is
  `0.5`). Per-dataset styling carries through dataset by dataset --
  `alpha=[1.0, 0.4]` gives forecasts at `[0.5, 0.2]`, and a dotted dataset
  gets a dotted forecast.

  This is a **visible change to existing forecast figures**, and it
  deliberately replaces the previous rule: every forecast used to be drawn
  `linestyle='--'` at a hard-coded `alpha=0.6` whatever its data looked
  like, so a forecast of a dotted, hairline or already-translucent dataset
  read as a *different* series rather than as its continuation. Both
  backends apply the identical policy (one shared constant,
  `hypertools.plot.forecast.FORECAST_ALPHA_SCALE`), so matplotlib and plotly
  cannot drift.

  `forecast_trail=` now fades from **that dataset's** live forecast alpha,
  down to a floor proportional to it rather than a fixed `0.08` — so a
  retained forecast can never come out *more* opaque than the live forecast
  it decays from, however faint the dataset. Depth, decay shape and the
  frame-index-derived (non-accumulating) fan are unchanged.

  Code that located forecast artists by their dashed linestyle must switch
  to the role tags, which exist for exactly this and are unchanged:
  `artist._hyp_forecast_role` (`'static'`/`'live'`/`'trail'`, plus
  `_hyp_forecast_age` on trails) on matplotlib, and
  `trace.meta['hyp_forecast_role']` / `hyp_forecast_age` /
  `hyp_forecast_alpha` on plotly. See "Identifying forecast artists" in
  `docs/animation.rst`.

- **Animated continuous-hue line plots with no explicit `linewidth=` now
  render at `1.0` instead of `1.5`.** This is a **visible change to
  existing animated hue figures**: the overlay now matches the width of the
  artist it replaces, which is what animated no-hue lines already used, so
  hue and no-hue animations finally agree. Pass `linewidth=1.5` to keep the
  old look.

### Bug fixes

- **The plotly backend's sliding animation window now matches matplotlib's,
  frame for frame.** `animate=True`/`'parallel'`/`'window'` paced every
  dataset against the LONGEST dataset in the plot and merely clamped the
  shorter ones into that one shared window, where matplotlib rescales the
  window onto each dataset's own rows. Four divergences came out of that,
  all of them plotly-only:

  - A **shorter dataset went blank for most of its own animation.** A 5-row
    marker dataset plotted beside a 15-row line drew nothing at all for 9 of
    its 15 frames -- 60% of the animation -- because the shared window slid
    off its end, while matplotlib kept a correctly-paced 2-point window alive
    to the final frame. Short and long datasets now advance side by side on
    both backends.
  - The **head window ran one point short** at every steady-state frame (a
    missing `- 1` in the window's start index). Beyond the count, this opened
    a one-segment **gap between a `chemtrails` trail and the opaque head it
    is supposed to join** -- the trail now ends exactly on the head's first
    vertex, as it always has in matplotlib.
  - **Frame 0 of a `precog` trail was one point short** (the revealed-row
    count floored at 2 where matplotlib floors at 1).
  - A **sub-frame request** (`duration * frame_rate` rounding below 1)
    produced a 2-frame plotly animation against matplotlib's single still.
    Because that count is also the denominator each dataset's window is
    paced against, the floor shifted the pacing of every frame, not just the
    frame count.

  Both backends now call one shared function
  (`hypertools.plot.trails.anim_window_bounds`), per dataset, per frame, so
  the two cannot drift apart again. Point counts in existing plotly
  `animate=`/`'window'` animations may shift by a point at some frames; no
  documented behavior changes.

- **A very short animation no longer comes out empty.** When
  `duration * frame_rate` rounded below 1, matplotlib asked for **zero
  frames** under `animate='serial'` and `animate='spin'` -- an animation
  that draws nothing at all -- because only its parallel/`'window'` path
  floored the count at one frame. Every style now floors at one frame on
  both backends.

- **`animate='spin'` paces its orbit over the frames it actually draws.** The
  matplotlib spin camera divided its rotation by the raw `frame_rate *
  duration` product rather than the number of frames drawn, so whenever that
  product was not a whole number the two backends pointed the camera
  somewhere different on the same call: at `frame_rate=7, duration=2.5` (18
  frames, product 17.5) matplotlib's last frame sat at 289.7 degrees and
  plotly's at 280.0. Overshooting also spoils a looping `rotations=1` spin --
  frames 0..N-1 are meant to span a full turn *exclusive*, so the animation
  does not draw the same angle twice when it wraps. Both backends now divide
  by the rounded frame count, which is what every other matplotlib animation
  path already did.

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

- **A partial-tag `animate='morph'` list (e.g. `animate=[None, 'morph',
  'morph']`) now names and reports the DATASET actually being shown, on
  both backends.** Per-segment `title=` and `FrameContext.current_index`
  used to index the morph hold/transition schedule by its position WITHIN
  THE MORPH SEQUENCE rather than mapping through the tag list, so an
  untagged (static) leading dataset silently shifted every title/index
  down by one -- the first hold was titled with the untagged dataset's
  name (never shown) and the true final dataset's title was unreachable.
  Scalar `animate='morph'` (every dataset tagged) was never affected,
  since sequence position and dataset index coincide there by
  construction.

- **`ctx.datasets` for `animate='morph'` is now the same, morph-sampled
  arrays on both backends.** plotly recorded the raw, uncapped input
  arrays; matplotlib already recorded the Hungarian-matched,
  `morph_samples`-capped clouds actually drawn from, per `FrameContext`'s
  own documented contract. The two now agree.

- **plotly trail traces (`chemtrails`/`precog`/`bullettime`) honor
  per-dataset `alpha=`.** They previously hardcoded a flat 0.3 opacity
  regardless of `alpha=`, while matplotlib already folded `alpha` into the
  0.3 trail fade (`0.3 * alpha`). A per-dataset `alpha=` list now fades
  plotly trails the same way.

- **`animate='spin'`/`'window'`, `order='serial'`, and a per-dataset
  `title=` list together now raise immediately with an accurate message.**
  Previously this combination ran the whole analyze/reduce pipeline,
  warned that `order='serial'` was being ignored (because `'spin'`/
  `'window'` have no serial reveal), and only then raised `TypeError`
  advising `order='serial'` -- exactly what had already been passed. The
  error now fires fail-fast and names the real reason (the style has no
  serial ordering to name segments by).

- **Kalman forecasts no longer diverge from a near-saturated fit.**
  `hyp.predict(x, model='Kalman', t=...)` could return values up to 1e7
  times the range of the data they were meant to continue (measured: 19 of
  432 fits on 40x3 drifting random walks exceeded 100x the data range). The
  delay-embedded transition operator was estimated by unconstrained least
  squares with nothing checking that it was non-explosive, and rolling a
  linear-Gaussian model forward with no observations is exactly
  `mean <- A @ mean`, so the forecast grows as `rho(A)**t`. When the number
  of predictors approaches the number of usable windows the fit is
  ill-conditioned and `rho` ran as high as 4.16.

- **A singleton-`hue=` warning now names the category the caller passed.**
  It read `hue category '_nolegend_' has only one observation ...` --
  matplotlib's sentinel for "keep this artist out of the legend", which
  `_regroup_categorical_lines` assigns to every REPEAT run of a category so
  each category gets exactly one legend entry. Any singleton run after the
  first of its category was therefore reported under a name the caller never
  supplied and could not go looking for. The warning now reads the real
  per-run category names.

## 1.0.0 (2026-07-24)

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
