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
  correspondence. All 138 of its examples are executed by the test suite
  rather than merely read. `docs/pipeline_order.rst` gains hierarchy
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

### Changed / validation

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

### Bug fixes

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
> the patch line.

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

  ANIMATED plots still draw no forecast under `hue=`/`cluster=`, but now say
  so instead of failing silently. The per-frame schedule maps frame-grid rows
  onto each dataset's raw observations, and regrouping leaves only per-run
  traces to reveal. **Both** backends now refuse: plotly's static block fires
  whenever there is no per-frame schedule -- exactly the state this refusal
  creates -- so it warned "no forecast is drawn" and then drew the
  full-history forecast, visible from frame 0.

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
