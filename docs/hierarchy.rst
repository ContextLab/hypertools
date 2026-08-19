.. _hierarchy:

Hierarchical DataFrames
========================

A pandas ``MultiIndex`` says that your observations, or your features, come
in *groups*: subjects nested in conditions, tickers nested in sectors,
sensors nested in rigs. Since 1.1, `hypertools.plot` and
`hypertools.predict` both read that structure instead of flattening it
away.

The whole feature rests on one distinction, so it is worth stating before
anything else:

* a **row** ``MultiIndex`` groups **observations**;
* a **column** ``MultiIndex`` groups **features** -- and its *innermost*
  level **is** the feature axis. Every level above the innermost one does
  the grouping.

Everything below follows from that. Every example on this page is
executable and its output is checked by ``tests/test_docs_hierarchy_guide.py``.

.. contents:: On this page
   :local:
   :depth: 1


Row versus column semantics
----------------------------

A **column** hierarchy names the same measurements in several groups. Here
is a market with two sectors, each measured three ways:

.. doctest::

   >>> import numpy as np
   >>> import pandas as pd
   >>> import hypertools as hyp
   >>> rng = np.random.default_rng(0)
   >>> columns = pd.MultiIndex.from_tuples(
   ...     [('Market', sector, measure)
   ...      for sector in ('Tech', 'Energy')
   ...      for measure in ('return', 'volatility', 'momentum')],
   ...     names=['Market', 'Sector', 'Measure'])
   >>> market = pd.DataFrame(
   ...     np.cumsum(rng.standard_normal((40, 6)), axis=0), columns=columns)
   >>> market.shape
   (40, 6)

``Measure`` is the innermost level, so ``Measure`` is the feature axis and
``(Market, Sector)`` does the grouping. Each group becomes one leaf, and
each leaf is **flattened onto the feature axis**: its columns become the
innermost level's values, keeping that level's name.

.. doctest::

   >>> from hypertools.core.hierarchy import group_columns
   >>> leaves, meta = group_columns(market)
   >>> meta['leaf_keys']
   [('Market', 'Tech'), ('Market', 'Energy')]
   >>> list(leaves[0].columns)
   ['return', 'volatility', 'momentum']
   >>> leaves[0].columns.name
   'Measure'
   >>> meta['n_levels'], meta['level_names'], meta['axis']
   (2, ['Market', 'Sector'], 'columns')

Two things to notice. ``n_levels`` counts the **grouping** levels (2), not
the frame's three column levels -- the innermost one is features, not
hierarchy. And grouping never modifies the frame you passed in:

.. doctest::

   >>> market.columns.nlevels
   3

A **row** hierarchy groups observations instead. Its innermost level is
part of a leaf's identity when plotting, and is the time axis when
forecasting -- which is the one place the two consumers genuinely diverge.

.. _hierarchy-comparison-table:

.. list-table:: How each axis is read, by consumer
   :header-rows: 1
   :widths: 24 24 22 30

   * - Axis and consumer
     - Innermost level means
     - Grouping
     - ``plot(..., predict=)``
   * - Row MultiIndex, plot
     - part of leaf identity
     - full tuple
     - only when every leaf and mean has at least 2 rows
   * - Row MultiIndex, predict
     - time/observation
     - all outer levels
     - n/a -- `hypertools.predict` groups by the outer levels instead
   * - Column MultiIndex, plot/predict
     - feature name
     - all outer levels
     - whenever the frame has at least 2 rows -- every group keeps all of them

For forecasting, a row group's innermost level survives as that group's
**flat** index, with its name and dtype intact -- so a datetime innermost
level is forecast against a ``DatetimeIndex`` and ``t`` may be a future
``Timestamp``:

.. doctest::

   >>> rows = pd.MultiIndex.from_product(
   ...     [['Tech', 'Energy'],
   ...      pd.date_range('2024-01-01', periods=30, freq='D')],
   ...     names=['Sector', 'date'])
   >>> daily = pd.DataFrame(np.cumsum(rng.standard_normal((60, 3)), axis=0),
   ...                      index=rows,
   ...                      columns=['open', 'close', 'volume'])
   >>> forecasts = hyp.predict(daily, model='Kalman',
   ...                         t=pd.Timestamp('2024-02-05'))
   >>> [f.shape for f in forecasts]
   [(6, 3), (6, 3)]
   >>> type(forecasts[0].index).__name__
   'DatetimeIndex'
   >>> forecasts[0].index[-1]
   Timestamp('2024-02-05 00:00:00')

The name travels with the *group*, not with the forecast. ``hyp.predict``
builds a fresh horizon index for what it returns, and that index is
unnamed -- on flat input too, so this is not a hierarchy rule:

.. doctest::

   >>> from hypertools.core.hierarchy import group_rows_for_forecast
   >>> groups, keys = group_rows_for_forecast(daily)
   >>> keys[0], groups[0].index.name, type(groups[0].index).__name__
   (('Tech',), 'date', 'DatetimeIndex')
   >>> forecasts[0].index.name is None
   True
   >>> flat = daily.loc['Tech']
   >>> flat.index.name
   'date'
   >>> hyp.predict(flat, model='Kalman', t=2).index.name is None
   True


Plotting versus forecasting
----------------------------

Hand the *same* ``(Sector, day)`` frame to each function and you get
different -- both correct -- answers.

.. doctest::

   >>> rows = pd.MultiIndex.from_product(
   ...     [['Tech', 'Energy'], range(10)], names=['Sector', 'day'])
   >>> panel = pd.DataFrame(rng.standard_normal((20, 4)), index=rows,
   ...                      columns=list('abcd'))
   >>> bundle = hyp.plot(panel, return_model=True)
   >>> len(bundle['trace_data'])
   22
   >>> sorted({a.shape for a in bundle['trace_data']})
   [(1, 3)]
   >>> [f.shape for f in hyp.predict(panel, model='Kalman', t=3)]
   [(3, 4), (3, 4)]

`hypertools.plot` draws **one trace per unique full index tuple**, so
``('Tech', 0)``, ``('Tech', 1)``, ... are twenty separate one-row leaves
plus two sector means: 22 traces. `hypertools.predict` treats ``day`` as
the time axis and groups by ``Sector``: two forecasts, each over the whole
ten-day history.

Neither rule is a bug. Plotting a panel of trajectories and forecasting a
panel of timeseries are different questions, and the innermost level plays
a different role in each.

When ``plot(..., predict=)`` is defined
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``plot(..., predict=)`` forecasts **every final trace** -- every leaf and
every derived mean -- so it is defined whenever every plotted trace has
**at least 2 rows**, on either axis. Forecasting needs history.

For a **column** hierarchy that means whenever the frame itself has at
least 2 rows, since every group keeps all of them. For a **row** hierarchy
it means every leaf *and* every derived mean, because row expansion draws
one trace per unique full index tuple.

A frame that qualifies has several timepoints per ``(cond, subj)`` -- that
is, the full index tuple repeats:

.. doctest::

   >>> rows = pd.MultiIndex.from_tuples(
   ...     [(cond, subj) for cond in ('A', 'B') for subj in ('s1', 's2')
   ...      for _ in range(12)], names=['cond', 'subj'])
   >>> trials = pd.DataFrame(np.cumsum(rng.standard_normal((48, 5)), axis=0),
   ...                       index=rows, columns=list('vwxyz'))
   >>> bundle = hyp.plot(trials, predict='Kalman', t=4, return_model=True)
   >>> bundle['trace_metadata']['keys']
   [('A', 's1'), ('A', 's2'), ('B', 's1'), ('B', 's2'), ('A',), ('B',)]
   >>> [f.shape for f in bundle['predict']['forecasts']]
   [(4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3)]

Note that ``t=`` is a `hypertools.plot` keyword in its own right, not a key
inside the ``predict=`` spec.

Two frames do **not** qualify. The first has an innermost row level that is
unique per row, so expansion yields one-row traces:

.. doctest::

   >>> rows = pd.MultiIndex.from_arrays(
   ...     [['Tech'] * 6 + ['Energy'] * 6, list(range(12))],
   ...     names=['Sector', 'row'])
   >>> unique_rows = pd.DataFrame(rng.standard_normal((12, 3)), index=rows,
   ...                            columns=list('abc'))
   >>> try:
   ...     hyp.plot(unique_rows, predict='Kalman', t=3)
   ... except ValueError as error:
   ...     print(error)
   plot(..., predict=...) needs at least 2 rows per trace, but trace 0 ('Tech', 0) has 1 row. Row-MultiIndex expansion draws one trace per unique FULL index tuple, so a frame whose innermost index level is unique per row yields one-row traces (and one-row per-level means). Either drop the hierarchy so the frame is one trajectory (df.reset_index(drop=True)), or move the grouping to the COLUMN axis, where every group keeps all of the frame's rows.

The second is a column hierarchy with a single observation:

.. doctest::

   >>> one_row = pd.DataFrame(
   ...     rng.standard_normal((1, 4)),
   ...     columns=pd.MultiIndex.from_tuples(
   ...         [(s, m) for s in ('Tech', 'Energy') for m in ('r', 'v')],
   ...         names=['Sector', 'Measure']))
   >>> try:
   ...     hyp.plot(one_row, predict='Kalman', t=3)
   ... except ValueError as error:
   ...     print(error)
   plot(..., predict=...) needs at least 2 rows per trace, but trace 0 ('Tech',) has 1 row. A column MultiIndex groups FEATURES, so every group keeps all 1 of the frame's rows -- the input itself has only one observation. Forecasting needs at least 2 observations (rows) to estimate how the data change over time; pass a frame with more rows.

The column message deliberately does **not** offer flattening as a remedy,
the way the row message offers ``reset_index``: flattening columns cannot
add a row. Only more data helps.

Both checks run on the **plotted** trajectories, so a row-count-changing
analysis stage (``manip='Resample'``, a smoother that trims edges) can
trigger them too. The error distinguishes the two causes and names the one
that applies.


Hue over a hierarchy
---------------------

A **continuous** ``hue=`` is carried *through* a column hierarchy rather
than superseded by it. Two forms are accepted, both stated relative to the
**input frame** rather than to the drawn figure:

* a flat sequence of ``len(x)`` values -- shared row-wise values, broadcast
  to every trace;
* one sequence per leaf, each of ``len(x)`` values.

The per-trace values come back as ``trace_metadata['aux']``:

.. doctest::

   >>> price = np.linspace(100.0, 140.0, len(market))
   >>> bundle = hyp.plot(market, hue=price, return_model=True)
   >>> aux = bundle['trace_metadata']['aux']
   >>> [a.shape for a in aux]
   [(40,), (40,), (40,)]

A mean trace takes the **element-wise mean of its members' hue**, computed
by the same operation that averages their data, so an auxiliary value can
never drift out of step with the trace it describes:

.. doctest::

   >>> bool(np.allclose(aux[2], (aux[0] + aux[1]) / 2))
   True

Colours are mapped over the concatenation of every trace's values -- leaves
and means together -- so one scale spans the figure and a ``colorbar=``
reads against all of it. The hierarchy still sets linewidth, alpha and
labels; only its colours step aside. Forecast overlays inherit the colour
of the trace they continue, taken at the anchor point.

A flat array sized to the **total drawn observations** is rejected, not
reinterpreted -- it is indistinguishable from the first form whenever a
frame has as many rows as the figure has points, and it would require you
to predict how many mean traces the expansion creates:

.. doctest::

   >>> try:
   ...     hyp.plot(market, hue=np.linspace(0.0, 1.0, 80))
   ... except ValueError as error:
   ...     print(error)
   hue over a column hierarchy must be a flat sequence of 40 row values (broadcast to every trace), or one hue sequence per leaf (2 sequences of 40 values); got 80 values.

Two exceptions. A **categorical** hue still defers to the grouping with a
``UserWarning``, because it regroups traces and the named leaves would stop
existing. And a **row** hierarchy still warns and ignores ``hue=``
altogether; ``trace_metadata['aux']`` is ``None`` in both cases.


Matrix hue: one blend per observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A per-leaf sequence may be **2-D**. Then each row is a set of **mixture
weights** over the palette, and the observation is drawn in the blend those
weights describe. This is the form that lets the hierarchy's own arithmetic
choose the colours: a mean trace averages its members' weights, and the mean
of mixture weights is itself a mixture weight -- so give each leaf one
*primary* and every parent comes out a *secondary* without anything
computing it.

.. doctest::

   >>> weights = [np.tile([1.0, 0.0, 0.0], (len(market), 1)),   # Tech: red
   ...            np.tile([0.0, 1.0, 0.0], (len(market), 1))]   # Energy: yellow
   >>> bundle = hyp.plot(market, hue=weights,
   ...                   palette=['#d92b2b', '#e8c72a', '#2f5fd0'],
   ...                   return_model=True)
   >>> aux = bundle['trace_metadata']['aux']
   >>> [a.shape for a in aux]
   [(40, 3), (40, 3), (40, 3)]
   >>> aux[2][0]                       # the mean of one red and one yellow
   array([0.5, 0.5, 0. ])

The rules, in full:

* weights are **normalized to sum to 1**, so only the ratio between
  components shows. Halving every weight in a row draws the identical
  colour -- a second quantity needs its own palette entry (a black one, say,
  for "darker = larger"), it cannot ride on the total magnitude;
* **negative** entries have no colour meaning, so each row is shifted by its
  own minimum: a signed matrix is coloured by within-row *contrast*;
* the palette must supply at least one colour **per column**. A shorter
  palette raises; a longer one leaves components unused;
* every per-leaf matrix must have the **same width** -- they share one
  palette;
* a **non-finite** entry draws that observation neutral grey and warns. A
  mean is the element-wise mean of its members, so one ``NaN`` greys the
  leaf *and every ancestor mean* at that row;
* more than **3 columns**, or any explicit ``color_reduce=``, switches to
  the literal-RGB route instead -- the matrix is reduced to three min-max
  scaled channels used directly as *(r, g, b)*. That is how you supply
  per-observation RGB under a hierarchy, and it means exactly what it means
  on a flat plot.


Mean trace construction
------------------------

`hypertools.plot` builds the final trace list as **leaves first, then the
derived means deepest-last**, so the top-level mean is the final trace.
Means are built from leaves, never from other means.

.. doctest::

   >>> bundle = hyp.plot(market, return_model=True)
   >>> trace_meta = bundle['trace_metadata']
   >>> trace_meta['keys']
   [('Market', 'Tech'), ('Market', 'Energy'), ('Market',)]
   >>> trace_meta['level_idx']
   [1, 1, 0]
   >>> trace_meta['is_mean']
   [False, False, True]

``level_idx`` is the depth the style formulas use: ``n_levels - 1`` for a
leaf, and ``k`` for a mean over levels ``0..k``. The formulas themselves
are exact:

* ``linewidth = 1 + (n_levels - 1 - level_idx)``
* ``alpha = min(1.0, 1.0 / (level_idx + 1) + 0.2)``
* only the **top-level mean** (``level_idx == 0`` and ``is_mean``) carries a
  legend label; everything else is ``'_nolegend_'``.

.. doctest::

   >>> from hypertools.plot.hierarchy import (build_hierarchy_styles,
   ...                                        build_hierarchy_traces)
   >>> leaves, meta = group_columns(market)
   >>> traces = build_hierarchy_traces([np.asarray(l) for l in leaves], meta)
   >>> style = build_hierarchy_styles(traces)
   >>> style['linewidths']
   [1.0, 1.0, 2.0]
   >>> [round(a, 2) for a in style['alphas']]
   [0.7, 0.7, 1.0]
   >>> style['labels']
   ['_nolegend_', '_nolegend_', 'Market']

There is one exception to the label rule. When ``n_levels == 1`` there *is*
no mean -- each leaf is itself a top-level group -- so each leaf carries its
own label:

.. doctest::

   >>> flat_hierarchy = pd.DataFrame(
   ...     rng.standard_normal((10, 4)),
   ...     columns=pd.MultiIndex.from_tuples(
   ...         [(g, f) for g in ('G1', 'G2') for f in ('a', 'b')],
   ...         names=['Group', 'Feature']))
   >>> leaves, meta = group_columns(flat_hierarchy)
   >>> traces = build_hierarchy_traces([np.asarray(l) for l in leaves], meta)
   >>> traces.is_mean
   [False, False]
   >>> build_hierarchy_styles(traces)['labels']
   ['G1', 'G2']

Colour is assigned per **top-level** index value, so every trace descending
from one top-level group shares its colour and is distinguished by weight
and opacity instead.


Limitations
------------

**Groups must share their innermost labels.** Feature correspondence is
nominal (see `Feature names and duplicates`_), so groups naming different
features -- ragged groups included -- are refused by name rather than
silently paired up by position.

**Unequal-length row groups are truncated to their overlap**, with a single
aggregated ``UserWarning`` per call (deduplicated even when a 3+-level tree
makes several groupings share the same short member):

.. doctest::

   >>> import warnings
   >>> ragged = pd.DataFrame(
   ...     rng.standard_normal((13, 3)),
   ...     index=pd.MultiIndex.from_tuples(
   ...         [('A', 's1')] * 8 + [('A', 's2')] * 5, names=['cond', 'subj']),
   ...     columns=list('abc'))
   >>> with warnings.catch_warnings(record=True) as caught:
   ...     warnings.simplefilter('always')
   ...     bundle = hyp.plot(ragged, return_model=True)
   >>> [a.shape for a in bundle['trace_data']]
   [(8, 3), (5, 3), (5, 3)]
   >>> print([str(w.message) for w in caught if 'unequal' in str(w.message)][0])
   MultiIndex group(s) with unequal-length members: 'A' has members of unequal length ([8, 5]), averaged over the overlapping prefix of 5 row(s).

The leaves keep their own lengths; only the **mean** is confined to the
overlapping prefix, because averaging assumes members line up by row
position at each timepoint.

**Some kwargs are refused, and some are ignored.** ``cluster=`` and
``n_clusters=`` raise ``ValueError`` on **either** axis -- both would fight
the hierarchy's colour assignment -- and the remedy differs by axis, which
is why the two messages differ. ``df.reset_index(drop=True)`` flattens a
ROW hierarchy, but it does not touch a column ``MultiIndex``, so a column
hierarchy needs ``df.columns = df.columns.map('_'.join)`` instead:

.. doctest::

   >>> for frame in (market, trials):
   ...     try:
   ...         hyp.plot(frame, cluster='KMeans', n_clusters=2)
   ...     except ValueError as error:
   ...         print(error)
   cluster=/n_clusters= is not compatible with a column-MultiIndex DataFrame: MultiIndex grouping already assigns colors by the top-level column index and would conflict with cluster-based grouping. Flatten the columns (df.columns = df.columns.map('_'.join)) before clustering, or drop cluster=/n_clusters= to use the MultiIndex grouping.
   cluster=/n_clusters= is not compatible with a row-MultiIndex DataFrame (GH #95): MultiIndex grouping already assigns colors by the top-level index and would conflict with cluster-based grouping. Reset the index (df.reset_index(drop=True)) before clustering, or drop cluster=/n_clusters= to use the MultiIndex grouping.

Taking the row remedy to a column hierarchy therefore leaves the call
refused, with the same message:

.. doctest::

   >>> try:
   ...     hyp.plot(market.reset_index(drop=True), cluster='KMeans',
   ...              n_clusters=2)
   ... except ValueError as error:
   ...     print(str(error).split(':')[0])
   cluster=/n_clusters= is not compatible with a column-MultiIndex DataFrame

``color=``/``colors=`` and ``linewidth=`` (and ``alpha=``) are ignored with
a ``UserWarning``, since the hierarchy owns them. A ``linestyle=`` **list**
must have exactly one entry per top-level group.

``legend=`` is the one overridden kwarg that is *honoured* rather than
warned away -- colour, width and alpha encode the hierarchy's structure, so
a caller's value would contradict the drawing, but legend text names groups
the hierarchy has no opinion about. A **list** renames the top-level groups
(one entry per unique top-level value, in first-appearance order, exactly
like ``linestyle=``); ``legend=False`` suppresses the automatic legend; and
``legend=True`` (or omitting it) labels the groups with the index values
themselves. ``legend=False`` suppresses the *legend* only -- ``colorbar=True``
still shows one named segment per top-level group, because the colorbar is
the colour key for the drawn groups rather than a second legend:

.. doctest::

   >>> style = build_hierarchy_styles(
   ...     traces, legend_labels=['Group one', 'Group two'])
   >>> style['labels']
   ['Group one', 'Group two']
   >>> try:
   ...     build_hierarchy_styles(traces, legend_labels=['only one'])
   ... except ValueError as error:
   ...     print(str(error).split(';')[0])
   legend= has 1 entries but there are 2 unique top-level MultiIndex value(s) (['G1', 'G2'])

``names=`` (per-INPUT-DATASET legend entries) does *not* apply to a
hierarchy: there is one input frame, drawn as leaves plus derived means, so
passing it raises ``ValueError`` pointing at ``legend=[...]`` instead:

.. doctest::

   >>> try:
   ...     hyp.plot(market, names=['a', 'b', 'c'])
   ... except ValueError as error:
   ...     print(str(error).split('.')[0])
   names= assigns one name per input dataset, but x has a column MultiIndex, so the drawn traces are hierarchy groups (2 leaf trajectory/ies + 1 derived per-level mean(s)), not input datasets

And ``predict=`` with
``animate='morph'`` raises ``NotImplementedError``: a morph interpolates
between point clouds and so has no time axis to forecast along.


Dual-axis and list inputs
--------------------------

A frame carrying a hierarchy on **both** axes is refused by both entry
points. Which hierarchy should win is genuinely ambiguous; before 1.1 the
row path silently won and the column hierarchy was discarded.

.. doctest::

   >>> dual = pd.DataFrame(
   ...     rng.standard_normal((8, 2)),
   ...     index=pd.MultiIndex.from_product([['A', 'B'], range(4)],
   ...                                      names=['g', 't']),
   ...     columns=pd.MultiIndex.from_product([['S'], ['x', 'y']],
   ...                                        names=['grp', 'feat']))
   >>> for function in (hyp.plot, hyp.predict):
   ...     try:
   ...         function(dual)
   ...     except ValueError as error:
   ...         print(error)
   x has both a row and a column MultiIndex. hypertools 1.1 does not define which hierarchy takes precedence. Flatten one axis (e.g. df.reset_index(drop=True), or df.columns = df.columns.map('_'.join)) and try again.
   x has both a row and a column MultiIndex. hypertools 1.1 does not define which hierarchy takes precedence. Flatten one axis (e.g. df.reset_index(drop=True), or df.columns = df.columns.map('_'.join)) and try again.

Hierarchy expansion is defined for a **bare** frame only, because the
hierarchy determines the entire trace list and that cannot be reconciled
with a caller-supplied list of datasets. A hierarchical frame nested in a
list is therefore rejected -- but **asymmetrically**:

* `hypertools.plot` rejects a **column** hierarchy in a list, and still
  warns-and-flattens a **row** one;
* `hypertools.predict` rejects **either** axis.

.. doctest::

   >>> try:
   ...     hyp.plot([market, market])
   ... except ValueError as error:
   ...     print(error)
   hyp.plot received a list whose element 0 is a DataFrame with a column MultiIndex. Hierarchy expansion is defined for a BARE DataFrame only, because the hierarchy determines the entire group list. Pass the frame on its own (hyp.plot(df, ...)), or flatten it first (df.reset_index(drop=True), or df.columns = df.columns.map('_'.join)).

   >>> with warnings.catch_warnings(record=True) as caught:
   ...     warnings.simplefilter('always')
   ...     figure = hyp.plot([trials, trials])
   >>> print([str(w.message) for w in caught
   ...        if 'only applied' in str(w.message)][0])
   MultiIndex grouping is only applied when a single DataFrame is passed; the MultiIndex on dataset 0 is being treated as a flat index.

   >>> try:
   ...     hyp.predict([trials, trials], t=2)
   ... except ValueError as error:
   ...     print(str(error).split('. ')[0])
   hyp.predict received a list whose element 0 is a DataFrame with a row MultiIndex

The asymmetry is deliberate. `hypertools.plot`'s row behaviour is
documented and depended upon, so it is preserved; its column behaviour was
never pinned (it silently flattened to a single line), so rejecting it is
purely additive. `hypertools.predict` had nothing to preserve on either
axis -- a row-hierarchical frame in a list raised a ``TypeError`` from deep
inside pandas, and a column-hierarchical one silently forecast the
flattened frame.

The flattening recipes are the ones the messages name:
``df.reset_index(drop=True)`` for a row hierarchy,
``df.columns = df.columns.map('_'.join)`` for a column one.


Return shapes
--------------

``return_model=True`` gives back a bundle whose two data entries answer
different questions.

``xform_data`` is the **analysed pipeline output**, one entry per analysed
input dataset -- for a hierarchy, one per leaf. ``trace_data`` is the final
**pre-center/pre-scale plotted trajectories**: for a hierarchy, the leaves
followed by the per-level means. Means are presentation artifacts built in
display space, which is why they are deliberately absent from
``xform_data``.

.. doctest::

   >>> bundle = hyp.plot(market, predict='Kalman', t=5, return_model=True)
   >>> len(bundle['xform_data']), len(bundle['trace_data'])
   (2, 3)
   >>> [a.shape for a in bundle['trace_data']]
   [(40, 3), (40, 3), (40, 3)]

For non-hierarchical input the two are **the same object** when no
display-only projection occurred, and diverge when a ``reduce=`` spec pins
more than three components: ``xform_data`` then keeps that many while
``trace_data`` is projected to the plotted dimensionality.

For a hierarchy they are never the same object. ``trace_data`` is a freshly
built trace list, and it is normally longer as well, because the per-level
means are appended to it. A hierarchy with a single grouping level is the
one case with no means at all -- and even then the two hold equal contents
as two distinct lists:

.. doctest::

   >>> one_level = pd.DataFrame(
   ...     np.cumsum(rng.standard_normal((30, 6)), axis=0),
   ...     columns=pd.MultiIndex.from_tuples(
   ...         [(g, f) for g in ('G1', 'G2') for f in ('a', 'b', 'c')],
   ...         names=['Group', 'Feature']))
   >>> single = hyp.plot(one_level, return_model=True)
   >>> single['trace_metadata']['is_mean']
   [False, False]
   >>> len(single['xform_data']), len(single['trace_data'])
   (2, 2)
   >>> single['xform_data'] is single['trace_data']
   False
   >>> all(bool(np.allclose(a, b)) for a, b in
   ...     zip(single['xform_data'], single['trace_data']))
   True

Neither is what the artists hold: those are centered, scaled and (unless
``antialias=False``) PCHIP-upsampled afterwards.

The bundle's ``pipeline`` re-applies to the frame that produced it. A
hierarchy's steps are fit on the frame's **groups** -- each as wide as one
group, not as the frame -- so the pipeline remembers the grouping and
reproduces it, returning one array per group. (``market``'s groups are
already 3 columns wide, so no reduction was needed for them at all; the
grouping is still what makes the answer meaningful.)

.. doctest::

   >>> [np.asarray(a).shape
   ...  for a in bundle['pipeline'].transform(market)]
   [(40, 3), (40, 3)]

Name matching applies to a **bare hierarchical frame**, and only to that.
A *list* is taken as already grouped and is matched positionally -- its
members ARE the datasets, whether they are arrays or labelled DataFrames --
so ``pipeline.transform([leaf_a, leaf_b])`` never reorders columns for you.
That is the same positional contract a list has always had elsewhere in
hypertools; hand the pipeline the frame if you want the labels honoured.

Given a frame, features are matched to the fitted pipeline **by name**, on
the same terms they are matched across groups, so reordering the innermost
labels changes nothing:

.. doctest::

   >>> shuffled = market[[('Market', sector, measure)
   ...                    for sector in ('Tech', 'Energy')
   ...                    for measure in ('momentum', 'return',
   ...                                    'volatility')]]
   >>> all(bool(np.allclose(a, b)) for a, b in
   ...     zip(bundle['pipeline'].transform(market),
   ...         bundle['pipeline'].transform(shuffled)))
   True

Handing it a **flattened** frame instead is refused by name, rather than
passed through as though the grouping had never mattered:

.. doctest::

   >>> flat = market.copy()
   >>> flat.columns = ['_'.join(label) for label in market.columns]
   >>> try:
   ...     bundle['pipeline'].transform(flat)
   ... except ValueError as error:
   ...     print(str(error).split('. ')[0])
   this Pipeline was fit on the 3-feature groups of a column-hierarchical DataFrame, but the data given here has [6] feature(s) per dataset

A pipeline you fitted yourself and passed in via ``pipeline=`` is handed back
in the bundle -- it is the same object -- and it re-applies on the same
terms: plotting a hierarchical frame records that frame's grouping on it,
unless it already carries one of its own.

.. doctest::

   >>> leaves, _ = group_columns(market)
   >>> _, mine = hyp.analyze([leaf.to_numpy() for leaf in leaves],
   ...                       reduce='PCA', ndims=3, return_model=True)
   >>> mine.input_hierarchy is None
   True
   >>> reused = hyp.plot(market, pipeline=mine, return_model=True)
   >>> reused['pipeline'] is mine
   True
   >>> mine.input_hierarchy['n_features']
   3
   >>> [np.asarray(a).shape for a in mine.transform(market)]
   [(40, 3), (40, 3)]

Bundled forecasts always correspond to ``trace_data``, so ``forecasts[i]``
equals ``hyp.predict(trace_data[i], model=..., t=t)`` for every ``i`` --
including the means, each forecast from its own averaged trajectory rather
than from an average of its members' forecasts:

.. doctest::

   >>> mean_forecast = hyp.predict(bundle['trace_data'][-1],
   ...                             model='Kalman', t=5)
   >>> bool(np.allclose(np.asarray(mean_forecast),
   ...                  np.asarray(bundle['predict']['forecasts'][-1])))
   True

``trace_metadata`` is ``None`` for non-hierarchical input. For a hierarchy
it describes every entry of ``trace_data`` positionally, with one entry per
trace in ``keys``, ``level_idx``, ``is_mean`` and (when a continuous
``hue=`` was carried through) ``aux``, plus the whole-hierarchy ``axis`` and
``level_names``:

.. doctest::

   >>> sorted(bundle['trace_metadata'])
   ['aux', 'axis', 'is_mean', 'keys', 'level_idx', 'level_names']
   >>> bundle['trace_metadata']['axis']
   'columns'
   >>> bundle['trace_metadata']['level_names']
   ['Market', 'Sector']
   >>> bundle['trace_metadata']['aux'] is None
   True

`hypertools.predict` returns the parallel shape: a **list** of forecasts,
one per group in input order, and with ``return_model=True`` the parallel
pair ``([f0, f1, ...], [m0, m1, ...])`` rather than a list of pairs.

.. doctest::

   >>> forecasts = hyp.predict(market, model='Kalman', t=5)
   >>> [f.shape for f in forecasts]
   [(5, 3), (5, 3)]
   >>> list(forecasts[0].columns)
   ['return', 'volatility', 'momentum']


Fitted model behaviour
-----------------------

Hierarchical forecasting needs one model per group, so `hypertools.predict`
states ownership explicitly. **The object you pass is never mutated**, and
``return_model=True`` returns one distinct object per group.

.. list-table:: What a ``model=`` spec means on hierarchical input
   :header-rows: 1
   :widths: 34 66

   * - ``model=``
     - Behaviour per group
   * - a name, a class, or a dict spec
     - one independent model is **fitted per group**
   * - an **unfitted** instance
     - deep-copied per group, then fitted; your object stays unfitted, and
       later groups cannot fall onto the reuse path just because an earlier
       group fitted a shared object
   * - an **already-fitted** instance
     - its learned parameters are **reused** on every group (via
       ``predict_new``), each through an independent deep copy -- not
       refitted

.. doctest::

   >>> from hypertools.predict.kalman import Kalman
   >>> shared = Kalman()
   >>> shared.is_fitted
   False
   >>> forecasts, models = hyp.predict(market, model=shared, t=5,
   ...                                 return_model=True)
   >>> [f.shape for f in forecasts]
   [(5, 3), (5, 3)]
   >>> shared.is_fitted
   False
   >>> [m.is_fitted for m in models]
   [True, True]
   >>> models[0] is models[1] or models[0] is shared
   False

Passing one of those fitted models back reuses its parameters on every
group, again through independent copies:

.. doctest::

   >>> reused, models2 = hyp.predict(market, model=models[0], t=5,
   ...                               return_model=True)
   >>> [f.shape for f in reused]
   [(5, 3), (5, 3)]
   >>> any(m is models[0] for m in models2)
   False

Arguments that describe the **whole call** -- ``t`` and the ``model=`` spec
-- are validated once, before the per-group loop, so a bad horizon or an
unknown model name is reported plainly instead of being blamed on the first
group. A per-group ``ValueError`` (too short a history, duplicated times)
*is* prefixed with the group's key, and a group's warnings are always
re-emitted with its key.


Backend parity
---------------

Both rendering backends draw the same hierarchy from the same data.
Grouping, mean construction and forecasting all happen upstream of drawing,
so nothing degrades silently when you switch.

.. doctest::

   >>> with hyp.set_interactive_backend('plotly'):
   ...     plotly_bundle = hyp.plot(market, predict='Kalman', t=5,
   ...                              return_model=True)
   >>> mpl_bundle = hyp.plot(market, predict='Kalman', t=5,
   ...                       return_model=True)
   >>> (plotly_bundle['trace_metadata']['keys']
   ...  == mpl_bundle['trace_metadata']['keys'])
   True
   >>> all(bool(np.allclose(a, b))
   ...     for a, b in zip(plotly_bundle['trace_data'],
   ...                     mpl_bundle['trace_data']))
   True
   >>> [f.shape for f in plotly_bundle['predict']['forecasts']]
   [(5, 3), (5, 3), (5, 3)]

The per-level linewidth, alpha and legend rules, the continuous-hue
colouring and the per-trace forecast overlays are all applied by both
backends. Animated hierarchical plots work on both as well.


Feature names and duplicates
-----------------------------

The innermost column labels are **feature identities**, and this is the one
rule most likely to surprise you: correspondence across groups is by
**name**, not by position.

Every group must carry the same feature labels, and later groups are
permuted into the first group's order before analysis. Reordering a group's
columns therefore changes nothing:

.. doctest::

   >>> permuted = pd.DataFrame(
   ...     np.arange(8.0).reshape(2, 4),
   ...     columns=pd.MultiIndex.from_tuples(
   ...         [('Tech', 'return'), ('Tech', 'volatility'),
   ...          ('Energy', 'volatility'), ('Energy', 'return')],
   ...         names=['Sector', 'Measure']))
   >>> leaves, meta = group_columns(permuted)
   >>> [list(leaf.columns) for leaf in leaves]
   [['return', 'volatility'], ['return', 'volatility']]
   >>> leaves[1].to_numpy().tolist()
   [[3.0, 2.0], [7.0, 6.0]]

The ``Energy`` group's columns were written ``volatility, return``; its
values were moved with their labels, so ``return`` still lines up with
``return``. Matching by position instead would quietly make column order
part of the statistical model.

Groups that name **different** features -- one ticker per sector, say --
raise ``ValueError`` naming both sides:

.. doctest::

   >>> tickers = pd.MultiIndex.from_tuples(
   ...     [('Tech', 'AAPL'), ('Tech', 'MSFT'), ('Tech', 'NVDA'),
   ...      ('Energy', 'XOM'), ('Energy', 'CVX'), ('Energy', 'BP')],
   ...     names=['Sector', 'Ticker'])
   >>> by_ticker = pd.DataFrame(rng.standard_normal((20, 6)), columns=tickers)
   >>> try:
   ...     hyp.plot(by_ticker)
   ... except ValueError as error:
   ...     print(str(error).split('. ')[0])
   column-hierarchy group ('Energy',) does not have the same features as the first group ('Tech',): missing ['AAPL', 'MSFT', 'NVDA'], unexpected ['XOM', 'CVX', 'BP']

The usual fix is to make the innermost level shared **measurements**
(``return``, ``volatility``) rather than per-group identifiers, moving the
identifiers up a level where they belong.

`hypertools.predict` groups the frame the same way, so it inherits the
whole rule -- the refusal and the permutation alike. Before 1.1 it had no
grouping to disagree with and forecast the flattened frame instead:

.. doctest::

   >>> try:
   ...     hyp.predict(by_ticker, model='Kalman', t=2)
   ... except ValueError as error:
   ...     print(str(error).split('. ')[0])
   column-hierarchy group ('Energy',) does not have the same features as the first group ('Tech',): missing ['AAPL', 'MSFT', 'NVDA'], unexpected ['XOM', 'CVX', 'BP']
   >>> [list(f.columns) for f in hyp.predict(permuted, model='Kalman', t=2)]
   [['return', 'volatility'], ['return', 'volatility']]

That message's escape-hatch snippet is spelled for `hypertools.plot`; from
`hypertools.predict`, take the same first step and forecast the leaves.

The escape hatch, and what it costs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If slot *i* really is the same feature in every group, you can say so
deliberately -- but the tool for that is a **lower-level escape hatch, not
positional hierarchy plotting**:

.. doctest::

   >>> leaves, meta = group_columns(by_ticker,
   ...                              feature_correspondence='position')
   >>> [list(leaf.columns) for leaf in leaves]
   [['AAPL', 'MSFT', 'NVDA'], ['XOM', 'CVX', 'BP']]
   >>> bundle = hyp.plot([leaf.to_numpy() for leaf in leaves],
   ...                   return_model=True)
   >>> len(bundle['trace_data'])
   2
   >>> bundle['trace_metadata'] is None
   True
   >>> [f.shape for f in hyp.predict(
   ...     [leaf.to_numpy() for leaf in leaves], model='Kalman', t=2)]
   [(2, 3), (2, 3)]

That is **not** equivalent to ``hyp.plot(df)``. It draws a plain list of
datasets, so there are no per-level mean traces (2 traces, not 3), no
hierarchy linewidth/alpha/legend styling, and no ``trace_metadata`` at all.
There is no hierarchy-preserving positional mode in 1.1.

Note also that ``align=`` does **not** substitute for correspondence. It
aligns the resulting spaces, but by then the reduction has already
interpreted arbitrary column positions as corresponding inputs.

Duplicate labels
~~~~~~~~~~~~~~~~~

Flattening can leave two identical labels inside one group -- two share
classes of one issuer, a repeated sensor. That is **permitted**: every
column survives, the group is neither split nor merged, and both plotting
and forecasting work normally.

.. doctest::

   >>> share_classes = pd.MultiIndex.from_tuples(
   ...     [('Alphabet', 'price'), ('Alphabet', 'price'),
   ...      ('Alphabet', 'volume'),
   ...      ('Berkshire', 'price'), ('Berkshire', 'price'),
   ...      ('Berkshire', 'volume')],
   ...     names=['Issuer', 'Measure'])
   >>> dupes = pd.DataFrame(rng.standard_normal((20, 6)),
   ...                      columns=share_classes)
   >>> leaves, meta = group_columns(dupes)
   >>> [leaf.shape[1] for leaf in leaves]
   [3, 3]
   >>> [np.asarray(leaf).shape for leaf in leaves]
   [(20, 3), (20, 3)]
   >>> [f.shape for f in hyp.predict(dupes, model='Kalman', t=1)]
   [(1, 3), (1, 3)]
   >>> type(hyp.plot(dupes)).__name__
   'Figure'

Such labels are matched across groups by ``(label, occurrence)`` -- the
first ``price`` to the first ``price``, the second to the second -- so each
group needs the same *number* of them.

Duplicates **across** different groups were always fine and stay fine.
``('Market', 'Tech', 'X')`` and ``('Market', 'Energy', 'X')`` are two
separate leaves that both happen to carry a feature called ``X``:

.. doctest::

   >>> shared_names = pd.DataFrame(
   ...     rng.standard_normal((20, 4)),
   ...     columns=pd.MultiIndex.from_tuples(
   ...         [('Market', 'Tech', 'X'), ('Market', 'Tech', 'Y'),
   ...          ('Market', 'Energy', 'X'), ('Market', 'Energy', 'Y')],
   ...         names=['Market', 'Sector', 'Measure']))
   >>> leaves, meta = group_columns(shared_names)
   >>> meta['leaf_keys']
   [('Market', 'Tech'), ('Market', 'Energy')]

.. _hierarchy-group-order:

Group order, which is a different question
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above is about correspondence **within** a group. The order of
the **groups** is a separate matter, and it is *not* neutralised: groups
become datasets, ``reduce=`` row-stacks every dataset and fits **one**
model on the stack (that is what makes the space shared), so group order is
row order in that stack. A reducer whose fit depends on row order will
embed a block-reordered frame differently -- and the default,
``IncrementalPCA``, is one of them, because it fits by ``partial_fit`` over
successive minibatches.

.. doctest::

   >>> blocks = pd.MultiIndex.from_tuples(
   ...     [('Market', sector, measure)
   ...      for sector in ('Tech', 'Energy', 'Health', 'Fin')
   ...      for measure in ('return', 'volatility', 'momentum', 'spread',
   ...                      'turnover')],
   ...     names=['Market', 'Sector', 'Measure'])
   >>> wide = pd.DataFrame(
   ...     np.cumsum(np.random.default_rng(0).standard_normal((40, 20)),
   ...               axis=0),
   ...     columns=blocks)
   >>> reordered = wide[[(top, sector, measure)
   ...                   for sector in ('Fin', 'Health', 'Energy', 'Tech')
   ...                   for top, _, measure in blocks[:5]]]
   >>> def drawn(frame, reducer):
   ...     bundle = hyp.plot(frame, reduce=reducer, return_model=True)
   ...     return dict(zip([tuple(k)
   ...                      for k in bundle['trace_metadata']['keys']],
   ...                     bundle['trace_data']))
   >>> def geometry(traces):
   ...     # distances between the plotted points: what "the same picture"
   ...     # means, and immune to a component's sign flipping
   ...     pts = np.vstack([traces[k] for k in sorted(traces)])
   ...     return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
   >>> def moved(reducer):
   ...     before, after = drawn(wide, reducer), drawn(reordered, reducer)
   ...     d0, d1 = geometry(before), geometry(after)
   ...     return float(np.abs(d0 - d1).max() / d0.max())
   >>> bool(moved('PCA') < 1e-8)             # same embedding
   True
   >>> bool(moved('IncrementalPCA') > 1e-6)  # a different one
   True

On that fixture, ``IncrementalPCA`` and ``TSNE`` produced **different
embeddings** after the blocks were reordered, while ``PCA``,
``TruncatedSVD``, ``FactorAnalysis``, ``Isomap`` and ``SpectralEmbedding``
preserved the embedding up to numerical and sign equivalence -- as does a
within-group column permutation under every one of them.

No percentage is quoted here on purpose. How far a figure moves depends on
the data, the scikit-learn version, the BLAS build and the platform, and for
a PCA-family embedding a flipped component sign is the same picture with a
large raw-coordinate difference. What is contractual is the *direction* of
the asymmetry, and that is what
``tests/test_hierarchy_group_order_and_pipeline.py`` pins.

This is a property of the **shared reduction space**, not of hierarchies:
``hyp.plot([A, B, C])`` and ``hyp.plot([C, B, A])`` differ in exactly the
same way, and did before 1.1.

It is documented rather than worked around for two reasons, neither of which
is "existing figures would move". Imposing a canonical group order would
mean inventing a total ordering over arbitrary hierarchy labels -- mixed
types, missing values, tuples of both -- and every choice there is arbitrary.
And it would make a labelled hierarchy behave differently from the
equivalent list of datasets, which has always been positional, adding one
more semantic distinction between two spellings of the same plot. Pass
``reduce='PCA'`` when block order must not matter; it is an explicit,
one-word remedy.


See also
---------

- :doc:`pipeline_order` -- where hierarchy expansion and mean construction
  sit relative to the canonical pipeline.
- :doc:`api` -- ``hypertools.plot``'s ``x``/``predict=``/``return_model=``
  entries and ``hypertools.predict``'s ``data``/``model=`` entries carry the
  same rules in reference form.
- :doc:`tutorials` -- the weather notebook works the bold-means /
  faint-leaves structure end-to-end. It builds it as an explicit list of
  loops rather than a row MultiIndex, precisely so that a continuous
  per-point ``hue=`` survives; this page's `Hue over a hierarchy`_ section
  explains when the MultiIndex path can carry one instead.
