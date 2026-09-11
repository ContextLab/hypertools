# CHANGELOG drafts from fixers (integrate into "Fixed during the release review")

## W2
- **`yahoo:` intraday bars keep their timestamps.** `interval='1h'` put every bar at midnight, so `hyp.predict` rejected the index. Intraday bars now carry their time, tz-aware in the exchange's timezone.
- **A fitted `Normalizer` accepts 1-D data.** A 1-D array, Series or list of numbers is one column in both fit and transform.
- **`set_autoinstall` handles are quiet at exit.** A live handle no longer prints "Exception ignored" at shutdown, and a re-entered handle keeps its call order.
- **Offline errors say what happened.** A missing 25+ character bare name lists the full resolution chain, and a cached copy that fails to parse raises `HypertoolsIOError` naming the file.
- **Extensionless remote `.npz` reports the `trust=True` error** instead of a parquet one.
- **`[density3d]` needs `scikit-image>=0.25.0`**, the first release with Python 3.13 wheels.
- **`load()`'s TypeError names polars frames.**
- **0-255 colour lists raise `ValueError`.** `palette=[[255,128,0],...]` used to be silently read as a data matrix; the error says to divide by 255 or pass a DataFrame.
- **`font='Noto Sans'` works in a fresh process.** The bundled faces are registered before the lookup.

## main (me)
- **A `hue=` surface matches the points beneath it.** Each hull vertex blended every point in its dataset with inverse-squared-distance weights; in 3-D the many distant points outweighed the near ones, so the hull took the dataset's washed-out mean colour. Vertices now blend their nearest points, on both backends.

## W1
- **Regular calendar data are forecast on their own calendar.** Business-day, month-start, weekly, quarterly and tz-aware daily indexes, and `PeriodIndex` data, are fitted on their own rows and forecast onto the next business days, month starts or periods. Before, business-day bars were interpolated onto calendar days and forecast onto weekends, month starts drifted, a fall DST change duplicated a day, and periods came back as timestamps. `step=` also accepts `'B'` and `'MS'`.
- **A fitted forecaster works across index kinds again.** A model fitted on an array and reused on dated rows, or the reverse, raised an error about `step`; it now steps in the new data's own units, as in 1.0.
- **ARIMA's minimum history includes `seasonal_order`.** A short seasonal fit now gets the "needs N observations" message instead of a bare `IndexError` or `LinAlgError`.
- **Time warnings appear once, and only when they apply.** A stacked panel warns "not sorted" once per call instead of three times, and an explicit `step=` on evenly spaced data no longer calls them irregular.
- (W1 notes: CHANGELOG lines 17-21 say one future step is always the median gap -> must update.)

## W4a
- **Forecasts and truth keep their own dataset's style with `'o-'`.** Each marker-plus-line dataset was drawn as two artists, so three datasets' forecasts came out red, red, green.
- **`ndims=1` `truth=` takes one column of values per trace.** A two-column truth used its first column as x, which stretched the date axis back to 1970. It now raises `ValueError`.
- **A one-column trace is drawn against its row index.** Antialiasing put a 40-row line at x 0..936, squashing its forecast 24x. `axis_scale='data'` also gave the value range to x.
- **Animated forecasts on two-column data no longer crash** with "too many values to unpack".
- **`forecast_fmt` markers mark only the forecast steps**, not all ~900 smoothed vertices.
- **Marker-only `hue=`/`cluster=` always refuses forecasts and warns**, even when the category count equals the dataset count.
- **`legend_colors=` accepts one colour per data entry beside forecast and truth entries.** A wrong count now closes the figure it opened.
- **plotly date axes show the same dates in every time zone.** Numeric dates were drawn in the viewer's local time.
- **`xlim=(None, date)` works on date axes**; the open side takes the data bound.
- **`panels=` accepts `forecast_trail=`** alongside `predict=`.
- **`transform=` fixes.** A bare array is one dataset instead of crashing, and a DataFrame with its own index no longer gives all-zero forecasts.
- **The 'truth' legend key is gray when truths span several colours**, instead of always showing dataset 0's colour.
- **A shuffled time index is drawn in time order**, so the forecast joins the end of the line.
- **`ndims=1` date ticks no longer collide**; matplotlib now uses concise date labels.
- (main) **A polars `transform=` frame works.** It raised `SchemaError` in the display scaling, with or without `predict=`.

## W7
- **Aligner classes accept arrays.** `HyperAlign().fit(xs).transform(ys)` on a list of NumPy arrays, or on a single array, raised "Unsupported datatype". The aligners now accept anything `hyp.align` does and return each dataset in its input's form.
- **`alignment_score(metric='dispersion')` rejects all-constant datasets.** Datasets each constant at a different value used to score exactly 1.0; they now raise, like `'isc'`.
- **Rows a `manip=` stage empties stop the pipeline at that stage.** A trailing `Smooth(center=False)` no longer triggers misleading PPCA imputation warnings or sklearn NaN errors. The error names the stage and suggests `min_periods=1`.
- **`hyp.plot(x, pipeline=p)` draws the pipeline's clusters.** A fitted trailing cluster step colours the figure with the fit figure's colours, where before it was dropped silently.
- **Align, impute and manip warnings point at your own line**, so deprecated spellings no longer go unseen.

## W10
- **A flat cluster spec no longer drops model parameters without a word.** `cluster={'model': 'KMeans', 'n_clusters': 4, 'random_state': 0}` ignored `random_state`, so the clusters changed on every call. Parameters go under `'kwargs'`, and any top-level key other than the `'n_clusters'` shortcut now raises `ValueError` naming it.
- **`legend=False` now wins over `names=`.** Passing dataset names used to force the legend on even when `legend=False` was given, on both backends.

## W3 batch 1
- **Plotly markers sit on the observations.** 'o-' and `forecast_fmt='ro:'` put a dot on every smoothed vertex, drawing the line as a solid tube; now only the samples are marked (in animations, the frame-grid vertex nearest each sample). A continuous hue with 'o-' in 1-D/2-D now shows its markers.
- **Plotly hue transparency.** Hue markers now honour `alpha=`, and translucent 3-D lines keep their colour instead of washing out to cyan.
- **Plotly hover labels name what you point at.** They read "trace 0"; every data trace now carries its legend label, a lone unlabelled dataset shows only its coordinates, and animated legends no longer grow entry by entry.
- **Plotly subplot cells.** Colorbars no longer land on the next cell, and an untitled call keeps the cell's title. A dimensionality mismatch raises a clear error, your own traces are left untouched, and a plotly `ax=` implies the plotly backend.
- **Plotly `frame_kwargs=`, `zoom=` and legend position.** `frame_kwargs=` styles the frame, static figures ignore `zoom=`, and `legend_kwargs={'x':0,'y':1}` anchors at that corner.
- **Plotly 1-D animations raise like matplotlib's.** Serial `FrameContext.window_bounds` now report the comet head.

## W4b
- **Dict-list palettes colour marker plots.** `fmt='o'` ignored a per-dataset list of {category: color} dicts.
- **Composing into `ax=` no longer repeats a palette colour.** 'hls' drawn 2 + 2 now gives the four-colour 'hls' set; `colors` in the bundle and the colorbar show the colours actually drawn.
- **`panels=` splits a plain `legend_colors=` list per panel.**
- **Two-column data draws into a 2-D `ax=`** instead of raising "the plot is 3D".
- **Per-dataset and nested `labels=` survive `hue=`/`cluster=`** instead of crashing; label arrays and Series are accepted.
- **An explicit `marker=` wins over the fmt marker on matplotlib.**
- **Markers on a smoothed line sit at the samples,** static and animated.
- **Continuous-hue markers honour `alpha=` on matplotlib.** (fix CHANGELOG ~:712 'marker colours deliberately do not' claim)
- **The NaN-hue warning counts observations** and points at the caller's line.
- **Label connectors and box edges are visible** on matplotlib.
- **Caller-axes and panel titles and axis labels use Noto Sans.**
- **A nested-list input's legend names its outer groups.**
- **Cluster and integer-hue line legends list categories in order** (0, 1, 2).

## W11
- **A dict model spec with a flat parameter raises instead of silently running defaults.** `{'model': 'PCA', 'whiten': True}` now raises `ValueError` naming the key and showing the corrected `'kwargs'` spec, in `reduce` (including streaming), `manip`, `align` (and `tools.align`), `impute`, `Pipeline`, `apply_model` and `text2mat`. Outer `**kwargs` next to a dict spec now reach `manip`/`align` models, and a spec's `'args'` now reach streaming and `text2mat` models.
