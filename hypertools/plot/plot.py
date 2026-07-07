#!/usr/bin/env python
import copy
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from .._shared.helpers import *
from .._shared.params import default_params
from ..tools.analyze import analyze
from ..cluster.cluster import cluster as clusterer, mixture_models
from .colors import mat2colors, colors2groups, get_palette_colors, continuous_colormap
from ..reduce.reduce import reduce as reducer
from ..tools.format_data import format_data
from .matplotlib_backend import _draw
from .backend import manage_backend
from .plotly_backend import resolve_backend
from .animate import _save_animation, _SVGFrameCollector, _save_animated_svg
from .surface import broadcast_surface, normalize_surface_arg
from .density import broadcast_density, normalize_density_arg
from .trails import broadcast_trail_flag
from .multiindex import expand_multiindex, build_multiindex_styles
from .morph import resolve_morph_rotations
from .fonts import resolve_font


# GH #206: the subset of mpl_kwargs keys `plotly_backend.plotly_draw` (and
# its trail/forecast helpers) actually reads and maps onto a plotly trace
# property, via the existing `_resolve_fmt`/`_trace_name` machinery -- see
# every `tkwargs.get(...)` call in `hypertools/plot/plotly_backend.py`.
# Any OTHER kwarg (arbitrary matplotlib-style passthrough, e.g. `zorder=`,
# `markeredgecolor=`, `dashes=`) has no plotly equivalent and is silently
# unusable there; `plot()` warns (once, naming every such kwarg) rather
# than silently dropping it with no feedback at all.
_PLOTLY_MAPPED_KWARGS = frozenset(
    {'color', 'alpha', 'linewidth', 'markersize', 'marker', 'linestyle',
     'label'})


def _apply_extra_kwargs(kwargs_list, extra):
    """Merge arbitrary extra matplotlib-style kwargs (GH #206's `**kwargs`
    passthrough) into every per-dataset dict in `kwargs_list`, IN PLACE.

    Deliberately bypasses `parse_kwargs`'s per-dataset list/tuple
    broadcasting entirely: `extra`'s values are applied VERBATIM,
    identically, to every dataset, whatever their type -- including a
    list/tuple value (e.g. `dashes=(4, 2)`), which is a single matplotlib
    property VALUE, not a per-dataset list of separate values. Running
    such a value through `parse_kwargs`'s broadcast machinery (designed
    for hypertools' own per-dataset style kwargs -- color/marker/
    linestyle/etc., where a list genuinely means "one value per dataset")
    would either misinterpret it (if its length happened to equal the
    dataset count) or now raise a spurious ``ValueError`` (since
    `parse_kwargs` was fixed, also for GH #206, to raise rather than
    silently drop a length that does NOT match the dataset count) -- both
    wrong for a kwarg whose natural value just happens to be tuple/list
    shaped. Callers needing genuine PER-DATASET control over an arbitrary
    property can already reach for the dedicated, per-dataset-aware kwargs
    (`color`/`marker`/`linestyle`/`markersize`/`linewidth`).

    A key already present in a given dataset's dict (set by a named
    parameter, e.g. `color=`, or by internal styling logic, e.g.
    MultiIndex/mixture-cluster `alpha`, `legend=`'s `label`, `explore=`'s
    `picker`) is left untouched -- named/internal styling always wins over
    a same-named extra kwarg.
    """
    if not extra:
        return
    for d in kwargs_list:
        for k, v in extra.items():
            if k not in d:
                d[k] = v


def _resolve_animate_mode(animate, n_datasets):
    """Resolve ``animate=`` for ``animate='morph'`` support (Hungarian
    point-cloud morphs between datasets, maintainer request): `animate` may
    be a single GLOBAL mode (``False``/``True``/``'parallel'``/``'spin'``/
    ``'serial'``/``'morph'``, unchanged from before) OR, ONLY for morph, a
    per-dataset list with ``'morph'``/``None``/``False`` entries (one per
    FINAL -- post cluster/hue-reshape -- dataset, matching `n_datasets`):
    ``'morph'``-tagged datasets join the morph sequence IN LIST ORDER;
    untagged datasets render as static (unanimated) backdrops.

    Returns
    -------
    (mode, morph_tags)
        `mode` is what every backend actually receives: the raw scalar
        `animate` unchanged, or ``'morph'`` if a list was given. `morph_tags`
        is ``None`` for every non-morph mode, or a list of `n_datasets` bool
        (``True`` where that dataset joins the morph sequence) whenever
        `mode` is ``'morph'`` (scalar ``animate='morph'`` tags every
        dataset).

    Raises
    ------
    ValueError
        A list entry is not ``'morph'``/``None``/``False``; a list's length
        doesn't match `n_datasets`; or fewer than 2 datasets end up tagged
        ``'morph'`` (scalar or list form).
    """
    if isinstance(animate, (list, tuple)):
        tags = []
        for item in animate:
            if item in (None, False):
                tags.append(False)
            elif item == "morph":
                tags.append(True)
            else:
                raise ValueError(
                    "animate list entries must be 'morph' or None/False "
                    "(per-dataset animate lists only support tagging "
                    f"datasets for animate='morph'); got {item!r}."
                )
        if len(tags) != n_datasets:
            raise ValueError(
                f"animate list has {len(tags)} entries but there are "
                f"{n_datasets} datasets to plot; pass a single mode to "
                "apply it to every dataset, or a list matching the "
                "dataset count."
            )
        if sum(tags) < 2:
            raise ValueError(
                "animate='morph' (per-dataset list form) requires at "
                f"least 2 datasets tagged 'morph'; got {sum(tags)}."
            )
        return "morph", tags
    if animate == "morph":
        if n_datasets < 2:
            raise ValueError(
                "animate='morph' requires at least 2 datasets to morph "
                f"between; got {n_datasets}."
            )
        return "morph", [True] * n_datasets
    return animate, None


@manage_backend
def plot(
    x,
    fmt="-",
    marker=None,
    markers=None,
    markersize=None,
    linewidth=None,
    linestyle=None,
    linestyles=None,
    color=None,
    colors=None,
    palette="hls",
    hue=None,
    labels=None,
    legend=None,
    colorbar=None,
    title=None,
    size=None,
    elev=10,
    azim=-60,
    ndims=3,
    reduce="IncrementalPCA",
    cluster=None,
    align=None,
    normalize=None,
    manip=None,
    pipeline=None,
    impute=None,
    resample=None,
    n_clusters=None,
    predict=None,
    t=10,
    save_path=None,
    animate=False,
    duration=30,
    tail_duration=2,
    rotations=1,
    zoom=1,
    chemtrails=False,
    precog=False,
    bullettime=False,
    frame_rate=30,
    morph_samples=None,
    interactive=False,
    explore=False,
    backend="auto",
    mpl_backend="auto",
    show=True,
    transform=None,
    vectorizer="CountVectorizer",
    semantic="LatentDirichletAllocation",
    corpus="wiki",
    ax=None,
    frame_kwargs=None,
    stream_init=10000,
    stream_chunk=100,
    stream_max=None,
    stream_window=None,
    return_model=False,
    surface=None,
    density=None,
    font=None,
    **kwargs,
):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame, String, Geo or mixed list
        Data for the plot. The form should be samples (rows) by features (cols).

        A DataFrame with a row **MultiIndex** (``x.index.nlevels >= 2``) is
        handled specially (GH #95): it is expanded, BEFORE the format_data/
        analyze/reduce pipeline runs, into one "leaf" dataset per unique full
        index combination (level order as given), so leaves flow through
        normalize/reduce/align exactly like any other list of datasets. AFTER
        that pipeline transforms them, one MEAN trajectory is computed (in
        the transformed/reduced space) for every unique value-combination of
        each non-leaf level -- from the deepest such level up to the top
        (outermost) level -- and appended as additional traces. For levels
        numbered 0 (top) through L-1 (leaf), where L = ``x.index.nlevels``,
        a trace whose deepest represented level is ``level_idx`` (``L - 1``
        for a leaf; ``k`` for a mean over the prefix ``levels[0:k+1]``)
        gets:

        - ``linewidth = 1 + (L - 1 - level_idx)`` -- i.e. 1 plus the number
          of levels averaged over: leaves are always 1, and each level
          higher up is one point thicker, so the TOP-level means are the
          thickest (``L``).
        - ``alpha = min(1.0, 1 / (level_idx + 1) + 0.2)`` -- leaves are the
          most transparent, the top-level mean is fully opaque (1.0), with
          intermediate levels smoothly in between.
        - ``color`` assigned purely by the trace's TOP-level index value
          (from `palette`, in order of that value's first appearance) --
          every leaf and every mean sharing the same top-level value shares
          one color.

        Example (2 levels, e.g. ``(condition, subject)``): leaves get
        lw=1, alpha=0.7; the condition-means (the only non-leaf level, which
        is also the top level here) get lw=2, alpha=1.0, and are the only
        traces with a legend label. Example (3 levels, e.g.
        ``(group, condition, subject)``): leaves lw=1, alpha=1/3+0.2≈0.533;
        (group, condition)-means lw=2, alpha=0.7; group-means (top level)
        lw=3, alpha=1.0.

        `legend` is automatically populated with one entry per unique
        top-level index value: only each top-level mean trace carries that
        label; every other trace (all leaves, and any intermediate-level
        means) is drawn with ``label='_nolegend_'`` (excluded from the
        legend, matching the convention `predict=`'s forecast overlay
        already uses). If `linestyle`/`linestyles` is given as a list, its
        length MUST equal the number of unique top-level index values (one
        style per top-level group, applied to every trace in that group);
        a mismatched length raises ``ValueError``. Any `color`/`colors`/
        `linewidth` kwarg is ignored (with a ``UserWarning``) since
        MultiIndex grouping owns those. `hue=` is superseded with a
        ``UserWarning`` (MultiIndex grouping takes precedence); `cluster=`/
        `n_clusters=` raise ``ValueError`` (both would fight the MultiIndex
        color assignment) -- reset the index first
        (``df.reset_index(drop=True)``) to cluster instead. `predict=` also
        raises ``ValueError`` when combined with MultiIndex expansion:
        forecasts are computed one-per-leaf BEFORE the per-level mean
        traces are appended, so the leaf count no longer matches the final
        trace count -- reset the index first to use `predict=`. Row
        averaging assumes member leaves align by row POSITION at each
        timepoint; leaves of unequal length are averaged over their
        overlapping prefix (the shortest member's length), with a single
        ``UserWarning`` per affected group (deduplicated even when a
        3+-level tree causes multiple groupings to share members). Works
        with both static and animated plots and both rendering backends,
        since the expansion happens upstream of drawing. A MultiIndex on
        the COLUMNS (as opposed to the row index) is unrelated to this and
        is unaffected -- it is handled by the existing column-formatting
        pipeline in `hypertools.tools.format_data`/`hypertools.tools.df2mat`.
        A single-level (or default `RangeIndex`) DataFrame, or a plain
        array/list input, is completely unaffected by any of the above.

        Expansion is ONLY applied when a single bare DataFrame is passed as
        `x`. If `x` is a LIST containing one or more MultiIndex DataFrames
        (whether alone or mixed with arrays/other DataFrames), the
        MultiIndex is silently treated as a flat index on each such element
        by the normal list-of-datasets pipeline -- a ``UserWarning`` is
        raised naming each offending element's position in the list.

    fmt : str or list of strings
        A list of format strings.  All matplotlib format strings are supported.

        A format string combining a LINE style with a MARKER (e.g. 'o-',
        's--') gets the SAME connecting-line smoothing/interpolation a
        pure line style (e.g. '-') gets (GH #141 follow-up; previously
        marker+line combos silently skipped interpolation, drawing
        straight/unsmoothed segments between raw points). The line and
        markers are drawn as two separate artists on the STATIC (non-
        animated) matplotlib backend: the smoothed/interpolated line, plus
        markers at the TRUE (pre-interpolation) sample points -- so
        markers never drift onto the dense interpolated curve. Pure line-
        only and pure marker-only styles are unaffected (still one
        artist, as before). For ANIMATED matplotlib plots, and for the
        plotly backend (static or animated -- it always draws a marker+
        line combo as a single 'lines+markers' trace), a marker+line
        combo's line is likewise now smoothed (the interpolation gate fix
        is backend-agnostic), but its markers currently render at the
        same (interpolated) points as the line rather than only the
        original samples -- splitting those into separate artists/traces
        for every animated style and for plotly is a follow-up.

    linestyle(s) : str or list of str
        A list of line styles

    marker(s) : str or list of str
        A list of marker types

    markersize : int or float
        Size of the markers in points (default: matplotlib's 6.0). Applies
        to both backends.

    linewidth : int or float
        Width of plotted lines in points (default: matplotlib's 1.5 for
        static plots, 1 for animations). Applies to both backends.

    color(s) : str or list of str
        A list of colors

    **kwargs : any other matplotlib-style keyword argument
        GH #206: any keyword argument that isn't one of `plot()`'s own
        named parameters above is passed straight through to each drawn
        artist -- e.g. `zorder=3`, `alpha=0.5`, `dashes=(4, 2)`,
        `markeredgecolor='k'`. Applied VERBATIM, identically, to every
        drawn dataset -- unlike `color`/`marker`/`linestyle`/etc. (see
        below), an extra kwarg's value is NEVER interpreted as "one entry
        per dataset" even if it happens to be a list/tuple (e.g.
        `dashes=(4, 2)` is a single dash-pattern VALUE, not per-dataset
        values `4` and `2`) -- so there is no per-dataset form for an
        extra kwarg; use one of the dedicated per-dataset-aware kwargs
        (`color`/`marker`/`linestyle`/`markersize`/`linewidth`) for that.
        Merged in AFTER the named style kwargs are resolved, so an
        explicit named kwarg (or internal styling logic, e.g. MultiIndex/
        mixture-cluster `alpha`, `legend=`'s `label`, `explore=`'s
        `picker`) always wins on a naming collision. On the matplotlib
        backend an unrecognized property surfaces matplotlib's OWN error
        (e.g. ``AttributeError: ... has no property ...``) -- no separate
        whitelist/validation is applied here. On the plotly backend, only
        a small subset maps onto an actual trace property (`color`,
        `alpha`, `linewidth`, `markersize`, `marker`, `linestyle`,
        `label`); anything else is ignored with a ``UserWarning`` naming
        every unmapped kwarg (rather than raising, since plotly's trace
        objects were never going to support the same kwarg surface as
        matplotlib).

        Every per-dataset list/tuple-valued kwarg `plot()` itself
        broadcasts this way (`color`/`colors`, `marker`/`markers`,
        `linestyle`/`linestyles` -- NOT the generic `**kwargs` passthrough
        above, which is never broadcast) has its length validated (GH
        #206): it MUST equal the number of datasets being drawn (the
        FINAL count, after any `cluster`/`hue`/MultiIndex reshaping), or
        ``ValueError`` is raised naming the kwarg, the length actually
        given, and the required dataset count. Previously a mismatched-
        length list silently degraded to `None` for every dataset with no
        error or warning at all.

    palette : str
        A matplotlib or seaborn color palette

    hue : list or numpy array
        Values used to color the plot. Accepts categorical labels (one per
        observation; grouped and colored by category), continuous numeric
        values (mapped through the palette; combined with a line format
        this produces multicolored lines whose color varies continuously
        along each trajectory), or a 2D matrix with one row per observation
        (e.g. mixture proportions or model weights; colors are blended per
        observation). To label a subset of points categorically, use None
        entries (i.e. ['a', None, 'b', 'a']).

    labels : list
        A list of labels for each point. Must be dimensionality of data (x).
        If no label is wanted for a particular point, input None.

        Supported on BOTH backends (GH #205/#F3): matplotlib draws these as
        `ax.annotate` call-outs; plotly draws the same points as
        `layout.scene.annotations` (3D) or `layout.annotations` (2D), at
        the same data coordinates, honoring the resolved `font=` (see
        below) the same way the legend/colorbar/title do.

    legend : list or bool
        If set to True, legend is implicitly computed from data. Passing a
        list will add string labels to the legend (one for each list item).

    colorbar : bool or dict
        If True, draws a colorbar reflecting the color mapping in use
        (GH #100). For a continuous 1D `hue` (or continuous `hue` combined
        with a line format, which produces multicolored lines), the
        colorbar is a continuous `ScalarMappable` spanning the ACTUAL
        `hue` value range, using the SAME palette as the lines/markers.
        For discrete groups (categorical `hue`, `cluster`/`n_clusters`, or
        a plain list of datasets with no `hue`/`cluster`), the colorbar is
        segmented (one BoundaryNorm-style block per group), with tick
        labels taken from the legend labels/group names (``1..n`` if no
        names are available). Pass a dict for finer control:
        ``{'label': str, 'ticks': [...], 'location': 'right'|'left'|'top'|
        'bottom'}`` (all keys optional; ``location`` defaults to
        ``'right'``, the same side as the legend -- when both a legend and
        a right-side colorbar are shown, the figure is widened so neither
        is clipped or overlaps the other). Raises ``ValueError`` if
        requested with no color mapping available at all (e.g. a single
        dataset with no `hue`/`cluster`). Default None (no colorbar).

    title : str
        A title for the plot

    font : None, str, or matplotlib.font_manager.FontProperties
        Controls the font used for every text surface hypertools draws,
        on BOTH backends (GH #205): point annotations (`labels=`), the
        legend, colorbar tick labels/axis label, and the plot title -- on
        matplotlib via `ax.annotate`/`ax.legend`/etc.; on plotly via
        `layout.scene.annotations`/`layout.annotations`, the legend,
        colorbar title/ticks, and the plot title.

        - `None` (default): AUTO-DETECT. If all text hypertools is about
          to draw is plain ASCII, this is a no-op (each backend's default
          font is used, exactly as before this kwarg existed). If any
          text contains non-ASCII characters (e.g. Japanese/Chinese/
          Korean/Cyrillic labels), hypertools scans installed fonts for
          one whose character map covers every character needed and uses
          it automatically -- so ``labels=["いち", "に", "さん"]`` "just
          works" without tofu (empty boxes) or missing-glyph warnings, as
          long as SOME covering font is installed (on most Linux distros,
          ``apt-get install fonts-noto-cjk`` provides one; macOS/Windows
          usually already have one, e.g. Hiragino Sans/Yu Gothic). If no
          installed font covers the needed characters, a single
          ``UserWarning`` is raised naming a few of the missing characters
          and rendering falls back to the backend's default (tofu).
        - `str`: either the name of an installed font FAMILY (e.g.
          ``'Noto Sans CJK JP'``), or a path to a ``.ttf``/``.otf``/
          ``.ttc`` font FILE (existing paths are detected automatically,
          relative or absolute). Raises ``ValueError`` if the string is
          neither a resolvable family name nor an existing file.
        - `matplotlib.font_manager.FontProperties`: used as-is.

        Backend semantics differ because matplotlib and plotly resolve
        fonts differently: matplotlib accepts a font FILE and sets a
        `FontProperties` object on each `Text` artist individually
        (exact glyph outlines, embedded at save time). plotly (rendered
        by a browser, or by Chromium via kaleido for static image export)
        only understands FAMILY NAMES -- there is no way to point it at a
        specific font file -- so hypertools takes the resolved font's
        family name (`FontProperties.get_name()`), wraps it in a small
        fallback chain (``'"<name>", "Noto Sans CJK JP", sans-serif'``),
        and sets it as `layout.font.family`, which every plotly text
        surface hypertools creates inherits unless it overrides its own
        `font.family` (none do, after this change). Static plotly image
        export (`save_path=...png/.jpg` etc., via kaleido) still depends
        on the exporting machine's OS having a font that actually covers
        the requested family/characters -- unlike matplotlib, hypertools
        cannot embed a specific font file into a plotly export.

    size : list
        A list of [width, height] in inches to resize the figure

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    manip : model spec or None
        A `hypertools.manip` spec (a registry name, dict spec, class/
        instance, or a `list` chaining several -- see
        `hypertools.manip.manip.manip`), run at the canonical `manip` stage
        position (GH #153): FIRST, before `normalize`/`reduce`/`align`/
        `cluster` -- e.g. ``hyp.plot(data, manip=[{'model': 'Smooth',
        'kwargs': {'kernel_width': 25}}, {'model': 'Resample', 'kwargs':
        {'n_samples': 1000}}], align={'model': 'HyperAlign'},
        reduce='UMAP')`` runs the whole cross-module pipeline in one call
        (GH #275). `resample=` (below) is independent sugar for a single
        `Resample` step applied BEFORE this stage's data reaches it (so
        resample sugar always runs first when both are given). Mutually
        exclusive with `pipeline=` (default: None).

    pipeline : hypertools.Pipeline or None
        A previously-FITTED `Pipeline` (e.g. from
        ``hyp.analyze(data, ..., return_model=True)`` or this function's
        own `return_model=True` bundle's `'pipeline'` key) to apply to `x`
        via `.transform` instead of fitting new `manip`/`normalize`/
        `reduce`/`align`/`cluster` models (GH #227) -- e.g. fit on dataset
        A via ``p = hyp.analyze(A, manip='Smooth', reduce='PCA',
        align='HyperAlign', return_model=True)[1]`` and reuse those exact
        fitted parameters on a structurally-identical dataset B via
        ``hyp.plot(B, pipeline=p)``. Mutually exclusive with `manip=`/
        `normalize=`/`reduce=`/`ndims=`/`align=`/`cluster=` (each must be
        left at its default) -- passing both raises `ValueError` naming the
        conflicting kwarg(s). `resample=` is still applied (as sugar, before
        `pipeline.transform` runs) since it is not one of the stage kwargs
        the fitted `Pipeline` itself covers (default: None).

    reduce : str or dict
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS. Can be
        passed as a string, but for finer control of the model parameters, pass
        as a dictionary, e.g. reduce={'model' : 'PCA', 'params' : {'whiten' : True}}.
        See scikit-learn specific model docs for details on parameters supported
        for each model.

    ndims : int
        An `int` representing the number of dims to reduce the data x
        to. If ndims > 3, will plot in 3 dimensions but return the higher
        dimensional data. Default is None, which will plot data in 3
        dimensions and return the data with the same number of dimensions
        possibly normalized and/or aligned according to normalize/align
        kwargs.

    align : str or dict or False/None
        If str, either 'hyper' or 'SRM'.  If 'hyper', alignment algorithm will be
        hyperalignment. If 'SRM', alignment algorithm will be shared response
        model.  You can also pass a dictionary for finer control, where the 'model'
        key is a string that specifies the model and the params key is a dictionary
        of parameter values (default : 'hyper').

    cluster : str or dict or False/None
        If cluster is passed, HyperTools will perform clustering using the
        specified clustering clustering model. Supportted algorithms are:
        KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch,
        FeatureAgglomeration, SpectralClustering and HDBSCAN (default: None).
        Can be passed as a string, but for finer control of the model
        parameters, pass as a dictionary, e.g.
        reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}. See
        scikit-learn specific model docs for details on parameters supported for
        each model. If no parameters are specified in the string a default set
        of parameters will be used.

    n_clusters : int
        If n_clusters is passed, HyperTools will perform k-means clustering
        with the k parameter set to n_clusters. The resulting clusters will
        be plotted in different colors according to the color palette.

    impute : str or dict or class or class instance or None
        Overrides the default PPCA fill for missing (NaN) values with a
        different `hypertools.impute` model, e.g. 'Kalman', 'KNNImputer'
        (default: None, i.e. PPCA -- byte-compatible with pre-1.0 behavior).
        See `hypertools.impute.impute` for accepted forms.

    resample : int or False/None
        If set to an integer `N` (GH #94), each input dataset is
        PCHIP-resampled to exactly `N` rows via the existing
        `hypertools.manip` ``Resample`` manipulator, applied right after
        `hypertools.tools.format_data` (so it sees whatever `x` has been
        normalized into -- a plain list of per-dataset numpy arrays) and
        BEFORE the normalize/reduce/align pipeline. `resample=500` on a
        100-row dataset produces per-dataset arrays with exactly 500 rows
        going into normalize/reduce/align/cluster/hue -- and the SAME
        values `hyp.manip(data, model='Resample', n_samples=500)` produces
        on that same input, since it is the identical manipulator call
        under the hood. This is independent
        of, and happens well before, the later line-smoothing
        interpolation (GH #141) applied for animation/line-drawing
        purposes. Default `None` (no resampling, unchanged pre-existing
        behavior); `False` is equivalent to `None`. Raises ``ValueError``
        if `resample` is anything other than `False`/`None` or an integer
        ``>= 2``.

    predict : str or dict or class or class instance or None
        If set, forecasts `t` new rows per input dataset (in the plotted,
        post normalize/reduce/align space) using the specified
        `hypertools.predict` model, e.g. 'Kalman', 'ARIMA', 'GaussianProcess'
        (see `hypertools.predict.predict` for accepted forms), and overlays
        one dashed, low-opacity (alpha 0.6) forecast trace per dataset in the
        SAME color as its source line (no separate legend entry). Only
        supported for STATIC plots (default: None; raises
        ``NotImplementedError`` if combined with ``animate``).

    t : int or datetime-like
        Forecast horizon passed to `predict` (see
        `hypertools.predict.common.resolve_t`); ignored unless `predict` is
        set (default: 10).

    save_path : str
        Path to save the image/movie. Must include the file extension in the
        save path (i.e. save_path='/path/to/file/image.png'). NOTE: If saving
        an animation, FFMPEG must be installed (this is a matplotlib req).
        FFMPEG can be easily installed on a mac via homebrew brew install
        ffmpeg or linux via apt-get apt-get install ffmpeg. If you don't
        have homebrew (mac only), you can install it like this:
        /usr/bin/ruby -e "$(curl -fsSL
        https://raw.githubusercontent.com/Homebrew/install/master/install)".

    animate : bool, 'parallel', 'spin', 'serial', 'morph', or list
        If True or 'parallel', plots the data as an animated trajectory, with
        each dataset plotted simultaneously. If 'spin', all the data is plotted
        at once but the camera spins around the plot. If 'serial', datasets
        appear ONE AT A TIME in list order: each grows point-by-point into
        place while all previous datasets stay fully drawn, and datasets are
        never connected to each other -- useful for e.g. conversation turns
        accumulating in a shared embedding space (default: False). This MODE
        is always GLOBAL -- there is exactly one camera and one frame loop
        driving every dataset in the animation, so it cannot vary per
        dataset (unlike `chemtrails`/`precog`/`bullettime` below, which CAN).

        If 'morph' (maintainer request, 2026-07-06), every dataset is
        treated as a POINT CLOUD (not a trajectory, regardless of `fmt`) and
        morphed ds_1 -> ds_2 -> ... -> ds_N through a hold/morph/hold/...
        schedule (``2N - 1`` segments: ``[hold_1, morph_1->2, hold_2, ...,
        hold_N]``). Every dataset keeps its FULL point count (maintainer
        request, 2026-07-06 follow-up): the target count `n` is the
        LARGEST morphing dataset's own size (after the optional
        `morph_samples` cap below), and any dataset with `m < n` points is
        padded up to `n` by duplicating `n - m` of its OWN points, chosen
        at random (seeded) -- no real data point is ever dropped. The
        duplicated (padding) points are hidden during that dataset's own
        HOLD segments (so semi-transparent markers alpha-composite exactly
        like a plain plot of that dataset's true points) and shown, like
        every other point, during MORPH segments. Consecutive (now equal-
        sized, `n`-point) clouds are chain-matched point-for-point with the
        Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on
        pairwise distances, so each point travels the shortest total
        distance to its partner in the next cloud -- exactly
        `examples/plot_shape_morph.py`'s original hand-rolled algorithm,
        now built into the library), and eased between clouds with
        smoothstep interpolation. A SINGLE point artist/trace is drawn
        (one per plot, not one per dataset): its color linearly (RGB)
        interpolates between the two datasets' own colors during a morph
        segment and is solid during a hold. Requires at least 2 datasets;
        raises `ValueError` otherwise. `surface=True` recomputes that one
        artist's hull every frame from its current interpolated positions
        (unaffected by which points are duplicates -- a duplicate is an
        exact copy of an existing point, so it never changes a convex
        hull's shape); 3-D only (`NotImplementedError` for 2-D data).

        `animate` may ALSO be a per-dataset LIST (length = the number of
        FINAL, post `cluster`/`hue`-reshape datasets), with each entry
        `'morph'`, `None`, or `False`: `'morph'`-tagged datasets join the
        morph sequence IN LIST ORDER; untagged datasets are drawn as STATIC
        (unanimated) backdrops, present in every frame. At least 2 entries
        must be `'morph'` (`ValueError` otherwise); any other mode string
        inside a list raises `ValueError` (list form only supports tagging
        datasets for `animate='morph'` -- 'spin'/'serial'/etc. cannot vary
        per dataset, see above). A scalar `animate='morph'` is equivalent to
        tagging every dataset `'morph'`.

    backend : str
        Rendering backend: 'matplotlib' (the classic renderer),
        'plotly' (interactive; requires plotly -- install with
        `pip install hypertools[interactive]`), or 'auto' (default), which
        uses plotly on Google Colab / Kaggle notebooks where interactivity
        matters most and matplotlib everywhere else. With the plotly backend,
        the return value is a plotly Figure (any animation frames are
        embedded directly in it, so no separate animation object is
        returned).

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float or list
        Number of rotations around the box over the course of the
        animation (default: 1 -- with the default 30-second duration,
        one revolution every 30 seconds). Identical pacing on both
        backends. A list is ONLY valid with `animate='morph'`: it must have
        exactly ``2N - 1`` entries (`N` = the number of morphing datasets),
        one per hold/morph segment (``[hold_1, morph_1->2, hold_2, ...,
        hold_N]``) -- e.g. `rotations=[1, 0.25, 2, 0.25, 1]` for 3 morphing
        datasets spins 1 full rotation during the first hold, a quarter
        rotation during the first morph, 2 rotations during the second
        hold, etc. Camera rotation speed (degrees/frame) is CONSTANT across
        the whole animation: each segment's SCREEN TIME (frame count) is
        proportional to its own rotation count -- not split evenly across
        segments -- so a segment with more rotations gets more time, never
        faster spinning (see :func:`hypertools.plot.morph.segment_frame_counts`
        and its `ZERO_ROTATION_FLOOR`: a segment with 0 rotations still gets
        a small amount of screen time so it stays visible). Within a
        segment, that segment's own rotation count is spread uniformly over
        its own frames, and the camera azimuth accumulates CONTINUOUSLY
        across segment boundaries (no jump). `N` is the number
        of morphing datasets AFTER the reduce/align/cluster/hue pipeline
        (the FINAL, drawn dataset count), which can differ from the number
        of datasets originally passed in. `ValueError` if a list is given
        with any `animate` mode other than `'morph'` (checked immediately,
        before the pipeline runs -- this only depends on `animate` itself),
        or if the list length doesn't match ``2N - 1`` (names the expected
        length; only knowable once `N` is, so checked after the pipeline
        runs).

    zoom (animation only) : float
        How far to zoom into the plot, positive numbers will zoom in (default: 0)

    chemtrails (animation only) : bool or list of bool
        A low-opacity trail is left behind the trajectory (default: False).
        Pass a list of bool (one entry per drawn dataset -- i.e. the FINAL
        count after any `cluster`/`hue`/`n_clusters` regrouping) for
        per-dataset control (GH #127): e.g. `chemtrails=[True, False]` turns
        chemtrails on for dataset 0 only. A bare bool is broadcast to every
        dataset. Raises `ValueError` if a list's length does not match the
        number of drawn datasets (naming both counts). Trail styles
        (`chemtrails`/`precog`/`bullettime`) only apply when
        `animate=True`/`'parallel'` -- see the note under `bullettime` below.

    precog (animation only) : bool or list of bool
        A low-opacity trail is plotted ahead of the trajectory (default:
        False). Accepts a per-dataset list exactly like `chemtrails` above,
        and the two may be mixed per dataset (e.g. dataset 0 chemtrails,
        dataset 1 precog, dataset 2 bullettime).

    bullettime (animation only) : bool or list of bool
        A low-opacity trail is plotted ahead and behind the trajectory
        (default: False). Accepts a per-dataset list exactly like
        `chemtrails` above. For any single dataset, `bullettime=True` (or
        `chemtrails=True` AND `precog=True` together) shows the FULL trail;
        `chemtrails` alone shows only the past window; `precog` alone shows
        only the future window; none of the three shows just the moving
        window (no separate trail artist/trace at all for that dataset).
        GH #127: trail styles apply ONLY to `animate=True`/`'parallel'`.
        `'spin'` has no "current position" for a trail to lead/follow (only
        the camera moves), and `'serial'`'s point-by-point reveal already
        communicates elapsed time, so `animate='spin'`/`'serial'` ignore
        `chemtrails`/`precog`/`bullettime` entirely (no trail artist/trace
        is created) and emit a `UserWarning` naming the mode, the ignored
        flag(s), and which dataset indices had them set.

    frame_rate (animation only) : int or float
        Frame rate for animation in frames per second (default: 30).
        Both backends generate exactly frame_rate * duration frames, so
        matplotlib and plotly animations play at identical speed,
        duration, and framerate.

    morph_samples (``animate='morph'`` only) : int or None
        An OPTIONAL cap on morphing-dataset size, applied BEFORE the
        duplicate-padding described under `animate` above: any morphing
        dataset larger than `morph_samples` is first downsampled (without
        replacement, seeded) to exactly `morph_samples` points. Default
        `None`: no cap -- every dataset keeps its full point count, and the
        target count is simply the largest dataset's own size. Since the
        Hungarian assignment's cost is roughly ``O(n^3)`` in the (post-cap)
        target point count, `morph_samples` is RECOMMENDED for clouds
        larger than ~2000 points (e.g. `morph_samples=1000`) -- the
        uncapped default can be slow, or memory-heavy, for very large
        datasets. Ignored for every other `animate` mode.

    interactive : bool
        If True, display the plot using an interactive matplotlib
        backend. Useful for inspecting and manipulating static plots. If
        animate=True, an interactive backend is required and this
        argument has no effect (default: False).

    explore : bool
        Displays user defined labels will appear on hover. If no labels are
        passed, the point index and coordinate will be plotted. To use,
        set explore=True. Note: Explore mode is currently only supported
        for 3D static plots, and is an experimental feature (i.e it may not yet
        work properly).

    mpl_backend : str
        The matplotlib backend used to create interactive and animated
        plots.  May be 'auto' (default), 'disable', or a backend key
        accepted by matplotlib. If 'auto', hypertools will use a backend
        determined automatically based on your environment
        (`hypertools.plot.backend.HYPERTOOLS_BACKEND`). If 'disable',
        experimental backend-switching is disabled and the current global
        matplotlib backend (`matplotlib.get_backend()`) is used.
        Otherwise, try to use the backend specified. NOTES: *This
        feature is experimental*. For a list of interactive matplotlib
        backends, see `matplotlib.rcsetup.interactive_bk`. For a list of
        backends available in IPython, run `%matplotlib --list`. Set the
        `$HYPERTOOLS_BACKEND` environment variable or use
        `hypertools.set_interactive_backend()` to override the backend
        used by 'auto' in non-IPython environments. If `animate=False`
        and `interactive=False`, this argument has no effect. Within the
        `hypertools.set_interactive_backend(backend)` context manager,
        the value of `backend` is prioritized over this argument.

    show : bool
        If set to False, the figure will not be displayed, but the figure,
        axis and data objects will still be returned (default: True).

    transform : list of numpy arrays or None
        The transformed data, bypasses transformations if this is set
        (default : None).

    vectorizer : str, dict, class or class instance
        The vectorizer to use. Built-in options are 'CountVectorizer' or
        'TfidfVectorizer'. To change default parameters, set to a dictionary
        e.g. {'model' : 'CountVectorizer', 'params' : {'max_features' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see here: http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to vectorizer_params. If
        a class instance, no parameters can be passed.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA). To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'params' :
        {'n_components' : 10}}. See
        http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see here:
        http://scikit-learn.org/stable/data_transforms.html).
        If a class, pass any parameters as a dictionary to text_params. If
        a class instance, no parameters can be passed.

    corpus : list (or list of lists) of text samples or 'wiki', 'nips', 'sotus'.
        Text to use to fit the semantic model (optional). If set to 'wiki', 'nips'
        or 'sotus' and the default semantic and vectorizer models are used, a
        pretrained model will be loaded which can save a lot of time.

    ax : matplotlib.Axes
        Axis handle to plot the figure

    frame_kwargs : dict
        Keyword arguments for styling the frame drawn around the plot.
        For 3D plots, the frame is a cube and `frame_kwargs` are
        forwarded to `mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe`.
        For 2D plots, the frame is a square and `frame_kwargs` are
        forwarded to `matplotlib.patches.Rectangle`.

    stream_init : int
        Streaming data only (iterators/generators and Hugging Face
        ``datasets.IterableDataset`` are detected automatically): number of
        initial samples used to estimate the normalization and reduction
        parameters (default: 10000). Those fitted models are then *applied*
        to all future samples, which are added to the plot dynamically.

    stream_chunk : int
        Streaming data only: number of new samples fetched from the stream
        per update (default: 100). Each fetched chunk is projected through
        the fitted models and rendered as one animation frame / live
        redraw, so this sets both the download batch size and the temporal
        resolution of the resulting animation.

    stream_max : int or None
        Streaming data only: stop streaming after this many samples.
        Default None streams continually until the stream is exhausted or
        the user interrupts (Ctrl-C); infinite streams render incoming
        data indefinitely, and any animation being saved via `save_path`
        is finalized whenever streaming stops (including on interrupt).

    stream_window : int or None
        Streaming data only: if set, only the most recent `stream_window`
        samples are displayed (comet style) while older samples scroll off;
        all consumed samples are still retained on the returned geometry.
        Default None displays the full accumulated trajectory.

    surface : bool, dict, or list of bool/dict, or None
        If set, overlays a smooth, lit surface over each dataset's convex
        hull (GH #109): a filled smooth outline for 2D data, or a shaded
        3D "blob" (inflated, subdivided, and Taubin-smoothed hull -- see
        `hypertools.plot.meshutil.smooth_hull_3d`) for 3D data. Pass
        ``True`` for the defaults below, a dict to override specific keys
        (unset keys use their default), or a list of bool/dict (one per
        *drawn* dataset, matching the final -- post `cluster`/`hue`
        regrouping -- dataset count) for per-dataset control; a bare
        ``False``/``None`` entry in the list disables that dataset's
        surface. Raises ``ValueError`` for 1D data (no hull concept), for
        an unrecognized dict key, or if a list's length does not match the
        number of drawn datasets. A dataset with too few points to form a
        hull (< 3 for 2D, < 4 for 3D) or whose points are exactly
        collinear/coplanar has its surface silently skipped with a
        ``UserWarning`` (never a crash). Default None (no surfaces).

        Accepted dict keys, with defaults:

        - ``alpha`` (float, default 0.6): surface opacity. Note that a
          translucent (< 1.0) 3D matplotlib surface REQUIRES the built-in
          backface culling (always applied) to avoid interior-face
          "cracks" showing through; plotly's translucent ``Mesh3d`` keeps
          the full mesh and may show mild WebGL self-overlap artifacts at
          silhouette edges (a known plotly limitation, not a hypertools
          bug) -- prefer ``alpha`` close to 1.0 if this is objectionable.
        - ``color`` (color spec or None, default None): surface base
          color. ``None`` inherits the dataset's own drawn line/marker
          color (resolved from `color`/`colors` if given, else the
          `palette` color cycle).
        - ``lighting`` (dict, default ``{}``): overrides the two-light
          Blinn-Phong lighting model BOTH backends use identically (see
          `hypertools.plot.meshutil.blinn_phong_colors`/
          `blinn_phong_vertex_colors`) -- matplotlib shades per-FACE;
          plotly shades per-VERTEX (precomputed and handed to
          ``go.Mesh3d`` as ``vertexcolor``, with plotly's own lighting
          engine forced to the identity so it reproduces those colors
          verbatim -- needed so the double-sided winding workaround below
          doesn't render dark self-shaded patches) -- so every key below
          visibly affects both backends the same way. Accepted keys:

          - ``ambient`` (float, default 0.45): flat, direction-independent
            base brightness; higher values flatten/wash out shading
            (matte look), 0 makes unlit faces fully black.
          - ``diffuse`` (float, default 0.55): key-light (Lambertian)
            contribution; scales how strongly faces facing the key light
            brighten relative to those facing away.
          - ``fill`` (float, default 0.25): weaker opposite-side fill-light
            contribution, so faces angled away from the key light are not
            rendered fully flat/black.
          - ``specular`` (float, default 0.30): strength of the glossy
            highlight; 0 gives a fully matte surface, higher values (e.g.
            0.9) give a glossy/wet look.
          - ``shininess`` (float, default 48): specular exponent -- higher
            values (e.g. 128) tighten the highlight into a small glossy
            spot; lower values spread it into a broad sheen.
          - ``lightdir`` (3-vector ``(x, y, z)`` or None, default None):
            explicit key-light direction in scene/data coordinates
            (need not be normalized; must not be the zero vector). ``None``
            (default) derives the key light automatically from the current
            camera view (offset above and to the side), matching each
            backend's own default camera-relative lighting.

          plotly's light position (for its own, identity-forced lighting
          engine, unrelated to the vertex-color computation above) is
          fixed at ``(2.5, -1.5, 3.0)`` in scene coordinates. Ignored for
          2D surfaces (flat fills have no lighting). Unrecognized keys
          (e.g. the pre-GH-109-round-3 plotly-only ``roughness``/
          ``fresnel``, which no longer affect either backend's rendering)
          raise ``ValueError`` rather than being silently accepted.
        - ``smoothing`` (int, default 3): number of interleaved
          [subdivide, Taubin-smooth] rounds for a 3D hull (face count
          scales as ``4 ** smoothing``); ignored for 2D.
        - ``pre_inflate`` (float, default 1.0): scale factor applied to
          the 3D hull about its centroid before smoothing (default: no
          blanket inflation). Any shrinkage smoothing introduces is
          instead recovered by a minimal, mathematically bounded (grow at
          most 10%) post-hoc rescale targeting the actual input points, so
          the surface hugs the data rather than ballooning past it;
          ignored for 2D.
        - ``keep_points`` (bool, default True): if False, hides that
          dataset's own line/marker (only the surface is shown).

        Animated plots (matplotlib and plotly, 3D only -- animation is not
        supported for 2D data at all, with or without surfaces) recompute
        each dataset's hull every frame from its CURRENTLY VISIBLE window:
        the revealed portion for ``animate='serial'``, the sliding
        head/tail window for ``animate=True``/``'parallel'`` (matching the
        window drawn by `chemtrails`/`tail_duration`), or the full,
        precomputed-once dataset for ``animate='spin'`` (only the camera
        orbits, so only per-frame shading/backface-culling -- not the mesh
        itself -- needs recomputing). Surfaces never gain a legend entry
        (``label='_nolegend_'`` / ``showlegend=False``) in either backend.

    density : bool, dict, or None
        If set, overlays a subtle KDE (kernel density estimate) "glow"
        behind the data (GH #108, #191): a 2-D alpha-ramped heatmap, or a
        3-D volumetric cloud, showing where each dataset's points are
        concentrated. Pass ``True`` for the defaults below, or a dict to
        override specific keys (unset keys use their default). Unlike
        `surface`, `density` has no per-dataset list form and no `color`
        override -- every density layer always inherits its dataset's own
        drawn color (or, with ``per_group=False``, a single neutral-gray
        layer is drawn for the pooled data). Raises ``ValueError`` for 1D
        data (no 2-D/3-D density concept) or an unrecognized dict key. A
        dataset with too few points (< 3) or degenerate (singular
        covariance -- e.g. exactly duplicated/collinear/coplanar points)
        has its density silently skipped with a ``UserWarning`` (never a
        crash). Default None (no density shading).

        Accepted dict keys, with defaults:

        - ``alpha`` (float, default 0.2): base opacity, kept subtle by
          design so the density layer never dominates the actual data.
          matplotlib's 2-D layer ramps linearly from fully transparent up
          to exactly this alpha at the KDE's peak; matplotlib's 3-D
          iso-surface/fog alphas and both plotly layers' opacities scale
          proportionally with it (see the backend-specific notes below).
        - ``levels`` (int, default 3): number of nested 3-D iso-surface
          shells. Wired into BOTH 3-D backends: matplotlib draws one
          `Poly3DCollection` per level, at density-fraction thresholds
          spaced evenly across ``[0.10, 0.65]`` via `numpy.linspace`
          (`levels=3`, the default, reproduces the original hand-tuned
          thresholds -- 10%/35%/65% of peak density, alphas 0.03/0.05/0.07
          -- EXACTLY, since evenly-spaced ``linspace(0.10, 0.65, 3)`` would
          instead give a 37.5%-not-35% middle shell); plotly's
          ``go.Volume`` layer uses ``surface_count=5*levels`` (15 at the
          default). **2-D density has no ``levels`` concept at all** --
          the 2-D layer is a single continuous alpha/heatmap ramp with no
          discrete shells, so ``levels`` is silently ignored for 2-D data
          (no error; the key is still valid, it's just a no-op there).
        - ``grid`` (int, default None): KDE evaluation grid resolution per
          axis. ``None`` auto-resolves to 200 for 2-D data or 50 for 3-D
          data (a 3-D grid is `grid**3` KDE evaluations, so much coarser by
          default).
        - ``per_group`` (bool, default True): fit and draw one density
          layer per drawn dataset. ``False`` pools every dataset's points
          into a single combined KDE, drawn as one neutral-gray layer
          instead.

        Backend rendering: matplotlib's 2-D layer is an alpha-ramped
        ``imshow`` (a `LinearSegmentedColormap` from transparent to the
        dataset's color at `alpha`, bilinear-interpolated, drawn below the
        data) -- not `contourf`, whose hard per-level boundaries read as
        banding rather than a smooth glow. matplotlib's 3-D layer is nested
        translucent iso-surfaces via `skimage.measure.marching_cubes`
        (`levels` shells spanning 10%-65% of peak density, alphas ramping
        0.03-0.07, both scaled by `alpha / 0.2`; see the `levels` entry
        above for the exact spacing) when scikit-image is installed
        (``pip install hypertools[density3d]``); otherwise it falls back to
        a translucent scatter "fog" (4000 points resampled from the KDE,
        `alpha` 0.03 scaled the same way) and emits a `UserWarning`
        suggesting the extra or ``backend='plotly'`` (which always renders
        a full volumetric `go.Volume`, no extra required). plotly's 2-D
        layer is a `go.Contour` heatmap (`coloring='heatmap'`, no
        contour lines, an alpha-ramped colorscale to `1.5 * alpha` --
        note this peak alpha is deliberately 1.5x the mpl 2-D layer's
        `alpha`, a documented cross-backend visibility difference, not a
        bug: plotly's heatmap reads fainter than mpl's `imshow` at the same
        alpha value, so the ramp is boosted to compensate); its
        3-D layer is a `go.Volume` (`isomin=0.05`, `isomax=1.0`,
        `surface_count=5*levels`, `opacity=min(3.0 * alpha, 0.6)`, a fixed
        `opacityscale` ramp tuned so the volume stays visible at plotly's
        3-D scene scale, a solid per-dataset colorscale). Density layers
        never gain a legend entry in either backend.

        3-D static-export caveat (both backends): when `per_group=True`
        (the default) draws more than one dataset's translucent 3-D density
        layer, the overlapping surfaces/volumes can composite unevenly in
        STATIC exports (PNG/SVG via matplotlib's Agg renderer or plotly's
        `kaleido`-based ``write_image``/``to_image``) -- a WebGL/rasterizer
        alpha-blending-order limitation, not a data or fitting bug. The
        interactive view (a live matplotlib window or plotly's browser/
        notebook widget) renders correctly; only static snapshots of
        multi-dataset 3-D density can look off.

        Animated plots (both backends, any `animate` style): the density
        is computed ONCE from the FULL dataset and drawn as a static
        background -- a single KDE evaluation is far too slow
        (~500ms at a 50^3 grid) to redo every animation frame, so, unlike
        `surface`, the density layer does not track the currently-visible
        window and is never touched by per-frame updates.

    return_model : bool
        If True, return a dict bundle
        ``{'fig': ..., 'xform_data': ..., 'animation': ..., 'pipeline': ...,
        'models': ..., 'predict': ...}`` instead of the bare figure, where
        ``xform_data`` is the normalized/reduced/aligned data, ``animation``
        is the ``matplotlib.animation.Animation`` handle (``None`` unless
        ``animate=True`` with the matplotlib backend), ``pipeline`` is a
        fitted `hypertools.Pipeline` covering whichever of `manip=`/
        `normalize=`/`reduce=`/`align=`/`cluster=` ran (the SAME `pipeline=`
        object passed in, if any; `None` when `transform=` was used, since
        then there is no raw data to have fit one on) -- pass it back in as
        `hyp.plot(new_data, pipeline=bundle['pipeline'])` to reuse these
        exact fitted parameters (GH #227), ``models`` holds the
        reduce/align/cluster/impute specs, and ``predict`` is ``None`` unless
        `predict` was set, in which case it is
        ``{'model': ..., 'params': {'t': t}, 'forecasts': [...]}`` (one
        forecast array per input dataset, in the analyzed/plotted --
        pre-center/scale -- space). Default False.

    Returns
    ----------
    fig : matplotlib.figure.Figure or plotly Figure
        The rendered figure. For animated matplotlib plots a
        ``(fig, animation)`` tuple is returned instead, so the caller can
        retain a reference to the ``matplotlib.animation.FuncAnimation``
        (required to keep the animation alive). When ``return_model=True``,
        a dict
        ``{'fig': ..., 'xform_data': ..., 'animation': ..., 'models': ...,
        'predict': ...}`` is returned (``animation`` included so the handle
        isn't dropped for animated plots).

    """

    # predict= + animate: forecast overlays are static-plot only in v1
    # (animating a growing/appended forecast trace is follow-up work).
    if predict is not None and animate:
        raise NotImplementedError(
            "predict= is not yet supported with animate: forecast traces "
            "are static-plot only in this release. Pass animate=False (the "
            "default) to use predict=, or omit predict= for an animated plot."
        )

    # rotations= as a per-SEGMENT list is only meaningful for
    # animate='morph' (every other mode has exactly one continuous camera
    # sweep, with no segment boundaries to assign rotations to). Whether
    # `animate` is IN morph mode at all is fully determined by the RAW
    # `animate` argument -- a scalar `'morph'`, or ANY list/tuple (which
    # `_resolve_animate_mode` below only ever uses to per-dataset-tag a
    # morph sequence) -- never by how many datasets end up being plotted,
    # so this mismatch is checked here, fail-fast, before the (expensive)
    # analyze/reduce/align/cluster/hue pipeline runs, mirroring the
    # colorbar=/surface=/density= early validation just below. The
    # COMPLEMENTARY checks that DO depend on the FINAL (post cluster/hue-
    # reshape) dataset count -- rotations' exact ``2N - 1`` length and the
    # "at least 2 morph-tagged datasets" minimum -- cannot be resolved yet
    # here and are still checked later, once `xform` (and so the final
    # dataset count) is known; see `_resolve_animate_mode`/
    # `resolve_morph_rotations` below.
    if isinstance(rotations, (list, tuple)) and not (
        animate == "morph" or isinstance(animate, (list, tuple))
    ):
        raise ValueError(
            "rotations as a list is only supported with animate='morph' "
            f"(got animate={animate!r}); pass a scalar rotations= for "
            "this animate mode."
        )

    # colorbar= kwarg validation (GH #100): fail fast with a clear message
    # before any of the (expensive) analyze/reduce/align pipeline runs.
    _VALID_COLORBAR_LOCATIONS = ('right', 'left', 'top', 'bottom')
    _VALID_COLORBAR_KEYS = {'label', 'ticks', 'location'}
    if colorbar is not None and colorbar is not False:
        if colorbar is True:
            colorbar = {}
        elif isinstance(colorbar, dict):
            unknown = set(colorbar) - _VALID_COLORBAR_KEYS
            if unknown:
                raise ValueError(
                    f"colorbar dict got unknown key(s) {sorted(unknown)}; "
                    f"valid keys are {sorted(_VALID_COLORBAR_KEYS)}."
                )
            loc = colorbar.get('location', 'right')
            if loc not in _VALID_COLORBAR_LOCATIONS:
                raise ValueError(
                    f"colorbar['location'] must be one of "
                    f"{_VALID_COLORBAR_LOCATIONS}; got {loc!r}."
                )
        else:
            raise ValueError(
                "colorbar must be True, False, None, or a dict with keys "
                f"a subset of {sorted(_VALID_COLORBAR_KEYS)}; got "
                f"{colorbar!r}."
            )
    else:
        colorbar = None

    # font= resolution (GH #205): resolved ONCE, here, from every piece of
    # text hypertools might draw that is knowable before the (expensive)
    # analyze/reduce/align pipeline runs -- labels=, a literal legend=
    # list, title=, colorbar['label']/['ticks'] if given, and `hue=` (when
    # `legend=True`, matplotlib's auto-legend draws one entry per unique
    # CATEGORICAL hue value, so non-ASCII hue values need to be scanned
    # here too -- their exact codepoints match what ends up in the legend
    # even though the deduplicated/sorted unique-value list itself isn't
    # known until the hue-grouping pipeline runs below). A continuous
    # (all-numeric) `hue` contributes no strings and is silently skipped
    # by the text-flattening helper. Resolving once up front means every
    # text surface `_draw`/`_add_colorbar` touches later shares the exact
    # same FontProperties, rather than each independently re-scanning
    # installed fonts.
    _font_texts = [labels, legend, title, hue]
    if colorbar is not None:
        _font_texts.append(colorbar.get('label'))
        _font_texts.append(colorbar.get('ticks'))
    resolved_font = resolve_font(font, _font_texts)

    # surface= kwarg validation (GH #109): fail fast (unknown dict keys)
    # before the expensive analyze/reduce pipeline runs. `_surface_norm` is
    # either None (disabled), a single validated dict (broadcast to every
    # dataset once the final dataset count is known), or a list of
    # dict-or-None (length-checked against that same final count below).
    _surface_norm = normalize_surface_arg(surface)

    # density= kwarg validation (GH #108/#191): fail fast (unknown dict
    # keys) before the expensive analyze/reduce pipeline runs, mirroring
    # `surface=`'s validation above. `_density_norm` is either None
    # (disabled) or a single validated dict, broadcast to every dataset (or
    # pooled into one layer, per its `per_group` key) once the final
    # dataset count is known.
    _density_norm = normalize_density_arg(density)

    # resample= kwarg validation (GH #94): fail fast before the expensive
    # analyze/reduce pipeline runs, mirroring colorbar=/surface=/density=
    # above. A valid `resample` is False/None (disabled, default) or an
    # int >= 2 (PCHIP interpolation -- what `hypertools.manip.Resample`
    # uses under the hood, same as the interpolation this module already
    # does for line smoothing -- needs at least 2 points to fit a curve).
    if resample is not None and resample is not False:
        if (isinstance(resample, bool)
                or not isinstance(resample, (int, np.integer))
                or resample < 2):
            raise ValueError(
                "resample= must be an integer >= 2 (the target sample "
                "count per dataset) or False/None to disable resampling; "
                f"got {resample!r}."
            )

    # pipeline= kwarg validation (GH #227): mutually exclusive with the
    # stage kwargs it replaces -- fail fast before the expensive analyze/
    # reduce pipeline runs, mirroring resample=/colorbar=/surface=/
    # density= above. Compared against plot()'s own LITERAL defaults
    # (reduce="IncrementalPCA", ndims=3) rather than `is not None` (unlike
    # hyp.analyze's own pipeline= check, whose stage kwargs all default to
    # None) since plot() always has a reduce=/ndims= value.
    if pipeline is not None:
        _conflicting = [name for name, default_value, value in (
            ('manip', None, manip),
            ('normalize', None, normalize),
            ('reduce', 'IncrementalPCA', reduce),
            ('ndims', 3, ndims),
            ('align', None, align),
            ('cluster', None, cluster),
        ) if value != default_value]
        if _conflicting:
            raise ValueError(
                "pipeline= is mutually exclusive with the stage kwarg(s) "
                f"{', '.join(_conflicting)} (a fitted Pipeline already "
                "encodes which stages run and their fitted parameters); "
                "pass pipeline= alone (resample= is still applied first, "
                "as sugar -- see pipeline='s docstring)."
            )

    # streaming inputs (issue #101): iterators/generators and Hugging Face
    # IterableDatasets are detected from the structure of the input -- no
    # flag needed. Models are fitted on the first `stream_init` samples and
    # every subsequent sample is projected through the fitted models and
    # added to the plot dynamically (fetched in chunks of `stream_chunk`),
    # continuing until the stream ends, `stream_max` samples have been
    # consumed, or the user interrupts.
    from ..io.streaming import is_stream, plot_stream
    if is_stream(x):
        return plot_stream(
            x, fmt, stream_init=stream_init, stream_chunk=stream_chunk,
            stream_max=stream_max, stream_window=stream_window,
            ndims=ndims, reduce=reduce,
            normalize=normalize, align=align, cluster=cluster,
            n_clusters=n_clusters, save_path=save_path, show=show,
            frame_rate=frame_rate, markersize=markersize,
            linewidth=linewidth, color=color, palette=palette, title=title,
            size=size, elev=elev, azim=azim, ax=ax)

    # remember whether the USER supplied an axis before `_draw` reassigns the
    # local `ax` to the axis it created (used by the GH #148 close below).
    _user_supplied_ax = ax is not None

    if ax is not None:
        if ndims > 2:
            if ax.name != "3d":
                raise ValueError(
                    "If passing ax and the plot is 3D, ax must " "also be 3d"
                )

    text_args = {"vectorizer": vectorizer, "semantic": semantic, "corpus": corpus}

    # nested lists (e.g. [[a, b], [c]]) are flattened into a flat list of
    # datasets while recording each leaf's outermost-group index and nesting
    # depth; these drive multilevel styling below (color by outer group,
    # thinner/fainter lines per deeper level)
    nested_groups = nested_depths = None
    if isinstance(x, list) and any(isinstance(el, list) for el in x) \
            and not all(isinstance(el, str) for el in x):
        x, nested_groups, nested_depths = _flatten_nested(x)

    # MultiIndex DataFrames (GH #95): a DataFrame with a row MultiIndex
    # (nlevels >= 2) is expanded HERE, before format_data/analyze/reduce, into
    # one "leaf" dataset per unique full index combination -- so the leaves
    # flow through the normal pipeline (normalize/reduce/align, streaming,
    # interpolation, animation) exactly like any other list of datasets.
    # After that pipeline transforms them (see the `_multiindex_meta is not
    # None` branch below, alongside cluster/hue), per-level MEAN trajectories
    # are computed in the TRANSFORMED space and appended, with per-dataset
    # color/linewidth/alpha/linestyle/label overrides (see
    # `hypertools.plot.multiindex` for the exact formulas). `cluster`/
    # `n_clusters` fight the MultiIndex color assignment (both try to own
    # the grouping-to-color mapping) and raise; `hue` is superseded with a
    # warning (MultiIndex grouping takes precedence).
    #
    # This expansion ONLY happens for a BARE single MultiIndex DataFrame.
    # A list containing MultiIndex DataFrame(s) (whether alone or mixed with
    # arrays/other DataFrames) does NOT trigger expansion -- each such
    # DataFrame is instead treated as a flat/plain dataset by the normal
    # list-of-datasets pipeline, silently dropping the MultiIndex grouping
    # unless we warn here.
    if isinstance(x, list):
        for _i, _el in enumerate(x):
            if isinstance(_el, pd.DataFrame) and _el.index.nlevels >= 2:
                warnings.warn(
                    "MultiIndex grouping is only applied when a single "
                    "DataFrame is passed; the MultiIndex on dataset "
                    f"{_i} is being treated as a flat index."
                )

    _multiindex_meta = None
    if isinstance(x, pd.DataFrame) and x.index.nlevels >= 2:
        if cluster is not None or n_clusters is not None:
            raise ValueError(
                "cluster=/n_clusters= is not compatible with a row-"
                "MultiIndex DataFrame (GH #95): MultiIndex grouping already "
                "assigns colors by the top-level index and would conflict "
                "with cluster-based grouping. Reset the index "
                "(df.reset_index(drop=True)) before clustering, or drop "
                "cluster=/n_clusters= to use the MultiIndex grouping."
            )
        if predict is not None:
            raise ValueError(
                "predict= is not supported with MultiIndex expansion in "
                "this release: forecasts are computed one-per-leaf before "
                "the per-level mean traces are appended, so the leaf count "
                "no longer matches the final trace count. Reset the index "
                "(df.reset_index(drop=True)) before using predict=, or "
                "drop predict= to use the MultiIndex grouping."
            )
        if hue is not None:
            warnings.warn(
                "x has a row MultiIndex (GH #95): MultiIndex grouping "
                "(leaf traces + per-level averages) takes precedence over "
                "hue=; ignoring hue."
            )
            hue = None
        x, _multiindex_meta = expand_multiindex(x)

    # analyze the data
    raw = None
    if transform is None:
        raw = format_data(x, impute=impute, **text_args)

        # resample= (GH #94): PCHIP-resample each dataset to exactly
        # `resample` rows via the existing `hyp.manip` `Resample`
        # manipulator, applied HERE -- right after `format_data` has
        # normalized `x` into a plain list of per-dataset numpy arrays,
        # and BEFORE `analyze` (normalize/reduce/align) runs -- so the
        # resampled row count is what normalize/reduce/align/cluster/hue
        # all see, and (mirroring `predict=`'s forecast values) resample=
        # values match `hyp.manip(data, model='Resample',
        # n_samples=resample)` on the SAME per-dataset array exactly
        # (`hyp.manip`'s dict model spec, e.g. ``{'model': 'Resample',
        # 'args': [], 'kwargs': {'n_samples': resample}}``, is equivalent
        # but the plain `model='Resample', n_samples=...` call form is
        # simpler and used here). Runs before, and independently of, the
        # later line-smoothing interpolation (GH #141) -- that step still
        # only densifies for ANIMATION/line-drawing purposes and operates
        # on whatever row count resample= (or the original data) leaves it.
        if resample:
            from ..manip.manip import manip as _manip
            raw = [
                np.asarray(_manip(ri, model='Resample', n_samples=resample))
                for ri in raw
            ]

        if pipeline is not None:
            # pipeline= (GH #227): apply the fitted Pipeline's stages via
            # .transform (never refit) instead of fitting new manip=/
            # normalize=/reduce=/align=/cluster= models. reduce=/ndims=/
            # normalize=/align=/cluster=/manip= were already validated
            # (above) to still be at their defaults, so they are safely
            # omitted here -- pipeline= governs every stage.
            xform, _ = analyze(raw, pipeline=pipeline, internal=True,
                               impute=impute, return_model=True)
        else:
            xform = analyze(
                raw,
                ndims=ndims,
                normalize=normalize,
                reduce=reduce,
                align=align,
                manip=manip,
                internal=True,
                impute=impute,
            )
    else:
        xform = transform

    # Return data that has been normalized and possibly reduced and/or aligned
    xform_data = copy.copy(xform)

    # catch all matplotlib kwargs here to pass on
    mpl_kwargs = {}

    # handle color (to be passed onto matplotlib). `colors` is treated as
    # an alias of `color` (like linestyle(s)/marker(s) below) and takes
    # priority when both are given -- but it must ALSO work on its own:
    # previously this block was nested inside `if color is not None`, so
    # `colors=` alone was silently ignored and fell back to the default
    # palette (GH #142 follow-up).
    if color is not None or colors is not None:
        mpl_kwargs["color"] = color
        if colors is not None:
            mpl_kwargs["color"] = colors
            if color is not None:
                warnings.warn(
                    "Both color and colors defined: color will be ignored \
                              in favor of colors."
                )

    # handle linestyle (to be passed onto matplotlib). `linestyles` is
    # treated as an alias of `linestyle` and takes priority when both are
    # given -- but it must ALSO work on its own: previously this block was
    # nested inside `if linestyle is not None`, so `linestyles=` alone was
    # silently ignored (GH #142 follow-up).
    if linestyle is not None or linestyles is not None:
        mpl_kwargs["linestyle"] = linestyle
        if linestyles is not None:
            mpl_kwargs["linestyle"] = linestyles
            if linestyle is not None:
                warnings.warn(
                    "Both linestyle and linestyles defined: linestyle  \
                              will be ignored in favor of linestyles."
                )

    # handle marker (to be passed onto matplotlib). `markers` is treated as
    # an alias of `marker` and takes priority when both are given -- but it
    # must ALSO work on its own: previously this block was nested inside
    # `if marker is not None`, so `markers=` alone was silently ignored
    # (GH #142 follow-up).
    if marker is not None or markers is not None:
        mpl_kwargs["marker"] = marker
        if markers is not None:
            mpl_kwargs["marker"] = markers
            if marker is not None:
                warnings.warn(
                    "Both marker and markers defined: marker will be \
                              ignored in favor of markers."
                )

    # handle marker size (to be passed onto matplotlib/plotly)
    if markersize is not None:
        mpl_kwargs["markersize"] = markersize

    # handle line width (to be passed onto matplotlib/plotly)
    if linewidth is not None:
        mpl_kwargs["linewidth"] = linewidth

    # reduce data to 3 dims for plotting, if ndims is None, return this.
    # xform was already formatted (and possibly reduced to ndims) by analyze()
    # above, so skip re-running format_data/PPCA here; reduce() returns the
    # data unchanged when it is already at the target dimensionality.
    if ndims and ndims < 3:
        xform = reducer(xform, ndims=ndims, reduce=reduce, internal=True,
                        format_data=False)
    else:
        xform = reducer(xform, ndims=3, reduce=reduce, internal=True,
                        format_data=False)

    # surface= (GH #109): no hull concept in 1D -- fail fast rather than
    # silently ignoring the kwarg.
    if _surface_norm is not None and xform[0].shape[1] == 1:
        raise ValueError(
            "surface= is not supported for 1D data (no hull concept in a "
            "single dimension)."
        )

    # density= (GH #108/#191): no 2D/3D density-grid concept in 1D -- fail
    # fast rather than silently ignoring the kwarg (mirrors surface= above).
    if _density_norm is not None and xform[0].shape[1] == 1:
        raise ValueError(
            "density= is not supported for 1D data (no 2D/3D density grid "
            "concept in a single dimension)."
        )

    # predict=: forecast `t` new rows per input dataset, in the plotted
    # (post normalize->reduce->align) space (GH #169). Computed here -- one
    # forecast per ORIGINAL input dataset, before any cluster/hue reshaping
    # -- so the forecasts correspond 1:1 with the datasets about to be
    # drawn. The final observed row of each dataset is prepended so the
    # dashed trace connects to the plotted trajectory (forecast length is
    # therefore t + 1). `bundle_forecasts` keeps this analyze-space copy
    # (for the return_model bundle); `raw_forecasts` is a working copy that
    # gets the SAME center/scale transform as `xform` below, so the drawn
    # dashed trace lines up with the drawn (centered/scaled) data.
    raw_forecasts = None
    bundle_forecasts = None
    if predict is not None:
        from ..predict.predict import predict as _predictor
        _fc = _predictor(xform, model=predict, t=t)
        if not isinstance(_fc, list):
            _fc = [_fc]
        raw_forecasts = [
            np.vstack([np.asarray(xi[-1:]), np.asarray(fc)])
            for xi, fc in zip(xform, _fc)
        ]
        bundle_forecasts = [np.array(fc) for fc in raw_forecasts]

    # per-point colors for multicolored lines (set by the hue branch below;
    # computed after interpolation). Dataset lengths are captured now so hue
    # values can be re-interpolated to match the interpolated trajectories.
    multicolor_hue = None
    pre_interp_lengths = [len(xi) for xi in xform]

    # original category NAMES for a categorical hue (set below, if
    # applicable), used by `legend=True` so the legend/colorbar show the
    # actual category strings rather than the integer group ids `hue` gets
    # reassigned to just below (group_by_category returns ints).
    hue_category_names = None

    # MultiIndex DataFrames (GH #95): xform currently holds the TRANSFORMED
    # leaf trajectories (post normalize/reduce/align), in the same order as
    # `_multiindex_meta['leaf_keys']` -- exactly what `build_multiindex_styles`
    # needs to compute per-level mean trajectories IN THE REDUCED SPACE and
    # append them. cluster=/n_clusters= were already rejected and hue=
    # already squelched (with a warning) above, so this always wins the
    # cluster/hue/nested_groups chain below.
    if _multiindex_meta is not None:
        if color is not None or colors is not None:
            warnings.warn(
                "x has a row MultiIndex (GH #95): MultiIndex grouping "
                "assigns color by the top-level index; ignoring "
                "color/colors."
            )
        if linewidth is not None:
            warnings.warn(
                "x has a row MultiIndex (GH #95): MultiIndex grouping "
                "assigns linewidth by level (leaves=1, thicker per level "
                "averaged over); ignoring linewidth."
            )
        xform, _mi_style = build_multiindex_styles(
            xform, _multiindex_meta, palette=palette,
            linestyle=linestyle, linestyles=linestyles)
        mpl_kwargs["color"] = _mi_style["colors"]
        mpl_kwargs["linewidth"] = _mi_style["linewidths"]
        mpl_kwargs["alpha"] = _mi_style["alphas"]
        if _mi_style["linestyles"] is not None:
            mpl_kwargs["linestyle"] = _mi_style["linestyles"]
        mpl_kwargs["label"] = _mi_style["labels"]
        legend = _mi_style["labels"]

    # find cluster and reshape if n_clusters
    elif cluster is not None:
        if hue is not None:
            warnings.warn("cluster overrides hue, ignoring hue.")
        if isinstance(cluster, (str, bytes)):
            model = cluster
            params = default_params(model) or {}
        elif isinstance(cluster, dict):
            model = cluster["model"]
            model_key = model if isinstance(model, str) \
                else getattr(model, "__name__", str(model))
            # canonical {'model': ..., 'args': [...], 'kwargs': {...}} (or
            # just 'kwargs', no 'args') vs LEGACY {'model': ...,
            # 'params': {...}} (accepted for backward compatibility, with
            # a DeprecationWarning) -- mirrors
            # hypertools.cluster.cluster._resolve_cluster_spec's own
            # dict-shape handling (round17 Task 6 fix: this used to only
            # read cluster.get('params', {}), silently DROPPING a
            # canonical {'model', 'kwargs'} dict's kwargs). The spec below
            # is always rebuilt in the canonical {'model', 'kwargs'} form
            # before being handed to `clusterer()` further down, so that
            # call never re-triggers this same warning -- do NOT
            # double-warn.
            if "args" in cluster or "kwargs" in cluster:
                _spec_kwargs = dict(cluster.get("kwargs", {}))
            elif "params" in cluster:
                warnings.warn(
                    "{'model': ..., 'params': {...}} is deprecated; use "
                    "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                    DeprecationWarning, stacklevel=2)
                _spec_kwargs = dict(cluster["params"])
            else:
                _spec_kwargs = {}
            params = default_params(model_key, _spec_kwargs) or {}
            if "n_clusters" in cluster and n_clusters is None:
                # top-level convenience:
                # cluster={'model': ..., 'n_clusters': k}
                n_clusters = cluster["n_clusters"]
        else:
            raise ValueError(
                "Invalid cluster model specified; should be" " string or dictionary!"
            )

        if n_clusters is not None:
            if _mixture_name(model) == "HDBSCAN":
                warnings.warn(
                    "n_clusters is not a valid parameter for "
                    "HDBSCAN clustering and will be ignored."
                )
            elif _mixture_name(model) in mixture_models:
                params["n_components"] = n_clusters
            else:
                params["n_clusters"] = n_clusters

        # canonical dict spec (not the legacy {'model','params'} form) so
        # this internal call doesn't trigger cluster()'s DeprecationWarning
        cluster_labels = clusterer(xform, cluster={"model": model, "kwargs": params})

        if _mixture_name(model) in mixture_models:
            # soft assignments: color each observation by the proportion-
            # weighted blend of its components' colors
            if legend is True:
                warnings.warn(
                    "legend is not supported for mixture-model clustering "
                    "(observations have blended colors, not discrete "
                    "groups); ignoring legend."
                )
                legend = None
            if not animate:
                # exact per-point colors (rendered via collections/scatter)
                multicolor_hue = np.asarray(cluster_labels,
                                            dtype=np.float64)
                hue = None
            else:
                # animations render one trace per group: quantize the
                # blended colors into (near-)identical-color groups
                blended = mat2colors(cluster_labels, palette=palette)
                group_ids, group_colors = colors2groups(blended)
                xform, labels = reshape_data(xform, group_ids, labels)
                mpl_kwargs["color"] = [
                    group_colors[gid]
                    for gid in sorted(set(group_ids), key=group_ids.index)
                ]
                hue = group_ids
        else:
            xform, labels = reshape_data(xform, cluster_labels, labels)
            hue = cluster_labels

    elif n_clusters is not None:
        # If cluster was None default to KMeans
        cluster_labels = clusterer(xform, cluster="KMeans", n_clusters=n_clusters)
        xform, labels = reshape_data(xform, cluster_labels, labels)
        if hue is not None:
            warnings.warn("n_clusters overrides hue, ignoring hue.")

    # group data if there is a grouping var
    elif hue is not None:
        if color is not None:
            warnings.warn("Using group, color keyword will be ignored.")

        # classify the hue argument: per-observation numeric matrix
        # (mixture proportions, model weights, ...), continuous 1D values,
        # or discrete grouping labels
        n_obs = sum(len(xi) for xi in xform)
        try:
            hue_array = np.asarray(hue)
        except Exception:
            hue_array = None
        hue_is_matrix = (hue_array is not None and hue_array.ndim == 2
                         and np.issubdtype(hue_array.dtype, np.number)
                         and hue_array.shape[0] == n_obs)
        hue_is_continuous = (hue_array is not None and hue_array.ndim == 1
                             and np.issubdtype(hue_array.dtype, np.number)
                             and hue_array.shape[0] == n_obs)

        if (hue_is_matrix or hue_is_continuous) and not animate:
            # EXACT PER-POINT COLORS: color varies continuously across
            # observations. Datasets stay intact (no group reshape, which
            # would fragment lines and quantize marker colors); per-point
            # colors are computed after interpolation, below, and rendered
            # via collections (lines) or scatter (markers).
            multicolor_hue = np.asarray(hue_array, dtype=np.float64)
            if legend is True:
                warnings.warn("legend is not supported for continuous or "
                              "matrix-valued hue; ignoring legend.")
                legend = None
            hue = None

        elif hue_is_matrix:
            # markers (or animated) path: blend colors per observation,
            # then group observations with (near-)identical colors into
            # traces
            blended = mat2colors(hue_array, palette=palette)
            group_ids, group_colors = colors2groups(blended)
            mpl_kwargs["color"] = [
                group_colors[gid]
                for gid in sorted(set(group_ids), key=group_ids.index)
            ]
            if legend is True:
                warnings.warn("legend is not supported for matrix-valued "
                              "hue; ignoring legend.")
                legend = None
            hue = group_ids

        else:
            # if list of lists, unpack
            if any(isinstance(el, list) for el in hue):
                hue = list(itertools.chain(*hue))

            # if all of the elements are numbers, map them to colors
            if not isinstance(hue[0], tuple):
                if all(isinstance(el, (int, float, np.integer, np.floating))
                       and not isinstance(el, bool) for el in hue):
                    hue = vals2bins(hue)
                elif all(isinstance(el, str) for el in hue):
                    hue_category_names = list(
                        sorted(set(hue), key=list(hue).index))
                    hue = group_by_category(hue)

        # reshape the data according to group
        if hue is not None:
            if n_clusters is None:
                xform, labels = reshape_data(xform, hue, labels)
            # interpolate lines if they are grouped
            if is_line(fmt):
                xform = patch_lines(xform)

    # multilevel styling for nested-list input: every leaf under the same
    # outermost group shares that group's color, and each additional nesting
    # level renders thinner and fainter (summary -> detail)
    elif nested_groups is not None and color is None and colors is None:
        import seaborn as sns
        n_outer = len(set(nested_groups))
        base_colors = sns.color_palette(palette, n_outer)
        mpl_kwargs["color"] = [base_colors[g] for g in nested_groups]
        min_depth = min(nested_depths)
        if any(d != min_depth for d in nested_depths):
            mpl_kwargs["linewidth"] = [
                max(0.5, 2.0 * (0.7 ** (d - min_depth))) for d in nested_depths
            ]
            mpl_kwargs["alpha"] = [
                max(0.3, 0.9 ** (d - min_depth)) for d in nested_depths
            ]

    # surface= (GH #109): broadcast to the FINAL (post cluster/hue-reshape)
    # dataset count -- reshaping above can change how many traces are
    # actually drawn, so this must run after it, not against the original
    # `x`.
    surface_list = (broadcast_surface(_surface_norm, len(xform))
                    if _surface_norm is not None else None)

    # density= (GH #108/#191): broadcast to the FINAL (post cluster/hue-
    # reshape) dataset count, same as surface= above.
    density_list = (broadcast_density(_density_norm, len(xform))
                    if _density_norm is not None else None)

    # animate='morph' (Hungarian-matched point-cloud morphs between
    # datasets, maintainer request): resolve a scalar or per-dataset LIST
    # `animate` into the actual backend style (`animate` from here on is
    # always one GLOBAL mode -- there is only ever one camera and one frame
    # loop, exactly as before) plus `morph_tags` (which FINAL datasets join
    # the morph sequence), now that `xform` reflects the final (post
    # cluster/hue-reshape) dataset count -- same timing as
    # surface_list/density_list above.
    animate, morph_tags = _resolve_animate_mode(animate, len(xform))
    if morph_tags is not None and xform[0].shape[1] != 3:
        raise NotImplementedError(
            "animate='morph' is only supported for 3-D plots; the data "
            f"being plotted is {xform[0].shape[1]}-D. Pass ndims=3 (the "
            "default) to use animate='morph'."
        )

    # `rotations` as a per-SEGMENT list ([hold_1, morph_1->2, hold_2, ...],
    # length 2 * n_morph_datasets - 1): the mode-mismatch check (list given
    # under a non-morph mode) was already raised, fail-fast, near the top
    # of this function -- it depends only on the raw `animate` argument,
    # not on `n_datasets`. The length check below, in contrast, genuinely
    # cannot happen any earlier: `n_morph_datasets` is the count of FINAL
    # (post cluster/hue-reshape) datasets tagged for morph, which is only
    # known now that `xform`/`morph_tags` exist.
    if morph_tags is not None:
        rotations = resolve_morph_rotations(rotations, sum(morph_tags))

    # chemtrails/precog/bullettime (GH #127): broadcast bool-or-list to the
    # FINAL (post cluster/hue-reshape) dataset count, same as surface=/
    # density= above -- each accepts a single bool (applied to every
    # dataset) or a list/tuple of bool (one entry per drawn dataset, mixed
    # per-dataset combinations allowed). `animate` itself stays a single
    # GLOBAL mode -- only these trail FLAGS become per-dataset.
    chemtrails = broadcast_trail_flag(chemtrails, len(xform), "chemtrails")
    precog = broadcast_trail_flag(precog, len(xform), "precog")
    bullettime = broadcast_trail_flag(bullettime, len(xform), "bullettime")

    # GH #127 (+ morph follow-up): 'spin' has no "current position" (only
    # the camera moves, so a trail has nothing to trail BEHIND or AHEAD of),
    # 'serial' already communicates elapsed time via its point-by-point
    # reveal, and 'morph' draws a single traveling point-cloud artist with
    # no per-dataset "current position" either -- trail styles are
    # semantically meaningless in all three, so warn once (naming the mode,
    # which flag(s) were set, and for which dataset indices) rather than
    # silently building frozen/invisible trail artists. `_draw`/
    # `plotly_draw` skip creating those artists entirely for these modes
    # (see their own `style`/`animate` branches), so this is purely
    # informational -- no flags are mutated here.
    if animate in ("spin", "serial", "morph"):
        _ignored_trail_flags = [
            (_name, [i for i, v in enumerate(_flags) if v])
            for _name, _flags in (
                ("chemtrails", chemtrails),
                ("precog", precog),
                ("bullettime", bullettime),
            )
        ]
        _ignored_trail_flags = [
            (name, idxs) for name, idxs in _ignored_trail_flags if idxs
        ]
        if _ignored_trail_flags:
            _detail = ", ".join(
                f"{name} for datasets {idxs}" for name, idxs in _ignored_trail_flags
            )
            warnings.warn(
                f"animate={animate!r} does not support trail styles; "
                f"ignoring {_detail}",
                UserWarning,
                stacklevel=2,
            )

    # handle legend
    if legend is not None:
        if legend is False:
            legend = None
        elif legend is True and hue is not None:
            if hue_category_names is not None:
                # categorical string hue: show the ORIGINAL category names,
                # not the integer group ids `hue` was reassigned to above.
                legend = hue_category_names
            else:
                legend = [item for item in
                         sorted(set(hue), key=list(hue).index)]
        elif legend is True and hue is None:
            legend = [i + 1 for i in range(len(xform))]

        mpl_kwargs["label"] = legend

    # colorbar (GH #100): resolve the color-mapping info (continuous hue
    # value range + palette, or discrete group colors + labels) now, while
    # `hue`/`multicolor_hue`/`xform`/`legend` reflect the FINAL grouping
    # decision (post cluster/hue reshape, post legend-label resolution) but
    # BEFORE interpolation (which doesn't change the mapping, only the
    # point density) -- shared by both the matplotlib and plotly backends.
    colorbar_info = _build_colorbar_info(
        colorbar, hue, multicolor_hue, cluster, n_clusters, xform,
        mpl_kwargs, legend, palette)

    # interpolate if its a line plot. animate='morph' treats every dataset
    # as a POINT CLOUD (Hungarian-matched to its neighbors in `morph.py`),
    # never as a time-ordered trajectory -- interpolating it here would
    # change its point count/order for no benefit and would desync the
    # (separately, seed-controlled) morph sampling downstream, so this
    # entire step is skipped for it.
    # GH #141: marker+line combo styles (e.g. 'o-') must get the SAME
    # connecting-line smoothing/interpolation pure line styles (e.g. '-')
    # already get -- gated on `has_line_component` (true whenever a line is
    # drawn at all) rather than the stricter `is_line` (true only when
    # there is NO marker), which previously skipped interpolation entirely
    # for any marker+line combo. `raw_xform` keeps a reference to the
    # PRE-interpolation per-dataset arrays so markers can still be drawn at
    # the true sample points (matplotlib_backend's static plot1D/2D/3D
    # split combo styles into a smoothed line artist + a markers-only
    # artist using this raw copy); it is carried through the SAME later
    # transforms (nan_to_num/center/scale, below) as `xform` so the two
    # stay in the same coordinate space. `interp_array`/`interp_array_list`
    # return NEW arrays rather than mutating their input, so this reference
    # stays valid even after `xform` itself is reassigned below.
    raw_xform = list(xform)
    pre_interp_point_counts = [xi.shape[0] for xi in xform]
    if animate == "morph":
        pass
    elif fmt is None or isinstance(fmt, str):
        if has_line_component(fmt):
            if xform[0].shape[0] > 1:
                xform = interp_array_list(
                    xform, interp_val=frame_rate * duration / (xform[0].shape[0] - 1)
                )
    elif isinstance(fmt, list):
        for idx, xi in enumerate(xform):
            if has_line_component(fmt[idx]):
                if xi.shape[0] > 1:
                    # interp_array (singular): xi is one dataset. The
                    # historical interp_array_list call here treated the
                    # 2D array as a LIST of rows, silently replacing the
                    # dataset with a list of per-row interpolations (latent
                    # for years because a bug made is_line() always False)
                    xform[idx] = interp_array(
                        xi, interp_val=frame_rate * duration / (xi.shape[0] - 1)
                    )

    # interpolation adds points, so per-point labels must be re-mapped onto
    # the interpolated trajectories (each label lands at its original
    # point's new index; in-between points get None)
    post_interp_point_counts = [xi.shape[0] for xi in xform]
    if labels is not None and post_interp_point_counts != pre_interp_point_counts:
        labels = _expand_labels(labels, pre_interp_point_counts,
                                post_interp_point_counts)

    # compute per-point colors for multicolored lines now that trajectories
    # have been interpolated (hue values are re-interpolated to match)
    line_colors = None
    if multicolor_hue is not None:
        line_colors = _multicolor_line_colors(
            multicolor_hue, pre_interp_lengths, xform, palette)

    # handle explore flag
    if explore:
        assert (
            xform[0].shape[1] == 3
        ), "Explore mode is currently only supported for 3D plots."
        mpl_kwargs["picker"] = True

    # predict= forecasts were computed per ORIGINAL input dataset; if
    # cluster/hue reshaping regrouped `xform` into a different number of
    # traces (by category rather than by dataset), the 1:1 correspondence
    # no longer holds -- skip drawing forecasts rather than mismatch traces.
    if raw_forecasts is not None and len(raw_forecasts) != len(xform):
        raw_forecasts = None

    # center + scale. When forecasts are drawn, the frame must contain
    # EVERYTHING drawn: compute the center/scale statistics from the FULL
    # stacked data (observed + forecasts, mirroring the animation principle
    # that limits/frame come from the full stacked data) and pass both
    # through the SAME transform. Otherwise forecasts that extend beyond
    # the observed data's range map outside [-1, 1] and render past the
    # square/cube frame (axes are off, so nothing clips them).
    # GH #141: `raw_xform` (the pre-interpolation sample points, used to
    # draw markers at their TRUE locations for marker+line combo styles --
    # see the interpolation block above) must land in the EXACT same
    # coordinate space as `xform`, so it is carried through the identical
    # center/scale statistics computed from `xform` (+ forecasts, when
    # present) below -- never its OWN, independently-computed stats.
    if raw_forecasts is not None:
        _joint = np.vstack([np.vstack(xform), np.vstack(raw_forecasts)])
        _mean = np.mean(_joint, 0)
        xform = [xi - _mean for xi in xform]
        raw_forecasts = [fc - _mean for fc in raw_forecasts]
        raw_xform = [xi - _mean for xi in raw_xform]

        _joint = np.vstack([np.vstack(xform), np.vstack(raw_forecasts)])
        _m1 = np.min(_joint)
        _m2 = np.max(_joint - _m1)
        _rescale = lambda a: 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
        raw_forecasts = [_rescale(fc) for fc in raw_forecasts]
        raw_xform = [_rescale(xi) for xi in raw_xform]
    else:
        # no forecasts: identical to the historical center()/scale() path,
        # but with the SAME stats also applied to raw_xform (rather than
        # calling center()/scale() a second time on raw_xform, which would
        # compute DIFFERENT stats from raw_xform's own, possibly narrower,
        # pre-interpolation range).
        _stacked = np.vstack(xform)
        _mean = np.mean(_stacked, 0)
        xform = [xi - _mean for xi in xform]
        raw_xform = [xi - _mean for xi in raw_xform]

        _stacked = np.vstack(xform)
        _m1 = np.min(_stacked)
        _m2 = np.max(_stacked - _m1)
        _rescale = lambda a: 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
        raw_xform = [_rescale(xi) for xi in raw_xform]

    # handle palette with seaborn
    import seaborn as sns
    if isinstance(palette, np.bytes_):
        palette = palette.decode("utf-8")

    # turn kwargs into a list
    kwargs_list = parse_kwargs(xform, mpl_kwargs)

    # GH #206: arbitrary extra matplotlib-style kwargs (anything not one
    # of plot()'s own named parameters, e.g. `zorder=`, `dashes=`,
    # `alpha=`, `markeredgecolor=`) are merged in AFTER the named/internal
    # style kwargs above (`_apply_extra_kwargs` never overwrites a key
    # already set), verbatim -- no per-dataset list broadcasting is
    # attempted for these (see `_apply_extra_kwargs`'s docstring for why).
    _apply_extra_kwargs(kwargs_list, kwargs)

    def _resolve_dataset_colors():
        """Resolve each dataset's OWN drawn color: an explicit color/colors
        kwarg if given (already in `kwargs_list`), or -- if none was given --
        the same per-dataset palette-cycle color both backends fall back to
        (matplotlib via `sns.set_palette` below; plotly via the `sns_local`
        fallback a few lines down). Shared by `surface_colors` (GH #109) and
        `density_colors` (GH #108/#191): both need the exact color each
        dataset will actually be drawn in on EITHER backend, resolved
        identically."""
        import matplotlib.colors as _mcolors
        if "color" in mpl_kwargs:
            _base_colors = [kwargs_list[i].get("color")
                            for i in range(len(xform))]
        else:
            _base_colors = list(sns.color_palette(palette, len(xform)))
        return [
            _mcolors.to_rgb(c) if c is not None
            else _mcolors.to_rgb(f"C{i % 10}")
            for i, c in enumerate(_base_colors)
        ]

    # surface= (GH #109): resolve each dataset's OWN drawn color now (used
    # when a dataset's surface spec has color=None, i.e. "inherit").
    surface_colors = (_resolve_dataset_colors()
                      if surface_list is not None else None)

    # density= (GH #108/#191): resolve each dataset's OWN drawn color the
    # SAME way as surface_colors above (density has no color-override key,
    # so this is always what gets drawn, per_group=True case only -- the
    # per_group=False pooled layer uses a fixed neutral gray instead).
    density_colors = (_resolve_dataset_colors()
                      if density_list is not None else None)

    # animate='morph': resolve each dataset's OWN drawn color the SAME way
    # as surface_colors/density_colors above -- the traveling morph cloud's
    # color is a linear RGB interpolation between two datasets' OWN colors
    # (see `hypertools.plot.morph.morph_color`), so both backends need
    # every dataset's resolved color regardless of whether surface=/
    # density= were requested.
    morph_colors = (_resolve_dataset_colors()
                    if morph_tags is not None else None)

    # handle format strings
    if fmt is not None:
        if not isinstance(fmt, list):
            draw_fmt = [fmt for i in xform]
        else:
            draw_fmt = fmt
    else:
        draw_fmt = ["-"] * len(x)

    # convert all nans to zeros
    for i, xi in enumerate(xform):
        xform[i] = np.nan_to_num(xi)
    raw_xform = [np.nan_to_num(xi) for xi in raw_xform]
    if raw_forecasts is not None:
        raw_forecasts = [np.nan_to_num(fc) for fc in raw_forecasts]

    # interactive (plotly) backend: render with plotly and skip the
    # matplotlib pipeline entirely. backend='auto' resolves to plotly only
    # on Colab/Kaggle (see hypertools.plot.plotly_backend for the policy).
    if resolve_backend(backend) == "plotly":
        from .plotly_backend import plotly_draw

        # GH #206: warn (once, listing every offending kwarg) about extra
        # kwargs that reached `mpl_kwargs` (via the `**kwargs` passthrough
        # above) but that the plotly backend has no property to map them
        # onto -- checked against the RAW `kwargs` the caller passed
        # (rather than `mpl_kwargs`, which also holds plotly-supported
        # named params like `color=`/`linewidth=`), so only genuinely
        # unmappable extras are reported.
        _unmapped_plotly_kwargs = sorted(set(kwargs) - _PLOTLY_MAPPED_KWARGS)
        if _unmapped_plotly_kwargs:
            warnings.warn(
                f"backend='plotly' cannot map the following extra "
                f"kwarg(s) to a trace property and will ignore them: "
                f"{_unmapped_plotly_kwargs}. Supported passthrough "
                f"kwargs for plotly are: {sorted(_PLOTLY_MAPPED_KWARGS)}."
            )

        if "color" not in mpl_kwargs:
            import seaborn as sns_local
            mpl_kwargs = dict(mpl_kwargs)
            mpl_kwargs["color"] = sns_local.color_palette(
                palette, len(xform))
            kwargs_list = parse_kwargs(xform, mpl_kwargs)
            _apply_extra_kwargs(kwargs_list, kwargs)
        fig = plotly_draw(
            xform,
            fmt=draw_fmt,
            kwargs_list=kwargs_list,
            labels=labels,
            legend=legend,
            title=title,
            animate=animate,
            size=size,
            show=show,
            save_path=save_path,
            frame_rate=frame_rate,
            duration=duration,
            rotations=rotations,
            elev=elev,
            azim=azim,
            point_colors=line_colors,
            tail_duration=tail_duration,
            chemtrails=chemtrails,
            precog=precog,
            bullettime=bullettime,
            zoom=zoom,
            forecasts=raw_forecasts,
            colorbar_info=colorbar_info,
            surface=surface_list,
            surface_colors=surface_colors,
            density=density_list,
            density_colors=density_colors,
            morph_tags=morph_tags,
            morph_colors=morph_colors,
            morph_samples=morph_samples,
            font=resolved_font,
        )
        ax = None
        data = xform
        line_ani = None
    else:
        # Apply the hypertools palette/style only for the duration of this
        # plot call. Previously sns.set_palette/sns.set_style mutated global
        # matplotlib rcParams as a side effect of plotting (GH issue #259);
        # rc_context restores the user's settings on exit. The figure's axes
        # and artists are created inside the context, so they keep the
        # hypertools styling.
        with plt.rc_context():
            sns.set_palette(palette=palette, n_colors=len(xform))
            sns.set_style(style="whitegrid")

            # draw the plot
            fig, ax, data, line_ani = _draw(
                xform,
                fmt=draw_fmt,
                kwargs_list=kwargs_list,
                labels=labels,
                legend=legend,
                title=title,
                animate=animate,
                raw_data=raw_xform,
                duration=duration,
                tail_duration=tail_duration,
                rotations=rotations,
                zoom=zoom,
                chemtrails=chemtrails,
                precog=precog,
                bullettime=bullettime,
                frame_rate=frame_rate,
                elev=elev,
                azim=azim,
                explore=explore,
                show=show,
                size=size,
                ax=ax,
                frame_kwargs=frame_kwargs,
                surface=surface_list,
                surface_colors=surface_colors,
                density=density_list,
                density_colors=density_colors,
                morph_tags=morph_tags,
                morph_colors=morph_colors,
                morph_samples=morph_samples,
                font=resolved_font,
            )

            # predict=: overlay one dashed, low-opacity (alpha 0.6) forecast
            # trace per input dataset (GH #169), in the SAME color as its
            # source line. Added AFTER `_draw` has already built the legend
            # (from the original data lines only, via ax.legend() inside
            # `_draw`), so these traces never gain a legend entry;
            # label='_nolegend_' mirrors the trail-artist precedent
            # (matplotlib_backend's animated trails) as a second guard.
            if raw_forecasts is not None:
                _src_lines = list(ax.lines)
                for _i, _fc in enumerate(raw_forecasts):
                    _fc_color = (_src_lines[_i].get_color()
                                if _i < len(_src_lines) else None)
                    _d = _fc.shape[1] if _fc.ndim > 1 else 1
                    if _d >= 3:
                        ax.plot(_fc[:, 0], _fc[:, 1], _fc[:, 2],
                               linestyle='--', color=_fc_color, alpha=0.6,
                               label='_nolegend_')
                    elif _d == 2:
                        ax.plot(_fc[:, 0], _fc[:, 1], linestyle='--',
                               color=_fc_color, alpha=0.6, label='_nolegend_')
                    else:
                        ax.plot(_fc[:, 0], linestyle='--', color=_fc_color,
                               alpha=0.6, label='_nolegend_')

            # exact per-point colors: swap the single-color artists for
            # per-segment-colored line collections or per-point-colored
            # scatter (the cube/square frame and axes from _draw are kept)
            if line_colors is not None:
                if is_line(fmt):
                    _apply_multicolor_lines(ax, xform, line_colors,
                                            kwargs_list)
                else:
                    _apply_multicolor_markers(ax, xform, line_colors,
                                              kwargs_list)

            # tighten layout (static plots only: animated axes are given
            # the full canvas so rotating zoomed cubes don't clip, and
            # tight_layout would shrink them back into subplot margins)
            if not animate:
                plt.tight_layout()

            # colorbar (GH #100): built once, here, from the (frame-
            # independent) color mapping. For animated plots this is never
            # touched by the per-frame update callbacks, so it stays static
            # across every frame. Added BEFORE the legend fit below so that
            # fit accounts for whatever room/reposition the colorbar just
            # consumed -- a 'left'/'top'/'bottom' colorbar reshapes `ax`
            # via matplotlib's own `make_axes` machinery, which can discard
            # an EARLIER legend fit; fitting the legend last, against
            # whatever the current layout actually is, sidesteps that.
            if colorbar_info is not None and ax is not None:
                _add_colorbar(fig, ax, colorbar_info, font=resolved_font)

            # legend fitting (GH #100/#95 follow-up): a right-side (outside)
            # legend can overflow the figure's right edge. `tight_layout`
            # reserves room for it on 2D axes but NOT on 3D axes, and
            # neither accounts for a colorbar sharing that edge (location=
            # 'right') or reshaping `ax` (location='left'/'top'/'bottom').
            # This previously only ran for STATIC plots, leaving the legend
            # fully clipped whenever animate=True (the legend is added by
            # `_draw` above regardless of `animate`, and is static across
            # every animation frame, so fitting it once here -- exactly like
            # the colorbar above -- is enough; no per-frame work needed).
            if legend is not None and ax is not None:
                _fit_right_legend(fig, ax)

            # save
            if save_path is not None:
                if animate:
                    _save_animation(line_ani, save_path, frame_rate)

                else:
                    plt.savefig(save_path)

    # Return shape (Jeremy decision #2):
    #   - static (matplotlib or plotly): return the Figure alone
    #   - animated matplotlib: return (fig, line_ani) so the caller can keep
    #     a reference to the FuncAnimation (needed to keep it alive); ax is
    #     recoverable as fig.axes[0], so it needs no separate return slot
    #   - animated plotly: frames are embedded in the Figure, so return fig
    #   - return_model=True: return a dict bundle exposing the analyzed
    #     xform_data plus the reduce/align/cluster model specs
    # GH #148: show=False must also remove the figure from pyplot's global
    # manager. `plt.ioff()` alone leaves it registered, so Jupyter's post-cell
    # flush_figures() still displays it (and a later plt.show() re-draws it).
    # Closing deregisters it; the returned Figure stays valid and savable.
    # Skip when the user supplied their own `ax` (their figure to manage),
    # skip plotly figures (not pyplot-managed), and skip ANIMATED figures:
    # the FuncAnimation's timer belongs to the live canvas, and closing
    # destroys it on GUI backends (e.g. TkAgg on Windows, where the backend
    # switch actually succeeds) -- the animation's pending first-draw hook
    # then crashes any later draw of the returned figure with
    # "'NoneType' object has no attribute 'start'".
    if (not show and not _user_supplied_ax and line_ani is None
            and isinstance(fig, plt.Figure)):
        plt.close(fig)

    if return_model:
        # gather reduce params (spec, not a fitted instance)
        if isinstance(reduce, dict):
            reduce_dict = reduce
        else:
            reduce_dict = {"model": reduce, "params": {"n_components": ndims}}
        # gather align params
        if isinstance(align, dict):
            align_dict = align
        else:
            align_dict = {"model": align, "params": {}}
        # 'pipeline' (GH #227, round17 Task 6): a fitted hypertools.Pipeline
        # covering whichever of manip/normalize/reduce/align/cluster ran,
        # so `hyp.plot(B, pipeline=bundle['pipeline'])` reuses these exact
        # fitted parameters on new data instead of refitting. When the
        # caller passed pipeline= themselves (reuse case), that SAME
        # (already-fitted) Pipeline is reused here too. When the caller
        # supplied transform= directly (bypassing format_data/analyze
        # entirely, so there is no `raw` this pipeline could have been fit
        # on), no pipeline can be reconstructed. resample= sugar is NOT
        # represented as a pipeline step (it is applied to `raw` before
        # this pipeline is fit, mirroring how format_data itself is not a
        # step either), so reusing this pipeline on new data does not
        # re-apply resample=.
        if pipeline is not None:
            bundle_pipeline = pipeline
        elif raw is not None:
            from ..core.pipeline import build_pipeline
            bundle_pipeline = build_pipeline(manip=manip, normalize=normalize,
                                             reduce=reduce, ndims=ndims,
                                             align=align, cluster=cluster)
            bundle_pipeline.fit_transform(raw)
        else:
            bundle_pipeline = None
        return {
            "fig": fig,
            "xform_data": xform_data,
            "animation": line_ani,
            "pipeline": bundle_pipeline,
            "models": {
                "reduce": reduce_dict,
                "align": align_dict,
                "cluster": cluster,
                "impute": impute,
            },
            "predict": None if predict is None else {
                "model": predict,
                "params": {"t": t},
                "forecasts": bundle_forecasts,
            },
        }

    # only animated matplotlib plots set line_ani; plotly and static plots
    # leave it None
    if line_ani is not None:
        return fig, line_ani

    return fig


def _build_colorbar_info(colorbar, hue, multicolor_hue, cluster, n_clusters,
                         xform, mpl_kwargs, legend, palette):
    """Resolve `colorbar=` into a backend-agnostic color-mapping dict, or
    None if no colorbar was requested (GH #100).

    Returns a dict with key ``'kind'`` of:
    - ``'continuous'``: ``vmin``/``vmax`` (the ACTUAL hue value range) and
      ``palette`` -- the caller builds a `ScalarMappable` from
      `continuous_colormap(palette)` + `Normalize(vmin, vmax)`, which is
      guaranteed to match `_multicolor_line_colors`'s per-point colors
      (same palette, same `mat2colors` default `n_bins`).
    - ``'discrete'``: ``colors`` ((n, 3) array, ORDER matching the drawn
      groups) and ``labels`` (tick labels, from `legend` if it is a list,
      else ``1..n``).
    Both kinds also carry the user-facing ``label``/``ticks``/``location``
    overrides (from the `colorbar` dict; see `plot`'s docstring).

    Raises ``ValueError`` if `colorbar` was requested but there is no
    color mapping to show (a single, ungrouped dataset), or the mapping is
    a per-observation blend with no discrete grouping (an unbounded color
    space -- nothing finite to put on a colorbar).
    """
    if colorbar is None:
        return None

    label = colorbar.get('label')
    ticks = colorbar.get('ticks')
    location = colorbar.get('location', 'right')

    if multicolor_hue is not None and multicolor_hue.ndim == 1 and hue is None:
        vals = np.asarray(multicolor_hue, dtype=np.float64)
        return {
            'kind': 'continuous',
            'vmin': float(np.min(vals)),
            'vmax': float(np.max(vals)),
            'palette': palette,
            'label': label,
            'ticks': ticks,
            'location': location,
        }

    if multicolor_hue is not None:
        raise ValueError(
            "colorbar is not supported for per-observation matrix/mixture"
            "-blended hue without discrete grouping (colors vary "
            "continuously over an unbounded blend space, so there is no "
            "finite set of colors to show). Use a 1D continuous hue, or "
            "combine with cluster= (with animate=True, which quantizes "
            "the blend into discrete groups) instead."
        )

    n_groups = len(xform)
    if n_groups <= 1 and hue is None and cluster is None and n_clusters is None:
        raise ValueError(
            "colorbar=True requires a color mapping (hue=, cluster=, or "
            "n_clusters=): a single, ungrouped dataset renders in one "
            "color, so there is nothing to map on a colorbar."
        )

    explicit_colors = mpl_kwargs.get('color')
    if (isinstance(explicit_colors, (list, tuple))
            and len(explicit_colors) == n_groups):
        # the mixture-blend paths (cluster=<mixture model> or matrix hue,
        # animated) set an EXPLICIT per-group color list -- reuse it
        # verbatim so the colorbar swatches exactly match the drawn lines.
        colors = np.asarray(explicit_colors)[:, :3]
    else:
        # everything else (categorical hue, non-mixture cluster/n_clusters,
        # or a plain list of datasets) is colored from the ambient palette
        # in dataset/group order -- exactly what sns.set_palette (mpl) /
        # the per-trace sns.color_palette (plotly) assign when drawing.
        colors = get_palette_colors(palette, n_groups)

    labels = (list(legend) if isinstance(legend, list)
              else [i + 1 for i in range(n_groups)])

    # A trace labeled '_nolegend_' (e.g. every MultiIndex leaf and
    # intermediate-level mean, GH #95 -- only the TOP-level mean of each
    # group carries a real label) must NEVER appear on the colorbar: filter
    # colors/labels down to the REAL (legend-worthy) entries together, so a
    # 2-level MultiIndex DataFrame (8 leaves + 2 top-level means) renders 2
    # colorbar segments (one per top-level group), not 10.
    if '_nolegend_' in labels:
        keep = [i for i, l in enumerate(labels) if l != '_nolegend_']
        if not keep:
            raise ValueError(
                "colorbar=True requires at least one labeled group, but "
                "every trace is unlabeled ('_nolegend_') -- there is no "
                "finite set of named groups to show."
            )
        colors = np.asarray(colors)[keep]
        labels = [labels[i] for i in keep]

    return {
        'kind': 'discrete',
        'colors': colors,
        'labels': labels,
        'label': label,
        'ticks': ticks,
        'location': location,
    }


def _apply_font_to_colorbar(cbar, font):
    """Apply the resolved `font=` (GH #205) to every text surface a
    colorbar draws: its tick labels (on whichever axis is the "long" one
    for its orientation -- x for a horizontal top/bottom colorbar, y for a
    vertical left/right one) and its axis label. Applied unconditionally
    to BOTH axes (harmless -- the unused one has no text) rather than
    branching on orientation, which keeps this correct regardless of
    location=. A no-op when `font` is None (no override requested/needed)."""
    if font is None:
        return
    for lbl in list(cbar.ax.get_xticklabels()) + list(cbar.ax.get_yticklabels()):
        lbl.set_fontproperties(font)
    cbar.ax.xaxis.label.set_fontproperties(font)
    cbar.ax.yaxis.label.set_fontproperties(font)


def _add_colorbar(fig, ax, colorbar_info, font=None):
    """Attach a matplotlib colorbar built from `_build_colorbar_info`'s
    output to `fig`/`ax` (GH #100): a continuous `ScalarMappable` for
    continuous hue, or a `BoundaryNorm`-segmented one (one block per
    group, tick labels = group names) for discrete groups."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    default_tick_labels = None
    if colorbar_info['kind'] == 'continuous':
        cmap = continuous_colormap(colorbar_info['palette'])
        norm = Normalize(vmin=colorbar_info['vmin'],
                         vmax=colorbar_info['vmax'])
        ticks = colorbar_info['ticks']
    else:
        n = len(colorbar_info['colors'])
        cmap = ListedColormap(colorbar_info['colors'])
        norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
        ticks = (colorbar_info['ticks'] if colorbar_info['ticks'] is not None
                 else list(range(n)))
        default_tick_labels = colorbar_info['labels']

    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    tick_kwargs = {} if ticks is None else {'ticks': ticks}
    # the FINAL tick label strings/title -- these can be much wider than the
    # numeric default tick labels matplotlib would otherwise draw (e.g. group
    # names), so they must be applied BEFORE any pixel-measurement-based
    # width fitting runs (see `_add_right_colorbar`), not after.
    ticklabels = (None if default_tick_labels is None or colorbar_info['ticks']
                 is not None else [str(l) for l in default_tick_labels])
    label = colorbar_info['label']

    if colorbar_info['location'] == 'right':
        cbar = _add_right_colorbar(fig, ax, mappable, ticklabels=ticklabels,
                                   label=label, font=font, **tick_kwargs)
    else:
        cbar = fig.colorbar(mappable, ax=ax,
                            location=colorbar_info['location'],
                            **tick_kwargs)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        if label:
            cbar.set_label(label)
        _apply_font_to_colorbar(cbar, font)

    # A VERTICAL discrete colorbar ('right'/'left') must read top-to-bottom
    # in the SAME order as the legend (first group at the TOP) -- matplotlib's
    # default low-value-at-bottom convention otherwise reverses it relative
    # to the legend, which reads first-to-last top-to-bottom (GH #100
    # follow-up). `invert_yaxis` only flips the DISPLAY orientation of the
    # existing data-to-color mapping -- the same data value still maps to
    # the same color, so color<->label pairing survives the flip exactly;
    # it just moves group 0 from the bottom to the top. A HORIZONTAL
    # discrete colorbar ('top'/'bottom') already reads left-to-right in
    # legend order (matplotlib's default), so it is left untouched.
    # Continuous colorbars are numeric and must keep the conventional
    # low-at-bottom orientation, so this only applies to 'discrete'.
    if (colorbar_info['kind'] == 'discrete'
            and colorbar_info['location'] in ('right', 'left')):
        cbar.ax.invert_yaxis()
    return cbar


def _tight_right_edge_in(fig):
    """The TRUE required figure width (inches) to avoid clipping ANY artist
    off the right edge -- e.g. a legend and/or a colorbar's tick labels/axis
    label.

    Uses `Figure.get_tightbbox`, which computes each artist's actual extent
    from its text/renderer metrics independent of the figure's CURRENT
    canvas size -- unlike measuring rasterized "inked" pixels (the
    previous approach here), which silently UNDER-reports whenever content
    already overflows the current canvas: pixels that fall outside the
    canvas are simply never in the rendered buffer, so a rasterize-based
    fit that only closes the gap by a small fixed margin each iteration
    converges far too slowly for long labels (needing many more than
    `max_iter` steps -- GH #100 follow-up) and can still leave content
    clipped. Measuring the true extent directly lets every caller compute
    the exact required width in one shot.

    hypertools draws inside a seaborn `rc_context`, but the figure is
    actually rendered downstream (the sphinx-gallery scraper, or a bare
    savefig/display after `plot()` returns) under the RESTORED default
    rcParams, whose font is WIDER than seaborn's. Measuring under the
    seaborn font would make content look like it fits when it clips in the
    real output, so this always measures under `matplotlib.rcParamsDefault`
    to match what actually gets saved/displayed.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with plt.rc_context(matplotlib.rcParamsDefault):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        return float(fig.get_tightbbox(renderer).x1)


def _add_right_colorbar(fig, ax, mappable, pad_in=0.2, width_in=0.35,
                        max_iter=3, ticklabels=None, label=None, font=None,
                        **colorbar_kwargs):
    """Add `mappable`'s colorbar in a NEW strip to the right of the figure
    -- to the right of an existing right-side legend, if any -- widening
    the figure (via `_tight_right_edge_in`'s true-extent measurement, then
    repositioning `ax` by its unchanged absolute inches so neither the plot
    nor an already-fitted legend need to shrink or move).

    `ticklabels`/`label` (the FINAL tick label strings and axis label, if
    any -- e.g. group names, which can be far wider than the default
    numeric tick labels matplotlib would otherwise draw) are applied
    IMMEDIATELY after the colorbar is created and BEFORE the width-fitting
    pass below runs -- fitting against the (short) default labels and only
    swapping in the real ones afterward would fit the wrong content and
    leave the real labels clipped.

    That fitting pass widens the figure to the exact true extent measured by
    `_tight_right_edge_in`, repositioning BOTH `ax` and the new colorbar
    axes by their unchanged absolute inches, so nothing is clipped off the
    right edge regardless of label length."""
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass

    right_edge_in = _tight_right_edge_in(fig)
    w, h = fig.get_size_inches()

    cbar_x0_in = right_edge_in + pad_in
    new_w = cbar_x0_in + width_in + pad_in
    if new_w > w:
        pos = ax.get_position()
        left_in, bottom_in = pos.x0 * w, pos.y0 * h
        w_in_ax, h_in_ax = pos.width * w, pos.height * h
        fig.set_size_inches(new_w, h)
        ax.set_position([left_in / new_w, bottom_in / h,
                         w_in_ax / new_w, h_in_ax / h])
        w = new_w

    pos = ax.get_position()
    cbar_ax = fig.add_axes([cbar_x0_in / w, pos.y0 + pos.height * 0.2,
                            width_in / w, pos.height * 0.6])
    cbar = fig.colorbar(mappable, cax=cbar_ax, **colorbar_kwargs)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    if label:
        cbar.set_label(label)
    # font (GH #205) applied BEFORE the width-fitting pass below so that
    # pass measures the ACTUAL (possibly wider, multibyte-covering) font
    # -- fitting against the default font first and swapping fonts
    # afterward could leave a multibyte label/ticklabel clipped.
    _apply_font_to_colorbar(cbar, font)

    # second pass: the colorbar's tick labels/title were just drawn and may
    # extend past the reserved `width_in` strip -- widen to the exact true
    # extent (in one shot, not a fixed-step guess) as needed.
    for _ in range(max_iter):
        w_cur, h_cur = fig.get_size_inches()
        new_w = _tight_right_edge_in(fig) + pad_in
        if new_w <= w_cur + 1e-3:
            break
        ax_pos, cbar_pos = ax.get_position(), cbar_ax.get_position()
        ax_left_in, ax_w_in = ax_pos.x0 * w_cur, ax_pos.width * w_cur
        cbar_left_in, cbar_w_in = cbar_pos.x0 * w_cur, cbar_pos.width * w_cur
        fig.set_size_inches(new_w, h_cur)
        ax.set_position([ax_left_in / new_w, ax_pos.y0,
                         ax_w_in / new_w, ax_pos.height])
        cbar_ax.set_position([cbar_left_in / new_w, cbar_pos.y0,
                              cbar_w_in / new_w, cbar_pos.height])
    return cbar


def _fit_right_legend(fig, ax, pad_in=0.15, max_iter=3):
    """Ensure a right-side (outside) legend stays fully within the figure.

    hypertools anchors its legend to the RIGHT of the plot via
    ``ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))``.
    ``tight_layout`` reserves horizontal room for such a legend on 2D axes but
    NOT on 3D (Axes3D) axes, and a wide legend (long labels or many entries)
    overflows the figure's right edge on either -- the earlier "shrink the
    axes" approach hit a floor and gave up, clipping the legend. Instead widen
    the figure, adding room only on the right while keeping the plot's absolute
    size and position, until the legend's right edge sits inside the canvas.
    Fixing the figure itself (rather than a save kwarg) means every downstream
    save path AND interactive/notebook display shows the full legend.

    Runs for BOTH static and animated plots (GH #100/#95 follow-up: the
    legend is added by `_draw` regardless of `animate` and is static across
    every animation frame, so a single fit here -- after the figure/axes
    already reflect whatever a colorbar in ANY location did to `ax` -- is
    enough; no per-frame work needed), and after ANY colorbar has already
    been added (see the call site in `plot()`), so this always fits against
    the figure's current, final layout rather than being undone by it.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    # tight_layout may install a persistent layout engine that re-runs on every
    # draw/save and would override the manual set_position below (undoing the
    # widening). Freeze it so our figure resizing sticks through savefig.
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    for _ in range(max_iter):
        w, h = fig.get_size_inches()
        new_w = min(_tight_right_edge_in(fig) + pad_in, 3.0 * w)
        if new_w <= w + 1e-3:
            return  # legend (and anything else) already fits
        pos = ax.get_position()
        left_in, plot_w_in = pos.x0 * w, pos.width * w
        # widen (add room only on the right) while keeping the plot's
        # absolute size and position, so the legend gains room without
        # shrinking the plot.
        fig.set_size_inches(new_w, h)
        ax.set_position([left_in / new_w, pos.y0,
                         plot_w_in / new_w, pos.height])


def _flatten_nested(x, _depth=1):
    """Flatten arbitrarily nested lists of datasets (arrays/DataFrames) into
    a flat list, recording each leaf's outermost-group index and nesting
    depth. Lists containing strings (text data) are returned un-flattened,
    since nested string lists denote text corpora, not grouped datasets."""
    if _contains_string(x):
        return x, None, None
    leaves, groups, depths = [], [], []
    for outer_idx, el in enumerate(x):
        for leaf, depth in _iter_leaves(el, _depth):
            leaves.append(leaf)
            groups.append(outer_idx)
            depths.append(depth)
    return leaves, groups, depths


def _iter_leaves(el, depth):
    if isinstance(el, list):
        for sub in el:
            yield from _iter_leaves(sub, depth + 1)
    else:
        yield el, depth


def _contains_string(el):
    if isinstance(el, str):
        return True
    if isinstance(el, list):
        return any(_contains_string(sub) for sub in el)
    return False


def _multicolor_line_colors(hue_src, orig_lengths, xform, palette):
    """Per-point RGB colors for multicolored lines.

    hue_src holds one value (or one row) per ORIGINAL observation; the
    trajectories in xform have since been interpolated to a higher temporal
    resolution, so each dataset's hue values are linearly re-interpolated to
    its new length before color mapping. Colors are mapped over the
    CONCATENATED hue values so the scale is shared across datasets.

    Returns a list of (n_i, 3) arrays, one per dataset in xform.
    """
    hue_src = np.asarray(hue_src, dtype=np.float64)
    if hue_src.ndim == 1:
        hue_src = hue_src[:, None]

    splits = np.cumsum(orig_lengths)[:-1]
    pieces = np.vsplit(hue_src, splits)

    interped = []
    for piece, xi in zip(pieces, xform):
        n_new = xi.shape[0]
        if n_new == piece.shape[0]:
            interped.append(piece)
            continue
        old_t = np.linspace(0.0, 1.0, piece.shape[0])
        new_t = np.linspace(0.0, 1.0, n_new)
        interped.append(np.column_stack(
            [np.interp(new_t, old_t, piece[:, c])
             for c in range(piece.shape[1])]))

    stacked = np.vstack(interped)
    colors = mat2colors(
        stacked.ravel() if stacked.shape[1] == 1 else stacked,
        palette=palette)

    out, start = [], 0
    for xi in xform:
        out.append(np.asarray(colors[start:start + xi.shape[0]]))
        start += xi.shape[0]
    return out


def _apply_multicolor_lines(ax, xform, line_colors, kwargs_list):
    """Replace single-color line artists with per-segment-colored
    collections (matplotlib backend)."""
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    for line in list(ax.lines):
        line.remove()

    is_3d = xform[0].shape[1] >= 3
    for i, (xi, ci) in enumerate(zip(xform, line_colors)):
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        lw = tkwargs.get('linewidth') or plt.rcParams['lines.linewidth']
        if xi.shape[1] == 1:
            pts = np.column_stack([np.arange(xi.shape[0]), xi[:, 0]])
        else:
            pts = xi[:, :3] if is_3d else xi[:, :2]
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_colors = (ci[:-1] + ci[1:]) / 2.0
        if is_3d:
            coll = Line3DCollection(segments, colors=seg_colors,
                                    linewidths=lw)
            ax.add_collection3d(coll)
        else:
            coll = LineCollection(segments, colors=seg_colors,
                                  linewidths=lw)
            ax.add_collection(coll)


def _expand_labels(labels, old_lengths, new_lengths):
    """Re-map per-point labels onto interpolated trajectories.

    Each original point's label is placed at that point's index in the
    interpolated (longer) trajectory; the interpolated in-between points get
    None (no annotation). Accepts flat label lists or lists nested per
    dataset; returns a flat list matching sum(new_lengths).
    """
    if any(isinstance(el, list) for el in labels):
        flat = list(itertools.chain(*labels))
    else:
        flat = list(labels)

    out = []
    start = 0
    for old_n, new_n in zip(old_lengths, new_lengths):
        piece = flat[start:start + old_n]
        start += old_n
        expanded = [None] * new_n
        for i, lab in enumerate(piece):
            if old_n == 1:
                j = 0
            else:
                j = min(new_n - 1, int(round(i * (new_n - 1) / (old_n - 1))))
            expanded[j] = lab
        out.extend(expanded)
    return out


def _apply_multicolor_markers(ax, xform, point_colors, kwargs_list):
    """Replace single-color marker artists with per-point-colored scatter
    (matplotlib backend). Gives exact per-observation colors -- e.g. mixture
    proportions render as true blends instead of quantized groups."""
    for line in list(ax.lines):
        line.remove()

    is_3d = xform[0].shape[1] >= 3
    for i, (xi, ci) in enumerate(zip(xform, point_colors)):
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        ms = float(tkwargs.get('markersize')
                   or plt.rcParams['lines.markersize'])
        s = ms ** 2  # scatter sizes are areas in points^2
        if xi.shape[1] == 1:
            ax.scatter(np.arange(xi.shape[0]), xi[:, 0], c=ci, s=s)
        elif is_3d:
            ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], c=ci, s=s,
                       depthshade=False)
        else:
            ax.scatter(xi[:, 0], xi[:, 1], c=ci, s=s)

def _mixture_name(model):
    """Registry name for a cluster-model spec (string or class)."""
    return model if isinstance(model, str) \
        else getattr(model, "__name__", str(model))

