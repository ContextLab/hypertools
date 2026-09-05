#!/usr/bin/env python
"""hyp.plot: the main HyperTools visualization entry point.

Contains the `plot()` dispatcher (input normalization, the
manip/normalize/reduce/align/cluster analysis pipeline, hue/cluster/
MultiIndex regrouping, color/legend/colorbar resolution, streaming
dispatch, and save/return handling) plus its private helpers. Low-level
drawing lives in `matplotlib_backend` and `plotly_backend`.
"""
import copy
import inspect
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# `np` also arrives via the star import below; naming it explicitly is
# what makes every `np.` reference in this file resolvable to a reader
# and to a linter, instead of 186 F405 "may be undefined" findings.
from .._shared.helpers import *
from .._shared.params import default_params
from ..core.model import external_stacklevel
from ..tools.analyze import analyze
from ..cluster.cluster import cluster as clusterer, mixture_models, \
    models as hard_cluster_models
from .colors import (mat2colors, colors2groups, get_palette_colors,
                     continuous_colormap, NAN_COLOR, is_missing_label)
from ..reduce.reduce import reduce as reducer
from ..tools.format_data import format_data
from .matplotlib_backend import _draw
from .backend import manage_backend
from .plotly_backend import resolve_backend
from .animation_context import FrameHooks
from .animate import _save_animation
from .surface import broadcast_surface, normalize_surface_arg
from .density import broadcast_density, normalize_density_arg
from .trails import broadcast_trail_flag
from .multiindex import expand_multiindex
from .hierarchy import build_hierarchy_styles, build_hierarchy_traces
from ..core.hierarchy import (group_columns, reject_dual_axis,
                              reject_hierarchical_in_list)
from .morph import resolve_morph_rotations
from .fonts import resolve_font, sans_serif_stack
# `_process_plot_format` is matplotlib's own fmt-string grammar, used so
# `forecast_fmt=` means exactly what the same string means in `fmt=`. Taken
# from `.forecast` rather than imported again here: that module VALIDATES
# `forecast_fmt=` with it, and a second guarded import is a second chance
# for the check and the application to disagree about what parses.
from .forecast import (FORECAST_ALPHA_SCALE, forecast_alpha,
                       _process_plot_format)


# GH #206: the subset of mpl_kwargs keys `plotly_backend.plotly_draw` (and
# its trail/forecast helpers) actually reads and maps onto a plotly trace
# property, via the existing `_resolve_fmt`/`_trace_name` machinery -- see
# every `tkwargs.get(...)` call in `hypertools/plot/plotly_backend.py`.
# Any OTHER kwarg (arbitrary matplotlib-style passthrough, e.g. `zorder=`,
# `markeredgecolor=`, `dashes=`) has no plotly equivalent and is silently
# unusable there; `plot()` warns (once, naming every such kwarg) rather
# than silently dropping it with no feedback at all.
def _is_plotly_figure(obj):
    """True for a plotly Figure (without importing plotly when it is absent)."""
    try:
        from plotly.basedatatypes import BaseFigure
    except ImportError:
        return False
    return isinstance(obj, BaseFigure)


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
    (`color`/`marker`/`linestyle`/`markersize`/`linewidth`/`alpha`).

    A key already present in a given dataset's dict (set by a named
    parameter, e.g. `color=` or `alpha=`, or by internal styling logic,
    e.g. MultiIndex grouping's `color`/`linewidth`/`alpha`, `legend=`'s
    `label`, `explore=`'s `picker`) is left untouched -- named/internal
    styling always wins over a same-named extra kwarg. `alpha` was itself
    a generic extra kwarg before 1.1; now that it is a named parameter, it
    can never appear in `extra` at all (Python binds `alpha=` to the
    parameter before `**kwargs` is assembled).
    """
    if not extra:
        return
    for d in kwargs_list:
        for k, v in extra.items():
            if k not in d:
                d[k] = v


# target vertex count for STATIC line smoothing -- matches the historical
# default density (frame_rate=30 * duration=30 = ~900 interpolated
# vertices), but as a fixed constant so animation kwargs no longer alter
# static rendering (release-1.0 audit, F01-007).
_STATIC_LINE_TARGET_VERTICES = 900

# de-emphasized color for UNLABELED points (the None entries of a
# partially-labeled categorical hue; release-1.0 audit, F02-013) -- the
# same neutral gray `colors.NAN_COLOR` uses for non-finite hue values, so
# "no information" reads consistently across the library. It IS that
# constant rather than a second copy of its value: the two were written as
# separate literals and are one drift away from disagreeing about what
# "no information" looks like.
_UNLABELED_HUE_COLOR = NAN_COLOR

#: Largest morphing-cloud size `animate='morph'` will accept without an
#: explicit `morph_samples=`. The one-to-one point matching is a Hungarian
#: assignment (`scipy.optimize.linear_sum_assignment`) over an n x n float64
#: cost matrix, costing roughly O(n^3): measured 0.10 s / 0.64 s / 4.99 s at
#: n = 1000 / 2000 / 4000, so the built-in zoo shapes (~30k points each,
#: 7.2 GB of cost matrix) do not finish in any usable time (measured: killed
#: at 10 min; `morph_samples=2000` renders the same call in 8.2 s). Above
#: this size `simplify=True` (the default) downsamples to it SILENTLY, and
#: `simplify=False` raises instead. Below it, nothing happens at all.
MORPH_SAMPLES_REQUIRED_ABOVE = 2000


def _seaborn_palette_arg(palette, n_colors):
    """`palette` in a form seaborn's `color_palette`/`set_palette` accept.

    plot() documents palette= as a name, a list of colors, or a matplotlib
    `Colormap` (F02-011); seaborn handles the first two natively but not a
    Colormap INSTANCE, so that one is pre-sampled to `n_colors` RGB tuples
    via `get_palette_colors` (the same resolution `mat2colors`/the colorbar
    use, keeping every path's colors identical).

    An ``'image:<path>'`` string is the same kind of case: seaborn has no
    idea what it means and raises "is not a valid palette name", so it is
    pre-resolved through `get_palette_colors` too. This is the SECOND
    interception `palette=` needs -- every call site below hands its palette
    straight to seaborn without going through `colors._get_palette`, so
    intercepting only there would leave `sns.set_palette` (and so EVERY
    matplotlib plot call) raising on an image palette."""
    from matplotlib.colors import Colormap

    from .colors import IMAGE_PALETTE_PREFIX
    if isinstance(palette, Colormap) or (
            isinstance(palette, str)
            and palette.startswith(IMAGE_PALETTE_PREFIX)):
        return [tuple(c) for c in get_palette_colors(palette, n_colors)]
    return palette


def _fmt_draws_line(fmt):
    """True if `fmt` draws a line for ANY trace (so the data must be
    segmented into contiguous runs rather than globally merged by category;
    GH #291). A per-trace fmt LIST counts if any entry has a line
    component; a bare string/None uses `has_line_component` directly."""
    if isinstance(fmt, (list, tuple, np.ndarray)):
        return any(has_line_component(f) for f in fmt)
    return has_line_component(fmt)


def _apply_forecast_override(style, override):
    """Overlay one dataset's `forecast_*=` override onto an inherited style.

    Sparse by design (see `forecast.resolve_forecast_overrides`): only the
    aspects the user actually named are replaced, so `forecast_fmt=':'`
    leaves the inherited colour alone and `forecast_palette=` leaves the
    inherited dash alone.

    An explicit override colour beats a colour letter inside `forecast_fmt=`
    (``forecast_fmt='r--'`` with `forecast_hue=`), matching matplotlib's own
    rule that a `color=` kwarg wins over a fmt string.
    """
    if not override:
        return style
    fmt = override.get('fmt')
    if fmt is not None:
        if _process_plot_format is not None:
            ls, marker, color = _process_plot_format(fmt)
        else:  # pragma: no cover - matplotlib moved its private parser
            ls, marker, color = fmt, 'None', None
        style['linestyle'] = ls
        if marker not in (None, 'None'):
            style['marker'] = marker
        if color is not None:
            style['color'] = color
    if 'color' in override:
        style['color'] = override['color']
    return style


def _forecast_style_from(src_line, alpha_scale=FORECAST_ALPHA_SCALE,
                         override=None, anchor_color=None):
    """Style a forecast to match the observed trace it continues.

    A forecast is the SAME series projected forward, so it inherits its
    observed trace's identity -- colour, linestyle and linewidth -- and
    differs only in transparency: ``alpha = observed_alpha * alpha_scale``
    (an unset alpha is matplotlib's "opaque", i.e. 1.0). See
    `hypertools.plot.forecast.FORECAST_ALPHA_SCALE` for why, and for the
    pre-1.1.0 always-dashed/always-0.6 rule this replaced.

    This is the ONE place matplotlib forecast styling is decided: the static
    overlay (`_draw_forecast_overlays`, which also serves ``animate='spin'``)
    and the per-frame live/trail artists all call it, so no two of them can
    drift apart. `plotly_backend._forecast_style_from` is its plotly twin
    (same policy, plotly's colour/width/dash/opacity vocabulary).

    Parameters
    ----------
    src_line : matplotlib.lines.Line2D or None
        The observed trace being continued. ``None`` (fewer drawn lines than
        datasets -- the same defensive case the call sites already guarded)
        falls back to matplotlib's own defaults, letting the colour cycle.
    alpha_scale : float, default `FORECAST_ALPHA_SCALE`
    override : dict or None
        This dataset's `forecast_*=` override
        (`forecast.resolve_forecast_overrides`). Sparse: only the aspects it
        names replace the inherited ones.
    anchor_color : RGB(A) tuple or None
        Under a CONTINUOUS `hue=` the observed trace has many colours, so
        "the same colour as its trace" resolves to the colour where the
        forecast starts: the final observed point's hue colour. Passed here
        (rather than read off `src_line`) because the line artist carries
        the per-dataset palette colour, which is a colour nothing visible is
        drawn in. A `forecast_*=` override still wins over it. Plotly's twin
        takes the same argument, so both backends anchor identically.

    Returns
    -------
    dict
        ``color``/``linestyle``/``linewidth``/``alpha`` kwargs for `ax.plot`
        (plus ``marker`` when `forecast_fmt=` asked for one).
    """
    if src_line is None:
        return _apply_forecast_override(
            dict(color=anchor_color, linestyle='-',
                 linewidth=plt.rcParams['lines.linewidth'],
                 alpha=forecast_alpha(None, alpha_scale)),
            override)
    linestyle = src_line.get_linestyle()
    if linestyle in ('None', 'none', ' ', '', None):
        # a MARKER-ONLY observed trace has no linestyle to inherit, and
        # `linestyle='None'` would draw the forecast as nothing at all. A
        # forecast is always a line, so fall back to solid -- which is also
        # what plotly's `_resolve_fmt` yields for a marker-only fmt, keeping
        # the two backends identical in this corner too.
        linestyle = '-'
    return _apply_forecast_override(
        dict(color=(src_line.get_color() if anchor_color is None
                    else anchor_color),
             linestyle=linestyle,
             linewidth=src_line.get_linewidth(),
             alpha=forecast_alpha(src_line.get_alpha(), alpha_scale)),
        override)


def _draw_forecast_overlays(ax, raw_forecasts, antialias=True,
                            owner=None, overrides=None):
    """Overlay one forecast trace per input dataset (GH #169), styled to
    match its source line (`_forecast_style_from`): same colour, linestyle
    and linewidth, at half its alpha.

    Called AFTER `_draw` has already built the legend (from the original data
    lines only), so these traces never gain a legend entry;
    `label='_nolegend_'` mirrors the trail-artist precedent
    (matplotlib_backend's animated trails) as a second guard. Shared by the
    STATIC path and the `animate='spin'` setup (which only rotates the camera
    around this same static overlay), so both draw identical artists from the
    identical seam-prepended forecast arrays.

    Returns
    -------
    list
        The created matplotlib line artists (so callers -- e.g. the
        `animate='spin'` path -- can `set_clip_on(False)` on them).
    """
    artists = []
    # (dataset index, artist) for every artist created, so the identity tag
    # below survives an `ax.plot` call returning more than one artist
    _artist_dataset = []
    src_lines = list(ax.lines)
    for i, fc in enumerate(raw_forecasts):
        # antialias (see `plot`'s `antialias=`): smooth the forecast the SAME
        # way as any other line, so a short forecast (e.g. t+1 = 5 vertices)
        # draws as a smooth curve rather than a few straight segments (the
        # seam-prepended first point and the final point stay exact, so it
        # still joins the trajectory).
        fc = np.asarray(fc)
        # rows BEFORE smoothing: a 1-D plot puts the row index on x, so the
        # forecast spans `n_rows - 1` row units no matter how many vertices
        # it is drawn with (see the 1-D branch below).
        _fc_rows = fc.shape[0]
        if antialias:
            fc = _interp_static_line(fc)
        # `owner` maps forecast -> the RUN it continues, when hue=/cluster=
        # regrouped the traces. Without it, forecast i continues trace i.
        _src = owner[i] if owner is not None and i < len(owner) else i
        _src_line = src_lines[_src] if _src < len(src_lines) else None
        style = _forecast_style_from(
            _src_line, override=overrides[i] if overrides is not None else None)
        d = fc.shape[1] if fc.ndim > 1 else 1
        _before = len(artists)
        if d >= 3:
            artists.extend(ax.plot(
                fc[:, 0], fc[:, 1], fc[:, 2], label='_nolegend_', **style))
        elif d == 2:
            artists.extend(ax.plot(
                fc[:, 0], fc[:, 1], label='_nolegend_', **style))
        else:
            # 1-D: x is the ROW INDEX, so the forecast has to be drawn over
            # the rows FOLLOWING its source run -- and it starts AT that
            # run's last drawn x, because the forecast array has the seam
            # observation prepended. Passing no x at all let matplotlib
            # default to `0..len(fc)-1`, painting every forecast back over
            # the START of the plot (measured: a 60-row frame drew its
            # forecast at x 0..3 while the observed line ran 0..59), and made
            # matplotlib disagree with plotly, which has always built this x
            # from the observed run's offset (`plotly_backend._aa_x(step,
            # arr.shape[0] - 1, ...)`). `linspace` rather than `arange`: the
            # antialiased curve has ~900 vertices spanning the same
            # `_fc_rows - 1` row units, which is exactly what `_aa_x`'s
            # `start + arange(n) / step` computes on the plotly side.
            _x0 = (float(np.asarray(_src_line.get_xdata())[-1])
                   if _src_line is not None
                   and len(np.asarray(_src_line.get_xdata())) else 0.0)
            _xs = _x0 + np.linspace(0.0, float(max(_fc_rows - 1, 0)),
                                    fc.shape[0])
            artists.extend(ax.plot(_xs, fc[:, 0], label='_nolegend_',
                                   **style))
        _artist_dataset.extend((i, _a) for _a in artists[_before:])
    # role tag (see hypertools/plot/forecast.py): forecast artists must be
    # identifiable WITHOUT guessing from linestyle -- user data drawn with
    # fmt='--' is dashed too, and trail artists also carry '_nolegend_'.
    # Tagged once over the whole list rather than per iteration, so an
    # `ax.plot` call that ever returns more than one artist cannot leave any
    # of them untagged.
    #
    # `_hyp_forecast_dataset` is the SOURCE DATASET's index -- the plotly
    # backend's `meta['hyp_dataset']` under matplotlib's naming. The role
    # tag says what an artist IS; this says which series it belongs to, so
    # downstream consumers pair forecasts with data by identity rather than
    # by list position (which only holds while forecasts stay
    # one-per-dataset in dataset order).
    for _ds, _a in _artist_dataset:
        _a._hyp_forecast_role = 'static'
        _a._hyp_forecast_dataset = _ds
    return artists


def _categorical_color_label_maps(hue, palette, explicit_colors,
                                  group_labels, sort_numeric):
    """Map each hue category -> (colour, legend label), in the SAME drawn
    order the marker/global grouping would use, so a LINE plot's per-category
    colours match an equivalent marker plot.

    `explicit_colors`/`group_labels` are the per-category values already
    resolved by the special hue sub-cases (partially-labeled hue, quantized
    matrix/mixture colours, cluster labels); when the colour was left to the
    palette (plain string/int categorical, hard clustering) it is resolved
    here from `palette`. `sort_numeric` selects sorted-numeric drawn order
    (integer hue / cluster ids) over first-appearance order (string hue)."""
    import seaborn as sns
    appear = list(dict.fromkeys(hue))
    if sort_numeric:
        try:
            drawn = sorted(appear)
        except TypeError:
            drawn = appear
    else:
        drawn = appear
    if (isinstance(explicit_colors, (list, tuple))
            and len(explicit_colors) == len(drawn)):
        cat_color = {c: explicit_colors[i] for i, c in enumerate(drawn)}
    else:
        pal = sns.color_palette(
            _seaborn_palette_arg(palette, len(drawn)), len(drawn))
        cat_color = {c: tuple(pal[i]) for i, c in enumerate(drawn)}
    if (isinstance(group_labels, (list, tuple))
            and len(group_labels) == len(drawn)):
        cat_label = {c: group_labels[i] for i, c in enumerate(drawn)}
    else:
        cat_label = {c: str(c) for c in drawn}
    return cat_color, cat_label


def _regroup_categorical_lines(xform, hue, labels, cat_color, cat_label):
    """Regroup LINE data by categorical `hue` into contiguous runs (GH #291).

    Splits each input dataset into maximal same-category runs (preserving
    order + dataset identity), colours each run by its category, bridges only
    runs adjacent WITHIN one input dataset, and gives each category exactly
    ONE legend entry (the first run carries the label; later runs of the same
    category get ``'_nolegend_'``). Returns
    ``(segments, seg_labels, run_colors, run_group_labels, seg_dataset,
    run_category_names, seg_lengths, run_bridged)``; `seg_dataset` is each
    run's source input-dataset index (for propagating per-dataset styles via
    `_expand_styles_to_runs`), `seg_lengths` is each run's row count BEFORE
    bridging, and `run_bridged` is whether `patch_lines` appended the next
    run's first row to it. The last two exist so the animation's reveal clock
    can tell a run's OWNED rows from its DRAWN geometry (see
    `ownership.TraceOwnership`).

    `run_category_names` carries every run's CATEGORY, which
    `run_group_labels` deliberately does not: that list is matplotlib's
    legend-label list, so every repeat run of a category holds the sentinel
    ``'_nolegend_'``. Reading a user-facing message out of it produced
    "hue category '_nolegend_' has only one observation" -- a sentinel
    meaning "keep this artist out of the legend", presented to a user as
    the name of a category that does not exist. One list cannot serve both
    matplotlib and a human; this is the human one."""
    segments, seg_labels, seg_cat, seg_bridge, seg_dataset = segment_by_run(
        xform, hue, labels)
    # BEFORE patch_lines, which appends the next run's first point to every
    # bridged run: TraceOwnership must not be told a run OWNS its neighbour's
    # first observation -- it needs the owned span and the drawn span kept
    # apart (see ownership.TraceOwnership.draw_span).
    seg_lengths = [len(s) for s in segments]
    breaks = {i + 1 for i in range(len(segments) - 1) if not seg_bridge[i]}
    segments = patch_lines(segments, breaks=breaks, labels=seg_labels)
    # what patch_lines ACTUALLY bridged, as ONE FLAG PER RUN. `seg_bridge`
    # is one flag per GAP -- `segment_by_run` documents its length as
    # `len(segments) - 1` -- so the final run has no entry at all, and
    # `seg_bridge[i] and i < n - 1` raises rather than guarding: Python
    # subscripts before it reaches the `and`. The last run is never bridged
    # (it has no successor), which is also what `TraceOwnership` requires of
    # each dataset's final run.
    run_bridged = [i < len(seg_bridge) and bool(seg_bridge[i])
                   for i in range(len(segments))]
    run_colors = [cat_color[c] for c in seg_cat]
    seen = set()
    run_group_labels = []
    # str() on the way out: a numpy string scalar reprs as
    # `np.str_('b')`, which is not what a reader should see in a
    # warning about their own data
    run_category_names = [str(cat_label.get(c, c)) for c in seg_cat]
    for c in seg_cat:
        if c in seen:
            run_group_labels.append('_nolegend_')
        else:
            seen.add(c)
            run_group_labels.append(cat_label.get(c, str(c)))
    return (segments, seg_labels, run_colors, run_group_labels, seg_dataset,
            run_category_names, seg_lengths, run_bridged)


def _expand_styles_to_runs(fmt, mpl_kwargs, seg_dataset, n_datasets):
    """Propagate per-INPUT-DATASET styles across run segmentation (GH #291
    follow-up).

    Contiguous-run segmentation turns N input datasets into >= N drawn runs,
    so a caller's per-dataset style list (`fmt` plus the NAMED styling kwargs
    that reach `mpl_kwargs` -- ``color``/``marker``/``linestyle``/
    ``markersize``/``linewidth``/``alpha``) would otherwise fail the later
    one-value-per-trace length checks. Any such list/tuple whose length
    equals the INPUT-dataset count is expanded to run length by repeating
    each dataset's value across the runs it produced; a list already at run
    length is left untouched (explicit per-run styling). ``alpha`` joined
    this set in 1.1 (it used to be a generic ``**kwargs`` passthrough
    applied verbatim per trace); any REMAINING generic passthrough value
    still never reaches `mpl_kwargs` and is unaffected here. Returns the
    (possibly expanded) `fmt`; mutates `mpl_kwargs` in place.

    `seg_dataset` gives each run's source-dataset index. When there is one
    run per dataset (``len(seg_dataset) == n_datasets``) the two layouts
    coincide and nothing needs changing."""
    n_runs = len(seg_dataset)
    if n_runs == n_datasets:
        return fmt

    def _expand(val):
        if isinstance(val, (list, tuple)) and len(val) == n_datasets:
            return [val[d] for d in seg_dataset]
        return val

    if isinstance(fmt, (list, tuple)) and len(fmt) == n_datasets:
        fmt = [fmt[d] for d in seg_dataset]
    for key in list(mpl_kwargs):
        mpl_kwargs[key] = _expand(mpl_kwargs[key])
    return fmt


def _interp_static_line(arr):
    """PCHIP-smooth a trajectory for STATIC line drawing, data-faithfully.

    Thin wrapper over `antialias_line` (`hypertools._shared.helpers`, which
    see) keeping only the densified array: the result has roughly
    `_STATIC_LINE_TARGET_VERTICES` vertices and contains every original sample
    exactly. This is the STATIC half of `plot`'s ``antialias=``.
    """
    return antialias_line(arr, _STATIC_LINE_TARGET_VERTICES)[0]


def _interp_anim_line(arr, n_frames):
    """Resample one trajectory onto the animation's exact frame grid.

    PCHIP-interpolates (the same monotone interpolant the static path uses)
    onto ``np.linspace(0, n - 1, n_frames)``: exactly `n_frames` rows -- one
    per animation frame -- for EVERY dataset (release-1.0 audit: the
    historical ``np.arange``-step grid was computed from the FIRST dataset
    only, so later datasets of a different length were silently truncated or
    ran out mid-animation, F04-003, and floating-point step error produced
    901/41 frames where the docstring promises exactly
    ``frame_rate * duration``, F04-004). Endpoints are exact, so the
    animation provably reaches the final sample.
    """
    from scipy.interpolate import PchipInterpolator as pchip
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n < 2:
        return arr
    grid = np.linspace(0.0, n - 1.0, max(2, int(n_frames)))
    out = pchip(np.arange(n), arr)(grid)
    out[0] = arr[0]
    out[-1] = arr[-1]
    return out


def _require_finite_for_line(xi, dataset_index):
    """Fail fast, with a hypertools-level message, when a line-styled
    trajectory still contains non-finite values after preprocessing.

    PCHIP interpolation (static smoothing and animation frame gridding)
    raises scipy's bare "`y` must contain only finite values." for NaN/inf
    input. Rows with ALL features missing are the usual cause: the default
    PPCA imputation cannot reconstruct them (it already warned), and the
    raw scipy traceback named neither the problem nor the fix
    (release-1.0 audit, F05-011).
    """
    if not np.isfinite(np.asarray(xi, dtype=float)).all():
        raise ValueError(
            f"dataset {dataset_index} still contains non-finite values "
            "(NaN/inf) after preprocessing, so its line cannot be smoothed/"
            "animated. This usually means some rows had ALL features "
            "missing -- the default PPCA imputation cannot fill those. "
            "Drop those rows, impute them first (e.g. hyp.impute(data, "
            "model='Kalman')), or plot markers only (fmt='.')."
        )


def _normalize_save_path(save_path):
    """Validate/normalize ``save_path=`` up front (release-1.0 audit,
    F09-004/F09-007).

    Accepts any path-like (``pathlib.Path`` included -- downstream writers
    call ``.lower()``/string slicing), expands ``~``, and fails fast --
    BEFORE the expensive analyze/reduce/align pipeline runs and before any
    figure is created -- on the misuses that previously surfaced as cryptic
    deep-stack errors or silent misbehavior: a non-path type (was
    ``AttributeError: 'int' object has no attribute 'write'``), an empty
    string (silently wrote a hidden ``'.png'`` file), an existing directory,
    and a missing parent directory.

    Returns
    -------
    str
        The normalized filesystem path.
    """
    try:
        sp = os.fspath(save_path)
    except TypeError:
        raise TypeError(
            "save_path must be a str or a path-like object (e.g. "
            f"pathlib.Path); got {type(save_path).__name__}: {save_path!r}."
        ) from None
    if isinstance(sp, bytes):
        sp = os.fsdecode(sp)
    if not sp.strip():
        raise ValueError(
            "save_path is an empty string; pass a real file path (e.g. "
            "save_path='figure.png')."
        )
    sp = os.path.expanduser(sp)
    if os.path.isdir(sp):
        raise ValueError(
            f"save_path points to an existing directory ({sp!r}); include "
            f"a file name, e.g. save_path={os.path.join(sp, 'figure.png')!r}."
        )
    parent = os.path.dirname(os.path.abspath(sp))
    if not os.path.isdir(parent):
        raise FileNotFoundError(
            f"save_path directory does not exist: {parent!r}. Create it "
            "first (e.g. os.makedirs) or point save_path at an existing "
            "directory."
        )
    return sp


def _is_numeric_matrix(x):
    """True when `x` is a plain python "matrix": a non-empty list whose
    entries are all non-empty lists/tuples of scalars, with equal row
    lengths -- e.g. ``[[1., 2.], [3., 4.]]``. Such input is ONE dataset
    (exactly like the equivalent ``np.array``), not a nested list of
    scalar "datasets" (release-1.0 audit, F01-004/F08-001)."""
    if not (isinstance(x, list) and x):
        return False
    for row in x:
        if not (isinstance(row, (list, tuple)) and len(row) > 0):
            return False
        if not all(isinstance(v, (int, float, np.number))
                   and not isinstance(v, bool) for v in row):
            return False
    return len({len(row) for row in x}) == 1


def _validate_labels_length(labels, dataset_lengths):
    """Raise a clear ValueError when `labels=` does not carry exactly one
    entry per observation (release-1.0 audit, F01-010/F10-011: a short
    list crashed with a bare IndexError; a long one was silently
    truncated). Accepts flat lists or lists nested per dataset."""
    n_obs = int(sum(dataset_lengths))
    if any(isinstance(el, (list, tuple)) for el in labels):
        n_labels = sum(len(el) if isinstance(el, (list, tuple)) else 1
                       for el in labels)
    else:
        n_labels = len(labels)
    if n_labels != n_obs:
        raise ValueError(
            f"labels has {n_labels} entr{'y' if n_labels == 1 else 'ies'} "
            f"but the data has {n_obs} observations; labels must have "
            "exactly one entry per observation (use None entries for "
            "points that should not be labeled).")


#: Resolved animate modes for which a per-dataset `title=` sequence means
#: "name each segment while it is the one being shown".
_SERIAL_TITLE_STYLES = ('serial', 'morph')


def _validate_title(title, style=None, order=None, n_datasets=None):
    """`title=` is one string for the whole figure, or -- for serial-style
    animations -- one string per dataset, shown while that dataset is the
    one being revealed (and blanked through morph transitions, so only
    fully formed clouds are named).

    A list/tuple used to be silently stringified onto the axes (a caller
    passing one title per dataset got the literal text "['a', 'b', 'c']"
    drawn on their figure) before per-dataset titles existed at all; a
    non-serial-style list still gets exactly that TypeError today, and so
    does any non-list/tuple type regardless of style (plan 1.1 Task 8 widens
    WHAT is accepted, but never lets a non-sequence -- e.g. a dict --
    silently iterate into a garbage one-entry title list).

    `style` is the raw (pre-`_resolve_animate_mode`) or resolved `animate=`
    value -- `_raw_animate_style` normalizes either. `order` is the raw or
    resolved `order=` value. The FIRST call (fail-fast, before the
    analyze/reduce pipeline) passes the raw values, which is enough for the
    type check; the SECOND call (once `len(xform)` -- the FINAL, post
    cluster/hue-reshape dataset count -- is known, beside
    `_resolve_animate_mode`) passes the resolved values and performs the
    length check below.

    `order == 'serial'` alone is only treated as "this will end up serial"
    when `style` could actually HONOR it: 'spin'/'window' have no
    dataset-by-dataset reveal, so `order='serial'` alongside either of them
    is silently folded back to 'parallel' (with a warning) by
    `_resolve_animate_mode` -- a per-dataset title list would then never be
    meaningful, so this raises immediately, at the FIRST (fail-fast) call,
    instead of letting the pipeline run and only discovering it once the
    SECOND call sees the already-folded-back `order`.

    Returns None for the scalar/None forms, or a list of `n_datasets`
    per-segment strings.
    """
    if title is None or isinstance(title, str):
        return None
    _style = _raw_animate_style(style)
    _order_wants_serial = order == 'serial'
    # truthy (animated) style that CANNOT honor order='serial' -- 'spin'/
    # 'window' today; `_resolve_animate_mode` warns and folds `order` back
    # to 'parallel' for exactly these, so a title list is never reachable
    # once that fold happens. A FALSY style (no animation at all) is left
    # alone here: `_resolve_order` already raises a clearer, dedicated
    # ValueError ("order='serial' requires an animated plot") for that
    # combination, and this function must not preempt it with a less
    # specific TypeError.
    _style_cannot_go_serial = (
        _order_wants_serial and bool(_style)
        and _style not in _SERIAL_CAPABLE_STYLES
    )
    serial_style = (_style in _SERIAL_TITLE_STYLES
                    or (_order_wants_serial and not _style_cannot_go_serial))
    if not serial_style or not isinstance(title, (list, tuple)):
        if _style_cannot_go_serial and isinstance(title, (list, tuple)):
            raise TypeError(
                f"title must be a string (or None), not "
                f"{type(title).__name__}. animate={_style!r} has no serial "
                "ordering (it does not reveal datasets one at a time), so "
                "order='serial' is ignored and per-dataset title lists are "
                "not meaningful for it -- pass a single string title "
                "instead, or use animate=True/'parallel'/'serial'/'morph' "
                "for a style that supports per-dataset titles."
            )
        raise TypeError(
            f"title must be a string (or None), not {type(title).__name__}. "
            "Per-dataset titles are only meaningful for serial-style "
            "animations (order='serial' or animate='morph'), and must be a "
            "list/tuple there. For a per-dataset legend entry use names=; "
            "for a per-observation annotation use labels=."
        )
    titles = [str(t) for t in title]
    if n_datasets is not None and len(titles) != n_datasets:
        raise ValueError(
            f"title has {len(titles)} entries but there are {n_datasets} "
            "datasets to plot; pass a single string for a fixed title, or "
            "one string per dataset.")
    return titles


HUE_MODES = ('mixture', 'rgb')


def _matrix_hue_wants_rgb(hue_array, color_reduce, hue_mode=None):
    """Is this matrix hue literal RGB rather than palette mixture weights?

    In one place because two call sites need to agree.

    `hue_mode=` decides it outright when given. Otherwise the AUTOMATIC rule
    applies, unchanged since before `hue_mode` existed: a matrix with more
    than 3 columns, or any matrix when `color_reduce=` is given, is RGB; a
    <=3-column matrix with no `color_reduce=` is mixture weights.

    The automatic rule is kept as the default because changing it would
    silently repaint every existing figure with a wide matrix hue. But it
    cannot be the ONLY rule, because it contradicts what mixture weights are
    for: one palette colour per component, one component per leaf. A market
    with six sectors needs six columns, and the automatic rule sends exactly
    that to the reducer -- measured on the Market candidate, whose 4-colour
    palette was ignored outright and whose blue-intended leaves drew red.
    `hue_mode='mixture'` is how a caller says "these are weights, however
    many there are"; `hue_mode='rgb'` forces the other branch for a narrow
    matrix.
    """
    if hue_mode is not None:
        return hue_mode == 'rgb'
    return np.asarray(hue_array).shape[1] > 3 or color_reduce is not None


def _validate_hue_mode(hue_mode, color_reduce, hue=None):
    """`hue_mode=` is only meaningful for a MATRIX hue, and only one of
    `hue_mode='mixture'` / `color_reduce=` can be honoured at once."""
    if hue_mode is None:
        return
    if hue is None:
        raise ValueError(
            f"hue_mode={hue_mode!r} says how to interpret a 2-D hue MATRIX "
            f"-- as palette mixture weights or as literal RGB -- but no "
            f"hue= was given.")
    if hue_mode not in HUE_MODES:
        raise ValueError(
            f"hue_mode= must be one of {HUE_MODES} or None (choose "
            f"automatically); got {hue_mode!r}.")
    if hue_mode == 'mixture' and color_reduce is not None:
        raise ValueError(
            "hue_mode='mixture' blends the hue matrix through palette= and "
            "color_reduce= reduces it to literal RGB channels; they cannot "
            "both apply. Drop color_reduce=, or pass hue_mode='rgb' to "
            "reduce.")


def _matrix_hue_to_rgb(hue_array, color_reduce):
    """Reduce a matrix hue to 3 min-max scaled columns used AS (r, g, b)."""
    rgb = np.asarray(hue_array, dtype=np.float64)
    if rgb.shape[1] > 3:
        # more than 3 columns: reduce to 3 (default IncrementalPCA;
        # color_reduce accepts any hyp.reduce spec). A <=3-column matrix
        # is NOT reduced -- hyp.reduce(ndims=3) can't synthesize more
        # dimensions than the input has, and doing so crashed for k<=3
        # (QC 2026-07 red-team); its columns are used directly instead.
        from ..reduce.reduce import reduce as _color_reducer
        try:
            rgb = np.asarray(
                _color_reducer(rgb, reduce=(color_reduce or 'IncrementalPCA'),
                               ndims=3),
                dtype=np.float64)
        except ValueError as exc:
            # name the kwarg the user actually passed (the underlying error
            # says 'reduce', which the user never typed; release-1.0 audit,
            # F02-008) and collapse any whitespace runs from wrapped lines
            raise ValueError(
                f"color_reduce={color_reduce!r} failed to reduce the matrix "
                f"hue to 3 color channels: "
                f"{' '.join(str(exc).split())}") from exc
        if rgb.ndim == 3 and rgb.shape[0] == 1:
            rgb = rgb[0]
    # min-max each column to [0, 1]
    lo = rgb.min(axis=0, keepdims=True)
    hi = rgb.max(axis=0, keepdims=True)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    rgb = np.clip((rgb - lo) / span, 0.0, 1.0)
    # pad to exactly 3 channels (a 1- or 2-column matrix given with an
    # explicit color_reduce=): fill the missing channel(s) with a neutral
    # 0.5 so the present columns still drive the color.
    if rgb.shape[1] < 3:
        rgb = np.hstack([rgb, np.full((rgb.shape[0], 3 - rgb.shape[1]), 0.5)])
    return rgb


def _hierarchy_hue_per_leaf(hue, n_rows, n_leaves):
    """Normalize a hue argument to ONE value sequence per hierarchy leaf.

    Only two forms are accepted, and both are stated relative to the INPUT
    frame rather than to the drawn figure:

    1. a flat sequence of ``n_rows`` values -- shared row-wise values,
       broadcast to every leaf;
    2. ``n_leaves`` sequences of ``n_rows`` values -- per-leaf values.

    A flat array sized to the TOTAL DRAWN observations is deliberately
    NOT accepted. It is indistinguishable from form 1 whenever a frame has
    as many rows as the figure has points, and it would require the caller
    to predict how many mean traces the expansion is going to create --
    which is exactly the bookkeeping a column hierarchy exists to remove.

    A per-leaf sequence may be 2-D, in which case it is a MATRIX hue: one
    row of MIXTURE WEIGHTS per observation, blended through the palette.
    The contract, all of it measured rather than assumed:

    * rows are NORMALIZED to sum to 1 by `mat2colors`, so only the ratio
      between components is visible -- halving every weight in a row draws
      the identical colour. A second quantity therefore needs its own
      palette entry (e.g. a black one for "darker = larger"); it cannot ride
      on the total magnitude;
    * negative entries have no colour meaning, so `mat2colors` shifts each
      row by its own minimum -- a signed matrix is coloured by within-row
      CONTRAST, not by absolute value;
    * the palette must supply at least one colour per COLUMN; a shorter one
      raises, a longer one simply leaves components unused;
    * a non-finite entry colours that observation neutral grey and warns.
      Because a derived mean is the element-wise mean of its children, one
      NaN greys the leaf AND every ancestor mean at that row;
    * every per-leaf matrix must have the same width -- they share one
      palette;
    * BY DEFAULT the width decides: more than 3 columns, or any explicit
      `color_reduce=`, switches to the literal-RGB route instead
      (`_matrix_hue_wants_rgb`), exactly as on a flat plot. That is how a
      caller supplies per-observation RGB under a hierarchy. The reduction
      runs on the concatenation, which already holds the derived means:
      mean-then-reduce, on one shared scale.
    * `hue_mode='mixture'` overrides that width rule and blends however
      many columns there are. It is what a hierarchy of more than three
      leaves actually needs -- the width rule and the one-primary-per-leaf
      idea contradict each other past three components, and the width rule
      wins silently.

    The mean is what makes matrix hue worth supporting here: the mean of
    mixture weights is itself a mixture weight, so giving each leaf one
    primary makes every sector come out a secondary and the market a
    tertiary, with nothing computing them.

    Returns
    -------
    (per_leaf, reason)
        `per_leaf` is a list of `n_leaves` float arrays -- length `n_rows`
        for a CONTINUOUS hue, `(n_rows, k)` for a MATRIX hue. When the hue
        is categorical, `per_leaf` is None and `reason` is a phrase naming
        the kind, for the caller's warning: that form regroups the traces,
        so it would destroy the very leaves the hierarchy names.
    """
    expected = (
        f"hue over a column hierarchy must be a flat sequence of {n_rows} "
        f"row values (broadcast to every trace), or one hue sequence per "
        f"leaf ({n_leaves} sequences of {n_rows} values)")

    # A sequence of sequences is detected WITHOUT np.asarray: unequal-length
    # per-leaf sequences build a ragged array, which numpy either refuses or
    # turns into an object array -- and that case has to reach its own
    # "every sequence must be length N" error, not a shape error.
    if isinstance(hue, (list, tuple)):
        nested = bool(len(hue)) and all(np.ndim(el) >= 1 for el in hue)
        flat_array = None
    else:
        flat_array = np.asarray(hue)
        nested = flat_array.ndim == 2 and flat_array.shape[0] == n_leaves

    if nested:
        seqs = [np.asarray(el) for el in hue]
        if len(seqs) != n_leaves:
            raise ValueError(
                f"one hue sequence per leaf is required ({n_leaves} "
                f"leaves), got {len(seqs)}. {expected}.")
        wrong = sorted({len(s) for s in seqs if len(s) != n_rows})
        if wrong:
            raise ValueError(
                f"every per-leaf hue sequence must have length {n_rows} "
                f"(one value per row of the frame); got length(s) "
                f"{wrong}. {expected}.")
        # A 2-D per-leaf entry is a MATRIX hue: one row of mixture weights
        # per observation, blended through the palette. It is kept AS a
        # matrix rather than ravelled -- ravelling passes the length check
        # above (the first dimension IS n_rows) and then reinterprets each
        # row's k weights as k consecutive CONTINUOUS values, which drew
        # every trace across ~220 degrees of hue instead of holding one.
        #
        # The per-level mean is what makes this worth supporting: a mean
        # trace takes the ELEMENT-WISE MEAN of its children's aux, and the
        # mean of mixture weights is itself a mixture weight. So giving the
        # leaves one primary each makes every sector come out a secondary
        # and the market a tertiary, with nothing computing them.
        ndims = {np.ndim(s) for s in seqs}
        if len(ndims) > 1:
            raise ValueError(
                f"per-leaf hue sequences must all have the same shape: got "
                f"a mix of {sorted(ndims)}-dimensional entries. Pass one "
                f"value per row for every leaf, or one weight ROW per row "
                f"for every leaf. {expected}.")
        if ndims == {2}:
            widths = sorted({np.asarray(s).shape[1] for s in seqs})
            if len(widths) > 1:
                raise ValueError(
                    f"every per-leaf hue MATRIX must have the same number "
                    f"of columns (they are blended through one shared "
                    f"palette); got width(s) {widths}. {expected}.")
            return [np.asarray(s, dtype=np.float64) for s in seqs], None
        values = np.concatenate([np.asarray(s).ravel() for s in seqs])
    else:
        flat = flat_array if flat_array is not None else np.asarray(hue)
        if flat.ndim != 1:
            raise ValueError(
                f"{expected}; got an array of shape {flat.shape}.")
        if flat.shape[0] != n_rows:
            raise ValueError(f"{expected}; got {flat.shape[0]} values.")
        seqs = [flat] * n_leaves
        values = flat

    # Categorical stays categorical: same rule as the flat-input classifier
    # below (non-numeric, or integer/bool with few enough distinct values
    # that adjacent labels would map to indistinguishable palette samples).
    n_unique = len(np.unique(values))
    if not np.issubdtype(values.dtype, np.number):
        return None, "a categorical hue"
    if ((np.issubdtype(values.dtype, np.integer)
         or np.issubdtype(values.dtype, np.bool_))
            and n_unique <= 12 and n_unique < values.shape[0]):
        return None, "a categorical (small-cardinality integer) hue"

    return [np.asarray(s, dtype=np.float64) for s in seqs], None


def _validate_alpha(alpha, n_datasets):
    """`alpha=` is a scalar applied to every dataset, or one value per
    dataset. Returns a list of `n_datasets` floats, or None.

    Promoted out of the GH #206 `**kwargs` passthrough (where a list raised
    matplotlib's bare "alpha must be numeric or None") so callers can fade
    backdrops behind a highlighted dataset without re-applying `set_alpha`
    on every frame. `n_datasets` is the dataset count at the CALL SITE --
    the INPUT dataset count when called before cluster/hue-reshape (see the
    call just before the MultiIndex/cluster/hue/nested_groups chain,
    below, in `plot()`), so that a per-dataset list is left at
    INPUT-dataset length for `_expand_styles_to_runs` (plot.py, which see)
    to widen to run length exactly like `color`/`linewidth` already are.
    """
    if alpha is None:
        return None
    values = [alpha] if np.isscalar(alpha) else list(alpha)
    try:
        values = [float(a) for a in values]
    except (TypeError, ValueError):
        raise ValueError(
            f"alpha must be a number, or one number per dataset; got "
            f"{alpha!r}.") from None
    if len(values) == 1:
        values = values * n_datasets
    if len(values) != n_datasets:
        raise ValueError(
            f"alpha has {len(values)} entries but there are {n_datasets} "
            "datasets to plot; pass a single value to apply it to every "
            "dataset, or one value per dataset.")
    for a in values:
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha values must be between 0 and 1; got {a}.")
    return values


def _valid_line2d_kwargs():
    """The set of keyword-argument names matplotlib line artists accept
    (full property names plus their aliases, e.g. both 'linewidth' and
    'lw'), used to validate the GH #206 ``**kwargs`` passthrough up
    front."""
    from matplotlib.lines import Line2D
    from matplotlib.artist import ArtistInspector
    insp = ArtistInspector(Line2D)
    valid = set(insp.get_setters())
    for prop, aliases in getattr(insp, "aliasd", {}).items():
        valid.add(prop)
        valid.update(aliases)
    return valid


#: Sentinel: `slow_warning_seconds` was not passed, so use the default.
_UNSET_SLOW_WARNING = object()


def _validate_forecast_trail(forecast_trail, predict):
    """`forecast_trail=` keeps earlier forecasts on screen as a fading fan.

    Returns the number of past forecasts to retain (0 = trail disabled).
    Validated early, beside the other fail-fast checks, so a bad value is
    reported before the analyze/reduce pipeline runs rather than after.
    """
    if forecast_trail in (False, None, 0):
        return 0
    if predict is None:
        raise ValueError(
            "forecast_trail= requires predict=; there are no forecasts to "
            "retain without a forecast model.")
    if forecast_trail is True:
        from .forecast import DEFAULT_FORECAST_TRAIL
        return DEFAULT_FORECAST_TRAIL
    if isinstance(forecast_trail, bool) or not isinstance(
            forecast_trail, (int, np.integer)):
        raise TypeError(
            "forecast_trail must be True/False or a positive int (the number "
            f"of past forecasts to keep); got {forecast_trail!r}.")
    if forecast_trail < 1:
        raise ValueError(
            f"forecast_trail must be >= 1 when given as an int; got "
            f"{forecast_trail}.")
    return forecast_trail


def _validate_extra_plot_kwargs(extra_kwargs):
    """Fail fast, BEFORE the analyze/reduce pipeline runs, on extra kwargs
    that no backend can use (release-1.0 audit, F01-012/F03-005):
    previously a renamed 0.x kwarg (``group=``) or a misspelled stage
    kwarg (``n_dims=``) ran the whole pipeline and then died with a
    cryptic ``AttributeError: Line2D.set() got an unexpected keyword
    argument ...``. Raises ``TypeError`` naming the kwarg, with a
    did-you-mean hint where one exists."""
    if not extra_kwargs:
        return
    if "group" in extra_kwargs:
        raise TypeError(
            "plot() got an unexpected keyword argument 'group'; group= "
            "was renamed to hue= in hypertools 1.0 -- pass hue= instead.")
    valid = _valid_line2d_kwargs() | set(_PLOTLY_MAPPED_KWARGS)
    unknown = [k for k in extra_kwargs if k not in valid]
    if unknown:
        import difflib
        import inspect
        param_names = set(inspect.signature(plot).parameters) - {"x", "kwargs"}
        candidates = sorted(param_names | valid)
        k = unknown[0]
        match = difflib.get_close_matches(k, candidates, n=1, cutoff=0.6)
        hint = f"; did you mean {match[0]!r}?" if match else ""
        raise TypeError(
            f"plot() got an unexpected keyword argument {k!r}{hint} "
            "(extra keyword arguments are passed through to matplotlib "
            "line artists -- see the **kwargs entry in plot's docstring).")


#: Animate STYLES that implement a dataset-by-dataset (serial) reveal.
#: Membership is tested against the RESOLVED style, never the raw argument:
#: `animate=['morph', None, 'morph']` resolves to 'morph', which is here.
_SERIAL_CAPABLE_STYLES = (True, 'parallel', 'serial', 'morph')
#: Styles that are serial by construction, so `order='parallel'` contradicts.
_INHERENTLY_SERIAL_STYLES = ('serial', 'morph')


def _raw_animate_style(animate):
    """The STYLE `animate=` names, before dataset-count-dependent resolution.

    A per-dataset list/tuple only ever tags datasets for a morph (see the
    list/tuple branch of `_resolve_animate_mode`, immediately below), so its
    style is 'morph' regardless of length or contents. Knowing this WITHOUT
    `n_datasets` is what lets ordering be validated fail-fast, before the
    pipeline.
    """
    if isinstance(animate, (list, tuple)):
        return 'morph'
    return animate


def _resolve_order(animate, order):
    """Validate and resolve ``order=`` against the requested animate STYLE.

    ``animate=`` names the STYLE ('spin'/'window'/'morph'/a parallel reveal)
    and ``order=`` names the ORDERING (all datasets at once, or one after
    another). ``animate='serial'`` predates this split and is a permanent
    alias for ``animate=True, order='serial'``.

    ``order=None`` (the default) means "whatever the style implies": parallel
    for the reveal styles, serial for 'serial'/'morph'. An EXPLICIT
    ``order='parallel'`` therefore contradicts an inherently serial style,
    and says so instead of being silently overridden.

    Returns 'parallel' or 'serial'. Called once, fail-fast, in `plot()`
    beside `_validate_title`, before the analyze/reduce pipeline runs.
    """
    if order is not None and order not in ('parallel', 'serial'):
        hint = (" (for matplotlib's draw-order property, pass zorder=)"
                if isinstance(order, (int, float, np.integer, np.floating))
                and not isinstance(order, bool) else "")
        raise ValueError(
            f"order must be 'parallel' or 'serial' (or None); got "
            f"{order!r}{hint}.")

    style = _raw_animate_style(animate)

    if style in _INHERENTLY_SERIAL_STYLES:
        if order == 'parallel':
            if style == 'serial':
                raise ValueError(
                    "animate='serial' is an alias for animate=True, "
                    "order='serial', so it conflicts with order='parallel'. "
                    "Pass animate=True, order='parallel' for a parallel "
                    "reveal.")
            raise ValueError(
                "animate='morph' is inherently serial (one cloud eases into "
                "the next), so it conflicts with order='parallel'. Drop "
                "order=, or pass animate=True, order='parallel' for a "
                "parallel reveal.")
        return 'serial'

    if order is None:
        return 'parallel'
    if order == 'serial' and not style:
        raise ValueError(
            "order='serial' requires an animated plot; pass animate=True "
            "(or 'serial'/'morph') alongside it. A static plot draws every "
            "dataset at once by definition.")
    return order


def _resolve_animate_mode(animate, n_datasets, order='parallel'):
    """Resolve ``animate=`` for ``animate='morph'`` support (Hungarian
    point-cloud morphs between datasets, maintainer request): `animate` may
    be a single GLOBAL mode (``False``/``True``/``'parallel'``/``'spin'``/
    ``'serial'``/``'morph'``, unchanged from before) OR, ONLY for morph, a
    per-dataset list with ``'morph'``/``None``/``False`` entries (one per
    FINAL -- post cluster/hue-reshape -- dataset, matching `n_datasets`):
    ``'morph'``-tagged datasets join the morph sequence IN LIST ORDER;
    untagged datasets render as static (unanimated) backdrops.

    `order` is `_resolve_order`'s already-validated result. It is folded
    INTO the returned `mode`, so `animate` from this function's call site in
    `plot()` onward is exactly what every backend and every downstream
    consumer should see -- the trail-ignore check (`_trail_ignoring_modes`),
    plotly draw (`plotly_draw`), matplotlib draw (`_draw`), and
    `_apply_multicolor_animation(style=...)` all read that one value, with
    no per-site substitution. `order` is ALSO returned, for consumers that
    need the ordering itself (FrameContext.order, per-segment titles).

    Returns
    -------
    (mode, morph_tags, order)
        `mode` is what every backend actually receives: the raw scalar
        `animate` unchanged, ``'morph'`` if a list was given, or ``'serial'``
        when `order='serial'` folds a parallel-reveal style into the serial
        backend mode. `morph_tags` is ``None`` for every non-morph mode, or a
        list of `n_datasets` bool (``True`` where that dataset joins the
        morph sequence) whenever `mode` is ``'morph'`` (scalar
        ``animate='morph'`` tags every dataset). `order` is `'parallel'` or
        `'serial'`, folded back to `'parallel'` (with a warning) when it does
        not apply to the resolved `mode` (e.g. 'spin'/'window').

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
        mode, morph_tags = "morph", tags
    elif animate == "morph":
        if n_datasets < 2:
            raise ValueError(
                "animate='morph' requires at least 2 datasets to morph "
                f"between; got {n_datasets}."
            )
        mode, morph_tags = "morph", [True] * n_datasets
    else:
        mode, morph_tags = animate, None

    if order == 'serial':
        if mode in (True, 'parallel'):
            # the whole point: a serial ORDERING of a parallel STYLE is
            # exactly the existing 'serial' backend mode
            mode = 'serial'
        elif mode not in _SERIAL_CAPABLE_STYLES:
            # 'spin'/'window' have no dataset-by-dataset reveal. Warn and
            # ignore, matching the established convention for a flag with no
            # meaning in the requested mode (`_trail_ignoring_modes` above)
            # rather than introducing a new hard error class.
            warnings.warn(
                f"animate={mode!r} has no serial ordering (it does not "
                f"reveal datasets one at a time); ignoring order='serial'. "
                "Use animate=True, order='serial' for a serial reveal.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
            order = 'parallel'
    return mode, morph_tags, order


@manage_backend
def plot(
    x,
    fmt="-",
    marker=None,
    markers=None,
    markersize=None,
    linewidth=None,
    alpha=None,
    linestyle=None,
    linestyles=None,
    color=None,
    colors=None,
    palette="hls",
    hue=None,
    color_reduce=None,
    hue_mode=None,
    labels=None,
    names=None,
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
    random_state=None,
    predict=None,
    t=10,
    save_path=None,
    animate=False,
    order=None,
    duration=30,
    tail_duration=2,
    rotations=1,
    zoom=1,
    chemtrails=False,
    precog=False,
    bullettime=False,
    forecast_trail=False,
    forecast_hue=None,
    forecast_cluster=None,
    forecast_n_clusters=None,
    forecast_palette=None,
    forecast_fmt=None,
    slow_warning_seconds=_UNSET_SLOW_WARNING,
    frame_rate=30,
    focused=None,
    morph_samples=None,
    on_frame=None,
    simplify=True,
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
    antialias=True,
    font=None,
    label_alpha=None,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    **kwargs,
):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame, String, or mixed list
        Data for the plot. The form should be samples (rows) by features
        (cols). A plain python list of equal-length numeric lists (e.g.
        ``[[1., 2.], [3., 4.]]``) is treated as ONE dataset, exactly like
        the equivalent ``np.array``. A bare scalar (e.g. ``hyp.plot(5)``)
        is likewise accepted and treated as a single one-column
        observation, drawn as a single point. When a list of several datasets is
        given, every dataset must have the same number of columns
        (features); to combine datasets with different feature counts,
        bring them into a shared space first (e.g.
        ``hyp.plot(hyp.align(data, align='hyper'), ...)``).

        Display space: static plots do NOT draw the input values in their
        original units. The (possibly reduced/aligned) coordinates are
        mean-centered and rescaled into ``[-1, 1]`` (a single shared
        affine transform across all datasets) to fit hypertools' unitless
        square/cube frame -- so coordinates read off the returned Figure
        are an affine image of the analyzed data, not the raw values, and
        scales are not comparable across separately-created figures. Use
        ``return_model=True`` to retrieve the analyzed (pre-rescale) data.

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
        `linewidth`/`alpha` kwarg is ignored (with a ``UserWarning``) since
        MultiIndex grouping owns those. `legend=` is the one overridden
        kwarg that is HONOURED rather than warned away: a list renames the
        top-level groups (one entry per unique top-level value, in
        first-appearance order -- see `legend` below) and ``legend=False``
        suppresses the automatic legend. `names=` (per-INPUT-DATASET
        entries) does not apply -- one frame is drawn as leaves plus
        derived means -- and raises ``ValueError`` pointing at `legend=`.
        `hue=` is superseded with a
        ``UserWarning`` (MultiIndex grouping takes precedence) -- on the
        COLUMN axis a CONTINUOUS hue is instead carried through the
        hierarchy, described below; `cluster=`/
        `n_clusters=` raise ``ValueError`` (both would fight the MultiIndex
        color assignment) -- reset the index first
        (``df.reset_index(drop=True)``) to cluster instead. `predict=` is
        supported since 1.1 (it used to raise): one forecast is computed per
        FINAL trace -- every leaf AND every derived mean, a mean forecast
        from its own averaged trajectory -- so the returned
        ``predict['forecasts']`` lines up 1:1 with ``trace_data``. It needs
        at least 2 rows per trace and raises ``ValueError`` otherwise --
        typically for this axis because the innermost index level is unique
        per row: expansion draws one trace per unique FULL index tuple, so
        such a frame yields one-row traces (and one-row means). The check
        runs on the PLOTTED trajectories, so a row-count-changing analysis
        stage (``manip='Resample'``, a smoother that trims edges) can
        trigger it too; the ``ValueError`` distinguishes the two causes and
        names the one that applies. Row
        averaging assumes member leaves align by row POSITION at each
        timepoint; leaves of unequal length are averaged over their
        overlapping prefix (the shortest member's length), with a single
        ``UserWarning`` per affected group (deduplicated even when a
        3+-level tree causes multiple groupings to share members). Works
        with both static and animated plots and both rendering backends,
        since the expansion happens upstream of drawing. A single-level (or
        default `RangeIndex`) DataFrame, or a plain array/list input, is
        completely unaffected by any of the above.

        A MultiIndex on the **COLUMNS** (``x.columns.nlevels >= 2``) is
        expanded too, since 1.1, under a DIFFERENT rule: the INNERMOST
        column level is the FEATURE axis and every level above it is the
        grouping hierarchy. A ``(Market, Sector, Measure)`` frame therefore
        groups by ``(Market, Sector)`` -- one leaf per sector, each holding
        that sector's measurements as its features -- and gains one market
        mean, for 4 traces. Styling, legend, `linestyle` and the `cluster=`/
        `color=`/`linewidth=` rules above all apply unchanged, reading
        ``n_levels`` as the number of GROUPING levels (2 here, not 3).
        Unlike the row rule, every group keeps all ``len(x)`` rows: column
        grouping never shortens a trace.

        `predict=` works on this axis too, per FINAL trace exactly as
        described above. Grouping never shortens a trace, so the >=
        2-rows-per-trace requirement fails here for one of exactly two
        reasons, and the ``ValueError`` names whichever applies: the INPUT
        frame has fewer than 2 rows (flattening the columns is deliberately
        not suggested for that case, since it cannot add a row), or a
        row-count-changing analysis stage -- ``manip='Resample'``, a
        smoother that trims edges -- shortened the trajectory between the
        input and the plot.

        One thing follows from that rule and is worth stating plainly.
        **Feature correspondence across groups is by NAME, not by
        position.** The innermost labels are feature identities: every group
        must carry the same feature labels, and groups are permuted into the
        first group's order before analysis, so reordering the columns
        within a group changes nothing. Duplicate labels inside one group
        are permitted and matched by ``(label, occurrence)``. A hierarchy
        whose groups hold DIFFERENT labels -- one ticker per sector, say --
        raises ``ValueError`` naming the missing and unexpected features,
        because arbitrary column positions are not corresponding variables
        merely by being written in the same slot. Make the innermost level
        shared measurements (``return``, ``volatility``), reduce each group
        yourself, or, if position *i* really does mean the same feature in
        every group, group them yourself and discard the labels
        deliberately::

            from hypertools.core.hierarchy import group_columns
            leaves, _ = group_columns(df, feature_correspondence='position')
            hyp.plot([leaf.to_numpy() for leaf in leaves])

        That last recipe is a LOWER-LEVEL escape hatch, not positional
        column-hierarchy plotting, and it is **not** equivalent to
        ``hyp.plot(df)``: it draws a plain list of datasets, so there are no
        per-level mean traces, no hierarchy linewidth/alpha/legend styling
        and no ``trace_metadata`` in the return bundle. There is no
        hierarchy-preserving positional mode in 1.1.

        That rule is about correspondence WITHIN a group. The order of the
        GROUPS themselves is still the order you wrote them, and it reaches
        the analysis pipeline: groups become datasets, and `reduce=`
        row-stacks every dataset and fits ONE model on the stack (a single
        shared space), so group order IS row order in that stack. A reducer
        whose fit depends on row order therefore embeds a block-reordered
        frame differently -- including the DEFAULT
        ``reduce='IncrementalPCA'``, which fits by `partial_fit` over
        successive minibatches. On a 40-row frame of 4 sector blocks x 5
        measures, permuting the BLOCKS gave a DIFFERENT embedding under
        ``IncrementalPCA`` and ``TSNE``, while ``PCA``, ``TruncatedSVD``,
        ``FactorAnalysis``, ``Isomap`` and ``SpectralEmbedding`` preserved
        it up to numerical and sign equivalence (a within-group column
        permutation is exactly invariant under all of them, per the rule
        above). No displacement percentage is quoted because it depends on
        the data, the scikit-learn version and the platform, and a flipped
        component sign is the same embedding.

        This is a property of the shared reduction space rather than of
        hierarchies -- ``hyp.plot([A, B, C])`` and ``hyp.plot([C, B, A])``
        differ the same way, and did before 1.1 -- so it is DOCUMENTED, not
        silently worked around. A canonical group order would mean inventing
        a total ordering over arbitrary, mixed-type, NA-bearing labels, and
        would make a labelled hierarchy behave differently from the
        equivalent positional list of datasets. Pass ``reduce='PCA'`` when
        block order must not matter.

        A **continuous** `hue=` is carried THROUGH a column hierarchy rather
        than superseded by it (since 1.1; a row hierarchy still warns and
        ignores). Two forms are accepted, both stated relative to the input
        frame rather than to the drawn figure:

        * a flat sequence of ``len(x)`` values -- shared row-wise values,
          broadcast to every trace;
        * one sequence per leaf, each of ``len(x)`` values -- per-leaf
          values.

        A mean trace takes the **element-wise mean of its members' hue**,
        computed by the same operation that averages their data, so an
        auxiliary value can never drift out of step with the trace it
        describes. Colours are mapped over the concatenation of every
        trace's values, leaves and means together, so one scale spans the
        figure and a `colorbar=` reads against all of it. The hierarchy
        still sets linewidth, alpha and labels; only its colours step aside.
        The per-trace values are returned as ``trace_metadata['aux']``.

        A flat array sized to the TOTAL DRAWN observations is **rejected**,
        not reinterpreted: it is indistinguishable from the first form
        whenever a frame has as many rows as the figure has points, and it
        would require the caller to predict how many mean traces the
        expansion creates. A CATEGORICAL hue still defers to the grouping
        with a ``UserWarning``, because it regroups traces and the named
        leaves would stop existing.

        Note also that `align=` does NOT recover discarded feature identity:
        it aligns the resulting spaces, but by then the reduction has
        already interpreted arbitrary positions as corresponding inputs.

        A frame carrying a hierarchy on **both** axes raises ``ValueError``:
        which one takes precedence is genuinely ambiguous, and before 1.1
        the row path silently won. Flatten one axis and try again.

        Expansion is ONLY applied when a single bare DataFrame is passed as
        `x`, because the hierarchy determines the entire trace list and that
        cannot be reconciled with a caller-supplied list of datasets. If `x`
        is a LIST containing a **row**-MultiIndex DataFrame, the MultiIndex
        is treated as a flat index on that element by the normal
        list-of-datasets pipeline, with a ``UserWarning`` naming its
        position. A **column**-MultiIndex DataFrame in a list instead raises
        ``ValueError`` -- before 1.1 it silently flattened to a single line
        with no warning at all. The asymmetry is deliberate: the row
        behaviour is documented and depended upon, the column one was not.

        See docs/hierarchy.rst for the user-facing guide to all of the
        above: the per-axis comparison table, worked examples of each rule,
        and the return shapes.

    fmt : str or list of strings
        A list of format strings.  All matplotlib format strings are
        supported, including color letters (e.g. ``'ro-'`` draws red
        markers joined by a red line, exactly as in matplotlib; an
        explicit `color=`/`colors=` kwarg wins over a fmt color letter).

        A single fmt string is broadcast to every drawn trace. A fmt LIST is
        distributed one-entry-per-DRAWN-TRACE, and normally there is one
        trace per input dataset. ``hue=``/``cluster=``/``n_clusters=`` (and a
        MultiIndex) regroup the data so the drawn-trace count can differ from
        the input-dataset count; in the ONE reconciled case -- a categorical
        (or cluster) LINE, which splits each dataset into one trace per
        contiguous same-category run so lines never join separate
        trajectories (GH #291) -- a fmt list given at INPUT-dataset length is
        automatically propagated to each dataset's runs, so
        ``hyp.plot([A, B], hue=h, fmt=['.', '-'])`` draws every run of A with
        markers and every run of B as a line. Otherwise (marker-only
        grouping, MultiIndex) the fmt list must match the drawn-trace count.
        A list matching neither the input-dataset count nor the drawn-trace
        count raises a ``ValueError`` naming fmt and both counts. A fmt tuple
        is accepted and treated exactly like the equivalent list.

        Static line rendering is DATA-FAITHFUL: line styles are smoothed
        by PCHIP interpolation, which only ever ADDS points between
        samples -- every original sample (including the final one) is
        always among the drawn line vertices, and trajectories with ~900+
        samples are drawn as-is (never decimated).

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

    alpha : float or list of float
        Opacity in [0, 1], either one value for every dataset or one value
        per dataset (e.g. ``alpha=[0.1, 0.1, 1.0]`` to fade two backdrops
        behind a highlighted third). Inputs that assign alpha internally --
        a row MultiIndex (per-level fading) or a nested list with varying
        nesting depth (per-depth fading) -- keep their own values and warn
        that ``alpha=`` was ignored, rather than silently dropping it. This
        is unconditional: an otherwise-invalid ``alpha=`` (wrong number of
        entries, non-numeric, out of range) is also just ignored-with-a-
        warning in this case, not validated against and raised on --
        whether ``alpha=`` will be used is decided before it is checked.

        A `predict=` forecast overlay is drawn at HALF its dataset's alpha
        (``alpha=[1.0, 0.4]`` gives forecasts at ``[0.5, 0.2]``); an unset
        alpha counts as opaque, so the default forecast alpha is 0.5. See
        `predict` below.

    color(s) : str or list of str
        A list of colors

    **kwargs : any other matplotlib-style keyword argument
        GH #206: any keyword argument that isn't one of `plot()`'s own
        named parameters above is passed straight through to each drawn
        artist -- e.g. `zorder=3`, `dashes=(4, 2)`, `markeredgecolor='k'`.
        Applied VERBATIM, identically, to every drawn dataset -- unlike
        `color`/`marker`/`linestyle`/etc. (see below), an extra kwarg's
        value is NEVER interpreted as "one entry per dataset" even if it
        happens to be a list/tuple (e.g. `dashes=(4, 2)` is a single
        dash-pattern VALUE, not per-dataset values `4` and `2`) -- so there
        is no per-dataset form for an extra kwarg; use one of the dedicated
        per-dataset-aware kwargs (`color`/`marker`/`linestyle`/
        `markersize`/`linewidth`/`alpha`) for that.
        Merged in AFTER the named style kwargs are resolved, so an
        explicit named kwarg (or internal styling logic, e.g. MultiIndex
        grouping's `color`/`linewidth`/`alpha`, `legend=`'s `label`,
        `explore=`'s `picker`) always wins on a naming collision. A kwarg
        that no backend can use (not a matplotlib line-artist property or alias,
        nor a plotly-mappable name) raises ``TypeError`` naming it -- with
        a did-you-mean hint for near-misses of plot's own parameters
        (e.g. ``n_dims`` -> ``ndims``) -- BEFORE the pipeline runs, rather
        than surfacing a cryptic matplotlib internals error after it
        (release-1.0 audit; the legacy 0.x ``group=`` kwarg gets a
        dedicated "renamed to hue=" message). On the plotly backend, only
        a small subset maps onto an actual trace property (`color`,
        `alpha`, `linewidth`, `markersize`, `marker`, `linestyle`,
        `label`); anything else is ignored with a ``UserWarning`` naming
        every unmapped kwarg (rather than raising, since plotly's trace
        objects were never going to support the same kwarg surface as
        matplotlib).

        Every list/tuple-valued NAMED styling kwarg `plot()` itself
        broadcasts (`color`/`colors`, `marker`/`markers`, `linestyle`/
        `linestyles`, `linewidth`, `markersize`, `alpha` -- NOT the generic
        `**kwargs` passthrough above, which is applied verbatim and never
        broadcast, so `zorder=`, `dashes=`, etc. must be a single value) is
        distributed one-entry-per-DRAWN-TRACE and its length is validated
        against the FINAL drawn-trace count (GH #206); a mismatch raises a
        ``ValueError`` naming the kwarg, the length given, and that count
        (previously it silently degraded to `None` for every trace).

        `cluster=`/`hue=`/`n_clusters=`/MultiIndex regroup the data, so the
        final drawn-trace count can differ from the number of INPUT datasets.
        In ONE case the two layouts are reconciled for you: a categorical
        (or cluster) LINE splits each dataset into one trace per contiguous
        same-category run (GH #291), and a style list given at INPUT-dataset
        length is automatically propagated to every run that dataset produced
        -- so ``hyp.plot([A, B], hue=h, linewidth=[1, 3])`` draws all of A's
        runs at width 1 and all of B's at width 3 (a list already at run
        length is used verbatim). For every OTHER regrouping -- marker-only
        hue/cluster grouping (which merges observations across datasets into
        one per-category trace, so a per-dataset style is not even well
        defined), and MultiIndex expansion -- a style list must match the
        resulting drawn-trace count, not the input-dataset count.

    palette : str, list of colors, or matplotlib.colors.Colormap
        A matplotlib or seaborn color palette (name), an explicit list of
        colors (hex strings like '#ff0000', named colors like 'red', or
        RGB(A) tuples -- usable on every path: categorical, continuous,
        matrix hue, and the colorbar), or a matplotlib `Colormap` instance
        (sampled evenly). For a CONTINUOUS `hue`, a short color list is
        blended into a smooth gradient using the listed colors as anchors
        (seaborn ``blend_palette`` semantics); for categorical/matrix hue
        the list must supply at least one color per category/component.
        Note the default 'hls' (like 'husl') is CYCLIC: for a continuous
        `hue` mapping, hypertools samples only ~5/6 of its hue circle so
        the minimum and maximum hue values stay visually distinguishable;
        categorical palettes are used as-is.
        A palette string of the form ``'image:<path>'`` extracts colors from
        a LOCAL image file instead (``palette='image:starry_night.jpg'``):
        six anchor colors, ordered most visually salient first, so a
        painting's vivid subject leads and its muted background follows.
        For a continuous ``hue`` those anchors are blended into a gradient
        exactly as any short color list is. See
        ``hypertools.plot.colors.image_palette`` for the extraction itself
        (and to choose a different number of colors). hypertools never
        downloads the image: fetch and cache it yourself, then pass the path.

    hue : list, numpy array, pandas Series/Index/Categorical, or 2D matrix
        Values used to color the plot, one per observation, matched to the
        observations POSITIONALLY (a pandas Series' index is ignored).
        Accepts categorical labels (one per observation; grouped and
        colored by category), continuous numeric values (mapped through
        the palette; combined with a line format this produces
        multicolored lines whose color varies continuously along each
        trajectory, and a marker+line combo format like ``'o-'`` keeps
        BOTH components -- the multicolored line plus per-point-colored
        markers at the true sample points), or a 2D matrix with one row
        per observation (e.g.
        mixture proportions or model weights; colors are blended per
        observation). Non-finite (NaN/inf) continuous/matrix hue values
        are drawn in a neutral light gray (with a warning) and are
        excluded from the color mapping, so the remaining observations
        keep their full color range. To label a subset of points
        categorically, use None entries (i.e. ['a', None, 'b', 'a']):
        the None-labeled points are drawn in the same de-emphasized
        neutral gray, get no legend entry, and do not consume a palette
        slot (the named categories keep the first palette colors, in
        first-appearance order).

        The categorical-vs-continuous choice: string labels always take
        the CATEGORICAL path (one trace per category, legend-able,
        categories in first-appearance order). A 1-D numeric hue takes
        the CONTINUOUS path (per-point palette-mapped colors, no
        legend), EXCEPT that integer (or boolean) values with at most 12
        unique values -- and fewer unique values than observations --
        are treated as categorical group ids (e.g. the cluster labels
        ``hyp.cluster`` returns): one trace per id, palette-colored and
        legend-labeled in sorted numeric order. Float-valued or
        higher-cardinality integer hues are always continuous. To force
        grouping, pass the ids as strings (``hue=[str(g) for g in
        ids]``); to force a continuous mapping, cast to float
        (``hue=np.asarray(ids, dtype=float)``).

        A SCALAR `hue` (a single string or number, e.g. ``hue='red'``) is
        broadcast to one group covering every observation -- a single
        color -- and emits a `UserWarning`, since this is usually a
        mistake (e.g. a DataFrame column NAME passed seaborn-style; pass
        the column's values, ``hue=df['col']``, instead).

        When the data is a list of datasets, `hue` may mirror that nesting --
        one hue sub-sequence per dataset, each matching that dataset's length
        (e.g. ``hyp.plot([d0, d1], hue=[h0, h1])``); it is flattened to one
        value (or matrix row) per observation.

        Over a **column MultiIndex** the same nesting means one sequence per
        LEAF (each of ``len(x)`` values), and a flat sequence of ``len(x)``
        values is broadcast to every trace; a mean trace's hue is the
        element-wise mean of its members'. A flat array sized to the total
        DRAWN observations is rejected, and a categorical hue still defers
        to the grouping. See `x` for the full rule.

        A 2D matrix hue is MIXTURE WEIGHTS by default when it has at most 3
        columns, and literal RGB when it has more (or when `color_reduce=`
        is given) -- see `color_reduce`. Pass `hue_mode=` to say which you
        meant instead of relying on the width.

    hue_mode : {'mixture', 'rgb'} or None
        How to interpret a 2D matrix `hue` (default: None -- choose by
        width, the historical rule described under `color_reduce`).

        ``'mixture'`` blends each row through `palette` as weights, one
        palette colour per COLUMN, whatever the width. This is what a
        hierarchy needs to make its own colour scheme: give each leaf one
        primary and every derived mean comes out the blend of its children,
        with nothing computing it. Six sectors need six columns, and the
        automatic rule would send exactly that to the reducer instead.

        ``'rgb'`` reduces the matrix to 3 min-max scaled channels used
        directly as (r, g, b), which is what a wide matrix does by default.

        Only meaningful for a matrix hue; anything else raises, as does
        combining ``'mixture'`` with `color_reduce=`. The default is left
        width-based deliberately: changing it would silently repaint every
        existing figure that passes a wide matrix hue.

    color_reduce : str, dict, class, instance, or None
        How to reduce an arbitrary high-dimensional matrix `hue` to the 3
        columns used as (r, g, b). Any `hyp.reduce` spec (default: None ->
        'IncrementalPCA'). Only applies when `hue` is a 2D matrix; the three
        reduced dimensions are min-max scaled to [0, 1] per column and used as
        the red/green/blue channels, so an arbitrary per-observation feature
        matrix becomes a continuous RGB coloring. A matrix `hue` with <=3
        columns is left on the palette-blend path unless `color_reduce=` is
        given explicitly.

    names : list or None
        Per-DATASET names, one per dataset in a list input (default: None).
        Distinct from `labels` (per-POINT text call-outs) and `hue` (per-
        observation coloring): each name labels its dataset's trace and turns
        the legend on, so `hyp.plot([raw, a, b], names=['raw', 'a', 'b'])`
        shows a legend naming the three datasets. Must have exactly one entry
        per dataset; mutually exclusive with passing a `legend=` list (use one
        or the other). Rendered on both the matplotlib and plotly backends.
        Incompatible with a CATEGORICAL `hue` (which regroups the data by
        category, so the drawn traces are no longer the named datasets);
        that combination raises ``ValueError`` -- label the hue categories
        with ``legend=[...]`` instead. Incompatible with a MultiIndex
        HIERARCHY for the same reason (one input frame, drawn as leaves
        plus derived per-level means); that combination also raises
        ``ValueError`` -- name the top-level groups with ``legend=[...]``
        or flatten the hierarchy.

    labels : list
        A list of point labels: exactly one entry per OBSERVATION (row)
        across all datasets, or a nested list with one sub-list per
        dataset; a length mismatch raises ``ValueError`` naming labels and
        both counts. If no label is wanted for a particular point, input
        None for that entry.

        In an ANIMATION whose frame grid is coarser than the data (fewer
        than one frame per sample), each label is attached to the nearest
        drawn frame point, so labels are never silently dropped.

        Supported on BOTH backends (GH #205/#F3): matplotlib draws these as
        `ax.annotate` call-outs; plotly draws the same points as
        `layout.scene.annotations` (3D) or `layout.annotations` (2D), at
        the same data coordinates, honoring the resolved `font=` (see
        below) the same way the legend/colorbar/title do.

    label_alpha : float or None
        Opacity of the translucent background box drawn behind each
        `labels=` point annotation (GH #103). `None` (default) keeps the
        historical opacity, 0.5, on both backends. Must be a number in
        ``[0, 1]``; any other value raises `ValueError`. On matplotlib
        this sets the annotation `bbox`'s `alpha`; on plotly it sets the
        alpha channel of the annotation's `bgcolor`
        (``'rgba(255,255,255,<label_alpha>)'``). Works for both static
        and animated plots (labels are drawn once, at the original data
        coordinates, and persist across every frame on both backends).

    legend : list, str, or bool
        If set to True, legend is implicitly computed from data. Passing a
        list will add string labels to the legend (one for each list
        item); the list must have exactly one entry per drawn dataset/
        group (``ValueError`` naming legend otherwise). A bare string is
        treated as a single-entry list (valid only for a single dataset).

        Under a MultiIndex HIERARCHY the drawn traces are leaves plus
        derived per-level means, and only the TOP-level groups carry a
        legend entry -- so there a list RENAMES those groups: one entry
        per unique top-level index value, in first-appearance order (the
        same convention ``linestyles=`` uses), with any other length
        raising ``ValueError``. ``legend=False`` suppresses the
        hierarchy's automatic legend; ``True``/omitted labels the groups
        with the index values themselves. On a column hierarchy combined
        with a continuous or matrix-valued `hue` any legend is dropped
        with a ``UserWarning`` (that path colors by value, so there are
        no discrete groups to name).

    colorbar : bool or dict
        If True, draws a colorbar reflecting the color mapping in use
        (GH #100). For a continuous 1D `hue` (or continuous `hue` combined
        with a line format, which produces multicolored lines), the
        colorbar is a continuous `ScalarMappable` spanning the ACTUAL
        `hue` value range, using the SAME palette as the lines/markers.
        For discrete groups (categorical `hue`, `cluster`/`n_clusters`, or
        a plain list of datasets with no `hue`/`cluster`), the colorbar is
        segmented (one BoundaryNorm-style block per group), with tick
        labels taken from an explicit ``legend=[...]`` list if given, else
        the categorical `hue`'s own category names (no ``legend=True``
        needed), else ``1..n``. Pass a dict for finer control:
        ``{'label': str, 'ticks': [...], 'location': 'right'|'left'|'top'|
        'bottom'}`` (all keys optional; ``location`` defaults to
        ``'right'``, the same side as the legend -- when both a legend and
        a right-side colorbar are shown, the figure is widened so neither
        is clipped or overlaps the other). Raises ``ValueError`` if
        requested with no color mapping available at all (e.g. a single
        dataset with no `hue`/`cluster`). Default None (no colorbar).

    title : str or list of str
        A title for the plot. Normally a single string. For serial-style
        animations (``order='serial'``, ``animate='serial'`` or
        ``animate='morph'``) you may pass one string per dataset: each is
        shown while its dataset is the one being revealed, and morph
        TRANSITIONS show a blank title so only fully-formed clouds are
        named (a hold and a transition both progress 0 -> 1, so the
        distinction is the segment itself, not how far through it you
        are). Anywhere else a non-string raises ``TypeError``: use
        ``names=`` for per-dataset legend entries, or ``labels=`` for
        per-observation annotations. Rendered identically on the
        matplotlib and plotly backends.

    font : None, str, or matplotlib.font_manager.FontProperties
        Controls the font used for every text surface hypertools draws,
        on BOTH backends (GH #205): point annotations (`labels=`), the
        legend, colorbar tick labels/axis label, and the plot title -- on
        matplotlib via `ax.annotate`/`ax.legend`/etc.; on plotly via
        `layout.scene.annotations`/`layout.annotations`, the legend,
        colorbar title/ticks, and the plot title.

        - `None` (default): hypertools uses its own sans-serif FALLBACK
          STACK, led by the Noto Sans face bundled with the package (SIL
          OFL 1.1, in ``hypertools/external/fonts``). matplotlib is handed
          that font FILE, so the MATPLOTLIB backend renders in the bundled
          Noto Sans identically on every platform. The PLOTLY backend can
          only pass a family NAME to the rendering browser (it cannot use a
          font file), so it *prefers* Noto Sans but falls back to the next
          installed system face when Noto isn't present -- plotly typography
          may therefore vary by platform. Both backends resolve their stack
          PER GLYPH (matplotlib walks a ``font.family`` list; a browser walks
          a CSS stack), so text mixing scripts renders completely from
          several faces (Latin from Noto Sans, Japanese from an installed
          CJK face, math symbols from DejaVu Sans) instead of showing tofu
          for whatever the primary face lacks -- and, crucially, the primary
          face stays Noto Sans, so a stray accent or Greek letter does NOT
          swap the whole plot onto some other font. Only when the stack has
          a genuine COVERAGE GAP (a script no stack family can draw) does
          hypertools scan for an installed font covering that gap and ADD it
          as an extra fallback (Noto stays primary) -- to matplotlib's
          ``font.family`` list and, appended near the end, to the plotly CSS
          stack (the latter still needs that family installed in the browser
          to take effect; see the backend note below). A ``UserWarning`` is
          raised only for characters NOTHING available can draw, naming them.
          Bundling every script is infeasible (a pan-CJK face alone is
          ~16 MB), so for full CJK coverage install a pan-Unicode font --
          ``apt-get install fonts-noto-cjk`` on most Linux distros;
          macOS/Windows usually already ship one (Hiragino Sans/Yu Gothic).
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
        family name (`FontProperties.get_name()`) and puts it at the FRONT
        of hypertools' curated sans-serif CSS stack (e.g.
        ``'"<name>", "Noto Sans", "Helvetica Neue", ..., sans-serif'``),
        and sets it as `layout.font.family`, which every plotly text
        surface hypertools creates inherits unless it overrides its own
        `font.family` (none do, after this change). Static plotly image
        export (`save_path=...png/.jpg` etc., via kaleido) still depends
        on the exporting machine's OS having a font that actually covers
        the requested family/characters -- unlike matplotlib, hypertools
        cannot embed a specific font file into a plotly export.

    xlabel, ylabel, zlabel : str or None
        Axis labels, on BOTH backends, for STATIC and ANIMATED plots, in
        2-D and 3-D (round17 #7). `None` (default): no label, EXCEPT that
        when a single DataFrame with named (non-default, non-duplicate)
        columns is plotted and the drawn axes correspond 1:1 to its
        (df2mat-transformed) columns -- a 2- or 3-column DataFrame drawn
        with no real dimensionality reduction -- the column names become
        the default axis labels (release-1.0 audit, F08-016). Explicitly
        passed labels always win (pass e.g. ``xlabel=''`` to suppress an
        inferred label), and nothing is inferred when `transform=` or
        `pipeline=` replace the standard analysis pipeline. matplotlib:
        `ax.set_xlabel`/`ax.set_ylabel`/`ax.set_zlabel`; hypertools draws
        its own cube/square frame in place of matplotlib's default axes
        box (ticks/spines/panes are hidden), so whenever any of these
        three is given, only the specific label Text artist(s) are kept
        visible rather than the whole axis (ticks/spines/gridlines/3-D
        panes stay hidden either way). plotly: `layout.scene.xaxis.title`/
        `.yaxis.title`/`.zaxis.title` for 3-D, `layout.xaxis.title`/
        `.yaxis.title` for 2-D -- again with only that axis's title shown
        (ticks/gridlines/zero-line stay hidden). `zlabel` on a 2-D plot
        (`ndims` < 3, or data that is intrinsically lower-dimensional)
        raises `ValueError` (no z-axis to label) -- pass `ndims=3` (the
        default) to use `zlabel=`, or use `xlabel=`/`ylabel=` for 2-D
        data.

    size : list
        A [width, height] pair of numbers, in inches, to resize the figure
        (anything else raises ``ValueError`` naming size)

    elev : int or float
        The camera elevation angle, in degrees, for 3-D plots: the angle
        above (positive) or below (negative) the x-y plane (default: 10,
        matplotlib's `Axes3D.view_init` convention). Must be a number;
        ignored for 2-D/1-D plots.

    azim : int or float
        The camera azimuth angle, in degrees, for 3-D plots: the rotation
        of the viewpoint about the z axis (default: -60, matplotlib's
        `Axes3D.view_init` convention). Must be a number; ignored for
        2-D/1-D plots. For every rotating 3-D animation style ('spin',
        'parallel'/True, 'window', 'serial', 'morph') this is the STARTING
        azimuth; the camera sweeps `rotations` full turns from it, so
        rotations=0 gives a fixed camera at exactly this angle.

    normalize : str, False, or None
        If set to 'across', the columns of the input data are z-scored
        across lists. If set to 'within', the columns are z-scored within
        each list that is passed. If set to 'row', each row of the input
        data is z-scored. If set to False or None, no normalization is
        applied (default: None).

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
        the fitted `Pipeline` itself covers. When `x` is a COLUMN-
        hierarchical frame, that grouping is recorded on the passed-in
        `Pipeline` itself (unless it already carries one), so the object
        `return_model=True` hands back re-applies to such a frame -- see
        `return_model` (default: None).

    reduce : str, dict, class, instance, or fitted Reducer
        Decomposition/manifold learning model to use (default:
        'IncrementalPCA'). Models supported: PCA, IncrementalPCA, SparsePCA,
        MiniBatchSparsePCA, KernelPCA, FastICA, FactorAnalysis, TruncatedSVD,
        DictionaryLearning, MiniBatchDictionaryLearning, TSNE, Isomap,
        SpectralEmbedding, LocallyLinearEmbedding, MDS, and UMAP; the mixture
        (soft-clustering) models GaussianMixture, BayesianGaussianMixture,
        LatentDirichletAllocation and NMF (which return per-observation
        membership proportions, GH #174); and the torch-backed autoencoders
        Autoencoder, DeepAutoencoder, SparseAutoencoder,
        ConvolutionalAutoencoder, SequenceAutoencoder and
        VariationalAutoencoder (GH #162, `pip install "hypertools[torch]"`).
        Can be passed as a string, or for finer control of the model
        parameters as a dictionary, e.g.
        reduce={'model': 'PCA', 'kwargs': {'whiten': True}}. See scikit-learn
        specific model docs for details on parameters supported for each model.
        A model INSTANCE (including an already-FITTED reducer, which is
        applied via `.transform` without refitting) is also accepted; if
        its output still has more than 3 dimensions (e.g.
        ``PCA(n_components=5)``), a second display-only reduction with the
        default reducer projects it to 3 dimensions for plotting. If None,
        no reduction is applied -- valid only when the data already has at
        most 3 (or `ndims`) dimensions; otherwise a ``ValueError``
        explains that the data cannot be drawn unreduced.

    ndims : int
        An `int` representing the number of dims to reduce the data x
        to. If ndims > 3, the data is analyzed at that dimensionality but
        plotted in 3 dimensions (a second, display-only reduction with the
        default reducer); use ``return_model=True`` to retrieve the
        higher-dimensional analyzed data. Default is 3 (plot in 3
        dimensions).

    align : str, dict, False, or None
        Alignment model to bring a list of datasets into a shared space.
        If str, 'hyper' (hyperalignment) or 'SRM' (shared response model).
        You can also pass a dictionary for finer control, where the 'model'
        key specifies the model and 'kwargs' holds its parameters, e.g.
        align={'model': 'HyperAlign', 'kwargs': {'n_iter': 10}}. If False or
        None, no alignment is applied (default: None).

    cluster : str, dict, class, instance, False, or None
        If cluster is passed, HyperTools will perform clustering using the
        specified clustering model (a registry name, dict spec, model
        class, or sklearn-API model instance -- an instance's own
        parameters are used, so `n_clusters=` is ignored, with a warning,
        alongside one). Supported algorithms are: KMeans,
        MiniBatchKMeans, AgglomerativeClustering, Birch,
        SpectralClustering, MeanShift, DBSCAN, OPTICS, AffinityPropagation and
        HDBSCAN, plus the mixture (soft-clustering) models GaussianMixture,
        BayesianGaussianMixture, LatentDirichletAllocation and NMF. Can be
        passed as a string, or for finer control of the model parameters as a
        dictionary, e.g. cluster={'model': 'KMeans', 'kwargs': {'max_iter':
        100}}. See scikit-learn specific model docs for details on parameters
        supported for each model. If no parameters are specified a default set
        of parameters will be used: 3 clusters/components for most models
        (the same default as `hyp.cluster`), 20 components for
        LatentDirichletAllocation and NMF (default: None). Clustering runs
        on the REDUCED (post `normalize=`/`reduce=`/`align=`) scores, not
        the raw input -- so LatentDirichletAllocation and NMF, which
        require non-negative data, fail here even when the raw data is
        non-negative (the reduced scores are signed); run `hyp.cluster`
        on the raw data instead to use those models. FeatureAgglomeration
        raises a ``ValueError``: it clusters features (columns), not
        observations, so its labels cannot group the plotted rows -- use
        `hyp.cluster(data, cluster='FeatureAgglomeration')` directly.
        Cluster labels are drawn as one trace per cluster (palette
        colors, legend entries in sorted label order).

    n_clusters : int
        If n_clusters is passed, HyperTools will perform clustering with
        the cluster count set to n_clusters, using k-means unless
        `cluster=` selects another model. The resulting clusters are
        plotted in different colors according to the color palette.
        Default None: each model's default count is used (3, matching
        `hyp.cluster`; 20 components for LatentDirichletAllocation/NMF).
        Ignored, with a ``UserWarning``, for models that discover the
        number of clusters themselves (HDBSCAN, MeanShift, DBSCAN,
        OPTICS, AffinityPropagation). If the `cluster=` spec itself
        carries a cluster count (an instance's own setting, or
        `n_clusters`/`n_components` in a dict spec's kwargs), the spec's
        value wins and a ``UserWarning`` notes the conflict -- the same
        precedence `hyp.cluster` applies.

    random_state : int, RandomState, or None
        Seed for reproducibility, threaded to the reduce/cluster stages: it is
        injected into any stage model whose constructor accepts a
        `random_state` (UMAP, TSNE, KMeans, GaussianMixture, ...), so e.g.
        `hyp.plot(x, reduce='UMAP', random_state=0)` gives a repeatable
        embedding. Deterministic models and pre-constructed instances are
        unaffected (default: None).

    impute : str or dict or class or class instance or None
        Overrides the default PPCA fill for missing (NaN) values with a
        different `hypertools.impute` model, e.g. 'Kalman', 'KNNImputer'
        (default: None, i.e. PPCA -- observed values are preserved
        exactly and only the NaN entries are reconstructed; see
        `hypertools.impute.ppca`). See `hypertools.impute.impute` for
        accepted forms.

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
        one forecast trace per dataset (no separate legend entry). A forecast
        is the SAME series projected forward, so it INHERITS the style of the
        observed trace it continues -- same color, same linestyle, same
        linewidth -- and differs only in transparency: ``forecast_alpha =
        observed_alpha * 0.5`` (an unset `alpha` is matplotlib's opaque 1.0,
        so the default is 0.5). Per-dataset styling carries through dataset
        by dataset, e.g. ``alpha=[1.0, 0.4]`` gives forecasts at
        ``[0.5, 0.2]``, and a dotted dataset gets a dotted forecast. Both
        backends apply the identical rule. (Before 1.1.0 every forecast was
        drawn dashed at a hard-coded alpha of 0.6 regardless of how its data
        was drawn.) The drawn overlay prepends the last observed row so the
        trace connects to the trajectory (`t + 1` drawn vertices); the forecast
        DATA itself -- e.g. in the ``return_model=True`` bundle -- has
        exactly `t` rows, matching `hyp.predict`. Supported for STATIC plots,
        for ``animate='spin'`` (which only rotates the camera around the
        static forecast overlay, so the trace simply rotates with the
        rest of the scene), and for the TIME-PROGRESSING animate modes
        (``True``/``'parallel'``/``'serial'``/``'window'``), where the
        forecast is recomputed from the history revealed so far and
        re-anchored on the last revealed observation, so it grows with the
        animation. Every one of those forecasts is computed BEFORE the first
        frame is drawn and folded into the plot's centre/scale statistics,
        so the whole fan lands inside the cube and nothing is clipped or
        clamped, and every frame is a lookup -- ``ani.save()`` and
        ``to_jshtml()`` replays render identically. NOT supported with
        ``animate='morph'`` (including the per-dataset morph list form),
        which interpolates between point CLOUDS and so has no time axis to
        forecast along; that combination raises ``NotImplementedError``.

        For a **hierarchical** `x` (a row or column MultiIndex; see `x`) the
        unit is the FINAL TRACE rather than the input dataset: one forecast
        per leaf AND one per derived per-level mean, a mean forecast from its
        own averaged trajectory rather than from an average of its members'
        forecasts. ``predict['forecasts'][i]`` in the ``return_model=True``
        bundle therefore equals ``hyp.predict(trace_data[i], model, t)`` for
        every ``i``. Forecasting needs history, so EVERY final trace -- both
        axes, leaves and means alike -- must have at least 2 rows; the first
        that does not raises ``ValueError`` naming the trace, its hierarchy
        key and its row count, before any model is fitted (and, for an
        animated plot, before the per-frame forecast schedule is built, since
        a one-row trace can never reach 2 rows at any frame). The remedy
        differs by axis: a ROW hierarchy draws one trace per unique FULL
        index tuple, so an innermost level that is unique per row yields
        one-row traces -- drop the hierarchy
        (``df.reset_index(drop=True)``) or move the grouping to the columns;
        a COLUMN hierarchy keeps all of the frame's rows in every group, so a
        short trace means the input itself has fewer than 2 observations and
        only more data helps. See docs/hierarchy.rst (default: None).

    t : int or datetime-like
        Forecast horizon passed to `predict` (see
        `hypertools.predict.common.resolve_t`); ignored unless `predict` is
        set. Measured in RAW observations of the analyzed data -- NOT in
        animation frames and NOT in drawn vertices. ``t=1`` forecasts only
        the next observation. Because an animation is paced on a resampled
        frame grid (see `duration`/`frame_rate`), an animated forecast joins
        the drawn trajectory to within one raw observation rather than
        exactly (default: 10).

    save_path : str or path-like
        Path to save the image/movie; the format is chosen by the file
        extension, which must be included (e.g.
        save_path='/path/to/file/image.png'). ``pathlib.Path`` objects work
        everywhere a str does, a leading ``~`` is expanded, and the target
        directory must already exist (a missing directory/empty path/
        non-path value fails fast with a clear error before the plot is
        computed). Supported formats: STATIC matplotlib plots accept any
        `matplotlib.pyplot.savefig` format (.png, .pdf, .svg, .eps, .jpg,
        ...); ANIMATED matplotlib plots accept .gif, .png/.apng (animated
        PNG), and .svg (animated vector graphics) with no extra
        dependencies, plus the video formats .mp4/.mov/.avi/.m4v/.mkv,
        which -- and ONLY which -- require FFmpeg (https://ffmpeg.org;
        e.g. ``brew install ffmpeg`` on macOS with Homebrew
        (https://brew.sh) or ``apt-get install ffmpeg`` on Debian/Ubuntu).
        The plotly backend saves .html natively (static or animated);
        its static image export (.png/.jpg/.svg/.pdf, via kaleido) and
        animated .gif/.png/.apng/video export render each frame through
        kaleido. Note on ANIMATION export cost: every frame is rendered
        and encoded, and the default `duration=30` x `frame_rate=30`
        yields 900 frames -- roughly a minute of encoding and a
        multi-MB file even for small datasets; encoding time and file
        size scale linearly with `duration * frame_rate`, so pass a
        shorter `duration=` (e.g. 2-5 seconds) for quick exports.

    animate : bool, 'parallel', 'spin', 'serial', 'window', 'morph', or list
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
        'serial' COMPOSES with those per-dataset trail flags (chemtrails-
        serial / precog-serial / bullettime-serial) on BOTH backends: the
        ONE dataset currently being revealed also traces out its own
        low-opacity trail (past / future / whole, per its flag) led by a
        short opaque comet-head, while already-revealed datasets stay fully
        drawn and future ones stay invisible -- see `bullettime` below.

        2-D animations (round17 #9, GH #123): every style EXCEPT `'spin'`
        works for `ndims=2` as well as `ndims=3`, in both backends, using a
        FIXED (non-rotating) viewport -- there is simply no camera-angle
        bookkeeping to do in 2-D. `'spin'` rotates the camera and nothing
        else, so it is meaningless for 2-D data and raises `ValueError`
        (naming the other styles) instead of silently doing nothing.
        `rotations=`/`zoom=` are 3-D camera controls with no 2-D
        equivalent; passing either as a non-default value alongside a
        2-D `animate` warns once (`UserWarning`) that it is ignored, in
        both backends -- including `animate='morph'`, whose `rotations`
        doubles as a per-segment PACING control in 3-D (not purely a
        camera control there -- see the note under `'morph'` below), but
        which is ignored the SAME way for consistency with every other
        2-D style: 2-D morphs always use even segment timing.

        If 'window' (round17 #8, GH #275 -- Jeremy's own definition:
        "like bullettime, but without the precog and chemtrail parts"), a
        sliding, FULLY-OPAQUE window of length `focused` (seconds; see
        `focused` below) moves along each trajectory -- nothing outside the
        window is drawn at all, not even a faded trail (unlike `bullettime`/
        `chemtrails`/`precog`, which paint a low-opacity backdrop outside
        their own in-focus window). In 3-D, the camera still rotates at a
        constant speed per `rotations`, exactly like `True`/`'parallel'`; in
        2-D there is no camera, so only the window itself moves. Any of
        `chemtrails`/`precog`/`bullettime` passed alongside `animate='window'`
        is ignored (`UserWarning`, naming the ignored flag(s) and dataset
        indices -- see the note under `bullettime` below), since 'window' has
        no trail artist/trace to configure. Both 2-D and 3-D, in both
        backends.

        If 'morph' (maintainer request, 2026-07-06), every dataset is
        treated as a POINT CLOUD (not a trajectory, regardless of `fmt`) and
        morphed ds_1 -> ds_2 -> ... -> ds_N through a hold/morph/hold/...
        schedule (``2N - 1`` segments: ``[hold_1, morph_1->2, hold_2, ...,
        hold_N]``). Every dataset keeps its FULL point count (maintainer
        request, 2026-07-06 follow-up): the target count `n` is the
        LARGEST morphing dataset's own size (after the optional
        `morph_samples` cap below), and any dataset with `m < n` points is
        padded up to `n` by duplicating `n - m` of its OWN points, chosen
        at random (seeded) -- the padding step itself never drops a real
        data point. Whether one was already dropped EARLIER, by sampling,
        is the caller's documented choice: with an explicit
        `morph_samples=`, or with `simplify=True` (the default) over clouds
        larger than `MORPH_SAMPLES_REQUIRED_ABOVE` = 2000 points, each cloud
        is first downsampled to that cap; with `simplify=False` and no
        `morph_samples=`, every dataset keeps its FULL point count and no
        real data point is ever dropped (see `simplify` below). The
        duplicated (padding) points are hidden during that dataset's own
        HOLD segments (so semi-transparent markers alpha-composite exactly
        like a plain plot of that dataset's true points) and shown, like
        every other point, during MORPH segments. Consecutive (now equal-
        sized, `n`-point) clouds are chain-matched point-for-point with the
        Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on
        pairwise distances, so each point travels the shortest total
        distance to its partner in the next cloud -- exactly
        the shape-morph gallery example's original hand-rolled algorithm
        (now `examples/animate_morph_zoo.py`),
        now built into the library), and eased between clouds with
        smoothstep interpolation. A SINGLE point artist/trace is drawn
        (one per plot, not one per dataset): its color linearly (RGB)
        interpolates between the two datasets' own colors during a morph
        segment and is solid during a hold. Requires at least 2 datasets;
        raises `ValueError` otherwise. Both 2-D and 3-D data are supported
        (round17 #9, GH #123 -- previously 2-D raised `NotImplementedError`);
        `surface=True` recomputes that one artist's hull every frame from
        its current interpolated positions (unaffected by which points are
        duplicates -- a duplicate is an exact copy of an existing point, so
        it never changes a convex hull's shape), but this hull-tracking is
        still 3-D only -- `surface=` is silently a no-op for an animated
        2-D `'morph'` (or any other 2-D animate style; see `surface`'s own
        docstring).

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

        `animate` may ALSO be a `dict` (GH #154 resolution): a mega-dict
        SPEC for the animation, mirroring the model-spec grammar used
        elsewhere in hypertools -- `'style'` plays the role of `model`
        (REQUIRED; the value is any of the scalar `animate` forms above,
        e.g. `'spin'`) and every OTHER key maps onto one of the flat
        animation kwargs below (`duration`, `tail_duration`, `rotations`,
        `zoom`, `chemtrails`, `precog`, `bullettime`, `frame_rate`,
        `focused`, `morph_samples`) -- e.g. ``animate={'style': 'spin', 'rotations':
        2, 'duration': 15}`` is exactly equivalent to
        ``animate='spin', rotations=2, duration=15``. The dict is unpacked
        into the flat kwargs at the very top of `plot()`, before anything
        else runs, so every downstream code path only ever sees the flat
        form. Raises `ValueError` if `'style'` is missing (message shows
        an example dict), if the dict has any key that isn't `'style'` or
        one of the flat animation kwargs above (message lists the valid
        keys), or if a dict key's value CONFLICTS with that same flat
        kwarg passed explicitly (a different value) -- naming the
        conflicting key and both values. This mega-dict form is additive
        sugar, not a new pipeline concept -- flat kwargs remain the
        primary/documented direction (GH #154); note that a `style=`/
        `labels=` mega-dict covering EVERY `plot()` kwarg (not just
        animation) was considered and explicitly rejected as unnecessary
        churn.

        `predict=` (forecast overlays; see below) is compatible with
        `animate='spin'` (which renders a STATIC scene and merely rotates the
        camera, so the fixed forecast overlay is drawn once and rotates along
        with everything else) AND with the time-progressing modes
        (`True`/`'parallel'`/`'serial'`/`'window'`), where the forecast is
        recomputed from the history revealed so far and re-anchored on the
        last revealed observation, so it grows with the animation. Only
        `'morph'` (including per-dataset morph lists) raises
        `NotImplementedError`: a morph interpolates between point CLOUDS, so
        there is no time axis to forecast along. See `forecast_trail=` to
        keep earlier forecasts on screen as a fading fan.

    order : {'parallel', 'serial'}
        Whether animated datasets are revealed all at once ('parallel') or
        one after another ('serial'). This is ORTHOGONAL to ``animate=``,
        which names the style, so it composes with the trail flags:
        ``animate=True, order='serial', chemtrails=True`` is the serial
        version of chemtrails, and renders identically on the matplotlib and
        plotly backends. The default (``None``) means "whatever the style
        implies": parallel for the reveal styles, serial for
        ``animate='serial'`` (a permanent alias for ``animate=True,
        order='serial'``) and ``animate='morph'`` (inherently serial).
        Passing ``order='parallel'`` alongside either of those raises
        ``ValueError``. ``'spin'`` and ``'window'`` have no dataset-by-dataset
        reveal, so ``order='serial'`` warns and is ignored there.
        ``order='serial'`` without an animation raises ``ValueError``.

    backend : str
        Rendering backend: 'matplotlib' (the classic renderer),
        'plotly' (interactive; the `[interactive]` extra installs itself on
        first use), or 'auto' (default), which
        uses plotly on Google Colab / Kaggle notebooks where interactivity
        matters most and matplotlib everywhere else. With the plotly backend,
        the return value is a plotly Figure (any animation frames are
        embedded directly in it, so no separate animation object is
        returned).

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds). Has no
        effect on static plots (static line smoothing uses a fixed
        density, independent of the animation kwargs). Note: when saving
        with `save_path=`, every frame is rendered and encoded
        (`duration * frame_rate` frames -- 900 at the defaults), so
        export time and file size scale linearly with `duration`; use a
        short duration (e.g. 2-5 seconds) for quick exports.

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
        How far to zoom into the plot, positive numbers will zoom in (default: 1)

    chemtrails (animation only) : bool or list of bool
        A low-opacity trail is left behind the trajectory (default: False).
        Pass a list of bool (one entry per drawn dataset -- i.e. the FINAL
        count after any `cluster`/`hue`/`n_clusters` regrouping) for
        per-dataset control (GH #127): e.g. `chemtrails=[True, False]` turns
        chemtrails on for dataset 0 only. A bare bool is broadcast to every
        dataset. Raises `ValueError` if a list's length does not match the
        number of drawn datasets (naming both counts). Trail styles
        (`chemtrails`/`precog`/`bullettime`) apply to `animate=True`/
        `'parallel'` and to `animate='serial'`, on BOTH backends (see the
        note under `bullettime` below).

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
        GH #127: trail styles apply to `animate=True`/`'parallel'` and to
        `animate='serial'`, on BOTH backends -- where they COMPOSE with the
        serial reveal: only the ONE dataset currently being revealed carries
        a trail (chemtrails = its revealed-so-far past, precog = its
        not-yet-revealed future, bullettime / chemtrails+precog = its whole
        trajectory), led by a short opaque comet-head near the reveal tip,
        while already-revealed datasets stay fully drawn and future ones stay
        invisible. `'spin'` has no "current position" for a trail to lead/
        follow (only the camera moves), `'morph'` draws a single traveling
        cloud, and `'window'` is bullettime MINUS its trail by definition, so
        `animate='spin'`/`'morph'`/`'window'` ignore `chemtrails`/`precog`/
        `bullettime` entirely (no trail artist/trace is created) and emit a
        `UserWarning` naming the mode, the ignored flag(s), and which dataset
        indices had them set.

    forecast_trail (animation only) : bool or int
        Keep earlier forecasts on screen as a fading fan -- the forecast
        analogue of `chemtrails` (default: False). Requires `predict=`;
        without it, `ValueError`. `True` retains 16 past forecasts; an int
        sets the cap (must be >= 1). Each retained forecast is drawn in its
        dataset's style, exactly like the live one, at an alpha that decays
        with age from THAT dataset's live forecast alpha down to a floor
        proportional to it -- so a retained forecast is never more opaque
        than the live forecast it fades from, however faint the dataset.
        A viewer can then see how the prediction CHANGED as history
        accumulated.

        The fan is recomputed from the frame index rather than accumulated
        in a buffer, so it depends only on which frame is being drawn: a
        saved GIF and an interactively-played animation are identical, and
        driving frames out of order (which `save()`/`to_jshtml()` do) gives
        the same picture.
        Retained forecasts need no extra room in the plot box: a retained
        forecast is just an earlier frame's, and the box is already built to
        contain every forecast the animation will draw.

    forecast_hue : sequence or None
        Group the FORECASTS by one value per dataset, colouring them
        independently of the observed data (default: `None` -- inherit).
        Requires `predict=`; without it, `ValueError`.

        One value per FORECAST, not per observation: a forecast is a single
        trace. Datasets sharing a value share a colour, drawn from
        `forecast_palette=`. Mutually exclusive with `forecast_cluster=`
        (both decide the same thing), exactly as `hue=` and `cluster=` are
        for the observed data.

        There is one forecast per DRAWN TRACE, which is one per input
        dataset until a hierarchical (MultiIndex) `x=` regroups the data:
        `plot()` forecasts every FINAL trace, so a hierarchy needs one value
        per leaf group PLUS one per derived mean. The same unit applies to
        `forecast_fmt=` and to `forecast_palette=`'s no-grouping case.

    forecast_cluster : str, class, instance, dict, or None
        Colour each forecast by WHERE IT IS PREDICTED TO END UP: the forecast
        ENDPOINTS are clustered, and each forecast takes its cluster's colour
        (default: `None` -- inherit). Requires `predict=`; without it,
        `ValueError`. Takes the same model specs as `cluster=`.

        So a forecast's colour answers "which of these series are heading to
        the same place?" -- a question the observed data cannot answer, which
        is the point of the separate kwarg. It deliberately does NOT
        recluster the observed data: inheriting the observed assignment is
        what the default already does, so that reading would make this a
        no-op. Nor does it cluster every predicted POINT (a single forecast
        would then change colour along its own short path) or whole
        flattened trajectories (sensitive to `t`, to sampling and to
        dimensionality, where an endpoint has one stable meaning).

        The endpoints are taken in the space the figure DRAWS -- after
        `reduce=`/`align=` -- so the grouping matches the geometry on screen
        rather than a pre-reduction one the viewer cannot see. With fewer
        than two forecasts it warns and inherits: every partition of a
        single point is the same partition. The endpoints clustered are
        those of the DRAWN traces' forecasts, so a hierarchical (MultiIndex)
        `x=` clusters its derived means' endpoints alongside its leaves'
        (see `forecast_hue=`).

        In an ANIMATION the groups are resolved ONCE, from the full-history
        forecasts (the ones `return_model=True` returns), and stay fixed for
        every frame -- they are not reclustered as the reveal progresses, so
        the colours hold still while the forecast geometry moves. Cluster
        labels are arbitrary names for groups, so per-frame reclustering
        would let a forecast change colour whenever a fit nudged its
        endpoint across a boundary, and would repaint a retained
        `forecast_trail=` fan whose earlier members were drawn under the old
        grouping.

    forecast_n_clusters : int or None
        Number of groups for `forecast_cluster=` (default: `None` -- the
        clusterer's own default). Separate from `n_clusters=` on purpose:
        the observed observations and the forecast endpoints are different
        point sets, and a good number of groups for one need not be a good
        number for the other. Without `forecast_cluster=` it warns and is
        ignored.

    forecast_palette : str, list of colors, matplotlib Colormap, or None
        Colours for the forecast grouping (default: `None` -- inherit the
        observed colours). Requires `predict=`; without it, `ValueError`.
        Takes the same forms as `palette=`, so the forecasts can use a
        different colormap from the data.

        With `forecast_hue=` or `forecast_cluster=`, one colour per group.
        With NEITHER, there is no forecast grouping to colour by, so it is
        spent one colour per forecast (see `forecast_hue=` on what counts as
        one for a hierarchical `x=`).

    forecast_fmt : str, sequence of str, or None
        Line/marker style for the forecast overlays, in the same format-string
        grammar as `fmt` (default: `None` -- inherit the observed style).
        Requires `predict=`; without it, `ValueError`. One string, or one per
        FORECAST -- which for a hierarchical (MultiIndex) `x=` means one per
        leaf group PLUS one per derived mean, not one per input dataset (see
        `forecast_hue=`).

        Overrides ONLY the style: a dotted forecast of a red trace is still
        red, unless a colour is also given (via `forecast_palette=`,
        `forecast_hue=`, `forecast_cluster=`, or a colour letter in the
        format string itself -- an explicit colour beats the format string's,
        matching matplotlib's own rule).

        Note that these four kwargs are independent, so observed and
        forecast data may differ in style, in grouping, in palette, or in
        any combination. Everything not named stays inherited: a forecast is
        its observed trace projected forward, drawn at half its alpha
        (`hypertools.plot.forecast.FORECAST_ALPHA_SCALE`), and each of these
        replaces exactly one aspect of that.

    slow_warning_seconds (animation only) : float or None
        Warn when an animated `predict=` schedule looks like it will take a
        long time to build (default: 10 seconds). Pass `None` to silence it.

        An animated forecast needs one fit per DISTINCT revealed history
        length, so cost grows with the data rather than with the frame
        count: 3 datasets x 60 rows x 900 frames is 177 fits (~5 s), but
        3 x 500 x 900 is 1497 fits (~330 s) because a longer series has both
        more distinct histories AND a costlier fit each. Nothing is skipped
        to make that faster -- sampling the reveal would change what is
        plotted -- so the notice exists to make a long wait expected rather
        than mysterious. It is emitted as soon as one real fit has been
        timed, not after the wait.

    frame_rate (animation only) : int or float
        Frame rate for animation in frames per second (default: 30).
        Both backends generate exactly ``round(frame_rate * duration)``
        frames -- or a single still frame when that rounds below one -- and
        both play them at ``1000 / frame_rate`` ms per frame, so matplotlib
        and plotly animations play at identical speed, duration, and
        framerate. Has no effect on static plots (static
        line smoothing uses a fixed density, independent of the animation
        kwargs).

    focused (animation only) : float or None
        Round17 #8 (GH #275): the length, in SECONDS -- the SAME unit as
        `tail_duration` -- of the "in-focus" (fully-opaque) window: the
        portion of a trajectory drawn opaque by default under `chemtrails`/
        `precog`/`bullettime`, or the sliding window size for the new
        `animate='window'` (see `animate` above). Default `None`: resolves
        to `tail_duration`'s own value -- today's hardcoded/`tail_duration`-
        derived focus length -- so omitting `focused=` never changes
        existing behavior; pass an explicit `focused=` to decouple the
        in-focus window's length from `tail_duration` (e.g. a wide
        `chemtrails` fade with a narrow opaque head, or vice versa).
        Silently ignored (no error, no warning -- this is the documented,
        expected no-op case) for `animate='spin'`/`'parallel'` (or `True`)
        with NO `chemtrails`/`precog`/`bullettime` flag set on any dataset,
        and for `animate='morph'` -- none of these has a separate "in-focus
        window distinct from the whole trajectory" concept for `focused` to
        control. Must be a non-negative number if given; raises `ValueError`
        otherwise.

    morph_samples (``animate='morph'`` only) : int or None
        An OPTIONAL cap on morphing-dataset size, applied BEFORE the
        duplicate-padding described under `animate` above: any morphing
        dataset larger than `morph_samples` is first downsampled (without
        replacement, seeded) to exactly `morph_samples` points.
        Default `None`: no cap -- every dataset keeps its full point count,
        and the target count is simply the largest dataset's own size (no
        real data point is ever dropped; see `hypertools.plot.morph`). The
        Hungarian assignment's cost is roughly ``O(n^3)`` in the (post-cap)
        target point count, so above 2000 points per cloud an uncapped morph
        is intractable: with the default ``simplify=True`` hypertools caps it
        at 2000 for you, and with ``simplify=False`` it raises ``ValueError``
        naming this parameter rather than appearing to hang. Pass
        ``morph_samples=1000`` (or whatever cap you want) to choose for
        yourself; an explicit value always wins over ``simplify``. Measured
        matching cost: 0.10 s at 1000 points, 0.64 s at 2000, 4.99 s at
        4000; the built-in zoo shapes (~30k points) would need a 7.2 GB cost
        matrix and were still running after 10 minutes.
        Must be a positive integer (or None); anything else
        raises ``ValueError``. Ignored for every other `animate` mode.

    on_frame : callable
        Called after each animation frame is drawn, with a single
        ``FrameContext`` argument exposing the frame index, the axes and
        drawn artists, the arrays being animated, and -- for serial-style
        animations -- which dataset is being revealed, how far through it,
        and the exact per-dataset reveal counts. For ``animate='morph'`` it
        also reports ``segment_index`` and ``segment_kind`` ('hold' or
        'transition'). Use this instead of reaching into matplotlib's
        private ``FuncAnimation._func``. On MATPLOTLIB, callbacks may also
        be attached afterwards via ``HyperAnimation.on_frame()``; this is
        not available on plotly, whose animated return is a plain
        ``go.Figure`` with its frames already built, so pass ``on_frame=``
        here for backend-portable code.

        Supported on BOTH backends, with the same per-frame context
        metadata but different call schedules: matplotlib calls back at
        render time, so a frame index may recur across a looping animation
        or a save; plotly calls back exactly once per frame index, while
        the frames are built. **Callbacks must be deterministic and
        idempotent for a given frame context. They must not depend on call
        count, call order, wall-clock time, or accumulated external
        state.**

        Mutating what the context hands you is the point of the hook and is
        fully supported -- the example below sets a title every frame.
        What is unsupported is accumulation (``count += 1``,
        ``alpha *= 0.9``), because a repeated frame would change the
        result. Precompute running quantities and index them by
        ``ctx.frame``.

        ``ctx.figure``, ``ctx.axes`` and ``ctx.artists`` are backend-native
        (``ctx.axes`` is ``None`` on plotly, whose ``ctx.artists`` are that
        frame's traces), so a callback that touches them is **not**
        portable across backends; every other field is identical across
        backends.

        >>> import numpy as np
        >>> import hypertools as hyp
        >>> data = [np.cumsum(np.random.default_rng(0).standard_normal(
        ...     (20, 3)), axis=0)]
        >>> def annotate(ctx):
        ...     ctx.axes.set_title(f'frame {ctx.frame} of {ctx.n_frames}')
        >>> anim = hyp.plot(data, animate=True, on_frame=annotate, show=False)

    simplify : bool
        Whether hypertools may silently downsample to keep a render
        tractable. Today this governs ``animate='morph'`` **only**: a morph
        over clouds larger than 2000 points is downsampled to 2000 with no
        warning (see ``morph_samples``), because the alternative is a plot
        that never appears. Pass ``simplify=False`` to get an explanatory
        ``ValueError`` instead, so that no real data point is ever dropped
        without you asking. Below the threshold, and whenever you pass
        ``morph_samples=`` yourself, ``simplify`` does nothing at all.

    interactive : bool
        If True, display the plot using an interactive matplotlib
        backend. Useful for inspecting and manipulating static plots. If
        animate=True, an interactive backend is required and this
        argument has no effect (default: False).

    explore : bool
        If True, hovering over a data point displays that point's
        user-defined label (from `labels=`); if no labels were passed, the
        point's index and coordinates are shown instead. Explore mode is
        currently only supported for 3D static plots (``ValueError``
        otherwise), and is an experimental feature (i.e. it may not yet
        work properly). Hover labels require an interactive matplotlib
        backend: under a non-interactive backend (e.g. Agg in scripts,
        CI, or the docs build) the figure is drawn as a static plot and a
        ``UserWarning`` explains that hover labels are unavailable.

    mpl_backend : str
        The matplotlib backend used to create interactive and animated
        plots.  May be 'auto' (default), 'disable', or a backend key
        accepted by matplotlib. If 'auto', hypertools will use a backend
        determined automatically based on your environment
        (`from hypertools.plot.backend import HYPERTOOLS_BACKEND`). If
        'disable',
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
        If set to False, the figure will not be displayed, but it is still
        returned (and remains valid/savable; see Returns). With show=False,
        hypertools also closes/deregisters its pyplot figure once drawing
        (and any `save_path=` export) is done -- including animated figures
        on non-GUI backends -- so batch-export loops never accumulate open
        figures. Note that show=True displays the figure in notebook/
        IPython contexts (the plotly backend displays it once, at the end
        of the cell, whether or not you keep the returned figure; in a
        script it calls its own renderer); in a plain non-interactive
        Python script the matplotlib
        backend registers the figure with pyplot but does not itself call
        ``plt.show()`` -- call ``plt.show()`` yourself to open a window.
        Default: True.

    transform : list of numpy arrays or None
        The transformed data, bypasses transformations if this is set
        (default : None).

    vectorizer : str, dict, class or class instance
        The vectorizer to use. Built-in options are 'CountVectorizer' or
        'TfidfVectorizer'. To change default parameters, set to a dictionary
        e.g. {'model' : 'CountVectorizer', 'kwargs' : {'max_features' : 10}}
        (the legacy {'model', 'params'} form is also still accepted). See
        https://scikit-learn.org/stable/api/sklearn.feature_extraction.html
        for details. You can also specify your own vectorizer model as a class,
        or class instance.  With either option, the class must have a
        fit_transform method (see https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

    semantic : str, dict, class or class instance
        Text model to use to transform text data. Built-in options are
        'LatentDirichletAllocation' or 'NMF' (default: LDA). To change default
        parameters, set to a dictionary e.g. {'model' : 'NMF', 'kwargs' :
        {'n_components' : 10}} (the legacy {'model', 'params'} form is also
        still accepted). See
        https://scikit-learn.org/stable/api/sklearn.decomposition.html
        for details on the two model options. You can also specify your own
        text model as a class, or class instance.  With either option, the class
        must have a fit_transform method (see
        https://scikit-learn.org/stable/data_transforms.html).
        To set parameters, use the dict form (or a configured class
        instance); a bare class is instantiated with its defaults.

    corpus : list (or list of lists) of text samples or 'wiki', 'nips', 'sotus'.
        Text to use to fit the semantic model (optional). If set to 'wiki', 'nips'
        or 'sotus' and the default semantic and vectorizer models are used, a
        pretrained model will be loaded which can save a lot of time.

    ax : matplotlib.Axes or plotly.graph_objects.Figure
        The surface to draw into: a matplotlib Axes for the matplotlib
        backend, or, with the plotly backend, the plotly Figure an earlier
        `hyp.plot` returned -- this call's traces are appended to it and it
        is returned (its layout is left alone). A matplotlib Axes under
        plotly raises `ValueError`; a plotly Figure under matplotlib raises
        `TypeError`.

        STATIC PLOTS ONLY. An animated plot (any truthy ``animate=``) owns
        its own figure: it creates one, draws there, and returns it, so an
        `ax` passed alongside ``animate=`` would be left empty. That
        combination raises ``ValueError`` rather than drawing the right data
        somewhere the caller is not looking. Style an animation through the
        figure it hands back -- ``anim.figure``, ``anim.figure.axes[0]`` --
        which persists: styling applied to that figure after the call
        survives both ``draw_frame`` and ``save`` (measured). Use
        ``on_frame=`` for decoration that must CHANGE with the frame, such
        as a marker tracking the head of a trace.
        Several animated panels in one figure are not supported; lay the
        panels out in the DATA instead (translate each column group into its
        own region of one shared frame) and make a single plot call.

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
        Only a subset of `plot`'s parameters applies to streaming inputs:
        `fmt`, the four `stream_*` parameters, `ndims`, `reduce`,
        `normalize`, `align`/`cluster`/`n_clusters` (rejected with a
        ``ValueError`` -- not yet supported for streams -- but accepted at
        their defaults), `save_path`, `show`, `frame_rate`, `markersize`,
        `linewidth`, `color`, `palette`, `title`, `size`, `elev`, `azim`,
        and `ax`. Any other parameter explicitly set alongside a
        streaming input is ignored, with a ``UserWarning`` naming it. In
        particular, streaming plots are always drawn with the matplotlib
        backend: a `backend=` request (e.g. ``backend='plotly'``) is
        ignored with that warning, and the return value is a matplotlib
        ``Figure`` even when the plotly backend was requested.

    stream_chunk : int
        Streaming data only: number of new samples fetched from the stream
        per update (default: 100). Each fetched chunk is projected through
        the fitted models and rendered as one animation frame / live
        redraw, so this sets both the download batch size and the temporal
        resolution of the resulting animation.

    stream_max : int or None
        Streaming data only: stop streaming after this many samples.
        Exactly `stream_max` samples are consumed from the stream (never
        more), and the returned figure's ``stream_info['truncated']`` is
        then True -- it means streaming was stopped (by `stream_max`, an
        interrupt, or an error) before the stream was observed to end.
        Default None streams continually until the stream is exhausted or
        the user interrupts (Ctrl-C); infinite streams render incoming
        data indefinitely, and any animation being saved via `save_path`
        is finalized whenever streaming stops (including on interrupt).
        For streams, `save_path` supports .gif/.png/.apng (Pillow) and,
        with FFmpeg installed, .mp4/.mov/.avi/.m4v/.mkv; other extensions
        raise ``ValueError`` before any samples are consumed.

    stream_window : int or None
        Streaming data only: if set, only the most recent `stream_window`
        samples are displayed (comet style) while older samples scroll off;
        all consumed samples are still retained on the returned figure's
        ``stream_info`` dict (its ``'data'``/``'xform_data'`` entries).
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
        an unrecognized dict key or an out-of-range dict value (see the
        per-key constraints below), or if a list's length does not match
        the number of drawn datasets. A dataset with too few points to form a
        hull (< 3 for 2D, < 4 for 3D) or whose points are exactly
        collinear/coplanar has its surface silently skipped with a
        ``UserWarning`` (never a crash). Default None (no surfaces).

        Accepted dict keys, with defaults:

        - ``alpha`` (float, default 0.6): surface opacity; must be in
          (0, 1]. A translucent (< 1.0) surface shows the enclosed data
          points through the hull on BOTH backends. Note that a
          translucent 3D matplotlib surface REQUIRES the built-in
          backface culling (always applied) to avoid interior-face
          "cracks" showing through; plotly renders a translucent surface
          as a genuinely translucent ``Mesh3d`` (its doubled-winding mesh
          gets per-layer opacity ``1 - sqrt(1 - alpha)``, compositing to
          exactly ``alpha`` total), which keeps the full mesh but may show
          per-triangle depth-sorting noise (a known WebGL/plotly
          limitation -- plotly.py issue #3554 -- not a hypertools bug) --
          prefer ``alpha=1.0`` if this is objectionable: at ``alpha >=
          0.999`` the plotly mesh instead renders through an artifact-free
          fully-opaque path (the alpha is baked into the surface color),
          and data points enclosed by their own opaque surface are hidden
          from that dataset's trace (they would be invisible behind it
          anyway, and hiding them avoids a WebGL "punch-through" defect).
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
          scales as ``4 ** smoothing``); must be in [0, 6] (beyond 6 the
          face count -- 4096x the raw hull's at 6 -- is a memory/time
          footgun with no visible smoothness gain); ignored for 2D.
        - ``pre_inflate`` (float, default 1.0): scale factor applied to
          the 3D hull about its centroid before smoothing (default: no
          blanket inflation); must be a positive, finite number. Any
          shrinkage smoothing introduces is instead recovered by a
          minimal, grow-only post-hoc rescale targeting ~99% containment
          of the actual input points, so the surface hugs the data rather
          than ballooning past it. The rescale is mathematically bounded
          (hard-capped at 3.0x growth): well-sampled clouds typically need
          at most ~1.25x, and only tiny (4-5 point) hulls -- whose coarse
          meshes lose proportionally far more of their bulge to smoothing
          -- approach the cap (see
          `hypertools.plot.meshutil.smooth_hull_3d`). Ignored for 2D.
        - ``keep_points`` (bool, default True): if False, hides that
          dataset's own line/marker (only the surface is shown). Note
          that on plotly, points enclosed by their own FULLY-OPAQUE
          (``alpha >= 0.999``) surface are hidden even when
          ``keep_points=True`` -- see the ``alpha`` entry above;
          translucent surfaces always show their points.

        Out-of-range values for any key above raise an eager
        ``ValueError`` (naming the key, the constraint, and the received
        value) BEFORE the analyze/reduce pipeline runs, exactly like
        `density`'s validation.

        Animated plots (matplotlib and plotly, 3D only -- round17 #9, GH
        #123: 2-D `animate` is now supported, but per-frame hull tracking
        is not, so `surface=` is silently a no-op on an animated 2-D plot,
        in both backends) recompute each dataset's hull every frame from
        its CURRENTLY VISIBLE window:
        the revealed portion for ``animate='serial'``, the sliding
        head/tail window for ``animate=True``/``'parallel'`` (matching the
        window drawn by `chemtrails`/`tail_duration`), or the full,
        precomputed-once dataset for ``animate='spin'`` (only the camera
        orbits, so only per-frame shading/backface-culling -- not the mesh
        itself -- needs recomputing). Animated surfaces keep the same
        per-vertex `hue` coloring static surfaces use (each frame's hull
        is colored from its currently-visible points' own hue colors) on
        both backends. Surfaces never gain a legend entry
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
        above for the exact spacing); scikit-image (the `[density3d]`
        extra) installs itself on first use, and only if that is not
        possible does it fall back to
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
        3-D layer is a `go.Volume` with, for a scene-filling dataset
        (boost=1), `isomin=0.05`, `isomax=1.0`, `surface_count=5*levels`,
        `opacity=min(2 * alpha, 0.4)`, and an `opacityscale` ramp tuned so
        the volume stays visible at plotly's 3-D scene scale, over a solid
        per-dataset colorscale. For a dataset SMALL relative to the scene
        (e.g. widely-separated clusters), the auto-boost shifts all of
        these together -- `opacity` and `surface_count` scale up (opacity
        capped at 0.75), `isomin` drops (down to 0.01), and the
        `opacityscale` breakpoints and the KDE grid's padding widen to
        expose more of the KDE's outer tail -- see
        `hypertools.plot.density.resolve_plotly_volume_params` for the
        exact formulas. Density layers never gain a legend entry in
        either backend.

        3-D static-export caveat (both backends): when `per_group=True`
        (the default) draws more than one dataset's translucent 3-D density
        layer, the overlapping surfaces/volumes can composite unevenly in
        STATIC exports (PNG/SVG via matplotlib's Agg renderer or plotly's
        `kaleido`-based ``write_image``/``to_image``) -- a WebGL/rasterizer
        alpha-blending-order limitation, not a data or fitting bug. The
        interactive view (a live matplotlib window or plotly's browser/
        notebook widget) renders correctly; only static snapshots of
        multi-dataset 3-D density can look off.

    antialias : bool
        Automatically smooth every drawn LINE so there are no sharp angles
        between successive observations. Default ``True`` (on for every
        plot, static and animated, in both backends).

        Trajectories are upsampled along a monotone PCHIP interpolant, which
        is C1 -- its tangent is continuous, so the drawn curve bends smoothly
        through each sample instead of turning a corner at it -- and every
        original sample remains an exact vertex of the drawn line, so this
        changes only how the data is DRAWN, never the data itself (returned
        arrays, `return_model=True` bundles, forecasts, hulls, densities and
        per-point labels/markers are all unaffected).

        Applied at the LAST stage before drawing, so it composes with
        everything upstream. In an ANIMATION each frame draws the smooth
        curve for exactly the portion of the trajectory that frame would
        have shown -- so a short animation of a finely-structured trajectory
        (many tight loops) renders as smooth curves rather than as one coarse
        straight segment per frame, at any `frame_rate`.

        Only applies to styles that draw a LINE (solid or dashed/dotted --
        e.g. ``'-'``, ``'--'``, ``':'``, and marker+line combos like
        ``'o-'``). MARKER-ONLY styles (e.g. ``'o'``, ``'.'``) are never
        touched: markers always render at the true sample points. Forecast
        overlays drawn by `predict=` are smoothed the same way.

        Pass ``antialias=False`` to draw raw straight segments between
        consecutive samples (the pre-1.1.0 behavior).

        Animated plots (both backends, any `animate` style): the density
        is computed ONCE from the FULL dataset and drawn as a static
        background -- a single KDE evaluation is far too slow
        (~500ms at a 50^3 grid) to redo every animation frame, so, unlike
        `surface`, the density layer does not track the currently-visible
        window and is never touched by per-frame updates.

    return_model : bool
        If True, return a dict bundle
        ``{'fig': ..., 'xform_data': ..., 'trace_data': ...,
        'trace_metadata': ..., 'animation': ..., 'pipeline': ...,
        'models': ..., 'predict': ...}`` instead of the bare figure, where
        ``xform_data`` is the normalized/reduced/aligned data, ``animation``
        is the ``matplotlib.animation.Animation`` handle (``None`` unless
        ``animate=True`` with the matplotlib backend) -- this is the RAW
        handle, never wrapped in a ``HyperAnimation``, so it has no
        ``.on_frame()``; pass ``on_frame=`` to this call instead, which
        still fires regardless of ``return_model=``. ``pipeline`` is a
        fitted `hypertools.Pipeline` covering whichever of `manip=`/
        `normalize=`/`reduce=`/`align=`/`cluster=` ran (the SAME `pipeline=`
        object passed in, if any; `None` when `transform=` was used, since
        then there is no raw data to have fit one on) -- pass it back in as
        `hyp.plot(new_data, pipeline=bundle['pipeline'])` to reuse these
        exact fitted parameters (GH #227). For a COLUMN-hierarchical frame
        the pipeline is fit on the frame's GROUPS, each as wide as ONE
        group, so it also remembers that grouping: calling
        ``bundle['pipeline'].transform(df)`` on a column-hierarchical frame
        groups it first and returns one array per group (before 1.1.0 this
        raised scikit-learn's "X has 20 features, but IncrementalPCA is
        expecting 5 features" -- or, when the reduce stage was a no-op
        because each group already had <= `ndims` columns, silently returned
        the ungrouped frame). A pipeline the caller passed in via `pipeline=`
        has that grouping recorded ON IT (in place -- it is the same object
        the bundle hands back) when it does not already carry one of its
        own, so it re-applies on the same terms. Feature correspondence is
        BY NAME there too:
        the frame's innermost column labels are matched to the ones the
        pipeline was fit on, so reordering them is harmless and naming
        different measurements raises. ``models`` holds the
        reduce/align/cluster/impute specs, and ``predict`` is ``None`` unless
        `predict` was set, in which case it is
        ``{'model': ..., 'params': {'t': t}, 'forecasts': [...]}`` (one
        forecast array per input dataset -- or, for a HIERARCHICAL input, one
        per FINAL TRACE, i.e. per leaf and per derived per-level mean, each
        mean forecast from its own averaged trajectory -- in the
        analyzed/plotted, pre-center/scale, space). Each bundled forecast has
        exactly `t` rows, matching what ``hyp.predict(xform_data,
        model=..., t=t)`` returns. A hierarchy additionally requires at least
        2 rows in EVERY final trace, on either axis, and raises otherwise;
        see `predict`.

        ``xform_data`` vs ``trace_data``. ``xform_data`` is the analysed
        pipeline output, one entry per analysed INPUT dataset.
        ``trace_data`` is the final PRE-CENTER/PRE-SCALE plotted
        trajectories -- for a hierarchical input, the leaves followed by the
        per-level means, which are presentation artifacts built in display
        space and so are deliberately absent from ``xform_data``. The two
        are the same object only when no display-only projection occurred;
        if a ``reduce=`` spec pins more than three components, ``xform_data``
        keeps that many while ``trace_data`` is projected to the plotted
        dimensionality. Neither is what the artists hold: those are centered,
        scaled and (unless ``antialias=False``) PCHIP-upsampled afterwards.
        Bundled forecasts always correspond to ``trace_data``, so
        ``forecasts[i]`` matches ``hyp.predict(trace_data[i], model=..., t=t)``
        for every i; they match ``hyp.predict(xform_data, ...)`` element-wise
        only when the two spaces coincide -- the usual case.

        ``trace_metadata`` is ``None`` for non-hierarchical input. For a
        hierarchy it describes every entry of ``trace_data`` positionally:
        ``{'keys', 'level_idx', 'is_mean', 'axis', 'level_names', 'aux'}``.
        See docs/hierarchy.rst.

        ``predict`` also carries ``'drawn'`` (bool) and ``'draw_reason'``
        (``None``, or a sentence naming the limitation). The forecasts are
        reported whenever the FIT succeeded, whether or not the figure could
        render them: `return_model=` hands back model output, and throwing a
        valid result away because a rendering combination is unsupported
        would discard the very thing it exists to return. ``drawn`` is what
        keeps the two cases distinguishable.

        There are three cases:

        1. **A plain forecast** (no `hue=`/`cluster=`), static or animated.
           Drawn, and ``drawn`` is True. For an ANIMATION the bundle is the
           END state: at the final frame the revealed history *is* the full
           history, so the full-history forecast the bundle carries is
           exactly the one the last frame draws; at every EARLIER frame the
           drawn forecast is computed from the history revealed so far and
           so is deliberately different. The bundle is not a per-frame
           record.

        2. **A STATIC plot with `hue=` or `cluster=`.** These regroup the
           drawn traces by category rather than by dataset, so there is no
           longer one trace per dataset -- but there is still one forecast
           per dataset, each anchored at that dataset's last observation and
           taking the style of the trace holding it, which is the trace it
           visually continues. That holds for a categorical `hue=`, a
           continuous `hue=` (where the colour is taken at the anchor, since
           the trace has many) and `cluster=`. Drawn, and ``drawn`` is True.

        3. **An ANIMATED plot with `hue=` or `cluster=`.** Not drawn, and
           warns: the per-frame schedule maps frame-grid rows onto each
           DATASET's raw observations, and regrouping leaves only per-run
           traces to reveal, with no per-dataset reveal to schedule against.
           ``drawn`` is False and ``draw_reason`` says why; ``forecasts``
           still holds the full-history fit. Plot statically to SEE
           forecasts alongside `hue=`/`cluster=`.

        The DRAWN overlay is not vertex-for-vertex the bundled
        array: it prepends the last observed row as a connector (`t + 1`
        vertices), and with `antialias=True` (the default) it is then
        PCHIP-densified well beyond that, so a short forecast still renders
        as a smooth curve rather than a few straight segments. Pass
        `antialias=False` to draw exactly the `t + 1` raw vertices.
        Default False.

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly Figure
        The rendered figure. Static plot coordinates are drawn in the
        centered/rescaled ``[-1, 1]`` display space described under `x`
        above. For animated matplotlib plots a ``HyperAnimation`` is
        returned instead: a ``(fig, animation)`` tuple subclass (so
        ``fig, anim = hyp.plot(...)`` unpacking works) that also exposes
        ``.figure``/``.to_html5_video()``/``.to_jshtml()``/``.save()`` and
        auto-plays inline in notebooks -- keep a reference to it so the
        underlying ``matplotlib.animation.FuncAnimation`` stays alive.
        When ``return_model=True``, a dict
        ``{'fig': ..., 'xform_data': ..., 'trace_data': ...,
        'trace_metadata': ..., 'animation': ..., 'pipeline': ...,
        'models': ..., 'predict': ...}`` is returned (``animation`` included
        so the handle isn't dropped for animated plots; ``pipeline`` is the
        fitted `hypertools.Pipeline` covering the stages that ran, reusable
        via ``hyp.plot(new_data, pipeline=...)``).

    Examples
    --------
    Plot a single high-dimensional dataset as a static 3-D trajectory (the
    data is reduced to 3 dimensions with the default reducer):

    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((50, 8)),
    ...               axis=0)
    >>> fig = hyp.plot(x, show=False)
    >>> fig.axes[0].name
    '3d'

    Plot two datasets as labeled point clouds (one legend entry each):

    >>> fig = hyp.plot([x, x + 10], '.', names=['a', 'b'], show=False)
    >>> [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    ['a', 'b']

    Color a trajectory continuously by time, in a 2-D projection:

    >>> fig = hyp.plot(x, ndims=2, hue=np.arange(50), show=False)
    >>> fig.axes[0].name
    'rectilinear'

    """

    # early kwarg validation (release-1.0 audit): catch renamed/misspelled/
    # unknown keyword arguments HERE, with a clear TypeError naming the
    # kwarg (plus a did-you-mean hint), BEFORE the expensive analyze/
    # reduce/align pipeline runs.
    _validate_extra_plot_kwargs(kwargs)

    # fmt: accept plain-bytes format strings like np.bytes_ (F01-017) --
    # decoded here once so every downstream fmt consumer sees str.
    if isinstance(fmt, bytes):
        fmt = fmt.decode("utf-8")
    # a fmt TUPLE is normalized to a list up front so every downstream
    # consumer (which tests `isinstance(fmt, list)`) handles it identically
    # -- previously a tuple passed the list/tuple validation below but then
    # fell through the list-only branches and surfaced an unrelated internal
    # error (e.g. "'<=' not supported between 'int' and 'str'") whose
    # occurrence depended on the hue pattern (reviewer follow-up).
    if isinstance(fmt, tuple):
        fmt = list(fmt)
    if isinstance(fmt, list):
        fmt = [f.decode("utf-8") if isinstance(f, bytes) else f for f in fmt]

    # fmt must be a format string (or a per-dataset list of them):
    # fmt=123 used to run the whole pipeline and die in a bare
    # "object of type 'int' has no len()" that never named the kwarg
    # (release-1.0 audit, X2-error-quality-015).
    if fmt is not None and not isinstance(fmt, str):
        if not (isinstance(fmt, (list, tuple))
                and all(isinstance(f, str) for f in fmt)):
            raise TypeError(
                f"fmt must be a matplotlib format string (e.g. '-', '.', "
                f"'o:') or a list of format strings (one per dataset); "
                f"got {fmt!r}.")

    # transform= is pre-transformed DATA (bypassing the analysis pipeline),
    # not a model spec: a bad value used to crash later with "'str' object
    # has no attribute 'shape'" (release-1.0 audit, X2-error-quality-015).
    if transform is not None:
        _xf_items = transform if isinstance(transform, (list, tuple)) \
            else [transform]
        for _xf in _xf_items:
            if not (hasattr(_xf, 'shape') or hasattr(_xf, '__array__')):
                raise TypeError(
                    f"transform= must be already-transformed data (a numpy "
                    f"array/DataFrame, or a list of them), or None; got "
                    f"{type(_xf).__name__}: {_xf!r}. To choose a "
                    "dimensionality-reduction model, pass reduce= instead "
                    "(transform= bypasses the analysis pipeline entirely).")

    # elev=/azim= must be numbers (degrees). Previously a bad value ran the
    # whole pipeline and only crashed at DRAW time with a message that never
    # named the kwarg (F10-014).
    for _angle_name, _angle_value in (("elev", elev), ("azim", azim)):
        if isinstance(_angle_value, bool) or not isinstance(
                _angle_value, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{_angle_name}= must be a number (the camera "
                f"{'elevation' if _angle_name == 'elev' else 'azimuth'} in "
                f"degrees); got {_angle_value!r}.")

    # size= must be a [width, height] pair of numbers; the raw matplotlib
    # unpack error never mentioned size= (F10-012).
    if size is not None:
        _size_ok = (not isinstance(size, (str, bytes))
                    and hasattr(size, "__len__") and len(size) == 2
                    and all(isinstance(v, (int, float, np.integer,
                                           np.floating))
                            and not isinstance(v, bool) for v in size))
        if not _size_ok:
            raise ValueError(
                "size= must be a [width, height] pair of numbers (the "
                f"figure size in inches); got {size!r}.")

    # ax= must be a matplotlib Axes; a bad value crashed deep inside the
    # backend with "'str' object has no attribute 'name'" (F10-015).
    # ax= names the surface to draw INTO: a matplotlib Axes for the
    # matplotlib backend, or a plotly Figure (the object an earlier
    # `hyp.plot` returned) for the plotly backend, whose traces are appended
    # to it. Until 1.1 the plotly backend silently ignored a matplotlib Axes
    # (the figure was built as a plotly Figure and the caller's axes left
    # empty; measured 2026-09-04 on Colab, where 'auto' resolves to plotly:
    # a two-panel before/after layout showed two empty 3-D cubes).
    _plotly_into = None
    if ax is not None:
        import matplotlib.axes as _mpl_axes
        _is_mpl_axes = isinstance(ax, _mpl_axes.Axes)
        _is_plotly_fig = _is_plotly_figure(ax)
        if not _is_mpl_axes and not _is_plotly_fig:
            raise TypeError(
                "ax= must be a matplotlib Axes (2-D) or Axes3D (3-D) "
                "instance, or a plotly Figure to draw into with the plotly "
                f"backend; got {type(ax).__name__!r}.")
        if resolve_backend(backend) == "plotly":
            if _is_mpl_axes:
                raise ValueError(
                    "ax= is a matplotlib Axes, and the plotly backend cannot "
                    "draw into it. Pass a plotly Figure instead (the one an "
                    "earlier hyp.plot returned) to draw into it, drop ax= to "
                    "get a new Figure, or draw this call with matplotlib: "
                    "hyp.plot(..., backend='matplotlib') or "
                    "`with hyp.set_interactive_backend('matplotlib'):`.")
            _plotly_into = ax
            ax = None
        elif _is_plotly_fig:
            raise TypeError(
                "ax= is a plotly Figure, but this call draws with matplotlib; "
                "pass backend='plotly' (or a matplotlib Axes).")

    # a bare scalar is plotted as a single 1-D point -- warn rather than
    # doing so silently (D11-014).
    if isinstance(x, (int, float, np.integer, np.floating)) \
            and not isinstance(x, bool):
        warnings.warn(
            "x is a single scalar value; hypertools will plot it as a "
            "single 1-D point. Pass an array/list of observations for a "
            "meaningful plot.", stacklevel=external_stacklevel())

    # align=False / cluster=False are documented as "no alignment" / "no
    # clustering" (same as None); normalize them here so the stage
    # dispatchers below never see a bare False (F03-003/F03-004).
    if align is False:
        align = None
    if cluster is False:
        cluster = None

    # a bare string for names=/legend= is ONE name, not a sequence of
    # single-character names (F10-009: names='ab' silently became
    # ['a', 'b']); wrap it so the per-dataset length checks below apply.
    if isinstance(names, str):
        names = [names]
    if isinstance(legend, str):
        legend = [legend]

    # legend= must be a bool, a label string, or a list of labels: any
    # other scalar (e.g. legend=7) was silently treated as truthy
    # (release-1.0 audit, X2-error-quality-016).
    if legend is not None and not isinstance(
            legend, (bool, np.bool_, list, tuple, np.ndarray, pd.Series,
                     pd.Index)):
        raise TypeError(
            f"legend= must be True/False, a label string, or a list of "
            f"labels (one per drawn trace/group); got "
            f"{type(legend).__name__}: {legend!r}.")

    # ...and every non-list container the check above ACCEPTS is normalised
    # to a plain list right here, so the label handling downstream (all of
    # which tests `isinstance(legend, (list, tuple))`, or just `list`) sees
    # one type. Measured on `hyp.plot([A, B], legend=<container>)`:
    # ndarray/Series/Index skipped the per-trace length check at
    # "legend= was given as a list of length ..." below and were handed to
    # matplotlib WHOLE as each artist's label, so both traces came out
    # named "['a' 'b']" / "Index(['a', 'b'], dtype='str')" plus two
    # "Passing label as a length 2 sequence" UserWarnings; a tuple labelled
    # the traces correctly but missed `_build_colorbar_info`'s
    # `isinstance(legend, list)` branch, so `legend=('A', 'B'),
    # colorbar=True` drew a colorbar reading ['1', '2']. The hierarchy path
    # already accepted all four containers correctly, so the two paths
    # disagreed about the same accepted input.
    # A 0-d ndarray (`np.array('a')`) is not iterable, so it becomes a
    # ONE-label list -- the same thing the `isinstance(legend, str)` wrap
    # above does with `legend='a'`, which then reports the length mismatch
    # instead of silently broadcasting that one label over every trace.
    if isinstance(legend, (tuple, np.ndarray, pd.Series, pd.Index)):
        legend = ([legend.item()]
                  if isinstance(legend, np.ndarray) and legend.ndim == 0
                  else list(legend))

    # Did the CALLER pass legend= as an explicit list of labels? Recorded
    # HERE, while `legend` still holds exactly what was passed in (the
    # string wrap above has already run, so legend='a' counts as a list of
    # one). Every later reader must consult this flag rather than
    # re-testing `isinstance(legend, list)`: the MultiIndex branch below
    # installs the hierarchy's own per-trace labels INTO `legend`, so from
    # that point on "legend is a list" no longer means "the user passed
    # one". Two user-visible bugs came from exactly that confusion --
    # `legend=[...]` was overwritten without a word (while every sibling
    # kwarg the hierarchy overrides warns), and `names=` ALONE raised
    # "pass dataset names via names= OR a legend= list, not both" on a
    # call that never mentioned legend=.
    _legend_user_list = isinstance(
        legend, (list, tuple, np.ndarray, pd.Series, pd.Index))

    # animate= dict form (GH #154 resolution): unpacked into the flat
    # animation kwargs HERE, at the very top of the function, before
    # anything else runs -- so every downstream code path (all of which
    # predates this feature) only ever sees the flat kwargs it already
    # understands; `animate` itself becomes the resolved style string/
    # bool/list from here on. `'style'` plays the role of `model` in
    # hypertools' usual spec-dict grammar; every other key must be one of
    # the flat animation kwargs below. A dict key CONFLICTING with the
    # same flat kwarg passed explicitly (a different value) is almost
    # certainly a mistake -- raise rather than silently pick one; compared
    # against each flat kwarg's own LITERAL default (mirroring the
    # pipeline=/stage-kwarg conflict check below) since there is no other
    # way to tell "explicitly passed, coincidentally equal to the default"
    # from "never passed" from inside the function body.
    if isinstance(animate, dict):
        _ANIMATE_DICT_DEFAULTS = {
            'duration': 30,
            'tail_duration': 2,
            'rotations': 1,
            'zoom': 1,
            'chemtrails': False,
            'precog': False,
            'bullettime': False,
            'frame_rate': 30,
            'focused': None,
            'morph_samples': None,
        }
        if 'style' not in animate:
            raise ValueError(
                "animate= dict form requires a 'style' key naming the "
                "animation style (e.g. animate={'style': 'spin', "
                "'rotations': 2, 'duration': 15}); got a dict with keys "
                f"{sorted(animate.keys())}."
            )
        _animate_dict = dict(animate)
        _animate_style = _animate_dict.pop('style')
        _unknown_animate_keys = set(_animate_dict) - set(_ANIMATE_DICT_DEFAULTS)
        if _unknown_animate_keys:
            raise ValueError(
                f"animate= dict got unknown key(s) "
                f"{sorted(_unknown_animate_keys)}; valid keys are 'style' "
                f"plus {sorted(_ANIMATE_DICT_DEFAULTS)}."
            )
        _animate_flat_locals = {
            'duration': duration,
            'tail_duration': tail_duration,
            'rotations': rotations,
            'zoom': zoom,
            'chemtrails': chemtrails,
            'precog': precog,
            'bullettime': bullettime,
            'frame_rate': frame_rate,
            'focused': focused,
            'morph_samples': morph_samples,
        }
        for _key, _dict_value in _animate_dict.items():
            _default_value = _ANIMATE_DICT_DEFAULTS[_key]
            _flat_value = _animate_flat_locals[_key]
            if _flat_value != _default_value and _flat_value != _dict_value:
                raise ValueError(
                    f"animate= dict specifies {_key}={_dict_value!r} but "
                    f"{_key}={_flat_value!r} was also passed explicitly as "
                    f"a flat kwarg with a different value; pass {_key}= in "
                    "only one place (either inside animate= or as its own "
                    "kwarg)."
                )
        duration = _animate_dict.get('duration', duration)
        tail_duration = _animate_dict.get('tail_duration', tail_duration)
        rotations = _animate_dict.get('rotations', rotations)
        zoom = _animate_dict.get('zoom', zoom)
        chemtrails = _animate_dict.get('chemtrails', chemtrails)
        precog = _animate_dict.get('precog', precog)
        bullettime = _animate_dict.get('bullettime', bullettime)
        frame_rate = _animate_dict.get('frame_rate', frame_rate)
        focused = _animate_dict.get('focused', focused)
        morph_samples = _animate_dict.get('morph_samples', morph_samples)
        animate = _animate_style

    # animate='chemtrails'/'precog'/'bullettime' sugar (QC 2026-07): these are
    # trail EFFECTS, not animation styles. Historically, passing one as the
    # animate STYLE silently produced a STATIC plot (the style whitelist in the
    # matplotlib backend did not recognize them). Map each to animate='parallel'
    # with the matching trail flag on -- what the effect actually needs (trails
    # apply to animate=True/'parallel').
    if isinstance(animate, str) and animate in ('chemtrails', 'precog',
                                                'bullettime'):
        if animate == 'chemtrails':
            chemtrails = True
        elif animate == 'precog':
            precog = True
        else:
            bullettime = True
        animate = 'parallel'

    # validate the animate style: an unrecognized string used to fall through
    # to a silent static plot (QC 2026-07). Fail fast with a clear message.
    if isinstance(animate, str) and animate not in ('parallel', 'spin',
                                                    'serial', 'morph', 'window'):
        raise ValueError(
            f"unknown animate style {animate!r}; valid styles are 'parallel', "
            "'spin', 'serial', 'morph', 'window' (or True/False). The trail "
            "effects 'chemtrails'/'precog'/'bullettime' are boolean kwargs, not "
            "styles -- e.g. animate='parallel', chemtrails=True.")

    # non-bool/non-string scalars (release-1.0 audit, F04-006/F05-004):
    # anything ==True/==False (np.True_, 1, 0, ...) is normalized to a real
    # bool; every OTHER scalar (e.g. animate=2 -- perhaps meant as
    # "2 rotations"?) used to slip past the string whitelist above, silently
    # render a STATIC plot (the backend dispatch is `animate in [True,
    # 'parallel', ...]`), and then crash with `AttributeError: 'NoneType'
    # object has no attribute 'save'` if save_path= was also set.
    if isinstance(animate, np.ndarray):
        animate = animate.tolist()  # per-dataset morph tags as an array
    if animate is not None and not isinstance(animate, (bool, str, dict,
                                                        list, tuple)):
        if animate == True or animate == False:  # noqa: E712 (np.bool_/0/1)
            animate = bool(animate)
        else:
            raise ValueError(
                f"animate={animate!r} is not a recognized animate value; "
                "use True/False, a style string ('parallel', 'spin', "
                "'serial', 'morph', 'window', or the sugar styles "
                "'chemtrails'/'precog'/'bullettime'), the dict form "
                "(animate={'style': ..., ...}), or a per-dataset list for "
                "animate='morph'. For extra camera rotations, pass "
                "rotations= instead.")

    # fail-fast on hue_mode= for the same reason title= fails fast here: a
    # typo or a contradictory pair should not survive the reduce pipeline.
    _validate_hue_mode(hue_mode, color_reduce, hue)

    # fail-fast on title= BEFORE the analyze/reduce pipeline (the same
    # precedent _validate_extra_kwargs sets) and before resolve_font() and the
    # plot_stream() return both consume it. Cited by SYMBOL, not line number:
    # tasks later in this plan add code above these, and stale line citations
    # have misdirected readers six times in this project. Passes the RAW
    # order= (not yet `_resolve_order`'d): a serial-style `title=` LIST needs
    # only that raw value to pass this early TYPE check (`n_datasets` isn't
    # known yet, so the length check is deferred to the second call, beside
    # `_resolve_animate_mode`, once `len(xform)` exists -- plan 1.1 Task 8).
    _n_forecast_trail = _validate_forecast_trail(forecast_trail, predict)
    # forecast_*= style the forecast overlays, so without predict= there
    # is nothing for them to style. Raising beats ignoring: a silently
    # dropped style kwarg leaves the user staring at an unchanged plot
    # with no clue which of their arguments did nothing.
    _forecast_style_kwargs = {
        'forecast_hue': forecast_hue,
        'forecast_cluster': forecast_cluster,
        'forecast_n_clusters': forecast_n_clusters,
        'forecast_palette': forecast_palette,
        'forecast_fmt': forecast_fmt,
    }
    if predict is None:
        _given = sorted(k for k, v in _forecast_style_kwargs.items()
                        if v is not None)
        if _given:
            raise ValueError(
                f"{', '.join(_given)} style the forecast overlays, but no "
                f"forecast was requested; pass predict= (e.g. "
                f"predict='Kalman') or drop "
                f"{'these' if len(_given) > 1 else 'it'}.")
    _segment_titles = _validate_title(title, style=animate, order=order)

    # fail-fast on order= (same precedent as title= above): it depends only
    # on the raw animate= argument (via `_raw_animate_style`), never on
    # `n_datasets`, so it validates/resolves here. Stored under a PRIVATE
    # name, not `order` itself: the streaming-kwarg diff check further down
    # this function compares each plot() parameter's locals()-visible value
    # against its literal signature default to detect explicit user input
    # (same mechanism the `cluster=False` note below it describes), and
    # `_resolve_order` always turns the default `order=None` into a
    # concrete 'parallel'/'serial' -- reassigning `order` itself here would
    # make every streaming call look like it had explicitly passed order=.
    # `_resolve_animate_mode` folds this resolved value into `order` (the
    # name every OTHER downstream consumer expects) once streaming inputs
    # have already returned early, below.
    _resolved_order = _resolve_order(animate, order)

    # fail-fast on on_frame= (plan 1.1 Task 7, same precedent as title=/
    # order= above): needs only its own callability and the raw animate=
    # truthiness, neither of which depends on the pipeline.
    if on_frame is not None:
        if not callable(on_frame):
            raise TypeError(
                f"on_frame must be callable; got {type(on_frame).__name__}.")
        if not animate:
            raise ValueError(
                "on_frame requires an animated plot; pass animate=True "
                "(or 'spin'/'serial'/'window'/'morph').")

    # fail-fast on simplify= (Contract 8, same precedent as title= above):
    # it needs no data, only its own type, so it must not wait for the
    # animate='morph' tractability guard further down the pipeline.
    if not isinstance(simplify, bool):
        raise TypeError(
            f"simplify must be True or False, not {type(simplify).__name__}. "
            "It controls whether hypertools may downsample to keep an "
            "animate='morph' render tractable; it does not downsample "
            "anything else.")

    # animations need a positive duration and frame rate (QC 2026-07: duration=0
    # or frame_rate=0 raised ZeroDivisionError, and a negative duration a cryptic
    # "zero-size array to reduction" error, from the frame-count math; release-1.0
    # audit F04-007: duration=None slipped through to a bare TypeError in the
    # frame-count multiplication, and F05-009: a negative tail_duration silently
    # suppressed the opaque head for the whole animation).
    if animate:
        if (duration is None or isinstance(duration, bool)
                or not isinstance(duration, (int, float, np.integer,
                                             np.floating))
                or duration <= 0):
            raise ValueError(
                f"duration must be a positive number of seconds for an "
                f"animation; got {duration!r}.")
        if (frame_rate is None or isinstance(frame_rate, bool)
                or not isinstance(frame_rate, (int, float, np.integer,
                                               np.floating))
                or frame_rate <= 0):
            raise ValueError(
                f"frame_rate must be a positive number; got {frame_rate!r}.")
        if not isinstance(tail_duration, (list, tuple)):
            if (tail_duration is None or isinstance(tail_duration, bool)
                    or not isinstance(tail_duration, (int, float, np.integer,
                                                      np.floating))
                    or tail_duration < 0):
                raise ValueError(
                    f"tail_duration must be a non-negative number of "
                    f"seconds (the trail/head-window length); got "
                    f"{tail_duration!r}.")
        # morph_samples=-5 used to leak numpy's internal 'negative
        # dimensions are not allowed' from the downsampling RNG without
        # ever naming the kwarg (release-1.0 audit, D03-gallery-basics-007)
        _ms_ok = (morph_samples is None
                  or (not isinstance(morph_samples, bool)
                      and isinstance(morph_samples, (int, float, np.integer,
                                                     np.floating))
                      and float(morph_samples) >= 1
                      and float(morph_samples).is_integer()))
        if not _ms_ok:
            raise ValueError(
                f"morph_samples must be a positive integer (the "
                f"per-dataset point cap for animate='morph') or None; got "
                f"{morph_samples!r}.")
        # rotations='two' used to be accepted silently and only crash at
        # SAVE time, deep inside matplotlib ("IndexError: list index out
        # of range") with no mention of the kwarg; zoom=-1 was silently
        # accepted despite the documented positive-zooms-in contract
        # (release-1.0 audit, X2-error-quality-014). Validate both eagerly,
        # next to the duration/frame_rate checks above.
        if isinstance(rotations, (list, tuple)):
            # per-segment morph pacing weights: each entry must be a
            # non-negative number (the list-only-with-morph check below
            # handles WHICH modes allow a list)
            if not all(isinstance(r, (int, float, np.integer, np.floating))
                       and not isinstance(r, bool) and r >= 0
                       for r in rotations):
                raise ValueError(
                    f"rotations, when given as a per-segment list (for "
                    f"animate='morph'), must contain only non-negative "
                    f"numbers; got {rotations!r}.")
        elif (rotations is None or isinstance(rotations, bool)
              or not isinstance(rotations, (int, float, np.integer,
                                            np.floating))):
            raise ValueError(
                f"rotations must be a number (of full camera rotations "
                f"over the animation; rotations=0 fixes the camera), or a "
                f"per-segment list with animate='morph'; got "
                f"{rotations!r}.")
        if (zoom is None or isinstance(zoom, bool)
                or not isinstance(zoom, (int, float, np.integer,
                                         np.floating))
                or zoom <= 0):
            raise ValueError(
                f"zoom must be a positive number (the camera zoom factor; "
                f"larger values zoom in, default 1); got {zoom!r}.")

    # save_path misuse fail-fast (F09-004/F09-007): normalize path-likes to
    # str (animated matplotlib and plotly writers do string operations on
    # it), expand ~, and reject non-paths/empty strings/missing directories
    # BEFORE the expensive pipeline runs or any figure exists.
    if save_path is not None:
        save_path = _normalize_save_path(save_path)

    # focused= resolution (GH #275 round17 #8): the length, in SECONDS (the
    # same unit as `tail_duration`), of the opaque "in-focus" window for
    # `animate='window'` and for any dataset with a `chemtrails`/`precog`/
    # `bullettime` trail. `None` (default) resolves to `tail_duration`'s own
    # value -- today's hardcoded/tail_duration-derived focus length -- so
    # omitting `focused=` never changes existing behavior. Silently ignored
    # (no error) for `animate='spin'`/`'parallel'` (with no trail flags set)/
    # `'morph'`, where the concept of a separate "in-focus" window distinct
    # from `tail_duration` doesn't apply -- see `matplotlib_backend
    # .animate_plot3D`/`plotly_backend._add_animation` for exactly where
    # `focused` vs. `tail_duration` is used.
    if focused is not None:
        if (isinstance(focused, bool)
                or not isinstance(focused, (int, float))
                or focused < 0):
            raise ValueError(
                f"focused= must be a non-negative number, or None (default: "
                f"tail_duration's value, {tail_duration!r}); got {focused!r}."
            )
        resolved_focused = focused
    else:
        resolved_focused = tail_duration

    # predict= + animate: a forecast over a STATIC scene is a fixed overlay,
    # which is why animate='spin' (camera-only) draws it once and rotates it.
    # Time-progressing modes now precompute a forecast per frame from the
    # history revealed so far (see hypertools/plot/forecast.py). 'morph' is
    # the one mode still refused: it interpolates between point clouds rather
    # than progressing along a time axis, so there is no history to forecast
    # from.
    #
    # BOTH morph spellings must be caught HERE. `_resolve_animate_mode` (which
    # maps a per-dataset list onto 'morph') is not called until much further
    # down this function, so at this point `animate` is still the raw list and
    # `animate == "morph"` is False for the list form.
    _is_morph_request = (animate == "morph"
                         or isinstance(animate, (list, tuple)))
    if predict is not None and _is_morph_request:
        raise NotImplementedError(
            "predict= is not supported with animate='morph' (including the "
            "per-dataset morph list form): a morph interpolates between "
            "point clouds rather than progressing along a time axis, so "
            "there is no history to forecast from. Use animate=True/"
            "'parallel'/'serial'/'window'/'spin', or omit predict=."
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
    # Font applied DIRECTLY to individual text artists (title/labels/legend/
    # colorbar). Only an EXPLICIT font= is applied that way -- it is the
    # caller's stated choice for every surface. An AUTO-detected font
    # (`resolved_font` set while `font is None`) merely fills a coverage GAP
    # and is added to the fallback STACK instead (see the rc_context below),
    # so the primary face stays the bundled Noto Sans and per-glyph fallback
    # supplies only the characters the stack lacks (maintainer font review).
    _artist_font = resolved_font if font is not None else None
    # The plotly backend has no rcParams stack, so an AUTO-detected gap family
    # must be handed to it EXPLICITLY (as a family name appended near the end
    # of its CSS stack) -- otherwise a character matplotlib renders via the
    # discovered font would silently show as tofu on plotly (maintainer font
    # review). `None` for the explicit-font and no-gap cases.
    _plotly_font_extra = (resolved_font.get_name()
                          if (resolved_font is not None and font is None)
                          else None)

    # label_alpha= resolution (GH #103): resolved ONCE, here, exactly like
    # font= above -- `None` (default) keeps the historical hardcoded 0.5
    # opacity on both backends; any other value must be a real alpha
    # (a number in [0, 1]), validated fail-fast before the expensive
    # analyze/reduce/align pipeline runs.
    if label_alpha is None:
        resolved_label_alpha = 0.5
    elif (isinstance(label_alpha, bool)
            or not isinstance(label_alpha, (int, float))
            or not (0 <= label_alpha <= 1)):
        raise ValueError(
            f"label_alpha= must be a number in [0, 1], or None (default: "
            f"0.5); got {label_alpha!r}."
        )
    else:
        resolved_label_alpha = label_alpha

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

    # A whole already-fitted Pipeline belongs in pipeline=, not a single stage
    # kwarg: passing one as reduce=/manip=/etc would apply it as that ONE stage
    # and then plot re-applies the reduce stage to enforce ndims, double-applying
    # it (QC 2026-07: this produced a cryptic "X has N features, but PCA is
    # expecting M features" error). Point the user at the dedicated reuse path.
    from ..core.pipeline import Pipeline as _HypPipeline
    for _stage_name, _stage_value in (('manip', manip), ('normalize', normalize),
                                      ('reduce', reduce), ('align', align),
                                      ('cluster', cluster)):
        if isinstance(_stage_value, _HypPipeline) and _stage_value.is_fitted:
            raise ValueError(
                f"{_stage_name}= received an already-fitted hypertools Pipeline. "
                "A whole fitted Pipeline encodes all of its own stages -- reuse "
                "it via hyp.plot(x, pipeline=<that Pipeline>), not as a single "
                "stage kwarg.")

    # streaming inputs (issue #101): iterators/generators and Hugging Face
    # IterableDatasets are detected from the structure of the input -- no
    # flag needed. Models are fitted on the first `stream_init` samples and
    # every subsequent sample is projected through the fitted models and
    # added to the plot dynamically (fetched in chunks of `stream_chunk`),
    # continuing until the stream ends, `stream_max` samples have been
    # consumed, or the user interrupts.
    from ..io.streaming import is_stream, plot_stream
    if is_stream(x):
        # only the parameters forwarded below have a streaming
        # implementation. Any OTHER parameter the caller explicitly set is
        # named in a UserWarning instead of being silently dropped
        # (F22-004) -- detected by comparing each plot() parameter's
        # current value against its signature default (`cluster=False`
        # etc. were already normalized to their None defaults above, so
        # documented no-op spellings do not warn).
        _stream_forwarded = {
            'x', 'fmt', 'stream_init', 'stream_chunk', 'stream_max',
            'stream_window', 'ndims', 'reduce', 'normalize', 'align',
            'cluster', 'n_clusters', 'save_path', 'show', 'frame_rate',
            'markersize', 'linewidth', 'color', 'palette', 'title',
            'size', 'elev', 'azim', 'ax'}
        _local_vals = locals()
        _stream_dropped = []
        for _pname, _param in inspect.signature(plot).parameters.items():
            if (_pname in _stream_forwarded
                    or _param.kind is inspect.Parameter.VAR_KEYWORD):
                continue
            _val = _local_vals.get(_pname, _param.default)
            try:
                _diff = not (_val is _param.default
                             or _val == _param.default)
            except Exception:
                _diff = True
            if _diff:
                _stream_dropped.append(_pname)
        _stream_dropped.extend(kwargs)
        if _stream_dropped:
            warnings.warn(
                "streaming input: the following plot() parameter(s) have "
                "no streaming implementation and will be ignored: "
                f"{', '.join(sorted(_stream_dropped))}. Parameters "
                "honored for streams: fmt, stream_init, stream_chunk, "
                "stream_max, stream_window, ndims, reduce, normalize, "
                "save_path, show, frame_rate, markersize, linewidth, "
                "color, palette, title, size, elev, azim, ax (see the "
                "stream_init docstring).", UserWarning, stacklevel=external_stacklevel())
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
        # An animated plot BUILDS ITS OWN FIGURE. Measured across every mode
        # -- parallel, serial, spin, chemtrails, precog, bullettime, window,
        # morph -- a passed `ax` is ignored: the animation is drawn on a new
        # figure, that figure is returned, and the caller's axes is left
        # empty. Silently drawing the right data in the wrong place is worse
        # than refusing, and refusing an unsupportable `ax=` is what the 3-D
        # check immediately below already does.
        #
        # Composing several animated plots into one figure is therefore not
        # possible today. An example that needs panels can lay them out in
        # the DATA instead -- translate each group into its own region of one
        # shared frame -- which keeps it to a single call and a single
        # animation (see `examples/`).
        if _raw_animate_style(animate):
            raise ValueError(
                "ax= cannot be combined with animate=: an animated plot owns "
                "its own figure, so the axes you passed would be left empty "
                "and the animation drawn somewhere else. Drop ax= and use the "
                "returned animation's .figure (or .figure.axes[0]) to style "
                "what was drawn; for several panels in one animation, lay the "
                "panels out in the data and make a single plot call."
            )
        if ndims > 2:
            if ax.name != "3d":
                raise ValueError(
                    "If passing ax and the plot is 3D, ax must " "also be 3d"
                )

    text_args = {"vectorizer": vectorizer, "semantic": semantic, "corpus": corpus}

    # a plain python "matrix" -- a list of equal-length rows of numbers,
    # e.g. [[1., 2.], [3., 4.]] -- is ONE dataset (exactly like the
    # equivalent np.array), NOT a nested list of scalar "datasets"
    # (F01-004/F08-001: the flattening below used to treat every NUMBER as
    # a leaf, then crash with a nonsensical error about a color= kwarg the
    # caller never passed). Ragged all-numeric rows are left as-is:
    # format_data treats each numeric list as its own (column-vector)
    # dataset.
    if _is_numeric_matrix(x):
        x = np.asarray(x, dtype=float)

    # nested lists (e.g. [[a, b], [c]]) are flattened into a flat list of
    # datasets while recording each leaf's outermost-group index and nesting
    # depth; these drive multilevel styling below (color by outer group,
    # thinner/fainter lines per deeper level)
    nested_groups = nested_depths = None
    if isinstance(x, list) and any(isinstance(el, list) for el in x) \
            and not all(isinstance(el, str) for el in x) \
            and not all(isinstance(el, (list, tuple)) and len(el) > 0
                        and all(isinstance(v, (int, float, np.number))
                                and not isinstance(v, bool) for v in el)
                        for el in x):
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
                , stacklevel=external_stacklevel())

    # A hierarchy in a LIST is rejected on the COLUMN axis only: before 1.1
    # it flattened to a single line with no warning at all (nothing pinned
    # it, so rejecting is purely additive), whereas a ROW hierarchy in a
    # list keeps today's documented warn-and-flatten path just above.
    # Dual-axis frames are rejected outright -- 1.1 declines to guess which
    # hierarchy wins, where before the row path silently won.
    reject_hierarchical_in_list(x, caller='hyp.plot', axes='columns')
    reject_dual_axis(x)

    _multiindex_meta = None
    # per-leaf continuous hue values for a COLUMN hierarchy (Task 6); None
    # for every other input, including a categorical hue that was warned
    # about and dropped.
    _mi_hue_per_leaf = None
    # innermost-level (feature) labels of a COLUMN hierarchy's leaves, kept
    # for the return_model bundle's pipeline; None on every other input.
    _mi_feature_labels = None
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
        # predict= used to be refused here because forecasts were computed
        # one-per-leaf BEFORE the per-level means were appended. 1.1 computes
        # them over the FINAL trace list instead (see the `_multiindex_meta
        # is not None` branch below), so the counts cannot disagree. What
        # replaces the blanket refusal is the narrower >= 2-rows-per-trace
        # precondition raised there.
        if hue is not None:
            warnings.warn(
                "x has a row MultiIndex (GH #95): MultiIndex grouping "
                "(leaf traces + per-level averages) takes precedence over "
                "hue=; ignoring hue."
            , stacklevel=external_stacklevel())
            hue = None
        x, _multiindex_meta = expand_multiindex(x)
    elif isinstance(x, pd.DataFrame) and x.columns.nlevels >= 2:
        # COLUMN hierarchy (1.1): the innermost column level is the FEATURE
        # axis and every level above it groups, so (Market, Sector, Ticker)
        # becomes one leaf per sector plus a market mean. Unlike the row
        # rule, every group keeps all len(df) rows.
        if cluster is not None or n_clusters is not None:
            raise ValueError(
                "cluster=/n_clusters= is not compatible with a column-"
                "MultiIndex DataFrame: MultiIndex grouping already assigns "
                "colors by the top-level column index and would conflict "
                "with cluster-based grouping. Flatten the columns "
                "(df.columns = df.columns.map('_'.join)) before clustering, "
                "or drop cluster=/n_clusters= to use the MultiIndex "
                "grouping."
            )
        # predict= is supported on this axis too (1.1); see the note on the
        # deleted row-axis refusal above.
        # NOMINAL correspondence: group_columns has already required that
        # every group carry the same feature labels and permuted the later
        # groups into the first group's order. Position therefore MEANS name
        # by the time the arrays are handed on, and a within-group column
        # permutation cannot move a trajectory.
        x, _multiindex_meta = group_columns(x)
        if hue is not None:
            # Classify hue HERE, before the MultiIndex branch below wins the
            # cluster/hue/nested_groups chain outright. A CONTINUOUS hue is
            # normalised to one value sequence per leaf and carried through
            # `FinalTraces.aux`, so a mean trace's colour is derived from
            # its members' hue by the same operation that derives its data.
            # A categorical hue still defers: it REGROUPS traces, so the
            # leaves the hierarchy names would stop existing.
            _mi_hue_per_leaf, _mi_hue_reason = _hierarchy_hue_per_leaf(
                hue, len(x[0]), len(_multiindex_meta['leaf_keys']))
            if _mi_hue_per_leaf is None:
                warnings.warn(
                    "x has a column MultiIndex: MultiIndex grouping (group "
                    "traces + per-level averages) takes precedence over "
                    f"{_mi_hue_reason}, which would regroup the traces; "
                    "ignoring hue. A CONTINUOUS hue is supported -- one "
                    "value per row, or one sequence per leaf."
                , stacklevel=external_stacklevel())
            # Either way `hue` itself is consumed here: what survives is
            # `_mi_hue_per_leaf`, which the branch below hands to the trace
            # builder as auxiliary values.
            hue = None
        # Hand the pipeline plain arrays rather than the labelled leaves.
        # format_data matches DataFrame features BY COLUMN NAME across
        # datasets (GH #132), which would duplicate the matching just done
        # -- and would reject the legitimate duplicate-label case (two share
        # classes of one issuer, a repeated sensor), which is matched here
        # by (label, occurrence) instead. Ragged groups never reach the
        # pipeline's equal-width check: unequal widths are already a
        # feature-label mismatch, reported by name.
        #
        # Keep the labels first, though: they are the ONLY record of what
        # each analysed column means once the leaves become bare arrays,
        # and the `return_model=True` bundle's pipeline needs them to match
        # features by name on re-application too (see
        # `Pipeline._fit_feature_order`). Every leaf shares this order --
        # `group_columns` permuted them into the first leaf's.
        if pipeline is not None and getattr(
                pipeline, 'input_hierarchy', None) is not None:
            # A caller-supplied pipeline that ALREADY carries a hierarchy
            # record must be checked against THIS frame while the labels
            # still exist. One line below, the leaves become bare arrays,
            # and a list is positional by contract -- so without this the
            # fit-time feature names are never consulted during plotting,
            # and a frame of the same WIDTH but different measurements
            # plotted happily against a pipeline fit on something else.
            # `bundle['pipeline'].transform(that_same_frame)` then raised
            # the nominal mismatch, contradicting the round-trip the
            # `return_model` docstring promises.
            #
            # `_fit_feature_order` is the one implementation of that rule:
            # it raises the missing/unexpected-feature error under 'name'
            # correspondence, and returns None under 'position' (where a
            # same-width frame IS allowed to mean something else -- that is
            # what opting out of nominal matching buys).
            _fit_order = pipeline._fit_feature_order(x)
            if _fit_order is not None:
                # Reordering matters as much as rejecting. `group_columns`
                # permutes into THIS frame's first group's order, which is a
                # property of the frame; the fitted steps are positional and
                # expect the FIT-time order. Feeding them the frame's order
                # transformed correct-looking coordinates that were silently
                # wrong -- the same defect nominal correspondence exists to
                # remove, one fit/transform pair over.
                x = [leaf.iloc[:, _fit_order] for leaf in x]
        _mi_feature_labels = list(x[0].columns)
        x = [leaf.to_numpy() for leaf in x]

    # Each leaf's row count BEFORE manip/normalize/reduce/align run. Read
    # only by the >= 2-rows-per-trace precondition far below, which
    # necessarily runs on the POST-pipeline arrays and so cannot otherwise
    # tell "the grouping/input gave this trace one row" from "a row-count-
    # changing stage (manip='Resample', a smoother that trims edges)
    # shortened it". Without it both axis messages stated something false
    # about the user's frame -- "the input itself has only one observation"
    # of a 30-row frame -- and offered a remedy that could not work. Same
    # quantity the sibling hue-length check above compares against.
    _mi_input_rows = (None if _multiindex_meta is None
                      else [len(xi) for xi in x])

    # default axis labels from DataFrame column names (release-1.0 audit,
    # F08-plot-inputs-016): when a SINGLE DataFrame with named (non-default,
    # non-duplicate) columns is passed, remember its df2mat-transformed
    # column labels; they become the default xlabel/ylabel(/zlabel) below
    # IF the drawn axes end up corresponding 1:1 to those columns (i.e. the
    # transformed data is 2-D or 3-D, so no real dimensionality reduction
    # mixes the columns). User-passed xlabel=/ylabel=/zlabel= always win,
    # and nothing is inferred when transform=/pipeline= replace the
    # standard analysis pipeline.
    _df_axis_labels = None
    if transform is None and pipeline is None:
        _lbl_df = None
        if isinstance(x, pd.DataFrame):
            _lbl_df = x
        elif (isinstance(x, (list, tuple)) and len(x) == 1
              and isinstance(x[0], pd.DataFrame)):
            _lbl_df = x[0]
        if (_lbl_df is not None and _lbl_df.shape[1] <= 3
                and _lbl_df.index.nlevels == 1
                and not isinstance(_lbl_df.columns,
                                   (pd.RangeIndex, pd.MultiIndex))
                and not _lbl_df.columns.duplicated().any()):
            try:
                from ..tools.df2mat import df2mat as _df2mat
                from ..tools.format_data import _prepare_df
                _df_axis_labels = _df2mat(_prepare_df(_lbl_df, warn=False),
                                          return_labels=True)[1]
            except Exception:
                _df_axis_labels = None  # never let label sugar break a plot

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

        # per-dataset feature counts must agree (F01-011/F03-008/F08-002):
        # the reduce stage stacks every dataset, so mismatched widths used
        # to die deep inside numpy ("all the input array dimensions ...
        # must match exactly") with no dataset info or fix hint. Fail fast
        # with a clear message BEFORE the pipeline runs.
        _widths = [ri.shape[1] for ri in raw]
        if len(set(_widths)) > 1:
            # when the ORIGINAL input mixed text and numeric datasets, the
            # real problem is a text/numeric sample-count mismatch (equal
            # counts would have been auto-hyperaligned by format_data), not
            # the embedded column counts -- say so (release-1.0 audit,
            # D08-tutorials-analysis-012 / D05-gallery-data-text-013).
            def _has_text(v):
                if isinstance(v, str):
                    return True
                if isinstance(v, (list, tuple)):
                    return any(_has_text(vi) for vi in v)
                return False
            _text_hint = (
                " (Note: text datasets are embedded into topic vectors -- "
                "hence the differing column counts -- and can only be "
                "combined with numeric datasets when every dataset has the "
                "SAME number of samples, which lets hypertools align them "
                "to a common space.)") if _has_text(x) else ""
            raise ValueError(
                "all datasets must have the same number of columns "
                "(features) to be analyzed/plotted together, but the "
                f"inputs have per-dataset column counts {_widths}. Either "
                "pass datasets with matching columns, or bring them into "
                "a shared space first and plot the result -- e.g. "
                "hyp.plot(hyp.align(data, align='hyper'), ...)."
                + _text_hint)

        # labels= carries one entry per observation (F01-010/F10-011):
        # validate BEFORE the pipeline runs, mirroring hue='s check.
        if labels is not None:
            _validate_labels_length(labels, [ri.shape[0] for ri in raw])

        # a per-dataset fmt LIST must match the dataset count
        # (F01-006/F10-003): fail fast here when no later regrouping
        # (hue=/cluster=/MultiIndex) can change the drawn-trace count; the
        # regrouped case is re-checked against the FINAL count below.
        if (isinstance(fmt, list) and hue is None and cluster is None
                and n_clusters is None and _multiindex_meta is None
                and len(fmt) != len(raw)):
            raise ValueError(
                f"fmt was given as a list of length {len(fmt)}, but there "
                f"are {len(raw)} dataset(s) to plot; pass one format "
                "string per dataset, or a single fmt string to broadcast "
                "it to every dataset.")

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
                # plot()'s ndims defaults to 3 (unlike analyze's None), so
                # forwarding it alongside reduce=None would trip analyze's
                # "ndims= was passed but reduce= is None" warning on EVERY
                # reduce=None plot -- including the internal streaming
                # redraw -- even though plot enforces its own display
                # dimensionality separately below (release-1.0 audit,
                # D1-code-residue regression).
                ndims=ndims if reduce is not None else None,
                normalize=normalize,
                reduce=reduce,
                align=align,
                manip=manip,
                internal=True,
                impute=impute,
                random_state=random_state,
            )
    else:
        xform = transform

    # Return data that has been normalized and possibly reduced and/or aligned
    xform_data = copy.copy(xform)

    # `trace_data` is whatever the PLOTTED trajectories are at the last point
    # before centering/scaling; `xform_data` is never reassigned after this
    # line. They start as the same object and diverge only where the drawn
    # trajectories stop being the analysed ones: the display-dimensionality
    # projection below (which rebinds `xform`), and the hierarchy branch,
    # which draws per-level means that the analysed leaf list does not
    # contain. `trace_metadata` describes those traces, or is None for
    # non-hierarchical input.
    trace_data = xform_data
    trace_metadata = None

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
                    "Both color and colors defined: color will be "
                    "ignored in favor of colors."
                , stacklevel=external_stacklevel())

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
                    "Both linestyle and linestyles defined: linestyle "
                    "will be ignored in favor of linestyles."
                , stacklevel=external_stacklevel())

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
                    "Both marker and markers defined: marker will be "
                    "ignored in favor of markers."
                , stacklevel=external_stacklevel())

    # handle marker size (to be passed onto matplotlib/plotly)
    if markersize is not None:
        mpl_kwargs["markersize"] = markersize

    # handle line width (to be passed onto matplotlib/plotly)
    if linewidth is not None:
        mpl_kwargs["linewidth"] = linewidth

    # alpha= (1.1): a first-class per-dataset style (GH #206 follow-up).
    # NOT validated/written here (unlike color/linewidth/marker/
    # markersize above), even though it is resolved against this same
    # INPUT dataset count -- validating here would run BEFORE `hue` is
    # finalised (the animate='morph' hue-drop, below, can still null it)
    # and BEFORE the MultiIndex/cluster/hue/nested_groups chain that may
    # override alpha internally has picked its branch, either of which
    # can disagree with a lookahead taken this early (task-6 second
    # review, NEW ISSUE). See the alpha= block right before that chain,
    # after hue is finalised, for the validation, the
    # `_alpha_overridden_internally` lookahead, and the full explanation.

    # reduce data to <=3 dims for DISPLAY. `analyze` above already applied
    # the requested reduce= spec; this pass only enforces the display
    # dimensionality (3, or ndims if lower) and is SKIPPED when the data is
    # already there -- re-applying a fitted/instance reducer here
    # re-transformed its own (already-reduced) output and crashed with
    # "X has 3 features, but ... is expecting N features" (F03-002); it
    # also re-fired one-shot warnings (e.g. the deprecated {'model',
    # 'params'} spec warning) twice per plot() call (F03-009). xform was
    # already formatted by analyze(), so format_data is skipped here.
    _display_ndims = ndims if (ndims and ndims < 3) else 3
    if xform[0].shape[1] > _display_ndims:
        if reduce is None:
            raise ValueError(
                f"the data to plot has {xform[0].shape[1]} dimensions, but "
                f"static plots support at most {_display_ndims}; "
                "reduce=None disables dimensionality reduction, so the "
                "data cannot be drawn. Pass a reduce model (e.g. the "
                "default reduce='IncrementalPCA'), or reduce the data "
                "yourself before plotting.")
        _display_reduce = reduce
        if (not isinstance(reduce, (str, dict, type))
                and hasattr(reduce, "fit_transform")):
            # a model INSTANCE (possibly fitted, possibly configured with
            # >3 components) must not be applied a second time -- project
            # to display space with the default reducer instead.
            _display_reduce = "IncrementalPCA"
        xform = reducer(xform, ndims=_display_ndims, reduce=_display_reduce,
                        internal=True, format_data=False)
        if xform[0].shape[1] > _display_ndims:
            # e.g. a dict spec pinning n_components > 3: fall back to the
            # default display projection rather than crash downstream.
            xform = reducer(xform, ndims=_display_ndims,
                            reduce="IncrementalPCA", internal=True,
                            format_data=False)
        # a display-ONLY projection just ran: the plotted trajectories are no
        # longer the analysed ones, so re-point `trace_data` at the rebound
        # list. `xform_data` deliberately keeps the pre-projection arrays --
        # it was captured before this block and is the analysed pipeline
        # output, which is what `pipeline.transform()` reproduces.
        trace_data = xform

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

    # zlabel= (round17 #7): no z-axis to label on a 2D (or 1D) plot -- fail
    # fast rather than silently ignoring the kwarg (mirrors surface=/
    # density= above), now that xform's FINAL dimensionality is known.
    # apply the DataFrame-column default axis labels (F08-016), now that
    # xform's FINAL dimensionality is known -- only when it matches the
    # DataFrame's transformed column count exactly (2-D or 3-D), so each
    # drawn axis IS one named column; user-passed labels take precedence
    # (pass e.g. xlabel='' to suppress a single inferred label).
    if (_df_axis_labels is not None
            and len(_df_axis_labels) == xform[0].shape[1]
            and len(_df_axis_labels) in (2, 3)):
        if xlabel is None:
            xlabel = str(_df_axis_labels[0])
        if ylabel is None:
            ylabel = str(_df_axis_labels[1])
        if len(_df_axis_labels) == 3 and zlabel is None:
            zlabel = str(_df_axis_labels[2])

    if zlabel is not None and xform[0].shape[1] < 3:
        raise ValueError(
            f"zlabel= is not supported for {xform[0].shape[1]}-D data (no "
            "z-axis in a 2-D or 1-D plot); pass ndims=3 (the default) to "
            "use zlabel=, or use xlabel=/ylabel= for 2-D data."
        )

    # predict=: forecast `t` new rows per input dataset, in the plotted
    # (post normalize->reduce->align) space (GH #169). Computed here -- one
    # forecast per ORIGINAL input dataset, before any cluster/hue reshaping
    # -- so the forecasts correspond 1:1 with the datasets about to be
    # drawn. For the DRAWN forecast trace only, the final observed row of
    # each dataset is prepended so the trace connects to the plotted
    # trajectory (drawn trace length is therefore t + 1).
    # `bundle_forecasts` keeps the UNPREPENDED analyze-space forecasts --
    # exactly `t` rows, matching what `hyp.predict(xform_data, ...)`
    # returns (release-1.0 audit, X1-api-consistency-016: the bundle used
    # to include the seam row, an off-by-one vs. hyp.predict);
    # `raw_forecasts` is the seam-prepended working copy that gets the
    # SAME center/scale transform as `xform` below, so the drawn forecast
    # trace lines up with the drawn (centered/scaled) data.
    #
    # A HIERARCHY is the one exception to "before any reshaping", and it is
    # the reason this is a function rather than straight-line code (1.1
    # Task 8). Its drawn traces are the leaves PLUS the derived per-level
    # means, so "one forecast per input dataset" would be one short per
    # mean -- which is exactly what the pre-1.1 blanket refusal said. The
    # MultiIndex branch below therefore calls this on `FinalTraces.arrays`
    # instead, once the final trace list exists; a mean is forecast from its
    # OWN averaged trajectory. Every other input keeps the historical
    # placement, because `_forecast_owner`/`DatasetRevealSchedule` below are
    # defined in terms of INPUT DATASETS and a blanket move past the
    # cluster/hue chain would silently redefine them.
    def _compute_forecasts(datasets):
        from ..predict.predict import predict as _predictor
        _fc = _predictor(datasets, model=predict, t=t)
        if not isinstance(_fc, list):
            _fc = [_fc]
        return (
            [np.asarray(fc, dtype=float) for fc in _fc],
            [np.vstack([np.asarray(xi[-1:]), np.asarray(fc)])
             for xi, fc in zip(datasets, _fc)],
            # ANALYZE-space copies for the animated per-frame schedule (see
            # hypertools/plot/forecast.py). Taken HERE, beside raw_forecasts,
            # so they keep the same 1:1 correspondence the regrouping guard
            # below checks -- and BEFORE `_interp_anim_line` resamples
            # `xform` onto the frame grid, because `t` is measured in RAW
            # analyze-space samples, not frame-grid rows.
            [np.array(xi, dtype=float, copy=True) for xi in datasets],
        )

    raw_forecasts = None
    bundle_forecasts = None
    analyze_histories = None
    if predict is not None and _multiindex_meta is None:
        bundle_forecasts, raw_forecasts, analyze_histories = \
            _compute_forecasts(xform)

    # per-point colors for multicolored lines (set by the hue branch below;
    # computed after interpolation). Dataset lengths are captured now so hue
    # values can be re-interpolated to match the interpolated trajectories.
    multicolor_hue = None
    # when a high-dim matrix hue is reduced to a 3-column RGB matrix (see the
    # color_reduce= branch below), multicolor_hue holds literal per-point RGB
    # values that must be used AS colors rather than blended over a palette.
    multicolor_hue_is_rgb = False
    pre_interp_lengths = [len(xi) for xi in xform]

    # morph interpolates between point CLOUDS, so hue (which colors fixed
    # observations) has no stable point to attach to across the morph -- every
    # hue form crashed here (IndexError from the data/label reshape, QC 2026-07,
    # pre-existing). Drop hue (with a warning) before the cluster/hue grouping
    # chain below rather than crash.
    if (((animate == 'morph') or isinstance(animate, list))
            and hue is not None):
        warnings.warn("hue is not supported with animate='morph'; "
                      "ignoring hue.", stacklevel=external_stacklevel())
        hue = None

    # original category NAMES for a categorical hue (set below, if
    # applicable), used by `legend=True` so the legend/colorbar show the
    # actual category strings rather than the integer group ids `hue` gets
    # reassigned to just below (group_by_category returns ints).
    hue_category_names = None
    # one label per drawn GROUP, in group order -- like hue_category_names
    # but with '_nolegend_' placeholders for unnamed groups (the None
    # entries of a partially-labeled hue; F02-013), so legend=True and the
    # discrete colorbar can label every trace without a length mismatch.
    hue_group_labels = None
    # A MultiIndex hierarchy's per-trace labels, kept for the colorbar
    # alone so that `legend=False` cannot un-name it (set in the hierarchy
    # branch below; stays None on every other input).
    _mi_colorbar_labels = None
    # HUMAN-readable name per drawn group, parallel to
    # `hue_group_labels` but never carrying matplotlib's
    # `'_nolegend_'` sentinel -- see `_regroup_categorical_lines`.
    _run_cat_names = None
    #: Each drawn run's SOURCE dataset index, when `hue=`/`cluster=`
    #: regrouped the traces. `None` when no regrouping happened.
    _seg_ds = None
    # each run's PRE-bridge row count and whether `patch_lines` bridged it;
    # set together with `_seg_ds` by `_regroup_categorical_lines`
    _seg_lengths = None
    _seg_bridged = None
    # (n_input_datasets, n_hue_groups) when a categorical hue regrouped the
    # data by category -- names= (one name per INPUT dataset) cannot apply
    # after that regrouping (F02-009).
    _hue_regrouped_counts = None
    # unfitted Clusterer built from the SAME resolved spec the figure's
    # cluster stage used (set in the cluster branch below), so the
    # return_model bundle's pipeline encodes the parameters the figure
    # was actually drawn with (F13-004)
    _bundle_cluster_stage = None

    # alpha= (1.1): a first-class per-dataset style, promoted out of the
    # GH #206 `**kwargs` passthrough (where a list raised matplotlib's bare
    # "alpha must be numeric or None"). Resolved against the INPUT dataset
    # count and validated/written HERE -- after `hue` has been finalised
    # (the animate='morph'/list-animate hue-drop just above already ran)
    # but still before the MultiIndex/cluster/hue/nested_groups chain
    # below, and still before `_expand_styles_to_runs` (see its
    # docstring), which only ever runs INSIDE that chain. Writing it here
    # means a per-input-dataset list is still at INPUT-dataset length when
    # `_expand_styles_to_runs` runs during hue/cluster contiguous-run
    # segmentation, so it gets widened to run length exactly like
    # color/linewidth already are, instead of being length-checked against
    # a run count it was never sized for.
    #
    # Internal per-trace alpha (row-MultiIndex level fading, nested-list
    # depth fading, further below) still wins over this -- the documented
    # rule at `_apply_extra_kwargs`'s docstring -- and each of those
    # branches warns before overwriting it, mirroring the MultiIndex
    # branch's existing linewidth= precedent (the `_multiindex_meta`
    # branch).
    #
    # Collision check BEFORE validation (task-6 review, Important finding):
    # this write runs before either overriding branch, so "alpha" is never
    # yet a key in mpl_kwargs here -- the brief's literal `"alpha" in
    # mpl_kwargs` check would always be False at this point and can't be
    # used. Instead, look ahead using the exact conditions those two
    # branches themselves gate on (mirrored, not called -- calling them
    # here would be premature) to decide whether one of them WILL fire and
    # overwrite alpha further down. When one will, a bad user alpha=
    # (wrong length, non-numeric, out of range) must not raise here: the
    # value is about to be discarded (with a warning, from the branch that
    # wins) regardless of whether it was valid, exactly as it was silently
    # discarded pre-1.1 -- raising here instead would change the meaning of
    # an existing call (a regression caught in review). The nested-list arm
    # mirrors `elif nested_groups is not None and color is None and colors
    # is None:` (plot.py, below) plus its own `if any(d != min_depth ...)`
    # guard, AND the fact that it only runs when neither of the two earlier
    # `elif` arms in that chain (cluster/n_clusters, hue) claims the input
    # first.
    #
    # Evaluated HERE, not beside color/linewidth above (task-6 SECOND
    # review, NEW ISSUE): the lookahead below reads `hue`, and `hue` is
    # NOT stable between that earlier site and the chain -- animate=
    # 'morph' (or a list animate=) nulls it (just above) AFTER that
    # earlier site but BEFORE the chain picks its branch, so a lookahead
    # computed that early judges the nested-list arm against the PRE-null
    # `hue`, disagreeing with the chain's actual POST-null choice:
    # nested_groups + hue=<array> + animate='morph' + a bad-length/
    # non-numeric alpha raised at the early site instead of losing (with a
    # warning) to depth fading, exactly like it did pre-1.1. Confirmed via
    # a scratch worktree at db02c64e (pre-task-6): that exact combination
    # succeeds silently there (no exception, no warning -- alpha was a
    # bare, unvalidated, unconditionally-overwritten **kwargs entry).
    #
    # The fix is timing, not a morph-specific special case: evaluate the
    # lookahead from the LAST point every value it reads is guaranteed
    # final, i.e. immediately before the chain whose choice it predicts.
    # Every other value the predicate reads was checked for the same class
    # of staleness: `_multiindex_meta`, `nested_groups` and
    # `nested_depths` are each assigned exactly once, during input
    # parsing, long before either alpha site; `color`/`colors` are plain
    # passed-through parameters, never reassigned anywhere in `plot()`.
    # `cluster`/`n_clusters` ARE reassigned (e.g. `cluster = "KMeans"`),
    # but only INSIDE the `elif cluster is not None or n_clusters is not
    # None:` arm itself, i.e. after that arm has already won the chain --
    # too late to change which arm wins, so it cannot desync the
    # lookahead. `hue` -- mutated by the morph-drop directly above, which
    # now runs BEFORE this point -- was the only one still live.
    _alpha_overridden_internally = (
        _multiindex_meta is not None
        or (nested_groups is not None and color is None and colors is None
            and cluster is None and n_clusters is None and hue is None
            and any(d != min(nested_depths) for d in nested_depths))
    )
    if alpha is not None and not _alpha_overridden_internally:
        mpl_kwargs["alpha"] = _validate_alpha(alpha, len(xform))

    # MultiIndex DataFrames (GH #95): xform currently holds the TRANSFORMED
    # leaf trajectories (post normalize/reduce/align), in the same order as
    # `_multiindex_meta['leaf_keys']` -- exactly what `build_multiindex_styles`
    # needs to compute per-level mean trajectories IN THE REDUCED SPACE and
    # append them. cluster=/n_clusters= were already rejected and hue=
    # already squelched (with a warning) above, so this always wins the
    # cluster/hue/nested_groups chain below.
    #
    # The hierarchy's final trace list, or None for every other input. Read
    # again far below, where it turns the forecast/trace count guard into an
    # assertion, so it must exist on every path.
    _ft = None
    if _multiindex_meta is not None:
        _mi_axis = _multiindex_meta.get('axis', 'rows')
        _mi_which = ("a row MultiIndex (GH #95)" if _mi_axis == 'rows'
                     else "a column MultiIndex")
        if color is not None or colors is not None:
            warnings.warn(
                f"x has {_mi_which}: MultiIndex grouping "
                "assigns color by the top-level index; ignoring "
                "color/colors."
            , stacklevel=external_stacklevel())
        if linewidth is not None:
            warnings.warn(
                f"x has {_mi_which}: MultiIndex grouping "
                "assigns linewidth by level (leaves=1, thicker per level "
                "averaged over); ignoring linewidth."
            , stacklevel=external_stacklevel())
        if alpha is not None:
            warnings.warn(
                f"x has {_mi_which}: MultiIndex grouping "
                "assigns alpha by level (leaves most transparent, "
                "top-level means fully opaque); ignoring alpha."
            , stacklevel=external_stacklevel())
        # ONE owner builds the final trace list (means, truncation, the
        # truncation warning); a SEPARATE, array-blind function styles it.
        # Before 1.1 a single function did both, so any second caller
        # appended every mean twice.
        if _mi_hue_per_leaf is not None:
            # The hue was validated against the INPUT frame's rows; the
            # analysis pipeline is what could have changed that count
            # (manip='Resample', a smoother that trims edges). Say which
            # stage broke the correspondence rather than letting the aux
            # arrays silently describe the wrong observations.
            _bad = [(i, len(a), len(xi))
                    for i, (a, xi) in enumerate(zip(_mi_hue_per_leaf, xform))
                    if len(a) != len(xi)]
            if _bad:
                raise ValueError(
                    "hue= has one value per row of the input frame, but the "
                    "analysis pipeline changed the row count before "
                    f"plotting: leaf {_bad[0][0]} has {_bad[0][1]} hue "
                    f"value(s) for {_bad[0][2]} plotted observation(s). "
                    "Resample or smooth the hue values the same way, or "
                    "drop the row-count-changing stage.")
        _ft = build_hierarchy_traces(xform, _multiindex_meta,
                                     aux=_mi_hue_per_leaf)
        if predict is not None:
            # Contract 10, checked over EVERY final trace -- leaves AND
            # derived means, on BOTH axes -- immediately after the trace
            # list exists and before anything calls into `hyp.predict` or
            # (further below) builds a `ForecastSchedule`. Raising here is
            # what makes the message about the user's DATA; letting it fall
            # through would surface `predict/common.py`'s internal
            # "cannot forecast from a single observation" shape error,
            # which says nothing about the hierarchy that produced the
            # one-row trace.
            #
            # NOT gated on the axis. A column hierarchy cannot SHORTEN a
            # trace, but it cannot lengthen one either: measured, a T=1
            # frame gives leaves of (1, 3) and a mean of (1, 3), so gating
            # on 'rows' would let it reach that internal error.
            #
            # WHICH remedy is offered depends on what actually made the
            # trace short, and that is NOT decided by the axis alone: this
            # loop sees post-pipeline arrays, so a row-count-changing stage
            # is a third cause. Compare each trace against its PRE-pipeline
            # length (`_mi_input_rows`) and name the stage when they differ;
            # the axis then selects between the two grouping/input cases --
            # a row hierarchy's short traces come from the expansion rule
            # (so flattening or moving to the column axis fixes it), a
            # column hierarchy's come from the input itself (so flattening
            # cannot help and is deliberately not offered).
            _n_leaves = len(_multiindex_meta['leaf_keys'])
            for _i, _arr in enumerate(_ft.arrays):
                _rows = np.asarray(_arr).shape[0]
                if _rows >= 2:
                    continue
                _plural = "row" if _rows == 1 else "rows"
                # Leaves come first and in leaf order (FinalTraces.arrays),
                # so a leaf maps straight onto its own input length. A mean
                # is the elementwise average over its members' overlapping
                # prefix, i.e. exactly `min` of their post-pipeline lengths
                # -- so once every leaf has cleared >= 2 rows above, no mean
                # can be short and this branch is unreachable. `min` over
                # all leaves is the conservative stand-in: it can only
                # UNDER-state the input length, so it never blames the
                # pipeline wrongly.
                _input_rows = (_mi_input_rows[_i] if _i < _n_leaves
                               else min(_mi_input_rows))
                if _rows != _input_rows:
                    _remedy = (
                        "The analysis pipeline changed the row count before "
                        f"plotting: this trace went from {_input_rows} row(s) "
                        f"in the input frame to {_rows}. Forecasting needs at "
                        "least 2 observations (rows) to estimate how the data "
                        "change over time; resample or smooth to at least 2 "
                        "rows, or drop the row-count-changing stage.")
                elif _mi_axis == 'rows':
                    _remedy = (
                        "Row-MultiIndex expansion draws one trace per unique "
                        "FULL index tuple, so a frame whose innermost index "
                        "level is unique per row yields one-row traces (and "
                        "one-row per-level means). Either drop the hierarchy "
                        "so the frame is one trajectory "
                        "(df.reset_index(drop=True)), or move the grouping "
                        "to the COLUMN axis, where every group keeps all of "
                        "the frame's rows.")
                else:
                    _observations = ("one observation" if _input_rows == 1
                                     else f"{_input_rows} observations")
                    _remedy = (
                        "A column MultiIndex groups FEATURES, so every group "
                        f"keeps all {_input_rows} of the frame's rows -- the "
                        f"input itself has only {_observations}. Forecasting "
                        "needs at least 2 observations (rows) to estimate "
                        "how the data change over time; pass a frame with "
                        "more rows.")
                raise ValueError(
                    "plot(..., predict=...) needs at least 2 rows per trace, "
                    f"but trace {_i} {_ft.keys[_i]} has {_rows} {_plural}. "
                    + _remedy)
            # One forecast per FINAL trace, computed from the same
            # pre-center/pre-scale arrays that become `trace_data` below --
            # so Contract 5's `forecasts[i] == hyp.predict(trace_data[i])`
            # holds by construction rather than by coincidence.
            bundle_forecasts, raw_forecasts, analyze_histories = \
                _compute_forecasts(_ft.arrays)
        # legend=[...] under a hierarchy RENAMES the top-level groups
        # rather than being discarded: the labelled traces are exactly the
        # top-level ones, so the caller's entries land on them one-for-one
        # (in `unique_top` first-appearance order, the same convention
        # `linestyles=` uses) and the leaves/intermediate means stay
        # unlabelled. This is the one hierarchy-overridden kwarg that is
        # HONOURED instead of warned away -- color/linewidth/alpha encode
        # the hierarchy's structure (which group, which level), so a
        # caller's value would contradict the drawing, whereas legend text
        # names groups the hierarchy has no opinion about.
        # ...except on the continuous-hue path just below, which drops the
        # legend entirely (with a warning): renaming groups whose legend is
        # about to be discarded would only leave the caller's entries in
        # `legend` at top-level-group length for the per-trace length check
        # further down to reject.
        _mi_legend_labels = (list(legend) if _legend_user_list
                             and _mi_hue_per_leaf is None else None)
        _mi_style = build_hierarchy_styles(
            _ft, palette=palette, linestyle=linestyle, linestyles=linestyles,
            legend_labels=_mi_legend_labels)
        xform = _ft.arrays
        # Contract 5: `trace_data` is the pre-center/pre-scale plotted
        # trajectory list, which for a hierarchy includes the derived means
        # -- deliberately NOT in `xform_data`, which holds only the analysed
        # leaves the returned pipeline can reproduce.
        trace_data = _ft.arrays
        trace_metadata = {'keys': _ft.keys, 'level_idx': _ft.level_idx,
                          'is_mean': _ft.is_mean, 'axis': _mi_axis,
                          'level_names': _ft.meta['level_names'],
                          'aux': _ft.aux}
        # recompute: the means were appended after the earlier pass, so the
        # cached lengths no longer cover every trace.
        pre_interp_lengths = [len(xi) for xi in xform]
        if _mi_hue_per_leaf is None:
            mpl_kwargs["color"] = _mi_style["colors"]
        else:
            # The hierarchy contributes width/alpha/label only: colour comes
            # from the hue, mapped over the CONCATENATION of every trace's
            # aux (leaves AND means) so one scale spans the whole figure.
            # `_mi_style['colors']` is dropped on this path only -- setting
            # it would give each line artist a flat colour that the
            # per-segment collections then replace, i.e. dead state that a
            # later reader would reasonably take for the real colours.
            # `np.concatenate` on the observation axis: a 1-D aux gives the
            # flat vector this has always produced, and a 2-D (matrix) aux
            # keeps its weight columns, which `_multicolor_line_colors`
            # already blends through `mat2colors`. `ravel()` here would
            # destroy the matrix form -- the same bug the per-leaf
            # normalizer had.
            _aux = [np.asarray(a, dtype=np.float64) for a in _ft.aux]
            multicolor_hue = np.concatenate(_aux, axis=0)
            # Same RGB rule as the flat path (`_matrix_hue_wants_rgb`), so
            # `hue=` and `color_reduce=` do not change meaning just because
            # the frame has a column hierarchy. Measured before this: a
            # 5-column hue became RGB on a flat plot and mixture weights
            # under a hierarchy, and color_reduce= was silently dropped.
            #
            # The reduction runs on the CONCATENATION, which already
            # contains the derived mean rows -- mean-then-reduce, not
            # reduce-then-mean. That is the only order available (the means
            # are what the hierarchy exists to produce) and it is the right
            # one: it keeps every trace on one shared color scale.
            multicolor_hue_is_rgb = False
            # the hierarchy path resolves its hue well before the flat
            # path's ndim check, so `hue_mode=` has to be validated here too
            if hue_mode is not None and multicolor_hue.ndim != 2:
                raise ValueError(
                    f"hue_mode={hue_mode!r} says how to interpret a 2-D hue "
                    f"MATRIX -- as palette mixture weights or as literal RGB "
                    f"-- but the per-leaf hue is "
                    f"{multicolor_hue.ndim}-dimensional. Drop hue_mode=, or "
                    f"pass one weight ROW per row for every leaf.")
            if multicolor_hue.ndim == 2 and _matrix_hue_wants_rgb(
                    multicolor_hue, color_reduce, hue_mode):
                multicolor_hue = _matrix_hue_to_rgb(multicolor_hue,
                                                    color_reduce)
                multicolor_hue_is_rgb = True
            if legend is True or _legend_user_list:
                # a legend LIST is dropped here for the same reason
                # legend=True is -- it used to survive to the per-trace
                # length check below and raise "legend= was given as a list
                # of length 2, but there are 6 dataset(s)/group(s)", which
                # blames the caller's length for a legend that this path
                # cannot draw at any length.
                warnings.warn("legend is not supported for continuous or "
                              "matrix-valued hue; ignoring legend.",
                              stacklevel=external_stacklevel())
                legend = None
        mpl_kwargs["linewidth"] = _mi_style["linewidths"]
        mpl_kwargs["alpha"] = _mi_style["alphas"]
        if _mi_style["linestyles"] is not None:
            mpl_kwargs["linestyle"] = _mi_style["linestyles"]
        mpl_kwargs["label"] = _mi_style["labels"]
        if _mi_hue_per_leaf is None:
            # The COLORBAR's copy of the hierarchy's per-trace labels, kept
            # in its own variable because it must survive `legend=False`:
            # the colorbar is the colour key for the drawn groups, not a
            # legend, so opting out of the legend must not change which
            # groups it names. `_build_colorbar_info` reads BOTH the group
            # names AND the '_nolegend_' entries that collapse leaves and
            # intermediate means down to the top-level groups off this
            # list; when the `legend is not False` guard below was the only
            # thing installing it, `colorbar=True, legend=False` lost both
            # at once and fell through to `labels = [i + 1 ...]`. Measured
            # on a 3-level column frame (US/EU x tech/fin x a,b,c),
            # matplotlib AND plotly: ticks ['US', 'EU'] -> ['1'..'6']; on a
            # ROW hierarchy (40 leaves + 6 means) 2 segments -> 46.
            _mi_colorbar_labels = _mi_style["labels"]
            if legend is not False:
                # `legend is not False` keeps the explicit opt-out alive:
                # this assignment used to run unconditionally, so it handed
                # the legend block below a non-empty label list and
                # `hyp.plot(df, legend=False)` drew a legend anyway. Every
                # OTHER value still resolves to the hierarchy's own
                # per-trace labels -- for a user list those are the renamed
                # ones `build_hierarchy_styles` just produced, so the drawn
                # legend says what the caller asked for.
                legend = _mi_style["labels"]

    # find cluster and reshape if cluster=/n_clusters= was given
    # (n_clusters= alone defaults to KMeans, matching the docstring)
    elif cluster is not None or n_clusters is not None:
        if hue is not None:
            warnings.warn(
                ("cluster" if cluster is not None else "n_clusters")
                + " overrides hue, ignoring hue.", stacklevel=external_stacklevel())
            hue = None
        if cluster is None:
            cluster = "KMeans"
        if isinstance(cluster, bytes):
            cluster = cluster.decode("utf-8")

        from ..cluster.cluster import _resolve_cluster_spec
        _n_clusters_explicit = n_clusters is not None
        _cluster_instance = None
        _spec_kwargs = {}
        _spec_top_n = None
        if isinstance(cluster, str):
            model = cluster
            params = default_params(model) or {}
        elif isinstance(cluster, dict):
            if "model" not in cluster:
                # the same instructive error hyp.cluster raises for a
                # model-less dict spec, instead of a bare KeyError
                # (F13-010)
                raise ValueError(
                    "If passing a dictionary, pass the model as the "
                    "value of the 'model' key and a dictionary of custom "
                    "parameters as the value of the 'kwargs' key (the "
                    "legacy 'params' key is also accepted).")
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
            # before being handed to `_resolve_cluster_spec` further down,
            # so that call never re-triggers this same warning -- do NOT
            # double-warn.
            if "args" in cluster or "kwargs" in cluster:
                _spec_kwargs = dict(cluster.get("kwargs", {}))
            elif "params" in cluster:
                warnings.warn(
                    "{'model': ..., 'params': {...}} is deprecated; use "
                    "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                    DeprecationWarning, stacklevel=external_stacklevel())
                _spec_kwargs = dict(cluster["params"])
            params = default_params(model_key, _spec_kwargs) or {}
            if "n_clusters" in cluster:
                # top-level convenience:
                # cluster={'model': ..., 'n_clusters': k} -- handed to
                # _resolve_cluster_spec below, which applies hyp.cluster's
                # documented precedence (the spec's value wins over
                # n_clusters=, with a UserWarning on explicit conflicts)
                _spec_top_n = cluster["n_clusters"]
        elif isinstance(cluster, type) or hasattr(cluster, "fit_predict") \
                or hasattr(cluster, "fit_transform"):
            # a class or (sklearn-API) model instance: hyp.cluster accepts
            # these directly (F03-014) -- resolve the registry name for
            # the mixture-model check below and pass the object through.
            model = cluster if isinstance(cluster, type) else type(cluster)
            _cluster_instance = None if isinstance(cluster, type) else cluster
            params = {}
            if n_clusters is not None and _cluster_instance is not None:
                warnings.warn(
                    "n_clusters= is ignored when cluster= is a model "
                    "INSTANCE (the instance's own parameters are used); "
                    "configure the instance directly (e.g. "
                    "KMeans(n_clusters=...)) or pass the model by name "
                    "(e.g. cluster='KMeans').", stacklevel=external_stacklevel())
                n_clusters = None
                _n_clusters_explicit = False
        else:
            raise ValueError(
                "invalid cluster model: expected a string, dict spec, "
                "class, or (sklearn-API) model instance; got "
                f"{cluster!r}.")

        # FeatureAgglomeration clusters FEATURES (columns), not
        # observations: it yields one label per COLUMN, so there is no
        # per-observation grouping to color/reshape the plot with --
        # regrouping by its labels silently drew n_features "points"
        # where the data's rows should be, or crashed downstream
        # (F13-001).
        if _mixture_name(model) == "FeatureAgglomeration":
            raise ValueError(
                "cluster='FeatureAgglomeration' is not supported by "
                "hyp.plot: FeatureAgglomeration clusters features "
                "(columns), not observations, so its labels (one per "
                "column) cannot color or group the plotted rows. Use "
                "hyp.cluster(data, cluster='FeatureAgglomeration') to "
                "get per-column labels directly.")

        # n_clusters= exemption for models that discover their own
        # cluster count (HDBSCAN, DBSCAN, MeanShift, OPTICS,
        # AffinityPropagation): warn-and-ignore instead of crashing in
        # the sklearn constructor, using the same signature-based check
        # as hyp.cluster's _resolve_cluster_spec (F13-002; generalizes
        # the old HDBSCAN-only special case).
        _model_cls = None
        if isinstance(model, str):
            _model_cls = (hard_cluster_models.get(model)
                          or mixture_models.get(model))
        elif isinstance(model, type):
            _model_cls = model
        if (_n_clusters_explicit and _model_cls is not None
                and _mixture_name(model) not in mixture_models
                and "n_clusters"
                not in inspect.signature(_model_cls).parameters):
            warnings.warn(
                f"n_clusters is not a valid parameter for "
                f"{_mixture_name(model)} clustering and will be ignored.", stacklevel=external_stacklevel())
            n_clusters = None
            _n_clusters_explicit = False

        # default_params() pre-fills a DEFAULT cluster count (KMeans
        # n_clusters=3, LDA/NMF n_components=20, ...). When the caller
        # supplied a count (n_clusters= or the dict's top-level
        # 'n_clusters'), drop that default -- it is not a user-typed
        # spec kwarg, so it must not win the resolver's spec-kwargs-
        # take-precedence rule (F13-009) or trigger a bogus conflict
        # warning.
        if _n_clusters_explicit or _spec_top_n is not None:
            for _count_key in ("n_clusters", "n_components"):
                if _count_key in params and _count_key not in _spec_kwargs:
                    del params[_count_key]

        # resolve the spec ONCE with hyp.cluster's own resolver -- same
        # grammar, same precedence (spec kwargs beat n_clusters=, with a
        # UserWarning on explicit conflicts), same signature-based
        # n_clusters exemption, and random_state= injection
        # (F13-002/-003/-009/-020) -- and build the return_model
        # bundle's cluster stage from the IDENTICAL resolved spec so the
        # bundled pipeline encodes the same parameters the figure was
        # drawn with (F13-004).
        if _cluster_instance is not None:
            _resolve_spec = _cluster_instance
        else:
            _resolve_spec = {"model": model, "kwargs": params}
            if _spec_top_n is not None:
                _resolve_spec["n_clusters"] = _spec_top_n
        _cluster_stage = _resolve_cluster_spec(
            _resolve_spec, n_clusters if n_clusters is not None else 3,
            random_state=random_state,
            n_clusters_explicit=_n_clusters_explicit)
        # a second, unfitted resolution of the same spec for the bundle
        # (n_clusters_explicit=False: any conflict was already warned
        # about just above -- values resolve identically either way)
        _bundle_cluster_stage = _resolve_cluster_spec(
            _resolve_spec, n_clusters if n_clusters is not None else 3,
            random_state=random_state)

        cluster_labels = clusterer(xform, cluster=_cluster_stage)

        if _mixture_name(model) in mixture_models:
            # soft assignments: color each observation by the proportion-
            # weighted blend of its components' colors
            if legend is True:
                warnings.warn(
                    "legend is not supported for mixture-model clustering "
                    "(observations have blended colors, not discrete "
                    "groups); ignoring legend."
                , stacklevel=external_stacklevel())
                legend = None
            if not animate:
                # exact per-point colors (rendered via collections/scatter)
                multicolor_hue = np.asarray(cluster_labels,
                                            dtype=np.float64)
                hue = None
            elif _fmt_draws_line(fmt):
                # LINE animation: segment each dataset into contiguous
                # same-group runs (never merging a group's non-adjacent
                # points or crossing a dataset boundary; GH #291), each run
                # coloured by its quantized blended group colour.
                blended = mat2colors(cluster_labels, palette=palette)
                group_ids, group_colors = colors2groups(blended)
                _cat_color = {gid: group_colors[gid]
                              for gid in dict.fromkeys(group_ids)}
                _cat_label = {gid: str(gid) for gid in dict.fromkeys(group_ids)}
                _nd = len(xform)
                (xform, labels, _run_colors, hue_group_labels, _seg_ds,
                 _run_cat_names, _seg_lengths,
                 _seg_bridged) = _regroup_categorical_lines(
                     xform, group_ids, labels, _cat_color, _cat_label)
                fmt = _expand_styles_to_runs(fmt, mpl_kwargs, _seg_ds, _nd)
                mpl_kwargs["color"] = _run_colors
                hue = group_ids
            else:
                # marker animations render one trace per group: quantize the
                # blended colors into (near-)identical-color groups
                blended = mat2colors(cluster_labels, palette=palette)
                group_ids, group_colors = colors2groups(blended)
                xform, labels = reshape_data(xform, group_ids, labels)
                mpl_kwargs["color"] = [
                    group_colors[gid]
                    for gid in sorted(set(group_ids), key=group_ids.index)
                ]
                hue = group_ids
        elif _fmt_draws_line(fmt):
            # LINE clustering: contiguous-run segmentation so a cluster's
            # non-adjacent points are NOT joined into one polyline and
            # separate datasets are never bridged (GH #291); each run is
            # coloured + labelled by its cluster id in sorted order, one
            # legend/colorbar entry per cluster.
            _cat_color, _cat_label = _categorical_color_label_maps(
                cluster_labels, palette, None, None, sort_numeric=True)
            _nd = len(xform)
            (xform, labels, _run_colors, hue_group_labels, _seg_ds,
             _run_cat_names, _seg_lengths,
             _seg_bridged) = _regroup_categorical_lines(
                 xform, cluster_labels, labels, _cat_color, _cat_label)
            fmt = _expand_styles_to_runs(fmt, mpl_kwargs, _seg_ds, _nd)
            mpl_kwargs["color"] = _run_colors
            hue = cluster_labels
            try:
                _cats_sorted = sorted(set(cluster_labels))
            except TypeError:
                _cats_sorted = list(dict.fromkeys(cluster_labels))
            hue_category_names = [str(c) for c in _cats_sorted]
        else:
            xform, labels = reshape_data(xform, cluster_labels, labels)
            # reshape_data returns groups in first-appearance order;
            # reorder the drawn groups (and their legend/colorbar
            # labels) into sorted label order so a legend reads
            # '0, 1, 2' rather than e.g. '1, 0, 2' (F13-022)
            _cats = list(sorted(set(cluster_labels),
                                key=list(cluster_labels).index))
            try:
                _order = sorted(range(len(_cats)), key=lambda i: _cats[i])
            except TypeError:
                _order = list(range(len(_cats)))
            xform = [xform[i] for i in _order]
            labels = [labels[i] for i in _order]
            hue = cluster_labels
            hue_group_labels = [str(_cats[i]) for i in _order]
            hue_category_names = list(hue_group_labels)

    # group data if there is a grouping var
    elif hue is not None:
        if color is not None:
            warnings.warn("hue= and color= were both given; color= will "
                          "be ignored in favor of hue=.", stacklevel=external_stacklevel())

        # pandas containers are used POSITIONALLY (their values, in order):
        # a categorical Series whose index does not contain the label 0
        # (e.g. a column sliced from a filtered DataFrame) crashed with a
        # bare `KeyError: 0` at the `hue[0]` tuple check below, because
        # `[]` on a Series is LABEL-based indexing (release-1.0 audit,
        # F02-003). A single-column DataFrame keeps its existing
        # matrix-hue handling via np.asarray below.
        if isinstance(hue, (pd.Series, pd.Index, pd.Categorical)):
            hue = hue.tolist()

        # NESTED per-dataset hue: when the data is a list of datasets, hue may
        # be given with the SAME nesting -- one hue sub-sequence per dataset,
        # each matching that dataset's length (the classic list-of-lists form,
        # e.g. examples/plot_hue.py). Flatten it to one value (or one matrix
        # row) per observation before classifying, so np.asarray doesn't read a
        # (n_datasets, len) block as a (3, ...) matrix hue. A genuinely flat or
        # (n_obs, k) matrix hue has len(hue) != n_datasets (or scalar elements),
        # so it is left untouched.
        if (isinstance(hue, (list, tuple)) and len(xform) > 1
                and len(hue) == len(xform)
                and all(np.ndim(h) >= 1 and len(h) == len(xi)
                        for h, xi in zip(hue, xform))):
            flat_hue = []
            for h in hue:
                flat_hue.extend(list(h))
            hue = flat_hue

        # classify the hue argument: per-observation numeric matrix
        # (mixture proportions, model weights, ...), continuous 1D values,
        # or discrete grouping labels
        n_obs = sum(len(xi) for xi in xform)
        try:
            hue_array = np.asarray(hue)
        except Exception:
            hue_array = None
        # a SCALAR hue -- a single string or number, e.g. hue='red' -- means
        # "put every observation in one group". Broadcast it to one value per
        # observation so it is not mis-measured as len('red') == 3 characters
        # (QC 2026-07 red-team: hue='red' on 20 points raised the nonsensical
        # "hue has 3 entries but the data has 20 observations"). Since a
        # single-group hue colors nothing differently, this is usually a
        # mistake -- e.g. a DataFrame COLUMN NAME passed seaborn-style --
        # so say so rather than silently accepting it (release-1.0 audit,
        # X2-error-quality-016).
        if hue_array is not None and hue_array.ndim == 0:
            warnings.warn(
                f"hue= was given a single scalar value ({hue!r}); all "
                "observations will be placed in ONE group (a single "
                "color). hue= takes the per-observation values themselves "
                "-- a list/array with one entry per observation (e.g. "
                "hue=df['col'] rather than a column name).",
                UserWarning, stacklevel=external_stacklevel())
            hue = [hue_array.item()] * n_obs
            hue_array = np.asarray(hue)
        # validate hue length (QC 2026-07): a hue that was too SHORT silently
        # truncated the plot (rendered only the first len(hue) points, no
        # warning); too LONG raised a cryptic IndexError deep in reshape_data.
        # hue must carry exactly one value/row per observation.
        _hue_len = (hue_array.shape[0]
                    if hue_array is not None and hue_array.ndim >= 1
                    else len(hue))
        if _hue_len != n_obs:
            raise ValueError(
                f"hue has {_hue_len} entr{'y' if _hue_len == 1 else 'ies'} but "
                f"the data has {n_obs} observations; hue must have exactly one "
                "value (or one row, for a matrix hue) per observation.")
        _hue_ndim = None if hue_array is None else hue_array.ndim
        if hue_mode is not None and _hue_ndim != 2:
            raise ValueError(
                f"hue_mode={hue_mode!r} says how to interpret a 2-D hue "
                f"MATRIX -- as palette mixture weights or as literal RGB -- "
                f"but hue= is "
                + ("not set" if hue_array is None
                   else f"{_hue_ndim}-dimensional")
                + ". Drop hue_mode=, or pass a matrix hue.")
        hue_is_matrix = (hue_array is not None and hue_array.ndim == 2
                         and np.issubdtype(hue_array.dtype, np.number)
                         and hue_array.shape[0] == n_obs)
        # small-cardinality integer (or boolean) hue -- e.g. the cluster
        # labels hyp.cluster returns -- is CATEGORICAL, not continuous: on
        # the continuous path, adjacent integer labels (0 and 1) map to
        # visually indistinguishable neighboring palette samples
        # (F13-005). Rule (documented in the hue docstring): integer/bool
        # dtype, at most 12 unique values, and fewer unique values than
        # observations; anything else numeric stays continuous.
        _hue_int_categorical = (
            hue_array is not None and hue_array.ndim == 1
            and (np.issubdtype(hue_array.dtype, np.integer)
                 or np.issubdtype(hue_array.dtype, np.bool_))
            and hue_array.shape[0] == n_obs
            and len(np.unique(hue_array)) <= 12
            and len(np.unique(hue_array)) < n_obs)
        hue_is_continuous = (hue_array is not None and hue_array.ndim == 1
                             and np.issubdtype(hue_array.dtype, np.number)
                             and hue_array.shape[0] == n_obs
                             and not _hue_int_categorical)

        # arbitrary matrix hue -> RGB. Shared with the column-hierarchy
        # path below so that `hue=` and `color_reduce=` mean the SAME thing
        # whether or not the frame has a hierarchy (they did not: measured,
        # color_reduce= changed a flat figure's colours and was silently
        # ignored under a hierarchy).
        if hue_is_matrix and _matrix_hue_wants_rgb(hue_array,
                                                   color_reduce,
                                                   hue_mode):
            hue_array = _matrix_hue_to_rgb(hue_array, color_reduce)
            multicolor_hue_is_rgb = True

        # set when a categorical INTEGER/boolean hue needs its groups
        # reordered from first-appearance to sorted numeric order after
        # reshape_data below (F13-005/F13-022)
        _hue_sort_numeric = False

        # morph animations tag/reshape datasets specially, so continuous/matrix
        # hue there keeps the grouped path below; every other animation (spin,
        # window, parallel, serial, True) uses the SAME exact-per-point-color
        # path as static plots (QC 2026-07). Excluding all animations here sent
        # continuous hue into the categorical regroup below, which split it into
        # single-point "groups" and crashed the frame interpolation
        # (`interp_array`: "x must contain at least 2 elements").
        _animate_is_morph = (animate == 'morph') or isinstance(animate, list)

        if (hue_is_matrix or hue_is_continuous) and not _animate_is_morph:
            # EXACT PER-POINT COLORS: color varies continuously across
            # observations. Datasets stay intact (no group reshape, which
            # would fragment lines and quantize marker colors); per-point
            # colors are computed after interpolation, below, and rendered
            # via collections (lines) or scatter (markers), and -- for
            # animations -- passed through to each frame as point_colors.
            multicolor_hue = np.asarray(hue_array, dtype=np.float64)
            if legend is True:
                warnings.warn("legend is not supported for continuous or "
                              "matrix-valued hue; ignoring legend.", stacklevel=external_stacklevel())
                legend = None
            hue = None

        elif hue_is_matrix:
            # markers (or animated) path: blend colors per observation,
            # then group observations with (near-)identical colors into
            # traces
            blended = (hue_array if multicolor_hue_is_rgb
                       else mat2colors(hue_array, palette=palette))
            group_ids, group_colors = colors2groups(blended)
            mpl_kwargs["color"] = [
                group_colors[gid]
                for gid in sorted(set(group_ids), key=group_ids.index)
            ]
            if legend is True:
                warnings.warn("legend is not supported for matrix-valued "
                              "hue; ignoring legend.", stacklevel=external_stacklevel())
                legend = None
            hue = group_ids

        else:
            # if list of lists, unpack
            if any(isinstance(el, list) for el in hue):
                hue = list(itertools.chain(*hue))

            # A MISSING label in a categorical hue means the point is
            # unlabeled, which this pipeline already spells `None` (the
            # partially-labeled branch below). Say it once, here: `nan !=
            # nan`, so two missing labels are not equal to each other and
            # became two SEPARATE saturated categories -- and, since
            # `np.nan` is a singleton while `float('nan')` is a fresh object
            # each time, WHICH of those happened depended on how the caller
            # spelled it. Guarded on "some entry is a string" so a purely
            # numeric hue (binned as continuous values, where non-finite
            # entries are already handled by `mat2colors`) is untouched.
            if any(isinstance(el, str) for el in hue):
                hue = [None if is_missing_label(el) else el for el in hue]

            # if all of the elements are numbers, map them to colors
            if not isinstance(hue[0], tuple):
                if _hue_int_categorical:
                    # categorical integer/boolean group ids (F13-005):
                    # grouped and palette-colored like string labels, with
                    # groups (and legend/colorbar labels) in sorted
                    # numeric order -- see the hue docstring's
                    # categorical-vs-continuous rule
                    _int_cats = sorted(set(hue_array.tolist()))
                    hue_category_names = [str(c) for c in _int_cats]
                    hue_group_labels = list(hue_category_names)
                    hue = hue_array.tolist()
                    _hue_sort_numeric = True
                elif all(isinstance(el, (int, float, np.integer,
                                         np.floating))
                         and not isinstance(el, bool) for el in hue):
                    hue = vals2bins(hue)
                elif all(isinstance(el, str) for el in hue):
                    hue_category_names = list(
                        sorted(set(hue), key=list(hue).index))
                    hue_group_labels = list(hue_category_names)
                    hue = group_by_category(hue)
                elif (any(el is None for el in hue)
                      and any(isinstance(el, str) for el in hue)
                      and all(el is None or isinstance(el, str)
                              for el in hue)):
                    # partially-labeled hue (the docstring's "label a subset
                    # of points" form, e.g. ['a', None, 'b', 'a']): the None
                    # entries mark UNLABELED points. They form their own
                    # group but are drawn in a de-emphasized neutral gray
                    # and get no legend entry, and the NAMED categories keep
                    # the first palette slots in first-appearance order --
                    # previously the None group consumed a fully-saturated
                    # palette slot the legend never explained, and shifted
                    # the named categories' colors (release-1.0 audit,
                    # F02-013).
                    _cats = list(sorted(set(hue), key=list(hue).index))
                    hue_category_names = [c for c in _cats if c is not None]
                    hue_group_labels = ['_nolegend_' if c is None else c
                                        for c in _cats]
                    # `None` marks UNLABELED points; they are a real group
                    # with no name, so say that rather than echoing the
                    # legend sentinel at the reader
                    _run_cat_names = ['the unlabeled group' if c is None
                                      else str(c) for c in _cats]
                    _base = get_palette_colors(palette,
                                               len(hue_category_names))
                    _named_idx = {c: i for i, c
                                  in enumerate(hue_category_names)}
                    mpl_kwargs["color"] = [
                        _UNLABELED_HUE_COLOR if c is None
                        else tuple(_base[_named_idx[c]]) for c in _cats]
                    hue = group_by_category(hue)

        # reshape the data according to group
        if hue is not None:
            # fail fast, naming hue=, on unhashable entries (e.g. a dict
            # passed as hue) -- previously a bare "TypeError: unhashable
            # type" escaped from deep inside reshape_data (F02-010)
            try:
                set(hue)
            except TypeError as exc:
                _bad = next((el for el in hue
                             if getattr(el, '__hash__', None) is None), None)
                raise TypeError(
                    "hue= entries must be hashable category labels, 1-D "
                    "numeric values, or the rows of a 2-D numeric matrix; "
                    f"got an entry of type {type(_bad).__name__}: {_bad!r}"
                ) from exc
            _n_datasets_before_hue = len(xform)
            if _fmt_draws_line(fmt):
                # LINE: contiguous-run segmentation preserving order AND
                # input-dataset identity (GH #291). Global category merging
                # (reshape_data) would fuse separate datasets that share a
                # category into one line, and collapse a category that
                # recurs along a trajectory (A A B B A A) into a tangled
                # polyline joining non-adjacent points. Segmenting keeps each
                # run separate, colours it by its category, bridges only runs
                # adjacent within one dataset, and gives each category ONE
                # legend entry.
                _cat_color, _cat_label = _categorical_color_label_maps(
                    hue, palette, mpl_kwargs.get("color"),
                    hue_group_labels, _hue_sort_numeric)
                (xform, labels, _run_colors, hue_group_labels, _seg_ds,
                 _run_cat_names, _seg_lengths,
                 _seg_bridged) = _regroup_categorical_lines(
                     xform, hue, labels, _cat_color, _cat_label)
                fmt = _expand_styles_to_runs(
                    fmt, mpl_kwargs, _seg_ds, _n_datasets_before_hue)
                mpl_kwargs["color"] = _run_colors
            else:
                # MARKER-only: global grouping (one trace per category) is
                # correct -- scatter has no connecting edges to fuse. Integer/
                # boolean hue is grouped in first-appearance order then
                # reordered into sorted numeric order (F13-005).
                xform, labels = reshape_data(xform, hue, labels)
                if _hue_sort_numeric:
                    _appear = list(sorted(set(hue), key=list(hue).index))
                    _order = sorted(range(len(_appear)),
                                    key=lambda i: _appear[i])
                    xform = [xform[i] for i in _order]
                    labels = [labels[i] for i in _order]
            _hue_regrouped_counts = (_n_datasets_before_hue, len(xform))
            # a PURE line cannot render a single-observation category -- it
            # draws NOTHING (and crashed animated interpolation, F02-002).
            # A non-bridged single-point run (dataset-boundary/last run, or a
            # singleton category) hits this; warn, naming the category. ('o-'
            # and other marker+line combos still show the marker, so this is
            # gated on the pure-line format only.)
            if is_line(fmt):
                _tiny = [i for i, xi in enumerate(xform) if xi.shape[0] < 2]
                if _tiny:
                    # name each singleton by its CATEGORY. Reading
                    # `hue_group_labels` here reported '_nolegend_' -- the
                    # matplotlib sentinel every repeat run of a category
                    # carries -- as though it were a category name.
                    _tiny_names = ", ".join(
                        repr(_run_cat_names[i])
                        if (_run_cat_names is not None
                            and i < len(_run_cat_names))
                        else f"group {i}" for i in _tiny)
                    warnings.warn(
                        f"hue categor{'y' if len(_tiny) == 1 else 'ies'} "
                        f"{_tiny_names} ha{'s' if len(_tiny) == 1 else 've'} "
                        "only one observation; a pure line format cannot "
                        "render a single point, so it will be invisible -- "
                        "pass fmt='.' or fmt='o-' to mark singleton "
                        "categories.", stacklevel=external_stacklevel())

    # multilevel styling for nested-list input: every leaf under the same
    # outermost group shares that group's color, and each additional nesting
    # level renders thinner and fainter (summary -> detail)
    elif nested_groups is not None and color is None and colors is None:
        import seaborn as sns
        n_outer = len(set(nested_groups))
        base_colors = sns.color_palette(
            _seaborn_palette_arg(palette, n_outer), n_outer)
        mpl_kwargs["color"] = [base_colors[g] for g in nested_groups]
        min_depth = min(nested_depths)
        if any(d != min_depth for d in nested_depths):
            mpl_kwargs["linewidth"] = [
                max(0.5, 2.0 * (0.7 ** (d - min_depth))) for d in nested_depths
            ]
            if alpha is not None:
                warnings.warn(
                    "x is a nested list with varying nesting depth: depth "
                    "fading assigns alpha by depth (summary levels "
                    "opaque, deeper/detail levels fainter); ignoring "
                    "alpha. Flatten the input if you want to set alpha "
                    "yourself.",
                    UserWarning, stacklevel=external_stacklevel())
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
    animate, morph_tags, order = _resolve_animate_mode(animate, len(xform),
                                                        order=_resolved_order)

    # title= per-segment sequence (plan 1.1 Task 8): re-validate/resolve now
    # that `len(xform)` (the FINAL, post cluster/hue-reshape dataset count)
    # and the FOLDED `animate`/`order` are both known -- the fail-fast call
    # above only confirmed title='s TYPE. `_segment_titles` is None for
    # every scalar/None title (the overwhelmingly common case); otherwise a
    # list of `len(xform)` per-dataset strings, and the STATIC axes title is
    # cleared (it is driven per frame instead, by `_make_title_updater`
    # below, through the SAME `_frame_hooks` registry Task 7 built for
    # `on_frame=`). See `_make_title_updater` for why segment PARITY
    # (`ctx.segment_kind`), never `ctx.current_fraction`, is the
    # hold/transition discriminator -- both sweep 0->1 over their own
    # segment, so a fraction alone cannot tell them apart.
    _segment_titles = _validate_title(title, style=animate, order=order,
                                      n_datasets=len(xform))
    if _segment_titles is not None:
        title = None      # the axes title is driven per frame, not statically

    # round17 #9 (GH #123): animate='morph' now supports 2-D as well as
    # 3-D data, matching every other animate style -- only 1-D (and any
    # higher-than-3-D result, which `plot.py` never actually produces for
    # plotting) has no hull/point-cloud concept to morph between.
    if morph_tags is not None and xform[0].shape[1] not in (2, 3):
        raise NotImplementedError(
            "animate='morph' is only supported for 2-D or 3-D plots; the "
            f"data being plotted is {xform[0].shape[1]}-D. Pass ndims=2 or "
            "ndims=3 (the default) to use animate='morph'."
        )

    # animate='morph' tractability guard: `morph_tags` marks which FINAL
    # (post cluster/hue-reshape) datasets join the morph sequence, so an
    # untagged static backdrop of any size is irrelevant here. An explicit
    # morph_samples= means the caller already chose, so `simplify` never
    # engages.
    if morph_tags is not None and morph_samples is None:
        _morph_sizes = [int(np.asarray(xform[i]).shape[0])
                        for i, _tagged in enumerate(morph_tags) if _tagged]
        _largest = max(_morph_sizes)
        if _largest > MORPH_SAMPLES_REQUIRED_ABOVE:
            if simplify:
                # SILENT by maintainer decision (2026-07-29): no warning, no
                # print. Hanging is worse than approximating, and the caller
                # who wants the guarantee back has simplify=False.
                morph_samples = MORPH_SAMPLES_REQUIRED_ABOVE
            else:
                raise ValueError(
                    f"animate='morph' received a cloud of {_largest} points. "
                    "The one-to-one point matching is a Hungarian assignment "
                    "(~O(n^3), with an n x n cost matrix), so this does not "
                    "finish in usable time or memory: measured, the built-in "
                    "zoo shapes were still running after 10 minutes, while "
                    f"morph_samples={MORPH_SAMPLES_REQUIRED_ABOVE} renders "
                    "the same call in 8.2 s. Set simplify=True (the default) "
                    "to let hypertools downsample to "
                    f"{MORPH_SAMPLES_REQUIRED_ABOVE} points per cloud "
                    "automatically, or pass morph_samples=<int> to choose "
                    "the cap yourself. With simplify=False and no "
                    "morph_samples=, every dataset keeps its full point "
                    "count and no real data point is ever dropped.")

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

    # 2-D animations (round17 #9, GH #123): fixed (non-rotating) viewport --
    # `rotations=`/`zoom=` are 3-D camera controls with no 2-D equivalent,
    # so whenever `animate` is truthy on 2-D data and either was set to a
    # non-default value, warn (once, here -- BEFORE dispatching to either
    # backend, so both backends behave identically) that it is ignored.
    # Applies uniformly to every animate style, including 'morph': its
    # `rotations` doubles as a per-segment PACING control in 3-D (see
    # `hypertools.plot.morph.segment_frame_counts`), not purely a camera
    # control, but that coupling is itself 3-D-camera-derived, so 2-D
    # morphs always use even segment timing for consistency with every
    # other 2-D animate style (both backends -- see
    # `matplotlib_backend.animate_plot2D`/`plotly_backend._add_animation`).
    if animate and xform[0].shape[1] == 2:
        _rotations_is_default = (
            rotations == 1 if not isinstance(rotations, (list, tuple))
            else False
        )
        if not _rotations_is_default:
            warnings.warn(
                "rotations= controls 3-D camera spin and has no effect on "
                "2-D animations, which use a fixed (non-rotating) "
                "viewport; ignoring.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
        if zoom != 1:
            warnings.warn(
                "zoom= controls the 3-D camera's distance/box-aspect zoom "
                "and has no 2-D equivalent; ignoring.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )

    # chemtrails/precog/bullettime (GH #127): broadcast bool-or-list to the
    # FINAL (post cluster/hue-reshape) dataset count, same as surface=/
    # density= above -- each accepts a single bool (applied to every
    # dataset) or a list/tuple of bool (one entry per drawn dataset, mixed
    # per-dataset combinations allowed). `animate` itself stays a single
    # GLOBAL mode -- only these trail FLAGS become per-dataset.
    chemtrails = broadcast_trail_flag(chemtrails, len(xform), "chemtrails")
    precog = broadcast_trail_flag(precog, len(xform), "precog")
    bullettime = broadcast_trail_flag(bullettime, len(xform), "bullettime")

    # trail flags on a STATIC plot (release-1.0 audit, F05-007): the same
    # user mistake the spin/serial/morph/window branch below already warns
    # about -- a user who forgot animate=True got no feedback about why
    # their trails were missing.
    if not animate:
        _static_trail_flags = [
            name for name, flags in (("chemtrails", chemtrails),
                                     ("precog", precog),
                                     ("bullettime", bullettime))
            if any(flags)
        ]
        if _static_trail_flags:
            warnings.warn(
                f"{'/'.join(_static_trail_flags)} only affect ANIMATED "
                "plots and will be ignored here; pass animate=True (or "
                "'parallel') to draw trails.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )

    # GH #127 (+ morph/window follow-up): 'spin' has no "current position"
    # (only the camera moves, so a trail has nothing to trail BEHIND or AHEAD
    # of), 'morph' draws a single traveling point-cloud artist with no
    # per-dataset "current position" either, and 'window' (round17 #8) is
    # explicitly bullettime MINUS its chemtrails/precog trail components
    # (Jeremy's own definition) -- trail styles are semantically meaningless
    # in all three, so warn once (naming the mode, which flag(s) were set, and
    # for which dataset indices) rather than silently building frozen/
    # invisible trail artists. `_draw`/`plotly_draw` skip creating those
    # artists entirely for these modes (see their own `style`/`animate`
    # branches), so this is purely informational -- no flags are mutated here.
    #
    # 'serial' COMPOSES with the trail flags on BOTH backends
    # (chemtrails-serial / precog-serial / bullettime-serial -- the currently-
    # revealing dataset traces out its own trail; see
    # `matplotlib_backend.update_lines_serial` and the matching serial branch
    # in `plotly_backend._add_animation`), so it is never ignored. Only the
    # styles that draw no per-dataset reveal at all still ignore trails.
    _trail_ignoring_modes = ("spin", "morph", "window")
    if animate in _trail_ignoring_modes:
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
                stacklevel=external_stacklevel(),
            )

    # names= (QC 2026-07): per-DATASET names, distinct from per-point `labels=`
    # (text call-outs on individual observations) and the `legend=True`
    # auto-numbering. Each name labels its dataset's trace and turns the legend
    # on, so `hyp.plot([raw, a, b, c], names=['raw','a','b','c'], ...)` shows a
    # legend naming the four datasets. Resolved BEFORE the legend block below so
    # it wins over a bare legend=True; explicit conflicting values raise.
    if names is not None:
        names = list(names)
        if _hue_regrouped_counts is not None:
            # a categorical hue stacks and REGROUPS the data by category,
            # so the drawn traces are hue groups, not the input datasets
            # names= labels -- previously this surfaced as a misleading
            # "names must have one entry per dataset (<group count>)"
            # error, or silently labeled category groups with dataset
            # names whenever the counts coincided (F02-009).
            _nd, _ng = _hue_regrouped_counts
            raise ValueError(
                "names= assigns one name per input dataset, but hue= "
                f"regrouped the data into hue groups ({_nd} dataset(s) -> "
                f"{_ng} hue group(s)), so per-dataset names cannot apply. "
                "Label the hue groups with legend=[...] (one entry per "
                "group, in first-appearance order) instead, or drop hue=.")
        if _multiindex_meta is not None:
            # Same shape of mistake as the hue case above, and for the same
            # reason: a hierarchy draws one trace per leaf PLUS one per
            # per-level mean, none of which is an input dataset (there is
            # exactly one input frame). Before this branch the caller got
            # either "names must have one entry per dataset (6)" -- a count
            # that names nothing they passed -- or, when the counts
            # happened to coincide, the flatly false "pass dataset names
            # via names= OR a legend= list, not both" on a call with no
            # legend= at all (the MultiIndex branch had already written the
            # hierarchy's labels into `legend`).
            _n_leaves = len(_multiindex_meta['leaf_keys'])
            _n_means = len(xform) - _n_leaves
            # the flatten remedy differs by AXIS, exactly as the cluster=
            # messages above do: df.reset_index(drop=True) does not touch a
            # column MultiIndex, so offering it there sends the caller
            # round the same error again.
            _flatten = ("df.reset_index(drop=True)" if _mi_axis == 'rows'
                        else "df.columns = df.columns.map('_'.join)")
            raise ValueError(
                "names= assigns one name per input dataset, but x has "
                f"{_mi_which}, so the drawn traces are hierarchy groups "
                f"({_n_leaves} leaf trajectory/ies + {_n_means} derived "
                "per-level mean(s)), not input datasets. Label the "
                "top-level groups with legend=[...] (one entry per unique "
                "top-level index value, in first-appearance order) "
                f"instead, or flatten the hierarchy ({_flatten}).")
        if len(names) != len(xform):
            raise ValueError(
                f"names must have one entry per dataset ({len(xform)}); got "
                f"{len(names)}")
        if _legend_user_list:
            # `_legend_user_list`, not `isinstance(legend, list)`: several
            # branches above REPLACE `legend` with an internally-derived
            # label list, and testing the current value made this fire on
            # calls that only ever passed names=.
            raise ValueError(
                "pass dataset names via names= OR a legend= list, not both")
        legend = names

    # handle legend
    if legend is not None:
        if legend is False:
            legend = None
        elif legend is True and hue is not None:
            if hue_group_labels is not None:
                # categorical string hue: show the ORIGINAL category names,
                # not the integer group ids `hue` was reassigned to above
                # ('_nolegend_' placeholders keep unnamed None-entry groups
                # out of the legend while matching the trace count).
                legend = list(hue_group_labels)
            else:
                legend = [item for item in
                         sorted(set(hue), key=list(hue).index)]
        elif legend is True and hue is None:
            legend = [i + 1 for i in range(len(xform))]

        # a legend LIST must carry one entry per drawn trace -- checked
        # here (naming legend=, the kwarg the user actually passed) rather
        # than letting parse_kwargs report a mismatch on the internal
        # 'label' kwarg (F10-010).
        if isinstance(legend, (list, tuple)) and len(legend) != len(xform):
            raise ValueError(
                f"legend= was given as a list of length {len(legend)}, "
                f"but there are {len(xform)} dataset(s)/group(s) to plot; "
                "pass one entry per drawn dataset, or legend=True to "
                "auto-number them.")

        mpl_kwargs["label"] = legend

    # colorbar (GH #100): resolve the color-mapping info (continuous hue
    # value range + palette, or discrete group colors + labels) now, while
    # `hue`/`multicolor_hue`/`xform`/`legend` reflect the FINAL grouping
    # decision (post cluster/hue reshape, post legend-label resolution) but
    # BEFORE interpolation (which doesn't change the mapping, only the
    # point density) -- shared by both the matplotlib and plotly backends.
    colorbar_info = _build_colorbar_info(
        colorbar, hue, multicolor_hue, cluster, n_clusters, xform,
        mpl_kwargs, legend, palette, hue_group_labels=hue_group_labels,
        hierarchy_labels=_mi_colorbar_labels)

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
    # a per-dataset fmt LIST re-checked against the FINAL trace count (the
    # early pre-pipeline check above cannot run when hue=/cluster=/
    # MultiIndex regrouping changes the number of drawn traces) -- a
    # mismatch used to surface as a bare IndexError from the loop below
    # (F01-005/F01-006/F10-003).
    if isinstance(fmt, list) and len(fmt) != len(xform):
        raise ValueError(
            f"fmt was given as a list of length {len(fmt)}, but there are "
            f"{len(xform)} trace(s) to draw (the drawn-trace count can "
            "differ from the input dataset count when hue=/cluster=/"
            "n_clusters= or a MultiIndex regroups the data); pass one "
            "format string per drawn trace, or a single fmt string to "
            "broadcast it to every trace.")
    # STATIC line smoothing is DATA-FAITHFUL (`_interp_static_line`): it
    # only ever ADDS points between samples, keeps every original sample
    # (including the final one) as a drawn vertex, and uses a fixed target
    # density -- so duration=/frame_rate= (animation kwargs) no longer
    # change static rendering (F01-001/F01-007). ANIMATED plots keep the
    # historical frame_rate*duration grid: there the interpolated rows ARE
    # the animation's frame-sampling.
    if animate == "morph":
        pass
    elif fmt is None or isinstance(fmt, str):
        if has_line_component(fmt):
            if any(xi.shape[0] > 1 for xi in xform):
                # rows with remaining NaN/inf would crash PCHIP with a bare
                # scipy message -- fail fast with a hypertools-level one
                # (release-1.0 audit, F05-011)
                for _i, _xi in enumerate(xform):
                    if _xi.shape[0] > 1:
                        _require_finite_for_line(_xi, _i)
                if animate:
                    # Every multi-row dataset is resampled onto the EXACT
                    # frame grid (release-1.0 audit): per-dataset
                    # interpolation (previously the step came from
                    # xform[0]'s length alone, silently truncating longer
                    # LATER datasets, F04-003) with exactly
                    # round(frame_rate * duration) rows (the docstring's
                    # promised frame count -- the old np.arange step
                    # produced 901/41 frames for some lengths, F04-004).
                    # Per-dataset singleton guard (F02-002/F05-012): a
                    # 1-point dataset (singleton hue category, or a
                    # reference point plotted beside a trajectory) cannot
                    # be PCHIP-interpolated (scipy needs >= 2 samples) --
                    # leave it as-is instead of crashing the whole plot;
                    # the backend paces it onto the frame grid.
                    _n_frames = max(2, int(round(frame_rate * duration)))
                    xform = [xi if xi.shape[0] < 2
                             else _interp_anim_line(xi, _n_frames)
                             for xi in xform]
                elif antialias:
                    # static antialiasing (see `plot`'s `antialias=`)
                    xform = [_interp_static_line(xi) for xi in xform]
    elif isinstance(fmt, list):
        for idx, xi in enumerate(xform):
            if has_line_component(fmt[idx]):
                if xi.shape[0] > 1:
                    # see the F05-011 note above
                    _require_finite_for_line(xi, idx)
                    # per-dataset exact frame grid -- see the F04-003/
                    # F04-004 note in the single-fmt branch above. (The
                    # historical interp_array_list call here treated the
                    # 2D array as a LIST of rows, silently replacing the
                    # dataset with a list of per-row interpolations --
                    # latent for years because a bug made is_line() always
                    # False.)
                    if animate:
                        xform[idx] = _interp_anim_line(
                            xi, max(2, int(round(frame_rate * duration))))
                    elif antialias:
                        # static antialiasing (see `plot`'s `antialias=`)
                        xform[idx] = _interp_static_line(xi)

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
            multicolor_hue, pre_interp_lengths, xform, palette,
            is_rgb=multicolor_hue_is_rgb)

    # handle explore flag (a real ValueError, not an assert -- asserts are
    # stripped under `python -O`, F01-016/F10-013)
    if explore:
        if xform[0].shape[1] != 3:
            raise ValueError(
                "explore mode is currently only supported for 3-D static "
                f"plots; the data being plotted is {xform[0].shape[1]}-D. "
                "Pass ndims=3 (the default) to use explore=True.")
        # headless/non-interactive backends (Agg in scripts and CI, the
        # doc-gallery build, ...) can render the figure but can never fire
        # hover events, so explore=True silently degraded to a static plot
        # with no hint why nothing pops up (release-1.0 audit,
        # D05-gallery-data-text-012).
        import matplotlib
        _backend_name = matplotlib.get_backend().lower()
        if any(_backend_name.endswith(nb) for nb in
               ("agg", "pdf", "svg", "ps", "template")) \
                and not _backend_name.endswith(("qtagg", "tkagg", "gtk3agg",
                                                "gtk4agg", "wxagg",
                                                "macosx")):
            warnings.warn(
                "explore=True shows labels on hover, which needs an "
                "interactive matplotlib backend; the current backend "
                f"({matplotlib.get_backend()!r}) is non-interactive, so "
                "the figure will be drawn as a static plot without hover "
                "labels. Run in an interactive session (or switch "
                "backends, e.g. matplotlib.use('QtAgg')) to use explore "
                "mode.", UserWarning, stacklevel=external_stacklevel())
        mpl_kwargs["picker"] = True

    # predict= forecasts were computed per ORIGINAL input dataset; if
    # cluster/hue reshaping regrouped `xform` into a different number of
    # traces (by category rather than by dataset), the 1:1 correspondence
    # no longer holds -- skip drawing forecasts rather than mismatch traces.
    _forecast_owner = None
    #: Why no forecast overlay was drawn, or `None` when one was. The
    #: `return_model=True` bundle reports this alongside the forecasts
    #: themselves: a fit that SUCCEEDED but could not be rendered is still a
    #: model result, and discarding it to make the bundle mirror the figure
    #: would throw away the very thing `return_model=` exists to hand back.
    _forecast_draw_reason = None
    if _ft is not None:
        # A HIERARCHY's forecasts were computed over `FinalTraces.arrays`
        # itself, so a count mismatch is not a legitimate regrouping (there
        # is none -- the MultiIndex branch wins the cluster/hue chain) but a
        # bug in this file. Assert rather than drop: the pre-1.1 guard
        # nulled `raw_forecasts` on any mismatch, which is precisely how a
        # missing per-trace forecast would become invisible.
        _ft.assert_consistent(raw_forecasts=raw_forecasts,
                              bundle_forecasts=bundle_forecasts)
    elif raw_forecasts is not None and len(raw_forecasts) != len(xform):
        # A forecast belongs to a DATASET and is anchored at that dataset's
        # last observation, so after regrouping it belongs to whichever run
        # holds that observation -- which is also the trace it visually
        # continues, and so the trace whose style it should inherit.
        # `segment_by_run` reports each run's source dataset, so that run is
        # the LAST one carrying this dataset's index.
        #
        # Dropping on a count mismatch (what this did before) made survival
        # an accident of how the categories happened to fall: 2 datasets
        # split into 2 runs kept their forecasts, into 8 runs lost them.
        if _seg_ds is not None and len(_seg_ds) == len(xform):
            _owner = {}
            for _run, _ds in enumerate(_seg_ds):
                _owner[_ds] = _run           # last write wins = final run
            if set(_owner) == set(range(len(raw_forecasts))):
                _forecast_owner = [_owner[i]
                                   for i in range(len(raw_forecasts))]
        if _forecast_owner is None:
            # No usable mapping -- in practice MARKER-only categorical
            # regrouping, which goes through `reshape_data` and groups
            # GLOBALLY by category, so 3 datasets under 2 categories become 2
            # traces that are not datasets at all. (A CONTINUOUS hue, named
            # here previously, is NOT an example: it colours one line artist
            # per dataset through a LineCollection overlay without changing
            # the trace count, so the correspondence holds and its forecasts
            # draw -- measured, and pinned by
            # `test_a_CONTINUOUS_hue_DRAWS_its_forecasts_and_never_refused`.)
            # Say so: a forecast the user asked for and did not get must not
            # vanish in silence.
            warnings.warn(
                f"predict= was given, but the forecast overlays could not be "
                f"matched to the {len(xform)} drawn trace(s): hue=/cluster= "
                f"regrouped the data and left no per-dataset trace to anchor "
                f"them to. No forecast is drawn. Plot without hue=/cluster=, "
                f"or use a CATEGORICAL hue (which keeps one trace per run "
                f"and so keeps its forecasts).",
                stacklevel=external_stacklevel())
            _forecast_draw_reason = (
                f"hue=/cluster= regrouped the data into {len(xform)} trace(s) "
                f"with no per-dataset trace to anchor the forecasts to")
            raw_forecasts = None
            analyze_histories = None

    # Run -> dataset -> original rows, for the animation's reveal clock and
    # (below) the forecast schedule. Built for regrouped LINE plots and for
    # unregrouped ones, so both take the same code path; left None for
    # anything whose drawn traces do not correspond to input datasets.
    from .ownership import TraceOwnership
    _ownership = None
    if _seg_ds is not None and _seg_lengths is not None:
        _ownership = TraceOwnership.from_segments(
            _seg_ds, _seg_lengths, _seg_bridged)
    elif (_hue_regrouped_counts is None and isinstance(xform, list)
            and len(xform) == len(raw_xform)):
        # nothing regrouped: one drawn trace per input dataset. MARKER-only
        # hue regrouping (`reshape_data`, plot.py:4503) also leaves `_seg_ds`
        # None while CHANGING the trace count -- it groups globally by
        # category, so its traces are not datasets at all and must not be
        # described as such.
        _ownership = TraceOwnership.identity([len(xi) for xi in raw_xform])

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
    # Animated predict= (CASE A -- STATIC data revealed over time): every
    # observation is known before the first frame, so every forecast the
    # animation will ever draw is knowable now. Precompute the whole schedule
    # HERE so (a) it can go into the centre/scale statistics below and land
    # inside the cube BY CONSTRUCTION -- no clamping, unlike the streaming
    # path in hypertools/io/streaming.py, where the box is frozen from the
    # head samples -- and (b) every frame is a pure lookup, so ani.save() and
    # to_jshtml() replays render identically.
    forecast_schedule = None
    # The ANIMATED forecast needs one drawn trace per dataset: the schedule
    # maps frame-grid rows onto each dataset's raw rows, and `hue=`/`cluster=`
    # regrouping replaces the per-dataset traces with per-RUN ones. The static
    # overlay copes (it draws once, anchored on the run holding each dataset's
    # last observation), but a per-frame reveal defined over runs cannot yet
    # be mapped back onto per-dataset histories -- building the schedule
    # anyway raises IndexError partway through the first frame, which is how
    # this was found.
    _reveal = None
    if (raw_forecasts is not None and analyze_histories is not None
            and animate and animate not in ('spin',)
            and len(analyze_histories) != len(xform)
            and _ownership is not None):
        # `hue=`/`cluster=` regrouped the data into one trace per RUN, and a
        # `TraceOwnership` says which dataset each run came from and from
        # which of its rows. `DatasetRevealSchedule` maps every frame onto
        # that dataset's own visible rows, read off the SAME `RunWindow`s the
        # backends slice their artists with -- so the fitted history and the
        # drawn trajectory cannot describe different states. The refusal
        # below now applies only when there is no such mapping at all.
        from .forecast import DatasetRevealSchedule
        from .trails import head_window_frames
        _reveal = DatasetRevealSchedule(
            _ownership, [np.asarray(xi).shape[0] for xi in xform],
            n_frames=max(1, int(round(frame_rate * duration))),
            # the length the BACKENDS will use, from the one function all
            # three callers share -- a schedule built on a different window
            # would disagree with the picture about what is on screen
            window_frames=head_window_frames(
                frame_rate, tail_duration, resolved_focused,
                animate == 'window', chemtrails, precog, bullettime),
            serial=(animate == 'serial' or order == 'serial'))
    if (raw_forecasts is not None and analyze_histories is not None
            and animate and animate not in ('spin',)
            and len(analyze_histories) != len(xform)
            and _reveal is None):
        _reveal_kind = ('serial' if (animate == 'serial' or order == 'serial')
                        else 'parallel')
        warnings.warn(
            f"predict= with an ANIMATED plot needs one drawn trace per "
            f"dataset, but hue=/cluster= regrouped {len(analyze_histories)} "
            f"dataset(s) into {len(xform)} trace(s), so no forecast overlay "
            f"is drawn (animate={animate!r}, {_reveal_kind} reveal); the observed "
            f"trajectories still animate. Plot statically (drop animate=), "
            f"drop hue=/cluster=, or render the forecast as a separate "
            f"plot.",
            stacklevel=external_stacklevel())
        _forecast_draw_reason = (
            f"an ANIMATED plot needs one drawn trace per dataset, but "
            f"hue=/cluster= regrouped {len(analyze_histories)} dataset(s) "
            f"into {len(xform)} trace(s)")
        analyze_histories = None
        # ...and drop the DRAWING copy too. Leaving it set drew the
        # full-history forecast as a STATIC overlay on the plotly backend
        # (whose static block fires whenever there is no schedule), i.e. the
        # final forecast visible from frame 0 -- on a figure whose warning
        # says none is drawn. It also inflated the centre/scale statistics
        # below to fit a forecast that is never rendered. `bundle_forecasts`
        # is untouched: the fit succeeded and is still reported.
        raw_forecasts = None
    if (raw_forecasts is not None and analyze_histories is not None
            and animate and animate not in ('spin',)):
        from .forecast import ForecastSchedule
        # `max(1, ...)`, matching what BOTH backends pace with (`total_frames`
        # in every updater and in `_add_animation`) -- NOT the `max(2, ...)`
        # used above for the interpolation grid, which is floored at 2 only
        # because PCHIP needs two samples to interpolate between. The two are
        # different quantities and were the same expression: at
        # `round(frame_rate * duration) == 1` the renderer drew a single frame
        # holding the WHOLE trajectory while a schedule built for 2 frames
        # reported one row revealed, so that frame showed all the data and no
        # forecast at all. Measured on an 8-row dataset at frame_rate=1,
        # duration=1: renderer 8 rows, schedule 1. Every other frame count
        # already agreed (checked at 2 and 12), so this changes nothing else.
        _n_frames = max(1, int(round(frame_rate * duration)))
        _grid_lengths = [len(xi) for xi in xform]
        _builder = (ForecastSchedule.for_serial
                    if (animate == 'serial' or order == 'serial')
                    else ForecastSchedule.for_parallel)
        from .forecast import DEFAULT_SLOW_WARNING_SECONDS
        _slow_secs = (DEFAULT_SLOW_WARNING_SECONDS
                      if slow_warning_seconds is _UNSET_SLOW_WARNING
                      else slow_warning_seconds)
        if _reveal is not None:
            # rows from the reveal, not counts from the drawn traces: a
            # dataset may now be spread over several of them
            forecast_schedule = ForecastSchedule.for_regrouped(
                analyze_histories, _reveal, model=predict, t=t,
                n_frames=_n_frames, slow_warning_seconds=_slow_secs)
        else:
            forecast_schedule = _builder(
                analyze_histories, _grid_lengths, model=predict, t=t,
                n_frames=_n_frames, slow_warning_seconds=_slow_secs)

    if raw_forecasts is not None:
        _fc_rows = [np.vstack(raw_forecasts)]
        if forecast_schedule is not None:
            _fc_rows.append(forecast_schedule.stacked_paths())
        _joint = np.vstack([np.vstack(xform)] + _fc_rows)
        _mean = np.mean(_joint, 0)
        xform = [xi - _mean for xi in xform]
        raw_forecasts = [fc - _mean for fc in raw_forecasts]
        raw_xform = [xi - _mean for xi in raw_xform]

        _joint = np.vstack([np.vstack(xform)]
                           + [r - _mean for r in _fc_rows])
        _m1 = np.min(_joint)
        _m2 = np.max(_joint - _m1) or 1.0  # degenerate (constant) data has
        # zero range: dividing by it emitted an 'invalid value encountered
        # in divide' RuntimeWarning and produced NaNs (release-1.0 audit,
        # C2 residual warnings); constant data maps to a finite fixed
        # position instead
        def _rescale(a):
            return 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
        raw_forecasts = [_rescale(fc) for fc in raw_forecasts]
        raw_xform = [_rescale(xi) for xi in raw_xform]
        if forecast_schedule is not None:
            # hand the schedule the SAME affine the data went through, so
            # `polyline()` returns display-box coordinates
            from .forecast import DisplayTransform
            forecast_schedule = forecast_schedule.to_display(
                DisplayTransform(_mean, _m1, _m2))
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
        _m2 = np.max(_stacked - _m1) or 1.0  # zero range (e.g. a single
        # observation reduced to zeros, or constant data) -> a finite
        # fixed position, instead of a divide-by-zero RuntimeWarning +
        # NaNs (release-1.0 audit, C2 residual warnings)
        def _rescale(a):
            return 2 * (np.divide(a - _m1, _m2)) - 1
        xform = [_rescale(xi) for xi in xform]
        raw_xform = [_rescale(xi) for xi in raw_xform]

    # handle palette with seaborn
    import seaborn as sns
    if isinstance(palette, np.bytes_):
        palette = palette.decode("utf-8")

    # a bare (r, g, b[, a]) tuple/list of floats is a SINGLE matplotlib
    # color (F10-004), not a per-dataset list -- broadcast it to every
    # dataset before parse_kwargs' per-dataset list handling sees it.
    _color_val = mpl_kwargs.get("color")
    if (isinstance(_color_val, (list, tuple))
            and len(_color_val) in (3, 4)
            and all(isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool) and 0 <= v <= 1
                    for v in _color_val)):
        mpl_kwargs["color"] = [tuple(_color_val)] * len(xform)

    # turn kwargs into a list
    kwargs_list = parse_kwargs(xform, mpl_kwargs)

    # GH #206: arbitrary extra matplotlib-style kwargs (anything not one
    # of plot()'s own named parameters, e.g. `zorder=`, `dashes=`,
    # `markeredgecolor=`) are merged in AFTER the named/internal style
    # kwargs above (`_apply_extra_kwargs` never overwrites a key already
    # set), verbatim -- no per-dataset list broadcasting is attempted for
    # these (see `_apply_extra_kwargs`'s docstring for why). `alpha=` is a
    # named parameter as of 1.1, so it never reaches this generic path.
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
        elif line_colors is not None:
            # hue= is set: use each dataset's MEAN per-point hue color as its
            # representative color, so surface=/density=/morph honor hue instead
            # of falling back to the palette cycle (QC 2026-07: surface=True
            # ignored hue -- the hull/mesh drew in a palette color while the
            # points were hue-colored).
            return [tuple(np.asarray(lc, dtype=float).mean(axis=0)[:3])
                    for lc in line_colors]
        else:
            _base_colors = list(sns.color_palette(
                _seaborn_palette_arg(palette, len(xform)), len(xform)))
        return [
            _mcolors.to_rgb(c) if c is not None
            else _mcolors.to_rgb(f"C{i % 10}")
            for i, c in enumerate(_base_colors)
        ]

    # surface= (GH #109): resolve each dataset's OWN drawn color now (used
    # when a dataset's surface spec has color=None, i.e. "inherit").
    surface_colors = (_resolve_dataset_colors()
                      if surface_list is not None else None)

    # surface= per-vertex coloring (QC 2026-07): when hue is set, color each
    # surface hull VERTEX by an inverse-distance-weighted blend of the enclosed
    # points' hue colors (meshutil.vertex_colors_from_points) rather than one
    # flat mean color (the old behavior painted the whole hull the average of
    # the points' colors -- e.g. gray for a rainbow hue). Bundle each dataset's
    # (points, per-point RGB); None where a dataset has no surface, no per-point
    # hue colors, or an EXPLICIT surface color= was given (an explicit color
    # wins over the inferred hue -- otherwise it would be silently ignored),
    # in which case surface_colors' flat color is used.
    def _surface_inherits_color(i):
        spec = surface_list[i] if i < len(surface_list) else None
        return spec is not None and spec.get('color') is None

    if surface_list is not None and line_colors is not None:
        surface_point_colors = [
            (np.asarray(xform[i])[:, :3], np.asarray(line_colors[i])[:, :3])
            if _surface_inherits_color(i) else None
            for i in range(len(xform))
        ]
    else:
        surface_point_colors = None

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
            # COPY the caller's list: the matplotlib backend rewrites
            # single-point line entries to '.' in place, which must not
            # leak back into the user's own fmt list
            # (X6-code-org-plot-008)
            draw_fmt = list(fmt)
    else:
        # sized from the FINAL trace count -- `x` is the ORIGINAL input,
        # whose length differs after hue=/cluster= regrouping (F01-005:
        # fmt=None + hue on a list input crashed with a bare IndexError)
        draw_fmt = ["-"] * len(xform)

    # convert all nans to zeros
    for i, xi in enumerate(xform):
        xform[i] = np.nan_to_num(xi)
    raw_xform = [np.nan_to_num(xi) for xi in raw_xform]
    if raw_forecasts is not None:
        raw_forecasts = [np.nan_to_num(fc) for fc in raw_forecasts]

    # forecast_*= overrides, resolved ONCE here rather than in each backend --
    # `forecast_cluster=` in particular must give both backends the same
    # labels, and clustering twice would not guarantee that. Resolved at this
    # point because the forecasts are now in the space they are DRAWN in, so
    # `forecast_cluster=` groups the geometry the user actually sees rather
    # than a pre-reduction one they do not.
    _forecast_overrides = None
    if raw_forecasts is not None and any(
            v is not None for v in _forecast_style_kwargs.values()):
        from .forecast import resolve_forecast_overrides
        _forecast_overrides = resolve_forecast_overrides(
            len(raw_forecasts), raw_forecasts,
            hue=forecast_hue, cluster=forecast_cluster,
            n_clusters=forecast_n_clusters,
            palette=forecast_palette, fmt=forecast_fmt,
            stacklevel=external_stacklevel())

    # on_frame= (plan 1.1 Task 7): ONE shared registry, created before
    # either backend's per-frame closures exist, and threaded into both --
    # `plotly_draw`/`_add_animation` dispatch it directly (plotly builds
    # every frame in a Python loop at IMPORT time, below); the matplotlib
    # path installs it as the outermost wrapper of `line_ani._func` further
    # down, and `HyperAnimation` adopts this SAME object so a callback
    # registered after construction (`anim.on_frame(cb)`) reaches the same
    # list the dispatcher reads (review C7 -- a list created fresh inside
    # `HyperAnimation.__new__` would never be seen by the closure).
    _frame_hooks = FrameHooks([on_frame] if on_frame is not None else [])

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
            , stacklevel=external_stacklevel())

        if "color" not in mpl_kwargs:
            import seaborn as sns_local
            mpl_kwargs = dict(mpl_kwargs)
            mpl_kwargs["color"] = sns_local.color_palette(
                _seaborn_palette_arg(palette, len(xform)), len(xform))
            kwargs_list = parse_kwargs(xform, mpl_kwargs)
            _apply_extra_kwargs(kwargs_list, kwargs)
        fig = plotly_draw(
            xform,
            into=_plotly_into,
            # the same run -> dataset -> rows mapping the matplotlib updaters
            # pace their reveal with, so neither backend re-derives it
            ownership=_ownership,
            # ...and the same reveal schedule its per-frame forecast colours
            # come from (Decision R3)
            forecast_reveal=_reveal,
            fmt=draw_fmt,
            antialias=antialias,
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
            focused=resolved_focused,
            chemtrails=chemtrails,
            precog=precog,
            bullettime=bullettime,
            zoom=zoom,
            forecasts=raw_forecasts,
            # which drawn run each forecast continues after hue=/cluster=
            # regrouping -- the SAME mapping `_draw_forecast_overlays` uses
            # on the matplotlib side, so neither backend has to re-derive it
            forecast_owner=_forecast_owner,
            # the SAME resolved per-dataset overrides the matplotlib path
            # applies, so `forecast_cluster=` cannot label differently here
            forecast_overrides=_forecast_overrides,
            # the SAME precomputed, display-mapped schedule the matplotlib
            # path drives its per-frame forecast artists from, so the two
            # backends read one table instead of two transcriptions of it
            forecast_schedule=forecast_schedule,
            forecast_trail=_n_forecast_trail,
            colorbar_info=colorbar_info,
            surface=surface_list,
            surface_colors=surface_colors,
            surface_point_colors=surface_point_colors,
            density=density_list,
            density_colors=density_colors,
            morph_tags=morph_tags,
            morph_colors=morph_colors,
            morph_samples=morph_samples,
            font=_artist_font,
            font_extra=_plotly_font_extra,
            label_alpha=resolved_label_alpha,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            frame_hooks=_frame_hooks,
            segment_titles=_segment_titles,
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
            sns.set_palette(
                palette=_seaborn_palette_arg(palette, len(xform)),
                n_colors=len(xform))
            sns.set_style(style="whitegrid")
            # Font, applied AFTER sns.set_style (which sets its own font
            # rcParams). A LIST gives matplotlib >= 3.6 PER-GLYPH fallback, so
            # text mixing scripts renders fully instead of showing "tofu" for
            # whatever the single active face lacks -- and an rcParam also
            # covers text hypertools never touches directly (tick labels, and
            # anything the user adds to the returned axes). Scoped by
            # rc_context, so the user's global rcParams are intact.
            #
            # An EXPLICIT font= is made primary; an AUTO-detected font (which
            # `resolve_font` only returns to fill a real coverage GAP -- e.g. a
            # script no stack family has) is appended as a FALLBACK so the
            # bundled Noto Sans stays the primary face and a stray accent/Greek
            # letter never swaps the whole plot onto a platform font
            # (maintainer font review).
            if resolved_font is None:
                _font_stack = sans_serif_stack()
            elif font is not None:
                _font_stack = sans_serif_stack(first=resolved_font.get_name())
            else:
                _font_stack = sans_serif_stack(extra=resolved_font.get_name())
            # BOTH keys: artists created with an explicit generic
            # `family='sans-serif'` (seaborn's style, and matplotlib's own
            # default) resolve through `font.sans-serif`, while artists that
            # inherit the rcParam resolve through `font.family` -- setting only
            # one leaves the other resolving through matplotlib's stock list.
            plt.rcParams['font.family'] = _font_stack
            plt.rcParams['font.sans-serif'] = _font_stack
            # The bundled Noto Sans has no U+2212 MINUS SIGN. hypertools' own
            # axes carry the whole stack, so matplotlib's per-glyph fallback
            # reaches DejaVu Sans; a caller-supplied `ax=` was created outside
            # this context, its tick labels carry the 'sans-serif' ALIAS, and
            # an alias resolves to ONE font (Noto Sans) with no fallback -- so
            # the layout pass below warned "Glyph 8722 missing from font(s)
            # Noto Sans" on every negative tick (1.1 feature tour, 2026-09-04).
            # Format negatives with ASCII '-' while hypertools draws.
            plt.rcParams['axes.unicode_minus'] = False

            # draw the plot
            fig, ax, data, line_ani = _draw(
                xform,
                fmt=draw_fmt,
                antialias=antialias,
                kwargs_list=kwargs_list,
                labels=labels,
                legend=legend,
                title=title,
                animate=animate,
                raw_data=raw_xform,
                duration=duration,
                tail_duration=tail_duration,
                focused=resolved_focused,
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
                surface_point_colors=surface_point_colors,
                density=density_list,
                density_colors=density_colors,
                morph_tags=morph_tags,
                morph_colors=morph_colors,
                morph_samples=morph_samples,
                font=_artist_font,
                label_alpha=resolved_label_alpha,
                xlabel=xlabel,
                ylabel=ylabel,
                zlabel=zlabel,
                frame_hooks=_frame_hooks,
                ownership=_ownership,
            )

            # predict=: overlay one forecast trace per input dataset
            # (GH #169), styled to match its source line -- same colour,
            # linestyle and linewidth, at half its alpha
            # (`_forecast_style_from`). Added AFTER `_draw` has built the
            # legend
            # (from the original data lines only, via ax.legend() inside
            # `_draw`), so these traces never gain a legend entry. The SAME
            # helper (and the SAME seam-prepended arrays) serve both the
            # static path and the animate='spin' path below.
            # The STATIC full-history overlay belongs only to modes that do
            # not reveal data over time: a static plot, or animate='spin'
            # (camera-only). Time-progressing modes get the per-frame artist
            # built below instead -- drawing both would put a frozen
            # full-history forecast on screen from frame 0.
            if raw_forecasts is not None and animate in (False, None, 'spin'):
                _forecast_artists = _draw_forecast_overlays(
                    ax, raw_forecasts, antialias=antialias,
                    owner=_forecast_owner, overrides=_forecast_overrides)
                # animate='spin' only rotates the camera around the fully-
                # drawn static scene, so these overlays rotate with everything
                # else once they exist -- no per-frame update needed. But they
                # must be unclipped, exactly like the other 3-D line artists
                # (see animate_plot3D's `set_clip_on(False)` block), so they
                # aren't clipped at wide rotation angles.
                if animate == 'spin':
                    for _artist in _forecast_artists:
                        _artist.set_clip_on(False)

            # ...and the time-progressing modes get one LIVE artist per
            # dataset instead, refilled every frame from the precomputed
            # schedule. Created EMPTY: frame 0 may legitimately have no
            # forecast for a dataset (too little history revealed), and
            # emptiness -- not alpha -- is how "nothing to draw" is said.
            if forecast_schedule is not None:
                # Colours come from the trajectory lines as they stand
                # BEFORE any forecast artist is added: `ax.lines` grows as
                # this loop runs, so snapshot it first or artist i would take
                # its colour from forecast i-1. (Same guard
                # `_draw_forecast_overlays` opens with.)
                from .forecast import trail_alpha, trail_frames
                _src_lines = list(ax.lines)
                _live_forecast_artists = []
                # [dataset][age-1] -> artist. Preallocated: allocating
                # artists mid-animation is what makes matplotlib animations
                # stutter. Every slot starts HIDDEN WITH EMPTY DATA --
                # emptiness, not alpha, is the "not yet written" signal,
                # because `trail_alpha` never returns 0 and a stale artist
                # would otherwise be indistinguishable from an empty one.
                _trail_forecast_artists = []
                # one per DATASET, not per drawn trace: `hue=`/`cluster=`
                # regrouping makes those different counts (a 30-row dataset
                # under A/B/A is three traces), and a forecast belongs to the
                # dataset. Without regrouping the two are equal and nothing
                # changes.
                for _i in range(len(raw_forecasts)):
                    # which drawn run this dataset's forecast continues: the
                    # run holding its LAST observation, the same trace the
                    # static overlay inherits from. Per FRAME the live artist
                    # re-takes the colour of whichever run is drawing the head
                    # (Decision R3); this is its build-time default.
                    _fc_src = (_forecast_owner[_i]
                               if _forecast_owner is not None
                               and _i < len(_forecast_owner) else _i)
                    # the SAME styling policy `_draw_forecast_overlays` uses
                    # (colour/linestyle/linewidth inherited from the observed
                    # trace, alpha halved), from the SAME helper -- so a
                    # paused animation is indistinguishable from the static
                    # plot and the two paths cannot drift.
                    # Under a CONTINUOUS hue the observed trace has many
                    # colours and `_src_lines[_fc_src]` is the HIDDEN
                    # single-colour artist that drives the reveal -- its
                    # palette colour is drawn nowhere. Anchor on the final
                    # observed hue colour instead, exactly as the static
                    # overlay does (`_apply_multicolor_lines`). Before this,
                    # a paused animation and a static plot of the same call
                    # showed the forecast in different colours, and the
                    # animated one continued a colour its trajectory never
                    # visibly had.
                    _fc_anchor = None
                    if (line_colors is not None and _fc_src < len(line_colors)
                            and len(line_colors[_fc_src])):
                        _fc_anchor = tuple(line_colors[_fc_src][-1])
                    _fc_style = _forecast_style_from(
                        _src_lines[_fc_src] if _fc_src < len(_src_lines)
                        else None,
                        override=(_forecast_overrides[_i]
                                  if _forecast_overrides is not None
                                  and _i < len(_forecast_overrides)
                                  else None),
                        anchor_color=_fc_anchor)
                    # trails FIRST, so the live forecast draws on top of its
                    # own fan rather than under it
                    _row = []
                    for _age in range(1, _n_forecast_trail + 1):
                        if _display_ndims >= 3:
                            _t, = ax.plot([], [], [], label='_nolegend_',
                                          **_fc_style)
                        elif _display_ndims == 2:
                            _t, = ax.plot([], [], label='_nolegend_',
                                          **_fc_style)
                        else:
                            _t, = ax.plot([], label='_nolegend_', **_fc_style)
                        # the fan decays from THIS dataset's live forecast
                        # alpha, not from a fixed one: a trail must never be
                        # more opaque than the live forecast it fades from.
                        _t.set_alpha(trail_alpha(
                            _age, _n_forecast_trail,
                            live_alpha=_fc_style['alpha']))
                        _t.set_clip_on(False)
                        _t.set_visible(False)
                        _t._hyp_forecast_role = 'trail'
                        _t._hyp_forecast_age = _age
                        # identity, like the static overlay and plotly's
                        # meta['hyp_dataset']: the animated artists ARE
                        # per-dataset, so every forecast artist on the axes
                        # answers "which series?" the same way.
                        _t._hyp_forecast_dataset = _i
                        _row.append(_t)
                    _trail_forecast_artists.append(_row)
                    # the SAME three-way split and label
                    # `_draw_forecast_overlays` uses. 1-D is a real branch:
                    # `_display_ndims` can be 1.
                    if _display_ndims >= 3:
                        _art, = ax.plot([], [], [], label='_nolegend_',
                                        **_fc_style)
                    elif _display_ndims == 2:
                        _art, = ax.plot([], [], label='_nolegend_',
                                        **_fc_style)
                    else:
                        _art, = ax.plot([], label='_nolegend_', **_fc_style)
                    _art.set_clip_on(False)
                    _art._hyp_forecast_role = 'live'
                    _art._hyp_forecast_dataset = _i
                    _live_forecast_artists.append(_art)

                # whether the user pinned this dataset's forecast colour
                # (`forecast_hue=`/`forecast_cluster=`/`forecast_palette=`);
                # an explicit grouping is resolved once from the full-history
                # forecasts and must stay fixed for every frame, so it wins
                # over the per-frame head-run colour below.
                # A CONTINUOUS hue pins it too, for the same reason: the
                # forecast's identity is the hue value where its trajectory
                # ends, which does not change frame to frame. Decision R3's
                # per-frame head-run colour remains correct for CATEGORICAL
                # regrouping, where the run colour is what the viewer sees.
                # `line_colors is not None` is DEFENSIVE, not load-bearing:
                # measured 2026-08-16, a continuous hue never regroups, so
                # `_reveal` above is None and `_run_colour` already returns
                # None on every frame. What actually gives an animated
                # continuous-hue forecast the anchor colour is the
                # `anchor_color=` at its build site. The clause states the
                # policy where the policy is decided, so that widening
                # `_reveal` to non-regrouped animations cannot silently
                # reintroduce the palette repaint; the invariant it rests on
                # is pinned by `test_a_continuous_hue_animation_is_NEVER_
                # regrouped_into_runs`.
                _override_colour = [
                    bool((_forecast_overrides is not None
                          and _i < len(_forecast_overrides)
                          and isinstance(_forecast_overrides[_i], dict)
                          and _forecast_overrides[_i].get('color') is not None)
                         or line_colors is not None)
                    for _i in range(len(raw_forecasts))]

                def _update_forecasts(ctx, _sched=forecast_schedule,
                                      _artists=_live_forecast_artists,
                                      _trails=_trail_forecast_artists,
                                      _retained=_n_forecast_trail,
                                      _antialias=antialias,
                                      _ndims=_display_ndims,
                                      _reveal_sched=_reveal,
                                      _lines=_src_lines,
                                      _pinned=_override_colour):
                    def _run_colour(dataset, frame):
                        """Decision R3: the colour of the run DRAWING the
                        head at `frame` -- the CURRENT frame for the live
                        forecast, and the frame it was FIT at for a retained
                        one, so crossing a category boundary does not repaint
                        the whole historical fan (which would make a saved
                        animation differ from a played one)."""
                        if _reveal_sched is None or _pinned[dataset]:
                            return None
                        run = _reveal_sched.head_run(dataset, frame)
                        if run is None or run >= len(_lines):
                            return None
                        return _lines[run].get_color()

                    def _blank(art):
                        art.set_visible(False)
                        if _ndims >= 3:
                            art.set_data_3d([], [], [])
                        else:
                            art.set_data([], [])

                    def _fill(art, pts):
                        if _antialias:
                            # documented parity with the static overlay
                            pts = _interp_static_line(pts)
                        art.set_visible(True)
                        # the SAME three-way split `_draw_forecast_overlays`
                        # uses. A 3-D forecast artist is a Line3D: set_data
                        # alone would leave its z-data at whatever it held
                        # last, drawing in the wrong place instead of failing.
                        if _ndims >= 3:
                            art.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
                        elif _ndims == 2:
                            art.set_data(pts[:, 0], pts[:, 1])
                        else:
                            art.set_data(np.arange(len(pts)), pts[:, 0])

                    # the fan is a PURE function of ctx.frame -- recomputed,
                    # never accumulated. FuncAnimation replays from frame 0
                    # for save()/to_jshtml() and may deliver frames out of
                    # order; a ring buffer would make a saved GIF differ from
                    # an interactively-played animation.
                    past = trail_frames(ctx.frame, _retained) if _retained \
                        else []
                    for i, art in enumerate(_artists):
                        pts = _sched.polyline(i, ctx.frame)
                        if pts is None or len(pts) < 2:
                            _blank(art)
                        else:
                            _colour = _run_colour(i, ctx.frame)
                            if _colour is not None:
                                art.set_color(_colour)
                            _fill(art, pts)
                        for _age, slot in enumerate(_trails[i], start=1):
                            # `past` is newest-first, so age N is past[N-1];
                            # ages beyond the frames available are blanked
                            # rather than left showing a stale fan
                            if _age > len(past):
                                _blank(slot)
                                continue
                            old_pts = _sched.polyline(i, past[_age - 1])
                            if old_pts is None or len(old_pts) < 2:
                                _blank(slot)
                            else:
                                # the head run AT THE FRAME THIS WAS FIT
                                _colour = _run_colour(i, past[_age - 1])
                                if _colour is not None:
                                    slot.set_color(_colour)
                                _fill(slot, old_pts)

                # INTERNAL phase: library updaters run before user callbacks,
                # so an on_frame= callback observes this frame's completed
                # forecast geometry rather than the previous frame's.
                _frame_hooks.add_internal(_update_forecasts)

            # exact per-point colors: swap the single-color artists for
            # per-segment-colored line collections or per-point-colored
            # scatter (the cube/square frame and axes from _draw are kept)
            if line_colors is not None:
                if (line_ani is not None and animate == 'morph'):
                    # morph draws its own single traveling artist; the
                    # static swap below would REMOVE it (and there is no
                    # per-point correspondence to color) -- warn instead
                    # of silently destroying the animation (F04-001
                    # follow-up)
                    warnings.warn(
                        "per-point (continuous/matrix) hue coloring is "
                        "not supported for animate='morph'; drawing the "
                        "morph with its default colors.",
                        UserWarning,
                        stacklevel=external_stacklevel())
                elif (line_ani is not None and animate != 'spin'
                        and has_line_component(fmt)):
                    # animated reveal styles (parallel/window/serial):
                    # per-frame multicolor rendering (F04-001/F05-002 --
                    # the static swap froze the animation; 'spin' keeps
                    # the static swap below, which is exactly right for a
                    # camera-only animation). Marker+line combos animate
                    # as a single line artist (see _draw's raw_data note),
                    # so they take this path too.
                    _apply_multicolor_animation(
                        ax, xform, line_colors, kwargs_list, line_ani,
                        style=animate, chemtrails=chemtrails,
                        precog=precog, bullettime=bullettime,
                        antialias=antialias,
                        total_frames=max(1, int(round(frame_rate
                                                      * duration))))
                elif is_line(fmt):
                    _apply_multicolor_lines(ax, xform, line_colors,
                                            kwargs_list)
                elif has_line_component(fmt):
                    # marker+line combo fmt (e.g. 'o-') with continuous/
                    # matrix hue (GH #141 x F02-004): keep BOTH components
                    # -- a multicolored smoothed connecting line PLUS
                    # per-point-colored markers at the TRUE (pre-
                    # interpolation) sample points, mirroring the no-hue
                    # combo rendering. Previously the line was silently
                    # dropped and a marker was scattered at every
                    # interpolated point (~45x more "data points" than
                    # exist).
                    _apply_multicolor_lines(ax, xform, line_colors,
                                            kwargs_list)
                    _marker_colors = _multicolor_line_colors(
                        multicolor_hue, pre_interp_lengths, raw_xform,
                        palette, is_rgb=multicolor_hue_is_rgb)
                    _apply_multicolor_markers(ax, raw_xform, _marker_colors,
                                              kwargs_list, fmt=fmt)
                else:
                    _apply_multicolor_markers(ax, xform, line_colors,
                                              kwargs_list, fmt=fmt)

            # on_frame= (plan 1.1 Task 7): the hook dispatcher goes on LAST,
            # so callbacks observe the FINAL artists -- _apply_multicolor_
            # animation (above) wraps line_ani._func itself and would
            # otherwise run after this dispatcher, handing hooks pre-
            # multicolor collections. Every animated updater in
            # matplotlib_backend already called `_frame_hooks.record(...)`
            # (a cheap no-op when there are no callbacks); this wrapper is
            # the ONE place that turns recorded state into an actual call.
            if line_ani is not None:
                _orig_frame_func = line_ani._func

                def _hyp_frame_with_hooks(num, *fargs,
                                          _orig=_orig_frame_func):
                    result = _orig(num, *fargs)
                    # updaters stash per-frame state as attributes on
                    # THEMSELVES (e.g. `update_morph.planes`, the just-drawn
                    # cube wireframe -- see matplotlib_backend's own
                    # `hasattr(update_lines_serial, "planes")` reuse-check),
                    # and a pre-existing test (test_morph_animation.py's
                    # TestBoxContainmentUnionHull) reads it back afterward via
                    # `ani._func.planes`. Since this wrapper -- not the
                    # original updater -- is now `line_ani._func`, mirror the
                    # updater's __dict__ onto the wrapper after every call so
                    # that access keeps working unchanged (measured: without
                    # this, `ani._func.planes` raised AttributeError, because
                    # a plain function wrapper carries no attributes of its
                    # own).
                    _hyp_frame_with_hooks.__dict__.update(_orig.__dict__)
                    _frame_hooks.dispatch(fig, ax)
                    return result

                line_ani._func = _hyp_frame_with_hooks

            # title= per-segment sequence (plan 1.1 Task 8): registered on
            # the SAME `_frame_hooks` registry as `on_frame=` (added last
            # above, so this callback also only ever sees final artists --
            # though it touches the title, not the artists, so ordering
            # relative to `on_frame=`'s own callbacks does not matter).
            if _segment_titles is not None and line_ani is not None:
                _frame_hooks.add(_make_title_updater(_segment_titles, ax))

            # animated 3-D titles need a reserved top margin or they render
            # entirely off-canvas -- animate_plot3D's full-canvas axes
            # leave zero room above the axes box, which IS the figure's own
            # top edge there (see _reserve_animated_3d_title_margin's
            # docstring for the full root-cause evidence). Gated on
            # "will a title actually be drawn" (scalar OR per-segment, same
            # condition the plotly fix uses) so a titleless 3-D animation
            # keeps the exact same maximised canvas as before -- and on
            # ndims >= 3 so 2-D animations (whose default, non-maximised
            # axes already leave normal title room) are never touched.
            if (ax is not None and line_ani is not None
                    and xform[0].shape[1] >= 3
                    and (title is not None or _segment_titles is not None)):
                _reserve_animated_3d_title_margin(fig, ax)

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
                _add_colorbar(fig, ax, colorbar_info, font=_artist_font)

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

            # save. `fig.savefig`, NOT `plt.savefig` (release-1.0 audit,
            # F09-001: `plt.savefig` writes pyplot's CURRENT figure, so
            # with a user-supplied ax= whose figure was not current the
            # exported file silently contained the WRONG figure). The
            # except-branch keeps a failing save (bad extension,
            # permissions, ...) from leaking the already-drawn figure into
            # pyplot's manager when the caller asked for show=False
            # (F09-006) -- the cleanup mirrors the normal-path close below.
            if save_path is not None:
                try:
                    if animate:
                        _save_animation(line_ani, save_path, frame_rate)
                    else:
                        fig.savefig(save_path)
                except Exception:
                    if (not show and not _user_supplied_ax
                            and isinstance(fig, plt.Figure)):
                        plt.close(fig)
                    # the exception propagates before the HyperAnimation
                    # wrapper is ever constructed, so its __del__ silencing
                    # (X4-warnings-012) can never run for this abandoned,
                    # never-rendered FuncAnimation -- without this it warned
                    # "Animation was deleted without rendering anything" at
                    # the next cyclic-gc pass, misattributed to whatever code
                    # ran later (release-1.0 audit, zero-warnings sweep).
                    if animate and line_ani is not None:
                        from .hyper_animation import mark_draw_started
                        mark_draw_started(line_ani)
                    raise

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
    # Skip when the user supplied their own `ax` (their figure to manage) and
    # skip plotly figures (not pyplot-managed). ANIMATED figures are closed
    # too (release-1.0 audit, F09-003: every animated show=False call leaked
    # one registered pyplot figure, growing without bound in batch-export
    # loops); the returned HyperAnimation stays fully usable afterward --
    # `.save()`/`.to_jshtml()` drive their frames explicitly on the existing
    # canvas, GUI-backed or not (verified on FigureCanvasMac; covered by
    # tests/test_plot_save_audit_fixes.py). Closing nulls the FuncAnimation's
    # event source but leaves its pending FIRST-DRAW hook connected, and a
    # later draw of the returned figure would fire it and dump a spurious
    # "'NoneType' object has no attribute 'add_callback'" traceback (the
    # historical GUI-backend crash this branch used to dodge by never
    # closing animated figures at all) -- so disconnect that hook explicitly;
    # Animation.save()/to_jshtml() never need it.
    if (not show and not _user_supplied_ax and isinstance(fig, plt.Figure)):
        # matplotlib >= 3.11: plt.close() DETACHES the figure's real canvas,
        # swapping in a bare FigureCanvasBase (draw() is a no-op and there is
        # no buffer_rgba), so the returned figure could no longer render or
        # re-save. Re-attach the original canvas after closing -- restoring
        # matplotlib <= 3.10's close semantics (canvas kept, figure
        # deregistered from pyplot), which is exactly the contract documented
        # above: the returned Figure stays valid, savable, and renderable.
        _live_canvas = fig.canvas
        plt.close(fig)
        if fig.canvas is not _live_canvas:
            fig.set_canvas(_live_canvas)
        if line_ani is not None:
            _first_draw_id = getattr(line_ani, '_first_draw_id', None)
            if _first_draw_id is not None:
                try:
                    fig.canvas.mpl_disconnect(_first_draw_id)
                except Exception:
                    pass
                line_ani._first_draw_id = None

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
        #
        # A COLUMN hierarchy is fit on the frame's GROUPS, not on the frame
        # -- `raw` is one dataset per group, each as wide as one group --
        # so record that, or re-applying the bundled pipeline to the very
        # frame that produced it fails inside scikit-learn ("X has 20
        # features, but IncrementalPCA is expecting 5") and, when the
        # reduce stage was a no-op (every leaf already <= ndims columns),
        # silently returns the UNGROUPED frame. See
        # `Pipeline._regroup_hierarchical_input`. A ROW hierarchy is
        # deliberately not recorded: its leaves keep the full row
        # MultiIndex (they re-expand to themselves), and its pipeline
        # already round-trips -- every leaf has the frame's own width.
        # Computed BEFORE the pipeline= branch because it applies to a
        # caller-supplied pipeline too: that one is handed back in the
        # bundle under the same documented promise, and without the record
        # `bundle['pipeline'].transform(df)` still raised the exact
        # pre-1.1.0 scikit-learn error the `return_model` docstring says it
        # no longer raises (measured: "X has 15 features, but
        # IncrementalPCA is expecting 5 features as input").
        _bundle_hierarchy = None
        if (_multiindex_meta is not None
                and _multiindex_meta.get('axis') == 'columns'
                and raw):
            _bundle_hierarchy = {
                'axis': 'columns',
                'n_features': int(raw[0].shape[1]),
                'feature_correspondence': _multiindex_meta.get(
                    'feature_correspondence', 'name'),
                'feature_labels': _mi_feature_labels,
            }
        if pipeline is not None:
            bundle_pipeline = pipeline
            if (_bundle_hierarchy is not None
                    and pipeline.input_hierarchy is None):
                # Recorded IN PLACE, on the caller's own object, because
                # the bundle hands back that same object by design (the
                # docstring says so, and `test_cross_module_kwargs.py:194`
                # asserts the identity). What is recorded is the grouping
                # the pipeline was just APPLIED under -- which is the only
                # grouping the bundled pipeline could have to reproduce --
                # and its width is not a guess: `analyze(raw,
                # pipeline=pipeline)` above already pushed these same
                # `raw[0].shape[1]`-wide groups through every fitted step.
                # An input_hierarchy the caller's pipeline ALREADY carries
                # belongs to its own fit and is left alone; it cannot
                # disagree about `n_features`, since a mismatch there would
                # have raised in `_regroup_hierarchical_input` during that
                # analyze() call, long before this bundle was built.
                from ..core.pipeline import _validate_input_hierarchy
                pipeline.input_hierarchy = _validate_input_hierarchy(
                    _bundle_hierarchy)
        elif raw is not None:
            from ..core.pipeline import build_pipeline
            # the cluster stage reuses the EXACT resolved spec the
            # figure's own cluster stage was built from (set in the
            # cluster branch above; None when no clustering ran) --
            # previously this path re-resolved the raw cluster= spec with
            # cluster.cluster()'s n_clusters=3 default, so the bundled
            # pipeline could encode a different cluster count/parameters
            # than the published figure (F13-004), and the n_clusters=-
            # only KMeans path was omitted from the pipeline entirely.
            cluster_spec = _bundle_cluster_stage
            # LOW (accepted tradeoff): this refits manip/normalize/reduce/
            # align/cluster a second time on `raw`, duplicating the work
            # already done above to produce `xform_data` for the figure --
            # kept because it is the only way to hand back a genuinely
            # fit-once-reusable `Pipeline` object (see the `pipeline=`
            # discussion above) without threading a Pipeline out of every
            # internal code path that can produce `xform_data`.
            # `_bundle_hierarchy` (the COLUMN-hierarchy record) is computed
            # above the pipeline= branch, since it applies to both.
            bundle_pipeline = build_pipeline(manip=manip, normalize=normalize,
                                             reduce=reduce, ndims=ndims,
                                             align=align, cluster=cluster_spec,
                                             input_hierarchy=_bundle_hierarchy)
            # this refit re-resolves the SAME reduce spec the figure was
            # drawn with, so any spec-conflict warning it emits (e.g.
            # "Unequal values passed to dims and n_components" when a
            # pre-configured reduce instance's n_components differs from
            # ndims) was ALREADY issued once by the analyze() call above --
            # suppress the duplicate (release-1.0 audit, R1).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='Unequal values passed to dims and '
                            'n_components',
                    category=UserWarning)
                bundle_pipeline.fit_transform(raw)
        else:
            bundle_pipeline = None
        # the bundle hands back the RAW FuncAnimation (never a
        # HyperAnimation), so the X4-warnings-012 __del__ silencing never
        # applies to it; without this, discarding the bundle leaked
        # matplotlib's "Animation was deleted without rendering anything"
        # UserWarning at the next cyclic-gc pass (release-1.0 audit,
        # zero-warnings sweep; see hyper_animation.mark_draw_started).
        if line_ani is not None:
            from .hyper_animation import mark_draw_started
            mark_draw_started(line_ani)
        return {
            "fig": fig,
            "xform_data": xform_data,
            "trace_data": trace_data,
            "trace_metadata": trace_metadata,
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
                # whether those forecasts reached the FIGURE. They are
                # reported either way -- `return_model=` hands back model
                # output, and a successful fit that a rendering combination
                # cannot display is still a successful fit.
                "drawn": _forecast_draw_reason is None,
                "draw_reason": _forecast_draw_reason,
            },
        }

    # only animated matplotlib plots set line_ani; plotly and static plots
    # leave it None. An animated plot returns a HyperAnimation (QC 2026-07): a
    # single object exposing .to_html5_video()/.to_jshtml()/.save()/.figure that
    # auto-plays inline in a notebook -- so `anim = hyp.plot(data, animate=...)`
    # then `anim.to_html5_video()` works (it used to fail on the bare tuple).
    # HyperAnimation still unpacks as the legacy (figure, animation) tuple, so
    # `fig, anim = hyp.plot(...)` keeps working too.
    if line_ani is not None:
        from .hyper_animation import HyperAnimation
        return HyperAnimation(fig, line_ani, frame_hooks=_frame_hooks)

    return fig


def _build_colorbar_info(colorbar, hue, multicolor_hue, cluster, n_clusters,
                         xform, mpl_kwargs, legend, palette,
                         hue_group_labels=None, hierarchy_labels=None):
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
      else from `hierarchy_labels` -- a MultiIndex hierarchy's per-trace
      labels, which reach here even under `legend=False` -- else from
      `hue_group_labels` -- the categorical hue's category names, known
      whether or not the user ALSO asked for a legend (F02-007) -- else
      ``1..n``).
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

    if isinstance(legend, list):
        labels = list(legend)
    elif (hierarchy_labels is not None
            and len(hierarchy_labels) == n_groups):
        # MultiIndex hierarchy under legend=False: `legend` is None here,
        # but the group names (and the '_nolegend_' entries the filter
        # below needs) are still known -- see `_mi_colorbar_labels`.
        labels = list(hierarchy_labels)
    elif (hue_group_labels is not None
            and len(hue_group_labels) == n_groups):
        # categorical hue: the category names are known even without
        # legend=True -- previously the colorbar fell back to 1..n unless
        # a (redundant) legend was also requested (F02-007)
        labels = list(hue_group_labels)
    else:
        labels = [i + 1 for i in range(n_groups)]

    # A trace labeled '_nolegend_' (e.g. every MultiIndex leaf and
    # intermediate-level mean, GH #95 -- only the TOP-level mean of each
    # group carries a real label) must NEVER appear on the colorbar: filter
    # colors/labels down to the REAL (legend-worthy) entries together, so a
    # 2-level MultiIndex DataFrame (8 leaves + 2 top-level means) renders 2
    # colorbar segments (one per top-level group), not 10.
    if '_nolegend_' in labels:
        keep = [i for i, lbl in enumerate(labels) if lbl != '_nolegend_']
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
                 is not None else [str(lbl) for lbl in default_tick_labels])
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


def _measurement_renderer(fig):
    """A real Agg `renderer` for `fig`, from a real draw of `fig` itself --
    needed by `_tight_right_edge_in`/`_legend_right_edge_in`, which measure
    ACTUAL legend/label extents that only exist once the real content is
    laid out (unlike `_animated_3d_title_line_height_in`, which measures a
    font-metrics-only property and can use a throwaway figure instead).

    Guards the draw exactly like matplotlib's OWN `Animation.save()` and
    `FigureCanvasBase.print_figure()` already guard THEIR internal draws
    (see `Animation.save`'s ``cbook._setattr_cm(self._fig.canvas,
    _is_saving=True, ...)`` and `print_figure`'s identical guard, both in
    `matplotlib/animation.py`/`matplotlib/backend_bases.py`): setting
    ``canvas._is_saving = True`` for the duration of this draw makes
    `Animation._start` (connected to every animated figure's
    ``'draw_event'`` at `FuncAnimation` construction time -- see
    `Animation.__init__`) a no-op (`if self._fig.canvas.is_saving():
    return`, deliberately WITHOUT disconnecting its own listener), which
    defers the animation's real "first draw" start to the next NON-
    measurement draw instead of firing it here.

    Without this guard (release QC for the patch line that shipped as
    1.1.0, found while verifying the 3-D title-margin fix's
    "neighbours"): a measurement draw here is often
    `fig`'s first draw ever, since `hyp.plot(..., show=False)` never draws
    the canvas itself -- and an unguarded first draw silently ran
    `FuncAnimation._init_draw()` -> a real, premature frame-0 update
    through `line_ani._func`, dispatching any `on_frame=` callback (and any
    `_frame_hooks`-driven per-segment `title=` schedule) one extra time,
    before the animation had otherwise started and before `plot()` even
    returned -- confirmed empirically (an animated 3-D plot with
    `legend=`/`colorbar=` plus `on_frame=` recorded one spurious call
    during construction, before any real frame was ever driven) and locked
    in by `tests/test_animation_margins.py::
    TestMeasurementDrawsDoNotStartTheAnimation`.

    `FigureCanvasAgg(fig)` (like every `FigureCanvasBase` subclass) rebinds
    `fig.canvas` to itself (`Figure.set_canvas`), so the guard applies to
    the canvas THIS CALL just created -- the same object `Figure.draw()`
    reads back via `self.canvas` when it fires `'draw_event'` at the end
    of the draw (`Figure.draw`, matplotlib source), not stale state left on
    whatever canvas existed before.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas._is_saving = True
    try:
        canvas.draw()
    finally:
        canvas._is_saving = False
    return canvas.get_renderer()


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
    with plt.rc_context(matplotlib.rcParamsDefault):
        renderer = _measurement_renderer(fig)
        return float(fig.get_tightbbox(renderer).x1)


def _legend_right_edge_in(fig, legend):
    """The legend's right edge (inches from the figure's left edge),
    measured under the restored default rcParams exactly like
    `_tight_right_edge_in` (see its docstring for why rcParamsDefault) --
    but for the LEGEND artist alone, so unclipped animated data/trail
    artists whose projected extent overshoots the canvas cannot inflate
    the legend fit (release-1.0 audit, F04-002)."""
    import matplotlib
    with plt.rc_context(matplotlib.rcParamsDefault):
        renderer = _measurement_renderer(fig)
        return float(legend.get_window_extent(renderer).x1) / fig.dpi


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


def _animated_3d_title_line_height_in(ax, probe='Xygj'):
    """The true rendered height (inches) of one line of `ax.title`'s
    resolved font.

    Measured on a THROWAWAY `Figure`/`Axes` -- never `ax`'s own real
    figure. The first attempt at this measured directly against the real
    `fig` (`FigureCanvasAgg(fig)` + `canvas.draw()`), which is wrong in a
    way that only shows up on an ANIMATED figure: `FuncAnimation.__init__`
    connects `fig.canvas.mpl_connect('draw_event', self._start)` so the
    animation only "starts" once the figure has genuinely been drawn for
    the first time (see `matplotlib.animation.Animation.__init__`) --
    `hyp.plot(..., show=False)` never draws the canvas itself, so that
    probe draw WAS the figure's first draw, firing `draw_event` ->
    `Animation._start()` -> `FuncAnimation._init_draw()` ->
    `self._draw_frame(next(self.new_frame_seq()))` -- i.e. a REAL,
    premature frame-0 update through `line_ani._func` (the
    `_hyp_frame_with_hooks` wrapper below), dispatching any `on_frame=`
    callback once, BEFORE the animation has otherwise started and before
    `plot()` even returns. Confirmed empirically (and locked in by
    `tests/test_animation_margins.py::
    TestMeasurementDrawsDoNotStartTheAnimation::
    test_reserving_the_title_margin_does_not_fire_on_frame`): this made
    `on_frame=` see one extra call, and made
    `_frame_hooks`-driven state (e.g. a per-segment `title=` list's
    schedule) drift out of sync with a caller's own frame count -- a real
    regression, not a measurement nicety. `figure.canvas`/its callback
    registry live on the FIGURE (`Figure._canvas_callbacks`), not on any
    one canvas object, so even swapping `fig.canvas` for a new one does
    not sidestep this -- a throwaway `Figure()` that `line_ani` never
    connected anything to is the only clean way to measure.

    Measured under `matplotlib.rcParamsDefault`, not whatever rcParams are
    active right now -- same reasoning as `_tight_right_edge_in` (see its
    docstring): hypertools's own font resolution (`resolve_font`, called
    from `plot()`) only ever sets a FontProperties `family=`, never a
    `size=`, so the title's actual rendered point size always comes from
    rcParams (`axes.titlesize`), resolved dynamically, and hypertools draws
    inside a seaborn/font `rc_context` that has already been exited by the
    time the figure is actually saved or displayed (confirmed there to be a
    real, not theoretical, discrepancy) -- and, for a per-segment `title=`
    list specifically, `_make_title_updater` sets the title with NO
    `fontproperties=` override at all (see its call site), so it is never
    anything BUT rcParams-driven. `ax.title`'s OWN resolved
    `FontProperties` (family only, per above) is copied onto the probe so
    the measurement still reflects whatever family the real title would
    use.

    `probe` (default ``'Xygj'``) replaces the probe title's text: a capital
    ascender, an x-height glyph, and two descenders, so the measured height
    is a safe upper bound regardless of the REAL title text's own
    ascender/descender mix -- this must work before any real per-frame
    title text exists at all (a per-segment list's first entry is only
    ever set once the animation actually plays; see `_make_title_updater`),
    and, even for an already-known scalar title, a representative probe
    avoids reserving a margin sized to that one string when OTHER
    per-segment strings sharing the same reserved margin might need
    slightly more.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    probe_fig = Figure()
    probe_ax = probe_fig.add_subplot()
    probe_ax.set_title(probe, fontproperties=ax.title.get_fontproperties())
    with plt.rc_context(matplotlib.rcParamsDefault):
        canvas = FigureCanvasAgg(probe_fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        height_px = probe_ax.title.get_window_extent(renderer).height
    return height_px / probe_fig.dpi


def _reserve_animated_3d_title_margin(fig, ax, pad_in=0.08):
    """Grow the figure so an animated 3-D plot's title has room to render.

    `matplotlib_backend.animate_plot3D` deliberately maximises the 3-D axes
    to ``ax.set_position([0, 0, 1, 1])`` (full canvas -- see its own
    comment) so a rotating zoomed cube never overflows the axes viewport
    and clips at some rotation angles. `Axes.set_title()` draws ABOVE the
    axes' own bounding box; with zero margin left above a full-canvas axes,
    that box top IS the figure's own top edge, so the title Text lands
    entirely off-canvas and never renders -- confirmed empirically
    (`tests/test_animation_margins.py`'s `TestAnimated3DTitleMargin`):
    `ax.title.get_window_extent().y0` sits past the canvas height, and a
    before/after pixel diff between a real title and an empty one shows
    ZERO changed pixels. This is true for BOTH a scalar `title=` (set once,
    statically, by the shared "add title" block in
    `matplotlib_backend._draw`) and a per-segment `title=` list (set every
    frame by `_make_title_updater` below) -- neither path ever reserved any
    margin, so both are reached the same way; the caller here gates on
    "will EITHER ever draw a non-empty title" (`title is not None or
    _segment_titles is not None`), mirroring the plotly-side fix
    (`plotly_backend.py`'s ``t=40 if (title or segment_titles) else 10``,
    `ccbb28c3`) exactly.

    Fix: grow the FIGURE height by the real measured title-line height
    (`_animated_3d_title_line_height_in`) plus `pad_in`, and reposition the
    already-created, already-populated axes to occupy exactly the same
    ABSOLUTE size it already had -- not a smaller, re-shrunk one -- at the
    BOTTOM of the now-taller canvas, leaving a blank strip above it for the
    title. This mirrors `_fit_right_legend`/`_add_right_colorbar` (same
    file), which already grow the canvas rather than shrink the maximised
    axes to make room for a legend/right-side colorbar, for the identical
    reason (see their docstrings).

    Why grow rather than shrink the existing full-canvas axes in place (the
    more "obvious" fix): `mpl_toolkits.mplot3d.axes3d.Axes3D.apply_aspect`
    derives its centered-square viewport from `ax.get_position(original=
    True)` and the FIGURE's own aspect ratio (`fig_aspect`). Shrinking the
    axes position's height fraction in place (e.g. `set_position([0, 0, 1,
    1 - m])`) shrinks the derived square -- and therefore the rendered cube
    -- by the same factor in both dimensions; that is provably proportional
    and safe IN ISOLATION (re-derived by hand against
    `Bbox.shrunk_to_aspect`), but it stacks with an already near-zero
    worst-case margin on wide/flat data at some rotation angles (the exact
    scenario `tests/test_animation_margins.py::TestAxesBoxNoClipping`
    stress-tests), risking reintroducing the very clipping bug the
    full-canvas positioning exists to prevent. Growing the figure instead
    and repositioning the axes to the SAME absolute inches keeps the cube's
    rendered geometry byte-for-byte identical to the title-less baseline at
    every rotation angle (verified: `TestAnimated3DTitleMargin` reruns
    `TestAxesBoxNoClipping`'s own wide/flat + chemtrails worst case, now
    WITH a title, and gets the same healthy margins) -- it cannot
    reintroduce that clipping bug, because nothing about the cube's own
    viewport changes at all.
    """
    w_in, h_in = fig.get_size_inches()
    pos = ax.get_position()
    bottom_in, height_in = pos.y0 * h_in, pos.height * h_in
    new_h_in = h_in + _animated_3d_title_line_height_in(ax) + pad_in
    # a persistent layout engine would silently undo the manual
    # set_position below on the next draw/save (see _fit_right_legend's
    # identical guard, same reasoning).
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    fig.set_size_inches(w_in, new_h_in)
    ax.set_position([pos.x0, bottom_in / new_h_in,
                     pos.width, height_in / new_h_in])


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
        # measure the LEGEND's own right edge, not the full-figure
        # tightbbox (release-1.0 audit, F04-002): animated data/trail
        # artists are deliberately UNCLIPPED (`set_clip_on(False)`, see
        # animate_plot3D's axes-box slicing fix), and a precog/bullettime
        # trail holds (nearly) the whole trajectory from the first frame --
        # its projected extent can reach far past the canvas, so a
        # tightbbox-driven fit ran away to the 3x cap and exported every
        # frame with the plot squashed into the left third. Only the legend
        # needs room here; for content that genuinely fits the canvas the
        # two measurements agree exactly.
        new_w = min(_legend_right_edge_in(fig, legend) + pad_in, 3.0 * w)
        if new_w <= w + 1e-3:
            return  # legend already fits
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


def _multicolor_line_colors(hue_src, orig_lengths, xform, palette, is_rgb=False):
    """Per-point RGB colors for multicolored lines.

    hue_src holds one value (or one row) per ORIGINAL observation; the
    trajectories in xform have since been interpolated to a higher temporal
    resolution, so each dataset's hue values are linearly re-interpolated to
    its new length before color mapping. Colors are mapped over the
    CONCATENATED hue values so the scale is shared across datasets.

    When `is_rgb` is True, `hue_src` already holds literal per-point RGB values
    (e.g. a matrix hue reduced to 3 columns via `plot`'s `color_reduce=`), so
    the re-interpolated values are used AS colors instead of being mapped
    through `mat2colors`.

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
    if is_rgb:
        colors = np.clip(stacked, 0.0, 1.0)
    else:
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

    # Remove the DATA lines this function replaces -- not every line on the
    # axes. `predict=` overlays are already drawn by now, and clearing them
    # here is why a continuous `hue=` silently produced no forecast: the
    # artists were created and then destroyed a few lines later. They are
    # tagged precisely so they can be told apart.
    _kept_forecasts = []
    for line in list(ax.lines):
        if getattr(line, '_hyp_forecast_role', None) is None:
            line.remove()
        else:
            _kept_forecasts.append(line)

    # A continuous hue gives the observed data MANY colours, so "the same
    # colour as its trace" resolves to the colour where the forecast starts:
    # the last segment of the trace it continues.
    #
    # Paired by the artist's DATASET tag, not by its position in
    # `_kept_forecasts`: position is only the right key while forecasts stay
    # one-per-dataset in dataset order, which anything that reorders or
    # filters them (`forecast_cluster=`, a per-dataset refusal) breaks
    # silently -- each forecast would simply take a neighbour's colour.
    for _fi, _fc_line in enumerate(_kept_forecasts):
        _ds = getattr(_fc_line, '_hyp_forecast_dataset', _fi)
        if _ds is not None and _ds < len(line_colors) and len(line_colors[_ds]):
            _anchor_color = line_colors[_ds][-1]
            _fc_line.set_color(_anchor_color)

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
        # Per-trace opacity has to travel WITH the colours: these
        # collections replace the line artists, so an `alpha` left on the
        # discarded `Line2D` is simply lost. That is how a hierarchy's
        # level-derived alphas -- and a plain `hue=` + `alpha=` -- silently
        # rendered fully opaque before 1.1.
        _alpha = tkwargs.get('alpha')
        if _alpha is not None:
            seg_colors = np.column_stack(
                [seg_colors[:, :3],
                 np.full(len(seg_colors), float(_alpha))])
        if is_3d:
            coll = Line3DCollection(segments, colors=seg_colors,
                                    linewidths=lw)
            ax.add_collection3d(coll)
        else:
            coll = LineCollection(segments, colors=seg_colors,
                                  linewidths=lw)
            ax.add_collection(coll)
        # Tag which TRACE this collection draws. `ax.collections` is not a
        # list of data artists -- the 3-D bounding cube is six
        # `Line3DCollection` wireframe faces (`matplotlib_backend.py`,
        # `_draw_cube`), so "the LineCollections on the axes" counts 6 more
        # than there are traces. Same purpose as the `_hyp_forecast_role`
        # tag on forecast lines.
        coll._hyp_trace_index = i


def _make_title_updater(titles, axes):
    """Set the axes title from the frame context (plan 1.1 Task 8).

    Morph transitions are blanked so only fully-formed clouds are named. The
    discriminator is `segment_kind` (from `morph.frame_to_segment`'s segment
    PARITY), never `current_fraction`: holds and transitions both sweep
    0->1 over their own segment, so a fraction cannot tell them apart. For
    non-morph serial reveals `segment_kind` is always None, so every frame
    falls through to the `current_index` branch below.
    """
    def _update(ctx):
        if ctx.segment_kind == 'transition':
            axes.set_title('')
            return
        idx = ctx.current_index
        if idx is None:
            return
        axes.set_title(titles[min(idx, len(titles) - 1)])
    return _update


def _apply_multicolor_animation(ax, xform, line_colors, kwargs_list,
                                line_ani, style, chemtrails, precog,
                                bullettime, total_frames, antialias=True):
    """Per-frame multicolored (continuous/matrix hue) line rendering for
    ANIMATED matplotlib plots (release-1.0 audit, F04-001/F05-002).

    The static multicolor path (`_apply_multicolor_lines`) swaps the
    single-color line artists for full-trajectory collections -- on an
    animated plot that removed the very artists the FuncAnimation updates
    each frame and drew the ENTIRE multicolored trajectory statically in
    every frame: no reveal, no chemtrails/precog/bullettime, and a 2-D
    animation collapsed to a single-frame gif (the plotly backend animated
    the identical call correctly). Instead: HIDE the single-color artists
    (they keep driving the reveal/window bookkeeping each frame) and add
    one initially-empty per-dataset collection for the head window plus one
    per trail; then wrap the animation's frame callback so that, after each
    original update runs, every collection is re-sliced to exactly the
    index window its hidden artist just moved to -- the multicolor
    rendering animates in lockstep with the no-hue animation.

    Only used for the parallel/'window'/'serial' reveal styles: 'spin'
    draws the full trajectory every frame (the static swap is already
    correct there), and 'morph' draws its own single traveling artist (the
    static swap would have REMOVED it -- callers skip morph entirely).
    """
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    is_3d = xform[0].shape[1] >= 3
    n = len(xform)
    # artist bookkeeping mirrors matplotlib_backend.animate_plot3D/2D: the
    # n head lines are created first, then one trail artist per dataset
    # that wants one (parallel/True AND serial create trails -- serial now
    # composes with the trail flags; 'window' never does), in dataset order.
    head_lines = list(ax.lines[:n])
    wants_trail = [
        style in (True, 'parallel', 'serial')
        and bool(chemtrails[i] or precog[i] or bullettime[i])
        for i in range(n)
    ]
    _trail_artists = list(ax.lines[n:n + sum(wants_trail)])
    trail_lines = {}
    for i in range(n):
        if wants_trail[i] and _trail_artists:
            trail_lines[i] = _trail_artists.pop(0)

    def _linewidth(i):
        # the hidden head artist already carries the caller's linewidth=:
        # `animate_plot3D`/`animate_plot2D` pop it out of kwargs_list ONCE per
        # dataset (matplotlib_backend.py:1602-1606 / :2197-2201, so it cannot
        # also ride along in **kwargs_list[idx] and collide) and pass it
        # explicitly to ax.plot. Reading it back off kwargs_list here found
        # nothing and silently fell through to rcParams['lines.linewidth'],
        # so every animated multicolour collection rendered at 1.5 regardless
        # of what the caller asked for. Reading the artist also guarantees the
        # overlay always matches the artist it replaces.
        if i < len(head_lines):
            return head_lines[i].get_linewidth()
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        return (tkwargs.get('linewidth')
                or plt.rcParams['lines.linewidth'])

    def _points(i):
        xi = xform[i]
        if xi.shape[1] == 1:
            return np.column_stack([np.arange(xi.shape[0]), xi[:, 0]])
        return xi[:, :3] if is_3d else xi[:, :2]

    def _make_collection(i, alpha=None, role='head'):
        if is_3d:
            coll = Line3DCollection([], linewidths=_linewidth(i),
                                    alpha=alpha)
            # autolim=False: the animation already fixed the axes limits
            # (cube_scale), and autoscaling an EMPTY collection crashes
            # inside Axes3D.add_collection3d
            ax.add_collection3d(coll, autolim=False)
        else:
            coll = LineCollection([], linewidths=_linewidth(i), alpha=alpha)
            ax.add_collection(coll)
        coll.set_label('_nolegend_')
        # match the line artists' unclipping (see animate_plot3D's
        # axes-box slicing fix)
        coll.set_clip_on(False)
        # Which TRACE this collection draws -- the same tag the STATIC path
        # sets (`_apply_multicolor_lines`), for the same reason: the 3-D
        # bounding cube is six `Line3DCollection` wireframe faces, so
        # `ax.collections` is not a list of data artists and an untagged
        # figure gives a reader no way to tell them apart. Without it an
        # ANIMATED continuous-hue figure carried zero tagged artists while
        # the static plot of the same call carried one per trace.
        # `_hyp_trace_role` distinguishes the head window from its trail,
        # mirroring `_hyp_forecast_role` on forecast artists.
        coll._hyp_trace_index = i
        coll._hyp_trace_role = role
        return coll

    head_colls = [_make_collection(i) for i in range(n)]
    trail_colls = {i: _make_collection(i, alpha=0.3, role='trail')
                   for i in trail_lines}

    for artist in head_lines + list(trail_lines.values()):
        artist.set_visible(False)

    def _artist_len(artist):
        return (len(artist.get_data_3d()[0]) if is_3d
                else len(artist.get_xdata()))

    def _set_segments(coll, pts, colors):
        if pts.shape[0] < 2:
            coll.set_segments([])
            return
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = np.asarray(colors)
        coll.set_segments(segments)
        coll.set_color((colors[:-1] + colors[1:]) / 2.0)

    # antialias (see `plot`'s `antialias=`): a dense, PCHIP-upsampled copy of
    # each trajectory, with its per-point colors resampled onto the SAME
    # parameterization, so every frame's collection draws a smooth curve over
    # exactly the rows the backend just drew. `step == 1` means "no
    # upsampling" and every slice below degrades to the raw rows.
    def _dense_for(i):
        pts = _points(i)
        ci = np.asarray(line_colors[i], dtype=float)
        if not antialias or pts.shape[0] < 2:
            return pts, ci, 1
        dense, step = antialias_line(pts)
        if step == 1:
            return pts, ci, 1
        grid = np.linspace(0.0, pts.shape[0] - 1.0, dense.shape[0])
        ci_dense = np.column_stack([
            np.interp(grid, np.arange(pts.shape[0]), ci[:, c])
            for c in range(ci.shape[1])])
        return dense, ci_dense, step

    _aa_cache = [_dense_for(i) for i in range(n)]

    def _aa_slice(i, a, b):
        """(points, colors) to DRAW for the original-row window ``[a:b]``."""
        dense, ci_dense, step = _aa_cache[i]
        if step == 1:
            return dense[a:b], ci_dense[a:b]
        if b <= a:
            return dense[0:0], ci_dense[0:0]
        return dense[a * step:(b - 1) * step + 1], \
            ci_dense[a * step:(b - 1) * step + 1]

    orig_func = line_ani._func

    def _multicolor_frame(num, *fargs):
        result = orig_func(num, *fargs)
        for i in range(n):
            pts = _points(i)
            n_pts = pts.shape[0]
            # the backend records the ORIGINAL row window each artist was just
            # drawn over (`_hyp_row_window`); prefer it, since antialiasing
            # decouples an artist's VERTEX count from its row count. The
            # length-based recovery below stays as the fallback for callers
            # that drive `_draw` without it.
            _win = getattr(head_lines[i], '_hyp_row_window', None)
            if _win is not None:
                start, end = _win
                _set_segments(head_colls[i], *_aa_slice(i, start, end))
                trail = trail_lines.get(i)
                if trail is not None:
                    _twin = getattr(trail, '_hyp_row_window', None)
                    if _twin is not None:
                        _set_segments(trail_colls[i], *_aa_slice(i, *_twin))
                    else:
                        trail_colls[i].set_segments([])
                continue
            # the hidden head artist was just set to the exact visible
            # window; recover its [start, end) indices from its length
            # plus the same frame->row mapping the backend used (see
            # hypertools.plot.trails.anim_window_bounds)
            head_len = _artist_len(head_lines[i])
            if style == 'serial':
                # serial: the head artist is the opaque comet-head near the
                # reveal tip (or, with no trail flag, the whole revealed
                # span). Recover its position from the SAME shared reveal
                # schedule the backend used to draw it (see
                # matplotlib_backend.update_lines_serial) -- a call, not a
                # second copy of the formula.
                from .matplotlib_backend import serial_reveal_counts
                _lengths = [_points(j).shape[0] for j in range(n)]
                shown = serial_reveal_counts(_lengths, num,
                                             int(total_frames))[i]
                end = shown
                start = max(0, shown - head_len)
            else:
                end = int(np.ceil((num + 1) * n_pts
                                  / max(1, int(total_frames))))
                end = max(1, min(n_pts, end))
                start = max(0, end - head_len)
            _set_segments(head_colls[i], *_aa_slice(i, start, end))

            trail = trail_lines.get(i)
            if trail is not None:
                trail_len = _artist_len(trail)
                if precog[i] and not (chemtrails[i] or bullettime[i]):
                    ts, te = n_pts - trail_len, n_pts  # anchored at the end
                else:
                    ts, te = 0, trail_len  # chemtrails/bullettime: from 0
                _set_segments(trail_colls[i], *_aa_slice(i, ts, te))
        return result

    line_ani._func = _multicolor_frame


def _expand_labels(labels, old_lengths, new_lengths):
    """Re-map per-point labels onto interpolated trajectories.

    Each original point's label is placed at that point's index in the
    interpolated (longer) trajectory; the interpolated in-between points get
    None (no annotation). When the trajectory was DOWN-sampled instead
    (animation frame grids can have fewer points than samples), each label
    lands on the nearest remaining point. Accepts flat label lists or lists
    nested per dataset; returns a flat list matching sum(new_lengths).
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
            # only REAL labels claim a slot: assigning the (mostly-None)
            # in-between entries too let a later None overwrite an
            # already-placed label whenever several original indices
            # mapped to the same new index (down-sampling), silently
            # dropping the user's labels (release-1.0 audit, F10-001).
            if lab is None:
                continue
            if old_n == 1:
                j = 0
            else:
                j = min(new_n - 1, int(round(i * (new_n - 1) / (old_n - 1))))
            expanded[j] = lab
        out.extend(expanded)
    return out


def _apply_multicolor_markers(ax, xform, point_colors, kwargs_list,
                              fmt=None):
    """Replace single-color marker artists with per-point-colored scatter
    (matplotlib backend). Gives exact per-observation colors -- e.g. mixture
    proportions render as true blends instead of quantized groups. When
    `fmt` is a single format string carrying a marker character (e.g. the
    'o' of 'o-'), that marker glyph is used for the scatter points
    (F02-004); otherwise the default circle is drawn."""
    # as in `_apply_multicolor_lines`: replace the DATA artists, keep the
    # tagged forecast overlays
    _kept_forecasts = []
    for line in list(ax.lines):
        if getattr(line, '_hyp_forecast_role', None) is None:
            line.remove()
        else:
            _kept_forecasts.append(line)

    # ...and, as in `_apply_multicolor_lines`, re-anchor each kept forecast
    # on the colour its own trace ENDS in (F14). This loop used to exist only
    # on the line path, so a MARKER-only fmt drew the forecast in the
    # per-dataset palette colour of an artist that was just removed -- and
    # since plotly anchors unconditionally (`_hue_anchor_color`, applied at
    # every trace's build regardless of fmt), the two backends drew the same
    # forecast in different colours for exactly `fmt='o'` (measured, at every
    # `ndims`: matplotlib rgb(59,82,139) vs plotly rgb(72,38,119) for the
    # first trace of a viridis column hierarchy). Same dataset-tag keying and
    # same reason as the line path: position breaks as soon as anything
    # reorders or filters the forecasts (`forecast_cluster=`, a per-dataset
    # refusal).
    for _fi, _fc_line in enumerate(_kept_forecasts):
        _ds = getattr(_fc_line, '_hyp_forecast_dataset', _fi)
        if (_ds is not None and _ds < len(point_colors)
                and len(point_colors[_ds])):
            _fc_line.set_color(point_colors[_ds][-1])

    marker = None
    if fmt is not None and not isinstance(fmt, (list, tuple, np.ndarray)):
        _, marker = split_marker_line_fmt(fmt)
    marker = marker or 'o'

    is_3d = xform[0].shape[1] >= 3
    for i, (xi, ci) in enumerate(zip(xform, point_colors)):
        tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
        ms = float(tkwargs.get('markersize')
                   or plt.rcParams['lines.markersize'])
        s = ms ** 2  # scatter sizes are areas in points^2
        if xi.shape[1] == 1:
            ax.scatter(np.arange(xi.shape[0]), xi[:, 0], c=ci, s=s,
                       marker=marker)
        elif is_3d:
            ax.scatter(xi[:, 0], xi[:, 1], xi[:, 2], c=ci, s=s,
                       depthshade=False, marker=marker)
        else:
            ax.scatter(xi[:, 0], xi[:, 1], c=ci, s=s, marker=marker)

def _mixture_name(model):
    """Registry name for a cluster-model spec (string or class)."""
    return model if isinstance(model, str) \
        else getattr(model, "__name__", str(model))

