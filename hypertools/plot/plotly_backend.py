#!/usr/bin/env python
"""Optional interactive (plotly) backend for hypertools.

matplotlib remains hypertools' default renderer everywhere. The plotly
backend exists for interactive exploration, primarily in Google Colab and
Kaggle notebooks (where hypertools sees most of its use, static matplotlib
output is limiting, and plotly ships preinstalled).

Backend policy (`backend` argument to hyp.plot):
- 'matplotlib' (default outside Colab/Kaggle): the classic renderer
- 'plotly': interactive renderer
- 'auto': plotly on Colab/Kaggle when plotly is importable; matplotlib
  everywhere else -- conservative so existing users see no change

Visual parity: this module reproduces the matplotlib renderer's signature
aesthetic exactly -- data pre-scaled to [-1, 1] (done upstream in plot()),
black wireframe cube (3D) or square frame (2D), axes fully hidden,
elev=10/azim=-60 camera, 1.5pt lines, 6pt markers, and the same seaborn
palette assignment per trace.
"""

import itertools
import os
import sys
import warnings

import numpy as np

from .meshutil import (blinn_phong_vertex_colors, points_enclosed,
                       vertex_colors_from_points)
from .surface import (
    PLOTLY_IDENTITY_LIGHTING,
    PLOTLY_LIGHTPOSITION,
    SURFACE_DEFAULTS,
    build_mesh_3d,
    build_outline_2d,
    mpl_lighting_kwargs,
    surface_cube_scale,
    view_vector,
)
from .density import (
    DENSITY_DEFAULTS,
    POOLED_COLOR,
    bbox_extent,
    density_alpha_boost,
    fit_kde,
    kde_grid_2d,
    kde_grid_3d,
    resolve_grid,
    resolve_plotly_volume_params,
)
from .trails import broadcast_trail_flag
from . import morph as _morph


VALID_BACKENDS = ('auto', 'matplotlib', 'plotly')

# matplotlib sizes are in points; plotly sizes are in pixels. hypertools'
# matplotlib figures render at dpi=100 (rcParams['figure.dpi'], never
# overridden by this codebase -- see `matplotlib_backend`'s lack of any
# `dpi=` kwarg), and this module deliberately sizes its own canvas at the
# same 100 px/inch (`layout['width'] = size[0] * 100` etc., below), so the
# exact points->pixels factor at that SHARED dpi is `100/72`, not the
# `4/3` (96/72, i.e. CSS/web dpi=96) previously used here (R2 fix --
# verified empirically: matplotlib 'o' markers rendered at dpi=100 and
# measured by pixel bounding-box gave e.g. markersize=100 -> ~139px
# diameter, matching `100 * 100/72 = 138.9` almost exactly; see
# `scripts/measure_marker_parity.py`).
PT_TO_PX = 100.0 / 72.0
DEFAULT_FIGSIZE = (6.4, 4.8)  # matplotlib rcParams['figure.figsize'] inches
DEFAULT_LINEWIDTH_PT = 1.5   # matplotlib rcParams['lines.linewidth']
DEFAULT_MARKERSIZE_PT = 6.0  # matplotlib rcParams['lines.markersize']
# Default CSS font stack for the plotly backend. It PREFERS the same
# Noto-Sans-first ordering as the matplotlib backend, but the two do NOT
# render identically: matplotlib is handed the bundled Noto Sans FILE, whereas
# plotly/kaleido can only be given a family NAME resolved by the rendering
# browser (Chrome/kaleido, Jupyter, ...). So "Noto Sans" here is used only if
# it also happens to be installed on that machine; otherwise the browser falls
# through to whatever system face matches next, and the result varies by
# platform. A browser resolves a CSS stack PER GLYPH, so the pan-CJK entries
# still keep mixed-script text rendering; `sans-serif` is the final fallback.
_PLOTLY_SANS_STACK = ('"Noto Sans", "Helvetica Neue", Helvetica, Arial, '
                      '"Noto Sans CJK JP", "Hiragino Sans", sans-serif')

# Animation Play/Pause control styling. The controls used to sit at paper
# (0, 0) anchored bottom-left, which in 2-D -- where the axes fill the paper
# area -- drew them ON TOP of the plot's bottom-left corner (maintainer report,
# Andy). They now hang BELOW the plotting area, laid out horizontally, with the
# bottom margin opened up so nothing is clipped.
_ANIM_BUTTON_MARGIN_B = 64   # bottom margin reserved for the controls (px)
CUBE_LINEWIDTH_PT = 1.5      # hypertools' frame linewidth, matching the
                             # matplotlib backend's ~2px frame (both the 3D
                             # wireframe cube and the 2D square)
# The 2D square is an SVG `shape` (honors its stroke width faithfully) but the
# 3D cube is a Scatter3d line, which plotly's gl line renderer draws at roughly
# 0.6x the requested width -- so at the same requested width the cube came out
# ~1px while the square came out ~2px, and the 2D frame looked visibly heavier
# than the 3D one (maintainer report, Andy). Boost ONLY the 3D cube's requested
# width so both render at the same ~2px as the matplotlib backend. Measured in
# the kaleido/Chrome renderer (which also produces every exported image and the
# docs gallery); the exact factor is not critical -- 1.3-1.7 all land on 2px.
_CUBE_GL_WIDTH_BOOST = 1.5

# matplotlib's '.' and ',' marker glyphs are defined with HALF the path
# scale of every other marker character (verified via
# `matplotlib.markers.MarkerStyle(ch).get_transform()`, whose scale is
# `0.25` for '.'/',' vs. `0.5` for 'o' and most others) -- so at the SAME
# `markersize`, matplotlib renders a '.' marker at exactly half the pixel
# diameter of an 'o' (confirmed by rendering both at dpi=100 and measuring
# pixel diameter: markersize=100 gave ~139px for 'o' vs ~71px for '.',
# `scripts/measure_marker_parity.py`). plotly has no equivalently-tiny
# "dot" symbol of its own -- both '.' and 'o' map to plotly's 'circle' (see
# `_MARKER_SYMBOLS`) -- so without this explicit discount plotly's dots
# render ~2x fatter than matplotlib's for `fmt='.'` (hypertools' most
# common scatter fmt, used throughout the density=/morph examples): this
# was half of the R2 "fat dots" bug.
_DOT_MARKER_CHARS = ('.', ',')
_DOT_MARKER_SCALE = 0.5

# matplotlib's animate='morph' traveling point cloud always draws with
# marker='.' and, when no explicit `markersize=` kwarg is given, a smaller
# default of 1.5pt -- NOT the general `DEFAULT_MARKERSIZE_PT` (6.0) used
# everywhere else -- see `matplotlib_backend.animate_plot3D`'s
# `morph_markersize = _mkw.get("markersize") or 1.5`. Without matching
# both that smaller default AND the `_DOT_MARKER_SCALE` above, plotly's
# default morph dots rendered ~8x fatter than matplotlib's (6.0 vs 1.5,
# doubled again for the missing dot-marker scale) -- this was the more
# severe half of the R2 bug (see
# `docs/images/v1.0-seven-features/morph_anim_plotly.png` before the fix).
MORPH_DEFAULT_MARKERSIZE_PT = 1.5

# plotly's `go.Scatter3d` (WebGL/gl3d) interprets `marker.size` differently
# from `go.Scatter`'s (SVG, 2-D) -- empirically verified (see
# `scripts/measure_marker_parity.py`'s 3-D calibration) by rendering
# isolated Scatter3d markers at a wide range of nominal `size` values
# (2..100) at several different camera distances (ruling out perspective
# as the cause -- the ratio is constant regardless of camera-to-origin
# distance) and fitting the nominal size -> measured pixel-diameter
# relationship: `diameter_px ~= 1.776 * size + 0.9` (R^2 ~1.0 across two
# orders of magnitude; the tiny intercept is negligible in practice).
# `go.Scatter`'s `marker.size` has NO such correction (it IS the rendered
# pixel diameter almost exactly, see the 2-D calibration in the same
# script) -- so every Scatter3d marker (all of hypertools' 3-D data
# traces, trails, and the animate='morph' traveling point cloud) needs
# its computed pixel size divided by this factor, or it renders ~1.8x
# fatter than the already-corrected 2-D/mpl-matching size.
_SCATTER3D_SIZE_FACTOR = 1.776

# matplotlib format-string characters -> plotly marker symbols. This MUST
# cover every marker character matplotlib's fmt grammar accepts (the printable
# keys of matplotlib.lines.Line2D.markers): a missing entry makes _parse_fmt
# treat that marker as "no marker" and silently fall through to a lines-only
# trace (this is how ',' rendered as solid lines instead of pixels).
_MARKER_SYMBOLS = {
    '.': 'circle', ',': 'circle', 'o': 'circle', 's': 'square',
    '^': 'triangle-up', 'v': 'triangle-down', '<': 'triangle-left',
    '>': 'triangle-right', '1': 'y-down', '2': 'y-up', '3': 'y-left',
    '4': 'y-right', '8': 'octagon', 'p': 'pentagon', 'P': 'cross',
    '*': 'star', 'h': 'hexagon', 'H': 'hexagon2', '+': 'cross-thin',
    'x': 'x-thin', 'X': 'x', 'D': 'diamond', 'd': 'diamond-tall',
    '|': 'line-ns', '_': 'line-ew',
}
# plotly's Scatter3d supports only a small symbol set; map unsupported 2D
# symbols to their closest 3D-legal equivalent
_SYMBOLS_3D = {'circle', 'circle-open', 'cross', 'diamond', 'diamond-open',
               'square', 'square-open', 'x'}
_SYMBOL_3D_FALLBACK = {
    'triangle-up': 'diamond', 'triangle-down': 'diamond',
    'triangle-left': 'diamond', 'triangle-right': 'diamond',
    'star': 'diamond-open', 'cross-thin': 'cross', 'x-thin': 'x',
    'diamond-tall': 'diamond', 'pentagon': 'circle', 'hexagon': 'circle',
    'hexagon2': 'circle', 'octagon': 'circle',
    'y-up': 'cross', 'y-down': 'cross', 'y-left': 'cross',
    'y-right': 'cross', 'line-ns': 'cross', 'line-ew': 'cross',
}

# matplotlib linestyles -> plotly dash styles ('-.' must be checked first)
_DASH_STYLES = (('-.', 'dashdot'), ('--', 'dash'), (':', 'dot'), ('-', 'solid'))
_LINESTYLE_NAMES = {'solid': 'solid', 'dashed': 'dash', 'dotted': 'dot',
                    'dashdot': 'dashdot', '-': 'solid', '--': 'dash',
                    ':': 'dot', '-.': 'dashdot'}


def detect_environment():
    """Return 'colab', 'kaggle', or 'other' for the current runtime."""
    if 'google.colab' in sys.modules:
        return 'colab'
    if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.path.isdir('/kaggle'):
        return 'kaggle'
    return 'other'


def resolve_backend(backend):
    """Resolve a user-requested backend to 'matplotlib' or 'plotly'."""
    # accept any case ('Plotly', 'MATPLOTLIB', ...): the canonical values are
    # lowercase, but matching case-insensitively avoids a surprising
    # "backend must be one of ..." error on an obvious spelling (QC 2026-07).
    if isinstance(backend, str):
        backend = backend.lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {VALID_BACKENDS}; got {backend!r}")
    if backend == 'auto':
        # a render preference set via hyp.set_interactive_backend('plotly' /
        # 'matplotlib') wins over the environment default (QC 2026-07). Lazy
        # import avoids a plotly_backend <-> backend import cycle at module load.
        from . import backend as _backend
        preferred = getattr(_backend, 'PREFERRED_RENDER_BACKEND', None)
        if preferred == 'plotly' and _has_plotly():
            return 'plotly'
        if preferred == 'matplotlib':
            return 'matplotlib'
        if detect_environment() in ('colab', 'kaggle') and _has_plotly():
            return 'plotly'
        return 'matplotlib'
    if backend == 'plotly' and not _has_plotly():
        raise ImportError(
            "The plotly backend requires plotly. Install it with:\n"
            "    pip install hypertools[interactive]")
    return backend


def _has_plotly():
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def _zoom_r(zoom):
    """Camera distance for a given matplotlib-style zoom: mpl image scale is
    ~10/(9 - zoom) (see matplotlib_backend's set_box_aspect conversion), so
    relative to zoom=1 the plotly camera moves in by (9 - zoom)/8."""
    return max(0.2, 1.95 * (9.0 - float(zoom)) / 8.0)


# ANIMATED plots pull the camera slightly farther back than static plots so
# the wireframe box keeps a comfortable margin at every rotation angle and is
# never clipped (Jeremy's animated-plot zoom-out request). Static plots are
# visually unchanged -- they keep using _zoom_r directly.
_ANIM_ZOOM_OUT = 1.1


def _anim_zoom_r(zoom):
    """Camera distance for ANIMATED plots: _zoom_r zoomed out by _ANIM_ZOOM_OUT."""
    return _zoom_r(zoom) * _ANIM_ZOOM_OUT


def _build_point_annotations(data, labels, ndims, font_family, label_alpha=0.5):
    """`labels=` point annotations (GH #205 F3): parity with
    `matplotlib_backend._draw`'s `annotate_plot`.

    Mirrors its semantics EXACTLY: `data` (the list of per-dataset (n_i, d)
    arrays) is stacked into one (N, d) array `X` via `np.vstack`; `labels`
    is flattened (via `itertools.chain`) if it is a list of per-dataset
    lists, else used as-is -- one label per row of `X`, in the same order.
    `labels[idx] is None` skips that point (no annotation), exactly like
    `annotate_plot`. A `labels` shorter than `X` raises `IndexError` (same
    as `annotate_plot`'s `labels[idx]` indexing); a longer `labels` simply
    has its extra entries ignored (the loop only runs `len(X)` times) --
    same mismatched-count behavior as matplotlib, not a new policy.

    Only 2-D (`ndims == 2`) and 3-D (`ndims == 3`) are supported, matching
    `annotate_plot`'s own two branches (`data[0].shape[-1] > 2` / `== 2`);
    other dimensionalities silently draw no annotations, exactly as
    `annotate_plot` does (neither of its branches match, so it draws
    nothing and raises nothing).

    Returns a list of plotly annotation dicts (3-D: includes `z`, meant for
    `layout.scene.annotations`; 2-D: no `z`, meant for `layout.annotations`).
    Style approximates `annotate_plot`'s matplotlib appearance (small text,
    a translucent white background box, a short straight connector with no
    arrowhead) as closely as plotly's annotation schema allows.
    """
    if ndims not in (2, 3):
        return []

    flat_labels = (list(itertools.chain(*labels))
                   if any(isinstance(el, list) for el in labels)
                   else list(labels))

    X = np.vstack(data)

    font = dict(size=10, color='black')
    if font_family is not None:
        font['family'] = font_family

    annotations = []
    for idx in range(X.shape[0]):
        label = flat_labels[idx]
        if label is None:
            continue
        ann = dict(
            text=str(label),
            x=float(X[idx, 0]),
            y=float(X[idx, 1]),
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor='rgba(0,0,0,0.6)',
            ax=-20,
            ay=-20,
            font=font,
            bgcolor=f'rgba(255,255,255,{label_alpha})',
            bordercolor='rgba(0,0,0,0.4)',
            borderwidth=1,
            borderpad=3,
        )
        if ndims == 3:
            ann['z'] = float(X[idx, 2])
        annotations.append(ann)
    return annotations


def _labeled_axis_layout(base, label, scene=False):
    """Build a plotly axis layout dict (2-D `layout.xaxis`/`.yaxis` or
    3-D `layout.scene.xaxis`/`.yaxis`/`.zaxis`), merged with `base` (e.g.
    a `range`).

    `visible=False` (when `label` is None -- the historical default,
    byte-identical to before `xlabel=`/`ylabel=`/`zlabel=` existed)
    hides EVERYTHING on that axis, including a title -- unlike
    matplotlib's `set_axis_off()`, whose axis label Text artist at least
    keeps its underlying text (`.get_text()` still returns it even when
    invisible). So when `label` is given (round17 #7), the axis is kept
    "visible" but every OTHER sub-property (ticks, gridlines, zero-line,
    and -- 3-D scene axes only -- the gray background pane) is hidden
    individually instead, leaving only `title` shown. `scene=True` adds
    `showbackground=False` (a scene-axis-only property; plain 2-D
    `layout.xaxis`/`.yaxis` has no such property and rejects it).
    """
    if label is None:
        return dict(visible=False, **base)
    layout = dict(
        showticklabels=False, showgrid=False, zeroline=False,
        showline=False, ticks='', title=dict(text=label), **base,
    )
    if scene:
        layout['showbackground'] = False
    return layout


def plotly_draw(data, fmt=None, kwargs_list=None, labels=None, legend=None,
                title=None, animate=False, size=None, show=True,
                save_path=None, frame_rate=30, duration=30, rotations=1,
                elev=10, azim=-60, point_colors=None, tail_duration=2,
                focused=None,
                chemtrails=False, precog=False, bullettime=False, zoom=1,
                forecasts=None, colorbar_info=None, surface=None,
                surface_colors=None, surface_point_colors=None,
                density=None, density_colors=None,
                morph_tags=None, morph_colors=None, morph_samples=None,
                font=None, label_alpha=0.5, xlabel=None, ylabel=None,
                zlabel=None):
    """Render grouped datasets with plotly, mirroring _draw's contract and
    the matplotlib renderer's appearance.

    Parameters mirror the relevant subset of
    hypertools.plot.matplotlib_backend._draw (D11 audit: every parameter
    listed; the notes further below expand on the non-obvious ones):

    Parameters
    ----------
    data : list of numpy.ndarray
        One (n_i, d) array per trace, d in (1, 2, 3), already centered and
        scaled to [-1, 1] by `plot.py`.
    fmt : list of str or None
        Matplotlib-style format strings, one per trace (None -> '-').
    kwargs_list : list of dict or None
        Per-trace matplotlib-style kwargs ('color', 'linewidth',
        'linestyle', 'marker', 'alpha', 'label', ...).
    labels : list or None
        Per-point annotation labels (one entry per observation; None
        entries mean "no label for this point").
    legend : list or None
        Legend labels (one per trace); None hides the legend.
    title : str or None
        Figure title.
    animate : bool or str
        Animation style (False for static; True/'parallel'/'spin'/
        'serial'/'window'/'morph').
    size : (width, height) or None
        Figure size in inches (converted to pixels at 100 dpi).
    show : bool
        Whether to call fig.show() (auto-suppressed in notebooks, where
        returning the figure already displays it).
    save_path : str or None
        Where to save the figure (.html for interactive output).
    frame_rate : int or float
        Animation frames per second (frame_rate * duration total frames).
    duration : float
        Animation length in seconds.
    rotations : float or list
        Camera revolutions over the animation (list form: morph only).
    elev : int or float
        Starting camera elevation, degrees (3-D only).
    azim : int or float
        Starting camera azimuth, degrees (3-D only).
    point_colors : list of numpy.ndarray or None
        Per-point RGB colors (continuous/matrix hue), one array per trace.
    tail_duration : float
        Trail length in seconds for trail styles.
    focused : float or None
        In-focus (opaque) window length in seconds; None -> tail_duration.
    chemtrails : bool or list of bool
        Past-trail flag(s), per trace.
    precog : bool or list of bool
        Future-trail flag(s), per trace.
    bullettime : bool or list of bool
        Past+future trail flag(s), per trace.
    zoom : float
        3-D camera zoom factor.
    forecasts : list of numpy.ndarray or None
        predict= forecast traces (see below).
    colorbar_info : dict or None
        Colorbar spec from `plot._build_colorbar_info` (see below).
    surface : list of dict or None
        Per-trace surface= specs (hull rendering).
    surface_colors : list or None
        Resolved per-trace surface colors.
    surface_point_colors : list or None
        Optional (points, per-point RGB) bundles for hue-colored hulls.
    density : list of dict or None
        Per-trace density= specs (KDE shading).
    density_colors : list or None
        Resolved per-trace density colors.
    morph_tags : list of bool or None
        Which traces join an animate='morph' sequence.
    morph_colors : list or None
        Resolved per-trace colors for morph interpolation.
    morph_samples : int or None
        Optional point-count cap for morphing datasets.
    font : matplotlib.font_manager.FontProperties or None
        Resolved font (family name is applied to layout.font; see below).
    label_alpha : float
        Opacity of the label annotations' background box (default 0.5).
    xlabel : str or None
        x-axis title.
    ylabel : str or None
        y-axis title.
    zlabel : str or None
        z-axis title (3-D only; rejected upstream otherwise).

    `font` (GH #205): the ALREADY-RESOLVED `matplotlib.font_manager.
    FontProperties` from `hypertools.plot.fonts.resolve_font` (or `None`
    -- no override), computed once in `plot.py` from every text source
    (labels/legend/title/hue) shared with the matplotlib backend. Unlike
    matplotlib, plotly text surfaces don't accept a font FILE -- only a
    FAMILY NAME -- so only `font.get_name()` is used here, wrapped in a
    small fallback chain (`'"<name>", "Noto Sans CJK JP", sans-serif'`)
    and set as `layout.font.family`; every plotly text surface hypertools
    creates (legend, colorbar title/ticks, plot title) inherits it unless
    it hardcodes its own `font.family` (only the title used to -- fixed
    below).

    `labels=` (GH #205 F3): point annotations, at parity with matplotlib's
    `annotate_plot` -- see `_build_point_annotations` for the exact
    label-to-point mapping semantics (mirrored from `annotate_plot`) and
    styling. Rendered as `layout.scene.annotations` (3-D) or
    `layout.annotations` (2-D); inherits the resolved `font=` family the
    same way the legend/colorbar/title do. Drawn unconditionally
    (including when `animate` is truthy) -- matplotlib never skips or
    raises for animated + `labels=` either (`_draw` calls `add_labels`
    after dispatching either the static or animated path); it just draws
    static annotations at the ORIGINAL (pre-animation) data coordinates
    on top of the animation, so this mirrors that by adding the SAME
    annotations to the base layout (not per-frame), anchored at the given
    data coordinates and persisting across every frame.

    `forecasts` (predict=, GH #169): an optional list of (t+1, d) arrays,
    one per dataset in `data` (same length, same coordinate space -- already
    center/scale-matched to `data` by the caller), each starting with the
    dataset's final observed row so the trace connects. Rendered as one
    dashed (`dash='dash'`), 0.6-opacity, `showlegend=False` trace per
    dataset, in the SAME color as its source trace.

    `colorbar_info` (GH #100): optional dict from
    `hypertools.plot.plot._build_colorbar_info` (``kind='continuous'`` with
    ``vmin``/``vmax``/``palette``, or ``kind='discrete'`` with
    ``colors``/``labels``; both carry ``label``/``ticks``/``location``
    overrides). Rendered as a colorbar attached to a hidden ("phantom")
    marker trace -- plotly colorbars are a `marker`/`line` property of a
    trace, not a figure-level artist, so a real (invisible) trace carries
    it without adding a visible point.

    `label_alpha` (GH #103): opacity of the translucent white background
    box drawn behind each `labels=` point annotation -- the alpha channel
    of `bgcolor='rgba(255,255,255,<label_alpha>)'` -- mirroring
    `annotate_plot`'s matplotlib `bbox` alpha exactly. Default 0.5 (the
    historical hardcoded value).

    `focused` (round17 #8, GH #275): the length, in seconds (same unit as
    `tail_duration`), of the opaque "in-focus" window for `animate='window'`
    and any chemtrails/precog/bullettime-flagged dataset -- see
    `hypertools.plot.plot.plot`'s `focused=` docstring for the full
    semantics (when it applies vs. is ignored). `None` (default) resolves to
    `tail_duration`'s own value here (defensively, mirroring the
    `chemtrails`/`precog`/`bullettime` re-broadcast above) when this
    function is called directly rather than through `plot.py`, which always
    resolves it first.

    `xlabel`/`ylabel`/`zlabel` (round17 #7): axis titles, in BOTH 2-D
    (`layout.xaxis.title`/`.yaxis.title`) and 3-D
    (`layout.scene.xaxis.title`/`.yaxis.title`/`.zaxis.title`) -- see
    `_labeled_axis_layout` for exactly which OTHER axis sub-properties
    stay hidden (ticks/gridlines/zero-line/background) so only the title
    itself becomes visible. `None` (default): axis fully hidden, byte-
    identical to before these kwargs existed. `zlabel` on a 2-D/1-D plot
    is rejected upstream in `plot.py` (`ValueError`, before this function
    is ever called).

    Returns the plotly Figure.
    """
    import plotly.graph_objects as go

    # input validation (release-1.0 audit, F24-012): plot() always calls
    # this with non-empty, already-reduced (<= 3-column) data and matching
    # per-trace lists, but plotly_draw is publicly reachable (re-exported
    # via the hypertools.plot.interactive shim) -- direct misuse previously
    # surfaced as a bare IndexError (empty data / mismatched fmt) or
    # SILENTLY drew only the first 3 columns of wider data.
    if data is None or len(data) == 0:
        raise ValueError(
            "plotly_draw requires at least one dataset in `data`; got an "
            "empty data list.")
    for _i, _d in enumerate(data):
        _shape = np.asarray(_d).shape
        _ncols = _shape[1] if len(_shape) > 1 else 1
        if len(_shape) not in (1, 2) or _ncols not in (1, 2, 3):
            raise ValueError(
                "plotly_draw supports 1-, 2-, or 3-column (already-"
                f"reduced) datasets; data[{_i}] has shape {_shape}. Reduce "
                "to <= 3 dimensions first (e.g. via hyp.plot's ndims=/"
                "reduce=).")
    if fmt is not None and len(fmt) != len(data):
        raise ValueError(
            f"fmt has {len(fmt)} entr{'y' if len(fmt) == 1 else 'ies'} but "
            f"there are {len(data)} dataset(s); pass one format string per "
            "dataset (or None).")
    if kwargs_list is not None and len(kwargs_list) != len(data):
        raise ValueError(
            f"kwargs_list has {len(kwargs_list)} "
            f"entr{'y' if len(kwargs_list) == 1 else 'ies'} but there are "
            f"{len(data)} dataset(s); pass one kwargs dict per dataset "
            "(or None).")

    fmt = fmt if fmt is not None else ['-'] * len(data)
    kwargs_list = kwargs_list if kwargs_list is not None else [{}] * len(data)

    # chemtrails/precog/bullettime (GH #127): normalize to one bool per
    # dataset. `plot.py` already broadcasts/validates against the FINAL
    # (post cluster/hue-reshape) dataset count before calling `plotly_draw`,
    # but this call is defensive (mirrors `matplotlib_backend._draw`'s same
    # normalization) so `plotly_draw` also works when called directly (as
    # several tests do) with a bare bool.
    chemtrails = broadcast_trail_flag(chemtrails, len(data), "chemtrails")
    precog = broadcast_trail_flag(precog, len(data), "precog")
    bullettime = broadcast_trail_flag(bullettime, len(data), "bullettime")

    ndims = data[0].shape[1] if data[0].ndim > 1 else 1

    # animate='morph' (Hungarian point-cloud morphs, maintainer request):
    # `plot.py` already raises `NotImplementedError` for 1-D (or higher
    # than 3-D) data before ever calling this backend; this is a defensive
    # re-check (mirrors `broadcast_trail_flag`'s own defensive
    # re-normalization above) for direct callers (tests) that bypass
    # `plot.py`. round17 #9 (GH #123): 2-D is now supported too, exactly
    # like every other animate style.
    if animate == "morph" and ndims not in (2, 3):
        raise NotImplementedError(
            "animate='morph' is only supported for 2-D or 3-D plots; got "
            f"{ndims}-D data."
        )

    # round17 #9 (GH #123): 'spin' rotates the 3-D camera and has no
    # meaning for 2-D data (2-D animations use a fixed, non-rotating
    # viewport, exactly like the matplotlib backend's `animate_plot2D`) --
    # without this check it would silently fall through to `_add_animation`'s
    # generic sliding-window branch instead of erroring.
    if animate == "spin" and ndims == 2:
        raise ValueError(
            "animate='spin' rotates the 3-D camera and has no meaning for "
            "2-D data (2-D animations use a fixed, non-rotating viewport). "
            "Use 'parallel'/True, 'serial', 'window', 'chemtrails', "
            "'precog', 'bullettime', or 'morph' instead."
        )
    morph_tags = (morph_tags if morph_tags is not None
                 else ([True] * len(data) if animate == "morph" else None))

    # density= (GH #108/#191), 2-D case: subtle KDE density layers must
    # render BELOW everything else (including surface= fills). Plotly's 2D
    # layering follows trace order in `fig.data` (no zorder), so these are
    # seeded at the very FRONT of `traces`. Unlike surface=, density is
    # supported WITH animate (it's computed once from the full data and
    # never touched by a frame update -- see `data_trace_start` below).
    n_density_traces_2d = 0
    if density is not None and ndims == 2:
        density_traces_2d = _build_density_traces_2d(go, data, density,
                                                      density_colors)
        n_density_traces_2d = len(density_traces_2d)
    else:
        density_traces_2d = []

    # surface= (GH #109), 2-D static case: smooth filled hull outlines must
    # render BELOW the data traces (but above any density layer). Plotly's
    # 2D layering follows trace order in `fig.data` (no zorder), so these
    # are seeded at the FRONT of `traces` (drawn first = bottom), after the
    # density layer. (2-D + animate is not supported for surfaces -- see the
    # 3-D branch below and the surface= docstring -- so this only runs for
    # static plots.)
    n_surface_traces_2d = 0
    if surface is not None and ndims == 2 and not animate:
        surface_traces_2d = _build_surface_traces_2d(go, data, surface,
                                                      surface_colors)
        n_surface_traces_2d = len(surface_traces_2d)
    else:
        surface_traces_2d = []
    traces = list(density_traces_2d) + list(surface_traces_2d)
    # absolute `fig.data` index where the DATA (non-background) traces
    # start: 0 unless 2-D density/surface layers were seeded at the front.
    data_trace_start = n_density_traces_2d + n_surface_traces_2d
    for i, arr in enumerate(data):
        if morph_tags is not None and morph_tags[i]:
            # animate='morph': this dataset joins the single traveling
            # point-cloud trace built below (after this loop) instead of
            # getting its own full-cloud trace -- nothing is appended for
            # it here (n_data_traces below is a plain COUNT, not a
            # positional map, so skipping entries is safe).
            continue
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        tkwargs = kwargs_list[i] or {}
        mode, symbol, dash, marker_char = _resolve_fmt(fmt[i], tkwargs)
        color = _to_plotly_color(tkwargs.get('color'), tkwargs.get('alpha'))
        width = float(tkwargs.get('linewidth')
                      or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
        msize = _marker_size_px(
            tkwargs.get('markersize') or DEFAULT_MARKERSIZE_PT, marker_char,
            ndims=ndims)
        name = _trace_name(legend, tkwargs, i)

        if ndims >= 3 and symbol not in _SYMBOLS_3D:
            symbol = _SYMBOL_3D_FALLBACK.get(symbol, 'circle')

        # multicolored lines: per-point colors along each trajectory
        trace_point_colors = None
        if point_colors is not None and i < len(point_colors) \
                and point_colors[i] is not None:
            trace_point_colors = [
                _rgb_string(c) for c in np.asarray(point_colors[i])]

        # surface= (GH #109) keep_points=False: hide this dataset's own
        # line/marker trace so only its surface shows.
        hide_points = (surface is not None and i < len(surface)
                      and surface[i] is not None
                      and not surface[i].get('keep_points', True))

        # surface= (GH #109 rendering-fix), 3-D only, FULLY-OPAQUE surfaces
        # only (release-1.0 audit, F07-001): plotly cannot always correctly
        # depth-composite Scatter3d points enclosed by an opaque Mesh3d
        # surface (they can visibly "punch through" the mesh as a hole --
        # see `_trim_faces_inside_other_meshes`'s docstring for the full
        # story and verification). Points a fully-opaque
        # (alpha >= SURFACE_OPAQUE_ALPHA) surface encloses are dropped (set
        # to NaN, plotly's standard "no point here" convention) from its
        # marker/line trace instead -- they would be hidden behind the
        # opaque surface anyway; any points the surface fails to enclose
        # (smoothing/inflation targets ~99% containment, not 100%) are left
        # visible as before. TRANSLUCENT surfaces (alpha < the threshold)
        # never hide their points: the mesh now renders with real Mesh3d
        # opacity (see `_mesh3d_trace`), so the data shows through it
        # exactly like the matplotlib reference behavior.
        if (ndims >= 3 and not hide_points and surface is not None
                and i < len(surface) and surface[i] is not None
                and surface[i].get('alpha', SURFACE_DEFAULTS['alpha'])
                    >= SURFACE_OPAQUE_ALPHA):
            mesh = build_mesh_3d(arr[:, :3], surface[i], dataset_label=f' {i}',
                                 quiet=True)
            if mesh is not None:
                mesh_verts, _mesh_faces = mesh
                enclosed = points_enclosed(arr[:, :3], mesh_verts)
                if enclosed.any():
                    arr = arr.copy()
                    arr[enclosed] = np.nan

        common = dict(
            mode=mode,
            name=name,
            showlegend=(legend is not None and name is not None
                       and not str(name).startswith('_')
                       and not hide_points),
            visible=not hide_points,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(color=color, size=msize, symbol=symbol),
        )
        if ndims >= 3:
            if trace_point_colors is not None:
                # Scatter3d supports per-point line colors natively
                common['line'] = dict(color=trace_point_colors, width=width,
                                      dash=dash)
                common['marker'] = dict(color=trace_point_colors,
                                        size=msize, symbol=symbol)
            traces.append(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], **common))
        elif ndims == 2:
            if trace_point_colors is not None and 'lines' in mode:
                # 2D Scatter has no per-point line colors; draw short
                # segment traces instead (grouped under one legend entry)
                traces.extend(_segment_traces_2d(
                    go, arr, trace_point_colors, width, dash, name))
                continue
            if trace_point_colors is not None:
                common['marker'] = dict(color=trace_point_colors,
                                        size=msize, symbol=symbol)
            traces.append(go.Scatter(x=arr[:, 0], y=arr[:, 1], **common))
        else:
            xs = np.arange(arr.shape[0])
            if trace_point_colors is not None and 'lines' in mode:
                pts = np.column_stack([xs, arr[:, 0]])
                traces.extend(_segment_traces_2d(
                    go, pts, trace_point_colors, width, dash, name))
                continue
            traces.append(go.Scatter(x=xs, y=arr[:, 0], **common))

    n_data_traces = len(traces) - n_surface_traces_2d - n_density_traces_2d

    # predict=: one dashed, low-opacity forecast trace per dataset, in the
    # same color as its source trace (GH #169; matplotlib parity).
    if forecasts is not None:
        for i, arr in enumerate(data):
            tkwargs = kwargs_list[i] or {}
            fc = np.atleast_2d(np.asarray(forecasts[i], dtype=np.float64))
            color = _to_plotly_color(tkwargs.get('color'), 0.6)
            width = float(tkwargs.get('linewidth')
                          or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
            fc_common = dict(mode='lines', showlegend=False,
                             hoverinfo='skip',
                             line=dict(color=color, width=width, dash='dash'))
            if ndims >= 3:
                traces.append(go.Scatter3d(
                    x=fc[:, 0], y=fc[:, 1], z=fc[:, 2], **fc_common))
            elif ndims == 2:
                traces.append(go.Scatter(
                    x=fc[:, 0], y=fc[:, 1], **fc_common))
            else:
                arr2 = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                start = arr2.shape[0] - 1
                traces.append(go.Scatter(
                    x=np.arange(start, start + fc.shape[0]), y=fc[:, 0],
                    **fc_common))

    # low-opacity trail traces for chemtrails (past) / precog (future) /
    # bullettime (both) on window animations, mirroring the matplotlib
    # renderer's alpha-0.3 trail artists. One per dataset THAT HAS ANY of
    # the three flags set (GH #127: previously all-or-nothing -- ANY flag
    # set anywhere created a trail trace for EVERY dataset). These do NOT
    # necessarily sit right after the data traces -- forecast traces
    # (predict=, above) are appended in between when both are present -- so
    # `trail_trace_start` records their real position, and
    # `trail_dataset_indices[k]` is the ORIGINAL dataset index that produced
    # `traces[trail_trace_start + k]`, so `_add_animation` can look up the
    # right dataset's data per frame.
    n_trail_traces = 0
    trail_trace_start = len(traces)
    trail_dataset_indices = [
        i for i in range(len(data))
        if chemtrails[i] or precog[i] or bullettime[i]
    ] if animate in (True, 'parallel') else []
    for i in trail_dataset_indices:
        tkwargs = kwargs_list[i] or {}
        mode, symbol, dash, marker_char = _resolve_fmt(fmt[i], tkwargs)
        color = _to_plotly_color(tkwargs.get('color'), 0.3)
        width = float(tkwargs.get('linewidth')
                      or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
        msize = _marker_size_px(
            tkwargs.get('markersize') or DEFAULT_MARKERSIZE_PT, marker_char,
            ndims=ndims)
        trail = dict(mode=mode, showlegend=False, hoverinfo='skip',
                     line=dict(color=color, width=width, dash=dash),
                     marker=dict(color=color, size=msize))
        if ndims >= 3:
            traces.append(go.Scatter3d(x=[], y=[], z=[], **trail))
        else:
            traces.append(go.Scatter(x=[], y=[], **trail))
    n_trail_traces = len(trail_dataset_indices)

    # surface= (GH #109), 3-D case: order doesn't matter here (plotly's 3-D
    # scene is depth-buffered, unlike 2-D's painter's-algorithm trace order),
    # so these are simply appended. `surface_dataset_indices[k]` records
    # which ORIGINAL dataset produced `surface_traces_3d[k]` (datasets whose
    # spec is None, or whose points are too few/degenerate, produce no
    # trace at all) -- `_add_animation` needs that mapping to recompute the
    # right dataset's window each frame.
    surface_trace_start_3d = len(traces)
    surface_dataset_indices = []

    # animate='morph': sampled once here (rather than down where the
    # traveling trace is built) so the cube_scale block below can reuse the
    # EXACT `sampled0` arrays `_add_animation`'s 'morph' branch will later
    # draw -- see the M3b box-containment note on `cube_scale` just below.
    # round17 #9 (GH #123): morph clouds/colors are resolved for 2-D too
    # now, not just 3-D -- names keep their historical "_3d" suffix (private
    # to this function) to minimize the diff, but `clouds0`'s column count
    # follows `ndims` below. surface= tracking is still 3-D only (see the
    # surface_for_static/cube_scale block just below, unchanged), so
    # `morph_surface_spec_3d` is only ever resolved when `ndims >= 3`.
    morph_indices_3d = None
    sampled0 = None
    dup_masks0 = None
    ds_colors0 = None
    morph_surface_spec_3d = None
    if morph_tags is not None and ndims in (2, 3):
        morph_indices_3d = [i for i, t in enumerate(morph_tags) if t]
        _morph_ncols = 3 if ndims >= 3 else 2
        clouds0 = [np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :_morph_ncols]
                  for i in morph_indices_3d]
        sampled0, dup_masks0 = _morph.sample_and_match_clouds(
            clouds0, morph_samples=morph_samples)
        ds_colors0 = [
            tuple(morph_colors[i]) if morph_colors is not None
            else (0.2, 0.4, 0.8)
            for i in morph_indices_3d
        ]
        if ndims >= 3:
            for i in morph_indices_3d:
                if surface is not None and i < len(surface) and surface[i] is not None:
                    morph_surface_spec_3d = surface[i]
                    break

    # cube_scale (GH #109 round 2): sized to whatever the built surface
    # meshes actually need (see `surface_cube_scale`), not assumed to be
    # the standard 1 -- otherwise a smoothed hull's pre_inflate/smoothing
    # overshoot can bulge past the drawn cube and axis ranges. `meshes`
    # holds every dataset's UNTRIMMED mesh (trimming only ever drops
    # faces, never moves vertices), so it is a fully-representative,
    # already-computed-once source for this bound.
    cube_scale = 1.0
    if surface is not None and ndims >= 3:
        # animate='morph': morph-tagged datasets never get their own STATIC
        # per-dataset mesh trace (they'd sit there, unmoving, duplicating
        # the single traveling morph mesh built below) -- excluded here via
        # a surface list with their entries forced to None. This ALSO means
        # `_build_surface_traces_3d` below never builds a mesh from a morph-
        # tagged dataset's FULL (unsampled) cloud at all -- their box-sizing
        # bound comes entirely from the sampled+union meshes computed below
        # (M3b/M4), never from the full-order cloud (which would be both a
        # correctness risk, see the M3b note just below, and, on a large raw
        # cloud, needless cost -- mirroring `matplotlib_backend
        # .animate_plot3D`'s identical M4 fix).
        surface_for_static = (
            [None if (morph_tags is not None and morph_tags[i]) else s
             for i, s in enumerate(surface)]
            if morph_tags is not None else surface
        )
        surface_traces_3d, surface_dataset_indices, surface_meshes = (
            _build_surface_traces_3d(go, data, surface_for_static,
                                     surface_colors, elev, azim,
                                     surface_point_colors=surface_point_colors))
        traces.extend(surface_traces_3d)
        # M3b box-containment fix: `full`-cloud meshes (built above, from
        # each morphing dataset's FULL, differently-ORDERED cloud) are NOT
        # a safe bound for the per-frame rebuilt mesh -- smooth_hull_3d's
        # ConvexHull/Taubin-smoothing pipeline is not invariant to input
        # row order for hulls with many coplanar/degenerate faces (e.g. a
        # cube's flat sides), so the SAME points in a different order can
        # produce a mesh whose extent exceeds the fixed 2% margin. Mid-morph
        # interpolated points are also convex combinations of two
        # consecutive `sampled0` clouds and so can lie outside either
        # endpoint's OWN hull even though they always lie inside the hull
        # of their UNION. Fix: size from meshes built with the EXACT
        # `sampled0` arrays that will actually be drawn (guaranteeing
        # hold-frame containment) plus one mesh built from the union of
        # every sampled cloud (a cheap, strictly-safe bound for every
        # interpolated frame).
        morph_full_meshes_for_scale = []
        if morph_surface_spec_3d is not None and sampled0 is not None:
            spec = morph_surface_spec_3d
            for cloud in sampled0:
                m = build_mesh_3d(cloud, spec, dataset_label=' morph',
                                  quiet=True)
                if m is not None:
                    morph_full_meshes_for_scale.append(m)
            union_cloud = np.concatenate(sampled0, axis=0)
            m_union = build_mesh_3d(union_cloud, spec,
                                    dataset_label=' morph-union', quiet=True)
            if m_union is not None:
                morph_full_meshes_for_scale.append(m_union)
        cube_scale = surface_cube_scale(
            list(surface_meshes.values()) + morph_full_meshes_for_scale)
        if morph_surface_spec_3d is not None:
            # full-sample duplication can make the endpoint+union sizing
            # bound above under-cover the worst actual mid-morph frame --
            # see `_morph.MORPH_SURFACE_SIZING_MARGIN`.
            cube_scale *= _morph.MORPH_SURFACE_SIZING_MARGIN

    # animate='morph': ONE traveling point-cloud trace (+ one Mesh3d trace
    # if any morphing dataset requests a surface), appended after every
    # normal data/trail/surface trace. `morph_trace_start_3d`/
    # `morph_mesh_trace_start_3d` record their positions for
    # `_add_animation`'s 'morph' branch.
    morph_trace_start_3d = None
    morph_mesh_trace_start_3d = None
    if morph_tags is not None and ndims in (2, 3):
        pts0 = sampled0[0]
        # full-sample morphs (maintainer request, 2026-07-06 follow-up):
        # this initial trace is frame 0 -- a HOLD frame of dataset 0 -- so
        # its own duplicated (padding) points are excluded here too, exactly
        # like every other hold frame (see `_add_animation`'s 'morph'
        # branch below and `hypertools.plot.morph.morph_visible_mask`).
        hide0 = _morph.morph_visible_mask(dup_masks0, 0)
        draw_pts0 = pts0[~hide0] if hide0 is not None else pts0
        color0_str = _rgb_string(ds_colors0[0])
        # matplotlib's morph trace always draws marker='.' (see
        # `MORPH_DEFAULT_MARKERSIZE_PT`'s docstring) -- so the plotly
        # counterpart always applies the dot-marker scale, and falls back
        # to the SAME smaller 1.5pt default (not the general 6.0pt
        # `DEFAULT_MARKERSIZE_PT`) when no explicit `markersize=` is given.
        msize0 = _marker_size_px(
            (kwargs_list[morph_indices_3d[0]] or {}).get('markersize')
            or MORPH_DEFAULT_MARKERSIZE_PT, '.', ndims=ndims)
        hide_morph_points = (morph_surface_spec_3d is not None and
                            not morph_surface_spec_3d.get('keep_points', True))
        morph_trace_start_3d = len(traces)
        # round17 #9 (GH #123): 2-D morphs use a plain go.Scatter marker
        # trace (no z, no scene) -- surface= tracking (the Mesh3d block
        # below) never runs for 2-D since `morph_surface_spec_3d` is only
        # ever resolved when `ndims >= 3` above.
        if ndims >= 3:
            traces.append(go.Scatter3d(
                x=draw_pts0[:, 0], y=draw_pts0[:, 1], z=draw_pts0[:, 2],
                mode='markers',
                marker=dict(color=color0_str, size=msize0, symbol='circle'),
                showlegend=False, visible=not hide_morph_points, hoverinfo='skip'))
        else:
            traces.append(go.Scatter(
                x=draw_pts0[:, 0], y=draw_pts0[:, 1],
                mode='markers',
                marker=dict(color=color0_str, size=msize0),
                showlegend=False, visible=not hide_morph_points, hoverinfo='skip'))

        if morph_surface_spec_3d is not None:
            view0 = view_vector(elev, azim)
            light_kw0 = mpl_lighting_kwargs(morph_surface_spec_3d)
            mesh0 = (build_mesh_3d(pts0, morph_surface_spec_3d,
                                   dataset_label=' morph', quiet=True)
                     if pts0.shape[0] >= 4 else None)
            if mesh0 is None:
                v0 = np.tile(pts0[-1] if len(pts0) else np.zeros(3), (4, 1))
                f0 = np.array([[0, 1, 2]])
            else:
                v0, f0 = mesh0
            morph_mesh_trace_start_3d = len(traces)
            traces.append(_mesh3d_trace(
                go, v0, f0, ds_colors0[0], morph_surface_spec_3d['alpha'],
                view0, light_kw0))

    # density= (GH #108/#191), 3-D case: one go.Volume trace per dataset (or
    # one pooled trace), computed ONCE from the full data. Appended here,
    # BEFORE the cube trace -- like the 3-D surface traces above, order
    # doesn't matter (depth-buffered scene) and, crucially, these traces are
    # NEVER added to `trace_indices`/`surface_trace_indices` in
    # `_add_animation`, so they are untouched by (and thus static across)
    # every animation frame.
    if density is not None and ndims >= 3:
        traces.extend(_build_density_traces_3d(go, data, density,
                                               density_colors))

    if ndims >= 3:
        traces.append(_cube_trace(go, scale=cube_scale))

    # colorbar (GH #100): appended LAST (after the cube trace) so it never
    # falls within `trace_indices = range(n_data_traces [+ n_trail_traces])`
    # -- the animation frame-update code below only ever touches those
    # indices, so this trace (and its colorbar) is never touched by a frame
    # update and stays static across the whole animation.
    has_colorbar = colorbar_info is not None
    if has_colorbar:
        traces.append(_colorbar_trace(go, colorbar_info, ndims,
                                      legend_present=legend is not None))

    fig = go.Figure(data=traces)

    # match matplotlib: centered black title (12pt, converted via the
    # module's shared PT_TO_PX = 100/72 rule and rounded to a whole pixel:
    # round(12 * 100/72) = 17px), default canvas
    # 6.4 x 4.8 inches at 100 dpi, legend to the RIGHT of the plot and
    # vertically centered on the box (same as the matplotlib renderer).
    # When a colorbar is ALSO shown on the (default) right side, it is
    # pushed further right than the legend (see `_colorbar_trace`) and the
    # right margin is widened further so neither is clipped.
    margin_r = 10
    if legend is not None:
        margin_r += 110
    if has_colorbar:
        margin_r += 110

    # font= (GH #205): plotly text surfaces take a FAMILY NAME (not a file
    # path), so only `font.get_name()` is used, wrapped in a fallback chain in
    # case the exact family name isn't installed in whatever renders this
    # (browser/Chromium via kaleido). Only an EXPLICIT `font=` is passed here
    # (auto-detected gap fonts are handled on the matplotlib side and are not
    # meaningful to the browser anyway); it leads, otherwise the default stack
    # is used. This PREFERS the same Noto-first face as the matplotlib backend
    # but cannot guarantee it -- the browser only resolves an installed family
    # name, never hypertools' bundled font FILE (see `_PLOTLY_SANS_STACK`). The
    # CSS stack is resolved PER GLYPH by the browser, so listing pan-CJK faces
    # after the Latin ones keeps mixed-script text rendering.
    font_family = (f'"{font.get_name()}", {_PLOTLY_SANS_STACK}'
                   if font is not None else _PLOTLY_SANS_STACK)

    layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=legend is not None,
        margin=dict(l=10, r=margin_r, t=40 if title else 10, b=10),
        legend=dict(bgcolor='rgba(255,255,255,0.8)',
                    x=1.02, y=0.5, xanchor='left', yanchor='middle'),
        # layout.font is plotly's inherited default for every text surface
        # (legend, colorbar title/ticks, plot title, annotations) that doesn't
        # set its own `font.family` -- so this one line covers all of them
        # except the title, which historically hardcoded its own family
        # (fixed just below).
        font=dict(family=font_family),
    )
    if title is not None:
        # centered over the plotting area (xref='paper'), like matplotlib
        # centers its title over the axes; same 12pt sans-serif appearance,
        # converted with the module's PT_TO_PX (100/72) rule and rounded to
        # a whole pixel (17px -- the old hardcoded 16 used the CSS 96/72
        # factor this module abandoned; release-1.0 audit, F08 follow-up).
        # family: the resolved font (GH #205) when given/auto-detected,
        # else the historical hardcoded default (ASCII-only regression:
        # byte-identical to before this kwarg existed).
        layout['title'] = dict(text=title, x=0.5, xanchor='center',
                               xref='paper',
                               y=0.97, yanchor='top',
                               font=dict(color='black',
                                         size=round(12 * PT_TO_PX),
                                         family=font_family if font_family
                                                is not None else
                                                'DejaVu Sans, Arial, '
                                                'sans-serif'))
    size = size if size is not None else DEFAULT_FIGSIZE
    layout['width'] = int(size[0] * 100)
    layout['height'] = int(size[1] * 100)

    if ndims >= 3:
        layout['scene'] = dict(
            xaxis=_labeled_axis_layout(
                {'range': [-cube_scale, cube_scale]}, xlabel, scene=True),
            yaxis=_labeled_axis_layout(
                {'range': [-cube_scale, cube_scale]}, ylabel, scene=True),
            zaxis=_labeled_axis_layout(
                {'range': [-cube_scale, cube_scale]}, zlabel, scene=True),
            camera=dict(eye=_camera_eye(
                elev, azim,
                r=_anim_zoom_r(zoom) if animate else _zoom_r(zoom))),
            # matplotlib's Axes3D uses a 4:4:3 box aspect by default; match
            # it so the cube renders wider than tall, exactly like the
            # matplotlib backend
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=0.75),
        )
    elif ndims == 2:
        # matplotlib stretches the 2D frame to fill the axes region (no
        # equal-aspect constraint), so the plotly frame does the same
        layout['xaxis'] = _labeled_axis_layout({'range': [-1.1, 1.1]}, xlabel)
        layout['yaxis'] = _labeled_axis_layout({'range': [-1.1, 1.1]}, ylabel)
        layout['shapes'] = [_square_shape()]
    else:
        layout['xaxis'] = _labeled_axis_layout({}, xlabel)
        layout['yaxis'] = _labeled_axis_layout({}, ylabel)

    # labels= (GH #205 F3): point annotations, at parity with matplotlib's
    # annotate_plot -- see _build_point_annotations for the exact mapping
    # semantics. 3-D annotations live in layout.scene.annotations (data
    # space, x/y/z); 2-D annotations live in layout.annotations (data
    # space via xref/yref='x'/'y', since the default paper-relative refs
    # would ignore the actual data coordinates).
    if labels is not None:
        point_annotations = _build_point_annotations(
            data, labels, ndims, font_family, label_alpha=label_alpha)
        if point_annotations:
            if ndims == 3:
                layout['scene']['annotations'] = point_annotations
            elif ndims == 2:
                for ann in point_annotations:
                    ann.setdefault('xref', 'x')
                    ann.setdefault('yref', 'y')
                layout['annotations'] = point_annotations

    fig.update_layout(**layout)

    if animate:
        _add_animation(fig, data, ndims, animate, frame_rate, duration,
                       rotations, elev, azim, n_data_traces,
                       tail_duration=tail_duration, focused=focused,
                       chemtrails=chemtrails,
                       precog=precog, bullettime=bullettime, zoom=zoom,
                       n_trail_traces=n_trail_traces,
                       trail_trace_start=trail_trace_start,
                       trail_dataset_indices=trail_dataset_indices,
                       surface=surface, surface_colors=surface_colors,
                       surface_trace_start=surface_trace_start_3d,
                       surface_dataset_indices=surface_dataset_indices,
                       surface_point_colors=surface_point_colors,
                       data_trace_start=data_trace_start,
                       morph_tags=morph_tags, morph_colors=morph_colors,
                       morph_samples=morph_samples,
                       morph_trace_start=morph_trace_start_3d,
                       morph_mesh_trace_start=morph_mesh_trace_start_3d,
                       morph_surface_spec=morph_surface_spec_3d,
                       morph_sampled=sampled0, morph_dup_masks=dup_masks0)

    if save_path is not None:
        ext = save_path.lower().rsplit('.', 1)[-1]
        if ext == 'html':
            fig.write_html(save_path)
        elif animate and ext in ('gif', 'png', 'apng', 'mp4', 'mov', 'avi',
                                 'svg'):
            _export_animation_file(fig, save_path, frame_rate, duration,
                                   size)
        else:
            fig.write_image(save_path)

    if show:
        import plotly.io as pio
        if 'sphinx_gallery' in str(pio.renderers.default or ''):
            # docs builds: plotly's sphinx-gallery renderer writes a static
            # png AND an interactive html from the full figure, and kaleido
            # serializes EVERY animation frame to render the one png -- a
            # 900-frame figure took ~an hour and produced tens-of-MB pages.
            # Write the pair ourselves: png from a frame-stripped snapshot,
            # html with the embedded frames capped (total duration and
            # rotations preserved, so pacing stays identical).
            _show_sphinx_gallery(fig)
        elif not _in_interactive_shell():
            # Plain script (no IPython frontend): nothing else will display the
            # figure, so show it here. In an interactive notebook we DON'T call
            # fig.show(): plot() RETURNS the Figure, and the notebook frontend
            # rich-displays that return value exactly once. Calling fig.show()
            # too rendered it TWICE, and -- since fig.show() fires mid-cell,
            # before matplotlib's end-of-cell flush -- made plotly figures jump
            # ahead of matplotlib ones (Jeremy's QC 2026-07 double-display /
            # ordering report). Assign the result or use show=False to hold the
            # Figure without displaying it.
            fig.show()

    return fig


def _in_interactive_shell():
    """True inside an interactive IPython/Jupyter frontend (where a returned
    figure is auto-displayed by the cell's rich-display hook)."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


_SG_MAX_EMBEDDED_FRAMES = 150


def _show_sphinx_gallery(fig):
    import itertools
    import math
    import os

    import plotly.graph_objects as go

    for i in itertools.count():
        base = os.path.join(os.getcwd(), f'hypertools_fig_{i:03d}')
        if not (os.path.exists(base + '.html')
                or os.path.exists(base + '.png')):
            break

    snapshot = go.Figure(fig)
    snapshot.frames = ()
    snapshot.layout.updatemenus = ()
    snapshot.write_image(base + '.png')

    light = fig
    if fig.frames and len(fig.frames) > _SG_MAX_EMBEDDED_FRAMES:
        step = math.ceil(len(fig.frames) / _SG_MAX_EMBEDDED_FRAMES)
        light = go.Figure(fig)
        light.frames = fig.frames[::step]
        # keep total playback duration identical: fewer embedded frames,
        # each shown proportionally longer
        for menu in light.layout.updatemenus or ():
            for button in menu.buttons or ():
                try:
                    button.args[1]['frame']['duration'] *= step
                except (IndexError, KeyError, TypeError):
                    pass
    light.write_html(base + '.html', include_plotlyjs='cdn',
                     auto_play=False)


import contextlib
import threading

# --- headless-Chrome (kaleido) animation export: hard timeout via subprocess -
# kaleido 1.x drives headless Chrome; its OWN per-render timeout only wraps the
# figure CALC (`asyncio.wait_for(tab._calc_fig(...), ...)`), NOT browser launch
# or tab acquisition, and its shared sync server blocks on an unbounded
# Queue.get -- so a wedged Chrome hangs a `to_image()` call FOREVER
# (test_animation_export.py::test_plotly_mp4_export sat pytest's full 1200s
# inside kaleido's sync-server call_function on Windows CI). A blocked native/
# browser call cannot be safely interrupted OR reclaimed from a Python thread
# (abandoning the thread and poking kaleido's private singleton state corrupts a
# later export in the same process), so frame rendering runs in a KILLABLE
# SUBPROCESS: it owns its own kaleido singleton + Chrome, the parent enforces a
# PROGRESS-SENSITIVE watchdog, and on a stall the whole process tree (Chrome
# included) is killed and the export retried in a fresh subprocess. A module
# lock serializes exports so at most one render subprocess (and its browser)
# exists at a time.
#
# The watchdog measures PROGRESS, not total elapsed time. A whole-export
# deadline scaled by frame count cannot tell "a long export that is steadily
# rendering" from "Chrome is wedged": the DEFAULT animation is duration=30 x
# frame_rate=30 = ~900 frames, so any per-frame-scaled deadline generous enough
# for a healthy 900-frame export is hours long -- far past pytest's 1200s cap,
# letting a real wedge hang CI for hours. Instead the worker renames each frame
# into place atomically as it finishes, the parent counts completed frames, and
# the inactivity timer resets on every new frame. A wedge is therefore caught in
# STALL_TIMEOUT seconds regardless of frame count, while a healthy export runs as
# long as it needs. A generous absolute ceiling remains as a second safeguard
# against pathological slow-drip progress.
_KALEIDO_EXPORT_ATTEMPTS = 2          # whole-export attempts before giving up
_KALEIDO_STALL_TIMEOUT = 120          # kill if no NEW frame lands in this long
_KALEIDO_POLL_INTERVAL = 2            # progress-poll cadence (seconds)
_KALEIDO_MIN_CEILING = 1800           # floor for the absolute ceiling (s)
_KALEIDO_CEILING_PER_FRAME = 30       # absolute-ceiling budget per frame (s)
_EXPORT_LOCK = threading.Lock()       # one render subprocess at a time


def _frame_snapshots(fig):
    """Yield one static, frameless `go.Figure` snapshot per animation frame in
    `fig.frames` -- a copy of the pristine base (embedded frames cleared,
    play/pause controls hidden) with that frame's layout/data updates applied,
    suitable for rendering to a single image when assembling a GIF/video/SVG
    export. Module-level so the export subprocess can reuse it."""
    import plotly.graph_objects as go
    base = go.Figure(fig)
    base.frames = ()
    base.layout.updatemenus = ()
    for frame in fig.frames:
        snapshot = go.Figure(base)
        if frame.layout:
            snapshot.update_layout(frame.layout)
        if frame.data:
            indices = frame.traces if frame.traces is not None \
                else range(len(frame.data))
            for idx, trace in zip(indices, frame.data):
                snapshot.data[idx].update(trace)
        yield snapshot


def _export_ceiling(n_frames):
    """Absolute wall-clock backstop for one render attempt -- deliberately far
    above the real cost (a frame renders in a few seconds even on a slow 2-core
    runner) so it never trips a healthy export. The PRIMARY guard is the
    progress watchdog in `_wait_with_progress`; this only catches pathological
    slow-drip progress that keeps resetting the inactivity timer."""
    return max(_KALEIDO_MIN_CEILING, n_frames * _KALEIDO_CEILING_PER_FRAME)


def _kill_process_tree(proc):
    """Kill a render subprocess AND its headless-Chrome children. On POSIX the
    subprocess is its own session leader, so the whole group is signalled; on
    Windows ``taskkill /T`` walks the tree. Best-effort and BOUNDED at every
    step -- including `taskkill` itself, which is a subprocess that can stall
    and would otherwise block the recovery path indefinitely."""
    import signal
    import subprocess
    try:
        if os.name == 'nt':
            try:
                killed = subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                    capture_output=True, timeout=15)
                if killed.returncode != 0:
                    # taskkill ran but did NOT kill the tree (access denied, a
                    # partially-exited tree, ...) -- fall back rather than
                    # assuming success and racing a retry against a live worker
                    proc.kill()
            except subprocess.TimeoutExpired:
                proc.kill()          # taskkill itself stalled
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(timeout=15)
    except Exception:  # noqa: BLE001
        pass


def _wait_with_progress(proc, count_completed,
                        stall_timeout=_KALEIDO_STALL_TIMEOUT,
                        ceiling=None, poll=_KALEIDO_POLL_INTERVAL):
    """Wait for `proc`, killing it only when it STOPS MAKING PROGRESS.

    `count_completed()` returns how many frames have finished (the worker
    renames each into place atomically, so the count only ever grows). The
    inactivity timer resets on every new frame, so a wedged headless Chrome is
    caught in `stall_timeout` seconds NO MATTER how many frames the export has
    -- while a healthy long export (the default animation is ~900 frames) runs
    as long as it needs. `ceiling`, if given, is an absolute backstop against
    pathological slow-drip progress.

    Returns the reason the wait ended: ``'exited'`` (the process finished on its
    own), ``'stalled'`` (no new frame within `stall_timeout` -- a wedged
    browser), or ``'ceiling'`` (still progressing, but past the absolute
    backstop). The two kill reasons are reported separately because they need
    different diagnosis: a wedge is a browser fault, while a ceiling hit means
    an export that really is rendering, just far too slowly.
    """
    import time
    start = time.monotonic()
    last_progress = start
    last_count = count_completed()
    while True:
        if proc.poll() is not None:
            return 'exited'
        time.sleep(poll)
        now = time.monotonic()
        count = count_completed()
        if count > last_count:
            last_count = count
            last_progress = now
        if now - last_progress > stall_timeout:
            _kill_process_tree(proc)
            return 'stalled'
        if ceiling is not None and now - start > ceiling:
            _kill_process_tree(proc)
            return 'ceiling'


def _render_frames_via_subprocess(fig, ext, width, height, n_frames):
    """Render every animation frame of `fig` to an image file (format `ext`) in
    a KILLABLE subprocess, guarded by a PROGRESS watchdog -- the only reliable
    way to bound a blocked headless-Chrome call and reclaim it (a Python thread
    cannot). A stalled render (no new frame for `_KALEIDO_STALL_TIMEOUT`) has
    its process tree, Chrome included, killed and the export retried; a
    non-zero exit or short frame count is likewise retried. Retries RESUME:
    frames already rendered are kept and skipped by the worker, so a wedge on
    frame 800 of 900 does not redo 800 successful renders. Returns the per-frame
    image BYTES in frame order, or raises the last error if every attempt fails.
    Exports are serialized (`_EXPORT_LOCK`) so a wedged browser never coexists
    with a fresh one."""
    import glob
    import shutil
    import subprocess
    import tempfile
    ceiling = _export_ceiling(n_frames)
    last_err = None
    # ONE working dir for the whole export (not per attempt) so a retry resumes
    # from the frames that already landed. mkdtemp + ignore_errors rmtree rather
    # than TemporaryDirectory: on Windows a just-killed worker/Chrome can still
    # hold frame files open, and a cleanup exception would REPLACE the
    # TimeoutError we actually want to report.
    workdir = tempfile.mkdtemp(prefix='hypertools-plotly-export-')
    try:
        with _EXPORT_LOCK:
            fig_json = os.path.join(workdir, 'figure.json')
            frames_dir = os.path.join(workdir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            with open(fig_json, 'w') as fh:
                fh.write(fig.to_json())

            def _completed():
                return len(glob.glob(os.path.join(frames_dir, f'*.{ext}')))

            for attempt in range(_KALEIDO_EXPORT_ATTEMPTS):
                err_path = os.path.join(workdir, f'stderr-{attempt}.log')
                # stderr -> file (not a PIPE) so a chatty Chrome can't deadlock
                # on a full pipe buffer while we watch for progress
                with open(err_path, 'wb') as errf:
                    proc = subprocess.Popen(
                        [sys.executable, '-m',
                         'hypertools.plot._kaleido_export_worker',
                         fig_json, frames_dir, ext, str(width), str(height)],
                        stdout=subprocess.DEVNULL, stderr=errf,
                        start_new_session=(os.name != 'nt'))
                    reason = _wait_with_progress(
                        proc, _completed,
                        stall_timeout=_KALEIDO_STALL_TIMEOUT,
                        ceiling=ceiling)
                # EVERY frame present wins, however the worker ended: it may
                # have rendered them all and then wedged during browser
                # TEARDOWN, after the last frame landed. Never discard a
                # complete export because the browser misbehaved on the way out.
                files = sorted(glob.glob(os.path.join(frames_dir, f'*.{ext}')))
                if len(files) == n_frames:
                    out = []
                    for fp in files:
                        with open(fp, 'rb') as fh:
                            out.append(fh.read())
                    return out
                if reason == 'stalled':
                    last_err = TimeoutError(
                        "plotly frame export stalled: no new frame for "
                        f"{_KALEIDO_STALL_TIMEOUT}s (headless Chrome wedged) "
                        f"after {len(files)}/{n_frames} frame(s); killed the "
                        "render subprocess and its browser, retrying")
                    continue
                if reason == 'ceiling':
                    last_err = TimeoutError(
                        "plotly frame export exceeded its absolute ceiling of "
                        f"{ceiling}s while still rendering ({len(files)}/"
                        f"{n_frames} frame(s) done) -- the export is making "
                        "progress but far too slowly; killed the render "
                        "subprocess and its browser, retrying")
                    continue
                if proc.returncode != 0:
                    tail = ''
                    try:
                        with open(err_path, encoding='utf-8',
                                  errors='replace') as fh:
                            tail = fh.read()[-2000:]
                    except OSError:
                        pass
                    last_err = RuntimeError(
                        "plotly frame export subprocess failed (exit "
                        f"{proc.returncode}): {tail}")
                    continue
                last_err = RuntimeError(
                    f"plotly frame export produced {len(files)} of "
                    f"{n_frames} frame image(s)")
                continue
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    raise last_err if last_err is not None else RuntimeError(
        "plotly frame export failed")


@contextlib.contextmanager
def _shared_kaleido_session():
    """Keep ONE kaleido browser session alive for the duration of the block.

    kaleido 1.x launches (and tears down) a full headless-Chrome process for
    EVERY ``to_image`` call unless its global sync server is running. A
    per-frame animation export makes one such call per frame, so a
    60-frame export paid ~60 Chrome cold starts -- ~3s each on a fast
    machine and far more on slow 2-core CI runners, where the plotly
    animated-SVG export blew through pytest's 1200s per-test timeout and
    killed the whole job (CI run 29582796739,
    tests/test_round3.py::test_animated_svg_plotly). Sharing one session
    across all frames removes every cold start after the first.

    Degrades gracefully: a no-op if kaleido is missing or predates the
    sync-server API (kaleido < 1.1, incl. 0.2.x, whose plotly integration
    already keeps a persistent scope), and if a server is already running
    (started by the caller) it is reused and NEVER stopped here. While a
    server is in use, plotly warns on every call that per-call kaleido
    launch options are ignored ("The kopts argument is ignored if using a
    server") -- expected and harmless here (we pass none), so that one
    specific message is suppressed rather than spamming once per frame.
    """
    try:
        import kaleido
        start = kaleido.start_sync_server
        stop = kaleido.stop_sync_server
        server = getattr(kaleido, '_global_server', None)
    except (ImportError, AttributeError):
        yield
        return
    started_here = False
    try:
        if server is None or not server.is_running():
            start(silence_warnings=True)
            started_here = True
    except Exception:
        # server startup is a pure optimization -- fall back to plotly's
        # ordinary per-call rendering rather than failing the export
        yield
        return
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning,
                message=r'The kopts argument is ignored')
            yield
    finally:
        if started_here:
            # This session runs INSIDE the export subprocess, whose whole
            # process tree the parent kills on a wedge -- so a plain stop is
            # safe here (if it ever blocked on a wedged Chrome, the process is
            # killed anyway; nothing in the parent depends on this cleanup).
            stop(silence_warnings=True)


def _export_animation_file(fig, save_path, frame_rate, duration, size):
    """Export a plotly animation to .gif, .png/.apng, or .mp4/.mov/.avi.

    Each frame is rendered to a PNG via kaleido and the sequence is
    assembled with Pillow (gif / animated png) or ffmpeg (video formats).
    """
    import io
    import os
    import subprocess
    import tempfile

    from PIL import Image

    size = size if size is not None else DEFAULT_FIGSIZE
    width, height = int(size[0] * 100), int(size[1] * 100)
    ext = save_path.lower().rsplit('.', 1)[-1]
    n_frames = len(fig.frames)

    # Every frame is rendered in a killable subprocess (see
    # _render_frames_via_subprocess) so a wedged headless Chrome is bounded by a
    # hard deadline and cannot hang the export -- the frame snapshots are built
    # from `fig` inside that subprocess via the module-level `_frame_snapshots`.
    if ext == 'svg':
        # vector export: render each frame as SVG and stitch them into one
        # SMIL-animated SVG
        from .._shared.animated_svg import combine_frames_svg
        frame_bytes = _render_frames_via_subprocess(
            fig, 'svg', width, height, n_frames)
        frame_svgs = [b.decode('utf-8') for b in frame_bytes]
        with open(save_path, 'w') as f:
            f.write(combine_frames_svg(frame_svgs, max(1.0, duration)))
        return

    # exported files contain EVERY animation frame (no subsampling). Frame
    # subsampling is reserved for the interactive-HTML embedding path
    # (_show_sphinx_gallery), where it caps embedded-file size; an exported
    # gif/png/mp4 must never be subsampled or it would play back too fast.
    frame_bytes = _render_frames_via_subprocess(
        fig, 'png', width, height, n_frames)
    images = [Image.open(io.BytesIO(b)).convert('RGB') for b in frame_bytes]

    # per-frame delay is the TRUE inter-frame interval (1000 / frame_rate),
    # tied to the requested framerate -- NOT 1000*duration/n_frames. With the
    # full frame set (n_frames == frame_rate*duration) the two agree, but
    # deriving the delay from frame_rate keeps real-time playback correct and
    # decoupled from the frame count (a regression guard against any future
    # subsample-and-compensate creeping into the export path). Delays
    # cumulatively round onto the format's timing grid (GIF stores delays
    # in CENTIseconds, APNG in milliseconds), mirroring the matplotlib
    # path's _RealTimePillowWriter: a uniform int(1000/30)=33 -> 30 ms GIF
    # delay made every default-framerate gif play ~10% fast (release-1.0
    # audit, D06-gallery-animation-007 / F04-010).
    def _grid_durations(n_frames, grid_ms):
        per_frame_ms = 1000.0 / max(float(frame_rate), 1e-6)
        durations, prev = [], 0
        for i in range(1, n_frames + 1):
            cum = int(round(i * per_frame_ms / grid_ms)) * grid_ms
            durations.append(cum - prev)
            prev = cum
        return durations

    if ext == 'gif':
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=_grid_durations(len(images), 10), loop=0)
    elif ext in ('png', 'apng'):
        # write to a UNIQUE temporary .png and rename onto the requested
        # name -- `save_path[:-5] + '.png'` silently destroyed a
        # pre-existing sibling .png whenever the caller asked for .apng
        # (release-1.0 audit, F09-002; same fix as animate._save_animation)
        target_dir = os.path.dirname(os.path.abspath(save_path))
        fd, tmp_path = tempfile.mkstemp(suffix='.png', dir=target_dir)
        os.close(fd)
        try:
            images[0].save(tmp_path, format='PNG', save_all=True,
                           append_images=images[1:],
                           duration=_grid_durations(len(images), 1),
                           loop=0)
            os.replace(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        fps = max(1, int(round(float(frame_rate))))
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(images):
                img.save(os.path.join(tmpdir, f'frame_{i:04d}.png'))
            subprocess.run(
                ['ffmpeg', '-y', '-framerate', str(fps), '-i',
                 os.path.join(tmpdir, 'frame_%04d.png'),
                 '-pix_fmt', 'yuv420p', save_path],
                check=True, capture_output=True)


def _cube_trace(go, scale=1.0, linewidth_pt=CUBE_LINEWIDTH_PT):
    """hypertools' signature black wireframe cube as a single 3D trace.

    Mirrors matplotlib_backend's plot_cube: 12 edges at +/-scale, black,
    1pt lines.
    Edges are chained with None separators so one trace draws them all.
    """
    s = scale
    edges = [
        # bottom face
        [(-s, -s, -s), (s, -s, -s)], [(s, -s, -s), (s, s, -s)],
        [(s, s, -s), (-s, s, -s)], [(-s, s, -s), (-s, -s, -s)],
        # top face
        [(-s, -s, s), (s, -s, s)], [(s, -s, s), (s, s, s)],
        [(s, s, s), (-s, s, s)], [(-s, s, s), (-s, -s, s)],
        # vertical edges
        [(-s, -s, -s), (-s, -s, s)], [(s, -s, -s), (s, -s, s)],
        [(s, s, -s), (s, s, s)], [(-s, s, -s), (-s, s, s)],
    ]
    xs, ys, zs = [], [], []
    for (x0, y0, z0), (x1, y1, z1) in edges:
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        # boosted so the gl-rendered cube matches the SVG square's ~2px stroke
        # (see _CUBE_GL_WIDTH_BOOST) -- the 2D square uses no boost
        line=dict(color='black',
                  width=linewidth_pt * PT_TO_PX * _CUBE_GL_WIDTH_BOOST),
        showlegend=False, hoverinfo='skip')


def _square_shape(scale=1.0, linewidth_pt=CUBE_LINEWIDTH_PT):
    """hypertools' 2D black square frame (mirrors matplotlib_backend's
    plot_square)."""
    return dict(type='rect', x0=-scale, y0=-scale, x1=scale, y1=scale,
                line=dict(color='black', width=linewidth_pt * PT_TO_PX),
                fillcolor='rgba(0,0,0,0)', layer='below')


def _surface_base_rgb(spec, fallback_rgb):
    """Base RGB for one dataset's surface (GH #109): `spec['color']` if
    given, otherwise the dataset's own drawn color (`fallback_rgb`)."""
    if spec['color'] is not None:
        import matplotlib.colors as mcolors
        return mcolors.to_rgb(spec['color'])
    return fallback_rgb


# `surface['alpha']` at/above this threshold renders through the
# artifact-free OPAQUE path (alpha baked into the base color via
# `_blend_toward_white`, enclosed data points hidden -- they would be
# invisible behind the opaque mesh anyway); anything below it renders a
# GENUINELY translucent Mesh3d so the enclosed data points stay visible,
# exactly like the matplotlib reference behavior (release-1.0 audit,
# F07-001: the point-hiding used to be unconditional, so a translucent
# plotly surface showed no data at all). 0.999 (not 1.0 exactly) because
# plotly's translucent rendering path engages -- speckle artifacts and all,
# see `_blend_toward_white` -- for ANY opacity < 1, so near-1 alphas are
# visually indistinguishable from 1.0 yet would pay its full artifact cost.
SURFACE_OPAQUE_ALPHA = 0.999


def _mesh_layer_opacity(alpha):
    """Per-LAYER Mesh3d opacity for a translucent surface: every face is
    emitted twice (both winding orders -- see `_mesh3d_trace`), so along any
    line of sight the two coincident copies alpha-composite twice. Giving
    each copy ``1 - sqrt(1 - alpha)`` makes the pair composite to exactly
    the requested total `alpha` (``1 - (1 - o)**2 == alpha``), matching the
    matplotlib renderer's single-layer (backface-culled) translucency -- and
    the lower per-layer opacity also softens plotly's depth-sort speckle
    (see `_blend_toward_white`), whose contrast scales with per-layer
    opacity."""
    return 1.0 - float(np.sqrt(max(0.0, 1.0 - float(alpha))))


def _blend_toward_white(color_rgb, alpha):
    """Alpha-composite `color_rgb` over a white background (matching this
    module's `paper_bgcolor='white'`), returning the resulting flat RGB.

    Used to FAKE a translucent look for the FULLY-OPAQUE
    (``alpha >= SURFACE_OPAQUE_ALPHA``) plotly surface mesh (GH #109
    rendering-fix) instead of asking plotly's Mesh3d for real `opacity <
    1`: plotly's WebGL renderer has a documented, currently-unfixed
    depth-sorting limitation for translucent meshes
    (https://github.com/plotly/plotly.py/issues/3554 -- "an overlay of
    multiple transparent surfaces may not perfectly be sorted in depth by
    the webgl API") that manifests as per-triangle speckle/faceting
    on these densely-tessellated smooth-hull meshes for ANY `opacity < 1`,
    even values as close to 1 as 0.999. Baking the requested alpha into an
    always-opaque color sidesteps that rendering path entirely -- plotly's
    own bug report confirms "setting opacity to 1 removes these artifacts".

    Translucent surfaces (``alpha < SURFACE_OPAQUE_ALPHA``) no longer use
    this blend (release-1.0 audit, F07-001): an opaque whitened mesh hides
    the data points it encloses, so they now render with REAL Mesh3d
    opacity (see `_mesh_layer_opacity`) against the dataset's true base
    color -- accepting the (milder, per-layer-opacity-scaled) speckle as
    the price of actually showing the data, exactly like the matplotlib
    reference. The plot() docstring documents the trade-off and recommends
    ``alpha=1.0`` where the artifacts are objectionable.
    """
    # array-aware so it handles BOTH a single (3,) base color and a per-VERTEX
    # (V, 3) color array (QC 2026-07 surface hue-per-vertex): each channel is
    # composited toward white independently, broadcasting over any shape.
    return alpha * np.asarray(color_rgb, dtype=float) + (1.0 - alpha) * 1.0


def _vertexcolor_strings(verts, faces, blended_rgb, view, light_kw):
    """Precomputed per-vertex Blinn-Phong shading (GH #109 round 3), as a
    list of plotly 'rgb(...)' strings -- one per vertex in `verts`, in
    order, suitable for ``go.Mesh3d(vertexcolor=...)``.

    Shades `blended_rgb` (the ALREADY alpha-composited-toward-white base
    color -- see `_blend_toward_white`) with the SAME two-light Blinn-Phong
    model the matplotlib renderer uses (`light_kw` is
    `mpl_lighting_kwargs(spec)`), so plotly's rendered surface matches the
    matplotlib one instead of plotly's own (face-based, doubled-winding-
    incompatible) lighting engine.
    """
    vertexcolor = blinn_phong_vertex_colors(
        verts, faces, blended_rgb, view, **light_kw)
    return [_rgb_string(c) for c in vertexcolor]


def _mesh3d_trace(go, verts, faces, color_rgb, opacity, view, light_kw):
    """A single ``go.Mesh3d`` surface trace (GH #109), lit via precomputed
    per-vertex Blinn-Phong shading (GH #109 round 3) that matches the
    matplotlib renderer's own lighting model exactly, with hypertools'
    verified parameters.

    `opacity` (the user-requested `surface['alpha']`) is handled two ways
    (release-1.0 audit, F07-001):

    - ``opacity >= SURFACE_OPAQUE_ALPHA``: the trace is rendered fully
      opaque (``opacity=1.0``), which plotly's Mesh3d renders correctly
      and without artifacts; the alpha-composite adjustment is baked into
      the base color BEFORE per-vertex shading (`_blend_toward_white`).
    - ``opacity < SURFACE_OPAQUE_ALPHA``: the trace is GENUINELY
      translucent -- per-layer ``opacity = 1 - sqrt(1 - alpha)`` (the
      doubled winding composites twice; see `_mesh_layer_opacity`) against
      the UNblended base color, exactly like the matplotlib path shades
      its own unblended base color and relies on real alpha compositing --
      so the data points the hull encloses stay visible through it.

    Double-sided (GH #109 round 2): plotly's Mesh3d back-face-culls each
    triangle independently against the camera. Our finely-tessellated
    smoothed-hull meshes are only outward-facing ON AVERAGE (verified via
    `face_normals`); Taubin smoothing routinely leaves small, genuinely
    concave dimples (an expected side effect of smoothing an irregular
    point cloud's hull, not a meshing bug -- see `smooth_hull_3d`'s
    docstring), and at some camera angles the dimpled triangles' true
    normals face far enough away from the camera to get culled -- verified
    via a live Chromium render (not just kaleido) that this reproduces on a
    SINGLE, non-overlapping mesh with NO other trace in the scene, so it is
    independent of the mesh-mesh interaction below. The culled triangles
    leave a visible hole clear through to the background (or, when another
    mesh happens to be trimmed/positioned behind it, that mesh's color
    instead) since Mesh3d has no depth-independent "this is definitely
    occluded" fallback. Emitting EVERY face twice, once with each winding
    order, means at least one of the two copies is always front-facing from
    any camera angle, so no triangle can ever go missing.

    GH #109 round 3: round 2's `flatshading=True` (per-face normals) fixed
    the holes but broke the LOOK of both meshes -- plotly computes each
    doubled face's shading from ITS OWN (possibly reversed) normal, so the
    reversed-winding copy renders dark/black wherever the original copy
    faces the light, producing large jagged dark patches wherever the two
    windings' triangles interleave in screen space. Fixed by shading per
    VERTEX instead (`flatshading=False` + precomputed `vertexcolor`, with
    plotly's own lighting engine set to the identity
    (`PLOTLY_IDENTITY_LIGHTING`) so it reproduces those colors verbatim
    rather than re-shading them): both windings of a doubled face share the
    SAME three vertex indices, so they are always colored identically,
    making the dark-patch defect structurally impossible regardless of
    camera angle.
    """
    faces_both_windings = np.vstack([faces, faces[:, [0, 2, 1]]])
    if opacity >= SURFACE_OPAQUE_ALPHA:
        base_rgb = _blend_toward_white(color_rgb, opacity)
        trace_opacity = 1.0
    else:
        base_rgb = np.asarray(color_rgb, dtype=float)
        trace_opacity = _mesh_layer_opacity(opacity)
    vertexcolor = _vertexcolor_strings(verts, faces, base_rgb, view, light_kw)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces_both_windings[:, 0], j=faces_both_windings[:, 1],
        k=faces_both_windings[:, 2],
        vertexcolor=vertexcolor, opacity=trace_opacity, flatshading=False,
        lighting=PLOTLY_IDENTITY_LIGHTING, lightposition=PLOTLY_LIGHTPOSITION,
        hoverinfo='skip', showlegend=False, showscale=False)


def _trim_faces_inside_other_meshes(i, meshes):
    """Boolean keep-mask (len(faces),) for dataset `i`'s mesh: drops faces
    whose centroid falls inside an EARLIER (lower dataset-index) mesh's
    volume (GH #109 rendering-fix; priority rule reworked in round 2).

    When two datasets' surfaces geometrically intersect (e.g. two
    overlapping point-cloud "blobs"), plotly cannot correctly depth-
    composite the two closed opaque ``Mesh3d`` volumes where they overlap
    -- confirmed in round 2 via a live Chromium render of two overlapping,
    fully-opaque meshes with NO trimming at all: the shared volume renders
    as a noisy, jagged interleaving of both colors (WebGL depth-buffer
    z-fighting), not a clean occlusion either way.

    Round 1 traded this off symmetrically (every dataset trimmed against
    every other), which left each mesh's cut boundary equally ragged and,
    since NEITHER side extends fully into the shared region, occasionally
    let a gap open onto the far mesh's own interior. Round 2 instead only
    ever trims a dataset against LOWER-indexed ones: the first (lowest-
    index) dataset in any overlapping cluster is always left completely
    intact, so it is guaranteed to be a closed, complete surface covering
    the whole overlap volume -- later datasets' cut edges are hidden
    against that intact surface instead of against another equally-cut
    one. This does not eliminate every possible camera-angle artifact for
    deep, near-symmetric overlaps (there is no discrete per-face trim that
    does, short of true CSG boolean geometry), but it is a strict
    improvement over mutual trimming for the common case.
    """
    verts_i, faces_i = meshes[i]
    keep = np.ones(len(faces_i), dtype=bool)
    centers = verts_i[faces_i].mean(axis=1)
    for j, (verts_j, _faces_j) in meshes.items():
        if j >= i:
            continue
        keep &= ~points_enclosed(centers, verts_j)
    return keep


def _build_surface_traces_3d(go, data, surface, surface_colors, elev, azim,
                             surface_point_colors=None):
    """Build one ``go.Mesh3d`` trace per dataset with a (non-None,
    non-degenerate) surface spec. Returns ``(traces, dataset_indices,
    meshes)`` where ``dataset_indices[k]`` is the ORIGINAL dataset index
    that produced ``traces[k]`` (datasets with no spec, or too few/
    degenerate points, contribute no trace, so the two lists can be
    shorter than `data`), and ``meshes`` is the ``{dataset_index: (verts,
    faces)}`` dict of every (untrimmed) built mesh, reused by the caller to
    size the axes cube (GH #109 round 2 -- see `surface_cube_scale`)
    without rebuilding every mesh a second time.

    All of the (non-degenerate) datasets' meshes are built FIRST, then each
    is trimmed against lower-indexed datasets' meshes (see
    `_trim_faces_inside_other_meshes`) before any trace is constructed --
    this needs the full set of meshes up front since datasets can overlap
    pairwise in either direction.
    """
    meshes = {}
    for i, (arr, spec) in enumerate(zip(data, surface)):
        if spec is None:
            continue
        pts = np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :3]
        mesh = build_mesh_3d(pts, spec, dataset_label=f' {i}')
        if mesh is None:
            continue
        meshes[i] = mesh

    view = view_vector(elev, azim)
    traces, dataset_indices = [], []
    for i, (arr, spec) in enumerate(zip(data, surface)):
        if i not in meshes:
            continue
        verts, faces = meshes[i]
        if len(meshes) > 1:
            faces = faces[_trim_faces_inside_other_meshes(i, meshes)]
            if len(faces) == 0:
                continue
        spc = (surface_point_colors[i]
               if surface_point_colors and i < len(surface_point_colors)
               else None)
        if spc is not None:
            # per-VERTEX hue coloring (QC 2026-07): inverse-distance-weighted
            # blend of the enclosed points' colors, one color per mesh vertex.
            pts_i, cols_i = spc
            base_rgb = vertex_colors_from_points(verts, pts_i, cols_i)
        else:
            base_rgb = _surface_base_rgb(spec, surface_colors[i])
        traces.append(_mesh3d_trace(go, verts, faces, base_rgb,
                                    spec['alpha'], view,
                                    mpl_lighting_kwargs(spec)))
        dataset_indices.append(i)
    return traces, dataset_indices, meshes


def _build_surface_traces_2d(go, data, surface, surface_colors):
    """Build one ``go.Scatter(fill='toself')`` smooth outline per dataset
    with a (non-None, non-degenerate) surface spec (GH #109, static 2-D)."""
    traces = []
    for i, (arr, spec) in enumerate(zip(data, surface)):
        if spec is None:
            continue
        pts = np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :2]
        outline = build_outline_2d(pts, spec, dataset_label=f' {i}')
        if outline is None:
            continue
        base_rgb = _surface_base_rgb(spec, surface_colors[i])
        r, g, b = (int(round(255 * c)) for c in base_rgb)
        alpha = spec['alpha']
        # explicitly close the loop (smooth_hull_2d's curve does not repeat
        # its first point) so the underlying path is verifiably closed,
        # even though plotly's fill='toself' would close it visually anyway
        xs = np.append(outline[:, 0], outline[0, 0])
        ys = np.append(outline[:, 1], outline[0, 1])
        traces.append(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            fillcolor=f'rgba({r},{g},{b},{alpha})',
            line=dict(color=f'rgba({r},{g},{b},{min(1.0, alpha + 0.15)})',
                      width=1),
            showlegend=False, hoverinfo='skip'))
    return traces


def _one_density_contour_trace(go, pts, spec, color_rgb, label=""):
    """One ``go.Contour`` heatmap-colored KDE layer (GH #108/#191, 2-D),
    or ``None`` if `pts` is too small/degenerate to fit a KDE."""
    kde = fit_kde(pts, dataset_label=label)
    if kde is None:
        return None
    gridsize = resolve_grid(spec, 2)
    xs, ys, Z, _ = kde_grid_2d(pts, kde, gridsize=gridsize)
    r, g, b = (int(round(255 * c)) for c in color_rgb)
    alpha = min(1.5 * spec['alpha'], 1.0)
    return go.Contour(
        x=xs, y=ys, z=Z,
        contours=dict(coloring='heatmap', showlines=False),
        colorscale=[[0, f'rgba({r},{g},{b},0)'],
                    [1, f'rgba({r},{g},{b},{alpha})']],
        line_width=0, showscale=False, hoverinfo='skip')


def _build_density_traces_2d(go, data, density, density_colors):
    """Build each dataset's (or, with ``per_group=False``, one pooled)
    ``go.Contour`` KDE density layer (GH #108/#191, 2-D)."""
    if density[0] is not None and not density[0].get('per_group', True):
        all_pts = np.vstack([
            np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :2]
            for arr in data])
        trace = _one_density_contour_trace(go, all_pts, density[0],
                                           POOLED_COLOR, label=' (pooled)')
        return [trace] if trace is not None else []
    traces = []
    for i, (arr, spec) in enumerate(zip(data, density)):
        if spec is None:
            continue
        pts = np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :2]
        trace = _one_density_contour_trace(go, pts, spec, density_colors[i],
                                           label=f' {i}')
        if trace is not None:
            traces.append(trace)
    return traces


def _one_density_volume_trace(go, pts, spec, color_rgb, label="", boost=1.0):
    """One ``go.Volume`` KDE iso-surface layer (GH #108/#191, 3-D), or
    ``None`` if `pts` is too small/degenerate to fit a KDE.

    hypertools' 3-D plotly scene fits ALL datasets into a shared [-1, 1]
    cube, so any single dataset's own KDE grid (bounded to just its own
    points) occupies only a modest fraction of that cube. `go.Volume`'s
    WebGL ray-marching renders a low, linearly-scaled opacity as nearly
    invisible at that scale (verified empirically: the naive
    `opacity=min(1.5*alpha, 0.5)` / `opacityscale=[[0,0],[0.3,0.3],[1,1]]`
    combination read as completely blank in real 2-dataset renders). These
    constants are instead tuned so the DEFAULT `alpha=0.2` renders a
    clearly-visible-but-still-subtle glow, confirmed against real evidence
    renders (docs/images/v1.0-seven-features/density_3d_plotly.png):
    `isomin=0.05` (vs. the higher 0.1) exposes more of the outer shells,
    `surface_count=5*levels` (15 at the `levels=3` default) gives finer
    gradation, and the opacity/opacityscale curve reaches
    meaningfully-visible mid-tones well before the peak rather than only
    right at it.

    R2 follow-up (maintainer request): even with the above, plotly's glow
    still read as heavier/denser than matplotlib's airy iso-surface shells
    when the two were compared side by side. `opacity`/`opacityscale`/
    `MAX_VOLUME_OPACITY` (see :func:`~.density.resolve_plotly_volume_params`
    and that constant's docstrings) were retuned further down for more
    transparency, re-verified against the same evidence images, with the
    small-in-scene auto-boost still keeping a separated cluster visible.

    `boost` (see :func:`~.density.density_alpha_boost`, GH #108 round 2) is
    ``1.0`` (a no-op) for a scene-filling dataset -- so a single dataset's
    params are unchanged from before -- and ramps up for a dataset that's
    small relative to the whole scene (e.g. one of several widely-separated
    clusters, jointly scaled into the same shared cube).

    Boosting `opacity`/`surface_count` alone is NOT enough to fix that
    small-in-scene case: hypertools' plotly scatter markers are large,
    fully-opaque, same-colored disks that -- for a small dataset -- cover
    almost its entire on-screen footprint, hiding any density volume drawn
    underneath except for a thin glow peeking out past the markers' edges.
    That glow's visibility is governed by how far out the KDE grid reaches
    (`pad`) and how much opacity its low density values get (`isomin`,
    `opacityscale`), not by the trace's overall `opacity` -- verified
    empirically: rendering an isolated small, separated cluster showed the
    glow stayed invisible even at `opacity` near its ceiling until `pad`
    and the opacity ramp were ALSO widened (see
    :func:`~.density.resolve_plotly_volume_params`). The boosted opacity is
    capped at :data:`~.density.MAX_VOLUME_OPACITY` so the volume never
    becomes fully opaque (the underlying data markers must stay the
    dominant visual element).
    """
    kde = fit_kde(pts, dataset_label=label)
    if kde is None:
        return None
    gridsize = resolve_grid(spec, 3)
    levels = spec.get('levels', DENSITY_DEFAULTS['levels'])
    pad, isomin, opacityscale, opacity, surface_count = (
        resolve_plotly_volume_params(spec['alpha'], levels, boost))
    X, Y, Z, D, _, _ = kde_grid_3d(pts, kde, gridsize=gridsize, pad=pad)
    dmax = D.max()
    if dmax <= 0:
        return None
    color = _rgb_string(color_rgb)
    return go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=(D / dmax).ravel(),
        isomin=isomin, isomax=1.0, surface_count=surface_count,
        opacity=opacity,
        opacityscale=opacityscale,
        colorscale=[[0, color], [1, color]],
        showscale=False, hoverinfo='skip')


def _build_density_traces_3d(go, data, density, density_colors):
    """Build each dataset's (or, with ``per_group=False``, one pooled)
    ``go.Volume`` KDE density layer (GH #108/#191, 3-D).

    Each per-dataset layer's opacity is boosted (GH #108 round 2) by how
    small that dataset's own bounding box is relative to the bounding box
    of the WHOLE scene (all datasets combined) -- see
    :func:`~.density.density_alpha_boost`."""
    if density[0] is not None and not density[0].get('per_group', True):
        all_pts = np.vstack([
            np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :3]
            for arr in data])
        trace = _one_density_volume_trace(go, all_pts, density[0],
                                          POOLED_COLOR, label=' (pooled)',
                                          boost=1.0)
        return [trace] if trace is not None else []
    scene_pts = np.vstack([
        np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :3]
        for arr in data])
    scene_extent = bbox_extent(scene_pts)
    traces = []
    for i, (arr, spec) in enumerate(zip(data, density)):
        if spec is None:
            continue
        pts = np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :3]
        boost = density_alpha_boost(bbox_extent(pts), scene_extent)
        trace = _one_density_volume_trace(go, pts, spec, density_colors[i],
                                          label=f' {i}', boost=boost)
        if trace is not None:
            traces.append(trace)
    return traces


def _degenerate_mesh3d_update(go, point, color_rgb=(0.5, 0.5, 0.5)):
    """A zero-area placeholder ``go.Mesh3d`` geometry update (GH #109):
    used for an animation frame whose current window is too small/degenerate
    to form a real hull, so the trace stays valid (and invisible) rather
    than being dropped (plotly frames cannot vary trace count).

    `vertexcolor` (GH #109 round 3) must be supplied explicitly here too,
    sized to these 4 placeholder vertices -- the base trace's own
    `vertexcolor` array is sized to its (very different) real vertex count,
    and plotly does not broadcast/truncate a stale per-vertex array across a
    frame's new geometry. The actual color is irrelevant (the placeholder
    triangle has zero area, so nothing is ever visibly drawn), but the
    array LENGTH must match `len(v)` or plotly errors.
    """
    v = np.tile(np.asarray(point, dtype=np.float64), (4, 1))
    f = np.array([[0, 1, 2]])
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                     i=f[:, 0], j=f[:, 1], k=f[:, 2],
                     vertexcolor=[_rgb_string(color_rgb)] * len(v))


def _mesh3d_geometry_update(go, verts, faces, color_rgb, opacity, view, light_kw):
    """A ``go.Mesh3d`` geometry-only frame update (x/y/z/i/j/k/vertexcolor),
    doubled to both winding orders like `_mesh3d_trace` (GH #109 round 2):
    frame updates only override the attributes given here, so the base
    trace's `flatshading=False`/`opacity`/`lighting` persist across frames --
    but the i/j/k arrays themselves are replaced wholesale each frame, so
    the double-sided fix must be reapplied here too, or every animated
    frame would silently revert to the single-sided (holes-prone) geometry
    the base (first) frame fixed.

    `vertexcolor` (GH #109 round 3) must ALSO be recomputed every frame,
    for two independent reasons: (1) the mesh's own vertex count/positions
    change every frame for the 'serial'/sliding-window animation modes
    (the base trace's array would be the wrong length), and (2) even in
    'spin' mode (mesh geometry frozen, only the camera orbits) the LIGHTING
    must still be recomputed every frame from the current camera view --
    exactly like the matplotlib renderer recomputes `blinn_phong_colors`
    every spin frame (see `matplotlib_backend._shade_and_cull_3d`) -- via
    `view`, the direction-towards-camera vector for THIS frame's angle.
    """
    faces_both_windings = np.vstack([faces, faces[:, [0, 2, 1]]])
    # match _mesh3d_trace's opacity handling (release-1.0 audit, F07-001):
    # a translucent base trace carries REAL Mesh3d opacity (which persists
    # across frame updates), so its per-frame vertexcolor must be computed
    # from the UNblended base color; only the fully-opaque path bakes the
    # alpha into the color.
    if opacity >= SURFACE_OPAQUE_ALPHA:
        base_rgb = _blend_toward_white(color_rgb, opacity)
    else:
        base_rgb = np.asarray(color_rgb, dtype=float)
    vertexcolor = _vertexcolor_strings(verts, faces, base_rgb, view, light_kw)
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces_both_windings[:, 0], j=faces_both_windings[:, 1],
                     k=faces_both_windings[:, 2], vertexcolor=vertexcolor)


def _parse_fmt(fmt_str, tkwargs):
    """Convert a matplotlib format string + kwargs into plotly
    (mode, marker symbol, line dash)."""
    mode, symbol, dash, _marker_char = _resolve_fmt(fmt_str, tkwargs)
    return mode, symbol, dash


def _resolve_fmt(fmt_str, tkwargs):
    """As `_parse_fmt`, but also returns the resolved single matplotlib
    marker CHARACTER (e.g. '.', 'o', 's'), or ``None`` if no marker is
    drawn at all -- so callers can look up character-specific size scaling
    (`_DOT_MARKER_CHARS`/`_DOT_MARKER_SCALE`) using the exact same
    character `_parse_fmt` used to pick the plotly symbol (explicit
    `marker=` kwarg takes priority over `fmt_str`, matching matplotlib)."""
    fmt_str = fmt_str or ''
    symbol = 'circle'
    dash = 'solid'
    has_marker = False
    has_line = False
    marker_char = None

    # explicit marker/linestyle kwargs take priority (matplotlib behavior)
    kw_marker = tkwargs.get('marker')
    kw_linestyle = tkwargs.get('linestyle')

    for ls, dash_name in _DASH_STYLES:
        if ls in fmt_str:
            has_line = True
            dash = dash_name
            fmt_str = fmt_str.replace(ls, '')
            break
    for ch in fmt_str:
        if ch in _MARKER_SYMBOLS:
            has_marker = True
            symbol = _MARKER_SYMBOLS[ch]
            marker_char = ch
            break

    if kw_marker is not None and kw_marker in _MARKER_SYMBOLS:
        has_marker = True
        symbol = _MARKER_SYMBOLS[kw_marker]
        marker_char = kw_marker
    if kw_linestyle is not None:
        has_line = True
        dash = _LINESTYLE_NAMES.get(kw_linestyle, 'solid')

    if has_marker and has_line:
        mode = 'lines+markers'
    elif has_marker:
        mode = 'markers'
    else:
        mode = 'lines'
    return mode, symbol, dash, marker_char


def _marker_size_px(markersize_pt, marker_char, ndims=2):
    """Convert an mpl `markersize` (points, diameter) to the `marker.size`
    value to pass to a plotly trace, matching matplotlib's rendered pixel
    diameter at hypertools' shared 100-dpi canvas.

    Applies the `_DOT_MARKER_SCALE` discount when `marker_char` is a
    '.'/',' (see that constant's docstring), then -- for `ndims >= 3`
    (`go.Scatter3d`, used for every 3-D data/trail/morph trace) -- divides
    by `_SCATTER3D_SIZE_FACTOR` to correct for Scatter3d's different
    `marker.size` -> rendered-pixel-diameter relationship (see that
    constant's docstring). `go.Scatter` (`ndims` 1 or 2) needs no such
    correction."""
    scale = _DOT_MARKER_SCALE if marker_char in _DOT_MARKER_CHARS else 1.0
    px = float(markersize_pt) * PT_TO_PX * scale
    if ndims >= 3:
        px /= _SCATTER3D_SIZE_FACTOR
    return px


def _colorbar_trace(go, colorbar_info, ndims, legend_present):
    """A hidden ("phantom") marker trace whose sole purpose is to carry a
    plotly colorbar (GH #100) -- plotly attaches colorbars to a trace's
    `marker`/`line`, not to the figure directly, so a real (invisible: a
    single ``None``-positioned point, `opacity=0`) trace is the standard
    way to show one without any visible marker of its own. `location`
    controls which side of the plot the colorbar sits on; the default
    ('right') is pushed further right than an existing legend so the two
    never overlap (mirrors the matplotlib backend's `_add_right_colorbar`)."""
    location = colorbar_info.get('location', 'right')
    # x: horizontal anchor for a vertical colorbar (location in
    # ('left', 'right')); orientation='h' + y for a horizontal one (top/bottom)
    if location == 'right':
        x, xanchor, orientation, y, yanchor = (
            1.25 if legend_present else 1.02, 'left', 'v', 0.5, 'middle')
    elif location == 'left':
        x, xanchor, orientation, y, yanchor = -0.15, 'right', 'v', 0.5, 'middle'
    elif location == 'top':
        x, xanchor, orientation, y, yanchor = 0.5, 'center', 'h', 1.15, 'bottom'
    else:  # 'bottom'
        x, xanchor, orientation, y, yanchor = 0.5, 'center', 'h', -0.15, 'top'

    cb = dict(x=x, xanchor=xanchor, y=y, yanchor=yanchor,
             orientation=orientation, len=0.75,
             thickness=15)
    if colorbar_info.get('label'):
        cb['title'] = dict(text=colorbar_info['label'])

    if colorbar_info['kind'] == 'continuous':
        # `continuous_colormap` (not `get_palette_colors`): the continuous
        # value mapping trims cyclic palettes so its endpoints stay
        # distinguishable (release-1.0 audit, F01-013) -- the colorbar must
        # show exactly the colors `mat2colors` assigned to the points.
        from .colors import continuous_colormap
        colors = continuous_colormap(colorbar_info['palette']).colors
        colorscale = _colors_to_plotly_colorscale(colors)
        cmin, cmax = colorbar_info['vmin'], colorbar_info['vmax']
        if colorbar_info.get('ticks') is not None:
            cb['tickvals'] = list(colorbar_info['ticks'])
    else:
        colors = colorbar_info['colors']
        n = len(colors)
        # A VERTICAL discrete colorbar ('right'/'left', orientation='v')
        # must read top-to-bottom in the SAME order as the legend (first
        # group at the TOP) -- plotly's default low-value-at-bottom
        # convention otherwise reverses it relative to the legend (GH #100
        # follow-up). Segment `i` (from the bottom) is built from
        # `colors[n - 1 - i]` (i.e. `colors` reversed) so the FIRST group's
        # color ends up in the TOP segment; `tickvals` are reversed to
        # match (tick for group `g`, at `ticktext[g]`, is placed at the
        # segment that now holds `colors[g]`), so label<->color pairing is
        # unchanged -- only the physical position each group occupies
        # flips. A HORIZONTAL discrete colorbar ('top'/'bottom',
        # orientation='h') already reads left-to-right in legend order
        # (plotly's default), so it is left untouched.
        scale_colors = colors[::-1] if orientation == 'v' else colors
        colorscale = _discrete_plotly_colorscale(scale_colors)
        cmin, cmax = -0.5, n - 0.5
        if colorbar_info.get('ticks') is not None:
            cb['tickvals'] = list(colorbar_info['ticks'])
        else:
            cb['tickvals'] = (list(range(n - 1, -1, -1)) if orientation == 'v'
                              else list(range(n)))
            cb['ticktext'] = [str(l) for l in colorbar_info['labels']]

    marker = dict(color=[cmin], colorscale=colorscale, cmin=cmin, cmax=cmax,
                 showscale=True, colorbar=cb, size=0.001, opacity=0)
    common = dict(mode='markers', marker=marker, hoverinfo='skip',
                 showlegend=False)
    if ndims >= 3:
        return go.Scatter3d(x=[None], y=[None], z=[None], **common)
    return go.Scatter(x=[None], y=[None], **common)


def _colors_to_plotly_colorscale(colors):
    """(n, 3) RGB array (evenly spaced over [0, 1]) -> a plotly continuous
    colorscale (list of [fraction, 'rgb(...)'] pairs)."""
    colors = np.asarray(colors)
    n = len(colors)
    if n == 1:
        c = _rgb_string(colors[0])
        return [[0.0, c], [1.0, c]]
    return [[i / (n - 1), _rgb_string(c)] for i, c in enumerate(colors)]


def _discrete_plotly_colorscale(colors):
    """(n, 3) RGB array -> a HARD-edged (BoundaryNorm-style) plotly
    colorscale: `n` equal-width segments, each a single flat color, so the
    colorbar shows `n` distinct blocks rather than a gradient."""
    colors = np.asarray(colors)
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        s = _rgb_string(c)
        scale.append([i / n, s])
        scale.append([(i + 1) / n, s])
    return scale


def _rgb_string(c):
    """(r, g, b) floats in [0, 1] -> plotly 'rgb(...)' string."""
    r, g, b = (int(round(255 * float(v))) for v in np.asarray(c)[:3])
    return f'rgb({r},{g},{b})'


def _segment_traces_2d(go, pts, colors, width, dash, name):
    """Per-segment colored 2D line, emitted as one small trace per segment
    (plotly's 2D Scatter lines accept only a single color per trace)."""
    segs = []
    for j in range(len(pts) - 1):
        segs.append(go.Scatter(
            x=pts[j:j + 2, 0], y=pts[j:j + 2, 1], mode='lines',
            line=dict(color=colors[j], width=width, dash=dash),
            showlegend=False, hoverinfo='skip',
            legendgroup=name or 'multicolor'))
    return segs


def _to_plotly_color(color, alpha=None):
    if color is None:
        return None
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    a = 1.0 if alpha is None else float(alpha)
    return f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a})'


def _trace_name(legend, tkwargs, i):
    label = tkwargs.get('label')
    if label is not None:
        return str(label)
    if isinstance(legend, (list, tuple)) and i < len(legend):
        return str(legend[i])
    return None


def _camera_eye(elev, azim, r=1.95):
    """Convert matplotlib elev/azim (degrees) to a plotly camera eye."""
    elev_r, azim_r = np.deg2rad(elev), np.deg2rad(azim)
    return dict(
        x=r * np.cos(elev_r) * np.cos(azim_r),
        y=r * np.cos(elev_r) * np.sin(azim_r),
        z=r * np.sin(elev_r),
    )


def _add_animation(fig, data, ndims, animate, frame_rate, duration,
                   rotations, elev, azim, n_data_traces, tail_duration=2,
                   focused=None,
                   chemtrails=None, precog=None, bullettime=None,
                   zoom=1, n_trail_traces=0, trail_trace_start=None,
                   trail_dataset_indices=None,
                   surface=None, surface_colors=None,
                   surface_trace_start=None,
                   surface_dataset_indices=None, data_trace_start=0,
                   morph_tags=None, morph_colors=None, morph_samples=None,
                   morph_trace_start=None, morph_mesh_trace_start=None,
                   morph_surface_spec=None, surface_point_colors=None,
                   morph_sampled=None, morph_dup_masks=None):
    """Attach frames + play controls: 'spin' rotates the camera; True /
    'parallel' reveals trajectories through a sliding time window; 'morph'
    eases the single traveling point-cloud trace (+ mesh, if surfaced)
    built by `plotly_draw` through the Hungarian-matched hold/morph
    schedule (see `hypertools.plot.morph`) while camera eye rotation
    follows `rotations` (scalar: uniform over the whole animation, exactly
    like every other style; list: per-segment, see
    `hypertools.plot.morph.segment_azimuths`). Frames only touch the data
    traces, so the cube/frame stays put.

    `surface`/`surface_colors`/`surface_trace_start`/`surface_dataset_indices`
    (GH #109, 3-D only): if surfaces are in play, each frame ALSO carries a
    full ``go.Mesh3d`` geometry update (x/y/z/i/j/k) per surfaced dataset,
    recomputed from that dataset's CURRENT visible window ('parallel'/
    'serial') or its precomputed full-data mesh ('spin', where only the
    camera moves) -- `surface_trace_start` + an index into
    `surface_dataset_indices` gives that trace's position in `fig.data`.
    `surface_point_colors` (release-1.0 audit, F07-005): the same
    per-dataset ``(points, per_point_rgb)`` hue bundles `plotly_draw`'s
    static path uses -- when present, every frame's mesh update keeps the
    per-vertex hue coloring (windowed to the frame's visible slice for
    'parallel'/'serial') instead of falling back to a flat mean color.

    `trail_trace_start`: the actual `fig.data` index where the trail traces
    (chemtrails/precog/bullettime) begin, as recorded by `plotly_draw`. This
    is NOT always `n_data_traces` -- when `predict=` forecast traces are
    also present they are appended between the data traces and the trail
    traces, so assuming contiguity (`range(n_data_traces + n_trail_traces)`)
    would target the forecast traces instead of the trail traces. Trail
    frame updates always address `range(trail_trace_start,
    trail_trace_start + n_trail_traces)` instead.

    `chemtrails`/`precog`/`bullettime` (GH #127): per-dataset bool lists
    (length `len(data)`, broadcast/validated by `plotly_draw`). Only
    datasets with at least one of the three flags set get a trail trace at
    all -- `trail_dataset_indices[k]` is the ORIGINAL dataset index that
    produced the trail trace at `fig.data[trail_trace_start + k]`, so each
    frame's trail geometry is built from `chemtrails[i]`/`precog[i]`/
    `bullettime[i]` for that SAME original dataset index `i`, not from the
    trail trace's own position `k`.

    `data_trace_start` (GH #108/#191): the actual `fig.data` index where the
    DATA traces begin -- 0, UNLESS a 2-D `density=`/`surface=` layer was
    seeded at the FRONT of `fig.data` (density= is the only one of the two
    that can coexist with `animate`, since surface= 2-D is static-only).
    Density traces themselves are deliberately never referenced by
    `trace_indices` below (nor `surface_trace_indices`): they are computed
    once from the full data and must stay untouched by every frame update.
    """
    import plotly.graph_objects as go

    # EXACTLY match the matplotlib renderer's pacing: frame_rate frames
    # per second of animation for the full duration (no frame cap), so the
    # two backends play at identical speed, duration, and framerate
    n_frames = max(2, int(round(frame_rate * duration)))
    frames = []
    trace_indices = list(range(data_trace_start, data_trace_start + n_data_traces))
    trail_dataset_indices = trail_dataset_indices or []
    chemtrails = chemtrails if chemtrails is not None else [False] * len(data)
    precog = precog if precog is not None else [False] * len(data)
    bullettime = bullettime if bullettime is not None else [False] * len(data)

    surface_dataset_indices = surface_dataset_indices or []
    surface_trace_indices = (
        list(range(surface_trace_start,
                   surface_trace_start + len(surface_dataset_indices)))
        if surface is not None and ndims >= 3 and surface_dataset_indices
        else []
    )

    def _surface_frame_data(windows_by_index, angle,
                            window_colors_by_index=None):
        """One Mesh3d geometry update per surfaced dataset, built from
        `windows_by_index[dataset_idx]` (that dataset's current window),
        shaded (GH #109 round 3) from `angle` -- THIS frame's camera azimuth
        (the camera rotates every frame in every animate mode, matching the
        matplotlib renderer -- see `_shade_and_cull_3d`), so the vertex
        colors stay lit consistently with wherever the camera actually is.

        `window_colors_by_index` (release-1.0 audit, F07-005): optional
        ``{dataset_idx: per_point_rgb}`` dict, windowed to the SAME slice
        as `windows_by_index` -- when present for a dataset, the mesh keeps
        the per-vertex hue coloring static plots use
        (`vertex_colors_from_points`) instead of falling back to one flat
        (for a rainbow hue: gray) mean color."""
        view = view_vector(elev, angle)
        out = []
        for i in surface_dataset_indices:
            window = windows_by_index[i]
            pts = np.atleast_2d(np.asarray(window, dtype=np.float64))[:, :3]
            spec = surface[i]
            base_rgb = _surface_base_rgb(spec, surface_colors[i])
            light_kw = mpl_lighting_kwargs(spec)
            mesh = build_mesh_3d(pts, spec, dataset_label=f' {i}',
                                 quiet=True) if len(pts) >= 4 else None
            if mesh is None:
                pt = pts[-1] if len(pts) else np.zeros(3)
                out.append(_degenerate_mesh3d_update(go, pt, base_rgb))
            else:
                v, f = mesh
                cols = (window_colors_by_index or {}).get(i)
                if cols is not None and len(cols) == len(pts):
                    base_rgb = vertex_colors_from_points(v, pts, cols)
                out.append(_mesh3d_geometry_update(
                    go, v, f, base_rgb, spec['alpha'], view, light_kw))
        return out

    def _window_colors(idx, start, stop):
        """Dataset `idx`'s per-point hue RGB array sliced to the
        ``[start:stop]`` row window (rows are aligned 1:1 with `data[idx]`
        -- both come from the same post-interpolation `xform`; see
        `plot.py`'s `surface_point_colors` construction), or ``None`` if
        that dataset has no per-point hue colors (F07-005)."""
        if not surface_point_colors or idx >= len(surface_point_colors):
            return None
        spc = surface_point_colors[idx]
        if spc is None:
            return None
        return np.asarray(spc[1])[start:stop]

    if animate == 'spin' and ndims >= 3:
        # the FULL dataset is static in 'spin' mode (only the camera
        # rotates) -- precompute each surfaced dataset's mesh once
        spin_meshes = None
        spin_base_rgbs = None
        if surface_trace_indices:
            spin_meshes = [
                build_mesh_3d(
                    np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :3],
                    surface[i], dataset_label=f' {i}', quiet=True)
                for i in surface_dataset_indices
            ]
            # per-vertex hue coloring (F07-005): 'spin' draws the FULL
            # dataset every frame with a frozen mesh, so each surfaced
            # dataset's per-vertex base colors are computed ONCE here (only
            # the per-frame Blinn-Phong shading changes with the camera) --
            # matching the static path's `_build_surface_traces_3d` exactly.
            spin_base_rgbs = []
            for idx, mesh in zip(surface_dataset_indices, spin_meshes):
                flat_rgb = _surface_base_rgb(surface[idx], surface_colors[idx])
                cols = _window_colors(idx, 0, None)
                if mesh is not None and cols is not None:
                    pts_i = np.atleast_2d(
                        np.asarray(data[idx], dtype=np.float64))[:, :3]
                    if len(cols) == len(pts_i):
                        spin_base_rgbs.append(vertex_colors_from_points(
                            mesh[0], pts_i, cols))
                        continue
                spin_base_rgbs.append(flat_rgb)
        for k in range(n_frames):
            angle = azim + 360.0 * rotations * k / n_frames
            frame_kwargs = dict(
                name=str(k),
                layout=dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom)))))
            if surface_trace_indices:
                # GH #109 round 3: the mesh itself is frozen in 'spin' mode
                # (only the camera orbits), but the LIGHTING still must be
                # recomputed every frame from the current camera angle --
                # exactly like the matplotlib renderer's spin animation
                # recomputes `blinn_phong_colors` every frame (see
                # `matplotlib_backend._shade_and_cull_3d`) -- or the
                # rendered surface would stay lit as if the camera were
                # still at the FIRST frame's angle while visibly orbiting.
                view = view_vector(elev, angle)
                surf_data = []
                for j, (idx, mesh) in enumerate(
                        zip(surface_dataset_indices, spin_meshes)):
                    spec = surface[idx]
                    # per-vertex hue colors precomputed once above
                    # (F07-005); flat dataset color otherwise
                    base_rgb = spin_base_rgbs[j]
                    if mesh is None:
                        surf_data.append(_degenerate_mesh3d_update(
                            go, np.zeros(3),
                            _surface_base_rgb(spec, surface_colors[idx])))
                    else:
                        v, f = mesh
                        light_kw = mpl_lighting_kwargs(spec)
                        surf_data.append(_mesh3d_geometry_update(
                            go, v, f, base_rgb, spec['alpha'], view, light_kw))
                frame_kwargs['data'] = surf_data
                frame_kwargs['traces'] = surface_trace_indices
            frames.append(go.Frame(**frame_kwargs))
    elif animate == 'morph' and ndims in (2, 3):
        # Hungarian-matched point-cloud morph (maintainer request): ONE
        # traveling Scatter3d/Scatter trace (+ one Mesh3d trace if surfaced,
        # 3-D only) eases through the hold/morph schedule. In 3-D the camera
        # eye rotates per `rotations` (scalar: uniform over the whole
        # animation; list: per-segment, continuous across boundaries -- see
        # `hypertools.plot.morph.segment_azimuths`). round17 #9 (GH #123):
        # 2-D morphs use a fixed (non-rotating) viewport -- `rotations=` has
        # no camera to drive in 2-D, so segment timing is always even
        # (`rotations=1`, matching `matplotlib_backend.animate_plot2D`'s
        # identical decision -- `plot.py` already warns once if the caller
        # passed a non-default `rotations=`/`zoom=` for 2-D data).
        morph_indices = [i for i, t in enumerate(morph_tags or []) if t]
        n_morph_datasets = len(morph_indices)
        _morph_ncols = 3 if ndims >= 3 else 2
        clouds = [np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :_morph_ncols]
                 for i in morph_indices]
        if morph_sampled is not None and morph_dup_masks is not None:
            # reuse the sampled/matched clouds `plotly_draw` already
            # computed for the static setup (same clouds, same
            # morph_samples) rather than re-running the O(n^3) Hungarian
            # matching a second time per figure (X6-code-org-plot-005)
            sampled, dup_masks = morph_sampled, morph_dup_masks
        else:
            sampled, dup_masks = _morph.sample_and_match_clouds(
                clouds, morph_samples=morph_samples)
        ds_colors = [
            tuple(morph_colors[i]) if morph_colors is not None
            else (0.2, 0.4, 0.8)
            for i in morph_indices
        ]
        if ndims >= 3:
            frame_counts, _, azimuths = _morph.morph_schedule(
                n_morph_datasets, n_frames, rotations, azim)
        else:
            frame_counts, _, azimuths = _morph.morph_schedule(
                n_morph_datasets, n_frames, 1, 0)
        n_frames = sum(frame_counts)

        morph_trace_indices = [morph_trace_start]
        if morph_mesh_trace_start is not None:
            morph_trace_indices.append(morph_mesh_trace_start)

        for k in range(n_frames):
            seg_idx, step, n_steps = _morph.frame_to_segment(frame_counts, k)
            pts = _morph.morph_positions(sampled, seg_idx, step, n_steps)
            color = _morph.morph_color(ds_colors, seg_idx, step, n_steps)
            angle = azimuths[k]

            # full-sample morphs (maintainer request, 2026-07-06 follow-up):
            # on a HOLD frame, the held dataset's own duplicated (padding)
            # points are sliced out of the DRAWN trace -- a per-frame
            # array-length change, which plotly frames support fine -- so
            # alpha compositing looks exactly like a plain plot of that
            # dataset's true points; `pts` itself (fed to the mesh below)
            # stays the FULL n-point cloud, since duplicates never change a
            # convex hull's shape. On a MORPH frame nothing is hidden.
            hide = _morph.morph_visible_mask(dup_masks, seg_idx)
            draw_pts = pts[~hide] if hide is not None else pts

            if ndims >= 3:
                frame_traces = [go.Scatter3d(
                    x=draw_pts[:, 0], y=draw_pts[:, 1], z=draw_pts[:, 2],
                    marker=dict(color=_rgb_string(color)))]
            else:
                frame_traces = [go.Scatter(
                    x=draw_pts[:, 0], y=draw_pts[:, 1],
                    marker=dict(color=_rgb_string(color)))]
            if morph_mesh_trace_start is not None:
                view = view_vector(elev, angle)
                light_kw = mpl_lighting_kwargs(morph_surface_spec)
                mesh = (build_mesh_3d(pts, morph_surface_spec,
                                      dataset_label=' morph', quiet=True)
                       if pts.shape[0] >= 4 else None)
                if mesh is None:
                    pt = pts[-1] if len(pts) else np.zeros(3)
                    frame_traces.append(
                        _degenerate_mesh3d_update(go, pt, color))
                else:
                    v, f = mesh
                    frame_traces.append(_mesh3d_geometry_update(
                        go, v, f, color, morph_surface_spec['alpha'],
                        view, light_kw))

            frame_kwargs = dict(
                name=str(k), data=frame_traces, traces=morph_trace_indices)
            if ndims >= 3:
                frame_kwargs['layout'] = dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            frames.append(go.Frame(**frame_kwargs))
    elif animate == 'serial':
        # datasets appear one at a time, each growing into place while
        # earlier ones stay fully drawn (never connected to each other)
        lengths = [np.atleast_2d(a).shape[0] for a in data]
        total_points = sum(lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        for k in range(n_frames):
            revealed = total_points * k / max(1, n_frames - 1)
            frame_traces = []
            windows_by_index = {}
            window_colors_by_index = {}
            for idx, (arr, start) in enumerate(zip(data, starts)):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                shown = int(np.clip(revealed - start, 0, arr.shape[0]))
                seg = arr[:shown]
                windows_by_index[idx] = seg
                cols = _window_colors(idx, 0, shown)
                if cols is not None:
                    window_colors_by_index[idx] = cols
                if ndims >= 3:
                    frame_traces.append(go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                elif ndims == 2:
                    frame_traces.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                else:
                    frame_traces.append(go.Scatter(
                        x=np.arange(seg.shape[0]), y=seg[:, 0]))
            frame_kwargs = dict(name=str(k), data=frame_traces,
                                traces=list(trace_indices))
            if ndims >= 3:
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(
                    scene_camera=dict(eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if surface_trace_indices:
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _surface_frame_data(
                                            windows_by_index, angle,
                                            window_colors_by_index))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            frames.append(go.Frame(**frame_kwargs))
    else:
        max_len = max(arr.shape[0] for arr in data)
        # focused=/tail_duration= (round17 #8, GH #275): `focused` governs
        # the visible window for `animate='window'` and for any dataset with
        # a chemtrails/precog/bullettime trail; plain `animate=True`/
        # `'parallel'` with no trail flag set anywhere keeps using
        # `tail_duration` alone (unaffected by `focused`), mirroring
        # `matplotlib_backend.animate_plot3D`'s identical `_uses_focus_window`
        # resolution exactly. `focused` may reach this function as `None`
        # when `plotly_draw`/`_add_animation` are called directly (bypassing
        # `plot.py`'s own resolution, which never passes `None` through) --
        # resolved defensively here the same way `chemtrails`/`precog`/
        # `bullettime` are re-broadcast defensively above.
        _focused = focused if focused is not None else tail_duration
        _uses_focus_window = (
            animate == 'window' or any(chemtrails) or any(precog)
            or any(bullettime)
        )
        _window_duration = _focused if _uses_focus_window else tail_duration
        # the visible window covers `_window_duration` seconds of the
        # duration-second animation, matching the matplotlib renderer's
        # window_frames = frame_rate * _window_duration frame window
        window = max(2, int(round(max_len * float(_window_duration)
                                  / max(float(duration), 1e-6))))
        has_trails = n_trail_traces > 0
        if has_trails:
            # trail traces are NOT guaranteed to sit right after the data
            # traces (predict= forecast traces may be appended in between)
            # -- address them by their recorded start index, not by
            # assuming contiguity with n_data_traces.
            trace_indices = list(trace_indices) + list(range(
                trail_trace_start, trail_trace_start + n_trail_traces))
        for k in range(n_frames):
            end = max(2, int(np.ceil((k + 1) * max_len / n_frames)))
            start = max(0, end - window)
            frame_traces = []
            windows_by_index = {}
            window_colors_by_index = {}
            for idx, arr in enumerate(data):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                seg = arr[start:min(end, arr.shape[0])]
                windows_by_index[idx] = seg
                cols = _window_colors(idx, start, min(end, arr.shape[0]))
                if cols is not None:
                    window_colors_by_index[idx] = cols
                if ndims >= 3:
                    frame_traces.append(go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                elif ndims == 2:
                    frame_traces.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                else:
                    frame_traces.append(go.Scatter(
                        x=np.arange(start, start + seg.shape[0]),
                        y=seg[:, 0]))

            # GH #127: trail traces exist (and are updated here) only for
            # datasets in `trail_dataset_indices`, in that SAME ascending
            # order (matching how `plotly_draw` created them, so this stays
            # aligned with the contiguous `trail_trace_start`-based trace
            # range below). Semantics per dataset `idx` mirror the
            # matplotlib renderer exactly: chemtrails AND precog together
            # (or bullettime alone) show the FULL trail; chemtrails alone
            # shows the past window; precog alone shows the future window.
            trail_traces = []
            if has_trails:
                for idx in trail_dataset_indices:
                    arr = np.atleast_2d(np.asarray(data[idx], dtype=np.float64))
                    ct, pc, bt = chemtrails[idx], precog[idx], bullettime[idx]
                    if (ct and pc) or bt:
                        trail = arr
                        t0 = 0
                    elif ct:
                        trail = arr[:start + 1]
                        t0 = 0
                    else:
                        trail = arr[min(end, arr.shape[0]) - 1:]
                        t0 = min(end, arr.shape[0]) - 1
                    if ndims >= 3:
                        trail_traces.append(go.Scatter3d(
                            x=trail[:, 0], y=trail[:, 1], z=trail[:, 2]))
                    elif ndims == 2:
                        trail_traces.append(go.Scatter(
                            x=trail[:, 0], y=trail[:, 1]))
                    else:
                        trail_traces.append(go.Scatter(
                            x=np.arange(t0, t0 + trail.shape[0]),
                            y=trail[:, 0]))
            frame_traces.extend(trail_traces)
            frame_kwargs = dict(name=str(k), data=frame_traces,
                                traces=list(trace_indices))
            if ndims >= 3:
                # matplotlib's sliding-window animation rotates the camera
                # while the window advances (matplotlib_backend's
                # update_lines_parallel); mirror that here
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(
                    scene_camera=dict(eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if surface_trace_indices:
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _surface_frame_data(
                                            windows_by_index, angle,
                                            window_colors_by_index))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            frames.append(go.Frame(**frame_kwargs))

    fig.frames = frames
    frame_ms = max(10, int(1000.0 * duration / n_frames))
    # Play/Pause controls: laid out horizontally BELOW the plotting area
    # (y < 0 in paper coords, anchored by their top edge) rather than at paper
    # (0, 0). In 3-D the scene floats above that corner so the old placement
    # merely looked cramped, but in 2-D the axes fill the paper area and the
    # controls landed ON the plot's bottom-left corner (maintainer report).
    # `margin.b` is opened up in the same call so they are never clipped;
    # update_layout merges nested dicts, so l/r/t margins are preserved.
    # Symmetric `pad` centers each label in its button (the default padding
    # made 'Play' sit noticeably off-center).
    fig.update_layout(
        margin=dict(b=_ANIM_BUTTON_MARGIN_B),
        updatemenus=[dict(
            type='buttons',
            direction='right',
            showactive=False,
            x=0, xanchor='left',
            y=-0.06, yanchor='top',
            pad=dict(l=8, r=8, t=6, b=6),
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(0,0,0,0.22)',
            borderwidth=1,
            font=dict(family=_PLOTLY_SANS_STACK, size=12, color='#2b2b2b'),
            buttons=[
                dict(label='Play', method='animate',
                     args=[None,
                           dict(frame=dict(duration=frame_ms, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0))]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')]),
            ])])
