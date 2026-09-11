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

import contextlib
import itertools
import json
import os
import re
import sys
import threading
import warnings

from .._shared.lazy_import import (lazy_import, ensure_kaleido_chrome,
                                   subprocess_env)
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
    _padded_bounds,
    POOLED_COLOR,
    bbox_extent,
    density_alpha_boost,
    fit_kde,
    kde_grid_2d,
    kde_grid_3d,
    resolve_grid,
    resolve_plotly_volume_params,
)
from .trails import (RunWindow, anim_window_bounds, broadcast_trail_flag,
                     dataset_window_bounds, head_window_frames)
from .._shared.helpers import (UNIT_FRAME_LIMIT, UNIT_FRAME_SCALE,
                               antialias_line, has_line_component,
                               row_index_x)
from . import morph as _morph


def _normalize_scatter3d_alpha(trace, inherited_mode=None, *, frame=False):
    """Keep a Scatter3d's hue when its colours carry transparency.

    Plotly's WebGL line/marker path composites an ``rgba(...)`` colour
    without premultiplying it, so a translucent colour ADDS to the white
    background instead of blending with it: steelblue at alpha 0.5 renders
    as (197, 255, 255), a pale cyan, rather than (162, 192, 217) (notebook
    visual review 2026-09; 1.1 release review). The trace-level `opacity`
    blends correctly, so:

    * UNIFORM alpha (every active colour the same) moves to `opacity`
      verbatim, with the colours made opaque -- true translucency.
    * NONUNIFORM alpha (a translucent line with opaque markers, per-vertex
      alpha, ...) cannot be one `opacity`. The largest alpha becomes the
      trace `opacity`, and every colour is first composited over the white
      paper (`_blend_toward_white`, the rule this module's Mesh3d surfaces
      use) by its share ``alpha / max_alpha`` of it. Over white the result
      is exactly the requested colour; what it cannot express is a
      lower-alpha part showing ANOTHER trace through it at its own lower
      opacity -- the price of drawing the right hue.

    Only active components participate: an unused marker colour must not
    prevent correcting a line. This operation is idempotent and also
    accepts partial frame traces.
    """
    if trace.type != 'scatter3d':
        return
    mode = trace.mode or inherited_mode or 'lines+markers'
    components = []
    for key, token in [('line', 'lines'), ('marker', 'markers')]:
        if token not in mode:
            continue
        obj = getattr(trace, key)
        color = obj.color
        if color is None:
            continue
        scalar = isinstance(color, str)
        values = [color] if scalar else list(color)
        converted, alphas = [], []
        for value in values:
            match = re.fullmatch(r'rgba\(([^,]+),([^,]+),([^,]+),([^,]+)\)',
                                 value.replace(' ', '')) if isinstance(value, str) else None
            if not match:
                converted.append(value)
                alphas.append(1.)
            else:
                converted.append('rgb(' + ','.join(match.groups()[:3]) + ')')
                alphas.append(float(match.group(4)))
        if alphas:
            components.append((obj, converted[0] if scalar else converted, alphas))
    alphas = [a for _, _, values in components for a in values]
    if not alphas:
        return
    top = max(alphas)
    if min(alphas) == top:
        if top == 1 and not frame:
            return
        for obj, color, _ in components:
            obj.color = color
        trace.opacity = (1 if trace.opacity is None else trace.opacity) * top
        return
    # nonuniform: opacity = the largest alpha, each colour pre-blended
    # toward white by its own share of it
    for obj, color, values in components:
        scalar = isinstance(color, str)
        colors = [color] if scalar else list(color)
        blended = []
        for value, alpha in zip(colors, values):
            rgb = _rgb_triplet(value) if isinstance(value, str) else None
            if not isinstance(rgb, tuple) or alpha == top:
                blended.append(value)
                continue
            share = alpha / top if top > 0 else 0.0
            mixed = _blend_toward_white(np.asarray(rgb) / 255.0, share)
            blended.append(_rgb_string(mixed))
        obj.color = blended[0] if scalar else blended
    trace.opacity = (1 if trace.opacity is None else trace.opacity) * top


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
_PLOTLY_SANS_FAMILIES = ('Noto Sans', 'Helvetica Neue', 'Helvetica', 'Arial',
                         'Noto Sans CJK JP', 'Hiragino Sans')
_PLOTLY_GENERIC_TAIL = 'sans-serif'   # CSS generic family; always last resort


def _plotly_font_family(explicit=None, extra=None):
    """Build a CSS ``font-family`` stack string for plotly text surfaces.

    The curated ``_PLOTLY_SANS_FAMILIES`` supply the body of the stack, with
    the generic ``sans-serif`` tail always last. Two optional roles bracket
    them:

    * ``explicit`` -- a caller-supplied ``font=`` family -- LEADS the stack
      (the caller's typography choice wins).
    * ``extra`` -- an auto-detected family filling a real coverage GAP (the
      matplotlib side adds the same family to its own fallback stack) -- is
      appended just before the ``sans-serif`` tail, so a browser resolving the
      stack per glyph uses it only for the glyphs the curated faces lack.

    Family names already present (case-sensitive) are not repeated, so passing
    an ``explicit``/``extra`` that is already curated leaves the stack tidy.
    Every family name is quoted; the generic tail is left bare, as CSS
    requires.
    """
    families = []

    def _add(name):
        if name and name not in families:
            families.append(name)

    _add(explicit)
    for name in _PLOTLY_SANS_FAMILIES:
        _add(name)
    _add(extra)
    quoted = ', '.join('"{}"'.format(name) for name in families)
    return '{}, {}'.format(quoted, _PLOTLY_GENERIC_TAIL)


# The default stack (no explicit face, no gap filler); the single source of
# truth other modules and tests compare against.
_PLOTLY_SANS_STACK = _plotly_font_family()

# Animation Play/Pause control styling. The controls used to sit at paper
# (0, 0) anchored bottom-left, which in 2-D -- where the axes fill the paper
# area -- drew them ON TOP of the plot's bottom-left corner (maintainer report,
# Andy). They now hang BELOW the plotting area, laid out horizontally, with the
# bottom margin opened up so nothing is clipped.
_ANIM_BUTTON_MARGIN_B = 64   # bottom margin reserved for the controls (px)
_ANIM_BUTTON_HEIGHT_PX = 30  # rendered height of the Play/Pause row (px)
_ANIM_BUTTON_GAP_PX = 6      # space above and below that row (px)


def _x_axis_band_px(fig, ndims):
    """Height (px) of what a 2-D figure's x axis draws below the plotting
    area -- tick marks and labels (two lines on a date axis) and the axis
    title -- or 0 when it draws nothing there (3-D; a hidden unit-scale
    axis). The Play/Pause controls go below this band. Generous by a few
    px: 12 px tick labels at plotly's 1.3 line height, 5 px outside ticks,
    and a title placed under the labels with plotly's automatic standoff.
    """
    if ndims >= 3:
        return 0
    axis = fig.layout.xaxis
    if axis is None or axis.visible is False:
        return 0
    band = 0
    if axis.showticklabels:
        lines = 2 if axis.type == 'date' else 1
        band += 5 + 4 + lines * 16
    title = axis.title.text if axis.title is not None else None
    if title:
        band += 30 if band else 22
    return band
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
# DATA lines in 3-D (every Scatter3d line hypertools draws but the cube:
# trajectories, trails, forecasts, truth) are asked for at their true width
# times this. Measured 2026-09-11 in kaleido (ink area / stroke length, a
# straight line, device scale 1 and 2): Scatter3d draws EXACTLY 0.50x the
# requested width from 1.4 to 12 px -- 2.08 px asked, 1.00 drawn -- while
# the SVG 2-D line draws what it is asked for. 1.1 release review (L1): a
# 3-D data line was half as thick as the same line in 2-D and matplotlib.
# The legend key is unaffected (`legend.itemsizing='constant'` draws every
# key line at plotly's fixed width).
_GL_LINE_WIDTH_BOOST = 2.0
#: `plot()`'s documented default `linewidth` for ANIMATIONS (points) --
#: what the matplotlib backend's animators pop (`linewidths = [... .pop(
#: "linewidth", 1)]`); static plots use `DEFAULT_LINEWIDTH_PT`
DEFAULT_ANIM_LINEWIDTH_PT = 1.0

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
# default -- `morph.MORPH_DEFAULT_MARKERSIZE_PT` (4pt), NOT the general
# `DEFAULT_MARKERSIZE_PT` (6.0) used everywhere else. Both backends read
# that one constant. Without matching both that smaller default AND the
# `_DOT_MARKER_SCALE` above, plotly's default morph dots rendered far
# fatter than matplotlib's -- the more severe half of the R2 bug (see
# `docs/images/v1.0-seven-features/morph_anim_plotly.png` before the fix).
# (1.1 visual review, L9: the old shared 1.5pt drew sub-pixel dots here.)
MORPH_DEFAULT_MARKERSIZE_PT = _morph.MORPH_DEFAULT_MARKERSIZE_PT

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
        # installs the [interactive] extra on demand (see _shared.lazy_import)
        lazy_import('plotly', purpose='the plotly backend')
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
                   if any(isinstance(el, (list, tuple)) for el in labels)
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


def _data_axis_layout(label, limit=None, date=False):
    """A VISIBLE plotly axis in the data's own units (GH #285,
    ``axis_scale='data'``).

    The opposite of `_labeled_axis_layout`'s historical default: ticks,
    tick labels and the zero line stay on, because reading real values off
    the axis is the entire point of `axis_scale='data'`. `limit` is an
    explicit ``(low, high)`` range (plotly autoranges when it is None) and
    `date=True` marks the axis as a date axis, for the epoch-millisecond x
    values `ndims=1` series mode emits from a `DatetimeIndex`.
    """
    layout = dict(visible=True, showticklabels=True, showgrid=False,
                  zeroline=False, showline=True, ticks='outside',
                  linecolor='black', mirror=False)
    if label is not None:
        layout['title'] = dict(text=label)
    if limit is not None:
        layout['range'] = ([str(v) for v in _epoch_ms_to_iso(limit)]
                           if date else [limit[0], limit[1]])
    if date:
        layout['type'] = 'date'
    return layout


def _epoch_ms_to_iso(values):
    """Epoch-millisecond x values as NAIVE ISO-8601 date strings.

    `plot()` carries a date x axis as epoch milliseconds internally (every
    stage -- antialiasing, bounds, forecasts -- needs numbers), but plotly.js
    renders a NUMERIC date in the viewer's LOCAL time zone: a series that
    starts 2026-01-01 00:00 drew at 19:00 Dec 31 in New York (1.1 release
    review, measured with kaleido under TZ=UTC vs TZ=America/New_York). A
    naive date STRING is rendered as written, in every time zone -- and the
    hover label then shows the true date. Non-finite or non-numeric
    entries (a legend proxy's ``None``) become ``None``.
    """
    arr = np.asarray(values)
    if arr.dtype.kind in 'iuf':
        num = arr.astype(float).ravel()
    else:
        num = np.array([float(v) if isinstance(v, (int, float, np.integer,
                                                   np.floating))
                        and not isinstance(v, bool) else np.nan
                        for v in arr.ravel()], dtype=float)
    out = np.full(num.shape, None, dtype=object)
    ok = np.isfinite(num)
    if ok.any():
        out[ok] = np.datetime_as_string(
            np.round(num[ok]).astype('int64').astype('datetime64[ms]'),
            unit='ms')
    return out.reshape(arr.shape) if arr.ndim else out


def _dates_as_iso(fig):
    """Rewrite every numeric x of `fig` -- its traces, its animation
    frames' traces and any x range -- from epoch milliseconds to naive ISO
    strings (`_epoch_ms_to_iso`), for a date x axis."""
    def _fix(trace):
        x = getattr(trace, 'x', None)
        if x is not None and len(x):
            trace.x = _epoch_ms_to_iso(x)
    for trace in fig.data:
        _fix(trace)
    for frame in fig.frames:
        for trace in frame.data:
            _fix(trace)
        _xaxis = getattr(frame.layout, 'xaxis', None) if frame.layout else None
        if _xaxis is not None and _xaxis.range is not None:
            _xaxis.range = [str(v) for v in _epoch_ms_to_iso(_xaxis.range)]


def _build_aa_curves(data, fmt, antialias, morph_tags=None):
    """One ``(dense, step)`` pair per dataset, for DRAW-TIME line smoothing.

    Mirrors `matplotlib_backend._draw`'s identical precomputation (see
    `plot`'s ``antialias=``): each LINE-styled dataset gets a dense,
    PCHIP-upsampled copy built ONCE here, so every static trace and every
    animation frame can draw a smooth curve for whatever window of ORIGINAL
    rows it would have shown (`_aa_window`) without re-interpolating.

    Datasets that draw no line are passed through untouched with
    ``step == 1``: `has_line_component` is True only when a linestyle token
    is present (solid/dashed/dotted, with or without a marker), so a
    marker-only 'o'/'.' dataset is never densified and its markers stay on
    the true samples. `animate='morph'` datasets are skipped too -- morph
    draws traveling point CLOUDS, not lines. ``step == 1`` makes every
    window mapping degrade to the raw row slice, so ``antialias=False``
    reproduces the pre-antialias figure exactly.
    """
    curves = []
    for i, arr in enumerate(data):
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        is_morph = morph_tags is not None and i < len(morph_tags) and morph_tags[i]
        if antialias and not is_morph and has_line_component(
                fmt[i] if i < len(fmt) else None):
            curves.append(antialias_line(arr))
        else:
            curves.append((arr, 1))
    return curves


def _aa_window(aa_curves, i, a, b):
    """The smooth polyline to DRAW for dataset `i`'s ORIGINAL-row window
    ``data[i][a:b]``.

    Because `antialias_line` subdivides UNIFORMLY (``dense[::step]`` is
    exactly the original array), a window of original rows maps onto the
    dense curve exactly as ``dense[a * step:(b - 1) * step + 1]``. With
    antialiasing off (or nothing to upsample, ``step == 1``) this is
    literally ``data[i][a:b]``, so the drawn vertices are unchanged.
    """
    dense, step = aa_curves[i]
    if step == 1:
        return dense[a:b]
    if b <= a:
        return dense[0:0]
    return dense[a * step:(b - 1) * step + 1]


def _aa_x(step, start_x, n_drawn):
    """The x positions accompanying a drawn 1-D window of `n_drawn` vertices
    whose first vertex sits at ORIGINAL row index `start_x`.

    1-D plots put the row index on x, so densifying the y values has to
    densify x the same way; `step` dense vertices span one original row.
    ``step == 1`` returns the historical integer `np.arange`, byte-identical
    to the pre-antialias behavior.
    """
    if step == 1:
        return np.arange(start_x, start_x + n_drawn)
    return start_x + np.arange(n_drawn) / step


def _aa_resample_colors(colors, n_orig, n_dense):
    """Per-point colors resampled onto a densified line's parameterization.

    `colors` are plotly color STRINGS (one per original point), so each dense
    vertex takes its NEAREST original point's color rather than a blended
    one. Returns `colors` unchanged when there is nothing to resample.
    """
    if colors is None or n_dense == n_orig:
        return colors
    grid = np.linspace(0, n_orig - 1, n_dense)
    return [colors[j] for j in np.round(grid).astype(int)]


def _observation_vertices(dense, raw, n_rows, aa_step):
    """Indices, into a dataset's drawn (dense) curve, of its TRUE
    observations.

    `dense` is the curve plotly draws: the `n_rows`-row array `plot()` hands
    over, subdivided `aa_step` times per row by `_build_aa_curves`. Those
    rows are the observations themselves unless `plot()` resampled them
    first -- static antialiasing (`plot._interp_static_line`) and the
    animation frame grid (`plot._interp_anim_line`) both do -- and `raw`
    (`plot()`'s pre-resampling ``raw_xform`` rows, in the same display
    space) says where the observations are. Matched from that relationship,
    not from any one grid's arithmetic:

    * EXACT: when every observation is a vertex of the curve (static
      antialiasing keeps each sample as an exact vertex, as does any frame
      grid that contains the samples), those vertices, found in order.
    * NEAREST: otherwise (a frame grid whose rows need not contain the
      samples), the vertex nearest each observation's position along the
      shared uniform parameter, at most half a grid row away -- the rule
      the matplotlib backend's animated markers follow.

    `raw` None (or with as many rows as `n_rows`) means the rows are the
    observations: every `aa_step`-th vertex.
    """
    aa_step = max(int(aa_step), 1)
    n_obs = None if raw is None else np.asarray(raw).shape[0]
    if n_obs is None or n_obs == n_rows or n_obs < 2 or n_rows < 2:
        return np.arange(n_rows) * aa_step
    dense = np.asarray(dense, dtype=np.float64).reshape(len(dense), -1)
    raw = np.asarray(raw, dtype=np.float64).reshape(n_obs, -1)
    if raw.shape[1] == dense.shape[1]:
        # EXACT: walk the curve once, taking each observation's first exact
        # match at or after the previous one's (a doubling search window
        # keeps this linear for any spacing)
        span = float(np.nanmax(np.abs(dense))) if dense.size else 1.0
        tol = 1e-9 * max(span, 1.0)
        found, j = [], 0
        for row in raw:
            hit, width = None, 16
            while j < len(dense):
                window = dense[j:j + width]
                close = np.flatnonzero(np.abs(window - row).max(axis=1)
                                       <= tol)
                if close.size:
                    hit = j + int(close[0])
                    break
                if j + width >= len(dense):
                    break
                width *= 2
            if hit is None:
                found = None
                break
            found.append(hit)
            j = hit + 1
        if found is not None:
            return np.asarray(found, dtype=int)
    # NEAREST: plot()'s resampling grids are uniform in the parameter, so
    # observation k sits at row k * (n_rows - 1) / (n_obs - 1)
    pos = np.arange(n_obs) * ((n_rows - 1) / (n_obs - 1)) * aa_step
    return np.unique(np.rint(pos).astype(int))


def _observation_marker(marker, n_vertices, vertices, ndims):
    """A trace's ``marker=`` dict with a marker at every TRUE OBSERVATION of
    a smoothed line and none at the vertices the smoothing added.

    `plot`'s ``antialias=`` promises that markers render at the true sample
    points; a ``'o-'`` trace drawn as ONE ``lines+markers`` trace over the
    dense curve would otherwise put a marker on every one of its ~900
    vertices, which draws the line as a thick tube of overlapping dots
    (1.1 release review). So the size becomes a per-vertex array: the
    marker's size at `vertices` (`_observation_vertices`, or a plain step
    for a curve whose every `step`-th vertex is a sample) and 0 elsewhere.
    Keeping ONE trace keeps the legend key (line AND marker) and every
    trace index an animation addresses unchanged; an animation frame sends
    the matching slice of this array (`_aa_window_sizes`).

    A per-point size array is what plotly calls a "bubble" trace, which
    changes two of its defaults: markers become 70% opaque and (in 2-D) gain
    a 1 px white outline. Both are pinned back to the scalar-size look here
    (``opacity=1``; ``line.width=0``), so an observation marker is drawn
    exactly as a marker-only trace draws it. The legend key is unaffected:
    `plotly_draw` sets ``legend.itemsizing='constant'``.

    `vertices` may also be an int step (every `step`-th vertex). `marker`
    is returned unchanged when every vertex is an observation (nothing was
    interpolated).
    """
    if marker is None:
        return marker
    if np.isscalar(vertices):
        vertices = np.arange(0, int(n_vertices), max(int(vertices), 1))
    vertices = np.asarray(vertices, dtype=int)
    vertices = vertices[(vertices >= 0) & (vertices < int(n_vertices))]
    if len(vertices) >= int(n_vertices):
        return marker
    marker = _bubble_safe_marker(marker, ndims)
    sizes = np.zeros(int(n_vertices))
    sizes[vertices] = marker.get('size') or 0
    marker['size'] = sizes
    return marker


def _aa_window_sizes(sizes, step, a, b):
    """The slice of a full-curve per-vertex marker-size array that goes with
    `_aa_window`'s drawn window for ORIGINAL rows ``[a, b)`` -- the same
    index arithmetic, so an animation frame's sizes line up with its
    vertices."""
    step = max(int(step), 1)
    if step == 1:
        return sizes[a:b]
    if b <= a:
        return sizes[0:0]
    return sizes[a * step:(b - 1) * step + 1]


def _bubble_safe_marker(marker, ndims):
    """A copy of `marker` that looks the same once its `size` becomes a
    per-point array: plotly's "bubble" defaults (70% opacity and, in 2-D, a
    white outline) pinned back to an ordinary marker's. Also used for the
    base of an animated trace whose FRAMES send such an array."""
    marker = dict(marker)
    marker.setdefault('opacity', 1)
    if ndims < 3:
        marker['line'] = dict(width=0)
    return marker


def _run_window(frame_windows, idx, n_rows, num, total_frames,
                window_frames):
    """This trace's `RunWindow` at one frame.

    `frame_windows` is what `trails.dataset_window_bounds` returned for this
    frame -- ONE clock per source dataset, so `hue=`/`cluster=` runs of one
    dataset reveal in row order -- or None when the caller has no ownership
    mapping (marker-only categorical regrouping, whose traces are not
    datasets). The fallback rebuilds the historical per-trace bounds as a
    `RunWindow` so both paths hand the drawing code one shape, and it is the
    SAME shared `anim_window_bounds` the matplotlib backend falls back to.
    """
    if frame_windows is not None:
        return frame_windows[idx]
    start, end, trail_stop = anim_window_bounds(
        num, total_frames, n_rows, window_frames)
    return RunWindow(start, end, trail_stop, max(0, end - 1), True, n_rows)


#: `title_kwargs=` keys (already alias-resolved by `plot._normalize_title_kwargs`)
#: that plotly's `layout.title` can express, and where each one lands.
#: `fontsize` is in POINTS on the matplotlib side and pixels here, so it goes
#: through the module's PT_TO_PX rule exactly like the default title size.
_PLOTLY_TITLE_PROPS = {
    'fontsize': ('font', 'size'),
    'fontfamily': ('font', 'family'),
    'fontweight': ('font', 'weight'),
    'fontstyle': ('font', 'style'),
    'fontvariant': ('font', 'variant'),
    'color': ('font', 'color'),
    'y': ('y',),
}


def _plotly_title_overrides(title_kwargs):
    """Split `title_kwargs=` into plotly `layout.title` properties (GH #285).

    Returns ``(title_props, font_props)``. Anything plotly's title cannot
    express (matplotlib-only placement/box options such as `pad`, `loc`,
    `backgroundcolor`, `linespacing`) is reported in ONE warning naming the
    backend and the keys, rather than silently rendering a title styled
    differently from the matplotlib one.
    """
    if not title_kwargs:
        return {}, {}
    title_props, font_props, unsupported = {}, {}, []
    for key, value in title_kwargs.items():
        target = _PLOTLY_TITLE_PROPS.get(key)
        if target is None:
            unsupported.append(key)
        elif target[0] == 'font':
            font_props[target[1]] = (round(value * PT_TO_PX)
                                     if target[1] == 'size'
                                     and isinstance(value, (int, float))
                                     else value)
        else:
            title_props[target[0]] = value
    import plotly.graph_objects as go
    valid = getattr(go.layout.title.Font, '_valid_props', None)
    if valid:
        rejected = [k for k in font_props if k not in valid]
        for key in rejected:
            font_props.pop(key)
        unsupported.extend(rejected)
    if unsupported:
        warnings.warn(
            f"backend='plotly' cannot map the following title_kwargs to a "
            f"layout.title property and will ignore them: "
            f"{sorted(unsupported)}. Supported (after alias resolution): "
            f"{sorted(_PLOTLY_TITLE_PROPS)}.",
            UserWarning, stacklevel=3)
    return title_props, font_props


def _plotly_title_text(text):
    """A title string as plotly draws it: newlines become ``<br>``.

    `plot()` promises a title renders identically on both backends, and
    matplotlib breaks a line on ``'\n'``; plotly's title is HTML-ish and
    draws a raw newline as nothing at all (one long line). Applied on
    every plotly title path -- static, per-segment and per-frame dynamic.
    """
    if text is None:
        return None
    return str(text).replace('\n', '<br>')


def _plotly_title_lines(*texts):
    """The most lines any of these plotly title strings needs (``<br>`` or
    ``'\n'`` separated); 1 for nothing at all."""
    n = 1
    for text in texts:
        if text is None:
            continue
        for entry in ([text] if isinstance(text, str) else list(text)):
            if isinstance(entry, str):
                n = max(n, _plotly_title_text(entry).count('<br>') + 1)
    return n


def _title_margin_top(n_lines, size_px, height_px):
    """`layout.margin.t` that keeps an `n_lines`-line title of `size_px`
    off the plotting area (GH #285, 1.1 release review T6).

    The title is anchored by its TOP at ``y=0.97`` of the container and
    grows downward, so the margin has to hold the 3% offset plus one line
    height (1.25 x the font size, plotly's line spacing) per line. The
    historical single-line/default-size case keeps its exact 40px so an
    un-styled figure is byte-identical to before; anything taller or
    larger is measured.
    """
    default_px = round(12 * PT_TO_PX)
    if n_lines <= 1 and size_px <= default_px:
        return 40
    return int(np.ceil(0.03 * height_px + n_lines * 1.25 * size_px + 6))


def _frame_title_dict(text, index, style, segment_colors):
    """One animation frame's `layout.title` (GH #285).

    With no `title_kwargs=`/`title_color=`/`font=` in play this is exactly
    the bare ``dict(text=...)`` plotly frames carried before; otherwise the
    resolved style is re-applied on EVERY frame, because a frame's layout
    patch replaces the title outright.
    """
    text = _plotly_title_text(text)
    if not style and not segment_colors:
        return dict(text=text)
    title = dict(style or {})
    title['text'] = text
    if segment_colors is not None:
        color = (segment_colors[min(index, len(segment_colors) - 1)]
                 if not callable(segment_colors) else None)
        if color is not None:
            title['font'] = {**title.get('font', {}),
                             'color': _to_plotly_color(color)}
    return title


def _plotly_legend_entry_traces(entries, ndims):
    """Invisible traces that exist only to carry an explicit legend entry
    (GH #285) -- plotly's equivalent of matplotlib's proxy `Line2D` handles.

    Used for a matrix/mixture `hue=`'s palette swatches and for
    `legend_colors=[(label, color), ...]`, neither of which corresponds to a
    drawn trace. A 3-D figure needs `Scatter3d` (a 2-D `Scatter` cannot live
    in a `scene`), so the trace type follows `ndims`.
    """
    import plotly.graph_objects as go
    traces = []
    for label, color in entries:
        common = dict(mode='lines', name=str(label), showlegend=True,
                      hoverinfo='skip',
                      line=dict(color=_to_plotly_color(color), width=2))
        if ndims >= 3:
            traces.append(go.Scatter3d(x=[None], y=[None], z=[None],
                                       **common))
        else:
            traces.append(go.Scatter(x=[None], y=[None], **common))
    return traces


def _legend_anchors_for(legend_kwargs):
    """``xanchor``/``yanchor`` that follow a caller's `legend_kwargs` x/y
    when the caller gave a position but no anchor.

    hypertools' default legend is anchored ``xanchor='left',
    yanchor='middle'`` for its x=1.02/y=0.5 spot outside the right edge;
    kept for a caller's ``{'x': 0, 'y': 1}`` those anchors put the legend's
    MIDDLE on the top edge, half of it off the plot (1.1 release review).
    Inside the paper the anchor follows the position by thirds (plotly's
    own ``'auto'`` rule: left/bottom near 0, right/top near 1); outside it,
    the legend hangs away from the plot (x > 1 -> left, x < 0 -> right,
    y > 1 -> bottom, y < 0 -> top). An anchor the caller gave is kept.
    """
    out = {}
    for axis, lo, mid, hi, key in (('x', 'left', 'center', 'right',
                                    'xanchor'),
                                   ('y', 'bottom', 'middle', 'top',
                                    'yanchor')):
        value = legend_kwargs.get(axis)
        if value is None or key in legend_kwargs:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 1:
            out[key] = lo
        elif value < 0:
            out[key] = hi
        elif value <= 1 / 3:
            out[key] = lo
        elif value >= 2 / 3:
            out[key] = hi
        else:
            out[key] = mid
    return out


#: plotly trace types drawn in a 3-D `scene` (every other trace type
#: hypertools can meet in a figure lives on 2-D axes)
_SCENE_TRACE_TYPES = frozenset({'scatter3d', 'mesh3d', 'volume',
                                'isosurface', 'surface', 'cone',
                                'streamtube'})


def _target_ndims(into):
    """3 when an `ax=` target is a 3-D surface, 2 when it is 2-D axes,
    None when there is nothing to tell (an empty figure).

    A `PlotlyCell` knows what it was built for; a bare figure is read from
    the traces it already draws."""
    if isinstance(into, PlotlyCell):
        return 3 if into.ndims >= 3 else 2
    types = {getattr(t, 'type', None) for t in getattr(into, 'data', ())}
    types.discard(None)
    if not types:
        return None
    return 3 if types & _SCENE_TRACE_TYPES else 2


def _compose_scope_traces(into):
    """The traces an `ax=` target already holds that a new call composes
    with: every trace of a bare Figure, or the traces attached to a
    `PlotlyCell`'s own legend; none for a fresh figure."""
    if into is None:
        return []
    if isinstance(into, PlotlyCell):
        key = cell_layout_keys(into.index)['legend']
        return [tr for tr in into.figure.data
                if getattr(tr, 'legend', None) == key
                or (key == 'legend' and getattr(tr, 'legend', None) is None)]
    return list(getattr(into, 'data', ()) or ())


def _rgb_triplet(color):
    """The ``(r, g, b)`` of a plotly colour string, opacity dropped (an
    ``rgba(...)``/``rgb(...)`` string as `_to_plotly_color` builds; any
    other spelling is returned as itself)."""
    text = str(color).strip()
    if text.startswith(('rgba(', 'rgb(')):
        parts = text[text.index('(') + 1:-1].split(',')
        return tuple(round(float(p)) for p in parts[:3])
    return text


def _rgba_with_alpha(color, alpha):
    """`color` (any plotly colour string, typically the ``rgba(r,g,b,a)``
    `_to_plotly_color` builds) with its alpha replaced by `alpha`."""
    text = str(color).strip()
    if text.startswith(('rgba(', 'rgb(')):
        parts = text[text.index('(') + 1:-1].split(',')
        r, g, b = (p.strip() for p in parts[:3])
        return f'rgba({r},{g},{b},{float(alpha)})'
    return _to_plotly_color(color, alpha)


def _forecast_legend_traces(specs, ndims):
    """One data-free legend trace per distinct forecast label -- the plotly
    twin of `hypertools.plot.plot._forecast_legend_handles`.

    `specs` is ``[(label, line, alpha[, mode, marker]), ...]``, one per
    forecast trace that carries a legend label (its model's name), in
    trace order; `line` is the trace's ``line=`` dict (colour with the
    forecast alpha baked in, width, dash), and `mode`/`marker` the
    trace's drawing mode and marker dict when `forecast_fmt=` added
    markers (`_forecast_marker`). Each entry wears the first such forecast's line style,
    and its colour when every forecast under that label shares one (a
    single dataset, or a `forecast_palette=` that colours by model);
    otherwise `forecast.FORECAST_LEGEND_COLOR` at the forecast's alpha --
    the entry then stands for the model's dash, not for any one dataset.
    The traces carry ``meta['hyp_legend_entry'] = <label>`` and NO
    ``hyp_forecast_role`` -- they are legend keys, not forecasts, and a
    reader pairing forecast traces with the matplotlib artists by role
    must not count them.
    """
    import plotly.graph_objects as go
    from .forecast import (FORECAST_LEGEND_COLOR, FORECAST_LEGEND_MIN_ALPHA,
                           group_forecast_labels)
    traces = []
    for label, members in group_forecast_labels([s[0] for s in specs]):
        first_line = specs[members[0]][1]
        # legible whatever the forecasts' own alpha (matplotlib parity:
        # `plot._forecast_legend_handles` floors it the same way)
        alpha = max([FORECAST_LEGEND_MIN_ALPHA]
                    + [float(specs[k][2]) for k in members])
        line = dict(first_line)
        line['width'] = max(float(specs[k][1].get('width') or 0)
                            for k in members) or first_line.get('width')
        # the same colour at different opacities is ONE colour (Codex
        # round 3: alpha=[1, .4] made every all-red key gray)
        if len({_rgb_triplet(specs[k][1].get('color'))
                for k in members}) != 1:
            line['color'] = _to_plotly_color(FORECAST_LEGEND_COLOR, alpha)
        else:
            line['color'] = _rgba_with_alpha(first_line.get('color'), alpha)
        # the key draws the forecasts' markers too (a `forecast_fmt='o:'`)
        mode = specs[members[0]][3] if len(specs[members[0]]) > 3 else 'lines'
        marker = specs[members[0]][4] if len(specs[members[0]]) > 4 else None
        common = dict(mode=mode, name=str(label), showlegend=True,
                      hoverinfo='skip', line=line,
                      meta=dict(hyp_legend_entry=str(label)))
        if marker is not None:
            common['marker'] = dict(marker, color=line['color'])
        if ndims >= 3:
            traces.append(go.Scatter3d(x=[None], y=[None], z=[None],
                                       **common))
        else:
            traces.append(go.Scatter(x=[None], y=[None], **common))
    return traces


def plotly_draw(data, fmt=None, kwargs_list=None, labels=None, legend=None,
                title=None, animate=False, size=None, show=True,
                save_path=None, frame_rate=30, duration=30, rotations=1,
                elev=10, azim=-60, point_colors=None, tail_duration=2,
                focused=None,
                chemtrails=False, precog=False, bullettime=False, zoom=1,
                forecasts=None, forecast_owner=None,
                forecast_overrides=None,
                forecast_schedule=None, forecast_trail=0,
                colorbar_info=None, surface=None,
                surface_colors=None, surface_point_colors=None,
                density=None, density_colors=None,
                morph_tags=None, morph_colors=None, morph_samples=None,
                morph_loop=False,
                dynamic_title=None,
                font=None, font_extra=None, label_alpha=0.5, xlabel=None,
                ylabel=None, zlabel=None, antialias=True, frame_hooks=None,
                segment_titles=None, ownership=None, forecast_reveal=None,
                into=None, title_kwargs=None, title_segment_colors=None,
                legend_kwargs=None, legend_entries=None,
                axis_scale='unit', xlim=None, ylim=None, x_date=False,
                truths=None, forecast_labels=None,
                forecast_datasets=None, datasets_drawn=None,
                legend_explicit=False, raw_data=None, frame_kwargs=None,
                trace_names=None, row_counts=None, before_show=None):
    """Render grouped datasets with plotly, mirroring _draw's contract and
    the matplotlib renderer's appearance.

    Parameters mirror the relevant subset of
    hypertools.plot.matplotlib_backend._draw (D11 audit: every parameter
    listed; the notes further below expand on the non-obvious ones):

    Parameters
    ----------
    data : list of numpy.ndarray
        One (n_i, d) array per trace, d in (1, 2, 3), already centered and
        scaled to [-1, 1] by `plot.py` -- unless `axis_scale='data'`, in
        which case these are the pipeline's own (unscaled) coordinates.
    axis_scale : {'unit', 'data'}
        GH #285. 'unit' (default, and everything before it) draws the frame
        square (half-width `UNIT_FRAME_SCALE`) and pins both 2-D axes to +-`UNIT_FRAME_LIMIT`. 'data' draws no
        square, leaves the axes visible with real ticks, and takes its
        ranges from `xlim`/`ylim` (or plotly's autorange when both are
        None) -- the matplotlib backend's `frame_2d` under plotly's
        vocabulary.
    xlim : (low, high) or None
        Explicit x range, applied under `axis_scale='data'`.
    ylim : (low, high) or None
        Explicit y range, applied under `axis_scale='data'`.
    x_date : bool
        The x values are epoch MILLISECONDS (what `plot()`'s ndims=1 series
        mode emits for a `DatetimeIndex` under the plotly backend); marks
        the x axis `type='date'` so plotly renders real dates, and hands
        every trace x (frames included) and the x range to plotly as naive
        date strings (`_dates_as_iso`), so the figure draws the same dates
        in every viewer's time zone.
    row_counts : list of int or None
        The ORIGINAL row count behind each trace of `data` (`plot()`
        antialiases static lines upstream, so a trace can hold more drawn
        vertices than rows). A 1-D trace puts the row index on x, so its
        vertices -- and the forecast/truth that continue it -- are placed
        in ROW units (`row_index_x`), matching matplotlib's plot1D. `None`
        treats every vertex as a row (the pre-1.1 x).
    truths : list of numpy.ndarray or None
        GH #285. One seam-prepended ACTUAL continuation per drawn trace
        (`plot`'s `truth=`), already in display space. Drawn as one solid,
        fully-opaque, marked trace per dataset, tagged
        ``meta['hyp_forecast_role'] = 'truth'`` -- the plotly half of
        `plot._draw_truth_overlays`.
    trace_names : list of str or None
        The name of every entry of `data` -- what its hover label shows --
        from `plot._plotly_hover_names`: the label its legend entry shows or
        would show under ``legend=True`` (a category for every run of it, a
        hierarchy's top-level group for its leaves, a series' column, else
        the dataset number), or None for a lone unlabelled dataset, which is
        then drawn with no hover name box at all (`_hover_identity`). A name
        shared by several traces becomes their `legendgroup`. Whether a
        legend entry is DRAWN stays decided by `legend`. `None` (a direct
        caller) keeps the historical naming (legend labels only).
    raw_data : list of numpy.ndarray or None
        The PRE-resampling observations, one per entry of `data`, in the
        same display space (`plot()`'s ``raw_xform`` -- the matplotlib
        backend's ``raw_data=``). `plot()` densifies a static line
        (`antialias=`) and resamples an animated one onto its frame grid
        before either backend sees it, so only this says where the true
        observations are: a marker+line fmt (``'o-'``) marks exactly those
        (the nearest frame-grid vertex, in an animation) and never the
        interpolated vertices; a continuous `hue=` line in 1-D/2-D draws its
        markers at these rows. `None` treats the rows of `data` as the
        observations.
    legend_explicit : bool
        Whether `legend_entries` came from a caller's
        ``legend_colors=[(label, color), ...]`` -- an explicit legend that
        the forecast/truth entries stay out of -- rather than from a
        mixture `hue=`'s automatic swatches, which they are added to (Codex
        round 4: mixture legends lost their forecast and truth entries).
    datasets_drawn : int or None
        How many datasets the figure holds once this call's are added --
        recorded as ``layout.meta['hyp_datasets_drawn']`` (before the
        figure is saved or shown, so a displayed figure carries it) for a
        later ``ax=<this figure>`` call to continue the palette from.
        Drawing into a `PlotlyCell` records it per cell instead
        (``layout.meta['hyp_cell_datasets_drawn'][str(index)]``): each cell
        keeps its own count, like a matplotlib axes of its own.
    before_show : callable or None
        Called as ``before_show(fig)`` with the finished figure, before it
        is saved or shown -- so whatever `plot()` records on it (the
        palette colours beside `datasets_drawn`, legend ranks) is in the
        displayed and saved figure too.
    forecast_datasets : list of int or None
        GH #285. Which SOURCE DATASET each forecast belongs to, for
        ``meta['hyp_dataset']``. `None` means "forecast i is dataset i"; the
        multi-model `predict=` form passes a real map, since it draws one
        forecast per (model, dataset) pair.
    forecast_labels : list of str or None
        GH #285. One legend label per forecast, for the multi-model
        `predict=['Kalman', 'ARIMA', ...]` form; the traces are
        `showlegend=True` (once per model) instead of the usual
        `showlegend=False`. `None` keeps every forecast out of the legend.
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
    into : plotly.graph_objects.Figure or None
        Draw INTO this figure (the caller's `ax=` under the plotly backend):
        the traces are appended to it and it is returned; its layout is left
        alone. None (default) builds a new figure. Not allowed with `animate`.
    animate : bool or str
        Animation style (False for static; True/'parallel'/'spin'/
        'serial'/'window'/'morph'). 'serial' composes with the
        chemtrails/precog/bullettime trail flags below, at parity with the
        matplotlib backend (backend parity, Task 4) -- see those params.
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
        Past-trail flag(s), per trace. Applies to `animate=True`/
        `'parallel'` AND `animate='serial'` (the currently-revealing
        dataset traces its own past), matching the matplotlib backend.
    precog : bool or list of bool
        Future-trail flag(s), per trace. Same `animate='serial'` support
        as `chemtrails` above.
    bullettime : bool or list of bool
        Past+future trail flag(s), per trace. Same `animate='serial'`
        support as `chemtrails` above.
    zoom : float
        3-D camera zoom factor, for ANIMATIONS only (as `plot()` documents
        it, and as the matplotlib backend applies it); a static figure keeps
        the default view.
    frame_kwargs : dict or None
        `plot()`'s ``frame_kwargs=`` -- matplotlib keywords for the cube
        (`plot_wireframe`) or square (`Rectangle`) frame, mapped onto the
        plotly frame by `_frame_style` (colour, width, dash, alpha, 2-D
        fill); unmappable keys are named in a warning.
    forecasts : list of numpy.ndarray or None
        predict= forecast traces (see below). ONE PER INPUT DATASET, which
        after `hue=`/`cluster=` regrouping is NOT one per drawn trace.
    forecast_overrides : list of dict or None
        One `forecast_*=` override per dataset
        (`forecast.resolve_forecast_overrides`), or `None` for pure
        inheritance.
    forecast_owner : list of int or None
        `forecast_owner[i]` is the index in `data` of the run that forecast
        `i` continues -- the run holding dataset `i`'s last observation, and
        so the run whose style it inherits. `None` when no regrouping
        happened, in which case forecast `i` continues run `i`.
    forecast_schedule : hypertools.plot.forecast.ForecastSchedule or None
        Every forecast a TIME-PROGRESSING animation will draw, precomputed by
        `plot()` and already mapped into the display box (see below). `None`
        for static plots and `animate='spin'`, which draw the frozen
        full-history `forecasts` overlay instead.
    forecast_trail : int
        Past forecasts kept on screen as a fading fan (`forecast_trail=`),
        0 (the default) for none. Only meaningful with `forecast_schedule`.
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
    morph_loop : bool
        `plot`'s ``loop=``: close an ``animate='morph'`` sequence by
        returning to the FIRST cloud, reusing its sampled points (GH #285).
    dynamic_title : dict or None
        Where a callable / ``{index...}`` `title=` leaves each frame's text
        (GH #285). `plot` registers an internal frame updater that writes
        ``dynamic_title['text']`` while `frame_hooks.dispatch` runs, and
        each frame-build branch below reads it back into that frame's
        ``layout.title`` -- the updater sees the right frame's context but
        cannot reach the ``go.Frame`` being assembled around it.
    font : matplotlib.font_manager.FontProperties or None
        Resolved font (family name is applied to layout.font; see below).
    font_extra : str or None
        An auto-detected extra family appended to the font stack to cover
        glyphs the resolved font lacks (see the family-stack note below).
    label_alpha : float
        Opacity of the label annotations' background box (default 0.5).
    xlabel : str or None
        x-axis title.
    ylabel : str or None
        y-axis title.
    zlabel : str or None
        z-axis title (3-D only; rejected upstream otherwise).
    antialias : bool
        Whether to smooth every drawn LINE (default True) -- see below.
    frame_hooks : hypertools.plot.animation_context.FrameHooks or None
        The public `on_frame=` registry (plan 1.1 Task 7), created once by
        `plot()` and threaded straight through to `_add_animation`, which
        records each frame's state and dispatches every registered
        callback immediately before that frame is appended to
        `fig.frames` -- see `_add_animation`'s docstring. `None` (the
        default) when no `on_frame=` was requested.
    forecast_reveal : hypertools.plot.forecast.DatasetRevealSchedule or None
        Which of each dataset's ORIGINAL rows are on screen at each frame,
        when `hue=`/`cluster=` regrouping means a dataset spans several
        traces. Used only to colour the animated forecast traces: a live
        forecast wears the colour of the run drawing the head at that frame,
        and a retained `forecast_trail=` member wears the one it was FIT
        with, so a boundary crossing does not repaint the historical fan
        (Decision R3, matching the matplotlib backend). `None` for
        unregrouped figures, where the forecast keeps its build-time colour.
    ownership : hypertools.plot.ownership.TraceOwnership or None
        Which source dataset each drawn trace came from and which of its
        rows (`None` when the traces do not correspond to input datasets --
        marker-only categorical regrouping groups globally by category).
        When given, the animation's head/trail windows come from
        `trails.dataset_window_bounds`, which paces every trace of ONE
        dataset from that dataset's single clock, so a `hue=`/`cluster=`
        regrouped trajectory sweeps once in row order rather than growing
        in several places at once. This is the SAME call the matplotlib
        updaters make, for the reason the `trails` module exists: a window
        arithmetic transcribed separately into this backend once blanked a
        5-row dataset for 9 of its 15 frames.
    title_kwargs : dict or None
        Alias-resolved title styling (`hyp.plot`'s `title_kwargs=`). Its
        size/family/weight/style/colour/y keys map onto `layout.title`
        (see `_plotly_title_overrides`, which warns by name about the
        matplotlib-only ones); applied to both the static title and every
        per-frame title of a serial/morph animation.

    title_segment_colors : list of colors or None
        `hyp.plot`'s `title_color=` in its per-segment form -- one colour
        per `segment_titles` entry, re-applied on every frame. The
        callable form is matplotlib-only and is rejected in `plot()`
        before reaching here.

    legend_kwargs : dict or None
        Extra `layout.legend` properties, merged over hypertools' defaults
        last so the caller's values win.

    legend_entries : list of (label, color) or None
        Explicit legend entries with no drawn trace behind them -- a
        matrix/mixture `hue=`'s palette swatches, or `legend_colors=` given
        as (label, color) pairs. Rendered as data-free traces (plotly's
        equivalent of matplotlib proxy handles); see
        `_plotly_legend_entry_traces`.

    segment_titles : list of str or None
        Per-segment titles for serial-style animations (`_validate_title`'s
        resolved list, plan 1.1 Task 8) -- `None` for every static or
        scalar-title plot (the overwhelmingly common case). Threaded
        straight through to `_add_animation`, which sets each frame's
        `layout.title` from the SAME hold/transition rule the matplotlib
        backend's `_make_title_updater` uses (segment PARITY, never a
        fraction) -- see `_add_animation`'s docstring. Also reserves the
        same top `margin` a scalar `title=` would (task-8 review, margin
        finding) -- `title` itself is None here (the static title text is
        never drawn), but a per-frame title still renders on every hold
        frame and needs the same vertical room or it clips at the canvas
        top edge; see the `margin=` local below.

    `antialias` (see `plot`'s `antialias=`): DRAW-TIME line smoothing, at
    parity with `matplotlib_backend._draw`'s identical option. Each
    line-styled dataset gets a dense, PCHIP-upsampled copy built ONCE
    (`_build_aa_curves`); the static traces and every animation frame then
    draw, for whatever window of ORIGINAL rows they would have shown,
    exactly the corresponding stretch of that smooth curve (`_aa_window`),
    so successive observations are joined by a smoothly bending curve
    instead of a sharp-angled chain of straight segments. The underlying
    `data` rows are deliberately left untouched, so frame pacing, window/row
    index math, `labels=` annotations, hover text, colorbars, `surface=`
    hulls and `density=` layers all keep indexing the REAL observations --
    only the drawn coordinate arrays change. Marker-only styles (e.g. 'o',
    '.') are never touched, so their markers stay on the true samples, and
    `animate='morph'` (traveling point CLOUDS, not lines) is excluded too.
    A marker+line style ('o-') keeps its markers on the true samples as
    well: its one trace gets a per-vertex marker size that is zero at every
    interpolated vertex (`_observation_marker`, located via `raw_data`).
    `antialias=False` reproduces the pre-antialias figure exactly (same
    traces, same frames, same coordinate arrays).

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
    `showlegend=False` trace per dataset, styled to match its source trace
    (`_forecast_style_from`): the SAME colour, width and dash, at half its
    opacity.

    `forecast_schedule`/`forecast_trail` (predict= during a TIME-PROGRESSING
    animation, 1.1): a frozen full-history overlay would show a prediction
    made from data the viewer has not been revealed yet, so when a schedule
    is given the static `forecasts` block above is skipped entirely (the same
    gate `plot.py` applies to the matplotlib overlay) and this function
    creates one EMPTY `live` trace per dataset -- plus `forecast_trail`
    fading `trail` traces per dataset, newest-first, exactly like matplotlib's
    preallocated trail artists -- which `_add_animation` then rewrites every
    frame from the schedule. Every one of them carries
    ``meta['hyp_forecast_role']`` (``'static'``/``'live'``/``'trail'``),
    ``meta['hyp_dataset']``, ``meta['hyp_forecast_age']`` and
    ``meta['hyp_forecast_alpha']``, so forecast traces are identifiable
    without guessing from `dash` (user data drawn with `fmt='--'` is dashed
    too) -- the plotly half of matplotlib's `_hyp_forecast_role` artist tag.

    The OBSERVED data traces are tagged the same way, with
    ``meta['hyp_trace_index']`` -- the index into `data` of the trajectory
    they draw, and the plotly half of matplotlib's `coll._hyp_trace_index`
    (`plot._apply_multicolor_lines`). It exists for the same reason: neither
    `fig.data` nor `ax.collections` is a list of data artists. `fig.data`
    also carries the black wireframe cube, 2-D density/surface layers, the
    forecast overlays above and the colorbar's phantom trace, and NONE of
    those is named, so counting "traces with a `name`" or "all but the last"
    is wrong as soon as any of them is present. Under a continuous `hue=` in
    2-D the tag is also what identifies the many one-segment traces
    (`_segment_traces_2d`) as ONE trajectory.

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
    if animate:
        # an animation's default width is 1 pt (`plot()`'s `linewidth`
        # docstring, and what matplotlib's animators draw), not the static
        # 1.5 -- set per dataset so the head, its trail and its forecast
        # all inherit it; an explicit `linewidth=` still wins
        kwargs_list = [
            dict(kw or {}, linewidth=(kw or {}).get('linewidth')
                 or DEFAULT_ANIM_LINEWIDTH_PT) for kw in kwargs_list]

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

    # `frame_kwargs=`: the cube/square's colour, width, dash (and 2-D fill)
    _frame = _frame_style(frame_kwargs, ndims)

    # ax= (`into`) must be the same KIND of surface as this plot: a 3-D
    # scene for 3-D data, 2-D axes for 1-/2-D data. A mismatched grid cell
    # used to die in `transplant_panel` with a bare "cannot unpack
    # non-iterable NoneType" (its 3-D domain read off 2-D axes, or the
    # reverse), and a mismatched figure silently overlaid a 2-D trace on a
    # 3-D scene -- the plotly half of `plot()`'s matplotlib `ax=` check
    # ("If passing ax and the plot is 3D, ax must also be 3d"), checked
    # before anything is drawn (1.1 release review)
    if into is not None:
        _into_nd = _target_ndims(into)
        if _into_nd is not None and (_into_nd >= 3) != (ndims >= 3):
            _kind = ('cell of a hyp.subplots grid' if isinstance(
                into, PlotlyCell) else 'plotly figure')
            _this = ('3-D' if ndims >= 3
                     else ('time-series (1-D)' if ndims == 1 else '2-D'))
            raise ValueError(
                f"ax= is a {'3-D' if _into_nd >= 3 else '2-D'} {_kind}, but "
                f"this call draws a {_this} plot ({ndims} column"
                f"{'s' if ndims != 1 else ''} after reduction). Pass a "
                f"{'3-D' if ndims >= 3 else '2-D'} target -- "
                f"hyp.subplots(..., ndims={3 if ndims >= 3 else 2}, "
                "backend='plotly') for a grid, or the Figure of a "
                f"{'3-D' if ndims >= 3 else '2-D'} hyp.plot -- or, if the "
                f"data has the dimensions for it, pass "
                f"ndims={3 if _into_nd >= 3 else 2} to draw into this one.")

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
    # every other style: a single-column trajectory has nothing to reveal
    # a path through -- the matplotlib backend's `_draw` refuses it with
    # this exact message, and plotly used to animate it silently, drawing
    # frame-grid row numbers as the x axis (1.1 release review). `ndims=1`
    # SERIES mode is unaffected: `plot()` hands it over as (index, value)
    # columns, a 2-D plot.
    if animate and ndims not in (2, 3):
        raise ValueError(
            "Animations are only supported for 2-D or 3-D plots (got "
            f"{ndims}-D data); pass ndims=2 or ndims=3 (the default).")

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

    # antialias= (see this function's docstring and `plot`'s `antialias=`):
    # build each line-styled dataset's dense, PCHIP-upsampled drawing curve
    # ONCE, here, before any trace is created -- the static traces below and
    # every animation frame in `_add_animation` then slice the SAME curves
    # via `_aa_window`, so a dataset is never interpolated twice and the
    # smoothing is identical across the static figure and its frames.
    aa_curves = _build_aa_curves(data, fmt, antialias, morph_tags=morph_tags)
    # where each dataset's TRUE observations are (see `raw_data` above); a
    # list that does not pair up with `data` is ignored rather than guessed
    if raw_data is not None and len(raw_data) != len(data):
        raw_data = None
    observations = [
        (raw_data[i] if raw_data is not None and raw_data[i] is not None
         and np.asarray(raw_data[i]).ndim > 0 else None)
        for i in range(len(data))]
    # full-curve per-vertex marker sizes for each observation-marked data
    # (and trail) trace, which every animation frame slices to its window
    # (`_aa_window_sizes`); None where the marker size is a plain scalar
    obs_marker_sizes = [None] * len(data)
    #: dataset -> the colour-bin representation of an ANIMATED multicoloured
    #: 1-D/2-D line (`_hue_line_bins`): full-curve x/y, each segment's bin,
    #: and the bins' colours -- what `_add_animation` re-slices every frame
    hue_units = {}
    #: dataset -> its drawn curve's per-vertex line colours, for an
    #: ANIMATED multicoloured 3-D line, whose frames send the window's slice
    hue_colors_3d = {}

    def _rows_of(i, arr):
        """The ORIGINAL row count behind drawn trace `i` (see
        `row_counts`); `arr` is its drawn array."""
        if row_counts is not None and i < len(row_counts):
            return int(row_counts[i])
        return np.atleast_2d(np.asarray(arr)).shape[0]

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
        # `legend_name` decides the legend entry (unchanged rules); `name`
        # is what the trace is CALLED -- its hover label -- which every
        # data trace gets, legend or not (`trace_names`)
        legend_name = _trace_name(legend, tkwargs, i)
        name = (legend_name if legend_name is not None
                else _hover_name(trace_names, i))

        if ndims >= 3 and symbol not in _SYMBOLS_3D:
            symbol = _SYMBOL_3D_FALLBACK.get(symbol, 'circle')

        # multicolored lines: per-point colors along each trajectory.
        #
        # The trace's `alpha=` lives in these per-point colours -- for the
        # LINE and the MARKERS alike, exactly as a single-coloured trace's
        # `alpha=` dims both its line and its markers (the reference every
        # hue path is held to). On matplotlib, `plot._apply_multicolor_lines`
        # gives the segment colours a 4th channel from `tkwargs['alpha']`
        # and `plot._apply_multicolor_markers` scatters with the same alpha;
        # serializing the colours through `_rgb_string` (which drops the 4th
        # channel) with no trace `opacity` is why a hierarchy's 0.7 leaves,
        # and a plain `hue=` + `alpha=`, once rendered fully opaque on plotly
        # alone. (Until the 1.1 release review the markers deliberately
        # kept opaque hue colours, copying a matplotlib path that dropped
        # the alpha; both backends now honour it.) In 3-D the uniform alpha
        # is then moved to the trace's native `opacity` by
        # `_normalize_scatter3d_alpha`, which keeps Scatter3d's hue intact.
        trace_point_colors = None
        trace_line_colors = None
        if point_colors is not None and i < len(point_colors) \
                and point_colors[i] is not None:
            _pt_alpha = tkwargs.get('alpha')
            trace_point_colors = (
                [_rgb_string(c) for c in np.asarray(point_colors[i])]
                if _pt_alpha is None else
                [_to_plotly_color(c, _pt_alpha)
                 for c in np.asarray(point_colors[i])])
            trace_line_colors = trace_point_colors

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
        enclosed_mask = None
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
                    enclosed_mask = enclosed

        # antialias=: draw the dense (smooth) curve for this dataset's FULL
        # row range instead of its raw rows. `arr` itself stays the original
        # rows -- it is what the surface/enclosure logic above and the point
        # count below reason about. Per-point colors are resampled onto the
        # same parameterization so they stay 1:1 with the drawn vertices;
        # a color array that does NOT align with the rows (defensive -- the
        # caller builds both from the same post-interpolation data) disables
        # smoothing for this trace rather than mismatching the two.
        aa_step = aa_curves[i][1]
        if (aa_step != 1 and trace_point_colors is not None
                and len(trace_point_colors) != arr.shape[0]):
            aa_step = 1
        if aa_step == 1:
            draw_arr = arr
        else:
            draw_arr = _aa_window(aa_curves, i, 0, arr.shape[0])
            if enclosed_mask is not None:
                # points an opaque surface hides are dropped from the drawn
                # line (NaN); on the dense curve each vertex follows its
                # NEAREST original point's visibility.
                draw_arr = draw_arr.copy()
                grid = np.linspace(0, arr.shape[0] - 1, draw_arr.shape[0])
                draw_arr[enclosed_mask[np.round(grid).astype(int)]] = np.nan
            # the line and marker colours follow the SAME resampling, so
            # they stay index-aligned with the drawn vertices
            trace_point_colors = trace_line_colors = _aa_resample_colors(
                trace_point_colors, arr.shape[0], draw_arr.shape[0])

        common = dict(
            mode=mode,
            name=name,
            # explicit `legend_entries` (legend_colors=[(label, color)])
            # define the legend outright, so the data traces stay out of
            # it (matplotlib parity; Codex round 3)
            showlegend=(legend is not None and legend_name is not None
                       and not str(legend_name).startswith('_')
                       and not hide_points and not legend_entries),
            **_hover_identity(name, trace_names, ndims),
            visible=not hide_points,
            line=dict(color=color, width=width, dash=dash),
            marker=dict(color=color, size=msize, symbol=symbol),
            # WHICH trace of `data` this draws -- the plotly half of
            # matplotlib's `coll._hyp_trace_index` tag
            # (`plot._apply_multicolor_lines`), and for the same reason:
            # `fig.data` is not a list of data traces. It also carries the
            # black wireframe cube, 2-D density/surface layers, forecast
            # overlays and an invisible colorbar carrier, none of which is
            # named, so "the traces with a name" or "all but the last"
            # miscounts as soon as any of those is present. Tagged
            # positively so a decoration added later cannot leak in.
            meta=dict(hyp_trace_index=i),
        )
        if trace_point_colors is not None:
            # per-point marker colours, in every dimensionality: the 1-D
            # branch used to fall through to the single `color`, so a
            # marker-only continuous hue drew all 60 points in ONE palette
            # colour there while matplotlib's `_apply_multicolor_markers`
            # scattered them per point
            common['marker'] = dict(color=trace_point_colors,
                                    size=msize, symbol=symbol)
        obs_vertices = _observation_vertices(
            arr if aa_step == 1 else aa_curves[i][0], observations[i],
            arr.shape[0], aa_step)
        if 'markers' in mode:
            # a marker on each OBSERVATION of a smoothed line, none on the
            # vertices antialiasing (or the animation frame grid) added --
            # `plot`'s `antialias=` contract
            common['marker'] = _observation_marker(
                common['marker'], draw_arr.shape[0], obs_vertices, ndims)
            if not np.isscalar(common['marker']['size']):
                obs_marker_sizes[i] = common['marker']['size']
        if ndims >= 3:
            if trace_point_colors is not None:
                # Scatter3d supports per-point line colors natively
                common['line'] = dict(color=trace_line_colors, width=width,
                                      dash=dash)
                hue_colors_3d[i] = list(trace_line_colors)
            traces.append(go.Scatter3d(
                x=draw_arr[:, 0], y=draw_arr[:, 1], z=draw_arr[:, 2],
                **common))
            continue
        # 1-D: x in ROW units (`row_counts`, `row_index_x`), matching
        # matplotlib's plot1D -- not the densified vertex index
        xs = (draw_arr[:, 0] if ndims == 2
              else _aa_x(aa_step, 0, draw_arr.shape[0]) if aa_step != 1
              else row_index_x(_rows_of(i, arr), draw_arr.shape[0]))
        ys = draw_arr[:, 1] if ndims == 2 else draw_arr[:, 0]
        if trace_point_colors is not None and 'lines' in mode:
            if animate:
                # an ANIMATED multicoloured 2-D line is re-drawn window by
                # window, so its colours must travel with it: a fixed set
                # of colour-BIN traces (`_hue_bin_units`), each drawing every
                # segment of its colour that the frame's window holds
                # (1.1 release review: one static trace per segment left
                # the whole trajectory on screen, and every frame
                # overwrote segment 0 with the window in one colour)
                _bins = _hue_line_bins(trace_line_colors)
                hue_units[i] = dict(xs=np.asarray(xs, dtype=float),
                                    ys=np.asarray(ys, dtype=float),
                                    bins=_bins, alpha=tkwargs.get('alpha'))
                for _k, _color in enumerate(_bins['colors']):
                    _bx, _by = _binned_polylines(
                        hue_units[i]['xs'], hue_units[i]['ys'],
                        _bins['seg_bin'], _k, 0, len(xs) - 1)
                    traces.append(go.Scatter(
                        x=_bx, y=_by, mode='lines', name=name,
                        showlegend=False, hoverinfo='skip',
                        visible=not hide_points,
                        legendgroup=name or 'multicolor',
                        line=dict(color=_color, width=width, dash=dash),
                        meta=dict(hyp_trace_index=i, hyp_hue_bin=_k)))
            else:
                # 2D Scatter has no per-point line colors; draw short
                # segment traces instead (grouped under one legend entry)
                traces.extend(_segment_traces_2d(
                    go, np.column_stack([xs, ys]), trace_line_colors, width,
                    dash, name, trace_index=i))
            if 'markers' in mode:
                # ... and, for a marker+line fmt ('o-'), the markers as ONE
                # marker-only trace on the observations themselves, each in
                # its own hue colour -- matplotlib's
                # `_apply_multicolor_markers` scatter beside its
                # LineCollection. The segments carry only the line, so
                # without this the markers were silently dropped.
                # (the observation vertices of the drawn curve: exactly the
                # samples for a static plot, whose densified rows keep every
                # one; their colours are the hue's own at those rows)
                _ov = obs_vertices[obs_vertices < len(xs)]
                obs_x, obs_y = xs[_ov], ys[_ov]
                obs_point_colors = [trace_point_colors[j] for j in _ov]
                traces.append(go.Scatter(
                    x=obs_x, y=obs_y, mode='markers', name=name,
                    showlegend=False, visible=not hide_points,
                    **{k: v for k, v in _hover_identity(
                        name, trace_names, ndims).items()
                       if k != 'legendgroup'},
                    legendgroup=name or 'multicolor',
                    marker=dict(color=obs_point_colors, size=msize,
                                symbol=symbol),
                    meta=dict(hyp_trace_index=i)))
                if i in hue_units:
                    # an animation re-draws the observations a window
                    # holds, each in its own colour
                    hue_units[i]['markers'] = dict(
                        vertices=np.asarray(_ov, dtype=int),
                        colors=list(trace_point_colors))
            continue
        traces.append(go.Scatter(x=xs, y=ys, **common))

    n_data_traces = len(traces) - n_surface_traces_2d - n_density_traces_2d

    # predict=: one forecast trace per dataset, styled to match its source
    # trace -- same colour, width and dash, at half its opacity
    # (`_forecast_style_from`; GH #169, matplotlib parity).
    # `forecast_trace_start`/`forecast_trace_specs` do for these traces what
    # `trail_trace_start`/`trail_dataset_indices` (just below) do for the
    # chemtrail traces: record the block's REAL position in `traces` and what
    # each entry draws, so `_add_animation` can rewrite them every frame.
    # `forecast_trace_specs[k]` is the ``(dataset, age)`` pair that produced
    # `traces[forecast_trace_start + k]`; age 0 is the live forecast.
    forecast_trace_start = len(traces)
    forecast_trace_specs = []
    #: parallel to `forecast_trace_specs`: {run -> colour} for each
    #: forecast trace, so a frame can repaint it in the head run's
    #: colour (Decision R3). Empty dict = the colour is pinned.
    forecast_frame_colors = []
    #: ``(label, line, alpha)`` per forecast trace that carries a legend
    #: label -- `_forecast_legend_traces` turns these into one data-free
    #: legend trace per distinct label, appended after every drawn trace
    #: (so the frame-index bookkeeping above is untouched)
    forecast_legend_specs = []
    if forecasts is not None and forecast_schedule is None:
        # Loop over the FORECASTS (one per input dataset), not over `data`
        # (one per drawn RUN). `hue=`/`cluster=` regrouping makes those two
        # counts differ, and looping over runs indexed `forecasts[i]` off
        # the end -- an IndexError that only became reachable once the
        # matplotlib side started keeping forecasts under regrouping.
        for i in range(len(forecasts)):
            # `forecast_owner` names the run this forecast continues; without
            # it, forecast i continues run i (the un-regrouped case).
            src = (forecast_owner[i]
                   if forecast_owner is not None and i < len(forecast_owner)
                   else i)
            src = src if src < len(data) else len(data) - 1
            arr = data[src]
            tkwargs = kwargs_list[src] or {}
            fc = np.atleast_2d(np.asarray(forecasts[i], dtype=np.float64))
            fc_line, fc_alpha = _forecast_style_from(
                tkwargs, fmt[src],
                override=(forecast_overrides[i]
                          if forecast_overrides is not None
                          and i < len(forecast_overrides) else None),
                # a continuous `hue=` draws this run in MANY colours, so the
                # forecast takes the one it starts from (matplotlib parity;
                # see `_hue_anchor_color`). `src` -- the run holding the
                # dataset's last observation -- is the right index into
                # `point_colors` for the same reason it is the right index
                # into `kwargs_list`/`data`.
                anchor_color=_hue_anchor_color(point_colors, src))
            # the forecast's legend entry (its model's name, GH #285) is a
            # separate data-free trace built by `_forecast_legend_traces`
            # from every forecast sharing the label -- so one model over
            # several datasets (several colours) gets ONE neutral entry,
            # not the first dataset's colour posing as the model's. The
            # forecast trace itself never lists.
            fc_name = (forecast_labels[i]
                       if forecast_labels is not None
                       and i < len(forecast_labels) else None)
            fc_mode, fc_marker = _forecast_marker(
                tkwargs, (forecast_overrides[i]
                          if forecast_overrides is not None
                          and i < len(forecast_overrides) else None),
                fc_line['color'], ndims)
            if fc_name is not None:
                forecast_legend_specs.append(
                    (fc_name, fc_line, fc_alpha, fc_mode, fc_marker))
            fc_common = dict(mode=fc_mode, showlegend=False,
                             hoverinfo='skip',
                             line=fc_line,
                             **({} if fc_marker is None
                                else dict(marker=fc_marker)),
                             meta=dict(
                                 hyp_forecast_role='static',
                                 hyp_dataset=(forecast_datasets[i]
                                              if forecast_datasets is not None
                                              else i),
                                 hyp_forecast_age=0,
                                 hyp_forecast_alpha=fc_alpha))
            if fc_name is not None:
                fc_common['name'] = fc_name
            # antialias=: a forecast trace is always a LINE, so smooth it the
            # same way as any other line (matching `plot._draw_forecast_
            # overlays`, which does exactly this on the matplotlib side) --
            # a short forecast (e.g. t+1 = 5 vertices) then draws as a smooth
            # curve rather than a few straight segments. The seam-
            # prepended first point and the final point stay exact, so it
            # still joins the trajectory.
            fc_draw, fc_step = (antialias_line(fc) if antialias else (fc, 1))
            if fc_marker is not None:
                # `forecast_fmt='ro:'` marks the forecast's STEPS (and its
                # seam), not every vertex of the smoothed curve -- which drew
                # the dotted forecast as a solid tube of dots
                fc_common['marker'] = _observation_marker(
                    fc_marker, fc_draw.shape[0], fc_step, ndims)
            if ndims >= 3:
                traces.append(go.Scatter3d(
                    x=fc_draw[:, 0], y=fc_draw[:, 1], z=fc_draw[:, 2],
                    **fc_common))
            elif ndims == 2:
                traces.append(go.Scatter(
                    x=fc_draw[:, 0], y=fc_draw[:, 1], **fc_common))
            else:
                start = _rows_of(src, arr) - 1
                traces.append(go.Scatter(
                    x=_aa_x(fc_step, start, fc_draw.shape[0]),
                    y=fc_draw[:, 0], **fc_common))
            forecast_trace_specs.append((i, 0))
    elif forecast_schedule is not None:
        # A time-progressing animation must draw the forecast made from the
        # history revealed SO FAR, so the full-history overlay above is
        # skipped and these EMPTY traces take its place, rewritten every
        # frame by `_add_animation`. Empty -- not zero-alpha -- is how
        # "nothing to draw here yet" is said, exactly as on the matplotlib
        # side: `trail_alpha` never returns 0, so a stale trace and an
        # unwritten one would otherwise be indistinguishable.
        from .forecast import forecast_alpha, trail_alpha
        n_retained = int(forecast_trail or 0)
        # over the FORECASTS (one per input dataset), not over `data` (one
        # per drawn RUN) -- the same rule the static branch above states, and
        # for the same reason: `hue=`/`cluster=` regrouping makes those
        # counts differ, `meta['hyp_dataset']` is what `_forecast_frame_data`
        # asks the schedule with, and a run index there indexed off the end
        # of the schedule (IndexError on the first frame of every regrouped
        # animated forecast).
        _n_forecasts = len(forecasts) if forecasts is not None else len(data)
        for i in range(_n_forecasts):
            # style from the run this dataset's forecast CONTINUES, exactly
            # as the static branch does
            _src = (forecast_owner[i]
                    if forecast_owner is not None and i < len(forecast_owner)
                    else i)
            _src = _src if _src < len(data) else len(data) - 1
            tkwargs = kwargs_list[_src] or {}
            # the LIVE forecast's alpha for this dataset -- the fan decays
            # from THIS, not from a fixed value, so a trail can never be more
            # opaque than the live forecast it fades from (matplotlib parity)
            from .forecast import forecast_alpha_scale_for
            live_alpha = forecast_alpha(
                tkwargs.get('alpha'),
                # a recoloured forecast keeps its trace's alpha here too
                # (Codex round 3: the animated branch still halved it)
                forecast_alpha_scale_for(
                    forecast_overrides[i]
                    if forecast_overrides is not None
                    and i < len(forecast_overrides) else None))
            # trails FIRST, so the live forecast draws on top of its own fan
            # rather than under it (matplotlib parity)
            for age in list(range(1, n_retained + 1)) + [0]:
                # the declared alpha and the one baked into the rgba string
                # are the SAME float, so a reader of `meta` can trust it
                alpha = trail_alpha(age, n_retained, live_alpha=live_alpha)
                fc_line, alpha = _forecast_style_from(
                    tkwargs, fmt[_src], alpha=alpha,
                    override=(forecast_overrides[i]
                              if forecast_overrides is not None
                              and i < len(forecast_overrides) else None),
                    # same anchor the STATIC branch above takes: under a
                    # continuous hue the run's own `line.color` is the
                    # per-dataset palette colour, which nothing is drawn in.
                    anchor_color=_hue_anchor_color(point_colors, _src))
                fc_mode, fc_marker = _forecast_marker(
                    tkwargs, (forecast_overrides[i]
                              if forecast_overrides is not None
                              and i < len(forecast_overrides) else None),
                    fc_line['color'], ndims)
                fc_common = dict(
                    mode=fc_mode, showlegend=False, hoverinfo='skip',
                    line=fc_line,
                    # each smoothed frame sends a per-vertex size array
                    # marking only the forecast's steps
                    # (`_forecast_frame_data`), so the base marker is made
                    # to look the same under one
                    **({} if fc_marker is None else dict(
                        marker=(_bubble_safe_marker(fc_marker, ndims)
                                if antialias else fc_marker))),
                    meta=dict(
                        hyp_forecast_role='live' if age == 0 else 'trail',
                        hyp_dataset=(forecast_datasets[i]
                                     if forecast_datasets is not None
                                     and i < len(forecast_datasets) else i),
                        hyp_forecast_age=age,
                        hyp_forecast_alpha=alpha))
                if ndims >= 3:
                    traces.append(go.Scatter3d(x=[], y=[], z=[], **fc_common))
                else:
                    traces.append(go.Scatter(x=[], y=[], **fc_common))
                forecast_trace_specs.append((i, age))
                if age == 0 and forecast_labels is not None \
                        and i < len(forecast_labels) \
                        and forecast_labels[i] is not None:
                    # the LIVE forecast's legend entry (static parity)
                    forecast_legend_specs.append(
                        (forecast_labels[i], fc_line, alpha, fc_mode,
                         fc_marker))
                # Decision R3: the colour a live/retained forecast wears is
                # the HEAD RUN's, which changes from frame to frame. Plotly
                # frames carry geometry, so the colour must be resolvable
                # per frame -- precompute what THIS trace would look like
                # continuing each possible run, through the same
                # `_forecast_style_from` the build above uses, so the two
                # cannot express different policies. Empty when the user
                # pinned the colour (forecast_hue=/_cluster=/_palette=):
                # an explicit grouping is fixed for the whole animation.
                _ov = (forecast_overrides[i]
                       if forecast_overrides is not None
                       and i < len(forecast_overrides) else None)
                # A continuous hue pins it too: the forecast's identity is
                # the hue value where its trajectory ENDS, which does not
                # change from frame to frame. Decision R3's per-frame
                # head-run colour stays correct for CATEGORICAL regrouping,
                # where the run colour is what the viewer actually sees.
                # Same rule, same reason, as matplotlib's `_override_colour`
                # -- including that this half is DEFENSIVE: a continuous hue
                # never regroups, so `forecast_reveal` is None and
                # `_forecast_frame_data` never consults this map at all
                # (measured 2026-08-16). The anchor an animated forecast
                # actually wears comes from `anchor_color=` above.
                from .forecast import override_has_color
                # a colour letter in forecast_fmt= pins the colour as an
                # explicit forecast_hue=/palette= does (Codex round 4:
                # 'ro:' forecasts were repainted in the head run's colour)
                _pinned = (override_has_color(_ov)
                           or _hue_anchor_color(point_colors, _src)
                           is not None)
                forecast_frame_colors.append({} if _pinned else {
                    _r: _forecast_style_from(
                        kwargs_list[_r] or {}, fmt[_r],
                        alpha=trail_alpha(
                            age, n_retained,
                            live_alpha=forecast_alpha(
                                (kwargs_list[_r] or {}).get('alpha'),
                                forecast_alpha_scale_for(_ov))),
                        override=_ov)[0].get('color')
                    for _r in range(len(data))})

    # truth= (GH #285): each trace's ACTUAL continuation, drawn beside the
    # forecast it is compared against -- the plotly half of
    # `plot._draw_truth_overlays`, with the same styling policy (the
    # observed trace's colour and width, SOLID, fully opaque, with markers
    # on the observations) and the same `hyp_forecast_role='truth'` tag.
    # Appended AFTER the forecast block and BEFORE the trail traces, and
    # never rewritten per frame: it is what happened, not a prediction being
    # refitted as the reveal advances.
    if truths is not None:
        from .plot import TRUTH_STYLE
        # composing into a figure/cell that already lists a truth entry:
        # one entry covers every call's truth (Codex round 4)
        _truth_already_listed = any(
            ((tr.meta or {}).get('hyp_forecast_role') == 'truth'
             or (tr.meta or {}).get('hyp_legend_entry') == 'truth')
            and tr.showlegend
            for tr in _compose_scope_traces(into))
        # the one 'truth' key stands for EVERY dataset's truth: when they
        # span several colours it is a neutral proxy (added with the
        # forecast keys below), not the first truth trace, which wore
        # dataset 0's colour (1.1 release review, F10; matplotlib parity)
        _truth_key_at = None
        _truth_rgbs = set()
        for i, tr in enumerate(truths):
            src = (forecast_owner[i]
                   if forecast_owner is not None and i < len(forecast_owner)
                   else i)
            src = src if src < len(data) else len(data) - 1
            tr = np.atleast_2d(np.asarray(tr, dtype=np.float64))
            tr_line, _ = _forecast_style_from(
                kwargs_list[src] or {}, fmt[src], alpha=1.0,
                anchor_color=_hue_anchor_color(point_colors, src))
            tr_line = dict(tr_line)
            tr_line['dash'] = 'solid'
            tr_draw, tr_step = (antialias_line(tr) if antialias else (tr, 1))
            # a marker on every OBSERVATION, not on every vertex of the
            # antialiased curve (matplotlib parity: its truth overlay
            # marks the raw rows and draws the smooth line marker-free).
            # Dense vertex `k * step` is where raw row k sits on the curve
            # (the same convention `_aa_x` builds the 1-D x from), so the
            # marker size is a per-vertex array that is 0 everywhere else
            # -- one trace, so a truth stays one trace per dataset.
            # (`_observation_marker` also keeps plotly's bubble defaults --
            # 70% opacity, a white outline in 2-D -- off these markers)
            tr_marker = _observation_marker(
                dict(size=_marker_size_px(TRUTH_STYLE['markersize'],
                                          TRUTH_STYLE['marker'], ndims),
                     color=tr_line.get('color')),
                tr_draw.shape[0], tr_step, ndims)
            tr_common = dict(
                mode='lines+markers',
                showlegend=bool(i == 0
                                and (legend is not None or legend_entries)
                                and not legend_explicit
                                and not _truth_already_listed),
                name='truth', hoverinfo='skip', line=tr_line,
                marker=tr_marker,
                # listed AFTER the forecast entries (which are appended as
                # the last traces), the order the matplotlib legend uses:
                # data, forecasts, truth
                legendrank=1001,
                meta=dict(hyp_forecast_role='truth', hyp_dataset=i,
                          hyp_forecast_age=0, hyp_forecast_alpha=1.0))
            _truth_rgbs.add(_rgb_triplet(tr_line.get('color')))
            if tr_common['showlegend']:
                _truth_key_at = (len(traces), tr_line)
            if ndims >= 3:
                traces.append(go.Scatter3d(x=tr_draw[:, 0], y=tr_draw[:, 1],
                                           z=tr_draw[:, 2], **tr_common))
            elif ndims == 2:
                traces.append(go.Scatter(x=tr_draw[:, 0], y=tr_draw[:, 1],
                                         **tr_common))
            else:
                traces.append(go.Scatter(
                    x=_aa_x(tr_step, _rows_of(src, data[src]) - 1,
                            tr_draw.shape[0]),
                    y=tr_draw[:, 0], **tr_common))
        if _truth_key_at is not None and len(_truth_rgbs) > 1:
            traces[_truth_key_at[0]].showlegend = False
        else:
            _truth_key_at = None

    # low-opacity trail traces for chemtrails (past) / precog (future) /
    # bullettime (both) on window animations, mirroring the matplotlib
    # renderer's alpha-0.3 trail artists. One per dataset THAT HAS ANY of
    # the three flags set (GH #127: previously all-or-nothing -- ANY flag
    # set anywhere created a trail trace for EVERY dataset). These do NOT
    # necessarily sit right after the data traces -- forecast traces
    # (predict=, above) are appended in between when both are present -- so
    # `trail_trace_start` records their real position, and every trail trace
    # carries ``meta['hyp_trail_index']`` -- the ORIGINAL dataset index that
    # produced it -- so `_add_animation` can look up the right dataset's data
    # per frame. A dataset's trail is ONE trace, except an animated
    # multicoloured 2-D line's, which is one trace per colour bin
    # (`_hue_line_bins`); `n_trail_traces` counts traces, not datasets.
    #
    # Backend parity (Task 4): 'serial' builds these too, not just
    # True/'parallel' -- each currently-revealing dataset traces out its own
    # trail as it grows, matching `matplotlib_backend.update_lines_serial`
    # exactly (the serial frame loop in `_add_animation` below is what
    # decides each frame's actual trail geometry per dataset; this list only
    # decides which datasets get a trail TRACE at all, same flags/rule as
    # parallel).
    n_trail_traces = 0
    trail_trace_start = len(traces)
    trail_dataset_indices = [
        i for i in range(len(data))
        if chemtrails[i] or precog[i] or bullettime[i]
    ] if animate in (True, 'parallel', 'serial') else []
    for i in trail_dataset_indices:
        tkwargs = kwargs_list[i] or {}
        mode, symbol, dash, marker_char = _resolve_fmt(fmt[i], tkwargs)
        # fold the 0.3 trail-fade factor into whatever alpha= this dataset
        # carries (default 1.0 -> 0.3, unchanged for the common no-alpha
        # case) -- mirrors matplotlib_backend.animate_plot3D/2D's
        # `_trail_kwargs` (`kw["alpha"] = 0.3 * kw.pop("alpha", 1.0)`)
        # exactly. Previously hardcoded to 0.3 regardless of alpha=, so a
        # per-dataset alpha list (unreachable before per-dataset alpha=
        # existed) never reached plotly's trail traces even though the
        # matching head trace already honors it (see `tkwargs.get('alpha')`
        # a few dozen lines above, in the head-trace loop).
        _trail_alpha = tkwargs.get('alpha')
        _trail_alpha = (0.3 if _trail_alpha is None
                        else 0.3 * float(_trail_alpha))
        color = _to_plotly_color(tkwargs.get('color'), _trail_alpha)
        width = float(tkwargs.get('linewidth')
                      or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
        msize = _marker_size_px(
            tkwargs.get('markersize') or DEFAULT_MARKERSIZE_PT, marker_char,
            ndims=ndims)
        trail_marker = dict(color=color, size=msize)
        if 'markers' in mode:
            # the observations of the dataset's whole smoothed curve; every
            # frame sends its trail window's slice of these sizes
            _n_rows = np.atleast_2d(np.asarray(data[i])).shape[0]
            trail_marker = _observation_marker(
                trail_marker, aa_curves[i][0].shape[0],
                _observation_vertices(aa_curves[i][0], observations[i],
                                      _n_rows, aa_curves[i][1]), ndims)
        trail = dict(mode=mode, showlegend=False, hoverinfo='skip',
                     line=dict(color=color, width=width, dash=dash),
                     marker=trail_marker,
                     # which dataset this trail belongs to (a multicoloured
                     # 2-D trail is several colour-bin traces)
                     meta=dict(hyp_trail_index=i))
        if i in hue_units:
            # a multicoloured 2-D trail: the head's colour bins at the
            # trail's opacity (matplotlib's trail collection keeps the
            # per-segment colours at 0.3 alpha), lines only, as matplotlib
            # draws it
            for _k, _color in enumerate(hue_units[i]['bins']['colors']):
                traces.append(go.Scatter(
                    x=[], y=[], mode='lines', showlegend=False,
                    hoverinfo='skip',
                    line=dict(color=_rgba_with_alpha(_color, _trail_alpha),
                              width=width, dash=dash),
                    meta=dict(hyp_trail_index=i, hyp_hue_bin=_k)))
            continue
        if i in hue_colors_3d:
            # a multicoloured 3-D trail: the head's per-vertex colours at
            # the trail's opacity; every frame sends its window's slice
            trail['line'] = dict(
                color=[_rgba_with_alpha(c, _trail_alpha)
                       for c in hue_colors_3d[i]],
                width=width, dash=dash)
        if ndims >= 3:
            traces.append(go.Scatter3d(x=[], y=[], z=[], **trail))
        else:
            traces.append(go.Scatter(x=[], y=[], **trail))
    n_trail_traces = len(traces) - trail_trace_start

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
            clouds0, morph_samples=morph_samples, loop=morph_loop)
        if morph_loop:
            # `loop=True` returns ONE more cloud than it was given (GH
            # #285): extend the sequence-position -> dataset-index map so
            # every consumer counts the closing repeat, exactly as
            # `matplotlib_backend.animate_plot3D` does.
            morph_indices_3d = morph_indices_3d + [morph_indices_3d[0]]
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
    morph_alphas0 = None
    if morph_tags is not None and ndims in (2, 3):
        pts0 = sampled0[0]
        # full-sample morphs (maintainer request, 2026-07-06 follow-up):
        # this initial trace is frame 0 -- a HOLD frame of dataset 0 -- so
        # its own duplicated (padding) points are excluded here too, exactly
        # like every other hold frame (see `_add_animation`'s 'morph'
        # branch below and `hypertools.plot.morph.morph_visible_mask`).
        hide0 = _morph.morph_visible_mask(dup_masks0, 0)
        draw_pts0 = pts0[~hide0] if hide0 is not None else pts0
        # GH #284: `alpha=` reaches the traveling cloud. The morph-tagged
        # datasets' own traces are never built (skipped in the data loop
        # above), so the alpha each of them carries in `kwargs_list` is
        # applied HERE instead -- folded into the marker's rgba string, the
        # way every other trace in this module carries its alpha (see
        # `_to_plotly_color` in the data loop) -- on the same hold/morph
        # schedule as the color (`_morph.morph_alpha`); all-`None` keeps
        # the plain opaque `_rgb_string` colour, as before.
        morph_alphas0 = [(kwargs_list[i] or {}).get('alpha')
                         for i in morph_indices_3d]
        alpha0 = _morph.morph_alpha(morph_alphas0, 0, 0, 1)
        color0_str = (_rgb_string(ds_colors0[0]) if alpha0 is None
                      else _to_plotly_color(ds_colors0[0], alpha0))
        # matplotlib's morph trace always draws marker='.' (see
        # `MORPH_DEFAULT_MARKERSIZE_PT`'s docstring) -- so the plotly
        # counterpart always applies the dot-marker scale, and falls back
        # to the SAME smaller 4pt default (not the general 6.0pt
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
                                               density_colors,
                                               limit=cube_scale))

    if ndims >= 3:
        # every 3-D DATA line so far (trajectories, forecasts, truth,
        # trails): Scatter3d draws half the width it is asked for (see
        # `_GL_LINE_WIDTH_BOOST`); the cube below carries its own boost
        for _tr in traces:
            if _tr.type == 'scatter3d' and _tr.line is not None \
                    and _tr.line.width is not None:
                _tr.line.width = _tr.line.width * _GL_LINE_WIDTH_BOOST
        traces.append(_cube_trace(
            go, scale=cube_scale, linewidth_pt=_frame['width_pt'],
            color=_frame['color'], dash=_frame['dash']))

    # colorbar (GH #100): appended LAST (after the cube trace) so it never
    # falls within `trace_indices = range(n_data_traces [+ n_trail_traces])`
    # -- the animation frame-update code below only ever touches those
    # indices, so this trace (and its colorbar) is never touched by a frame
    # update and stays static across the whole animation.
    has_colorbar = colorbar_info is not None
    if has_colorbar:
        traces.append(_colorbar_trace(go, colorbar_info, ndims,
                                      legend_present=legend is not None))

    fig = _hyper_figure_class()(data=traces)

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
    # path), so a family name is wrapped in a fallback chain in case it isn't
    # installed in whatever renders this (browser/Chromium via kaleido). The
    # CSS stack is resolved PER GLYPH by the browser, so listing pan-CJK faces
    # after the Latin ones keeps mixed-script text rendering.
    #   * `font` -- an EXPLICIT font= -- LEADS the stack (the caller's choice).
    #   * `font_extra` -- an AUTO-detected family filling a real coverage GAP
    #     (the matplotlib side adds the same family to its fallback stack) --
    #     is appended near the END, so it supplies only the glyphs the Latin
    #     faces lack rather than replacing the primary typography. Without
    #     this, a character matplotlib renders via the discovered font would
    #     silently show as tofu on plotly (maintainer font review).
    # This PREFERS the same Noto-first face as the matplotlib backend but
    # cannot guarantee it -- the browser only resolves an installed family
    # NAME, never hypertools' bundled font FILE (see `_plotly_font_family`).
    font_family = _plotly_font_family(
        explicit=font.get_name() if font is not None else None,
        extra=font_extra)

    # t=40 reserves room for a title; t=10 assumes no title will ever be
    # drawn. `segment_titles` (plan 1.1 Task 8) must ALSO trigger the wider
    # margin: `plot.py` nulls the static `title` for segment-titled
    # serial/morph animations (the title is drawn PER FRAME instead, by the
    # 'morph'/'serial' branches of `_add_animation`, below), but a title
    # still renders on every hold frame -- keying this off `title` alone
    # left those figures at the "no title" t=10 margin even though most
    # frames DO show one (task-8 review, margin finding: measured via a
    # real kaleido PNG render of the first hold frame -- ink starting at
    # canvas row 0 of a 480px-tall render at t=10 (clipped), vs. row 6+ at
    # t=40 (clean); reproduced in
    # tests/plot/test_serial_titles.py). This only reserves the SPACE a
    # per-frame title will use -- it does not un-null `title` itself, so no
    # stray static title is drawn (see `if title is not None:` below).
    layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=legend is not None or bool(legend_entries),
        margin=dict(l=10, r=margin_r,
                    t=40 if (title or segment_titles) else 10, b=10),
        # `itemsizing='constant'`: a legend key is drawn at plotly's fixed
        # key size rather than at the trace's own marker size, so a '.'
        # (2 px) marker still gets a readable dot in the key, as it does in
        # a matplotlib legend (1.1 release review, feature-tour 9.8/9.16)
        legend=dict(bgcolor='rgba(255,255,255,0.8)', itemsizing='constant',
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
        _title_font = dict(color='black',
                           size=round(12 * PT_TO_PX),
                           family=font_family if font_family is not None
                           else 'DejaVu Sans, Arial, sans-serif')
        # title_kwargs= (GH #285): size/colour/family/weight/style/y, the
        # plotly-expressible half of matplotlib's set_title kwargs.
        _title_props, _title_font_props = _plotly_title_overrides(
            title_kwargs)
        _title_font.update(_title_font_props)
        layout['title'] = dict(text=_plotly_title_text(title), x=0.5,
                               xanchor='center', xref='paper',
                               y=0.97, yanchor='top',
                               font=_title_font)
        layout['title'].update(_title_props)
        _title_size_px = _title_font.get('size', round(12 * PT_TO_PX))
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
                # `zoom=` is animation-only (plot()'s docstring; the
                # matplotlib static view ignores it too)
                r=_anim_zoom_r(zoom) if animate else _zoom_r(1))),
            # matplotlib's Axes3D uses a 4:4:3 box aspect by default; match
            # it so the cube renders wider than tall, exactly like the
            # matplotlib backend
            aspectmode='manual',
            aspectratio=dict(x=1.0, y=1.0, z=0.75),
        )
    elif ndims == 2 and axis_scale != 'data':
        # matplotlib stretches the 2D frame to fill the axes region (no
        # equal-aspect constraint), so the plotly frame does the same
        layout['xaxis'] = _labeled_axis_layout(
            {'range': [-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT]}, xlabel)
        layout['yaxis'] = _labeled_axis_layout(
            {'range': [-UNIT_FRAME_LIMIT, UNIT_FRAME_LIMIT]}, ylabel)
        layout['shapes'] = [_square_shape(
            scale=UNIT_FRAME_SCALE, linewidth_pt=_frame['width_pt'],
            color=_frame['color'], dash=_frame['dash'],
            fill=_frame['fill'])]
    elif axis_scale == 'data':
        # GH #285: real units. No frame square, no unit range, and the axes
        # keep plotly's own ticks/labels -- the plotly half of
        # `matplotlib_backend._draw`'s axis_scale='data' branch.
        layout['xaxis'] = _data_axis_layout(xlabel, xlim, date=x_date)
        layout['yaxis'] = _data_axis_layout(ylabel, ylim)
    else:
        layout['xaxis'] = _labeled_axis_layout({}, xlabel)
        layout['yaxis'] = _labeled_axis_layout({}, ylabel)

    # An ANIMATION's legend rides on data-free proxy traces (1.1 release
    # review, L7): plotly omits the legend item of a trace with no points,
    # and a data trace is empty until the reveal reaches it (a later
    # dataset, a later cluster's first run), so its entry appeared and
    # vanished frame by frame while matplotlib's legend is complete from
    # frame 0. Each proxy wears its data trace's style and shares its
    # `legendgroup`, so a legend click still toggles the data; the data
    # traces keep their `name` for hover.
    if animate and animate != 'spin':
        _proxies = []
        for _k in range(data_trace_start, data_trace_start + n_data_traces):
            _tr = fig.data[_k]
            if not _tr.showlegend or _tr.name is None:
                continue
            _group = _tr.legendgroup or _tr.name
            _tr.legendgroup = _group
            _tr.showlegend = False
            _proxies.append(_legend_proxy_for(_tr, ndims, _group))
        if _proxies:
            fig.add_traces(_proxies)

    # labels= (GH #205 F3): point annotations, at parity with matplotlib's
    # annotate_plot -- see _build_point_annotations for the exact mapping
    # semantics. 3-D annotations live in layout.scene.annotations (data
    # space, x/y/z); 2-D annotations live in layout.annotations (data
    # space via xref/yref='x'/'y', since the default paper-relative refs
    # would ignore the actual data coordinates).
    # explicit legend entries (GH #285): a matrix/mixture hue's palette
    # swatches, or `legend_colors=[(label, color), ...]`. Added as
    # data-free traces, plotly's equivalent of matplotlib proxy handles.
    if legend_entries:
        fig.add_traces(_plotly_legend_entry_traces(legend_entries, ndims))
    # predict= legend entries: one per model name, after the explicit
    # entries (the matplotlib legend lists them in the same order). Only
    # with a legend to list in -- like the matplotlib proxies, which exist
    # only on an axes that has one -- so a legend-less figure's traces are
    # exactly its drawn ones.
    if forecast_legend_specs and not legend_explicit and (
            legend is not None or legend_entries):
        # composing into a figure/cell that already lists some of these
        # models: hide the earlier keys and decide the new key's colour
        # over EVERY forecast of that model in the scope (Codex round 4:
        # three calls into one cell listed 'Kalman' three times)
        names = {s[0] for s in forecast_legend_specs}
        for tr in _compose_scope_traces(into):
            meta = tr.meta or {}
            if meta.get('hyp_legend_entry') in names:
                tr.showlegend = False
            elif (meta.get('hyp_forecast_role') in ('static', 'live')
                    and tr.name in names):
                marker = (tr.marker.to_plotly_json()
                          if tr.marker is not None and tr.marker.symbol
                          else None)
                if marker is not None and not np.isscalar(
                        marker.get('size', 0)):
                    # an observation-marked (per-vertex size) forecast: its
                    # legend key takes the marker's one real size
                    marker['size'] = float(np.max(marker['size']))
                forecast_legend_specs.append(
                    (tr.name, tr.line.to_plotly_json(),
                     meta.get('hyp_forecast_alpha'), tr.mode or 'lines',
                     marker))
        fig.add_traces(_forecast_legend_traces(forecast_legend_specs, ndims))
    if truths is not None and _truth_key_at is not None:
        # the neutral 'truth' key (see the truth block): data-free, after
        # the forecast keys, so the drawn traces' indices are untouched
        from .forecast import FORECAST_LEGEND_COLOR
        from .plot import TRUTH_STYLE
        _gray = _to_plotly_color(FORECAST_LEGEND_COLOR, 1.0)
        _key = dict(mode='lines+markers', name='truth', showlegend=True,
                    hoverinfo='skip', legendrank=1001,
                    line=dict(_truth_key_at[1], color=_gray),
                    marker=dict(color=_gray, size=_marker_size_px(
                        TRUTH_STYLE['markersize'], TRUTH_STYLE['marker'],
                        ndims)),
                    meta=dict(hyp_legend_entry='truth'))
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **_key)
                      if ndims >= 3 else go.Scatter(x=[None], y=[None],
                                                    **_key))

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

    # legend_kwargs= (GH #285): plotly `layout.legend` properties, applied
    # last so a caller's x/y/orientation/font/bgcolor wins over the
    # hypertools defaults above -- the same precedence the matplotlib
    # backend gives it over its own `Axes.legend` defaults.
    if legend_kwargs:
        layout['legend'] = {**layout['legend'],
                            **_legend_anchors_for(legend_kwargs),
                            **legend_kwargs}

    fig.update_layout(**layout)

    # GH #285: per-frame (serial/morph) titles carry the same resolved
    # font/`title_kwargs=` styling the static title does -- previously each
    # frame's title reset to plotly's defaults, which is the plotly half of
    # the same bug matplotlib's `_make_title_updater` had.
    _segment_title_style = None
    if segment_titles is not None:
        _seg_props, _seg_font = _plotly_title_overrides(title_kwargs)
        _seg_font.setdefault('family',
                             font_family if font_family is not None
                             else 'DejaVu Sans, Arial, sans-serif')
        _seg_font.setdefault('size', round(12 * PT_TO_PX))
        _seg_font.setdefault('color', 'black')
        _segment_title_style = dict(x=0.5, xanchor='center', xref='paper',
                                    y=0.97, yanchor='top', font=_seg_font)
        _segment_title_style.update(_seg_props)
        _title_size_px = _seg_font.get('size', round(12 * PT_TO_PX))

    # a multi-line (explicit '\n', or `title_wrap=`) or enlarged title
    # needs more than the 40px single-line margin, or it overlaps the
    # plotting area (1.1 release review T6): reserve per line and per
    # font size, exactly as the matplotlib backend's probe does. A
    # dynamic (callable / pattern) title is measured over EVERY frame at
    # the end of `_add_animation`, once its text exists.
    if title is not None or segment_titles is not None:
        _n_title_lines = _plotly_title_lines(title, segment_titles)
        _needed = _title_margin_top(_n_title_lines, _title_size_px,
                                    layout['height'])
        if _needed > 40:
            fig.update_layout(margin=dict(t=_needed))

    if animate:
        _add_animation(fig, data, ndims, animate, frame_rate, duration,
                       rotations, elev, azim, n_data_traces,
                       tail_duration=tail_duration, focused=focused,
                       chemtrails=chemtrails,
                       precog=precog, bullettime=bullettime, zoom=zoom,
                       n_trail_traces=n_trail_traces,
                       trail_trace_start=trail_trace_start,
                       trail_dataset_indices=trail_dataset_indices,
                       forecast_schedule=forecast_schedule,
                       forecast_trace_start=forecast_trace_start,
                       forecast_trace_specs=forecast_trace_specs,
                       forecast_frame_colors=forecast_frame_colors,
                       forecast_reveal=forecast_reveal,
                       forecast_datasets=forecast_datasets,
                       forecast_trail=forecast_trail,
                       forecast_antialias=antialias,
                       surface=surface, surface_colors=surface_colors,
                       surface_trace_start=surface_trace_start_3d,
                       surface_dataset_indices=surface_dataset_indices,
                       surface_point_colors=surface_point_colors,
                       data_trace_start=data_trace_start,
                       morph_tags=morph_tags, morph_colors=morph_colors,
                       morph_samples=morph_samples,
                       morph_loop=morph_loop,
                       dynamic_title=dynamic_title,
                       morph_trace_start=morph_trace_start_3d,
                       morph_mesh_trace_start=morph_mesh_trace_start_3d,
                       morph_surface_spec=morph_surface_spec_3d,
                       morph_sampled=sampled0, morph_dup_masks=dup_masks0,
                       morph_alphas=morph_alphas0,
                       aa_curves=aa_curves, frame_hooks=frame_hooks,
                       segment_titles=segment_titles,
                       segment_title_style=_segment_title_style,
                       segment_title_colors=title_segment_colors,
                       # the run -> dataset -> rows mapping the reveal
                       # clock is driven from (see `_run_window`)
                       ownership=ownership,
                       # an animated multicoloured line's colours travel
                       # with its window (`_hue_line_bins`)
                       hue_units=hue_units, hue_colors_3d=hue_colors_3d)

    # Notebook visual review 2026-09: Scatter3d's RGBA colour path can
    # change hue under transparency. Use RGB + native opacity instead.
    # Include frame payloads, which may override the base trace colours.
    # Done HERE, on this call's own figure, before any `ax=` composition:
    # a caller's figure keeps its own traces exactly as they were (1.1
    # release review: this loop used to run over the composed figure and
    # rewrote the caller's rgba traces too).
    for frame in fig.frames:
        indices = frame.traces if frame.traces is not None else range(len(frame.data))
        for index, trace in zip(indices, frame.data):
            _normalize_scatter3d_alpha(
                trace, fig.data[index].mode
                if fig.data[index].type == 'scatter3d' else None, frame=True)
    for trace in fig.data:
        _normalize_scatter3d_alpha(trace)
    if x_date:
        # dates as naive ISO strings, not epoch ms: plotly.js draws numeric
        # dates in the VIEWER's local time zone (see `_epoch_ms_to_iso`)
        _dates_as_iso(fig)

    if into is not None:
        if animate:
            raise ValueError(
                "ax= (a plotly Figure or hyp.subplots cell) cannot be "
                "combined with animate=: an animated plot builds its own "
                "figure and frames.")
        if isinstance(into, PlotlyCell):
            # `ax=<hyp.subplots(backend='plotly') cell>`: the whole drawn
            # panel -- traces, axis layout, frame, annotations, its own
            # legend and colorbar -- moves into that cell of the grid, the
            # plotly form of drawing into one matplotlib Axes of a grid.
            transplant_panel(into.figure, fig, into.row, into.col,
                             into.index, ndims)
            fig = into.figure
        else:
            # `ax=<plotly Figure>`: draw INTO the caller's figure. The
            # traces (data, legend and colorbar entries) are appended; the
            # caller's layout is theirs to keep.
            into.add_traces(list(fig.data))
            fig = into

    if datasets_drawn is not None:
        _meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        if isinstance(into, PlotlyCell):
            _cells = dict(_meta.get('hyp_cell_datasets_drawn') or {})
            _cells[str(into.index)] = int(datasets_drawn)
            fig.layout.meta = {**_meta, 'hyp_cell_datasets_drawn': _cells}
        else:
            fig.layout.meta = {**_meta,
                               'hyp_datasets_drawn': int(datasets_drawn)}

    if before_show is not None:
        before_show(fig)

    if save_path is not None:
        ext = save_path.lower().rsplit('.', 1)[-1]
        if ext == 'html':
            fig.write_html(save_path)
        elif animate and ext in ('gif', 'png', 'apng', 'mp4', 'mov', 'avi',
                                 'svg'):
            _export_animation_file(fig, save_path, frame_rate, duration,
                                   size)
        else:
            # kaleido renders through a headless Chrome; provision one (and,
            # on Colab/Kaggle, the libraries it needs) on demand.
            ensure_kaleido_chrome()
            fig.write_image(save_path)

    if show:
        show_figure(fig)

    return fig


#: Horizontal room (px) reserved beside a subplot cell for its own legend
#: and for its own colorbar -- the single-axes path widens its right
#: margin by the same amount for each (see `plotly_draw`'s `margin_r`).
PANEL_LEGEND_PX = 110
PANEL_COLORBAR_PX = 110
PANEL_GUTTER_PAD_PX = 8
#: The base margin round a panel grid (the untitled single-axes figure's).
PANEL_MARGIN_PX = 10
#: Gap between neighbouring 3-D cells: what matplotlib's `tight_layout`
#: leaves between two `Axes3D` panels (measured 2026-09-07: 20-26 px
#: between 193-297 px cells at 100 dpi).
PANEL_GAP_PX = 20
#: Gap between neighbouring 2-D/1-D cells, wider for the tick labels an
#: `axis_scale='data'` panel draws (matplotlib: ~43 px between two 2-D
#: panels of a 2x2 grid).
PANEL_AXIS_GAP_PX = 40
#: Room above a titled row for a one-line title: the single-axes plotly
#: figure reserves 40 px of top margin for one (`_title_margin_top`), 10
#: of which is the base margin.
PANEL_TITLE_PX = 30
#: Width of the drawn 3-D cube relative to its SCENE'S HEIGHT at
#: hypertools' default view. Plotly sizes a 3-D scene by its domain's
#: height alone and clips it at the domain's sides (measured 2026-09-07:
#: the same 267 px wide x 209 px tall cube in 600x300 and 1200x300
#: scenes; 179x140 in 800x200; a 300x600 scene's cube is 415 px tall and
#: cut off at the 300 px width), so a cube is ~0.89 scene-heights wide and
#: ~0.70 tall. With a little margin, this is what `transplant_panel` uses
#: to keep a cube inside a cell narrower than it is tall. (Until the 1.1
#: release review it was 1.4 -- the cube's width relative to its OWN
#: height rather than the scene's -- which backed the camera off ~1.5x
#: further than a narrow cell needed, and every square cell by 1.4x.)
SCENE_CUBE_WIDTH_PER_HEIGHT = 0.92


def panel_gutter_px(legend_present, colorbar_present):
    """Pixels to reserve to the RIGHT of every subplot cell (`panels=` and
    `hyp.subplots(backend='plotly')` cells alike) so a per-panel legend
    and/or colorbar sits beside its own panel instead of over the next
    one."""
    px = 0
    if legend_present:
        px += PANEL_LEGEND_PX
    if colorbar_present:
        px += PANEL_COLORBAR_PX
    return px + (PANEL_GUTTER_PAD_PX if px else 0)


def cell_layout_keys(index):
    """Plotly's layout keys for subplot cell number `index` (0-based,
    row-major, as `plotly.subplots.make_subplots` numbers them): the 2-D
    axis layout keys and axis ids, the 3-D scene key, and the legend key
    that panel's traces are attached to (plotly >= 5.15 supports several
    legends: ``layout.legend``, ``layout.legend2``, ...)."""
    suffix = '' if index == 0 else str(index + 1)
    return dict(xaxis=f'xaxis{suffix}', yaxis=f'yaxis{suffix}',
                xref=f'x{suffix}', yref=f'y{suffix}',
                scene=f'scene{suffix}', legend=f'legend{suffix}')


class PlotlyCell:
    """One cell of a ``hyp.subplots(..., backend='plotly')`` grid -- the
    plotly counterpart of the matplotlib ``Axes`` that helper returns, and
    what ``hyp.plot(..., ax=cell)`` draws into (via `transplant_panel`).

    Attributes
    ----------
    figure : plotly.graph_objects.Figure
        The `make_subplots` grid figure the cell belongs to (the figure
        `hyp.subplots` returned; every cell of one grid shares it).
    row, col : int
        1-based grid position, as `make_subplots` numbers cells.
    index : int
        0-based row-major cell number (``layout.xaxis``/``scene``/``legend``
        for 0, ``xaxis2``/``scene2``/``legend2`` for 1, ...).
    ndims : int
        The dimensionality the cell was built for (3 -> a ``'scene'``
        cell, 1 or 2 -> an ``'xy'`` cell).
    """

    __slots__ = ('figure', 'row', 'col', 'index', 'ndims')

    def __init__(self, figure, row, col, index, ndims):
        self.figure = figure
        self.row = int(row)
        self.col = int(col)
        self.index = int(index)
        self.ndims = int(ndims)

    def __repr__(self):
        return (f"PlotlyCell(row={self.row}, col={self.col}, "
                f"index={self.index}, ndims={self.ndims})")


def make_panel_grid(nrows, ncols, ndims, titles=None, size=None,
                    gutter_px=0, title_px=None, **make_subplots_kw):
    """The empty plotly grid `panels=` and `hyp.subplots(backend='plotly')`
    fill: a `plotly.subplots.make_subplots` figure with ``'scene'`` cells
    for 3-D and ``'xy'`` cells otherwise, sized like the matplotlib grid
    (`size` inches x 100 px, default `DEFAULT_FIGSIZE`), with `gutter_px`
    reserved to the right of EVERY cell (and in the right margin) for a
    per-panel legend/colorbar (see `panel_gutter_px`). When `size` is not
    given the figure is widened by the gutters, so the default grid stays
    as roomy as it is without them.

    The cells are laid out the way matplotlib's `tight_layout` lays out
    the matplotlib grid (1.1 release review: the plotly grid used
    `make_subplots`' default spacing -- 10-15 % of the figure between
    cells -- and full-height cells, so three 3-D panels sat in tall
    narrow cells with their titles far above small cubes): `PANEL_GAP_PX`
    (`PANEL_AXIS_GAP_PX` for 2-D/1-D cells) between neighbours,
    `title_px` above every row (default `PANEL_TITLE_PX` when `titles`
    has one, else 0; `panels=` passes what its panels' own titles need),
    and -- for 3-D grids -- SQUARE cells, as an `Axes3D`'s equal box
    aspect makes them, sized by whichever of the width or the height
    binds and centred in the figure. 2-D cells fill the figure.

    Extra keywords go to `make_subplots` (``shared_xaxes=``, ...); a
    caller's ``horizontal_spacing=``/``vertical_spacing=`` replaces the
    pixel-derived one.
    """
    from plotly.subplots import make_subplots
    cell = {'type': 'scene'} if ndims >= 3 else {'type': 'xy'}
    if size is not None:
        width, height = int(size[0] * 100), int(size[1] * 100)
    else:
        width = int(DEFAULT_FIGSIZE[0] * 100) + gutter_px * ncols
        height = int(DEFAULT_FIGSIZE[1] * 100)
    titles = list(titles) if titles is not None else []
    if title_px is None:
        title_px = PANEL_TITLE_PX if any(t for t in titles) else 0
    title_px = int(title_px)
    gap = PANEL_GAP_PX if ndims >= 3 else PANEL_AXIS_GAP_PX
    base = PANEL_MARGIN_PX
    cell_w = (width - 2 * base - gutter_px * ncols
              - gap * (ncols - 1)) / ncols
    cell_h = (height - 2 * base - title_px * nrows
              - gap * (nrows - 1)) / nrows
    cell_w, cell_h = max(cell_w, 1.0), max(cell_h, 1.0)
    if ndims >= 3:
        cell_w = cell_h = min(cell_w, cell_h)
    # the gutter after the LAST column and the title room above the FIRST
    # row live in the margins; the grid is centred in what is left
    plot_w = ncols * cell_w + (ncols - 1) * (gap + gutter_px)
    plot_h = nrows * cell_h + (nrows - 1) * (gap + title_px)
    side = max((width - plot_w - gutter_px) / 2, 0.0)
    vert = max((height - plot_h - title_px) / 2, 0.0)
    margin = dict(l=int(round(side)), t=int(round(vert)) + title_px)
    margin['r'] = max(int(width - plot_w - margin['l']), 0)
    margin['b'] = max(int(height - plot_h - margin['t']), 0)
    make_kw = {}
    if ncols > 1:
        # make_subplots refuses a spacing wider than the cells allow
        make_kw['horizontal_spacing'] = min((gap + gutter_px) / plot_w,
                                            0.98 / (ncols - 1))
    if nrows > 1:
        make_kw['vertical_spacing'] = min((gap + title_px) / plot_h,
                                          0.98 / (nrows - 1))
    if titles:
        make_kw['subplot_titles'] = [t if t is not None else ''
                                     for t in titles]
    make_kw.update(make_subplots_kw)
    fig = make_subplots(rows=nrows, cols=ncols,
                        specs=[[dict(cell) for _ in range(ncols)]
                               for _ in range(nrows)], **make_kw)
    fig.update_layout(width=width, height=height, margin=margin,
                      paper_bgcolor='white', plot_bgcolor='white')
    return fig


def _grid_spec(target):
    """The `make_panel_grid` arguments a `hyp.subplots(backend='plotly')`
    grid was built with (kept in ``layout.meta['hyp_grid']``), or None
    for a grid that was not built that way (a `panels=` grid, which
    sizes its gutters up front from the panels it has already drawn)."""
    meta = target.layout.meta
    if isinstance(meta, dict) and isinstance(meta.get('hyp_grid'), dict):
        return dict(meta['hyp_grid'])
    return None


def ensure_panel_layout(target, gutter_px=None, title_px=None):
    """`ensure_panel_gutter` for both dimensions a drawn cell can grow: the
    gutter beside every cell and the title room above every row. Either
    one growing rebuilds the grid (Codex round 4: a three-line title
    widened the top margin but left the rows 37 px apart)."""
    spec = _grid_spec(target)
    if spec is None:
        return False
    new_gutter = max(int(spec.get('gutter_px', 0)), int(gutter_px or 0))
    new_title = max(int(spec.get('title_px') or 0), int(title_px or 0))
    if (new_gutter == int(spec.get('gutter_px', 0))
            and new_title == int(spec.get('title_px') or 0)):
        return False
    spec['title_px'] = new_title
    return _rebuild_panel_grid(target, spec, new_gutter)


def ensure_panel_gutter(target, gutter_px):
    """Give a `hyp.subplots(backend='plotly')` grid at least `gutter_px`
    of room beside every cell -- rebuilding its layout (width, margins,
    every cell's domain) from the arguments it was built with, and
    re-placing the legends, colorbars and titles of the cells already
    drawn -- the first time a cell actually receives a legend or a
    colorbar. The grid is built WITHOUT gutters (1.1 release review,
    feature-tour 9.8: a legend-less two-cell grid reserved a 118 px gutter
    beside each cell, so its cubes were three quarters the size of the
    matplotlib pair's and sat left-heavy), so a grid whose cells never
    ask for one stays as tight as `panels=` draws it.
    """
    spec = _grid_spec(target)
    if spec is None or int(spec.get('gutter_px', 0)) >= int(gutter_px):
        return False
    return _rebuild_panel_grid(target, spec, int(gutter_px))


def _rebuild_panel_grid(target, spec, gutter_px):
    """Re-lay `target` out from `spec` with `gutter_px` (see
    `ensure_panel_layout`) and re-place every drawn cell's furniture."""
    spec['gutter_px'] = int(gutter_px)
    grid = make_panel_grid(spec['nrows'], spec['ncols'], spec['ndims'],
                           size=spec.get('size'), gutter_px=spec['gutter_px'],
                           title_px=spec.get('title_px'),
                           **dict(spec.get('make_subplots_kw') or {}))
    target.layout.update(width=grid.layout.width, height=grid.layout.height,
                         margin=grid.layout.margin.to_plotly_json())
    for i in range(spec['nrows'] * spec['ncols']):
        keys = cell_layout_keys(i)
        if spec['ndims'] >= 3:
            target.layout[keys['scene']].domain = \
                grid.layout[keys['scene']].domain.to_plotly_json()
        else:
            for axis in ('xaxis', 'yaxis'):
                target.layout[keys[axis]].domain = \
                    grid.layout[keys[axis]].domain
    meta = dict(target.layout.meta) if isinstance(target.layout.meta,
                                                  dict) else {}
    target.layout.meta = {**meta, 'hyp_grid': spec}
    for i in range(spec['nrows'] * spec['ncols']):
        _place_cell_furniture(target, i, spec['ndims'])
    return True


def _explicit_legend_position(legend):
    """``{'lx', 'ly'}`` when a single-figure legend dict carries a
    position other than hypertools' own default (``x=1.02, y=0.5``, the
    outside-right anchor `plotly_draw` sets), i.e. a caller's
    `legend_kwargs` placed it; else None."""
    x, y = legend.get('x'), legend.get('y')
    if x is None or y is None:
        return None
    if abs(float(x) - 1.02) < 1e-9 and abs(float(y) - 0.5) < 1e-9:
        return None
    return {'lx': float(x), 'ly': float(y)}


def _cell_domain(target, keys, ndims):
    if ndims >= 3:
        domain = target.layout[keys['scene']].domain
        return domain.x[0], domain.x[1], domain.y[0], domain.y[1]
    x0, x1 = target.layout[keys['xaxis']].domain
    y0, y1 = target.layout[keys['yaxis']].domain
    return x0, x1, y0, y1


def _place_cell_furniture(target, index, ndims):
    """Place cell `index`'s legend, colorbars and title from its CURRENT
    domain (the placement rules `transplant_panel` applies), so a cell
    can be re-placed after `ensure_panel_gutter` moved it."""
    keys = cell_layout_keys(index)
    x0, x1, y0, y1 = _cell_domain(target, keys, ndims)
    if x0 is None or x1 is None:
        return
    plot_w = max((target.layout.width or int(DEFAULT_FIGSIZE[0] * 100))
                 - (target.layout.margin.l or 0)
                 - (target.layout.margin.r or 0), 1)
    y_mid = 0.5 * (y0 + y1)
    try:
        legend = target.layout[keys['legend']]
    except Exception:  # noqa: BLE001 - a cell never drawn has no legendN
        legend = None
    placed = legend is not None and legend.x is not None
    meta = target.layout.meta if isinstance(target.layout.meta, dict) else {}
    explicit = (meta.get('hyp_cell_legends') or {}).get(str(index))
    # whether the cell SHOWS a legend -- what `transplant_panel` recorded
    # from its traces -- not whether a `legendN` layout exists: every drawn
    # cell gets one placed, so reading that pushed a legend-less cell's
    # colorbar a legend's width right, onto the next cell (1.1 release
    # review)
    furniture = (meta.get('hyp_cell_furniture') or {}).get(str(index))
    if furniture is not None:
        has_legend = bool(furniture.get('legend'))
    else:
        has_legend = any(
            bool(t.showlegend) for t in target.data
            if getattr(t, 'legend', None) == keys['legend']
            or (index == 0 and getattr(t, 'legend', None) in (None,
                                                               'legend')))
    if placed and explicit:
        legend.update(x=x0 + float(explicit['lx']) * (x1 - x0),
                      y=y0 + float(explicit['ly']) * (y1 - y0))
    elif placed:
        legend.update(x=x1 + PANEL_GUTTER_PAD_PX / plot_w, y=y_mid)
    cb_offset = PANEL_GUTTER_PAD_PX + (PANEL_LEGEND_PX if has_legend else 0)
    for trace in target.data:
        if getattr(trace, 'legend', None) != keys['legend'] \
                and not (index == 0 and getattr(trace, 'legend', None)
                         in (None, 'legend')):
            continue
        marker = getattr(trace, 'marker', None)
        if marker is None or not getattr(marker, 'showscale', None) \
                or marker.colorbar is None or marker.colorbar.x is None:
            continue
        cb = marker.colorbar
        if cb.orientation in (None, 'v') and cb.xanchor == 'right':
            cb.update(x=x0 - PANEL_GUTTER_PAD_PX / plot_w, y=y_mid,
                      len=0.75 * (y1 - y0))
        elif cb.orientation in (None, 'v'):
            cb.update(x=x1 + cb_offset / plot_w, y=y_mid,
                      len=0.75 * (y1 - y0))
        else:
            on_top = cb.yanchor == 'bottom'
            cb.update(x=0.5 * (x0 + x1), len=0.75 * (x1 - x0),
                      y=(y1 if on_top else y0))
    title_spec = (meta.get('hyp_cell_titles') or {}).get(str(index))
    if title_spec:
        for ann in target.layout.annotations:
            if ann.name == f'hyp-cell-title-{index}':
                ann.x = x0 + float(title_spec['tx']) * (x1 - x0)
                ty = title_spec.get('ty')
                ann.y = y1 if ty is None else y0 + float(ty) * (y1 - y0)


def transplant_panel(target, panel, row, col, index, ndims):
    """Move one drawn single-axes plotly figure into cell ``(row, col)`` of
    a `make_subplots` figure, at parity with what a matplotlib `ax=` panel
    keeps: its traces, its axis layout (the 2-D unit-frame ranges, hidden
    ticks and axis titles, or the visible ``axis_scale='data'`` axes; the
    3-D scene), its frame square and point annotations (re-referenced to
    the cell's own axes), and ITS OWN legend and colorbar, placed just
    right of the cell rather than merged into one figure-wide legend or
    stacked on one figure-wide colorbar (1.1 release review: three panels
    with ``legend=True`` listed '1, 1, 1' in a single legend, and two
    ``colorbar=True`` panels drew both colorbars on top of each other).

    `index` is the cell's 0-based row-major number. `target` must already
    carry its final ``width``/``height`` and margins (the legend/colorbar
    offsets are pixel distances converted to paper fractions), and its
    ``horizontal_spacing`` should reserve `panel_gutter_px` beside each
    cell. Returns the layout keys the cell uses: ``'scene'`` (3-D) or
    ``'xaxis'``/``'yaxis'`` (2-D), plus ``'legend'``.

    The shared implementation behind `plot(..., panels=)` on this backend
    and the `hyp.subplots(backend='plotly')` cells that `ax=` accepts.
    """
    keys = cell_layout_keys(index)
    # what this cell holds beside it once this panel is in -- a legend
    # and/or a colorbar, from THIS call or an earlier one into the same
    # cell -- recorded per cell, so the grid's gutter is sized for the
    # busiest cell and a colorbar arriving after a legend goes beside it
    # rather than on top of it (Codex round 4)
    _meta = (dict(target.layout.meta)
             if isinstance(target.layout.meta, dict) else {})
    furniture = dict(_meta.get('hyp_cell_furniture') or {})
    cell_furniture = dict(furniture.get(str(index))
                          or {'legend': False, 'colorbar': False})
    cell_furniture['legend'] = bool(
        cell_furniture['legend']
        or any(bool(trace.showlegend) for trace in panel.data))
    cell_furniture['colorbar'] = bool(
        cell_furniture['colorbar']
        or any(getattr(getattr(trace, 'marker', None), 'showscale', None)
               for trace in panel.data))
    furniture[str(index)] = cell_furniture
    target.layout.meta = {**_meta, 'hyp_cell_furniture': furniture}
    # a `hyp.subplots` grid is built without gutters and one title line
    # per row; the first legend/colorbar, or a taller title, a cell brings
    # makes the grid grow (`ensure_panel_layout`), BEFORE this cell's
    # domain is read below
    ensure_panel_layout(
        target,
        gutter_px=max(panel_gutter_px(f.get('legend'), f.get('colorbar'))
                      for f in furniture.values()),
        title_px=(max(0, int(panel.layout.margin.t or 0) - PANEL_MARGIN_PX)
                  if panel.layout.title is not None
                  and panel.layout.title.text else 0))
    plot_w = (target.layout.width or int(DEFAULT_FIGSIZE[0] * 100)) \
        - (target.layout.margin.l or 0) - (target.layout.margin.r or 0)
    plot_w = max(plot_w, 1)

    if ndims >= 3:
        scene = (panel.layout.scene.to_plotly_json()
                 if panel.layout.scene is not None else {})
        scene.pop('domain', None)
        # a cell drawn into twice keeps the earlier call's `labels=`
        # (updating the scene would replace its annotation list, leaving
        # the first dataset's points visible but unlabelled; round 2)
        earlier = [a.to_plotly_json()
                   for a in target.layout[keys['scene']].annotations]
        if earlier:
            scene['annotations'] = earlier + list(scene.get('annotations',
                                                            []))
        target.layout[keys['scene']].update(scene)
        domain = target.layout[keys['scene']].domain
        x0, x1 = domain.x
        y0, y1 = domain.y
        # plotly sizes a 3-D scene by its domain's HEIGHT alone (the cube
        # is ~0.70 of it tall and ~0.89 of it wide at hypertools' view,
        # see `SCENE_CUBE_WIDTH_PER_HEIGHT`) and clips at the sides, so in
        # a cell narrower than it is tall -- a caller's own row_heights=
        # or a tall `size=` -- the cube spilled out of the cell's sides.
        # Back the camera off (apparent size ~ 1/distance, measured) by
        # exactly what the cell's aspect needs. The grid's own cells are
        # square (`make_panel_grid`), where no back-off is needed and the
        # cube fills the cell's width like the matplotlib panel's does.
        plot_h = (target.layout.height or int(DEFAULT_FIGSIZE[1] * 100)) \
            - (target.layout.margin.t or 0) - (target.layout.margin.b or 0)
        cell_w = max(plot_w * (x1 - x0), 1.0)
        cell_h = max(plot_h * (y1 - y0), 1.0)
        back_off = max(1.0, SCENE_CUBE_WIDTH_PER_HEIGHT * cell_h / cell_w)
        camera = target.layout[keys['scene']].camera
        if back_off > 1.0 and camera is not None and camera.eye is not None:
            eye = camera.eye
            target.layout[keys['scene']].camera.eye = dict(
                x=(eye.x or 0.0) * back_off, y=(eye.y or 0.0) * back_off,
                z=(eye.z or 0.0) * back_off)
    else:
        for src, dst in (('xaxis', keys['xaxis']), ('yaxis', keys['yaxis'])):
            axis = panel.layout[src].to_plotly_json()
            axis.pop('domain', None)
            axis.pop('anchor', None)
            target.layout[dst].update(axis)
        # the frame square (unit scale) and `labels=` annotations refer to
        # the panel's own 'x'/'y'; re-point them at this cell's axes
        for shape in panel.layout.shapes:
            spec = shape.to_plotly_json()
            spec['xref'] = keys['xref']
            spec['yref'] = keys['yref']
            target.add_shape(spec)
        for ann in panel.layout.annotations:
            spec = ann.to_plotly_json()
            if spec.get('xref', 'x') == 'x':
                spec['xref'] = keys['xref']
            if spec.get('yref', 'y') == 'y':
                spec['yref'] = keys['yref']
            target.add_annotation(spec)
        x0, x1 = target.layout[keys['xaxis']].domain
        y0, y1 = target.layout[keys['yaxis']].domain

    y_mid = 0.5 * (y0 + y1)
    legend_entries = bool(cell_furniture['legend'])
    # the panel's colorbar goes right of its legend when there is one,
    # else right of the cell, spanning the cell's height like the
    # single-axes colorbar spans the plot's (`len=0.75` of the paper there)
    cb_offset = PANEL_GUTTER_PAD_PX + (PANEL_LEGEND_PX if legend_entries
                                       else 0)
    for trace in panel.data:
        # every trace of this panel lists in THIS panel's legend
        trace.update(legend=keys['legend'])
        marker = getattr(trace, 'marker', None)
        if marker is not None and getattr(marker, 'showscale', None) \
                and marker.colorbar is not None:
            # (before `add_trace`, which COPIES the trace into `target`)
            cb = marker.colorbar
            if cb.orientation in (None, 'v') and cb.xanchor == 'right':
                # `location='left'`: keep it on the cell's LEFT
                cb.update(x=x0 - PANEL_GUTTER_PAD_PX / plot_w,
                          xanchor='right', y=y_mid, yanchor='middle',
                          len=0.75 * (y1 - y0))
            elif cb.orientation in (None, 'v'):
                cb.update(x=x1 + cb_offset / plot_w, xanchor='left',
                          y=y_mid, yanchor='middle', len=0.75 * (y1 - y0))
            else:
                on_top = cb.y is not None and cb.y > 0.5
                cb.update(x=0.5 * (x0 + x1), xanchor='center',
                          len=0.75 * (x1 - x0), y=(y1 if on_top else y0),
                          yanchor=('bottom' if on_top else 'top'))
        target.add_trace(trace, row=row, col=col)

    # the panel's legend, beside its own cell (same styling as the
    # single-axes legend, whose x=1.02/y=0.5 meant "just right of the one
    # plot, vertically centred on it")
    legend = (panel.layout.legend.to_plotly_json()
              if panel.layout.legend is not None else {})
    _explicit = _explicit_legend_position(legend)
    if _explicit is not None:
        # a caller's `legend_kwargs` x/y (paper fractions of the single
        # figure) mean the same place INSIDE the cell (Codex round 3:
        # transplanting overwrote them with the gutter placement)
        legend.update(x=x0 + _explicit['lx'] * (x1 - x0),
                      y=y0 + _explicit['ly'] * (y1 - y0))
    else:
        legend.update(x=x1 + PANEL_GUTTER_PAD_PX / plot_w, y=y_mid,
                      xanchor='left', yanchor='middle')
    _meta = (dict(target.layout.meta)
             if isinstance(target.layout.meta, dict) else {})
    _legends = dict(_meta.get('hyp_cell_legends') or {})
    _legends[str(index)] = _explicit
    target.layout.meta = {**_meta, 'hyp_cell_legends': _legends}
    # the panel's inherited text font (`font=`, GH #205) is MATERIALIZED
    # on this cell's text -- legend, title, axis titles/ticks, colorbar --
    # property by property under any explicit override, so two cells with
    # different fonts stay independent (round 2: a `legend_kwargs=` font
    # size dropped the family, and the grid-wide default made cell two
    # inherit cell one's family)
    panel_font = (panel.layout.font.to_plotly_json()
                  if panel.layout.font is not None else {})
    if panel_font:
        legend['font'] = _with_base_font(legend.get('font'), panel_font)
        _materialize_cell_fonts(target, keys, ndims, panel_font)
        if not target.layout.font.to_plotly_json():
            target.layout.font = dict(panel_font)
    target.layout[keys['legend']] = legend

    # the panel's `title=`, already formatted by the single-axes path
    # (newlines, `title_wrap=`, `title_kwargs=`), as this cell's title --
    # a `make_subplots`-style annotation above the cell, positioned by
    # the same `x`/`y`/anchors the title carries, mapped from the single
    # figure's paper into the cell's domain. One per cell: drawing into
    # the cell again with a `title=` REPLACES it (as a matplotlib axes
    # title is replaced), and an untitled call leaves it alone (as an
    # untitled `hyp.plot(..., ax=ax)` leaves the axes title; 1.1 release
    # review: the second call deleted it), while `labels=` annotations
    # keep accumulating.
    title_name = f'hyp-cell-title-{index}'
    title = panel.layout.title
    if title is not None and title.text:
        target.layout.annotations = tuple(
            a for a in target.layout.annotations if a.name != title_name)
        tx = 0.5 if title.x is None else float(title.x)
        default_y = title.y is None or abs(float(title.y) - 0.97) < 1e-9
        spec = dict(text=title.text, name=title_name,
                    x=x0 + tx * (x1 - x0), xref='paper', yref='paper',
                    xanchor=title.xanchor or 'center', showarrow=False)
        if default_y:
            spec.update(y=y1, yanchor='bottom')
        else:
            spec.update(y=y0 + float(title.y) * (y1 - y0),
                        yanchor=title.yanchor or 'top')
        # where in its cell the title sits, so `_place_cell_furniture`
        # can put it back after the cell moves (`ensure_panel_gutter`)
        _meta = (dict(target.layout.meta)
                 if isinstance(target.layout.meta, dict) else {})
        _titles = dict(_meta.get('hyp_cell_titles') or {})
        _titles[str(index)] = {'tx': tx,
                               'ty': None if default_y else float(title.y)}
        target.layout.meta = {**_meta, 'hyp_cell_titles': _titles}
        title_font = (title.font.to_plotly_json()
                      if title.font is not None else {})
        merged_font = _with_base_font(title_font, panel_font)
        if merged_font:
            spec['font'] = merged_font
        target.add_annotation(spec)
        # the title sits in the top margin: reserve what the single-axes
        # path computed for it (per line and per font size), never less
        # than the 40 px a one-line title needs
        needed = max(40, int(panel.layout.margin.t or 0))
        if (target.layout.margin.t or 0) < needed:
            target.layout.margin.t = needed
    # reconcile this cell's furniture with what it already held (a
    # colorbar arriving beside an earlier legend, or the reverse)
    _place_cell_furniture(target, index, ndims)
    return keys


def _with_base_font(explicit, base):
    """A plotly font dict: `base` (a panel's inherited `layout.font`)
    under `explicit`'s own properties."""
    merged = dict(base or {})
    merged.update(explicit or {})
    return merged


def _materialize_cell_fonts(target, keys, ndims, panel_font):
    """Write `panel_font` under every text element of one cell that has
    no explicit family/size/color of its own: axis titles and tick labels
    (2-D axes or the 3-D scene's) and the cell's colorbar titles/ticks."""
    if ndims >= 3:
        scene = target.layout[keys['scene']]
        axes_ = [scene.xaxis, scene.yaxis, scene.zaxis]
    else:
        axes_ = [target.layout[keys['xaxis']], target.layout[keys['yaxis']]]
    for axis in axes_:
        axis.tickfont = _with_base_font(axis.tickfont.to_plotly_json(),
                                        panel_font)
        if axis.title is not None:
            axis.title.font = _with_base_font(
                axis.title.font.to_plotly_json(), panel_font)
    for trace in target.data:
        marker = getattr(trace, 'marker', None)
        if marker is None or not getattr(marker, 'showscale', None):
            continue
        if (ndims >= 3 and getattr(trace, 'scene', None) != keys['scene']) \
                or (ndims < 3 and (getattr(trace, 'xaxis', None) or 'x')
                    != keys['xref']):
            continue
        cb = marker.colorbar
        cb.tickfont = _with_base_font(cb.tickfont.to_plotly_json(),
                                      panel_font)
        if cb.title is not None:
            cb.title.font = _with_base_font(cb.title.font.to_plotly_json(),
                                            panel_font)


def show_figure(fig):
    """Display `fig` the way ``plot(..., show=True)`` does on this backend.

    Shared by the single-axes path and `panels=` (1.1 review, P6), so both
    go through the same three cases:

    - a docs build (plotly's sphinx-gallery renderer): plotly's own
      renderer writes a static png AND an interactive html from the full
      figure, and kaleido serializes EVERY animation frame to render the
      one png -- a 900-frame figure took ~an hour and produced tens-of-MB
      pages. Write the pair ourselves instead: png from a frame-stripped
      snapshot, html with the embedded frames capped (total duration and
      rotations preserved, so pacing stays identical).
    - an interactive notebook: display at the END of the cell (after
      matplotlib-inline's own flush, so plotly figures keep their place
      behind matplotlib ones drawn in the same cell) and only if the cell's
      rich-display hook has not already shown this figure as its last
      expression. See `HyperPlotlyFigure`.
    - a plain script (no IPython frontend): nothing else will display the
      figure, so show it here.
    """
    import plotly.io as pio
    if 'sphinx_gallery' in str(pio.renderers.default or ''):
        _show_sphinx_gallery(fig)
    elif _in_interactive_shell():
        _display_at_cell_end(fig)
    else:
        fig.show()


_HYPER_FIGURE_CLASS = None


def _hyper_figure_class():
    """The ``plotly.graph_objects.Figure`` subclass ``plot()`` returns, built
    on first use because plotly is an optional dependency."""
    global _HYPER_FIGURE_CLASS
    if _HYPER_FIGURE_CLASS is None:
        import plotly.graph_objects as go

        class HyperPlotlyFigure(go.Figure):
            """A ``plotly.graph_objects.Figure`` that ``plot()`` displays ONCE, at
            the end of the notebook cell.

            ``plot(..., show=True)`` on the matplotlib backend shows the figure at the
            end of the cell whether or not the caller keeps it (``fig = hyp.plot(x)``
            still draws). Until 1.1 the plotly backend never called ``fig.show()``
            inside IPython: calling it drew the figure TWICE when the figure was
            also the cell's last expression, and drew it mid-cell, ahead of the
            matplotlib figures flushed at the end (Jeremy's QC 2026-07 report). The
            cost was that ``fig = hyp.plot(x)`` drew nothing (measured 2026-09-04 on
            Colab, where ``backend='auto'`` resolves to plotly: 29 of the feature
            tour's plot cells were blank).

            Now ``plot()`` queues the figure for a one-shot IPython ``post_execute``
            callback, registered after matplotlib-inline's flush so it runs after
            it, and the callback skips any figure this hook has already displayed.
            Both usages draw exactly once, in cell order.
            """

            def _ipython_display_(self):
                self._hyp_displayed = True
                super()._ipython_display_()

        _HYPER_FIGURE_CLASS = HyperPlotlyFigure
    return _HYPER_FIGURE_CLASS


def _in_interactive_shell():
    """True inside an interactive IPython/Jupyter frontend (where a returned
    figure is auto-displayed by the cell's rich-display hook)."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


_PENDING_DISPLAY = []       # figures queued for the end of the running cell


def _display_at_cell_end(fig):
    """Queue ``fig`` for display when the current IPython cell finishes."""
    from IPython import get_ipython
    shell = get_ipython()
    if shell is None or not hasattr(shell, 'events'):
        fig.show()
        return
    # register on THIS shell unless already registered (keyed on the shell's
    # own callback list, not on the queue being empty: a callback that raised
    # in an earlier cell must not leave later cells unregistered)
    callbacks = getattr(shell.events, 'callbacks', {})
    if _flush_pending_display not in callbacks.get('post_execute', []):
        shell.events.register('post_execute', _flush_pending_display)
    # once per FIGURE: several `ax=` calls into one grid queue the same
    # figure, which must display once, as the matplotlib grid does
    if not any(queued is fig for queued in _PENDING_DISPLAY):
        _PENDING_DISPLAY.append(fig)


def _flush_pending_display():
    """The one-shot ``post_execute`` callback: show every queued figure the
    rich-display hook has not shown, then unregister."""
    from IPython import get_ipython
    import plotly.io as pio
    figs, _PENDING_DISPLAY[:] = list(_PENDING_DISPLAY), []
    shell = get_ipython()
    if shell is not None and hasattr(shell, 'events'):
        try:
            shell.events.unregister('post_execute', _flush_pending_display)
        except ValueError:
            pass
    for fig in figs:
        if getattr(fig, '_hyp_displayed', False):
            fig._hyp_displayed = False
            continue
        pio.show(fig)


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
    ensure_kaleido_chrome()
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


def _worker_error(frames_dir):
    """The exception the export worker reported through its error file, as
    the type the caller is promised (`ImportError` for a missing extra with
    installation off, `HypertoolsIOError` when no Chrome could be provided),
    or None when the worker failed some other way (rendering, a kill)."""
    from ._kaleido_export_worker import ERROR_FILE
    from ..core.exceptions import HypertoolsIOError
    path = os.path.join(frames_dir, ERROR_FILE)
    try:
        with open(path, encoding='utf-8') as fh:
            info = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        os.remove(path)
    except OSError:
        pass
    types = {'ImportError': ImportError, 'ModuleNotFoundError': ImportError,
             'HypertoolsIOError': HypertoolsIOError}
    cls = types.get(info.get('type'))
    if cls is None:
        return None
    return cls(f"plotly frame export: {info.get('message', '')}")


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
                # on a full pipe buffer while we watch for progress.
                # env: the worker provisions kaleido/Chrome itself (it may
                # pip-install and download), so it must start from THIS
                # process's effective set_autoinstall() setting, which lives
                # in Python and is not inherited by a fresh interpreter
                # (release audit 2026-09-07).
                with open(err_path, 'wb') as errf:
                    proc = subprocess.Popen(
                        [sys.executable, '-m',
                         'hypertools.plot._kaleido_export_worker',
                         fig_json, frames_dir, ext, str(width), str(height)],
                        stdout=subprocess.DEVNULL, stderr=errf,
                        env=subprocess_env(),
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
                    reported = _worker_error(frames_dir)
                    if reported is not None:
                        # the worker could not import or provision what it
                        # needs (a missing kaleido with installation off, no
                        # usable Chrome): that is not a render failure to
                        # retry, and the caller is promised the documented
                        # exception type (ImportError naming the manual
                        # command; HypertoolsIOError for Chrome)
                        raise reported
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


#: `frame_kwargs=` keys `_frame_style` maps (matplotlib spellings, as
#: `matplotlib_backend.plot_cube`'s `plot_wireframe` and `plot_square`'s
#: `Rectangle` take them)
_FRAME_COLOR_KEYS = ('color', 'colors', 'edgecolor', 'edgecolors', 'ec')
_FRAME_WIDTH_KEYS = ('linewidth', 'linewidths', 'lw')
_FRAME_STYLE_KEYS = ('linestyle', 'linestyles', 'ls')
_FRAME_FACE_KEYS = ('facecolor', 'fc')
#: matplotlib's own default frame width (`plot_cube`/`plot_square`), which
#: `CUBE_LINEWIDTH_PT` is calibrated to match on screen
_MPL_FRAME_LINEWIDTH_PT = 1.0


def _frame_style(frame_kwargs, ndims):
    """The plotly styling of the cube/square frame from `plot()`'s
    ``frame_kwargs=`` (matplotlib's `plot_wireframe`/`Rectangle` keywords).

    Returns ``dict(color, width_pt, dash, fill)`` -- `color` a plotly colour
    string (``alpha=`` folded in), `width_pt` the frame width in the
    points `_cube_trace`/`_square_shape` take (a matplotlib ``linewidth``
    scaled by the same factor that makes the default 1 pt frame match
    matplotlib's on screen), `dash` a plotly dash name, `fill` the 2-D
    square's fill colour or None. With no `frame_kwargs` this is exactly the
    historical black frame. Keywords with no plotly equivalent (``zorder``,
    ``rstride``, ...) are named in one warning instead of being dropped
    silently (1.1 release review: plotly ignored `frame_kwargs=` outright,
    so the cube stayed black whatever colour was asked for).
    """
    kw = dict(frame_kwargs or {})
    alpha = kw.pop('alpha', None)
    # `color`/`colors` style the whole frame (a Rectangle's edge AND face);
    # the edge spellings style only its outline
    both = next((kw.get(k) for k in ('color', 'colors')
                 if kw.get(k) is not None), None)
    edge = next((kw.get(k) for k in ('edgecolor', 'edgecolors', 'ec')
                 if kw.get(k) is not None), None)
    color = edge if edge is not None else both
    for k in _FRAME_COLOR_KEYS:
        kw.pop(k, None)
    width = next((kw.pop(k) for k in _FRAME_WIDTH_KEYS
                  if kw.get(k) is not None), None)
    for k in _FRAME_WIDTH_KEYS:
        kw.pop(k, None)
    style = next((kw.pop(k) for k in _FRAME_STYLE_KEYS
                  if kw.get(k) is not None), None)
    for k in _FRAME_STYLE_KEYS:
        kw.pop(k, None)
    face = next((kw.pop(k) for k in _FRAME_FACE_KEYS
                 if kw.get(k) is not None), None)
    for k in _FRAME_FACE_KEYS:
        kw.pop(k, None)
    fill = kw.pop('fill', None)
    # matplotlib's plot_wireframe/Rectangle defaults that are meaningless
    # (or already implied) here
    for k in ('rstride', 'cstride'):
        kw.pop(k, None)
    if kw:
        warnings.warn(
            "backend='plotly' cannot map the following frame_kwargs to the "
            f"plotly frame and will ignore them: {sorted(kw)}. Supported: "
            "color/edgecolor, linewidth, linestyle, alpha (and, for the 2-D "
            "square, facecolor/fill).", UserWarning, stacklevel=3)
    def _one(c):
        # a `colors=` list (one per wireframe line): one colour here
        if isinstance(c, (list, tuple)) and c and not isinstance(
                c[0], (int, float, np.integer, np.floating)):
            return c[0]
        return c
    color, both = _one(color), _one(both)
    line_color = ('black' if color is None and alpha is None
                  else _to_plotly_color(color if color is not None
                                        else 'black', alpha))
    width_pt = (CUBE_LINEWIDTH_PT if width is None
                else float(width) * CUBE_LINEWIDTH_PT
                / _MPL_FRAME_LINEWIDTH_PT)
    dash = ('solid' if style is None
            else _LINESTYLE_NAMES.get(style, 'solid'))
    fill_color = None
    if ndims < 3:
        # matplotlib's `plot_square`: a `color=` (or a face colour) fills
        # the square unless `fill=False`; with neither it is an outline
        face_color = face if face is not None else both
        if face_color is not None and fill is not False:
            fill_color = _to_plotly_color(face_color, alpha)
        elif fill is True:
            fill_color = _to_plotly_color('C0', alpha)
    return dict(color=line_color, width_pt=width_pt, dash=dash,
                fill=fill_color)


def _cube_trace(go, scale=1.0, linewidth_pt=CUBE_LINEWIDTH_PT, color='black',
                dash='solid'):
    """hypertools' signature black wireframe cube as a single 3D trace.

    Mirrors matplotlib_backend's plot_cube: 12 edges at +/-scale, black,
    1pt lines (or the `frame_kwargs=` style `_frame_style` resolved).
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
        line=dict(color=color,
                  width=linewidth_pt * PT_TO_PX * _CUBE_GL_WIDTH_BOOST,
                  **({} if dash == 'solid' else dict(dash=dash))),
        showlegend=False, hoverinfo='skip')


def _square_shape(scale=1.0, linewidth_pt=CUBE_LINEWIDTH_PT, color='black',
                  dash='solid', fill=None):
    """hypertools' 2D black square frame (mirrors matplotlib_backend's
    plot_square; `color`/`dash`/`fill` from `_frame_style`)."""
    return dict(type='rect', x0=-scale, y0=-scale, x1=scale, y1=scale,
                line=dict(color=color, width=linewidth_pt * PT_TO_PX,
                          **({} if dash == 'solid' else dict(dash=dash))),
                fillcolor=fill if fill is not None else 'rgba(0,0,0,0)',
                layer='below')


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
    ``go.Contour`` KDE density layer (GH #108/#191, 2-D); each grid reaches
    `KDE_GRID_BANDWIDTHS` kernel widths past its own cloud (see
    `kde_grid_2d`)."""
    points = [np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :2]
              for arr in data]
    if density[0] is not None and not density[0].get('per_group', True):
        all_pts = np.vstack(points)
        trace = _one_density_contour_trace(go, all_pts, density[0],
                                           POOLED_COLOR, label=' (pooled)')
        return [trace] if trace is not None else []
    traces = []
    for i, (pts, spec) in enumerate(zip(points, density)):
        if spec is None:
            continue
        trace = _one_density_contour_trace(go, pts, spec, density_colors[i],
                                           label=f' {i}')
        if trace is not None:
            traces.append(trace)
    return traces


def _one_density_volume_trace(go, pts, spec, color_rgb, label="", boost=1.0,
                              limit=None):
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
    if limit is None:
        X, Y, Z, D, _, _ = kde_grid_3d(pts, kde, gridsize=gridsize, pad=pad)
    else:
        # the grid, padded past the data so the glow fades out, is CLIPPED
        # to the scene's cube (1.1 release review, L2): a grid reaching
        # past the scene range (x to +-1.3) drew its translucent shells
        # over the cube's edges, which rendered stippled (1076 of 3551
        # dark cube/marker pixels survived on the reviewer's case; all of
        # them do clipped). The same `gridsize` samples the clipped box.
        lo, hi = _padded_bounds(np.asarray(pts, dtype=float), pad)
        lo, hi = np.maximum(lo, -limit), np.minimum(hi, limit)
        axes_ = [np.linspace(lo[i], hi[i], gridsize) for i in range(3)]
        X, Y, Z = np.meshgrid(*axes_, indexing='ij')
        D = kde(np.vstack([X.ravel(), Y.ravel(), Z.ravel()])).reshape(
            X.shape)
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


def _build_density_traces_3d(go, data, density, density_colors, limit=None):
    """Build each dataset's (or, with ``per_group=False``, one pooled)
    ``go.Volume`` KDE density layer (GH #108/#191, 3-D).

    Each per-dataset layer's opacity is boosted (GH #108 round 2) by how
    small that dataset's own bounding box is relative to the bounding box
    of the WHOLE scene (all datasets combined) -- see
    :func:`~.density.density_alpha_boost`. `limit` (the scene cube's
    half-width) clips every layer's grid to the cube (see
    `_one_density_volume_trace`)."""
    if density[0] is not None and not density[0].get('per_group', True):
        all_pts = np.vstack([
            np.atleast_2d(np.asarray(arr, dtype=np.float64))[:, :3]
            for arr in data])
        trace = _one_density_volume_trace(go, all_pts, density[0],
                                          POOLED_COLOR, label=' (pooled)',
                                          boost=1.0, limit=limit)
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
                                          label=f' {i}', boost=boost,
                                          limit=limit)
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


def _forecast_style_from(tkwargs, fmt_str, alpha=None, override=None,
                         anchor_color=None):
    """Style a forecast trace to match the observed trace it continues.

    The plotly twin of `hypertools.plot.plot._forecast_style_from`, sharing
    its policy constant (`forecast.FORECAST_ALPHA_SCALE`) so the two backends
    cannot drift: colour, line WIDTH and DASH are inherited verbatim from the
    observed trace, and only the opacity changes -- ``observed_alpha *
    FORECAST_ALPHA_SCALE``, with an unset alpha counting as fully opaque.
    The dash comes from the same `_resolve_fmt` call the observed trace is
    built with, so a solid dataset yields a solid forecast and a dotted one a
    dotted forecast (pre-1.1.0 every forecast was ``dash='dash'`` at a
    hard-coded 0.6 opacity, regardless of how the data was drawn).

    Parameters
    ----------
    tkwargs : dict
        This dataset's resolved per-trace kwargs (`kwargs_list[i]`).
    fmt_str : str or None
        This dataset's fmt (`fmt[i]`), the same value the observed trace's
        `_resolve_fmt` gets.
    alpha : float, optional
        Override the computed alpha -- used by the animated TRAIL traces,
        whose alpha is `forecast.trail_alpha` of the live one.
    override : dict, optional
        This dataset's `forecast_*=` override
        (`forecast.resolve_forecast_overrides`) -- the SAME dict the
        matplotlib side applies, resolved once and translated here into
        plotly's dash/rgba vocabulary. Sparse: only the aspects it names
        replace the inherited ones.
    anchor_color : tuple of float, optional
        The source trace's FINAL per-point colour, from `_hue_anchor_color`,
        when a continuous `hue=` gave that trace many colours. It replaces
        `tkwargs['color']` (which is then only plotly's per-dataset palette
        fallback, not a colour the trace is actually drawn in) and is still
        overruled by an explicit `forecast_color=`, so the precedence reads
        override > anchor > inherited, exactly as on matplotlib.

    Returns
    -------
    (dict, float)
        A trace ``line=`` dict, and the alpha baked into its rgba colour --
        the SAME float callers record in ``meta['hyp_forecast_alpha']``, so a
        reader of `meta` can trust it.
    """
    from .forecast import forecast_alpha
    override = override or {}
    # a `forecast_fmt=` is read through the SAME `_resolve_fmt` the observed
    # trace's own fmt goes through, so the two strings mean the same thing --
    # but WITHOUT the observed trace's linestyle=/marker= kwargs. Inside
    # `_resolve_fmt` an explicit style kwarg beats the fmt string, which is
    # right for the observed trace's own fmt and wrong for an override whose
    # entire purpose is to overrule the observed style: `linestyle='--'` with
    # `forecast_fmt=':'` drew a DASHED forecast here and a dotted one on
    # matplotlib, which applies the override last.
    _fmt_kwargs = tkwargs
    if 'fmt' in override:
        _fmt_kwargs = {k: v for k, v in tkwargs.items()
                       if k not in ('linestyle', 'ls', 'marker')}
    _mode, _symbol, dash, _marker_char = _resolve_fmt(
        override.get('fmt', fmt_str), _fmt_kwargs)
    if alpha is None:
        # a recoloured forecast keeps its trace's alpha (matplotlib
        # parity: `plot._forecast_style_from` applies the same
        # `forecast.forecast_alpha_scale_for` rule)
        from .forecast import forecast_alpha_scale_for
        alpha = forecast_alpha(tkwargs.get('alpha'),
                               forecast_alpha_scale_for(override))
    width = float(tkwargs.get('linewidth') or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
    color = override.get(
        'color',
        anchor_color if anchor_color is not None else tkwargs.get('color'))
    if 'color' not in override:
        # a colour letter in `forecast_fmt=` ('r:') recolours the forecast,
        # as it does on matplotlib (Codex round 3: plotly dropped it)
        fmt_color = _fmt_color_letter(override.get('fmt'))
        if fmt_color is not None:
            color = fmt_color
    line = dict(color=_to_plotly_color(color, alpha), width=width, dash=dash)
    return line, alpha


def _fmt_color_letter(fmt):
    """The colour a matplotlib format string names (``'r:'`` -> ``'r'``),
    or None when it names none (or is not a string)."""
    if not isinstance(fmt, str) or not fmt:
        return None
    try:
        from matplotlib.axes._base import _process_plot_format
        return _process_plot_format(fmt)[2]
    except Exception:  # noqa: BLE001 - an unparseable fmt names no colour
        return None


def _forecast_marker(tkwargs, override, line_color, ndims):
    """``(mode, marker)`` for a forecast trace: ``('lines', None)`` unless
    `forecast_fmt=` asked for a marker (``'o:'``), in which case the trace
    draws ``'lines+markers'`` with that marker at the observed trace's
    marker size, in the forecast's own colour -- what the matplotlib
    overlay draws for the same string (Codex round 3: plotly dropped the
    marker). A forecast never inherits the OBSERVED trace's marker: it is
    a line, and only its own format string can add markers to it."""
    fmt = (override or {}).get('fmt')
    if not isinstance(fmt, str) or not fmt:
        return 'lines', None
    _mode, symbol, _dash, marker_char = _resolve_fmt(
        fmt, {k: v for k, v in tkwargs.items()
              if k not in ('linestyle', 'ls', 'marker')})
    if marker_char is None or symbol is None:
        return 'lines', None
    size = _marker_size_px(
        tkwargs.get('markersize') or DEFAULT_MARKERSIZE_PT, marker_char,
        ndims=ndims)
    # the parsed mode: 'markers' for a marker-only string ('ro'), as the
    # matplotlib overlay draws it (Codex round 4), else lines+markers
    mode = 'markers' if 'lines' not in _mode else 'lines+markers'
    return mode, dict(symbol=symbol, size=size, color=line_color)


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
        # A CONTINUOUS colorbar needs ONE palette to sample across the
        # value range. A `{category: color}` dict resolves here (read as an
        # ordered list of colours); a PER-DATASET palette LIST (GH #285) is
        # n separate palettes with no single ramp between them, and
        # `colors._get_palette` already rejects it by name -- from
        # `mat2colors`, when the per-point colours are computed, which is
        # strictly before this colorbar is built. No second check here: it
        # could never fire.
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
            cb['ticktext'] = [str(lbl) for lbl in colorbar_info['labels']]

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


#: Most colour-bin traces an ANIMATED multicoloured 2-D line gets
#: (`_hue_line_bins`). A plotly 2-D line has ONE colour per trace, and every
#: frame must rewrite every trace of the line, so the per-segment colours
#: are drawn with at most this many traces. A continuous `hue=` maps through
#: a 100-colour ramp (`colors.continuous_colormap`'s `n_bins`), so it is
#: drawn EXACTLY; only a line with more distinct colours than this (a
#: matrix/RGB hue's blends) has each segment take its bin's mean colour
#: (k-means over the distinct colours; measured 2026-09-11: 48 bins over a
#: 100-colour ramp were at most 7-15/255 per channel off, 100 bins are exact).
HUE_ANIM_MAX_BINS = 100


class PlotlyTraceGroup(tuple):
    """The several frame traces that draw ONE dataset -- an animated
    multicoloured 2-D line is one trace per colour bin (`_hue_line_bins`)
    -- handed to an `on_frame=` callback as that dataset's single entry of
    `FrameContext.artists`.

    It is a tuple of `go.Scatter` traces (iterate it to reach each one), and
    ASSIGNING an attribute sets it on every member, so ``artist.opacity =
    0.4`` -- what `dataset_fade=` and a portable callback do -- fades the
    whole dataset. Reading an attribute reads the first member's.
    """

    def __setattr__(self, name, value):
        for trace in self:
            setattr(trace, name, value)

    def __getattr__(self, name):
        if not self:
            raise AttributeError(name)
        return getattr(self[0], name)


def _parse_rgba(color):
    """``(r, g, b, a)`` (0-255 channels, alpha 0-1) of a plotly
    ``rgb(...)``/``rgba(...)`` string."""
    text = str(color).strip()
    parts = [float(p) for p in text[text.index('(') + 1:-1].split(',')]
    return tuple(parts[:3]) + ((parts[3] if len(parts) > 3 else 1.0),)


def _hue_line_bins(colors, max_bins=HUE_ANIM_MAX_BINS):
    """Group a multicoloured line's SEGMENT colours into a bounded set of
    colour bins, for the animated 2-D representation (see
    `HUE_ANIM_MAX_BINS`).

    `colors` is one plotly colour string per drawn VERTEX (segment ``j``
    wears vertex ``j``'s colour, as `_segment_traces_2d` draws it). Returns
    ``dict(seg_bin=<int array, one per segment>, colors=[bin colour
    strings])``. Deterministic: the k-means starts from distinct colours
    spread evenly over their first-appearance order.
    """
    seg_colors = list(colors[:-1]) if len(colors) > 1 else list(colors)
    distinct = list(dict.fromkeys(seg_colors))
    index_of = {c: k for k, c in enumerate(distinct)}
    seg_distinct = np.array([index_of[c] for c in seg_colors], dtype=int)
    if len(distinct) <= max_bins:
        return dict(seg_bin=seg_distinct, colors=distinct)
    rgba = np.array([_parse_rgba(c) for c in distinct], dtype=float)
    weights = np.bincount(seg_distinct, minlength=len(distinct)).astype(float)
    centres = rgba[np.linspace(0, len(distinct) - 1, max_bins).astype(int)]
    for _ in range(25):
        dist = ((rgba[:, None, :3] - centres[None, :, :3]) ** 2).sum(axis=2)
        label = dist.argmin(axis=1)
        moved = centres.copy()
        for k in range(max_bins):
            member = label == k
            if member.any():
                w = weights[member][:, None]
                moved[k] = (rgba[member] * w).sum(axis=0) / w.sum()
        if np.allclose(moved, centres):
            break
        centres = moved
    used = sorted(set(label.tolist()))
    renumber = {k: n for n, k in enumerate(used)}
    bin_colors = []
    for k in used:
        r, g, b, a = centres[k]
        bin_colors.append(f'rgba({int(round(r))},{int(round(g))},'
                          f'{int(round(b))},{float(a)})')
    return dict(seg_bin=np.array([renumber[label[d]] for d in seg_distinct],
                                 dtype=int),
                colors=bin_colors)


def _binned_polylines(xs, ys, seg_bin, k, v0, v1):
    """The x/y of colour bin `k`'s share of vertices ``v0..v1`` (inclusive)
    of a multicoloured line: its runs of consecutive segments, each drawn
    as one polyline, separated by NaN gaps (plotly breaks a line at a gap).
    Empty arrays when the window holds none of the bin's segments."""
    if v1 <= v0:
        return np.zeros(0), np.zeros(0)
    segs = np.arange(v0, v1)
    segs = segs[seg_bin[v0:v1] == k]
    if segs.size == 0:
        return np.zeros(0), np.zeros(0)
    # split into runs of consecutive segment indices
    breaks = np.flatnonzero(np.diff(segs) > 1) + 1
    out_x, out_y = [], []
    for run in np.split(segs, breaks):
        verts = np.arange(run[0], run[-1] + 2)
        out_x.extend([xs[verts], [np.nan]])
        out_y.extend([ys[verts], [np.nan]])
    return np.concatenate(out_x[:-1]), np.concatenate(out_y[:-1])


def _segment_traces_2d(go, pts, colors, width, dash, name, trace_index=None):
    """Per-segment colored 2D line, emitted as one small trace per segment
    (plotly's 2D Scatter lines accept only a single color per trace).

    Every segment carries the SAME ``meta['hyp_trace_index']`` as the
    trajectory it is a piece of, so a reader counting data traces by that tag
    sees one trace rather than ``len(pts) - 1`` of them."""
    segs = []
    for j in range(len(pts) - 1):
        segs.append(go.Scatter(
            x=pts[j:j + 2, 0], y=pts[j:j + 2, 1], mode='lines',
            line=dict(color=colors[j], width=width, dash=dash),
            showlegend=False, hoverinfo='skip',
            meta=(None if trace_index is None
                  else dict(hyp_trace_index=trace_index)),
            legendgroup=name or 'multicolor'))
    return segs


def _to_plotly_color(color, alpha=None):
    # ROUNDS each channel, like `_rgb_string`. These are the module's two
    # colour serializers and they must agree: a forecast whose colour is
    # anchored to its source trace's final per-point colour
    # (`_hue_anchor_color`) goes through THIS function while the trace's own
    # per-point strings go through `_rgb_string`, so truncating here made the
    # "same" colour print one channel unit darker (measured on viridis's last
    # stop: 0.9932*255 -> rgb(253,...) rounded vs rgb(252,...) truncated).
    if color is None:
        return None
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    a = 1.0 if alpha is None else float(alpha)
    return (f'rgba({int(round(r * 255))},{int(round(g * 255))},'
            f'{int(round(b * 255))},{a})')


def _hue_anchor_color(point_colors, src):
    """The single colour a forecast inherits from a MULTI-coloured trace.

    A continuous `hue=` gives the observed trace one colour per point, so
    "the same colour as its trace" resolves to the colour where the forecast
    begins: the source run's LAST per-point colour. The matplotlib twin is
    the `_kept_forecasts` loop in `plot._apply_multicolor_lines`, which
    anchors on `line_colors[dataset][-1]` -- without this, plotly styled the
    forecast from `kwargs_list[src]['color']`, which under a continuous hue
    is the per-dataset PALETTE fallback `plot.py` fills in for plotly
    (`plot.py`, "if 'color' not in mpl_kwargs"). Measured on a 3-trace column
    hierarchy with `hue=linspace(0,1)` and `palette='viridis'`: the observed
    traces all ended at rgb(253,231,37) while their forecasts drew
    rgb(59,82,139)/rgb(33,145,140)/rgb(94,201,97) -- seaborn's 3-colour
    viridis cycle, unrelated to the hue.

    Returns None (leaving the inherited single colour in place) when this
    trace has no per-point colours, which is every non-continuous-hue plot.
    """
    if point_colors is None or src is None or src >= len(point_colors):
        return None
    pc = point_colors[src]
    if pc is None or len(pc) == 0:
        return None
    return tuple(float(v) for v in np.asarray(pc[-1], dtype=np.float64)[:3])


def _legend_proxy_for(trace, ndims, group):
    """A data-free trace carrying `trace`'s legend entry (name, line and
    marker style, `legendgroup`) -- how an animation keeps its legend
    complete while its data traces are still empty (see `plotly_draw`)."""
    import plotly.graph_objects as go

    def _scalar(value):
        if value is None or isinstance(value, str) or np.isscalar(value):
            return value
        values = [v for v in value if v is not None]
        if not values:
            return None
        return (max(values) if all(np.isscalar(v) and not isinstance(v, str)
                                   for v in values) else values[0])

    line = trace.line.to_plotly_json() if trace.line is not None else {}
    line['color'] = _scalar(line.get('color'))
    common = dict(mode=trace.mode or 'lines', name=trace.name,
                  showlegend=True, legendgroup=group, hoverinfo='skip',
                  line=line,
                  # NOT `hyp_legend_entry`, which marks the forecast
                  # model keys (`_forecast_legend_traces`)
                  meta=dict(hyp_legend_proxy=str(trace.name)))
    if trace.opacity is not None:
        common['opacity'] = trace.opacity
    if trace.mode and 'markers' in trace.mode and trace.marker is not None:
        marker = {k: v for k, v in trace.marker.to_plotly_json().items()
                  if k in ('color', 'size', 'symbol', 'opacity')}
        marker['color'] = _scalar(marker.get('color'))
        marker['size'] = _scalar(marker.get('size'))
        common['marker'] = marker
    if ndims >= 3:
        return go.Scatter3d(x=[None], y=[None], z=[None], **common)
    return go.Scatter(x=[None], y=[None], **common)


def _hover_name(trace_names, i):
    """Data trace `i`'s name from `plotly_draw`'s `trace_names` (None when
    it has none, or when no names were given)."""
    if trace_names is None or i >= len(trace_names):
        return None
    name = trace_names[i]
    return None if name is None else str(name)


def _hover_identity(name, trace_names, ndims):
    """Extra properties that give a data trace its hover identity (1.1
    release review, maintainer finding: plotly showed "trace 0", "trace 1",
    ... on hover because unlabelled data traces had no `name`).

    * a name shared by several data traces (the runs of one hue/cluster
      category, a hierarchy group's leaves and means) becomes their
      `legendgroup`, so the one legend entry of the group toggles all of
      them;
    * a trace with no name at all (a lone, unlabelled dataset) gets a
      `hovertemplate` with an empty ``<extra></extra>``: plotly would
      otherwise print "trace 0" in the name box, a label that names
      nothing -- the coordinates alone are shown.

    Returns ``{}`` when `trace_names` was not given (a direct `plotly_draw`
    caller), keeping that path exactly as it was.
    """
    if trace_names is None:
        return {}
    if name is None:
        coords = ('x: %{x}<br>y: %{y}<br>z: %{z}' if ndims >= 3
                  else '(%{x}, %{y})')
        return dict(hovertemplate=coords + '<extra></extra>')
    if sum(1 for n in trace_names if n is not None and str(n) == name) > 1:
        return dict(legendgroup=name)
    return {}


def _trace_name(legend, tkwargs, i):
    """This trace's plotly `name`, or None when it has no legend entry.

    `'_nolegend_'` (and any other leading-underscore label) is
    MATPLOTLIB's convention for "keep this artist out of the legend" --
    `plot.py` uses it for every hierarchy leaf, every intermediate mean,
    every unnamed hue group, forecasts and trails. plotly has no such
    convention: a name is just text, so passing the sentinel through made
    it the trace's actual name -- rendered in hover labels ("_nolegend_"
    beside the cursor on every leaf of a MultiIndex plot) and written into
    exported HTML, where a plain list of arrays leaves `name=None`.
    Normalising to None here fixes both while keeping the sentinel's
    meaning: `showlegend` at the call site already excludes a `None` name,
    so exactly the same traces stay out of the legend (its own
    `startswith('_')` test is kept as a belt-and-braces guard for any
    future caller that sets `name` without coming through here).
    """
    label = tkwargs.get('label')
    if label is not None:
        name = str(label)
    elif isinstance(legend, (list, tuple)) and i < len(legend):
        name = str(legend[i])
    else:
        return None
    return None if name.startswith('_') else name


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
                   forecast_schedule=None, forecast_trace_start=None,
                   forecast_trace_specs=None, forecast_trail=0,
                   forecast_antialias=True, forecast_datasets=None,
                   surface=None, surface_colors=None,
                   surface_trace_start=None,
                   surface_dataset_indices=None, data_trace_start=0,
                   morph_tags=None, morph_colors=None, morph_samples=None,
                   morph_loop=False,
                   dynamic_title=None,
                   morph_trace_start=None, morph_mesh_trace_start=None,
                   morph_surface_spec=None, surface_point_colors=None,
                   morph_sampled=None, morph_dup_masks=None,
                   morph_alphas=None, aa_curves=None,
                   frame_hooks=None, segment_titles=None,
                   segment_title_style=None, segment_title_colors=None,
                   ownership=None,
                   forecast_frame_colors=None, forecast_reveal=None,
                   hue_units=None, hue_colors_3d=None):
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
    all -- each trail trace's ``meta['hyp_trail_index']`` is the ORIGINAL
    dataset index that produced it (an animated multicoloured 2-D line's
    trail is several colour-bin traces, like its head), so each frame's
    trail geometry is built from `chemtrails[i]`/`precog[i]`/
    `bullettime[i]` for that SAME original dataset index `i`, not from the
    trail trace's own position. This applies to `animate=True`/
    `'parallel'` AND `animate='serial'` (backend parity, Task 4): the
    `'serial'` branch below builds the SAME per-dataset trail semantics as
    `matplotlib_backend.update_lines_serial` (the ONE currently-revealing
    dataset carries a trail; already-revealed/not-yet-started datasets
    don't), rather than the sliding-window semantics used elsewhere in this
    function.

    `forecast_schedule`/`forecast_trace_start`/`forecast_trace_specs`/
    `forecast_trail`/`forecast_antialias` (predict= during a time-progressing
    animation, 1.1): the forecast traces `plotly_draw` created empty are
    rewritten every frame from the precomputed schedule, by the SAME
    mechanism the chemtrail traces use -- a recorded trace range whose data
    each frame replaces. `forecast_trace_specs[k]` is the ``(dataset, age)``
    pair for `fig.data[forecast_trace_start + k]`; age 0 draws
    ``schedule.polyline(dataset, frame)`` and age N draws the forecast from
    ``trail_frames(frame, forecast_trail)[N - 1]``, so the fan is a PURE
    function of the frame index -- never accumulated -- and an exported
    animation is identical to an interactively-played one. A frame with no
    forecast for a slot (too little history revealed, or fewer past frames
    than the fan is deep) gets EMPTY x/y/z, matching matplotlib's
    hidden-artist state. Wired into BOTH the `'serial'` branch and the
    trailing parallel/`'window'` branch below: a forecast wired into only one
    would be frozen in the other. `'spin'` never receives a schedule (it
    reveals nothing over time, so it keeps the static full-history overlay)
    and `'morph'` refuses `predict=` outright.

    `data_trace_start` (GH #108/#191): the actual `fig.data` index where the
    DATA traces begin -- 0, UNLESS a 2-D `density=`/`surface=` layer was
    seeded at the FRONT of `fig.data` (density= is the only one of the two
    that can coexist with `animate`, since surface= 2-D is static-only).
    Density traces themselves are deliberately never referenced by
    `trace_indices` below (nor `surface_trace_indices`): they are computed
    once from the full data and must stay untouched by every frame update.

    `aa_curves` (antialias=, see `plotly_draw`'s docstring): the per-dataset
    ``(dense, step)`` smoothing curves `plotly_draw` built once via
    `_build_aa_curves`. Every frame draws `_aa_window(aa_curves, idx, a, b)`
    -- the smooth stretch spanning exactly the ORIGINAL rows ``[a:b]`` the
    frame would otherwise have sliced -- for both its head/window traces and
    its chemtrails/precog/bullettime trail traces. All the surrounding index
    math (frame pacing, `window_frames`/`start`/`end`, `revealed`, trail
    bounds) and
    everything derived from it (`windows_by_index` for `surface=` meshes,
    `_window_colors`) keeps working in ORIGINAL rows, untouched. ``None``
    (direct callers) or ``step == 1`` means the raw row slices are drawn,
    exactly as before `antialias=` existed.

    `frame_hooks` (plan 1.1 Task 7, the public `on_frame=` hook): unlike
    matplotlib, where a per-frame updater is CALLED by `FuncAnimation`
    later, plotly builds every frame in the Python loops below, right now,
    before this function returns -- so `on_frame=` fires here too, once
    per frame index, in order. Each of the four branches below (`'spin'`,
    `'morph'`, `'serial'`, and the trailing `else:` covering the shared
    parallel/serial-window/'window' reveal) calls
    ``frame_hooks.record(...)`` then ``frame_hooks.dispatch(fig, None)``
    (plotly has no `Axes` object) immediately BEFORE its own
    ``frames.append(go.Frame(**frame_kwargs))`` -- dispatching before the
    append means a callback that mutates a trace object is captured by the
    `go.Frame` that stores it (mutating afterward would be silently
    dropped). `None` (the default) short-circuits every one of these
    blocks to a no-op via `FrameHooks.dispatch`'s own guard.

    `segment_titles` (plan 1.1 Task 8, `title=` as a per-dataset sequence
    for serial-style animations): only the `'morph'` and `'serial'`
    branches below set a per-frame `layout.title` from it -- `'spin'` and
    the trailing `else:` (parallel/window) never receive a non-`None` value
    here, because `_validate_title` rejects a `title=` list for those
    styles long before `plotly_draw` is even called. The `'morph'` branch
    derives its own segment index from data it already has in hand
    (`seg_idx`) independently of the `frame_hooks` block below it, since
    either may run without the other. The `'serial'` branch instead calls
    `serial_current_index` over the already-built `_shown`/`lengths` AT
    MOST ONCE per frame and shares the result between its `segment_titles`
    and `frame_hooks` consumers (task-8 review, minor finding: these used
    to each call it separately with identical arguments -- byte-identical,
    purely duplicate work) -- guarded so the call itself is still skipped
    entirely when neither consumer is active.
    """
    import plotly.graph_objects as go

    if aa_curves is None:
        aa_curves = [(np.atleast_2d(np.asarray(a, dtype=np.float64)), 1)
                     for a in data]

    # EXACTLY match the matplotlib renderer's pacing: frame_rate frames per
    # second of animation for the full duration (no frame cap), so the two
    # backends play at identical speed, duration, and framerate -- down to
    # the `max(1, ...)` floor, which this used to spell `max(2, ...)`. That
    # lone character made a sub-frame request (`duration * frame_rate`
    # rounding below 1) a 2-frame plotly animation against matplotlib's
    # single still, and -- because this count is also the `total_frames`
    # every dataset's window is paced against -- shifted the pacing itself.
    #
    # This ONE count is resolved before the style branches below, so the
    # floor necessarily applies to all four styles. Matplotlib floored only
    # its parallel/'window' path, so aligning here surfaced that its
    # 'serial' and 'spin' asked `FuncAnimation` for ZERO frames on the same
    # input -- an animation that draws nothing. Both were given the same
    # floor rather than reproducing them here; see `matplotlib_backend`.
    n_frames = max(1, int(round(frame_rate * duration)))
    frames = []
    trace_indices = list(range(data_trace_start, data_trace_start + n_data_traces))
    trail_dataset_indices = trail_dataset_indices or []

    # observation markers (`_observation_marker`): a data or trail trace
    # whose marker size is a per-vertex array over the dataset's whole
    # smoothed curve gets, in every frame, the slice of it that matches the
    # frame's window -- the markers stay on the observations that window
    # holds instead of sliding with the window's start
    _full_sizes = {}
    for _k in range(len(fig.data)):
        _m = fig.data[_k].marker if hasattr(fig.data[_k], 'marker') else None
        _s = None if _m is None else _m.size
        if _s is not None and not np.isscalar(_s):
            _full_sizes[_k] = np.asarray(_s, dtype=float)

    def _marker_window(trace_index, idx, a, b):
        """``{'marker': {'size': ...}}`` for trace `trace_index` (drawing
        dataset `idx`) over ORIGINAL rows ``[a, b)``, or ``{}`` when that
        trace's marker size is a plain scalar."""
        sizes = _full_sizes.get(trace_index)
        if sizes is None or aa_curves is None:
            return {}
        return dict(marker=dict(size=_aa_window_sizes(
            sizes, aa_curves[idx][1], a, b)))

    # A dataset's head (and trail) may be SEVERAL traces -- an animated
    # multicoloured 2-D line is one trace per colour bin (`_hue_line_bins`)
    # -- so frames address each dataset's own traces, found by their tags,
    # in trace order.
    hue_units = hue_units or {}
    _head_traces, _trail_traces = {}, {}
    for _k in range(data_trace_start, data_trace_start + n_data_traces):
        _i = (fig.data[_k].meta or {}).get('hyp_trace_index')
        if _i is not None:
            _head_traces.setdefault(_i, []).append(_k)
    if trail_trace_start is not None:
        for _k in range(trail_trace_start, trail_trace_start + n_trail_traces):
            _i = (fig.data[_k].meta or {}).get('hyp_trail_index')
            if _i is not None:
                _trail_traces.setdefault(_i, []).append(_k)
    if _head_traces:
        trace_indices = [k for i in sorted(_head_traces)
                         for k in _head_traces[i]]
    # per-vertex colour arrays a trace carries over its dataset's whole
    # curve (a multicoloured 3-D line and its trail): every frame sends the
    # slice matching its window, so the colours travel with the data
    # (1.1 release review: frames rewrote only the geometry, painting a late
    # window with the colours of the trajectory's first rows)
    _full_colors = {}
    for _k in list(trace_indices) + [k for ks in _trail_traces.values()
                                     for k in ks]:
        _tr = fig.data[_k]
        for _part in ('line', 'marker'):
            _obj = getattr(_tr, _part, None)
            _c = None if _obj is None else _obj.color
            if _c is not None and not isinstance(_c, str) \
                    and len(_c) > 1:
                _full_colors[(_k, _part)] = list(_c)

    def _dense_span(idx, a, b):
        """Vertices ``v0..v1`` (inclusive) of dataset `idx`'s drawn curve
        that ORIGINAL rows ``[a, b)`` span (`_aa_window`'s arithmetic)."""
        step = max(int(aa_curves[idx][1]), 1)
        return a * step, (b - 1) * step

    def _entries(idx, trace_ids, a, b):
        """The frame payload for the traces `trace_ids` that draw dataset
        `idx`'s head (or trail) over ORIGINAL rows ``[a, b)``, one per
        trace, in order."""
        if not trace_ids:
            return []
        if idx in hue_units:
            unit = hue_units[idx]
            v0, v1 = _dense_span(idx, a, b)
            out = []
            for k in trace_ids:
                meta = fig.data[k].meta or {}
                if 'hyp_hue_bin' in meta:
                    bx, by = _binned_polylines(
                        unit['xs'], unit['ys'], unit['bins']['seg_bin'],
                        meta['hyp_hue_bin'], v0, v1)
                    out.append(go.Scatter(x=bx, y=by))
                else:
                    # the observation markers the window holds
                    mk = unit.get('markers')
                    verts = (np.zeros(0, dtype=int) if mk is None
                             or b <= a else mk['vertices'][
                                 (mk['vertices'] >= v0)
                                 & (mk['vertices'] <= v1)])
                    out.append(go.Scatter(
                        x=unit['xs'][verts], y=unit['ys'][verts],
                        marker=dict(color=[mk['colors'][j] for j in verts]
                                    if mk is not None else [])))
            return out
        k = trace_ids[0]
        seg = _aa_window(aa_curves, idx, a, b)
        extra = dict(_marker_window(k, idx, a, b))
        for part in ('line', 'marker'):
            full = _full_colors.get((k, part))
            if full is not None:
                extra.setdefault(part, {})['color'] = _aa_window_sizes(
                    full, aa_curves[idx][1], a, b)
        if ndims >= 3:
            return [go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                                 **extra)]
        if ndims == 2:
            return [go.Scatter(x=seg[:, 0], y=seg[:, 1], **extra)]
        return [go.Scatter(x=_aa_x(aa_curves[idx][1], a, seg.shape[0]),
                           y=seg[:, 0], **extra)]

    def _artists(entries_by_unit):
        """`FrameContext.artists`: ONE artist per dataset head (then per
        trail) -- a dataset drawn by several colour-bin traces is handed
        over as one `PlotlyTraceGroup`, so a callback (`dataset_fade=`)
        styling ``ctx.artists[i]`` styles the whole of dataset `i`."""
        return tuple(e[0] if len(e) == 1 else PlotlyTraceGroup(e)
                     for e in entries_by_unit if e)
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

    forecast_trace_specs = forecast_trace_specs or []
    has_forecasts = (forecast_schedule is not None
                     and forecast_trace_start is not None
                     and bool(forecast_trace_specs))
    forecast_trace_indices = (
        list(range(forecast_trace_start,
                   forecast_trace_start + len(forecast_trace_specs)))
        if has_forecasts else [])

    def _forecast_frame_data(k, anchor_rows):
        """One geometry update per forecast trace at frame `k`, in
        `forecast_trace_specs` order (so it lines up with
        `forecast_trace_indices`).

        `anchor_rows[dataset]` is that dataset's last REVEALED row this
        frame -- the 1-D x offset the forecast hangs off, generalizing the
        static path's `start = arr.shape[0] - 1` (where everything is
        revealed) to a partially-revealed one. Unused for 2-D/3-D, which
        carry their own coordinates.
        """
        from .forecast import trail_frames
        past = trail_frames(k, forecast_trail) if forecast_trail else []
        out = []
        _colors = forecast_frame_colors or []
        for _spec, (dataset, age) in enumerate(forecast_trace_specs):
            # `dataset` is the FORECAST's index (model-major for a
            # collection); the reveal schedule and the anchor rows are per
            # SOURCE dataset (Codex round 3: an IndexError for two models
            # x hue regrouping)
            _src = (forecast_datasets[dataset]
                    if forecast_datasets is not None
                    and dataset < len(forecast_datasets) else dataset)
            if age == 0:
                fit_frame = k
                pts = forecast_schedule.polyline(dataset, k)
            elif age <= len(past):
                fit_frame = past[age - 1]
                pts = forecast_schedule.polyline(dataset, past[age - 1])
            else:
                fit_frame = None
                pts = None          # fewer past frames than the fan is deep
            # Decision R3, matplotlib parity: the live forecast takes the
            # colour of the run drawing the head NOW; a retained one takes
            # the run that was drawing it when it was FIT, so a boundary
            # crossing does not repaint the historical fan.
            _line = None
            if (forecast_reveal is not None and fit_frame is not None
                    and _spec < len(_colors) and _colors[_spec]):
                _run = forecast_reveal.head_run(_src, fit_frame)
                _colour = _colors[_spec].get(_run)
                if _colour is not None:
                    _line = dict(color=_colour)
            if pts is None or len(pts) < 2:
                out.append(go.Scatter3d(x=[], y=[], z=[]) if ndims >= 3
                           else go.Scatter(x=[], y=[]))
                continue
            # antialias=: a forecast is a LINE, smoothed exactly like the
            # static overlay above (and like matplotlib's per-frame artist,
            # whose `_interp_static_line` is this same call at this same
            # 900-vertex target), so a paused animation is indistinguishable
            # from a static plot
            draw, step = (antialias_line(pts) if forecast_antialias
                          else (pts, 1))
            _extra = {} if _line is None else dict(line=_line)
            _base = fig.data[forecast_trace_indices[_spec]]
            if step != 1 and _base.mode and 'markers' in _base.mode:
                # a `forecast_fmt='o:'` marker on each forecast STEP, not on
                # every vertex of this frame's smoothed curve (the static
                # overlay's rule, `_observation_marker`)
                _sizes = np.zeros(draw.shape[0])
                _sizes[::int(step)] = float(_base.marker.size)
                _extra['marker'] = dict(size=_sizes)
            if ndims >= 3:
                out.append(go.Scatter3d(x=draw[:, 0], y=draw[:, 1],
                                        z=draw[:, 2], **_extra))
            elif ndims == 2:
                out.append(go.Scatter(x=draw[:, 0], y=draw[:, 1], **_extra))
            else:
                out.append(go.Scatter(
                    x=_aa_x(step, anchor_rows.get(_src, 0),
                            draw.shape[0]),
                    y=draw[:, 0], **_extra))
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
            # on_frame= (plan 1.1 Task 7): 'spin' never builds a
            # `frame_traces` list of its own (the FULL dataset is static;
            # only the camera/lighting move) -- publish the traces the
            # frame ACTUALLY renders instead: the figure's shared static
            # data traces, plus (when surfaced) this frame's own re-shaded
            # Mesh3d updates. The leading traces are SHARED across every
            # frame (mutating one is figure-wide); the trailing mesh
            # entries, if any, are per-frame -- see FrameContext's
            # artist-lifetime table.
            _frame_artists = tuple(fig.data[i] for i in trace_indices)
            if surface_trace_indices:
                _frame_artists = _frame_artists + tuple(surf_data)
            if frame_hooks is not None:
                frame_hooks.record(
                    frame=k, n_frames=n_frames, artists=_frame_artists,
                    datasets=tuple(data), style='spin', order='parallel',
                    current_index=None, current_fraction=None,
                    # 'spin' draws every dataset in FULL on every frame --
                    # matplotlib's `update_lines_spin` publishes the same
                    # counts (GH #285).
                    revealed_counts=tuple(
                        np.atleast_2d(np.asarray(a)).shape[0] for a in data),
                    window_bounds=tuple(
                        (0, np.atleast_2d(np.asarray(a)).shape[0])
                        for a in data))
                frame_hooks.dispatch(fig, None)
                if dynamic_title is not None and 'text' in dynamic_title:
                    # GH #285: a callable / `{index...}` title=, computed
                    # for THIS frame by the internal updater `dispatch`
                    # just ran (plot.py). `.setdefault` for the same reason
                    # the per-segment branches use it: a 3-D scene_camera
                    # layout may already be in `frame_kwargs['layout']`.
                    frame_kwargs.setdefault('layout', {})['title'] = (
                        _frame_title_dict(dynamic_title['text'], 0,
                                          segment_title_style,
                                          segment_title_colors))
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
                clouds, morph_samples=morph_samples, loop=morph_loop)
        if morph_loop:
            # see the note in `plotly_draw`'s own setup pass above. Done
            # AFTER the reuse branch too, because `morph_sampled` was
            # already built with `loop=morph_loop` there and so already
            # carries the closing cloud.
            morph_indices = morph_indices + [morph_indices[0]]
        n_morph_datasets = len(morph_indices)
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

            # GH #284: the cloud's `alpha=` rides in the marker's rgba
            # string on the same schedule as its colour (see the initial
            # morph trace in `plotly_draw`); `None` keeps it opaque.
            alpha = _morph.morph_alpha(morph_alphas, seg_idx, step, n_steps)
            color_str = (_rgb_string(color) if alpha is None
                         else _to_plotly_color(color, alpha))
            if ndims >= 3:
                frame_traces = [go.Scatter3d(
                    x=draw_pts[:, 0], y=draw_pts[:, 1], z=draw_pts[:, 2],
                    marker=dict(color=color_str))]
            else:
                frame_traces = [go.Scatter(
                    x=draw_pts[:, 0], y=draw_pts[:, 1],
                    marker=dict(color=color_str))]
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

            # `seg_idx // 2` is a position WITHIN THE MORPH SEQUENCE (0, 1,
            # 2, ... for the 1st, 2nd, 3rd morph-tagged dataset), not a
            # FINAL dataset index -- those only coincide when every dataset
            # is tagged (scalar animate='morph'). `morph_indices` (built
            # above from `morph_tags`) maps sequence position back to the
            # actual dataset index for a partial-tag list (e.g.
            # animate=[None, 'morph', 'morph']), exactly like the simplify
            # guard in `plot.py` already does -- so `title=`'s per-segment
            # lookup and `FrameContext.current_index` agree with each other
            # and with the matplotlib backend's `update_morph`.
            _dataset_idx = morph_indices[seg_idx // 2]
            frame_kwargs = dict(
                name=str(k), data=frame_traces, traces=morph_trace_indices)
            if ndims >= 3:
                frame_kwargs['layout'] = dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if segment_titles is not None:
                # segment PARITY discriminates hold vs. transition, never a
                # fraction (both sweep 0->1 over their own segment) -- the
                # same rule `_make_title_updater` uses on the matplotlib
                # backend, applied here via `seg_idx % 2` directly since
                # this branch already has it in hand. `.setdefault` (not a
                # plain assignment) so this does not clobber the
                # scene_camera layout key the ndims>=3 block above may have
                # just set, and is not clobbered by it either.
                _seg_i = min(_dataset_idx, len(segment_titles) - 1)
                _text = '' if seg_idx % 2 else segment_titles[_seg_i]
                frame_kwargs.setdefault('layout', {})['title'] = (
                    _frame_title_dict(_text, _seg_i, segment_title_style,
                                      segment_title_colors))
            if frame_hooks is not None:
                frame_hooks.record(
                    frame=k, n_frames=n_frames, artists=tuple(frame_traces),
                    # the morph-SAMPLED (morph_samples-capped/matched)
                    # clouds -- what this loop actually draws from -- not
                    # the raw `data`, matching matplotlib's `update_morph`
                    # (`FrameContext.datasets`' own contract: "the arrays
                    # the animation actually DRAWS FROM ... not the raw
                    # input").
                    datasets=tuple(sampled), style='morph', order='serial',
                    current_index=_dataset_idx,
                    current_fraction=step / max(1, n_steps - 1),
                    revealed_counts=None, segment_index=seg_idx,
                    segment_kind='hold' if seg_idx % 2 == 0 else 'transition')
                frame_hooks.dispatch(fig, None)
                if dynamic_title is not None and 'text' in dynamic_title:
                    # GH #285: a callable / `{index...}` title=, computed
                    # for THIS frame by the internal updater `dispatch`
                    # just ran (plot.py). `.setdefault` for the same reason
                    # the per-segment branches use it: a 3-D scene_camera
                    # layout may already be in `frame_kwargs['layout']`.
                    frame_kwargs.setdefault('layout', {})['title'] = (
                        _frame_title_dict(dynamic_title['text'], 0,
                                          segment_title_style,
                                          segment_title_colors))
            frames.append(go.Frame(**frame_kwargs))
    elif animate == 'serial':
        # datasets appear one at a time, each growing into place while
        # earlier ones stay fully drawn (never connected to each other).
        # Trail composition mirrors `matplotlib_backend.update_lines_serial`
        # (backend parity, Task 4): the ONE dataset currently being revealed
        # leads with a short opaque comet-head and trails the rest at 0.3
        # opacity; already-revealed datasets stay fully drawn, and
        # not-yet-started ones stay empty.
        lengths = [np.atleast_2d(a).shape[0] for a in data]
        total_points = sum(lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        has_trails = n_trail_traces > 0
        if has_trails:
            trace_indices = list(trace_indices) + list(range(
                trail_trace_start, trail_trace_start + n_trail_traces))
        # head length in FRAMES, resolved exactly as
        # `matplotlib_backend.animate_plot3D` resolves its own `window_frames`
        _focused = focused if focused is not None else tail_duration
        _uses_focus_window = (any(chemtrails) or any(precog)
                              or any(bullettime))
        _window_duration = _focused if _uses_focus_window else tail_duration
        window_frames = (1 if _window_duration == 0
                         else int(frame_rate * _window_duration))

        for k in range(n_frames):
            revealed = total_points * k / max(1, n_frames - 1)
            frame_traces = []
            trail_traces = []
            # this frame's payload per dataset head / trail, for
            # `FrameContext.artists` (`_artists`)
            head_units, trail_units = [], []
            windows_by_index = {}
            window_colors_by_index = {}
            head_bounds_by_index = {}
            forecast_anchors = {}
            _shown = []
            for idx, (arr, start) in enumerate(zip(data, starts)):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                n_pts = arr.shape[0]
                shown = int(np.clip(revealed - start, 0, n_pts))
                _shown.append(shown)
                ct, pc, bt = chemtrails[idx], precog[idx], bullettime[idx]
                has_trail = has_trails and (ct or pc or bt)

                trail_bounds = None
                if not has_trail:
                    head_bounds = (0, shown)          # plain serial, UNCHANGED
                elif shown <= 0:
                    head_bounds = (0, 0)
                elif shown >= n_pts:
                    head_bounds = (0, n_pts)
                else:
                    w = max(1, int(round(window_frames * n_pts
                                         / max(1, total_points))))
                    head_bounds = (max(0, shown - 1 - w), shown)
                    if (ct and pc) or bt:
                        trail_bounds = (0, n_pts)              # bullettime
                    elif ct:
                        trail_bounds = (0, shown)              # chemtrails
                    else:
                        trail_bounds = (max(0, shown - 1), n_pts)  # precog
                head_bounds_by_index[idx] = head_bounds

                # surface/hue windows follow the FULL revealed portion, as in
                # `matplotlib_backend.update_lines_serial` -- independent of
                # the comet-head trimming above
                windows_by_index[idx] = arr[:shown]
                forecast_anchors[idx] = max(0, shown - 1)
                cols = _window_colors(idx, 0, shown)
                if cols is not None:
                    window_colors_by_index[idx] = cols

                _heads = _entries(idx, _head_traces.get(idx, []),
                                  *head_bounds)
                frame_traces.extend(_heads)
                head_units.append(_heads)

                if has_trails and idx in _trail_traces:
                    # (0, 0) -- an empty window -- when this dataset has a
                    # trail TRACE but no trail THIS frame
                    t0, t1 = (trail_bounds if has_trail
                              and trail_bounds is not None else (0, 0))
                    _trails = _entries(idx, _trail_traces[idx], t0, t1)
                    trail_traces.extend(_trails)
                    trail_units.append(_trails)

            frame_traces.extend(trail_traces)
            frame_kwargs = dict(name=str(k), data=frame_traces,
                                traces=list(trace_indices))
            if ndims >= 3:
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if surface_trace_indices:
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _surface_frame_data(
                                            windows_by_index, angle,
                                            window_colors_by_index))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            if has_forecasts:
                # appended to `frame_kwargs` rather than to `frame_traces`,
                # exactly like the surface meshes above: `frame_traces` is
                # what `frame_hooks.record` publishes as `ctx.artists`, and
                # matplotlib's `ctx.artists` does not include its forecast
                # artists either (they are added by `plot.py` after the
                # animation's own artist list is built).
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _forecast_frame_data(
                                            k, forecast_anchors))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + forecast_trace_indices)
            # `_shown`/`lengths` are ALREADY built above (one entry per
            # dataset, from the per-idx loop just above) -- reused here
            # rather than re-derived, same as `lengths`/`starts` themselves
            # are never re-derived from `data` a second time anywhere in
            # this branch. Computed AT MOST ONCE per frame and shared
            # between the `segment_titles`/`frame_hooks` consumers below
            # (task-8 review, minor finding: these used to each call
            # `serial_current_index` separately with the identical
            # `(_shown, lengths)` arguments -- byte-identical results, so
            # purely duplicate work).
            _serial_idx = _serial_frac = None
            if segment_titles is not None or frame_hooks is not None:
                from .matplotlib_backend import serial_current_index
                _serial_idx, _serial_frac = serial_current_index(_shown,
                                                                  lengths)
            if segment_titles is not None:
                # `.setdefault` (not a plain assignment) so a 3-D
                # scene_camera layout set by the ndims>=3 block above is
                # preserved, not clobbered.
                _seg_i = min(_serial_idx, len(segment_titles) - 1)
                frame_kwargs.setdefault('layout', {})['title'] = (
                    _frame_title_dict(segment_titles[_seg_i], _seg_i,
                                      segment_title_style,
                                      segment_title_colors))
            if frame_hooks is not None:
                frame_hooks.record(
                    frame=k, n_frames=n_frames,
                    artists=_artists(head_units + trail_units),
                    datasets=tuple(data), style='serial', order='serial',
                    current_index=_serial_idx, current_fraction=_serial_frac,
                    revealed_counts=tuple(_shown),
                    # the DRAWN head window -- a trailed dataset's comet
                    # head starts after its trail, not at row 0 (the
                    # matplotlib serial updater reports the same; the
                    # FrameContext contract says every field but the
                    # figure/axes/artists agrees across backends)
                    window_bounds=tuple(
                        tuple(int(v) for v in head_bounds_by_index.get(
                            i, (0, c)))
                        for i, c in enumerate(_shown)))
                frame_hooks.dispatch(fig, None)
                if dynamic_title is not None and 'text' in dynamic_title:
                    # GH #285: a callable / `{index...}` title=, computed
                    # for THIS frame by the internal updater `dispatch`
                    # just ran (plot.py). `.setdefault` for the same reason
                    # the per-segment branches use it: a 3-D scene_camera
                    # layout may already be in `frame_kwargs['layout']`.
                    frame_kwargs.setdefault('layout', {})['title'] = (
                        _frame_title_dict(dynamic_title['text'], 0,
                                          segment_title_style,
                                          segment_title_colors))
            frames.append(go.Frame(**frame_kwargs))
    else:
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
        # shared with matplotlib AND with plot()'s reveal schedule
        # the visible head window is `window_frames` FRAMES long, resolved
        # byte-identically to `matplotlib_backend.animate_plot3D`/`2D` (same
        # int() truncation, same `_window_duration == 0` special case) and
        # then mapped onto each dataset's own rows by `anim_window_bounds`.
        # It used to be derived from the LONGEST dataset's row count and
        # rounded, which mis-sized the window whenever that count was not the
        # frame count AND left every dataset sharing one window -- see
        # `trails.anim_window_bounds`.
        window_frames = head_window_frames(
            frame_rate, tail_duration, focused, animate == 'window',
            chemtrails, precog, bullettime)
        has_trails = n_trail_traces > 0
        if has_trails:
            # trail traces are NOT guaranteed to sit right after the data
            # traces (predict= forecast traces may be appended in between)
            # -- address them by their recorded start index, not by
            # assuming contiguity with n_data_traces.
            trace_indices = list(trace_indices) + list(range(
                trail_trace_start, trail_trace_start + n_trail_traces))
        # every trace's drawn row count, once: the reveal clock needs them
        # every frame and they do not change between frames
        _grid_lengths = [
            np.atleast_2d(np.asarray(a, dtype=np.float64)).shape[0]
            for a in data]
        for k in range(n_frames):
            # ONE clock per source dataset -- the same call the matplotlib
            # updater makes, so the two backends cannot drift (see the
            # `trails` module docstring for the drift this rule prevents).
            frame_windows = None
            if ownership is not None:
                frame_windows = dataset_window_bounds(
                    k, n_frames, ownership, _grid_lengths, window_frames)
            frame_traces = []
            head_units, trail_units = [], []
            windows_by_index = {}
            window_colors_by_index = {}
            forecast_anchors = {}
            # GH #285: the drawn head window per dataset, published as
            # `FrameContext.revealed_counts` / `.window_bounds` below, from
            # the same `_run_window` call that slices the traces.
            head_bounds = []
            for idx, arr in enumerate(data):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                # PER DATASET, exactly as the matplotlib renderer paces it:
                # a 5-row marker dataset beside a 15-row line gets its own
                # rescaled window instead of being clamped into the longest
                # dataset's, which used to slide off its end and leave it
                # blank for most of the animation (`trails` docstring).
                _win = _run_window(frame_windows, idx, arr.shape[0],
                                   k, n_frames, window_frames)
                start, end = _win.head_start, _win.head_end
                head_bounds.append((start, end))
                seg = arr[start:end]
                windows_by_index[idx] = seg
                forecast_anchors[idx] = max(0, end - 1)
                cols = _window_colors(idx, start, end)
                if cols is not None:
                    window_colors_by_index[idx] = cols
                # antialias=: `seg` (ORIGINAL rows) still drives the surface
                # mesh/hue windows above; only the DRAWN vertices are smoothed
                _heads = _entries(idx, _head_traces.get(idx, []), start, end)
                frame_traces.extend(_heads)
                head_units.append(_heads)

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
                    # this dataset's OWN bounds again -- the head loop above
                    # computed the same triple, but only the trailed datasets
                    # need `trail_stop`, and re-deriving beats threading a
                    # parallel dict through the head loop for it
                    _twin = _run_window(frame_windows, idx, arr.shape[0],
                                        k, n_frames, window_frames)
                    if (ct and pc) or bt:
                        t0, t1 = 0, arr.shape[0]
                    elif ct:
                        # `trail_stop = max(0, end - w)`, independently clamped
                        # at 0 rather than derived from the already-clamped
                        # `start` (a `start + 1` off-by-one that used to
                        # overstate this trail by one point at every frame,
                        # not just the early, fully-clamped ones)
                        t0, t1 = 0, _twin.past_stop
                    else:
                        # precog shares the head's last vertex (F05-008) --
                        # via `future_start`, never `end - 1`: a run this
                        # dataset's clock has not reached has `end == 0`, and
                        # `data[-1:]` would draw one point of a future
                        # category from the first frame (Decision R5).
                        t0, t1 = _twin.future_start, arr.shape[0]
                    # antialias=: trail bounds stay ORIGINAL-row indices; the
                    # smooth curve spanning exactly those rows is drawn
                    _trails = _entries(idx, _trail_traces.get(idx, []),
                                       t0, t1)
                    trail_traces.extend(_trails)
                    trail_units.append(_trails)
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
            if has_forecasts:
                # see the identical block in the 'serial' branch above for
                # why this appends to `frame_kwargs`, not to `frame_traces`
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _forecast_frame_data(
                                            k, forecast_anchors))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + forecast_trace_indices)
            if frame_hooks is not None:
                frame_hooks.record(
                    frame=k, n_frames=n_frames,
                    artists=_artists(head_units + trail_units),
                    datasets=tuple(data), style=animate, order='parallel',
                    current_index=None, current_fraction=None,
                    revealed_counts=tuple(e for _, e in head_bounds),
                    window_bounds=tuple(head_bounds))
                frame_hooks.dispatch(fig, None)
                if dynamic_title is not None and 'text' in dynamic_title:
                    # GH #285: a callable / `{index...}` title=, computed
                    # for THIS frame by the internal updater `dispatch`
                    # just ran (plot.py). `.setdefault` for the same reason
                    # the per-segment branches use it: a 3-D scene_camera
                    # layout may already be in `frame_kwargs['layout']`.
                    frame_kwargs.setdefault('layout', {})['title'] = (
                        _frame_title_dict(dynamic_title['text'], 0,
                                          segment_title_style,
                                          segment_title_colors))
            frames.append(go.Frame(**frame_kwargs))

    fig.frames = frames

    # a dynamic (callable / `{index...}`) title only has text now that
    # every frame is built: reserve top margin for the TALLEST title any
    # frame draws (1.1 release review T7) -- every frame is a plain dict,
    # so this is one pass over strings, and it can never clip a later
    # frame the way a frame-0-only measurement could.
    _frame_titles = []
    for _frame in frames:
        _t = getattr(getattr(_frame.layout, 'title', None), 'text', None)
        if _t:
            _frame_titles.append(_t)
    # ...and a title an `on_frame=` callback set on the figure itself: with
    # no title= the layout reserved only 10 px, so it rendered cut off at
    # the top of the canvas (1.1 visual review L11)
    _layout_title = getattr(getattr(fig.layout, 'title', None), 'text', None)
    if frame_hooks is not None and _layout_title:
        _frame_titles.append(_layout_title)
    if _frame_titles:
        _size_px = round(12 * PT_TO_PX)
        if segment_title_style and segment_title_style.get('font'):
            _size_px = segment_title_style['font'].get('size', _size_px)
        elif fig.layout.title and fig.layout.title.font \
                and fig.layout.title.font.size:
            _size_px = fig.layout.title.font.size
        _needed = _title_margin_top(_plotly_title_lines(*_frame_titles),
                                    _size_px, fig.layout.height or 504)
        _current = (fig.layout.margin.t
                    if fig.layout.margin and fig.layout.margin.t is not None
                    else 10)
        if _needed > _current:
            fig.update_layout(margin=dict(t=_needed))
    # Play-button pacing is the TRUE inter-frame interval, `1000 / frame_rate`
    # -- byte-identical to the `interval=` matplotlib hands `FuncAnimation`,
    # and the same rule the GIF/APNG export path above already documents ("NOT
    # 1000*duration/n_frames"). This used to be that forbidden form, which
    # agrees only when `frame_rate * duration` is a whole number: at
    # frame_rate=3, duration=1.4 the browser played 350 ms/frame against
    # matplotlib's 333.33 (5% slow), and a sub-frame request played 50 against
    # matplotlib's 100 (2x fast). The export path was right; this was the one
    # place left deriving playback speed from the frame count.
    frame_ms = 1000.0 / float(frame_rate)
    # Play/Pause controls: laid out horizontally BELOW the plotting area
    # (y < 0 in paper coords, anchored by their top edge) rather than at paper
    # (0, 0). In 3-D the scene floats above that corner so the old placement
    # merely looked cramped, but in 2-D the axes fill the paper area and the
    # controls landed ON the plot's bottom-left corner (maintainer report).
    # `margin.b` is opened up in the same call so they are never clipped;
    # update_layout merges nested dicts, so l/r/t margins are preserved.
    # Symmetric `pad` centers each label in its button (the default padding
    # made 'Play' sit noticeably off-center).
    #
    # A VISIBLE 2-D x axis (`axis_scale='data'`, an `ndims=1` series, a date
    # axis) draws its tick labels -- two lines on a date axis -- and its
    # title exactly where y=-0.06 put the controls, which covered them (1.1
    # release review: the "2020" under the first date tick). The controls
    # then go below that band, and the margin grows to hold both.
    _menu_y, _margin_b = -0.06, _ANIM_BUTTON_MARGIN_B
    _band = _x_axis_band_px(fig, ndims)
    if _band:
        _margin_b = max(_ANIM_BUTTON_MARGIN_B,
                        _band + _ANIM_BUTTON_HEIGHT_PX + 2 * _ANIM_BUTTON_GAP_PX)
        _height = fig.layout.height or int(DEFAULT_FIGSIZE[1] * 100)
        _top = (fig.layout.margin.t if fig.layout.margin
                and fig.layout.margin.t is not None else 10)
        _plot_h = max(_height - _top - _margin_b, 1)
        _menu_y = -(_band + _ANIM_BUTTON_GAP_PX) / _plot_h
    fig.update_layout(
        margin=dict(b=_margin_b),
        updatemenus=[dict(
            type='buttons',
            direction='right',
            showactive=False,
            x=0, xanchor='left',
            y=_menu_y, yanchor='top',
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
