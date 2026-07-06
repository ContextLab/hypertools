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

import os
import sys
import warnings

import numpy as np

from .meshutil import blinn_phong_vertex_colors, points_enclosed
from .surface import (
    PLOTLY_IDENTITY_LIGHTING,
    PLOTLY_LIGHTPOSITION,
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

# matplotlib sizes are in points; plotly sizes are in pixels (1pt = 4/3 px)
PT_TO_PX = 4.0 / 3.0
DEFAULT_FIGSIZE = (6.4, 4.8)  # matplotlib rcParams['figure.figsize'] inches
DEFAULT_LINEWIDTH_PT = 1.5   # matplotlib rcParams['lines.linewidth']
DEFAULT_MARKERSIZE_PT = 6.0  # matplotlib rcParams['lines.markersize']
CUBE_LINEWIDTH_PT = 1.5      # hypertools' wireframe cube linewidth (1pt in
                             # matplotlib; slightly heavier here because
                             # plotly's 3D line antialiasing renders lighter)

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
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {VALID_BACKENDS}; got {backend!r}")
    if backend == 'auto':
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
    ~10/(9 - zoom) (see draw.py's set_box_aspect conversion), so relative to
    zoom=1 the plotly camera moves in by (9 - zoom)/8."""
    return max(0.2, 1.95 * (9.0 - float(zoom)) / 8.0)


# ANIMATED plots pull the camera slightly farther back than static plots so
# the wireframe box keeps a comfortable margin at every rotation angle and is
# never clipped (Jeremy's animated-plot zoom-out request). Static plots are
# visually unchanged -- they keep using _zoom_r directly.
_ANIM_ZOOM_OUT = 1.1


def _anim_zoom_r(zoom):
    """Camera distance for ANIMATED plots: _zoom_r zoomed out by _ANIM_ZOOM_OUT."""
    return _zoom_r(zoom) * _ANIM_ZOOM_OUT


def plotly_draw(data, fmt=None, kwargs_list=None, labels=None, legend=None,
                title=None, animate=False, size=None, show=True,
                save_path=None, frame_rate=30, duration=30, rotations=1,
                elev=10, azim=-60, point_colors=None, tail_duration=2,
                chemtrails=False, precog=False, bullettime=False, zoom=1,
                forecasts=None, colorbar_info=None, surface=None,
                surface_colors=None, density=None, density_colors=None,
                morph_tags=None, morph_colors=None, morph_samples=None):
    """Render grouped datasets with plotly, mirroring _draw's contract and
    the matplotlib renderer's appearance.

    Parameters mirror the relevant subset of hypertools.plot.matplotlib_backend._draw:
    `data` is a list of (n_i, d) arrays with d in (1, 2, 3), already
    centered and scaled to [-1, 1]; `fmt` is a list of matplotlib-style
    format strings (one per trace); `kwargs_list` carries per-trace
    matplotlib kwargs ('color', 'linewidth', 'linestyle', 'marker',
    'alpha', 'label').

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

    Returns the plotly Figure.
    """
    import plotly.graph_objects as go

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
    # `plot.py` already raises `NotImplementedError` for 2-D data before
    # ever calling this backend; this is a defensive re-check (mirrors
    # `broadcast_trail_flag`'s own defensive re-normalization above) for
    # direct callers (tests) that bypass `plot.py`.
    if animate == "morph" and ndims != 3:
        raise NotImplementedError(
            "animate='morph' is only supported for 3-D plots; got "
            f"{ndims}-D data."
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
        mode, symbol, dash = _parse_fmt(fmt[i], tkwargs)
        color = _to_plotly_color(tkwargs.get('color'), tkwargs.get('alpha'))
        width = float(tkwargs.get('linewidth')
                      or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
        msize = float(tkwargs.get('markersize')
                      or DEFAULT_MARKERSIZE_PT) * PT_TO_PX
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

        # surface= (GH #109 rendering-fix), 3-D only: plotly cannot always
        # correctly depth-composite Scatter3d points enclosed by an opaque
        # Mesh3d surface (they can visibly "punch through" the mesh as a
        # hole -- see `_trim_faces_inside_other_meshes`'s docstring for the
        # full story and verification). Points this dataset's own surface
        # encloses are dropped (set to NaN, plotly's standard "no point
        # here" convention) from its marker/line trace instead -- they
        # would be hidden behind the opaque surface anyway; any points the
        # surface fails to enclose (smoothing/inflation only targets ~96%+
        # containment, not 100%) are left visible as before.
        if (ndims >= 3 and not hide_points and surface is not None
                and i < len(surface) and surface[i] is not None):
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
        mode, symbol, dash = _parse_fmt(fmt[i], tkwargs)
        color = _to_plotly_color(tkwargs.get('color'), 0.3)
        width = float(tkwargs.get('linewidth')
                      or DEFAULT_LINEWIDTH_PT) * PT_TO_PX
        msize = float(tkwargs.get('markersize')
                      or DEFAULT_MARKERSIZE_PT) * PT_TO_PX
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
        # a surface list with their entries forced to None; their FULL
        # (unsampled) cloud's hull is still measured (below) so the axes
        # cube is sized to contain it as a safe upper bound, mirroring
        # `matplotlib_backend.animate_plot3D`'s identical tradeoff.
        surface_for_static = (
            [None if (morph_tags is not None and morph_tags[i]) else s
             for i, s in enumerate(surface)]
            if morph_tags is not None else surface
        )
        surface_traces_3d, surface_dataset_indices, surface_meshes = (
            _build_surface_traces_3d(go, data, surface_for_static,
                                     surface_colors, elev, azim))
        traces.extend(surface_traces_3d)
        morph_full_meshes_for_scale = []
        if morph_tags is not None:
            for i, tag in enumerate(morph_tags):
                if tag and surface[i] is not None:
                    pts_i = np.atleast_2d(
                        np.asarray(data[i], dtype=np.float64))[:, :3]
                    m = build_mesh_3d(pts_i, surface[i], dataset_label=f' {i}',
                                      quiet=True)
                    if m is not None:
                        morph_full_meshes_for_scale.append(m)
        cube_scale = surface_cube_scale(
            list(surface_meshes.values()) + morph_full_meshes_for_scale)

    # animate='morph': ONE traveling point-cloud trace (+ one Mesh3d trace
    # if any morphing dataset requests a surface), appended after every
    # normal data/trail/surface trace. `morph_trace_start_3d`/
    # `morph_mesh_trace_start_3d` record their positions for
    # `_add_animation`'s 'morph' branch.
    morph_trace_start_3d = None
    morph_mesh_trace_start_3d = None
    morph_surface_spec_3d = None
    if morph_tags is not None and ndims >= 3:
        morph_indices_3d = [i for i, t in enumerate(morph_tags) if t]
        clouds0 = [np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :3]
                  for i in morph_indices_3d]
        sampled0 = _morph.sample_and_match_clouds(
            clouds0, morph_samples=morph_samples)
        ds_colors0 = [
            tuple(morph_colors[i]) if morph_colors is not None
            else (0.2, 0.4, 0.8)
            for i in morph_indices_3d
        ]
        for i in morph_indices_3d:
            if surface is not None and i < len(surface) and surface[i] is not None:
                morph_surface_spec_3d = surface[i]
                break

        pts0 = sampled0[0]
        color0_str = _rgb_string(ds_colors0[0])
        msize0 = float((kwargs_list[morph_indices_3d[0]] or {}).get('markersize')
                       or DEFAULT_MARKERSIZE_PT) * PT_TO_PX
        hide_morph_points = (morph_surface_spec_3d is not None and
                            not morph_surface_spec_3d.get('keep_points', True))
        morph_trace_start_3d = len(traces)
        traces.append(go.Scatter3d(
            x=pts0[:, 0], y=pts0[:, 1], z=pts0[:, 2], mode='markers',
            marker=dict(color=color0_str, size=msize0, symbol='circle'),
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

    # match matplotlib: centered black title (12pt = 16px), default canvas
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
    layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=legend is not None,
        margin=dict(l=10, r=margin_r, t=40 if title else 10, b=10),
        legend=dict(bgcolor='rgba(255,255,255,0.8)',
                    x=1.02, y=0.5, xanchor='left', yanchor='middle'),
    )
    if title is not None:
        # centered over the plotting area (xref='paper'), like matplotlib
        # centers its title over the axes; same 12pt sans-serif appearance
        layout['title'] = dict(text=title, x=0.5, xanchor='center',
                               xref='paper',
                               y=0.97, yanchor='top',
                               font=dict(color='black', size=16,
                                         family='DejaVu Sans, Arial, '
                                                'sans-serif'))
    size = size if size is not None else DEFAULT_FIGSIZE
    layout['width'] = int(size[0] * 100)
    layout['height'] = int(size[1] * 100)

    if ndims >= 3:
        layout['scene'] = dict(
            xaxis=dict(visible=False, range=[-cube_scale, cube_scale]),
            yaxis=dict(visible=False, range=[-cube_scale, cube_scale]),
            zaxis=dict(visible=False, range=[-cube_scale, cube_scale]),
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
        layout['xaxis'] = dict(visible=False, range=[-1.1, 1.1])
        layout['yaxis'] = dict(visible=False, range=[-1.1, 1.1])
        layout['shapes'] = [_square_shape()]
    else:
        layout['xaxis'] = dict(visible=False)
        layout['yaxis'] = dict(visible=False)

    fig.update_layout(**layout)

    if animate:
        _add_animation(fig, data, ndims, animate, frame_rate, duration,
                       rotations, elev, azim, n_data_traces,
                       tail_duration=tail_duration, chemtrails=chemtrails,
                       precog=precog, bullettime=bullettime, zoom=zoom,
                       n_trail_traces=n_trail_traces,
                       trail_trace_start=trail_trace_start,
                       trail_dataset_indices=trail_dataset_indices,
                       surface=surface, surface_colors=surface_colors,
                       surface_trace_start=surface_trace_start_3d,
                       surface_dataset_indices=surface_dataset_indices,
                       data_trace_start=data_trace_start,
                       morph_tags=morph_tags, morph_colors=morph_colors,
                       morph_samples=morph_samples,
                       morph_trace_start=morph_trace_start_3d,
                       morph_mesh_trace_start=morph_mesh_trace_start_3d,
                       morph_surface_spec=morph_surface_spec_3d)

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
        else:
            fig.show()

    return fig


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


def _export_animation_file(fig, save_path, frame_rate, duration, size):
    """Export a plotly animation to .gif, .png/.apng, or .mp4/.mov/.avi.

    Each frame is rendered to a PNG via kaleido and the sequence is
    assembled with Pillow (gif / animated png) or ffmpeg (video formats).
    """
    import io
    import os
    import subprocess
    import tempfile

    import plotly.graph_objects as go
    from PIL import Image

    size = size if size is not None else DEFAULT_FIGSIZE
    width, height = int(size[0] * 100), int(size[1] * 100)
    ext = save_path.lower().rsplit('.', 1)[-1]

    def frame_snapshots():
        for frame in fig.frames:
            snapshot = go.Figure(fig)
            snapshot.frames = ()
            # hide the interactive play/pause controls in exported frames
            # (update_layout(updatemenus=[]) is a no-op; assign directly)
            snapshot.layout.updatemenus = ()
            if frame.layout:
                snapshot.update_layout(frame.layout)
            if frame.data:
                indices = frame.traces if frame.traces is not None \
                    else range(len(frame.data))
                for idx, trace in zip(indices, frame.data):
                    snapshot.data[idx].update(trace)
            yield snapshot

    if ext == 'svg':
        # vector export: render each frame as SVG and stitch them into one
        # SMIL-animated SVG
        from .._shared.animated_svg import combine_frames_svg
        frame_svgs = [
            snapshot.to_image(format='svg', width=width,
                              height=height).decode('utf-8')
            for snapshot in frame_snapshots()]
        with open(save_path, 'w') as f:
            f.write(combine_frames_svg(frame_svgs, max(1.0, duration)))
        return

    # exported files contain EVERY animation frame (frame_snapshots iterates
    # the full fig.frames -- no subsampling). Frame subsampling is reserved
    # for the interactive-HTML embedding path (_show_sphinx_gallery), where
    # it caps embedded-file size; an exported gif/png/mp4 must never be
    # subsampled or it would play back too fast.
    images = []
    for snapshot in frame_snapshots():
        png = snapshot.to_image(format='png', width=width, height=height)
        images.append(Image.open(io.BytesIO(png)).convert('RGB'))

    # per-frame delay is the TRUE inter-frame interval (1000 / frame_rate),
    # tied to the requested framerate -- NOT 1000*duration/n_frames. With the
    # full frame set (n_frames == frame_rate*duration) the two agree, but
    # deriving the delay from frame_rate keeps real-time playback correct and
    # decoupled from the frame count (a regression guard against any future
    # subsample-and-compensate creeping into the export path).
    frame_ms = max(1, int(round(1000.0 / max(float(frame_rate), 1e-6))))

    if ext == 'gif':
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=frame_ms, loop=0)
    elif ext in ('png', 'apng'):
        target = save_path if ext == 'png' else save_path[:-5] + '.png'
        images[0].save(target, format='PNG', save_all=True,
                       append_images=images[1:], duration=frame_ms, loop=0)
        if target != save_path:
            os.replace(target, save_path)
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

    Mirrors draw.py's plot_cube: 12 edges at +/-scale, black, 1pt lines.
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
        line=dict(color='black', width=linewidth_pt * PT_TO_PX),
        showlegend=False, hoverinfo='skip')


def _square_shape(scale=1.0, linewidth_pt=CUBE_LINEWIDTH_PT):
    """hypertools' 2D black square frame (mirrors draw.py's plot_square)."""
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


def _blend_toward_white(color_rgb, alpha):
    """Alpha-composite `color_rgb` over a white background (matching this
    module's `paper_bgcolor='white'`), returning the resulting flat RGB.

    Used to FAKE a translucent look for the plotly surface mesh (GH #109
    rendering-fix) instead of asking plotly's Mesh3d for real `opacity <
    1`: plotly's WebGL renderer has a documented, currently-unfixed
    depth-sorting limitation for translucent meshes
    (https://github.com/plotly/plotly.py/issues/3554 -- "an overlay of
    multiple transparent surfaces may not perfectly be sorted in depth by
    the webgl API") that manifests as severe per-triangle speckle/faceting
    on these densely-tessellated smooth-hull meshes for ANY `opacity < 1`,
    even values as close to 1 as 0.999. Baking the requested alpha into an
    always-opaque color sidesteps that rendering path entirely -- plotly's
    own bug report confirms "setting opacity to 1 removes these artifacts".
    """
    return tuple(alpha * c + (1.0 - alpha) * 1.0 for c in color_rgb)


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

    `opacity` (the user-requested `surface['alpha']`) is NOT passed through
    to plotly's own opacity/blending -- see `_blend_toward_white` -- the
    trace is always rendered fully opaque (`opacity=1.0`), which plotly's
    Mesh3d renders correctly and without artifacts. The alpha-composite
    adjustment is instead baked into the base color BEFORE per-vertex
    shading (so the shading itself is computed against the correct,
    already-blended base color, exactly like the matplotlib path shades its
    own unblended base color and then relies on real alpha compositing).

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
    blended = _blend_toward_white(color_rgb, opacity)
    vertexcolor = _vertexcolor_strings(verts, faces, blended, view, light_kw)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces_both_windings[:, 0], j=faces_both_windings[:, 1],
        k=faces_both_windings[:, 2],
        vertexcolor=vertexcolor, opacity=1.0, flatshading=False,
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


def _build_surface_traces_3d(go, data, surface, surface_colors, elev, azim):
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
    blended = _blend_toward_white(color_rgb, opacity)
    vertexcolor = _vertexcolor_strings(verts, faces, blended, view, light_kw)
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces_both_windings[:, 0], j=faces_both_windings[:, 1],
                     k=faces_both_windings[:, 2], vertexcolor=vertexcolor)


def _parse_fmt(fmt_str, tkwargs):
    """Convert a matplotlib format string + kwargs into plotly
    (mode, marker symbol, line dash)."""
    fmt_str = fmt_str or ''
    symbol = 'circle'
    dash = 'solid'
    has_marker = False
    has_line = False

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
            break

    if kw_marker is not None and kw_marker in _MARKER_SYMBOLS:
        has_marker = True
        symbol = _MARKER_SYMBOLS[kw_marker]
    if kw_linestyle is not None:
        has_line = True
        dash = _LINESTYLE_NAMES.get(kw_linestyle, 'solid')

    if has_marker and has_line:
        mode = 'lines+markers'
    elif has_marker:
        mode = 'markers'
    else:
        mode = 'lines'
    return mode, symbol, dash


def _colorbar_trace(go, colorbar_info, ndims, legend_present):
    """A hidden ("phantom") marker trace whose sole purpose is to carry a
    plotly colorbar (GH #100) -- plotly attaches colorbars to a trace's
    `marker`/`line`, not to the figure directly, so a real (invisible: a
    single ``None``-positioned point, `opacity=0`) trace is the standard
    way to show one without any visible marker of its own. `location`
    controls which side of the plot the colorbar sits on; the default
    ('right') is pushed further right than an existing legend so the two
    never overlap (mirrors the matplotlib backend's `_add_right_colorbar`)."""
    from .colors import get_palette_colors

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
             thickness=15 if orientation == 'v' else 15)
    if colorbar_info.get('label'):
        cb['title'] = dict(text=colorbar_info['label'])

    if colorbar_info['kind'] == 'continuous':
        colors = get_palette_colors(colorbar_info['palette'], 100)
        colorscale = _colors_to_plotly_colorscale(colors)
        cmin, cmax = colorbar_info['vmin'], colorbar_info['vmax']
        if colorbar_info.get('ticks') is not None:
            cb['tickvals'] = list(colorbar_info['ticks'])
    else:
        colors = colorbar_info['colors']
        n = len(colors)
        colorscale = _discrete_plotly_colorscale(colors)
        cmin, cmax = -0.5, n - 0.5
        if colorbar_info.get('ticks') is not None:
            cb['tickvals'] = list(colorbar_info['ticks'])
        else:
            cb['tickvals'] = list(range(n))
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
                   chemtrails=None, precog=None, bullettime=None,
                   zoom=1, n_trail_traces=0, trail_trace_start=None,
                   trail_dataset_indices=None,
                   surface=None, surface_colors=None,
                   surface_trace_start=None,
                   surface_dataset_indices=None, data_trace_start=0,
                   morph_tags=None, morph_colors=None, morph_samples=None,
                   morph_trace_start=None, morph_mesh_trace_start=None,
                   morph_surface_spec=None):
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

    def _surface_frame_data(windows_by_index, angle):
        """One Mesh3d geometry update per surfaced dataset, built from
        `windows_by_index[dataset_idx]` (that dataset's current window),
        shaded (GH #109 round 3) from `angle` -- THIS frame's camera azimuth
        (the camera rotates every frame in every animate mode, matching the
        matplotlib renderer -- see `_shade_and_cull_3d`), so the vertex
        colors stay lit consistently with wherever the camera actually is."""
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
                out.append(_mesh3d_geometry_update(
                    go, v, f, base_rgb, spec['alpha'], view, light_kw))
        return out

    if animate == 'spin' and ndims >= 3:
        # the FULL dataset is static in 'spin' mode (only the camera
        # rotates) -- precompute each surfaced dataset's mesh once
        spin_meshes = None
        if surface_trace_indices:
            spin_meshes = [
                build_mesh_3d(
                    np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :3],
                    surface[i], dataset_label=f' {i}', quiet=True)
                for i in surface_dataset_indices
            ]
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
                for idx, mesh in zip(surface_dataset_indices, spin_meshes):
                    spec = surface[idx]
                    base_rgb = _surface_base_rgb(spec, surface_colors[idx])
                    if mesh is None:
                        surf_data.append(_degenerate_mesh3d_update(
                            go, np.zeros(3), base_rgb))
                    else:
                        v, f = mesh
                        light_kw = mpl_lighting_kwargs(spec)
                        surf_data.append(_mesh3d_geometry_update(
                            go, v, f, base_rgb, spec['alpha'], view, light_kw))
                frame_kwargs['data'] = surf_data
                frame_kwargs['traces'] = surface_trace_indices
            frames.append(go.Frame(**frame_kwargs))
    elif animate == 'morph' and ndims >= 3:
        # Hungarian-matched point-cloud morph (maintainer request): ONE
        # traveling Scatter3d trace (+ one Mesh3d trace if surfaced) eases
        # through the hold/morph schedule; camera eye rotates per
        # `rotations` (scalar: uniform over the whole animation; list:
        # per-segment, continuous across boundaries -- see
        # `hypertools.plot.morph.segment_azimuths`).
        morph_indices = [i for i, t in enumerate(morph_tags or []) if t]
        n_morph_datasets = len(morph_indices)
        clouds = [np.atleast_2d(np.asarray(data[i], dtype=np.float64))[:, :3]
                 for i in morph_indices]
        sampled = _morph.sample_and_match_clouds(
            clouds, morph_samples=morph_samples)
        ds_colors = [
            tuple(morph_colors[i]) if morph_colors is not None
            else (0.2, 0.4, 0.8)
            for i in morph_indices
        ]
        frame_counts = _morph.segment_frame_counts(n_morph_datasets, n_frames)
        rotations_resolved = _morph.resolve_morph_rotations(
            rotations, n_morph_datasets)
        azimuths = _morph.segment_azimuths(frame_counts, rotations_resolved, azim)
        n_frames = sum(frame_counts)

        morph_trace_indices = [morph_trace_start]
        if morph_mesh_trace_start is not None:
            morph_trace_indices.append(morph_mesh_trace_start)

        for k in range(n_frames):
            seg_idx, step, n_steps = _morph.frame_to_segment(frame_counts, k)
            pts = _morph.morph_positions(sampled, seg_idx, step, n_steps)
            color = _morph.morph_color(ds_colors, seg_idx, step, n_steps)
            angle = azimuths[k]

            frame_traces = [go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
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

            frames.append(go.Frame(
                name=str(k), data=frame_traces, traces=morph_trace_indices,
                layout=dict(scene_camera=dict(
                    eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))))
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
            for idx, (arr, start) in enumerate(zip(data, starts)):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                shown = int(np.clip(revealed - start, 0, arr.shape[0]))
                seg = arr[:shown]
                windows_by_index[idx] = seg
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
                                        + _surface_frame_data(windows_by_index, angle))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            frames.append(go.Frame(**frame_kwargs))
    else:
        max_len = max(arr.shape[0] for arr in data)
        # the visible window covers tail_duration seconds of the
        # duration-second animation, matching the matplotlib renderer's
        # tail_duration * frame_rate frame window
        window = max(2, int(round(max_len * float(tail_duration)
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
            for idx, arr in enumerate(data):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                seg = arr[start:min(end, arr.shape[0])]
                windows_by_index[idx] = seg
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
                # while the window advances (draw.py update_lines_parallel);
                # mirror that here
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(
                    scene_camera=dict(eye=_camera_eye(elev, angle, r=_anim_zoom_r(zoom))))
            if surface_trace_indices:
                frame_kwargs['data'] = (list(frame_kwargs['data'])
                                        + _surface_frame_data(windows_by_index, angle))
                frame_kwargs['traces'] = (list(frame_kwargs['traces'])
                                          + surface_trace_indices)
            frames.append(go.Frame(**frame_kwargs))

    fig.frames = frames
    frame_ms = max(10, int(1000.0 * duration / n_frames))
    fig.update_layout(updatemenus=[dict(
        type='buttons',
        showactive=False,
        y=0, x=0, xanchor='left', yanchor='bottom',
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate')]),
        ])])
