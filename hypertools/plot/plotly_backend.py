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


def plotly_draw(data, fmt=None, kwargs_list=None, labels=None, legend=None,
                title=None, animate=False, size=None, show=True,
                save_path=None, frame_rate=30, duration=30, rotations=1,
                elev=10, azim=-60, point_colors=None, tail_duration=2,
                chemtrails=False, precog=False, bullettime=False, zoom=1):
    """Render grouped datasets with plotly, mirroring _draw's contract and
    the matplotlib renderer's appearance.

    Parameters mirror the relevant subset of hypertools.plot.matplotlib_backend._draw:
    `data` is a list of (n_i, d) arrays with d in (1, 2, 3), already
    centered and scaled to [-1, 1]; `fmt` is a list of matplotlib-style
    format strings (one per trace); `kwargs_list` carries per-trace
    matplotlib kwargs ('color', 'linewidth', 'linestyle', 'marker',
    'alpha', 'label').

    Returns the plotly Figure.
    """
    import plotly.graph_objects as go

    fmt = fmt if fmt is not None else ['-'] * len(data)
    kwargs_list = kwargs_list if kwargs_list is not None else [{}] * len(data)

    ndims = data[0].shape[1] if data[0].ndim > 1 else 1
    traces = []
    for i, arr in enumerate(data):
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

        common = dict(
            mode=mode,
            name=name,
            showlegend=legend is not None and name is not None,
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

    n_data_traces = len(traces)

    # low-opacity trail traces for chemtrails (past) / precog (future) /
    # bullettime (both) on window animations, mirroring the matplotlib
    # renderer's alpha-0.3 trail artists. One per dataset, updated per
    # frame; they sit between the data traces and the cube so frame trace
    # indices stay contiguous.
    n_trail_traces = 0
    if animate in (True, 'parallel') and (chemtrails or precog or
                                          bullettime):
        for i, arr in enumerate(data):
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
        n_trail_traces = len(data)

    if ndims >= 3:
        traces.append(_cube_trace(go))

    fig = go.Figure(data=traces)

    # match matplotlib: centered black title (12pt = 16px), default canvas
    # 6.4 x 4.8 inches at 100 dpi, legend to the RIGHT of the plot and
    # vertically centered on the box (same as the matplotlib renderer)
    layout = dict(
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=legend is not None,
        # reserve right margin for the outside legend when one is shown
        margin=dict(l=10, r=120 if legend is not None else 10,
                    t=40 if title else 10, b=10),
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
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            zaxis=dict(visible=False, range=[-1, 1]),
            camera=dict(eye=_camera_eye(elev, azim, r=_zoom_r(zoom))),
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
                       n_trail_traces=n_trail_traces)

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
                   chemtrails=False, precog=False, bullettime=False,
                   zoom=1, n_trail_traces=0):
    """Attach frames + play controls: 'spin' rotates the camera; True /
    'parallel' reveals trajectories through a sliding time window. Frames
    only touch the data traces, so the cube/frame stays put."""
    import plotly.graph_objects as go

    # EXACTLY match the matplotlib renderer's pacing: frame_rate frames
    # per second of animation for the full duration (no frame cap), so the
    # two backends play at identical speed, duration, and framerate
    n_frames = max(2, int(round(frame_rate * duration)))
    frames = []
    trace_indices = list(range(n_data_traces))

    if animate == 'spin' and ndims >= 3:
        for k in range(n_frames):
            angle = azim + 360.0 * rotations * k / n_frames
            frames.append(go.Frame(
                name=str(k),
                layout=dict(scene_camera=dict(eye=_camera_eye(elev, angle, r=_zoom_r(zoom))))))
    elif animate == 'serial':
        # datasets appear one at a time, each growing into place while
        # earlier ones stay fully drawn (never connected to each other)
        lengths = [np.atleast_2d(a).shape[0] for a in data]
        total_points = sum(lengths)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        for k in range(n_frames):
            revealed = total_points * k / max(1, n_frames - 1)
            frame_traces = []
            for arr, start in zip(data, starts):
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                shown = int(np.clip(revealed - start, 0, arr.shape[0]))
                seg = arr[:shown]
                if ndims >= 3:
                    frame_traces.append(go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                elif ndims == 2:
                    frame_traces.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                else:
                    frame_traces.append(go.Scatter(
                        x=np.arange(seg.shape[0]), y=seg[:, 0]))
            frame_kwargs = dict(name=str(k), data=frame_traces,
                                traces=trace_indices)
            if ndims >= 3:
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(
                    scene_camera=dict(eye=_camera_eye(elev, angle, r=_zoom_r(zoom))))
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
            trace_indices = list(range(n_data_traces + n_trail_traces))
        for k in range(n_frames):
            end = max(2, int(np.ceil((k + 1) * max_len / n_frames)))
            start = max(0, end - window)
            frame_traces = []
            trail_traces = []
            for arr in data:
                arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
                seg = arr[start:min(end, arr.shape[0])]
                if ndims >= 3:
                    frame_traces.append(go.Scatter3d(
                        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2]))
                elif ndims == 2:
                    frame_traces.append(go.Scatter(x=seg[:, 0], y=seg[:, 1]))
                else:
                    frame_traces.append(go.Scatter(
                        x=np.arange(start, start + seg.shape[0]),
                        y=seg[:, 0]))
                if has_trails:
                    # chemtrails: past; precog: future; bullettime: both
                    if bullettime:
                        trail = arr
                        t0 = 0
                    elif chemtrails:
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
                                traces=trace_indices)
            if ndims >= 3:
                # matplotlib's sliding-window animation rotates the camera
                # while the window advances (draw.py update_lines_parallel);
                # mirror that here
                angle = azim + 360.0 * rotations * k / n_frames
                frame_kwargs['layout'] = dict(
                    scene_camera=dict(eye=_camera_eye(elev, angle, r=_zoom_r(zoom))))
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
