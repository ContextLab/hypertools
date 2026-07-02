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
"""

import os
import sys
import warnings

import numpy as np


VALID_BACKENDS = ('auto', 'matplotlib', 'plotly')


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


# hypertools' signature aesthetic, mirrored from the matplotlib renderer:
# clean white background, unobtrusive cube outline, no tick labels
_AXIS_STYLE = dict(
    showticklabels=False,
    title='',
    showgrid=False,
    zeroline=False,
    showline=True,
    linecolor='black',
    mirror=True,
)
_SCENE_AXIS_STYLE = dict(
    showticklabels=False,
    title='',
    showgrid=True,
    gridcolor='rgb(220,220,220)',
    zeroline=False,
    showbackground=False,
    showline=True,
    linecolor='black',
    mirror=True,
)


def plotly_draw(data, fmt=None, kwargs_list=None, labels=None, legend=None,
                title=None, animate=False, size=None, show=True,
                save_path=None, frame_rate=50, duration=30, rotations=2,
                elev=10, azim=-60):
    """Render grouped datasets with plotly, mirroring _draw's contract.

    Parameters mirror the relevant subset of hypertools.plot.draw._draw:
    `data` is a list of (n_i, d) arrays with d in (1, 2, 3); `fmt` is a list
    of matplotlib-style format strings (one per trace); `kwargs_list` carries
    per-trace matplotlib kwargs ('color', 'linewidth', 'alpha', 'label').

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
        mode = _fmt_to_mode(fmt[i])
        color = _to_plotly_color(tkwargs.get('color'), tkwargs.get('alpha'))
        width = tkwargs.get('linewidth') or 2
        name = _trace_name(legend, tkwargs, i)

        common = dict(
            mode=mode,
            name=name,
            showlegend=legend is not None and name is not None,
            line=dict(color=color, width=width),
            marker=dict(color=color, size=4),
        )
        if ndims >= 3:
            traces.append(go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], **common))
        elif ndims == 2:
            traces.append(go.Scatter(x=arr[:, 0], y=arr[:, 1], **common))
        else:
            traces.append(go.Scatter(
                x=np.arange(arr.shape[0]), y=arr[:, 0], **common))

    fig = go.Figure(data=traces)

    layout = dict(
        title=title,
        template='plotly_white',
        showlegend=legend is not None,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
    )
    if size is not None:
        layout['width'] = int(size[0] * 100)
        layout['height'] = int(size[1] * 100)

    if ndims >= 3:
        layout['scene'] = dict(
            xaxis=_SCENE_AXIS_STYLE, yaxis=_SCENE_AXIS_STYLE,
            zaxis=_SCENE_AXIS_STYLE,
            camera=dict(eye=_camera_eye(elev, azim)),
            aspectmode='cube',
        )
    else:
        layout['xaxis'] = _AXIS_STYLE
        layout['yaxis'] = _AXIS_STYLE

    fig.update_layout(**layout)

    if animate:
        _add_animation(fig, data, ndims, animate, frame_rate, duration,
                       rotations, elev, azim)

    if save_path is not None:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)

    if show:
        fig.show()

    return fig


def _fmt_to_mode(fmt_str):
    """Map a matplotlib format string to a plotly scatter mode."""
    if fmt_str is None:
        return 'lines'
    has_marker = any(m in fmt_str for m in '.o^vs*+xDdph')
    has_line = ('-' in fmt_str) or (':' in fmt_str)
    if has_marker and has_line:
        return 'lines+markers'
    if has_marker:
        return 'markers'
    return 'lines'


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


def _camera_eye(elev, azim, r=2.0):
    """Convert matplotlib elev/azim (degrees) to a plotly camera eye."""
    elev_r, azim_r = np.deg2rad(elev), np.deg2rad(azim)
    return dict(
        x=r * np.cos(elev_r) * np.cos(azim_r),
        y=r * np.cos(elev_r) * np.sin(azim_r),
        z=r * np.sin(elev_r),
    )


def _add_animation(fig, data, ndims, animate, frame_rate, duration,
                   rotations, elev, azim):
    """Attach frames + play controls: 'spin' rotates the camera; True /
    'parallel' reveals trajectories through a sliding time window."""
    import plotly.graph_objects as go

    n_frames = 90
    frames = []

    if animate == 'spin' and ndims >= 3:
        for k in range(n_frames):
            angle = azim + 360.0 * rotations * k / n_frames
            frames.append(go.Frame(
                name=str(k),
                layout=dict(scene_camera=dict(eye=_camera_eye(elev, angle)))))
    else:
        max_len = max(arr.shape[0] for arr in data)
        window = max(2, max_len // 10)
        for k in range(n_frames):
            end = max(2, int(np.ceil((k + 1) * max_len / n_frames)))
            start = max(0, end - window)
            frame_traces = []
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
            frames.append(go.Frame(name=str(k), data=frame_traces))

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
