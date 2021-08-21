# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio

import sys

from scipy.spatial.distance import pdist, squareform

from ..core import get_default_options, apply_model, get, has_all_attributes, fullfact, eval_dict
from ..align import align, pad
from ..cluster import cluster
from ..manip import manip
from ..reduce import reduce

from .static import static_plot, group_mean, match_color
from .animate import Animator

defaults = get_default_options()


def get_env():
    if 'ipykernel_launcher.py' in sys.argv[0]:
        return 'jupyter'
    else:
        return 'python'


def update_plotly_renderer(backend=None):
    if (backend is None) and (get_env() == 'python'):
        pio.renderers.default = 'browser'
    elif backend is not None:
        pio.renderers.default = backend


def get_cmap(cmap, **kwargs):
    n_colors = kwargs.pop('n_colors', eval(defaults['plot']['n_colors']))

    if type(cmap) is str:
        cmap = sns.color_palette(cmap, n_colors=n_colors, as_cmap=False)
    if type(cmap) is sns.palettes._ColorPalette:
        cmap = np.array(cmap)
    return cmap


def colorize_rgb(x, cmap, **kwargs):
    cmap = get_cmap(cmap, **kwargs)
    if cmap is None:
        return x

    colors = np.unique(x.reshape([x.shape[0] * x.shape[1], x.shape[2]]), axis=0)
    colors = colors[np.lexsort(colors.T[::-1])]  # colors sorted by row

    color_bins = np.digitize(np.arange(n_colors), np.linspace(0, n_colors, num=n_colors + 1))
    colorized = np.zeros([x.shape[0], x.shape[1], cmap.shape[1]])

    for b in range(1, n_colors):
        inds = match_color(x, colors[color_bins == b, :])
        colorized[inds[0], inds[1], :] = cmap[b, :]
    return colorized


def mat2colors(m, **kwargs):
    if not (dw.zoo.is_array(m) or dw.zoo.is_dataframe(m)):
        if type(m) is str:
            return np.atleast_2d(mpl.colors.to_rgb(m))
        elif type(m) is list:
            stacked_m = dw.stack(m)
            stacked_colors = pd.DataFrame(mat2colors(stacked_m), index=stacked_m.index)
            return dw.unstack(stacked_colors)
        else:
            return np.atleast_2d(mpl.colors.to_rgb(m))

    cmap = get_cmap(kwargs.pop('cmap', eval(defaults['plot']['cmap'])))
    n_colors = cmap.shape[0]

    m = np.squeeze(np.array(m))
    if m.ndim < 2:
        _, edges = np.histogram(m, bins=n_colors-1)
        bins = np.digitize(m, edges) - 1

        colors = np.zeros([len(m), cmap.shape[1]])
        for i in range(len(edges)):
            colors[bins == i, :] = cmap[i, :]
        return colors

    reducer = kwargs.pop('reduce', 'IncrementalPCA')
    if type(reduce) is not dict:
        reducer = {'model': reduce, 'args': [], 'kwargs': {'n_components': 3}}
    else:
        assert has_all_attributes(reduce, ['model', 'args', 'kwargs']), ValueError(f'invalid reduce model: {reducer}')
        reducer['kwargs'] = dw.core.update_dict(reducer['kwargs'], {'n_components': 3})
    m = reduce(m, model=reducer)
    return colorize_rgb(m, cmap)


def mpl2plotly_color(c):
    if type(c) is list:
        return [mpl2plotly_color(i) for i in c]
    else:
        color = mpl.colors.to_rgb(c)
        return f'rgb({color[0]}, {color[1]}, {color[2]})'


def get_colors(data):
    def safe_len(x):
        if type(x) is list:
            return len(x)
        else:
            return 1

    def helper(x, idx=1):
        if type(x) is list:
            return [helper(d, idx=idx + i) for i, d in enumerate(x)]
        elif dw.zoo.is_multiindex_dataframe(x):
            return x.index.to_frame()
        elif dw.zoo.is_dataframe(x):
            return pd.DataFrame(idx * np.ones_like(x.values[:, 0]), index=x.index)
        else:
            raise ValueError(f'cannot get colors for datatype: {type(x)}')

    if type(data) is list:
        return [helper(d, i * safe_len(d) + 1) for i, d in enumerate(data)]
    else:
        return helper(data)


def parse_style(fmt):
    default_style = eval_dict(defaults['plot'])

    if type(fmt) is not str:
        return dw.core.update_dict({'color': None, 'linestyle': None, 'marker': None}, default_style)

    def pop_string(s, sub_s):
        if sub_s in s:
            return sub_s, s.replace(sub_s, '', 1)
        else:
            return None, s

    markers = list('.,ov^>12348spP*hH+xXDd|_')
    line_styles = ['-', '--', '-.', ':']

    color = None
    for m in markers:
        marker, fmt = pop_string(fmt, m)
        if marker:
            break

    for i in line_styles:
        linestyle, fmt = pop_string(fmt, i)
        if linestyle:
            break

    try:
        color = mpl.colors.to_rgb(fmt)
    except ValueError:
        pass

    # noinspection PyUnboundLocalVariable
    return dw.core.update_dict(default_style, {'color': color, 'marker': marker, 'linestyle': linestyle})


def get_bounds(data):
    x = dw.stack(data)
    return np.vstack([np.nanmin(x, axis=0), np.nanmax(x, axis=0)])


def get_empty_canvas(fig=None):
    if fig is None:
        fig = go.Figure()
    fig = fig.to_dict()

    for axis in ['xaxis', 'yaxis', 'zaxis']:
        fig['layout']['template']['layout']['scene'][axis]['showbackground'] = False
        fig['layout']['template']['layout']['scene'][axis]['showgrid'] = False
        fig['layout']['template']['layout']['scene'][axis]['showticklabels'] = False
        fig['layout']['template']['layout']['scene'][axis]['title'] = ''

    return go.Figure(fig)


def plot_bounding_box(bounds, color='k', width=4, opacity=0.75, fig=None):
    fig = get_empty_canvas(fig=fig)

    color = mpl2plotly_color(color)

    n_dims = bounds.shape[1]

    # TODO: could also pass in a reduction model; if >3D, reduce to 3D prior to plotting
    assert n_dims in [2, 3], ValueError(f'only 2D or 3D coordinates are supported; given: {n_dims}D')

    n_vertices = np.power(2, n_dims)

    lengths = np.abs(np.diff(bounds, axis=0))
    vertices = fullfact(n_dims * [2]) - 1
    vertices = np.multiply(vertices, np.repeat(lengths, n_vertices, axis=0))
    vertices += np.repeat(np.atleast_2d(np.min(bounds, axis=0)), n_vertices, axis=0)

    edges = []
    for i in range(n_vertices):
        for j in range(i):
            # check for adjacent vertex (match every coordinate except 1)
            if np.sum([a == b for a, b in zip(vertices[i], vertices[j])]) == n_dims - 1:
                x = np.concatenate([vertices[i], vertices[j]], axis=0)
                if n_dims == 2:
                    edges.append(go.Scatter(x=x[:, 0].ravel().tolist()[0], y=x[:, 1].ravel().tolist()[0],
                                            mode='lines', showlegend=False, hoverinfo='skip', name='bounding box',
                                            opacity=opacity, line=dict(width=width, color=color)))
                elif n_dims == 3:
                    edges.append(go.Scatter3d(x=x[:, 0].ravel().tolist()[0], y=x[:, 1].ravel().tolist()[0],
                                              z=x[:, 2].ravel().tolist()[0], mode='lines', showlegend=False,
                                              hoverinfo='skip', name='bounding box', opacity=opacity,
                                              line=dict(width=width, color=color)))

    fig.add_traces(edges)
    return fig


@dw.decorate.funnel
def plot(data, *fmt, **kwargs):
    pipeline = kwargs.pop('pipeline', None)

    manipulators = kwargs.pop('manip', None)
    aligners = kwargs.pop('align', None)
    reducers = kwargs.pop('reduce', {'model': 'IncrementalPCA', 'args': [], 'kwargs': {'n_components': 3}})
    clusterers = kwargs.pop('cluster', None)

    assert len(fmt) == 0 or len(fmt) == 1, ValueError(f'invalid format: {fmt}')
    if len(fmt) == 1:
        kwargs = dw.core.update_dict(parse_style(fmt[0]), kwargs)

    if pipeline is not None:
        data = apply_model(data, model=pipeline)

    if manipulators is not None:
        data = manip(data, model=manipulators)

    if aligners is not None:
        data = align(data, model=aligners)

    if reducers is not None:
        data = reduce(data, model=reducers)

    if clusterers is not None:
        colors = cluster(data, model=clusterers)
    else:
        colors = kwargs.pop('color', get_colors(data))

    cmap = kwargs.pop('cmap', eval(defaults['plot']['cmap']))
    color_kwargs = kwargs.pop('color_kwargs', kwargs)
    colors = mat2colors(colors, cmap=cmap, **color_kwargs)
    kwargs['color'] = colors

    if type(data) is list:
        c = np.max([*[d.shape[1] for d in data], 2])
    else:
        c = np.max([data.shape[1], 2])

    renderer = kwargs.pop('renderer', None)
    update_plotly_renderer(backend=renderer)

    fig = kwargs.pop('fig', go.Figure())

    bounding_box = kwargs.pop('bounding_box', True)
    data = pad(data, c=c)

    if bounding_box:
        fig = plot_bounding_box(get_bounds(data), color='k', width=2, fig=fig)

    kwargs['fig'] = fig
    animate = kwargs.pop('animate', False)

    if animate:
        animation_opts = dw.core.update_dict(eval_dict(defaults['animate']), kwargs)
        camera_angles = np.linspace(0, animation_opts['rotations'] * 360,
                                    animation_opts['duration'] * animation_opts['framerate'] + 1)[:-1]
        animator = Animator(ax, data, mode, angles, zooms, animation_opts)
        return animation.FuncAnimation(plt.gcf(), animator, frames=len(camera_angles),
                                       interval=1000 / animation_opts['framerate'], blit=True)

    return static_plot(data, **kwargs)
