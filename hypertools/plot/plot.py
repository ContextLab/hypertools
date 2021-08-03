# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.animation as animation

from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

from ..core import get_default_options, apply_model, get, has_all_attributes, fullfact, eval_dict
from ..align import align, pad
from ..cluster import cluster
from ..manip import manip
from ..reduce import reduce

from .static import static_plot
from .animate import Animator

defaults = get_default_options()


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

    def match_color(img, c):
        all_inds = np.squeeze(np.zeros_like(img)[:, :, 0])
        for i in range(c.shape[0]):
            # noinspection PyShadowingNames
            inds = np.zeros_like(img)
            for j in range(c.shape[1]):
                inds[:, :, j] = np.isclose(img[:, :, j], c[i, j])
            all_inds = (all_inds + np.sum(inds, axis=2) == c.shape[1]) > 0
        return np.where(all_inds)

    colors = np.unique(x.reshape([x.shape[0] * x.shape[1], x.shape[2]]), axis=0)
    colors = colors[np.lexsort(colors.T[::-1])]  # colors sorted by row

    color_bins = np.digitize(np.arange(n_colors), np.linspace(0, n_colors, num=n_colors + 1))
    colorized = np.zeros([x.shape[0], x.shape[1], cmap.shape[1]])

    for b in range(1, n_colors):
        inds = match_color(x, colors[color_bins == b, :])
        colorized[inds[0], inds[1], :] = cmap[b, :]
    return colorized


def mat2colors(m, **kwargs):
    if not dw.zoo.is_array(m):
        if type(m) is str:
            return np.atleast_2d(mpl.colors.to_rgb(m))
        elif type(m) is list:
            return np.concatenate([mat2colors(c) for c in m], axis=0)
        else:
            return np.atleast_2d(mpl.colors.to_rgb(m))

    cmap = get_cmap(kwargs.pop('cmap', eval(defaults['plot']['cmap'])))
    n_colors = cmap.shape[0]

    m = np.array(m)
    if m.ndim < 2:
        _, edges = np.histogram(m, bins=n_colors)
        bins = np.digitize(m, edges)

        colors = np.zeros([len(m), cmap.shape[1]])
        for i in range(1, len(edges)):
            colors[bins == i, :] = cmap[i, :]
        return colors

    reducer = kwargs.pop('reduce', eval(defaults['reduce']['model']))
    if type(reduce) is not dict:
        reducer = {'model': reduce, 'args': [], 'kwargs': {'n_components': 3}}
    else:
        assert has_all_attributes(reduce, ['model', 'args', 'kwargs']), ValueError(f'invalid reduce model: {reducer}')
        reducer['kwargs'] = dw.core.update_dict(reducer['kwargs'], {'n_components': 3})
    m = reduce(m, model=reducer)
    return colorize_rgb(m, cmap)


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
    return np.concatenate([np.nanmin(data, axis=0), np.nanmax(data, axis=0)], axis=0)


def plot_bounding_box(bounds, color='k', linewidth=2, ax=None):
    if ax is None:
        if bounds.shape[1] == 3:
            ax = plt.gca(projection='3d')
        else:
            assert bounds.shape[1] == 2, ValueError('bounding box must be either 2d or 3d')
            ax = plt.gca()

    n_dims = bounds.shape[1]
    n_vertices = np.power(2, n_dims)

    lengths = np.abs(np.diff(bounds, axis=0))
    vertices = fullfact(n_dims * [2]) - 1
    vertices = np.multiply(vertices, np.repeat(lengths, n_vertices, axis=0))
    vertices += np.repeat(np.atleast_2d(np.min(bounds, axis=0)), n_vertices, axis=0)

    for i in range(n_vertices):
        for j in range(i):
            # check for adjacent vertex (match every coordinate except 1)
            if np.sum([a == b for a, b in zip(vertices[i], vertices[j])]) == n_dims - 1:
                next_edge = np.concatenate([vertices[i], vertices[j]], axis=0)
                ax.plot(*[np.array(x).squeeze() for x in np.split(next_edge, n_dims, axis=1)], color=color,
                        linewidth=linewidth)
    return ax


@dw.decorate.funnel
def plot(data, *fmt, **kwargs):
    pipeline = kwargs.pop('pipeline', None)

    manipulators = kwargs.pop('manip', None)
    aligners = kwargs.pop('align', None)
    reducers = kwargs.pop('reduce', eval(defaults['reduce']['model']))
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

    data = reduce(data, model=reducers)

    if clusterers is not None:
        colors = cluster(data, model=clusterers)
    else:
        colors = kwargs.pop('color', None)

    cmap = kwargs.pop('cmap', eval(defaults['plot']['cmap']))
    if colors is not None:
        color_kwargs = kwargs.pop('color_kwargs', kwargs)
        colors = mat2colors(colors, cmap=cmap, **color_kwargs)
    kwargs['color'] = colors

    if type(data) is list:
        c = np.max([*[d.shape[1] for d in data], 2])
    else:
        c = np.max([data.shape[1], 2])

    if c == 2:
        ax = kwargs.pop('ax', plt.gca())
    elif c == 3:
        ax = kwargs.pop('ax', plt.gca(projection='3d'))
    else:
        raise ValueError(f'data must be 2D or 3D; given: {c}D')

    bounding_box = kwargs.pop('bounding_box', True)
    data = pad(data, c=c)

    if bounding_box:
        plot_bounding_box(get_bounds(data), color='k', linewidth=2, ax=ax)

    animate = kwargs.pop('animate', False)

    if animate:
        animation_opts = dw.core.update_dict(eval_dict(defaults['animate']), kwargs)
        camera_angles = np.linspace(0, animation_opts['rotations'] * 360,
                                    animation_opts['duration'] * animation_opts['framerate'] + 1)[:-1]
        animator = Animator(ax, data, mode, angles, zooms, animation_opts)
        return animation.FuncAnimation(plt.gcf(), animator, frames=len(camera_angles),
                                       interval=1000 / animation_opts['framerate'], blit=True)

    return static_plot(data, **kwargs)
