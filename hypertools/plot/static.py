# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib as mpl

from ..core import get_default_options, eval_dict, get, fullfact

defaults = eval_dict(get_default_options()['plot'])


def match_color(img, c):
    c = np.atleast_2d(c)

    if np.ndim(img) == 3:
        all_inds = np.squeeze(np.zeros_like(img)[:, :, 0])
    else:
        all_inds = np.squeeze(np.zeros_like(img[:, 0]))
    for i in range(c.shape[0]):
        # noinspection PyShadowingNames
        inds = np.zeros_like(img)
        for j in range(c.shape[1]):
            if np.ndim(img) == 3:
                inds[:, :, j] = np.isclose(img[:, :, j], c[i, j])
            else:
                inds[:, j] = np.isclose(img[:, j], c[i, j])

        all_inds = (all_inds + (np.sum(inds, axis=np.ndim(img) - 1) == c.shape[1])) > 0
    return np.where(all_inds)


def group_mean(x):
    @dw.decorate.apply_unstacked
    def helper(y):
        return pd.DataFrame(y.mean(axis=0)).T

    means = helper(x)
    if hasattr(means.index, 'levels'):
        n_levels = len(means.index.levels)
        if n_levels > 1:
            index = pd.MultiIndex.from_frame(means.index.to_frame().iloc[:, :-1])
            return pd.DataFrame(data=means.values, columns=means.columns, index=index)
    return means


def get_continuous_inds(x):
    x = np.sort(x.ravel())
    diffs = np.diff(x)
    breaks = np.where(diffs > 1)[0]
    if len(breaks) == 0:
        return [x]
    else:
        breaks = np.concatenate([[0], breaks + 1, [len(x)]])
        return [x[breaks[i]:breaks[i+1]] for i in range(len(breaks) - 1)]


def get_empty_canvas(fig=None):
    if fig is None:
        fig = go.Figure()
    fig = fig.to_dict()

    # set 3D properties
    for axis in ['xaxis', 'yaxis', 'zaxis']:
        fig['layout']['template']['layout']['scene'][axis]['showbackground'] = False
        fig['layout']['template']['layout']['scene'][axis]['showgrid'] = False
        fig['layout']['template']['layout']['scene'][axis]['showticklabels'] = False
        fig['layout']['template']['layout']['scene'][axis]['title'] = ''

    # set 2D properties
    for axis in ['xaxis', 'yaxis']:
        fig['layout']['template']['layout'][axis]['showgrid'] = False
        fig['layout']['template']['layout'][axis]['showticklabels'] = False
        fig['layout']['template']['layout'][axis]['title'] = ''
    fig['layout']['template']['layout']['plot_bgcolor'] = 'white'

    return go.Figure(fig)


def mpl2plotly_color(c):
    if type(c) is list:
        return [mpl2plotly_color(i) for i in c]
    else:
        color = mpl.colors.to_rgb(c)
        return f'rgb({color[0]}, {color[1]}, {color[2]})'


def plot_bounding_box(bounds, color='k', width=3, opacity=0.9, fig=None, buffer=0.025):
    def expand_range(x, b):
        length = np.max(x) - np.min(x)
        return [np.min(x) - b * length, np.max(x) + b * length]

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
                edges.append(get_plotly_shape(np.concatenate([vertices[i], vertices[j]], axis=0), mode='lines',
                                              showlegend=False, hoverinfo='skip', name='bounding box', opacity=opacity,
                                              linewidth=width, color=color))
    fig.add_traces(edges)
    if n_dims == 2:
        fig.update_xaxes(range=expand_range(bounds[:, 0], buffer))
        fig.update_yaxes(range=expand_range(bounds[:, 1], buffer))

    return fig


def get_plotly_shape(x, **kwargs):
    def flatten(y):
        if type(y) is list:
            if len(y) == 1:
                return flatten(y[0])
            else:
                return [flatten(i) for i in y]
        elif (not np.isscalar(y)) and dw.zoo.is_array(y):
            return flatten(y.ravel().tolist())
        elif dw.zoo.is_dataframe(y):
            return flatten(y.values)
        else:
            return y

    mode = kwargs.pop('mode', defaults['mode'])
    color = kwargs.pop('color', defaults['color'])

    width = kwargs.pop('linewidth', defaults['linewidth'])
    size = kwargs.pop('markersize', defaults['markersize'])
    edgewidth = kwargs.pop('markeredgewidth', None)

    edgecolor = kwargs.pop('edgecolor', None)
    if edgecolor is None:
        edgecolor = color

    facecolor = kwargs.pop('facecolor', None)
    if facecolor is None:
        facecolor = color

    shape = {}
    if 'line' in mode:
        shape['line'] = {'width': width, 'color': color}
    if 'marker' in mode:
        shape['marker'] = {'color': facecolor, 'size': size}
        if edgewidth is not None:
            shape['marker']['line'] = {'width': edgewidth, 'color': edgecolor}

    shape['x'] = flatten(x[:, 0])
    shape['y'] = flatten(x[:, 1])

    if x.shape[1] == 2:
        return go.Scatter(**dw.core.update_dict(kwargs, shape), mode=mode)
    elif x.shape[1] == 3:
        shape['z'] = flatten(x[:, 2])
        return go.Scatter3d(**dw.core.update_dict(kwargs, shape), mode=mode)
    else:
        raise ValueError(f'data must be 2D or 3D (given: {data.shape[1]}D)')


def static_plot(data, **kwargs):
    kwargs = dw.core.update_dict(defaults, kwargs)

    fig = kwargs.pop('fig', go.Figure())
    color = kwargs.pop('color', None)

    if type(data) is list:
        names = kwargs.pop('name', [str(d) for d in range(len(data))])
        for i, d in enumerate(data):
            opts = {'color': get(color, i), 'fig': fig, 'name': get(names, i)}
            fig = static_plot(d, **dw.core.update_dict(kwargs, opts))
        return fig
    kwargs = dw.core.update_dict({'name': ''}, kwargs)

    color = get(color, range(data.shape[0]), axis=0)
    if dw.zoo.is_multiindex_dataframe(data):
        color_df = pd.DataFrame(color, index=data.index)
        group_means = group_mean(data)
        group_colors = group_mean(color_df).values

        scale_properties = ['linewidth', 'markersize', '_alpha']
        group_kwargs = kwargs.copy()
        for s in scale_properties:
            if s[0] == '_':
                group_kwargs[s] /= defaults['plot']['scale']
            else:
                group_kwargs[s] *= defaults['plot']['scale']
        group_kwargs['color'] = group_colors

        static_plot(group_means, **group_kwargs)

    # remove defaults that shouldn't be passed to plot
    remove_params = ['n_colors', 'scale', 'cmap']
    for r in remove_params:
        kwargs.pop(r, None)

    # TODO: write a helper function to manage 2d and 3d plotting:
    #  - use line collections when multiple colors are specified for a multicolored line plot:
    #    https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html
    #  - check whether we're in scatter mode or line mode
    #  - check whether we're in 3D or 2D mode

    # note on animations and styles: it'd be nice to be able to *separately* specify an animation style
    # and plot style for each dataset (e.g. if data is passed as a list or stacked dataframe)

    # also, could add support for using arbitrary text as markers

    unique_colors = np.unique(color, axis=0)

    for i in range(unique_colors.shape[0]):
        c = unique_colors[i, :]
        c_inds = match_color(color, c)[0]

        for inds in get_continuous_inds(c_inds):
            if len(inds) == 1:
                if inds[0] < data.shape[0] - 1:
                    inds = np.array([inds[0], inds[0] + 1])
                else:
                    inds = np.array([inds[0], inds[0]])

            fig.add_trace(get_plotly_shape(data.values[inds, :], **kwargs, color=mpl2plotly_color(c)))

    return fig
