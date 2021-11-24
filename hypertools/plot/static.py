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
    img = np.atleast_2d(img)
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
        return [x[breaks[i]:breaks[i + 1]] for i in range(len(breaks) - 1)]


def get_empty_canvas(fig=None):
    if fig is None:
        fig = go.Figure()
    fig = fig.to_dict()

    colors = ['#BCBEC0', '#D1D3D4', '#E6E7E8']

    # set 3D properties
    for i, axis in enumerate(['xaxis', 'yaxis', 'zaxis']):
        # fig['layout']['template']['layout']['scene'][axis]['showbackground'] = False
        # fig['layout']['template']['layout']['scene'][axis]['showgrid'] = False
        fig['layout']['template']['layout']['scene'][axis]['backgroundcolor'] = get(colors, i)
        fig['layout']['template']['layout']['scene'][axis]['showticklabels'] = False
        fig['layout']['template']['layout']['scene'][axis]['title'] = ''

    # set 2D properties
    for axis in ['xaxis', 'yaxis']:
        # fig['layout']['template']['layout'][axis]['showgrid'] = False
        fig['layout']['template']['layout'][axis]['showticklabels'] = False
        fig['layout']['template']['layout'][axis]['title'] = ''
    fig['layout']['template']['layout']['plot_bgcolor'] = get(colors, 0)

    return go.Figure(fig)


def mpl2plotly_color(c):
    if type(c) is list:
        return [mpl2plotly_color(i) for i in c]
    else:
        color = mpl.colors.to_rgb(c)
        return f'rgb({color[0]}, {color[1]}, {color[2]})'


def get_bounds(data):
    x = dw.stack(data)
    return np.vstack([np.nanmin(x, axis=0), np.nanmax(x, axis=0)])


def expand_range(x, b=0.025):
    length = np.max(x) - np.min(x)
    return [np.min(x) - b * length, np.max(x) + b * length]


def plot_bounding_box(bounds, color='k', width=3, opacity=0.9, fig=None, buffer=0.025, simplify=False):
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
                edges.append(get_plotly_shape(np.array(np.concatenate([vertices[i], vertices[j]], axis=0)),
                                              mode='lines', showlegend=False, hoverinfo='skip', name='bounding box',
                                              opacity=opacity, linewidth=width, color=color))

    if simplify:
        return edges

    fig = get_empty_canvas(fig=fig)
    fig.add_traces(edges)
    if n_dims == 2:
        fig.update_xaxes(range=expand_range(bounds[:, 0], b=buffer))
        fig.update_yaxes(range=expand_range(bounds[:, 1], b=buffer))

    return fig


# noinspection PyIncorrectDocstring
def flatten(y, depth=0):
    """
    Turn an array, series, or dataframe into a flat list

    Parameters
    ----------
    :param y: the object to flatten
    Returns
    -------
    :return: the flattened object
    """
    if type(y) is list:
        if len(y) == 1:
            return [flatten(y[0], depth=depth + 1)]
        else:
            return [flatten(i, depth=depth + 1) for i in y]
    elif (not np.isscalar(y)) and dw.zoo.is_array(y):
        return flatten(y.ravel().tolist(), depth=depth + 1)
    elif dw.zoo.is_dataframe(y):
        return flatten(y.values, depth=depth + 1)
    else:
        if depth == 0:
            return [y]
        else:
            return y


def get_plotly_shape(x, **kwargs):
    mode = kwargs.pop('mode', defaults['mode'])
    color = kwargs.pop('color', defaults['color'])

    width = kwargs.pop('linewidth', defaults['linewidth'])
    size = kwargs.pop('markersize', defaults['markersize'])
    symbol = kwargs.pop('marker', defaults['marker'])
    edgewidth = kwargs.pop('markeredgewidth', None)

    edgecolor = kwargs.pop('edgecolor', None)
    if edgecolor is None:
        edgecolor = color

    facecolor = kwargs.pop('facecolor', None)
    if facecolor is None:
        facecolor = color

    dash = kwargs.pop('dash', None)

    shape = {}
    if 'line' in mode:
        shape['line'] = {'width': width, 'color': color}
        if dash is not None:
            shape['line']['dash'] = dash
    if 'marker' in mode:
        shape['marker'] = {'color': facecolor, 'size': size, 'symbol': symbol}
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

    legend_override = kwargs.pop('legend_override', None)
    if (legend_override is not None) and ('showlegend' not in kwargs.keys() or kwargs['showlegend']):
        for n, s in legend_override['styles'].items():
            if (hasattr(data, 'shape') and data.shape[1] == 3) or \
                    any([(hasattr(d, 'shape') and d.shape[1] == 3) for d in data]):
                dummy_coords = np.atleast_2d([None, None, None])
            else:
                dummy_coords = np.atleast_2d([None, None])
            fig.add_trace(get_plotly_shape(dummy_coords, **s, name=n, legendgroup=n))
        kwargs['showlegend'] = False

    # FIXME: fill this in-- add "null" objects to the legend, and then force showlegend to be False for all other
    #  shapes.  also need to change how trace "legendgroups" are inferred; if legend_override is specified (and not
    #  None), shape legendgroups should correspond to cluster labels rather than traces

    if type(data) is list:
        names = kwargs.pop('name', [str(d) for d in range(len(data))])  # FIXME: flagging for updating...
        for i, d in enumerate(data):
            opts = {'color': get(color, i), 'fig': fig, 'name': get(names, i), 'legendgroup': get(names, i)}

            if legend_override is not None:
                lo = legend_override.copy()
                lo['labels'] = lo['labels'][i]

                opts['legend_override'] = lo

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

        fig = static_plot(group_means, **group_kwargs)

    # remove defaults that shouldn't be passed to plot
    remove_params = ['n_colors', 'scale', 'cmap', 'bigmarkersize', 'smallmarkersize']
    for r in remove_params:
        kwargs.pop(r, None)

    # note on animations and styles: it'd be nice to be able to *separately* specify an animation style
    # and plot style for each dataset (e.g. if data is passed as a list or stacked dataframe)

    # also, could add support for using arbitrary text as markers

    unique_colors = np.unique(color, axis=0)
    for i in range(unique_colors.shape[0]):
        c = unique_colors[i, :]
        c_inds = match_color(color, c)[0]

        for j, inds in enumerate(get_continuous_inds(c_inds)):
            if i > 0 or j > 0:
                opts = {'showlegend': False}
            else:
                opts = {}

            if len(inds) == 1:
                if inds[0] < data.shape[0] - 1:
                    inds = np.array([inds[0], inds[0] + 1])
                else:
                    inds = np.array([inds[0], inds[0]])

            if legend_override is not None:
                next_labels = legend_override['labels'].iloc[inds].values
                for k in np.unique(next_labels):
                    group_inds = np.where(next_labels == k)[0]
                    group_opts = opts.copy()
                    group_opts['legendgroup'] = legend_override['names'][k]
                    fig.add_trace(get_plotly_shape(data.values[inds[group_inds], :],
                                                   **dw.core.update_dict(kwargs, group_opts),
                                                   color=mpl2plotly_color(c)))
            else:
                fig.add_trace(get_plotly_shape(data.values[inds, :],
                                               **dw.core.update_dict(kwargs, opts),
                                               color=mpl2plotly_color(c)))

    return fig
