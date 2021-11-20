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

from ..core import get_default_options, apply_model, get, has_all_attributes, eval_dict
from ..align import align, pad
from ..cluster import cluster
from ..manip import manip
from ..reduce import reduce

from .static import static_plot, group_mean, match_color, mpl2plotly_color, plot_bounding_box, get_bounds,\
    get_empty_canvas
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

    if dw.zoo.is_dataframe(x):
        x = x.values

    if np.ndim(x) == 3:
        img_colors = np.unique(x.reshape([x.shape[0] * x.shape[1], x.shape[2]]), axis=0)
    else:
        img_colors = np.unique(x, axis=0)
    img_colors = img_colors[np.lexsort(img_colors.T[::-1])]  # colors sorted by row

    n_colors = cmap.shape[0]
    color_bins = np.digitize(np.arange(img_colors.shape[0]), np.linspace(0, img_colors.shape[0], num=n_colors))

    if np.ndim(x) == 3:
        colorized = np.zeros([x.shape[0], x.shape[1], cmap.shape[1]])
    else:
        colorized = np.zeros([x.shape[0], cmap.shape[1]])

    for b in range(1, n_colors):
        inds = match_color(x, img_colors[color_bins == b, :])

        if np.ndim(x) == 3:
            colorized[inds[0], inds[1], :] = cmap[b, :]
        else:
            colorized[inds, :] = cmap[b, :]
    return colorized


def mat2colors(m, **kwargs):
    if type(m) is str:
        return np.atleast_2d(mpl.colors.to_rgb(m))
    elif type(m) is list:
        stacked_m = dw.stack(m)
        stacked_colors = pd.DataFrame(data=mat2colors(stacked_m, **kwargs), index=stacked_m.index)
        return dw.unstack(stacked_colors)
    elif dw.zoo.is_dataframe(m):
        return mat2colors(m.values, **kwargs)

    cmap = get_cmap(kwargs.pop('cmap', eval(defaults['plot']['cmap'])))
    n_colors = cmap.shape[0]

    m = np.squeeze(np.array(m))
    if m.ndim < 2:
        _, edges = np.histogram(m, bins=n_colors - 1)
        bins = np.digitize(m, edges) - 1

        colors = np.zeros([len(m), cmap.shape[1]])
        for i in range(len(edges)):
            colors[bins == i, :] = cmap[i, :]
        return colors

    reducer = kwargs.pop('reduce', 'IncrementalPCA')
    if type(reduce) is not dict:
        reducer = {'model': reducer, 'args': [], 'kwargs': {'n_components': 3}}
    else:
        assert has_all_attributes(reducer, ['model', 'args', 'kwargs']), ValueError(f'invalid reduce model: {reducer}')
        # noinspection PyTypeChecker
        reducer['kwargs'] = dw.core.update_dict(reducer['kwargs'], {'n_components': 3})
    m = reduce(m, model=reducer)
    return colorize_rgb(m, cmap)


def labels2colors(c, **kwargs):
    # noinspection PyShadowingNames
    def helper(x, cmap):
        colors = np.zeros((x.shape[0], cmap.shape[1]))

        if x.shape[1] == 1:
            for i, c in enumerate(np.unique(x)):
                colors[x == c] = colors[i, :]  # FIXME: check this...
        else:
            for i in range(x.shape[0]):
                colors[i, :] = np.dot(x.iloc[i].values, cmap)

        return colors

    stacked_labels = dw.stack(c)
    if stacked_labels.shape[0] == 1:  # discrete labels
        groups = np.unique(stacked_labels)
        max_labels = stacked_labels.copy()
    else:  # mixture proportions
        groups = list(stacked_labels.columns)
        max_labels = pd.DataFrame(index=stacked_labels.index, data=np.argmax(stacked_labels.values, axis=1))
    n_colors = len(groups)
    cmap = get_cmap(kwargs.pop('cmap', eval(defaults['plot']['cmap'])), n_colors=n_colors)

    style = {'mode': kwargs.pop('mode', eval(defaults['plot']['mode']))}

    if 'line' in style['mode'] and 'dash' in kwargs.keys():
        style['dash'] = kwargs['dash']
    if 'marker' in style['mode']:
        style['marker'] = kwargs.pop('marker', eval(defaults['plot']['marker']))
        style['markersize'] = kwargs.pop('markersize', eval(defaults['plot']['markersize']))

    cluster_names = kwargs.pop('cluster_names', [str(g) for g in groups])
    legend_override = {'styles': {n: {'color': mpl2plotly_color(cmap[i, :]),
                                      **style} for i, n in enumerate(cluster_names)},
                       'names': cluster_names}

    if type(c) is list:
        legend_override['labels'] = dw.unstack(max_labels)
    else:
        legend_override['labels'] = dw.unstack(max_labels)[0]

    colors = [x.values for x in dw.unstack(pd.DataFrame(index=stacked_labels.index, data=helper(stacked_labels, cmap)))]
    if dw.zoo.is_dataframe(c):
        colors = colors[0]
    return colors, legend_override


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
    def combo_merge(a, b):
        combos = []
        # noinspection PyShadowingNames
        for i in a:
            for j in b:
                combos.append(i + j)
        return combos

    def sort_by_anti_substring(xs):
        sorted_strings = []
        for a in xs:
            if any([a in x for x in sorted_strings]):
                sorted_strings.append(a)
            else:
                sorted_strings.insert(0, a)
        return sorted_strings

    marker_shapes = ['scatter', 'marker', 'markers', 'bigmarker', 'bigmarkers', 'circle', 'square', 'diamond', 'cross',
                     'triangle-up', 'triangle-down', 'triangle-left', 'triangle-ne', 'triangle-se', 'triangle-sw',
                     'triangle-nw', 'pentagon', 'hexagon', 'hexagon2', 'octagon', 'star', 'hexagram',
                     'star-triangle-up', 'star-triangle-down', 'star-square', 'star-diamond', 'diamond-tall',
                     'diamond-wide', 'hourglass', 'bowtie', 'circle-cross', 'circle-x', 'square-cross', 'square-x',
                     'diamond-cross', 'diamond-x', 'cross-thin', 'x-thin', 'asterisk', 'hash', 'y-up', 'y-down',
                     'y-left', 'y-right', 'line-ew', 'line-ns', 'line-ne', 'line-nw', 'arrow-up', 'arrow-down',
                     'arrow-left', 'arrow-right', 'arrow-bar-up', 'arrow-bar-down', 'arrow-bar-left', 'arrow-bar-right']
    marker_shapes_dict = {'.': 'circle', ',': 'circle', 'o': 'circle', 'v': 'triangle-down', '^': 'triangle-up',
                          '<': 'triangle-left', '>': 'triangle-right', '1': 'star-triangle-down',
                          '2': 'star-triangle-up', '3': 'star-triangle-left', '4': 'star-triangle-right',
                          '8': 'octagon', 's': 'square', 'p': 'pentagon', 'P': 'cross', '*': 'star', 'h': 'hexagon',
                          'H': 'hexagon2', '+': 'cross-thin', 'x': 'x', 'X': 'x-thin', 'd': 'diamond',
                          'D': 'diamond-wide', '|': 'line-ns', '_': 'line-ew'}
    marker_styles = ['', '-open', '-dot', '-open-dot']
    markers = combo_merge(marker_shapes, marker_styles)
    markers.extend(list('.,ov^<>12348spP*hH+xXDd|_'))
    markers = sort_by_anti_substring(markers)
    line_styles = sort_by_anti_substring(['--', '-:', ':', '-', 'line', 'lines'])
    big_markers = ['o', 'big']
    small_markers = [',']
    dash_styles = {'--': 'dash', '-:': 'dashdot', ':': 'dot'}

    # noinspection PyShadowingNames
    def substr_list(style, x):
        """
        style: a style string
        x: a list of substrings

        return: true if any of the strings in x is a substring of s, and false otherwise
        """
        if style is None:
            return ''

        inds = np.array([s in style for s in x])
        if np.any(inds):
            return x[np.where(inds)[0][0]]
        else:
            return ''

    def is_bigmarker(s):
        return substr_list(s, big_markers)

    def is_smallmarker(s):
        return substr_list(s, small_markers)

    def is_matplotlib_marker(s):
        return s in marker_shapes_dict.keys()

    def is_dashed(s):
        return substr_list(s, list(dash_styles.keys()))

    def pop_string(s, sub_s):
        if sub_s in s:
            return sub_s, s.replace(sub_s, '', 1)
        else:
            return None, s

    color = None
    marker = None
    linestyle = None
    dash = None

    # need to parse markers first so that hyphens in marker shape names are parsed correctly
    marker_opts = {}
    for m in markers:
        marker, fmt = pop_string(fmt, m)
        if marker:
            if is_bigmarker(marker):
                marker_opts = {'markersize': eval(defaults['plot']['bigmarkersize'])}
            elif is_smallmarker(marker):
                marker_opts = {'markersize': eval(defaults['plot']['smallmarkersize'])}
            else:
                marker_opts = {'markersize': eval(defaults['plot']['markersize'])}
            if is_matplotlib_marker(marker):
                marker = marker_shapes_dict[marker]
            marker_opts['marker'] = marker
            break

    for i in line_styles:
        linestyle, fmt = pop_string(fmt, i)
        if linestyle:
            if is_dashed(linestyle):
                dash = dash_styles[linestyle]
            break

    try:
        color = mpl.colors.to_rgb(fmt)
    except ValueError:
        pass

    if dash:
        line_opts = {'dash': dash}
    else:
        line_opts = {}

    if marker and linestyle:
        mode = 'lines+markers'
    elif marker:
        mode = 'markers'
    else:
        mode = 'lines'

    # noinspection PyUnboundLocalVariable
    return dw.core.update_dict(eval_dict(defaults['plot'].copy()), {'color': color, 'mode': mode, **marker_opts,
                                                                    **line_opts})


def plot(original_data, *fmt, **kwargs):
    wrangle_ops = ['array', 'dataframe', 'text', 'impute', 'interp']
    wrangle_kwargs = {f'{w}_kwargs': kwargs.pop(f'{w}_kwargs', {}) for w in wrangle_ops}

    # noinspection PyUnusedLocal
    @dw.decorate.interpolate
    def wrangle(f, **opts):
        return f

    data = wrangle(original_data, **wrangle_kwargs)

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

    cmap = kwargs.pop('cmap', eval(defaults['plot']['cmap']))
    color_kwargs = kwargs.pop('color_kwargs', kwargs)
    if clusterers is not None:
        cluster_labels = cluster(data, model=clusterers)
        colors, kwargs['legend_override'] = labels2colors(cluster_labels, cmap=cmap,
                                                          **dw.core.update_dict(kwargs, color_kwargs))
    else:
        if 'color' not in kwargs.keys() or kwargs['color'] is None:
            colors = get_colors(data)
        else:
            colors = kwargs.pop('color', ValueError('error parsing color argument'))

        colors = mat2colors(colors, cmap=cmap, **color_kwargs)
    kwargs['color'] = colors

    if type(data) is list:
        c = np.max([*[d.shape[1] for d in data], 2])
    else:
        c = np.max([data.shape[1], 2])

    renderer = kwargs.pop('renderer', None)
    update_plotly_renderer(backend=renderer)

    fig = kwargs.pop('fig', go.Figure())

    bounding_box = kwargs.pop('bounding_box', False)
    data = pad(data, c=c)

    if bounding_box:
        fig = plot_bounding_box(get_bounds(data), fig=fig)
    else:
        fig = get_empty_canvas(fig=fig)

    kwargs['fig'] = fig
    animate = kwargs.pop('animate', False)

    if animate:
        return Animator(data, **kwargs).build_animation()

    return static_plot(data, **kwargs)
