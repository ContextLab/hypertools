# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from matplotlib import pyplot as plt

from ..core import get_default_options, apply_model, get, has_all_attributes
from ..align import align
from ..cluster import cluster
from ..manip import manip
from ..reduce import reduce

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

    cmap = get_cmap(kwargs.pop('cmap', eval(defaults['plot']['cmap'])))
    # FIXME: STOPPED HERE...

    m = np.array(m)
    if m.ndim < 2:
        pass

    reducer = kwargs.pop('reduce', eval(defaults['reduce']['model']))
    if type(reduce) is not dict:
        reducer = {'model': reduce, 'args': [], 'kwargs': {'n_components': 3}}
    else:
        assert has_all_attributes(reduce, ['model', 'args', 'kwargs']), ValueError(f'invalid reduce model: {reducer}')
        reducer['kwargs'] = dw.core.update_dict(reducer['kwargs'], {'n_components': 3})
    m = reduce(m, model=reducer)
    return colorize_rgb(m, cmap)


def parse_style(fmt):
    if type(fmt) is not str:
        return {'color': None, 'linestyle': None, 'marker': None}

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
    return {'color': color, 'marker': marker, 'linestyle': linestyle}


@dw.decorate.funnel
def plot(data, *fmt, **kwargs):
    pipeline = kwargs.pop('pipeline', None)

    manipulators = kwargs.pop('manip', None)
    aligners = kwargs.pop('align', None)
    reducers = kwargs.pop('reduce', eval(defaults['reduce']['model']))
    clusterers = kwargs.pop('cluster', None)

    if len(fmt) > 0:
        kwargs = dw.core.update_dict(parse_style(fmt[0]), kwargs)

    if pipeline is not None:
        data = apply_model(data, model=pipeline)

    if manipulators is not None:
        data = manip(data, model=manipulators)

    if aligners is not None:
        data = align(data, model=aligners)

    data = reduce(data, model=reducers)

    if clusterers is not None:
        clusters = cluster(data, model=clusterers)
    else:
        clusters = None

    cmap = kwargs.pop('cmap', eval(defaults['plot']['cmap']))
    colors = kwargs.pop('color', None)



# TODO: copy relevant stuff from hypertools_revamp notebook.  key things to do:
#  1.) funnel data (DONE)
#  2.) specify default reduce args if n_dims > 3 after applying other stuff
#  3.) allow optional calls to reduce (overwrite), cluster, manipulate, and align.
#      user specifies order via a list of models (similar to sklearn Pipeline)
#  4.) parse plot-specific arguments (defaults specified in config.ini)
#  5.) handle multi-index dataframes
#  6.) draw bounding box
#  7.) move/position the camera as needed
#  8.) given current plotting backend, generate the plot
#    a.) to generate an animation, do steps 1--6 and then filter data and/or adjust camera for each animation frame

# import holoviews as hv
#
#
# def backend(engine, *args, **kwargs):
#     hv.extension(engine, *args, **kwargs)
#
#
# def init_notebook_mode():
#     hv.notebook_extension()
#
#
# class Plot(object):
#     # noinspection PyShadowingNames
#     def __init__(self, backend, *args, **kwargs):
#         assert (backend in engine.keys()), 'Unknown backend plot engine: ' + backend
#         self.backend = engine[backend]
#         self.args = args
#         self.kwargs = kwargs
#
#     def draw(self, data):
#         return self.plotter(data, *self.args, **self.kwargs)
#
#     def save(self, *args, **kwargs):
#         pass
#
#     def get_args(self):
#         return self.args
#
#     def set_args(self, **args):
#         self.args = args
#
#     def get_kwargs(self):
#         return self.kwargs
#
#     def set_kwargs(self, **kwargs):
#         self.kwargs = kwargs
#
#     def update_kwargs(self, **kwargs):
#         for key, val in kwargs:
#             self.kwargs[key] = val
