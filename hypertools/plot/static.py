# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go

from ..core import get_default_options, eval_dict, get

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


def static_plot(data, **kwargs):
    kwargs = dw.core.update_dict(defaults, kwargs)

    ax = kwargs.pop('ax', plt.gca())
    color = kwargs.pop('color', None)

    if type(data) is list:
        for i, d in enumerate(data):
            opts = {'color': get(color, i), 'ax': ax}
            static_plot(d, **dw.core.update_dict(kwargs, opts))
        return ax

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

            if data.shape[1] == 2:
                ax.plot(data.values[inds, 0], data.values[inds, 1], color=c, **kwargs)
            elif data.shape[1] == 3:
                ax.plot3D(data.values[inds, 0], data.values[inds, 1], data.values[inds, 2], color=c, **kwargs)
            else:
                raise ValueError(f'data must be 2D or 3D (given: {data.shape[1]}D)')

    return ax
