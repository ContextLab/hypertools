# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from .backend import manage_backend

from ..core import get_default_options, eval_dict, get

defaults = eval_dict(get_default_options()['plot'])


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


@manage_backend
def static_plot(data, **kwargs):
    kwargs = dw.core.update_dict(defaults, kwargs)

    ax = kwargs.pop('ax', plt.gca())
    color = kwargs.pop('color', None)

    if type(data) is list:
        for i, d in enumerate(data):
            opts = {'color': get(color, i), 'ax': ax}
            return static_plot(d, **dw.core.update_dict(kwargs, opts))

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
    if data.shape[1] == 2:
        ax.plot(data.values[:, 0], data.values[:, 1], color=color, **kwargs)
    elif data.shape[1] == 3:
        ax.plot3D(data.values[:, 0], data.values[:, 1], data.values[:, 2], color=color, **kwargs)
    else:
        raise ValueError(f'data must be 2D or 3D (given: {data.shape[1]}D)')

    return ax.get_lines()