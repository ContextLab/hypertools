# noinspection PyPackageRequirements
import datawrangler as dw

from ..core import get_default_options, eval_dict

defaults = eval_dict(get_default_options()['plot'])


def static_plot(data, **kwargs):
    kwargs = dw.core.update_dict(defaults, kwargs)

    ax = kwargs.pop('ax', plt.gca())
    color = kwargs.pop('color', None)

    if type(data) is list:
        for i, d in enumerate(data):
            opts = {'color': get(color, i), 'ax': ax}
            return static_plot(d, **dw.core.update_dict(kwargs, opts))

    return ax.plot(*data.split(data.shape[1], axis=1), color=color, **kwargs)