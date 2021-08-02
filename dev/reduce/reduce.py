# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np

from ..core.model import apply_model
from ..core import get_default_options
from ..align.common import pad


defaults = get_default_options()


@dw.decorate.apply_stacked
def reduce(data, model='IncrementalPCA', **kwargs):
    n_components = kwargs.pop('n_components', eval(defaults['reduce']['n_components']))

    if data.shape[1] > n_components:
        return apply_model(data, model, search=['decomposition', 'manifold', 'mixture', 'umap', 'ppca'],
                           n_components=n_components, **kwargs)
    elif data.shape[1] == n_components:
        return data
    else:
        return pad(data, c=n_components)
