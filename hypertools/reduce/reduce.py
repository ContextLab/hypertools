# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np

from ..core.model import apply_model
from ..core import get_default_options
from ..align.common import pad


defaults = get_default_options()


def get_n_components(model, **kwargs):
    if 'n_components' in kwargs.keys():
        return kwargs['n_components']

    if type(model) is str:
        if model in ['SparseCoder']:
            if 'dictionary' in kwargs.keys():
                return kwargs['dictionary'].shape[1]
        elif model == 'PPCA':
            return None
        else:
            return defaults[model].copy().pop('n_components', None)
    elif hasattr(model, '__name__'):
        return get_n_components(getattr(model, '__name__'), **kwargs)
    elif type(model) is dict and all([k in ['model', 'args', 'kwargs'] for k in model.keys()]):
        return get_n_components(model['model'], **model['kwargs'])
    else:
        return None


@dw.decorate.apply_stacked
def reduce(data, model='IncrementalPCA', **kwargs):
    # noinspection PyTypeChecker
    n_components = get_n_components(model, **kwargs)

    if (n_components is None) or (data.shape[1] > n_components):
        return apply_model(data, model, search=['sklearn.decomposition', 'sklearn.manifold', 'sklearn.mixture',
                                                'umap', 'ppca'],
                           **dw.core.update_dict(get_default_options()['reduce'], kwargs))
    elif data.shape[1] == n_components:
        transformed_data = data.copy()
    else:
        transformed_data = pad(data, c=n_components)

    return_model = kwargs.pop('return_model', False)
    if return_model:
        return transformed_data, {'model': model, 'args': [], 'kwargs': kwargs}
    else:
        return transformed_data
