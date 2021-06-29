from .srm import SRM
from .procrustes import Procrustes
from .hyperalign import Hyperalign
from .null import NullAlign

import numpy as np

from ..decorate import apply_defaults

def pad(x, c, max_rows=None):
    if not max_rows:
        max_rows = x.shape[0]

    y = np.zeros([max_rows, c])
    y[:, :x.shape[1]] = x[:max_rows, :]
    return y


def trim_and_pad(data):
    r = np.min([x.shape[0] for x in data])
    c = np.max([x.shape[1] for x in data])
    x = [pad(d, c, max_rows=r) for d in data]
    return x


def align(data, target=None, model='hyper', return_model=False, **kwargs):
    model_dict = {'hyper': Hyperalign,
                  'procrustes': Procrustes,
                  'SRM': SRM,
                  None: NullAlign}

    if (model is None) or (type(model) == str):
        if model not in model_dict.keys():
            raise RuntimeError('Model not supported: ' + model)
        model = model_dict[model]
    else:
        if not (hasattr(model, 'fit') and hasattr(model, 'transform') and hasattr(model, 'fit_transform')):
            raise RuntimeError('Model must have fit, transform, and fit_transform methods')

    aligner = apply_defaults(model(**kwargs))
    data = aligner.fit_transform(trim_and_pad(data))
    if return_model:
        return data, aligner
    else:
        return data
    

