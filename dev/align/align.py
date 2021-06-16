from .srm import SRM
from .procrustes import Procrustes
from .hyperalign import Hyperalign

def align(data, target=None, model='hyper', return_model=False, **kwargs):
    model_dict = {'hyper': Hyperalign,
                  'procrustes': Procrustes,
                  'SRM': SRM}

    if type(model) == str:
        if model not in model_dict.keys():
            raise RuntimeError('Model not supported: ' + model)
        model = model_dict[model]
    else:
        if not (hasattr(model, 'fit') and hasattr(model, 'transform') and hasattr(model, 'fit_transform')):
            raise RuntimeError('Model must have fit, transform, and fit_transform methods')

    # deal with Procrustes as a special case-- either pass in template or use first observation as template
    # in this function, either simply align the data or return model + align data
    # could implement as a class
    # also deal with padding and formatting

