# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import os
import sklearn, flair, umap, hdbscan

defaults = dw.core.get_default_options('config.ini')


def apply_model(data, model, *args, mode='fit_transform', return_model=False, **kwargs):
    if type(model) is list:
        fitted_models = []
        for m in model:
            data, next_fitted = apply_model(data, m, return_model=True, **kwargs)
            fitted_models.append(next_fitted)
        if return_model:
            return data, fitted_models
        else:
            return data

    elif type(model) is dict:
        assert all([hasattr(model, k) for k in ['model', 'args', 'kwargs']]),\
            ValueError('model must have keys "model", "args", and "kwargs"')

        return apply_model(data, model['model'], return_model=return_model,
                           *[model['args'], *args], **dw.core.update_dict(model['kwargs'], kwargs))

    else:
        model = extract_model(model)

    elif callable(model):
        model = apply_defaults(model)(*args, **kwargs)

        # scikit-learn-like?
        if dw.zoo.text.is_sklearn_model(model):
            assert mode in ['fit', 'transform', 'fit_transform']




