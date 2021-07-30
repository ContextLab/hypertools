# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import os
import sklearn
import flair
import umap
from sklearn.experimental import enable_hist_gradient_boosting, enable_iterative_imputer, enable_halving_search_cv

defaults = dw.core.get_default_options('config.ini')
sklearn_modules = ['calibration', 'cluster', 'compose', 'covariance', 'cross_decompositoin', 'decomposition',
                   'discriminant_analysis', 'ensemble', 'experimental', 'feature_extraction',
                   'feature_extraction.image', 'feature_extraction.text', 'feature_selection',
                   'gaussian_process', 'impute', 'inspection', 'isotonic', 'kernel_approximation',
                   'kernel_ridge', 'linear_model', 'manifold', 'metrics', 'mixture', 'model_selection', 'multiclass',
                   'multioutput', 'naive_bayes', 'neighbors', 'neural_network', 'pipeline', 'preprocessing',
                   'random_projection', 'semi_supervised', 'svm', 'tree']
sklearn_modules = [f'sklearn.{m}' for m in sklearn_modules]
sklearn_modules.append('umap')
flair_embeddings = [f'flair.embeddings.{f}' for f in dir(flair.embeddings) if 'embedding' in f.lower()]


def has_all_attributes(x, attributes):
    return all([hasattr(x, a) for a in attributes])


def has_any_attributes(x, attributes):
    return any([hasattr(x, a) for a in attributes])


def get_model(x):
    if type(x) is str:
        for m in [*sklearn_modules, *flair_embeddings]:
            try:
                exec(f'import {m}.{x}', globals())
                return eval(f'{m}.{x}')
            except ModuleNotFoundError:
                continue
        return None
    elif callable(x):
        if hasattr(x, 'fit') and has_any_attributes(x, ['transform', 'fit_transform', 'predict', 'fit_predict',
                                                        'predict_proba']) or \
                hasattr(x, 'embed'):
            return x
    raise ValueError(f'unknown model: {x}')


def get_sklearn_method(x, mode):
    def helper(m):
        if hasattr(x, m):
            return getattr(x, m)
        elif ('fit' in mode) and has_all_attributes(x, ['fit', m.replace('fit_', '')]):
            return [getattr(x, 'fit'), getattr(x, m.replace('fit_', ''))]
        else:
            raise ValueError(f'mode not supported for {x}: {m}')

    if 'transform' in mode:  # transform or fit_transform
        if hasattr(model, 'transform'):
            return helper(mode)
        elif hasattr(model, 'predict'):
            return helper(mode.replace('transform', 'predict'))
        elif hasattr(model, 'predict_proba'):
            return helper(mode.replace('transform', 'predict_proba'))
    elif 'predict_proba' in mode:  # predict_proba or fit_predict_proba
        if hasattr(model, 'predict_proba'):
            return helper(mode)
        elif hasattr(model, 'transform'):
            return helper(mode.replace('predict_proba', 'transform'))
        elif hasattr(model, 'predict'):
            return helper(mode.replace('predict_proba', 'predict'))
    elif 'predict' in mode:  # predict or fit_predict
        if hasattr(model, 'predict'):
            return helper(mode)
        elif hasattr(model, 'transform'):
            return helper(mode.replace('predict', 'transform'))
        elif hasattr(model, 'predict_proba'):
            return helper(mode.replace('transform', 'predict_proba'))
    return helper(mode)


def apply_model(data, model, *args, return_model=False, **kwargs):
    mode = kwargs.pop('mode', 'fit_transform')

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
        assert all([hasattr(model, k) for k in ['model', 'args', 'kwargs']]), \
            ValueError('model must have keys "model", "args", and "kwargs"')

        return apply_model(data, model['model'], return_model=return_model,
                           *[model['args'], *args], **dw.core.update_dict(model['kwargs'], kwargs))

    else:
        model = dw.core.apply_defaults(get_model(model))(*args, **kwargs)
        if dw.zoo.text.is_hugging_face_model(model):
            return dw.zoo.text.apply_text_model(model, data, *args, mode=mode, return_model=return_model, **kwargs)
        f = get_sklearn_method(model, mode)
        if type(f) is list:
            assert len(f) == 2, ValueError(f'bad mode: {mode}')
            f[0](data)
            transformed_data = f[1](data)
        else:
            transformed_data = f(data)

        if return_model:
            return transformed_data, model
        else:
            return transformed_data
