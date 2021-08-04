# noinspection PyPackageRequirements
import datawrangler as dw
# noinspection PyPackageRequirements
import umap

import numpy as np
import os
import sklearn
import flair
from sklearn.experimental import enable_hist_gradient_boosting, enable_iterative_imputer, enable_halving_search_cv

from ..external import ppca, brainiak

defaults = dw.core.get_default_options('config.ini')
sklearn_modules = ['calibration', 'cluster', 'compose', 'covariance', 'cross_decompositoin', 'decomposition',
                   'discriminant_analysis', 'ensemble', 'experimental', 'feature_extraction',
                   'feature_extraction.image', 'feature_extraction.text', 'feature_selection',
                   'gaussian_process', 'impute', 'inspection', 'isotonic', 'kernel_approximation',
                   'kernel_ridge', 'linear_model', 'manifold', 'metrics', 'mixture', 'model_selection', 'multiclass',
                   'multioutput', 'naive_bayes', 'neighbors', 'neural_network', 'pipeline', 'random_projection',
                   'semi_supervised', 'svm', 'tree']
sklearn_modules = [f'sklearn.{m}' for m in sklearn_modules]
sklearn_modules.append('umap')
flair_embeddings = [f'flair.embeddings.{f}' for f in dir(flair.embeddings) if 'embedding' in f.lower()]
externals = ['ppca', 'brainiak']


def has_all_attributes(x, attributes):
    """
    Check if the given object has *all* of the given attributes

    Parameters
    ----------
    :param x: object to check
    :param attributes: a list of strings specifying the attributes to check

    Returns
    -------
    :return: True if x contains *all* of the given attributes and False otherwise.
    """
    return all([hasattr(x, a) for a in attributes])


def has_any_attributes(x, attributes):
    """
    Check if the given object has *any* of the given attributes

    Parameters
    ----------
    :param x: object to check
    :param attributes: a list of strings specifying the attributes to check

    Returns
    -------
    :return: True if x contains *any* of the given attributes and False otherwise.
    """
    return any([hasattr(x, a) for a in attributes])


def get_model(x, search=None):
    """
    Return a scikit-learn or hugging-face model

    Parameters
    ----------
    :param x: either a valid model object or a string with a model's name
    :param search: (optional) specify which Python modules to search over to find the model (default: None; directs
      the function to search over all available modules)

    Returns
    -------
    :return: an instance of the given model if found, and None otherwise
    """
    if search is None:
        search = [*sklearn_modules, *flair_embeddings, *aligners, *manipulators, *externals]

    if type(x) is str:
        for m in search:
            try:
                if type(m) is str:
                    exec(f'import {m}.{x}', globals())
                elif hasattr(m, x):
                    exec(f'{x} = m.{x}', globals())
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


# noinspection PyIncorrectDocstring
def apply_model(data, model, *args, return_model=False, search=None, **kwargs):
    """
    Apply one or more models to a dataset.  Similar to scikit-learn Pipelines:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline

    Parameters
    ----------
    :param data: any hypertools-compatible dataset
    :param model: any scikit-learn compatible model, any hugging-face model, any string (naming a scikit-learn or
      hugging-face model), or a list of models to be applied in sequence (each model fits and then transforms the output
      of the previous step in the pipeline).  For additional customization, models may be specified as dictionaries
      with the following fields:
        - 'model': one or more models
        - 'args': a list of unnamed arguments to be passed into the fit function (after the data argument)
        - 'kwargs': a list of keyword arguments, to be passed to the model's initializer function
    :param args: a list of unnamed arguments to be passed into *all* model's fit functions (appended to model['args'])
    :param return_model: if True, return the fitted model (or list of fitted models) in addition to the transformed
      dataset (default: False).
    :param search: used to narrow the scope of the search for models, which can result in faster runtimes.  This is
      passed to the get_model function.  (Default: None)
    :param mode: one of: 'fit', 'predict', 'predict_proba', 'embed', 'fit_transform', 'fit_predict', or
      'fit_predict_proba' (default: 'fit_transform').  Specifies whether to fit (only), transform/predict/embed (only),
      or fit AND transform/predict.

    Returns
    -------
    :return: either the transformed data (if return_model is False) or the transformed data and the fitted model(s) (if
      return_model is True)
    """
    mode = kwargs.pop('mode', 'fit_transform')
    custom = kwargs.pop('custom', False)

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
    elif custom and callable(model):
        transformed_data = model(data, *args, **kwargs)
        if return_model:
            return tranformed_data, {'model': model, 'args': args, 'kwargs': kwargs}
    else:
        model = dw.core.apply_defaults(get_model(model, search=search))(*args, **kwargs)
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
            return transformed_data, {'model': model, 'args': args, 'kwargs': kwargs}
        else:
            return transformed_data
