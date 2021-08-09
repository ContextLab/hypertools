# noinspection PyPackageRequirements
import datawrangler as dw
# noinspection PyPackageRequirements
import umap
import numpy as np
import pandas as pd
import os
import importlib
import sklearn
import flair
from sklearn.experimental import enable_hist_gradient_boosting, enable_iterative_imputer, enable_halving_search_cv

from .configurator import get_default_options

from ..external import ppca, brainiak

defaults = dw.core.get_default_options('config.ini')
sklearn_modules = ['calibration', 'cluster', 'compose', 'covariance', 'cross_decomposition', 'decomposition',
                   'discriminant_analysis', 'ensemble', 'experimental', 'feature_extraction',
                   'feature_extraction.image', 'feature_extraction.text', 'feature_selection',
                   'gaussian_process', 'impute', 'inspection', 'isotonic', 'kernel_approximation',
                   'kernel_ridge', 'linear_model', 'manifold', 'metrics', 'mixture', 'model_selection', 'multiclass',
                   'multioutput', 'naive_bayes', 'neighbors', 'neural_network', 'pipeline', 'preprocessing',
                   'random_projection', 'semi_supervised', 'svm', 'tree']
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
        search = [*sklearn_modules, *flair_embeddings, *externals]

    if type(x) is str:
        for m in search:
            try:
                if type(m) is str:
                    if hasattr(importlib.import_module(m), x):
                        exec(f'from {m} import {x}', globals())
                        return eval(x)
                elif hasattr(m, x):
                    return getattr(m, x)
            except ModuleNotFoundError:
                continue
        return None
    elif callable(x):
        return x
    raise ValueError(f'unknown model: {x}')


def get_sklearn_method(x, mode):
    def helper(m):
        if hasattr(x, m):
            return getattr(x, m)
        elif ('fit' in mode) and has_all_attributes(x, ['fit', m.replace('fit_', '')]):
            return [getattr(x, 'fit'), getattr(x, m.replace('fit_', ''))]
        elif ('transform' in mode) and hasattr(x, m.replace('transform', 'predict')):
            return getattr(x, m.replace('transform', 'predict'))
        elif ('transform' in mode) and hasattr(x, m.replace('transform', 'predict_proba')):
            return getattr(x, m.replace('transform', 'predict_proba'))
        else:
            raise ValueError(f'mode not supported for {x}: {m}')

    if 'transform' in mode:  # transform or fit_transform
        if hasattr(x, 'transform'):
            return helper(mode)
        elif hasattr(x, 'predict'):
            return helper(mode.replace('transform', 'predict'))
        elif hasattr(x, 'predict_proba'):
            return helper(mode.replace('transform', 'predict_proba'))
    elif 'predict_proba' in mode:  # predict_proba or fit_predict_proba
        if hasattr(x, 'predict_proba'):
            return helper(mode)
        elif hasattr(x, 'transform'):
            return helper(mode.replace('predict_proba', 'transform'))
        elif hasattr(x, 'predict'):
            return helper(mode.replace('predict_proba', 'predict'))
    elif 'predict' in mode:  # predict or fit_predict
        if hasattr(x, 'predict'):
            return helper(mode)
        elif hasattr(x, 'transform'):
            return helper(mode.replace('predict', 'transform'))
        elif hasattr(x, 'predict_proba'):
            return helper(mode.replace('transform', 'predict_proba'))

    if hasattr(x, 'fit_predict'):
        return helper('fit_predict')
    elif hasattr(x, 'fit_transform'):
        return helper('fit_transform')
    elif hasattr(x, 'fit_predict_proba'):
        return helper('fit_predict_proba')
    else:
        return helper(mode)


# noinspection PyIncorrectDocstring
def apply_model(data, model, *args, return_model=False, search=None, **kwargs):
    """
    Apply one or more models to a dataset.  Similar to scikit-learn Pipelines:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline

    Parameters
    ----------
    :param data: a pandas DataFrame, 2D numpy array, or a list of DataFrames or arrays (must have the same numbers of
      columns).  Only numerical data is supported.
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

    # noinspection PyShadowingNames
    def unpack_result(x, template, return_model):
        def safe_unstack(d, unpack):
            if unpack:
                return dw.unstack(d[0]), d[1]
            else:
                return dw.unstack(d)

        def safe_df(d, idx, unpack):
            if unpack:
                return pd.DataFrame(d[0], index=idx), d[1]
            else:
                return pd.DataFrame(d, index=idx)

        if return_model:
            data = x[0]
        else:
            data = x
        if type(template) is list:
            if type(data) is list:
                return x
            elif dw.zoo.is_multiindex_dataframe(x):
                return safe_unstack(x, return_model)
            elif dw.zoo.is_array(data):
                index = dw.stack(template).index
                return safe_unstack(safe_df(x, index, return_model), return_model)
        elif dw.zoo.is_dataframe(template):
            if dw.zoo.is_dataframe(data):
                return x
            elif dw.zoo.is_array(data):
                return safe_df(x, template.index, return_model)
        else:
            return x

    mode = kwargs.pop('mode', 'fit_transform')
    custom = kwargs.pop('custom', False)

    if type(data) is list:
        stacked_data = dw.stack(data)
    elif dw.zoo.is_dataframe(data):
        stacked_data = data
    elif dw.zoo.is_array(data):
        stacked_data = pd.DataFrame(data)
    else:
        raise ValueError(f'unsupported datatype: {type(data)}')

    if type(model) is list:
        fitted_models = []
        for m in model:
            stacked_data, next_fitted = apply_model(stacked_data, m, return_model=True, **kwargs)
            fitted_models.append(next_fitted)
        if return_model:
            return unpack_result(stacked_data, data, return_model), fitted_models
        else:
            return unpack_result(stacked_data, data, return_model)

    elif type(model) is dict:
        assert all([k in model.keys() for k in ['model', 'args', 'kwargs']]), \
            ValueError('model must have keys "model", "args", and "kwargs"')

        return unpack_result(apply_model(stacked_data, model['model'], return_model=return_model, mode=mode,
                                         custom=custom, *[*model['args'], *args], **dw.core.update_dict(model['kwargs'],
                                                                                                        kwargs)),
                             data, return_model)
    elif custom and callable(model):
        transformed_data = model(stacked_data, *args, **kwargs)
        if return_model:
            return unpack_result(tranformed_data, data, return_model), {'model': model, 'args': args, 'kwargs': kwargs}
        else:
            return unpack_result(transformed_data, data, return_model)
    else:
        model = dw.core.apply_defaults(get_model(model, search=search), get_default_options())(*args, **kwargs)
        if dw.zoo.text.is_hugging_face_model(model):
            return unpack_result(dw.zoo.text.apply_text_model(model, stacked_data, *args, mode=mode,
                                                              return_model=return_model, **kwargs), data, return_model)
        f = get_sklearn_method(model, mode)
        if type(f) is list:
            assert len(f) == 2, ValueError(f'bad mode: {mode}')
            f[0](stacked_data)
            transformed_data = f[1](stacked_data)
        else:
            transformed_data = f(stacked_data)

        if return_model:
            return unpack_result(transformed_data, data, False), {'model': model, 'args': args, 'kwargs': kwargs}
        else:
            return unpack_result(transformed_data, data, False)
