# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import os
import sklearn, flair, umap
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




