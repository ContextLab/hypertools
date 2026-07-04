#!/usr/bin/env python
"""Unified stack/unstack model application -- the hypertools 2.0 core.

Every model-applying operation in hypertools follows the same recipe:

    format -> STACK the list of datasets into one array -> apply the model
    ONCE -> UNSTACK the result back into the input's list structure

Fitting a single model to the stacked data is what makes results comparable
across datasets (separate per-dataset fits would produce incomparable
embeddings/labels). `apply_model` exposes that recipe directly, supporting
model specification as a registry name, a dict with params, a scikit-learn
style instance, or a list of any of these (applied as a pipeline).

The registry is an explicit whitelist -- the earlier refactor resolved model
strings with eval() over sklearn's namespace, which was fragile and
security-sensitive; named models here are imported explicitly.
"""

import numpy as np


def _build_registry():
    from ..reduce.reduce import models as reduce_models
    from ..cluster.cluster import models as cluster_models, mixture_models
    registry = {}
    registry.update(reduce_models)
    registry.update(cluster_models)
    registry.update(mixture_models)
    # UMAP resolves lazily (importing umap triggers slow numba JIT)
    registry['UMAP'] = None
    return registry


def supported_models():
    """Names accepted as string model specifications."""
    return sorted(_build_registry().keys())


def apply_model(data, model, mode='auto', return_model=False,
                format_data=True, stack=True, ndims=None):
    """Apply a model (or pipeline of models) to one or more datasets.

    Parameters
    ----------
    data : numpy array, pandas DataFrame, or list of arrays/DataFrames
        The dataset(s) to transform.

    model : str, dict, sklearn-style instance, or list
        - str: a registered model name (see `supported_models()`),
          e.g. 'PCA', 'KMeans', 'GaussianMixture'
        - dict: {'model': <str or instance>, 'params': {...}}
        - instance: any object exposing fit/transform/fit_transform/
          fit_predict/predict_proba (scikit-learn convention)
        - list: a pipeline; each element is applied in sequence, with each
          stage's output feeding the next stage's input

    mode : str
        How to apply the model: 'fit_transform', 'fit_predict',
        'predict_proba', or 'auto' (default), which prefers fit_transform,
        then predict_proba (fitting first), then fit_predict.

    return_model : bool
        If True, also return the fitted model (or list of fitted models for
        a pipeline) so it can be reused on held-out data (default: False).

    format_data : bool
        Whether to run hypertools' format_data on the input (default: True).

    stack : bool
        If True (default), datasets are vertically stacked and the model is
        fit ONCE across all of them, then results are split back to match
        the input structure. If False, a separate model is fit per dataset.

    ndims : int or None
        Convenience: sets n_components on models that accept it.

    Returns
    -------
    result (and fitted model(s) if return_model=True). Lists in, lists out:
    a single input dataset returns a single result.
    """
    single_input = not isinstance(data, list)
    if format_data:
        # lazy: importing tools.format_data at module load time would create
        # a circular import (tools -> ..._shared.exceptions -> core.exceptions
        # -> core (package init) -> core.model -> tools), since core.model is
        # now exported from core/__init__.py.
        from ..tools.format_data import format_data as formatter
        data = formatter(data, ppca=True)
    elif single_input:
        data = [data]

    # pipeline: thread the data through each stage
    if isinstance(model, list):
        fitted = []
        for stage in model:
            data, stage_model = apply_model(
                data, stage, mode=mode, return_model=True,
                format_data=False, stack=stack, ndims=ndims)
            if not isinstance(data, list):
                data = [data]
            fitted.append(stage_model)
        result = data if not single_input else data[0]
        return (result, fitted) if return_model else result

    model_instance = _resolve_model(model, ndims)

    lengths = [np.asarray(d).shape[0] for d in data]
    arrays = [np.asarray(d, dtype=np.float64) for d in data]

    if stack:
        stacked = np.vstack(arrays)
        applied, fitted = _apply_single(model_instance, stacked, mode)
        pieces = np.vsplit(np.asarray(applied), np.cumsum(lengths)[:-1]) \
            if np.asarray(applied).ndim > 1 else \
            np.split(np.asarray(applied), np.cumsum(lengths)[:-1])
        result = [np.asarray(p) for p in pieces]
    else:
        result, fitted = [], []
        for arr in arrays:
            applied, fm = _apply_single(_clone(model_instance), arr, mode)
            result.append(np.asarray(applied))
            fitted.append(fm)

    if single_input:
        result = result[0]
    return (result, fitted) if return_model else result


def _resolve_model(model, ndims):
    """Turn a model specification into a ready-to-fit instance."""
    params = {}
    if isinstance(model, dict):
        # dev-2.0 form: {'model', 'params'}; fork form: {'model', 'args', 'kwargs'}
        params = dict(model.get('params', model.get('kwargs', {})))
        model = model['model']

    if isinstance(model, str):
        registry = _build_registry()
        if model not in registry:
            raise ValueError(
                f'unknown model {model!r}; supported names: '
                f'{", ".join(supported_models())} (or pass a scikit-learn '
                f'style instance directly)')
        if model == 'UMAP':
            from umap import UMAP as model_cls
        else:
            model_cls = registry[model]
        if ndims is not None:
            params.setdefault('n_components', ndims)
        return model_cls(**params)

    # instance: duck-type on the sklearn convention
    if not any(hasattr(model, m) for m in
               ('fit_transform', 'fit_predict', 'transform', 'predict')):
        raise ValueError(
            'model instances must follow the scikit-learn convention '
            '(fit/transform/fit_transform/fit_predict/predict_proba)')
    if params:
        model.set_params(**params)
    if ndims is not None and hasattr(model, 'n_components'):
        model.set_params(n_components=ndims)
    return model


def _apply_single(model, stacked, mode):
    """Fit and apply one model to one (stacked) array."""
    if mode == 'auto':
        if hasattr(model, 'predict_proba'):
            mode = 'predict_proba'
        elif hasattr(model, 'fit_transform') or hasattr(model, 'transform'):
            mode = 'fit_transform'
        else:
            mode = 'fit_predict'

    if mode == 'fit_transform':
        if hasattr(model, 'fit_transform'):
            return model.fit_transform(stacked), model
        model.fit(stacked)
        return model.transform(stacked), model
    if mode == 'predict_proba':
        model.fit(stacked)
        return model.predict_proba(stacked), model
    if mode == 'fit_predict':
        if hasattr(model, 'fit_predict'):
            return model.fit_predict(stacked), model
        model.fit(stacked)
        return model.predict(stacked), model
    raise ValueError(f'unknown mode {mode!r}; use fit_transform, '
                     f'fit_predict, predict_proba, or auto')


def _clone(model):
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        return model
