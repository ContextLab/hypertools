"""Autoregressive forecaster: any scikit-learn regressor over lagged features.

Builds a lagged design matrix (each row is the flattened previous `lags`
observations; each target is the next observation) and fits `model` on it.
Multi-step forecasts are produced recursively: predict one step, append it
to the trailing window, predict the next step, and so on. `model` may be a
registry string (Ridge, Lasso, LinearRegression, RandomForestRegressor,
GradientBoostingRegressor, SVR, KNeighborsRegressor), a scikit-learn
regressor class, or an already-constructed instance. Multivariate targets
use the regressor's native multi-output support when available, falling
back to `sklearn.multioutput.MultiOutputRegressor` otherwise.
"""
import inspect

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

from .common import Forecaster

# registry of common sklearn regressors, resolved by name (mirrors the
# registry style of hypertools/cluster/cluster.py)
MODELS = {
    'Ridge': Ridge,
    'Lasso': Lasso,
    'LinearRegression': LinearRegression,
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'SVR': SVR,
    'KNeighborsRegressor': KNeighborsRegressor,
}


def _resolve_estimator(model, model_kwargs):
    if isinstance(model, str):
        assert model in MODELS, ValueError(
            f'unknown model: {model!r}; supported string names are {list(MODELS)}')
        return MODELS[model](**model_kwargs)
    if inspect.isclass(model):
        return model(**model_kwargs)
    return model  # already a (possibly unfitted) estimator instance


def _lagged_matrix(x, lags):
    n = x.shape[0]
    rows = [x[i - lags:i].reshape(-1) for i in range(lags, n)]
    targets = [x[i] for i in range(lags, n)]
    return np.asarray(rows), np.asarray(targets)


def _fit_with_multioutput_fallback(estimator, x, y):
    """Fit `estimator` on (x, y); if `y` is multivariate and the estimator
    does not natively support it, fall back to `MultiOutputRegressor`."""
    if y.shape[1] == 1:
        estimator.fit(x, y.ravel())
        return estimator

    try:
        estimator.fit(x, y)
        return estimator
    except (ValueError, TypeError):
        wrapped = MultiOutputRegressor(clone(estimator))
        wrapped.fit(x, y)
        return wrapped


def fitter(data, **kwargs):
    model = kwargs.get('model', 'Ridge')
    lags = kwargs.get('lags', 10)
    model_kwargs = kwargs.get('model_kwargs', {}) or {}

    x = data.to_numpy(dtype=float)
    n, d = x.shape
    assert n > lags, ValueError(
        f'AutoRegressor needs more than lags={lags} observations to fit; got {n}')

    x_mat, y_mat = _lagged_matrix(x, lags)
    estimator = _resolve_estimator(model, model_kwargs)
    fitted = _fit_with_multioutput_fallback(estimator, x_mat, y_mat)

    return {'estimator': fitted, 'lags': lags, 'history': x[-lags:].copy(), 'n_features': d}


def forecaster(data, n_steps, future_index, **kwargs):
    estimator = kwargs['estimator']
    d = kwargs['n_features']
    history = kwargs['history'].copy()

    rows = []
    for _ in range(n_steps):
        pred = np.asarray(estimator.predict(history.reshape(1, -1))).reshape(-1)[:d]
        rows.append(pred)
        history = np.vstack([history[1:], pred])

    return pd.DataFrame(rows, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: reuse the already-fit `estimator` unchanged (no
    refitting); the recursion is reseeded from the trailing `lags`
    observations of `new_data` instead of the original fit's history."""
    from .common import resolve_t

    estimator = fitted_params['estimator']
    lags = fitted_params['lags']
    d = fitted_params['n_features']

    n_steps, future_index = resolve_t(new_data, t)
    if n_steps < 0:
        return new_data.loc[future_index]

    x_new = new_data.to_numpy(dtype=float)
    assert len(x_new) >= lags, ValueError(
        f'predict_new needs at least lags={lags} observations of new data; got {len(x_new)}')
    history = x_new[-lags:].copy()

    rows = []
    for _ in range(n_steps):
        pred = np.asarray(estimator.predict(history.reshape(1, -1))).reshape(-1)[:d]
        rows.append(pred)
        history = np.vstack([history[1:], pred])

    return pd.DataFrame(rows, index=future_index, columns=new_data.columns)


class AutoRegressor(Forecaster):
    """Recursive multi-step forecaster: any scikit-learn regressor over
    lagged features.

    Parameters
    ----------
    model : str, class, or instance
        Regressor to use. String names are resolved from a small registry
        (Ridge, Lasso, LinearRegression, RandomForestRegressor,
        GradientBoostingRegressor, SVR, KNeighborsRegressor); a scikit-learn
        regressor class or an already-constructed instance also works.
    lags : int
        Number of trailing observations used as predictors (default: 10).
    **model_kwargs
        Passed through to `model` when it is a string or class.
    """

    def __init__(self, model='Ridge', lags=10, **model_kwargs):
        required = ['estimator', 'lags', 'history', 'n_features']
        super().__init__(model=model, lags=lags, model_kwargs=model_kwargs, fitter=fitter,
                          forecaster=forecaster, applier=applier, data=None, required=required)

        self.model = model
        self.lags = lags
        self.model_kwargs = model_kwargs
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
