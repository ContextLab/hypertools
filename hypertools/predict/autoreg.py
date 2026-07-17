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
        # real raise (not `assert ..., ValueError(...)`, which raises
        # AssertionError and is stripped under `python -O`) -- QC 2026-07.
        if model not in MODELS:
            raise ValueError(
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
    """Fit a lagged-feature regressor for the `AutoRegressor` forecaster.

    Builds a lagged design matrix (each row: the flattened previous
    `lags` observations; target: the next observation) and fits `model`
    on it, falling back to `MultiOutputRegressor` if `model` does not
    natively support a multivariate target.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on.
    **kwargs
        `model` : str, class, or instance, the regressor to use
        (default: `'Ridge'`). `lags` : int, number of trailing
        observations used as predictors (default: 10). `model_kwargs` :
        dict, passed to `model` when it is a string or class.

    Returns
    -------
    dict
        `{'estimator': <fitted regressor>, 'lags': lags, 'history':
        <last `lags` rows of data>, 'n_features': <number of columns>}`.

    Raises
    ------
    ValueError
        If `data` has `lags` or fewer observations, or contains NaN
        (AutoRegressor is not NaN-tolerant).
    """
    model = kwargs.get('model', 'Ridge')
    lags = kwargs.get('lags', 10)
    model_kwargs = kwargs.get('model_kwargs', {}) or {}

    x = data.to_numpy(dtype=float)
    n, d = x.shape
    if not n > lags:
        raise ValueError(
            f'AutoRegressor needs more than lags={lags} observations to fit; '
            f'got {n}. Pass more data or a smaller lags= value.')
    # sklearn's raw "Input y contains NaN." names neither the forecaster nor
    # the fix (QC 2026-07 red-team F16-predict-014).
    if np.isnan(x).any():
        raise ValueError(
            f'AutoRegressor cannot fit data containing NaN '
            f'({int(np.isnan(x).sum())} missing value(s) found). Fill missing '
            'values first (e.g. hyp.impute(data)), or use a NaN-tolerant '
            "forecaster (model='Kalman' or model='ARIMA').")

    x_mat, y_mat = _lagged_matrix(x, lags)
    estimator = _resolve_estimator(model, model_kwargs)
    fitted = _fit_with_multioutput_fallback(estimator, x_mat, y_mat)

    return {'estimator': fitted, 'lags': lags, 'history': x[-lags:].copy(), 'n_features': d}


def forecaster(data, n_steps, future_index, **kwargs):
    """Recursively forecast `n_steps` ahead using the fitted lagged-feature regressor.

    Predicts one step from the trailing `lags`-observation window,
    appends the prediction to that window (dropping the oldest entry),
    and repeats `n_steps` times.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `estimator`, `n_features`, `history` : fitted state from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted values, indexed by `future_index`, columns matching `data`.
    """
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
    if n_steps <= 0:
        return new_data.loc[future_index]

    x_new = new_data.to_numpy(dtype=float)
    if not len(x_new) >= lags:
        raise ValueError(
            f'predict_new needs at least lags={lags} observations of new '
            f'data; got {len(x_new)}')
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
        Must be a positive integer: 0, negative, boolean, or non-integer
        values (e.g. ``lags=2.5``) raise a `ValueError` at construction
        (fitting also requires strictly more than `lags` observations).
    model_kwargs : dict or None
        Parameters passed to `model` when it is a string or class
        (default: None). Direct keyword arguments (e.g.
        ``AutoRegressor(model='SVR', C=2.0)``) are equivalent and are
        merged with `model_kwargs` (direct kwargs win on conflicts).
        (QC 2026-07 red-team F16-predict-006: an explicit
        ``model_kwargs={...}`` -- the form this docstring documents -- used
        to be double-nested into the estimator's constructor and crash
        with ``TypeError: ... unexpected keyword argument 'model_kwargs'``.)
    **kwargs
        Additional parameters for `model` (see `model_kwargs`).

    Notes
    -----
    The inner estimator-choosing parameter is named ``model``, which
    collides with `hypertools.predict.predict`'s own ``model=`` argument;
    to choose the inner regressor through ``hyp.predict``, use the dict
    spec form, e.g. ``model={'model': 'AutoRegressor', 'kwargs':
    {'model': 'Ridge'}}``.
    """

    def __init__(self, model='Ridge', lags=10, model_kwargs=None, **kwargs):
        # validate lags up front (2026-07 release audit, final wave item 12):
        # lags=0 leaked sklearn's "Found array with 0 feature(s)", negative
        # values built nonsense designs, and a float crashed with a raw
        # "'float' object cannot be interpreted as an integer".
        if (isinstance(lags, bool) or not isinstance(lags, (int, np.integer))
                or lags < 1):
            raise ValueError(
                f'lags must be a positive integer (the number of trailing '
                f'observations used as predictors); got {lags!r}. Pass e.g. '
                'lags=10.')
        lags = int(lags)
        # merge direct kwargs into model_kwargs (direct kwargs win); the
        # original object is kept when there is nothing to merge so sklearn's
        # clone() identity check still passes.
        if kwargs:
            model_kwargs = {**(model_kwargs or {}), **kwargs}
        required = ['estimator', 'lags', 'history', 'n_features']
        super().__init__(model=model, lags=lags, model_kwargs=model_kwargs,
                          fitter=fitter, forecaster=forecaster, applier=applier,
                          data=None, required=required)

        self.model = model
        self.lags = lags
        self.model_kwargs = model_kwargs
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
