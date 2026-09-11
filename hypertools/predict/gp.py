"""Gaussian-process forecaster (scikit-learn).

Fits a `GaussianProcessRegressor` against actual observation times and
predicts `t` steps beyond it. Default kernel is
`DotProduct() + RBF(10.0) + WhiteKernel()`: the DotProduct (linear) component
lets forecasts EXTRAPOLATE trends -- with a stationary-only kernel (e.g. plain
RBF), predictions beyond the training range revert to the training mean, so a
drifting trajectory's forecast bends back toward the data cloud instead of
continuing its sweep. `normalize_y=True` by default. `kernel`/`alpha`/
`normalize_y` pass through to `GaussianProcessRegressor` -- no optional
dependency (scikit-learn is a base requirement).

NaN policy: GaussianProcess cannot fit data containing NaN; a clear
`ValueError` is raised (fill missing values first, e.g. with `hyp.impute`,
or use the NaN-tolerant `Kalman`/`ARIMA` forecasters).
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
from .time import (TIME_STEP_ATTR, is_time_index, resolve_step,
                   time_coordinates)

from .common import Forecaster


def _check_nan(y, context):
    # sklearn's raw "Input y contains NaN." names neither the forecaster nor
    # the fix (QC 2026-07 red-team F16-predict-014).
    if np.isnan(y).any():
        raise ValueError(
            f'GaussianProcess cannot {context} data containing NaN '
            f'({int(np.isnan(y).sum())} missing value(s) found). Fill missing '
            'values first (e.g. hyp.impute(data)), or use a NaN-tolerant '
            "forecaster (model='Kalman' or model='ARIMA').")


def fitter(data, **kwargs):
    """Fit a `GaussianProcessRegressor` against observation times for `data`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit; the target `y` is `data`'s values, regressed
        against elapsed times in units of the fitted step. Categorical or
        repeated numeric row IDs use observation positions (0..n-1).
    **kwargs
        `kernel` : sklearn kernel or None, covariance kernel (default:
        `DotProduct() + RBF(10.0) + WhiteKernel()`). `alpha` : float,
        diagonal regularization (default: 1e-10). `normalize_y` : bool,
        whether to normalize targets before fitting (default: True).

    Returns
    -------
    dict
        `{'gp': <fitted GaussianProcessRegressor>, 'n': <number of
        fit-time observations>}`.

    Raises
    ------
    ValueError
        If `data` contains NaN (GaussianProcess is not NaN-tolerant).
    """
    kernel = kwargs.get('kernel', None)
    if kernel is None:
        kernel = DotProduct() + RBF(10.0) + WhiteKernel()
    alpha = kwargs.get('alpha', 1e-10)
    normalize_y = kwargs.get('normalize_y', True)

    n = len(data)
    step = data.attrs.get(TIME_STEP_ATTR)
    if step is None:
        step = resolve_step(data.index)
    origin = data.index[0]
    timed = is_time_index(data.index)
    x = (time_coordinates(data.index, origin, step) if timed
         else np.arange(n)).reshape(-1, 1)
    y = data.to_numpy(dtype=float)
    _check_nan(y, 'fit')

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y).fit(x, y)
    return {'gp': gp, 'n': n, 'time_origin': origin, 'time_step': step,
            'timed': timed}


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead by extending the fitted GP's time index.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `gp`, `n` : the fitted `GaussianProcessRegressor` and fit-time
        observation count from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted values, indexed by `future_index`, columns matching `data`.
    """
    gp = kwargs['gp']
    n = kwargs['n']

    x_future = (time_coordinates(future_index, kwargs['time_origin'],
                                 kwargs['time_step']) if kwargs.get('timed')
                else np.arange(n, n + n_steps)).reshape(-1, 1)
    y_pred = gp.predict(x_future)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return pd.DataFrame(y_pred, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: condition the LEARNED kernel on the NEW series.

    The reusable learned parameters of a GP are its optimized kernel
    hyperparameters (`gp.kernel_`). A fresh `GaussianProcessRegressor`
    with that fixed kernel (``optimizer=None`` -- the hyperparameters are
    NOT re-optimized) is conditioned on `new_data`'s values over its own
    time index, then extended forward, so the forecast actually continues
    the new series. (QC 2026-07 red-team F16-predict-007: reuse previously
    ignored the new data's values entirely and replayed the original fit's
    forecast for ANY new dataset.)"""
    from .common import resolve_t

    gp = fitted_params['gp']
    n_steps, future_index = resolve_t(new_data, t)
    if n_steps <= 0:
        return new_data.loc[future_index]

    n_new = len(new_data)
    # `predict_new` resolved the step for THIS index (the learned interval
    # when it is in the new index's units, else the new data's own); a
    # fitted row count cannot measure a dated index, nor a duration an array.
    step = fitted_params.get('time_step', resolve_step(new_data.index))
    timed = is_time_index(new_data.index)
    origin = new_data.index[0]
    x_new = (time_coordinates(new_data.index, origin, step) if timed
             else np.arange(n_new)).reshape(-1, 1)
    y_new = new_data.to_numpy(dtype=float)
    _check_nan(y_new, 'be conditioned on')

    conditioned = GaussianProcessRegressor(
        kernel=gp.kernel_, alpha=gp.alpha, normalize_y=gp.normalize_y,
        optimizer=None).fit(x_new, y_new)

    x_future = (time_coordinates(future_index, origin, step) if timed
                else np.arange(n_new, n_new + n_steps)).reshape(-1, 1)
    y_pred = conditioned.predict(x_future)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return pd.DataFrame(y_pred, index=future_index, columns=new_data.columns)


class GaussianProcess(Forecaster):
    """Gaussian-process forecaster over the time index.

    Parameters
    ----------
    step : number, duration string, Timedelta, calendar offset, or None
        One future step. None infers it per dataset: a datetime index's
        calendar frequency when it has one (business days, month starts, a
        ``PeriodIndex``'s periods...), else the median positive gap between
        sorted observation times. Numerical indexes use their own units;
        datetime/duration indexes need a duration such as '1h' (datetime
        indexes also take a calendar frequency such as 'B' or 'MS').
        See `hypertools.predict` for the interpolation and reuse policies.
    kernel : sklearn.gaussian_process.kernels.Kernel or None
        Covariance kernel (default: `DotProduct() + RBF(10.0) + WhiteKernel()`;
        the linear DotProduct term lets forecasts extrapolate trends rather
        than reverting to the training mean beyond the data).
    alpha : float
        Value added to the diagonal of the kernel matrix during fitting
        (default: 1e-10).
    normalize_y : bool
        Whether to normalize the target values before fitting (default: True).

    Notes
    -----
    Reuse (``return_model=True`` round-trip): the learned kernel
    hyperparameters are conditioned on the NEW series without
    re-optimization (see `applier`). Unknown keyword arguments raise
    `TypeError` (they were previously swallowed silently -- QC 2026-07
    red-team F16-predict-009).
    """

    # The forecaster evaluates arbitrary future coordinates directly.
    _uses_observation_times = True

    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True, step=None):
        required = ['gp', 'n']
        super().__init__(step=step, kernel=kernel, alpha=alpha, normalize_y=normalize_y, fitter=fitter,
                          forecaster=forecaster, applier=applier, data=None,
                          required=required)

        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
