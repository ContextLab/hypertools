"""Kalman-filter forecaster (pykalman).

Fits a linear-Gaussian state-space model via EM (`n_iter` iterations, default
5), then extends the fitted filter forward with `filter_update` (no new
observations) to produce a `t`-step-ahead forecast. NaNs in the input are
tolerated via `np.ma.masked_invalid`, which pykalman treats as missing
observations during both EM and filtering.

`pykalman` is a core hypertools dependency (pure-python, lightweight), so the
`Kalman` forecaster works out of the box. It is still imported lazily (inside
the fitter) so `hypertools.predict` stays importable even in an environment
where the core deps were stripped, raising a friendly `ImportError` only then.
"""
import numpy as np
import pandas as pd

from .common import Forecaster


def _import_kalman_filter():
    try:
        from pykalman import KalmanFilter
    except ImportError as e:
        raise ImportError(
            'pykalman is required for the Kalman forecaster. It is normally a '
            'core hypertools dependency; reinstall hypertools, or install it '
            'directly with `pip install pykalman`.'
        ) from e
    return KalmanFilter


def fitter(data, **kwargs):
    """EM-fit a linear-Gaussian Kalman filter on `data` (missing entries masked).

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit on; NaN entries are masked (treated as missing) via
        `numpy.ma.masked_invalid`.
    **kwargs
        `n_iter` : int, number of EM iterations (default: 5).

    Returns
    -------
    dict
        `{'kf': <fitted pykalman.KalmanFilter>, 'mean': <final filtered
        state mean>, 'cov': <final filtered state covariance>}`.
    """
    kalman_filter_cls = _import_kalman_filter()
    n_iter = kwargs.get('n_iter', 5)

    x = np.ma.masked_invalid(data.to_numpy(dtype=float))
    n, d = x.shape

    kf = kalman_filter_cls(n_dim_obs=d, n_dim_state=d).em(x, n_iter=n_iter)
    means, covs = kf.filter(x)

    return {'kf': kf, 'mean': means[-1], 'cov': covs[-1]}


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead by iterating `filter_update` with no new observations.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `kf`, `mean`, `cov` : the fitted filter and final filtered state
        from `fitter`.

    Returns
    -------
    pandas.DataFrame
        Forecasted state means, indexed by `future_index`, columns
        matching `data`.
    """
    kf = kwargs['kf']
    mean, cov = kwargs['mean'], kwargs['cov']

    rows = []
    for _ in range(n_steps):
        mean, cov = kf.filter_update(mean, cov)
        rows.append(np.asarray(mean))

    return pd.DataFrame(rows, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: filter the NEW series with the LEARNED
    transition/observation matrices (no EM -- `kf` is reused unchanged),
    then iterate `filter_update` forward to forecast beyond it."""
    from .common import resolve_t

    kf = fitted_params['kf']
    n_steps, future_index = resolve_t(new_data, t)
    if n_steps < 0:
        return new_data.loc[future_index]

    x = np.ma.masked_invalid(new_data.to_numpy(dtype=float))
    means, covs = kf.filter(x)
    mean, cov = means[-1], covs[-1]

    rows = []
    for _ in range(n_steps):
        mean, cov = kf.filter_update(mean, cov)
        rows.append(np.asarray(mean))

    return pd.DataFrame(rows, index=future_index, columns=new_data.columns)


class Kalman(Forecaster):
    """Kalman-filter forecaster: EM-fit a linear-Gaussian state-space model,
    then iterate `filter_update` (no observations) to forecast forward.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations used to fit the transition/observation
        matrices and their noise covariances (default: 5).
    """

    def __init__(self, n_iter=5, **kwargs):
        required = ['kf', 'mean', 'cov']
        super().__init__(n_iter=n_iter, fitter=fitter, forecaster=forecaster, applier=applier,
                          data=None, required=required, **kwargs)

        self.n_iter = n_iter
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
