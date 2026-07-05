"""Kalman-filter forecaster (pykalman).

Fits a linear-Gaussian state-space model via EM (`n_iter` iterations, default
5), then extends the fitted filter forward with `filter_update` (no new
observations) to produce a `t`-step-ahead forecast. NaNs in the input are
tolerated via `np.ma.masked_invalid`, which pykalman treats as missing
observations during both EM and filtering.

`pykalman` ships via the optional `[predict]` extra; it is imported lazily
(inside the fitter) so `hypertools.predict` stays importable without it, and
a friendly `ImportError` is raised only when a `Kalman` forecaster is
actually fit.
"""
import numpy as np
import pandas as pd

from .common import Forecaster


def _import_kalman_filter():
    try:
        from pykalman import KalmanFilter
    except ImportError as e:
        raise ImportError(
            'pykalman is required for the Kalman forecaster; install it with '
            'pip install "hypertools[predict]"'
        ) from e
    return KalmanFilter


def fitter(data, **kwargs):
    kalman_filter_cls = _import_kalman_filter()
    n_iter = kwargs.get('n_iter', 5)

    x = np.ma.masked_invalid(data.to_numpy(dtype=float))
    n, d = x.shape

    kf = kalman_filter_cls(n_dim_obs=d, n_dim_state=d).em(x, n_iter=n_iter)
    means, covs = kf.filter(x)

    return {'kf': kf, 'mean': means[-1], 'cov': covs[-1]}


def forecaster(data, n_steps, future_index, **kwargs):
    kf = kwargs['kf']
    mean, cov = kwargs['mean'], kwargs['cov']

    rows = []
    for _ in range(n_steps):
        mean, cov = kf.filter_update(mean, cov)
        rows.append(np.asarray(mean))

    return pd.DataFrame(rows, index=future_index, columns=data.columns)


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
        super().__init__(n_iter=n_iter, fitter=fitter, forecaster=forecaster, data=None,
                          required=required, **kwargs)

        self.n_iter = n_iter
        self.fitter = fitter
        self.forecaster = forecaster
        self.data = None
        self.required = required
