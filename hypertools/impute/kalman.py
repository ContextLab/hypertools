"""Kalman-smoothing imputer (pykalman).

Fits a linear-Gaussian state-space model via EM
(`KalmanFilter(...).em(np.ma.masked_invalid(X), n_iter=5)`), then `.smooth`s
the entire masked series and splices the smoothed values back in ONLY where
the input was missing -- every non-missing entry passes through unchanged.

Unlike `hypertools.impute.ppca.PPCA`, which cannot reconstruct rows with NO
observed features at all, the Kalman smoother's state propagates across
time, so it CAN fill rows where every feature is missing, as long as
neighboring rows have some observations -- this closes the GH #169 gap.

`pykalman` ships via the optional `[predict]` extra (shared with
`hypertools.predict.kalman`); imported lazily so `hypertools.impute` stays
importable without it, and a friendly `ImportError` is raised only when a
`Kalman` imputer is actually fit.
"""
import numpy as np
import pandas as pd

from .common import Imputer


def _import_kalman_filter():
    try:
        from pykalman import KalmanFilter
    except ImportError as e:
        raise ImportError(
            'pykalman is required for the Kalman imputer; install it with '
            'pip install "hypertools[predict]"'
        ) from e
    return KalmanFilter


def fitter(data, **kwargs):
    kalman_filter_cls = _import_kalman_filter()
    n_iter = kwargs.get('n_iter', 5)

    x = np.ma.masked_invalid(data.to_numpy(dtype=float))
    n, d = x.shape

    kf = kalman_filter_cls(n_dim_obs=d, n_dim_state=d).em(x, n_iter=n_iter)
    return {'kf': kf}


def transformer(data, **kwargs):
    kf = kwargs['kf']
    x = data.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(x)

    smoothed, _ = kf.smooth(masked)

    mask = np.isnan(x)
    out = x.copy()
    out[mask] = smoothed[mask]
    return pd.DataFrame(out, index=data.index, columns=data.columns)


class Kalman(Imputer):
    """Kalman-smoothing imputer: EM-fit a linear-Gaussian state-space model,
    then replace ONLY the missing entries with the smoothed estimates.

    Unlike PPCA, this fills rows where EVERY feature is missing (the
    smoother's state propagates across time from neighboring observed
    rows) -- the GH #169 gap.

    Parameters
    ----------
    n_iter : int
        Number of EM iterations (default: 5).
    """

    def __init__(self, n_iter=5, **kwargs):
        required = ['kf']
        super().__init__(n_iter=n_iter, fitter=fitter, transformer=transformer, data=None,
                          required=required, **kwargs)
        self.n_iter = n_iter
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
