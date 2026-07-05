"""Gaussian-process forecaster (scikit-learn).

Fits a `GaussianProcessRegressor` against the time index (0..n-1) and
predicts `t` steps beyond it. Default kernel is `RBF(10.0) + WhiteKernel()`;
`normalize_y=True` by default. `kernel`/`alpha`/`normalize_y` pass through to
`GaussianProcessRegressor` -- no optional dependency (scikit-learn is a base
requirement).
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .common import Forecaster


def fitter(data, **kwargs):
    kernel = kwargs.get('kernel', None)
    if kernel is None:
        kernel = RBF(10.0) + WhiteKernel()
    alpha = kwargs.get('alpha', 1e-10)
    normalize_y = kwargs.get('normalize_y', True)

    n = len(data)
    x = np.arange(n).reshape(-1, 1)
    y = data.to_numpy(dtype=float)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y).fit(x, y)
    return {'gp': gp, 'n': n}


def forecaster(data, n_steps, future_index, **kwargs):
    gp = kwargs['gp']
    n = kwargs['n']

    x_future = np.arange(n, n + n_steps).reshape(-1, 1)
    y_pred = gp.predict(x_future)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return pd.DataFrame(y_pred, index=future_index, columns=data.columns)


class GaussianProcess(Forecaster):
    """Gaussian-process forecaster over the time index.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel or None
        Covariance kernel (default: `RBF(10.0) + WhiteKernel()`).
    alpha : float
        Value added to the diagonal of the kernel matrix during fitting
        (default: 1e-10).
    normalize_y : bool
        Whether to normalize the target values before fitting (default: True).
    """

    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True, **kwargs):
        required = ['gp', 'n']
        super().__init__(kernel=kernel, alpha=alpha, normalize_y=normalize_y, fitter=fitter,
                          forecaster=forecaster, data=None, required=required, **kwargs)

        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.fitter = fitter
        self.forecaster = forecaster
        self.data = None
        self.required = required
