"""Gaussian-process forecaster (scikit-learn).

Fits a `GaussianProcessRegressor` against the time index (0..n-1) and
predicts `t` steps beyond it. Default kernel is
`DotProduct() + RBF(10.0) + WhiteKernel()`: the DotProduct (linear) component
lets forecasts EXTRAPOLATE trends -- with a stationary-only kernel (e.g. plain
RBF), predictions beyond the training range revert to the training mean, so a
drifting trajectory's forecast bends back toward the data cloud instead of
continuing its sweep. `normalize_y=True` by default. `kernel`/`alpha`/
`normalize_y` pass through to `GaussianProcessRegressor` -- no optional
dependency (scikit-learn is a base requirement).
"""
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct

from .common import Forecaster


def fitter(data, **kwargs):
    """Fit a `GaussianProcessRegressor` against the time index (0..n-1) for `data`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit; the target `y` is `data`'s values, regressed
        against the integer time index.
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
    """
    kernel = kwargs.get('kernel', None)
    if kernel is None:
        kernel = DotProduct() + RBF(10.0) + WhiteKernel()
    alpha = kwargs.get('alpha', 1e-10)
    normalize_y = kwargs.get('normalize_y', True)

    n = len(data)
    x = np.arange(n).reshape(-1, 1)
    y = data.to_numpy(dtype=float)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y).fit(x, y)
    return {'gp': gp, 'n': n}


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

    x_future = np.arange(n, n + n_steps).reshape(-1, 1)
    y_pred = gp.predict(x_future)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return pd.DataFrame(y_pred, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: GP conditions on the ORIGINAL fit (the already-fit
    `gp` is reused unchanged, over the time index it was actually trained
    on -- it is not re-fit against `new_data`'s values). Forecasting
    continues from where the original fit's time index left off; only the
    returned index/columns come from `new_data` (via `resolve_t`)."""
    from .common import resolve_t

    gp = fitted_params['gp']
    n = fitted_params['n']
    n_steps, future_index = resolve_t(new_data, t)
    if n_steps < 0:
        return new_data.loc[future_index]

    x_future = np.arange(n, n + n_steps).reshape(-1, 1)
    y_pred = gp.predict(x_future)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    return pd.DataFrame(y_pred, index=future_index, columns=new_data.columns)


class GaussianProcess(Forecaster):
    """Gaussian-process forecaster over the time index.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels.Kernel or None
        Covariance kernel (default: `DotProduct() + RBF(10.0) + WhiteKernel()`;
        the linear DotProduct term lets forecasts extrapolate trends rather
        than reverting to the training mean beyond the data).
    alpha : float
        Value added to the diagonal of the kernel matrix during fitting
        (default: 1e-10).
    normalize_y : bool
        Whether to normalize the target values before fitting (default: True).
    """

    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True, **kwargs):
        required = ['gp', 'n']
        super().__init__(kernel=kernel, alpha=alpha, normalize_y=normalize_y, fitter=fitter,
                          forecaster=forecaster, applier=applier, data=None,
                          required=required, **kwargs)

        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
