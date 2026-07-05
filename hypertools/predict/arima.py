"""ARIMA forecaster (statsmodels).

Fits a univariate `statsmodels.tsa.arima.model.ARIMA` model independently to
each column (ARIMA has no native multivariate support), then forecasts `t`
steps ahead per column via `.forecast(steps=t)`. Default order `(1, 1, 1)`;
`order` and any other `ARIMA` constructor kwargs pass through.

`statsmodels` ships via the optional `[predict]` extra; it is imported
lazily (inside the fitter) so `hypertools.predict` stays importable without
it, and a friendly `ImportError` is raised only when an `ARIMA` forecaster is
actually fit.

Convergence warnings (non-invertible starting MA parameters, failure to
fully converge on small/synthetic series, etc.) are common and harmless for
short forecasts; they are suppressed narrowly around the `fit()` call only
(not globally), so genuine warnings elsewhere in the process are unaffected.
"""
import warnings

import numpy as np
import pandas as pd

from .common import Forecaster


def _import_arima():
    try:
        from statsmodels.tsa.arima.model import ARIMA as SMArima
    except ImportError as e:
        raise ImportError(
            'statsmodels is required for the ARIMA forecaster; install it with '
            'pip install "hypertools[predict]"'
        ) from e
    return SMArima


def _import_convergence_warning():
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    return ConvergenceWarning


def fitter(data, **kwargs):
    sm_arima = _import_arima()
    convergence_warning = _import_convergence_warning()
    order = kwargs.get('order', (1, 1, 1))
    arima_kwargs = {k: v for k, v in kwargs.items() if k not in ('order', 'n_iter')}

    results = []
    for col in data.columns:
        x = data[col].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            warnings.simplefilter('ignore', category=convergence_warning)
            fit_result = sm_arima(x, order=order, **arima_kwargs).fit()
        results.append(fit_result)

    return {'results': results}


def forecaster(data, n_steps, future_index, **kwargs):
    results = kwargs['results']

    columns = {}
    for col, fit_result in zip(data.columns, results):
        columns[col] = np.asarray(fit_result.forecast(steps=n_steps))

    return pd.DataFrame(columns, index=future_index, columns=data.columns)


def applier(fitted_params, new_data, t):
    """`predict_new` path: apply each column's already-fit ARIMA parameters
    to the new series via `MLEResults.apply` (statsmodels' documented
    no-re-estimation path -- it reuses the fitted parameters against new
    data rather than re-optimizing them), then forecast forward."""
    from .common import resolve_t

    results = fitted_params['results']
    n_steps, future_index = resolve_t(new_data, t)
    if n_steps < 0:
        return new_data.loc[future_index]

    columns = {}
    for col, fit_result in zip(new_data.columns, results):
        new_series = new_data[col].to_numpy(dtype=float)
        applied = fit_result.apply(new_series)
        columns[col] = np.asarray(applied.forecast(steps=n_steps))

    return pd.DataFrame(columns, index=future_index, columns=new_data.columns)


class ARIMA(Forecaster):
    """Per-column ARIMA forecaster (statsmodels).

    Parameters
    ----------
    order : tuple of (p, d, q)
        ARIMA order (default: ``(1, 1, 1)``).
    **kwargs
        Passed through to ``statsmodels.tsa.arima.model.ARIMA``.
    """

    def __init__(self, order=(1, 1, 1), **kwargs):
        required = ['results']
        super().__init__(order=order, fitter=fitter, forecaster=forecaster, applier=applier,
                          data=None, required=required, **kwargs)

        self.order = order
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
