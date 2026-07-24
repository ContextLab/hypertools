"""ARIMA forecaster (statsmodels).

Fits a univariate `statsmodels.tsa.arima.model.ARIMA` model independently to
each column (ARIMA has no native multivariate support), then forecasts `t`
steps ahead per column via `.forecast(steps=t)`. Default order `(1, 1, 1)`;
`order` and any other `ARIMA` constructor kwargs pass through.

IMPORTANT -- the default order only suits drift/random-walk-like signals:
with d=1 differencing and no trend term, an ARIMA(1, 1, 1) forecast damps
toward a constant within a few steps, so it cannot continue oscillatory
(seasonal) signals or extrapolate a linear trend (QC 2026-07 red-team
F16-predict-005: on a strong noisy sine the default's 30-step forecast
anti-correlated with the held-out truth, while ``order=(4, 0, 0)`` tracked
it at r~0.93). For oscillatory or strongly-trending data, pass a suitable
``order=`` (and/or ``trend=``), or use ``model='AutoRegressor'``,
``'GaussianProcess'``, or ``'Kalman'``, which handle those signals with
their defaults.

`statsmodels` is a core hypertools dependency, so the `ARIMA` forecaster works
out of the box. It is still imported lazily (inside the fitter) so
`hypertools.predict` stays importable even where the core deps were stripped,
raising a friendly `ImportError` only then.

Convergence warnings (non-invertible starting MA parameters, failure to
fully converge on small/synthetic series, etc.) are common and harmless for
short forecasts; they are suppressed narrowly around the `fit()` call only
(not globally) and only for statsmodels' OWN routine fit-time noise (its
ConvergenceWarning/ValueWarning categories plus the two specific
starting-parameter UserWarnings -- see `_import_statsmodels_warnings`), so
genuine warnings, including unrelated UserWarnings raised during the fit,
are unaffected.
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
            'statsmodels is required for the ARIMA forecaster. It is normally a '
            'core hypertools dependency; reinstall hypertools, or install it '
            'directly with `pip install statsmodels`.'
        ) from e
    return SMArima


def _import_statsmodels_warnings():
    """The statsmodels warning categories (plus specific UserWarning message
    prefixes) that the fitter suppresses around each column's `.fit()` --
    kept NARROW so genuine, unrelated warnings still propagate (release-1.0
    audit, X4-warnings-015: the fitter used to blanket-suppress ALL
    UserWarnings during fit)."""
    from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                                 ValueWarning)
    # statsmodels raises these fit-time diagnostics as plain UserWarnings
    # (module statsmodels.tsa.statespace.sarimax); they are routine for
    # short/synthetic series and harmless for short forecasts.
    messages = ('Non-invertible starting MA parameters found',
                'Non-stationary starting autoregressive parameters found')
    return (ConvergenceWarning, ValueWarning), messages


def fitter(data, **kwargs):
    """Fit an independent `statsmodels` ARIMA model per column of `data`.

    statsmodels' routine fit-time warnings (ConvergenceWarning,
    ValueWarning, and the specific starting-parameter UserWarnings) raised
    during each column's `.fit()` are suppressed -- narrowly, only around
    the fit call, and only those; other warnings propagate.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to fit; one univariate ARIMA model is fit per column.
    **kwargs
        `order` : tuple of (p, d, q), ARIMA order (default: `(1, 1,
        1)`). Remaining kwargs (excluding `order`/`n_iter`) are
        forwarded to `statsmodels.tsa.arima.model.ARIMA`.

    Returns
    -------
    dict
        `{'results': [<fitted statsmodels ARIMAResults>, ...]}`, one
        entry per column of `data`, in column order.
    """
    sm_arima = _import_arima()
    sm_categories, sm_messages = _import_statsmodels_warnings()
    order = kwargs.get('order', (1, 1, 1))
    arima_kwargs = {k: v for k, v in kwargs.items() if k not in ('order', 'n_iter')}

    results = []
    for col in data.columns:
        x = data[col].to_numpy(dtype=float)
        with warnings.catch_warnings():
            # NARROW suppression (release-1.0 audit, X4-warnings-015):
            # only statsmodels' own routine fit-time noise is silenced --
            # its ConvergenceWarning/ValueWarning categories and the two
            # specific starting-parameter UserWarnings -- so any OTHER
            # UserWarning raised during the fit still reaches the caller.
            for category in sm_categories:
                warnings.filterwarnings('ignore', category=category)
            for message in sm_messages:
                warnings.filterwarnings('ignore', message=message,
                                        category=UserWarning)
            fit_result = sm_arima(x, order=order, **arima_kwargs).fit()
        results.append(fit_result)

    return {'results': results}


def forecaster(data, n_steps, future_index, **kwargs):
    """Forecast `n_steps` ahead per column using each column's fitted ARIMA model.

    Parameters
    ----------
    data : pandas.DataFrame
        The (fit-time) data; only its column names/order are used.
    n_steps : int
        Number of steps to forecast ahead.
    future_index : pandas.Index
        Index to assign to the forecasted rows.
    **kwargs
        `results` : list of fitted `ARIMAResults`, one per column (from
        `fitter`).

    Returns
    -------
    pandas.DataFrame
        Forecasted values, indexed by `future_index`, columns matching `data`.
    """
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
    if n_steps <= 0:
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
        ARIMA order (default: ``(1, 1, 1)``). The default suits
        drift/random-walk-like signals only -- it damps to a near-constant
        forecast within a few steps and cannot continue oscillations or
        extrapolate trends (see the module docstring for alternatives).
    **kwargs
        Passed through to ``statsmodels.tsa.arima.model.ARIMA`` (unknown
        keyword arguments therefore raise ``TypeError`` from statsmodels).
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
