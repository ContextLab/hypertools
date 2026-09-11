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


def _lag_order(component):
    """An ARIMA ``order`` component as a lag count: an int as is, a
    statsmodels sparse lag sequence (``[1, 3]``) as its highest lag (``0``
    for an empty one)."""
    if isinstance(component, (list, tuple, np.ndarray)):
        return int(max((int(v) for v in component), default=0))
    return int(component)


class ARIMA(Forecaster):
    """Per-column ARIMA forecaster (statsmodels).

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
    order : tuple of (p, d, q)
        ARIMA order (default: ``(1, 1, 1)``). The default suits
        drift/random-walk-like signals only -- it damps to a near-constant
        forecast within a few steps and cannot continue oscillations or
        extrapolate trends (see the module docstring for alternatives).
    **kwargs
        Passed through to ``statsmodels.tsa.arima.model.ARIMA`` (unknown
        keyword arguments therefore raise ``TypeError`` from statsmodels).

    Notes
    -----
    ``min_history`` (see `Forecaster.min_history`) is computed from the
    order by `min_history_for`: 3 rows for the default ``(1, 1, 1)``. `fit`
    raises a ``ValueError`` naming the model, its order and that count for a
    shorter history (statsmodels used to raise a bare ``IndexError``), and
    an animated ``hyp.plot(..., predict='ARIMA')`` draws no forecast on the
    frames that have revealed fewer rows than that.
    """

    @classmethod
    def min_history_for(cls, order=(1, 1, 1), **kwargs):
        """The fewest rows an ARIMA of this ``order`` can be fit on.

        ``max(d + 2, p + q + 1)``: statsmodels differences the series ``d``
        times and needs at least TWO rows left afterwards (measured on
        statsmodels 0.14: every order with ``d >= 1`` raised a bare
        ``IndexError`` from a ``d + 1``-row history, and ``d = 0`` a
        ``ValueError`` from one row), and a fit with fewer rows than ARMA
        coefficients plus one has nothing to estimate them from. The default
        ``(1, 1, 1)`` therefore needs 3 rows; ``(4, 0, 0)`` needs 5.

        ``p`` and ``q`` may also be statsmodels' SPARSE lag form -- a
        sequence of the lags to include, ``([1, 3], 0, 0)`` -- in which
        case the highest lag is the order that counts (the fit needs that
        many earlier rows), so the check accepts every order the fitter
        does (1.1 release review, round 2).
        """
        p, d, q = order
        return max(_lag_order(d) + 2, _lag_order(p) + _lag_order(q) + 1)

    @property
    def min_history(self):
        """`min_history_for(self.order)` -- see `Forecaster.min_history`."""
        return self.min_history_for(self.order)

    def _min_history_detail(self):
        return f'(order={tuple(self.order)!r})'

    _regular_time_grid = True

    def __init__(self, order=(1, 1, 1), step=None, **kwargs):
        required = ['results']
        super().__init__(step=step, order=order, fitter=fitter, forecaster=forecaster, applier=applier,
                          data=None, required=required, **kwargs)

        self.order = order
        self.fitter = fitter
        self.forecaster = forecaster
        self.applier = applier
        self.data = None
        self.required = required
