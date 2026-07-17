"""hyp.predict dispatcher: resolve a forecaster spec and fit_predict it.

The public `predict` first normalizes obviously-degenerate or ambiguous
inputs (None, scalars, empty inputs, 1-D series -- see `predict`'s data
docs), then hands off to a datawrangler-funnel-wrapped core so any input
(array / DataFrame / list / text / polars) arrives as DataFrame(s); the
resolved Forecaster (DataFrame-based) is applied directly rather than via
the array-based core.apply_model.

Model specs may be: a registered name (``FORECASTERS``' ``__name__``\\ s), a
dict in either ``{'model': ..., 'params': {...}}`` or the fork-style
``{'model': ..., 'args': [...], 'kwargs': {...}}`` form, a Forecaster
subclass, or a Forecaster instance. An instance that has already been fit
(``instance.is_fitted``) is routed to `Forecaster.predict_new` instead of
`fit_predict` -- the no-re-estimation path behind ``return_model=True``
(see `hypertools.predict.common.Forecaster.predict_new`): a fitted
forecaster returned by an earlier ``predict(..., return_model=True)`` call
can be passed back as ``model=`` on NEW data without re-estimating its
learned parameters.
"""
import numbers
import warnings

import numpy as np
import pandas as pd
import datawrangler as dw

from .common import Forecaster
from .kalman import Kalman
from .gp import GaussianProcess
from .autoreg import AutoRegressor
from .arima import ARIMA
from .laplace import Laplace
from .chronos import Chronos
from ..core.shared import unpack_model


FORECASTERS = [Kalman, GaussianProcess, AutoRegressor, ARIMA, Laplace, Chronos]

#: short-name aliases for forecasters, resolved before the registry lookup
#: (QC 2026-07: `hyp.predict(x, model='GP')` used to raise "unknown predict
#: model 'GP'" -- only the full 'GaussianProcess' name was registered).
_FORECASTER_ALIASES = {'GP': 'GaussianProcess'}


def _supported_names():
    return [f.__name__ for f in FORECASTERS]


def _spec_help():
    return (f'supported names: {", ".join(_supported_names())} (or pass a '
            "dict {'model': ..., 'args': [...], 'kwargs': {...}}, a "
            'Forecaster subclass, or a Forecaster instance directly)')


def _coerce_dataset(d):
    """Normalize ONE dataset-like object before wrangling: a 1-D array or a
    pandas Series is a UNIVARIATE TIMESERIES -- n observations of 1 feature,
    i.e. an (n, 1) column -- matching format_data/plot's convention. (QC
    2026-07 red-team F16-predict-002/-003: the funnel used to wrangle a
    (200,) array into ONE row of 200 features -- crashing the default model
    or silently echoing the input as a (t, 200) "forecast" -- and wrangled a
    Series into an EMPTY (0, 0) DataFrame, silently losing the data.)"""
    if isinstance(d, pd.Series):
        return d.to_frame()
    if isinstance(d, np.ndarray) and d.ndim == 1:
        return d.reshape(-1, 1)
    return d


def _normalize_data(data):
    """Validate/normalize `data` before it reaches the datawrangler funnel,
    turning degenerate inputs into clear errors instead of silent nonsense
    forecasts or cryptic library internals (QC 2026-07 red-team
    F16-predict-002/-003/-013, X2-error-quality-002/-004)."""
    if data is None:
        raise TypeError(
            'Unsupported data type passed to predict: None. Supported types: '
            'Numpy Array, Pandas DataFrame/Series, String, List of strings, '
            'List of numbers, or a list of datasets.')
    if isinstance(data, (bool, np.bool_)) or (isinstance(data, numbers.Number)
                                              and not isinstance(data, bool)):
        raise ValueError(
            f'cannot forecast from a single scalar observation ({data!r}); '
            'pass a timeseries with at least 2 observations (rows).')
    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            raise ValueError(
                f'cannot forecast from a single scalar observation '
                f'({data!r}); pass a timeseries with at least 2 observations '
                '(rows).')
        if data.size == 0:
            raise ValueError(
                f'cannot forecast an empty dataset (got an array of shape '
                f'{tuple(data.shape)}); pass at least 2 observations (rows).')
        return _coerce_dataset(data)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(
                'cannot forecast an empty dataset (got an empty list); pass '
                'a timeseries (or a list of timeseries) with at least 2 '
                'observations (rows) each.')
        # a flat list of numbers is a single univariate series, NOT one
        # dataset per scalar (which returned a list of n constant 1x1
        # "forecasts" -- QC 2026-07 red-team F16-predict-002)
        if all(isinstance(v, numbers.Number) and not isinstance(v, (bool, np.bool_))
               for v in data):
            return np.asarray(data, dtype=float).reshape(-1, 1)
        return [_coerce_dataset(d) for d in data]
    return _coerce_dataset(data)


@dw.decorate.funnel
def _wrangled_predict(data, model='Kalman', t=10, return_model=False, **kwargs):
    """Funnel-wrapped core of `predict` (see its docstring)."""
    # validate the forecast horizon (QC 2026-07: a numeric t<=0 silently
    # returned an empty (0, n_features) forecast). t may ALSO be a target
    # datetime/Timestamp (forecast up to that time), which passes through.
    # normalize a 0-d numpy array (np.array(5)) to its Python scalar so the
    # checks below see the value, not an array (QC 2026-07 red-team: a 0-d array
    # slipped past every check and hit a misleading downstream message).
    if isinstance(t, np.ndarray) and t.ndim == 0:
        t = t.item()
    # t=None used to fall through to misleading downstream errors ("requires
    # a DatetimeIndex" / "cannot convert float NaN to integer") -- QC 2026-07
    # red-team F16-predict-011.
    if t is None:
        raise ValueError('t (forecast horizon) must be a positive integer or '
                         'a target datetime; got None')
    # bool is an int subclass, and np.bool_ (np.True_) is a separate type that
    # is NOT caught by isinstance(t, bool) -- reject both explicitly.
    if isinstance(t, (bool, np.bool_)):
        raise ValueError(f"t (forecast horizon) must be a positive integer or "
                         f"a target datetime; got {t!r}")
    if isinstance(t, (int, np.integer)):
        if t < 1:
            raise ValueError(f"t (forecast horizon) must be >= 1; got {t}")
    elif isinstance(t, (float, np.floating)):
        raise ValueError(f"t (forecast horizon) must be an integer number of "
                         f"steps or a target datetime, not a float; got {t!r}")

    if isinstance(model, dict) and 'kwargs' not in model and 'args' not in model:
        # {'model': ..., 'params': {...}} form: unpack before handing the
        # inner model spec to unpack_model (which only auto-unpacks the
        # fork-style {'model', 'args', 'kwargs'} triple). Warn here (round17
        # Task 6 fix): this shortcut used to bypass unpack_model entirely
        # for this dict shape, silently swallowing its DeprecationWarning.
        if 'model' not in model:
            raise ValueError(
                f"a model spec dict must include a 'model' key; got keys "
                f'{sorted(model)}. Pass e.g. '
                "{'model': 'Kalman', 'kwargs': {...}}.")
        if 'params' in model:
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)
        kwargs = {**dict(model.get('params', {})), **kwargs}
        model = model['model']
    elif isinstance(model, dict) and 'model' not in model:
        raise ValueError(
            f"a model spec dict must include a 'model' key; got keys "
            f'{sorted(model)}. Pass e.g. '
            "{'model': 'Kalman', 'kwargs': {...}}.")

    if isinstance(model, str):
        model = _FORECASTER_ALIASES.get(model, model)
    try:
        resolved = unpack_model(model, valid=FORECASTERS, parent_class=Forecaster)
    except ValueError as e:
        # unpack_model's terse "unknown model: 42" (for non-string specs)
        # gave no hint about accepted spec forms (QC 2026-07 red-team
        # F16-predict-018); re-raise with the same rich message unknown
        # strings get, plus a pointer for raw sklearn regressors.
        raise ValueError(
            f'unknown predict model {model!r}; {_spec_help()}. To forecast '
            'with an arbitrary scikit-learn regressor, wrap it in '
            "AutoRegressor: model={'model': 'AutoRegressor', 'kwargs': "
            "{'model': <regressor>}}.") from e

    if isinstance(resolved, type):
        resolved = resolved(**kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        # merge outer **kwargs into the dict form's kwargs (outer kwargs win,
        # matching the deprecated params-dict form); they used to be silently
        # dropped for this spec form only (QC 2026-07 red-team
        # F16-predict-010).
        resolved = cls(*resolved.get('args', []),
                       **{**resolved.get('kwargs', {}), **kwargs})
    elif isinstance(resolved, str):
        raise ValueError(f'unknown predict model {resolved!r}; {_spec_help()}')
    elif kwargs:
        # an already-constructed instance cannot absorb constructor kwargs;
        # warn instead of silently ignoring them (QC 2026-07 red-team).
        warnings.warn(
            f'ignoring keyword argument(s) {sorted(kwargs)}: model= is '
            'already a constructed instance, so constructor parameters '
            'cannot be applied. Pass the class (or a name/dict spec) to '
            'set parameters.')

    if isinstance(resolved, Forecaster) and resolved.is_fitted:
        forecasts = resolved.predict_new(data, t)
    else:
        forecasts = resolved.fit_predict(data, t)

    return (forecasts, resolved) if return_model else forecasts


def predict(data, model='Kalman', t=10, return_model=False, **kwargs):
    """Forecast `t` new rows continuing each input dataset.

    Parameters
    ----------
    data : DataFrame/array or list of these
        Dataset(s) to forecast from, each shaped (n_observations,
        n_features) with at least 2 observations (rows). A 1-D array, a
        flat list of numbers, or a pandas Series is treated as a
        UNIVARIATE timeseries -- n observations of 1 feature, i.e. an
        (n, 1) column -- matching `hyp.plot`/`format_data`'s convention.
        Degenerate inputs (None, a scalar, empty data, fewer than 2 rows)
        raise a clear error instead of returning meaningless forecasts.

    model : str, dict, class, or Forecaster instance
        Which forecaster to use (default: 'Kalman'). A string is one of
        `FORECASTERS`' names (Kalman, GaussianProcess, AutoRegressor, ARIMA,
        Laplace, Chronos); the short alias `'GP'` also resolves to
        GaussianProcess. A dict may be ``{'model': ..., 'params': {...}}``
        (deprecated) or ``{'model': ..., 'args': [...], 'kwargs': {...}}``.
        A class or an already-constructed (unfitted) instance is used
        directly. An ALREADY-FITTED Forecaster instance (returned from a
        previous ``return_model=True`` call) is applied to `data` via
        `predict_new` (its learned parameters are reused -- not
        re-estimated).

        Note that the models' default configurations suit different
        signals: Kalman/AutoRegressor/GaussianProcess track oscillatory
        and trending data out of the box, while ARIMA(1, 1, 1) and Laplace
        defaults only suit drift/random-walk-like signals (see
        `hypertools.predict.arima` / `hypertools.predict.laplace`).

        NaN policy: Kalman, ARIMA, and Chronos tolerate missing (NaN)
        entries; GaussianProcess, AutoRegressor, and Laplace raise a clear
        `ValueError` (fill missing values first, e.g. with `hyp.impute`).

    t : int or datetime-like
        Forecast horizon (see `hypertools.predict.common.resolve_t`). A
        datetime-like `t` at or before the data's last timestamp truncates
        the history up to `t` instead of forecasting.

    return_model : bool
        If True, also return the fitted (or reused) Forecaster instance, so
        it can be passed back as `model=` on future calls with new data
        (default: False).

    **kwargs
        Passed through to the forecaster's constructor when `model`
        resolves to a name, class, or dict spec (merged into a dict spec's
        own ``kwargs``, with these outer values winning). When `model` is
        already a constructed instance they cannot be applied, and a
        warning is raised. Unknown parameter names raise `TypeError`
        (typos used to be swallowed silently for some models).

    Returns
    -------
    forecasts (and the fitted Forecaster if return_model=True). Lists in,
    lists out: a single input dataset returns a single forecast DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((40, 3)),
    ...               axis=0)
    >>> forecast = hyp.predict(x, model='Kalman', t=5)
    >>> forecast.shape
    (5, 3)
    """
    data = _normalize_data(data)
    return _wrangled_predict(data, model=model, t=t, return_model=return_model,
                             **kwargs)
