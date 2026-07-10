"""hyp.predict dispatcher: resolve a forecaster spec and fit_predict it.

Wrapped by datawrangler's funnel so any input (array / DataFrame / list /
text / polars) arrives as DataFrame(s); the resolved Forecaster (DataFrame-
based) is applied directly rather than via the array-based core.apply_model.

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
import warnings

import numpy as np
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


@dw.decorate.funnel
def predict(data, model='Kalman', t=10, return_model=False, **kwargs):
    """Forecast `t` new rows continuing each input dataset.

    Parameters
    ----------
    data : DataFrame/array or list of these
        Dataset(s) to forecast from.

    model : str, dict, class, or Forecaster instance
        Which forecaster to use (default: 'Kalman'). A string is one of
        `FORECASTERS`' names (Kalman, GaussianProcess, AutoRegressor, ARIMA,
        Laplace, Chronos); the short alias `'GP'` also resolves to
        GaussianProcess. A dict may be ``{'model': ..., 'params': {...}}``
        or ``{'model': ..., 'args': [...], 'kwargs': {...}}``. A class or an
        already-constructed (unfitted) instance is used directly. An
        ALREADY-FITTED Forecaster instance (returned from a previous
        ``return_model=True`` call) is applied to `data` via `predict_new`
        (its learned parameters are reused -- not re-estimated).

    t : int or datetime-like
        Forecast horizon (see `hypertools.predict.common.resolve_t`).

    return_model : bool
        If True, also return the fitted (or reused) Forecaster instance, so
        it can be passed back as `model=` on future calls with new data
        (default: False).

    **kwargs
        Passed through to the forecaster's constructor when `model`
        resolves to a class (ignored when `model` is already an instance).

    Returns
    -------
    forecasts (and the fitted Forecaster if return_model=True). Lists in,
    lists out: a single input dataset returns a single forecast DataFrame.
    """
    # validate the forecast horizon (QC 2026-07: a numeric t<=0 silently
    # returned an empty (0, n_features) forecast). t may ALSO be a target
    # datetime/Timestamp (forecast up to that time), which passes through.
    # normalize a 0-d numpy array (np.array(5)) to its Python scalar so the
    # checks below see the value, not an array (QC 2026-07 red-team: a 0-d array
    # slipped past every check and hit a misleading downstream message).
    if isinstance(t, np.ndarray) and t.ndim == 0:
        t = t.item()
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

    args = []
    if isinstance(model, dict) and 'kwargs' not in model and 'args' not in model:
        # {'model': ..., 'params': {...}} form: unpack before handing the
        # inner model spec to unpack_model (which only auto-unpacks the
        # fork-style {'model', 'args', 'kwargs'} triple). Warn here (round17
        # Task 6 fix): this shortcut used to bypass unpack_model entirely
        # for this dict shape, silently swallowing its DeprecationWarning.
        if 'params' in model:
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)
        kwargs = {**dict(model.get('params', {})), **kwargs}
        model = model['model']

    if isinstance(model, str):
        model = _FORECASTER_ALIASES.get(model, model)
    resolved = unpack_model(model, valid=FORECASTERS, parent_class=Forecaster)

    if isinstance(resolved, type):
        resolved = resolved(*args, **kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        resolved = cls(*resolved.get('args', []), **resolved.get('kwargs', {}))
    elif isinstance(resolved, str):
        raise ValueError(
            f'unknown predict model {resolved!r}; supported names: '
            f'{", ".join(_supported_names())} (or pass a dict '
            "{'model': ..., 'params': {...}}, a Forecaster subclass, or a "
            'Forecaster instance directly)')

    if isinstance(resolved, Forecaster) and resolved.is_fitted:
        forecasts = resolved.predict_new(data, t)
    else:
        forecasts = resolved.fit_predict(data, t)

    return (forecasts, resolved) if return_model else forecasts
