"""hyp.predict dispatcher: resolve a forecaster spec and fit_predict it.

The public `predict` first normalizes obviously-degenerate or ambiguous
inputs (None, scalars, empty inputs, 1-D series -- see `predict`'s data
docs), then hands off to a datawrangler-funnel-wrapped core so any input
(array / DataFrame / list / text / polars) arrives as DataFrame(s); the
resolved Forecaster (DataFrame-based) is applied directly rather than via
the array-based core.apply_model.

Model specs may be: a registered name (``FORECASTERS``' ``__name__``\\ s), a
dict in either ``{'model': ..., 'params': {...}}`` (deprecated) or the
fork-style ``{'model': ..., 'args': [...], 'kwargs': {...}}`` form, a Forecaster
subclass, or a Forecaster instance. An instance that has already been fit
(``instance.is_fitted``) is routed to `Forecaster.predict_new` instead of
`fit_predict` -- the no-re-estimation path behind ``return_model=True``
(see `hypertools.predict.common.Forecaster.predict_new`): a fitted
forecaster returned by an earlier ``predict(..., return_model=True)`` call
can be passed back as ``model=`` on NEW data without re-estimating its
learned parameters.
"""
import copy
import numbers
import warnings

import numpy as np
import pandas as pd
import datawrangler as dw

from .backtest import backtest_predict, model_collection, spec_name
from .common import Forecaster
from .kalman import Kalman
from .gp import GaussianProcess
from .autoreg import AutoRegressor
from .arima import ARIMA
from .laplace import Laplace
from .chronos import Chronos
from ..core.shared import supported_names, unpack_model
from ..core.model import external_stacklevel


FORECASTERS = [Kalman, GaussianProcess, AutoRegressor, ARIMA, Laplace, Chronos]

#: short-name aliases for forecasters, resolved before the registry lookup
#: (QC 2026-07: `hyp.predict(x, model='GP')` used to raise "unknown predict
#: model 'GP'" -- only the full 'GaussianProcess' name was registered).
_FORECASTER_ALIASES = {'GP': 'GaussianProcess'}


def _spec_help():
    return (f'supported names: {", ".join(supported_names(FORECASTERS))} (or pass a '
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
    from ..core.shared import require_data, no_observations_message
    require_data(data, 'predict')
    if isinstance(data, tuple):
        # a tuple of datasets is accepted exactly like a list (2026-07
        # release audit, final wave item 15: it used to be funneled as one
        # opaque object and die with a misleading empty-dataset error)
        data = list(data)
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
                no_observations_message(
                    'forecast', f'got an array of shape {tuple(data.shape)}')
                + ' Pass at least 2 observations (rows).')
        return _coerce_dataset(data)
    if isinstance(data, pd.DataFrame) and (data.shape[0] == 0
                                           or data.shape[1] == 0):
        raise ValueError(
            no_observations_message(
                'forecast', f'got a DataFrame of shape {tuple(data.shape)}')
            + ' Pass at least 2 observations (rows).')
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(
                no_observations_message('forecast', 'got an empty list')
                + ' Pass a timeseries (or a list of timeseries) with at '
                'least 2 observations (rows) each.')
        # a flat list of numbers is a single univariate series, NOT one
        # dataset per scalar (which returned a list of n constant 1x1
        # "forecasts" -- QC 2026-07 red-team F16-predict-002)
        if all(isinstance(v, numbers.Number) and not isinstance(v, (bool, np.bool_))
               for v in data):
            return np.asarray(data, dtype=float).reshape(-1, 1)
        return [_coerce_dataset(d) for d in data]
    return _coerce_dataset(data)


def _validate_horizon(t):
    """Check `t` (the forecast horizon) and return it normalized.

    Split out of `_wrangled_predict` so the HIERARCHICAL path can run it
    ONCE, before the per-group loop: `t` describes the whole call, and
    routing its errors through the recursion made `hyp.predict(hier_df,
    t=0)` say "group ('Tech',): t (forecast horizon) must be >= 1", which
    sends a reader looking for a problem in that group's data (review of the
    Task 7/8 commits). Idempotent, so `_wrangled_predict` re-running it on
    each group is free.
    """
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
    return t


def _resolve_forecaster_spec(model, kwargs):
    """Normalize a `model=` spec into what `unpack_model` resolves it to.

    Returns `(resolved, kwargs)` where `resolved` is a class, a
    ``{'model': cls, ...}`` dict, or an already-constructed forecaster --
    deliberately BEFORE construction, so the hierarchical path can re-use
    this as a pre-flight validation of the SPEC (which describes the whole
    call) without building a forecaster it would throw away, and without
    reaching a `Chronos`-style constructor twice.

    Split out of `_wrangled_predict` for the same reason as
    `_validate_horizon`: an unknown model name is the caller's mistake, not
    a group's, and it used to be reported as "group ('Tech',): unknown
    predict model 'Kalmann'" (review of the Task 7/8 commits).
    """
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
                DeprecationWarning, stacklevel=external_stacklevel())
        kwargs = {**dict(model.get('params', {})), **kwargs}
        model = model['model']
    elif isinstance(model, dict) and 'model' not in model:
        raise ValueError(
            f"a model spec dict must include a 'model' key; got keys "
            f'{sorted(model)}. Pass e.g. '
            "{'model': 'Kalman', 'kwargs': {...}}.")

    if isinstance(model, str):
        model = _FORECASTER_ALIASES.get(model, model)
        # case-insensitive fallback (release-1.0 audit,
        # D09-tutorials-applied-014: model='kalman' used to raise "unknown
        # predict model 'kalman'" while listing 'Kalman' as supported)
        if model not in supported_names(FORECASTERS):
            _by_lower = {n.lower(): n for n in supported_names(FORECASTERS)}
            _by_lower.update({k.lower(): v
                              for k, v in _FORECASTER_ALIASES.items()})
            model = _by_lower.get(model.lower(), model)
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

    # `unpack_model` hands an UNRESOLVABLE name straight back as a string;
    # raised here rather than in the construction dispatch below so the
    # pre-flight validation sees it too.
    if isinstance(resolved, str):
        raise ValueError(f'unknown predict model {resolved!r}; {_spec_help()}')
    return resolved, kwargs


@dw.decorate.funnel
def _wrangle(data, **kwargs):
    """Run `data` through the SAME datawrangler funnel `_wrangled_predict`
    uses, and hand back the resulting DataFrame(s) instead of forecasting.

    The backtesting path (``holdout=``) has to SPLIT the data before any
    forecaster sees it, which means it needs the wrangled frames -- their
    index carries the time axis the held-out rows are compared on. Doing it
    through the funnel (rather than `pd.DataFrame(...)`) keeps `holdout=`
    accepting exactly what `predict` accepts.
    """
    return data


def _holdout_datasets(data):
    """The wrangled dataset(s) `holdout=` scores, always as a list."""
    frames = _wrangle(data)
    return frames if isinstance(frames, list) else [frames]


@dw.decorate.funnel
def _wrangled_predict(data, model='Kalman', t=10, return_model=False, **kwargs):
    """Funnel-wrapped core of `predict` (see its docstring)."""
    t = _validate_horizon(t)
    resolved, kwargs = _resolve_forecaster_spec(model, kwargs)

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
    elif kwargs:
        # an already-constructed instance cannot absorb constructor kwargs;
        # warn instead of silently ignoring them (QC 2026-07 red-team).
        warnings.warn(
            f'ignoring keyword argument(s) {sorted(kwargs)}: model= is '
            'already a constructed instance, so constructor parameters '
            'cannot be applied. Pass the class (or a name/dict spec) to '
            'set parameters.', stacklevel=external_stacklevel())

    if isinstance(resolved, Forecaster) and resolved.is_fitted:
        forecasts = resolved.predict_new(data, t)
    else:
        forecasts = resolved.fit_predict(data, t)

    return (forecasts, resolved) if return_model else forecasts


def predict(data, model='Kalman', t=10, return_model=False, holdout=None,
            metrics=None, per_column=False, return_forecasts=False, **kwargs):
    """Forecast `t` new rows continuing each input dataset.

    Parameters
    ----------
    data : DataFrame/array or list/tuple of these
        Dataset(s) to forecast from, each shaped (n_observations,
        n_features) with at least 2 observations (rows). A 1-D array, a
        flat list of numbers, or a pandas Series is treated as a
        UNIVARIATE timeseries -- n observations of 1 feature, i.e. an
        (n, 1) column -- matching `hyp.plot`/`format_data`'s convention;
        a tuple of datasets is treated exactly like a list.
        Degenerate inputs (None, a scalar, empty data, fewer than 2 rows)
        raise a clear error instead of returning meaningless forecasts.

        HIERARCHICAL input (1.1). A BARE DataFrame carrying a MultiIndex on
        one axis is split into groups and forecast one group at a time (see
        ``docs/hierarchy.rst``):

        - a COLUMN MultiIndex groups by every level ABOVE the innermost one;
          the innermost level is the FEATURE axis, so each group keeps all
          `len(data)` observations. A ``(Market, Sector, Measure)`` frame
          gives one forecast per sector, each with the three Measure columns.
          Feature correspondence across groups is NOMINAL (the shared rule in
          `hypertools.core.hierarchy.group_columns`, which the plotting path
          needs so per-level means are meaningful): every group must carry the
          SAME innermost labels -- duplicates matched by (label, occurrence)
          -- so a frame whose groups name different features (different
          sensors per rig) or have different widths raises a `ValueError`
          instead of forecasting; if the groups really are incommensurable,
          slice them yourself and forecast one flat frame per group. When
          groups share labels in a DIFFERENT order, later groups are permuted
          into the first group's order, so a group's forecast columns follow
          that order rather than its own.
        - a ROW MultiIndex groups by every level ABOVE the innermost one;
          the innermost level is the TIME axis and SURVIVES as each group's
          index (it is dropped from the grouping levels, not reset), so a
          datetime innermost level keeps its `DatetimeIndex` and `t` may be a
          future `Timestamp`. Because the index survives, a group whose times
          are out of order raises the usual "not sorted in ascending order"
          warning with its group name prepended, and a group whose TIME index
          (`DatetimeIndex`/`TimedeltaIndex`/`PeriodIndex`) has DUPLICATED
          entries raises a `ValueError` naming it (the horizon is ill-defined
          when several observations share one position on the time axis).
          That rejection is `resolve_t`'s, so it applies to FLAT time-indexed
          input too -- a change from 1.0, which forecast such frames.
          Integer-indexed panels are not affected, whether or not their index
          repeats (a stacked ``pd.concat([run_a, run_b])`` still forecasts).

        Each group is forecast by a RECURSIVE `predict` call. That
        terminates because grouping FLATTENS each group on the axis it
        grouped along, so no group can be re-detected as hierarchical
        (`hypertools.core.hierarchy`'s one invariant). Arguments that
        describe the WHOLE call -- `t` and the `model=` spec -- are checked
        ONCE before that loop, so a bad horizon or an unknown model name is
        reported plainly instead of being blamed on the first group. What
        the loop prefixes with the group's key is a per-group `ValueError`
        (the type every data-shaped rejection raises: too short a history,
        duplicated times), so a group that is too short to forecast says
        which one it was. Other exception types propagate unchanged; a
        group's WARNINGS, however, are always re-emitted with its key,
        whatever the group failed with.

        A frame with a MultiIndex on BOTH axes raises (which hierarchy wins
        is undefined), and so does a hierarchical frame nested inside a
        list/tuple -- the hierarchy determines the whole group list, which
        cannot be reconciled with a caller-supplied list of datasets.

    model : str, dict, class, Forecaster instance, or a COLLECTION of these
        Which forecaster to use (default: 'Kalman').

        SEVERAL MODELS AT ONCE (1.2). A LIST or TUPLE of specs forecasts
        each of them and returns a ``{name: forecast}`` dict in the order
        given -- the shape ``hyp.plot(..., predict=['Kalman', 'ARIMA'])``
        consumes. Names come from the specs (a string's registry spelling,
        a dict spec's inner model, a class/instance's ``__name__``) and
        repeats are numbered ``Kalman``, ``Kalman (2)``, ...; pass a DICT
        MAPPING NAMES TO SPECS -- ``model={'tiny Chronos': {'model':
        'Chronos', 'kwargs': {...}}}`` -- to name them yourself. (A dict
        carrying any of ``model``/``args``/``kwargs``/``params`` is a
        single spec, not a mapping.) ``'truth'`` is reserved as a name.
        With ``holdout=``, the collection becomes one scored ROW per model
        instead.

        A string is one of
        `FORECASTERS`' names (Kalman, GaussianProcess, AutoRegressor, ARIMA,
        Laplace, Chronos); the short alias `'GP'` also resolves to
        GaussianProcess, and names are matched case-insensitively
        ('kalman' works too). A dict may be ``{'model': ..., 'params': {...}}``
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

        Ownership on HIERARCHICAL input, where one model is needed per
        group. The caller's instance is never mutated either way, and
        ``return_model=True`` returns one distinct object per group:

        - a name, a class, a dict spec, or an UNFITTED instance: one
          independent model is FITTED PER GROUP. An unfitted instance is
          deep-copied per group, so the caller's object is never fitted and
          later groups do not fall onto the `predict_new` path just because
          an earlier group fitted the shared object.
        - an ALREADY-FITTED instance: its learned parameters are REUSED on
          every group (via `predict_new`), each through an independent deep
          copy -- the reuse promised above, applied group-wise rather than
          refitting.

    t : int or datetime-like
        Forecast horizon (see `hypertools.predict.common.resolve_t`). A
        datetime-like `t` at or before the data's last timestamp truncates
        the history up to `t` instead of forecasting; a `t` BEFORE the
        data's first timestamp raises a `ValueError` (there is no data to
        truncate to and nothing to forecast -- it used to silently return
        an empty frame).

    return_model : bool
        If True, also return the fitted (or reused) Forecaster instance, so
        it can be passed back as `model=` on future calls with new data
        (default: False). On hierarchical input this returns PARALLEL
        SEQUENCES -- ``([f0, f1, ...], [m0, m1, ...])``, mirroring the flat
        ``(forecast, model)`` shape -- rather than a list of pairs. With a
        COLLECTION of models it returns ``({name: forecast}, {name:
        model})``; it is not supported with ``holdout=``.

    holdout : int, float, True, or None
        BACKTEST instead of forecasting (1.2). Fit each model on the HEAD
        of the data, forecast the held-out TAIL, and return a scores frame
        comparing every model against the rows that actually happened. An
        int holds out that many rows; a float in (0, 1) holds out that
        FRACTION of them (rounded, at least 1); ``True`` holds out exactly
        `t` rows. The head must keep at least 2 rows.

        **`t` is not consulted** for an int/float `holdout`: the horizon IS
        the number of held-out rows, or the forecast would not line up
        one-to-one with the truth. (Use ``holdout=True`` to say "hold out
        `t` rows".) The horizon used is reported in the frame's ``horizon``
        column.

        Not available on HIERARCHICAL input (which group is scored would
        have to become a fourth axis of the frame); slice the groups and
        backtest them one at a time.

    metrics : str or sequence of str
        Which metrics to score, from ``'mae'``, ``'rmse'``, ``'mape'``
        (default: all three, in that order). Column labels are the
        upper-cased names. MAPE is a PERCENTAGE and is nan-safe on zeros:
        entries whose actual value is exactly 0 are dropped from its
        average (all-zero truth gives NaN) rather than turning the column
        into ``inf``. The FIRST metric is the RANKING metric behind
        ``attrs['best']``.

    per_column : bool
        False (default) averages each metric over the columns (and, for a
        list of datasets, over the datasets), giving ONE row per model.
        True returns the LONG form instead: one row per model x
        (dataset x) column, indexed by a MultiIndex -- ``.reset_index()``
        for a fully tidy frame. The verdict in ``attrs`` is computed from
        the averaged form either way.

    return_forecasts : bool
        With ``holdout=``, also return the forecasts that were scored:
        ``(scores, {name: forecast, 'naive': baseline, 'truth': held_out})``
        (default: False). For a LIST of datasets each value is a list of
        frames, one per dataset, in input order.

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
    A hierarchical DataFrame (see `data`) returns a LIST of forecasts, one
    per group in input order -- or, with ``return_model=True``, the parallel
    ``([f0, f1, ...], [m0, m1, ...])`` pair described there.

    A COLLECTION of models (see `model`) returns a ``{name: forecast}``
    dict whose values are exactly what a single-model call would have
    returned (a DataFrame, or a list of them for list/hierarchical input).

    With ``holdout=``, a scores DataFrame instead (see Notes) -- or
    ``(scores, forecasts)`` with ``return_forecasts=True``.

    Notes
    -----
    **Backtesting (``holdout=``).** The returned frame has one ROW per
    model, in the order the models were given, plus an always-present
    ``'naive'`` row: the last-value-carried-forward baseline (each column's
    last OBSERVED training value, repeated over the horizon), which is the
    textbook random-walk benchmark for a timeseries. Its COLUMNS are the
    requested metrics (``MAE``, ``RMSE``, ``MAPE``), ``n`` (how many
    predicted/actual pairs were scored), ``unscored`` (held-out values the
    model failed to produce -- returned as NaN -- which are counted, and
    warned about, rather than silently dropped, since a model scored on
    fewer entries is not comparable to one scored on all of them) and
    ``horizon`` (the number of held-out rows). Multi-column data is scored
    per column and averaged; ``per_column=True`` keeps the per-column rows.

    HOW TO READ IT. ``scores.attrs`` carries the verdict:
    ``attrs['best']`` (the row with the lowest value of the ranking metric
    -- the FIRST entry of `metrics` -- among the real models, EXCLUDING the
    baseline, so what it names can always be handed back to `predict`),
    ``attrs['best_score']``, ``attrs['baseline']`` (``'naive'``),
    ``attrs['baseline_score']``, and ``attrs['beats_baseline']`` -- True
    only when the best model's score is strictly BELOW the baseline's. A
    model that does not beat the naive baseline has not earned its
    complexity, whatever its absolute error looks like. TIES on the ranking
    metric go to the model listed FIRST (the comparison is stable, so
    re-running with the same specs gives the same verdict); models whose
    ranking metric is NaN (nothing scoreable) are never chosen as best, and
    ``attrs['best']`` is None if no model scored at all.

    The verdict lives in ``attrs`` rather than in a ``best`` column on
    purpose: it is one fact about the whole comparison, and a column would
    repeat it on every row, force a non-numeric column into an otherwise
    all-numeric frame (breaking ``scores.mean()``, ``.plot.bar()`` and the
    describe-style bar chart), and become meaningless on a per-column
    slice. Note that pandas does not propagate ``attrs`` through every
    operation -- read the verdict from the frame `predict` returned, not
    from a slice of it.


    **Forecasts are not scale-invariant, so normalize heterogeneous
    features first.** These are fitted statistical models, not scale-free
    transforms: their priors, initial covariances and convergence criteria
    are expressed in the units you supply, so multiplying an input column
    by a constant does not simply multiply its forecast by the same
    constant. Measured on a 60x2 seasonal series, scaling the input by 100
    and dividing the one-step forecast change back by 100 moves it by 41%
    on one column for ``Kalman`` and 32% for ``ARIMA`` -- and raising
    ``n_iter`` from 1 to 100 does not close the gap, because `pykalman`'s
    EM starts from identity covariances and settles on a different
    optimum. Columns measured in wildly different units (millimetres beside
    percentages, say) will therefore not be weighted the way you expect.
    Pass ``normalize=`` upstream, or `hypertools.normalize` the data, when
    the features are not already comparable.

    Examples
    --------
    >>> import numpy as np
    >>> import hypertools as hyp
    >>> x = np.cumsum(np.random.default_rng(0).standard_normal((40, 3)),
    ...               axis=0)
    >>> forecast = hyp.predict(x, model='Kalman', t=5)
    >>> forecast.shape
    (5, 3)

    One forecast per group of a column hierarchy (the innermost level is
    the feature axis, so every group keeps all 3 measures):

    >>> import pandas as pd
    >>> columns = pd.MultiIndex.from_tuples(
    ...     [(sector, measure) for sector in ('Tech', 'Energy')
    ...      for measure in ('open', 'close', 'volume')],
    ...     names=['Sector', 'Measure'])
    >>> df = pd.DataFrame(np.cumsum(
    ...     np.random.default_rng(0).standard_normal((40, 6)), axis=0),
    ...     columns=columns)
    >>> [f.shape for f in hyp.predict(df, model='Kalman', t=5)]
    [(5, 3), (5, 3)]

    Several models at once -- one forecast per model, keyed by name:

    >>> forecasts = hyp.predict(x, model=['Kalman', 'AutoRegressor'], t=5)
    >>> {name: f.shape for name, f in forecasts.items()}
    {'Kalman': (5, 3), 'AutoRegressor': (5, 3)}

    Backtest them against the last 5 rows, naive baseline included:

    >>> scores = hyp.predict(x, model=['Kalman', 'AutoRegressor'], holdout=5)
    >>> list(scores.index)
    ['Kalman', 'AutoRegressor', 'naive']
    >>> list(scores.columns)
    ['MAE', 'RMSE', 'MAPE', 'n', 'unscored', 'horizon']
    >>> scores.attrs['best'] in ('Kalman', 'AutoRegressor')
    True
    >>> isinstance(scores.attrs['beats_baseline'], bool)
    True
    """
    data = _normalize_data(data)

    # MULTI-MODEL / BACKTEST dispatch (GH #285). Both are strictly opt-in:
    # `model` is a collection only when it is a list/tuple/name-mapping, and
    # the scoring path only runs for a non-None `holdout`, so the
    # single-model forecast path below is byte-for-byte what it was.
    collection = model_collection(
        model, valid=supported_names(FORECASTERS),
        aliases=_FORECASTER_ALIASES, caller='hyp.predict')
    if return_forecasts and holdout is None:
        raise ValueError(
            'return_forecasts=True only applies to a backtest; pass '
            'holdout= (the rows to hold out and score against), or drop '
            'return_forecasts to get the forecasts themselves.')
    if holdout is not None:
        if return_model:
            raise ValueError(
                'return_model=True is not supported with holdout=: a '
                'backtest fits one model per model spec and reports scores. '
                'Use return_forecasts=True for the scored forecasts, then '
                'refit on the full data with the winning spec.')
        if isinstance(data, pd.DataFrame) and (data.index.nlevels >= 2
                                               or data.columns.nlevels >= 2):
            raise ValueError(
                'holdout= is not supported on hierarchical (MultiIndex) '
                'input; slice the groups and backtest them one at a time.')
        if collection is not None:
            names, specs = collection
        else:
            names = [spec_name(model, supported_names(FORECASTERS),
                               _FORECASTER_ALIASES)]
            specs = [model]
        return backtest_predict(
            _holdout_datasets(data), predict, t, holdout, names, specs,
            metrics=metrics, per_column=per_column,
            return_forecasts=return_forecasts, kwargs=kwargs)
    if collection is not None:
        names, specs = collection
        results = {name: predict(data, model=spec, t=t,
                                 return_model=return_model, **kwargs)
                   for name, spec in zip(names, specs)}
        if return_model:
            return ({name: r[0] for name, r in results.items()},
                    {name: r[1] for name, r in results.items()})
        return results

    # imported INSIDE the function on purpose: `predict` re-enters itself
    # once per group below, and a function-level import resolves each name on
    # `hypertools.core.hierarchy` at CALL time -- which is what lets
    # tests/predict/test_predict_multiindex.py's observing wrappers (patched
    # onto that module) see the calls. A module-level import would bind the
    # originals once and the wrappers would never run.
    from ..core.hierarchy import (group_columns, group_rows_for_forecast,
                                  reject_dual_axis, reject_hierarchical_in_list)
    # this block runs AFTER `_normalize_data` (the plan leaves the order
    # open): the degenerate-input checks have to come first, or a frame with
    # a column MultiIndex but zero columns would group into ZERO leaves and
    # return an empty LIST of forecasts instead of `_normalize_data`'s clear
    # "no observations" error. `reject_hierarchical_in_list` accepts a list
    # or a tuple, so it is indifferent to `_normalize_data`'s tuple->list
    # conversion either way.
    reject_hierarchical_in_list(data, caller='hyp.predict', axes='both')
    if isinstance(data, pd.DataFrame) and (data.index.nlevels >= 2
                                           or data.columns.nlevels >= 2):
        reject_dual_axis(data)
        if data.columns.nlevels >= 2:
            # `group_columns` returns (leaves, META); the group LABELS live in
            # meta['leaf_keys']. Unpacking it as `keys` would label each group
            # with a dict key ('n_levels', 'leaf_keys', ...) in the messages
            # below. Labels come from the grouping key, NEVER from a leaf's
            # columns -- flattening (Contract 11) removes the tuples anyway.
            groups, _meta = group_columns(data)
            keys = _meta['leaf_keys']
        else:
            # NOT `expand_multiindex`: the innermost row level is the TIME
            # axis and survives as each group's index, so a datetime-like `t`
            # still works per group (hypertools/core/hierarchy.py's
            # `group_rows_for_forecast`).
            groups, keys = group_rows_for_forecast(data)

        # WHOSE mistake is it? `t` and the `model=` spec describe the WHOLE
        # CALL, but every check on them used to run inside the per-group
        # recursion, so `hyp.predict(hier_df, t=0)` reported "group
        # ('Tech',): t (forecast horizon) must be >= 1" and a misspelled name
        # reported "group ('Tech',): unknown predict model 'Kalmann'" --
        # blaming a group for an argument it had no part in (review of the
        # Task 7/8 commits). Validate both HERE, once, so those errors reach
        # the caller unprefixed; the `group {key}:` prefix below then means
        # what it says: something about THAT group's data.
        t = _validate_horizon(t)
        # VALIDATION ONLY -- the result is discarded, because each group
        # constructs (or deep-copies) its own forecaster below. Its warnings
        # are suppressed for the same reason: the real per-group calls emit
        # them, prefixed, and a pre-flight copy would double every one (the
        # deprecated ``{'model': ..., 'params': {...}}`` spec warns here).
        # `dict(kwargs)` because the deprecated form merges into `kwargs`.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _resolve_forecaster_spec(model, dict(kwargs))

        # TERMINATION (Contract 11): every group above is flat on the axis it
        # was grouped along -- flat columns, or the innermost level as a flat
        # index -- so the recursive `predict(group, ...)` below cannot
        # re-detect it as hierarchical. Never route `expand_multiindex`
        # leaves through here: those keep their row MultiIndex and re-expand
        # to themselves, so the recursion would not terminate.
        forecasts, models = [], []
        for key, group in zip(keys, groups):
            # OWNERSHIP: an UNFITTED instance is deep-copied so each group
            # fits independently and the caller's object is never fitted (and
            # later groups never fall onto `predict_new`); a FITTED instance
            # is deep-copied so each group reuses the same learned parameters
            # via `predict_new` (:216-219) without the groups sharing mutable
            # state. A NAME or a CLASS is stateless -- `_wrangled_predict`
            # constructs a fresh forecaster from it per call -- so copying it
            # would only obscure that. A DICT spec is NOT: `{'model':
            # <instance>}` is an accepted form (:149-166 unpacks it), and
            # passing dicts through un-copied fitted the caller's instance,
            # shared one model across every group, and dropped groups 2..n
            # onto `predict_new` -- all three promises above at once (review
            # of this task's first commit). Deep-copying a stateless dict is
            # behaviourally identical (measured: `copy.deepcopy({'model':
            # 'Kalman', 'kwargs': {}})` compares equal), so `dict` left the
            # passthrough rather than growing a "does it hold an instance?"
            # special case.
            group_model = (model if isinstance(model, (str, type))
                           else copy.deepcopy(model))
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    # `catch_warnings` SUPPRESSES what it records, so
                    # everything captured here must be re-emitted below or it
                    # is silently lost; 'always' defeats the
                    # __warningregistry__ dedup that would otherwise drop the
                    # 2nd..nth group's copy of an identical warning.
                    warnings.simplefilter('always')
                    result = predict(group, model=group_model, t=t,
                                     return_model=True, **kwargs)
                failure = None
            except ValueError as err:
                result, failure = None, err
            finally:
                # a `finally`, not straight-line code after the `try`: only
                # ValueError is caught (the plan leaves other types
                # un-prefixed on purpose), so a TypeError/NotFittedError used
                # to propagate past the re-emission and take the group's
                # warnings with it -- the flat path emits them, the
                # hierarchical one lost them. Re-emitted OUTSIDE the
                # `catch_warnings` context (inside it they would just be
                # re-captured) and BEFORE any exception leaves this block.
                # F8: the group name is prepended; the category is preserved
                # so a DeprecationWarning does not silently become a
                # UserWarning.
                for w in caught:
                    warnings.warn(f'group {key}: {w.message}', w.category,
                                  stacklevel=external_stacklevel())
            if failure is not None:
                raise ValueError(f'group {key}: {failure}') from failure
            forecasts.append(result[0])
            models.append(result[1])
        # PARALLEL SEQUENCES, mirroring the flat shape (forecast, model)
        # rather than a list of (forecast, model) pairs.
        return (forecasts, models) if return_model else forecasts

    return _wrangled_predict(data, model=model, t=t, return_model=return_model,
                             **kwargs)
