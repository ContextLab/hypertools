"""Base class for hypertools forecasters (scikit-learn compatible).

A Forecaster wraps a (fitter, forecaster, applier, required-params)
quadruple, mirroring `hypertools.manip.common.Manipulator` but fitting ONE
model PER dataset: `fit` runs the fitter separately on each dataset (a list
of datasets yields a list of fitted param dicts, stored in ``models_``);
`predict` returns a forecast with `t` new rows continuing each dataset's
index; `fit_predict` chains the two. Child classes (Kalman, GaussianProcess,
AutoRegressor, ARIMA, Laplace, Chronos) supply the fitter/forecaster
callables plus their own defaults.

``predict_new(data, t)`` (used by the ``return_model=True`` round-trip: see
`hypertools.predict.predict`) applies the LEARNED parameters from a previous
`fit` to a NEW dataset without re-estimating them, via a child-supplied
``applier(fitted_params, new_data, t)`` callable; ``applier=None`` falls back
to conditioning on the new data directly (see `Forecaster.predict_new`).
"""
import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..core.shared import as_dataframe as _as_dataframe
from ..core.exceptions import _InsufficientHistoryError


def _infer_step(index):
    """Median positive gap between sorted observation times."""
    from .time import infer_step
    return infer_step(index)


def resolve_t(data, t, step=None):
    """Resolve a forecast horizon into a step count and a continued index.

    Implements GH #169's ``t`` semantics:

    - ``t`` an int: forecast ``t`` timesteps ahead. One timestep is the
      index's CALENDAR frequency when it has one -- stored, inferable with
      ``pd.infer_freq``, a ``PeriodIndex``'s own, or business days for
      weekday-only sessions -- so business-day data continue onto the next
      business days and month starts onto month starts (tz-aware days keep
      their local wall-clock time across DST); otherwise it is the median
      positive gap between sorted observations (a plain ``RangeIndex`` uses
      a step of 1). See `hypertools.predict.time`.
    - ``t`` a datetime-like value on time-indexed (``DatetimeIndex``) data:
      the number of steps (using the inferred step) from the last
      observation up to ``t``. If ``t`` is at or before the last
      observation, ``t`` is IN THE PAST (or exactly AT the end): a
      non-positive count is returned, meaning "truncate" (no forecasting
      model is needed) -- callers should slice the data instead of
      forecasting. In that case ``future_index`` is the (past-inclusive)
      index sliced up to ``t``, not an extension. A ``t`` BEFORE the first
      observation raises a `ValueError` (there is no data to truncate to
      and nothing to forecast; it used to silently return an empty frame).
      A ``t`` strictly after
      the last observation always forecasts at least one step (a target
      less than one full step ahead rounds up to a single step). A
      tz-naive ``t`` on tz-aware data is localized to the data's timezone.
      On a ``PeriodIndex`` the periods' start times are compared, and ``t``
      may also be a ``pd.Period``.

    An index that is not sorted ascending WARNS and is sorted together with
    its observations before fitting (forecasts continue from the latest time), and a TIME index (`DatetimeIndex`, `TimedeltaIndex`
    or `PeriodIndex`) carrying DUPLICATE entries raises a `ValueError`: the
    horizon is ill-defined when several observations share one position on
    the time axis. This is checked for every time-indexed input, flat or
    grouped (hypertools 1.1; `hyp.predict` names the offending group when the
    input was hierarchical). A duplicated NON-time index -- the stacked
    `pd.concat([run_a, run_b])` panel, whose index repeats 0..n-1 -- is
    unaffected: its step is still the median positive gap and the
    forecast still continues from the last row.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset whose index is being extended (or truncated).
    t : int or datetime-like
        The forecast horizon.
    step : number, duration, calendar offset, or None
        One step (see `hypertools.predict.time.resolve_step`); None uses the
        step `data` was prepared with, else infers it from the index.

    Returns
    -------
    n_steps : int
        Number of steps to forecast; zero or negative means "truncate"
        (see above).
    future_index : pandas.Index
        The continued index (or, for truncation, the sliced index). A
        ``PeriodIndex`` input is described by its start timestamps here;
        `Forecaster` returns forecasts as periods again.
    """
    from .time import (default_step, future_times, is_calendar_step,
                       order_time_data, resolve_step, time_coordinates)
    data = order_time_data(data)
    index = data.index

    if t is None:
        raise ValueError('t (forecast horizon) must be a positive integer '
                         'or a target datetime; got None')

    # duplicate observation TIMES make the horizon ill-defined: `_infer_step`
    # would take the (zero-length) gap between the repeats out of the running
    # and forecast on a step that no longer describes the data, and a
    # datetime-like `t` would truncate to an ambiguous position (1.1 plan,
    # Decisions #4: "preserve, warn, reject duplicates").
    #
    # SCOPED to time-like indexes on purpose (review of the Task 7 commit).
    # An unconditional check also rejected the `pd.concat([run_a, run_b])`
    # idiom, whose index repeats 0..n-1 -- measured at ea5d9b5e, that frame
    # forecast fine, and nothing about its horizon is ambiguous: the step is
    # the median positive gap (1) and the forecast continues from the
    # last row. Rejecting it contradicts Decisions #4's own "legitimate
    # integer-indexed panels are not rejected", and the message's argument
    # ("one position on the time axis") does not describe a positional index.
    # A datetime-like `t` on a non-time index already raises below.
    if (len(index) > 1 and not index.is_unique
            and isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex,
                                   pd.PeriodIndex))):
        if isinstance(index, pd.DatetimeIndex) and index.nunique() == 1:
            # the FULLY degenerate case (every observation at one timestamp)
            # is `_infer_step`'s, and tests/test_predict_audit_fixes.py:186
            # pins its wording. This check runs BEFORE `_infer_step`, so hand
            # the case over rather than copying the string -- a copy left
            # that branch dead code with a test that only pinned the copy.
            _infer_step(index)
        duplicated = index[index.duplicated()].unique()
        raise ValueError(
            f'the dataset index has {len(duplicated)} duplicated '
            f'entr{"y" if len(duplicated) == 1 else "ies"} (e.g. '
            f'{duplicated[0]!r}), so the forecast horizon is ill-defined: '
            'several observations share one position on the time axis. '
            'Aggregate the repeats (e.g. df.groupby(level=-1).mean()) or '
            'give them distinct times before forecasting.')

    step = resolve_step(index, step if step is not None else default_step(data))

    if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
        n_steps = int(t)
        last = index[-1]
        # Release review 2026-09-09: NumPy integer addition can wrap a
        # valid future time into the past at the dtype's upper bound.
        if isinstance(last, (int, np.integer)):
            last = int(last)
        if not (isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))
                or pd.api.types.is_numeric_dtype(index.dtype)):
            return n_steps, pd.RangeIndex(len(index), len(index) + n_steps)

        if isinstance(index, pd.RangeIndex) and isinstance(step, int):
            future_index = pd.RangeIndex(start=last + step, stop=last + step * (n_steps + 1), step=step)
        else:
            future_index = future_times(last, step, n_steps)
        return n_steps, future_index

    # a real raise (not `assert ..., ValueError(...)`, which raises
    # AssertionError and is stripped under `python -O`) -- QC 2026-07 red-team.
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f'a datetime-like t requires a time-indexed '
                         f'(DatetimeIndex) dataset; got t={t!r} on a '
                         f'{type(index).__name__}. For numerically-indexed '
                         'data, pass t as a positive integer number of steps.')

    target = t.start_time if isinstance(t, pd.Period) else pd.Timestamp(t)
    # tz-aware index + tz-naive t raised a raw pandas "Cannot compare
    # tz-naive and tz-aware timestamps" (QC 2026-07 red-team
    # F16-predict-020): localize the naive target to the data's timezone
    # (the unambiguous intent); a tz-aware t on tz-naive data is ambiguous,
    # so explain rather than guess.
    if index.tz is not None and target.tz is None:
        target = target.tz_localize(index.tz)
    elif index.tz is None and target.tz is not None:
        raise ValueError(
            f'the dataset index is timezone-naive but t={t!r} is '
            'timezone-aware; pass a tz-naive t (or localize the data index).')
    last = index[-1]

    # a target BEFORE the first observation used to silently return an
    # empty (0, n_features) frame (2026-07 release audit, final wave item
    # 14): there is nothing to truncate to and nothing to forecast, so
    # raise instead.
    first = index.min()
    if target < first:
        raise ValueError(
            f'the target time t={t!r} is before the first observation '
            f'({first}): there is no data to truncate to and nothing to '
            'forecast. Pass a t within the observed range (to truncate the '
            f'history) or after the last observation ({last}) to forecast.')

    if target <= last:
        # t at (or before) the last observation: truncate. n_steps == 0
        # (t exactly at the end) keeps the full history (QC 2026-07
        # red-team F16-predict-004: this used to fall through to the model
        # forecaster with n_steps=0, silently returning an all-NaN frame or
        # crashing model-dependently).
        keep = index <= target
        n_steps = -(len(index) - int(keep.sum()))
        future_index = index[keep]
        return n_steps, future_index

    # target is strictly after the last observation: always forecast at
    # least one step (a target within half a step of the end used to round
    # to n_steps=0 and crash downstream -- QC 2026-07 red-team).
    if is_calendar_step(step):
        # the nearest point of the calendar grid (business days, month
        # starts, ...), measured the way the model's time axis is
        elapsed = time_coordinates(pd.DatetimeIndex([target]), last, step)[0]
    else:
        elapsed = (target - last) / step
    n_steps = max(1, int(np.round(elapsed)))
    future_index = pd.DatetimeIndex(future_times(last, step, n_steps))
    return n_steps, future_index


class Forecaster(BaseEstimator):
    """Scikit-learn-compatible base class for hypertools forecasters.

    Wraps a `(fitter, forecaster, applier, required)` quadruple, fitting
    ONE model PER dataset: `fit` runs `fitter` separately on each
    dataset (a list of datasets yields a list of fitted param dicts,
    stored in `models_`); `predict` returns a forecast with `t` new rows
    continuing each dataset's index; `fit_predict` chains the two. Child
    classes (Kalman, GaussianProcess, AutoRegressor, ARIMA, Laplace,
    Chronos) supply the fitter/forecaster callables plus their own
    defaults via `**kwargs` to `__init__`.

    Parameters
    ----------
    **kwargs
        `data` : the dataset(s) to forecast (may be `None` until `fit`
        is called). `fitter` : callable that fits forecasting parameters
        and returns a dict. `forecaster` : callable that produces a
        forecast from fitted parameters. `applier` : optional callable
        that applies learned parameters to NEW data (for `predict_new`).
        `required` : list of parameter names `fitter` must return. Any
        remaining kwargs are forwarded to `fitter`/`forecaster` on every
        call.
    """

    #: The fewest observations (rows) a single dataset needs before this
    #: forecaster can be fit on it. `fit` raises a `ValueError` naming the
    #: model and this count for anything shorter, instead of letting the
    #: underlying library fail deep inside its own code (statsmodels' ARIMA
    #: raised a bare ``IndexError`` from a 2-row history). Subclasses whose
    #: floor depends on their hyperparameters override `min_history_for`
    #: (and, for instances, this attribute) -- see `ARIMA`. The animated
    #: `predict=` overlay in `hypertools.plot` reads it through
    #: `hypertools.plot.forecast.model_min_history`, so early frames draw no
    #: forecast until enough history has been revealed.
    min_history = 2

    @classmethod
    def min_history_for(cls, *args, **kwargs):
        """The `min_history` an instance built with these constructor
        arguments would have. The base rule ignores the arguments and returns
        the class attribute; `ARIMA` computes its floor from ``order``."""
        return cls.min_history

    def _check_min_history(self, d, which):
        """Raise a `ValueError` when dataset `d` (a DataFrame) has fewer rows
        than this forecaster needs; `which` names the dataset in the message."""
        needed = int(self.min_history)
        if d.shape[0] < needed:
            name = type(self).__name__
            detail = self._min_history_detail()
            raise _InsufficientHistoryError(
                f'cannot forecast with {name} from {d.shape[0]} row(s): '
                f'{which} is shorter than the {needed} observation(s) '
                f'(rows) {name}{detail} needs. Pass a longer history, or a '
                'model with a smaller minimum history (e.g. Kalman, which '
                'needs 2 rows).')

    def _min_history_detail(self):
        """Text appended to the model name in `_check_min_history`'s message
        (e.g. ARIMA's order); the base class adds nothing."""
        return ''

    def __init__(self, **kwargs):
        self.step = kwargs.pop('step', None)
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.forecaster = kwargs.pop('forecaster', None)
        self.applier = kwargs.pop('applier', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        """Fit a separate forecasting model on each dataset in `data`.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit. Each is fit independently via
            `self.fitter`, producing one fitted param dict per dataset
            (stored in `self.models_`).

        Returns
        -------
        Forecaster
            `self`, for chaining.

        Raises
        ------
        ValueError
            If `data` is `None`, empty, or has fewer than 2 observations
            (rows) -- or fewer than this forecaster's own `min_history`
            (the message names the model and the rows it needs); if
            `self.fitter` does not return a dict; or if any name in
            `self.required` is missing from a returned dict.
        """
        # real raises (not `assert ..., ValueError(...)`, which raises
        # AssertionError and is stripped under `python -O`) -- QC 2026-07.
        from ..core.shared import no_observations_message
        if data is None:
            raise ValueError(
                no_observations_message('forecast', 'data is None'))
        single = not isinstance(data, list)
        datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]

        from .time import prepare_time_data
        models = []
        for i, d in enumerate(datasets):
            # degenerate inputs used to fall through to model internals
            # (raw sklearn/pykalman errors) or return silent constant
            # "forecasts" (QC 2026-07 red-team F16-predict-013 /
            # X2-error-quality-002).
            which = 'the dataset' if single else f'dataset {i}'
            if d.shape[0] == 0 or d.shape[1] == 0:
                raise ValueError(
                    no_observations_message(
                        'forecast', f'{which} has shape {tuple(d.shape)}')
                    + ' Pass at least 2 observations (rows) of at least 1 '
                    'feature (column).')
            if d.shape[0] < 2:
                raise ValueError(
                    f'cannot forecast from a single observation: {which} has '
                    f'only {d.shape[0]} row. Forecasting needs at least 2 '
                    'observations (rows) to estimate how the data change '
                    'over time.')
            # a model-specific floor above the universal 2 (ARIMA's order
            # decides how many rows statsmodels can difference and fit)
            self._check_min_history(d, which)
            if self.fitter is None:
                models.append({})
                continue
            observed, model_data, delta = prepare_time_data(
                d, self.step, regular=getattr(self, '_regular_time_grid', False))
            self._check_min_history(model_data, which)
            datasets[i] = observed
            params = self.fitter(model_data, **self.kwargs)
            if not isinstance(params, dict):
                raise ValueError('fit function must return a dictionary')
            if not all(r in params for r in self.required):
                raise ValueError('one or more required fields not returned')
            params['_time_data'] = model_data
            params['_time_step'] = delta
            models.append(params)

        self.data = datasets[0] if single else datasets
        self.models_ = models
        return self

    def predict(self, t):
        """Forecast `t` steps beyond each fitted dataset's end.

        Parameters
        ----------
        t : int or datetime-like
            Forecast horizon, resolved per-dataset via `resolve_t`. A
            `t` at or before a dataset's last observation truncates
            that dataset's history up to `t` (inclusive) instead of
            forecasting.

        Returns
        -------
        A forecast DataFrame (or list of them, matching the structure of
        the data `fit` was called with).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit` has not been called yet, or a required fitted
            attribute is missing for a dataset.
        """
        if self.data is None or not hasattr(self, 'models_'):
            raise NotFittedError('must fit forecaster before predicting')

        from .time import finalize_forecast
        single = not isinstance(self.data, list)
        datasets = [self.data] if single else self.data

        forecasts = []
        for d, params in zip(datasets, self.models_):
            for r in self.required:
                if r not in params:
                    raise NotFittedError(f'missing fitted attribute: {r}')

            n_steps, future_index = resolve_t(d, t)

            if n_steps <= 0:
                # t is at or before the last observation: truncate rather
                # than forecast (n_steps == 0 used to fall through to the
                # model with a zero-step horizon -- QC 2026-07 red-team
                # F16-predict-004).
                forecasts.append(finalize_forecast(d.loc[future_index], d))
                continue

            if self.forecaster is None:
                forecasts.append(finalize_forecast(d, d))
                continue

            merged = {**params, **self.kwargs}
            forecasts.append(finalize_forecast(
                self.forecaster(params.get('_time_data', d), n_steps,
                                future_index, **merged), d))

        return forecasts[0] if single else forecasts

    def fit_predict(self, data, t):
        """Fit a forecasting model on `data`, then immediately forecast `t` steps ahead.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit and forecast.
        t : int or datetime-like
            Forecast horizon (see `predict`/`resolve_t`).

        Returns
        -------
        A forecast DataFrame (or list of them, matching `data`'s structure).
        """
        self.fit(data)
        return self.predict(t)

    def for_dataset(self, index):
        """A view of this fitted forecaster bound to ONE of its fitted
        datasets: `predict_new` on a single new series then reuses that
        dataset's learned parameters instead of refusing a count mismatch.
        `plot()`'s animated forecast schedule forecasts each dataset's
        revealed history on its own, so a forecaster fitted on several
        datasets (``hyp.predict([a, b], return_model=True)``) is applied
        dataset by dataset through this (Codex round 3)."""
        if not self.is_fitted:
            raise NotFittedError('must fit forecaster before calling for_dataset')
        fitted = self.data if isinstance(self.data, list) else [self.data]
        if len(self.models_) == 1:
            return self
        if not 0 <= int(index) < len(self.models_):
            raise IndexError(
                f'for_dataset({index}): the forecaster was fitted on '
                f'{len(self.models_)} dataset(s).')
        view = copy.copy(self)
        view.models_ = [self.models_[int(index)]]
        view.data = fitted[int(index)]
        return view

    @property
    def is_fitted(self):
        """Whether `fit` has already been run (so `predict_new` can reuse
        the learned parameters on new data without re-estimating them)."""
        return self.data is not None and hasattr(self, 'models_')

    def predict_new(self, data, t):
        """Apply the LEARNED parameters from a previous `fit` to a NEW
        dataset, forecasting `t` steps beyond its end -- no re-estimation.

        This is the no-re-estimation path behind ``return_model=True``: a
        forecaster returned by ``hyp.predict(A, ..., return_model=True)`` can
        be passed back as ``model=`` on a later call with new data ``B``, and
        the dispatcher routes it here instead of calling `fit_predict` again.

        Child classes that have genuinely reusable learned parameters (e.g.
        Kalman's transition/observation matrices, a fit GP, an already-fit
        sklearn regressor, a fit ARIMA result) supply an
        ``applier(fitted_params, new_data, t)`` callable that applies those
        parameters to ``new_data`` without re-fitting. Child classes with no
        reusable learned state -- Laplace and Chronos are context-conditioned
        online/foundation-model estimators with nothing to "fit" beyond the
        raw series -- leave ``applier=None``: reuse for those simply means
        re-deriving the (trivial) fitted params from the new series via the
        original ``fitter`` and forecasting from there ("condition on the new
        data" rather than "replay learned parameters").

        Parameters
        ----------
        data : DataFrame/array or list of these
            New dataset(s) to forecast from. If the number of new datasets
            matches the number of models fit previously, they are paired by
            position; if only one model was fit, it is reused for every new
            dataset.
        t : int or datetime-like
            Forecast horizon (see `resolve_t`).

        Returns
        -------
        A forecast DataFrame (or list of them, matching `data`'s structure).
        """
        if not self.is_fitted:
            raise NotFittedError('must fit forecaster before calling predict_new')

        single = not isinstance(data, list)
        new_datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]

        fitted_datasets = [self.data] if not isinstance(self.data, list) else self.data
        if len(self.models_) == len(new_datasets):
            paired_models = self.models_
            paired_fitted = fitted_datasets
        elif len(self.models_) == 1:
            paired_models = [self.models_[0]] * len(new_datasets)
            paired_fitted = [fitted_datasets[0]] * len(new_datasets)
        else:
            raise ValueError(
                f'predict_new got {len(new_datasets)} new dataset(s) but the '
                f'fitted forecaster has {len(self.models_)} fitted model(s); '
                'pass either a matching number of new datasets or reuse a '
                'forecaster that was fit on a single dataset.')

        # a feature-count mismatch used to surface as cryptic pandas/numpy
        # broadcast errors deep inside the model (QC 2026-07 red-team
        # F16-predict-012).
        for i, (d, fitted_d) in enumerate(zip(new_datasets, paired_fitted)):
            which = f'new dataset {i}' if len(new_datasets) > 1 else 'the new dataset'
            if d.shape[1] != fitted_d.shape[1]:
                raise ValueError(
                    f'the fitted forecaster expects {fitted_d.shape[1]} '
                    f'feature(s) (columns) but {which} has {d.shape[1]}; '
                    'reuse a fitted forecaster only on data with the same '
                    'columns it was fit on.')
            if d.shape[0] == 0:
                from ..core.shared import no_observations_message
                raise ValueError(
                    no_observations_message('forecast', f'{which} has 0 rows'))
            # the minimum history is what a FIT needs; a model with an
            # applier reuses its learned parameters and conditions on
            # whatever context the new data offers (statsmodels applies a
            # fitted AR(4) to two rows), so only the re-derive path below,
            # which refits on `d`, is held to it (release review, round 2)
            if self.applier is None:
                self._check_min_history(d, which)

        from .time import finalize_forecast, prepare_time_data, step_matches_index
        forecasts = []
        for d, params in zip(new_datasets, paired_models):
            for r in self.required:
                if r not in params:
                    raise NotFittedError(f'missing fitted attribute: {r}')

            # The learned interval is kept when it is expressed in the new
            # index's units (a one-hour transition never silently becomes a
            # three-hour one). Across index KINDS -- fit on an array, reused
            # on dated rows, or the reverse -- a row count and a duration
            # cannot be converted into each other, so the new data step in
            # their own units, as they did in 1.0 (release review
            # 2026-09-11: those reuses raised about `step` instead).
            step = next((s for s in (params.get('_time_step'), self.step)
                         if step_matches_index(s, d.index)), None)
            observed, model_data, _ = prepare_time_data(
                d, step, regular=getattr(self, '_regular_time_grid', False))
            n_steps, future_index = resolve_t(observed, t)
            if n_steps <= 0:
                forecasts.append(finalize_forecast(observed.loc[future_index],
                                                   observed))
                continue
            d = model_data

            if self.applier is not None:
                merged = {**params, **self.kwargs}
                forecasts.append(finalize_forecast(self.applier(merged, d, t),
                                                   observed))
                continue

            # No reusable learned parameters: condition on the new data
            # directly (re-derive fitted params from `d` via the same
            # fitter/hyperparameters, then forecast forward).
            n_steps, future_index = resolve_t(d, t)
            if n_steps <= 0:
                forecasts.append(finalize_forecast(d.loc[future_index], observed))
                continue
            if self.forecaster is None:
                forecasts.append(finalize_forecast(d, observed))
                continue
            new_params = self.fitter(d, **self.kwargs) if self.fitter is not None else {}
            merged = {**new_params, **self.kwargs}
            forecasts.append(finalize_forecast(
                self.forecaster(d, n_steps, future_index, **merged), observed))

        return forecasts[0] if single else forecasts
