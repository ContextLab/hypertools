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
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..core.shared import as_dataframe as _as_dataframe


def _infer_step(index):
    """The minimum non-zero difference between any pair of observations.

    Adjacent (sorted) observations always yield the smallest gaps, so this
    is computed from successive differences of the sorted index.
    """
    if len(index) < 2:
        return pd.Timedelta(1, unit='s') if isinstance(index, pd.DatetimeIndex) else 1

    values = index.sort_values()
    diffs = values[1:] - values[:-1]

    if isinstance(index, pd.DatetimeIndex):
        nonzero = diffs[diffs != pd.Timedelta(0)]
        # a real raise (not `assert ..., ValueError(...)`, which raises
        # AssertionError and is stripped under `python -O`) -- QC 2026-07.
        if len(nonzero) == 0:
            raise ValueError('cannot infer a timestep: all observations '
                             'share one timestamp')
        return nonzero.min()

    diffs = np.asarray(diffs)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 1
    return nonzero.min()


def resolve_t(data, t):
    """Resolve a forecast horizon into a step count and a continued index.

    Implements GH #169's ``t`` semantics:

    - ``t`` an int: forecast ``t`` timesteps ahead. The timestep duration is
      the minimum non-zero difference between any pair of observations
      (index-aware for time-indexed data; a plain ``RangeIndex`` uses a step
      of 1).
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

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset whose index is being extended (or truncated).
    t : int or datetime-like
        The forecast horizon.

    Returns
    -------
    n_steps : int
        Number of steps to forecast; zero or negative means "truncate"
        (see above).
    future_index : pandas.Index
        The continued index (or, for truncation, the sliced index).
    """
    index = data.index

    if t is None:
        raise ValueError('t (forecast horizon) must be a positive integer '
                         'or a target datetime; got None')

    # a descending (e.g. newest-first CSV export) or otherwise unsorted index
    # silently produced a "forecast" from the OLDEST observation, landing
    # inside the observed range (QC 2026-07 red-team F16-predict-016).
    if len(index) > 1 and not index.is_monotonic_increasing:
        from ..core.model import external_stacklevel
        warnings.warn(
            'the dataset index is not sorted in ascending order; forecasts '
            'continue from the LAST row. If your data are newest-first, sort '
            'them (e.g. df.sort_index()) before forecasting.',
            stacklevel=external_stacklevel())

    if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
        n_steps = int(t)
        step = _infer_step(index)
        last = index[-1]

        if isinstance(index, pd.RangeIndex):
            future_index = pd.RangeIndex(start=last + step, stop=last + step * (n_steps + 1), step=step)
        else:
            future_index = pd.Index([last + step * (i + 1) for i in range(n_steps)])
        return n_steps, future_index

    # a real raise (not `assert ..., ValueError(...)`, which raises
    # AssertionError and is stripped under `python -O`) -- QC 2026-07 red-team.
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f'a datetime-like t requires a time-indexed '
                         f'(DatetimeIndex) dataset; got t={t!r} on a '
                         f'{type(index).__name__}. For numerically-indexed '
                         'data, pass t as a positive integer number of steps.')

    target = pd.Timestamp(t)
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
    step = _infer_step(index)
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
    n_steps = max(1, int(np.round((target - last) / step)))
    future_index = pd.DatetimeIndex([last + step * (i + 1) for i in range(n_steps)])
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

    def __init__(self, **kwargs):
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
            (rows); if `self.fitter` does not return a dict; or if any
            name in `self.required` is missing from a returned dict.
        """
        # real raises (not `assert ..., ValueError(...)`, which raises
        # AssertionError and is stripped under `python -O`) -- QC 2026-07.
        from ..core.shared import no_observations_message
        if data is None:
            raise ValueError(
                no_observations_message('forecast', 'data is None'))
        single = not isinstance(data, list)
        datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]

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
            if self.fitter is None:
                models.append({})
                continue
            params = self.fitter(d, **self.kwargs)
            if not isinstance(params, dict):
                raise ValueError('fit function must return a dictionary')
            if not all(r in params for r in self.required):
                raise ValueError('one or more required fields not returned')
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
                forecasts.append(d.loc[future_index])
                continue

            if self.forecaster is None:
                forecasts.append(d)
                continue

            merged = {**params, **self.kwargs}
            forecasts.append(self.forecaster(d, n_steps, future_index, **merged))

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

        forecasts = []
        for d, params in zip(new_datasets, paired_models):
            for r in self.required:
                if r not in params:
                    raise NotFittedError(f'missing fitted attribute: {r}')

            if self.applier is not None:
                merged = {**params, **self.kwargs}
                forecasts.append(self.applier(merged, d, t))
                continue

            # No reusable learned parameters: condition on the new data
            # directly (re-derive fitted params from `d` via the same
            # fitter/hyperparameters, then forecast forward).
            n_steps, future_index = resolve_t(d, t)
            if n_steps <= 0:
                forecasts.append(d.loc[future_index])
                continue
            if self.forecaster is None:
                forecasts.append(d)
                continue
            new_params = self.fitter(d, **self.kwargs) if self.fitter is not None else {}
            merged = {**new_params, **self.kwargs}
            forecasts.append(self.forecaster(d, n_steps, future_index, **merged))

        return forecasts[0] if single else forecasts
