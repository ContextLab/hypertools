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
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


def _as_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(np.asarray(data))


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
        assert len(nonzero) > 0, ValueError('cannot infer a timestep: all observations share one timestamp')
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
      observation, ``t`` is IN THE PAST: a negative count is returned,
      meaning "truncate" (no forecasting model is needed) -- callers should
      slice the data instead of forecasting. In that case ``future_index``
      is the (past-inclusive) index sliced up to ``t``, not an extension.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset whose index is being extended (or truncated).
    t : int or datetime-like
        The forecast horizon.

    Returns
    -------
    n_steps : int
        Number of steps to forecast; negative means "truncate" (see above).
    future_index : pandas.Index
        The continued index (or, for truncation, the sliced index).
    """
    index = data.index

    if isinstance(t, (int, np.integer)) and not isinstance(t, bool):
        n_steps = int(t)
        step = _infer_step(index)
        last = index[-1]

        if isinstance(index, pd.RangeIndex):
            future_index = pd.RangeIndex(start=last + step, stop=last + step * (n_steps + 1), step=step)
        else:
            future_index = pd.Index([last + step * (i + 1) for i in range(n_steps)])
        return n_steps, future_index

    assert isinstance(index, pd.DatetimeIndex), \
        ValueError('a datetime-like t requires a time-indexed (DatetimeIndex) dataset')

    target = pd.Timestamp(t)
    step = _infer_step(index)
    last = index[-1]

    if target <= last:
        keep = index <= target
        n_steps = -(len(index) - int(keep.sum()))
        future_index = index[keep]
        return n_steps, future_index

    n_steps = int(np.round((target - last) / step))
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
            If `data` is `None`, if `self.fitter` does not return a
            dict, or if any name in `self.required` is missing from a
            returned dict.
        """
        assert data is not None, ValueError('cannot forecast an empty dataset')
        single = not isinstance(data, list)
        datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]

        models = []
        for d in datasets:
            if self.fitter is None:
                models.append({})
                continue
            params = self.fitter(d, **self.kwargs)
            assert isinstance(params, dict), ValueError('fit function must return a dictionary')
            assert all(r in params for r in self.required), \
                ValueError('one or more required fields not returned')
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
            that dataset's history up to `t` instead of forecasting.

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

            if n_steps < 0:
                # t is in the past: truncate rather than forecast
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

        if len(self.models_) == len(new_datasets):
            paired_models = self.models_
        elif len(self.models_) == 1:
            paired_models = [self.models_[0]] * len(new_datasets)
        else:
            raise ValueError(
                f'predict_new got {len(new_datasets)} new dataset(s) but the '
                f'fitted forecaster has {len(self.models_)} fitted model(s); '
                'pass either a matching number of new datasets or reuse a '
                'forecaster that was fit on a single dataset.')

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
            if n_steps < 0:
                forecasts.append(d.loc[future_index])
                continue
            if self.forecaster is None:
                forecasts.append(d)
                continue
            new_params = self.fitter(d, **self.kwargs) if self.fitter is not None else {}
            merged = {**new_params, **self.kwargs}
            forecasts.append(self.forecaster(d, n_steps, future_index, **merged))

        return forecasts[0] if single else forecasts
