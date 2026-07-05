"""Base class for hypertools forecasters (scikit-learn compatible).

A Forecaster wraps a (fitter, forecaster, required-params) triple, mirroring
`hypertools.manip.common.Manipulator` but fitting ONE model PER dataset:
`fit` runs the fitter separately on each dataset (a list of datasets yields a
list of fitted param dicts, stored in ``models_``); `predict` returns a
forecast with `t` new rows continuing each dataset's index; `fit_predict`
chains the two. Child classes (Kalman, GaussianProcess, AutoRegressor,
ARIMA, Laplace, Chronos) supply the fitter/forecaster callables plus their
own defaults.
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
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.forecaster = kwargs.pop('forecaster', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
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
        self.fit(data)
        return self.predict(t)
