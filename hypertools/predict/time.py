"""Observation times and the explicit linear regular-grid forecast policy.

Release review 2026-09-08: times describe observations, not additional
response columns. GaussianProcess uses the coordinates directly; the discrete
time models fit a grid ending at the last observation, with linear
interpolation inside the observed range (never extrapolated training rows).
"""
import warnings

import numpy as np
import pandas as pd
from ..core.exceptions import _InsufficientHistoryError


def is_time_index(index):
    """Datetime/duration/period indexes and unique numerical coordinates."""
    return (isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
            or (pd.api.types.is_numeric_dtype(index.dtype) and index.is_unique))


def order_time_data(data):
    """Sort timed observations together; leave categorical/duplicate IDs alone."""
    if len(data) > 1 and not data.index.is_monotonic_increasing:
        action = ('observations are sorted before forecasting' if is_time_index(data.index)
                  else 'repeated/categorical row IDs retain their input order')
        warnings.warn('the dataset index is not sorted in ascending order; '
                      + action, UserWarning, stacklevel=3)
    if not is_time_index(data.index):
        return data.copy()
    if data.index.hasnans:
        raise ValueError('observation times must be finite and cannot contain NaT/NaN')
    if pd.api.types.is_numeric_dtype(data.index.dtype):
        if not np.isfinite(np.asarray(data.index, dtype=float)).all():
            raise ValueError('observation times must be finite')
    result = data.sort_index(kind='stable').copy()
    if isinstance(result.index, pd.PeriodIndex):
        result.index = result.index.to_timestamp()
    return result


def infer_step(index):
    """Median positive gap between sorted observation times."""
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    temporal = isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))
    if len(index) < 2:
        return pd.Timedelta(seconds=1) if temporal else 1
    if not temporal and not pd.api.types.is_numeric_dtype(index.dtype):
        return 1
    ordered = index.sort_values()
    gaps = ordered[1:] - ordered[:-1]
    if temporal:
        gaps = gaps[gaps > pd.Timedelta(0)]
        if not len(gaps):
            raise ValueError('cannot infer a timestep: all observations share one timestamp')
        return gaps.median()
    gaps = np.asarray(gaps, dtype=float)
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return 1
    value = float(np.median(gaps))
    return int(value) if value.is_integer() else value


def resolve_step(index, step=None):
    """A positive step in the index's own units (a duration for time indexes)."""
    if step is None:
        return infer_step(index)
    if isinstance(step, (bool, np.bool_)):
        raise ValueError('step must be a positive number or time duration')
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
        if isinstance(step, (int, float, np.number)):
            raise ValueError('step for a time index must specify a duration, e.g. "1h"')
        step = pd.Timedelta(step)
        if pd.isna(step) or step <= pd.Timedelta(0):
            raise ValueError('step must be a positive time duration')
        return step
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be positive and finite')
    return int(step) if step.is_integer() else step


def time_coordinates(index, origin, step):
    """Elapsed times in units of one model step, independent of calendar epoch."""
    return np.asarray((index - origin) / step, dtype=float)


def prepare_time_data(data, step=None, regular=False):
    """Return sorted observations, fitting data, and the resolved step.

    Interpolation is linear, column by column, on a grid anchored at the
    latest observation. NaN values are retained, not imputed. Duplicate
    numerical row IDs remain positional for compatibility with stacked runs.
    """
    from .common import resolve_t

    observed = order_time_data(data)
    delta = resolve_step(observed.index, step)
    observed.attrs['_hypertools_time_step'] = delta
    # Validate duplicated time stamps before fitting, including native-time
    # models. Horizon resolution owns the public diagnostic.
    resolve_t(observed, 1)
    if not regular or not is_time_index(observed.index) or len(observed) < 2:
        return observed, observed, delta
    x = time_coordinates(observed.index, observed.index[-1], delta)
    if np.allclose(np.diff(x), 1., rtol=1e-8, atol=1e-8):
        return observed, observed, delta
    count = int(np.floor(-x[0] + 1e-8)) + 1
    if count < 2:
        raise _InsufficientHistoryError(
            'step exceeds the observed time span; use a smaller step')
    if count > 1_000_000:
        raise ValueError('interpolation would exceed 1,000,000 rows; use a larger step')
    grid = np.arange(1 - count, 1, dtype=float)
    values = observed.to_numpy(dtype=float)
    interpolated = np.column_stack([np.interp(grid, x, col) for col in values.T])
    index = pd.Index([observed.index[-1] + int(i) * delta for i in grid],
                     name=observed.index.name)
    fitted = pd.DataFrame(interpolated, index=index, columns=observed.columns)
    fitted.attrs['_hypertools_time_step'] = delta
    warnings.warn(
        'Irregular observation times were linearly interpolated onto a regular '
        f'grid with step={delta} before fitting this discrete-time forecaster. '
        'Pass step= to choose the grid interval; GaussianProcess uses the '
        'actual observation times without interpolation.', UserWarning, stacklevel=3)
    return observed, fitted, delta
