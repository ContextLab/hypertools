"""Observation times and the explicit linear regular-grid forecast policy.

Release review 2026-09-08: times describe observations, not additional
response columns. GaussianProcess uses the coordinates directly; the discrete
time models fit a grid ending at the last observation, with linear
interpolation inside the observed range (never extrapolated training rows).

Release review 2026-09-11: a step is a NUMBER (numerical indexes), a fixed
DURATION (``pd.Timedelta``: hourly, minutely, tz-naive daily data...) or a
CALENDAR OFFSET (a ``pandas`` ``DateOffset``: business days, month starts,
quarters, and local calendar days on a tz-aware index). A datetime index whose
frequency is stored (``index.freq``), inferable (``pd.infer_freq``) or given
by its periods (``PeriodIndex``) steps on that calendar, so a business-day or
month-start series is REGULAR -- fitted on its own rows, forecast onto the
next business days / month starts -- rather than an "irregular" series with
uneven absolute gaps. Weekday-only data whose sessions skip a few weekdays
(exchange holidays) step in business days; the skipped weekdays are the only
rows the regular grid fills. A ``PeriodIndex`` is handled on its start
timestamps internally and comes back as periods of the same frequency.
"""
import contextlib
import contextvars
import datetime
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from ..core.exceptions import _InsufficientHistoryError

#: ``DataFrame.attrs`` keys used between the helpers below and
#: `hypertools.predict.common` (internal; stripped from returned forecasts).
TIME_STEP_ATTR = '_hypertools_time_step'
PERIOD_FREQ_ATTR = '_hypertools_period_freq'
ORDERED_ATTR = '_hypertools_time_ordered'

# Messages already issued inside one `warn_once_per_call()` scope (one
# hyp.plot call): plot re-checks the same data along several internal paths,
# and a shuffled index warned twice for one call (2026-09-11 review).
_WARNED_IN_CALL = contextvars.ContextVar('_hypertools_time_warned',
                                         default=None)


@contextlib.contextmanager
def warn_once_per_call():
    """Issue each distinct time-policy warning at most once inside this
    block (outermost scope wins, so nested calls share one record)."""
    if _WARNED_IN_CALL.get() is not None:
        yield
        return
    token = _WARNED_IN_CALL.set(set())
    try:
        yield
    finally:
        _WARNED_IN_CALL.reset(token)


def _warn_time(message):
    """Warn about a time policy at the caller's line, once per message per
    `warn_once_per_call()` scope."""
    seen = _WARNED_IN_CALL.get()
    if seen is not None:
        if message in seen:
            return
        seen.add(message)
    from ..core.model import external_stacklevel
    warnings.warn(message, UserWarning, stacklevel=external_stacklevel())

#: the most calendar grid points one coordinate computation may generate
_MAX_CALENDAR_POINTS = 2_000_000


def is_time_index(index):
    """Datetime/duration/period indexes and unique numerical coordinates."""
    return (isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
            or (pd.api.types.is_numeric_dtype(index.dtype) and index.is_unique))


def is_calendar_step(step):
    """Whether `step` is a calendar offset (business day, month start, ...)
    rather than a number or a fixed ``pd.Timedelta`` duration. Resolved steps
    never hold a fixed-length pandas ``Tick``: those become Timedeltas."""
    return isinstance(step, pd.offsets.BaseOffset)


def _step_text(step):
    """A step for messages: a calendar offset as its alias (``'B'``,
    ``'MS'`` -- what ``step=`` accepts), anything else as ``str()``."""
    if isinstance(step, pd.offsets.BaseOffset):
        return repr(step.freqstr)
    return str(step)


def order_time_data(data):
    """Sort timed observations together; leave categorical/duplicate IDs alone.

    The "not sorted" warning is issued once per dataset: the returned copy is
    marked, so the internal re-checks a forecast makes on it stay silent. A
    ``PeriodIndex`` is replaced by its start timestamps, and its frequency is
    recorded so forecasts can be returned as periods (`finalize_forecast`).
    """
    if (len(data) > 1 and not data.index.is_monotonic_increasing
            and not data.attrs.get(ORDERED_ATTR, False)):
        action = ('observations are sorted before forecasting' if is_time_index(data.index)
                  else 'repeated/categorical row IDs retain their input order')
        _warn_time('the dataset index is not sorted in ascending order; '
                   + action)
    if not is_time_index(data.index):
        result = data.copy()
        result.attrs[ORDERED_ATTR] = True
        return result
    if data.index.hasnans:
        raise ValueError('observation times must be finite and cannot contain NaT/NaN')
    if pd.api.types.is_numeric_dtype(data.index.dtype):
        if not np.isfinite(np.asarray(data.index, dtype=float)).all():
            raise ValueError('observation times must be finite')
    result = data.sort_index(kind='stable').copy()
    if isinstance(result.index, pd.PeriodIndex):
        result.attrs[PERIOD_FREQ_ATTR] = result.index.freqstr
        result.index = result.index.to_timestamp()
    result.attrs[ORDERED_ATTR] = True
    return result


def finalize_forecast(frame, observed):
    """A forecast (or truncated history) as returned to the caller: periods
    again for a ``PeriodIndex`` input, and without the internal ordering
    attributes."""
    freq = observed.attrs.get(PERIOD_FREQ_ATTR)
    if freq is not None and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = frame.index.to_period(freq)
    if PERIOD_FREQ_ATTR in frame.attrs or ORDERED_ATTR in frame.attrs:
        frame = frame.copy()
        frame.attrs.pop(PERIOD_FREQ_ATTR, None)
        frame.attrs.pop(ORDERED_ATTR, None)
    return frame


def _normalize_offset(offset, tz=None):
    """A pandas offset as a step: fixed-length offsets become Timedeltas;
    a day on a tz-aware index is a LOCAL calendar day (``DateOffset(days=n)``,
    identical in pandas 2 and 3), so days across a DST change stay at the
    same wall-clock time."""
    if not isinstance(offset, pd.offsets.BaseOffset):
        offset = to_offset(offset)
    if isinstance(offset, pd.offsets.Day):
        return (pd.DateOffset(days=int(offset.n)) if tz is not None
                else pd.Timedelta(days=int(offset.n)))
    if isinstance(offset, pd.offsets.Tick):
        return pd.Timedelta(offset)
    return offset


def _period_step(freqstr):
    """The calendar step between the START timestamps of consecutive periods
    of frequency `freqstr` (monthly periods start on month starts, weekly
    'W-SUN' periods on Mondays, ...), or None if pandas cannot name it."""
    starts = pd.period_range('2000-01-01', periods=3, freq=freqstr).to_timestamp()
    freq = pd.infer_freq(starts)
    return None if freq is None else _normalize_offset(freq)


def _gapped_calendar(index):
    """A calendar step for a sorted, tz-aware or naive DatetimeIndex with no
    inferable frequency, or None.

    Recognizes observations taken at one time of day on (a) month starts or
    month ends, stepping by the median number of months; (b) weekdays only,
    one business day apart at the median -- trading sessions, whose skipped
    weekdays are holidays; (c) tz-aware local calendar days.
    """
    local = index.tz_localize(None) if index.tz is not None else index
    days = local.normalize()
    time_of_day = local - days
    if not (time_of_day == time_of_day[0]).all():
        return None
    day_numbers = np.asarray(days.values.astype('datetime64[D]'))
    day_gaps = np.diff(day_numbers).astype(np.int64)
    if not len(day_gaps) or (day_gaps <= 0).any():
        return None
    month_start = bool((local.day == 1).all())
    if month_start or bool(local.is_month_end.all()):
        months = np.diff(np.asarray(local.year * 12 + local.month, dtype=np.int64))
        if (months > 0).all():
            count = max(1, int(np.round(np.median(months))))
            return (pd.offsets.MonthBegin(count) if month_start
                    else pd.offsets.MonthEnd(count))
    if bool((local.dayofweek < 5).all()):
        sessions = np.busday_count(day_numbers[:-1], day_numbers[1:])
        if np.median(sessions) == 1 and (day_gaps > sessions).any():
            return pd.offsets.BDay(1)
    if index.tz is not None:
        median = float(np.median(day_gaps))
        if median.is_integer():
            return pd.DateOffset(days=int(median))
    return None


def _calendar_offset(index):
    """The calendar step of a sorted DatetimeIndex, a fixed Timedelta for a
    fixed-frequency index, or None when the observations follow neither."""
    freq = index.freq
    if freq is None and len(index) >= 3:
        try:
            freq = pd.infer_freq(index)
        except (TypeError, ValueError):
            freq = None
    if freq is not None:
        return _normalize_offset(freq, index.tz)
    if len(index) < 3:
        return None
    return _gapped_calendar(index)


def infer_step(index):
    """One step of `index`: its calendar frequency when it has one (see the
    module docstring), otherwise the median positive gap between sorted
    observation times."""
    if isinstance(index, pd.PeriodIndex):
        step = _period_step(index.freqstr)
        if step is not None:
            return step
        index = index.to_timestamp()
    temporal = isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex))
    if len(index) < 2:
        return pd.Timedelta(seconds=1) if temporal else 1
    if not temporal and not pd.api.types.is_numeric_dtype(index.dtype):
        return 1
    ordered = index.sort_values()
    if isinstance(ordered, pd.DatetimeIndex):
        calendar = _calendar_offset(ordered)
        if calendar is not None:
            return calendar
    # Release review 2026-09-09: integer differences must not wrap at the
    # signed/unsigned dtype bounds. Subtract before converting to float so
    # large epoch offsets do not erase small gaps either.
    coordinates = (ordered.astype(object)
                   if pd.api.types.is_integer_dtype(ordered.dtype) else ordered)
    gaps = coordinates[1:] - coordinates[:-1]
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


def default_step(data):
    """The step `data` carries from an earlier resolution, else -- for a
    ``PeriodIndex`` input observed at every period -- one period (exact even
    for two observations, where no calendar can be inferred); None means
    infer from the observation times (periods sampled every other hour, or
    quarterly months, step at their own cadence)."""
    step = data.attrs.get(TIME_STEP_ATTR)
    freq = data.attrs.get(PERIOD_FREQ_ATTR)
    if step is None and freq is not None and len(data) > 1:
        period = _period_step(freq)
        if period is not None and isinstance(data.index, pd.DatetimeIndex):
            x = time_coordinates(data.index, data.index[-1], period)
            if np.allclose(np.diff(x), 1., rtol=1e-8, atol=1e-8):
                step = period
    return step


def step_matches_index(step, index):
    """Whether a (fitted or requested) `step` is expressed in `index`'s units:
    a duration or calendar offset for a datetime/period index, a duration for
    a timedelta index, a number otherwise. A fitted forecaster reused on a
    different KIND of index (array-fit, dated reuse and vice versa) cannot
    keep its training interval, so it steps in the new data's own units."""
    if step is None or isinstance(step, (bool, np.bool_)):
        return False
    durational = isinstance(step, (pd.Timedelta, datetime.timedelta,
                                   np.timedelta64, str))
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return durational or is_calendar_step(step)
    if isinstance(index, pd.TimedeltaIndex):
        return durational or isinstance(step, pd.offsets.Tick)
    return isinstance(step, (int, float, np.number)) and not durational


def _time_step(step, tz=None, calendar=True):
    """Validate a user/fitted step for a datetime (`calendar`) or timedelta
    index: a duration, or (datetime only) a calendar offset/alias."""
    if isinstance(step, (int, float, np.number)):
        raise ValueError('step for a time index must specify a duration, e.g. "1h"')
    offset = None
    if isinstance(step, pd.offsets.BaseOffset):
        offset = step
    elif isinstance(step, str):
        try:
            pd.Timedelta(step)
        except ValueError:
            try:
                offset = to_offset(step)
            except ValueError:
                raise ValueError(
                    f'step={step!r} is neither a duration (e.g. "1h") nor a '
                    'calendar frequency (e.g. "B" or "MS")') from None
    if offset is not None:
        resolved = _normalize_offset(offset, tz)
        if is_calendar_step(resolved):
            if not calendar:
                raise ValueError(
                    f'a calendar step ({step!r}) needs a datetime index; a '
                    'timedelta index takes a duration such as "1h"')
            reference = pd.Timestamp('2000-01-03')
            if not reference + resolved > reference:
                raise ValueError('step must be a positive time duration')
            return resolved
        step = resolved
    step = pd.Timedelta(step)
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError('step must be a positive time duration')
    return step


def resolve_step(index, step=None):
    """A positive step in the index's own units: a number for numerical
    indexes, a duration for timedelta indexes, and a duration or calendar
    offset for datetime/period indexes (see the module docstring)."""
    if step is None:
        return infer_step(index)
    if isinstance(step, (bool, np.bool_)):
        raise ValueError('step must be a positive number or time duration')
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return _time_step(step, tz=getattr(index, 'tz', None))
    if isinstance(index, pd.TimedeltaIndex):
        return _time_step(step, calendar=False)
    if isinstance(step, (pd.Timedelta, datetime.timedelta, np.timedelta64,
                         pd.offsets.BaseOffset)):
        raise ValueError(f'step for a numerical index must be a positive '
                         f'number; got {step!r}')
    try:
        step = float(step)
    except (TypeError, ValueError):
        raise ValueError(f'step for a numerical index must be a positive '
                         f'number; got {step!r}') from None
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be positive and finite')
    return int(step) if step.is_integer() else step


def calendar_points(origin, offset, first, last):
    """The calendar grid ``origin + k * offset`` for k = first..last (k != 0
    runs are generated by `pd.date_range`, whose lattice equals repeated
    offset arithmetic; ``k = 0`` is `origin` itself even off the lattice)."""
    points = []
    if first < 0:
        stop = min(last, -1)
        points.append(pd.date_range(end=origin + stop * offset,
                                    periods=stop - first + 1, freq=offset))
    if first <= 0 <= last:
        points.append(pd.DatetimeIndex([origin]))
    if last > 0:
        start = max(first, 1)
        points.append(pd.date_range(start=origin + start * offset,
                                    periods=last - start + 1, freq=offset))
    if not points:
        return pd.DatetimeIndex([], tz=origin.tz)
    result = points[0]
    for part in points[1:]:
        result = result.append(part)
    return result


def _calendar_coordinates(index, origin, offset):
    """Positions of `index` on the grid ``origin + k * offset``: integer on
    grid points, linear in elapsed time between neighbouring points."""
    index = pd.DatetimeIndex(index)
    origin = pd.Timestamp(origin)
    if not len(index):
        return np.zeros(0)
    low, high = min(index.min(), origin), max(index.max(), origin)
    sample = calendar_points(origin, offset, 1, 8)
    shortest = min((sample[1:] - sample[:-1]).min(), sample[0] - origin)
    shortest = max(shortest, pd.Timedelta(seconds=1))
    ahead = int(np.ceil((high - origin) / shortest)) + 1
    behind = int(np.ceil((origin - low) / shortest)) + 1
    if ahead + behind > _MAX_CALENDAR_POINTS:
        raise ValueError('these observation times span more than '
                         f'{_MAX_CALENDAR_POINTS:,} calendar steps; use a larger step')
    grid = calendar_points(origin, offset, -behind, ahead)
    positions = np.arange(-behind, ahead + 1, dtype=float)
    slot = np.clip(grid.searchsorted(index, side='right') - 1, 0, len(grid) - 2)
    left, right = grid[slot], grid[slot + 1]
    fraction = np.asarray((index - left) / (right - left), dtype=float)
    return positions[slot] + fraction


def time_coordinates(index, origin, step):
    """Elapsed times in units of one model step, independent of calendar epoch."""
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    if isinstance(origin, pd.Period):
        origin = origin.start_time
    if is_calendar_step(step):
        return _calendar_coordinates(index, origin, step)
    if pd.api.types.is_integer_dtype(index.dtype):
        index = index.astype(object)
    if isinstance(origin, (int, np.integer)):
        origin = int(origin)
    return np.asarray((index - origin) / step, dtype=float)


def future_times(last, step, n_steps):
    """The `n_steps` observation times after `last`, one `step` apart."""
    if is_calendar_step(step):
        return calendar_points(pd.Timestamp(last), step, 1, n_steps)
    return pd.Index([last + step * (i + 1) for i in range(n_steps)])


def prepare_time_data(data, step=None, regular=False):
    """Return sorted observations, fitting data, and the resolved step.

    Interpolation is linear, column by column, on a grid anchored at the
    latest observation. NaN values are retained, not imputed. Duplicate
    numerical row IDs remain positional for compatibility with stacked runs.
    Observations already one step apart -- including a calendar step such as
    business days or month starts -- are fitted as they are.
    """
    from .common import resolve_t

    observed = order_time_data(data)
    delta = resolve_step(observed.index,
                         step if step is not None else default_step(observed))
    observed.attrs[TIME_STEP_ATTR] = delta
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
    origin = observed.index[-1]
    if is_calendar_step(delta):
        index = calendar_points(origin, delta, 1 - count, 0)
        index.name = observed.index.name
    else:
        if isinstance(origin, (int, np.integer)):
            origin = int(origin)
        index = pd.Index([origin + int(i) * delta for i in grid],
                         name=observed.index.name)
    fitted = pd.DataFrame(interpolated, index=index, columns=observed.columns)
    fitted.attrs[TIME_STEP_ATTR] = delta
    own = _own_regular_step(observed)
    if step is not None and own is not None:
        message = (f'Regularly spaced observations (one every '
                   f'{_step_text(own)}) were linearly interpolated onto a grid '
                   f'with step={_step_text(delta)} before fitting this '
                   'discrete-time forecaster.')
    else:
        message = ('Irregular observation times were linearly interpolated onto a '
                   f'regular grid with step={_step_text(delta)} before fitting '
                   'this discrete-time forecaster. Pass step= to choose the grid '
                   'interval; GaussianProcess uses the actual observation times '
                   'without interpolation.')
    # attributed to the caller's line, not a hypertools frame: a fixed
    # stacklevel printed '.../hypertools/predict/common.py:435' in tutorials
    _warn_time(message)
    return observed, fitted, delta


def _own_regular_step(observed):
    """The observations' own step when they are regularly spaced on it
    (so a DIFFERENT requested step is a resampling, not a repair), else None."""
    try:
        own = infer_step(observed.index)
    except ValueError:
        return None
    x = time_coordinates(observed.index, observed.index[-1], own)
    return own if np.allclose(np.diff(x), 1., rtol=1e-8, atol=1e-8) else None
