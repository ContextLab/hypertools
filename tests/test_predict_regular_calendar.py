"""Regular CALENDAR data are forecast on their own calendar (release review
2026-09-11, finding 1).

A business-day, month-start, weekly or quarterly index has uneven absolute
gaps (a weekend, a 28-to-31-day month), so the median-gap rule used to call it
"irregular": business-day bars were interpolated onto every calendar day
(fabricating weekend rows) and forecast onto Saturdays, month starts drifted
(03-04, 04-04), a tz-aware daily index grew a duplicated day across the fall
DST change, and a PeriodIndex came back as a DatetimeIndex.

Every check here is a real observable: the rows the fitted model actually saw,
the forecast's own index, and equality with the same values forecast by
position (a regular index must fit exactly like observation order).
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.holiday import USFederalHolidayCalendar

import hypertools as hyp
from tests._netskip import skip_on_transient_network

MODELS = ['Kalman', 'ARIMA', 'GaussianProcess',
          {'model': 'AutoRegressor', 'kwargs': {'lags': 2}}]


def _walk(n, cols=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, cols)).cumsum(axis=0)


def _forecast(frame, model, t=4):
    """Forecast, returning (forecast, fitted model, interpolation warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        forecast, fitted = hyp.predict(frame, model=model, t=t,
                                       return_model=True)
    interpolated = [str(w.message) for w in caught
                    if 'interpolated' in str(w.message)]
    return forecast, fitted, interpolated


def _unfrequenced(index):
    """The same timestamps WITHOUT a stored `freq`, so the calendar must be
    inferred from the observations (a CSV/API index never carries one)."""
    return pd.DatetimeIndex(list(index), name=index.name)


@pytest.mark.parametrize('model', MODELS, ids=lambda m: str(m)[:20])
@pytest.mark.parametrize('freq', ['B', 'MS', 'W-SUN', 'QS', 'ME', 'h'])
def test_regular_calendar_index_fits_observed_rows_and_steps_its_calendar(freq, model):
    index = _unfrequenced(pd.date_range('2023-01-02', periods=40, freq=freq))
    frame = pd.DataFrame(_walk(40, seed=3), index=index, columns=['a', 'b'])
    forecast, fitted, interpolated = _forecast(frame, model)

    assert interpolated == []
    # the model saw exactly the observed rows -- nothing fabricated
    assert list(fitted.models_[0]['_time_data'].index) == list(index)
    expected = pd.date_range(index[-1], periods=5, freq=freq)[1:]
    assert list(forecast.index) == list(expected)
    # a regular index fits exactly like observation order
    positional = hyp.predict(frame.reset_index(drop=True), model=model, t=4)
    np.testing.assert_allclose(forecast.to_numpy(), positional.to_numpy())


def test_business_day_forecasts_never_land_on_a_weekend():
    index = _unfrequenced(pd.bdate_range('2024-01-01', periods=60))
    frame = pd.DataFrame(_walk(60, seed=4), index=index)
    forecast, fitted, interpolated = _forecast(frame, 'Kalman', t=7)
    assert interpolated == []
    assert (forecast.index.dayofweek < 5).all()
    assert len(fitted.models_[0]['_time_data']) == 60
    # a datetime target counts BUSINESS days: Friday -> next Wednesday is 3
    friday = index[index.dayofweek == 4][-1]
    history = frame.loc[:friday]
    target = friday + pd.Timedelta(days=5)
    steps = hyp.predict(history, model='Kalman', t=target)
    assert list(steps.index) == list(pd.bdate_range(friday, periods=4)[1:])


@pytest.mark.parametrize('start,periods', [('2024-02-20', 30),    # spring DST in history
                                           ('2024-10-07', 25)])   # fall DST in forecast
def test_tz_aware_days_across_dst_step_local_calendar_days(start, periods):
    tz = 'America/New_York'
    index = _unfrequenced(pd.date_range(start, periods=periods, freq='D', tz=tz))
    frame = pd.DataFrame(_walk(periods, seed=5), index=index)
    forecast, fitted, interpolated = _forecast(frame, 'Kalman', t=6)
    assert interpolated == []
    assert len(fitted.models_[0]['_time_data']) == periods
    assert str(forecast.index.tz) == tz
    assert forecast.index.is_unique
    assert (forecast.index.hour == 0).all()
    assert list(forecast.index) == list(
        pd.date_range(index[-1], periods=7, freq='D', tz=tz)[1:])
    positional = hyp.predict(frame.reset_index(drop=True), model='Kalman', t=6)
    np.testing.assert_allclose(forecast.to_numpy(), positional.to_numpy())


@pytest.mark.parametrize('model', ['Kalman', 'GaussianProcess', 'ARIMA'])
@pytest.mark.parametrize('freq', ['M', 'Q', 'D', 'W'])
def test_period_index_forecasts_continue_as_periods(freq, model):
    index = pd.period_range('2020-01-01', periods=24, freq=freq)
    frame = pd.DataFrame(_walk(24, seed=6), index=index, columns=['a', 'b'])
    forecast, fitted, interpolated = _forecast(frame, model, t=3)
    assert interpolated == []
    assert isinstance(forecast.index, pd.PeriodIndex)
    assert forecast.index.freqstr == index.freqstr
    assert list(forecast.index) == list(pd.period_range(index[-1] + 1,
                                                        periods=3, freq=freq))
    positional = hyp.predict(frame.reset_index(drop=True), model=model, t=3)
    np.testing.assert_allclose(forecast.to_numpy(), positional.to_numpy())
    # truncation returns the ORIGINAL periods, too
    truncated = hyp.predict(frame, model=model, t=index[10].start_time)
    assert isinstance(truncated.index, pd.PeriodIndex)
    assert list(truncated.index) == list(index[:11])
    # a fitted model reused on new periods keeps producing periods
    again = hyp.predict(frame.iloc[:12], model=fitted, t=2)
    assert list(again.index) == list(pd.period_range(index[11] + 1, periods=2,
                                                     freq=freq))


def test_periods_step_at_their_observed_cadence():
    # hourly periods observed every second hour step two hours, uninterpolated
    index = pd.period_range('2026-01-01 00:00', periods=20, freq='2h').asfreq('h')
    frame = pd.DataFrame(_walk(20, seed=11), index=index)
    forecast, fitted, interpolated = _forecast(frame, 'Kalman', t=2)
    assert interpolated == []
    assert len(fitted.models_[0]['_time_data']) == 20
    assert list(forecast.index) == [index[-1] + 2, index[-1] + 4]
    # two monthly periods are one period apart: the next ones are months,
    # not 31-day strides that drift through the calendar
    pair = pd.DataFrame([[1.], [2.]], index=pd.period_range('2021-01', periods=2, freq='M'))
    ahead = hyp.predict(pair, model='Kalman', t=60)
    assert list(ahead.index) == list(pd.period_range('2021-03', periods=60, freq='M'))


@pytest.mark.parametrize('animate', [False, True])
def test_series_plot_draws_calendar_forecasts_on_their_dates(animate):
    """A plotted PeriodIndex / business-day series draws its forecast at the
    forecast's own dates (the static overlay used to place period forecasts
    at x = -1 once they came back as periods)."""
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    cases = [pd.period_range('2020-01', periods=24, freq='M'),
             _unfrequenced(pd.bdate_range('2024-01-01', periods=40))]
    for index in cases:
        frame = pd.DataFrame(_walk(len(index), seed=12), index=index)
        expected = hyp.predict(frame, model='Kalman', t=3)
        stamps = (expected.index.to_timestamp()
                  if isinstance(expected.index, pd.PeriodIndex) else expected.index)
        kwargs = dict(animate=True, duration=1, frame_rate=4,
                      slow_warning_seconds=None) if animate else {}
        out = hyp.plot(frame, ndims=1, reduce=None, predict='Kalman', t=3,
                       return_model=True, show=False, antialias=False, **kwargs)
        try:
            role = 'live' if animate else 'static'
            if animate:
                # the final frame of a 1 s, 4 fps animation reveals every row
                out['animation']._func(3, *out['animation']._args)
            lines = [line for line in out['fig'].axes[0].lines
                     if getattr(line, '_hyp_forecast_role', None) == role]
            assert len(lines) == 2
            for line in lines:
                np.testing.assert_allclose(np.asarray(line.get_xdata())[-1],
                                           date2num(stamps[-1].to_pydatetime()))
        finally:
            plt.close(out['fig'])


def test_trading_days_with_holidays_never_fabricate_or_forecast_weekends():
    """Market sessions: weekdays minus exchange holidays, so no frequency is
    inferable. The business-day calendar still governs -- only the missing
    holiday sessions may be filled, never a Saturday or a Sunday."""
    sessions = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    index = _unfrequenced(pd.date_range('2024-06-03', '2024-09-13', freq=sessions))
    holidays = pd.bdate_range(index[0], index[-1]).difference(index)
    assert len(holidays) == 3          # Juneteenth, July 4th, Labor Day
    frame = pd.DataFrame(_walk(len(index), seed=7), index=index)
    for model in ['Kalman', 'ARIMA']:
        forecast, fitted, _ = _forecast(frame, model, t=5)
        seen = fitted.models_[0]['_time_data'].index
        assert (seen.dayofweek < 5).all()
        assert len(seen) <= len(index) + len(holidays)
        assert list(forecast.index) == list(pd.bdate_range(index[-1], periods=6)[1:])


def test_genuinely_irregular_times_still_use_the_interpolated_grid():
    index = pd.to_datetime('2024-01-01') + pd.to_timedelta(
        [0, 1, 3, 6, 7, 9, 13, 15, 17, 20, 23, 24], unit='h')
    frame = pd.DataFrame(_walk(12, seed=8), index=index)
    _, fitted, interpolated = _forecast(frame, 'Kalman', t=2)
    assert interpolated and 'Irregular' in interpolated[0]
    assert len(fitted.models_[0]['_time_data']) != len(frame)


def test_backtest_on_business_days_scores_the_held_out_sessions_directly():
    index = _unfrequenced(pd.bdate_range('2024-01-01', periods=50))
    frame = pd.DataFrame(_walk(50, seed=9), index=index)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scores, forecasts = hyp.predict(frame, model='Kalman', holdout=6,
                                        return_forecasts=True)
    assert not [w for w in caught if 'interpolated' in str(w.message)]
    positional_scores, positional = hyp.predict(
        frame.reset_index(drop=True), model='Kalman', holdout=6,
        return_forecasts=True)
    np.testing.assert_allclose(forecasts['Kalman'].to_numpy(),
                               positional['Kalman'].to_numpy())
    assert list(forecasts['Kalman'].index) == list(index[-6:])


def test_period_backtest_returns_periods():
    index = pd.period_range('2020-01', periods=30, freq='M')
    frame = pd.DataFrame(_walk(30, seed=10), index=index)
    _, forecasts = hyp.predict(frame, model='Kalman', holdout=4,
                               return_forecasts=True)
    for name in ('Kalman', 'naive', 'truth'):
        assert isinstance(forecasts[name].index, pd.PeriodIndex)
        assert list(forecasts[name].index) == list(index[-4:])


def test_yahoo_daily_bars_forecast_trading_days():
    """The review's live case: yahoo:AAPL daily closes."""
    with skip_on_transient_network('loading yahoo:AAPL'):
        bars = hyp.load('yahoo:AAPL')
    closes = bars[['close']].iloc[-70:]
    assert (closes.index.dayofweek < 5).all()
    forecast, fitted, _ = _forecast(closes, 'Kalman', t=5)
    seen = fitted.models_[0]['_time_data'].index
    assert (seen.dayofweek < 5).all()
    missing = pd.bdate_range(closes.index[0], closes.index[-1]).difference(closes.index)
    assert len(seen) == len(closes) + len(missing)
    assert (forecast.index.dayofweek < 5).all()
    assert forecast.index[0] == closes.index[-1] + pd.offsets.BDay(1)
