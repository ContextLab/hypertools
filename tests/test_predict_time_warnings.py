"""Time-policy warnings are issued once, and only when they apply (release
review 2026-09-11, finding 4).

A stacked ``pd.concat([run_a, run_b])`` panel warned "not sorted" THREE times
per call (fit, the fit-time duplicate check, and the forecast horizon each
re-checked the same frame), and an explicit ``step=`` on regularly spaced
data called the observations "Irregular" when they were only being resampled
onto the requested grid.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def _messages(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return result, [str(w.message) for w in caught]


def _walk(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 2)).cumsum(axis=0)


@pytest.mark.parametrize('model', ['Kalman', 'ARIMA', 'GaussianProcess'])
def test_a_stacked_panel_warns_once_per_call(model):
    run = pd.DataFrame(_walk(10))
    stacked = pd.concat([run, run])
    forecast, messages = _messages(lambda: hyp.predict(stacked, model=model, t=3))
    assert sum('not sorted' in m for m in messages) == 1
    assert list(forecast.index) == [10, 11, 12]
    # one per dataset in a list, and once for a backtest
    _, messages = _messages(lambda: hyp.predict([stacked, stacked], model=model, t=2))
    assert sum('not sorted' in m for m in messages) == 2
    _, messages = _messages(lambda: hyp.predict(stacked, model=model, holdout=3))
    assert sum('not sorted' in m for m in messages) == 1


def test_shuffled_times_warn_once_and_reuse_warns_once():
    frame = pd.DataFrame(_walk(20, seed=1),
                         index=pd.date_range('2026-01-01', periods=20, freq='h'))
    shuffled = frame.iloc[np.random.default_rng(3).permutation(20)]
    (_, fitted), messages = _messages(
        lambda: hyp.predict(shuffled, t=2, return_model=True))
    assert sum('not sorted' in m for m in messages) == 1
    _, messages = _messages(lambda: hyp.predict(shuffled, model=fitted, t=2))
    assert sum('not sorted' in m for m in messages) == 1


def test_explicit_step_on_regular_data_is_not_called_irregular():
    frame = pd.DataFrame(_walk(20, seed=2),
                         index=pd.timedelta_range(0, periods=20, freq='250ms'))
    forecast, messages = _messages(
        lambda: hyp.predict(frame, model='Kalman', t=3, step='500ms'))
    interpolated = [m for m in messages if 'interpolated' in m]
    assert len(interpolated) == 1
    assert 'Irregular' not in interpolated[0]
    assert 'Regularly spaced' in interpolated[0]
    assert forecast.index[0] - frame.index[-1] == pd.Timedelta('500ms')
    # the step the data already have: nothing to resample, nothing to say
    _, messages = _messages(
        lambda: hyp.predict(frame, model='Kalman', t=3, step='250ms'))
    assert not [m for m in messages if 'interpolated' in m]


def test_irregular_data_with_an_explicit_step_are_still_called_irregular():
    times = pd.to_timedelta([0, 1, 3, 6, 7, 9, 13, 15, 17, 20], unit='h')
    frame = pd.DataFrame(_walk(10, seed=4), index=times)
    _, messages = _messages(lambda: hyp.predict(frame, model='Kalman', t=2, step='2h'))
    assert any('Irregular' in m for m in messages)


# ---------------------------------------------------------------------------
# Attribution (2026-09-11 tutorial re-execution): the time-policy warnings
# used a fixed stacklevel that landed on hypertools' own frames, so the
# tutorials printed '~/hypertools/hypertools/predict/common.py:435:
# UserWarning' and a shuffled index under hyp.plot warned twice (from two
# different library lines). They must point at the caller's line.

def _trading_days():
    idx = pd.bdate_range('2026-06-01', '2026-09-10')
    holidays = pd.to_datetime(['2026-06-19', '2026-07-03', '2026-09-07'])
    idx = idx[~idx.isin(holidays)]
    return pd.DataFrame({'v': _walk(len(idx))[:, 0]}, index=idx)


def _caught(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        call()
    return caught


def test_interpolation_warning_names_the_callers_line_and_a_readable_step():
    caught = [w for w in _caught(
        lambda: hyp.predict(_trading_days(), model='Kalman', t=3))
        if 'interpolated' in str(w.message)]
    assert caught
    assert all(w.filename == __file__ for w in caught), \
        [(w.filename, w.lineno) for w in caught]
    # a calendar step reads as its alias, not an object repr
    assert "step='B'" in str(caught[0].message), str(caught[0].message)


def test_backtest_interpolation_warning_names_the_callers_line():
    caught = [w for w in _caught(
        lambda: hyp.predict(_trading_days(), model='Kalman', holdout=5))
        if 'interpolated' in str(w.message)]
    assert caught
    assert all(w.filename == __file__ for w in caught), \
        [(w.filename, w.lineno) for w in caught]


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_unsorted_index_under_plot_warns_once_at_the_callers_line(backend):
    import matplotlib.pyplot as plt
    data = _trading_days().sample(frac=1.0, random_state=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default')   # Python's display filter
        hyp.plot(data, ndims=1, reduce=None, predict='Kalman', t=3,
                 backend=backend, show=False)
    unsorted = [w for w in caught if 'not sorted' in str(w.message)]
    assert len(unsorted) == 1, [(w.filename, w.lineno) for w in unsorted]
    assert unsorted[0].filename == __file__
    plt.close('all')
