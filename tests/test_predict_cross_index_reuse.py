"""A fitted forecaster reused on a different KIND of index (release review
2026-09-11, finding 2 -- a regression against 1.0).

In 1.0 all eight cross cases below worked: a model fit on an array and reused
on dated rows forecast the dated continuation, and the reverse forecast the
array continuation. The 1.1 time-step bookkeeping carried the training step
across kinds -- a row count onto a DatetimeIndex (``ValueError: step for a time
index must specify a duration``) and a duration onto a RangeIndex
(``TypeError: float() argument must be ... not 'Timedelta'``).

A row count and a duration cannot be converted into each other, so reuse
across kinds steps in the NEW data's own units. The observable check: the
same fitted model, reused on the same values, gives the same forecast values
whether those values carry a positional or a regular dated index.
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp

MODELS = ['Kalman', 'ARIMA', 'GaussianProcess', 'AutoRegressor']


def _data(n=40, seed=1):
    values = np.random.default_rng(seed).normal(size=(n, 2)).cumsum(axis=0)
    dated = pd.DataFrame(values, index=pd.date_range('2024-01-01', periods=n,
                                                     freq='h'))
    return values, dated


@pytest.mark.parametrize('model', MODELS)
def test_array_fit_reused_on_dated_rows(model):
    values, dated = _data()
    _, fitted = hyp.predict(values, model=model, t=3, return_model=True)
    out = hyp.predict(dated, model=fitted, t=3)
    assert list(out.index) == list(pd.date_range(dated.index[-1], periods=4,
                                                 freq='h')[1:])
    positional = hyp.predict(values, model=fitted, t=3)
    np.testing.assert_allclose(out.to_numpy(), positional.to_numpy())


@pytest.mark.parametrize('model', MODELS)
def test_dated_fit_reused_on_an_array(model):
    values, dated = _data()
    _, fitted = hyp.predict(dated, model=model, t=3, return_model=True)
    out = hyp.predict(values, model=fitted, t=3)
    assert list(out.index) == [40, 41, 42]
    same_kind = hyp.predict(dated, model=fitted, t=3)
    np.testing.assert_allclose(out.to_numpy(), same_kind.to_numpy())


@pytest.mark.parametrize('model', ['Kalman', 'GaussianProcess'])
def test_dated_fit_reused_on_categorical_labels_and_timedeltas(model):
    values, dated = _data()
    _, fitted = hyp.predict(dated, model=model, t=3, return_model=True)
    labelled = pd.DataFrame(values, index=[f'r{i}' for i in range(40)])
    out = hyp.predict(labelled, model=fitted, t=2)
    assert list(out.index) == [40, 41]
    # a timedelta index takes the learned one-hour duration as it is
    elapsed = pd.DataFrame(values, index=pd.timedelta_range(0, periods=40,
                                                            freq='h'))
    out = hyp.predict(elapsed, model=fitted, t=2)
    assert list(out.index) == [pd.Timedelta(hours=40), pd.Timedelta(hours=41)]


def test_same_kind_reuse_still_keeps_the_learned_interval():
    values, dated = _data()
    _, fitted = hyp.predict(dated, model='Kalman', t=3, return_model=True)
    three_hourly = pd.DataFrame(values, index=pd.date_range('2024-01-01',
                                                            periods=40, freq='3h'))
    with pytest.warns(UserWarning, match='interpolated'):
        out = hyp.predict(three_hourly, model=fitted, t=2)
    assert out.index[0] - three_hourly.index[-1] == pd.Timedelta(hours=1)


def test_a_mismatched_explicit_step_is_a_clear_error():
    values, _ = _data()
    with pytest.raises(ValueError, match='numerical index must be a positive number'):
        hyp.predict(values, model='Kalman', step='1h', t=2)
