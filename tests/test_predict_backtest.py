# -*- coding: utf-8 -*-
"""`hyp.predict(x, holdout=k)`: hold-out backtesting (GH #285).

Replaces the hand-rolled hold-out split, naive last-value baseline,
MAE/MAPE rows, model x ticker pivot and best-vs-naive verdict of
``docs/tutorials/stock_forecasting.ipynb`` cells 6-9.

Real forecasters on real (seeded) data; the "a perfect forecast scores 0"
case uses a genuine `Forecaster` -- a least-squares line extrapolator -- on
an exactly linear series, where being perfect is a property of the data and
the model, not of a stub.
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.backtest import mae, mape, rmse
from hypertools.predict.common import Forecaster


def _series(n=60, seed=0):
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'a': 0.05 * t + np.sin(t / 5.0) + 0.01 * rng.standard_normal(n),
                         'b': 0.05 * t - np.sin(t / 5.0) + 0.01 * rng.standard_normal(n)})


# --- a perfect forecaster (real model, exactly-linear data) ---------------

def _fit_line(data, **kwargs):
    x = np.arange(len(data))
    return {'coeffs': {c: np.polyfit(x, data[c].to_numpy(dtype=float), 1)
                       for c in data.columns},
            'n_fit': len(data)}


def _forecast_line(data, n_steps, future_index, coeffs=None, n_fit=None,
                   **kwargs):
    steps = np.arange(n_fit, n_fit + n_steps)
    return pd.DataFrame({c: np.polyval(coeffs[c], steps) for c in data.columns},
                        index=future_index, columns=data.columns)


class LinearExtrapolator(Forecaster):
    """Least-squares line per column, extrapolated -- exact on linear data."""

    def __init__(self, **kwargs):
        super().__init__(fitter=_fit_line, forecaster=_forecast_line,
                         required=['coeffs', 'n_fit'], **kwargs)


# --- metric definitions ---------------------------------------------------

def test_metrics_are_zero_for_an_exact_match():
    truth = np.array([1.0, 2.0, 4.0, 8.0])
    assert mae(truth, truth) == 0.0
    assert rmse(truth, truth) == 0.0
    assert mape(truth, truth) == 0.0


def test_mape_is_nan_safe_on_zeros():
    truth = np.array([0.0, 2.0])
    pred = np.array([1.0, 3.0])
    # the zero entry is dropped rather than producing inf
    assert np.isfinite(mape(pred, truth))
    assert mape(pred, truth) == pytest.approx(50.0)
    assert np.isnan(mape(np.array([1.0, 1.0]), np.array([0.0, 0.0])))


def test_metrics_ignore_missing_pairs():
    truth = np.array([1.0, np.nan, 3.0])
    pred = np.array([1.0, 100.0, 3.0])
    assert mae(pred, truth) == 0.0


# --- the scores frame -----------------------------------------------------

def test_holdout_returns_scores_with_a_naive_row():
    df = _series()
    scores = hyp.predict(df, model=['AutoRegressor', 'Kalman'], holdout=10)
    assert isinstance(scores, pd.DataFrame)
    assert list(scores.index) == ['AutoRegressor', 'Kalman', 'naive']
    assert list(scores.columns) == ['MAE', 'RMSE', 'MAPE', 'n', 'unscored',
                                    'horizon']
    assert (scores['horizon'] == 10).all()
    # 10 held-out rows x 2 columns scored per model
    assert (scores['n'] == 20).all()
    assert scores.to_numpy().dtype.kind == 'f'


def test_single_model_holdout_also_gets_the_baseline():
    scores = hyp.predict(_series(), model='Kalman', holdout=8)
    assert list(scores.index) == ['Kalman', 'naive']


def test_naive_row_is_the_last_value_carried_forward():
    df = _series(n=40)
    k = 6
    scores = hyp.predict(df, model='AutoRegressor', holdout=k)
    held = df.iloc[-k:].to_numpy()
    last = df.iloc[-k - 1].to_numpy()
    expected = np.mean([np.mean(np.abs(last[j] - held[:, j]))
                        for j in range(df.shape[1])])
    assert scores.loc['naive', 'MAE'] == pytest.approx(expected)


def test_a_perfect_forecast_scores_zero_and_beats_the_baseline():
    # exactly linear data: a line extrapolator reproduces the held-out rows
    linear = pd.DataFrame({'a': 2.0 + 3.0 * np.arange(30),
                           'b': -1.0 + 0.5 * np.arange(30)})
    scores = hyp.predict(linear, model={'oracle': LinearExtrapolator},
                         holdout=5)
    assert scores.loc['oracle', 'MAE'] == pytest.approx(0.0, abs=1e-8)
    assert scores.loc['oracle', 'RMSE'] == pytest.approx(0.0, abs=1e-8)
    assert scores.loc['oracle', 'MAPE'] == pytest.approx(0.0, abs=1e-8)
    assert scores.loc['naive', 'MAE'] > 1.0
    assert scores.attrs['best'] == 'oracle'
    assert scores.attrs['beats_baseline'] is True
    assert scores.attrs['baseline'] == 'naive'
    assert scores.attrs['metric'] == 'MAE'


def test_verdict_excludes_the_baseline_even_when_it_wins():
    # a random walk is exactly what the naive baseline is built for
    rng = np.random.default_rng(3)
    walk = pd.DataFrame({'x': np.cumsum(rng.standard_normal(60))})
    scores = hyp.predict(walk, model=['AutoRegressor'], holdout=10)
    assert scores.attrs['best'] == 'AutoRegressor'  # never 'naive'
    assert scores.attrs['beats_baseline'] == (
        scores.loc['AutoRegressor', 'MAE'] < scores.loc['naive', 'MAE'])


def test_ties_go_to_the_first_listed_model():
    df = _series(n=40)
    scores = hyp.predict(df, model=['AutoRegressor', 'AutoRegressor'],
                         holdout=5)
    assert list(scores.index) == ['AutoRegressor', 'AutoRegressor (2)', 'naive']
    assert scores.loc['AutoRegressor', 'MAE'] == pytest.approx(
        scores.loc['AutoRegressor (2)', 'MAE'])
    assert scores.attrs['best'] == 'AutoRegressor'


def test_ranking_metric_follows_the_first_requested_metric():
    df = _series(n=40)
    scores = hyp.predict(df, model=['AutoRegressor', 'Kalman'], holdout=6,
                         metrics=('rmse', 'mae'))
    assert list(scores.columns) == ['RMSE', 'MAE', 'n', 'unscored', 'horizon']
    assert scores.attrs['metric'] == 'RMSE'
    assert scores.attrs['best'] == scores.drop(index='naive')['RMSE'].idxmin()


def test_metrics_accepts_a_single_name():
    scores = hyp.predict(_series(n=40), model='Kalman', holdout=5, metrics='mae')
    assert list(scores.columns) == ['MAE', 'n', 'unscored', 'horizon']


# --- horizon resolution ---------------------------------------------------

def test_float_holdout_is_a_fraction():
    df = _series(n=50)
    scores = hyp.predict(df, model='AutoRegressor', holdout=0.2)
    assert (scores['horizon'] == 10).all()


def test_holdout_true_uses_t():
    df = _series(n=40)
    scores = hyp.predict(df, model='AutoRegressor', t=7, holdout=True)
    assert (scores['horizon'] == 7).all()


def test_t_is_ignored_for_an_int_holdout():
    df = _series(n=40)
    a = hyp.predict(df, model='AutoRegressor', holdout=5)
    b = hyp.predict(df, model='AutoRegressor', t=99, holdout=5)
    assert (a['horizon'] == 5).all()
    assert np.allclose(a.to_numpy(), b.to_numpy())


# --- forecasts and long form ---------------------------------------------

def test_return_forecasts_hands_back_what_was_scored():
    df = _series(n=40)
    scores, forecasts = hyp.predict(df, model=['Kalman'], holdout=6,
                                    return_forecasts=True)
    assert set(forecasts) == {'Kalman', 'naive', 'truth'}
    assert forecasts['truth'].shape == (6, 2)
    assert np.allclose(forecasts['truth'].to_numpy(), df.iloc[-6:].to_numpy())
    assert forecasts['Kalman'].shape == (6, 2)
    # the reported score really is that forecast's error
    err = np.mean(np.abs(forecasts['Kalman'].to_numpy()
                         - forecasts['truth'].to_numpy()), axis=0)
    assert scores.loc['Kalman', 'MAE'] == pytest.approx(np.mean(err))
    # the baseline forecast is a constant row of the last training values
    assert np.allclose(forecasts['naive'].to_numpy(),
                       df.iloc[-7].to_numpy()[None, :])


def test_per_column_long_form_averages_to_the_wide_form():
    df = _series(n=40)
    wide = hyp.predict(df, model=['Kalman'], holdout=6)
    long = hyp.predict(df, model=['Kalman'], holdout=6, per_column=True)
    assert list(long.index.names) == ['model', 'column']
    assert list(long.loc['Kalman'].index) == ['a', 'b']
    assert long.loc['Kalman', 'MAE'].mean() == pytest.approx(
        wide.loc['Kalman', 'MAE'])
    assert long.attrs['best'] == wide.attrs['best']
    assert long.reset_index().columns.tolist()[:2] == ['model', 'column']


def test_list_of_datasets_adds_a_dataset_level():
    a, b = _series(n=40, seed=0), _series(n=40, seed=1)
    wide = hyp.predict([a, b], model=['Kalman'], holdout=5)
    long = hyp.predict([a, b], model=['Kalman'], holdout=5, per_column=True)
    assert list(long.index.names) == ['model', 'dataset', 'column']
    assert sorted(long.loc['Kalman'].index.get_level_values('dataset').unique()) == [0, 1]
    assert long.loc['Kalman', 'MAE'].mean() == pytest.approx(
        wide.loc['Kalman', 'MAE'])
    assert wide.loc['Kalman', 'n'] == 2 * 5 * 2
    scores, forecasts = hyp.predict([a, b], model='Kalman', holdout=5,
                                    return_forecasts=True)
    assert isinstance(forecasts['truth'], list) and len(forecasts['truth']) == 2
    assert len(forecasts['Kalman']) == 2


def test_numpy_and_series_inputs_are_accepted():
    x = np.cumsum(np.random.default_rng(1).standard_normal((40, 3)), axis=0)
    scores = hyp.predict(x, model='AutoRegressor', holdout=5)
    assert list(scores.index) == ['AutoRegressor', 'naive']
    assert scores.loc['AutoRegressor', 'n'] == 15
    univariate = hyp.predict(pd.Series(x[:, 0]), model='AutoRegressor', holdout=5)
    assert univariate.loc['AutoRegressor', 'n'] == 5


def test_mape_survives_zeros_in_the_held_out_truth():
    # a series that crosses exactly through 0 in the held-out tail
    values = np.arange(-20.0, 20.0)
    scores = hyp.predict(pd.DataFrame({'x': values}), model='AutoRegressor',
                         holdout=25)
    assert np.isfinite(scores['MAPE']).all()


# --- errors ---------------------------------------------------------------

def test_holdout_too_large_for_the_data_raises():
    with pytest.raises(ValueError, match='at least 2 observations'):
        hyp.predict(_series(n=10), model='AutoRegressor', holdout=9)


def test_bad_holdout_values_raise():
    df = _series(n=40)
    with pytest.raises(ValueError, match='between 0 and 1'):
        hyp.predict(df, model='Kalman', holdout=1.5)
    with pytest.raises(ValueError, match='holdout must be'):
        hyp.predict(df, model='Kalman', holdout='10')
    with pytest.raises(ValueError, match='holdout=False'):
        hyp.predict(df, model='Kalman', holdout=False)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match='unknown metric'):
        hyp.predict(_series(n=30), model='Kalman', holdout=5, metrics='r2')


def test_return_model_with_holdout_raises():
    with pytest.raises(ValueError, match='not supported with holdout'):
        hyp.predict(_series(n=30), model='Kalman', holdout=5, return_model=True)


def test_return_forecasts_without_holdout_raises():
    with pytest.raises(ValueError, match='only applies to a backtest'):
        hyp.predict(_series(n=30), model='Kalman', t=3, return_forecasts=True)


def test_hierarchical_input_with_holdout_raises():
    columns = pd.MultiIndex.from_tuples(
        [(sector, measure) for sector in ('Tech', 'Energy')
         for measure in ('open', 'close')], names=['Sector', 'Measure'])
    df = pd.DataFrame(np.cumsum(
        np.random.default_rng(0).standard_normal((40, 4)), axis=0),
        columns=columns)
    with pytest.raises(ValueError, match='not supported on hierarchical'):
        hyp.predict(df, model='Kalman', holdout=5)


def test_a_model_named_naive_is_rejected():
    with pytest.raises(ValueError, match='reserved'):
        hyp.predict(_series(n=30), model={'naive': 'Kalman'}, holdout=5)


# --- the tutorial's shape -------------------------------------------------

def test_replaces_the_stock_tutorial_comparison():
    """One call replaces stock_forecasting cells 6-9: per-ticker hold-out,
    naive baseline, MAE/MAPE rows and the best-vs-naive verdict."""
    rng = np.random.default_rng(0)
    tickers = {f'T{i}': pd.DataFrame(
        {'log_close': np.cumsum(0.01 * rng.standard_normal(80)) + 5})
        for i in range(3)}
    scores = hyp.predict(list(tickers.values()),
                         model=['Kalman', 'AutoRegressor'], holdout=10,
                         metrics=('mape', 'mae'))
    assert list(scores.index) == ['Kalman', 'AutoRegressor', 'naive']
    assert scores.attrs['best'] in ('Kalman', 'AutoRegressor')
    # the per-ticker table the tutorial pivoted by hand
    per_ticker = hyp.predict(list(tickers.values()),
                             model=['Kalman', 'AutoRegressor'], holdout=10,
                             metrics=('mape',), per_column=True)
    table = per_ticker.reset_index().pivot(index='model', columns='dataset',
                                           values='MAPE')
    assert table.shape == (3, 3)
    assert np.isfinite(table.to_numpy()).all()
