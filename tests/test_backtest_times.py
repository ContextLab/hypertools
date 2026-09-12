"""Score real forecasts at held-out times, without using held-out values."""
import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

import hypertools as hyp
from hypertools.predict import GaussianProcess


def _irregular():
    times = np.array([0., 1., 3., 6., 7., 9., 13., 15., 15.5, 19., 23.5])
    return pd.DataFrame({'a': 2 * times + 1, 'b': np.sin(times)}, index=times)


def test_gp_backtest_evaluates_the_actual_held_out_coordinates():
    frame = _irregular()
    kernel = DotProduct(sigma_0=1, sigma_0_bounds='fixed')
    spec = GaussianProcess(kernel=kernel, alpha=1e-6, normalize_y=False)
    scores, result = hyp.predict(frame, model=spec, holdout=3, return_forecasts=True)
    train, held = frame.iloc[:-3], frame.iloc[-3:]
    reference = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=False).fit(
            np.asarray(train.index)[:, None] / 2, train)
    expected = reference.predict(np.asarray(held.index)[:, None] / 2)
    actual = result['GaussianProcess']
    pd.testing.assert_index_equal(actual.index, held.index)
    np.testing.assert_allclose(actual, expected)
    assert scores.loc['GaussianProcess', 'MAE'] == pytest.approx(
        np.abs(expected - held.to_numpy()).mean())
    assert not spec.is_fitted


@pytest.mark.parametrize('model', ['Kalman', 'ARIMA',
                                  {'model': 'AutoRegressor', 'kwargs': {'lags': 2}}])
@pytest.mark.parametrize('step', [None, 3])
def test_discrete_backtests_interpolate_a_covering_grid(model, step):
    frame = _irregular()
    train, held = frame.iloc[:-3], frame.iloc[-3:]
    interval = 2 if step is None else step
    x = (held.index.to_numpy() - 15) / interval
    grid_forecast = hyp.predict(train, model=model, step=step, t=int(np.ceil(x[-1])))
    values = np.vstack([train.iloc[-1], grid_forecast])
    expected = np.column_stack([np.interp(x, np.arange(len(values)), col)
                                for col in values.T])
    with pytest.warns(UserWarning, match='interpolated to the held-out'):
        scores, result = hyp.predict(frame, model=model, step=step, holdout=3,
                                      return_forecasts=True)
    name = model if isinstance(model, str) else model['model']
    np.testing.assert_allclose(result[name], expected)
    pd.testing.assert_index_equal(result[name].index, held.index)
    pd.testing.assert_frame_equal(result['truth'], held)
    assert scores.loc[name, 'MAE'] == pytest.approx(np.abs(expected - held).to_numpy().mean())
    assert scores.loc[name, 'horizon'] == 3  # observations, not generated steps


@pytest.mark.parametrize('model', ['GaussianProcess', 'Kalman'])
def test_shuffling_rows_and_changing_held_out_values_cannot_change_the_fit(model):
    frame = _irregular()
    original = frame.copy(deep=True)
    _, expected = hyp.predict(frame, model=model, holdout=3, return_forecasts=True)
    altered = frame.copy()
    altered.iloc[-3:] += 1000
    altered = altered.sample(frac=1, random_state=5)
    with pytest.warns(UserWarning, match='sorted'):
        scores, actual = hyp.predict(altered, model=model, holdout=3, return_forecasts=True)
    pd.testing.assert_frame_equal(actual[model], expected[model])
    assert scores.loc[model, 'MAE'] > 900
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize('kind', ['datetime', 'timedelta', 'period'])
def test_native_time_indexes_are_retained_in_backtest_results(kind):
    frame = _irregular()
    if kind == 'datetime':
        frame.index = pd.Timestamp('2026-03-07', tz='America/New_York') + pd.to_timedelta(frame.index, unit='h')
    elif kind == 'timedelta':
        frame.index = pd.to_timedelta(frame.index, unit='h')
    else:
        # Unique, irregular periods; normalization uses their start times.
        frame.index = pd.PeriodIndex(pd.Timestamp('2026-01-01') +
                                      pd.to_timedelta(frame.index * 2, unit='h'), freq='h')
    _, result = hyp.predict(frame, model=['Kalman', 'GaussianProcess'],
                            holdout=3, return_forecasts=True)
    # Periods are RETAINED too: a PeriodIndex input comes back as periods
    # (release review 2026-09-11; this used to expect `.to_timestamp()`).
    expected_index = frame.index[-3:]
    for forecast in result.values():
        pd.testing.assert_index_equal(forecast.index, expected_index)


def test_datasets_and_model_step_overrides_have_independent_grids():
    a = _irregular()
    b = a.copy()
    b.index = b.index * 3 + 100
    specs = {'auto': 'Kalman', 'explicit': {'model': 'Kalman', 'kwargs': {'step': 3}}}
    _, results = hyp.predict([a, b], model=specs, holdout=3, return_forecasts=True)
    for i, frame in enumerate([a, b]):
        for name, spec in specs.items():
            _, single = hyp.predict(frame, model=spec, holdout=3, return_forecasts=True)
            pd.testing.assert_frame_equal(results[name][i], single['Kalman'])


@pytest.mark.parametrize('index', [pd.Index(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
                                  pd.Index([0, 10, 30, 60, 100, 0, 10, 30])])
def test_positional_row_ids_keep_their_meaning_when_the_split_is_unique(index):
    frame = pd.DataFrame(np.random.default_rng(2).normal(size=(8, 2)), index=index)
    expected = hyp.predict(frame.iloc[:-3].reset_index(drop=True), t=3)
    _, result = hyp.predict(frame, holdout=3, return_forecasts=True)
    np.testing.assert_allclose(result['Kalman'], expected)
    pd.testing.assert_index_equal(result['Kalman'].index, frame.index[-3:])


def test_duplicate_timestamps_across_the_split_are_rejected():
    frame = pd.DataFrame(np.arange(8.), index=pd.to_datetime(
        ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04',
         '2026-01-05', '2026-01-05', '2026-01-06', '2026-01-07']))
    with pytest.raises(ValueError, match='duplicated'):
        hyp.predict(frame, holdout=3)


def test_regular_backtest_is_identical_to_direct_forecasting():
    frame = pd.DataFrame(np.random.default_rng(9).normal(size=(30, 2)),
                         index=pd.date_range('2026-01-01', periods=30, freq='2h'))
    for name in ['Kalman', 'ARIMA', 'AutoRegressor', 'GaussianProcess']:
        expected = hyp.predict(frame.iloc[:-3], model=name, t=3)
        _, result = hyp.predict(frame, model=name, holdout=3, return_forecasts=True)
        # The backtest retains the truth index's frequency metadata;
        # ordinary predict() returns an equivalent index without a freq.
        pd.testing.assert_frame_equal(result[name], expected, check_freq=False)


def test_missing_whole_steps_select_the_matching_forecasts_without_interpolation():
    frame = pd.DataFrame(np.sin(np.arange(14.)), index=list(range(12)) + [12, 19])
    expected = hyp.predict(frame.iloc[:-2], t=8).iloc[[0, 7]]
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        _, result = hyp.predict(frame, holdout=2, return_forecasts=True)
    assert not any('interpolated' in str(w.message) for w in caught)
    pd.testing.assert_frame_equal(result['Kalman'], expected)


def test_missing_training_anchor_is_not_filled_with_held_out_values():
    frame = pd.DataFrame([1., 2., 3., 4., 5., 6., 7., np.nan, 9., 10.],
                         index=[0., 1., 2., 3., 4., 5., 6., 7., 7.5, 8.])
    with pytest.warns(UserWarning, match='not directly comparable'):
        scores, result = hyp.predict(frame, holdout=2, return_forecasts=True)
    assert np.isnan(result['Kalman'].iloc[0, 0])
    assert np.isfinite(result['Kalman'].iloc[1, 0])
    assert scores.loc['Kalman', 'unscored'] == 1


def test_unreasonably_large_backtest_grid_has_a_clear_step_error():
    frame = pd.DataFrame(np.arange(9.), index=list(range(8)) + [100_000_000])
    with pytest.raises(ValueError, match='1,000,000 forecast steps.*larger step'):
        hyp.predict(frame, holdout=1)
