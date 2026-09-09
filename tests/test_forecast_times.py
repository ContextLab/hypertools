"""Forecasts must use observation times, independently of their display units."""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

import hypertools as hyp
from hypertools.predict import GaussianProcess, Kalman


@pytest.mark.parametrize('model', ['Kalman', 'ARIMA',
                                  {'model': 'AutoRegressor', 'kwargs': {'lags': 2}}])
def test_discrete_models_fit_the_documented_interpolated_history(model):
    times = np.array([0., 1., 3., 6., 7., 9., 13., 15.])
    values = np.column_stack([times ** 2, np.sin(times)])
    # Median gap is 2; the grid ends at 15 and stays inside [0, 15].
    grid = np.arange(1., 16., 2.)
    expected_input = pd.DataFrame(
        np.column_stack([np.interp(grid, times, col) for col in values.T]), index=grid)
    expected = hyp.predict(expected_input, model=model, t=3)
    frame = pd.DataFrame(values, index=times)
    with pytest.warns(UserWarning, match='linearly interpolated'):
        actual, fitted = hyp.predict(frame, model=model, t=3, return_model=True)
    pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_allclose(actual.index, [17, 19, 21])
    # Truncation returns observed rows, never synthetic interpolation rows.
    pd.testing.assert_frame_equal(fitted.data, frame, check_flags=False)


def test_gp_fits_actual_times_against_a_real_sklearn_reference():
    times = np.array([0., 1., 3., 6., 7., 9., 13., 15.])
    values = np.column_stack([2 * times + 1, -times + 3])
    frame = pd.DataFrame(values, index=times)
    kernel = DotProduct(sigma_0=1, sigma_0_bounds='fixed')
    gp = GaussianProcess(kernel=kernel, alpha=1e-6, normalize_y=False)
    result = gp.fit_predict(frame, 3)
    reference = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                        normalize_y=False).fit((times / 2)[:, None], values)
    expected = reference.predict(np.array([17., 19., 21.])[:, None] / 2)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(gp.models_[0]['gp'].X_train_.ravel(), times / 2)


@pytest.mark.parametrize('model', ['Kalman', 'GaussianProcess'])
def test_permuted_timestamped_observations_have_the_same_forecasts(model):
    times = pd.date_range('2026-01-01', periods=20, freq='2h')
    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(20, 2)), index=times)
    expected = hyp.predict(frame, model=model, t=3)
    shuffled = frame.iloc[np.random.default_rng(11).permutation(len(frame))]
    with pytest.warns(UserWarning, match='sorted'):
        actual = hyp.predict(shuffled, model=model, t=3)
    pd.testing.assert_frame_equal(actual, expected)


def test_steps_are_per_dataset_and_reused_models_keep_their_time_scale():
    values = np.random.default_rng(2).normal(size=(20, 1)).cumsum(axis=0)
    frames = [pd.DataFrame(values, index=pd.date_range('2026-01-01', periods=20, freq=f))
              for f in ['1h', '3h']]
    predictions, model = hyp.predict(frames, t=2, return_model=True)
    for frame, forecast, hours in zip(frames, predictions, [1, 3]):
        assert forecast.index[0] - frame.index[-1] == pd.Timedelta(hours=hours)
    pd.testing.assert_frame_equal(model.for_dataset(1).predict(2), predictions[1])
    # An explicit 2-hour grid determines both fitting and future labels.
    with pytest.warns(UserWarning, match='interpolated'):
        overridden = Kalman(step='2h').fit_predict(frames[0], 2)
    assert overridden.index[0] - frames[0].index[-1] == pd.Timedelta(hours=2)


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
@pytest.mark.parametrize('scale', [1., 100.])
def test_series_plot_forecasts_match_joint_signal_forecasts(backend, scale):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    values = np.random.default_rng(42).normal(size=(20, 2)).cumsum(axis=0)
    frame = pd.DataFrame(values, index=np.arange(20.) * scale)
    expected = hyp.predict(frame, t=3)
    result = hyp.plot(frame, ndims=1, reduce=None, predict='Kalman', t=3,
                      backend=backend, return_model=True, antialias=False, show=False)
    try:
        np.testing.assert_allclose(result['predict']['forecasts'][0], expected)
        if backend == 'matplotlib':
            traces = [line for line in result['fig'].axes[0].lines
                      if getattr(line, '_hyp_forecast_role', None) == 'static']
            for col, line in enumerate(traces):
                np.testing.assert_allclose(line.get_xdata()[1:], expected.index)
                np.testing.assert_allclose(line.get_ydata()[1:], expected.iloc[:, col])
    finally:
        if backend == 'matplotlib':
            plt.close(result['fig'])


def test_animated_series_final_forecasts_use_the_same_timed_joint_model():
    frame = pd.DataFrame(np.random.default_rng(6).normal(size=(12, 2)),
                         index=np.arange(12.) * 3)
    expected = hyp.predict(frame, t=3)
    result = hyp.plot(frame, ndims=1, reduce=None, predict='Kalman', t=3,
                      animate=True, duration=1, frame_rate=4, return_model=True,
                      antialias=False, show=False, slow_warning_seconds=None)
    try:
        animation = result['animation']
        animation._func(3, *animation._args)
        traces = [line for line in result['fig'].axes[0].lines
                  if getattr(line, '_hyp_forecast_role', None) == 'live']
        assert len(traces) == 2
        for col, line in enumerate(traces):
            np.testing.assert_allclose(line.get_xdata()[1:], expected.index)
            np.testing.assert_allclose(line.get_ydata()[1:], expected.iloc[:, col])
    finally:
        plt.close(result['fig'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_datetime_horizon_uses_each_datasets_own_interval(backend):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    frames = [pd.DataFrame(np.arange(10.)[:, None],
                          index=pd.date_range('2026-01-01', periods=10, freq=f))
              for f in ['1h', '3h']]
    target = frames[1].index[-1] + pd.Timedelta(hours=6)
    expected = hyp.predict(frames, t=target)
    bundle = hyp.plot(frames, ndims=1, reduce=None, predict='Kalman', t=target,
                      return_model=True, backend=backend, show=False)
    for actual, reference in zip(bundle['predict']['forecasts'], expected):
        np.testing.assert_allclose(actual, reference)
    assert [len(f) for f in expected] == [24, 2]
    if backend == 'matplotlib':
        plt.close(bundle['fig'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_column_hierarchy_forecasts_use_the_original_times(backend):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    columns = pd.MultiIndex.from_product([['A', 'B'], ['a', 'b'], ['x', 'y']])
    index = pd.to_datetime('2026-01-01') + pd.to_timedelta(
        [0, 1, 3, 6, 7, 9, 13, 15, 17, 20, 23, 24], unit='h')
    data = pd.DataFrame(np.random.default_rng(15).normal(size=(12, 8)),
                        columns=columns, index=index)
    bundle = hyp.plot(data, ndims=2, reduce=None, predict='Kalman', t=3,
                      backend=backend, return_model=True, show=False)
    for observed, actual in zip(bundle['trace_data'], bundle['predict']['forecasts']):
        expected = hyp.predict(pd.DataFrame(observed, index=index), t=3)
        np.testing.assert_allclose(actual, expected)
    if backend == 'matplotlib':
        plt.close(bundle['fig'])


def test_explicit_step_also_places_truth_and_forecast_on_the_same_grid():
    data = pd.DataFrame(np.arange(12.)[:, None], index=np.arange(12.) * 2)
    result = hyp.plot(data, ndims=1, reduce=None,
                      predict={'model': 'Kalman', 'kwargs': {'step': 4}},
                      t=3, truth=np.array([12., 13., 14.]),
                      return_model=True, show=False, antialias=False)
    try:
        overlays = [next(line for line in result['fig'].axes[0].lines
                         if getattr(line, '_hyp_forecast_role', None) == role)
                    for role in ('static', 'truth')]
        for line in overlays:
            np.testing.assert_allclose(line.get_xdata(), [22, 26, 30, 34])
    finally:
        plt.close(result['fig'])


def test_arima_animation_waits_for_enough_interpolated_history():
    times = np.array([0, 1, 11, 21, 31, 41, 51, 61], dtype=float)
    frame = pd.DataFrame(np.sin(times)[:, None], index=times)
    bundle = hyp.plot(frame, ndims=1, reduce=None, animate=True,
                      predict={'model': 'ARIMA', 'kwargs': {'order': (4, 0, 0)}},
                      t=2, duration=1, frame_rate=8, return_model=True,
                      slow_warning_seconds=None, show=False)
    try:
        animation = bundle['animation']
        animation._func(7, *animation._args)
        live = [line for line in bundle['fig'].axes[0].lines
                if getattr(line, '_hyp_forecast_role', None) == 'live']
        assert len(live) == 1
        assert len(live[0].get_xdata()) > 0
    finally:
        plt.close(bundle['fig'])


@pytest.mark.parametrize('model', [Kalman(), GaussianProcess()])
def test_reuse_at_a_different_cadence_preserves_the_learned_interval(model):
    train = pd.DataFrame(np.sin(np.arange(24.) / 3), index=np.arange(24.))
    new = pd.DataFrame(np.cos(np.arange(12.) / 4), index=np.arange(12.) * 2)
    model.fit(train)
    actual = model.predict_new(new, 3)
    np.testing.assert_allclose(actual.index, [23, 24, 25])
    if isinstance(model, Kalman):
        grid = np.arange(23.)
        dense = pd.DataFrame(np.interp(grid, new.index, new[0]), index=grid)
        pd.testing.assert_frame_equal(actual, model.predict_new(dense, 3))
    else:
        learned = model.models_[0]['gp']
        reference = GaussianProcessRegressor(
            kernel=learned.kernel_, alpha=learned.alpha,
            normalize_y=learned.normalize_y, optimizer=None)
        reference.fit(np.asarray(new.index)[:, None], new)
        np.testing.assert_allclose(actual.to_numpy().ravel(),
                                   reference.predict(np.array([[23], [24], [25]])))


@pytest.mark.parametrize('index', [pd.timedelta_range('0h', periods=12, freq='2h'),
                                  pd.period_range('2026-01-01', periods=12, freq='D')])
def test_duration_and_period_indexes_have_equivalent_time_coordinates(index):
    data = pd.DataFrame(np.sin(np.arange(12.)), index=index)
    expected = hyp.predict(data.reset_index(drop=True), t=2)
    actual = hyp.predict(data, t=2)
    np.testing.assert_allclose(actual, expected)
    observed = index.to_timestamp() if isinstance(index, pd.PeriodIndex) else index
    assert actual.index[0] == observed[-1] + (observed[1] - observed[0])


@pytest.mark.parametrize('step', [0, -1, float('inf'), float('nan'), True])
def test_invalid_steps_fail_before_fitting(step):
    with pytest.raises(ValueError, match='step must'):
        hyp.predict(np.arange(12.), step=step, t=2)


@pytest.mark.parametrize('index', [pd.Index([0., 1., np.nan]),
                                  pd.Index([0., 1., np.inf]),
                                  pd.DatetimeIndex(['2026-01-01', '2026-01-02', pd.NaT])])
def test_missing_or_infinite_observation_times_are_rejected(index):
    with pytest.raises(ValueError, match='observation times must be finite'):
        hyp.predict(pd.DataFrame([1., 2., 3.], index=index), t=2)


def test_truth_preserves_its_explicit_observation_times():
    data = pd.DataFrame(np.arange(12.), index=np.arange(12.) * 2)
    truth = pd.DataFrame([12., 13., 14.], index=[23., 27., 29.])
    bundle = hyp.plot(data, ndims=1, reduce=None, predict='Kalman', t=3,
                      truth=truth, return_model=True, antialias=False, show=False)
    try:
        line = next(line for line in bundle['fig'].axes[0].lines
                    if getattr(line, '_hyp_forecast_role', None) == 'truth')
        np.testing.assert_allclose(line.get_xdata(), [22, 23, 27, 29])
    finally:
        plt.close(bundle['fig'])


@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_series_column_hierarchy_forecasts_values_at_actual_times(backend):
    if backend == 'plotly':
        pytest.importorskip('plotly')
    columns = pd.MultiIndex.from_product([['A', 'B'], ['x']])
    times = pd.date_range('2026-01-01', periods=12, freq='2h')
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(12, 2)),
                         index=times, columns=columns)
    bundle = hyp.plot(frame, ndims=1, reduce=None, predict='Kalman', t=3,
                      return_model=True, backend=backend, show=False)
    try:
        for observed, actual in zip(bundle['trace_data'], bundle['predict']['forecasts']):
            expected = hyp.predict(pd.DataFrame(observed[:, 1:], index=times), t=3)
            assert actual.shape == (3, 1)
            np.testing.assert_allclose(actual, expected)
    finally:
        if backend == 'matplotlib':
            plt.close(bundle['fig'])


def test_animation_cost_counts_joint_fits_once_for_multiple_columns():
    data = np.random.default_rng(4).normal(size=(20, 2))
    with pytest.warns(UserWarning, match='needs 4 forecast fits'):
        bundle = hyp.plot(data, ndims=1, reduce=None, predict='Kalman', t=3,
                          animate=True, duration=1, frame_rate=4,
                          slow_warning_seconds=0., show=False, return_model=True)
    plt.close(bundle['fig'])


class _CountedKalman(Kalman):
    """An ordinary Kalman forecaster recording its real fits for this test."""

    fitted_histories = []

    def fit(self, data):
        type(self).fitted_histories.append(data)
        return super().fit(data)


def test_repeated_animation_model_entries_each_fit_their_own_histories():
    _CountedKalman.fitted_histories = []
    frame = pd.DataFrame(np.random.default_rng(4).normal(size=(20, 2)))
    bundle = hyp.plot(frame, ndims=1, reduce=None,
                      predict=[_CountedKalman, _CountedKalman], t=3,
                      animate=True, duration=1, frame_rate=4,
                      slow_warning_seconds=None, show=False, return_model=True)
    try:
        # Two static full-history fits, then three visible histories per
        # model. Each two-column history must still be fitted jointly.
        assert len(_CountedKalman.fitted_histories) == 8
        assert all((d[0] if isinstance(d, list) else d).shape[1] == 2
                   for d in _CountedKalman.fitted_histories)
    finally:
        plt.close(bundle['fig'])


def test_animation_reuses_each_datasets_model_inside_a_dictionary_spec():
    frames = [pd.DataFrame(np.random.default_rng(i).normal(size=(12, 1)),
                           index=np.arange(12.) * step)
              for i, step in enumerate([1, 3])]
    _, fitted = hyp.predict(frames, t=2, return_model=True)
    expected = hyp.predict(frames, model=fitted, t=2)
    bundle = hyp.plot(frames, ndims=1, reduce=None, predict={'model': fitted},
                      animate=True, t=2, duration=1, frame_rate=3,
                      slow_warning_seconds=None, antialias=False,
                      return_model=True, show=False)
    try:
        animation = bundle['animation']
        animation._func(2, *animation._args)
        lines = [line for line in bundle['fig'].axes[0].lines
                 if getattr(line, '_hyp_forecast_role', None) == 'live']
        assert len(lines) == 2
        for line, reference in zip(lines, expected):
            np.testing.assert_allclose(line.get_xdata()[1:], reference.index)
            np.testing.assert_allclose(line.get_ydata()[1:], reference.iloc[:, 0])
    finally:
        plt.close(bundle['fig'])
