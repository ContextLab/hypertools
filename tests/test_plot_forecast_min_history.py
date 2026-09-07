"""Forecast overlays under `hyp.plot` -- 1.1 release-review fixes.

F1  ``predict='ARIMA'`` with a time-progressing ``animate=`` crashed with a
    raw ``IndexError`` out of statsmodels: the per-frame schedule fit the
    2-row history the earliest frames reveal, and an ARIMA(1, 1, 1) needs 3.
    Forecasters now carry a ``min_history`` (ARIMA's from its order), `fit`
    refuses a shorter history with a ``ValueError`` naming the model and the
    rows it needs, and the animated schedule draws no forecast until enough
    history is revealed.
F2  A datetime-like ``t=`` never worked inside `plot()` (the forecaster was
    handed bare arrays), although ``hyp.predict(df, t=Timestamp)`` did.
F3  A ``predict=`` collection on a hierarchical (MultiIndex) frame raised an
    internal "hierarchy trace/bundle_forecasts mismatch" error.
X1  `Forecaster.fit` returns the instance (sklearn chaining).
X2  A finite input that a trailing ``Smooth(center=False)`` left NaN at the
    head was reported as "rows had ALL features missing" -- the input's
    fault, not the pipeline's.

Every check reads real observables: exception types and messages, drawn
artists, returned bundles. No mocks, no monkeypatching.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import pytest                                             # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.dates import date2num                     # noqa: E402

import hypertools as hyp                                  # noqa: E402
from hypertools.predict.arima import ARIMA                # noqa: E402
from hypertools.predict.autoreg import AutoRegressor      # noqa: E402
from hypertools.predict.kalman import Kalman              # noqa: E402
from hypertools.predict.common import Forecaster          # noqa: E402
from hypertools.plot.forecast import (                    # noqa: E402
    DEFAULT_MIN_HISTORY, ForecastSchedule, model_min_history)


def _walk(rows=30, cols=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


def _by_role(fig, role):
    return [line for line in fig.axes[0].lines
            if getattr(line, '_hyp_forecast_role', None) == role]


def _draw_every_frame(anim):
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
    return anim.n_frames


# --- F1: min_history ----------------------------------------------------

def test_arima_min_history_follows_its_order():
    assert Forecaster.min_history == DEFAULT_MIN_HISTORY == 2
    assert ARIMA.min_history_for() == 3                    # (1, 1, 1)
    assert ARIMA().min_history == 3
    assert ARIMA(order=(4, 0, 0)).min_history == 5
    assert ARIMA(order=(2, 1, 2)).min_history == 5
    assert ARIMA(order=(0, 1, 0)).min_history == 3
    assert Kalman().min_history == 2
    assert model_min_history('ARIMA') == 3
    assert model_min_history('Kalman') == 2
    assert model_min_history({'model': 'ARIMA',
                              'kwargs': {'order': (4, 0, 0)}}) == 5
    assert model_min_history(ARIMA(order=(3, 2, 1))) == 5


@pytest.mark.parametrize('order, rows', [((1, 1, 1), 3), ((4, 0, 0), 5),
                                         ((2, 1, 2), 5), ((3, 2, 1), 5)])
def test_arima_fits_at_its_floor_and_statsmodels_agrees(order, rows):
    """The floor is what statsmodels can actually fit: `rows` rows fit and
    forecast finite values, every shorter history the base check catches
    first raises hypertools' own ValueError (never statsmodels' IndexError)."""
    x = _walk(rows=rows, cols=1, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.predict(x, model={'model': 'ARIMA',
                                    'kwargs': {'order': order}}, t=2)
    assert out.shape == (2, 1) and np.isfinite(out.to_numpy()).all()
    for short in range(2, rows):
        with pytest.raises(ValueError, match='ARIMA'):
            hyp.predict(x[:short], model={'model': 'ARIMA',
                                          'kwargs': {'order': order}}, t=2)


def test_predict_on_a_two_row_history_names_arima_and_three_rows():
    x = _walk()
    with pytest.raises(ValueError) as info:
        hyp.predict(x[:2], model='ARIMA', t=2)
    message = str(info.value)
    assert 'ARIMA' in message and '3 observation' in message
    assert 'order=(1, 1, 1)' in message
    assert '2 row' in message


def test_forecaster_fit_rejects_short_history_before_the_fitter_runs():
    x = _walk(rows=4)
    model = ARIMA(order=(2, 1, 2))              # needs 5 rows
    with pytest.raises(ValueError, match=r'ARIMA\(order=\(2, 1, 2\)\)'):
        model.fit(x)
    assert not model.is_fitted
    model.fit(_walk(rows=5, seed=4))
    assert model.is_fitted


def test_animated_arima_renders_every_frame():
    """The F1 repro: the same call used to raise IndexError out of the
    per-frame schedule before the first frame existed."""
    x = _walk()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        anim = hyp.plot(x, reduce=None, ndims=2, predict='ARIMA', t=3,
                        animate=True, duration=1, show=False)
    try:
        assert _draw_every_frame(anim) > 1
    finally:
        plt.close(anim.figure)


def test_schedule_uses_the_model_floor_and_skips_short_histories():
    x = _walk(rows=8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schedule = ForecastSchedule(
            [x], counts=[[k] for k in range(1, 9)], model='ARIMA', t=2)
    assert schedule.min_history == 3
    assert schedule.path(0, 0) is None            # 1 row revealed
    assert schedule.path(0, 1) is None            # 2 rows: below ARIMA's floor
    assert schedule.path(0, 2) is not None        # 3 rows: fit
    assert schedule.path(0, 2).shape == (3, 2)


def test_animated_kalman_and_arima_collection_renders():
    x, y = _walk(seed=1), _walk(rows=25, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        anim = hyp.plot([x, y], reduce=None, ndims=2,
                        predict=['Kalman', 'ARIMA'], t=3, animate=True,
                        duration=1, show=False)
    try:
        assert _draw_every_frame(anim) > 1
    finally:
        plt.close(anim.figure)


# --- X1: fit returns the instance ---------------------------------------

@pytest.mark.parametrize('cls', [Kalman, AutoRegressor, ARIMA])
def test_fit_returns_the_same_instance_for_chaining(cls):
    x = _walk(seed=5)
    model = cls()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = model.fit(x)
        assert fitted is model
        chained = cls().fit(x).predict(2)
    assert chained.shape == (2, 2)


# --- F2: datetime-like t= inside plot() ---------------------------------

@pytest.fixture
def dated():
    index = pd.date_range('2020-01-01', periods=30, freq='D')
    frame = pd.DataFrame(_walk(cols=1, seed=7), index=index, columns=['val'])
    return frame, index


def test_datetime_t_forecasts_up_to_that_date_in_series_mode(dated):
    frame, index = dated
    target = pd.Timestamp('2020-02-05')                  # 6 days past the end
    fig = hyp.plot(frame, ndims=1, predict='Kalman', t=target,
                   antialias=False, show=False)
    try:
        (overlay,) = _by_role(fig, 'static')
        xs = np.asarray(overlay.get_xdata(), dtype=float)
        assert len(xs) == 7                              # seam + 6 steps
        assert xs[0] == pytest.approx(date2num(index[-1].to_pydatetime()))
        assert xs[-1] == pytest.approx(date2num(target.to_pydatetime()))
    finally:
        plt.close(fig)


def test_datetime_t_matches_hyp_predict_step_count_with_return_model(dated):
    frame, _ = dated
    target = '2020-02-03'
    bundle = hyp.plot(frame, ndims=1, predict='Kalman', t=target,
                      return_model=True, show=False)
    try:
        expected = hyp.predict(frame, model='Kalman', t=target)
        assert len(expected) == 4
        (forecast,) = bundle['predict']['forecasts']
        assert forecast.shape == (4, 1)
    finally:
        plt.close(bundle['fig'])


def test_datetime_t_on_two_dated_datasets_static_and_animated(dated):
    frame, _ = dated
    other = frame * 2.0 + 5.0
    target = pd.Timestamp('2020-02-02')
    fig = hyp.plot([frame, other], ndims=1, predict='Kalman', t=target,
                   antialias=False, show=False)
    try:
        overlays = _by_role(fig, 'static')
        assert len(overlays) == 2
        for line in overlays:
            assert np.asarray(line.get_xdata())[-1] == pytest.approx(
                date2num(target.to_pydatetime()))
    finally:
        plt.close(fig)
    anim = hyp.plot([frame, other], ndims=1, predict='Kalman', t=target,
                    animate=True, duration=1, show=False)
    try:
        assert _draw_every_frame(anim) > 1
    finally:
        plt.close(anim.figure)


def test_datetime_t_in_two_dimensions(dated):
    frame, index = dated
    wide = pd.DataFrame(_walk(cols=2, seed=8), index=index,
                        columns=['a', 'b'])
    fig = hyp.plot(wide, reduce=None, ndims=2, predict='Kalman',
                   t=pd.Timestamp('2020-02-04'), antialias=False, show=False)
    try:
        (overlay,) = _by_role(fig, 'static')
        assert len(np.asarray(overlay.get_xdata())) == 6      # seam + 5
    finally:
        plt.close(fig)


def test_datetime_t_without_a_datetime_index_is_refused(dated):
    frame, _ = dated
    with pytest.raises(ValueError, match='DatetimeIndex'):
        hyp.plot(frame.to_numpy(), ndims=1, predict='Kalman',
                 t=pd.Timestamp('2020-02-05'), show=False)
    with pytest.raises(ValueError, match='at or before'):
        hyp.plot(frame, ndims=1, predict='Kalman',
                 t=pd.Timestamp('2020-01-20'), show=False)
    shifted = frame.copy()
    shifted.index = shifted.index + pd.Timedelta(days=3)
    with pytest.raises(ValueError, match='different number of steps'):
        hyp.plot([frame, shifted], ndims=1, predict='Kalman',
                 t=pd.Timestamp('2020-02-05'), show=False)


# --- F3: a predict= collection on a hierarchy ---------------------------

MODELS = ['Kalman', 'ARIMA']
T = 3


@pytest.fixture
def column_hierarchy():
    columns = pd.MultiIndex.from_product([['g1', 'g2'], ['a', 'b', 'c']])
    return pd.DataFrame(_walk(cols=6, seed=9), columns=columns)


@pytest.fixture
def row_hierarchy():
    rows = pd.MultiIndex.from_arrays(
        [['r1'] * 15 + ['r2'] * 15, ['s'] * 30], names=['grp', 'sub'])
    return pd.DataFrame(_walk(cols=3, seed=10), index=rows,
                        columns=['a', 'b', 'c'])


@pytest.mark.parametrize('frame_fixture', ['column_hierarchy',
                                           'row_hierarchy'])
def test_hierarchy_gets_one_forecast_per_trace_per_model(frame_fixture,
                                                         request):
    frame = request.getfixturevalue(frame_fixture)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = hyp.plot(frame, predict=MODELS, t=T, return_model=True,
                          show=False)
    try:
        traces = bundle['trace_data']
        n_traces = len(traces)
        assert n_traces >= 2                     # every leaf (and any mean)
        forecasts = bundle['predict']['forecasts']
        assert list(forecasts) == MODELS         # keyed like the flat case
        for name in MODELS:
            assert len(forecasts[name]) == n_traces
            for forecast, trace in zip(forecasts[name], traces):
                assert forecast.shape == (T, np.asarray(trace).shape[1])
        overlays = _by_role(bundle['fig'], 'static')
        assert len(overlays) == n_traces * len(MODELS)
        assert bundle['predict']['drawn'] is True
    finally:
        plt.close(bundle['fig'])


@pytest.mark.parametrize('frame_fixture', ['column_hierarchy',
                                           'row_hierarchy'])
def test_hierarchy_collection_animates(frame_fixture, request):
    frame = request.getfixturevalue(frame_fixture)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        anim = hyp.plot(frame, predict=MODELS, t=T, animate=True,
                        duration=1, show=False)
    try:
        assert _draw_every_frame(anim) > 1
    finally:
        plt.close(anim.figure)


def test_hierarchy_mapping_form_keeps_the_callers_names(column_hierarchy):
    specs = {'kal': 'Kalman', 'ar': 'ARIMA'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = hyp.plot(column_hierarchy, predict=specs, t=T,
                          return_model=True, show=False)
    try:
        assert list(bundle['predict']['forecasts']) == ['kal', 'ar']
    finally:
        plt.close(bundle['fig'])


# --- X2: NaN introduced by a trailing smoother --------------------------

def test_nan_introduced_by_a_trailing_smoother_is_blamed_on_the_manip():
    from hypertools.manip.smooth import Smooth
    x = _walk(seed=12)
    assert np.isfinite(x).all()
    with pytest.raises(ValueError) as info:
        hyp.plot(x, reduce=None, ndims=2,
                 manip=Smooth(kernel='boxcar', kernel_width=12, center=False),
                 show=False)
    message = str(info.value)
    assert 'finite on input' in message
    assert 'manip= stage' in message and 'Smooth' in message
    assert 'min_periods=1' in message
    assert 'ALL features missing' not in message
    fig = hyp.plot(x, reduce=None, ndims=2,
                   manip=Smooth(kernel='boxcar', kernel_width=12,
                                center=False, min_periods=1),
                   antialias=False, show=False)
    try:
        (line,) = [ln for ln in fig.axes[0].lines
                   if getattr(ln, '_hyp_forecast_role', None) is None]
        assert np.isfinite(np.asarray(line.get_xydata())).all()
    finally:
        plt.close(fig)


def test_nan_in_the_input_keeps_the_all_features_missing_message():
    x = _walk(seed=13)
    x[5] = np.nan
    with pytest.raises(ValueError, match='ALL features missing'):
        hyp.plot(x, reduce=None, ndims=2, show=False)


def test_arima_min_history_accepts_statsmodels_sparse_lag_orders():
    """`order=([1, 3], 0, 0)` is statsmodels' sparse AR form (include lags
    1 and 3); the fitter accepts it, so the minimum-history check must
    too, counting the highest lag (release review, round 2)."""
    import numpy as np
    import pandas as pd
    from hypertools.predict.arima import ARIMA
    assert ARIMA.min_history_for(order=([1, 3], 0, 0)) == 4
    assert ARIMA.min_history_for(order=(2, 1, [1, 2])) == 5
    assert ARIMA.min_history_for(order=([], 0, 0)) == 2
    rng = np.random.default_rng(0)
    series = pd.DataFrame(np.cumsum(rng.normal(size=(60, 1)), axis=0))
    forecast = hyp.predict(series, model={'model': 'ARIMA',
                                          'kwargs': {'order': ([1, 3], 0, 0)}},
                           t=2)
    assert forecast.shape == (2, 1)
    assert np.isfinite(forecast.to_numpy()).all()


def test_a_fitted_forecaster_reuses_its_parameters_on_a_short_context():
    """The minimum history is what a FIT needs. A fitted ARIMA(4, 0, 0)
    applied to two new rows conditions on those rows with its learned
    parameters (statsmodels does exactly that), so reuse must not be held
    to the five-row fit floor (release review, round 2)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(1)
    series = pd.DataFrame(np.cumsum(rng.normal(size=(40, 1)), axis=0))
    _, fitted = hyp.predict(series, model={'model': 'ARIMA',
                                           'kwargs': {'order': (4, 0, 0)}},
                            t=2, return_model=True)
    again = hyp.predict(series.iloc[:2], model=fitted, t=2)
    assert again.shape == (2, 1)
    assert np.isfinite(again.to_numpy()).all()
