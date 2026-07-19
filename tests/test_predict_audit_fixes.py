"""Regression tests for the 2026-07 release-audit fixes to hyp.predict
(unit F16-predict plus the predict parts of X2-error-quality-002/-004).

Every test uses real data and real forecaster runs (no mocks); each mirrors
a repro that was CONFIRMED failing on the pre-fix code by the independent
audit verifiers (see notes/audit-1.0-2026-07/verdicts/F16-predict.json).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.predict import predict
from hypertools.predict.common import Forecaster, resolve_t


def _sine(n=200, period=40.0, noise=0.1, seed=42):
    full = np.sin(2 * np.pi * np.arange(n + 30) / period)
    rng = np.random.default_rng(seed)
    hist = (full[:n] + noise * rng.standard_normal(n)).reshape(-1, 1)
    return hist, full[n:n + 30]


# --- F16-predict-001: default Kalman actually learns dynamics --------------

def test_kalman_learns_dynamics_sine_forecast_tracks_truth():
    pytest.importorskip('pykalman')
    hist, truth = _sine()
    fc, fitted = predict(hist, model='Kalman', t=30, return_model=True)

    values = np.asarray(fc).ravel()
    assert fc.shape == (30, 1)
    # the pre-fix model returned a constant flat line (std exactly 0)
    assert float(np.std(values)) > 0.0
    assert np.corrcoef(values, truth)[0, 1] > 0.5
    # the transition matrix is genuinely estimated (pre-fix: identity)
    A = np.asarray(fitted.models_[0]['kf'].transition_matrices)
    assert not np.allclose(A, np.eye(A.shape[0]))


def test_kalman_default_tracks_linear_trend():
    pytest.importorskip('pykalman')
    rng = np.random.default_rng(0)
    hist = (0.05 * np.arange(200) + 0.1 * rng.standard_normal(200)).reshape(-1, 1)
    truth = 0.05 * np.arange(200, 230)
    fc = predict(hist, model='Kalman', t=30)
    assert np.corrcoef(np.asarray(fc).ravel(), truth)[0, 1] > 0.5


# --- F16-predict-002: 1-D input is a univariate series ---------------------

def test_1d_array_is_univariate_series():
    pytest.importorskip('pykalman')
    hist, truth = _sine(noise=0.0)
    fc = predict(hist.ravel(), t=30)  # default model, (200,) input
    assert np.asarray(fc).shape == (30, 1)
    assert np.corrcoef(np.asarray(fc).ravel(), truth)[0, 1] > 0.5


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_flat_scalar_list_is_one_series_not_many_datasets():
    values = [float(v) for v in np.sin(np.arange(60) / 5.0)]
    fc = predict(values, model='GaussianProcess', t=3)
    # pre-fix: a LIST of 60 constant single-point "forecasts"
    assert isinstance(fc, pd.DataFrame)
    assert np.asarray(fc).shape == (3, 1)


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_1d_datasets_inside_a_list_are_columns():
    out = predict([np.sin(np.arange(60) / 5.0), np.cos(np.arange(80) / 5.0)],
                  model='GaussianProcess', t=3)
    assert isinstance(out, list)
    assert [np.asarray(o).shape for o in out] == [(3, 1), (3, 1)]


# --- F16-predict-003: pandas Series input -----------------------------------

# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_series_input_forecasts_instead_of_silently_vanishing():
    s = pd.Series(np.arange(50, dtype=float))
    fc = predict(s, model='GaussianProcess', t=3)
    assert np.asarray(fc).shape == (3, 1)


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_series_with_datetimeindex_keeps_time_semantics():
    idx = pd.date_range('2024-01-01', periods=40, freq='D')
    s = pd.Series(np.arange(40, dtype=float), index=idx)
    fc = predict(s, model='GaussianProcess', t=2)
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert fc.index[0] == idx[-1] + pd.Timedelta(days=1)


# --- F16-predict-004: datetime horizon at/near the last observation --------

# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
# (only the GaussianProcess parameter case emits it)
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('model,extra', [('GaussianProcess', None),
                                         ('Kalman', 'pykalman'),
                                         ('ARIMA', 'statsmodels')])
def test_datetime_t_at_last_timestamp_truncates_consistently(model, extra):
    if extra is not None:
        pytest.importorskip(extra)
    idx = pd.date_range('2024-01-01', periods=40, freq='D')
    df = pd.DataFrame({'a': np.sin(np.arange(40) / 5.0)}, index=idx)
    out = predict(df, model=model, t=str(idx[-1]))
    # t == the last observation: the full history up to t, never NaN
    pd.testing.assert_frame_equal(out, df)


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_datetime_t_half_step_ahead_forecasts_one_step():
    idx = pd.date_range('2024-01-01', periods=40, freq='D')
    df = pd.DataFrame({'a': np.arange(40, dtype=float)}, index=idx)
    out = predict(df, model='GaussianProcess', t=str(idx[-1] + pd.Timedelta(hours=11)))
    assert out.shape == (1, 1)
    assert out.index[0] == idx[-1] + pd.Timedelta(days=1)


# --- F16-predict-005 (doc fix): defaults' suitability documented ------------

def test_arima_and_laplace_document_default_suitability():
    # NOTE: `import hypertools.predict.arima as ...` would resolve through
    # the shadowing `predict` FUNCTION attribute (F16-predict-017, out of
    # scope here); the from-import resolves via sys.modules.
    from hypertools.predict import arima as arima_mod
    from hypertools.predict import laplace as laplace_mod
    assert 'drift/random-walk' in arima_mod.__doc__
    assert 'drifting/trending' in laplace_mod.__doc__ or 'drift' in laplace_mod.__doc__


def test_arima_custom_order_tracks_sine():
    pytest.importorskip('statsmodels')
    hist, truth = _sine()
    fc = predict(hist, model={'model': 'ARIMA', 'kwargs': {'order': (4, 0, 0)}}, t=30)
    assert np.corrcoef(np.asarray(fc).ravel(), truth)[0, 1] > 0.5


# --- F16-predict-006: explicit model_kwargs= --------------------------------

def test_autoregressor_explicit_model_kwargs_accepted():
    from hypertools.predict import AutoRegressor
    X = pd.DataFrame(np.sin(np.arange(50) / 5.0).reshape(-1, 1))
    out = AutoRegressor(model='SVR', model_kwargs={'C': 2.0}).fit_predict(X, 3)
    assert out.shape == (3, 1)
    # direct kwargs and the explicit dict are merged (direct kwargs win)
    ar = AutoRegressor(model='SVR', model_kwargs={'C': 2.0}, gamma='auto')
    assert ar.model_kwargs == {'C': 2.0, 'gamma': 'auto'}


# --- F16-predict-008: real ValueErrors, not assert-based --------------------

def test_autoregressor_too_few_observations_raises_valueerror():
    with pytest.raises(ValueError, match='more than lags'):
        predict(np.random.RandomState(0).randn(3, 2), model='AutoRegressor', t=2)


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_all_duplicate_timestamps_raise_valueerror():
    df = pd.DataFrame({'a': np.ones(5)},
                      index=pd.DatetimeIndex(['2024-01-01'] * 5))
    with pytest.raises(ValueError, match='share one timestamp'):
        predict(df, model='GaussianProcess', t=3)


# --- F16-predict-009 / X2-error-quality-003: typo'd kwargs rejected ---------

def test_kalman_typo_kwarg_raises_typeerror():
    pytest.importorskip('pykalman')
    with pytest.raises(TypeError, match='n_itr'):
        predict(np.random.RandomState(0).randn(60, 2), model='Kalman', t=2, n_itr=99)


def test_gp_typo_kwarg_raises_typeerror():
    with pytest.raises(TypeError, match='kernal'):
        predict(np.random.RandomState(0).randn(60, 2), model='GaussianProcess',
                t=2, kernal='rbf')


def test_laplace_unknown_kwarg_raises_typeerror():
    from hypertools.predict import Laplace
    with pytest.raises(TypeError):
        Laplace(bogus_flag=True)


def test_chronos_sampling_controls_forwarded():
    pytest.importorskip('chronos')
    x = np.sin(np.arange(60) / 6.0).reshape(-1, 1)
    fc = predict(x, model='Chronos', t=3, num_samples=3)
    assert np.asarray(fc).shape == (3, 1)
    with pytest.raises(TypeError, match='num_sampels'):
        predict(x, model='Chronos', t=3, num_sampels=3)


# --- F16-predict-010: fork-style dict spec keeps outer kwargs ---------------

def test_fork_dict_spec_merges_outer_kwargs():
    pytest.importorskip('pykalman')
    X = np.random.RandomState(0).randn(40, 2)
    _, m1 = predict(X, model={'model': 'Kalman', 'kwargs': {}}, t=2, n_iter=1,
                    return_model=True)
    assert m1.n_iter == 1  # pre-fix: silently reset to the default (5)


# --- F16-predict-011: t=None gets a clear error ------------------------------

def test_t_none_clear_error_rangeindex_and_datetimeindex():
    with pytest.raises(ValueError, match='got None'):
        predict(np.random.RandomState(0).randn(30, 2), model='GaussianProcess', t=None)
    df = pd.DataFrame({'a': np.random.RandomState(0).randn(30)},
                      index=pd.date_range('2024-01-01', periods=30))
    with pytest.raises(ValueError, match='got None'):
        predict(df, model='GaussianProcess', t=None)


# --- F16-predict-012: feature-count mismatch on reuse ------------------------

def test_predict_new_feature_count_mismatch_clear_error():
    pytest.importorskip('pykalman')
    _, m = predict(np.random.RandomState(0).randn(40, 2), model='Kalman', t=3,
                   n_iter=2, return_model=True)
    with pytest.raises(ValueError, match='expects 2 feature'):
        predict(np.random.RandomState(1).randn(40, 5), model=m, t=3)


# --- F16-predict-013 / X2-error-quality-002/-004: degenerate inputs ---------

def test_empty_dataframe_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        predict(pd.DataFrame(np.empty((0, 2))), model='GaussianProcess', t=3)


def test_empty_list_clear_error_never_reaches_text_pipeline():
    with pytest.raises(ValueError, match='empty'):
        predict([], model='GaussianProcess', t=3)


def test_empty_array_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        predict(np.array([]), t=5)


def test_scalar_input_clear_error():
    with pytest.raises(ValueError, match='scalar'):
        predict(5, t=2)


def test_none_input_clear_error():
    with pytest.raises(TypeError, match='Unsupported data type'):
        predict(None)


def test_single_row_clear_error():
    with pytest.raises(ValueError, match='single observation'):
        predict(np.array([[1.0, 2.0]]), model='GaussianProcess', t=3)


# --- F16-predict-014: clear NaN policy ---------------------------------------

def _nan_data():
    X = np.column_stack([np.sin(np.arange(100) / 6), np.cos(np.arange(100) / 6)])
    X[10:14, 0] = np.nan
    return X


def test_gp_and_autoregressor_nan_error_names_impute():
    for model in ('GaussianProcess', 'AutoRegressor'):
        with pytest.raises(ValueError, match='hyp.impute'):
            predict(_nan_data(), model=model, t=3)


def test_laplace_nan_error_is_clear_valueerror():
    pytest.importorskip('skaters')
    with pytest.raises(ValueError, match='NaN'):
        predict(_nan_data(), model='Laplace', t=3)


def test_kalman_and_arima_still_tolerate_nan():
    pytest.importorskip('pykalman')
    pytest.importorskip('statsmodels')
    for model in ('Kalman', 'ARIMA'):
        out = predict(_nan_data(), model=model, t=3)
        assert np.asarray(out).shape == (3, 2)
        assert np.isfinite(np.asarray(out)).all()


# --- F16-predict-016: descending DatetimeIndex warns -------------------------

def test_descending_datetimeindex_warns():
    pytest.importorskip('statsmodels')
    idx = pd.date_range('2024-01-01', periods=60, freq='D')[::-1]
    df = pd.DataFrame({'p': 100 + 0.5 * np.arange(60)}, index=idx)
    with pytest.warns(UserWarning, match='not sorted'):
        predict(df, model='ARIMA', t=5)


# --- F16-predict-018: rich unknown-model errors for every spec type ---------

@pytest.mark.parametrize('bad_model', [42, None])
def test_unknown_model_nonstring_lists_options(bad_model):
    with pytest.raises(ValueError) as exc_info:
        predict(np.random.RandomState(0).randn(30, 2), model=bad_model, t=3)
    message = str(exc_info.value)
    assert 'Kalman' in message and 'Forecaster' in message


def test_raw_sklearn_regressor_class_hints_autoregressor():
    from sklearn.linear_model import Ridge
    with pytest.raises(ValueError, match='AutoRegressor'):
        predict(np.random.RandomState(0).randn(30, 2), model=Ridge, t=3)


def test_dict_spec_missing_model_key_clear_error():
    with pytest.raises(ValueError, match="'model' key"):
        predict(np.random.RandomState(0).randn(30, 2), model={'params': {}}, t=3)


def test_instance_spec_with_kwargs_warns():
    from hypertools.predict.gp import GaussianProcess
    with pytest.warns(UserWarning, match='ignoring keyword'):
        predict(np.random.RandomState(0).randn(30, 2),
                model=GaussianProcess(), t=2, alpha=1.0)


# --- F16-predict-019: dead args removed (behavioral: dict spec still works) --

def test_autoregressor_inner_estimator_via_dict_spec():
    fc = predict(np.sin(np.arange(60) / 5.0).reshape(-1, 1),
                 model={'model': 'AutoRegressor',
                        'kwargs': {'model': 'Ridge', 'lags': 5}}, t=3)
    assert np.asarray(fc).shape == (3, 1)


# --- F16-predict-020: tz-aware index with naive t ----------------------------

# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures,
# and its lbfgs optimizer intermittently stops at the 20-iteration cap on
# them too -- both are contrived-fixture noise, not hypertools behavior
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
@pytest.mark.filterwarnings(
    'ignore:lbfgs failed to converge'
    ':sklearn.exceptions.ConvergenceWarning')
def test_tz_aware_index_localizes_naive_t():
    idx = pd.date_range('2024-01-01', periods=100, freq='D', tz='US/Eastern')
    df = pd.DataFrame({'a': np.arange(100, dtype=float)}, index=idx)
    out = predict(df, model='GaussianProcess', t='2024-04-20')
    assert len(out) > 0
    assert out.index.tz is not None


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_tz_aware_t_on_naive_index_clear_error():
    df = pd.DataFrame({'a': np.arange(50, dtype=float)},
                      index=pd.date_range('2024-01-01', periods=50, freq='D'))
    with pytest.raises(ValueError, match='timezone'):
        predict(df, model='GaussianProcess',
                t=pd.Timestamp('2024-03-01', tz='US/Eastern'))
