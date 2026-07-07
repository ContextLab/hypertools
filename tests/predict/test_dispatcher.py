import numpy as np
import pandas as pd
import pytest

from hypertools.predict.predict import predict, FORECASTERS
from hypertools.predict.gp import GaussianProcess


def _make_df(n=60, ncols=2, index=None):
    t = np.arange(n)
    trend = 0.05 * t
    sine = np.sin(t / 5.0)
    cols = {'a': trend + sine, 'b': trend - sine}
    df = pd.DataFrame({k: cols[k] for k in list(cols)[:ncols]})
    if index is not None:
        df.index = index
    return df


# --- every registered forecaster name resolves (extras skip-gated) --------

@pytest.mark.parametrize('name,extra', [
    ('Kalman', 'pykalman'),
    ('GaussianProcess', None),
    ('AutoRegressor', None),
    ('ARIMA', 'statsmodels'),
    ('Laplace', 'skaters'),
    ('Chronos', 'chronos'),
])
def test_all_forecaster_names_resolve(name, extra):
    if extra is not None:
        pytest.importorskip(extra)
    df = _make_df(n=60)
    out = predict(df, model=name, t=4)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (4, 2)


def test_forecaster_names_match_registry():
    assert {f.__name__ for f in FORECASTERS} == {
        'Kalman', 'GaussianProcess', 'AutoRegressor', 'ARIMA', 'Laplace', 'Chronos'}


# --- dict (both forms) / class / instance resolution -----------------------

def test_dict_params_form():
    df = _make_df(n=60)
    with pytest.warns(DeprecationWarning, match="'params'"):
        out = predict(df, model={'model': 'GaussianProcess', 'params': {'alpha': 1e-6}}, t=4)
    assert out.shape == (4, 2)


def test_dict_args_kwargs_form():
    df = _make_df(n=60)
    out = predict(df, model={'model': 'GaussianProcess', 'args': [], 'kwargs': {'alpha': 1e-6}}, t=4)
    assert out.shape == (4, 2)


def test_class_form():
    df = _make_df(n=60)
    out = predict(df, model=GaussianProcess, t=4, alpha=1e-6)
    assert out.shape == (4, 2)


def test_instance_form():
    df = _make_df(n=60)
    out = predict(df, model=GaussianProcess(alpha=1e-6), t=4)
    assert out.shape == (4, 2)


# --- t: int and datetime horizons ------------------------------------------

def test_t_int_horizon():
    df = _make_df(n=60)
    out = predict(df, model='GaussianProcess', t=5)
    assert list(out.index) == list(range(60, 65))


def test_t_datetime_horizon():
    idx = pd.date_range('2026-01-01', periods=60, freq='D')
    df = _make_df(n=60, index=idx)
    target = idx[-1] + pd.Timedelta(days=4)
    out = predict(df, model='GaussianProcess', t=target)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert len(out) == 4


# --- list-in / list-out ------------------------------------------------------

def test_list_in_list_out():
    dfs = [_make_df(n=50), _make_df(n=70)]
    out = predict(dfs, model='GaussianProcess', t=3)
    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (3, 2)
        assert list(fc.index) == list(range(len(src), len(src) + 3))


# --- unknown model name --------------------------------------------------

def test_unknown_model_name_lists_options():
    df = _make_df(n=60)
    with pytest.raises(ValueError) as exc_info:
        predict(df, model='NotARealForecaster', t=4)
    message = str(exc_info.value)
    assert 'NotARealForecaster' in message
    for name in ('Kalman', 'GaussianProcess', 'AutoRegressor', 'ARIMA', 'Laplace', 'Chronos'):
        assert name in message


# --- return_model round trip: no re-estimation on new data -----------------

def test_return_model_roundtrip_kalman_no_reestimation(monkeypatch):
    pytest.importorskip('pykalman')
    from pykalman import KalmanFilter

    a = _make_df(n=70)
    fc_a, fitted = predict(a, model='Kalman', t=5, return_model=True, n_iter=5)
    assert isinstance(fc_a, pd.DataFrame)
    original_kf = fitted.models_[0]['kf']

    def _boom(self, *args, **kwargs):
        raise AssertionError('em() must not be called during predict_new (no re-estimation)')

    monkeypatch.setattr(KalmanFilter, 'em', _boom)

    b = _make_df(n=40)
    fc_b = predict(b, model=fitted, t=5)

    assert isinstance(fc_b, pd.DataFrame)
    assert fc_b.shape == (5, b.shape[1])
    assert list(fc_b.index) == list(range(40, 45))
    # learned parameters are the SAME object -- never rebuilt/re-estimated
    assert fitted.models_[0]['kf'] is original_kf


def test_return_model_roundtrip_gp_no_reestimation(monkeypatch):
    from sklearn.gaussian_process import GaussianProcessRegressor

    a = _make_df(n=70)
    fc_a, fitted = predict(a, model='GaussianProcess', t=5, return_model=True)
    original_gp = fitted.models_[0]['gp']

    def _boom(self, *args, **kwargs):
        raise AssertionError('fit() must not be called during predict_new (no re-estimation)')

    monkeypatch.setattr(GaussianProcessRegressor, 'fit', _boom)

    b = _make_df(n=40)
    fc_b = predict(b, model=fitted, t=5)

    assert fc_b.shape == (5, b.shape[1])
    assert list(fc_b.index) == list(range(40, 45))
    assert fitted.models_[0]['gp'] is original_gp


def test_return_model_roundtrip_autoregressor_no_reestimation(monkeypatch):
    from sklearn.linear_model import Ridge

    a = _make_df(n=70)
    fc_a, fitted = predict(a, model='AutoRegressor', t=5, return_model=True, lags=10)
    original_estimator = fitted.models_[0]['estimator']

    def _boom(self, *args, **kwargs):
        raise AssertionError('fit() must not be called during predict_new (no re-estimation)')

    monkeypatch.setattr(Ridge, 'fit', _boom)

    b = _make_df(n=40)
    fc_b = predict(b, model=fitted, t=5)

    assert fc_b.shape == (5, b.shape[1])
    assert list(fc_b.index) == list(range(40, 45))
    assert fitted.models_[0]['estimator'] is original_estimator
