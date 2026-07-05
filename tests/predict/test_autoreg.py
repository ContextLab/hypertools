import numpy as np
import pandas as pd
import pytest

from hypertools.predict.autoreg import AutoRegressor


def _make_df(n=70, index=None):
    t = np.arange(n)
    trend = 0.05 * t
    sine = np.sin(t / 5.0)
    df = pd.DataFrame({'a': trend + sine, 'b': trend - sine, 'c': sine * 2})
    if index is not None:
        df.index = index
    return df


def test_forecast_shape_and_rangeindex_continuation_default_ridge():
    df = _make_df(n=70)
    out = AutoRegressor(lags=10).fit_predict(df, t=10)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(range(70, 80))


def test_forecast_datetimeindex_continuation():
    idx = pd.date_range('2026-01-01', periods=70, freq='D')
    df = _make_df(n=70, index=idx)
    out = AutoRegressor(lags=10).fit_predict(df, t=5)

    assert isinstance(out.index, pd.DatetimeIndex)
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=i) for i in range(1, 6)])
    assert list(out.index) == list(expected)


def test_trend_direction_sanity():
    df = _make_df(n=70)
    out = AutoRegressor(model='Ridge', lags=10).fit_predict(df, t=15)

    assert out['a'].mean() > df['a'].iloc[:35].mean()


def test_list_in_list_out():
    dfs = [_make_df(n=60), _make_df(n=80)]
    out = AutoRegressor(lags=8).fit_predict(dfs, t=6)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (6, 3)
        assert list(fc.index) == list(range(len(src), len(src) + 6))


@pytest.mark.parametrize('model', [
    'Ridge', 'Lasso', 'LinearRegression', 'RandomForestRegressor',
    'GradientBoostingRegressor', 'SVR', 'KNeighborsRegressor',
])
def test_string_registry_models_run(model):
    df = _make_df(n=60)
    out = AutoRegressor(model=model, lags=8).fit_predict(df, t=4)
    assert out.shape == (4, 3)
    assert np.isfinite(out.to_numpy()).all()


def test_class_and_instance_forms():
    from sklearn.linear_model import Ridge

    df = _make_df(n=60)

    out_cls = AutoRegressor(model=Ridge, lags=8, alpha=0.5).fit_predict(df, t=4)
    assert out_cls.shape == (4, 3)

    out_inst = AutoRegressor(model=Ridge(alpha=0.5), lags=8).fit_predict(df, t=4)
    assert out_inst.shape == (4, 3)


def test_univariate_series():
    df = _make_df(n=60)[['a']]
    out = AutoRegressor(model='SVR', lags=8).fit_predict(df, t=4)
    assert out.shape == (4, 1)
    assert np.isfinite(out.to_numpy()).all()


def test_unknown_model_name_raises():
    df = _make_df(n=60)
    with pytest.raises(Exception):
        AutoRegressor(model='NotARealModel', lags=8).fit_predict(df, t=4)
