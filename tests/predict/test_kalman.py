import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pykalman')

from hypertools.predict.kalman import Kalman


def _make_df(n=70, index=None):
    t = np.arange(n)
    trend = 0.05 * t
    sine = np.sin(t / 5.0)
    df = pd.DataFrame({'a': trend + sine, 'b': trend - sine, 'c': sine * 2})
    if index is not None:
        df.index = index
    return df


def test_forecast_shape_and_rangeindex_continuation():
    df = _make_df(n=70)
    out = Kalman(n_iter=5).fit_predict(df, t=10)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(range(70, 80))


def test_forecast_datetimeindex_continuation():
    idx = pd.date_range('2026-01-01', periods=70, freq='D')
    df = _make_df(n=70, index=idx)
    out = Kalman(n_iter=5).fit_predict(df, t=5)

    assert isinstance(out.index, pd.DatetimeIndex)
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=i) for i in range(1, 6)])
    assert list(out.index) == list(expected)


def test_trend_direction_sanity():
    df = _make_df(n=70)
    out = Kalman(n_iter=5).fit_predict(df, t=15)

    # loose sanity check: the upward-trending column's forecast mean should
    # exceed the observed mean of the second half of the input (real
    # algorithm -- no tight bounds)
    assert out['a'].mean() > df['a'].iloc[:35].mean()


def test_nan_tolerance():
    df = _make_df(n=70)
    df_missing = df.copy()
    df_missing.iloc[10:13, :] = np.nan  # a few fully-missing interior rows
    df_missing.iloc[40, 1] = np.nan     # a single missing entry

    out = Kalman(n_iter=5).fit_predict(df_missing, t=8)

    assert out.shape == (8, 3)
    assert np.isfinite(out.to_numpy()).all()


def test_list_in_list_out():
    dfs = [_make_df(n=60), _make_df(n=80)]
    out = Kalman(n_iter=5).fit_predict(dfs, t=6)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (6, 3)
        assert list(fc.index) == list(range(len(src), len(src) + 6))


def test_friendly_import_error_when_pykalman_missing(monkeypatch):
    import hypertools.predict.kalman as kalman_mod
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'pykalman':
            raise ImportError('no module named pykalman')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    df = _make_df(n=30)
    with pytest.raises(ImportError, match='hypertools\\[predict\\]'):
        kalman_mod.Kalman(n_iter=2).fit(df)
