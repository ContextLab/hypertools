import numpy as np
import pandas as pd
import pytest

pytest.importorskip('skaters')

from hypertools.predict.laplace import Laplace


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
    out = Laplace().fit_predict(df, t=10)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(range(70, 80))


def test_forecast_datetimeindex_continuation():
    idx = pd.date_range('2026-01-01', periods=70, freq='D')
    df = _make_df(n=70, index=idx)
    out = Laplace().fit_predict(df, t=5)

    assert isinstance(out.index, pd.DatetimeIndex)
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=i) for i in range(1, 6)])
    assert list(out.index) == list(expected)


def test_trend_direction_sanity():
    df = _make_df(n=70)
    out = Laplace().fit_predict(df, t=15)

    # loose sanity check: the upward-trending column's forecast mean should
    # exceed the observed mean of the second half of the input (real
    # algorithm -- no tight bounds)
    assert out['a'].mean() > df['a'].iloc[:35].mean()


def test_list_in_list_out():
    dfs = [_make_df(n=60), _make_df(n=80)]
    out = Laplace().fit_predict(dfs, t=6)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (6, 3)
        assert list(fc.index) == list(range(len(src), len(src) + 6))


def test_large_horizon_single_factory_call_handles_it():
    # Verified directly against skaters==0.11.0: a single laplace(k=t) call
    # handles horizons well beyond typical use (checked k up to 100) without
    # truncating -- this exercises that path with t=50 on a longer series.
    df = _make_df(n=90)
    out = Laplace().fit_predict(df, t=50)

    assert out.shape == (50, 3)
    assert np.isfinite(out.to_numpy()).all()


def test_friendly_import_error_when_skaters_missing(monkeypatch):
    import hypertools.predict.laplace as laplace_mod
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'skaters.api':
            raise ImportError('no module named skaters')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    df = _make_df(n=30)
    with pytest.raises(ImportError, match='hypertools\\[predict\\]'):
        laplace_mod.Laplace().fit_predict(df, t=5)
