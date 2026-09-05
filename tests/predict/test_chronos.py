import numpy as np
import pandas as pd
import pytest

pytest.importorskip('chronos')

from hypertools.predict.chronos import Chronos


def _make_df(n=60, index=None):
    t = np.arange(n)
    trend = 0.05 * t
    sine = np.sin(t / 5.0)
    df = pd.DataFrame({'a': trend + sine, 'b': trend - sine})
    if index is not None:
        df.index = index
    return df


def test_forecast_shape_and_rangeindex_continuation():
    df = _make_df(n=60)
    out = Chronos().fit_predict(df, t=5)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (5, 2)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(range(60, 65))
    assert np.isfinite(out.to_numpy()).all()


def test_forecast_datetimeindex_continuation():
    idx = pd.date_range('2026-01-01', periods=60, freq='D')
    df = _make_df(n=60, index=idx)
    out = Chronos().fit_predict(df, t=3)

    assert isinstance(out.index, pd.DatetimeIndex)
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=i) for i in range(1, 4)])
    assert list(out.index) == list(expected)


def test_list_in_list_out():
    dfs = [_make_df(n=50), _make_df(n=70)]
    out = Chronos().fit_predict(dfs, t=4)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (4, 2)
        assert list(fc.index) == list(range(len(src), len(src) + 4))


def test_friendly_import_error_when_chronos_missing(monkeypatch):
    """With automatic installation disabled, a missing optional module raises
    an ImportError naming the manual `pip install "hypertools[...]"` command
    (with it enabled, `lazy_import` would install the extra instead -- not
    something a unit test should do). `sys.modules[name] = None` is how the
    import system marks a module as unimportable."""
    from hypertools.predict import chronos as mod
    monkeypatch.setenv('HYPERTOOLS_AUTO_INSTALL', '0')
    monkeypatch.setitem(__import__('sys').modules, 'chronos', None)

    df = _make_df(n=30)
    with pytest.raises(ImportError, match=r'hypertools\[predict-hf\]'):
        mod.Chronos().fit(df)
