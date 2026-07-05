import numpy as np
import pandas as pd

from hypertools.predict.gp import GaussianProcess


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
    out = GaussianProcess().fit_predict(df, t=10)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3)
    assert list(out.columns) == list(df.columns)
    assert list(out.index) == list(range(70, 80))


def test_forecast_datetimeindex_continuation():
    idx = pd.date_range('2026-01-01', periods=70, freq='D')
    df = _make_df(n=70, index=idx)
    out = GaussianProcess().fit_predict(df, t=5)

    assert isinstance(out.index, pd.DatetimeIndex)
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=i) for i in range(1, 6)])
    assert list(out.index) == list(expected)


def test_trend_direction_sanity():
    df = _make_df(n=70)
    out = GaussianProcess().fit_predict(df, t=15)

    assert out['a'].mean() > df['a'].iloc[:35].mean()


def test_list_in_list_out():
    dfs = [_make_df(n=60), _make_df(n=80)]
    out = GaussianProcess().fit_predict(dfs, t=6)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert fc.shape == (6, 3)
        assert list(fc.index) == list(range(len(src), len(src) + 6))


def test_kernel_and_alpha_kwargs_pass_through():
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    df = _make_df(n=50)
    custom_kernel = RBF(5.0) + WhiteKernel(noise_level=0.5)
    model = GaussianProcess(kernel=custom_kernel, alpha=1e-6, normalize_y=False)
    out = model.fit_predict(df, t=4)

    assert out.shape == (4, 3)
    fitted_gp = model.models_[0]['gp']
    assert fitted_gp.alpha == 1e-6
    assert fitted_gp.normalize_y is False
