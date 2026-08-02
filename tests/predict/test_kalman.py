import numpy as np
import pandas as pd
import pytest

pytest.importorskip('pykalman')

from hypertools.predict.kalman import Kalman, _companion_transition, _resolve_lags


def _random_walk(seed=3, n=40, d=3, drift=0.6):
    """A drift-y random walk: a unit-root process, so the least-squares
    companion estimate sits right at the edge of the stable region and
    tips over it when the regression is near-saturated."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((n, d)) + drift, axis=0)


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


def test_estimated_transition_is_non_explosive():
    """The delay-embedded transition operator is applied once per forecast
    step, so a spectral radius above 1 compounds as rho ** t. Unconstrained
    least squares returns rho up to ~4 when the regression is near-saturated
    (n - lags close to lags * n_features); the estimate must be constrained
    to the non-explosive region instead."""
    x = _random_walk(seed=3, n=40, d=3, drift=0.6)

    offenders = {}
    for k in range(5, 41):
        window = x[:k]
        n, d = window.shape
        lags = _resolve_lags(None, n, d)
        A, _ = _companion_transition(window, lags)
        rho = float(np.abs(np.linalg.eigvals(A)).max())
        if rho > 1 + 1e-8:
            offenders[k] = round(rho, 4)

    assert not offenders, (
        f'explosive transition matrices (spectral radius > 1) at history '
        f'lengths {offenders}')


def test_forecast_bounded_on_short_random_walk_history():
    """A 12-step forecast must stay in the neighbourhood of the data it
    continues. History lengths near 20 saturate the least-squares design
    (n - lags ~ lags * n_features) and used to produce forecasts many
    thousands of times the data range."""
    x = _random_walk(seed=3, n=40, d=3, drift=0.6)
    data_range = float(x.max() - x.min())

    df = pd.DataFrame(x[:20], columns=['a', 'b', 'c'])
    out = Kalman(n_iter=5).fit_predict(df, t=12)

    ratio = float(np.abs(out.to_numpy()).max()) / data_range
    assert np.isfinite(out.to_numpy()).all()
    assert ratio < 10, (
        f'12-step forecast reaches {ratio:.1f}x the data range '
        f'(max |forecast| = {float(np.abs(out.to_numpy()).max()):.4g}, '
        f'data range = {data_range:.4g})')


@pytest.mark.parametrize('seed,k', [(1, 19), (4, 20), (7, 19), (11, 20)])
def test_forecast_bounded_across_saturating_history_lengths(seed, k):
    """The blow-up is not monotonic in history length: it strikes wherever
    the number of regression rows lands near the number of predictors.
    These (seed, k) pairs each produced rho > 3.6 before the fix."""
    x = _random_walk(seed=seed, n=40, d=3, drift=0.5)
    data_range = float(x.max() - x.min())

    df = pd.DataFrame(x[:k], columns=['a', 'b', 'c'])
    out = Kalman(n_iter=5).fit_predict(df, t=12)

    ratio = float(np.abs(out.to_numpy()).max()) / data_range
    assert ratio < 10, f'12-step forecast reaches {ratio:.1f}x the data range'


def test_genuinely_explosive_dynamics_are_still_followed():
    """The stability constraint must not cost the forecaster its ability to
    follow real growth. A clean exponential is an exactly-representable
    order-1 process fitted from a strongly over-determined regression (20
    rows for 5 predictors), so the explosive estimate is real and must be
    kept -- unlike the near-saturated random-walk fits, which sit at ~1 row
    per predictor. A blanket rho <= 1 constraint fails this test."""
    steps = np.arange(25.0)
    df = pd.DataFrame({'a': 1.3 ** steps})
    truth = 1.3 ** np.arange(25.0, 31.0)

    lags = _resolve_lags(None, *df.shape)
    A, _ = _companion_transition(df.to_numpy(dtype=float), lags)
    rho = float(np.abs(np.linalg.eigvals(A)).max())
    assert rho == pytest.approx(1.3, abs=0.05), (
        f'explosive dynamics supported by a strongly over-determined fit '
        f'were flattened to rho={rho:.3f}')

    out = Kalman(n_iter=5).fit_predict(df, t=6).to_numpy().ravel()
    rel_err = float(np.abs(out - truth).max() / truth.max())
    assert rel_err < 0.05, f'forecast lost the growth (rel. error {rel_err:.3f})'


def test_noisy_exponential_growth_accuracy_is_retained():
    """Regression guard for the same trade-off on realistic (noisy) growth:
    this case scored MAE 1.75 before the stability work and 20.8 under a
    blanket constraint."""
    rng = np.random.default_rng(0)
    growth = 1.15 ** np.arange(31.0)
    observed = growth[:25] * (1 + 0.05 * rng.standard_normal(25))
    df = pd.DataFrame({'a': observed})

    out = Kalman(n_iter=5).fit_predict(df, t=6).to_numpy().ravel()
    mae = float(np.abs(out - growth[25:]).mean())

    assert mae < 5.0, f'noisy-exponential MAE regressed to {mae:.2f}'


def test_stability_constraint_preserves_delay_embedding_structure():
    """Constraining the estimate must scale the fitted AR coefficient
    blocks, not the companion shift block -- the lower rows encode 'this
    state entry is the observation from one step earlier' and are structural,
    not estimated."""
    x = _random_walk(seed=4, n=40, d=3, drift=0.5)
    window = x[:20]
    n, d = window.shape
    lags = _resolve_lags(None, n, d)

    A, H = _companion_transition(window, lags)

    assert A.shape == (lags * d, lags * d)
    # shift block: identity mapping older observations down the state vector
    np.testing.assert_allclose(A[d:, :-d], np.eye((lags - 1) * d))
    np.testing.assert_allclose(A[d:, -d:], np.zeros(((lags - 1) * d, d)))
    # observation matrix still reads the newest block
    np.testing.assert_allclose(H[:, :d], np.eye(d))


def test_stability_constraint_is_inert_on_well_conditioned_dynamics():
    """Trend + sine is a legitimate unit-root/oscillatory process whose
    least-squares companion estimate already sits at rho == 1. The
    constraint must leave such fits untouched (no flattening -- that was the
    failure mode of the EM-fitted transition matrix, see module docstring)."""
    df = _make_df(n=70)
    x = df.to_numpy(dtype=float)
    lags = _resolve_lags(None, *x.shape)

    A, _ = _companion_transition(x, lags)
    rho = float(np.abs(np.linalg.eigvals(A)).max())

    assert rho == pytest.approx(1.0, abs=1e-3)
    # and the forecast still tracks the oscillation rather than flat-lining
    out = Kalman(n_iter=5).fit_predict(df, t=15)
    assert out['c'].std() > 0.5 * df['c'].iloc[-15:].std()


def test_friendly_import_error_when_pykalman_missing(monkeypatch):
    # `hypertools.predict` (the attribute) is shadowed by `hyp.predict` the
    # dispatcher function (see hypertools/__init__.py); `import a.b.c as x`
    # walks attributes from the top, so it would resolve the shadowed
    # attribute. `from a.b import c as x` resolves via sys.modules instead.
    from hypertools.predict import kalman as kalman_mod
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'pykalman':
            raise ImportError('no module named pykalman')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    df = _make_df(n=30)
    with pytest.raises(ImportError, match='pip install pykalman'):
        kalman_mod.Kalman(n_iter=2).fit(df)
