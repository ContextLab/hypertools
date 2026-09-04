import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from hypertools.predict.common import Forecaster, resolve_t


def _constant_fitter(data, **kwargs):
    return {"last_row": data.iloc[-1]}


def _constant_forecaster(data, n_steps, future_index, **kwargs):
    last_row = kwargs["last_row"]
    rows = [last_row.to_numpy() for _ in range(n_steps)]
    return pd.DataFrame(rows, index=future_index, columns=data.columns)


def _make_forecaster():
    return Forecaster(fitter=_constant_fitter, forecaster=_constant_forecaster, required=["last_row"])


def _make_df(n=10, ncols=3, index=None):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.rand(n, ncols), columns=list("abc")[:ncols])
    if index is not None:
        df.index = index
    return df


# --- base fit/predict/fit_predict contract -----------------------------

def test_fit_predict_single_in_single_out():
    df = _make_df(n=10)
    out = _make_forecaster().fit_predict(df, t=5)

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (5, 3)
    assert list(out.columns) == list(df.columns)
    # constant-continuation forecaster: every forecast row equals the last observed row
    for _, row in out.iterrows():
        assert np.allclose(row.to_numpy(), df.iloc[-1].to_numpy())
    # index continues the RangeIndex
    assert list(out.index) == [10, 11, 12, 13, 14]


def test_fit_predict_list_in_list_out():
    dfs = [_make_df(n=8), _make_df(n=12)]
    out = _make_forecaster().fit_predict(dfs, t=3)

    assert isinstance(out, list)
    assert len(out) == 2
    for src, fc in zip(dfs, out):
        assert isinstance(fc, pd.DataFrame)
        assert fc.shape == (3, 3)
        assert list(fc.index) == list(range(len(src), len(src) + 3))


def test_predict_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        _make_forecaster().predict(5)


def test_fit_accepts_ndarray():
    arr = np.random.RandomState(1).rand(6, 2)
    out = _make_forecaster().fit_predict(arr, t=2)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (2, 2)


# --- resolve_t: int on RangeIndex ---------------------------------------

def test_resolve_t_int_on_rangeindex():
    df = _make_df(n=10)
    n_steps, future_index = resolve_t(df, 4)

    assert n_steps == 4
    assert list(future_index) == [10, 11, 12, 13]


# --- resolve_t: int on DatetimeIndex (hourly + irregular) ---------------

def test_resolve_t_int_on_hourly_datetimeindex():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    df = _make_df(n=5, index=idx)

    n_steps, future_index = resolve_t(df, 3)

    assert n_steps == 3
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=i) for i in (1, 2, 3)])
    assert list(future_index) == list(expected)


def test_resolve_t_int_on_irregular_datetimeindex_uses_min_nonzero_diff():
    # gaps (minutes): 1, 2, 1, 6 -> minimum non-zero diff is 1 minute
    base = pd.Timestamp("2026-01-01")
    idx = pd.DatetimeIndex([base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=3),
                             base + pd.Timedelta(minutes=4), base + pd.Timedelta(minutes=10)])
    df = _make_df(n=5, index=idx)

    n_steps, future_index = resolve_t(df, 2)

    assert n_steps == 2
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(minutes=1), idx[-1] + pd.Timedelta(minutes=2)])
    assert list(future_index) == list(expected)


# --- resolve_t: datetime t (future and past) ----------------------------

def test_resolve_t_future_datetime():
    idx = pd.date_range("2026-01-01", periods=5, freq="h")
    df = _make_df(n=5, index=idx)

    target = idx[-1] + pd.Timedelta(hours=3)
    n_steps, future_index = resolve_t(df, target)

    assert n_steps == 3
    expected = pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=i) for i in (1, 2, 3)])
    assert list(future_index) == list(expected)


def test_resolve_t_past_datetime_signals_truncation():
    idx = pd.date_range("2026-01-01", periods=10, freq="h")
    df = _make_df(n=10, index=idx)

    target = idx[4]  # 5th observation: strictly in the past relative to idx[-1]
    n_steps, future_index = resolve_t(df, target)

    assert n_steps < 0
    assert n_steps == -5  # 5 trailing rows would be dropped
    assert list(future_index) == list(idx[:5])


# --- resolve_t: duplicated index entries, and the scope of that check ---

def test_resolve_t_rejects_duplicated_timestamps():
    """A repeated TIMESTAMP makes the horizon ill-defined: `_infer_step`
    would drop the (zero-length) gap between the repeats and forecast on a
    step that no longer describes the data, and a datetime-like `t` would
    truncate to an ambiguous position."""
    idx = pd.DatetimeIndex(
        sorted(list(pd.date_range("2026-01-01", periods=5, freq="h")) * 2))
    df = _make_df(n=10, index=idx)

    with pytest.raises(ValueError, match=r"duplicated entr.*ill-defined"):
        resolve_t(df, 3)


def test_resolve_t_keeps_a_duplicated_integer_index():
    """SCOPE: the rejection above is about the TIME axis (1.1 plan,
    *Decisions (resolved)* #4: "legitimate integer-indexed panels are not
    rejected"). A stacked panel -- `pd.concat([run_a, run_b])`, whose index
    repeats 0..n-1 -- has a perfectly well-defined horizon: the step is the
    minimum non-zero difference (1) and the forecast continues from the last
    row, exactly as in 1.0."""
    df = _make_df(n=10, index=pd.Index(sorted(list(range(5)) * 2)))

    n_steps, future_index = resolve_t(df, 2)

    assert n_steps == 2
    assert list(future_index) == [5, 6]


def test_all_identical_timestamps_message_comes_from_live_infer_step(monkeypatch):
    """The fully-degenerate case (every observation at ONE timestamp) is
    `_infer_step`'s: `tests/test_predict_audit_fixes.py` pins its wording.
    `resolve_t`'s duplicate check runs FIRST, so it must hand this case to
    `_infer_step` rather than raise a copied string -- a copy would leave
    that branch dead code with a test that only pins the copy."""
    from hypertools.predict import common as common_module

    calls = []
    real_infer_step = common_module._infer_step

    def spy(index):
        calls.append(index)
        return real_infer_step(index)

    monkeypatch.setattr(common_module, "_infer_step", spy)
    df = _make_df(n=5, index=pd.DatetimeIndex(["2026-01-01"] * 5))

    with pytest.raises(ValueError, match="share one timestamp"):
        resolve_t(df, 3)

    assert calls, "the message must come from live _infer_step code, not a copy"


def test_forecaster_predict_truncates_on_past_datetime_without_calling_forecaster():
    idx = pd.date_range("2026-01-01", periods=10, freq="h")
    df = _make_df(n=10, index=idx)

    def _boom(*_, **__):
        raise AssertionError("forecaster should not be invoked for a past-date truncation")

    forecaster = Forecaster(fitter=_constant_fitter, forecaster=_boom, required=["last_row"])
    out = forecaster.fit_predict(df, t=idx[4])

    assert isinstance(out, pd.DataFrame)
    assert list(out.index) == list(idx[:5])
    pd.testing.assert_frame_equal(out, df.iloc[:5])
