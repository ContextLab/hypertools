# noinspection PyPackageRequirements
"""Tests for GH #285 item 2: the `Delay` (Takens time-delay embedding)
manipulator, `manip='Delay'` / `Delay(tau=1, dims=2, drop_edges=True)`.

All tests use real data and real hypertools calls (no mocks). The core
claim is an EXACT match against the hand-built embedding in
`docs/tutorials/modern_sklearn_dynamics.ipynb` cell 6:
``np.column_stack([x[i * tau:i * tau + n] for i in range(dims)])``.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.manip import Delay
from hypertools.manip.manip import _supported_names
from hypertools.tools.analyze import analyze


def _lorenz(n=400, dt=0.01, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """A real (if short) Lorenz trajectory, matching the shape of data used
    by docs/tutorials/modern_sklearn_dynamics.ipynb -- deterministic, no
    RNG needed."""
    state = np.array([1.0, 1.0, 1.0])
    out = np.empty((n, 3))
    for i in range(n):
        out[i] = state
        x, y, z = state
        d = np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
        state = state + dt * d
    return out


def _tutorial_delay_matrix(x, tau, dims):
    """The exact construction from modern_sklearn_dynamics.ipynb cell 6."""
    n = len(x) - tau * (dims - 1)
    return np.column_stack([x[i * tau:i * tau + n] for i in range(dims)])


# --- core claim: exact match against the tutorial's hand-built embedding --

def test_delay_matches_tutorial_column_stack_construction():
    trajectory = _lorenz()
    x = trajectory[:, 0]
    tau, dims = 5, 20
    expected = _tutorial_delay_matrix(x, tau, dims)

    got = np.asarray(hyp.manip(x[:, None], model='Delay', tau=tau, dims=dims))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected)


def test_delay_class_matches_tutorial_construction_directly():
    x = np.arange(37, dtype=float) ** 1.3  # not perfectly linear, real values
    tau, dims = 3, 6
    expected = _tutorial_delay_matrix(x, tau, dims)

    df = pd.DataFrame({'x': x})
    out = Delay(tau=tau, dims=dims).fit_transform(df)
    np.testing.assert_allclose(out.to_numpy(), expected)


# --- registration: supported_models()/string specs find it ----------------

def test_delay_registered_in_manip_supported_names():
    assert 'Delay' in _supported_names()


def test_delay_exported_from_manip_package():
    from hypertools.manip import Delay as ExportedDelay
    assert ExportedDelay is Delay


# --- column naming / order -------------------------------------------------

def test_delay_column_names_and_order():
    df = pd.DataFrame({'x': np.arange(10, dtype=float)})
    out = Delay(tau=2, dims=3).fit_transform(df)
    assert list(out.columns) == ['x_lag4', 'x_lag2', 'x_lag0']
    # last column (lag0) is always the undelayed value
    np.testing.assert_allclose(out['x_lag0'].to_numpy(), np.arange(4, 10, dtype=float))


# --- multi-column input: each column embedded independently ---------------

def test_delay_multicolumn_embeds_each_column_independently():
    x = np.arange(20, dtype=float)
    y = np.arange(20, dtype=float) * 10
    df = pd.DataFrame({'x': x, 'y': y})
    out = Delay(tau=2, dims=3).fit_transform(df)
    assert list(out.columns) == ['x_lag4', 'x_lag2', 'x_lag0', 'y_lag4', 'y_lag2', 'y_lag0']
    expected_x = _tutorial_delay_matrix(x, 2, 3)
    expected_y = _tutorial_delay_matrix(y, 2, 3)
    np.testing.assert_allclose(out[['x_lag4', 'x_lag2', 'x_lag0']].to_numpy(), expected_x)
    np.testing.assert_allclose(out[['y_lag4', 'y_lag2', 'y_lag0']].to_numpy(), expected_y)


# --- DataFrame in -> DataFrame out, with generated column names -----------

def test_delay_dataframe_in_dataframe_out_generated_columns():
    df = pd.DataFrame({'temp': np.linspace(0, 1, 15)})
    out = Delay(tau=1, dims=4).fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ['temp_lag3', 'temp_lag2', 'temp_lag1', 'temp_lag0']
    assert out.shape == (15 - 3, 4)


def test_delay_dataframe_index_preserved_for_kept_rows():
    idx = pd.date_range('2020-01-01', periods=12, freq='D')
    df = pd.DataFrame({'x': np.arange(12, dtype=float)}, index=idx)
    out = Delay(tau=1, dims=3, drop_edges=True).fit_transform(df)
    assert list(out.index) == list(idx[2:])


# --- drop_edges=False pads with NaN ----------------------------------------

def test_drop_edges_false_pads_with_nan_same_row_count():
    x = np.arange(10, dtype=float)
    df = pd.DataFrame({'x': x})
    out = Delay(tau=2, dims=3, drop_edges=False).fit_transform(df)
    assert out.shape == (10, 3)  # same number of rows as input
    # the first max_lag=4 rows must have at least one NaN (insufficient
    # history); the undelayed (lag0) column is always fully populated
    assert out['x_lag0'].isna().sum() == 0
    assert out['x_lag4'].isna().sum() == 4
    assert out['x_lag2'].isna().sum() == 2
    # rows with full history match the drop_edges=True (dropped) values
    dropped = Delay(tau=2, dims=3, drop_edges=True).fit_transform(df)
    np.testing.assert_allclose(out.iloc[4:].to_numpy(), dropped.to_numpy())


def test_drop_edges_true_raises_when_no_rows_survive():
    df = pd.DataFrame({'x': np.arange(5, dtype=float)})
    with pytest.raises(ValueError, match='drop_edges'):
        Delay(tau=3, dims=3, drop_edges=True).fit_transform(df)  # needs 7 rows


# --- dims=1 is a no-op identity (renamed column, unchanged values) --------

def test_dims_1_is_identity():
    x = np.linspace(-2, 2, 12)
    df = pd.DataFrame({'v': x})
    out = Delay(tau=5, dims=1).fit_transform(df)
    assert list(out.columns) == ['v_lag0']
    np.testing.assert_allclose(out['v_lag0'].to_numpy(), x)


# --- list input: per-dataset (no history bleeding across datasets) --------

def test_delay_list_input_is_per_dataset():
    a = np.arange(20, dtype=float)
    b = np.arange(20, dtype=float) + 1000  # far away, bleed would be obvious
    out = hyp.manip([a[:, None], b[:, None]], model='Delay', tau=2, dims=3)
    assert isinstance(out, list) and len(out) == 2
    expected_a = _tutorial_delay_matrix(a, 2, 3)
    expected_b = _tutorial_delay_matrix(b, 2, 3)
    np.testing.assert_allclose(np.asarray(out[0]), expected_a)
    np.testing.assert_allclose(np.asarray(out[1]), expected_b)


# --- constructor validation --------------------------------------------

@pytest.mark.parametrize('tau', [0, -1, 1.5, True])
def test_invalid_tau_raises(tau):
    with pytest.raises(ValueError, match='tau'):
        Delay(tau=tau)


@pytest.mark.parametrize('dims', [0, -1, 2.5, True])
def test_invalid_dims_raises(dims):
    with pytest.raises(ValueError, match='dims'):
        Delay(dims=dims)


# --- works through hyp.manip(x, model='Delay', ...) ------------------------

def test_manip_delay_string_spec():
    x = np.arange(30, dtype=float)
    out = np.asarray(hyp.manip(x[:, None], model='Delay', tau=1, dims=2))
    expected = _tutorial_delay_matrix(x, 1, 2)
    np.testing.assert_allclose(out, expected)


# --- cross-module kwargs: manip='Delay' inside hyp.plot / hyp.analyze -----

def test_plot_manip_delay_dict_spec():
    x = np.arange(30, dtype=float).reshape(-1, 1) + np.array([0, 100])
    fig = hyp.plot(x, manip={'model': 'Delay', 'kwargs': {'tau': 2, 'dims': 3}}, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_analyze_manip_delay_dict_spec():
    x = np.column_stack([np.arange(30, dtype=float), np.arange(30, dtype=float) * 2])
    result = np.asarray(analyze(x, manip={'model': 'Delay', 'kwargs': {'tau': 2, 'dims': 3}},
                               reduce=None))
    # 2 input columns x 3 dims = 6 output columns, 30 - 2*2 = 26 rows
    assert result.shape == (26, 6)
