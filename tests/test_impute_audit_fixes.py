"""Regression tests for the 2026-07 release-audit fixes to hyp.impute
(unit F17-impute, D05-gallery-data-text-001, and the impute parts of
X2-error-quality-003/-004).

Every test uses real data and real imputer runs (no mocks); each mirrors a
repro that was CONFIRMED failing on the pre-fix code by the independent
audit verifiers (see notes/audit-1.0-2026-07/verdicts/F17-impute.json and
D05-gallery-data-text.json).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.impute.impute import impute


def _latent_sine(D, n=100, k=3, frac=0.1, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    latents = np.column_stack(
        [np.sin(2 * np.pi * t / p) for p in (23.0, 41.0, 67.0)])[:, :k]
    truth = latents @ rng.randn(k, D) + 0.05 * rng.randn(n, D)
    holes = rng.rand(n, D) < frac
    for i in range(n):
        if holes[i].all():
            holes[i, rng.randint(D)] = False
    x = truth.copy()
    x[holes] = np.nan
    return truth, x, holes


def _rank3_benchmark(seed=0, n=120, D=8, k=3, frac=0.10):
    rng = np.random.RandomState(seed)
    truth = rng.randn(n, k) @ rng.randn(k, D) + 0.05 * rng.randn(n, D)
    holes = rng.rand(n, D) < frac
    for i in range(n):
        if holes[i].all():
            holes[i, rng.randint(D)] = False
    x = truth.copy()
    x[holes] = np.nan
    return truth, x, holes


# --- D05-gallery-data-text-001: Kalman on wide data --------------------------

@pytest.mark.parametrize('D', [20, 50, 100])
def test_kalman_wide_data_recovers_instead_of_zero_filling(D):
    pytest.importorskip('pykalman')
    truth, x, holes = _latent_sine(D)
    est = np.asarray(impute(x.copy(), model='Kalman'))

    filled = est[holes]
    # pre-fix: every fill was exactly 0.0 for D >= 50 (std 0, 1 unique value)
    assert filled.std() > 0.0
    assert len(np.unique(filled)) > 1
    assert np.corrcoef(filled, truth[holes])[0, 1] > 0.9
    # observed entries are never altered (byte-identical)
    assert np.array_equal(est[~holes], truth[~holes])
    assert not np.isnan(est).any()


def test_kalman_fills_fully_missing_rows_from_neighbors():
    pytest.importorskip('pykalman')
    truth, x, holes = _latent_sine(20)
    x[40:43, :] = np.nan
    est = np.asarray(impute(x.copy(), model='Kalman'))
    assert not np.isnan(est).any()
    # the fills track the (smooth) truth, i.e. genuinely use neighboring rows
    r = np.corrcoef(est[40:43].ravel(), truth[40:43].ravel())[0, 1]
    assert r > 0.9


# --- F17-impute-009: Kalman single-row input ---------------------------------

def test_kalman_single_row_clear_error():
    pytest.importorskip('pykalman')
    with pytest.raises(ValueError, match='at least 2 rows'):
        impute(np.array([[1.0, np.nan, 3.0]]), model='Kalman')


# --- F17-impute-002: default PPCA recovers low-rank structure ----------------

def test_ppca_default_recovers_low_rank_structure():
    truth, x, holes = _rank3_benchmark()
    out = np.asarray(impute(x.copy(), model='PPCA', random_state=0))
    # pre-fix (full-rank default): r ~0.13, numerically identical to plain
    # column-mean fills
    assert np.corrcoef(out[holes], truth[holes])[0, 1] > 0.9
    assert np.array_equal(out[~holes], truth[~holes])


# --- F17-impute-003: PPCA random_state ---------------------------------------

def test_ppca_random_state_gives_deterministic_imputations():
    _, x, _ = _rank3_benchmark()
    a = np.asarray(impute(x.copy(), model='PPCA', random_state=7))
    b = np.asarray(impute(x.copy(), model='PPCA', random_state=7))
    assert np.array_equal(a, b)


def test_ppca_random_state_does_not_disturb_global_rng():
    _, x, _ = _rank3_benchmark()
    np.random.seed(123)
    expected = np.random.rand(3)
    np.random.seed(123)
    impute(x.copy(), model='PPCA', random_state=7)
    assert np.array_equal(np.random.rand(3), expected)


# --- F17-impute-004: constant column no longer poisons the EM ---------------

def test_ppca_constant_column_imputed_exactly_without_warning():
    rng = np.random.RandomState(1)
    cst = rng.randn(40, 4) * 2 + 1
    cst[:, 3] = 7.0
    cst[4, 3] = np.nan
    cst[6, 0] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = np.asarray(impute(cst.copy(), model='PPCA', random_state=0))
    # pre-fix: nondeterministically a 'did not converge (relative change nan)'
    # warning, a LinAlgError crash, or a silent full mean-fill collapse
    assert not any('did not converge' in str(w.message) for w in caught)
    assert out[4, 3] == 7.0  # the constant column's fill is the constant
    assert not np.isnan(out).any()


def test_ppca_ordinary_dense_data_no_convergence_warning():
    _, x, _ = _rank3_benchmark()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        impute(x.copy(), model='PPCA', random_state=0)
    assert not any('did not converge' in str(w.message) for w in caught)


# --- F17-impute-005: reuse with wrong column count ---------------------------

def test_ppca_reuse_wrong_column_count_raises_valueerror():
    rng = np.random.RandomState(2)
    train = rng.randn(60, 4) * 3 + 10
    train[rng.rand(60, 4) < 0.1] = np.nan
    _, fitted = impute(train, model='PPCA', d=2, return_model=True)
    with pytest.raises(ValueError, match='same columns'):
        impute(rng.randn(10, 7), model=fitted)


# --- F17-impute-006 (doc fix): observed values ARE preserved exactly ---------

@pytest.mark.parametrize('model', ['PPCA', 'SimpleImputer', 'KNNImputer',
                                   'IterativeImputer', 'Kalman'])
def test_every_model_preserves_observed_entries_exactly(model):
    if model == 'Kalman':
        pytest.importorskip('pykalman')
    truth, x, holes = _rank3_benchmark(seed=5)
    out = np.asarray(impute(x.copy(), model=model, random_state=0)
                     if model in ('PPCA', 'IterativeImputer')
                     else impute(x.copy(), model=model))
    assert np.array_equal(out[~holes], truth[~holes])
    assert not np.isnan(out).any()


def test_ppca_module_docstring_documents_the_splice():
    # NOTE: `import hypertools.impute.ppca as ...` would resolve through the
    # shadowing `impute` FUNCTION attribute (F16-predict-017, out of scope
    # here); the from-import resolves via sys.modules.
    from hypertools.impute import ppca as ppca_mod
    assert 'preserved exactly' in ppca_mod.__doc__
    assert 'NOT guaranteed' not in ppca_mod.__doc__


# --- F17-impute-007 / X2-error-quality-003: kwargs handling ------------------

def test_typo_kwarg_raises_typeerror_for_every_imputer():
    from hypertools.impute import (PPCA, SimpleImputer, KNNImputer,
                                   IterativeImputer, Kalman)
    for cls, typo in [(PPCA, {'dd': 3}), (SimpleImputer, {'strateggy': 'median'}),
                      (KNNImputer, {'n_neighbours': 3}),
                      (IterativeImputer, {'random_stat': 0}),
                      (Kalman, {'n_itr': 3})]:
        with pytest.raises(TypeError):
            cls(**typo)


def test_typo_kwarg_via_dispatcher_raises_typeerror():
    _, x, _ = _rank3_benchmark()
    with pytest.raises(TypeError, match='strateggy'):
        impute(x.copy(), model='SimpleImputer', strateggy='median')


def test_fork_dict_spec_merges_outer_kwargs():
    rng = np.random.RandomState(2)
    x = rng.randn(40, 4)
    x[5, 2] = np.nan
    out = impute(x.copy(), model={'model': 'SimpleImputer', 'kwargs': {}},
                 strategy='constant', fill_value=-99.0)
    # pre-fix: the outer kwargs were silently dropped for this spec form
    assert np.asarray(out)[5, 2] == -99.0


def test_instance_spec_with_kwargs_warns():
    from hypertools.impute import SimpleImputer
    rng = np.random.RandomState(2)
    x = rng.randn(40, 4)
    x[5, 2] = np.nan
    with pytest.warns(UserWarning, match='ignoring keyword'):
        impute(x.copy(), model=SimpleImputer(), strategy='median')


# --- F17-impute-008 / X2-error-quality-004: degenerate inputs ----------------

def test_none_input_clear_error():
    with pytest.raises(TypeError, match='Unsupported data type'):
        impute(None)


def test_empty_list_clear_error_never_reaches_text_pipeline():
    with pytest.raises(ValueError, match='no observations'):
        impute([])


def test_empty_array_clear_error():
    with pytest.raises(ValueError, match='no observations'):
        impute(np.empty((0, 3)))


def test_all_nan_dataset_clear_error():
    with pytest.raises(ValueError, match='entirely missing'):
        impute(np.full((10, 3), np.nan), model='PPCA')


def test_scalar_input_clear_error():
    with pytest.raises(ValueError, match='scalar'):
        impute(5)


def test_imputer_fit_none_raises_real_valueerror():
    from hypertools.impute import SimpleImputer
    with pytest.raises(ValueError, match='empty dataset'):
        SimpleImputer().fit(None)


# --- F17-impute-010: 1-D input is a univariate series ------------------------

def test_1d_input_is_univariate_series_filled_with_series_statistic():
    out = np.asarray(impute(np.array([1.0, np.nan, 3.0, 4.0]),
                            model='SimpleImputer'))
    # pre-fix: shape (1, 4) with the NaN zero-filled
    assert out.shape == (4, 1)
    assert out[1, 0] == pytest.approx((1.0 + 3.0 + 4.0) / 3.0)


def test_series_input_matches_1d_convention():
    s = pd.Series([1.0, np.nan, 3.0, 4.0])
    out = np.asarray(impute(s, model='SimpleImputer'))
    assert out.shape == (4, 1)
    assert out[1, 0] == pytest.approx((1.0 + 3.0 + 4.0) / 3.0)


# --- F17-impute-001: mismatched columns are never widened --------------------

def test_mismatched_columns_impute_independently_with_warning():
    rng = np.random.RandomState(3)
    a = pd.DataFrame(rng.randn(12, 4) + 5, columns=list('wxyz'))
    a.iloc[3, 1] = np.nan
    b = pd.DataFrame(rng.randn(10, 3) - 5, columns=list('wxy'))
    b.iloc[2, 0] = np.nan
    with pytest.warns(UserWarning, match='share columns'):
        res = impute([a, b], model='SimpleImputer')
    # pre-fix: [(12, 4), (10, 4)] with an invented constant 'z' column
    assert [r.shape for r in res] == [(12, 4), (10, 3)]
    assert list(res[1].columns) == list('wxy')
    # each dataset is filled from ITS OWN statistics
    assert np.asarray(res[1])[2, 0] == pytest.approx(
        b['w'].dropna().mean())
    assert not any(r.isna().any().any() for r in res)


def test_mismatched_width_arrays_keep_their_shapes():
    rng = np.random.RandomState(4)
    a, b = rng.randn(10, 3), rng.randn(10, 4)
    a[2, 1] = np.nan
    with pytest.warns(UserWarning, match='share columns'):
        res = impute([a, b], model='SimpleImputer')
    assert [r.shape for r in res] == [(10, 3), (10, 4)]


def test_shared_columns_still_imputed_jointly():
    rng = np.random.RandomState(5)
    a = pd.DataFrame(rng.randn(30, 4), columns=list('wxyz'))
    b = pd.DataFrame(rng.randn(20, 4), columns=list('wxyz'))
    a.iloc[3, 1] = np.nan
    res = impute([a, b], model='SimpleImputer')
    pooled_mean = pd.concat([a, b])['x'].dropna().mean()
    assert np.asarray(res[0])[3, 1] == pytest.approx(pooled_mean)


# --- F17-impute-012: list path preserves each dataset's index ----------------

def test_list_of_dataframes_keeps_indexes():
    rng = np.random.RandomState(2)
    a = pd.DataFrame(rng.randn(30, 4), columns=list('wxyz'), index=range(100, 130))
    a.iloc[3, 1] = np.nan
    b = pd.DataFrame(rng.randn(25, 4), columns=list('wxyz'), index=range(500, 525))
    b.iloc[10, 2] = np.nan
    res = impute([a, b], model='SimpleImputer')
    assert list(res[0].index) == list(range(100, 130))
    assert list(res[1].index) == list(range(500, 525))
    assert list(res[0].columns) == list('wxyz')


# --- F17-impute-011: rich model-spec errors ----------------------------------

def test_raw_sklearn_imputer_class_error_names_the_wrapper():
    import sklearn.impute as ski
    _, x, _ = _rank3_benchmark()
    with pytest.raises(ValueError, match='hypertools.impute.SimpleImputer'):
        impute(x.copy(), model=ski.SimpleImputer)


def test_unknown_name_error_recommends_canonical_dict_form():
    _, x, _ = _rank3_benchmark()
    with pytest.raises(ValueError, match="'kwargs'"):
        impute(x.copy(), model='banana')


def test_dict_spec_missing_model_key_clear_error():
    _, x, _ = _rank3_benchmark()
    with pytest.raises(ValueError, match="'model' key"):
        impute(x.copy(), model={'params': {}})


# --- F17-impute-014: PPCA d validation ---------------------------------------

@pytest.mark.parametrize('bad_d', [999, 1, 0, -3, 2.5])
def test_ppca_invalid_d_clear_error(bad_d):
    rng = np.random.RandomState(2)
    x = rng.randn(40, 4)
    x[5, 2] = np.nan
    with pytest.raises(ValueError, match='latent dimensions'):
        impute(x, model='PPCA', d=bad_d)


# --- F17-impute-015: inf pre-check -------------------------------------------

def test_inf_input_clear_error():
    rng = np.random.RandomState(1)
    x = rng.randn(40, 4)
    x[0, 0] = np.inf
    x[1, 1] = np.nan
    for model in ('SimpleImputer', 'PPCA'):
        with pytest.raises(ValueError, match='infinite values'):
            impute(x.copy(), model=model)
