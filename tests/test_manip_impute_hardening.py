"""manip / normalize / impute robustness (QC 2026-07 release hunt).

- impute of an all-missing column crashed (SimpleImputer/KNN/Iterative);
- ZScore/Normalize of a constant column produced all-NaN silently;
- PPCA impute returned the LATENT PCA scores (rotating observed values / dropping
  columns) instead of the imputed data;
- manip() lacked the cross-module stage kwargs normalize()/analyze() accept;
- Smooth surfaced raw scipy errors for bad kernel_width.

Real data + sklearn/numeric cross-checks, no mocks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.manip.manip import manip as manipulate


def _rng():
    return np.random.default_rng(0)


# --- impute all-missing column (M1) ------------------------------------

@pytest.mark.parametrize('model', ['SimpleImputer', 'KNNImputer', 'IterativeImputer'])
def test_impute_all_missing_column_no_crash(model):
    df = pd.DataFrame(_rng().normal(size=(40, 4)), columns=list('abcd'))
    df['b'] = np.nan
    # the all-missing column deliberately provokes the no-observed-values
    # notice
    with pytest.warns(UserWarning, match='no observed values at all'):
        out = np.asarray(hyp.impute(df, model=model))
    assert out.shape == (40, 4)
    assert np.isnan(out).sum() == 0
    assert np.allclose(out[:, 1], 0.0)  # all-missing column filled with 0


# --- constant-column ZScore / Normalize (M2) ---------------------------

def test_zscore_constant_column_is_zero_not_nan():
    df = pd.DataFrame(_rng().normal(size=(30, 3)), columns=list('abc'))
    df['a'] = 5.0
    z = np.asarray(manipulate(df, model='ZScore'))
    assert not np.isnan(z).any()
    assert np.allclose(z[:, 0], 0.0)


def test_normalize_manip_constant_column_finite():
    df = pd.DataFrame(_rng().normal(size=(30, 3)), columns=list('abc'))
    df['a'] = 3.0
    assert np.isfinite(np.asarray(manipulate(df, model='Normalize'))).all()


# --- PPCA impute contract (M3 + observed-value preservation) -----------

def test_ppca_impute_preserves_observed_values_and_shape():
    x = _rng().normal(size=(50, 5))
    xm = x.copy()
    mask = _rng().random(x.shape) < 0.1
    xm[mask] = np.nan
    out = np.asarray(hyp.impute(xm, model='PPCA'))
    assert out.shape == (50, 5)
    assert np.allclose(out[~mask], x[~mask], atol=1e-9)  # observed preserved
    assert np.isnan(out).sum() == 0


def test_ppca_impute_rank_deficient_preserves_shape():
    base = _rng().normal(size=(50, 2))
    x = base @ _rng().normal(size=(2, 5))  # rank-2, 5 columns
    xm = x.copy()
    xm[_rng().random(x.shape) < 0.1] = np.nan
    out = np.asarray(hyp.impute(xm, model='PPCA'))
    assert out.shape == (50, 5)  # was (50, 2) -- latent scores


def test_ppca_impute_dropped_column_is_dense_and_feeds_reduce():
    """Red-team regression: a column with < min_obs observations is DROPPED by
    PPCA; the splice used to leave its missing entries NaN, which then crashed
    hyp.reduce/hyp.plot with 'Input X contains NaN'. Dropped columns must be
    densified (observed-mean fill) while observed values stay exact."""
    x = _rng().normal(size=(40, 5))
    xm = x.copy()
    xm[5:, 2] = np.nan  # column 2 keeps only 5 (< min_obs=10) observations
    mask = np.isnan(xm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = np.asarray(hyp.impute(xm, model='PPCA'))
        red = np.asarray(hyp.reduce(xm, reduce='PCA', ndims=2))
    assert out.shape == (40, 5)
    assert np.isnan(out).sum() == 0            # dense: dropped column filled
    assert np.allclose(out[~mask], x[~mask], atol=1e-9)  # observed preserved
    assert red.shape == (40, 2)                # no longer crashes downstream


def test_ppca_impute_single_column_clear_error():
    """A single (or sub-2) valid column can't support PPCA's cross-column model;
    it used to crash with a cryptic LinAlgError ('0-dimensional array')."""
    df = pd.DataFrame({'a': ([1.0, 2.0, np.nan, 4.0, 5.0] * 4)})
    # the NaN cells sit in single-column rows with nothing else observed,
    # deliberately provoking the cannot-fill notice before the ValueError
    with pytest.warns(UserWarning, match='cannot fill'), \
         pytest.raises(ValueError, match='at least 2 columns'):
        hyp.impute(df, model='PPCA')


# --- manip cross-module stage kwargs (M4) ------------------------------

def test_manip_accepts_cross_module_stage_kwargs():
    x = np.cumsum(_rng().normal(size=(60, 5)), axis=0)
    out = manipulate(x, model='ZScore', reduce='PCA', ndims=2)
    assert np.asarray(out).shape == (60, 2)
    out2, model = manipulate(x, model='Smooth', kernel_width=15, reduce='PCA',
                             ndims=3, return_model=True)
    assert np.asarray(out2).shape == (60, 3)
    assert [n for n, _ in model.steps] == ['manip', 'reduce']


# --- Smooth kernel_width errors (M5) -----------------------------------

def test_smooth_kernel_width_too_large_clear_error():
    x = np.cumsum(_rng().normal(size=(60, 3)), axis=0)
    # the even kernel_width deliberately provokes the must-be-odd notice
    # before the too-large ValueError
    with pytest.warns(UserWarning, match='Increasing smoothing kernel width'), \
         pytest.raises(ValueError, match='larger than the number of samples'):
        manipulate(x, model='Smooth', kernel_width=200)


def test_smooth_savgol_kernel_width_le_order_clear_error():
    x = np.cumsum(_rng().normal(size=(60, 3)), axis=0)
    with pytest.raises(ValueError, match='kernel_width'):
        manipulate(x, model='Smooth', kernel='savgol', kernel_width=3)
