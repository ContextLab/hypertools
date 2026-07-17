"""Regression tests for the 2026-07 release audit findings on hypertools.align
(unit F12-align + the align part of X2-error-quality-003). Real data, real
calls, deterministic seeds -- no mocks.

Covered findings:
- F12-align-001 (critical): trim_and_pad silently scrambled row order for any
  non-RangeIndex index (DatetimeIndex, strings, shuffled ints).
- F12-align-002 (doc): SRM-family `features` default docstring said "minimum
  number of columns" but the actual default is the padded common (max) width.
- F12-align-003 (minor): the "keeps only N row(s)" trim warning fired twice
  per hyp.align() call.
- F12-align-004 (minor): hyp.align(single bare array) returned a list-of-one
  instead of a bare array (docstring promises reduce-style shape matching).
- F12-align-006 (minor): hyp.align([]) fell into the text-corpus funnel and
  died with a cryptic LatentDirichletAllocation error.
- F12-align-007 (minor): HyperAlign n_iter validated with assert (stripped
  under -O); float n_iter silently truncated.
- F12-align-008 (minor): Procrustes error messages contained giant embedded
  whitespace runs / dev-jargon wording.
- F12-align-009 (minor): zero index overlap yielded silent empty (0, k) output.
- X2-error-quality-003 (align part): misspelled constructor kwargs (e.g.
  n_itr=) were silently swallowed, so typo'd analysis parameters were ignored.
- Binding contract: model=False / model=None (and the deprecated align= alias)
  are no-ops returning the data unchanged.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.align.common import trim_and_pad
from hypertools.align.hyperalign import HyperAlign
from hypertools.align.null import NullAlign
from hypertools.align.srm import (SharedResponseModel,
                                  DeterministicSharedResponseModel,
                                  RobustSharedResponseModel)


def _identifiable(n=10, k=3):
    """Rows are identifiable: column 0 of row i equals i."""
    base = np.zeros((n, k))
    base[:, 0] = np.arange(n)
    return base


# --- F12-align-001: row (observation/time) order preservation ---------------

def test_align_preserves_datetimeindex_row_order():
    """CRITICAL regression: DatetimeIndex (fMRI/timeseries-style) input must
    come back in input/time order for EVERY output dataset -- previously
    list(set(...)) hash order silently scrambled the rows."""
    n = 10
    tidx = pd.date_range('2020-01-01', periods=n, freq='s')
    d0 = pd.DataFrame(_identifiable(n), index=tidx)
    d1 = pd.DataFrame(_identifiable(n), index=tidx)
    out = hyp.align([d0, d1], model='NullAlign')
    for o in out:
        assert list(np.asarray(o)[:, 0].astype(int)) == list(range(n))


def test_align_preserves_string_index_row_order():
    n = 12
    idx = [f't{i:02d}' for i in range(n)]
    d0 = pd.DataFrame(_identifiable(n), index=idx)
    d1 = pd.DataFrame(_identifiable(n), index=idx)
    out = hyp.align([d0, d1], model='NullAlign')
    for o in out:
        assert list(np.asarray(o)[:, 0].astype(int)) == list(range(n))


def test_align_preserves_shuffled_integer_index_row_order():
    """Non-contiguous / shuffled integer index: output must follow the FIRST
    dataset's index order (deterministic), not hash order."""
    n = 10
    rng = np.random.RandomState(0)
    idx = list(rng.permutation(np.arange(100, 100 + n)))
    d0 = pd.DataFrame(_identifiable(n), index=idx)
    d1 = pd.DataFrame(_identifiable(n), index=idx)
    out = hyp.align([d0, d1], model='NullAlign')
    for o in out:
        assert list(np.asarray(o)[:, 0].astype(int)) == list(range(n))


def test_trim_keeps_first_dataset_order_on_partial_overlap():
    """When trimming to common rows, the kept rows must appear in the FIRST
    dataset's index order."""
    n = 12
    tidx = pd.date_range('2021-06-01', periods=n, freq='h')
    d0 = pd.DataFrame(_identifiable(n), index=tidx)
    # second dataset only shares timepoints 3..11
    d1 = pd.DataFrame(_identifiable(n)[3:], index=tidx[3:])
    with pytest.warns(UserWarning, match='keeps only'):
        out = hyp.align([d0, d1], model='NullAlign')
    for o in out:
        assert list(np.asarray(o)[:, 0].astype(int)) == list(range(3, n))


def test_trim_and_pad_order_preserving_directly():
    n = 8
    idx = [f'obs{i}' for i in range(n)]
    dfs = [pd.DataFrame(_identifiable(n, 2), index=idx) for _ in range(3)]
    out = trim_and_pad(dfs)
    for o in out:
        assert list(o.index) == idx
        assert list(np.asarray(o)[:, 0].astype(int)) == list(range(n))


def test_hyperalign_row_order_preserved_end_to_end():
    """Order preservation must hold through a real (non-null) aligner too:
    align 3 rotated copies of an identifiable trajectory on a DatetimeIndex
    and confirm output rows follow the input time order (monotone col 0 of
    the common trajectory reconstructed via near-perfect alignment)."""
    n = 30
    rng = np.random.RandomState(0)
    t = np.linspace(0, 4 * np.pi, n)
    common = np.stack([t / 10, np.sin(t), np.cos(t)], axis=1)
    tidx = pd.date_range('2022-01-01', periods=n, freq='min')
    dsets = []
    for seed in range(3):
        r = np.random.RandomState(seed)
        rot, _ = np.linalg.qr(r.randn(3, 3))
        dsets.append(pd.DataFrame(common @ rot, index=tidx))
    out = hyp.align(dsets, model='HyperAlign')
    # all datasets share identical (rotated) content, so aligned outputs
    # must be identical row-for-row; the row ORDER must match input order:
    # reconstruct by comparing against the aligned version of dataset 0
    # computed with a plain RangeIndex (known-good order).
    ref = hyp.align([pd.DataFrame(np.asarray(d)) for d in dsets],
                    model='HyperAlign')
    for o, r_ in zip(out, ref):
        assert np.allclose(np.asarray(o), np.asarray(r_), atol=1e-8)


# --- binding contract: model=False / model=None are no-ops ------------------

@pytest.mark.parametrize('noop', [False, None])
def test_align_model_false_and_none_are_noops(noop):
    rng = np.random.RandomState(0)
    d0, d1 = rng.randn(20, 4), rng.randn(20, 4)
    out = hyp.align([d0, d1], model=noop)
    assert isinstance(out, list) and len(out) == 2
    for o, d in zip(out, (d0, d1)):
        assert np.allclose(np.asarray(o), d)


@pytest.mark.parametrize('noop', [False, None])
def test_align_deprecated_align_kwarg_false_and_none_are_noops(noop):
    rng = np.random.RandomState(1)
    d0, d1 = rng.randn(15, 3), rng.randn(15, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        out = hyp.align([d0, d1], align=noop)
    assert isinstance(out, list) and len(out) == 2
    for o, d in zip(out, (d0, d1)):
        assert np.allclose(np.asarray(o), d)


def test_align_false_no_op_returns_model_none():
    rng = np.random.RandomState(2)
    d0, d1 = rng.randn(10, 3), rng.randn(10, 3)
    out, model = hyp.align([d0, d1], model=False, return_model=True)
    assert model is None
    for o, d in zip(out, (d0, d1)):
        assert np.allclose(np.asarray(o), d)


def test_align_model_true_raises_clear_error():
    rng = np.random.RandomState(3)
    with pytest.raises(ValueError, match='model=True'):
        hyp.align([rng.randn(10, 3), rng.randn(10, 3)], model=True)


# --- F12-align-003: trim warning fires exactly once per call ----------------

def test_trim_warning_emitted_once_per_align_call():
    rng = np.random.RandomState(0)
    ds = [rng.randn(40, 5), rng.randn(30, 5)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hyp.align(ds, model='NullAlign')
    assert sum('keeps only' in str(x.message) for x in w) == 1


def test_trim_warning_still_fires_on_held_out_transform():
    """A reused fitted model transforming NEW ragged data must still warn."""
    rng = np.random.RandomState(0)
    train = [pd.DataFrame(rng.randn(30, 4)), pd.DataFrame(rng.randn(30, 4))]
    model = NullAlign()
    model.fit(train)
    held_out = [pd.DataFrame(rng.randn(25, 4)),
                pd.DataFrame(rng.randn(20, 4))]
    with pytest.warns(UserWarning, match='keeps only'):
        model.transform(held_out)


# --- F12-align-004: single bare array returns a bare array ------------------

def test_align_single_bare_array_returns_bare_array():
    X = np.random.RandomState(0).randn(30, 5)
    out = hyp.align(X, model='NullAlign')
    assert isinstance(out, np.ndarray)
    assert out.shape == (30, 5)
    assert np.allclose(out, X)


def test_align_list_input_still_returns_list():
    rng = np.random.RandomState(0)
    ds = [rng.randn(20, 4), rng.randn(20, 4)]
    out = hyp.align(ds, model='NullAlign')
    assert isinstance(out, list) and len(out) == 2


# --- F12-align-006: empty list rejected up front -----------------------------

def test_align_empty_list_raises_clear_error():
    with pytest.raises(ValueError, match='empty'):
        hyp.align([], model='HyperAlign')


# --- F12-align-007: n_iter validation ----------------------------------------

def test_hyperalign_negative_n_iter_raises_valueerror():
    rng = np.random.RandomState(0)
    ds = [rng.randn(20, 4) for _ in range(3)]
    with pytest.raises(ValueError, match='n_iter'):
        hyp.align(ds, model='HyperAlign', n_iter=-1)


def test_hyperalign_float_n_iter_raises_valueerror():
    rng = np.random.RandomState(0)
    ds = [rng.randn(20, 4) for _ in range(3)]
    with pytest.raises(ValueError, match='n_iter'):
        hyp.align(ds, model='HyperAlign', n_iter=2.5)


def test_hyperalign_bool_n_iter_raises_valueerror():
    with pytest.raises(ValueError, match='n_iter'):
        HyperAlign(n_iter=True)


def test_hyperalign_numpy_integer_n_iter_accepted():
    rng = np.random.RandomState(0)
    ds = [rng.randn(20, 4) for _ in range(2)]
    out = hyp.align(ds, model='HyperAlign', n_iter=np.int64(2))
    assert len(out) == 2


# --- F12-align-008: Procrustes error message quality -------------------------

def test_procrustes_reduction_error_message_clean_and_actionable():
    rng = np.random.RandomState(0)
    src = pd.DataFrame(rng.randn(20, 8))
    tgt = pd.DataFrame(rng.randn(20, 3))
    with pytest.raises(ValueError) as excinfo:
        hyp.align([src], model='Procrustes', target=tgt, reduction=False)
    msg = str(excinfo.value)
    assert '  ' not in msg  # no embedded whitespace runs
    assert 'reduction=True' in msg  # names the fix
    assert '8' in msg and '3' in msg  # names the offending shapes


def test_procrustes_invariant_dataset_error_message():
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError) as excinfo:
        hyp.align([np.zeros((20, 4)), rng.randn(20, 4)], model='Procrustes')
    msg = str(excinfo.value)
    assert 'variance' in msg  # names the problem in user terms
    assert 'For now do not handle' not in msg  # dev-jargon phrasing removed
    assert '  ' not in msg


# --- F12-align-009: zero index overlap raises instead of empty output --------

def test_align_zero_index_overlap_raises():
    rng = np.random.RandomState(0)
    a = pd.DataFrame(rng.randn(4, 4), index=[0, 1, 2, 3])
    b = pd.DataFrame(rng.randn(4, 4), index=[10, 11, 12, 13])
    with pytest.raises(ValueError, match='common'):
        hyp.align([a, b], model='HyperAlign')


# --- X2-error-quality-003 (align part): misspelled kwargs must raise ---------

def test_align_misspelled_kwarg_raises_typeerror_naming_it():
    rng = np.random.RandomState(0)
    X, Y = rng.rand(20, 4), rng.rand(20, 4)
    with pytest.raises(TypeError, match='n_itr'):
        hyp.align([X, Y], n_itr=5)  # typo for n_iter


def test_srm_misspelled_kwarg_raises_typeerror_naming_it():
    rng = np.random.RandomState(0)
    X, Y = rng.rand(20, 4), rng.rand(20, 4)
    with pytest.raises(TypeError, match='featurse'):
        hyp.align([X, Y], model='SRM', featurse=3)  # typo for features


def test_nullalign_unknown_kwarg_raises():
    with pytest.raises(TypeError, match='bogus_kwarg'):
        hyp.align([np.random.RandomState(0).rand(10, 3)] * 2,
                  model='NullAlign', bogus_kwarg=1)


def test_dict_spec_misspelled_kwarg_raises():
    rng = np.random.RandomState(0)
    X, Y = rng.rand(20, 4), rng.rand(20, 4)
    with pytest.raises(TypeError, match='n_itr'):
        hyp.align([X, Y], model={'model': 'HyperAlign',
                                 'kwargs': {'n_itr': 5}})


def test_correctly_spelled_kwargs_still_work():
    rng = np.random.RandomState(0)
    X, Y = rng.rand(20, 4), rng.rand(20, 4)
    out, model = hyp.align([X, Y], model='HyperAlign', n_iter=3,
                           return_model=True)
    assert model.kwargs['n_iter'] == 3
    out, model = hyp.align([X, Y], model='SRM', features=2,
                           return_model=True)
    assert model.kwargs['features'] == 2


# --- F12-align-002: SRM features default doc matches actual behavior ---------

def test_srm_features_default_is_padded_common_width():
    """Behavioral pin: unequal-width datasets are zero-padded to the MAX
    width before fitting, so the default `features` is that padded common
    width (8 here), not the pre-pad minimum (4)."""
    rng = np.random.RandomState(0)
    ds = [rng.randn(30, 6), rng.randn(30, 4), rng.randn(30, 8)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out, m = hyp.align(ds, model='SRM', return_model=True)
    assert m.features == 8  # fitted default = padded common (max) width
    assert np.asarray(out[0]).shape[1] == 8


@pytest.mark.parametrize('cls', [SharedResponseModel,
                                 DeterministicSharedResponseModel,
                                 RobustSharedResponseModel])
def test_srm_docstring_no_longer_claims_minimum_columns(cls):
    assert 'minimum number of' not in cls.__doc__
    assert 'maximum' in cls.__doc__ or 'padded' in cls.__doc__


# --- F12-align-005: documented import path for aligner classes ---------------

def test_aligner_classes_importable_from_align_subpackage():
    from hypertools.align import (HyperAlign as H, Procrustes as P,
                                  SharedResponseModel as S, NullAlign as N)
    from hypertools.align.common import Aligner
    for cls in (H, P, S, N):
        assert issubclass(cls, Aligner)


def test_align_docstring_documents_class_import_path():
    assert 'from hypertools.align import' in hyp.align.__doc__
