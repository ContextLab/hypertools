# -*- coding: utf-8 -*-
"""Regression tests for the F15-analyze audit findings (release-1.0 QC,
2026-07): analyze()'s stage routing/validation, the pipeline= reuse
contract, and impute=/ndims= handling. Real data, real calls, no mocks.
"""
import warnings

import numpy as np
import pytest

import hypertools as hyp
from hypertools.tools.analyze import analyze


def _rng(seed=1):
    return np.random.default_rng(seed)


def _two(seed=1, shape=(30, 6)):
    rng = _rng(seed)
    return [rng.standard_normal(shape) for _ in range(2)]


def _allclose_lists(a, b):
    return all(np.allclose(x, y) for x, y in zip(a, b))


# --- F15-analyze-001: False disables a stage, True raises curated error ----

def test_align_false_skips_alignment_legacy_path():
    data = _two()
    out_false = analyze(data, normalize='within', align=False)
    out_none = analyze(data, normalize='within', align=None)
    assert _allclose_lists(out_false, out_none)


def test_align_false_skips_alignment_pipeline_path():
    data = _two()
    out, model = analyze(data, normalize='within', align=False,
                         return_model=True)
    assert 'align' not in [name for name, _ in model.steps]
    assert _allclose_lists(out, analyze(data, normalize='within'))


def test_align_true_curated_error_both_paths():
    data = _two()
    with pytest.raises(ValueError, match='align=True was removed'):
        analyze(data, align=True)
    with pytest.raises(ValueError, match='align=True was removed'):
        analyze(data, align=True, return_model=True)


def test_reduce_false_and_cluster_false_skip_stage():
    data = _two()
    out = analyze(data, normalize='across', reduce=False)
    assert _allclose_lists(out, analyze(data, normalize='across'))
    out2, model = analyze(data, normalize='across', cluster=False,
                          return_model=True)
    assert 'cluster' not in [name for name, _ in model.steps]
    assert _allclose_lists(out2, analyze(data, normalize='across'))


# --- F15-analyze-002: no-transform reducers (TSNE/MDS) + cluster= ----------

def test_tsne_plus_cluster_returns_embedded_data():
    small = _two(seed=7, shape=(25, 5))
    out, model = analyze(small,
                         reduce={'model': 'TSNE', 'kwargs': {'perplexity': 5}},
                         ndims=2,
                         cluster={'model': 'KMeans',
                                  'kwargs': {'n_clusters': 2, 'n_init': 10}},
                         random_state=0, return_model=True)
    assert [np.asarray(o).shape for o in out] == [(25, 2), (25, 2)]
    labels = np.asarray(model.named_steps['cluster'].transform(out))
    assert labels.shape == (50,)
    assert set(np.unique(labels)) == {0, 1}
    # and without return_model (same code path must not crash either)
    out2 = analyze(small,
                   reduce={'model': 'TSNE', 'kwargs': {'perplexity': 5}},
                   ndims=2,
                   cluster={'model': 'KMeans',
                            'kwargs': {'n_clusters': 2, 'n_init': 10}},
                   random_state=0)
    assert [np.asarray(o).shape for o in out2] == [(25, 2), (25, 2)]


# --- F15-analyze-003: pipeline= reuse of a cluster-bearing model -----------

def test_pipeline_reuse_with_cluster_returns_data_not_labels():
    rng = _rng(0)
    train, test = rng.standard_normal((60, 10)), rng.standard_normal((60, 10))
    out_tr, model = analyze(
        train, normalize='across', reduce='PCA', ndims=2,
        cluster={'model': 'KMeans',
                 'kwargs': {'n_clusters': 3, 'n_init': 10, 'random_state': 0}},
        return_model=True)
    out_te = analyze(test, pipeline=model)
    assert np.asarray(out_tr).shape == (60, 2)
    assert np.asarray(out_te).shape == (60, 2)
    # the reuse output must equal the fitted non-cluster steps applied in order
    expected = model.named_steps['reduce'].transform(
        model.named_steps['normalize'].transform(test))
    assert np.allclose(np.asarray(out_te), np.asarray(expected))
    # labels stay recoverable via the documented recipe
    labels = np.asarray(model.named_steps['cluster'].transform(out_te))
    assert labels.shape == (60,)


# --- F15-analyze-004: pipeline= honors impute=, warns on ndims= ------------

def test_pipeline_path_honors_impute():
    data = _two(seed=7)
    nan_data = [x.copy() for x in data]
    nan_data[0][3, 2] = np.nan
    nan_data[1][10, 5] = np.nan
    _, m = analyze(data, normalize='across', reduce='PCA', ndims=2,
                   return_model=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ra = analyze(nan_data, pipeline=m, impute='Kalman')
        rb = analyze(nan_data, pipeline=m)
    assert not _allclose_lists(ra, rb)  # Kalman fill != PPCA fill
    assert not any(np.isnan(np.asarray(a)).any() for a in ra)


def test_pipeline_path_warns_that_ndims_is_ignored():
    data = _two()
    _, m = analyze(data, normalize='across', reduce='PCA', ndims=2,
                   return_model=True)
    with pytest.warns(UserWarning, match='ndims'):
        out = analyze(data, pipeline=m, ndims=4)
    assert [np.asarray(o).shape[1] for o in out] == [2, 2]


# --- F15-analyze-011: pipeline= type validation -----------------------------

def test_pipeline_wrong_type_raises_named_typeerror():
    with pytest.raises(TypeError, match='pipeline='):
        analyze(np.random.rand(10, 3), pipeline='PCA')


# --- F15-analyze-017: impute= honored without normalize= --------------------

def test_impute_honored_on_reduce_only_path():
    nan_data = _two(seed=7)
    nan_data[0][3, 2] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r1 = analyze(nan_data, reduce='PCA', ndims=2, impute='Kalman')
        r2 = analyze(nan_data, reduce='PCA', ndims=2)
    assert not _allclose_lists(r1, r2)  # Kalman fill != PPCA fill
    assert not any(np.isnan(np.asarray(a)).any() for a in r1)


def test_impute_with_normalize_unchanged_and_consistent_across_paths():
    nan_data = _two(seed=7)
    nan_data[0][3, 2] = np.nan
    legacy = analyze(nan_data, normalize='across', impute='Kalman')
    built, _ = analyze(nan_data, normalize='across', impute='Kalman',
                       return_model=True)
    assert _allclose_lists(legacy, built)
    assert not any(np.isnan(np.asarray(a)).any() for a in legacy)


# --- F15-analyze-006: ndims without reduce warns (was a silent no-op) ------

def test_ndims_without_reduce_warns_and_passes_through():
    data = _two(seed=7)
    with pytest.warns(UserWarning, match='ndims'):
        out = analyze(data, ndims=3)
    assert [o.shape for o in out] == [(30, 6), (30, 6)]


# --- binding contract: staged order normalize -> reduce -> align -----------

def test_staged_order_normalize_then_reduce_then_align():
    data = _two(seed=2)
    combo = analyze(data, normalize='across', reduce='PCA', ndims=3,
                    align='hyper')
    # the staged hyp.align call passes the legacy 'hyper' alias to model=
    # directly, deliberately provoking its deprecation notice (the analyze()
    # stage-kwarg path above resolves the alias without warning)
    with pytest.warns(DeprecationWarning,
                      match="'hyper' is a deprecated alias"):
        staged = hyp.align(
            hyp.reduce(hyp.normalize(data, normalize='across'),
                       reduce='PCA', ndims=3),
            model='hyper')
    assert _allclose_lists(combo, staged)


# --- binding contract: unknown stage kwargs raise with the kwarg name ------

def test_unknown_kwarg_raises_typeerror_naming_it():
    data = _two()
    with pytest.raises(TypeError, match='foo'):
        analyze(data, foo=1)
    with pytest.raises(TypeError, match='model'):
        analyze(data, model='PCA')


# --- binding contract: empty input raises a clear no-data error ------------

def test_empty_list_raises_no_data_error(capsys):
    with pytest.raises(ValueError, match='no observations'):
        analyze([], normalize='across')
    captured = capsys.readouterr()
    assert 'loading corpus' not in captured.out


# --- F15-analyze-012: 3-D input raises a clear shape error -----------------

def test_3d_array_raises_clear_shape_error():
    with pytest.raises(ValueError, match='2-D'):
        analyze(np.zeros((4, 5, 6)), normalize='across')
    with pytest.raises(ValueError, match='2-D'):
        analyze(np.arange(120.).reshape(4, 5, 6), normalize='across')


# --- F15-analyze-006/015: docstring accuracy --------------------------------

def test_docstring_mentions_random_state_and_correct_reduce_default():
    doc = ' '.join(analyze.__doc__.split())
    assert 'random_state' in doc
    assert "(default: 'IncrementalPCA')" not in doc
