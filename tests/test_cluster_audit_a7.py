# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release-audit fixes to hypertools.cluster
(audit unit F13-cluster, fix batch A7-cluster-reduce).

Real data and real scikit-learn models throughout -- no mocks.
"""
import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans

import hypertools as hyp
from hypertools.cluster.cluster import cluster
from hypertools.cluster.common import Clusterer


def _blobs(n_per=50, cols=4, k=2, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal((n_per, cols)) + 8.0 * i
                      for i in range(k)])


# --- F13-cluster-001: FeatureAgglomeration clusters FEATURES ---------------

def test_feature_agglomeration_warns_and_returns_per_feature_labels():
    X = _blobs(100, 5, 3)
    with pytest.warns(UserWarning, match='clusters features'):
        labels = cluster(X, cluster='FeatureAgglomeration', n_clusters=3)
    # one label per COLUMN, by (now documented) design
    assert len(labels) == 5


def test_feature_agglomeration_reuse_same_data_returns_feature_labels():
    X = _blobs(60, 5, 2)
    with pytest.warns(UserWarning, match='clusters features'):
        labels, model = cluster(X, cluster='FeatureAgglomeration',
                                n_clusters=2, return_model=True)
    # reuse on the SAME data must return the same per-feature labels, not the
    # old misleading "new data (a different number of rows)" error
    again = cluster(X, cluster=model)
    assert list(again) == list(labels)


def test_feature_agglomeration_reuse_different_width_raises_clearly():
    X = _blobs(60, 5, 2)
    with pytest.warns(UserWarning, match='clusters features'):
        _, model = cluster(X, cluster='FeatureAgglomeration', n_clusters=2,
                           return_model=True)
    with pytest.raises(NotImplementedError, match='column'):
        model.transform(np.random.default_rng(1).standard_normal((60, 7)))


def test_cluster_docstring_documents_feature_agglomeration_semantics():
    assert 'FeatureAgglomeration' in cluster.__doc__
    assert 'column' in cluster.__doc__


# --- F13-cluster-006: no-predict reuse must not silently alias stale labels

def test_no_predict_reuse_same_data_is_silent_and_correct():
    X = _blobs(30, 4, 2, seed=5)
    labels, model = cluster(X, cluster='AgglomerativeClustering',
                            n_clusters=2, return_model=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        again = cluster(X, cluster=model)
    assert list(again) == list(labels)


def test_no_predict_reuse_different_data_same_rows_warns():
    rng = np.random.default_rng(5)
    A = np.vstack([rng.standard_normal((30, 4)) + 6 * i for i in range(2)])
    labels_A, model = cluster(A, cluster='AgglomerativeClustering',
                              n_clusters=2, return_model=True)
    B = rng.standard_normal((60, 4)) * 50 + 100  # unrelated, same row count
    with pytest.warns(UserWarning, match='fit-time'):
        labels_B = cluster(B, cluster=model)
    # documented recovery behavior: the stored fit-time labels come back
    assert list(labels_B) == list(labels_A)


def test_no_predict_reuse_different_row_count_still_raises():
    A = _blobs(30, 4, 2, seed=5)
    _, model = cluster(A, cluster='AgglomerativeClustering', n_clusters=2,
                       return_model=True)
    with pytest.raises(NotImplementedError):
        model.transform(np.random.default_rng(2).standard_normal((10, 4)))


# --- F13-cluster-008: silent conflict resolution now warns ------------------

def test_instance_spec_with_conflicting_n_clusters_warns():
    X = _blobs(75, 4, 2, seed=1)
    with pytest.warns(UserWarning, match='n_clusters=3'):
        labels = cluster(X, cluster=KMeans(n_clusters=5, random_state=0,
                                           n_init=10), n_clusters=3)
    assert len(set(labels)) == 5  # the instance still wins (documented)


def test_instance_spec_without_explicit_n_clusters_no_warning():
    X = _blobs(75, 4, 2, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        labels = cluster(X, cluster=KMeans(n_clusters=2, random_state=0,
                                           n_init=10))
    assert len(set(labels)) == 2


def test_dict_with_both_params_and_kwargs_warns_params_ignored():
    X = _blobs(75, 4, 3, seed=1)
    with pytest.warns(UserWarning, match="'params'"):
        labels = cluster(X, cluster={'model': 'KMeans',
                                     'params': {'n_clusters': 2},
                                     'kwargs': {'n_clusters': 4,
                                                'random_state': 0}})
    assert len(set(labels)) == 4  # kwargs still wins (documented)


# --- F13-cluster-009 (cluster side): spec kwargs beat n_clusters=, loudly ---

def test_spec_kwargs_n_clusters_beats_explicit_kwarg_with_warning():
    X = _blobs(100, 4, 3, seed=3)
    spec = {'model': 'KMeans', 'kwargs': {'n_clusters': 2, 'random_state': 0}}
    with pytest.warns(UserWarning, match='n_clusters'):
        labels = cluster(X, cluster=spec, n_clusters=4)
    assert len(set(labels)) == 2


# --- F13-cluster-011: invalid instance specs get real errors ----------------

def test_non_estimator_spec_raises_value_error_naming_value():
    X = _blobs(30, 3, 2)
    with pytest.raises(ValueError, match='42'):
        cluster(X, cluster=42)


def test_sklearn_pipeline_spec_clusters_via_fit_predict():
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    X = _blobs(50, 3, 2, seed=0)
    pipe = SkPipeline([('scale', StandardScaler()),
                       ('kmeans', KMeans(n_clusters=2, n_init=10,
                                         random_state=0))])
    labels = cluster(X, cluster=pipe)
    assert isinstance(labels, list) and len(labels) == 100
    assert len(set(labels)) == 2


# --- F13-cluster-012: mixed-width dataset lists get a hypertools error ------

def test_mixed_width_dataset_list_raises_hypertools_error():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match='column'):
        cluster([rng.standard_normal((50, 4)),
                 rng.standard_normal((50, 3))], n_clusters=2)


# --- F13-cluster-014: HDBSCAN default run is warning-free -------------------

def test_hdbscan_default_run_emits_no_future_warning():
    X = _blobs(100, 5, 2, seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = cluster(X, cluster='HDBSCAN')
    assert len(labels) == 200
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


# --- F13-cluster-015: random_state is documented -----------------------------

def test_cluster_docstring_documents_random_state():
    assert 'random_state :' in cluster.__doc__


# --- F13-cluster-019: ndims without reduce warns instead of silently no-oping

def test_ndims_without_reduce_warns_and_is_ignored():
    X = np.random.default_rng(0).standard_normal((80, 10))
    with pytest.warns(UserWarning, match='ndims'):
        a = cluster(X, n_clusters=3, ndims=2, random_state=0)
    b = cluster(X, n_clusters=3, random_state=0)
    assert list(a) == list(b)


def test_ndims_with_reduce_takes_effect_without_ndims_warning():
    X = np.random.default_rng(0).standard_normal((80, 10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = cluster(X, n_clusters=3, reduce='PCA', ndims=2,
                         random_state=0)
    assert len(labels) == 80
    assert not any('ndims' in str(w.message) for w in caught)


# --- binding contract: cluster=False means "skip stage, return input" -------

def test_cluster_false_returns_input_unchanged():
    X = np.random.default_rng(0).standard_normal((20, 4))
    out = cluster(X, cluster=False)
    assert np.array_equal(np.asarray(out), X)
    out2, model = cluster(X, cluster=False, return_model=True)
    assert model is None
    assert np.array_equal(np.asarray(out2), X)


# --- F13-cluster-021 (cluster side): the default k is documented -------------

def test_cluster_docstring_documents_default_n_clusters():
    # n_clusters=None (signature default) is documented as meaning 3
    assert 'None' in cluster.__doc__
    labels = cluster(_blobs(50, 4, 3, seed=2), random_state=0)
    assert len(set(labels)) == 3
