"""Reusing a model handed back by `return_model=True` on NEW data must not
crash, across every dispatcher (reduce/cluster/align) and both flavors of
returned model: a single-stage wrapper (Reducer/Clusterer/Aligner) and a whole
cross-module `hypertools.Pipeline`.

Regression coverage for QC 2026-07: Jeremy reused a fitted cross-module cluster
model,

    _, model = hyp.cluster(X, cluster='KMeans', n_clusters=3, reduce='PCA',
                           ndims=3, manip='ZScore', return_model=True)
    hyp.cluster(Y, cluster=model, reduce='PCA', ndims=3, manip='ZScore')

which crashed with ``AttributeError: 'Pipeline' object has no attribute
'labels_'`` -- the fitted Pipeline was wrapped in a fresh Clusterer instead of
being reused via ``.transform``. The same class of bug hit
``reduce(reduce=fitted_Pipeline)`` and ``align(model=fitted_Pipeline)``, plus a
double-wrap when a fitted ``Clusterer`` was reused alongside cross-module
stages. All data is real (small) numeric arrays -- no mocks.
"""
import warnings

import numpy as np
import pytest

import hypertools as hyp
from hypertools.core.pipeline import Pipeline


def _xy():
    r = np.random.default_rng(0)
    x = r.normal(size=(40, 6))
    return x, x + 5.0


def _xy_lists():
    r = np.random.default_rng(1)
    xl = [r.normal(size=(30, 5)), r.normal(size=(30, 5))]
    return xl, [a + 5.0 for a in xl]


# --- Jeremy's exact reported case + variants ---------------------------

def test_cluster_reuse_fitted_pipeline_cross_module_does_not_crash():
    x, y = _xy()
    _, model = hyp.cluster(x, cluster='KMeans', n_clusters=3, reduce='PCA',
                           ndims=3, manip='ZScore', return_model=True)
    assert isinstance(model, Pipeline)
    # reusing the whole fitted pipeline on new data (used to raise
    # AttributeError: 'Pipeline' object has no attribute 'labels_')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = hyp.cluster(y, cluster=model, reduce='PCA', ndims=3,
                             manip='ZScore')
    assert len(np.asarray(labels)) == 40


def test_cluster_reuse_fitted_pipeline_warns_about_redundant_stage_kwargs():
    x, y = _xy()
    _, model = hyp.cluster(x, cluster='KMeans', n_clusters=3, reduce='PCA',
                           ndims=3, manip='ZScore', return_model=True)
    with pytest.warns(UserWarning, match='redundant'):
        hyp.cluster(y, cluster=model, reduce='PCA', ndims=3, manip='ZScore')


def test_cluster_reuse_fitted_pipeline_alone_no_warning():
    x, y = _xy()
    _, model = hyp.cluster(x, cluster='KMeans', n_clusters=3, reduce='PCA',
                           ndims=3, return_model=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning fails the test
        labels = hyp.cluster(y, cluster=model)
    assert len(np.asarray(labels)) == 40


def test_reduce_reuse_fitted_pipeline():
    x, y = _xy()
    _, model = hyp.reduce(x, reduce='PCA', ndims=2, manip='ZScore',
                          return_model=True)
    out = hyp.reduce(y, reduce=model)
    assert np.asarray(out).shape == (40, 2)


def test_align_reuse_fitted_pipeline():
    xl, yl = _xy_lists()
    _, model = hyp.align(xl, model='HyperAlign', reduce='PCA', ndims=3,
                         return_model=True)
    out = hyp.align(yl, model=model)
    assert len(out) == 2


# --- idempotent resolver: fitted single-stage wrapper is not re-wrapped -

def test_resolve_cluster_spec_is_idempotent_on_clusterer():
    from hypertools.cluster.cluster import _resolve_cluster_spec
    from hypertools.cluster.common import Clusterer
    x, _ = _xy()
    _, cc = hyp.cluster(x, cluster='KMeans', n_clusters=3, return_model=True)
    assert isinstance(cc, Clusterer)
    assert _resolve_cluster_spec(cc, 3) is cc  # not re-wrapped


def test_cluster_reuse_fitted_clusterer_dim_mismatch_raises_clear_error():
    # A Clusterer fitted on 6-D data cannot be reused after a reduce stage
    # squeezes the input to 3-D -- the fitted centroids live in 6-D. This is
    # ill-posed, and now raises a CLEAR sklearn dimension-mismatch ValueError
    # instead of the old cryptic 'labels_' AttributeError.
    x, y = _xy()
    _, cc = hyp.cluster(x, cluster='KMeans', n_clusters=3, return_model=True)
    with pytest.raises(ValueError, match='features'):
        hyp.cluster(y, cluster=cc, reduce='PCA', ndims=3, manip='ZScore')


# --- still-working single-stage reuse (no regression) ------------------

def test_reduce_reuse_fitted_reducer_single_stage():
    x, y = _xy()
    _, rr = hyp.reduce(x, reduce='PCA', ndims=2, return_model=True)
    assert np.asarray(hyp.reduce(y, reduce=rr)).shape == (40, 2)


def test_align_reuse_fitted_aligner_single_stage():
    xl, yl = _xy_lists()
    _, aa = hyp.align(xl, model='SharedResponseModel', return_model=True)
    assert len(hyp.align(yl, model=aa)) == 2
