"""Top-level `random_state` for reproducibility (QC 2026-07).

Stochastic stages (UMAP/TSNE/MDS reductions, KMeans/GaussianMixture clustering)
were only reproducible via the verbose dict spec
``reduce={'model': 'UMAP', 'kwargs': {'random_state': 1}}``. A top-level
``random_state=`` on reduce/cluster/analyze/plot now injects it into any stage
model whose constructor accepts it, without disturbing models that don't.

Real data, no mocks, headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import warnings

import numpy as np
import pytest

import hypertools as hyp


def _rng():
    return np.random.default_rng(0)


def _arr(x):
    return np.asarray(x[0] if isinstance(x, list) else x)


def test_reduce_random_state_reproducible():
    X = _rng().normal(size=(60, 10))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = _arr(hyp.reduce(X, reduce='UMAP', ndims=2, random_state=1))
        b = _arr(hyp.reduce(X, reduce='UMAP', ndims=2, random_state=1))
    assert np.allclose(a, b)


def test_reduce_random_state_noop_for_models_without_it():
    """IncrementalPCA has no random_state -> injection must be skipped, no crash."""
    X = _rng().normal(size=(40, 6))
    out = _arr(hyp.reduce(X, reduce='IncrementalPCA', ndims=3, random_state=1))
    assert out.shape == (40, 3)


def test_reduce_user_kwargs_random_state_wins():
    X = _rng().normal(size=(50, 8))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = _arr(hyp.reduce(X, reduce={'model': 'UMAP', 'kwargs': {'random_state': 9}},
                            ndims=2, random_state=1))
        b = _arr(hyp.reduce(X, reduce={'model': 'UMAP', 'kwargs': {'random_state': 9}},
                            ndims=2, random_state=1))
    assert np.allclose(a, b)


def test_cluster_random_state_reproducible():
    X = _rng().normal(size=(80, 6))
    a = np.asarray(hyp.cluster(X, cluster='GaussianMixture', n_clusters=3,
                               random_state=5))
    b = np.asarray(hyp.cluster(X, cluster='GaussianMixture', n_clusters=3,
                               random_state=5))
    assert np.allclose(a, b)


def test_cluster_random_state_noop_for_density_clusterers():
    """DBSCAN has no random_state -> no crash."""
    X = _rng().normal(size=(60, 5))
    out = np.asarray(hyp.cluster(X, cluster='DBSCAN', random_state=1))
    assert out.shape[0] == 60


def test_analyze_and_plot_random_state_reproducible():
    X = _rng().normal(size=(80, 12))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = _arr(hyp.analyze(X, reduce='UMAP', ndims=3, random_state=3))
        b = _arr(hyp.analyze(X, reduce='UMAP', ndims=3, random_state=3))
        assert np.allclose(a, b)
        f1 = hyp.plot(X, reduce='UMAP', ndims=3, random_state=8, show=False)
        f2 = hyp.plot(X, reduce='UMAP', ndims=3, random_state=8, show=False)
    d1 = np.asarray(f1.axes[0].lines[0].get_data_3d())
    d2 = np.asarray(f2.axes[0].lines[0].get_data_3d())
    assert np.allclose(d1, d2)


def test_pipeline_path_random_state_reproducible():
    """Cross-module (build_pipeline) reduce+cluster stages get random_state."""
    X = _rng().normal(size=(80, 12))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ra, _ = hyp.analyze(X, reduce='UMAP', ndims=3, cluster='KMeans',
                            random_state=2, return_model=True)
        rb, _ = hyp.analyze(X, reduce='UMAP', ndims=3, cluster='KMeans',
                            random_state=2, return_model=True)
    assert np.allclose(_arr(ra), _arr(rb))
