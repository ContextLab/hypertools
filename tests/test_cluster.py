# -*- coding: utf-8 -*-

import numpy as np
import pytest
from hypertools.cluster.cluster import cluster
from hypertools.plot.plot import plot

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = np.vstack([cluster1, cluster2])
labels = cluster(data, n_clusters=2)


def test_cluster_n_clusters():
    assert len(set(labels))==2


def test_cluster_returns_list():
    assert type(labels) is list


def test_cluster_hdbscan():
    # HDBSCAN ships with scikit-learn (>=1.3), so it is always available
    hdbscan_labels = cluster(data, cluster='HDBSCAN')
    assert len(set(hdbscan_labels)) == 2


def test_cluster_density_models_by_name():
    # regression test for GH #146 / #190: density/bandwidth clusterers
    # (no n_clusters param) must resolve by string name and must not have
    # n_clusters force-injected into their constructor
    for name in ('MeanShift', 'DBSCAN'):
        result = cluster(data, cluster=name)
        assert isinstance(result, list)
        assert len(result) == len(data)
        assert all(isinstance(v, (int, np.integer)) for v in result)


def test_cluster_gaussian_mixture_returns_proportions():
    props = cluster(data, cluster='GaussianMixture', n_clusters=2)
    assert props.shape == (200, 2)
    assert np.allclose(props.sum(axis=1), 1)
    # the two well-separated blobs should be assigned near-deterministically
    assert np.mean(props.max(axis=1) > 0.99) > 0.95


def test_cluster_bayesian_gaussian_mixture():
    props = cluster(data, cluster='BayesianGaussianMixture', n_clusters=2)
    assert props.shape == (200, 2)
    assert np.allclose(props.sum(axis=1), 1)


def test_cluster_lda_nonnegative_proportions():
    props = cluster(np.abs(data), cluster='LatentDirichletAllocation',
                    n_clusters=2)
    assert props.shape == (200, 2)
    assert np.allclose(props.sum(axis=1), 1)
    assert props.min() >= 0


def test_cluster_nmf_custom_params():
    # legacy 'params' dict spec exercised deliberately; assert the
    # deprecation notice fires
    with pytest.warns(DeprecationWarning, match=r"'params'.*deprecated"):
        props = cluster(np.abs(data),
                        cluster={'model': 'NMF',
                                 'params': {'n_components': 2,
                                            'max_iter': 500}})
    assert props.shape == (200, 2)
    assert props.min() >= 0


def test_cluster_mixture_via_plot():
    # end-to-end: mixture clustering through the plot pipeline
    geo = plot(data, '.', cluster='GaussianMixture', n_clusters=2, show=False)
    assert geo is not None


# --- flat dict-spec keys (1.1 review) ----------------------------------
#
# `{'model': 'KMeans', 'n_clusters': 4, 'random_state': 0}` used to drop
# `random_state` without a word, so every call drew different clusters.
# Model parameters belong under 'kwargs' (the convention every dispatcher
# documents); 'n_clusters' is the one documented top-level convenience.
# Anything else at the top level now raises, naming the key.

def _unseparated_blobs():
    # overlapping draws, so KMeans' result genuinely depends on its seed
    rng = np.random.default_rng(1)
    return rng.standard_normal((300, 5))


def _kmeans_runs(spec, n=6, via='cluster'):
    x = _unseparated_blobs()
    runs = set()
    for _ in range(n):
        if via == 'cluster':
            labels = cluster(x, cluster=spec)
        else:
            import hypertools as hyp
            _, model = hyp.analyze(x, cluster=spec, return_model=True)
            labels = model.named_steps['cluster'].transform(x)
        runs.add(tuple(int(v) for v in labels))
    return runs


def test_unseeded_kmeans_on_this_data_varies_between_runs():
    # the observable the flat-spec tests rely on: without a seed, one-init
    # KMeans on this data does NOT reproduce, so a dropped random_state
    # shows up as more than one distinct labelling
    runs = _kmeans_runs({'model': 'KMeans',
                         'kwargs': {'n_clusters': 4, 'n_init': 1}})
    assert len(runs) > 1


def test_nested_kwargs_spec_is_reproducible():
    spec = {'model': 'KMeans',
            'kwargs': {'n_clusters': 4, 'n_init': 1, 'random_state': 0}}
    assert len(_kmeans_runs(spec)) == 1
    assert len(_kmeans_runs(spec, via='analyze')) == 1


@pytest.mark.parametrize('via', ['cluster', 'analyze', 'plot'])
def test_flat_spec_model_kwargs_raise_naming_the_key(via):
    import hypertools as hyp
    x = _unseparated_blobs()
    spec = {'model': 'KMeans', 'n_clusters': 4, 'random_state': 0,
            'n_init': 1}
    with pytest.raises(ValueError) as err:
        if via == 'cluster':
            hyp.cluster(x, cluster=spec)
        elif via == 'analyze':
            hyp.analyze(x, cluster=spec)
        else:
            hyp.plot(x, '.', cluster=spec, show=False)
    msg = str(err.value)
    assert "'n_init'" in msg and "'random_state'" in msg
    # the message spells out the corrected spec, and 'n_clusters' (the
    # documented top-level convenience) is not reported as unknown
    assert "'kwargs': {'n_init': 1, 'random_state': 0}" in msg
    assert "['n_init', 'random_state']" in msg


def test_flat_spec_error_suggestion_is_a_working_spec():
    # the corrected spec the error proposes really is reproducible
    x = _unseparated_blobs()
    with pytest.raises(ValueError, match="'random_state'"):
        cluster(x, cluster={'model': 'KMeans', 'n_clusters': 4,
                            'random_state': 0, 'n_init': 1})
    fixed = {'model': 'KMeans', 'n_clusters': 4,
             'kwargs': {'n_init': 1, 'random_state': 0}}
    assert len(_kmeans_runs(fixed)) == 1
    labels = cluster(x, cluster=fixed)
    assert len(set(labels)) == 4


def test_flat_spec_extra_key_raises_with_canonical_kwargs_too():
    # a stray top-level key next to a canonical 'kwargs' dict is just as
    # silently dropped, so it raises too (and the suggestion merges it in)
    x = _unseparated_blobs()
    with pytest.raises(ValueError) as err:
        cluster(x, cluster={'model': 'KMeans', 'kwargs': {'n_clusters': 2},
                            'random_state': 3})
    assert "'kwargs': {'n_clusters': 2, 'random_state': 3}" in str(err.value)


def test_flat_spec_top_level_n_clusters_still_accepted():
    # the documented convenience keeps working, without any warning
    import warnings
    x = _unseparated_blobs()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        labels = cluster(x, cluster={'model': 'KMeans', 'n_clusters': 5,
                                     'kwargs': {'random_state': 0}})
    assert len(set(labels)) == 5
