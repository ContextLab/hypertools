import numpy as np


def test_cluster_new_path_hard_labels():
    from hypertools.cluster.cluster import cluster
    rng = np.random.RandomState(0)
    data = np.vstack([rng.randn(50, 3), rng.randn(50, 3) + 100])
    labels = cluster(data, n_clusters=2)
    assert type(labels) is list and len(set(labels)) == 2


def test_cluster_registry_dicts():
    from hypertools.cluster.cluster import models, mixture_models
    assert 'KMeans' in models and 'GaussianMixture' in mixture_models


def test_cluster_soft_mixture_proportions():
    from hypertools.cluster.cluster import cluster
    rng = np.random.RandomState(1)
    data = np.vstack([rng.randn(40, 3), rng.randn(40, 3) + 50])
    props = cluster(data, cluster='GaussianMixture', n_clusters=2)
    assert props.shape == (80, 2) and np.allclose(props.sum(axis=1), 1)
