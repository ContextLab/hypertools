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
    props = cluster(np.abs(data),
                    cluster={'model': 'NMF',
                             'params': {'n_components': 2, 'max_iter': 500}})
    assert props.shape == (200, 2)
    assert props.min() >= 0


def test_cluster_mixture_via_plot():
    # end-to-end: mixture clustering through the plot pipeline
    geo = plot(data, '.', cluster='GaussianMixture', n_clusters=2, show=False)
    assert geo is not None
