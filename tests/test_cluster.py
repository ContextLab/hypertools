# -*- coding: utf-8 -*-

import numpy as np
import pytest
from hypertools.tools.cluster import cluster
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
