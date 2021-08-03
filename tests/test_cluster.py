# -*- coding: utf-8 -*-

import numpy as np
import pytest
from hypertools import cluster
from hypertools import plot

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = np.vstack([cluster1, cluster2])
labels = cluster(data, n_clusters=2)


def test_cluster_n_clusters():
    assert len(set(labels))==2


def test_cluster_returns_list():
    assert type(labels) is list


def test_cluster_hdbscan():
    try:
        from hdbscan import HDBSCAN
        _has_hdbscan = True
    except:
        _has_hdbscan = False

    if _has_hdbscan:
        hdbscan_labels = cluster(data, cluster='HDBSCAN')
        assert len(set(hdbscan_labels)) == 2
    else:
        with pytest.raises(ImportError):
            hdbscan_labels = cluster(data, cluster='HDBSCAN')
