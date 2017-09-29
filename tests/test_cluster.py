# -*- coding: utf-8 -*-

import pytest

from scipy.stats import multivariate_normal
import numpy as np

from hypertools.tools.cluster import cluster

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = np.vstack([cluster1,cluster2])
labels = cluster(data,n_clusters=2)
labels_aligned = cluster(data, n_clusters=2, align=True)
labels_normalized = cluster(data, n_clusters=2, normalize='across')

def test_cluster_n_clusters():
    assert len(set(labels))==2
    assert len(set(labels_aligned))==2
    assert len(set(labels_normalized))==2
    
    

def test_cluster_returns_list():
    assert type(labels) is list
    assert type(labels_aligned) is list
    assert type(labels_normalized) is list
