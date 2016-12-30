# -*- coding: utf-8 -*-

import pytest

from scipy.stats import multivariate_normal
import numpy as np

from hypertools.util.cluster import cluster

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = np.vstack([cluster1,cluster2])
labels = cluster(data,n_clusters=2)

def test_cluster_n_clusters():
    assert len(set(labels))==2

def test_cluster_returns_list():
    assert type(labels) is list
