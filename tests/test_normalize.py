# -*- coding: utf-8 -*-

import numpy as np

from hypertools.tools.normalize import normalize
from hypertools.plot.plot import plot

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = [cluster1, cluster2]


def test_normalize_returns_list():
    assert type(normalize(data)) is list


def test_normalize_across():
    norm_data = normalize(data, normalize='across')
    assert np.allclose(np.mean(np.vstack(norm_data),axis=0),0)


def test_normalize_within():
    norm_data = normalize(data, normalize='within')
    assert np.allclose([np.mean(i,axis=0) for i in norm_data],0)


def test_normalize_row():
    norm_data = normalize(data, normalize='row')
    assert np.allclose(np.mean(np.vstack(norm_data), axis=1),0)


def test_normalize_geo():
    geo = plot(data, show=False)
    norm_data = normalize(geo, normalize='row')
    assert np.allclose(np.mean(np.vstack(norm_data), axis=1),0)
