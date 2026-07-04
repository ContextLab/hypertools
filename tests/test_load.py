# # -*- coding: utf-8 -*-

import pandas as pd

from hypertools.io.load import load


def test_load_weights_avg():
    data = load('weights_avg')
    assert isinstance(data, list)


def test_load_weights_sample():
    data = load('weights_sample')
    assert isinstance(data, list)


def test_load_weights():
    data = load('weights')
    assert isinstance(data, list)


def test_load_mushrooms():
    data = load('mushrooms')
    assert isinstance(data, pd.DataFrame)


def test_load_spiral():
    data = load('spiral')
    assert isinstance(data, list)


def test_weights():
    data = load('weights_sample')
    assert all(wt.shape == (300, 100) for wt in data)


def test_weights_ndim3():
    # Should return 3 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=3)
    assert all(wt.shape == (100, 3) for wt in data)


def test_weights_ndim2():
    # Should return 2 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=2)
    assert all(wt.shape == (100, 2) for wt in data)


def test_weights_ndim1():
    # Should return 1 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=1)
    assert all(wt.shape == (100, 1) for wt in data)


def test_weights_ndim3_align():
    # Should return aligned 3 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=3, align='hyper')
    assert all(wt.shape == (100, 3) for wt in data)


def test_weights_ndim2_align():
    # Should return aligned 2 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=2, align='hyper')
    assert all(wt.shape == (100, 2) for wt in data)


def test_weights_ndim1_align():
    # Should return aligned 1 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=1, align='hyper')
    assert all(wt.shape == (100, 1) for wt in data)
