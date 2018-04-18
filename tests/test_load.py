# # -*- coding: utf-8 -*-
#
import pytest
import numpy as np

from hypertools.tools.load import load
from hypertools.datageometry import DataGeometry
# from sklearn.pipeline import Pipeline
#
# def test_load_wiki_model():
#     geo = load('wiki_model')
#     assert isinstance(geo, Pipeline)
#
def test_load_weights_avg():
    geo = load('weights_avg')
    assert isinstance(geo, DataGeometry)

def test_load_weights_sample():
    geo = load('weights_sample')
    assert isinstance(geo, DataGeometry)

def test_load_weights():
    geo = load('weights')
    assert isinstance(geo, DataGeometry)

def test_load_mushrooms():
    geo = load('mushrooms')
    assert isinstance(geo, DataGeometry)

def test_load_spiral():
    geo = load('spiral')
    assert isinstance(geo, DataGeometry)

# def test_load_sotus():
#     geo = load('sotus')
#     assert isinstance(geo, DataGeometry)
#
# def test_load_wiki():
#     geo = load('wiki')
#     assert isinstance(geo, DataGeometry)
#
# def test_load_nips():
#     geo = load('nips')
#     assert isinstance(geo, DataGeometry)
#
def test_weights():
    geo = load('weights_sample')
    assert all(wt.shape == (300, 100) for wt in geo.get_data())

def test_weights_ndim3():
    # Should return 3 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=3)
    print(geo.transform()[0].shape)
    assert all(wt.shape == (100, 3) for wt in geo.transform())

def test_weights_ndim2():
    # Should return 2 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=2)
    assert all(wt.shape == (100, 2) for wt in geo.transform())

def test_weights_ndim1():
    # Should return 1 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=1)
    assert all(wt.shape == (100, 1) for wt in geo.transform())

def test_weights_ndim3_align():
    # Should return aligned 3 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=3, align=True)
    assert all(wt.shape == (100, 3) for wt in geo.transform())

def test_weights_ndim2_align():
    # Should return aligned 2 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=2, align=True)
    assert all(wt.shape == (100, 2) for wt in geo.transform())

def test_weights_ndim1_align():
    # Should return aligned 1 dimensional data
    geo = load('weights_avg', reduce='PCA', ndims=1, align=True)
    assert all(wt.shape == (100, 1) for wt in geo.transform())
