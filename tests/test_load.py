# -*- coding: utf-8 -*-

import pytest
import numpy as np

from hypertools.tools.load import load

def test_weights():
    weights, labels = load('weights_sample')
    assert all(wt.shape == (300, 100) for wt in weights)

def test_weights_ndim3():
    # Should return 3 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=3)
    assert all(wt.shape == (300, 3) for wt in weights)

def test_weights_ndim2():
    # Should return 2 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=2)
    assert all(wt.shape == (300, 2) for wt in weights)

def test_weights_ndim1():
    # Should return 1 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=1)
    assert all(wt.shape == (300, 1) for wt in weights)

def test_weights_ndim3_align():
    # Should return aligned 3 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=3, align=True)
    assert all(wt.shape == (300, 3) for wt in weights)

def test_weights_ndim2_align():
    # Should return aligned 2 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=2, align=True)
    assert all(wt.shape == (300, 2) for wt in weights)

def test_weights_ndim1_align():
    # Should return aligned 1 dimensional data
    weights, labels = load('weights_avg', reduce='PCA', ndims=1, align=True)
    assert all(wt.shape == (300, 1) for wt in weights)
