# -*- coding: utf-8 -*-

import pytest
import numpy as np

from hypertools.tools.load import load

def test_weights():
<<<<<<< HEAD
    weights = load('weights_sample')
=======
    weights = load('weights')
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 100) for wt in weights)

def test_weights_ndim3():
    # Should return 3 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=3)
=======
    weights = load('weights', ndims=3)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 3) for wt in weights)

def test_weights_ndim2():
    # Should return 2 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=2)
=======
    weights = load('weights', ndims=2)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 2) for wt in weights)

def test_weights_ndim1():
    # Should return 1 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=1)
=======
    weights = load('weights', ndims=1)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 1) for wt in weights)

def test_weights_ndim3_align():
    # Should return aligned 3 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=3, align=True)
=======
    weights = load('weights', ndims=3, align=True)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 3) for wt in weights)

def test_weights_ndim2_align():
    # Should return aligned 2 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=2, align=True)
=======
    weights = load('weights', ndims=2, align=True)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 2) for wt in weights)

def test_weights_ndim1_align():
    # Should return aligned 1 dimensional data
<<<<<<< HEAD
    weights = load('weights_avg', reduce='PCA', ndims=1, align=True)
=======
    weights = load('weights', ndims=1, align=True)
>>>>>>> 44fe07e96e8f109b3023a70c8716b20c71f07764
    assert all(wt.shape == (300, 1) for wt in weights)
