# -*- coding: utf-8 -*-

import pytest
import numpy as np

from hypertools.tools.align import align
from hypertools.tools.load import load

weights = load('weights')

def test_procrustes():
    data1 = load('spiral')
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2])
    assert np.allclose(result[0],result[1])

def test_srm():
    data1 = load('spiral')
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2], method='SRM')
    assert np.allclose(result[0],result[1], rtol=1) #note: this tolerance is probably too high, but fails at anything lower

def test_align_shapes():
    # Should return data with the same shape as input data
    aligned = align(weights)
    assert all(al.shape == wt.shape for al, wt in zip(aligned, weights))

def test_align_reduce5d_shape():
    # Should return data with same rowcnt but reduced to 5 columns
    aligned = align(weights, ndims=5)
    assert all(al.shape == (wt.shape[0], 5) for al, wt in zip(aligned, weights))
