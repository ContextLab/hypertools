# -*- coding: utf-8 -*-

import pytest

import scipy.io as sio
import numpy as np

from hypertools.util.align import align
def test_procrustes():
    data = sio.loadmat('examples/sample_data/test_data.mat')
    data1 = data['spiral']
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2])
    assert np.allclose(result[0],result[1])

# def test_srm():
#     data = sio.loadmat('examples/sample_data/test_data.mat')
#     data1 = data['spiral']
#     rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
#            [-0.43426149,  0.87492975, -0.21427761],
#            [-0.10761949,  0.18578133,  0.97667976]])
#     data2 = np.dot(data1, rot)
#     result = align([data1,data2],method='SRM')
#     assert np.allclose(result[0],result[1])
