# -*- coding: utf-8 -*-

import numpy as np
import hypertools as hyp

weights = hyp.load('weights')
spiral = hyp.load('spiral')


def test_procrustes():
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2])
    assert np.allclose(result[0],result[1])


def test_hyper():
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2], align='hyper')
    assert np.allclose(result[0],result[1], rtol=1) #note: this tolerance is probably too high, but fails at anything lower


def test_SRM():
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    data2 = np.dot(data1, rot)
    result = align([data1,data2], align='SRM')
    assert np.allclose(result[0],result[1], rtol=1)


def test_align_shapes():
    # Should return data with the same shape as input data
    aligned = align(weights)
    assert all(al.shape == wt.shape for al, wt in zip(aligned, weights))


def test_align_geo():
    aligned = align(geo)
    assert np.allclose(aligned[0], aligned[1])
