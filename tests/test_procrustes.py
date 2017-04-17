# -*- coding: utf-8 -*-

import pytest

import scipy.io as sio
import numpy as np

from hypertools.tools.procrustes import procrustes
from hypertools.tools.load import load

def test_procrustes_func():
    target = load('spiral')
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
           [-0.43426149,  0.87492975, -0.21427761],
           [-0.10761949,  0.18578133,  0.97667976]])
    source = np.dot(target, rot)
    source_aligned = procrustes(source,target)
    assert np.allclose(target,source_aligned)
