# # -*- coding: utf-8 -*-

import pytest

## LIBRARIES ##

import numpy as np

## LOAD TEST DATA ##

# import scipy.io.loadmat as load
# test_dataset = load('examples/test_data.mat')
# test_data = test_dataset['spiral']

## HELPERS ##

from hypertools._shared.helpers import center
def test_center():
    assert np.array_equal(center([np.array([[1,2,3],[4,5,6]])]),[np.array([[-1.5,-1.5,-1.5],[1.5,1.5,1.5]])])
