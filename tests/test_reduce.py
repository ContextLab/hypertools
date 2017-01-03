# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertools.tools.reduce import reduce as reduc

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i in range(2)]
reduced_data_3d = reduc(data)
reduced_data_2d = reduc(data,ndims=2)
reduced_data_1d = reduc(data,ndims=1)

def test_reduce_is_list():
    assert type(reduced_data_3d) is list

def test_reduce_is_array():
    assert isinstance(reduced_data_3d[0],np.ndarray)

def test_reduce_dims_3d():
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_dims_2d():
    assert reduced_data_2d[0].shape==(100,2)

def test_reduce_dims_1d():
    assert reduced_data_1d[0].shape==(100,1)

def test_reduce_assert_exception():
    with pytest.raises(Exception) as e_info:
        reduc(data,ndims=4)
