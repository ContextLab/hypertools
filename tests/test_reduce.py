# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertools.util.reduce import reduce as reduc

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i in range(2)]
reduced_data = reduc(data)

def test_reduce_is_list():
    assert type(reduced_data) is list

def test_reduce_is_array():
    assert isinstance(reduced_data[0],np.ndarray)

def test_reduce_dims():
    assert reduced_data[0].shape==(100,3)
