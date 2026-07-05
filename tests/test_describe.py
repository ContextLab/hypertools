# -*- coding: utf-8 -*-

import numpy as np

from hypertools.reduce.describe import describe

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)


def test_describe_data_is_dict():
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict


def test_describe_geo():
    # describe() operates on raw data directly (no geo round-trip in 1.0)
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict
