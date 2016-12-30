# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertools.util.describe_pca import describe_pca

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)
attrs = describe_pca(data, show=False)

def test_describe_pca_data_is_dict():
    assert type(attrs) is dict
