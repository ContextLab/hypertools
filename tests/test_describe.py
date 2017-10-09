# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertools.tools.describe_pca import describe_pca
from hypertools.tools.describe import describe

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)

def test_describe_pca_data_is_dict():
    result = describe_pca(data, show=False)
    assert type(result) is dict

def test_describe_data_is_dict():
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict
