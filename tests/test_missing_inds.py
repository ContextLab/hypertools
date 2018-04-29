# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_equal

from hypertools.tools.missing_inds import missing_inds


def test_missing_inds_correct_inds():
    data = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
    data[3, 0] = np.nan
    data[9, 1] = np.nan
    missing_data = missing_inds(data)
    expected = np.array([3, 9])
    assert_array_equal(expected, missing_data)


def test_missing_inds_handles_no_missing_data():
    data = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
    no_missing_data = missing_inds(data)
    expected = np.array([])
    assert_array_equal(expected, no_missing_data)


def test_missing_inds_multiple_arrays():
    data1 = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
    data2 = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
    data1[5, 0] = np.nan
    missing_data = missing_inds([data1, data2])
    assert len(missing_data) == 2
    assert_array_equal(missing_data[0], np.array([5]))
    assert missing_data[1] is None
