# -*- coding: utf-8 -*-

import pytest

import numpy as np

from hypertools.util.missing_inds import missing_inds

data = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
data[3,0]=np.nan
data[9,1]=np.nan
missing_data = missing_inds(data)

def test_missing_inds_correct_inds():
    assert missing_data==[3,9]

data1 = np.random.multivariate_normal(np.zeros(2), np.eye(2), size=10)
no_missing_data = missing_inds(data1)

def test_missing_inds_handles_no_missing_data():
    assert no_missing_data==[]
