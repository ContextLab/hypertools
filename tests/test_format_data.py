# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd

from hypertools.tools import format_data

def test_np_array():
    data = np.random.rand(100,10)
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)

def test_df():
    data = pd.DataFrame(np.random.rand(100,10))
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)

def test_mixed_list():
    data1 = np.random.rand(100,10)
    data2 = pd.DataFrame(np.random.rand(100,10))
    assert isinstance(format_data([data1, data2]), list)
    assert isinstance(format_data([data1, data2])[0], np.ndarray)
    assert isinstance(format_data([data1, data2])[1], np.ndarray)
