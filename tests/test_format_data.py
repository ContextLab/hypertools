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

def test_text():
    data = ['here is some test text', 'and a little more', 'and more']
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)

def test_str():
    res = format_data('here is some test text')
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)

def test_mixed_list():
    mat = np.random.rand(3,20)
    df = pd.DataFrame(np.random.rand(3,20))
    text = ['here is some test text', 'and a little more', 'and more']
    string = 'a string'
    res = format_data([mat, df, text, string])
    assert isinstance(res, list)
    assert all(map(lambda x: isinstance(x, np.ndarray), res))

def test_force_align():
    mat = np.random.rand(4,4)
    df = pd.DataFrame(np.random.rand(4,3))
    text = ['here is some test text', 'and a little more', 'and more', 'just a bit more']
    res = format_data([mat, df, text])
    assert isinstance(res, list)
    assert all(map(lambda x: isinstance(x, np.ndarray), res))
    assert all(map(lambda x: x.shape[1]==20, res))
