# -*- coding: utf-8 -*-

import pytest

import pandas as pd
import numpy as np

from hypertools.tools.pandas_to_matrix import pandas_to_matrix

def test_pandas_to_matrix():
    df = pd.DataFrame(['a','b'])
    assert np.array_equal(pandas_to_matrix(df),np.array([[1,0],[0,1]]))
