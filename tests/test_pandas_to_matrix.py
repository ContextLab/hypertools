# -*- coding: utf-8 -*-

import pytest

import pandas as pd
import numpy as np

from hypertools.tools.df2mat import df2mat

def test_df2mat():
    df = pd.DataFrame(['a','b'])
    assert np.array_equal(df2mat(df),np.array([[1,0],[0,1]]))
