# -*- coding: utf-8 -*-

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


def test_missing_data():
    # format_data fills missing values via PPCA (no geo round-trip in 1.0)
    data = np.random.rand(100,10)
    data[0][0]=np.nan
    assert isinstance(format_data(data), list)
    assert isinstance(format_data(data)[0], np.ndarray)


def test_force_align():
    mat = np.random.rand(4, 3)
    df = pd.DataFrame(np.random.rand(4, 3))
    text = ['here is some test text', 'and a little more', 'and more', 'just a bit more']
    res = format_data([mat, df, text])
    assert isinstance(res, list)
    assert all(map(lambda x: isinstance(x, np.ndarray), res))
    assert all(map(lambda x: x.shape[1] == 50, res))


# --- GH #132: align DataFrame columns by NAME across datasets ---------------

def test_df_columns_reordered_by_name_across_datasets():
    # same columns, different order: dataset 2 must be reordered so features
    # align by name (previously consumed positionally -> silent misalignment)
    import pytest
    df1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [10., 20., 30.]})
    df2 = pd.DataFrame({'b': [100., 200., 300.], 'a': [1000., 2000., 3000.]})
    with pytest.warns(UserWarning, match='reordering'):
        out = format_data([df1, df2])
    # column 0 must be 'a' for BOTH datasets
    assert np.allclose(out[0][:, 0], [1., 2., 3.])
    assert np.allclose(out[1][:, 0], [1000., 2000., 3000.])
    assert np.allclose(out[1][:, 1], [100., 200., 300.])


def test_df_column_set_mismatch_raises():
    import pytest
    df1 = pd.DataFrame({'a': [1., 2., 3.], 'b': [10., 20., 30.]})
    df3 = pd.DataFrame({'a': [1., 2., 3.], 'c': [5., 6., 7.]})
    with pytest.raises(ValueError, match='columns do not match'):
        format_data([df1, df3])


def test_df_default_integer_columns_stay_positional():
    # DataFrames wrapping plain arrays (RangeIndex columns) keep the historical
    # positional behavior -- no reorder, no error
    a = pd.DataFrame(np.random.rand(10, 3))
    b = pd.DataFrame(np.random.rand(10, 3))
    out = format_data([a, b])
    assert len(out) == 2 and out[0].shape == (10, 3)


def test_df_matching_column_order_unchanged():
    df1 = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]})
    df2 = pd.DataFrame({'a': [5., 6.], 'b': [7., 8.]})
    out = format_data([df1, df2])
    assert np.allclose(out[1][:, 0], [5., 6.])
