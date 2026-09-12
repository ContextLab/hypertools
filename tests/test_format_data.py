# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from hypertools.tools import format_data


def test_np_array():
    data = np.random.rand(100,10)
    res = format_data(data)
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)
    # a plain numpy array should pass through unchanged (shape and values)
    assert res[0].shape == (100, 10)
    assert np.allclose(res[0], data)


def test_df():
    data = pd.DataFrame(np.random.rand(100,10))
    res = format_data(data)
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)
    # DataFrame values must be preserved exactly, not just wrapped
    assert res[0].shape == (100, 10)
    assert np.allclose(res[0], data.values)


def test_text():
    data = ['here is some test text', 'and a little more', 'and more']
    res = format_data(data)
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)
    # one row per text sample, projected into the (fixed) text-model
    # feature space -- shape must reflect both, not just "be an array"
    assert res[0].shape == (len(data), 50)


def test_str():
    res = format_data('here is some test text')
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)
    # a bare string is treated as a single document
    assert res[0].shape == (1, 50)


def test_mixed_list():
    import pytest
    mat = np.random.rand(3,20)
    df = pd.DataFrame(np.random.rand(3,20))
    text = ['here is some test text', 'and a little more', 'and more']
    string = 'a string'
    # mismatched text/numeric sample counts deliberately provoke the
    # cannot-auto-align notice
    with pytest.warns(UserWarning, match='cannot be auto-aligned'):
        res = format_data([mat, df, text, string])
    assert isinstance(res, list)
    assert all(map(lambda x: isinstance(x, np.ndarray), res))


def test_missing_data():
    import pytest
    # format_data fills missing values via PPCA (no geo round-trip in 1.0);
    # the NaN deliberately provokes the missing-data notice
    data = np.random.rand(100,10)
    data[0][0]=np.nan
    with pytest.warns(UserWarning, match='filling missing values'):
        res = format_data(data)
    assert isinstance(res, list)
    assert isinstance(res[0], np.ndarray)
    # the whole point of this path is that the NaN gets filled in, and the
    # shape of the data must be unchanged by the imputation
    assert res[0].shape == (100, 10)
    assert not np.isnan(res[0]).any()


def test_force_align():
    import pytest
    mat = np.random.rand(4, 3)
    df = pd.DataFrame(np.random.rand(4, 3))
    text = ['here is some test text', 'and a little more', 'and more', 'just a bit more']
    # matched text/numeric sample counts deliberately provoke the
    # aligning-to-common-space notice
    with pytest.warns(UserWarning, match='Aligning data to a common space'):
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


# --- release-1.0 audit regressions (2026-07) --------------------------------

def test_empty_list_raises_no_data_error_before_text_path(capsys):
    # X2-error-quality-005: [] must raise the standard no-data error, NOT be
    # routed into the text/LDA corpus pipeline (which printed 'loading
    # corpus: minipedia...' to stdout and crashed inside sklearn's LDA)
    import pytest
    with pytest.raises(ValueError, match='no observations'):
        format_data([])
    assert 'loading corpus' not in capsys.readouterr().out


def test_3d_array_raises_clear_shape_error():
    # F15-analyze-012: 3-D input was silently axis-mangled (constant values)
    # or crashed deep inside normalize (typical values)
    import pytest
    with pytest.raises(ValueError, match='2-D'):
        format_data(np.zeros((4, 5, 6)))
    with pytest.raises(ValueError, match='2-D'):
        format_data(np.arange(120.).reshape(4, 5, 6))


def test_df2mat_categorical_dataframe_warning_free():
    # X4-warnings-001: select_dtypes(include=['object']) relied on deprecated
    # pandas back-compat (Pandas4Warning, a DeprecationWarning subclass, on
    # every categorical-DataFrame plot, e.g. the mushrooms demo)
    import warnings
    from hypertools.tools.df2mat import df2mat
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': ['x', 'y', 'x']})
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        out = df2mat(df)
    # 1 numeric column + 2 dummy columns for 'x'/'y', all float
    assert out.shape == (3, 3)
    assert out.dtype.kind == 'f'
    assert not np.isnan(out).any()


def test_format_data_categorical_dataframe_warning_free():
    # end-to-end: the same DataFrame through format_data itself
    import warnings
    df = pd.DataFrame({'a': [1., 2., 3.], 'b': ['x', 'y', 'x']})
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        out = format_data(df)
    assert out[0].shape == (3, 3)


# --- datatype audit (2026-09-08): classification defers to datawrangler ----

def test_string_naming_an_existing_file_is_a_document_not_a_path():
    # format_data classifies text with a plain str/bytes test on purpose:
    # datawrangler's `dw.zoo.is_text` interprets a string as a file path or
    # URL first (loading it, or raising 'Unknown datatype: md' on an existing
    # file with an unknown extension), and a user's document that happens to
    # name a file must still be embedded as the text it IS. Pinned so the
    # dw-predicate refactor of the coercion layer can never route documents
    # through the loader.
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    readme = os.path.join(root, 'readme.md')     # lowercase in this repo: macOS
    # resolves 'README.md' too, the Linux CI runners do not (2026-09-08)
    pyproject = os.path.join(root, 'pyproject.toml')
    assert os.path.exists(readme) and os.path.exists(pyproject)
    out = format_data(readme)
    assert isinstance(out, list) and out[0].shape == (1, 50)
    out = format_data([readme, pyproject])
    assert out[0].shape == (2, 50)


def test_series_like_and_nested_inputs_keep_their_1_0_form():
    # the dw-predicate coercion layer must hand back exactly what the
    # isinstance ladders did for the pandas/numpy inputs the 1.0 tests
    # exercise: a Series (top-level or nested) is one 1-D column dataset
    # with its values untouched, nested groups flatten, a masked array's
    # masked cells become NaN
    values = np.arange(6.)
    s = pd.Series(values, index=list('abcdef'), name='s')
    out = format_data(s)
    assert out[0].shape == (6, 1) and np.array_equal(out[0][:, 0], values)
    out = format_data([s, values])
    assert len(out) == 2 and all(o.shape == (6, 1) for o in out)
    arr = np.arange(12.).reshape(6, 2)
    out = format_data([[arr, (arr * 2,)], s])
    assert [o.shape for o in out] == [(6, 2), (6, 2), (6, 1)]
    masked = np.ma.masked_array(arr, mask=arr == 4.)
    import pytest
    with pytest.warns(UserWarning, match='masked array with 1 masked'):
        out = format_data(masked, ppca=False)
    assert np.isnan(out[0][2, 0]) and np.isnan(out[0]).sum() == 1
