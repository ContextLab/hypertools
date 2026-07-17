# -*- coding: utf-8 -*-
"""Release-1.0 audit, final wave: F08-plot-inputs fixes in format_data
(and get_type).

Covers:
- F08-003: DataFrame Categorical columns are dummy-coded like object dtype
- F08-006: DataFrame datetime64 columns convert to epoch seconds + warning
- F08-008: unsupported-type errors name the type and the list index
- F08-009: numpy MaskedArray masked entries -> NaN (missing data) + warning
- F08-010: nested lists of arrays are flattened (like hyp.plot)
- F08-011: typo'd vectorizer=/semantic= raises a clear ValueError instead of
  a raw HuggingFace network error
- F08-013: python lists of bools are accepted like numpy bool arrays
"""

import numpy as np
import pandas as pd
import pytest

from hypertools.tools.format_data import format_data
from hypertools._shared.helpers import get_type


# --- F08-003: Categorical columns --------------------------------------------

def test_categorical_column_dummy_coded_like_object():
    df_obj = pd.DataFrame({'v': [1., 2., 3., 4.],
                           'c': ['a', 'b', 'a', 'b']})
    df_cat = df_obj.copy()
    df_cat['c'] = df_cat['c'].astype('category')
    out_obj = format_data(df_obj)[0]
    out_cat = format_data(df_cat)[0]
    assert out_cat.shape == out_obj.shape == (4, 3)
    np.testing.assert_allclose(out_cat, out_obj)


def test_categorical_numeric_categories_dummy_coded():
    df = pd.DataFrame({'v': [1., 2., 3., 4.],
                       'c': pd.Categorical([10, 20, 10, 20])})
    out = format_data(df)[0]
    # 1 numeric column + one dummy column per category
    assert out.shape == (4, 3)
    assert np.asarray(out, dtype=float).dtype.kind == 'f'


def test_categorical_input_dataframe_not_mutated():
    df = pd.DataFrame({'v': [1., 2.], 'c': pd.Categorical(['a', 'b'])})
    format_data(df)
    assert isinstance(df['c'].dtype, pd.CategoricalDtype)


# --- F08-006: datetime64 columns ----------------------------------------------

def test_datetime_column_converts_to_epoch_seconds_with_warning():
    df = pd.DataFrame({'t': pd.date_range('2020', periods=4),
                       'v': [1., 2., 3., 4.]})
    with pytest.warns(UserWarning, match=r"\['t'\].*seconds since the Unix"):
        out = format_data(df)[0]
    assert out.shape == (4, 2)
    # 2020-01-01 00:00:00 UTC epoch seconds
    assert out[0, 0] == pytest.approx(1577836800.0)
    # consecutive days are 86400 s apart
    assert out[1, 0] - out[0, 0] == pytest.approx(86400.0)


def test_datetime_tz_aware_and_nat_become_utc_seconds_and_nan():
    t = pd.to_datetime(['2020-01-01', None, '2020-01-03'])
    df = pd.DataFrame({'t': t.tz_localize('US/Eastern'), 'v': [1., 2., 3.]})
    with pytest.warns(UserWarning, match='datetime'):
        out = format_data(df, ppca=False)[0]
    # 2020-01-01 00:00 US/Eastern == 05:00 UTC
    assert out[0, 0] == pytest.approx(1577836800.0 + 5 * 3600)
    assert np.isnan(out[1, 0])  # NaT -> NaN (missing), never a sentinel int


def test_datetime_input_dataframe_not_mutated():
    df = pd.DataFrame({'t': pd.date_range('2020', periods=3),
                       'v': [1., 2., 3.]})
    with pytest.warns(UserWarning, match='datetime'):
        format_data(df)
    assert pd.api.types.is_datetime64_any_dtype(df['t'])


# --- F08-008: unsupported-type error quality ----------------------------------

def test_unsupported_type_names_received_type():
    with pytest.raises(TypeError, match=r"Unsupported data type 'dict'"):
        format_data({'a': 1})


def test_unsupported_list_element_names_index_and_type():
    with pytest.raises(TypeError,
                       match=r"dataset 1 of the input list.*'NoneType'"):
        format_data([np.random.randn(3, 2), None])


def test_get_type_message_keeps_legacy_prefix():
    # other audit suites match on this prefix -- it must survive
    with pytest.raises(TypeError, match='Unsupported data type'):
        get_type(None)


def test_get_type_lists_current_supported_types():
    with pytest.raises(TypeError, match='pandas Series'):
        get_type(None)


# --- F08-009: numpy masked arrays ----------------------------------------------

def _masked_dataset():
    rng = np.random.RandomState(0)
    base = rng.randn(20, 3)
    m = np.ma.masked_array(base, mask=np.zeros((20, 3), dtype=bool))
    m.mask[0, 1] = True
    m.mask[2, 0] = True
    return base, m


def test_masked_entries_become_nan_with_warning_ppca_false():
    base, m = _masked_dataset()
    with pytest.warns(UserWarning, match='masked array with 2 masked'):
        out = format_data(m, ppca=False)[0]
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, np.ma.MaskedArray)
    assert np.isnan(out[0, 1]) and np.isnan(out[2, 0])
    # observed entries are untouched
    assert np.allclose(out[~m.mask], base[~m.mask])


def test_masked_entries_flow_into_ppca_impute_path():
    base, m = _masked_dataset()
    # one call provokes TWO deliberate notices: the masked-entries notice and
    # the PPCA missing-data notice (nested pytest.warns asserts both; the
    # inner block re-emits whichever the outer one matches)
    with pytest.warns(UserWarning, match='masked'), \
         pytest.warns(UserWarning, match='filling missing values'):
        out = format_data(m)[0]
    # imputed: no NaNs remain, observed values preserved exactly, and the
    # masked cells were NOT taken from the raw underlying values
    assert not np.isnan(out).any()
    assert np.allclose(out[~m.mask], base[~m.mask])
    assert not np.allclose(out[m.mask], base[m.mask])


def test_masked_array_without_masked_entries_no_warning():
    rng = np.random.RandomState(1)
    m = np.ma.masked_array(rng.randn(10, 2), mask=False)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        out = format_data(m)[0]
    assert not isinstance(out, np.ma.MaskedArray)


# --- F08-010: nested lists of arrays -------------------------------------------

def test_nested_list_of_arrays_is_flattened():
    a, b = np.random.randn(3, 2), np.random.randn(4, 2)
    out = format_data([[a, b]])
    assert len(out) == 2
    np.testing.assert_allclose(out[0], a)
    np.testing.assert_allclose(out[1], b)


def test_deeper_nesting_and_mixed_levels_flatten_like_plot():
    a, b, c = (np.random.randn(3, 2) for _ in range(3))
    out = format_data([[a, [b]], c])
    assert len(out) == 3


def test_nested_list_mixing_arrays_and_text_raises_clear_error():
    with pytest.raises(ValueError, match='mixes numpy arrays with'):
        format_data([[np.random.randn(3, 2), 'hello']])


def test_nested_string_lists_still_text_corpora():
    # a nested list of strings must NOT be flattened into scalar "datasets"
    out = format_data([['some text here', 'more text here']], corpus=None)
    assert len(out) == 1
    assert out[0].shape[0] == 2


# --- F08-011: typo'd vectorizer=/semantic= -------------------------------------

def test_typo_vectorizer_raises_clear_valueerror_not_raw_hf_error():
    with pytest.raises(ValueError,
                       match=r"vectorizer='NotARealHypertoolsModel123'"
                             r".*CountVectorizer") as excinfo:
        format_data(['some text here', 'more text here'],
                    vectorizer='NotARealHypertoolsModel123', corpus=None)
    # the network-layer error is chained, not swallowed
    assert excinfo.value.__cause__ is not None


def test_typo_semantic_raises_clear_valueerror():
    with pytest.raises(ValueError,
                       match=r"semantic='NotARealHypertoolsModel123'"
                             r".*LatentDirichletAllocation"):
        format_data(['some text here', 'more text here'],
                    semantic='NotARealHypertoolsModel123', corpus=None)


def test_builtin_vectorizer_names_still_work():
    out = format_data(['some text here', 'more text here'],
                      vectorizer='CountVectorizer', corpus=None)
    assert out[0].shape[0] == 2


# --- F08-013: python lists of bools --------------------------------------------

def test_bool_list_accepted_as_numeric():
    out = format_data([True, False, True])[0]
    np.testing.assert_allclose(out.ravel(), [1., 0., 1.])
    # same data as the numpy bool array, which has always been accepted
    np.testing.assert_allclose(
        out, format_data(np.array([True, False, True]))[0])


def test_numpy_bool_scalar_list_accepted():
    out = format_data([np.bool_(True), np.bool_(False)])[0]
    np.testing.assert_allclose(out.ravel(), [1., 0.])


def test_get_type_bool_list_is_list_num():
    assert get_type([True, False]) == 'list_num'


# --- regression guards ----------------------------------------------------------

def test_series_inside_list_accepted():
    out = format_data([pd.Series([1., 2., 3.])])
    assert out[0].shape == (3, 1)


def test_flat_numeric_list_and_scalar_still_work():
    assert format_data([1., 2., 3.])[0].shape == (3, 1)
    assert format_data(np.array(5))[0].shape == (1, 1)


def test_multiple_arrays_unchanged():
    out = format_data([np.random.randn(4, 3), np.random.randn(4, 3)])
    assert len(out) == 2
