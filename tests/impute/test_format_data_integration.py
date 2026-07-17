import numpy as np
import pytest

from hypertools.tools.format_data import format_data, fill_missing


def test_format_data_missing_data_unchanged_behavior():
    # format_data fills missing values via PPCA (no geo round-trip in 1.0);
    # this mirrors tests/test_format_data.py::test_missing_data, re-run here
    # against the impute-routed implementation.
    data = np.random.rand(100, 10)
    data[0][0] = np.nan
    out = format_data(data)
    assert isinstance(out, list)
    assert isinstance(out[0], np.ndarray)
    assert not np.isnan(out[0]).any()


def test_format_data_warns_on_missing_data():
    data = np.random.rand(100, 10)
    data[0][0] = np.nan
    # F17-006: the old 'Inexact solution' text was stale -- PPCA preserves
    # observed values exactly and only reconstructs the NaN entries
    with pytest.warns(UserWarning, match='filling missing values'):
        format_data(data)


def test_fill_missing_returns_list_of_arrays_matching_row_counts():
    a = np.random.rand(30, 5)
    b = np.random.rand(20, 5)
    a[0, 0] = np.nan
    b[5, 2] = np.nan

    filled = fill_missing([a, b])

    assert isinstance(filled, list)
    assert len(filled) == 2
    assert all(isinstance(f, np.ndarray) for f in filled)
    assert filled[0].shape[0] == 30
    assert filled[1].shape[0] == 20


def test_fill_missing_single_array_returns_single_item_list():
    a = np.random.rand(40, 6)
    a[3, 1] = np.nan
    filled = fill_missing([a])
    assert isinstance(filled, list)
    assert len(filled) == 1
    assert isinstance(filled[0], np.ndarray)
    assert filled[0].shape[0] == 40


def test_fill_missing_fully_missing_row_stays_nan():
    a = np.random.rand(50, 6)
    a[10, :] = np.nan  # fully missing row -- PPCA cannot reconstruct it

    filled = fill_missing([a])

    assert np.isnan(filled[0][10]).all()
    other = np.delete(filled[0], 10, axis=0)
    assert not np.isnan(other).any()
