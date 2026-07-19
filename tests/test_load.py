# # -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from hypertools.io.load import load


def test_load_weights_avg():
    data = load('weights_avg')
    assert isinstance(data, list)


def test_load_weights_sample():
    data = load('weights_sample')
    assert isinstance(data, list)


def test_load_weights():
    data = load('weights')
    assert isinstance(data, list)


def test_load_mushrooms():
    data = load('mushrooms')
    assert isinstance(data, pd.DataFrame)


def test_load_spiral():
    data = load('spiral')
    assert isinstance(data, list)


def test_weights():
    data = load('weights_sample')
    assert all(wt.shape == (300, 100) for wt in data)


def test_weights_ndim3():
    # Should return 3 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=3)
    assert all(wt.shape == (100, 3) for wt in data)


def test_weights_ndim2():
    # Should return 2 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=2)
    assert all(wt.shape == (100, 2) for wt in data)


def test_weights_ndim1():
    # Should return 1 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=1)
    assert all(wt.shape == (100, 1) for wt in data)


def test_weights_ndim3_align():
    # Should return aligned 3 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=3, align='hyper')
    assert all(wt.shape == (100, 3) for wt in data)


def test_weights_ndim2_align():
    # Should return aligned 2 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=2, align='hyper')
    assert all(wt.shape == (100, 2) for wt in data)


def test_weights_ndim1_align():
    # Should return aligned 1 dimensional data
    data = load('weights_avg', reduce='PCA', ndims=1, align='hyper')
    assert all(wt.shape == (100, 1) for wt in data)


def test_load_reduce_dict_form(tmp_path):
    # Regression: reduce={'model': ..., 'params': {...}} used to crash with
    # "TypeError: unhashable type: 'dict'" because the analyze-trigger guard
    # built a set literal from the arg values (any({reduce, ndims, ...})),
    # and dicts aren't hashable. Use a local .npy so no network is needed.
    arr = np.random.RandomState(0).randn(50, 6)
    path = tmp_path / "local_array.npy"
    np.save(path, arr)

    # legacy 'params' dict spec exercised deliberately; assert the
    # deprecation notice fires
    with pytest.warns(DeprecationWarning, match=r"'params'.*deprecated"):
        data = load(str(path),
                    reduce={'model': 'PCA', 'params': {'whiten': True}},
                    ndims=3)
    assert np.asarray(data).shape == (50, 3)


def test_load_align_dict_form(tmp_path):
    # Regression companion to test_load_reduce_dict_form: align also accepts
    # a dict form (e.g. {'model': 'SharedResponseModel', 'params': {...}})
    # and must not trip the same unhashable-set guard.
    arr = np.random.RandomState(1).randn(50, 6)
    path = tmp_path / "a.npy"
    np.save(path, arr)

    data = load(str(path),
               reduce='PCA', ndims=3,
               align={'model': 'hyper', 'params': {}})
    assert all(np.asarray(wt).shape == (50, 3) for wt in data)
