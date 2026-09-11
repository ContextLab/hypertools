# -*- coding: utf-8 -*-

import numpy as np

from hypertools.tools.normalize import normalize, Normalizer

cluster1 = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=100)
cluster2 = np.random.multivariate_normal(np.zeros(3)+100, np.eye(3), size=100)
data = [cluster1, cluster2]


def test_normalize_returns_list():
    assert type(normalize(data)) is list


def test_normalize_across():
    norm_data = normalize(data, normalize='across')
    assert np.allclose(np.mean(np.vstack(norm_data),axis=0),0)


def test_normalize_within():
    norm_data = normalize(data, normalize='within')
    assert np.allclose([np.mean(i,axis=0) for i in norm_data],0)


def test_normalize_row():
    norm_data = normalize(data, normalize='row')
    assert np.allclose(np.mean(np.vstack(norm_data), axis=1),0)


def test_normalize_geo():
    # normalize() operates on raw data directly (no geo round-trip in 1.0)
    norm_data = normalize(data, normalize='row')
    assert np.allclose(np.mean(np.vstack(norm_data), axis=1),0)


# --- return_model reuse (Normalizer.transform on new data) — QC P0-1 regression ---

def _fit_new(mode):
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5, scale=3, size=(50, 4))
    normed, model = normalize(X, normalize=mode, return_model=True)
    assert isinstance(model, Normalizer)
    new = rng.normal(loc=5, scale=3, size=(10, 4))
    return normed, model, new


def test_normalizer_reuse_across_bare_array():
    # a fitted Normalizer must apply to a bare 2-D array (the documented
    # return_model reuse pattern) without crashing, single-in -> single-out.
    _, model, new = _fit_new('across')
    out = model.transform(new)                 # bare 2-D array in
    assert isinstance(out, np.ndarray) and out.shape == (10, 4)
    # 'across' reuse applies FIT-TIME mean/std (not recomputed from `new`)
    manual = (new - model.mean_) / model.std_
    assert np.allclose(out, manual)


def test_normalizer_reuse_within_and_row_bare_array():
    for mode in ('within', 'row'):
        _, model, new = _fit_new(mode)
        out = model.transform(new)
        assert isinstance(out, np.ndarray) and out.shape == (10, 4)


def test_normalizer_reuse_list_returns_list():
    # list input still returns a list (internal normalize() path unchanged)
    _, model, new = _fit_new('across')
    out = model.transform([new, new + 1.0])
    assert isinstance(out, list) and len(out) == 2 and out[0].shape == (10, 4)


# --- a 1-D dataset is ONE column in fit and transform alike (review 2026-09-11)

def _one_d_inputs():
    import pandas as pd
    import polars as pl
    values = np.array([1., 2., 3., 4., 6.])
    return values, [
        ('1-D array', values),
        ('pandas Series', pd.Series(values, name='v')),
        ('polars Series', pl.Series('v', values)),
        ('flat list', values.tolist()),
    ]


def test_fitted_normalizer_accepts_the_1d_data_it_was_fit_on():
    # normalize() reads a 1-D array/Series/flat list as one column (via
    # format_data), so its fitted Normalizer is fit on 1 column. Before the
    # fix, .transform() on the SAME data turned it into a single ROW and
    # raised "Normalizer was fit on 1 column(s) but got 5" (the flat list
    # became five one-value datasets).
    values, inputs = _one_d_inputs()
    expected = ((values - values.mean()) / values.std()).reshape(-1, 1)
    for label, obj in inputs:
        for mode in ('across', 'within'):
            normed, model = normalize(obj, normalize=mode, return_model=True)
            assert np.allclose(normed, expected), (label, mode)
            out = model.transform(obj)
            assert isinstance(out, np.ndarray), (label, mode)
            assert out.shape == (5, 1), (label, mode)
            assert np.allclose(out, expected), (label, mode)


def test_normalizer_fit_directly_on_1d_data_reads_one_column():
    values, inputs = _one_d_inputs()
    for label, obj in inputs:
        model = Normalizer('across').fit(obj)
        assert model.mean_.shape == (1,), label
        assert np.allclose(model.mean_, values.mean()), label
        # held-out 1-D data of a different length reuses the fit-time stats
        new = np.array([2., 4.])
        assert np.allclose(model.transform(new),
                           ((new - values.mean()) / values.std())[:, None])


def test_normalizer_list_of_1d_datasets_reads_each_as_one_column():
    a, b = np.array([1., 2., 3.]), np.array([4., 5., 6., 7.])
    normed, model = normalize([a, b], normalize='across', return_model=True)
    out = model.transform([a, b])
    assert [o.shape for o in out] == [(3, 1), (4, 1)]
    for got, want in zip(out, normed):
        assert np.allclose(got, want)
