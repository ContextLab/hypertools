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
