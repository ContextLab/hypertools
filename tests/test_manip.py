# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

import pytest
import hypertools as hyp

weights = hyp.load('weights_sample')


def test_normalize():
    x1 = hyp.manip(weights, model='Normalize')
    assert all([dw.util.btwn(w, -0.0001, 1.0001) for w in x1])
    assert len(x1) == len(weights)
    assert all([x.shape == w.shape for x, w in zip(x1, weights)])

    x2 = hyp.manip(weights, model='Normalize', min=-1, max=36)
    assert all([not dw.util.btwn(w, -0.0001, 1.0001) for w in x2])
    assert all([dw.util.btwn(w, -1.0001, 36.0001) for w in x2])
    assert len(x2) == len(weights)
    assert all([x.shape == w.shape for x, w in zip(x2, weights)])


def test_zscore():
    x1 = hyp.manip(weights, model='ZScore')
    assert np.allclose(dw.stack(x1).mean(axis=0), 0, atol=1e-5)
    assert np.allclose(dw.stack(x1).std(axis=0), 1, atol=1e-5)
    assert len(x1) == len(weights)
    assert all([x.shape == w.shape for x, w in zip(x1, weights)])

    x2 = hyp.manip(weights, model='ZScore', axis=1)
    assert all([np.allclose(w.mean(axis=1), 0, atol=1e-5) for w in x2])
    assert all([np.allclose(w.std(axis=1), 1, atol=1e-5) for w in x2])
    assert len(x2) == len(weights)
    assert all([x.shape == w.shape for x, w in zip(x2, weights)])


def test_resample():
    n_samples = 500
    x1 = hyp.manip(weights, model='Resample', n_samples=n_samples)
    assert all([w.shape[0] == n_samples for w in x1])
    assert all([p.shape[1] == q.shape[1] for p, q in zip(x1, weights)])
    assert len(x1) == len(weights)

    m_samples = 10
    x2 = hyp.manip(weights, model='Resample', n_samples=m_samples, axis=1)
    assert all([w.shape[1] == m_samples for w in x2])
    assert all([p.shape[0] == q.shape[0] for p, q in zip(x2, weights)])
    assert len(x2) == len(weights)

    # test resampling back to original shape (along both axes, in succession)
    resample_axis0 = {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': weights[0].shape[0], 'axis': 0}}
    resample_axis1 = {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': weights[0].shape[1], 'axis': 1}}
    x3 = hyp.manip(weights, model=[resample_axis0, resample_axis1])
    assert all([x.shape == w.shape for x, w in zip(x3, weights)])
    assert all([np.allclose(x, w) for x, w in zip(x3, weights)])
    assert len(x3) == len(weights)


def test_smooth():
    x1 = hyp.manip(weights, model='Smooth', maintain_bounds=True)
    assert all([p.shape == w.shape for p, w in zip(x1, weights)])
    assert all([dw.util.btwn(p, np.min(w), np.max(w)) for p, w in zip(x1, weights)])

    model1 = {'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 10}}
    x2 = hyp.manip(weights, model=model1, maintain_bounds=True)
    assert all([np.allclose(a, b) for a, b in zip(x1, x2)])

    model2 = {'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 10.5}}
    x3 = hyp.manip(weights, model=model2, maintain_bounds=True)
    assert all([np.allclose(a, b) for a, b in zip(x2, x3)])

    x4 = hyp.manip(weights, model='Smooth', maintain_bounds=True, axis=1)
    assert all([x.shape == w.shape for x, w in zip(x4, weights)])


def test_zscore_smooth_resample_smooth():
    x = hyp.manip(weights, model=['ZScore', 'Smooth', 'Resample', 'Smooth'])
    assert all([w.shape == (100, weights[0].shape[1]) for w in x])
    assert all([type(w) is pd.DataFrame for w in x])


def test_preprocessing():
    models = ['Binarizer', 'MaxAbsScaler']
    x1 = hyp.manip(weights, model=models)
    assert all([x.shape == w.shape for x, w in zip(x1, weights)])

    x2 = hyp.manip(weights, model=[*models, 'Smooth'])
    assert all([x.shape == w.shape for x, w in zip(x2, weights)])
