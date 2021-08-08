# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

import hypertools as hyp

weights = hyp.load('weights_sample')


def test_normalize():
    x1 = hyp.manip(weights, model='Normalize')
    assert all([dw.util.btwn(w, -0.0001, 1.0001) for w in x1])
    assert len(x1) == len(weights)

    x2 = hyp.manip(weights, model='Normalize', min=-1, max=36)
    assert all([not dw.util.btwn(w, -0.0001, 1.0001) for w in x2])
    assert all([dw.util.btwn(w, -1.0001, 36.0001) for w in x2])
    assert len(x2) == len(weights)


def test_zscore():
    x1 = hyp.manip(weights, model='ZScore')
    assert all([np.allclose(w.mean(axis=0), 0, atol=1e-5) for w in x1])
    assert all([np.allclose(w.std(axis=0), 1, atol=1e-5) for w in x1])
    assert len(x1) == len(weights)

    x2 = hyp.manip(weights, model='ZScore', axis=1)
    assert all([np.allclose(w.mean(axis=1), 0, atol=1e-5) for w in x2])
    assert all([np.allclose(w.std(axis=1), 1, atol=1e-5) for w in x2])
    assert len(x1) == len(weights)


def test_resample():
    n_samples = 500
    x1 = hyp.manip(weights, model='Resample', n_samples=n_samples)
    assert all([w.shape[0] == n_samples for w in x1])
    assert all([p.shape[1] == q.shape[1] for p, q in zip(x1, weights)])

    m_samples = 10
    x2 = hyp.manip(weights, model='Resample', n_samples=m_samples, axis=1)
    assert all([w.shape[1] == m_samples for w in x2])
    assert all([p.shape[0] == q.shape[0] for p, q in zip(x2, weights)])


def test_smooth():
    x1 = hyp.manip(weights, model='Smooth', maintain_bounds=True)
    assert all([p.shape == w.shape for p, w in zip(x1, weights)])
    assert all([dw.util.btwn(p, np.min(w), np.max(w)) for p, w in zip(x1, weights)])


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


test_normalize()
test_zscore()
test_resample()
test_smooth()
test_zscore_smooth_resample_smooth()
test_preprocessing()
