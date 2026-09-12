"""``fit()`` returns the fitted instance on every model base (release review
of 1.1.0): the sklearn chain ``Model().fit(x).transform(y)`` must work for
manipulators, aligners and imputers alike. Real data, no mocks."""

import numpy as np
import pandas as pd
import pytest

from hypertools.align.hyperalign import HyperAlign
from hypertools.align.null import NullAlign
from hypertools.align.procrustes import Procrustes
from hypertools.impute.ppca import PPCA
from hypertools.manip.delay import Delay
from hypertools.manip.normalize import Normalize
from hypertools.manip.resample import Resample
from hypertools.manip.smooth import Smooth
from hypertools.manip.zscore import ZScore


def _walk(seed, rows=40, cols=3):
    # the model classes take DataFrames (the hyp.* dispatchers wrap arrays)
    return pd.DataFrame(
        np.cumsum(np.random.default_rng(seed).normal(size=(rows, cols)), 0))


@pytest.mark.parametrize('make', [
    Normalize, ZScore, lambda: Smooth(kernel='boxcar', kernel_width=5),
    lambda: Resample(n_samples=20), lambda: Delay(tau=2, dims=3),
], ids=['Normalize', 'ZScore', 'Smooth', 'Resample', 'Delay'])
def test_manipulator_fit_returns_self_and_chains(make):
    model = make()
    x = _walk(0)
    assert model.fit(x) is model
    chained = make().fit(x).transform(_walk(1))
    separate = make()
    separate.fit(x)
    np.testing.assert_array_equal(np.asarray(chained),
                                  np.asarray(separate.transform(_walk(1))))


@pytest.mark.parametrize('make', [HyperAlign, Procrustes, NullAlign],
                         ids=['HyperAlign', 'Procrustes', 'NullAlign'])
def test_aligner_fit_returns_self_and_chains(make):
    xs = [_walk(0), _walk(1), _walk(2)]
    model = make()
    assert model.fit(xs) is model
    chained = make().fit(xs).transform(xs)
    separate = make()
    separate.fit(xs)
    for a, b in zip(chained, separate.transform(xs)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_imputer_fit_returns_self_and_chains():
    x = _walk(0, rows=60, cols=4)
    x.iloc[5, 1] = np.nan
    x.iloc[17, 3] = np.nan
    model = PPCA()
    assert model.fit(x) is model
    filled = PPCA().fit(x).transform(x)
    assert not np.isnan(np.asarray(filled)).any()
