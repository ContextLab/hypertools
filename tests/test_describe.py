# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.reduce.describe import describe

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)


def test_describe_data_is_dict():
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict


def test_describe_geo():
    # describe() operates on raw data directly (no geo round-trip in 1.0)
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict


def test_describe_exposed_at_top_level():
    # QC 2026-07: confirm hyp.describe is the public entry point
    assert hyp.describe is describe


def test_describe_matplotlib_removes_top_and_right_spines():
    # QC 2026-07 (Jeremy): the describe plot should drop the top/right spines
    # (seaborn sns.despine(top=True, right=True)).
    plt.close('all')
    describe(data, reduce='PCA', max_dims=6, show=True, backend='matplotlib')
    ax = plt.gca()
    assert ax.spines['top'].get_visible() is False
    assert ax.spines['right'].get_visible() is False
    # the data axes stay
    assert ax.spines['left'].get_visible() is True
    assert ax.spines['bottom'].get_visible() is True
    plt.close('all')


def test_describe_plotly_backend_runs_and_returns_dict(monkeypatch):
    # the plotly backend renders an interactive go.Figure (Jeremy's "also
    # support plotly") and still returns the same dict. Suppress the actual
    # display so the test is headless.
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    monkeypatch.setattr(go.Figure, 'show', lambda self, *a, **k: None)
    result = describe(data, reduce='PCA', max_dims=6, show=True,
                      backend='plotly')
    assert type(result) is dict
    # 'fig' added by the 2026-07 release audit (F11-reduce-describe-015):
    # describe() now hands back its figure so it can be saved/embedded
    assert set(result.keys()) == {'average', 'individual', 'fig'}
    assert isinstance(result['fig'], go.Figure)


# ---------------------------------------------------------------------------
# get_corr / get_cdist (2026-07 release audit, X7-code-org-rest-026: these
# public helpers had no direct tests -- describe() itself was tested, but a
# regression in either helper would only surface indirectly)
# ---------------------------------------------------------------------------

def test_get_cdist_matches_scipy_on_known_points():
    from scipy.spatial.distance import cdist as scipy_cdist
    from hypertools.reduce.describe import get_cdist

    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    out = get_cdist(pts)

    assert out.shape == (3, 3)
    # hand-computed Euclidean distances: |p0-p1| = 5, |p0-p2| = 10, |p1-p2| = 5
    expected = np.array([[0.0, 5.0, 10.0],
                         [5.0, 0.0, 5.0],
                         [10.0, 5.0, 0.0]])
    assert np.allclose(out, expected)
    assert np.allclose(out, scipy_cdist(pts, pts))
    # metric properties on real random data
    rng = np.random.RandomState(0)
    x = rng.rand(15, 4)
    d = get_cdist(x)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert (d >= 0).all()


def test_get_corr_perfect_and_known_correlations():
    from hypertools.reduce.describe import get_cdist, get_corr

    rng = np.random.RandomState(0)
    x = rng.rand(12, 5)
    d = get_cdist(x)

    # identical matrices correlate perfectly
    assert np.isclose(get_corr(d, d), 1.0)
    # an exact linear rescaling also correlates perfectly (Pearson)
    assert np.isclose(get_corr(2.5 * d + 1.0, d), 1.0)
    # agreement with an independent Pearson computation on real matrices
    other = get_cdist(rng.rand(12, 5))
    expected = np.corrcoef(d.ravel(), other.ravel())[0, 1]
    assert np.isclose(get_corr(other, d), expected)
    # correlation of distances between reduced and full data is high for a
    # faithful reduction of intrinsically low-dimensional data
    low_d = rng.rand(20, 2)
    embedded = np.hstack([low_d, low_d @ rng.rand(2, 3)])
    reduced = hyp.reduce(embedded, reduce='PCA', ndims=2)
    r = get_corr(get_cdist(np.asarray(reduced)), get_cdist(embedded))
    assert r > 0.95
