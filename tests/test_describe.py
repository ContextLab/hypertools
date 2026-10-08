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
    assert set(result.keys()) == {'average', 'individual', 'pooled', 'fig'}
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


# ---------------------------------------------------------------------------
# 'average' is the element-wise mean of the per-dataset curves; the old
# pooled (stacked) curve moved to 'pooled' (1.1 release review). The pooled
# curve mixes in between-dataset distances, so the dashed line labelled
# 'average' was not the mean of the curves drawn beside it.
# ---------------------------------------------------------------------------

def _walks(n=3, rows=40, cols=8, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((rows, cols)), axis=0) + 5 * i
            for i in range(n)]


def test_describe_average_is_mean_of_individual_curves():
    walks = _walks()
    result = describe(walks, reduce='PCA', max_dims=6, show=False)
    plt.close('all')
    individual = np.asarray(result['individual'], dtype=float)
    assert individual.shape == (3, 4)
    np.testing.assert_allclose(result['average'], individual.mean(axis=0))
    # the average lies within the individual curves at every component count
    avg = np.asarray(result['average'])
    assert (avg >= individual.min(axis=0) - 1e-12).all()
    assert (avg <= individual.max(axis=0) + 1e-12).all()


def test_describe_pooled_is_the_stacked_curve():
    walks = _walks()
    result = describe(walks, reduce='PCA', max_dims=6, show=False)
    stacked = describe(np.vstack(walks), reduce='PCA', max_dims=6,
                       show=False)
    plt.close('all')
    np.testing.assert_allclose(result['pooled'], stacked['individual'][0])
    # the pooled curve is a different quantity from the average here
    assert not np.allclose(result['pooled'], result['average'])


def test_describe_single_dataset_average_and_pooled_equal_individual():
    x = _walks(n=1)[0]
    result = describe(x, reduce='PCA', max_dims=6, show=False)
    plt.close('all')
    assert len(result['individual']) == 1
    np.testing.assert_allclose(result['average'], result['individual'][0])
    np.testing.assert_allclose(result['pooled'], result['individual'][0])


def test_describe_average_with_unequal_curve_lengths():
    # with max_dims=None each dataset's sweep is capped at its own
    # dimensionality, so curves can differ in length; the average at each
    # component count is the mean over the datasets that reach it
    rng = np.random.default_rng(1)
    short = rng.standard_normal((4, 6))     # min(shape) = 4 -> dims 2..3
    long = rng.standard_normal((30, 6))     # min(shape) = 6 -> dims 2..5
    result = describe([short, long], reduce='PCA', show=False)
    plt.close('all')
    a, b = result['individual']
    assert len(a) < len(b)
    expected = [np.mean([a[i], b[i]]) if i < len(a) else b[i]
                for i in range(len(b))]
    np.testing.assert_allclose(result['average'], expected)


def test_describe_matplotlib_average_line_draws_the_average():
    walks = _walks()
    result = describe(walks, reduce='PCA', max_dims=6, show=False,
                      backend='matplotlib')
    ax = result['fig'].axes[0]
    avg_lines = [ln for ln in ax.get_lines() if ln.get_label() == 'average']
    assert len(avg_lines) == 1
    np.testing.assert_allclose(avg_lines[0].get_ydata(), result['average'])
    np.testing.assert_array_equal(avg_lines[0].get_xdata(), [2, 3, 4, 5])


def test_describe_plotly_average_trace_draws_the_average():
    pytest.importorskip('plotly')
    walks = _walks()
    result = describe(walks, reduce='PCA', max_dims=6, show=False,
                      backend='plotly')
    avg = [tr for tr in result['fig'].data if tr.name == 'average']
    assert len(avg) == 1
    np.testing.assert_allclose(avg[0].y, result['average'])
    assert list(avg[0].x) == [2, 3, 4, 5]


def test_describe_matplotlib_x_ticks_are_whole_component_counts():
    walks = _walks()
    for max_dims in (4, 5, 6, 8):
        result = describe(walks, reduce='PCA', max_dims=max_dims,
                          show=False, backend='matplotlib')
        fig = result['fig']
        fig.canvas.draw()
        ax = fig.axes[0]
        lo, hi = ax.get_xlim()
        ticks = [t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9]
        assert ticks, max_dims
        assert all(float(t).is_integer() for t in ticks), (max_dims, ticks)
        labels = [lbl.get_text() for lbl in ax.get_xticklabels()
                  if lbl.get_text()]
        assert all('.' not in lbl for lbl in labels), (max_dims, labels)


def test_describe_plotly_x_ticks_are_whole_component_counts():
    pytest.importorskip('plotly')
    walks = _walks()
    for max_dims in (4, 5, 6, 8):
        result = describe(walks, reduce='PCA', max_dims=max_dims,
                          show=False, backend='plotly')
        xaxis = result['fig'].layout.xaxis
        assert xaxis.tickmode == 'array'
        vals = list(xaxis.tickvals)
        assert vals and all(float(v).is_integer() for v in vals), vals
        assert min(vals) >= 2 and max(vals) <= max_dims - 1


# ---------------------------------------------------------------------------
# describe(show=False) returns the drawn figure without displaying it, like
# hyp.plot(show=False) (1.1 release review; it used to return fig=None)
# ---------------------------------------------------------------------------

def test_describe_show_false_returns_matplotlib_figure_without_registering():
    import io
    plt.close('all')
    before = set(plt.get_fignums())
    result = describe(_walks(), reduce='PCA', max_dims=6, show=False,
                      backend='matplotlib')
    fig = result['fig']
    assert isinstance(fig, plt.Figure)
    # closed like plot(show=False): not left registered with pyplot
    assert set(plt.get_fignums()) == before
    # ... but still renderable and savable
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    assert buf.getvalue()[:8] == b'\x89PNG\r\n\x1a\n'
    assert len(fig.axes[0].get_lines()) >= 4  # 3 datasets + average


def test_describe_show_false_returns_plotly_figure_without_showing(
        monkeypatch):
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    shown = []
    monkeypatch.setattr(go.Figure, 'show',
                        lambda self, *a, **k: shown.append(self))
    result = describe(_walks(), reduce='PCA', max_dims=6, show=False,
                      backend='plotly')
    assert isinstance(result['fig'], go.Figure)
    assert shown == []
    assert len(result['fig'].data) == 4  # 3 datasets + average
