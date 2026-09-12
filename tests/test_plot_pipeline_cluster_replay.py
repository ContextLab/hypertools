"""`hyp.plot(x, pipeline=p)` applies a fitted pipeline's trailing cluster
step (1.1 release review, 2026-09-11, L13).

`plot()` documents `pipeline=` as running "in place of the manip/normalize/
reduce/align/cluster stages", and the `return_model=True` bundle's pipeline
ends in a fitted 'cluster' step when the figure was clustered. Reusing it
drew every point in one colour with no warning: `analyze(x, pipeline=p)`
returns the transformed DATA (documented -- the labels are recovered with
``p.named_steps['cluster'].transform(data)``), and `plot()` never did that
recovery. The reuse figure is now coloured by the fitted clusters, with the
fit figure's label-to-colour mapping, on both backends. Real data, real
calls, no mocks.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

import hypertools as hyp


def _walks():
    a = np.asarray(hyp.load('random_walk', n_samples=48, n_features=6,
                            random_state=41))
    b = np.asarray(hyp.load('random_walk', n_samples=48, n_features=6,
                            random_state=42))
    return a, b


def _fit_bundle(a, cluster='KMeans', **kwargs):
    return hyp.plot(a, '.', manip='ZScore', normalize='across',
                    reduce='PCA', ndims=2, cluster=cluster, random_state=0,
                    return_model=True, backend='matplotlib', show=False,
                    **kwargs)


def _line_colors(fig):
    """One colour per drawn marker trace, in drawing order."""
    return [to_hex(line.get_color()) for line in fig.axes[0].lines]


def test_reused_pipeline_colours_the_figure_by_its_fitted_clusters():
    a, b = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    pipe = fit['pipeline']
    assert list(pipe.named_steps)[-1] == 'cluster'
    fit_colors = _line_colors(fit['fig'])
    assert len(set(fit_colors)) == 3

    expected = pipe.named_steps['cluster'].transform(
        hyp.analyze(b, pipeline=pipe))
    reuse = hyp.plot(b, '.', pipeline=pipe, return_model=True,
                     backend='matplotlib', show=False)
    try:
        assert list(reuse['models']['cluster_labels']) == list(expected)
        present = sorted(set(expected))
        assert len(present) > 1
        # one marker trace per cluster present, each in the colour the FIT
        # figure gave that cluster (labels sort 0, 1, 2 in both)
        assert _line_colors(reuse['fig']) == [fit_colors[k] for k in present]
    finally:
        plt.close(fit['fig'])
        plt.close(reuse['fig'])


def test_reused_pipeline_on_the_fit_data_redraws_the_fit_figure_colours():
    a, _ = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    fig = hyp.plot(a, '.', pipeline=fit['pipeline'], backend='matplotlib',
                   show=False)
    try:
        assert _line_colors(fig) == _line_colors(fit['fig'])
    finally:
        plt.close(fit['fig'])
        plt.close(fig)


def test_reused_pipeline_colours_a_line_plot_too():
    a, b = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    fig = hyp.plot(b, pipeline=fit['pipeline'], backend='matplotlib',
                   show=False)
    try:
        assert len(set(_line_colors(fig))) > 1
    finally:
        plt.close(fit['fig'])
        plt.close(fig)


def test_reused_pipeline_colours_the_plotly_figure():
    pytest.importorskip('plotly')
    a, b = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    plt.close(fit['fig'])
    fig = hyp.plot(b, '.', pipeline=fit['pipeline'], backend='plotly',
                   show=False)
    colors = {str(trace.marker.color) for trace in fig.data
              if getattr(trace, 'marker', None) is not None
              and trace.marker.color is not None}
    assert len(colors) > 1, colors


def test_reused_mixture_pipeline_blends_its_fitted_memberships():
    a, b = _walks()
    fit = _fit_bundle(a, cluster='GaussianMixture', n_clusters=3)
    pipe = fit['pipeline']
    expected = np.asarray(pipe.named_steps['cluster'].transform(
        hyp.analyze(b, pipeline=pipe)))
    assert expected.ndim == 2 and expected.shape[1] == 3
    reuse = hyp.plot(b, '.', pipeline=pipe, return_model=True,
                     backend='matplotlib', show=False)
    try:
        np.testing.assert_allclose(
            np.asarray(reuse['models']['cluster_labels']), expected)
        colors = {tuple(np.round(c, 4)) for coll in reuse['fig'].axes[0].collections
                  for c in coll.get_facecolors()}
        assert len(colors) > 1
    finally:
        plt.close(fit['fig'])
        plt.close(reuse['fig'])


def test_a_cluster_step_that_cannot_label_new_data_warns_clearly():
    """AgglomerativeClustering has no out-of-sample predict: the fitted
    step cannot label a dataset with a different row count, so the figure
    is drawn without cluster colours and a warning says why (it used to
    drop the clusters silently)."""
    a, b = _walks()
    fit = _fit_bundle(a, cluster='AgglomerativeClustering', n_clusters=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(b[:30], '.', pipeline=fit['pipeline'],
                       backend='matplotlib', show=False)
    try:
        notes = [str(w.message) for w in caught
                 if 'cluster step' in str(w.message)]
        assert len(notes) == 1, [str(w.message) for w in caught]
        assert 'AgglomerativeClustering' in notes[0]
        assert caught[[str(w.message) for w in caught].index(notes[0])
                      ].filename == __file__
        assert len(set(_line_colors(fig))) == 1
    finally:
        plt.close(fit['fig'])
        plt.close(fig)


def test_the_bundle_reports_the_fitted_clusterer_it_replayed():
    from hypertools.cluster.common import Clusterer
    a, b = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    reuse = hyp.plot(b, '.', pipeline=fit['pipeline'], return_model=True,
                     backend='matplotlib', show=False)
    try:
        assert isinstance(reuse['models']['cluster'], Clusterer)
        assert reuse['models']['cluster'].is_fitted
        assert reuse['pipeline'] is fit['pipeline']
    finally:
        plt.close(fit['fig'])
        plt.close(reuse['fig'])


def test_an_explicit_hue_wins_over_the_pipelines_clusters():
    a, b = _walks()
    fit = _fit_bundle(a, n_clusters=3)
    hue = ['x'] * 24 + ['y'] * 24
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(b, '.', pipeline=fit['pipeline'], hue=hue,
                       backend='matplotlib', show=False)
    try:
        assert not [w for w in caught if 'overrides hue' in str(w.message)]
        assert len(fig.axes[0].lines) == 2
    finally:
        plt.close(fit['fig'])
        plt.close(fig)


def test_a_pipeline_without_a_cluster_step_is_unchanged():
    a, b = _walks()
    _, pipe = hyp.analyze(a, reduce='PCA', ndims=2, return_model=True)
    fig = hyp.plot(b, '.', pipeline=pipe, backend='matplotlib', show=False)
    try:
        assert len(set(_line_colors(fig))) == 1
    finally:
        plt.close(fig)
