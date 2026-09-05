# -*- coding: utf-8 -*-
"""Regression tests for the batch-B5 audit fixes (release-1.0 audit,
2026-07): plot()'s cluster integration (F13-cluster-001/-002/-003/-004/
-009/-010/-020/-021/-022), small-cardinality integer hue (F13-cluster-005),
streaming kwarg handling (F22-io-004), and plot-package code-org fixes
(X6-code-org-plot-005/-008/-009).

All tests draw real figures on the Agg backend with real data -- no mocks.
"""
import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors
import numpy as np
import pytest

import hypertools as hyp


def three_blobs(seed=3, n=100, d=4, spread=7):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.standard_normal((n, d)) + spread * i
                      for i in range(3)])


def point_groups(fig):
    """Sorted per-artist x-coordinate tuples for a '.'-format plot (one
    Line3D/Line2D artist per drawn group)."""
    out = []
    for ln in fig.axes[0].lines:
        try:
            xs = np.asarray(ln.get_data_3d()[0])
        except Exception:
            xs = np.asarray(ln.get_data()[0])
        if xs.size:
            out.append(tuple(np.round(np.sort(xs), 6)))
    return sorted(out)


def artist_colors(fig):
    return [tuple(np.round(mcolors.to_rgb(ln.get_color()), 3))
            for ln in fig.axes[0].lines]


# --- F13-cluster-001: FeatureAgglomeration must be rejected by plot() ---

@pytest.mark.parametrize('spec', [
    'FeatureAgglomeration',
    {'model': 'FeatureAgglomeration', 'n_clusters': 2},
])
def test_plot_featureagglomeration_raises_instructive_error(spec):
    data = three_blobs()[:, :4]
    with pytest.raises(ValueError, match='clusters features'):
        hyp.plot(data, '.', cluster=spec, show=False)


# --- F13-cluster-002: density/bandwidth clusterers ignore n_clusters= ---

@pytest.mark.parametrize('model', ['DBSCAN', 'MeanShift', 'OPTICS',
                                   'AffinityPropagation', 'HDBSCAN'])
def test_plot_density_clusterers_ignore_n_clusters_with_warning(model):
    data = three_blobs(n=30)
    with pytest.warns(UserWarning,
                      match='n_clusters is not a valid parameter'):
        fig = hyp.plot(data, '.', cluster=model, n_clusters=3, show=False)
    assert len(point_groups(fig)) >= 1  # rendered, did not crash


# --- F13-cluster-003: random_state= threads into the cluster stage ---

def test_plot_random_state_makes_cluster_plots_reproducible():
    noise = np.random.default_rng(3).standard_normal((120, 10))
    f1 = hyp.plot(noise, '.', cluster='KMeans', n_clusters=6,
                  random_state=0, show=False)
    f2 = hyp.plot(noise, '.', cluster='KMeans', n_clusters=6,
                  random_state=0, show=False)
    g1, g2 = point_groups(f1), point_groups(f2)
    assert len(g1) == 6
    assert g1 == g2


# --- F13-cluster-004/-021: bundle pipeline matches the figure's k ---

def test_plot_return_model_bundle_matches_figure_default_k():
    data = three_blobs()
    res = hyp.plot(data, '.', cluster='KMeans', random_state=0,
                   return_model=True, show=False)
    fig_groups = point_groups(res['fig'])
    pipe_k = len(set(map(int, res['pipeline'].transform(data))))
    # shared default k=3 (same as hyp.cluster) on BOTH sides
    assert len(fig_groups) == 3
    assert pipe_k == 3


def test_plot_return_model_bundle_matches_explicit_n_clusters():
    data = three_blobs()
    res = hyp.plot(data, '.', cluster='KMeans', n_clusters=4,
                   random_state=0, return_model=True, show=False)
    assert len(point_groups(res['fig'])) == 4
    assert len(set(map(int, res['pipeline'].transform(data)))) == 4


def test_plot_return_model_bundle_includes_n_clusters_only_kmeans():
    data = three_blobs()
    res = hyp.plot(data, '.', n_clusters=4, random_state=0,
                   return_model=True, show=False)
    assert len(point_groups(res['fig'])) == 4
    assert len(set(map(int, res['pipeline'].transform(data)))) == 4


def test_plot_return_model_bundle_mixture_components_match_figure():
    data = three_blobs()
    res = hyp.plot(data, '.', cluster='GaussianMixture', random_state=0,
                   return_model=True, show=False)
    props = np.asarray(res['pipeline'].transform(data))
    # figure blends the default 3 components; bundle must agree
    assert props.shape == (data.shape[0], 3)


# --- F13-cluster-009: spec kwargs beat n_clusters=, with a warning ---

def test_plot_cluster_spec_kwargs_beat_n_clusters_with_warning():
    data = three_blobs()
    spec = {'model': 'KMeans', 'kwargs': {'n_clusters': 2,
                                          'random_state': 0}}
    with pytest.warns(UserWarning, match='conflicts with the cluster spec'):
        fig = hyp.plot(data, '.', cluster=spec, n_clusters=4, show=False)
    assert len(point_groups(fig)) == 2  # the spec's k, matching hyp.cluster


def test_plot_cluster_dict_top_level_n_clusters():
    data = three_blobs()
    fig = hyp.plot(data, '.', random_state=0,
                   cluster={'model': 'KMeans', 'n_clusters': 4}, show=False)
    assert len(point_groups(fig)) == 4


# --- F13-cluster-010: dict spec without 'model' -> instructive error ---

def test_plot_cluster_dict_missing_model_raises_valueerror():
    with pytest.raises(ValueError, match="value of the 'model' key"):
        hyp.plot(np.random.default_rng(0).standard_normal((60, 4)), '.',
                 cluster={'kwargs': {'n_clusters': 3}}, show=False)


# --- F13-cluster-020: full cluster spec grammar accepted by plot() ---

def test_plot_cluster_accepts_bare_class():
    from sklearn.cluster import KMeans
    data = three_blobs()
    fig = hyp.plot(data, '.', cluster=KMeans, n_clusters=3,
                   random_state=0, show=False)
    assert len(point_groups(fig)) == 3


def test_plot_cluster_accepts_instance_with_own_params():
    from sklearn.cluster import KMeans
    data = three_blobs()
    fig = hyp.plot(data, '.',
                   cluster=KMeans(n_clusters=4, n_init=10, random_state=0),
                   show=False)
    assert len(point_groups(fig)) == 4


# --- F13-cluster-022: cluster legend entries in sorted label order ---

def test_plot_cluster_legend_sorted_numerically():
    data = three_blobs()
    fig = hyp.plot(data, '.', cluster='KMeans', n_clusters=3,
                   random_state=0, legend=True, show=False)
    entries = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert entries == ['0', '1', '2']


# --- F13-cluster-005: small-cardinality integer hue is categorical ---

def test_integer_cluster_labels_hue_renders_distinct_colors():
    # verbatim workflow from examples/plot_clusters.py (hyp.cluster cell): cluster labels
    # passed to hue= must NOT collapse into near-identical adjacent reds
    rng = np.random.default_rng(0)
    data = np.vstack([rng.standard_normal((100, 3)),
                      rng.standard_normal((100, 3)) + 3])
    labels = hyp.cluster(data, n_clusters=2, random_state=0)
    fig = hyp.plot(data, '.', hue=list(labels), show=False)
    colors = artist_colors(fig)
    assert len(fig.axes[0].lines) == 2  # one trace per label group
    assert len(set(colors)) == 2
    deltas = np.abs(np.asarray(colors[0]) - np.asarray(colors[1]))
    assert deltas.max() > 0.2  # visually distinct (was 0.031 pre-fix)


def test_integer_hue_groups_and_legend_sorted_numerically():
    rng = np.random.default_rng(1)
    # first appearance order 2, 0, 1 -- groups must still sort 0, 1, 2
    hue = [2] * 10 + [0] * 10 + [1] * 10
    data = rng.standard_normal((30, 4))
    fig = hyp.plot(data, '.', hue=hue, legend=True, show=False)
    entries = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert entries == ['0', '1', '2']
    assert len(fig.axes[0].lines) == 3


def test_boolean_hue_is_categorical():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((40, 4))
    hue = [False] * 20 + [True] * 20
    fig = hyp.plot(data, '.', hue=hue, legend=True, show=False)
    entries = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert entries == ['False', 'True']


def test_high_cardinality_integer_hue_stays_continuous():
    # n_unique == n_obs (an index/time axis): continuous per-point colors,
    # rendered via collections, not one-trace-per-value grouping
    rng = np.random.default_rng(0)
    data = np.cumsum(rng.standard_normal((50, 8)), axis=0)
    fig = hyp.plot(data, hue=np.arange(50), show=False)
    assert len(fig.axes[0].lines) == 0
    assert len(fig.axes[0].collections) > 0


def test_float_hue_stays_continuous_even_at_low_cardinality():
    # the categorical shortcut is for INTEGER/boolean dtypes only
    rng = np.random.default_rng(0)
    data = rng.standard_normal((40, 4))
    fig = hyp.plot(data, '.', hue=np.asarray([0.0, 1.0] * 20), show=False)
    assert len(fig.axes[0].collections) > 0


# --- F22-io-004: streaming inputs name their dropped parameters ---

def _stream(n=60, d=5, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield rng.standard_normal(d)


def test_plot_stream_warns_on_unsupported_kwargs():
    with pytest.warns(UserWarning, match='no streaming implementation'):
        fig = hyp.plot(_stream(), stream_init=20, stream_max=40, show=False,
                       hue=np.arange(40), legend=True)
    assert fig.stream_info['n_samples'] == 40


def test_plot_stream_supported_kwargs_do_not_warn():
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter('always')
        fig = hyp.plot(_stream(), stream_init=20, stream_chunk=10,
                       stream_max=50, linewidth=2, title='s', show=False)
    assert not [w for w in rec
                if 'no streaming implementation' in str(w.message)]
    assert fig.stream_info['n_samples'] == 50


# --- X6-code-org-plot-009: animated trails honor the user's linewidth ---

@pytest.mark.parametrize('ndims', [3, 2])
def test_animated_trail_honors_linewidth(ndims):
    traj = np.cumsum(np.random.default_rng(0).standard_normal((40, 3)),
                     axis=0)
    fig, ani = hyp.plot(traj, ndims=ndims, animate=True, chemtrails=True,
                        linewidth=5, duration=1, show=False)
    widths = sorted(ln.get_linewidth() for ln in fig.axes[0].lines)
    assert widths == [5.0, 5.0]  # head AND trail (trail was 1.0 pre-fix)


# --- X6-code-org-plot-008: the caller's fmt list is never mutated ---

def test_plot_does_not_mutate_callers_fmt_list():
    rng = np.random.default_rng(0)
    single = rng.standard_normal((1, 4))
    multi = rng.standard_normal((10, 4))
    user_fmt = ['-', '-']
    # NOTE (H1 polish wave): this used to be wrapped in pytest.warns
    # (UserWarning) on the mistaken premise that single-point line datasets
    # warn -- they are silently drawn as points. The wrapper only ever
    # matched a STRAY test-order-dependent UserWarning (matplotlib's
    # 'Animation was deleted without rendering anything' fired at GC by
    # earlier animation tests -- the exact noise X4-warnings-012 removed),
    # so it failed on a pristine tree in isolation. The point here is only
    # that the caller's list object stays untouched.
    hyp.plot([single, multi], user_fmt, show=False)
    assert user_fmt == ['-', '-']


# --- X6-code-org-plot-005: plotly morph reuses the matched clouds ---

def test_plotly_morph_animation_runs_hungarian_once(monkeypatch):
    from hypertools.plot import morph as morph_mod
    calls = {'n': 0}
    real = morph_mod.sample_and_match_clouds

    def counting(*args, **kwargs):
        calls['n'] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(morph_mod, 'sample_and_match_clouds', counting)
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((30, 3)), rng.standard_normal((30, 3)) + 4]
    fig = hyp.plot(data, '.', animate='morph', duration=1,
                   backend='plotly', show=False)
    assert fig is not None
    assert calls['n'] == 1  # was 2 (once static setup, once animation)
