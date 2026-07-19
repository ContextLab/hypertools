"""Hue/color coloring for hyp.plot (QC 2026-07, Jeremy's notes).

B1  a mixture/proportion matrix hue must BLEND the component colors -- a
    [0.5, 0.5] row is a true 50/50 blend, not the argmax component. (The old
    mat2colors subtracted the per-row min before normalizing, collapsing every
    row onto a pure palette vertex, so soft clusters never blended.)
B2  surface=True must honor hue -- the hull color follows the dataset's mean
    hue color instead of the palette cycle.
B3  an arbitrary high-dimensional matrix hue (>3 columns, or any matrix with
    color_reduce=) is reduced to 3 columns and mapped directly to (r, g, b).

All data is real numeric arrays -- no mocks. Plots run headless (Agg).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp
from hypertools.plot.colors import mat2colors


# --- B1: proportion-matrix hue blends ----------------------------------

def test_mat2colors_blends_proportions_not_argmax():
    import seaborn as sns
    props = np.array([[0.5, 0.5], [0.9, 0.1], [0.466, 0.534]])
    cols = mat2colors(props, palette='hls')
    base = np.asarray(sns.color_palette('hls', 2))[:, :3]
    # [0.5, 0.5] is exactly the midpoint of the two component colors
    assert np.allclose(cols[0], 0.5 * base[0] + 0.5 * base[1], atol=1e-6)
    # a near-even row is NOT the same color as a lopsided row (old bug: both
    # collapsed to the pure argmax component)
    assert not np.allclose(cols[2], cols[1], atol=0.05)


def test_mat2colors_signed_matrix_still_shifts():
    # signed matrices (negative entries) are still shifted to non-negative
    # before blending -- must not crash and must return valid RGB
    signed = np.array([[-1.0, 1.0], [2.0, -2.0], [0.0, 0.0]])
    cols = mat2colors(signed, palette='hls')
    assert cols.shape == (3, 3)
    assert cols.min() >= 0 and cols.max() <= 1


def test_gaussian_mixture_soft_cluster_produces_many_blended_colors():
    rng = np.random.default_rng(0)
    xc = np.vstack([rng.normal(loc=[0, 0], scale=10, size=(100, 2)),
                    rng.normal(loc=[5, 5], scale=7, size=(100, 2)),
                    rng.normal(loc=[0, 5], scale=6, size=(100, 2))])
    soft = np.asarray(hyp.cluster(xc, cluster='GaussianMixture', n_clusters=2))
    cols = mat2colors(soft, palette='hls')
    # overlapping blobs -> lots of intermediate memberships -> many blended
    # colors (the old code produced exactly 2)
    assert len(np.unique(np.round(cols, 2), axis=0)) > 10


# --- B2: surface honors hue --------------------------------------------

def test_surface_hue_renders_without_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    fig = hyp.plot(x, ndims=3, surface=True, hue=np.linspace(0, 1, 200),
                   palette='viridis', show=False)
    # the surface mesh color should track the mean hue color, not palette C0;
    # here we just assert it renders (the visual check is in the PR evidence)
    assert fig is not None


# --- B3: color_reduce / high-dim matrix hue -> RGB ---------------------

def test_matrix_hue_gt3_columns_reduces_to_rgb():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(120, 4)), axis=0)
    h10 = rng.normal(size=(120, 10))
    fig = hyp.plot(x, '.', ndims=3, hue=h10, show=False)  # auto IncrementalPCA->RGB
    assert fig is not None


def test_color_reduce_kwarg_accepts_reduce_spec():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(120, 4)), axis=0)
    h5 = rng.normal(size=(120, 5))
    fig = hyp.plot(x, '.', ndims=3, hue=h5, color_reduce='PCA', show=False)
    assert fig is not None


@pytest.mark.parametrize('k', [1, 2, 3, 5, 10])
def test_color_reduce_handles_any_column_count(k):
    # red-team follow-up: color_reduce= with a <=3-column matrix used to crash
    # (hyp.reduce(ndims=3) can't synthesize dims from <=3 features). Now <=3
    # columns are used directly (scaled + padded); >3 are reduced.
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(60, 4)), axis=0)
    hue = rng.normal(size=(60, k)) if k > 1 else rng.normal(size=(60,))
    # both marker and line paths
    assert hyp.plot(x, '.', ndims=3, hue=hue, color_reduce='PCA',
                    show=False) is not None
    assert hyp.plot(x, '-', ndims=3, hue=hue, color_reduce='PCA',
                    show=False) is not None


def test_matrix_hue_le3_columns_stays_palette_blend():
    # a <=3-column matrix without color_reduce keeps blending over the palette
    # (mixture-proportion semantics), not the RGB-reduce path
    rng = np.random.default_rng(0)
    h3 = np.abs(rng.normal(size=(60, 3)))
    h3 /= h3.sum(axis=1, keepdims=True)
    fig = hyp.plot(np.cumsum(rng.normal(size=(60, 3)), axis=0), '.', ndims=3,
                   hue=h3, show=False)
    assert fig is not None


# --- nested per-dataset hue (examples/plot_hue.py) ---------------------

def test_nested_per_dataset_hue_flattens():
    # when the data is a list of datasets, hue may be given with the SAME
    # nesting: one hue sub-list per dataset (the classic list-of-lists form
    # from examples/plot_hue.py). It must flatten to one value per observation,
    # NOT be misread as a (n_datasets, len) matrix hue -> used to raise
    # "hue has 3 entries but the data has 900 observations".
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((300, 10)) for _ in range(3)]
    nested_hue = [[int(rng.integers(1000)) for _ in range(300)]
                  for _ in range(3)]
    fig = hyp.plot(data, '.', hue=nested_hue, show=False)
    assert fig is not None


def test_nested_per_dataset_matrix_hue_flattens():
    # nested form also works when each dataset carries a per-observation matrix
    # hue: 3 datasets each (300, 4) -> flatten to a (900, 4) matrix hue
    rng = np.random.default_rng(1)
    data = [rng.standard_normal((300, 8)) for _ in range(3)]
    nested_hue = [rng.random((300, 4)) for _ in range(3)]
    fig = hyp.plot(data, '.', hue=nested_hue, show=False)
    assert fig is not None


def test_flat_and_matrix_hue_on_multidataset_unaffected():
    # a genuinely flat (n_obs,) hue and a (n_obs, k) matrix hue on multi-dataset
    # data must still be accepted as-is (not swept up by the nesting rule)
    rng = np.random.default_rng(2)
    data = [rng.standard_normal((300, 10)) for _ in range(3)]
    assert hyp.plot(data, '.', hue=rng.integers(0, 5, 900), show=False) is not None
    assert hyp.plot(data, '.', hue=rng.random((900, 4)), show=False) is not None


def test_wrong_length_nested_hue_still_errors():
    # a nested hue whose sub-lists DON'T match dataset lengths is not a valid
    # per-dataset hue and must still raise (no silent truncation)
    rng = np.random.default_rng(3)
    data = [rng.standard_normal((300, 10)) for _ in range(3)]
    bad = [[0] * 299 for _ in range(3)]  # 299 != 300 per dataset
    with pytest.raises(ValueError, match="observations"):
        hyp.plot(data, '.', hue=bad, show=False)


# --- GH #291: categorical/cluster LINES must not join separate trajectories

def _capture_draw(plot_call):
    """Run a hyp.plot(...) call and capture what actually reaches the drawing
    routine: the per-trace arrays (post hue/cluster regroup + interpolation),
    and the per-trace color/label kwargs. plot.py binds `_draw` locally, so
    patch it there."""
    import importlib
    plot_mod = importlib.import_module('hypertools.plot.plot')
    cap = {}
    orig = plot_mod._draw

    def spy(x, *a, **k):
        cap['x'] = [np.asarray(xi) for xi in x]
        kwl = k.get('kwargs_list')
        cap['colors'] = [d.get('color') for d in kwl] if kwl else None
        cap['labels'] = [d.get('label') for d in kwl] if kwl else None
        return orig(x, *a, **k)

    plot_mod._draw = spy
    try:
        cap['fig'] = plot_call()
    finally:
        plot_mod._draw = orig
    return cap


def _any_endpoint_touches_start(traces):
    """True if any trace ENDS exactly where a DIFFERENT trace STARTS -- the
    signature of a spurious bridge between two separate trajectories."""
    for i, ti in enumerate(traces):
        for j, tj in enumerate(traces):
            if i != j and np.allclose(ti[-1], tj[0]):
                return (i, j)
    return None


def _walks(n_datasets, seed=0, n=20, ndims=2, spread=60.0):
    """`n_datasets` well-separated random-walk trajectories."""
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.standard_normal((n, ndims)), axis=0)
            + k * spread for k in range(n_datasets)]


def test_gh291_two_datasets_different_categories():
    A, B = _walks(2)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', hue=['a'] * 20 + ['b'] * 20, show=False))
    assert len(cap['x']) == 2
    assert _any_endpoint_touches_start(cap['x']) is None


def test_gh291_two_datasets_same_category():
    # the case the first fix MISSED: one category over two datasets was
    # merged into a single connected line. Must stay two separate traces,
    # same colour, one legend entry.
    A, B = _walks(2)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', hue=['same'] * 40, legend=True, show=False))
    assert len(cap['x']) == 2
    assert _any_endpoint_touches_start(cap['x']) is None
    assert np.allclose(cap['colors'][0], cap['colors'][1])   # same colour
    # one legend entry for the shared category (the rest '_nolegend_')
    real = [l for l in cap['labels'] if l != '_nolegend_']
    assert real == ['same']


def test_gh291_two_datasets_reverse_numeric_categories():
    # reordering group flags by sorted category once produced a REVERSE
    # bridge from the second dataset into the first.
    A, B = _walks(2)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', hue=[2] * 20 + [1] * 20, show=False))
    assert len(cap['x']) == 2
    assert _any_endpoint_touches_start(cap['x']) is None


def test_gh291_category_repeated_across_datasets():
    A, B, C = _walks(3)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B, C], '-', hue=['a'] * 20 + ['b'] * 20 + ['a'] * 20, show=False))
    assert len(cap['x']) == 3
    assert _any_endpoint_touches_start(cap['x']) is None
    # the two 'a' datasets share a colour, the 'b' dataset differs
    ca, cb, ca2 = cap['colors']
    assert np.allclose(ca, ca2) and not np.allclose(ca, cb)


def test_gh291_ABA_within_one_trajectory_preserves_run_order():
    # A A B B A A in ONE dataset -> three runs in source order, drawn as a
    # CONTINUOUS line (bridged within the dataset), never collapsing the two
    # A runs into one polyline joining their non-adjacent points.
    rng = np.random.default_rng(2)
    T = np.cumsum(rng.standard_normal((30, 2)), axis=0)
    cap = _capture_draw(lambda: hyp.plot(
        T, '-', hue=['A'] * 10 + ['B'] * 10 + ['A'] * 10,
        legend=True, show=False))
    assert len(cap['x']) == 3                       # runs A, B, A
    # continuous within the trajectory: each run bridges to the next
    assert np.allclose(cap['x'][0][-1], cap['x'][1][0])
    assert np.allclose(cap['x'][1][-1], cap['x'][2][0])
    # both A runs share a colour; only one 'A' legend entry
    assert np.allclose(cap['colors'][0], cap['colors'][2])
    real = [l for l in cap['labels'] if l != '_nolegend_']
    assert real == ['A', 'B']


def test_gh291_three_datasets_no_cross_bridge():
    A, B, C = _walks(3)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B, C], '-', hue=['a'] * 20 + ['b'] * 20 + ['c'] * 20, show=False))
    assert len(cap['x']) == 3
    assert _any_endpoint_touches_start(cap['x']) is None


def test_gh291_single_point_runs_warn_and_dont_bridge_datasets():
    # alternating categories -> single-point runs; within one dataset they
    # bridge into a continuous line, and a pure line warns they'd otherwise
    # be invisible.
    rng = np.random.default_rng(3)
    T = np.cumsum(rng.standard_normal((5, 2)), axis=0)
    with pytest.warns(UserWarning, match="one observation"):
        cap = _capture_draw(lambda: hyp.plot(
            T, '-', hue=['a', 'b', 'a', 'b', 'a'], show=False))
    assert len(cap['x']) == 5


@pytest.mark.parametrize("fmt,expected_traces", [
    ('.', 1),      # marker-only: global grouping (one trace per category)
    ('-', 2),      # line-only: one run per dataset
    ('o-', 2),     # marker+line combo: segmented like a line
])
def test_gh291_marker_line_combo_formats(fmt, expected_traces):
    A, B = _walks(2)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], fmt, hue=['same'] * 40, show=False))
    assert len(cap['x']) == expected_traces
    if fmt != '.':
        assert _any_endpoint_touches_start(cap['x']) is None


@pytest.mark.parametrize("animate", ['spin', True, 'window'])
def test_gh291_animated_categorical_hue_no_cross_bridge(animate):
    A, B = _walks(2, ndims=3)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', hue=['a'] * 20 + ['b'] * 20,
        animate=animate, duration=1, show=False))
    assert len(cap['x']) == 2
    assert _any_endpoint_touches_start(cap['x']) is None


def test_gh291_plotly_categorical_hue_no_cross_bridge():
    A, B = _walks(2)
    fig = hyp.plot([A, B], '-', hue=['a'] * 20 + ['b'] * 20,
                   backend='plotly', show=False)
    # each dataset is its own line trace; none is joined to the other
    line_traces = [np.column_stack([t.x, t.y]) for t in fig.data
                   if getattr(t, 'x', None) is not None
                   and t.mode is not None and 'lines' in t.mode]
    assert len(line_traces) >= 2
    assert _any_endpoint_touches_start(line_traces) is None


def test_gh291_cluster_line_traces_do_not_bridge():
    # cluster-generated categorical groups must be segmented too: a cluster
    # spanning two datasets must not draw a line between them, and a cluster's
    # non-adjacent points within a dataset must not be joined.
    A, B = _walks(2)
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', n_clusters=2, show=False))
    assert _any_endpoint_touches_start(cap['x']) is None


def test_gh291_cluster_line_segments_stay_within_one_dataset():
    # two well-separated walks + 2 clusters: after segmentation every drawn
    # run belongs to a single input dataset, so no run's point span covers
    # BOTH clouds (which a cross-dataset bridge would produce). Uses reduce/
    # normalize=None so drawn coords stay in the input space for the check.
    A = np.cumsum(np.random.default_rng(4).standard_normal((20, 2)), axis=0)
    B = A + np.array([200.0, 0.0])          # cloud B far to the right
    cap = _capture_draw(lambda: hyp.plot(
        [A, B], '-', n_clusters=2, reduce=None, normalize=None, show=False))
    # a run bridging A<->B would span ~200 in x; a within-dataset run spans
    # only the walk's own extent (well under 100)
    for t in cap['x']:
        assert t[:, 0].max() - t[:, 0].min() < 100.0
