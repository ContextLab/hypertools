# -*- coding: utf-8 -*-
"""MultiIndex row-index DataFrames -> leaf traces + per-level averages
(GH #95).

A DataFrame with a row MultiIndex (``nlevels >= 2``) is expanded by
`hyp.plot`, before the format_data/analyze/reduce pipeline runs, into one
leaf dataset per unique full index combination; after that pipeline
transforms the leaves, one mean trajectory per non-leaf level grouping is
computed (in the TRANSFORMED space) and appended, with per-dataset color/
linewidth/alpha/linestyle/label overrides. See
`hypertools.plot.multiindex` for the exact formulas:

- ``linewidth = 1 + (levels aggregated over)`` -- leaves are 1; a level-k
  mean (aggregating over levels below it) gets progressively thicker up to
  the top level.
- ``alpha = min(1.0, 1/(level_idx + 1) + 0.2)`` where `level_idx` is 0 for
  the top level, increasing toward the leaf -- leaves are most transparent,
  the top-level mean is fully opaque (1.0).
- color is assigned purely by TOP-level index value (palette order of first
  appearance); linestyle(s), if a list, must have one entry per unique
  top-level value; legend shows only the top-level means.

These tests exercise the public `hyp.plot` API with real renders on both
backends (no mocks), plus direct unit tests of the pure `multiindex.py`
helpers for exact numeric ground truth.
"""

import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.multiindex import build_multiindex_styles, expand_multiindex


# ---------------------------------------------------------------------------
# synthetic data builders
# ---------------------------------------------------------------------------

def _make_2level_df(n_time=10, seed=0, lengths=None):
    """2 conditions x 4 subjects, smooth 3D trajectories (noisy random
    walks), offset per condition so the two groups are visually/numerically
    distinguishable. `lengths`, if given, is a dict subj -> n_time override
    (used to test unequal-length groups)."""
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for ci, cond in enumerate(['condA', 'condB']):
        for si in range(4):
            subj = f'S{si}'
            t = (lengths or {}).get(subj, n_time)
            base = rng.standard_normal((t, 3)).cumsum(axis=0) + ci * 5.0
            rows.append(base)
            tuples.extend([(cond, subj)] * t)
    data = np.vstack(rows)
    index = pd.MultiIndex.from_tuples(tuples, names=['cond', 'subj'])
    return pd.DataFrame(data, index=index, columns=['x', 'y', 'z'])


def _make_3level_df(n_time=8, seed=0):
    """2 groups x 2 conditions x 3 subjects."""
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for gi, grp in enumerate(['grpX', 'grpY']):
        for ci, cond in enumerate(['condA', 'condB']):
            for si in range(3):
                subj = f'S{si}'
                base = (rng.standard_normal((n_time, 3)).cumsum(axis=0)
                        + gi * 8.0 + ci * 2.0)
                rows.append(base)
                tuples.extend([(grp, cond, subj)] * n_time)
    data = np.vstack(rows)
    index = pd.MultiIndex.from_tuples(tuples, names=['grp', 'cond', 'subj'])
    return pd.DataFrame(data, index=index, columns=['x', 'y', 'z'])


def _leaf_lines_3d(ax, n):
    return [np.array(l.get_data_3d()).T for l in ax.get_lines()[:n]]


# ---------------------------------------------------------------------------
# unit tests: expand_multiindex / build_multiindex_styles (exact ground truth)
# ---------------------------------------------------------------------------

def test_expand_multiindex_leaf_order_and_counts():
    df = _make_2level_df()
    leaf_dfs, meta = expand_multiindex(df)
    assert len(leaf_dfs) == 8
    assert meta['n_levels'] == 2
    assert meta['leaf_keys'] == [
        ('condA', 'S0'), ('condA', 'S1'), ('condA', 'S2'), ('condA', 'S3'),
        ('condB', 'S0'), ('condB', 'S1'), ('condB', 'S2'), ('condB', 'S3'),
    ]
    for leaf_df, key in zip(leaf_dfs, meta['leaf_keys']):
        assert (leaf_df.index.get_level_values(0) == key[0]).all()
        assert (leaf_df.index.get_level_values(1) == key[1]).all()


def test_expand_multiindex_requires_2plus_levels():
    df = pd.DataFrame(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        expand_multiindex(df)


def test_build_styles_2level_exact():
    df = _make_2level_df()
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    arrays, style = build_multiindex_styles(leaf_arrays, meta)

    assert len(arrays) == 10  # 8 leaves + 2 cond means
    assert style['linewidths'] == [1.0] * 8 + [2.0] * 2
    assert style['alphas'] == pytest.approx([0.7] * 8 + [1.0] * 2)
    assert style['labels'] == ['_nolegend_'] * 8 + ['condA', 'condB']
    assert style['unique_top'] == ['condA', 'condB']

    # means EXACTLY equal np.mean of member leaves
    mean_a = np.mean(np.stack(leaf_arrays[0:4]), axis=0)
    mean_b = np.mean(np.stack(leaf_arrays[4:8]), axis=0)
    assert np.array_equal(arrays[8], mean_a)
    assert np.array_equal(arrays[9], mean_b)

    # colors: all condA traces (leaves 0-3 + mean 8) share one color,
    # distinct from condB (leaves 4-7 + mean 9)
    colors = style['colors']
    assert len(set(colors[0:4] + [colors[8]])) == 1
    assert len(set(colors[4:8] + [colors[9]])) == 1
    assert colors[0] != colors[4]


def test_build_styles_3level_exact():
    df = _make_3level_df()
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    arrays, style = build_multiindex_styles(leaf_arrays, meta)

    n_leaves = 2 * 2 * 3
    n_grp_cond_means = 2 * 2
    n_grp_means = 2
    assert len(arrays) == n_leaves + n_grp_cond_means + n_grp_means

    assert style['linewidths'][:n_leaves] == [1.0] * n_leaves
    gc_lws = style['linewidths'][n_leaves:n_leaves + n_grp_cond_means]
    assert gc_lws == [2.0] * n_grp_cond_means
    grp_lws = style['linewidths'][-n_grp_means:]
    assert grp_lws == [3.0] * n_grp_means

    assert style['alphas'][:n_leaves] == pytest.approx(
        [1 / 3 + 0.2] * n_leaves)
    assert style['alphas'][n_leaves:n_leaves + n_grp_cond_means] == \
        pytest.approx([0.7] * n_grp_cond_means)
    assert style['alphas'][-n_grp_means:] == pytest.approx(
        [1.0] * n_grp_means)

    # only the (deepest) grp-level means carry real legend labels
    assert style['labels'][:n_leaves + n_grp_cond_means] == \
        ['_nolegend_'] * (n_leaves + n_grp_cond_means)
    assert style['labels'][-n_grp_means:] == ['grpX', 'grpY']

    # (grp, cond)-mean exactly equals np.mean of its 3 member leaves
    leaf_keys = meta['leaf_keys']
    gc_key = ('grpX', 'condA')
    member_idx = [i for i, k in enumerate(leaf_keys) if k[:2] == gc_key]
    expected = np.mean(np.stack([leaf_arrays[i] for i in member_idx]), axis=0)
    gc_arr = arrays[n_leaves:n_leaves + n_grp_cond_means][0]
    assert np.array_equal(gc_arr, expected)

    # grp-mean exactly equals np.mean of ALL 6 leaves under that group
    grp_member_idx = [i for i, k in enumerate(leaf_keys) if k[0] == 'grpX']
    expected_grp = np.mean(
        np.stack([leaf_arrays[i] for i in grp_member_idx]), axis=0)
    assert np.array_equal(arrays[-2], expected_grp)


def test_build_styles_mismatched_leaf_count_raises():
    df = _make_2level_df()
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs][:-1]  # drop one
    with pytest.raises(ValueError):
        build_multiindex_styles(leaf_arrays, meta)


def test_build_styles_unequal_lengths_warns_and_truncates():
    df = _make_2level_df(lengths={'S0': 10, 'S1': 10, 'S2': 10, 'S3': 6})
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    with pytest.warns(UserWarning, match="unequal"):
        arrays, style = build_multiindex_styles(leaf_arrays, meta)
    cond_a_mean = arrays[8]
    assert cond_a_mean.shape[0] == 6  # truncated to shortest member
    expected = np.mean(
        np.stack([a[:6] for a in leaf_arrays[0:4]]), axis=0)
    assert np.array_equal(cond_a_mean, expected)


def test_build_styles_linestyle_list_wrong_length_raises():
    df = _make_2level_df()
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    with pytest.raises(ValueError, match="linestyle"):
        build_multiindex_styles(leaf_arrays, meta,
                                linestyle=['-', '--', '-.'])


def test_build_styles_linestyle_list_correct_length():
    df = _make_2level_df()
    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    _, style = build_multiindex_styles(leaf_arrays, meta,
                                       linestyles=['-', '--'])
    assert style['linestyles'] == ['-'] * 4 + ['--'] * 4 + ['-', '--']


# ---------------------------------------------------------------------------
# integration: matplotlib backend, full hyp.plot() pipeline
# ---------------------------------------------------------------------------

def test_plot_2level_mpl_counts_styles_legend():
    df = _make_2level_df()
    fig = hyp.plot(df, fmt='.', show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 10

    lws = [round(l.get_linewidth(), 6) for l in lines]
    assert lws == [1.0] * 8 + [2.0] * 2
    alphas = [round(l.get_alpha(), 6) for l in lines]
    assert alphas == pytest.approx([0.7] * 8 + [1.0] * 2)

    colors = [l.get_color() for l in lines]
    assert len(set(colors[0:4] + [colors[8]])) == 1
    assert len(set(colors[4:8] + [colors[9]])) == 1
    assert colors[0] != colors[4]

    labels = [l.get_label() for l in lines]
    assert labels == ['_nolegend_'] * 8 + ['condA', 'condB']

    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ['condA', 'condB']
    plt.close(fig)


def test_plot_2level_mpl_means_exactly_equal_np_mean_post_transform():
    """Core numeric contract, exercised through the REAL hyp.plot pipeline
    (post normalize/reduce/align/center/scale): the drawn mean line for each
    condition equals np.mean of that condition's drawn leaf lines. `fmt='.'`
    disables line interpolation (a nonlinear pchip resample that would not
    commute exactly with averaging) so only the pipeline's LINEAR
    center/scale steps -- which commute exactly with averaging since the
    same affine map is applied to every trace -- remain."""
    df = _make_2level_df()
    fig = hyp.plot(df, fmt='.', show=False)
    ax = fig.axes[0]
    leaves = _leaf_lines_3d(ax, 8)
    means = _leaf_lines_3d(ax, 10)[8:]

    mean_a = np.mean(np.stack(leaves[0:4]), axis=0)
    mean_b = np.mean(np.stack(leaves[4:8]), axis=0)
    assert np.allclose(means[0], mean_a, atol=1e-9)
    assert np.allclose(means[1], mean_b, atol=1e-9)
    plt.close(fig)


def test_plot_3level_mpl_counts_and_linewidths():
    df = _make_3level_df()
    fig = hyp.plot(df, fmt='.', show=False)
    lines = fig.axes[0].get_lines()
    assert len(lines) == 12 + 4 + 2  # leaves + (grp,cond)-means + grp-means
    lws = [round(l.get_linewidth(), 6) for l in lines]
    assert lws == [1.0] * 12 + [2.0] * 4 + [3.0] * 2
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ['grpX', 'grpY']
    plt.close(fig)


def test_plot_linestyle_list_cycles_per_top_group():
    df = _make_2level_df()
    fig = hyp.plot(df, fmt='-', linestyle=['-', '--'], show=False)
    linestyles = [l.get_linestyle() for l in fig.axes[0].get_lines()]
    assert linestyles == ['-'] * 4 + ['--'] * 4 + ['-', '--']
    plt.close(fig)


def test_plot_linestyle_list_wrong_length_raises():
    df = _make_2level_df()
    with pytest.raises(ValueError, match="linestyle"):
        hyp.plot(df, linestyle=['-', '--', '-.', ':'], show=False)


def test_plot_unequal_group_lengths_warns():
    df = _make_2level_df(lengths={'S0': 10, 'S1': 10, 'S2': 10, 'S3': 6})
    with pytest.warns(UserWarning, match="unequal"):
        fig = hyp.plot(df, fmt='.', show=False)
    plt.close(fig)


# ---------------------------------------------------------------------------
# interaction with hue= / cluster=
# ---------------------------------------------------------------------------

def test_hue_plus_multiindex_warns_and_ignores_hue():
    df = _make_2level_df()
    fake_hue = list(range(len(df)))
    with pytest.warns(UserWarning, match="MultiIndex"):
        fig = hyp.plot(df, hue=fake_hue, fmt='.', show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    # still grouped by top-level index (10 traces), not exploded by hue
    assert len(lines) == 10
    colors = [l.get_color() for l in lines]
    assert len(set(colors[0:4] + [colors[8]])) == 1
    plt.close(fig)


def test_cluster_plus_multiindex_raises():
    df = _make_2level_df()
    with pytest.raises(ValueError, match="MultiIndex"):
        hyp.plot(df, cluster='KMeans', show=False)


def test_n_clusters_plus_multiindex_raises():
    df = _make_2level_df()
    with pytest.raises(ValueError, match="MultiIndex"):
        hyp.plot(df, n_clusters=2, show=False)


# ---------------------------------------------------------------------------
# plotly parity
# ---------------------------------------------------------------------------

def test_plot_2level_plotly_parity():
    df = _make_2level_df()
    fig = hyp.plot(df, fmt='.', show=False, backend='plotly')
    data_traces = [t for t in fig.data if t.type in ('scatter3d', 'scatter')]
    # 10 data traces + 1 cube wireframe trace (3D)
    assert len(data_traces) == 11
    traces = data_traces[:10]

    widths = [t.line.width for t in traces]
    leaf_w = widths[0]
    mean_w = widths[8]
    assert all(w == pytest.approx(leaf_w) for w in widths[:8])
    assert all(w == pytest.approx(mean_w) for w in widths[8:10])
    assert mean_w > leaf_w

    # alpha is baked into the rgba() string (`_to_plotly_color`) -- compare
    # the RGB portion only (alpha itself is asserted separately below).
    def _rgb_only(rgba):
        return rgba.rsplit(',', 1)[0]

    colors = [_rgb_only(t.line.color) for t in traces]
    assert len(set(colors[0:4] + [colors[8]])) == 1
    assert len(set(colors[4:8] + [colors[9]])) == 1
    assert colors[0] != colors[4]

    def _alpha(rgba):
        return float(rgba.rstrip(')').rsplit(',', 1)[1])

    alphas = [_alpha(t.line.color) for t in traces]
    assert alphas[:8] == pytest.approx([0.7] * 8)
    assert alphas[8:10] == pytest.approx([1.0] * 2)

    names = [t.name for t in traces]
    assert names[8] == 'condA'
    assert names[9] == 'condB'
    showlegends = [t.showlegend for t in traces]
    assert showlegends[:8] == [False] * 8
    assert showlegends[8:10] == [True, True]


# ---------------------------------------------------------------------------
# animated smoke test (matplotlib): means animate too, no crash
# ---------------------------------------------------------------------------

def test_plot_2level_animated_mpl_smoke():
    df = _make_2level_df(n_time=12)
    bundle = hyp.plot(df, animate=True, duration=1, tail_duration=0.3,
                      frame_rate=3, show=False, return_model=True)
    line_ani = bundle['animation']
    assert line_ani is not None
    data_lines = line_ani._args[0]
    assert len(data_lines) == 10  # 8 leaves + 2 means, all animated
    plt.close('all')


@pytest.mark.parametrize('ndims', [3, 2])
@pytest.mark.parametrize('trail', ['bullettime', 'chemtrails'])
def test_multiindex_trail_alpha_no_collision(ndims, trail, tmp_path):
    """MultiIndex expansion assigns a per-trace ``alpha`` (faint leaf traces
    vs. opaque group-mean traces). The animated trail artists
    (chemtrails/precog/bullettime) used to pass a hardcoded ``alpha=0.3``
    alongside ``**kwargs_list[idx]``, so that per-trace alpha collided ->
    ``TypeError: ... got multiple values for keyword argument 'alpha'``. The
    0.3 fade is now folded into any pre-existing alpha, so building AND
    rendering the animation must not raise (both ndims=3 and ndims=2)."""
    df = _make_2level_df(n_time=12)
    fig, ani = hyp.plot(df, animate=True, duration=1, tail_duration=0.3,
                        frame_rate=3, ndims=ndims, legend=True, show=False,
                        **{trail: True})
    assert ani is not None
    # the collision used to raise while *building* the trail artists inside
    # hyp.plot(); saving forces a real render of the faded trail frames too.
    out = tmp_path / f'mi_{trail}_{ndims}d.gif'
    ani.save(str(out))
    assert out.exists() and out.stat().st_size > 0
    plt.close('all')


# ---------------------------------------------------------------------------
# regression: single-level DataFrame / plain arrays unchanged
# ---------------------------------------------------------------------------

def test_single_level_df_regression():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((20, 3)), columns=['x', 'y', 'z'])
    fig = hyp.plot(df, show=False)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    assert ax.get_legend() is None
    plt.close(fig)


def test_plain_array_list_regression():
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((20, 3)), rng.standard_normal((20, 3))]
    fig = hyp.plot(data, show=False)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    plt.close(fig)


def test_single_level_range_index_df_not_treated_as_multiindex():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((15, 3)), columns=['x', 'y', 'z'])
    assert df.index.nlevels == 1
    fig = hyp.plot(df, hue=['a'] * 7 + ['b'] * 8, show=False)
    ax = fig.axes[0]
    # normal hue grouping still works (2 groups), unaffected by MultiIndex path
    assert len(ax.get_lines()) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# follow-up fix: list-input bypass warning, predict= + MultiIndex ValueError,
# deduped unequal-length warning (review of 85f263de, GH #95)
# ---------------------------------------------------------------------------

def test_list_with_multiindex_df_warns_and_flattens():
    """A LIST containing a MultiIndex DataFrame (whether alone or mixed with
    plain arrays) does NOT get the MultiIndex expansion -- only a BARE single
    DataFrame does. This must raise a UserWarning naming the offending
    element, and the MultiIndex must be treated as a flat index (i.e. one
    plain line per list element, not one line per leaf + per-level means)."""
    df = _make_2level_df()  # 8 unique (cond, subj) leaves
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 3))

    with pytest.warns(UserWarning, match="MultiIndex grouping is only applied"):
        fig = hyp.plot([df, arr], show=False)
    ax = fig.axes[0]
    # flat treatment: exactly one line per list element (2), NOT 8 leaves + means
    assert len(ax.get_lines()) == 2
    plt.close(fig)

    # also fires for a list containing ONLY a MultiIndex df (still a list, not
    # a bare DataFrame)
    with pytest.warns(UserWarning, match="MultiIndex grouping is only applied"):
        fig2 = hyp.plot([df], show=False)
    ax2 = fig2.axes[0]
    assert len(ax2.get_lines()) == 1
    plt.close(fig2)


def test_predict_plus_multiindex_raises():
    df = _make_2level_df()
    with pytest.raises(ValueError, match="predict="):
        hyp.plot(df, predict='Kalman', show=False)


def test_multiindex_colorbar_shows_only_top_level_segments():
    """SEVERE regression (GH #100/#95): colorbar for a MultiIndex-expanded
    plot must show exactly ONE SEGMENT PER TOP-LEVEL GROUP (2, for a 2-level
    df with 8 leaves + 2 top-level means -- NOT 10, one per drawn trace),
    labeled with the top-level values, and must NEVER show '_nolegend_'
    (the label every leaf/intermediate-level mean carries)."""
    df = _make_2level_df()
    fig = hyp.plot(df, colorbar=True, show=False)
    plt.close(fig)

    assert len(fig.axes) == 2, "expected the plot axes + one colorbar axes"
    _, cbar_ax = fig.axes
    labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert labels == ['condA', 'condB']
    assert '_nolegend_' not in labels

    from matplotlib.collections import QuadMesh
    mesh = next(c for c in cbar_ax.collections if isinstance(c, QuadMesh))
    # BoundaryNorm over exactly 2 groups -> 3 boundaries
    assert np.allclose(mesh.norm.boundaries, [-0.5, 0.5, 1.5])


def test_multiindex_colorbar_3level_shows_only_top_level_segments():
    df = _make_3level_df()
    fig = hyp.plot(df, colorbar=True, show=False)
    plt.close(fig)

    _, cbar_ax = fig.axes
    labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert labels == ['grpX', 'grpY']
    assert '_nolegend_' not in labels


def test_multiindex_colorbar_top_level_order_matches_legend_order():
    """GH #100 follow-up: the MultiIndex-driven discrete colorbar (one
    segment per TOP-level group) must inherit the top-to-bottom
    legend-order fix, same as any other discrete colorbar -- the top-level
    group order must read the same top-to-bottom as the legend, not
    matplotlib's default bottom-up order."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    df = _make_2level_df()
    fig = hyp.plot(df, colorbar=True, legend=True, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    ax, cbar_ax = fig.axes
    _, legend_labels = ax.get_legend_handles_labels()
    assert list(legend_labels) == ['condA', 'condB']

    ticklabels = cbar_ax.get_yticklabels()
    entries = sorted(
        ((t.get_window_extent(renderer).y0, t.get_text()) for t in ticklabels),
        key=lambda e: -e[0])  # descending y = top first
    cbar_order = [label for _, label in entries]
    assert cbar_order == list(legend_labels), (
        "MultiIndex colorbar's top-level order must match the legend order")
    plt.close(fig)


def test_build_styles_3level_unequal_lengths_warns_exactly_once():
    """A 3-level tree where one leaf is short: the leaf is a member of BOTH
    its (grp, cond) prefix group AND its grp prefix group, so a naive
    per-group warning would fire twice for one underlying issue. Assert
    exactly one matching UserWarning record is emitted."""
    rng = np.random.default_rng(0)
    tuples, rows = [], []
    for gi, grp in enumerate(['grpX', 'grpY']):
        for ci, cond in enumerate(['condA', 'condB']):
            for si in range(3):
                subj = f'S{si}'
                t = 5 if (grp == 'grpX' and cond == 'condA' and subj == 'S0') else 8
                base = (rng.standard_normal((t, 3)).cumsum(axis=0)
                        + gi * 8.0 + ci * 2.0)
                rows.append(base)
                tuples.extend([(grp, cond, subj)] * t)
    data = np.vstack(rows)
    index = pd.MultiIndex.from_tuples(tuples, names=['grp', 'cond', 'subj'])
    df = pd.DataFrame(data, index=index, columns=['x', 'y', 'z'])

    leaf_dfs, meta = expand_multiindex(df)
    leaf_arrays = [d.values for d in leaf_dfs]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arrays, style = build_multiindex_styles(leaf_arrays, meta)
    unequal_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning) and "unequal" in str(w.message)
    ]
    assert len(unequal_warnings) == 1
