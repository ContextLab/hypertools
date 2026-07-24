"""Tests for `colorbar=` (GH #100): continuous-hue and discrete-group
colorbars on both the matplotlib and plotly backends.

All assertions are numeric, against REAL rendered artists/traces -- no
mocks. The matplotlib checks locate the actual `QuadMesh` a `Colorbar`
draws (matplotlib also adds a divider `LineCollection` with an unrelated
default colormap to the same axes, so `axes.collections[0]` is NOT
reliably the colorbar's mesh -- see `_quadmesh`) and compare its cmap/norm
directly against the SAME palette-resolution helpers hypertools uses to
color the plotted lines, so a passing test proves the colorbar reflects
the ACTUAL rendered colors, not just "a colorbar was drawn somewhere".
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pytest

import hypertools as hyp
from hypertools.plot.colors import continuous_colormap, get_palette_colors


def _quadmesh(ax):
    """The colorbar's actual color mesh (as opposed to matplotlib's
    divider-line `LineCollection`, which also lives in `ax.collections`
    but carries an unrelated default colormap)."""
    for c in ax.collections:
        if isinstance(c, QuadMesh):
            return c
    raise AssertionError(f"no QuadMesh found on {ax!r}; collections were "
                         f"{[type(c).__name__ for c in ax.collections]}")


def _line_collection(ax):
    """The multicolor Line3DCollection/LineCollection hypertools draws for
    continuously-hued lines (added AFTER the cube's wireframe collections,
    so it is always the LAST collection on the axes)."""
    return ax.collections[-1]


# --------------------------------------------------------------- fixtures
def _continuous_hue_data(n=50):
    # repeated boundary values -> the first/last two POINTS share an exact
    # hue value, so the first/last drawn LINE SEGMENT (an average of its
    # two endpoint colors) is EXACTLY equal to the endpoint color, letting
    # the test assert exact (not approximate) equality against the colorbar.
    hue = np.concatenate([np.zeros(5), np.linspace(0.0, 1.0, n - 10),
                          np.ones(5)])
    x = np.column_stack([np.linspace(0, 1, n)] * 3)
    return x, hue


def _discrete_hue_data(n_per_group=20):
    hue = ['a'] * n_per_group + ['b'] * n_per_group + ['c'] * n_per_group
    x = np.column_stack([np.linspace(0, 1, 3 * n_per_group)] * 3)
    return x, hue


# ------------------------------------------------------- continuous (mpl)
def test_continuous_colorbar_mpl_matches_line_colors():
    x, hue = _continuous_hue_data()
    fig = hyp.plot(x, hue=hue, fmt='-', colorbar=True, show=False)
    plt.close(fig)

    assert len(fig.axes) == 2, "expected the plot axes + one colorbar axes"
    ax, cbar_ax = fig.axes
    mesh = _quadmesh(cbar_ax)

    expected_cmap = continuous_colormap('hls')  # 'hls' is plot()'s default palette
    assert np.allclose(mesh.cmap.colors, expected_cmap.colors)
    assert mesh.norm.vmin == pytest.approx(0.0)
    assert mesh.norm.vmax == pytest.approx(1.0)

    # the colorbar's own endpoint colors (sampled at vmin/vmax through its
    # own cmap+norm) must equal the ACTUAL first/last drawn line-segment
    # colors -- i.e. the colorbar really does reflect what's on the plot.
    coll = _line_collection(ax)
    edge_colors = coll.get_edgecolor()
    cbar_first = mesh.cmap(mesh.norm(mesh.norm.vmin))
    cbar_last = mesh.cmap(mesh.norm(mesh.norm.vmax))
    assert np.allclose(edge_colors[0], cbar_first)
    assert np.allclose(edge_colors[-1], cbar_last)


def test_continuous_colorbar_label_and_custom_ticks_mpl():
    x, hue = _continuous_hue_data()
    fig = hyp.plot(x, hue=hue, fmt='-',
                   colorbar={'label': 'my value', 'ticks': [0.25, 0.75]},
                   show=False)
    plt.close(fig)
    _, cbar_ax = fig.axes
    assert cbar_ax.get_ylabel() == 'my value'
    tick_locs = [t for t in cbar_ax.get_yticks()]
    assert np.allclose(sorted(tick_locs), [0.25, 0.75])


# --------------------------------------------------------- discrete (mpl)
def test_discrete_colorbar_mpl_matches_group_colors_and_legend_labels():
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, colorbar=True, legend=True, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    ax, cbar_ax = fig.axes
    mesh = _quadmesh(cbar_ax)

    expected_colors = get_palette_colors('hls', 3)
    assert np.allclose(mesh.cmap.colors, expected_colors)
    # BoundaryNorm over 3 groups -> 4 boundaries at -0.5, 0.5, 1.5, 2.5
    assert np.allclose(mesh.norm.boundaries, [-0.5, 0.5, 1.5, 2.5])

    _, legend_labels = ax.get_legend_handles_labels()
    cbar_labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert cbar_labels == list(legend_labels)
    plt.close(fig)


def test_discrete_colorbar_default_labels_are_category_names_mpl():
    # release-1.0 audit F02-007: a categorical hue's category names are
    # KNOWN whether or not legend=True was also passed, so the colorbar
    # shows them directly (previously it fell back to 1..n unless a
    # redundant legend was requested). The 1..n fallback is still
    # exercised by test_discrete_colorbar_list_of_datasets_no_hue_mpl
    # (multiple datasets, no hue -> no names available).
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, colorbar=True, show=False)  # no legend=
    plt.close(fig)
    _, cbar_ax = fig.axes
    labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert labels == ['a', 'b', 'c']


def test_discrete_colorbar_list_of_datasets_no_hue_mpl():
    # brief: "discrete groups (... list-of-datasets ...)" -- multiple
    # datasets with no hue/cluster still get a meaningful discrete colorbar
    # (one swatch per dataset, colored from the same ambient palette).
    rng = np.random.default_rng(0)
    datasets = [rng.standard_normal((10, 3)) for _ in range(4)]
    fig = hyp.plot(datasets, colorbar=True, show=False)
    plt.close(fig)
    assert len(fig.axes) == 2
    _, cbar_ax = fig.axes
    mesh = _quadmesh(cbar_ax)
    assert np.allclose(mesh.cmap.colors, get_palette_colors('hls', 4))
    labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert labels == ['1', '2', '3', '4']


# ------------------------------------------------------------- error case
def test_colorbar_requires_color_mapping_raises():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match='color mapping'):
        hyp.plot(x, colorbar=True, show=False)


def test_colorbar_invalid_value_raises():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError):
        hyp.plot(x, colorbar='yes-please', show=False)


def test_colorbar_invalid_dict_key_raises():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError):
        hyp.plot(x, colorbar={'labl': 'typo'}, show=False)


def test_colorbar_invalid_location_raises():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError):
        hyp.plot(x, colorbar={'location': 'diagonal'}, show=False)


# ------------------------------------------------ legend + colorbar (mpl)
def test_colorbar_and_legend_coexist_no_overlap_mpl():
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, colorbar=True, legend=True, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    ax, cbar_ax = fig.axes
    legend = ax.get_legend()
    assert legend is not None
    legend_bbox = legend.get_window_extent(renderer)
    cbar_bbox = cbar_ax.get_window_extent(renderer)

    # colorbar sits strictly to the right of the legend, and both are
    # fully inside the (rasterized) canvas -- nothing clipped or stacked.
    assert cbar_bbox.x0 >= legend_bbox.x1 - 1.0, (
        f"colorbar (x0={cbar_bbox.x0}) overlaps legend (x1={legend_bbox.x1})")
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    assert cbar_bbox.x1 <= fig_w_px + 1.0, "colorbar clipped off right edge"
    assert legend_bbox.x0 >= 0, "legend clipped off the canvas"
    plt.close(fig)


# ------------------------------------------------------------ animated mpl
def test_colorbar_present_and_static_across_animation_frames():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    hue = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
    fig, ani = hyp.plot(walk, hue=hue, animate=True, duration=1,
                        frame_rate=5, colorbar=True, show=False)

    assert len(fig.axes) == 2
    cbar_ax = fig.axes[-1]
    mesh_before = _quadmesh(cbar_ax)
    colors_before = mesh_before.cmap.colors.copy()

    # advance the REAL FuncAnimation a few frames (no mocks: this is
    # matplotlib's own frame-update machinery)
    for frame in (0, 2, 4):
        ani._draw_frame(frame)

    assert len(fig.axes) == 2, "animation must not add/remove axes per frame"
    assert fig.axes[-1] is cbar_ax, "colorbar axes identity must be stable"
    mesh_after = _quadmesh(cbar_ax)
    assert mesh_after is mesh_before, "colorbar mesh must not be recreated"
    assert np.array_equal(mesh_after.cmap.colors, colors_before)
    plt.close(fig)


# --------------------------------------------------------------- plotly
def test_continuous_colorbar_plotly_matches_palette():
    x, hue = _continuous_hue_data()
    fig = hyp.plot(x, hue=hue, fmt='-', colorbar=True, backend='plotly',
                   show=False)
    cb_trace = fig.data[-1]
    assert cb_trace.marker.showscale is True

    # continuous mapping: cyclic palettes are trimmed so the colorbar's
    # two ends stay distinguishable (release-1.0 audit, F01-013) -- the
    # ground truth is the SAME continuous colormap mat2colors uses
    expected = continuous_colormap('hls').colors
    first_stop, first_color = cb_trace.marker.colorscale[0]
    last_stop, last_color = cb_trace.marker.colorscale[-1]
    assert first_stop == pytest.approx(0.0)
    assert last_stop == pytest.approx(1.0)

    def _parse_rgb(s):
        return tuple(int(v) for v in s[len('rgb('):-1].split(','))

    assert _parse_rgb(first_color) == tuple(
        int(round(255 * c)) for c in expected[0])
    assert _parse_rgb(last_color) == tuple(
        int(round(255 * c)) for c in expected[-1])
    assert cb_trace.marker.cmin == pytest.approx(0.0)
    assert cb_trace.marker.cmax == pytest.approx(1.0)


def test_discrete_colorbar_plotly_matches_groups():
    # NOTE (GH #100 follow-up): a VERTICAL discrete colorbar must read
    # top-to-bottom in the SAME order as the legend, i.e. group 0 (the
    # FIRST group) must sit at the TOP -- plotly's vertical orientation
    # places cmax (the largest value) at the top and cmin at the bottom, so
    # group 0's tick now lives at the LARGEST value (n - 1) instead of 0,
    # and the hard-edged colorscale segments are built from `colors[::-1]`
    # so the segment nearest cmax (top) holds group 0's color. This was
    # previously pinned to the old bottom-up order ([0, 1, 2] / segments in
    # forward order); updated to assert the new top-down order instead.
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, colorbar=True, backend='plotly', show=False)
    cb_trace = fig.data[-1]

    assert list(cb_trace.marker.colorbar.tickvals) == [2, 1, 0]
    # release-1.0 audit F02-007: category names are shown without needing
    # legend=True (previously '1'/'2'/'3' unless a legend was also drawn)
    assert list(cb_trace.marker.colorbar.ticktext) == ['a', 'b', 'c']
    assert cb_trace.marker.cmin == pytest.approx(-0.5)
    assert cb_trace.marker.cmax == pytest.approx(2.5)

    expected = get_palette_colors('hls', 3)

    def _parse_rgb(s):
        return tuple(int(v) for v in s[len('rgb('):-1].split(','))

    # first stop of each of the 3 hard-edged segments -- segment 0 (nearest
    # cmin, i.e. the BOTTOM of the vertical colorbar) now holds the LAST
    # group's color, and segment 2 (nearest cmax, the TOP) holds the FIRST
    # group's color, since group 0 must render at the top to match the
    # legend.
    scale = cb_trace.marker.colorscale
    seg0_color = _parse_rgb(scale[0][1])
    seg1_color = _parse_rgb(scale[2][1])
    seg2_color = _parse_rgb(scale[4][1])
    for seg_color, exp in zip((seg0_color, seg1_color, seg2_color),
                              reversed(expected)):
        assert seg_color == tuple(int(round(255 * c)) for c in exp)


def test_colorbar_and_legend_coexist_plotly_distinct_positions():
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, colorbar=True, legend=True, backend='plotly',
                   show=False)
    legend_x = fig.layout.legend.x
    cbar_x = fig.data[-1].marker.colorbar.x
    assert cbar_x - legend_x >= 0.1, "colorbar must sit clearly right of legend"
    assert fig.layout.margin.r >= 200, "right margin must widen for both"


def test_colorbar_requires_color_mapping_raises_plotly_too():
    x = np.random.default_rng(0).standard_normal((20, 3))
    with pytest.raises(ValueError, match='color mapping'):
        hyp.plot(x, colorbar=True, backend='plotly', show=False)


# ------------------------------------------------------------------------
# regression: legend fitting must run in EVERY layout (GH #100 follow-up)
# --------------------------------------------------------------------------
# `_fit_right_legend` previously only ran for STATIC plots, and only BEFORE
# a colorbar was added -- so a 'left'/'top' colorbar (which reshapes `ax`
# via matplotlib's own `make_axes`) or `animate=True` (which skipped the fit
# entirely) left the right-side legend fully clipped off the canvas. These
# assertions rasterize the ACTUAL figure and check the rightmost/leftmost
# inked pixel sits strictly inside the canvas -- not just "a legend exists
# somewhere" -- so a passing test proves nothing is cut off in the real
# rendered output.

def _rightmost_leftmost_inked(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[..., :3]
    inked_cols = np.where((buf < 245).any(axis=(0, 2)))[0]
    assert len(inked_cols), "nothing rendered at all"
    return int(inked_cols.max()), int(inked_cols.min()), buf.shape[1]


def test_legend_fully_inside_canvas_colorbar_location_left():
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, legend=True,
                   colorbar={'location': 'left'}, show=False)
    rightmost, leftmost, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, "legend clipped off the right edge"
    assert leftmost > 0, "colorbar clipped off the left edge"
    plt.close(fig)


def test_legend_fully_inside_canvas_colorbar_location_top():
    x, hue = _discrete_hue_data()
    fig = hyp.plot(x, hue=hue, legend=True,
                   colorbar={'location': 'top'}, show=False)
    rightmost, leftmost, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, "legend clipped off the right edge"
    plt.close(fig)


def test_legend_fully_inside_canvas_animate():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    hue = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
    fig, ani = hyp.plot(walk, hue=hue, animate=True, duration=1,
                        frame_rate=5, legend=True, show=False)
    ani._draw_frame(0)
    rightmost, _, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, "legend clipped off the right edge (animate=True)"
    plt.close(fig)


def test_legend_and_colorbar_fully_inside_canvas_animate():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((30, 3)), axis=0)
    hue = ['a'] * 10 + ['b'] * 10 + ['c'] * 10
    fig, ani = hyp.plot(walk, hue=hue, animate=True, duration=1,
                        frame_rate=5, legend=True, colorbar=True, show=False)
    ani._draw_frame(0)
    rightmost, _, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, (
        "legend/colorbar clipped off the right edge (animate=True)")
    plt.close(fig)


def test_long_labels_legend_and_colorbar_fully_inside_canvas():
    """The width fit must measure the ACTUAL final label strings, not a
    fixed per-iteration guess -- a naive small-step fit converges far too
    slowly for long labels and leaves them clipped (GH #100 follow-up)."""
    rng = np.random.default_rng(0)
    g1 = rng.standard_normal((50, 3))
    g2 = rng.standard_normal((50, 3)) + np.array([5, 0, 0])
    g3 = rng.standard_normal((50, 3)) + np.array([0, 5, 0])
    long_labels = ['very long group label A', 'longer label B',
                   'the longest label C']
    fig = hyp.plot([g1, g2, g3], legend=long_labels, colorbar=True,
                   show=False)
    rightmost, leftmost, w_px = _rightmost_leftmost_inked(fig)
    assert rightmost < w_px - 1, "long label(s) clipped off the right edge"
    assert leftmost > 0

    _, cbar_ax = fig.axes
    cbar_labels = [t.get_text() for t in cbar_ax.get_yticklabels()]
    assert cbar_labels == long_labels


# ------------------------------------------------------------------------
# discrete colorbar segment order must match the legend order (GH #100
# follow-up): legend reads first-to-last TOP-to-BOTTOM, so a VERTICAL
# discrete colorbar must too (first group at the top); a HORIZONTAL one
# must read first-to-last LEFT-to-RIGHT (first group leftmost).
# --------------------------------------------------------------------------

def test_discrete_colorbar_mpl_vertical_reads_top_to_bottom_in_legend_order():
    x, hue = _discrete_hue_data()
    legend = ['group A', 'group B', 'group C']
    fig = hyp.plot(x, hue=hue, colorbar=True, legend=legend, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    ax, cbar_ax = fig.axes

    # ground truth: the legend's own artists, top-to-bottom by DISPLAY y
    # (descending y = higher on the canvas = earlier), and each group's
    # actual drawn color (the legend proxy handles' colors).
    handles, legend_texts = ax.get_legend().legend_handles, ax.get_legend().texts
    legend_entries = sorted(
        zip((t.get_window_extent(renderer).y0 for t in legend_texts),
            (t.get_text() for t in legend_texts),
            (h.get_color() for h in handles)),
        key=lambda e: -e[0])
    legend_order = [label for _, label, _ in legend_entries]
    color_by_label = {label: color for _, label, color in legend_entries}
    assert legend_order == legend, "sanity check: legend itself reads top-down"

    # colorbar: tick labels sorted by DESCENDING display y (top first)
    ticklabels = cbar_ax.get_yticklabels()
    ticks = cbar_ax.get_yticks()
    entries = sorted(
        zip((t.get_window_extent(renderer).y0 for t in ticklabels),
            (t.get_text() for t in ticklabels), ticks),
        key=lambda e: -e[0])
    cbar_order = [label for _, label, _ in entries]
    assert cbar_order == legend_order, (
        "discrete colorbar must read top-to-bottom in the SAME order as "
        "the legend")

    # critical invariant: the order flip must not break color<->label
    # pairing -- each tick's face color (sampled through the colorbar's
    # own cmap/norm at that tick's data value) must equal the actual
    # drawn color for that group.
    mesh = _quadmesh(cbar_ax)
    for _, label, tickval in entries:
        face = mesh.cmap(mesh.norm(tickval))[:3]
        assert np.allclose(face, color_by_label[label], atol=1e-6), (
            f"{label}'s colorbar segment color does not match its line color")
    plt.close(fig)


def test_discrete_colorbar_mpl_horizontal_reads_left_to_right_in_legend_order():
    x, hue = _discrete_hue_data()
    legend = ['group A', 'group B', 'group C']
    fig = hyp.plot(x, hue=hue, colorbar={'location': 'bottom'},
                   legend=legend, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    ax, cbar_ax = fig.axes

    handles, legend_texts = ax.get_legend().legend_handles, ax.get_legend().texts
    color_by_label = {t.get_text(): h.get_color()
                      for t, h in zip(legend_texts, handles)}

    ticklabels = cbar_ax.get_xticklabels()
    ticks = cbar_ax.get_xticks()
    entries = sorted(
        zip((t.get_window_extent(renderer).x0 for t in ticklabels),
            (t.get_text() for t in ticklabels), ticks),
        key=lambda e: e[0])  # ascending x = left first
    cbar_order = [label for _, label, _ in entries]
    assert cbar_order == legend, (
        "horizontal discrete colorbar must read left-to-right in legend order")

    mesh = _quadmesh(cbar_ax)
    for _, label, tickval in entries:
        face = mesh.cmap(mesh.norm(tickval))[:3]
        assert np.allclose(face, color_by_label[label], atol=1e-6)
    plt.close(fig)


def test_discrete_colorbar_plotly_vertical_reads_top_to_bottom_in_legend_order():
    x, hue = _discrete_hue_data()
    legend = ['group A', 'group B', 'group C']
    fig = hyp.plot(x, hue=hue, colorbar=True, legend=legend,
                   backend='plotly', show=False)
    cb_trace = fig.data[-1]
    cb = cb_trace.marker.colorbar

    assert list(cb.ticktext) == legend
    # tickvals descending -> tick for legend[0] sits at the HIGHEST value,
    # which plotly renders nearest cmax, i.e. the TOP of a vertical colorbar
    tickvals = list(cb.tickvals)
    assert tickvals == sorted(tickvals, reverse=True)
    assert tickvals[0] == max(tickvals), "first legend entry must be at the top"

    # color pairing: the segment at each tickval's position holds the
    # SAME group's actual drawn color
    expected = get_palette_colors('hls', 3)

    def _parse_rgb(s):
        return tuple(int(v) for v in s[len('rgb('):-1].split(','))

    cmin, cmax = cb_trace.marker.cmin, cb_trace.marker.cmax
    scale = cb_trace.marker.colorscale
    for label, tickval, exp in zip(legend, tickvals, expected):
        frac = (tickval - cmin) / (cmax - cmin)
        # find the segment whose [start, end) fraction range contains frac
        seg_color = None
        for i in range(0, len(scale), 2):
            lo, hi = scale[i][0], scale[i + 1][0]
            if lo - 1e-9 <= frac <= hi + 1e-9:
                seg_color = _parse_rgb(scale[i][1])
                break
        assert seg_color is not None, f"no segment found for {label} at frac={frac}"
        assert seg_color == tuple(int(round(255 * c)) for c in exp), (
            f"{label}'s colorbar segment color does not match its group color")


# --------------------------------------------------- continuous regression
def test_continuous_colorbar_mpl_orientation_unchanged():
    """Regression: CONTINUOUS colorbars are numeric and must keep the
    conventional low-at-bottom orientation (untouched by the discrete
    top-to-bottom legend-order fix, GH #100 follow-up)."""
    x, hue = _continuous_hue_data()
    fig = hyp.plot(x, hue=hue, fmt='-', colorbar=True, show=False)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    _, cbar_ax = fig.axes

    assert not cbar_ax.yaxis_inverted(), (
        "continuous colorbar's y-axis must not be inverted")

    ticklabels = cbar_ax.get_yticklabels()
    ticks = cbar_ax.get_yticks()
    # the tick with the SMALLEST data value must be at the BOTTOM (smallest
    # display y0) -- i.e. vmin at bottom, vmax at top (unchanged convention)
    pairs = sorted(zip(ticks, ticklabels), key=lambda p: p[0])
    y_positions = [t.get_window_extent(renderer).y0 for _, t in pairs]
    assert y_positions == sorted(y_positions), (
        "vmin must render at the bottom and vmax at the top")
    plt.close(fig)


def test_continuous_colorbar_plotly_orientation_unchanged():
    x, hue = _continuous_hue_data()
    fig = hyp.plot(x, hue=hue, fmt='-', colorbar=True, backend='plotly',
                   show=False)
    cb_trace = fig.data[-1]
    # stop 0.0 (bottom of a vertical colorbar) must be the LOW end of the
    # palette (vmin's color), unchanged from before the discrete-only fix
    first_stop, _ = cb_trace.marker.colorscale[0]
    last_stop, _ = cb_trace.marker.colorscale[-1]
    assert first_stop == pytest.approx(0.0)
    assert last_stop == pytest.approx(1.0)
    assert cb_trace.marker.cmin < cb_trace.marker.cmax
