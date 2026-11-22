"""`panels=` grids: color means one thing across the whole figure, and the
legend and colorbar are drawn once for the figure (owner decision
2026-11-22, for 1.2.0).

Every test draws a real grid and reads the real artists (or plotly traces)
back; the layout tests measure window extents after a real draw.
"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb, to_rgba

import hypertools as hyp
from hypertools.plot.plot import (CAPTION_GID, SHARED_COLORBAR_GID,
                                  SHARED_LEGEND_GID)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _walks(n=4, rows=30, cols=5, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)).cumsum(axis=0) for _ in range(n)]


def _quiet_plot(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, show=False, **kwargs)


def _shared_legend(fig):
    found = [lg for lg in fig.legends if lg.get_gid() == SHARED_LEGEND_GID]
    assert len(found) <= 1
    return found[0] if found else None


def _shared_colorbars(fig):
    return [a for a in fig.axes if a.get_gid() == SHARED_COLORBAR_GID]


def _entries(legend):
    """``[(label, rgb)]`` of a legend's entries, rounded for comparison."""
    out = []
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        color = (handle.get_color() if hasattr(handle, 'get_color')
                 else handle.get_facecolor())
        color = np.asarray(color, dtype=object)
        rgb = (to_rgb(color.tolist()) if color.ndim <= 1
               else to_rgb(np.asarray(color[0], dtype=float)))
        out.append((text.get_text(), tuple(np.round(rgb, 4))))
    return out


def _line_colors(ax):
    return [tuple(np.round(to_rgb(line.get_color()), 4)) for line in ax.lines]


def _assert_strip_clear(fig, panel_axes):
    """The shared legend/colorbar overlap no panel, title or caption and lie
    inside the canvas (window extents after a real draw)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    strip = []
    legend = _shared_legend(fig)
    if legend is not None:
        strip.append(legend.get_window_extent(renderer))
    for cax in _shared_colorbars(fig):
        strip.append(cax.get_tightbbox(renderer))
    assert strip, "no shared legend or colorbar to measure"
    others = []
    for ax in panel_axes:
        others.append(ax.get_window_extent(renderer))
        if ax.get_title():
            title = ax.title.get_window_extent(renderer)
            # the grid is laid out exactly as it is without the strip, so
            # every title stays on the canvas
            assert title.y1 <= canvas.y1 + 0.5 and title.x0 >= canvas.x0
            others.append(title)
    others += [t.get_window_extent(renderer) for t in fig.texts
               if t.get_gid() == CAPTION_GID]
    for box in strip:
        assert box.x0 >= canvas.x0 - 0.5 and box.x1 <= canvas.x1 + 0.5, box
        assert box.y0 >= canvas.y0 - 0.5 and box.y1 <= canvas.y1 + 0.5, box
        for other in others:
            assert not box.overlaps(other), (box, other)
    for a, b in zip(strip, strip[1:]):
        assert not a.overlaps(b)
    return strip


# -- per-dataset colors ------------------------------------------------------

@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
@pytest.mark.parametrize('ndims', [2, 3])
def test_each_dataset_keeps_its_single_axes_color(panel_fit, ndims):
    x = _walks()
    single = _quiet_plot(x, ndims=ndims, return_model=True)
    expected = _line_colors(single['fig'].axes[0])
    assert len(set(expected)) == 4
    grid = _quiet_plot(x, ndims=ndims, panels=True, panel_fit=panel_fit,
                       return_model=True)
    drawn = [_line_colors(ax)[0] for ax in grid['axes']]
    assert drawn == expected
    # the bundle reports what was drawn: each panel its own color, the grid
    # every dataset's
    for model, color in zip(grid['panel_models'], expected):
        assert np.allclose(model['colors']['colors'], [color], atol=1e-3)
    assert np.allclose(grid['colors']['colors'], expected, atol=1e-3)


def test_explicit_colors_and_fmt_letters_are_honored_as_on_one_axes():
    x = _walks(3)
    for kw in (dict(fmt=['r-', '-', 'b--']),
               dict(color=['k', 'orange', 'purple']),
               dict(palette='viridis'),
               dict(palette=['#111111', '#22aa22', '#3333ff'])):
        single = _quiet_plot(x, ndims=2, **kw)
        expected = _line_colors(single.axes[0])
        grid = _quiet_plot(x, ndims=2, panels=True, return_model=True, **kw)
        assert [_line_colors(ax)[0] for ax in grid['axes']] == expected, kw


# -- one legend ---------------------------------------------------------------

def test_one_legend_for_the_figure_with_every_dataset():
    x = _walks()
    names = ['a', 'b', 'c', 'd']
    single = _quiet_plot(x, legend=names)
    expected = _entries(single.axes[0].get_legend())
    grid = _quiet_plot(x, legend=names, panels=True, return_model=True)
    fig = grid['fig']
    assert all(ax.get_legend() is None for ax in grid['axes'])
    legend = _shared_legend(fig)
    assert legend is not None
    assert _entries(legend) == expected
    assert [label for label, _ in _entries(legend)] == names


def test_legend_true_numbers_the_datasets_as_the_single_axes_call_does():
    x = _walks()
    single = _quiet_plot(x, legend=True)
    grid = _quiet_plot(x, legend=True, panels=True, return_model=True)
    legend = _shared_legend(grid['fig'])
    assert _entries(legend) == _entries(single.axes[0].get_legend())
    assert [label for label, _ in _entries(legend)] == ['1', '2', '3', '4']


def test_a_label_that_looks_different_in_two_panels_keeps_panel_legends():
    """One shared entry would be wrong for one of the panels, so the grid
    does not claim one."""
    x = _walks(2)
    grid = _quiet_plot(x, legend=['same', 'same'], panels=True,
                       return_model=True)
    fig = grid['fig']
    assert fig.legends == []
    assert [[t.get_text() for t in ax.get_legend().get_texts()]
            for ax in grid['axes']] == [['same'], ['same']]
    # ...and the datasets still keep their own colors
    assert _line_colors(grid['axes'][0]) != _line_colors(grid['axes'][1])


def test_legend_kwargs_style_the_one_legend_and_can_place_it():
    x = _walks(3)
    fig = _quiet_plot(x, legend=['a', 'b', 'c'], panels=True,
                      legend_kwargs={'title': 'who', 'fontsize': 14})
    legend = _shared_legend(fig)
    assert legend.get_title().get_text() == 'who'
    assert legend.get_texts()[0].get_fontsize() == 14
    _assert_strip_clear(fig, fig.axes[:3])
    placed = _quiet_plot(x, legend=['a', 'b', 'c'], panels=True,
                         legend_kwargs={'loc': 'lower center', 'ncol': 3})
    legend = _shared_legend(placed)
    placed.canvas.draw()
    box = legend.get_window_extent(placed.canvas.get_renderer())
    # figure coordinates: centered under the whole grid, and no strip
    assert abs(0.5 * (box.x0 + box.x1) - 0.5 * placed.bbox.width) < 2
    assert box.y0 < 0.1 * placed.bbox.height
    assert placed.get_size_inches()[0] == pytest.approx(
        _quiet_plot(x, panels=True).get_size_inches()[0])


def test_shared_legend_clears_a_3d_panels_axis_labels():
    """A 3-D panel's z-axis label hangs right of its box: the strip starts
    right of the last column's label, not on top of it."""
    import pandas as pd
    rng = np.random.default_rng(3)
    x = [pd.DataFrame(rng.normal(size=(40, 3)).cumsum(0),
                      columns=['first axis', 'second axis', 'third axis'])
         for _ in range(3)]
    grid = _quiet_plot(x, reduce=None, legend=['a', 'b', 'c'],
                       panels=(1, 3), return_model=True)
    fig = grid['fig']
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = _shared_legend(fig).get_window_extent(renderer)
    assert legend.x1 <= fig.bbox.x1
    labels = [axis.label.get_window_extent(renderer)
              for ax in grid['axes'] for axis in ax._axis_map.values()
              if axis.label.get_text()]
    # CHANGED for 1.2.0 (owner decision 2026-11-22): shared axis labels are
    # drawn on the outer panels only -- in this 1 x 3 row the two floor
    # labels under every panel and the z label on the right-most one,
    # which is the one beside the legend (this asserted all 9)
    assert len(labels) == 7
    assert grid['axes'][-1].zaxis.label.get_text() == 'third axis'
    for box in labels:
        assert not box.overlaps(legend)


def test_legend_false_or_absent_adds_nothing():
    x = _walks()
    for kw in ({}, dict(legend=False), dict(legend=None)):
        fig = _quiet_plot(x, panels=True, **kw)
        assert fig.legends == []
        assert all(ax.get_legend() is None for ax in fig.axes)
        assert _shared_colorbars(fig) == []
        assert len(fig.axes) == 4


def test_single_panel_grid_has_one_shared_legend():
    x = _walks(2)
    grid = _quiet_plot(x, reduce=['PCA'], panels=True, legend=['a', 'b'],
                       return_model=True)
    assert len(grid['axes']) == 1
    legend = _shared_legend(grid['fig'])
    assert [label for label, _ in _entries(legend)] == ['a', 'b']
    assert grid['axes'][0].get_legend() is None
    _assert_strip_clear(grid['fig'], grid['axes'])


def test_reducer_list_grid_has_one_legend():
    x = _walks()
    names = ['a', 'b', 'c', 'd']
    grid = _quiet_plot(x, reduce=['PCA', 'IncrementalPCA', 'FastICA'],
                       panels=True, legend=names, return_model=True)
    legend = _shared_legend(grid['fig'])
    assert [label for label, _ in _entries(legend)] == names
    assert all(ax.get_legend() is None for ax in grid['axes'])
    # every panel draws every dataset in the legend's colors
    colors = [color for _, color in _entries(legend)]
    for ax in grid['axes']:
        assert _line_colors(ax)[:4] == colors
    _assert_strip_clear(grid['fig'], grid['axes'])


def test_forecast_and_truth_entries_are_listed_once():
    x = _walks(3, rows=40, cols=2)
    rng = np.random.default_rng(1)
    truth = [xi[-1] + rng.normal(size=(5, 2)).cumsum(axis=0) for xi in x]
    kw = dict(predict='ARIMA', t=5, truth=truth, legend=['a', 'b', 'c'],
              ndims=2)
    single = _quiet_plot(x, **kw)
    expected = _entries(single.axes[0].get_legend())
    grid = _quiet_plot(x, panels=True, return_model=True, **kw)
    legend = _shared_legend(grid['fig'])
    assert all(ax.get_legend() is None for ax in grid['axes'])
    labels = [label for label, _ in _entries(legend)]
    assert labels == ['a', 'b', 'c', 'ARIMA', 'truth']
    # the forecast/truth keys stand for every dataset's, so they wear the
    # neutral glyph the single-axes legend gives them
    assert _entries(legend) == expected


# -- continuous hue: one norm, one colorbar ----------------------------------

@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_continuous_hue_shares_one_norm_and_one_colorbar(panel_fit):
    x = _walks(3, rows=21)
    # panel k's values run k..k+2, so 2.0 is the last point of panel 0, the
    # middle of panel 1 and the first of panel 2
    hue = [np.linspace(k, k + 2, 21) for k in range(3)]
    grid = _quiet_plot(x, hue=hue, colorbar=True, panels=True, fmt='o',
                       ndims=2, panel_fit=panel_fit, return_model=True)
    fig = grid['fig']
    bars = _shared_colorbars(fig)
    assert len(bars) == 1
    assert len(fig.axes) == 3 + 1          # no per-panel colorbar axes
    for model in grid['panel_models']:
        assert model['colors']['vmin'] == pytest.approx(0.0)
        assert model['colors']['vmax'] == pytest.approx(4.0)
    assert grid['colors']['vmin'] == pytest.approx(0.0)
    assert grid['colors']['vmax'] == pytest.approx(4.0)

    def _point_colors(ax):
        return np.vstack([c.get_facecolors() for c in ax.collections
                          if len(c.get_facecolors()) > 1])

    c0, c1, c2 = (_point_colors(ax) for ax in grid['axes'])
    assert np.allclose(c0[-1], c1[10], atol=1e-6)
    assert np.allclose(c1[10], c2[0], atol=1e-6)
    # ...and it is the color the single-axes call gives that value
    single = _quiet_plot(x, hue=hue, fmt='o', ndims=2, return_model=True)
    cmap, norm = single['colors']['cmap'], single['colors']['norm']
    assert (norm.vmin, norm.vmax) == (0.0, 4.0)
    assert np.allclose(c0[-1][:3], cmap(norm(2.0))[:3], atol=0.02)
    # different values are different colors
    assert not np.allclose(c0[0], c2[-1], atol=0.05)
    _assert_strip_clear(fig, grid['axes'])


# -- categorical hue ----------------------------------------------------------

@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_categorical_hue_mapping_is_shared(panel_fit):
    x = _walks(4)
    hue = [['p'] * 15 + ['q'] * 15, ['q'] * 30,
           ['r'] * 10 + ['p'] * 20, ['q'] * 30]
    single = _quiet_plot(x, hue=hue, fmt='o', legend=True, return_model=True)
    expected = {k: tuple(np.round(v, 4))
                for k, v in single['colors']['categories'].items()}
    assert list(expected) == ['p', 'q', 'r']
    grid = _quiet_plot(x, hue=hue, fmt='o', legend=True, panels=True,
                       panel_fit=panel_fit, return_model=True)
    for model in grid['panel_models']:
        for label, color in model['colors']['categories'].items():
            assert tuple(np.round(color, 4)) == expected[label]
    legend = _shared_legend(grid['fig'])
    assert _entries(legend) == list(expected.items())
    assert all(ax.get_legend() is None for ax in grid['axes'])
    assert {k: tuple(np.round(v, 4)) for k, v
            in grid['colors']['categories'].items()} == expected


def test_categorical_hue_colorbar_is_one_bar_with_every_category():
    x = _walks(3)
    hue = [['p'] * 30, ['q'] * 30, ['r'] * 15 + ['p'] * 15]
    grid = _quiet_plot(x, hue=hue, fmt='o', colorbar=True, panels=True,
                       return_model=True)
    bars = _shared_colorbars(grid['fig'])
    assert len(bars) == 1
    assert len(grid['fig'].axes) == 4
    assert [t.get_text() for t in bars[0].get_yticklabels()] == ['p', 'q',
                                                                 'r']
    _assert_strip_clear(grid['fig'], grid['axes'])


def test_palette_dict_names_every_category_of_the_grid():
    """A `palette=` dict names the whole grid's categories; a panel that
    holds only some of them used to refuse the others' keys."""
    x = _walks(3)
    hue = [['p'] * 30, ['q'] * 30, ['r'] * 15 + ['p'] * 15]
    palette = {'p': 'red', 'q': 'green', 'r': 'blue'}
    grid = _quiet_plot(x, hue=hue, fmt='o', legend=True, palette=palette,
                       panels=True, return_model=True)
    assert _entries(_shared_legend(grid['fig'])) == [
        ('p', (1.0, 0.0, 0.0)), ('q', (0.0, 0.502, 0.0)),
        ('r', (0.0, 0.0, 1.0))]


@pytest.mark.parametrize('location', ['left', 'top', 'bottom'])
def test_colorbar_on_another_side_is_still_one_bar(location):
    x = _walks(3, rows=21)
    hue = [np.linspace(k, k + 2, 21) for k in range(3)]
    grid = _quiet_plot(x, hue=hue, colorbar={'location': location},
                       panels=True, ndims=2, return_model=True)
    fig = grid['fig']
    (bar,) = _shared_colorbars(fig)
    assert len(fig.axes) == 4
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = bar.get_tightbbox(renderer)
    assert box.x0 >= 0 and box.x1 <= fig.bbox.x1
    assert box.y0 >= 0 and box.y1 <= fig.bbox.y1
    for ax in grid['axes']:
        assert not box.overlaps(ax.get_window_extent(renderer))


# -- clusters -----------------------------------------------------------------

def test_shared_fit_clusters_have_one_legend():
    x = _walks(3)
    grid = _quiet_plot(x, n_clusters=3, fmt='o', legend=True, panels=True,
                       random_state=0, return_model=True)
    legend = _shared_legend(grid['fig'])
    assert legend is not None
    assert all(ax.get_legend() is None for ax in grid['axes'])
    labels = [label for label, _ in _entries(legend)]
    assert len(labels) == len(set(labels)) == 3


def test_independently_fit_clusters_keep_per_panel_legends():
    """The exception: clusters fit per panel are different clusters in every
    panel, so one shared legend would claim a meaning that is false."""
    x = _walks(3)
    grid = _quiet_plot(x, n_clusters=2, fmt='o', legend=True, colorbar=True,
                       panels=True, panel_fit='independent', random_state=0,
                       return_model=True)
    fig = grid['fig']
    assert _shared_legend(fig) is None and fig.legends == []
    assert _shared_colorbars(fig) == []
    assert all(ax.get_legend() is not None for ax in grid['axes'])
    assert len(fig.axes) == 3 + 3          # one colorbar per panel
    # a reducer list clusters each panel's own embedding: also per panel
    grid = _quiet_plot(x, n_clusters=2, fmt='o', legend=True, panels=True,
                       reduce=['PCA', 'FastICA'], random_state=0,
                       return_model=True)
    assert _shared_legend(grid['fig']) is None
    assert all(ax.get_legend() is not None for ax in grid['axes'])


# -- layout -------------------------------------------------------------------

@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('size', [None, (9, 4)])
def test_shared_legend_colorbar_caption_and_titles_do_not_overlap(ndims,
                                                                  size):
    x = _walks(3)
    hue = [['alpha'] * 30, ['beta'] * 30, ['gamma'] * 15 + ['alpha'] * 15]
    grid = _quiet_plot(
        x, hue=hue, fmt='o', legend=True, colorbar=True, panels=(1, 3),
        ndims=ndims, size=size, title=['one', 'two', 'three'],
        caption='a caption under the whole grid', return_model=True)
    fig = grid['fig']
    assert _shared_legend(fig) is not None
    assert len(_shared_colorbars(fig)) == 1
    strip = _assert_strip_clear(fig, grid['axes'])
    # right of every panel, and vertically centered on the grid
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer) for ax in grid['axes']]
    right = max(b.x1 for b in boxes)
    middle = 0.5 * (min(b.y0 for b in boxes) + max(b.y1 for b in boxes))
    for box in strip:
        assert box.x0 >= right
        assert abs(0.5 * (box.y0 + box.y1) - middle) <= 0.12 * fig.bbox.height
    if size is not None:
        assert tuple(fig.get_size_inches()[:1]) == (9.0,)


def test_default_figure_reserves_one_strip_not_one_per_column():
    x = _walks(4)
    plain = _quiet_plot(x, panels=(1, 4), ndims=2)
    with_legend = _quiet_plot(x, panels=(1, 4), ndims=2,
                              legend=['a', 'b', 'c', 'd'])
    extra = with_legend.get_size_inches()[0] - plain.get_size_inches()[0]
    assert 0.3 < extra < 2.0               # was 1.1 in per column (4.4 in)
    assert with_legend.get_size_inches()[1] == plain.get_size_inches()[1]
    _assert_strip_clear(with_legend, with_legend.axes[:4])


def test_2x2_grid_legend_is_clear():
    x = _walks(4)
    grid = _quiet_plot(x, panels=(2, 2), ndims=2,
                       legend=['first', 'second', 'third', 'fourth'],
                       title=['a', 'b', 'c', 'd'], return_model=True)
    _assert_strip_clear(grid['fig'], grid['axes'])


# -- animated grids -----------------------------------------------------------

def test_animated_grid_shared_legend_is_static_and_clear_of_the_cubes():
    x = _walks(4, rows=40)
    names = ['a', 'b', 'c', 'd']
    single = _quiet_plot(x, legend=names)
    expected = _entries(single.axes[0].get_legend())
    grid = _quiet_plot(x, panels=(1, 4), animate=True, legend=names,
                       duration=2, frame_rate=10, rotations=1,
                       caption='caption', return_model=True)
    fig, anim = grid['fig'], grid['animation']
    legend = _shared_legend(fig)
    assert legend is not None
    assert _entries(legend) == expected
    assert [_line_colors(ax)[0] for ax in grid['axes']] == [
        color for _, color in expected]
    n_frames = int(anim._save_count)
    assert n_frames == 20
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box0 = legend.get_window_extent(renderer)
    for frame in range(n_frames):
        anim._func(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        assert _shared_legend(fig) is legend
        assert legend.get_visible()
        box = legend.get_window_extent(renderer)
        assert np.allclose(box.extents, box0.extents, atol=0.5)
        assert all(ax.get_legend() is None for ax in grid['axes'])
        for ax in grid['axes']:
            # everything the cube draws (its panes, lines and data)
            for artist in (list(ax.lines) + list(ax.collections)
                           + [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]):
                if not artist.get_visible():
                    continue
                extent = artist.get_window_extent(renderer)
                if not np.all(np.isfinite(extent.extents)):
                    continue
                assert extent.x1 <= box.x0, (frame, artist)
    _assert_strip_clear(fig, grid['axes'])


def test_animated_grid_colorbar_is_shared():
    x = _walks(3, rows=30)
    hue = [np.linspace(k, k + 2, 30) for k in range(3)]
    grid = _quiet_plot(x, hue=hue, colorbar=True, panels=(1, 3),
                       animate=True, duration=1, frame_rate=5,
                       return_model=True)
    fig = grid['fig']
    assert len(_shared_colorbars(fig)) == 1
    assert len(fig.axes) == 4
    assert grid['colors']['vmin'] == pytest.approx(0.0)
    assert grid['colors']['vmax'] == pytest.approx(4.0)
    _assert_strip_clear(fig, grid['axes'])


# -- plotly -------------------------------------------------------------------

def _plotly_legend_names(fig):
    return [t.name for t in fig.data if t.showlegend]


def _plotly_rgb(color):
    import re
    if color.startswith('#'):
        return tuple(np.round(to_rgb(color), 2))
    nums = [float(v) for v in re.findall(r'[\d.]+', color)[:3]]
    return tuple(np.round(np.asarray(nums) / 255.0, 2))


def test_plotly_grid_shows_each_legend_entry_once_in_dataset_colors():
    x = _walks()
    names = ['a', 'b', 'c', 'd']
    single = _quiet_plot(x, legend=names, return_model=True)
    expected = [tuple(np.round(c, 2)) for c in single['colors']['colors']]
    fig = _quiet_plot(x, legend=names, panels=True, backend='plotly')
    assert _plotly_legend_names(fig) == names
    shown = [t for t in fig.data if t.showlegend]
    assert [_plotly_rgb(t.line.color) for t in shown] == expected
    # one legend for the figure, right of the grid
    assert {t.legend for t in shown} <= {None, 'legend'}
    assert fig.layout.legend.x >= 1.0
    assert fig.layout.showlegend is True


def test_plotly_grid_has_one_colorbar_and_a_shared_color_range():
    x = _walks(3, rows=21)
    hue = [np.linspace(k, k + 2, 21) for k in range(3)]
    grid = _quiet_plot(x, hue=hue, colorbar=True, panels=True, fmt='o',
                       ndims=2, backend='plotly', return_model=True)
    fig = grid['fig']
    scales = [t for t in fig.data
              if getattr(getattr(t, 'marker', None), 'showscale', None)]
    assert len(scales) == 1
    assert (scales[0].marker.cmin, scales[0].marker.cmax) == (0.0, 4.0)
    assert scales[0].marker.colorbar.x >= 1.0
    for model in grid['panel_models']:
        assert (model['colors']['vmin'], model['colors']['vmax']) == (0.0,
                                                                     4.0)
    # hue value 2.0 is one color in all three panels
    points = [t for t in fig.data if t.mode == 'markers'
              and t.marker.color is not None and len(t.x) == 21]
    assert len(points) == 3
    assert points[0].marker.color[-1] == points[1].marker.color[10]
    assert points[1].marker.color[10] == points[2].marker.color[0]


def test_plotly_grid_categorical_hue_and_legend_false():
    x = _walks(3)
    hue = [['p'] * 30, ['q'] * 30, ['r'] * 15 + ['p'] * 15]
    single = _quiet_plot(x, hue=hue, fmt='o', legend=True,
                         return_model=True)
    expected = {k: tuple(np.round(v, 2))
                for k, v in single['colors']['categories'].items()}
    fig = _quiet_plot(x, hue=hue, fmt='o', legend=True, panels=True,
                      backend='plotly')
    assert _plotly_legend_names(fig) == ['p', 'q', 'r']
    for trace in fig.data:
        if trace.name in expected and trace.marker.color is not None \
                and isinstance(trace.marker.color, str):
            assert _plotly_rgb(trace.marker.color) == expected[trace.name]
    # ...and an explicit legend=False suppresses it: the categorical hue
    # still names and colors the traces, but none is listed and the
    # figure shows no legend (review round 2: this combination, which the
    # test is named for, was never exercised)
    off = _quiet_plot(x, hue=hue, fmt='o', legend=False, panels=True,
                      backend='plotly')
    named = [t for t in off.data if t.name in expected]
    assert {t.name for t in named} == {'p', 'q', 'r'}
    assert all(t.showlegend is False for t in off.data)
    assert off.layout.showlegend is False
    for trace in named:
        if isinstance(trace.marker.color, str):
            assert _plotly_rgb(trace.marker.color) == expected[trace.name]
    bare = _quiet_plot(x, panels=True, backend='plotly')
    assert not any(t.showlegend for t in bare.data)
    assert not bare.layout.showlegend


@pytest.mark.parametrize('location', ['right', 'left', 'top', 'bottom'])
def test_plotly_shared_colorbar_stays_on_the_canvas(location):
    x = _walks(3, rows=21)
    hue = [np.linspace(k, k + 2, 21) for k in range(3)]
    fig = _quiet_plot(x, hue=hue, colorbar={'location': location},
                      panels=True, backend='plotly',
                      title=['one', 'two', 'three'])
    (bar,) = [t.marker.colorbar for t in fig.data
              if getattr(getattr(t, 'marker', None), 'showscale', None)]
    margin = fig.layout.margin
    plot_w = fig.layout.width - margin.l - margin.r
    plot_h = fig.layout.height - margin.t - margin.b
    if location == 'right':
        assert bar.x >= 1.0 and (bar.x - 1.0) * plot_w + 60 <= margin.r
    elif location == 'left':
        assert bar.x <= 0.0 and -bar.x * plot_w + 60 <= margin.l
    elif location == 'top':
        assert bar.y >= 1.0 and (bar.y - 1.0) * plot_h + 50 <= margin.t
    else:
        assert bar.y <= 0.0 and -bar.y * plot_h + 50 <= margin.b


def test_plotly_forecast_keys_are_listed_once():
    x = _walks(3, rows=40, cols=2)
    kw = dict(predict='ARIMA', t=5, legend=['a', 'b', 'c'], ndims=2,
              backend='plotly')
    single = _quiet_plot(x, **kw)
    fig = _quiet_plot(x, panels=True, **kw)
    assert sorted(_plotly_legend_names(fig)) == sorted(
        _plotly_legend_names(single)) == ['ARIMA', 'a', 'b', 'c']
    key = [t for t in fig.data if t.showlegend and t.name == 'ARIMA'][0]
    ref = [t for t in single.data if t.showlegend and t.name == 'ARIMA'][0]
    # one model continuing three datasets: the neutral key, as on one axes
    assert _plotly_rgb(key.line.color) == _plotly_rgb(ref.line.color)


def test_plotly_independent_clusters_keep_per_panel_legends():
    x = _walks(3)
    fig = _quiet_plot(x, n_clusters=2, fmt='o', legend=True, panels=True,
                      panel_fit='independent', random_state=0,
                      backend='plotly')
    shown = [t for t in fig.data if t.showlegend]
    assert len({t.legend for t in shown}) == 3
