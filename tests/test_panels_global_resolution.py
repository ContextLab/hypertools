"""`panels=` grids resolve their colors ONCE, for the whole grid (1.2.0,
review round 2).

A panel grid's colors, hue classification (continuous vs categorical), hue
range, category-to-color mapping, palette validation, legend names and
their validation are resolved globally, exactly as the equivalent
single-axes call of all the datasets resolves them, and each panel is handed
its share; no panel re-decides any of these from its own subset of the
data. Every test here compares a real grid (matplotlib artists or plotly
traces, read back after a real draw) with that single-axes call, on calls
such as ``hyp.plot(x, hue=hue, legend=True, panels=True)``.
"""
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import hypertools as hyp
from hypertools.plot.colors import NAN_COLOR
from hypertools.plot.plot import SHARED_COLORBAR_GID, SHARED_LEGEND_GID

BACKENDS = ['matplotlib', 'plotly']


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _xy():
    """Two 2-D walks of 20 rows (the review's reproduction data)."""
    x = np.random.default_rng(0).normal(size=(20, 2)).cumsum(0)
    return [x, x + 3]


def _three():
    """Three 4-column clouds of 30 rows, offset from each other."""
    rng = np.random.default_rng(1)
    return [rng.normal(size=(30, 4)) + i for i in range(3)]


K = dict(reduce=None, ndims=2, show=False)


def _plot(*args, **kwargs):
    """`hyp.plot`, with every warning it raises recorded: ``(result,
    [messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = hyp.plot(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _quiet(*args, **kwargs):
    """`hyp.plot` with its warnings ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, **kwargs)


def _mpl_legends(fig):
    """The label lists of every legend on a matplotlib figure: the
    figure-level (shared) ones first, then each axes' own."""
    out = [[t.get_text() for t in lg.get_texts()] for lg in fig.legends]
    out += [[t.get_text() for t in ax.get_legend().get_texts()]
            for ax in fig.axes if ax.get_legend() is not None]
    return out


def _mpl_legend_colors(legend):
    """``{label: rgb}`` of a matplotlib legend's entries."""
    out = {}
    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        color = (handle.get_color() if hasattr(handle, 'get_color')
                 else handle.get_facecolor())
        color = np.asarray(color, dtype=object)
        rgb = (to_rgb(color.tolist()) if color.ndim <= 1
               else to_rgb(np.asarray(color[0], dtype=float)))
        out[text.get_text()] = tuple(np.round(rgb, 3))
    return out


def _plotly_legend_names(fig):
    """The names a plotly figure's legend lists, in the order it lists
    them (by ``legendrank``, then trace order)."""
    listed = [(t.legendrank if t.legendrank is not None else 1000, i,
               str(t.name))
              for i, t in enumerate(fig.data) if t.showlegend]
    return [name for _, _, name in sorted(listed)]


def _plotly_rgb(color):
    """A plotly ``rgb(...)``/``rgba(...)`` string as a 0-1 rgb tuple."""
    numbers = [float(v) for v in re.findall(r'[\d.]+', color)[:3]]
    return tuple(np.round(np.asarray(numbers) / 255.0, 3))


def _point_colors(fig, backend):
    """The per-observation marker colors of every dataset a figure draws
    with a continuous hue, one (n, 3) array per drawn dataset in drawing
    order."""
    if backend == 'plotly':
        out = []
        for trace in fig.data:
            colors = getattr(trace.marker, 'color', None)
            if colors is None or isinstance(colors, str):
                continue
            if getattr(trace.marker, 'showscale', None):
                continue  # the colorbar's phantom trace
            out.append(np.asarray([_plotly_rgb(c) for c in colors]))
        return out
    return [np.round(np.asarray(coll.get_facecolors())[:, :3], 3)
            for ax in fig.axes for coll in ax.collections
            if len(coll.get_offsets()) > 1]


# --- Codex 1: the hue's classification is the grid's ----------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_integer_hue_is_classified_once_for_the_grid(backend):
    hue = [np.tile(np.arange(10), 2), np.tile(np.arange(10, 20), 2)]
    kw = dict(fmt='o', hue=hue, colorbar=True, return_model=True,
              backend=backend, **K)
    single = _quiet(_xy(), **kw)
    grid = _quiet(_xy(), panels=True, **kw)
    # 20 distinct integers are continuous, though each panel holds 10
    assert single['colors']['kind'] == 'continuous'
    assert grid['colors']['kind'] == 'continuous'
    assert [m['colors']['kind'] for m in grid['panel_models']] \
        == ['continuous', 'continuous']
    assert (grid['colors']['vmin'], grid['colors']['vmax']) == (0.0, 19.0)
    expected = _point_colors(single['fig'], backend)
    drawn = _point_colors(grid['fig'], backend)
    assert len(expected) == len(drawn) == 2
    for want, got in zip(expected, drawn):
        np.testing.assert_allclose(got, want, atol=2e-3)
    # the palette does not restart in the second panel
    assert not np.allclose(drawn[0][0], drawn[1][0], atol=1e-3)


@pytest.mark.parametrize('backend', BACKENDS)
def test_small_integer_hue_stays_categorical_in_every_panel(backend):
    # 4 categories over the grid; the second panel has as many distinct
    # values as observations, which alone would read as continuous
    x = [np.random.default_rng(2).normal(size=(12, 2)),
         np.random.default_rng(3).normal(size=(2, 2))]
    hue = [np.repeat([0, 1], 6), np.array([2, 3])]
    kw = dict(fmt='o', hue=hue, legend=True, return_model=True,
              backend=backend, **K)
    single = _quiet(x, **kw)
    grid = _quiet(x, panels=True, **kw)
    assert single['colors']['kind'] == 'discrete'
    assert grid['colors']['kind'] == 'discrete'
    assert [m['colors']['kind'] for m in grid['panel_models']] \
        == ['discrete', 'discrete']
    assert grid['colors']['labels'] == single['colors']['labels']


# --- Codex 2: the bundle's colors are the grid's --------------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_bundle_colors_keep_categories_when_each_panel_has_one(backend):
    kw = dict(fmt='o', hue=[['a'] * 20, ['b'] * 20], legend=True,
              return_model=True, backend=backend, **K)
    single = _quiet(_xy(), **kw)
    grid = _quiet(_xy(), panels=True, **kw)
    assert grid['colors']['labels'] == ['a', 'b']
    assert list(grid['colors']['categories']) == ['a', 'b']
    for name in 'ab':
        np.testing.assert_allclose(grid['colors']['categories'][name],
                                   single['colors']['categories'][name])


# --- Codex 3: plotly legend entries that look different stay apart --------

def test_plotly_legend_entries_of_different_line_width_are_not_merged():
    fig = _quiet(_xy(), panels=True, color='red', linewidth=[1, 8],
                 legend=['same', 'same'], backend='plotly', **K)
    listed = [t for t in fig.data if t.showlegend]
    assert [t.name for t in listed] == ['same', 'same']
    assert listed[0].line.width != listed[1].line.width
    assert fig.layout.showlegend
    # ...as on matplotlib, which keeps a legend per panel
    mpl = _quiet(_xy(), panels=True, color='red', linewidth=[1, 8],
                 legend=['same', 'same'], backend='matplotlib', **K)
    assert _mpl_legends(mpl) == [['same'], ['same']]


def test_plotly_legend_entries_of_different_marker_size_are_not_merged():
    fig = _quiet(_xy(), panels=True, fmt='o', color='red',
                 markersize=[3, 14], legend=['same', 'same'],
                 backend='plotly', **K)
    listed = [t for t in fig.data if t.showlegend]
    assert [t.name for t in listed] == ['same', 'same']
    assert listed[0].marker.size != listed[1].marker.size


def test_plotly_identical_entries_are_still_merged():
    fig = _quiet(_xy(), panels=True, fmt='o',
                 hue=[['a'] * 20, ['a'] * 10 + ['b'] * 10], legend=True,
                 backend='plotly', **K)
    assert _plotly_legend_names(fig) == ['a', 'b']


# --- Codex 4 / R6: a panel with no finite hue value -----------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_all_nan_panel_draws_gray_under_the_grid_colorbar(backend):
    hue = [np.full(20, np.nan), np.linspace(0., 1., 20)]
    kw = dict(fmt='o', hue=hue, colorbar=True, return_model=True,
              backend=backend, **K)
    single, single_warned = _plot(_xy(), **kw)
    grid, warned = _plot(_xy(), panels=True, **kw)
    assert (grid['colors']['vmin'], grid['colors']['vmax']) == (0.0, 1.0)
    drawn = _point_colors(grid['fig'], backend)
    np.testing.assert_allclose(drawn[0], np.tile(NAN_COLOR, (20, 1)),
                               atol=2e-3)
    np.testing.assert_allclose(drawn[1],
                               _point_colors(single['fig'], backend)[1],
                               atol=2e-3)
    if backend == 'matplotlib':
        assert len([a for a in grid['fig'].axes
                    if a.get_gid() == SHARED_COLORBAR_GID]) == 1
    else:
        assert len([t for t in grid['fig'].data
                    if getattr(getattr(t, 'marker', None), 'showscale',
                               None)]) == 1
    # one warning, the single-axes call's (all 20 of them counted)
    nan_warnings = [m for m in warned if 'non-finite' in m]
    assert nan_warnings == [m for m in single_warned if 'non-finite' in m]
    assert len(nan_warnings) == 1 and nan_warnings[0].startswith('20 ')


# --- Codex 5: a colorbar for the datasets' own colors ---------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('legend', [['a', 'b'], None])
def test_dataset_color_colorbar_is_drawn_once_for_the_grid(backend, legend):
    kw = dict(colorbar=True, return_model=True, backend=backend, **K)
    if legend is not None:
        kw['legend'] = legend
    single = _quiet(_xy(), **kw)
    grid = _quiet(_xy(), panels=True, **kw)
    labels = legend if legend is not None else [1, 2]
    assert grid['colors']['labels'] == single['colors']['labels'] == labels
    np.testing.assert_allclose(np.asarray(grid['colors']['colors']),
                               np.asarray(single['colors']['colors']))
    fig = grid['fig']
    if backend == 'matplotlib':
        (bar,) = [a for a in fig.axes if a.get_gid() == SHARED_COLORBAR_GID]
        fig.canvas.draw()
        ticks = [t.get_text() for t in bar.get_yticklabels()]
        assert ticks == [str(label) for label in labels]
        # the bar's swatches are the colors the panels' lines are drawn in
        lines = [to_rgb(ax.lines[0].get_color())
                 for ax in grid['axes']]
        np.testing.assert_allclose(lines,
                                   np.asarray(grid['colors']['colors']),
                                   atol=1e-6)
    else:
        (bar,) = [t for t in fig.data
                  if getattr(getattr(t, 'marker', None), 'showscale', None)]
        assert list(bar.marker.colorbar.ticktext) \
            == [str(label) for label in labels]


# --- Codex 6: the palette is narrowed before independent fits -------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_fit_takes_the_grid_palette_mapping(backend):
    grid = _quiet(_xy(), panels=True, panel_fit='independent', fmt='o',
                  hue=[['a'] * 20, ['b'] * 20],
                  palette={'a': 'red', 'b': 'blue'}, legend=True,
                  return_model=True, backend=backend, **K)
    categories = grid['colors']['categories']
    assert tuple(categories['a']) == to_rgb('red')
    assert tuple(categories['b']) == to_rgb('blue')
    if backend == 'matplotlib':
        (legend,) = [lg for lg in grid['fig'].legends
                     if lg.get_gid() == SHARED_LEGEND_GID]
        assert _mpl_legend_colors(legend) == {'a': (1.0, 0.0, 0.0),
                                              'b': (0.0, 0.0, 1.0)}
    else:
        assert _plotly_legend_names(grid['fig']) == ['a', 'b']
        colors = {t.name: _plotly_rgb(t.marker.color)
                  for t in grid['fig'].data if t.showlegend}
        assert colors == {'a': (1.0, 0.0, 0.0), 'b': (0.0, 0.0, 1.0)}


@pytest.mark.parametrize('backend', BACKENDS)
def test_independent_fit_still_refuses_a_category_no_panel_has(backend):
    with pytest.raises(ValueError, match="does not have"):
        _quiet(_xy(), panels=True, panel_fit='independent', fmt='o',
               hue=[['a'] * 20, ['b'] * 20],
               palette={'a': 'red', 'b': 'blue', 'zzz': 'green'},
               backend=backend, **K)


# --- Codex 7 / R2 / R3 / R4: legend lists are the grid's ------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('legend, length', [(['only'], 1), ('x', 1),
                                            (['x', 'y'], 2)])
def test_dataset_legend_list_is_validated_against_the_grid(
        backend, legend, length):
    x = _three()
    with pytest.raises(ValueError) as single:
        _quiet(x, '-', legend=legend, show=False, backend=backend)
    with pytest.raises(ValueError) as grid:
        _quiet(x, '-', legend=legend, panels=(1, 3), show=False,
               backend=backend)
    assert str(grid.value) == str(single.value)
    assert (f"list of length {length}, but there are 3 dataset(s)"
            in str(grid.value))
    assert plt.get_fignums() == []


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt', ['o', '-'])
def test_legend_list_names_the_grid_categories(backend, fmt):
    x = _three()
    hue = [['a'] * 15 + ['b'] * 15, ['b'] * 10 + ['c'] * 20, ['d'] * 30]
    names = ['A', 'B', 'C', 'D']
    kw = dict(hue=hue, ndims=2, show=False, return_model=True,
              backend=backend)
    single = _quiet(x, fmt, legend=names, **kw)
    plain = _quiet(x, fmt, legend=True, **kw)
    grid = _quiet(x, fmt, legend=names, panels=(1, 3), **kw)
    assert grid['colors']['labels'] == single['colors']['labels'] == names
    # each name keeps its category's color
    for name, category in zip(names, 'abcd'):
        np.testing.assert_allclose(
            grid['colors']['categories'][name],
            plain['colors']['categories'][category])
    if backend == 'matplotlib':
        (legend,) = [lg for lg in grid['fig'].legends
                     if lg.get_gid() == SHARED_LEGEND_GID]
        assert [t.get_text() for t in legend.get_texts()] == names
        (single_legend,) = [ax.get_legend() for ax in single['fig'].axes
                            if ax.get_legend() is not None]
        assert _mpl_legend_colors(legend) \
            == _mpl_legend_colors(single_legend)
        assert all(ax.get_legend() is None for ax in grid['axes'])
    else:
        assert _plotly_legend_names(grid['fig']) == names
        assert _plotly_legend_names(single['fig']) == names


@pytest.mark.parametrize('backend', BACKENDS)
def test_categorical_legend_list_of_the_wrong_length_names_the_grid(backend):
    hue = [['a'] * 10 + ['b'] * 10, ['b'] * 10 + ['c'] * 10]
    with pytest.raises(ValueError, match=r"length 2, but there are 3 "):
        _quiet(_xy(), panels=True, fmt='o', hue=hue,
               legend=['Alpha', 'Beta'], backend=backend, **K)


@pytest.mark.parametrize('backend', BACKENDS)
def test_single_axes_line_legend_list_names_categories_not_runs(backend):
    # 'b' is drawn as two runs (one per dataset): three names for three
    # categories were refused as "3 entries for 4 dataset(s)/group(s)"
    hue = [['a'] * 10 + ['b'] * 10, ['b'] * 10 + ['c'] * 10]
    names = ['Alpha', 'Beta', 'Gamma']
    kw = dict(hue=hue, return_model=True, backend=backend, **K)
    lines = _quiet(_xy(), '-', legend=names, **kw)
    markers = _quiet(_xy(), 'o', legend=names, **kw)
    assert lines['colors']['labels'] == markers['colors']['labels'] == names
    for name in names:
        np.testing.assert_allclose(lines['colors']['categories'][name],
                                   markers['colors']['categories'][name])
    if backend == 'matplotlib':
        assert _mpl_legends(lines['fig']) == [names]
    else:
        assert _plotly_legend_names(lines['fig']) == names
    with pytest.raises(ValueError, match=r"length 2, but there are 3 "):
        _quiet(_xy(), '-', hue=hue, legend=['x', 'y'], backend=backend, **K)


def test_single_axes_line_legend_list_follows_the_legend_order():
    # integer categories are listed sorted (1, 3) though 3 is drawn first:
    # the names follow the legend, as on the marker path
    x = np.random.default_rng(0).normal(size=(20, 2)).cumsum(0)
    hue = np.repeat([3, 1], 10)
    for fmt in ('o', '-'):
        result = _quiet(x, fmt, hue=hue, legend=['one', 'three'],
                        return_model=True, **K)
        plain = _quiet(x, fmt, hue=hue, legend=True, return_model=True, **K)
        assert _mpl_legends(result['fig']) == [['one', 'three']]
        np.testing.assert_allclose(result['colors']['categories']['one'],
                                   plain['colors']['categories']['1'])
        np.testing.assert_allclose(result['colors']['categories']['three'],
                                   plain['colors']['categories']['3'])


# --- Codex 8: names= alone asks for the legend on plotly too --------------

def test_plotly_grid_shows_the_legend_names_asks_for():
    single = _quiet(_xy(), names=['a', 'b'], backend='plotly', **K)
    grid = _quiet(_xy(), names=['a', 'b'], panels=True, backend='plotly',
                  **K)
    assert single.layout.showlegend
    assert grid.layout.showlegend
    assert _plotly_legend_names(grid) == _plotly_legend_names(single) \
        == ['a', 'b']
    # every trace is on the figure's ONE legend
    assert {t.legend for t in grid.data if t.showlegend} <= {None, 'legend'}
    off = _quiet(_xy(), names=['a', 'b'], legend=False, panels=True,
                 backend='plotly', **K)
    assert not off.layout.showlegend
    assert not any(t.showlegend for t in off.data)
    mpl = _quiet(_xy(), names=['a', 'b'], panels=True,
                 backend='matplotlib', **K)
    assert _mpl_legends(mpl) == [['a', 'b']]


# --- R5: the shared legend lists clusters in label order ------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('fmt', ['o', '-'])
def test_shared_cluster_legend_is_in_label_order(backend, fmt):
    kw = dict(n_clusters=3, legend=True, random_state=0, ndims=2,
              show=False, backend=backend)
    single = _quiet(_three(), fmt, **kw)
    grid = _quiet(_three(), fmt, panels=(1, 3), **kw)
    if backend == 'matplotlib':
        assert _mpl_legends(single) == [['0', '1', '2']]
        assert _mpl_legends(grid) == [['0', '1', '2']]
    else:
        assert _plotly_legend_names(single) == ['0', '1', '2']
        assert _plotly_legend_names(grid) == ['0', '1', '2']


def test_animated_shared_cluster_legend_is_in_label_order():
    anim = _quiet(_three(), 'o', n_clusters=3, legend=True, random_state=0,
                  ndims=2, show=False, panels=(1, 3), animate=True,
                  duration=1, frame_rate=4)
    assert _mpl_legends(anim.figure) == [['0', '1', '2']]


@pytest.mark.parametrize('backend', BACKENDS)
def test_cluster_legend_list_names_the_grid_clusters(backend):
    names = ['low', 'mid', 'high']
    kw = dict(n_clusters=3, legend=names, random_state=0, ndims=2,
              show=False, backend=backend)
    single = _quiet(_three(), 'o', **kw)
    grid = _quiet(_three(), 'o', panels=(1, 3), **kw)
    if backend == 'matplotlib':
        assert _mpl_legends(grid) == _mpl_legends(single) == [names]
    else:
        assert _plotly_legend_names(grid) \
            == _plotly_legend_names(single) == names


# --- R7 / R10: warnings about the whole call are issued once --------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_nan_hue_warning_is_issued_once_with_the_total(backend, panel_fit):
    x = _three()
    hue = [np.r_[np.nan, np.linspace(0, 1, 29)] for _ in x]
    kw = dict(hue=hue, ndims=2, show=False, backend=backend)
    _, single = _plot(x, 'o', **kw)
    _, grid = _plot(x, 'o', panels=(1, 3), panel_fit=panel_fit, **kw)
    single = [m for m in single if 'non-finite' in m]
    grid = [m for m in grid if 'non-finite' in m]
    assert len(single) == 1 and single[0].startswith('3 observation(s)')
    assert grid == single


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('panel_fit', ['shared', 'independent'])
def test_continuous_hue_legend_warning_is_issued_once(backend, panel_fit):
    x = _three()
    hue = [np.linspace(0, 1, 30) for _ in x]
    kw = dict(hue=hue, legend=True, ndims=2, show=False, backend=backend)
    _, single = _plot(x, 'o', **kw)
    _, grid = _plot(x, 'o', panels=(1, 3), panel_fit=panel_fit, **kw)
    single = [m for m in single if 'legend is not supported' in m]
    grid = [m for m in grid if 'legend is not supported' in m]
    assert len(single) == 1
    assert grid == single


# --- a panel whose every label is missing ---------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
def test_panel_of_unlabeled_points_does_not_take_a_category_color(backend):
    hue = [[None] * 20, ['b'] * 10 + ['c'] * 10]
    kw = dict(fmt='o', hue=hue, legend=True, return_model=True,
              backend=backend, **K)
    single = _quiet(_xy(), **kw)
    grid = _quiet(_xy(), panels=True, **kw)
    assert grid['colors']['labels'] == ['b', 'c']
    for name in 'bc':
        np.testing.assert_allclose(grid['colors']['categories'][name],
                                   single['colors']['categories'][name])
    category_colors = [tuple(np.round(c, 3))
                       for c in grid['colors']['categories'].values()]
    if backend == 'matplotlib':
        first = grid['axes'][0]
        drawn = [tuple(np.round(to_rgb(line.get_color()), 3))
                 for line in first.lines]
        assert drawn and all(c not in category_colors for c in drawn)
        assert _mpl_legends(grid['fig']) == [['b', 'c']]
    else:
        unlabeled = [t for t in grid['fig'].data
                     if t.name not in ('b', 'c')
                     and isinstance(t.marker.color, str)]
        assert unlabeled
        assert all(_plotly_rgb(t.marker.color) not in category_colors
                   for t in unlabeled)
        assert _plotly_legend_names(grid['fig']) == ['b', 'c']


# --- a shared colorbar beside per-panel legends; one explicit color -------

@pytest.mark.parametrize('backend', BACKENDS)
def test_colorbar_of_one_explicit_color_reports_that_color(backend):
    # (single axes too: the swatches were the palette's, which nothing drew)
    for extra in ({}, {'panels': True}):
        result = _quiet(_xy(), color='red', colorbar=True,
                        return_model=True, backend=backend, **extra, **K)
        np.testing.assert_allclose(np.asarray(result['colors']['colors']),
                                   [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


@pytest.mark.parametrize('backend', BACKENDS)
def test_grid_colorbar_is_drawn_once_beside_per_panel_legends(backend):
    # the two 'same' entries look different (line width), so the legends
    # stay per panel; the colorbar is still the grid's one
    fig = _quiet(_xy(), panels=True, color='red', linewidth=[1, 8],
                 legend=['same', 'same'], colorbar=True, backend=backend,
                 **K)
    if backend == 'matplotlib':
        assert _mpl_legends(fig) == [['same'], ['same']]
        assert len([a for a in fig.axes
                    if a.get_gid() == SHARED_COLORBAR_GID]) == 1
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        (bar,) = [a for a in fig.axes if a.get_gid() == SHARED_COLORBAR_GID]
        bar_box = bar.get_tightbbox(renderer)
        assert bar_box.x1 <= fig.bbox.width + 1 and bar_box.x0 >= 0
        for ax in fig.axes:
            if ax is bar:
                continue
            assert not ax.get_legend().get_window_extent(
                renderer).overlaps(bar_box)
            assert ax.get_legend().get_window_extent(renderer).x1 \
                <= fig.bbox.width + 1
    else:
        assert [t.name for t in fig.data if t.showlegend] \
            == ['same', 'same']
        assert len([t for t in fig.data
                    if getattr(getattr(t, 'marker', None), 'showscale',
                               None)]) == 1


# --- ndims=1 series grids name their datasets as the single call does -----

def test_series_grid_legend_names_each_dataset():
    x = _three()
    single = _quiet(x, ndims=1, legend=True, show=False)
    grid = _quiet(x, ndims=1, legend=True, panels=(1, 3), show=False)
    assert _mpl_legends(single) == [['dataset 1', 'dataset 2', 'dataset 3']]
    assert _mpl_legends(grid) == _mpl_legends(single)
    (legend,) = grid.legends
    (single_legend,) = [ax.get_legend() for ax in single.axes
                        if ax.get_legend() is not None]
    assert _mpl_legend_colors(legend) == _mpl_legend_colors(single_legend)
