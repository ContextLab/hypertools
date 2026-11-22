"""Shared axis labels of a `panels=` grid are drawn on the outer panels
only (1.2.0, owner decision 2026-11-22), and a static grid and an animated
one are laid out by one path.

The rule, for the occupied panels of a grid (spare cells do not count):

- an axis label that runs along the BOTTOM of a panel (x in 2-D and in
  ``ndims=1`` series panels; x and y, the floor axes, in 3-D) is drawn on
  the lowest occupied panel of each column, when every occupied panel of
  that column has the same text for it;
- an axis label that runs up the SIDE of a panel (y in 2-D; z in 3-D) is
  drawn on one panel of each row, when every occupied panel of that row
  has the same text for it: the left-most one in 2-D, and in 3-D the
  panel at the end of the row on the side of the cube matplotlib draws
  the z label on at the grid's view (the right-most at the default view);
- where the texts of a column (or row) differ, every panel of it keeps
  its own.

Everything is measured on real draws of real figures (Agg, and kaleido-free
plotly figure objects); nothing is mocked or patched.
"""
import itertools

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mpl_toolkits.mplot3d import proj3d

import hypertools as hyp
from hypertools.plot.plot import (CAPTION_GID, PLOTLY_Z_TITLE_NAME,
                                  SHARED_COLORBAR_GID, SHARED_LEGEND_GID)


def _walk(seed, rows=24, cols=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


def _frames(n, cols=('alpha', 'beta', 'gamma')):
    """`n` DataFrames whose columns are named `cols` (as many columns)."""
    return [pd.DataFrame(_walk(seed, cols=len(cols)), columns=list(cols))
            for seed in range(n)]


ANIM = dict(animate=True, duration=1, frame_rate=4, show=False)
#: datasets whose column names differ share no features, so they are fitted
#: per panel (a shared fit refuses them: features are matched by name)
OWN = dict(panel_fit='independent')

#: grid shape -> (number of panels, the lowest occupied panel of each
#: column, the left-most panel of each row, the right-most occupied panel
#: of each row), all as row-major panel numbers
GRIDS = {
    (1, 4): (4, {0, 1, 2, 3}, {0}, {3}),
    (2, 2): (4, {2, 3}, {0, 2}, {1, 3}),
    (4, 1): (4, {3}, {0, 1, 2, 3}, {0, 1, 2, 3}),
    (2, 3): (5, {3, 4, 2}, {0, 3}, {2, 4}),
    (3, 5): (13, {10, 11, 12, 8, 9}, {0, 5, 10}, {4, 9, 12}),
}
GRID_IDS = ['1x4', '2x2', '4x1', '2x3-with-5', '3x5-with-13']


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _figure(result):
    """The matplotlib figure of a static or animated grid call."""
    return getattr(result, 'figure', result)


def _plot(data, animated, **kwargs):
    """A static grid's figure, or an animated grid's (showing frame 0)."""
    extra = dict(ANIM) if animated else dict(show=False)
    extra.update(kwargs)
    return _figure(hyp.plot(data, **extra))


def _drawn(ax):
    """``{'x': text, ...}`` of the axis labels `ax` actually draws."""
    out = {}
    for name in ('x', 'y', 'z'):
        axis = getattr(ax, name + 'axis', None)
        if axis is None:
            continue
        label = axis.label
        if label.get_text() and label.get_visible():
            out[name] = label.get_text()
    return out


def _carriers(fig, n, name):
    """The panels (row-major numbers) that draw the `name` axis label."""
    return {i for i, ax in enumerate(fig.axes[:n]) if name in _drawn(ax)}


def _draw(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _cube_box(ax):
    """The drawn box (pixels) of a 3-D axes' cube at its current view."""
    corners = np.array(list(itertools.product(
        ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))
    px, py, _ = proj3d.proj_transform(
        corners[:, 0], corners[:, 1], corners[:, 2], ax.get_proj())
    points = ax.transData.transform(np.column_stack([px, py]))
    return matplotlib.transforms.Bbox.from_extents(
        points[:, 0].min(), points[:, 1].min(),
        points[:, 0].max(), points[:, 1].max())


def _shrunk(box, pixels=2.0):
    """`box` less a couple of pixels of antialiasing on every side."""
    return matplotlib.transforms.Bbox.from_extents(
        box.x0 + pixels, box.y0 + pixels, box.x1 - pixels, box.y1 - pixels)


# --------------------------------------------------------------------------
# which panels carry which labels
# --------------------------------------------------------------------------

@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
@pytest.mark.parametrize('shape', list(GRIDS), ids=GRID_IDS)
def test_2d_shared_labels_are_drawn_on_the_outer_panels(shape, animated):
    n, lowest, leftmost, _ = GRIDS[shape]
    fig = _plot(_frames(n, ('alpha', 'beta')), animated, ndims=2,
                panels=shape)
    assert _carriers(fig, n, 'x') == lowest
    assert _carriers(fig, n, 'y') == leftmost
    for ax in fig.axes[:n]:
        assert set(_drawn(ax).values()) <= {'alpha', 'beta'}
        assert _drawn(ax).get('x', 'alpha') == 'alpha'
        assert _drawn(ax).get('y', 'beta') == 'beta'


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
@pytest.mark.parametrize('shape', list(GRIDS), ids=GRID_IDS)
def test_3d_shared_labels_are_drawn_on_the_outer_panels(shape, animated):
    n, lowest, _, rightmost = GRIDS[shape]
    fig = _plot(_frames(n), animated, panels=shape)
    # the floor axes: under the lowest panel of each column
    assert _carriers(fig, n, 'x') == lowest
    assert _carriers(fig, n, 'y') == lowest
    # the vertical axis: once per row, on the panel at the end of the row
    # the z label is drawn toward (the cube's right at the default view)
    assert _carriers(fig, n, 'z') == rightmost
    for ax in fig.axes[:n]:
        drawn = _drawn(ax)
        assert drawn == {key: {'x': 'alpha', 'y': 'beta',
                               'z': 'gamma'}[key] for key in drawn}


@pytest.mark.parametrize('azim', [-60, 30, 120, 210])
def test_the_z_label_goes_to_the_end_of_the_row_it_is_drawn_toward(azim):
    """matplotlib draws a cube's z label at its left or its right edge,
    depending on the view; the row's one z label is on the panel at that
    end of the row, so it sits at the grid's outer edge."""
    fig = hyp.plot(_frames(4), panels=(2, 2), azim=azim, show=False)
    carriers = _carriers(fig, 4, 'z')
    assert carriers in ({0, 2}, {1, 3})
    renderer = _draw(fig)
    for i in carriers:
        ax = fig.axes[i]
        label = ax.zaxis.label.get_window_extent(renderer)
        cube = _cube_box(ax)
        center = 0.5 * (cube.x0 + cube.x1)
        drawn_right = 0.5 * (label.x0 + label.x1) > center
        assert drawn_right == (i % 2 == 1), (azim, i)
    if azim == -60:
        assert carriers == {1, 3}


def test_static_and_animated_grids_carry_the_same_labels():
    for ndims, cols in ((2, ('alpha', 'beta')),
                        (3, ('alpha', 'beta', 'gamma'))):
        data = _frames(5, cols)
        static = _plot(data, False, ndims=ndims, panels=(2, 3))
        moving = _plot(data, True, ndims=ndims, panels=(2, 3))
        assert ([_drawn(ax) for ax in static.axes[:5]]
                == [_drawn(ax) for ax in moving.axes[:5]])


def test_an_animated_grid_keeps_its_labels_on_every_frame():
    anim = hyp.plot(_frames(4), panels=(2, 2), **ANIM)
    first = None
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        drawn = [_drawn(ax) for ax in anim.figure.axes[:4]]
        first = drawn if first is None else first
        assert drawn == first, frame
    assert first == [{}, {'z': 'gamma'}, {'x': 'alpha', 'y': 'beta'},
                     {'x': 'alpha', 'y': 'beta', 'z': 'gamma'}]


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
def test_labels_that_differ_in_a_column_or_row_are_all_kept(animated):
    """Column 0's x labels differ (both kept), column 1's agree (bottom
    only); row 0's y labels differ (both kept), row 1's agree (left
    only)."""
    names = [('a', 'p'), ('b', 'q'), ('A', 'r'), ('b', 'r')]
    data = [pd.DataFrame(_walk(i, cols=2), columns=list(cols))
            for i, cols in enumerate(names)]
    fig = _plot(data, animated, ndims=2, panels=(2, 2), **OWN)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {'x': 'a', 'y': 'p'}, {'y': 'q'},
        {'x': 'A', 'y': 'r'}, {'x': 'b'}]


def test_3d_labels_that_differ_are_all_kept():
    names = [('a', 'b', 'c'), ('a', 'b', 'C'),
             ('a', 'B', 'c'), ('a', 'b', 'c')]
    data = [pd.DataFrame(_walk(i), columns=list(cols))
            for i, cols in enumerate(names)]
    fig = hyp.plot(data, panels=(2, 2), show=False, **OWN)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        # x agrees in both columns; y differs in column 0; z differs in
        # row 0 and agrees in row 1
        {'y': 'b', 'z': 'c'}, {'z': 'C'},
        {'x': 'a', 'y': 'B'}, {'x': 'a', 'y': 'b', 'z': 'c'}]


def test_a_panel_without_a_label_makes_its_column_differ():
    """A column whose panels are not all labeled alike shares nothing:
    the labeled panels keep their labels, the unlabeled one has none."""
    data = [pd.DataFrame(_walk(0, cols=2), columns=['a', 'b']),
            _walk(1, cols=2),
            pd.DataFrame(_walk(2, cols=2), columns=['a', 'b']),
            pd.DataFrame(_walk(3, cols=2), columns=['a', 'b'])]
    fig = hyp.plot(data, ndims=2, panels=(2, 2), show=False)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        # column 0 agrees on x (bottom only); column 1 does not; row 0
        # does not agree on y; row 1 does (left only)
        {'y': 'b'}, {}, {'x': 'a', 'y': 'b'}, {'x': 'a'}]


def test_every_panel_is_labeled_when_every_panel_differs():
    """The existing way to label every panel: labels that are the
    panels' own (here each DataFrame's own column names)."""
    data = [pd.DataFrame(_walk(i, cols=2), columns=[f'x{i}', f'y{i}'])
            for i in range(4)]
    for animated in (False, True):
        fig = _plot(data, animated, ndims=2, panels=(2, 2), **OWN)
        assert [_drawn(ax) for ax in fig.axes[:4]] == [
            {'x': f'x{i}', 'y': f'y{i}'} for i in range(4)]


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
def test_explicit_labels_are_shared_labels(animated):
    arrays = [_walk(i) for i in range(4)]
    fig = _plot(arrays, animated, panels=(2, 2), xlabel='one',
                ylabel='two', zlabel='three')
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {}, {'z': 'three'}, {'x': 'one', 'y': 'two'},
        {'x': 'one', 'y': 'two', 'z': 'three'}]
    flat = _plot([a[:, :2] for a in arrays], animated, ndims=2,
                 panels=(2, 2), xlabel='one', ylabel='two')
    assert [_drawn(ax) for ax in flat.axes[:4]] == [
        {'y': 'two'}, {}, {'x': 'one', 'y': 'two'}, {'x': 'one'}]


def test_an_explicit_label_beside_derived_ones():
    """`xlabel=` is the whole grid's; the y labels are each panel's own
    column name and differ in row 0 only."""
    names = [('a', 'p'), ('b', 'q'), ('c', 'r'), ('d', 'r')]
    data = [pd.DataFrame(_walk(i, cols=2), columns=list(cols))
            for i, cols in enumerate(names)]
    fig = hyp.plot(data, ndims=2, panels=(2, 2), xlabel='time',
                   show=False, **OWN)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {'y': 'p'}, {'y': 'q'}, {'x': 'time', 'y': 'r'}, {'x': 'time'}]


def test_empty_labels_draw_nothing():
    fig = hyp.plot(_frames(4), panels=(2, 2), xlabel='', ylabel='',
                   zlabel='', show=False)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [{}, {}, {}, {}]


def test_one_label_dropped_the_others_shared():
    fig = hyp.plot(_frames(4), panels=(2, 2), ylabel='', show=False)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {}, {'z': 'gamma'}, {'x': 'alpha'}, {'x': 'alpha', 'z': 'gamma'}]


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
def test_series_panels_follow_the_2d_rule(animated):
    """``ndims=1``: the x label is the index's name and the y label the
    series' name, both derived by each panel's own call."""
    index = pd.Index(np.arange(24), name='trial')
    data = [pd.DataFrame({'signal': _walk(i, cols=1)[:, 0]}, index=index)
            for i in range(4)]
    fig = _plot(data, animated, ndims=1, panels=(2, 2))
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {'y': 'signal'}, {}, {'x': 'trial', 'y': 'signal'},
        {'x': 'trial'}]


def test_reducer_list_panels_share_explicit_labels():
    fig = hyp.plot([_walk(i, cols=6) for i in range(3)], ndims=2,
                   reduce=['PCA', 'IncrementalPCA', 'FastICA'],
                   panels=(1, 3), xlabel='first', ylabel='second',
                   show=False)
    assert [_drawn(ax) for ax in fig.axes[:3]] == [
        {'x': 'first', 'y': 'second'}, {'x': 'first'}, {'x': 'first'}]


def test_the_subplots_alias_follows_the_same_rule():
    fig = hyp.plot(_frames(4, ('alpha', 'beta')), ndims=2,
                   subplots=(2, 2), show=False)
    assert [_drawn(ax) for ax in fig.axes[:4]] == [
        {'y': 'beta'}, {}, {'x': 'alpha', 'y': 'beta'}, {'x': 'alpha'}]


def test_tick_labels_are_left_alone():
    """Tick labels are not part of the rule: with ``axis_scale='data'``
    every panel keeps its own ticks."""
    fig = hyp.plot(_frames(4, ('alpha', 'beta')), ndims=2, panels=(2, 2),
                   axis_scale='data', show=False)
    _draw(fig)
    for ax in fig.axes[:4]:
        assert [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert _carriers(fig, 4, 'x') == {2, 3}
    assert _carriers(fig, 4, 'y') == {0, 2}


# --------------------------------------------------------------------------
# the one decision function
# --------------------------------------------------------------------------

def test_the_decision_function_on_plain_texts():
    from hypertools.plot.plot import _outer_label_panels
    same = [{'x': 'a', 'y': 'b'}] * 5
    assert _outer_label_panels(same, 2, 3, ndims=2) == [
        {'y'}, set(), {'x'}, {'x', 'y'}, {'x'}]
    cube = [{'x': 'a', 'y': 'b', 'z': 'c'}] * 5
    assert _outer_label_panels(cube, 2, 3, ndims=3, z_side='right') == [
        set(), set(), {'x', 'y', 'z'}, {'x', 'y'}, {'x', 'y', 'z'}]
    assert _outer_label_panels(cube, 2, 3, ndims=3, z_side='left') == [
        {'z'}, set(), {'x', 'y'}, {'x', 'y', 'z'}, {'x', 'y'}]
    # None and '' are both "no label"; a column that mixes a label and
    # none differs
    mixed = [{'x': 'a', 'y': None}, {'x': '', 'y': 'b'},
             {'x': 'a', 'y': 'b'}, {'x': 'a', 'y': 'b'}]
    assert _outer_label_panels(mixed, 2, 2, ndims=2) == [
        set(), {'y'}, {'x', 'y'}, {'x'}]


# --------------------------------------------------------------------------
# the plotly grid
# --------------------------------------------------------------------------

def _plotly_titles(bundle, ndims):
    """Per panel, the axis titles the plotly grid draws."""
    out = []
    for axes in bundle['axes']:
        if ndims == 3:
            named = {'x': axes.xaxis, 'y': axes.yaxis, 'z': axes.zaxis}
        else:
            named = {'x': axes[0], 'y': axes[1]}
        out.append({name: axis.title.text for name, axis in named.items()
                    if axis.visible is not False and axis.title.text})
    # a 3-D grid's first-column z titles are written in the left margin
    for note in bundle['fig'].layout.annotations:
        for i in range(len(out)):
            if note.name == PLOTLY_Z_TITLE_NAME.format(i):
                assert 'z' not in out[i]
                out[i]['z'] = note.text
    return out


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (1, 4)],
                         ids=['2x2', '2x3-with-5', '1x4'])
def test_plotly_2d_grid_draws_shared_titles_on_the_outer_panels(shape):
    n, lowest, leftmost, _ = GRIDS[shape]
    bundle = hyp.plot(_frames(n, ('alpha', 'beta')), ndims=2, panels=shape,
                      backend='plotly', show=False, return_model=True)
    titles = _plotly_titles(bundle, 2)
    assert {i for i, t in enumerate(titles) if 'x' in t} == lowest
    assert {i for i, t in enumerate(titles) if 'y' in t} == leftmost
    assert all(set(t.values()) <= {'alpha', 'beta'} for t in titles)


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (1, 4)],
                         ids=['2x2', '2x3-with-5', '1x4'])
def test_plotly_3d_grid_draws_shared_titles_on_the_outer_panels(shape):
    from hypertools.plot.plot import _PLOTLY_Z_LABEL_SIDE
    n, lowest, leftmost, rightmost = GRIDS[shape]
    bundle = hyp.plot(_frames(n), panels=shape, backend='plotly',
                      show=False, return_model=True)
    titles = _plotly_titles(bundle, 3)
    assert {i for i, t in enumerate(titles) if 'x' in t} == lowest
    assert {i for i, t in enumerate(titles) if 'y' in t} == lowest
    # plotly writes a scene's z title LEFT of its cube: one per row, at
    # the row's left end
    assert _PLOTLY_Z_LABEL_SIDE == 'left'
    assert {i for i, t in enumerate(titles) if 'z' in t} == leftmost
    assert all(t['z'] == 'gamma' for t in titles if 'z' in t)


@pytest.mark.parametrize('size', [None, [8, 6]], ids=['default', 'sized'])
def test_plotly_3d_grid_z_titles_are_written_in_the_left_margin(size):
    """A scene's z title falls outside a grid cell's scene and is clipped
    away, so the first column's are written beside the grid, in room the
    default figure is widened by (an explicit `size=` is kept)."""
    plain = hyp.plot([_walk(i) for i in range(4)], panels=(2, 2),
                     backend='plotly', show=False, size=size)
    fig = hyp.plot(_frames(4), panels=(2, 2), backend='plotly',
                   show=False, size=size)
    notes = {note.name: note for note in fig.layout.annotations
             if note.name and note.name.startswith('hyp_panel_ztitle_')}
    assert sorted(notes) == ['hyp_panel_ztitle_0', 'hyp_panel_ztitle_2']
    assert fig.layout.margin.l >= 30
    if size is None:
        assert fig.layout.width == plain.layout.width + (
            fig.layout.margin.l - plain.layout.margin.l)
    else:
        assert (fig.layout.width, fig.layout.height) == (800, 600)
    plot_w = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
    for i, key in ((0, 'scene'), (2, 'scene3')):
        note = notes[f'hyp_panel_ztitle_{i}']
        domain = fig.layout[key].domain
        assert note.text == 'gamma' and note.textangle == -90
        # left of its scene, inside the canvas, level with its middle
        assert note.x == domain.x[0] and note.xshift < 0
        assert fig.layout.margin.l + note.x * plot_w + note.xshift > 5
        assert note.y == pytest.approx(0.5 * (domain.y[0] + domain.y[1]))
        assert not fig.layout[key].zaxis.title.text


def test_plotly_grid_keeps_labels_that_differ():
    names = [('a', 'p'), ('b', 'q'), ('A', 'r'), ('b', 'r')]
    data = [pd.DataFrame(_walk(i, cols=2), columns=list(cols))
            for i, cols in enumerate(names)]
    bundle = hyp.plot(data, ndims=2, panels=(2, 2), backend='plotly',
                      show=False, return_model=True, **OWN)
    assert _plotly_titles(bundle, 2) == [
        {'x': 'a', 'y': 'p'}, {'y': 'q'},
        {'x': 'A', 'y': 'r'}, {'x': 'b'}]


def test_plotly_and_matplotlib_grids_agree_in_2d():
    data = _frames(5, ('alpha', 'beta'))
    fig = hyp.plot(data, ndims=2, panels=(2, 3), show=False)
    bundle = hyp.plot(data, ndims=2, panels=(2, 3), backend='plotly',
                      show=False, return_model=True)
    assert _plotly_titles(bundle, 2) == [_drawn(ax) for ax in fig.axes[:5]]


# --------------------------------------------------------------------------
# where the kept labels are drawn
# --------------------------------------------------------------------------

def _assert_labels_in_their_cells(fig, n, shape, where=None):
    """Every drawn axis label is on the canvas, inside its own panel's
    grid cell, and clear of every other panel's drawing, title and
    labels, and of the shared legend, colorbar and caption."""
    renderer = _draw(fig)
    width, height = fig.canvas.get_width_height()
    panels = fig.axes[:n]
    nrows, ncols = shape
    furniture = [legend.get_window_extent(renderer)
                 for legend in fig.legends]
    furniture += [ax.get_tightbbox(renderer) for ax in fig.axes
                  if ax.get_gid() == SHARED_COLORBAR_GID]
    furniture += [text.get_window_extent(renderer) for text in fig.texts
                  if text.get_gid() == CAPTION_GID]
    labels = {}
    for i, ax in enumerate(panels):
        labels[i] = [
            getattr(ax, name + 'axis').label.get_window_extent(renderer)
            for name in _drawn(ax)]
    assert any(labels.values())
    # a panel's cell: half way to the next panels' axes on its inner
    # sides, and out to the canvas' edge on its outer ones (where the
    # shared legend, colorbar and caption are checked separately)
    boxes = [ax.get_window_extent(renderer) for ax in panels]
    for i, ax in enumerate(panels):
        row, col = divmod(i, ncols)
        left = 0.0 if col == 0 else 0.5 * (boxes[i - 1].x1 + boxes[i].x0)
        right = (width if col == ncols - 1 or i + 1 >= n
                 else 0.5 * (boxes[i].x1 + boxes[i + 1].x0))
        top = (height if row == 0
               else 0.5 * (boxes[i - ncols].y0 + boxes[i].y1))
        bottom = (0.0 if row == nrows - 1 or i + ncols >= n
                  else 0.5 * (boxes[i].y0 + boxes[i + ncols].y1))
        for label in labels[i]:
            tag = (where, i)
            assert label.x0 >= -0.5 and label.x1 <= width + 0.5, tag
            assert label.y0 >= -0.5 and label.y1 <= height + 0.5, tag
            assert label.x0 >= left - 1.0 and label.x1 <= right + 1.0, tag
            assert label.y0 >= bottom - 1.0 and label.y1 <= top + 1.0, tag
            label = _shrunk(label)
            for box in furniture:
                assert not label.overlaps(box), tag
            for j, other in enumerate(panels):
                if j == i:
                    continue
                theirs = (_cube_box(other) if hasattr(other, 'get_proj')
                          else boxes[j])
                assert not label.overlaps(theirs), (tag, j)
                if other.get_title():
                    assert not label.overlaps(
                        other.title.get_window_extent(renderer)), (tag, j)
                for one in labels[j]:
                    assert not label.overlaps(_shrunk(one)), (tag, j)


FURNITURE = [
    {}, dict(legend=True),
    dict(legend=True, caption='a caption under the whole grid',
         title=['one', 'two', 'three', 'four', 'five']),
    dict(hue='rows', colorbar=True, caption='a caption'),
]
FURNITURE_IDS = ['bare', 'legend', 'legend-caption-titles',
                 'colorbar-caption']


def _extras(extras, n, rows=24):
    """`extras` cut to `n` panels (titles; a per-row hue for a colorbar)."""
    out = dict(extras)
    if 'title' in out:
        out['title'] = out['title'][:n]
    if out.get('hue') == 'rows':
        out['hue'] = [np.arange(rows)] * n
    return out


@pytest.mark.parametrize('extras', FURNITURE, ids=FURNITURE_IDS)
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 3)],
                         ids=['1x4', '2x2', '2x3-with-5'])
def test_static_labels_stay_in_their_cells(shape, ndims, extras):
    n = GRIDS[shape][0]
    cols = ('first axis', 'second axis', 'third axis')[:ndims]
    fig = hyp.plot(_frames(n, cols), ndims=ndims, panels=shape, show=False,
                   **_extras(extras, n))
    _assert_labels_in_their_cells(fig, n, shape)


@pytest.mark.parametrize('extras', [FURNITURE[0], FURNITURE[2],
                                    FURNITURE[3]],
                         ids=['bare', 'legend-caption-titles',
                              'colorbar-caption'])
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 3)],
                         ids=['1x4', '2x2', '2x3-with-5'])
def test_animated_labels_stay_in_their_cells_through_a_full_rotation(
        shape, ndims, extras):
    n = GRIDS[shape][0]
    cols = ('first axis', 'second axis', 'third axis')[:ndims]
    turn = dict(rotations=1) if ndims == 3 else {}
    anim = hyp.plot(_frames(n, cols), ndims=ndims, panels=shape,
                    **dict(ANIM, frame_rate=12, **turn,
                           **_extras(extras, n)))
    assert anim.n_frames >= 12
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        _assert_labels_in_their_cells(anim.figure, n, shape, frame)


# --------------------------------------------------------------------------
# the room of labels that are not drawn is given back
# --------------------------------------------------------------------------

def _cube_share(ax):
    """A cube's drawn width as a share of its axes' width."""
    return _cube_box(ax).width / ax.get_window_extent().width


LONG = ('first axis', 'second axis', 'third axis')


@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 3)],
                         ids=['1x4', '2x2', '2x3-with-5'])
@pytest.mark.parametrize('mode', ['static', 'still', 'rotating'])
def test_the_cubes_of_a_grid_are_all_drawn_the_same_size(shape, mode):
    """The panels are read side by side: the panels that draw labels do
    not get a smaller cube than the ones that draw none."""
    n = GRIDS[shape][0]
    extra = {'static': {}, 'still': dict(ANIM, rotations=0),
             'rotating': dict(ANIM, rotations=1)}[mode]
    fig = _figure(hyp.plot(_frames(n, LONG), panels=shape,
                           **dict(dict(show=False), **extra)))
    _draw(fig)
    assert len({frozenset(_drawn(ax)) for ax in fig.axes[:n]}) > 1
    widths = [_cube_box(ax).width for ax in fig.axes[:n]]
    assert max(widths) - min(widths) <= 0.01 * max(widths), widths


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 3)],
                         ids=['1x4', '2x2', '2x3-with-5'])
def test_3d_cubes_get_the_room_of_the_undrawn_labels_back(shape, animated):
    """Seen from one view (a static grid, or ``rotations=0``), the outer
    panels' labels hang toward the grid's outer edge, where the figure
    gives them room, so no cube is drawn smaller for the z label and the
    cubes keep most of the size they have without any label -- they were
    all drawn at 65-70% of it when every panel drew every label. (What
    still costs a little is a floor label of an inner column, which
    reaches sideways toward the next panel.)"""
    n = GRIDS[shape][0]
    extra = dict(rotations=0) if animated else {}
    plain = _plot([_walk(i) for i in range(n)], animated, panels=shape,
                  **extra)
    labeled = _plot(_frames(n), animated, panels=shape, **extra)
    short = _plot(_frames(n, ('x', 'y', 'z')), animated, panels=shape,
                  **extra)
    _draw(plain), _draw(labeled), _draw(short)
    for a, b, c in zip(plain.axes[:n], labeled.axes[:n], short.axes[:n]):
        assert _cube_share(b) >= 0.9 * _cube_share(a)
        assert _cube_share(c) >= 0.99 * _cube_share(a)
    # the room is added at the outer edges; the panels keep their size
    assert np.all(labeled.get_size_inches() >= plain.get_size_inches())
    assert np.all(labeled.get_size_inches()
                  <= plain.get_size_inches() + [1.0, 0.6])
    for a, b in zip(plain.axes[:n], labeled.axes[:n]):
        size_a = a.get_window_extent().size
        size_b = b.get_window_extent().size
        assert np.allclose(size_a, size_b, rtol=0.02), (size_a, size_b)


def test_labels_kept_on_inner_panels_are_fitted_inside_their_cells():
    """Labels that differ are drawn on inner panels too; there they have
    no outer edge to hang toward, so the grid's cubes are drawn smaller
    (all of them alike) until those labels fit their own cells."""
    names = ('alpha', 'beta', 'gamma')
    data = [pd.DataFrame(_walk(i), columns=[f'{c} {i}' for c in names])
            for i in range(4)]
    own = hyp.plot(data, panels=(2, 2), show=False, **OWN)
    shared = hyp.plot(_frames(4, names), panels=(2, 2), show=False)
    _draw(own), _draw(shared)
    assert all(len(_drawn(ax)) == 3 for ax in own.axes[:4])
    assert _cube_share(own.axes[0]) < 0.9 * _cube_share(shared.axes[0])
    _assert_labels_in_their_cells(own, 4, (2, 2))


def test_3d_figure_enlargement_reads_the_labels_actually_drawn():
    """The default figure grows for the labels that are drawn, where
    they are drawn: a grid with shared labels is the unlabeled grid plus
    the outer labels' room at its edges, with larger cubes than the same
    grid with every panel's own (different) labels on every panel."""
    cols = ('first axis', 'second axis', 'third axis')
    shared = hyp.plot(_frames(4, cols), panels=(2, 2), show=False, **OWN)
    own = hyp.plot(
        [pd.DataFrame(_walk(i), columns=[f'{c} {i}' for c in cols])
         for i in range(4)], panels=(2, 2), show=False, **OWN)
    plain = hyp.plot([_walk(i) for i in range(4)], panels=(2, 2),
                     show=False)
    assert tuple(plain.get_size_inches()) == (6.4, 6.4)
    size = shared.get_size_inches()
    assert 6.4 <= size[0] <= 6.4 + 1.0 and 6.4 <= size[1] <= 6.4 + 0.6
    _draw(shared), _draw(own)
    for a, b in zip(shared.axes[:4], own.axes[:4]):
        assert _cube_box(a).width > 1.05 * _cube_box(b).width


def test_a_small_explicit_size_warns_only_for_labels_that_are_drawn():
    """A static 3-D grid at an explicit size too small for labels on every
    panel: with shared labels the warning is decided by the panels that
    draw them, and a grid whose labels are all dropped never warns."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        hyp.plot(_frames(4), panels=(2, 2), size=[4, 4], xlabel='',
                 ylabel='', zlabel='', show=False)
    with pytest.warns(UserWarning, match='larger size='):
        hyp.plot(_frames(4), panels=(1, 4), size=[3, 1], show=False)


def test_2d_inner_panels_get_their_label_room_back():
    """With the x labels on the bottom row only, a 2 x 2 grid's panels
    are taller than the same grid's with a label under every panel."""
    shared = hyp.plot(_frames(4, ('alpha', 'beta')), ndims=2,
                      panels=(2, 2), show=False, **OWN)
    own = hyp.plot(
        [pd.DataFrame(_walk(i, cols=2), columns=[f'alpha{i}', f'beta{i}'])
         for i in range(4)], ndims=2, panels=(2, 2), show=False, **OWN)
    assert tuple(shared.get_size_inches()) == tuple(own.get_size_inches())
    for a, b in zip(shared.axes[:4], own.axes[:4]):
        assert a.get_position().height > b.get_position().height + 0.005
        assert a.get_position().width > b.get_position().width + 0.005


# --------------------------------------------------------------------------
# one layout path: static == animated with rotations=0
# --------------------------------------------------------------------------

LAYOUTS = {
    'bare': {},
    'titles': dict(title=['one', 'two', 'three', 'four', 'five']),
    'legend': dict(legend=True),
    'colorbar': dict(hue='rows', colorbar=True),
    'caption': dict(caption='a caption under the whole grid'),
    'everything': dict(legend=True, caption='a caption',
                       title=['one', 'two', 'three', 'four', 'five']),
}


def _layout(fig, n):
    """A grid's figure size and its panels' axes positions."""
    return (np.array(fig.get_size_inches()),
            np.array([ax.get_position().bounds for ax in fig.axes[:n]]))


@pytest.mark.parametrize('labels', [False, True],
                         ids=['unlabeled', 'labeled'])
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (2, 3)],
                         ids=['1x4', '2x2', '2x3-with-5'])
def test_static_layout_is_the_animated_layout_without_rotation(
        shape, layout, ndims, labels):
    """A static grid and frame 0 of the same grid animated with
    ``rotations=0`` have the same figure size, the same axes positions,
    the same labels, the same shared legend/colorbar/caption boxes and (in
    3-D) the same cubes."""
    n = GRIDS[shape][0]
    cols = ('first axis', 'second axis', 'third axis')[:ndims]
    data = (_frames(n, cols) if labels
            else [_walk(i, cols=ndims) for i in range(n)])
    extras = _extras(LAYOUTS[layout], n)
    static = hyp.plot(data, ndims=ndims, panels=shape, show=False,
                      **extras)
    # (`rotations=` is the 3-D camera's; a 2-D grid has one view anyway)
    still = dict(rotations=0) if ndims == 3 else {}
    anim = hyp.plot(data, ndims=ndims, panels=shape,
                    **dict(ANIM, **extras, **still))
    anim.draw_frame(0)
    moving = anim.figure
    size_a, boxes_a = _layout(static, n)
    size_b, boxes_b = _layout(moving, n)
    assert np.allclose(size_a, size_b, atol=1e-6), (size_a, size_b)
    assert np.allclose(boxes_a, boxes_b, atol=1e-6)
    assert ([_drawn(ax) for ax in static.axes[:n]]
            == [_drawn(ax) for ax in moving.axes[:n]])
    ra, rb = _draw(static), _draw(moving)

    def _furniture(fig, renderer):
        boxes = [legend.get_window_extent(renderer).bounds
                 for legend in fig.legends
                 if legend.get_gid() == SHARED_LEGEND_GID]
        boxes += [ax.get_position().bounds for ax in fig.axes
                  if ax.get_gid() == SHARED_COLORBAR_GID]
        boxes += [text.get_window_extent(renderer).bounds
                  for text in fig.texts if text.get_gid() == CAPTION_GID]
        return np.array(boxes)

    fa, fb = _furniture(static, ra), _furniture(moving, rb)
    assert fa.shape == fb.shape
    assert np.allclose(fa, fb, atol=0.5)
    if ndims == 3:
        for a, b in zip(static.axes[:n], moving.axes[:n]):
            assert np.allclose(_cube_box(a).bounds, _cube_box(b).bounds,
                               atol=0.5)
            for name in _drawn(a):
                assert np.allclose(
                    getattr(a, name + 'axis').label
                    .get_window_extent(ra).bounds,
                    getattr(b, name + 'axis').label
                    .get_window_extent(rb).bounds, atol=0.5)


@pytest.mark.parametrize('layout', ['bare', 'everything'])
@pytest.mark.parametrize('labels', [False, True],
                         ids=['unlabeled', 'labeled'])
def test_a_2d_grid_is_laid_out_the_same_static_or_rotating(layout, labels):
    """2-D panels have no camera, so `rotations=` changes nothing."""
    data = (_frames(4, ('alpha', 'beta')) if labels
            else [_walk(i, cols=2) for i in range(4)])
    extras = _extras(LAYOUTS[layout], 4)
    static = hyp.plot(data, ndims=2, panels=(2, 2), show=False, **extras)
    with pytest.warns(UserWarning, match='no effect on 2-D animations'):
        moving = hyp.plot(data, ndims=2, panels=(2, 2), rotations=1,
                          **dict(ANIM, **extras)).figure
    size_a, boxes_a = _layout(static, 4)
    size_b, boxes_b = _layout(moving, 4)
    assert np.allclose(size_a, size_b, atol=1e-6)
    assert np.allclose(boxes_a, boxes_b, atol=1e-6)


@pytest.mark.parametrize('layout', ['bare', 'everything'])
def test_rotation_changes_only_the_cubes_of_an_unlabeled_3d_grid(layout):
    """With rotation, an unlabeled 3-D grid keeps the static grid's
    figure size and axes positions; only each cube is drawn smaller, by
    the clearance its swept extent needs."""
    data = [_walk(i) for i in range(4)]
    extras = _extras(LAYOUTS[layout], 4)
    static = hyp.plot(data, panels=(2, 2), show=False, **extras)
    anim = hyp.plot(data, panels=(2, 2), rotations=1,
                    **dict(ANIM, **extras))
    anim.draw_frame(0)
    size_a, boxes_a = _layout(static, 4)
    size_b, boxes_b = _layout(anim.figure, 4)
    assert np.allclose(size_a, size_b, atol=1e-6)
    assert np.allclose(boxes_a, boxes_b, atol=1e-6)
    _draw(static), _draw(anim.figure)
    for a, b in zip(static.axes[:4], anim.figure.axes[:4]):
        assert _cube_box(b).width <= _cube_box(a).width + 0.5
        assert _cube_box(b).width >= 0.8 * _cube_box(a).width
