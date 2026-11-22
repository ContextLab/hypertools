"""A static `panels=` grid's default figure is sized BY THE GRID (1.2.0).

Owner decision 2026-11-22: with no `size=`, a static `panels=`/`subplots=`
grid gets the figure an animated one already had -- one 3.2 in square cell
per panel, never narrower than matplotlib's 6.4 in -- instead of the fixed
6.4 x 4.8 in figure that left a 1 x 4 row of cubes small between two white
bands. An explicit `size=` is honored exactly, `hyp.subplots()` called
directly keeps its own default, and the plotly grid follows the same rule
at its 100 px per inch.

Real figures, really drawn (Agg) or really built (plotly): every assertion
reads sizes, artists and pixels back off the returned figure. Nothing is
mocked.
"""
import itertools
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot import plot as plot_module

RNG = np.random.default_rng(295)
DATA = [RNG.normal(size=(30, 3)).cumsum(axis=0) for _ in range(6)]
FRAMES = [pd.DataFrame(d, columns=['first axis', 'second axis',
                                   'third axis']) for d in DATA]
ANIM = dict(animate='spin', duration=1, frame_rate=4, show=False)

#: grid shape -> the default figure of a bare grid (inches)
SHAPES = [((1, 2), (6.4, 3.2)), ((1, 3), (9.6, 3.2)), ((1, 4), (12.8, 3.2)),
          ((2, 2), (6.4, 6.4)), ((2, 3), (9.6, 6.4)), ((3, 1), (6.4, 9.6)),
          ((4, 1), (6.4, 12.8))]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _size(fig):
    return tuple(float(v) for v in fig.get_size_inches())


def _draw(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


# --------------------------------------------------------------------------
# matplotlib: the default size
# --------------------------------------------------------------------------

def test_the_cell_constant_is_not_named_for_animation():
    assert plot_module._PANEL_CELL_INCHES == 3.2
    assert not hasattr(plot_module, '_ANIMATED_PANEL_CELL_INCHES')


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape, expected', SHAPES)
def test_a_bare_static_grid_is_sized_by_its_grid(shape, expected, ndims):
    n = shape[0] * shape[1]
    fig = hyp.plot(DATA[:n], panels=shape, ndims=ndims, show=False)
    assert _size(fig) == pytest.approx(expected)


def test_a_grid_with_an_empty_cell_is_sized_by_the_grid_not_the_panels():
    fig = hyp.plot(DATA[:5], panels=(2, 3), show=False)
    assert _size(fig) == pytest.approx((9.6, 6.4))


def test_subplots_alias_is_sized_the_same_way():
    fig = hyp.plot(DATA[:4], subplots=(1, 4), show=False)
    assert _size(fig) == pytest.approx((12.8, 3.2))


@pytest.mark.parametrize('extras', [
    {}, dict(title=['a', 'b', 'c', 'd']), dict(legend=True),
    dict(legend=['one', 'two', 'three', 'four'], title=['a', 'b', 'c', 'd']),
    dict(hue=[np.arange(30)] * 4, colorbar=True),
], ids=['bare', 'titles', 'legend', 'legend-titles', 'colorbar'])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (4, 1)])
def test_static_and_animated_default_sizes_are_equal(shape, extras):
    """Nothing here is animation-specific (no axis labels for a rotating
    cube to swing, no caption), so the two figures are the same size."""
    static = hyp.plot(DATA[:4], panels=shape, show=False, **extras)
    animated = hyp.plot(DATA[:4], panels=shape, **dict(ANIM, **extras))
    assert _size(static) == pytest.approx(_size(animated.figure))


@pytest.mark.parametrize('shape', [(1, 4), (2, 2)])
def test_static_and_animated_2d_default_sizes_are_equal(shape):
    static = hyp.plot(DATA[:4], panels=shape, ndims=2, show=False)
    animated = hyp.plot(DATA[:4], panels=shape, ndims=2, animate=True,
                        duration=1, frame_rate=4, show=False)
    assert _size(static) == pytest.approx(_size(animated.figure))


def test_a_shared_legend_widens_the_grid_sized_figure_by_its_strip():
    bare = hyp.plot(DATA[:4], panels=(1, 4), show=False)
    legend = hyp.plot(DATA[:4], panels=(1, 4), legend=True, show=False)
    assert _size(legend)[0] > _size(bare)[0]
    assert _size(legend)[1] == pytest.approx(_size(bare)[1])


@pytest.mark.parametrize('extras', [
    {}, dict(legend=True), dict(title=['a', 'b', 'c', 'd']),
    dict(legend=True, title=['a', 'b', 'c', 'd']),
    dict(hue=[np.arange(30)] * 4, colorbar=True),
], ids=['bare', 'legend', 'titles', 'legend-titles', 'colorbar'])
@pytest.mark.parametrize('size', [(6.4, 4.8), (8, 3), (5, 9), (12.8, 3.2)])
@pytest.mark.parametrize('ndims', [2, 3])
def test_an_explicit_size_is_honored_exactly(size, extras, ndims):
    fig = hyp.plot(DATA[:4], panels=(1, 4), ndims=ndims, size=size,
                   show=False, **extras)
    assert _size(fig) == pytest.approx(size)


@pytest.mark.parametrize('size', [(6.4, 4.8), (8, 3), (12.8, 3.2)])
@pytest.mark.parametrize('ndims', [2, 3])
def test_a_caption_only_adds_its_own_room_to_an_explicit_size(size, ndims):
    """`caption=` makes any figure taller by the caption's height where
    the canvas has no room for it (its documented rule, unchanged): the
    width given is kept, and the height grows by one line at most."""
    fig = hyp.plot(DATA[:4], panels=(1, 4), ndims=ndims, size=size,
                   caption='a caption', show=False)
    width, height = _size(fig)
    assert width == pytest.approx(size[0])
    assert size[1] - 1e-9 <= height <= size[1] + 0.3


# --------------------------------------------------------------------------
# panels=True/'auto': the shape comes from the aspect, then the size from
# the shape
# --------------------------------------------------------------------------

@pytest.mark.parametrize('n, shape, expected', [
    (2, (1, 2), (6.4, 3.2)), (3, (1, 3), (9.6, 3.2)),
    (4, (2, 2), (6.4, 6.4)), (5, (2, 3), (9.6, 6.4)),
    (6, (2, 3), (9.6, 6.4))])
@pytest.mark.parametrize('panels', [True, 'auto'])
def test_auto_grid_shape_is_chosen_first_then_sized(n, shape, expected,
                                                    panels):
    """The shape is the one the default 6.4 x 4.8 in ASPECT picks (as
    documented and as before); the default figure is then sized by it."""
    result = hyp.plot(DATA[:n], panels=panels, show=False,
                      return_model=True)
    assert tuple(result['panels']) == shape
    assert _size(result['fig']) == pytest.approx(expected)


def test_auto_grid_with_an_explicit_size_uses_that_aspect_and_that_size():
    result = hyp.plot(DATA[:4], panels=True, size=(12, 3), show=False,
                      return_model=True)
    assert tuple(result['panels']) == (1, 4)
    assert _size(result['fig']) == pytest.approx((12, 3))


# --------------------------------------------------------------------------
# what the sizing is for
# --------------------------------------------------------------------------

def test_a_row_of_cubes_fills_most_of_the_canvas_height():
    """A 1 x 4 row of cubes spanned under a fifth of the 6.4 x 4.8 in
    figure's height, between two white bands; in a 3.2 in tall figure it
    spans what matplotlib draws a cube at in a square 3-D axes."""
    fig = hyp.plot(DATA[:4], panels=(1, 4), show=False)
    renderer = _draw(fig)
    ink = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2) < 128
    rows = np.flatnonzero(ink.any(axis=1))
    assert (rows[-1] - rows[0]) / ink.shape[0] > 0.5
    # ...and each panel's square is its whole cell: about 3.2 in a side
    for ax in fig.axes[:4]:
        box = ax.get_window_extent(renderer)
        assert min(box.width, box.height) / fig.dpi > 2.4


@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (3, 1)])
def test_axis_labels_enlarge_a_static_grid_little(shape):
    """The grid-sized cells hold a labeled cube almost as they are: the
    figure grows by a tenth at most (the old fixed figure grew a 1 x 4
    row from 6.4 to more than 10 in wide), and the cube keeps a real
    share of its cell."""
    n = shape[0] * shape[1]
    plain = hyp.plot(DATA[:n], panels=shape, show=False)
    labeled = hyp.plot(FRAMES[:n], panels=shape, show=False)
    assert _size(labeled)[0] <= 1.1 * _size(plain)[0]
    assert _size(labeled)[1] <= 1.1 * _size(plain)[1]
    assert _size(labeled)[0] >= _size(plain)[0]
    assert _size(labeled)[1] >= _size(plain)[1]


def test_a_default_size_labeled_grid_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        hyp.plot(FRAMES[:4], panels=(1, 4), show=False)


def _furniture_boxes(ax, renderer):
    """The drawn boxes of everything a 2-D panel hangs outside its axes:
    title, axis labels and tick labels."""
    texts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    texts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    return [t.get_window_extent(renderer) for t in texts
            if t.get_visible() and t.get_text()]


def _overlap(a, b, slack=1.0):
    return (min(a.x1, b.x1) - max(a.x0, b.x0) > slack
            and min(a.y1, b.y1) - max(a.y0, b.y0) > slack)


@pytest.mark.parametrize('legend', [False, True], ids=['plain', 'legend'])
@pytest.mark.parametrize('shape, n', [((1, 4), 4), ((2, 2), 4), ((4, 1), 4),
                                      ((2, 3), 6), ((2, 3), 5)])
def test_2d_grid_furniture_does_not_collide(shape, n, legend):
    """Titles, axis labels (from DataFrame columns) and tick labels of a
    2-D grid at its default size: none is drawn over another panel's axes
    or furniture, over the shared legend, or off the canvas."""
    frames = [f[['first axis', 'second axis']] for f in FRAMES[:n]]
    fig = hyp.plot(frames, panels=shape, ndims=2, reduce=None,
                   title=[f'panel {i}' for i in range(n)],
                   legend=True if legend else None, show=False)
    renderer = _draw(fig)
    panels = [ax for ax in fig.axes[:n] if ax.get_visible()]
    assert len(panels) == n
    # CHANGED for 1.2.0 (owner decision 2026-11-22): shared axis labels are
    # drawn on the outer panels only -- the x label under the lowest panel
    # of each column, the y label on the left-most panel of each row (this
    # asserted both labels on every panel)
    nrows, ncols = shape
    for i, ax in enumerate(panels):
        assert ax.get_xlabel() == ('first axis' if i + ncols >= n else '')
        assert ax.get_ylabel() == ('second axis' if i % ncols == 0 else '')
    width, height = fig.canvas.get_width_height()
    furniture = [_furniture_boxes(ax, renderer) for ax in panels]
    for i, boxes in enumerate(furniture):
        for box in boxes:
            assert box.x0 >= -1 and box.y0 >= -1, i
            assert box.x1 <= width + 1 and box.y1 <= height + 1, i
            for legend_artist in fig.legends:
                assert not _overlap(
                    box, legend_artist.get_window_extent(renderer)), i
        # a panel's own furniture: no two pieces on each other
        for one, other in itertools.combinations(boxes, 2):
            assert not _overlap(one, other), i
    for (i, a), (j, b) in itertools.combinations(enumerate(panels), 2):
        for box in furniture[i]:
            assert not _overlap(box, b.get_window_extent(renderer)), (i, j)
            for theirs in furniture[j]:
                assert not _overlap(box, theirs), (i, j)
        for box in furniture[j]:
            assert not _overlap(box, a.get_window_extent(renderer)), (i, j)
    # square-ish, roomy cells: no panel's axes is squeezed thin
    for ax in panels:
        box = ax.get_window_extent(renderer)
        assert box.width / fig.dpi > 1.6 and box.height / fig.dpi > 1.6


# --------------------------------------------------------------------------
# hyp.subplots() called directly keeps its own default
# --------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (4, 1)])
def test_hyp_subplots_keeps_matplotlibs_default_size(shape):
    fig, axes = hyp.subplots(*shape, backend='matplotlib')
    assert _size(fig) == pytest.approx((6.4, 4.8))
    for ax, data in zip(axes, DATA):
        hyp.plot(data, ax=ax, show=False)
    assert _size(fig) == pytest.approx((6.4, 4.8))


@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (4, 1)])
def test_hyp_subplots_plotly_keeps_its_default_size(shape):
    pytest.importorskip('plotly')
    fig, _ = hyp.subplots(*shape, backend='plotly')
    assert (fig.layout.width, fig.layout.height) == (640, 480)


# --------------------------------------------------------------------------
# plotly: the same rule at 100 px per inch
# --------------------------------------------------------------------------

@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape, expected', SHAPES)
def test_a_bare_plotly_grid_is_sized_by_its_grid(shape, expected, ndims):
    pytest.importorskip('plotly')
    n = shape[0] * shape[1]
    fig = hyp.plot(DATA[:n], panels=shape, ndims=ndims, backend='plotly',
                   show=False)
    assert (fig.layout.width, fig.layout.height) == (
        round(expected[0] * 100), round(expected[1] * 100))


def test_a_plotly_grid_is_as_large_as_the_matplotlib_one():
    pytest.importorskip('plotly')
    for extras in ({}, dict(title=['a', 'b', 'c', 'd'])):
        mpl = hyp.plot(DATA[:4], panels=(1, 4), show=False, **extras)
        ply = hyp.plot(DATA[:4], panels=(1, 4), backend='plotly',
                       show=False, **extras)
        assert (ply.layout.width, ply.layout.height) == tuple(
            round(100 * v) for v in _size(mpl))


def test_a_plotly_shared_legend_widens_the_grid_sized_figure():
    pytest.importorskip('plotly')
    from hypertools.plot.plotly_backend import (PANEL_GUTTER_PAD_PX,
                                                PANEL_LEGEND_PX)
    fig = hyp.plot(DATA[:4], panels=(1, 4), legend=True, backend='plotly',
                   show=False)
    assert fig.layout.width == 1280 + PANEL_LEGEND_PX + PANEL_GUTTER_PAD_PX
    assert fig.layout.height == 320


def test_plotly_3d_cells_are_square_and_fill_the_row():
    """Each scene of a 1 x 4 row is a square nearly the figure's height
    (it was a 147 px square in a 480 px tall figure)."""
    pytest.importorskip('plotly')
    fig = hyp.plot(DATA[:4], panels=(1, 4), backend='plotly', show=False)
    width, height = fig.layout.width, fig.layout.height
    plot_w = width - fig.layout.margin.l - fig.layout.margin.r
    plot_h = height - fig.layout.margin.t - fig.layout.margin.b
    for key in ('scene', 'scene2', 'scene3', 'scene4'):
        domain = fig.layout[key].domain
        side_w = (domain.x[1] - domain.x[0]) * plot_w
        side_h = (domain.y[1] - domain.y[0]) * plot_h
        assert side_w == pytest.approx(side_h, abs=1.5)
        assert side_h > 0.85 * height


@pytest.mark.parametrize('size', [(6.4, 4.8), (8, 3), (5, 9)])
@pytest.mark.parametrize('extras', [{}, dict(legend=True)],
                         ids=['bare', 'legend'])
def test_plotly_explicit_size_is_honored_exactly(size, extras):
    pytest.importorskip('plotly')
    fig = hyp.plot(DATA[:4], panels=(1, 4), size=size, backend='plotly',
                   show=False, **extras)
    assert (fig.layout.width, fig.layout.height) == (
        int(size[0] * 100), int(size[1] * 100))
