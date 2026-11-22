"""Layout of `panels=` grids, review round 1 (GH #295).

Axis labels of 3-D panels (static and animated) stay inside their own
panel; an animated labeled grid is enlarged no more than its labels need,
accepts the size hypertools picks itself, and names a size that works when
it refuses one; and the grid's one shared legend goes where
`legend_kwargs` asks, with room reserved only where it is drawn.

Everything is measured on real draws of real figures (Agg), frame by
frame for animations; nothing is mocked or patched.
"""
import itertools
import re

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mpl_toolkits.mplot3d import proj3d

import hypertools as hyp
from hypertools.plot.plot import (CAPTION_GID, SHARED_COLORBAR_GID,
                                  SHARED_LEGEND_GID)


def _walk(seed, rows=30, cols=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


DATA = [_walk(seed) for seed in range(4)]
XYZ = [pd.DataFrame(d, columns=list('xyz')) for d in DATA]
LONG = [pd.DataFrame(d, columns=['first axis', 'second axis',
                                 'third axis']) for d in DATA]
ANIM = dict(animate=True, duration=1, frame_rate=6, show=False)
GRIDS = [((1, 4), 4), ((2, 2), 4), ((3, 1), 3)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


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


def _assert_labels_clear(fig, n, frame=None):
    """Every panel's axis labels are on the canvas and clear of every
    OTHER panel's cube, axis labels and title, and of the shared legend,
    colorbar and caption."""
    renderer = _draw(fig)
    width, height = fig.canvas.get_width_height()
    panels = fig.axes[:n]
    furniture = [legend.get_window_extent(renderer)
                 for legend in fig.legends]
    furniture += [ax.get_tightbbox(renderer) for ax in fig.axes
                  if ax.get_gid() == SHARED_COLORBAR_GID]
    furniture += [text.get_window_extent(renderer) for text in fig.texts
                  if text.get_gid() == CAPTION_GID]
    labels = {}
    for i, ax in enumerate(panels):
        labels[i] = [axis.label.get_window_extent(renderer)
                     for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
                     if axis.label.get_text()]
        assert labels[i], i
    for i, ax in enumerate(panels):
        for label in labels[i]:
            where = (frame, i)
            assert label.x0 >= -0.5 and label.x1 <= width + 0.5, where
            assert label.y0 >= -0.5 and label.y1 <= height + 0.5, where
            label = _shrunk(label)
            for box in furniture:
                assert not label.overlaps(box), where
            for j, other in enumerate(panels):
                if j == i:
                    continue
                assert not label.overlaps(_cube_box(other)), (where, j)
                if other.get_title():
                    assert not label.overlaps(
                        other.title.get_window_extent(renderer)), (where, j)
                for theirs in labels[j]:
                    assert not label.overlaps(_shrunk(theirs)), (where, j)


# --------------------------------------------------------------------------
# A1 / T5: static 3-D grids
# --------------------------------------------------------------------------

@pytest.mark.parametrize('extras', [
    {}, dict(legend=True), dict(title=['a', 'b', 'c', 'd']),
    dict(legend=True, caption='a caption', title=['a', 'b', 'c', 'd']),
    dict(hue=[np.arange(30)] * 4, colorbar=True),
], ids=['bare', 'legend', 'titles', 'legend-caption-titles', 'colorbar'])
@pytest.mark.parametrize('frames', [XYZ, LONG], ids=['short', 'long'])
@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_static_3d_grid_axis_labels_stay_in_their_own_panel(shape, n,
                                                            frames, extras):
    extras = {key: (value[:n] if isinstance(value, list) else value)
              for key, value in extras.items()}
    fig = hyp.plot(frames[:n], panels=shape, show=False, **extras)
    _assert_labels_clear(fig, n)


def test_static_3d_grid_z_label_is_clear_of_the_legend():
    """T5: the last column's z-axis label was drawn over the legend."""
    fig = hyp.plot(XYZ, panels=(1, 4), legend=True, show=False)
    renderer = _draw(fig)
    legend = fig.legends[0].get_window_extent(renderer)
    for ax in fig.axes[:4]:
        assert not ax.zaxis.label.get_window_extent(renderer).overlaps(
            legend)


def test_static_grid_with_an_explicit_size_keeps_that_size():
    fig = hyp.plot(LONG, panels=(1, 4), size=[8, 3], show=False)
    assert tuple(fig.get_size_inches()) == (8.0, 3.0)
    _assert_labels_clear(fig, 4)


def test_static_grid_without_axis_labels_is_not_resized_or_rezoomed():
    plain = hyp.plot(DATA, panels=(2, 2), show=False)
    unlabeled = hyp.plot(XYZ, panels=(2, 2), xlabel='', ylabel='',
                         zlabel='', show=False)
    assert tuple(plain.get_size_inches()) == (6.4, 4.8)
    assert tuple(unlabeled.get_size_inches()) == (6.4, 4.8)
    for a, b in zip(plain.axes, unlabeled.axes):
        assert np.allclose(a.get_box_aspect(), b.get_box_aspect())


def test_static_labeled_cubes_keep_a_share_of_their_cells():
    fig = hyp.plot(LONG, panels=(1, 4), show=False)
    _draw(fig)
    for ax in fig.axes[:4]:
        cube, cell = _cube_box(ax), ax.get_window_extent()
        assert cube.width >= 0.3 * cell.width


# --------------------------------------------------------------------------
# A1 / A3: animated 3-D grids
# --------------------------------------------------------------------------

@pytest.mark.parametrize('extras', [
    {}, dict(legend=True, caption='a caption'),
    dict(title=lambda ctx: f'frame {ctx.frame}'),
], ids=['bare', 'legend-caption', 'titles'])
@pytest.mark.parametrize('frames', [XYZ, LONG], ids=['short', 'long'])
@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_animated_3d_grid_axis_labels_stay_in_their_own_panel(
        shape, n, frames, extras):
    anim = hyp.plot(frames[:n], panels=shape, **dict(ANIM, **extras))
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        _assert_labels_clear(anim.figure, n, frame)


@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_axis_labels_enlarge_an_animated_grid_no_more_than_they_need(
        shape, n):
    """A3: a 3 x 1 grid with axis labels became 6.4 x 13 in, mostly
    blank, with cubes a small part of their cells."""
    plain = hyp.plot(DATA[:n], panels=shape, **ANIM)
    labeled = hyp.plot(XYZ[:n], panels=shape, **ANIM)
    plain_size = plain.figure.get_size_inches()
    size = labeled.figure.get_size_inches()
    assert size[0] <= 1.1 * plain_size[0]
    assert size[1] <= 1.1 * plain_size[1]
    # the cube keeps a real share of its cell at every view
    fig = labeled.figure
    for frame in range(labeled.n_frames):
        labeled.draw_frame(frame)
        _draw(fig)
        for ax in fig.axes[:n]:
            cube, cell = _cube_box(ax), ax.get_window_extent()
            assert cube.width >= 0.38 * cell.width, frame
            assert cube.width <= cell.width and cube.height <= cell.height


# --------------------------------------------------------------------------
# T2: explicit size= on an animated labeled grid
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frames', [XYZ, LONG], ids=['short', 'long'])
@pytest.mark.parametrize('extras', [{}, dict(legend=True, caption='c')],
                         ids=['bare', 'legend-caption'])
@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_the_size_hypertools_picks_is_accepted_as_size(shape, n, extras,
                                                       frames):
    picked = hyp.plot(frames[:n], panels=shape, **dict(ANIM, **extras))
    size = [float(v) for v in picked.figure.get_size_inches()]
    again = hyp.plot(frames[:n], panels=shape, size=size,
                     **dict(ANIM, **extras))
    assert again.n_frames == picked.n_frames


@pytest.mark.parametrize('frames', [XYZ, LONG], ids=['short', 'long'])
@pytest.mark.parametrize('size', [[3, 1], [2, 2], [4, 1.5]])
@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_the_size_a_refusal_suggests_works(shape, n, size, frames):
    with pytest.raises(ValueError, match='larger size=') as err:
        hyp.plot(frames[:n], panels=shape, size=size, **ANIM)
    message = str(err.value)
    assert "xlabel='', ylabel='', zlabel=''" in message
    match = re.search(r'\(\(([\d.]+), ([\d.]+)\) or more\)', message)
    assert match, message
    suggested = [float(match.group(1)), float(match.group(2))]
    assert suggested[0] >= size[0] and suggested[1] >= size[1]
    anim = hyp.plot(frames[:n], panels=shape, size=suggested, **ANIM)
    for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
        anim.draw_frame(frame)
        _assert_labels_clear(anim.figure, n, frame)
    # ...and so does the advice to drop the labels
    hyp.plot(frames[:n], panels=shape, size=size, xlabel='', ylabel='',
             zlabel='', **ANIM)


def test_a_grid_that_does_not_rotate_needs_only_the_room_of_its_view():
    """`rotations=0`: the labels do not swing, so the cube is fitted for
    the one view and a size that a full turn refuses can be enough."""
    turning = hyp.plot(XYZ, panels=(1, 4), **ANIM)
    still = hyp.plot(XYZ, panels=(1, 4), rotations=0, **ANIM)
    still.draw_frame(0)
    turning.draw_frame(0)
    _draw(still.figure)
    _draw(turning.figure)
    # the one view's cube is drawn larger than the turning one's
    assert (_cube_box(still.figure.axes[0]).width
            > _cube_box(turning.figure.axes[0]).width)
    for frame in range(still.n_frames):
        still.draw_frame(frame)
        _assert_labels_clear(still.figure, 4, frame)
    # ...and it is fitted for that one view: the views sampled for a
    # panel that does not rotate are its own camera angle alone
    from hypertools.plot.plot import _camera_azimuths
    axis = still.figure.axes[0]
    assert _camera_azimuths(axis, 0, 3.0) == [float(axis.azim)]
    assert len(_camera_azimuths(axis, 1, 3.0)) == 121
    assert len(_camera_azimuths(axis, 0.25, 3.0)) == 31
    # a size the turning grid accepts is accepted without the turn, and
    # the smallest size refused without the turn is refused with it
    for size in ([8.0, 2.2], [7.0, 2.0], [6.0, 1.8], [5.0, 1.5]):
        try:
            hyp.plot(LONG, panels=(1, 4), size=size, **ANIM)
            turning_ok = True
        except ValueError:
            turning_ok = False
        try:
            hyp.plot(LONG, panels=(1, 4), size=size, rotations=0, **ANIM)
            still_ok = True
        except ValueError:
            still_ok = False
        assert still_ok or not turning_ok, size


# --------------------------------------------------------------------------
# A2: where the shared legend goes
# --------------------------------------------------------------------------

NAMES = ['alpha', 'beta', 'gamma', 'delta']


def _legend(fig):
    found = [legend for legend in fig.legends
             if legend.get_gid() == SHARED_LEGEND_GID]
    assert len(found) == 1
    return found[0]


def _grid_boxes(fig, n=4):
    renderer = _draw(fig)
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes[:n]]
    boxes += [ax.title.get_window_extent(renderer) for ax in fig.axes[:n]
              if ax.get_title()]
    return renderer, boxes


SIDES = {
    'right': 'right', 'center right': 'right', 'upper right': 'right',
    'lower right': 'right', 'center left': 'left',
    'upper left': 'left', 'lower left': 'left', 'upper center': 'top',
    'lower center': 'bottom', 9: 'top', 6: 'left',
}


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
@pytest.mark.parametrize('loc', list(SIDES), ids=[str(k) for k in SIDES])
def test_a_named_legend_location_is_a_strip_on_that_side(loc, animated,
                                                         ndims):
    extra = dict(ANIM, caption='a caption') if animated else dict(
        show=False)
    result = hyp.plot(DATA, ndims=ndims, panels=(2, 2), legend=NAMES,
                      title=['a', 'b', 'c', 'd'],
                      legend_kwargs={'loc': loc}, **extra)
    fig = getattr(result, 'figure', result)
    frames = range(result.n_frames) if animated else [None]
    for frame in frames:
        if frame is not None:
            result.draw_frame(frame)
        renderer, boxes = _grid_boxes(fig)
        width, height = fig.canvas.get_width_height()
        legend = _legend(fig).get_window_extent(renderer)
        # on the canvas, clear of every panel and title (and the caption)
        assert legend.x0 >= 0 and legend.x1 <= width, frame
        assert legend.y0 >= 0 and legend.y1 <= height, frame
        for box in boxes:
            assert not legend.overlaps(box), frame
        for text in fig.texts:
            assert not legend.overlaps(text.get_window_extent(renderer))
        side = SIDES[loc]
        if side == 'right':
            assert legend.x0 >= max(b.x1 for b in boxes) - 1
        elif side == 'left':
            assert legend.x1 <= min(b.x0 for b in boxes) + 1
        elif side == 'top':
            assert legend.y0 >= max(b.y1 for b in boxes) - 1
        else:
            assert legend.y1 <= min(b.y0 for b in boxes) + 1


def test_upper_and_lower_right_align_the_legend_with_the_grid():
    tops = {}
    for loc in ('upper right', 'center right', 'lower right'):
        fig = hyp.plot(DATA, ndims=2, panels=(2, 2), legend=NAMES,
                       legend_kwargs={'loc': loc}, show=False)
        renderer = _draw(fig)
        tops[loc] = _legend(fig).get_window_extent(renderer).y1
        plt.close(fig)
    assert tops['upper right'] > tops['center right'] > tops['lower right']


def test_default_legend_strip_is_unchanged_by_loc_right():
    default = hyp.plot(DATA, panels=(2, 2), legend=NAMES, show=False)
    named = hyp.plot(DATA, panels=(2, 2), legend=NAMES, show=False,
                     legend_kwargs={'loc': 'center right'})
    assert tuple(default.get_size_inches()) == tuple(named.get_size_inches())
    a = _legend(default).get_window_extent(_draw(default))
    b = _legend(named).get_window_extent(_draw(named))
    assert np.allclose(a.extents, b.extents, atol=0.5)


@pytest.mark.parametrize('loc,grows', [
    ('center left', 0), ('upper center', 1), ('lower center', 1),
    ('upper right', 0)])
def test_a_legend_strip_widens_the_default_figure_on_its_side_only(loc,
                                                                   grows):
    plain = hyp.plot(DATA, panels=(2, 2), show=False)
    fig = hyp.plot(DATA, panels=(2, 2), legend=NAMES, show=False,
                   legend_kwargs={'loc': loc})
    before, after = plain.get_size_inches(), fig.get_size_inches()
    assert after[grows] > before[grows] + 0.2
    assert after[1 - grows] == pytest.approx(before[1 - grows])


@pytest.mark.parametrize('loc', ['center left', 'lower center',
                                 'upper right'])
def test_an_explicit_size_holds_the_legend_strip_inside_it(loc):
    fig = hyp.plot(DATA, panels=(1, 4), legend=NAMES, size=[9, 3],
                   show=False, legend_kwargs={'loc': loc})
    assert tuple(fig.get_size_inches()) == (9.0, 3.0)
    renderer, boxes = _grid_boxes(fig)
    legend = _legend(fig).get_window_extent(renderer)
    assert legend.x0 >= 0 and legend.x1 <= 900 and legend.y0 >= 0
    for box in boxes:
        assert not legend.overlaps(box)


@pytest.mark.parametrize('kwargs', [
    {'bbox_to_anchor': (0.5, 0.5), 'loc': 'center'},
    {'bbox_to_anchor': (0.02, 0.98), 'loc': 'upper left'},
    {'loc': 'center'}, {'loc': (0.3, 0.4)}, {'loc': 10},
], ids=['anchor-center', 'anchor-corner', 'center', 'pair', 'code'])
def test_a_legend_the_caller_positions_gets_no_strip(kwargs):
    """No room is reserved for a legend the caller placed: the figure and
    the panels are exactly the legend-less grid's."""
    plain = hyp.plot(DATA, panels=(2, 2), show=False)
    fig = hyp.plot(DATA, panels=(2, 2), legend=NAMES, show=False,
                   legend_kwargs=kwargs)
    assert tuple(fig.get_size_inches()) == tuple(plain.get_size_inches())
    for a, b in zip(plain.axes[:4], fig.axes[:4]):
        # (a reserved strip would move the panels by a tenth of the figure)
        assert np.allclose(a.get_position().bounds, b.get_position().bounds,
                           atol=0.01)
    assert len(_legend(fig).get_texts()) == 4


def test_bbox_to_anchor_is_in_figure_fractions():
    fig = hyp.plot(DATA, panels=(2, 2), legend=NAMES, show=False,
                   legend_kwargs={'bbox_to_anchor': (0.25, 0.75),
                                  'loc': 'lower left'})
    renderer = _draw(fig)
    width, height = fig.canvas.get_width_height()
    box = _legend(fig).get_window_extent(renderer)
    assert box.x0 / width == pytest.approx(0.25, abs=0.01)
    assert box.y0 / height == pytest.approx(0.75, abs=0.01)


def test_a_bottom_legend_is_laid_out_in_columns_that_fit():
    many = [f'dataset number {i}' for i in range(12)]
    data = [_walk(seed) for seed in range(12)]
    fig = hyp.plot(data, panels=(3, 4), legend=many, show=False,
                   legend_kwargs={'loc': 'lower center'})
    renderer = _draw(fig)
    width, _ = fig.canvas.get_width_height()
    box = _legend(fig).get_window_extent(renderer)
    assert box.x0 >= 0 and box.x1 <= width
    assert _legend(fig)._ncols > 1


def test_plotly_legend_the_caller_positions_gets_no_strip():
    plain = hyp.plot(DATA, panels=(2, 2), backend='plotly', show=False)
    strip = hyp.plot(DATA, panels=(2, 2), backend='plotly', show=False,
                     legend=NAMES)
    placed = hyp.plot(DATA, panels=(2, 2), backend='plotly', show=False,
                      legend=NAMES, legend_kwargs={'x': 0.5, 'y': 1.1})
    assert strip.layout.width > plain.layout.width
    assert placed.layout.width == plain.layout.width
    assert placed.layout.margin.r == plain.layout.margin.r
    assert (placed.layout.legend.x, placed.layout.legend.y) == (0.5, 1.1)
    assert {trace.name for trace in placed.data if trace.showlegend} == set(
        NAMES)


@pytest.mark.parametrize('kwargs', [{}, {'loc': 'lower center'}],
                         ids=['right', 'bottom'])
def test_a_many_panel_grid_with_a_legend_lays_out_without_warnings(kwargs):
    """Each panel was first laid out around its own (transient) legend,
    which a small cell cannot hold: one "Tight layout not applied"
    warning per panel."""
    import warnings
    many = [f'dataset number {i}' for i in range(12)]
    data = [_walk(seed) for seed in range(12)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(data, panels=(3, 4), legend=many, show=False,
                       legend_kwargs=kwargs)
    assert not [str(w.message) for w in caught]
    assert len(_legend(fig).get_texts()) == 12


@pytest.mark.parametrize('shape,n', GRIDS, ids=['1x4', '2x2', '3x1'])
def test_static_long_axis_labels_do_not_run_into_each_other(shape, n):
    """Fitting the labels by shrinking the cube alone left three long
    labels on top of each other around a tiny cube: the default figure
    grows instead."""
    fig = hyp.plot(LONG[:n], panels=shape, legend=True, show=False)
    renderer = _draw(fig)
    for ax in fig.axes[:n]:
        boxes = [_shrunk(axis.label.get_window_extent(renderer))
                 for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
        for a, b in itertools.combinations(boxes, 2):
            assert not a.overlaps(b)
