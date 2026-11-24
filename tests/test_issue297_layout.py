"""GH #297: layout points found in the gallery figures.

A. A single animated 3-D plot's cube stays inside the canvas at every
   camera angle. On a square figure it reached (and passed) the left and
   right canvas edges when seen corner-on.
B. ...and stays off a legend placed in a corner of its axes
   (``legend_kwargs={'loc': 'upper left'}``), static or animated.
C. Observation labels (``labels=``) are not drawn on top of each other.
D. An animated markers-only plot with a continuous ``hue=`` is revealed
   frame by frame like every other animation. It drew every observation
   on every frame.

Real figures and real frames, rendered headless. Nothing is mocked. No
assertion depends on a font's absolute metrics: boxes are measured where
they were drawn and compared with each other and with the canvas.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import hypertools as hyp                                        # noqa: E402


def _walk(seed, rows=60, cols=3):
    return np.random.default_rng(seed).normal(size=(rows, cols)).cumsum(0)


X = _walk(0)
DATA = [_walk(seed) for seed in range(5)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _ink(fig):
    """Boolean image (row 0 at the TOP) of the pixels `fig` draws on."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return (rgba[..., :3] < 250).any(axis=-1)


def _ink_margins(fig):
    """(left, right, bottom, top) blank margins around what is drawn, px."""
    ink = _ink(fig)
    cols = np.flatnonzero(ink.any(axis=0))
    rows = np.flatnonzero(ink.any(axis=1))
    height, width = ink.shape
    return (cols[0], width - 1 - cols[-1], height - 1 - rows[-1], rows[0])


def _box_px(artist, fig):
    return artist.get_window_extent(fig.canvas.get_renderer())


# --------------------------------------------------------------------------
# A. the cube and the canvas edge
# --------------------------------------------------------------------------

@pytest.mark.parametrize('size', [(8, 8), (5, 8), (6, 6)])
def test_animated_cube_stays_inside_a_square_or_portrait_canvas(size):
    anim = hyp.plot(X, '-', animate=True, duration=3, frame_rate=12,
                    size=size, elev=16, show=False)
    fig = anim.figure
    width = fig.bbox.width
    least = np.inf
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        left, right, _bottom, _top = _ink_margins(fig)
        least = min(least, left, right)
    # the cube keeps a visible margin (2% of the canvas is asked for; the
    # frame's line width and the frame sampling take a little of it)
    assert least >= 0.01 * width, (
        f'the cube comes within {least} px of the side of a '
        f'{width:.0f} px canvas')


def test_default_landscape_animation_is_not_zoomed_out():
    """The cube already clears a landscape canvas: nothing changes."""
    anim = hyp.plot(X, '-', animate=True, duration=2, frame_rate=10,
                    show=False)
    assert anim.animation._hyp_zoom_scale == [1.0]


def test_cube_fills_the_width_it_is_given():
    """The fit takes no more than it needs: at its widest view the cube
    reaches to within the margin of one side of the canvas."""
    anim = hyp.plot(X, '-', animate=True, duration=3, frame_rate=12,
                    size=(8, 8), show=False)
    fig = anim.figure
    least = np.inf
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        least = min(least, *_ink_margins(fig)[:2])
    assert least <= 0.035 * fig.bbox.width


def test_an_explicit_zoom_in_is_honored():
    """`zoom=` above 1 asks for a closer view: the cube is not shrunk
    back to the canvas."""
    anim = hyp.plot(X, '-', animate=True, duration=2, frame_rate=10,
                    size=(8, 8), zoom=3, show=False)
    assert anim.animation._hyp_zoom_scale == [1.0]


def test_cube_fit_does_not_run_the_users_hook():
    """Fitting the cube draws frame 0 through the updater alone."""
    calls = []
    anim = hyp.plot(X, '-', animate=True, duration=2, frame_rate=10,
                    size=(8, 8), on_frame=lambda ctx: calls.append(ctx.frame),
                    show=False)
    assert calls == []
    anim.draw_frame(3)
    assert calls == [3]


# --------------------------------------------------------------------------
# B. the cube and a legend inside the axes
# --------------------------------------------------------------------------

def _legend_area_ink(fig, legend):
    """How many pixels are drawn where `legend` goes, with the legend
    itself hidden: the cube or the data reaching under it."""
    box = _box_px(legend, fig)
    legend.set_visible(False)
    try:
        ink = _ink(fig)
    finally:
        legend.set_visible(True)
    height = ink.shape[0]
    rows = slice(max(0, int(height - box.y1)), int(np.ceil(height - box.y0)))
    cols = slice(max(0, int(box.x0)), int(np.ceil(box.x1)))
    return int(ink[rows, cols].sum())


#: ten entries: tall enough, in a corner of an 8 x 8 in figure, to reach
#: the cube as it was drawn (measured: the cube has to come down to about
#: three quarters of its size to clear it)
KEYED = [_walk(seed) for seed in range(10)]
LEGEND = [f'A rather long name for dataset {i}' for i in range(len(KEYED))]


@pytest.mark.parametrize('loc', ['upper left', 'upper right', 'lower left'])
def test_animated_cube_stays_off_a_corner_legend(loc):
    anim = hyp.plot(KEYED, '-', animate=True, duration=3, frame_rate=12,
                    size=(8, 8), legend=LEGEND,
                    legend_kwargs={'loc': loc, 'fontsize': 8}, show=False)
    fig = anim.figure
    legend = fig.axes[0].get_legend()
    assert legend is not None
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        assert _legend_area_ink(fig, legend) == 0, (
            f'frame {frame}: the cube is drawn under the {loc} legend')
    # the legend, not the canvas edge, is what set the cube's size here
    edge_only = hyp.plot(KEYED, '-', animate=True, duration=3, frame_rate=12,
                         size=(8, 8), show=False)
    assert (anim.animation._hyp_zoom_scale[0]
            < 0.95 * edge_only.animation._hyp_zoom_scale[0])


def test_animated_cube_stays_off_the_default_legend():
    """The default legend sits right of the plot, where the cube's right
    corner reached into its box at the widest view."""
    anim = hyp.plot(DATA, '-', animate=True, duration=3, frame_rate=12,
                    legend=LEGEND[:len(DATA)], show=False)
    fig = anim.figure
    legend = fig.axes[0].get_legend()
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        assert _legend_area_ink(fig, legend) == 0, f'frame {frame}'
    # the legend is still whole, on the canvas
    box = _box_px(legend, fig)
    assert box.x0 >= 0 and box.x1 <= fig.bbox.width


def test_animated_cube_stays_off_its_colorbar():
    anim = hyp.plot(X, '-', hue=np.arange(len(X)), colorbar=True,
                    animate=True, duration=3, frame_rate=12, show=False)
    fig = anim.figure
    plot_ax, bar_ax = fig.axes
    bar = bar_ax.get_position()
    edge = bar.x0 * fig.bbox.width
    bar_ax.set_visible(False)
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        ink = _ink(fig)
        columns = np.flatnonzero(ink.any(axis=0))
        assert columns[-1] < edge, f'frame {frame}'


def test_cube_is_not_shrunk_without_limit_for_a_huge_legend():
    """A legend that would leave the cube under 55% of its size is not
    cleared: the cube stays at that share. (70% until review round 5:
    under matplotlib 3.11's taller lines the ten-entry legend of
    `test_animated_cube_stays_off_a_corner_legend` needs 61%.)"""
    data = [_walk(seed) for seed in range(24)]
    anim = hyp.plot(data, '-', animate=True, duration=2, frame_rate=10,
                    size=(8, 8), legend=[f'A long name for dataset {i}'
                                         for i in range(len(data))],
                    legend_kwargs={'loc': 'upper left', 'fontsize': 8},
                    show=False)
    assert anim.animation._hyp_zoom_scale[0] == pytest.approx(0.55, abs=0.01)


def test_static_cube_stays_off_a_corner_legend():
    fig = hyp.plot(DATA, '-', size=(5, 5), legend=LEGEND[:len(DATA)],
                   legend_kwargs={'loc': 'upper left', 'fontsize': 8},
                   show=False)
    legend = fig.axes[0].get_legend()
    assert _legend_area_ink(fig, legend) == 0


def test_static_plot_without_a_corner_legend_keeps_its_cube():
    """The default (outside, right) legend and no legend at all leave a
    static cube at matplotlib's own size."""
    plain = hyp.plot(DATA, '-', show=False)
    keyed = hyp.plot(KEYED, '-', legend=LEGEND, show=False)
    reference = plt.figure().add_subplot(projection='3d').get_box_aspect()
    for fig in (plain, keyed):
        np.testing.assert_allclose(fig.axes[0].get_box_aspect(), reference)


# --------------------------------------------------------------------------
# C. observation labels
# --------------------------------------------------------------------------

def _label_boxes(fig):
    """Display boxes (round frame included) of the visible labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [text.get_bbox_patch().get_window_extent(renderer)
            for ax in fig.axes for text in ax.texts
            if text.get_visible() and text.get_text()
            and text.get_bbox_patch() is not None]


def _overlapping_pairs(boxes):
    return [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
            if boxes[i].overlaps(boxes[j])]


def _crowded(n=4, cols=3):
    """`n` datasets whose FIRST rows nearly coincide: their labels, all
    hung on the same side, land on each other."""
    out = []
    for seed in range(n):
        walk = _walk(seed, rows=30, cols=cols)
        out.append(walk - walk[0] + 0.02 * seed)
    return out


@pytest.mark.parametrize('ndims', [2, 3])
def test_labels_of_neighboring_points_do_not_overlap(ndims):
    names = ['Subject 0', 'Subject 1', 'Subject 2', 'Subject 3']
    fig = hyp.plot(_crowded(), '.', labels=names, ndims=ndims, show=False)
    boxes = _label_boxes(fig)
    assert len(boxes) == len(names)
    assert _overlapping_pairs(boxes) == []
    # ...and every one of them is still on the canvas
    for box in boxes:
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height


def test_labels_that_are_clear_of_each_other_stay_where_they_were():
    """A label that covers no other keeps its place up and to the left
    of its point, so such a figure is drawn as it always was."""
    apart = [np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]]),
             np.array([[1.0, 1.0, 1.0], [0.0, 0.1, 0.0]])]
    fig = hyp.plot(apart, '.', labels=['low', 'high'], show=False)
    fig.canvas.draw()
    labels = [t for t in fig.axes[0].texts if t.get_text()]
    assert len(labels) == 2
    for label in labels:
        assert tuple(label.xyann) == (-20.0, 20.0)
        assert label.get_ha() == 'right' and label.get_va() == 'bottom'


def test_first_label_keeps_its_place_and_later_ones_move():
    fig = hyp.plot(_crowded(), '.', labels=['a' * 9, 'b' * 9, 'c' * 9,
                                           'd' * 9], show=False)
    fig.canvas.draw()
    labels = [t for t in fig.axes[0].texts if t.get_text()]
    assert tuple(labels[0].xyann) == (-20.0, 20.0)
    assert any(tuple(label.xyann) != (-20.0, 20.0) for label in labels[1:])


def test_label_placement_is_the_same_on_every_draw():
    fig = hyp.plot(_crowded(), '.', labels=list('abcd'), show=False)
    first = [tuple(np.round(box.extents, 3)) for box in _label_boxes(fig)]
    again = [tuple(np.round(box.extents, 3)) for box in _label_boxes(fig)]
    assert first == again


def test_labels_are_placed_for_the_layout_the_figure_ends_up_with():
    """The placement is made at draw time: resizing the figure after the
    call (which moves the points, not the labels' offsets) still leaves
    the labels clear of each other."""
    fig = hyp.plot(_crowded(), '.', labels=['Subject 0', 'Subject 1',
                                           'Subject 2', 'Subject 3'],
                   show=False)
    fig.set_size_inches(3.5, 3.0)
    assert _overlapping_pairs(_label_boxes(fig)) == []


def test_a_label_the_reader_moved_is_left_alone():
    fig = hyp.plot(_crowded(cols=2), '.', labels=list('abcd'), ndims=2,
                   show=False)
    fig.canvas.draw()
    label = [t for t in fig.axes[0].texts if t.get_text()][1]
    label.xyann = (55.0, -35.0)                 # what a drag does
    fig.canvas.draw()
    fig.canvas.draw()
    assert tuple(label.xyann) == (55.0, -35.0)


def test_animated_labels_do_not_overlap_on_any_frame():
    data = _crowded(3)
    anim = hyp.plot(data, '-', labels=['first', 'second', 'third'],
                    animate='spin', duration=2, frame_rate=10, show=False)
    fig = anim.figure
    seen = 0
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        boxes = _label_boxes(fig)
        seen = max(seen, len(boxes))
        assert _overlapping_pairs(boxes) == [], f'frame {frame}'
    assert seen == 3


def test_an_animation_frame_places_its_labels_the_same_whenever_drawn():
    """Frames are idempotent: the placement depends on the frame alone,
    not on the frames drawn before it."""
    anim = hyp.plot(_crowded(3), '-', labels=['first', 'second', 'third'],
                    animate='spin', duration=2, frame_rate=10, show=False)
    fig = anim.figure

    def boxes(frame):
        anim.draw_frame(frame)
        return [tuple(np.round(b.extents, 2)) for b in _label_boxes(fig)]

    forward = boxes(5)
    boxes(15)
    boxes(0)
    assert boxes(5) == forward


# --------------------------------------------------------------------------
# D. animated markers with a continuous hue
# --------------------------------------------------------------------------

HUE = np.linspace(0.0, 1.0, len(X))


def _scatters(ax):
    """The per-point-colored scatters of the data (not the cube frame)."""
    from matplotlib.collections import PathCollection
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _offsets(coll, ndims):
    if ndims == 3:
        return np.column_stack([np.asarray(v) for v in coll._offsets3d])
    return np.asarray(coll.get_offsets())


def _static_points_and_colors(ndims):
    """Where a STATIC plot of the same call puts each observation, and
    the color it gives it."""
    fig = hyp.plot(X, 'o', hue=HUE, ndims=ndims, show=False)
    (coll,) = _scatters(fig.axes[0])
    # the colors depend on hue= alone; they are read off the 2-D scatter,
    # which keeps them in row order (a 3-D one sorts them by depth)
    flat = hyp.plot(X, 'o', hue=HUE, ndims=2, show=False)
    (flat_coll,) = _scatters(flat.axes[0])
    colors = np.asarray(flat_coll.get_facecolors())
    assert len(colors) == len(X)
    return _offsets(coll, ndims), colors


def _drawn_colors(coll, fig):
    """The RGB rows `coll` is drawn with, sorted (a 3-D scatter hands its
    colors back in depth order, as of its last draw)."""
    fig.canvas.draw()
    colors = np.round(np.asarray(coll.get_facecolors())[:, :3], 9)
    return colors[np.lexsort(colors.T)]


def _sorted_rows(colors):
    colors = np.round(np.asarray(colors)[:, :3], 9)
    return colors[np.lexsort(colors.T)]


def _rows_of(points, reference):
    """Which rows of `reference` the rows of `points` are."""
    rows = []
    for point in points:
        distance = np.abs(reference - point).sum(axis=1)
        assert distance.min() < 1e-9
        rows.append(int(distance.argmin()))
    return rows


@pytest.mark.parametrize('ndims', [2, 3])
def test_animated_marker_hue_is_revealed_frame_by_frame(ndims):
    reference, colors = _static_points_and_colors(ndims)
    anim = hyp.plot(X, 'o', hue=HUE, ndims=ndims, animate=True, duration=3,
                    frame_rate=10, show=False)
    plain = hyp.plot(X, 'o', ndims=ndims, animate=True, duration=3,
                     frame_rate=10, show=False)
    ax = anim.figure.axes[0]
    (head,) = _scatters(ax)
    shown = []
    for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
        anim.draw_frame(frame)
        plain.draw_frame(frame)
        assert head.get_visible()
        rows = _rows_of(_offsets(head, ndims), reference)
        # the window of the same animation without hue=
        line = plain.figure.axes[0].lines[0]
        expected = len(line.get_data_3d()[0] if ndims == 3
                       else line.get_xdata())
        assert len(rows) == expected
        # consecutive observations, each in its own color
        assert rows == list(range(rows[0], rows[0] + len(rows)))
        np.testing.assert_allclose(_drawn_colors(head, anim.figure),
                                   _sorted_rows(colors[rows]), atol=1e-8)
        shown.append(rows)
    assert shown[0] != shown[1] != shown[2]
    assert len(shown[0]) < len(X)


def test_animated_marker_hue_frames_differ_in_2d():
    """Every frame of the 2-D animation was the same picture."""
    anim = hyp.plot(X, 'o', hue=HUE, ndims=2, animate=True, duration=3,
                    frame_rate=10, show=False)
    fig = anim.figure
    pictures = []
    for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
        anim.draw_frame(frame)
        pictures.append(_ink(fig).copy())
    assert (pictures[0] != pictures[1]).any()
    assert (pictures[1] != pictures[2]).any()


@pytest.mark.parametrize('ndims', [2, 3])
def test_animated_marker_hue_draws_its_chemtrail(ndims):
    reference, colors = _static_points_and_colors(ndims)
    anim = hyp.plot(X, 'o', hue=HUE, ndims=ndims, animate=True, duration=3,
                    frame_rate=10, chemtrails=True, tail_duration=0.5,
                    trail_alpha=0.4, show=False)
    ax = anim.figure.axes[0]
    head, trail = _scatters(ax)
    assert trail._hyp_trace_role == 'trail'
    anim.draw_frame(anim.n_frames - 1)
    assert head.get_visible() and trail.get_visible()
    head_rows = _rows_of(_offsets(head, ndims), reference)
    trail_rows = _rows_of(_offsets(trail, ndims), reference)
    # the trail holds what the head has left behind, from the first row
    assert trail_rows[0] == 0
    assert len(trail_rows) > len(head_rows)
    assert head_rows[-1] == len(X) - 1
    assert trail.get_alpha() == pytest.approx(0.4)
    np.testing.assert_allclose(_drawn_colors(trail, anim.figure),
                               _sorted_rows(colors[trail_rows]), atol=1e-8)


def test_animated_marker_hue_keeps_its_colorbar_and_marker():
    anim = hyp.plot(X, 's', hue=HUE, ndims=2, animate=True, duration=2,
                    frame_rate=10, markersize=9, colorbar=True, show=False)
    fig = anim.figure
    assert len(fig.axes) == 2                       # the plot and its colorbar
    (head,) = _scatters(fig.axes[0])
    anim.draw_frame(anim.n_frames - 1)
    np.testing.assert_allclose(head.get_sizes(), [81.0])
    square = matplotlib.markers.MarkerStyle('s')
    np.testing.assert_allclose(
        head.get_paths()[0].vertices,
        square.get_path().transformed(square.get_transform()).vertices)


def test_spinning_marker_hue_still_shows_every_observation():
    """'spin' moves the camera only: all the points, on every frame."""
    anim = hyp.plot(X, 'o', hue=HUE, animate='spin', duration=2,
                    frame_rate=10, show=False)
    (coll,) = _scatters(anim.figure.axes[0])
    for frame in (0, anim.n_frames - 1):
        anim.draw_frame(frame)
        assert len(_offsets(coll, 3)) == len(X)


def test_on_frame_receives_the_drawn_marker_scatter():
    seen = []
    anim = hyp.plot(X, 'o', hue=HUE, ndims=2, animate=True, duration=2,
                    frame_rate=10, show=False,
                    on_frame=lambda ctx: seen.append(list(ctx.artists)))
    anim.draw_frame(4)
    (head,) = _scatters(anim.figure.axes[0])
    assert seen and seen[-1][0] is head
