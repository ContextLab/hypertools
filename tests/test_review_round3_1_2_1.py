"""1.2.1 review round 3: what the third review found in the round 2 fixes.

A. ``labels=`` with ``animate='morph'`` reads arrays and Series like
   every other reader of ``labels=`` (the check now goes through the
   shared datatype predicates; `tests/test_datatype_gate.py`).
B. The mouse callbacks of a labeled plot: one pair per axes, and they
   work on the labels the axes holds now, so an ``ax.clear()`` followed
   by a new labeled plot survives a mouse click.
C. A label's measured size follows everything that changes its drawn
   extent (rotation, ``parse_math``), so labels turned on their side are
   still kept off each other.
D. Continuous ``hue=`` markers with a half ``fillstyle`` are whole
   outlined markers with one half filled, as matplotlib draws them on a
   line, static and animated, 2-D and 3-D.

Real figures, real mouse events and real frames, rendered headless.
Nothing is mocked, and no assertion depends on a font's absolute
metrics.
"""
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.backend_bases import MouseButton, MouseEvent    # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg     # noqa: E402
from matplotlib.collections import PathCollection               # noqa: E402
from matplotlib.colors import to_rgb                            # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.matplotlib_backend import _label_size_pt   # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _entries(ax):
    return [entry for entry in ax._hyp_labels_and_points
            if entry[0].axes is ax]


# ---------------------------------------------------------------------------
# A. labels= with animate='morph': arrays and Series
# ---------------------------------------------------------------------------

def _morph_label_warnings(labels):
    rng = np.random.default_rng(0)
    clouds = [rng.normal(size=(12, 3)), rng.normal(size=(12, 3)) + 2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(clouds, 'o', animate='morph', labels=labels, ndims=3,
                 duration=1, frame_rate=5, show=False)
    return [str(w.message) for w in caught
            if "animate='morph'" in str(w.message)
            and 'ignoring labels=' in str(w.message)]


@pytest.mark.parametrize('make', [
    lambda names: [np.array(names), np.array(names)],
    lambda names: [pd.Series(names), pd.Series(names)],
    lambda names: (tuple(names), tuple(names)),
], ids=['arrays', 'series', 'tuples'])
def test_morph_says_that_array_and_series_labels_are_not_drawn(make):
    names = [f'p{i}' for i in range(12)]
    assert len(_morph_label_warnings(make(names))) == 1


@pytest.mark.parametrize('labels', [
    [[None] * 12, [None] * 12],
    [np.array([None] * 12, dtype=object)] * 2,
], ids=['lists', 'arrays'])
def test_morph_says_nothing_when_no_label_is_given(labels):
    assert _morph_label_warnings(labels) == []


# ---------------------------------------------------------------------------
# B. one pair of mouse callbacks per labeled axes
# ---------------------------------------------------------------------------

def _connected(fig, name):
    return len(fig.canvas.callbacks.callbacks.get(name, {}))


def _click(fig, ax):
    """Press and release the left mouse button over `ax` (real events)."""
    x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
    for name in ('button_press_event', 'button_release_event'):
        fig.canvas.callbacks.process(name, MouseEvent(
            name, fig.canvas, x, y, button=MouseButton.LEFT))


def _labeled_axes(ndims):
    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    return fig, ax, np.random.default_rng(0).normal(size=(6, ndims))


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_click_after_clearing_and_relabeling_an_axes_works(ndims):
    """The first call's callbacks stayed connected with its removed
    labels (``axes is None``) and raised AttributeError out of the
    button release, which also left the new labels hidden."""
    fig, ax, x = _labeled_axes(ndims)
    hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
             reduce=None, ndims=ndims)
    removed = [entry[0] for entry in _entries(ax)]
    ax.clear()
    hyp.plot(x + 1, 'o', ax=ax, labels=list('uvwxyz'), show=False,
             reduce=None, ndims=ndims)
    fig.canvas.draw()
    labels = [entry[0] for entry in _entries(ax)]
    assert [label.get_text() for label in labels] == list('uvwxyz')
    assert all(label.axes is None for label in removed)

    x0, y0 = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
    fig.canvas.callbacks.process('button_press_event', MouseEvent(
        'button_press_event', fig.canvas, x0, y0, button=MouseButton.LEFT))
    assert not any(label.get_visible() for label in labels)
    fig.canvas.callbacks.process('button_release_event', MouseEvent(
        'button_release_event', fig.canvas, x0, y0,
        button=MouseButton.LEFT))
    assert all(label.get_visible() for label in labels)
    # ...and a second click, and a draw after it
    _click(fig, ax)
    fig.canvas.draw()
    assert all(label.get_visible() for label in labels)


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_click_on_a_cleared_axes_that_was_not_relabeled_works(ndims):
    fig, ax, x = _labeled_axes(ndims)
    hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
             reduce=None, ndims=ndims)
    ax.clear()
    _click(fig, ax)
    fig.canvas.draw()
    assert len(ax.texts) == 0


@pytest.mark.parametrize('ndims', [2, 3])
def test_labeled_calls_into_one_axes_share_one_pair_of_callbacks(ndims):
    """Every call connected a press and a release callback of its own."""
    fig, ax, x = _labeled_axes(ndims)
    hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
             reduce=None, ndims=ndims)
    _click(fig, ax)     # (matplotlib drops what it no longer needs here)
    press = _connected(fig, 'button_press_event')
    release = _connected(fig, 'button_release_event')
    for _ in range(3):
        ax.clear()
        hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
                 reduce=None, ndims=ndims)
        _click(fig, ax)
    assert _connected(fig, 'button_press_event') == press
    # (the same six labels each time, so the same number of matplotlib's
    # own per-label callbacks, which it disconnects at the click)
    assert _connected(fig, 'button_release_event') == release
    # a second call WITHOUT a clear adds labels, and no callback of ours
    hyp.plot(x + 1, 'o', ax=ax, labels=list('uvwxyz'), show=False,
             reduce=None, ndims=ndims)
    assert _connected(fig, 'button_press_event') == press
    labels = [entry[0] for entry in _entries(ax)]
    assert len(labels) == 12
    _click(fig, ax)
    assert all(label.get_visible() for label in labels)


def test_a_labeled_figure_given_another_canvas_answers_the_mouse_there():
    fig, ax, x = _labeled_axes(2)
    hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
             reduce=None, ndims=2)
    FigureCanvasAgg(fig)
    assert _connected(fig, 'button_press_event') == 0
    hyp.plot(x + 1, 'o', ax=ax, labels=list('uvwxyz'), show=False,
             reduce=None, ndims=2)
    labels = [entry[0] for entry in _entries(ax)]
    assert len(labels) == 12
    x0, y0 = ax.bbox.x0 + 5, ax.bbox.y0 + 5
    fig.canvas.callbacks.process('button_press_event', MouseEvent(
        'button_press_event', fig.canvas, x0, y0, button=MouseButton.LEFT))
    assert not any(label.get_visible() for label in labels)
    fig.canvas.callbacks.process('button_release_event', MouseEvent(
        'button_release_event', fig.canvas, x0, y0,
        button=MouseButton.LEFT))
    assert all(label.get_visible() for label in labels)


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_labeled_figure_that_was_clicked_can_be_pickled(ndims):
    fig, ax, x = _labeled_axes(ndims)
    hyp.plot(x, 'o', ax=ax, labels=list('abcdef'), show=False,
             reduce=None, ndims=ndims)
    _click(fig, ax)
    # (bytes this test has just made from its own figure)
    again = pickle.loads(pickle.dumps(fig))
    assert [text.get_text() for text in again.axes[0].texts] == list(
        'abcdef')


def test_each_labeled_axes_of_a_figure_answers_the_mouse():
    fig, (left, right) = plt.subplots(1, 2)
    x = np.random.default_rng(0).normal(size=(5, 2))
    hyp.plot(x, 'o', ax=left, labels=list('abcde'), show=False,
             reduce=None, ndims=2)
    hyp.plot(x, 'o', ax=right, labels=list('vwxyz'), show=False,
             reduce=None, ndims=2)
    labels = [entry[0] for ax in (left, right) for entry in _entries(ax)]
    assert len(labels) == 10
    x0, y0 = left.bbox.x0 + 5, left.bbox.y0 + 5
    fig.canvas.callbacks.process('button_press_event', MouseEvent(
        'button_press_event', fig.canvas, x0, y0, button=MouseButton.LEFT))
    assert not any(label.get_visible() for label in labels)
    fig.canvas.callbacks.process('button_release_event', MouseEvent(
        'button_release_event', fig.canvas, x0, y0,
        button=MouseButton.LEFT))
    assert all(label.get_visible() for label in labels)


# ---------------------------------------------------------------------------
# C. a label's size follows its rotation and its parse_math
# ---------------------------------------------------------------------------

def _overlapping_pairs(fig, labels):
    """How many pairs of the label boxes overlap, as they were drawn."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer)
             for label in labels if label.get_visible()]
    count = 0
    for i, one in enumerate(boxes):
        for other in boxes[i + 1:]:
            if (min(one.x1, other.x1) - max(one.x0, other.x0) > 0.5
                    and min(one.y1, other.y1) - max(one.y0, other.y0) > 0.5):
                count += 1
    return count


def _stacked_labels():
    """Two points one above the other, far enough apart that their
    (wide, flat) labels are clear of each other."""
    x = np.array([[0.0, 0.0], [0.0, 0.25], [1.0, 1.0], [-1.0, -1.0]])
    fig = hyp.plot(x, 'o', reduce=None, ndims=2, show=False,
                   labels=['a long label below', 'a long label above',
                           None, None])
    ax = fig.axes[0]
    labels = [entry[0] for entry in _entries(ax)]
    return fig, labels


def test_a_rotated_label_is_measured_as_it_is_drawn():
    fig, labels = _stacked_labels()
    fig.canvas.draw()
    label = labels[0]
    wide, flat = _label_size_pt(label)
    assert wide > 3 * flat
    label.set_rotation(90)
    fig.canvas.draw()
    assert _label_size_pt(label) == pytest.approx((flat, wide), rel=0.05)
    label.set_rotation(45)
    fig.canvas.draw()
    width, height = _label_size_pt(label)
    assert width == pytest.approx(height, rel=0.05)
    assert flat < width < wide
    label.set_rotation(0)
    fig.canvas.draw()
    assert _label_size_pt(label) == pytest.approx((wide, flat))


def test_a_label_switched_to_plain_text_is_measured_again():
    fig, labels = _stacked_labels()
    label = labels[0]
    label.set_text(r'$\frac{a}{b}$ and $\sum_{i}^{n} x_i^2$')
    fig.canvas.draw()
    as_math = _label_size_pt(label)
    label.set_parse_math(False)
    fig.canvas.draw()
    as_text = _label_size_pt(label)
    # (the plain text is every character of the source, on one line)
    assert as_text[0] > as_math[0]
    assert as_text[1] < as_math[1]


def test_labels_turned_on_their_side_are_kept_off_each_other():
    """The size kept from before the rotation said the two labels were
    still clear of each other, so they were left overlapping."""
    fig, labels = _stacked_labels()
    assert _overlapping_pairs(fig, labels) == 0
    before = [tuple(label.xyann) for label in labels]
    for label in labels:
        label.set_rotation(90)
    fig.canvas.draw()
    assert _overlapping_pairs(fig, labels) == 0
    # (they did overlap where they were: one of them has moved)
    assert [tuple(label.xyann) for label in labels] != before


# ---------------------------------------------------------------------------
# D. half-filled markers colored by a continuous hue
# ---------------------------------------------------------------------------

XY = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
# (in 3-D, no observation on an edge of the cube the data spans: an
# animation draws those edges, and a line through a marker is not the
# marker's own look)
XYZ = np.array([[0.0, 0.1, -0.2], [1.0, 0.3, 0.2], [-1.0, -0.2, 0.1],
                [0.2, 1.0, 0.3], [-0.3, -1.0, -0.1], [0.3, -0.2, 1.0],
                [-0.2, 0.2, -1.0]])
HALF = dict(reduce=None, fillstyle='left', markeredgecolor='k',
            markersize=30, show=False)


def _half_scatter(ndims, animated, fmt='o', **kwargs):
    data = XYZ if ndims == 3 else XY
    options = dict(HALF, ndims=ndims,
                   hue=np.linspace(0.0, 1.0, len(data)), **kwargs)
    if animated:
        anim = hyp.plot(data, fmt, animate=True, duration=1, frame_rate=4,
                        **options)
        anim.draw_frame(anim.n_frames - 1)
        fig = anim.figure
    else:
        fig = hyp.plot(data, fmt, **options)
    ax = fig.axes[0]
    dots = [coll for coll in ax.collections
            if isinstance(coll, PathCollection) and coll.get_visible()]
    assert len(dots) == 1
    return fig, ax, dots[0]


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0


def _marker_looks(fig, ax, dots):
    """Per marker, from the drawn pixels: the color inside its left
    half, the color inside its right half, and the darkest pixel on the
    way out of the marker through its right rim."""
    image = _pixels(fig)
    centers = dots.get_offset_transform().transform(
        np.asarray(dots.get_offsets(), dtype=float))
    radius = 0.5 * np.sqrt(dots.get_sizes()[0]) * fig.dpi / 72.0
    height = image.shape[0]
    looks = []
    for cx, cy in centers:
        row = int(round(height - cy))
        left = image[row, int(round(cx - 0.5 * radius))]
        right = image[row, int(round(cx + 0.5 * radius))]
        rim = image[row, int(round(cx + 0.7 * radius)):
                    int(round(cx + 1.3 * radius)) + 1]
        looks.append((left, right, float(rim.sum(axis=1).min()) / 3.0))
    return looks


def _right_outline_pixels(fig, dots):
    """Per marker, how many dark pixels lie in the right part of its
    box, clear of the straight edge down its middle: the outline of the
    right half, about two and a half radii long when it is drawn."""
    image = _pixels(fig)
    centers = dots.get_offset_transform().transform(
        np.asarray(dots.get_offsets(), dtype=float))
    radius = 0.5 * np.sqrt(dots.get_sizes()[0]) * fig.dpi / 72.0
    height = image.shape[0]
    counts = []
    for cx, cy in centers:
        row = height - cy
        part = image[int(round(row - 1.3 * radius)):
                     int(round(row + 1.3 * radius)) + 1,
                     int(round(cx + 0.3 * radius)):
                     int(round(cx + 1.3 * radius)) + 1]
        counts.append(int((part.mean(axis=2) < 0.35).sum()) / radius)
    return counts


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt', ['o', '-o'])
def test_half_filled_hue_markers_are_whole_outlined_markers(
        ndims, animated, fmt):
    """The scatter drew the filled half alone: half-discs with nothing
    where the other half of the marker is."""
    fig, ax, dots = _half_scatter(ndims, animated, fmt=fmt)
    looks = _marker_looks(fig, ax, dots)
    count = 7 if ndims == 3 else 3
    assert len(looks) == count
    hues = np.asarray(dots.get_facecolors())[:, :3]
    assert len(hues) == count
    # the black edge goes round the right half of every marker
    assert min(_right_outline_pixels(fig, dots)) > 1.5
    if fmt == 'o':
        # (a line format draws its line through the markers, static
        # over them, so single pixels are read on markers alone)
        for left, right, rim in looks:
            # the left half wears one of the hue colors...
            assert min(np.abs(hues - left).max(axis=1)) < 0.05
            # ...the right half is not filled...
            assert min(np.abs(hues - right).max(axis=1)) > 0.2
            # ...and the edge is met on the way out through it
            assert rim < 0.35
    # one artist with one offset per observation, like any scatter
    assert len(dots.get_offsets()) == count
    assert len(dots.get_paths()) == 1


def test_half_filled_hue_markers_match_matplotlibs_own_on_a_line():
    """The same picture as ``ax.plot(..., fillstyle='left')`` draws,
    marker for marker, with the hue in the place of the line's color."""
    fig, ax, dots = _half_scatter(2, False)
    ours = _marker_looks(fig, ax, dots)
    hues = np.asarray(dots.get_facecolors())[:, :3]

    ref_fig, ref_ax = plt.subplots(figsize=fig.get_size_inches(),
                                   dpi=fig.dpi)
    for (px, py), color in zip(XY, hues):
        ref_ax.plot([px], [py], 'o', color=color, fillstyle='left',
                    markeredgecolor='k', markersize=30)
    image = _pixels(ref_fig)
    radius = 0.5 * 30 * ref_fig.dpi / 72.0
    for (px, py), (left, right, rim) in zip(XY, ours):
        cx, cy = ref_ax.transData.transform((px, py))
        row = int(round(image.shape[0] - cy))
        assert image[row, int(round(cx - 0.5 * radius))] == pytest.approx(
            left, abs=0.05)
        assert image[row, int(round(cx + 0.5 * radius))] == pytest.approx(
            right, abs=0.05)
        ref_rim = image[row, int(round(cx + 0.7 * radius)):
                        int(round(cx + 1.3 * radius)) + 1]
        assert float(ref_rim.sum(axis=1).min()) / 3.0 < 0.35
        assert rim < 0.35


@pytest.mark.parametrize('animated', [False, True],
                         ids=['static', 'animated'])
def test_the_other_half_wears_markerfacecoloralt(animated):
    fig, ax, dots = _half_scatter(2, animated, markerfacecoloralt='gold')
    for _left, right, rim in _marker_looks(fig, ax, dots):
        assert right == pytest.approx(to_rgb('gold'), abs=0.05)
        assert rim < 0.35


def test_a_full_fillstyle_and_an_unfilled_marker_draw_as_before():
    fig, ax, dots = _half_scatter(2, False, fillstyle='full')
    assert type(dots) is PathCollection
    for left, right, rim in _marker_looks(fig, ax, dots):
        assert right == pytest.approx(left, abs=0.05)
        assert rim < 0.35
    # a marker with nothing to fill has no other half
    fig, ax, dots = _half_scatter(2, False, fmt='x')
    assert type(dots) is PathCollection
    assert _pixels(fig).min() < 0.35


def test_a_figure_with_half_filled_hue_markers_can_be_pickled():
    fig, ax, dots = _half_scatter(2, False)
    before = _pixels(fig).copy()
    # (bytes this test has just made from its own figure)
    again = pickle.loads(pickle.dumps(fig))
    FigureCanvasAgg(again)
    assert np.array_equal(_pixels(again), before)


def test_hidden_half_filled_markers_draw_nothing():
    fig, ax, dots = _half_scatter(2, False)
    shown = _pixels(fig).copy()
    dots.set_visible(False)
    hidden = _pixels(fig)
    fig_plain, ax_plain, dots_plain = _half_scatter(2, False)
    dots_plain.remove()
    assert np.array_equal(hidden, _pixels(fig_plain))
    assert not np.array_equal(hidden, shown)
