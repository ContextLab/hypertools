"""The room hypertools reserves for text is the room matplotlib draws it in.

How tall a line of text is belongs to matplotlib and to the font: through
matplotlib 3.10 a line is as tall as the glyphs ``lp``, from 3.11 on as
tall as the font's own ascent plus descent (a third more in Noto Sans,
hypertools' default face), and 3.11 no longer places a 3-D axis label that
has no text. Three places assumed the older behavior (found on the
matplotlib 3.11.2 leg of CI, 2026-11-22):

* a caption's room was ``lines * 1.2 * font size``, so a wrapped caption
  ended below the canvas;
* a shared axis label emptied on an inner 3-D panel kept the position of
  an earlier layout, outside its panel;
* a static labeled 3-D grid whose labels' taller boxes touched was reported
  as too small for its ``size=``, with blank room left beside every panel.

Every test here draws real figures headless and reads the extents off the
real renderer (nothing is mocked or patched), and asserts the invariant
itself rather than a number that holds under one matplotlib version.
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
from hypertools.plot.plot import (CAPTION_GID, _CAPTION_PAD_IN,
                                  _OUTER_LABEL_PAD_IN, _PANEL_CUBE_FILL,
                                  _caption_text_height_in,
                                  _label_gutter_room)


def _walk(seed, rows=30, cols=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


X = _walk(0)
DATA = [_walk(seed) for seed in range(4)]
LONG = [pd.DataFrame(d, columns=['first axis', 'second axis',
                                 'third axis']) for d in DATA]
SENTENCE = ('A caption sentence that is far too long for one line of a '
            'small figure, so that it has to be wrapped to the width of '
            'the figure it is drawn under, and stay there. ')


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _caption(fig):
    found = fig.findobj(lambda a: a.get_gid() == CAPTION_GID)
    assert len(found) == 1
    return found[0]


def _drawn_box(fig, artist):
    """`artist`'s box (pixels) on a real draw of `fig`."""
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer())


# --------------------------------------------------------------------------
# a caption's room is the height its text is drawn at
# --------------------------------------------------------------------------

@pytest.mark.parametrize('font', [None, 'DejaVu Sans', 'DejaVu Serif',
                                  'DejaVu Sans Mono'])
@pytest.mark.parametrize('text', ['short', SENTENCE * 2, SENTENCE * 5,
                                  'W' * 400],
                         ids=['1-line', 'wrapped', 'long', 'one-word'])
@pytest.mark.parametrize('call', [
    dict(x=X), dict(x=X, ndims=2), dict(x=DATA, panels=(2, 2)),
    dict(x=X, animate=True, duration=1, frame_rate=4),
], ids=['3d', '2d', 'grid', 'animated'])
def test_caption_room_holds_the_drawn_text_with_its_pad(call, text, font):
    """Under the caption's last line there is the caption's own pad, in
    every font: the room is sized from the drawn text's height."""
    call = dict(call, show=False)
    if font is not None:
        call['font'] = font
    result = hyp.plot(call.pop('x'), caption=text, **call)
    fig = getattr(result, 'figure', result)
    box = _drawn_box(fig, _caption(fig))
    pad_px = _CAPTION_PAD_IN * fig.dpi
    # (one pixel for the renderer's rounding of the canvas height)
    assert box.y0 >= pad_px - 1.0, (box.y0, pad_px)
    assert box.y1 <= fig.canvas.get_width_height()[1]


@pytest.mark.parametrize('text', ['one line', 'two\nlines', SENTENCE * 3])
def test_caption_height_is_measured_as_it_is_drawn(text):
    """`_caption_text_height_in` (which draws nothing) reports the height
    the figure's real renderer then draws the same text at."""
    fig = hyp.plot(X, caption=text, show=False)
    artist = _caption(fig)
    shown = artist.get_text()
    measured = _caption_text_height_in(artist, shown)
    assert artist.get_text() == shown and artist.get_visible()
    drawn = _drawn_box(fig, artist).height / fig.dpi
    assert measured == pytest.approx(drawn, abs=0.5 / fig.dpi)


def test_caption_that_grows_over_the_frames_keeps_its_pad():
    """The sampled frames (first, middle, last) of a per-frame caption fit
    the room with the pad under them."""
    anim = hyp.plot(X, caption=lambda ctx: SENTENCE * (1 + ctx.frame),
                    animate=True, duration=1, frame_rate=5, show=False)
    fig = anim.figure
    pad_px = _CAPTION_PAD_IN * fig.dpi
    for frame in sorted({0, anim.n_frames // 2, anim.n_frames - 1}):
        anim.draw_frame(frame)
        assert _drawn_box(fig, _caption(fig)).y0 >= pad_px - 1.0, frame


# --------------------------------------------------------------------------
# an emptied shared label is where matplotlib would draw it
# --------------------------------------------------------------------------

def _emptied_labels(fig, n):
    return [(ax, axis.label) for ax in fig.axes[:n]
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
            if not axis.label.get_text()]


@pytest.mark.parametrize('call', [
    dict(panels=(2, 2)), dict(panels=(1, 4)), dict(panels=(2, 2), size=(9, 8)),
    dict(panels=(2, 2), caption='a caption'),
    dict(panels=(1, 4), animate=True, duration=1, frame_rate=4),
], ids=['2x2', '1x4', '2x2-sized', '2x2-caption', 'animated'])
def test_emptied_shared_3d_labels_sit_where_they_would_be_drawn(call):
    """Shared axis labels are drawn on the outer panels only. The labels
    emptied on the other panels report the place matplotlib draws that
    panel's label at in the finished layout, at their own panel."""
    result = hyp.plot(LONG, show=False, **call)
    fig = getattr(result, 'figure', result)
    found = _emptied_labels(fig, 4)
    assert found
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    reported = [label.get_window_extent(renderer).get_points()[0]
                for _, label in found]
    for (ax, label), where in zip(found, reported):
        box = ax.get_window_extent()
        # (a drawn label may hang a little past its panel's square where
        # nothing else is drawn; so may the place of an emptied one)
        slack = 0.06 * box.width
        assert box.x0 - slack <= where[0] <= box.x1 + slack
        assert box.y0 - slack <= where[1] <= box.y1 + slack
    # matplotlib's own placement, read off a draw with a text in each
    for _, label in found:
        label.set_text('x')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for (_, label), where in zip(found, reported):
        placed = label.get_transform().transform(label.get_position())
        assert np.allclose(placed, where, atol=0.5), (placed, where)


# --------------------------------------------------------------------------
# labels are given the blank room beside their panel before a size= is
# reported as too small
# --------------------------------------------------------------------------

def _label_boxes(fig, n):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [[axis.label.get_window_extent(renderer)
             for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
             if axis.label.get_text()] for ax in fig.axes[:n]]


def test_gutter_room_is_the_blank_room_beside_each_square():
    fig = hyp.plot(LONG, '-', panels=(1, 4), size=(8, 3), show=False)
    fig.canvas.draw()
    axes = fig.axes[:4]
    limit = 0.5 * _PANEL_CUBE_FILL
    rooms = _label_gutter_room(
        fig, [{'axes': ax} for ax in axes], limit)
    squares = [ax.bbox for ax in axes]
    width = fig.canvas.get_width_height()[0]
    pad_px = _OUTER_LABEL_PAD_IN * fig.dpi
    for i, (room, square) in enumerate(zip(rooms, squares)):
        center = 0.5 * (square.x0 + square.x1)
        for side in ('left', 'right'):
            assert room[side] >= limit
        # never past the middle of the gutter to the next square, nor
        # nearer the canvas' edge than the labels' pad
        if i + 1 < len(squares):
            middle = 0.5 * (square.x1 + squares[i + 1].x0)
            assert center + room['right'] * square.width <= middle
            assert room['right'] > 0.5
        else:
            assert (center + room['right'] * square.width
                    == pytest.approx(width - pad_px))
        if i:
            middle = 0.5 * (squares[i - 1].x1 + square.x0)
            assert center - room['left'] * square.width >= middle
        else:
            assert (center - room['left'] * square.width
                    == pytest.approx(pad_px))


def test_gutter_room_stops_at_a_shared_legend():
    fig = hyp.plot(LONG, '-', panels=(1, 4), size=(10, 3), legend=True,
                   show=False)
    fig.canvas.draw()
    axes = fig.axes[:4]
    legend = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    square = axes[-1].bbox
    room = _label_gutter_room(
        fig, [{'axes': ax} for ax in axes], 0.5 * _PANEL_CUBE_FILL)[-1]
    reach_px = 0.5 * (square.x0 + square.x1) + room['right'] * square.width
    assert reach_px <= max(legend.x0, square.x1)


@pytest.mark.parametrize('size', [(8, 3), (8.4, 3), (9, 3.2)])
def test_a_static_grid_with_room_beside_its_panels_fits_its_size(size):
    """Four labeled cubes side by side, with blank room between and
    beside their squares: no warning, no label laid out over another of
    its own panel, every label on the canvas and clear of the other
    panels' labels."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(LONG, '-', panels=(1, 4), size=size, show=False)
    assert not [str(w.message) for w in caught]
    assert tuple(fig.get_size_inches()) == tuple(float(v) for v in size)
    boxes = _label_boxes(fig, 4)
    width, height = fig.canvas.get_width_height()
    for own in boxes:
        for box in own:
            assert box.x0 >= 0 and box.x1 <= width
            assert box.y0 >= 0 and box.y1 <= height
        for a, b in itertools.combinations(own, 2):
            assert not (min(a.x1, b.x1) - max(a.x0, b.x0) > 4.0
                        and min(a.y1, b.y1) - max(a.y0, b.y0) > 4.0)
    for own, theirs in itertools.combinations(boxes, 2):
        for a, b in itertools.product(own, theirs):
            assert not a.overlaps(b)
