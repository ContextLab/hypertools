"""1.2.1 review round 5: an animation's labels are placed on every frame.

The owner's rule (2026-11-23): axis labels move with the rotation and
stay with their axes; the labels of individual datapoints have correct
positioning in every frame, may shift from frame to frame but must not
wobble back and forth; legends, titles and other static text are laid
out once, for the worst case over all frames, and then do not move.

A. On every frame the datapoint labels are at their points, off the
   legend, the titles, the suptitle and the caption, and on the canvas;
   they cover each other far less than labels left at the usual place.
B. They do not wobble: no label's offset turns back within the time of
   one glide, and a label never jumps (it glides) unless a glide would
   carry it over static content.
C. A frame is the same picture whatever was drawn before it, in a new
   identical call and in a saved file.
D. Static content has the same extent on every frame, and a 3-D plot's
   axis labels turn with the cube without leaving the canvas or
   touching static content.

Real figures and real frames, rendered headless; saved files are read
back. Nothing is mocked, and no assertion depends on a font's absolute
metrics: boxes are measured where they were drawn and compared with
each other.
"""
import itertools
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.animation import PillowWriter                   # noqa: E402
from PIL import Image                                           # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot import matplotlib_backend as backend       # noqa: E402
from hypertools.plot.matplotlib_backend import label_tracks     # noqa: E402


def _walk(seed, rows=40, cols=3):
    return np.random.default_rng(seed).normal(size=(rows, cols)).cumsum(0)


CLOUD = np.random.default_rng(3).standard_normal((30, 3))
NAMES = [f'Subject {i}' for i in range(len(CLOUD))]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, show=False, **kwargs)


def _spin8(**kwargs):
    return _quiet(CLOUD[:8], 'o', labels=NAMES[:8], animate='spin',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _spin12_dressed(**kwargs):
    return _quiet([CLOUD[:6], CLOUD[6:12]], 'o',
                  labels=[NAMES[:6], NAMES[6:12]], animate='spin',
                  legend=['first', 'second'], title='A title',
                  suptitle='Above it all', caption='A caption under it',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _spin30_dressed(**kwargs):
    return _quiet([CLOUD[:15], CLOUD[15:]], 'o',
                  labels=[NAMES[:15], NAMES[15:]], animate='spin',
                  legend=['first', 'second'], title='A title',
                  caption='A caption under it',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _window3d(**kwargs):
    return _quiet(CLOUD, 'o', labels=NAMES, animate='window',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _window2d(**kwargs):
    data = [_walk(1, cols=2), _walk(2, cols=2)]
    names = [[f'{letter}{k}' if k % 3 == 0 else None for k in range(40)]
             for letter in 'ab']
    return _quiet(data, '-o', labels=names, legend=['a', 'b'],
                  title='Two walks', animate='window',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _serial2d(**kwargs):
    return _quiet(CLOUD[:, :2], 'o', labels=NAMES, animate='serial',
                  title='One at a time',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


def _panels(**kwargs):
    data = [np.random.default_rng(seed).normal(size=(12, 3))
            for seed in range(4)]
    names = [[f'p{seed}-{i}' for i in range(12)] for seed in range(4)]
    return _quiet(data, 'o', panels=(2, 2), labels=names,
                  legend=list('abcd'), title=list('ABCD'), suptitle='S',
                  caption='A caption', animate='spin',
                  **{'duration': 4, 'frame_rate': 15, **kwargs})


ANIMATIONS = {
    'spin, 8 labels': _spin8,
    'spin, 12 labels, legend, title, suptitle, caption': _spin12_dressed,
    'spin, 30 labels, legend, title, caption': _spin30_dressed,
    'window, 3-D': _window3d,
    'window, 2-D, two lines': _window2d,
    'serial, 2-D': _serial2d,
    '2 x 2 panels': _panels,
}
#: the animations whose points pass each other on screen (the cube
#: turns), with few enough labels that every label has a free place
PASSING = ['spin, 30 labels, legend, title, caption',
           '2 x 2 panels']


def _labeled_axes(fig):
    return [ax for ax in fig.axes
            if getattr(ax, '_hyp_labels_and_points', None)]


def _labels(ax):
    return [entry[0] for entry in ax._hyp_labels_and_points]


def _static_artists(fig):
    """Everything that must not move during an animation: legends,
    figure texts (suptitle, caption) and the panels' titles."""
    found = list(fig.legends) + [text for text in fig.texts
                                 if text.get_text().strip()]
    for ax in fig.axes:
        if ax.get_legend() is not None:
            found.append(ax.get_legend())
        if ax.title.get_text().strip():
            found.append(ax.title)
    return found


def _show(anim, frame):
    anim.draw_frame(int(frame))
    anim.figure.canvas.draw()
    return anim.figure.canvas.get_renderer()


def _pixels(anim, frame):
    _show(anim, frame)
    return np.asarray(anim.figure.canvas.buffer_rgba()).copy()


def _census(anim):
    """Over every frame: pairs of label boxes of one axes that overlap,
    label boxes on static content, label boxes that leave the canvas,
    and how many label boxes were looked at."""
    fig = anim.figure
    counts = dict(pairs=0, on_static=0, off_canvas=0, boxes=0)
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        static = [artist.get_window_extent(renderer)
                  for artist in _static_artists(fig)]
        for ax in _labeled_axes(fig):
            boxes = [label.get_bbox_patch().get_window_extent(renderer)
                     for label in _labels(ax) if label.get_visible()]
            counts['boxes'] += len(boxes)
            counts['pairs'] += sum(
                1 for one, other in itertools.combinations(boxes, 2)
                if one.overlaps(other))
            counts['on_static'] += sum(
                1 for box in boxes for other in static
                if box.overlaps(other))
            # (half a pixel: a box edge is drawn across a pixel)
            counts['off_canvas'] += sum(
                1 for box in boxes
                if box.x0 < -0.5 or box.y0 < -0.5
                or box.x1 > fig.bbox.width + 0.5
                or box.y1 > fig.bbox.height + 0.5)
    return counts


def _leave_labels_at_the_usual_place(anim):
    """Replace every table of `anim` by one that leaves each label up
    and to the left of its point on every frame (where 1.2.0 drew it)."""
    home = backend._LABEL_CANDIDATES.index(backend._LABEL_HOME)
    for ax in _labeled_axes(anim.figure):
        tracks = label_tracks(ax)
        sizes = np.array([backend._label_size_pt(ref())
                          for ref in tracks['labels']])
        usual = backend._label_center_offsets(
            np.full(tracks['places'].shape, home), sizes)
        tracks['offsets'] = np.where(tracks['shown'][..., None], usual,
                                     np.nan)


def _turns_back(track, window, tolerance=1.0):
    """How many times the series `track` (one component of a label's
    offset, in points, NaN while the label is hidden) goes more than
    `tolerance` one way and then more than `tolerance` back, both within
    `window` frames of the turn."""
    count, inside = 0, False
    for t in range(1, len(track) - 1):
        before = track[max(0, t - window):t]
        after = track[t + 1:t + 1 + window]
        turn = False
        if not (np.isnan(track[t]) or np.isnan(before).any()
                or np.isnan(after).any()):
            peak = (track[t] - before.min() > tolerance
                    and track[t] - after.min() > tolerance)
            trough = (before.max() - track[t] > tolerance
                      and after.max() - track[t] > tolerance)
            turn = peak or trough
        count += turn and not inside
        inside = turn
    return count


def _wobbles(anim):
    """Turns back within one glide's time, summed over every label and
    both components; and the largest change of an offset between two
    frames, in points, outside the frames a label was made to jump."""
    turns, step = 0, 0.0
    for ax in _labeled_axes(anim.figure):
        tracks = label_tracks(ax)
        offsets, window = tracks['offsets'], tracks['ease_frames']
        if tracks['periodic']:
            # (round the loop: the last frame leads into the first)
            offsets = np.concatenate([offsets, offsets[:window + 1]])
        for k in range(offsets.shape[1]):
            for component in range(2):
                turns += _turns_back(offsets[:, k, component], window)
        moved = np.abs(np.diff(offsets, axis=0)).max(axis=2)
        forced = tracks['forced']
        if tracks['periodic']:
            forced = np.concatenate([forced, forced[:window + 1]])
        moved[forced[1:] | forced[:-1]] = np.nan
        if np.isfinite(moved).any():
            step = max(step, float(np.nanmax(moved)))
    return turns, step


# --------------------------------------------------------------------------
# A. every frame's labels are placed for that frame

@pytest.mark.parametrize('name', list(ANIMATIONS))
def test_labels_keep_off_static_content_and_on_the_canvas_every_frame(name):
    anim = ANIMATIONS[name]()
    assert anim.n_frames == 60
    counts = _census(anim)
    assert counts['boxes'] > 60
    assert counts['on_static'] == 0
    assert counts['off_canvas'] == 0


def _pairs_per_frame(anim):
    """Per frame, how many pairs of label boxes of one axes overlap, as
    the frame is drawn."""
    fig = anim.figure
    pairs = []
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        count = 0
        # (all the figure's labels together: in a grid a panel's labels
        # must keep off the next panel's too)
        for boxes in [[label.get_bbox_patch().get_window_extent(renderer)
                       for ax in _labeled_axes(fig)
                       for label in _labels(ax) if label.get_visible()]]:
            count += sum(1 for one, other in itertools.combinations(boxes, 2)
                         if one.overlaps(other))
        pairs.append(count)
    return np.array(pairs)


@pytest.mark.parametrize('name', [
    'spin, 8 labels',
    'spin, 12 labels, legend, title, suptitle, caption',
    'window, 3-D',
    'window, 2-D, two lines',
    'serial, 2-D'])
def test_no_two_labels_overlap_on_any_frame(name):
    """With a dozen labels on a plot (the sliding windows show fewer at
    a time) no label's box is on another's on any frame, glides from
    place to place included. Review round 5 had 20 to 30 frames of 60
    with stacked labels in the spins (a label was left under another
    for up to a second and a half rather than moved)."""
    anim = ANIMATIONS[name]()
    pairs = _pairs_per_frame(anim)
    assert len(pairs) == 60
    assert pairs.max() == 0, pairs.tolist()


def test_a_grids_labels_overlap_on_few_frames():
    """A 2 x 2 grid with 12 labels a panel (48 on one default-size
    figure, each panel's kept out of the other panels and off the other
    panels' labels) has less room than a single plot: at most two pairs
    of labels overlap at a time, on at most a quarter of the frames
    (measured: one pair on 4 frames of 60 with matplotlib 3.10, whose
    text is narrower, more with 3.11; review round 5 had pairs on 55
    frames, up to 7 at once)."""
    anim = _panels()
    pairs = _pairs_per_frame(anim)
    assert pairs.max() <= 2, pairs.tolist()
    assert (pairs > 0).sum() <= 15, pairs.tolist()


def test_thirty_labels_overlap_less_than_stills_of_each_frame():
    """30 labels on a default-size cube that turns once in four seconds
    are more than can be kept apart on every frame without hopping, and
    the bound is what the rule of a static figure reaches when it places
    each frame on its own, free to move every label on every frame
    (4 to 10 pairs a frame, 5.6 on average, with 164 turn-backs): the
    planned labels have at most one pair more than that on any frame,
    never more than its worst frame, under a third as many in all, and
    no pair at all on a quarter of the frames (measured: 0.9 a frame,
    at most 3, no pair on 23 frames of 60 with matplotlib 3.10; 1.3 a
    frame and at most 6 with the wider text of 3.11)."""
    anim = ANIMATIONS['spin, 30 labels, legend, title, caption']()
    planned = _pairs_per_frame(anim)
    for ax in _labeled_axes(anim.figure):
        backend.solve_label_tracks(ax, _independent=True)
    stills = _pairs_per_frame(anim)
    assert stills.sum() > 120
    assert (planned <= stills + 1).all(), (planned - stills).tolist()
    assert planned.max() <= stills.max(), planned.tolist()
    assert planned.sum() <= stills.sum() / 3, (planned.sum(), stills.sum())
    assert (planned == 0).sum() >= 15


@pytest.mark.parametrize('name, share', [
    ('spin, 8 labels', 0.25),
    ('spin, 12 labels, legend, title, suptitle, caption', 0.25),
    ('spin, 30 labels, legend, title, caption', 0.25),
    ('window, 3-D', 0.25),
    ('2 x 2 panels', 0.25)])
def test_labels_cover_each_other_far_less_than_at_the_usual_place(
        name, share):
    """Summed over all frames, the pairs of labels that overlap are a
    fraction of what they are with every label left up and to the left
    of its point (1.2.0, and what a label placed once for frame 0 drifts
    back toward as the cube turns)."""
    anim = ANIMATIONS[name]()
    placed = _census(anim)['pairs']
    _leave_labels_at_the_usual_place(anim)
    usual = _census(anim)['pairs']
    assert usual > 0
    assert placed <= share * usual, (placed, usual)


def test_a_label_is_drawn_where_the_table_has_it():
    """The table's offsets are what is drawn: on every frame the middle
    of each label's box is that far from the label's point."""
    anim = _spin8()
    fig = anim.figure
    (ax,) = _labeled_axes(fig)
    tracks = label_tracks(ax)
    scale = fig.dpi / 72.0
    for frame in range(0, anim.n_frames, 7):
        renderer = _show(anim, frame)
        for k, label in enumerate(_labels(ax)):
            box = label.get_bbox_patch().get_window_extent(renderer)
            anchor = ax.transData.transform(label.xy)
            drawn = (0.5 * (box.x0 + box.x1) - anchor[0],
                     0.5 * (box.y0 + box.y1) - anchor[1])
            np.testing.assert_allclose(
                drawn, tracks['offsets'][frame, k] * scale, atol=1.5)


def test_labels_follow_their_points_as_the_cube_turns():
    """Each label's arrow ends on its point's projection in the frame's
    own view, and the points do move."""
    from mpl_toolkits.mplot3d import proj3d
    anim = _spin8()
    (ax,) = _labeled_axes(anim.figure)
    seen = []
    for frame in (0, 15, 30, 45):
        _show(anim, frame)
        spots = []
        for label, x, y, z in ax._hyp_labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            np.testing.assert_allclose(
                ax.transData.transform(label.xy),
                ax.transData.transform((x2, y2)), atol=0.5)
            spots.append(ax.transData.transform(label.xy))
        seen.append(np.array(spots))
    assert np.abs(seen[0] - seen[2]).max() > 20.0


@pytest.mark.parametrize('make', [_window3d, _window2d, _serial2d])
def test_labels_of_points_not_shown_are_hidden(make):
    """A label is shown on exactly the frames the table has an offset
    for it, some labels are hidden on some frames, and a label that has
    come into view is at a real place."""
    anim = make()
    (ax,) = _labeled_axes(anim.figure)
    tracks = label_tracks(ax)
    assert not tracks['shown'].all() and tracks['shown'].any()
    for frame in range(0, anim.n_frames, 5):
        _show(anim, frame)
        visible = [label.get_visible() for label in _labels(ax)]
        assert visible == list(tracks['shown'][frame])
        assert np.array_equal(np.isfinite(tracks['offsets'][frame, :, 0]),
                              tracks['shown'][frame])


# --------------------------------------------------------------------------
# B. no wobble

@pytest.mark.parametrize('name', list(ANIMATIONS))
def test_no_label_turns_back_within_a_glide(name):
    """Points pass each other as the cube turns, so labels have to
    change places; no label's offset goes one way and then back by more
    than a point within the time of one glide (round the loop too, for
    an animation that loops). Placing every frame on its own, which is
    what the rule would be without the table, does turn back."""
    anim = ANIMATIONS[name]()
    turns, _step = _wobbles(anim)
    assert turns == 0
    if name in PASSING:
        moved = sum(int((label_tracks(ax)['places'][1:]
                         != label_tracks(ax)['places'][:-1]).sum())
                    for ax in _labeled_axes(anim.figure))
        assert moved > 0                    # (labels did change places)
        for ax in _labeled_axes(anim.figure):
            backend.solve_label_tracks(ax, _independent=True)
        alone, _step = _wobbles(anim)
        assert alone > 0


@pytest.mark.parametrize('name', PASSING)
def test_a_label_glides_to_its_next_place(name):
    """Between two frames a label moves by no more than 0.35 of the way
    from one place to the next (a glide takes at least five frames),
    where placing every frame on its own jumps the whole way at once."""
    anim = ANIMATIONS[name]()
    _turns, step = _wobbles(anim)
    jump = 0.0
    for ax in _labeled_axes(anim.figure):
        tracks = label_tracks(ax)
        assert tracks['ease_frames'] >= 5
        sizes = np.array([backend._label_size_pt(ref())
                          for ref in tracks['labels']])
        targets = backend._label_center_offsets(
            tracks['places'], sizes, tracks['candidates'])
        change = np.abs(np.diff(targets, axis=0)).max(axis=2)
        jump = max(jump, float(change.max()))
    assert jump > 20.0                      # (some label changed places)
    assert 0.0 < step <= 0.35 * jump, (step, jump)
    for ax in _labeled_axes(anim.figure):
        backend.solve_label_tracks(ax, _independent=True)
    _turns, alone = _wobbles(anim)
    assert alone > 2.0 * step


def test_a_looping_spin_has_no_seam():
    """A spin of whole turns leads from its last frame back into its
    first: the labels are placed round the loop, so no label is further
    from where it is on frame 0 than one frame's glide."""
    anim = _spin8()
    (ax,) = _labeled_axes(anim.figure)
    tracks = label_tracks(ax)
    assert tracks['periodic']
    offsets = tracks['offsets']
    seam = np.abs(offsets[0] - offsets[-1]).max()
    inside = np.abs(np.diff(offsets, axis=0)).max()
    assert seam <= inside + 1e-9
    # ...and a reveal, whose last frame shows everything and whose first
    # shows one point, is not taken for a loop
    (window_ax,) = _labeled_axes(_window3d().figure)
    assert not label_tracks(window_ax)['periodic']


# --------------------------------------------------------------------------
# C. a frame is a function of the animation and its index

class _KeepFrames(PillowWriter):
    """Pillow's own gif writer, which also keeps the frames it was given
    so they can be compared with frames drawn one at a time."""

    def finish(self):
        self.kept = [np.asarray(frame.convert('RGBA')).copy()
                     for frame in self._frames]
        super().finish()


@pytest.mark.parametrize('name', ['spin, 8 labels',
                                  'spin, 12 labels, legend, title, '
                                  'suptitle, caption',
                                  'window, 3-D', 'window, 2-D, two lines',
                                  '2 x 2 panels'])
def test_a_frame_is_the_same_picture_however_it_is_reached(name, tmp_path):
    """Frame k drawn in order, in a shuffled order, backwards, as the
    only frame ever drawn of a second identical call, and as handed to
    the writer of a saved gif: the same pixels."""
    make = ANIMATIONS[name]
    anim = make(duration=2, frame_rate=10)
    n = anim.n_frames
    forward = [_pixels(anim, i) for i in range(n)]
    assert any(not np.array_equal(forward[0], other)
               for other in forward[1:])
    order = list(np.random.default_rng(0).permutation(n)) \
        + list(range(n))[::-1]
    for i in order:
        assert np.array_equal(_pixels(anim, i), forward[i]), i
    for k in (n // 2, n - 1, 0):
        fresh = make(duration=2, frame_rate=10)
        assert np.array_equal(_pixels(fresh, k), forward[k]), k
        plt.close(fresh.figure)
    _show(anim, n // 3)
    writer = _KeepFrames(fps=10)
    path = tmp_path / 'saved.gif'
    anim.animation.save(str(path), writer=writer)
    assert len(writer.kept) == n
    for i, saved in enumerate(writer.kept):
        assert np.array_equal(saved, forward[i]), i
    with Image.open(path) as saved:
        assert saved.n_frames == n
        saved.seek(n // 2)
        middle = np.asarray(saved.convert('RGBA')).astype(float)
    # (a gif stores 256 colors, so the file is compared loosely)
    assert np.abs(middle - forward[n // 2]).mean() < 4.0
    assert (np.abs(middle - forward[n // 2]).mean()
            < np.abs(middle - forward[0]).mean())


def test_the_table_is_the_same_whenever_it_is_worked_out():
    """Worked out again after frames were drawn in any order, the table
    is the same, number for number."""
    anim = _spin12_dressed()
    (ax,) = _labeled_axes(anim.figure)
    first = label_tracks(ax)['offsets'].copy()
    for frame in (41, 3, 59, 17):
        _show(anim, frame)
    again = backend.solve_label_tracks(ax)['offsets']
    assert np.array_equal(first, again, equal_nan=True)
    twin = _spin12_dressed()
    (twin_ax,) = _labeled_axes(twin.figure)
    assert np.array_equal(first, label_tracks(twin_ax)['offsets'],
                          equal_nan=True)


def test_a_resized_figure_has_its_labels_placed_again():
    """The table is made for the figure's size: after the figure is
    given another size the next frame works it out again, and the
    labels are on the canvas and off static content at the new size."""
    anim = _spin12_dressed()
    fig = anim.figure
    (ax,) = _labeled_axes(fig)
    before = label_tracks(ax)
    width, height = fig.get_size_inches()
    fig.set_size_inches(0.8 * width, 0.8 * height)
    _show(anim, 5)
    assert label_tracks(ax) is not before
    counts = _census(anim)
    assert counts['off_canvas'] == 0 and counts['on_static'] == 0


# --------------------------------------------------------------------------
# D. static content is laid out once; axis labels turn with the cube

def _layout(fig, renderer):
    extents = [tuple(np.round(artist.get_window_extent(renderer).extents, 3))
               for artist in _static_artists(fig)]
    boxes = [tuple(np.round(ax.get_position().bounds, 9))
             for ax in fig.axes]
    zooms = [tuple(np.round(ax.get_box_aspect(), 9))
             for ax in fig.axes if hasattr(ax, 'get_proj')]
    return extents, boxes, zooms, tuple(fig.get_size_inches())


@pytest.mark.parametrize('name', list(ANIMATIONS))
def test_static_content_is_the_same_on_every_frame(name):
    """Legends, titles, suptitle, caption, every axes' box, every cube's
    size and the figure's size: identical on all frames."""
    anim = ANIMATIONS[name]()
    fig = anim.figure
    first = _layout(fig, _show(anim, 0))
    assert len(first[0]) + len(first[1]) >= 1
    for frame in range(1, anim.n_frames):
        assert _layout(fig, _show(anim, frame)) == first, frame


def _axis_texts(ax):
    return [axis3d.label for axis3d in ax._axis_map.values()
            if axis3d.label.get_text().strip()]


AXIS_LABELED = {
    'landscape': dict(),
    'square, legend': dict(size=(5, 5), legend=['first', 'second']),
    'square, legend inside': dict(
        size=(5, 5), legend=['first', 'second'],
        legend_kwargs={'loc': 'upper left'}),
    'legend, title, suptitle, caption': dict(
        legend=['first', 'second'], title='A title', suptitle='Above',
        caption='A caption under the plot'),
    'colorbar': dict(hue=[np.linspace(0, 1, 15)] * 2, colorbar=True),
    'two turns back': dict(rotations=-2),
}


@pytest.mark.parametrize('name', list(AXIS_LABELED))
def test_axis_labels_turn_with_the_cube_inside_the_canvas(name):
    """A single animated 3-D plot's axis labels were fitted for no view
    at all: on a square figure "the z axis" was 61 px off the canvas for
    part of the turn, and with a legend, a title and a caption the
    labels crossed them on 6 frames of 60. On EVERY frame each axis
    label is on the canvas and off the static content, the static
    content does not move, and the labels do move with the cube."""
    anim = _quiet([CLOUD[:15], CLOUD[15:]], 'o', animate='spin',
                  xlabel='the x axis', ylabel='the y axis',
                  zlabel='the z axis', duration=4, frame_rate=15,
                  **AXIS_LABELED[name])
    fig = anim.figure
    ax = fig.axes[0]
    width, height = fig.bbox.width, fig.bbox.height
    first, places = None, []
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        layout = _layout(fig, renderer)
        first = layout if first is None else first
        assert layout == first, frame
        static = [artist.get_window_extent(renderer)
                  for artist in _static_artists(fig)]
        static += [other.get_tightbbox(renderer) for other in fig.axes
                   if other is not ax]
        texts = _axis_texts(ax)
        assert len(texts) == 3
        boxes = [text.get_window_extent(renderer) for text in texts]
        places.append([box.extents for box in boxes])
        for text, box in zip(texts, boxes):
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= width \
                and box.y1 <= height, (frame, text.get_text(), box.extents)
            assert not any(box.overlaps(other) for other in static), (
                frame, text.get_text())
    places = np.array(places)
    assert np.ptp(places[:, :, 0], axis=0).min() > 5.0


def test_datapoint_labels_keep_off_the_axis_labels_as_they_turn():
    """The cube's axis labels hang where the labels of points near its
    edges want to go: on every frame no datapoint label covers one."""
    anim = _quiet(CLOUD[:12], 'o', labels=NAMES[:12], animate='spin',
                  xlabel='the x axis', ylabel='the y axis',
                  zlabel='the z axis', legend=['data'], title='A title',
                  duration=4, frame_rate=15)
    fig = anim.figure
    (ax,) = _labeled_axes(fig)
    covered = glided = 0
    tracks = label_tracks(ax)
    half = tracks['ease_frames']
    moving = (tracks['places'] != np.roll(tracks['places'], half, axis=0))
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        texts = [text.get_window_extent(renderer)
                 for text in _axis_texts(ax)]
        for k, label in enumerate(_labels(ax)):
            box = label.get_bbox_patch().get_window_extent(renderer)
            if any(box.overlaps(text) for text in texts):
                # (a label gliding from one place to the next may pass
                # an axis label that is itself turning)
                covered += not moving[frame, k]
                glided += bool(moving[frame, k])
    assert covered == 0
    assert glided <= 12


def test_a_label_does_not_reach_into_the_next_panel():
    """A label that reached into the next panel of a grid was painted
    over by that panel's background, cut off at its edge. On every frame
    each label's box stays off every other panel."""
    anim = _panels()
    fig = anim.figure
    panels = _labeled_axes(fig)
    assert len(panels) == 4
    inside = 0
    for frame in range(0, anim.n_frames, 3):
        renderer = _show(anim, frame)
        for ax in panels:
            others = [other.get_window_extent(renderer)
                      for other in panels if other is not ax]
            for label in _labels(ax):
                box = label.get_bbox_patch().get_window_extent(renderer)
                inside += any(box.overlaps(other) for other in others)
    assert inside == 0


@pytest.mark.parametrize('dressed', [False, True])
def test_a_grids_axis_labels_stay_on_the_canvas_and_apart_every_frame(
        dressed):
    """An animated grid's cubes are fitted with their axis labels at a
    sample of the views (every 30 degrees): checked here on EVERY frame,
    no axis label leaves the canvas, lies on static content or meets a
    label of another panel."""
    data = [np.random.default_rng(seed).normal(size=(12, 3))
            for seed in range(4)]
    extra = dict(legend=list('abcd'), title=list('ABCD'), suptitle='S',
                 caption='A caption') if dressed else {}
    anim = _quiet(data, 'o', panels=(2, 2), xlabel='the x axis',
                  ylabel='the y axis', zlabel='the z axis', animate='spin',
                  duration=4, frame_rate=15, **extra)
    fig = anim.figure
    panels = [ax for ax in fig.axes if hasattr(ax, 'get_proj')]
    first = None
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        layout = _layout(fig, renderer)
        first = layout if first is None else first
        assert layout == first, frame
        static = [artist.get_window_extent(renderer)
                  for artist in _static_artists(fig)]
        boxes = [[text.get_window_extent(renderer)
                  for text in _axis_texts(ax) if text.get_visible()]
                 for ax in panels]
        assert sum(map(len, boxes)) >= 3
        for k, own in enumerate(boxes):
            for box in own:
                assert box.x0 >= 0 and box.y0 >= 0 \
                    and box.x1 <= fig.bbox.width \
                    and box.y1 <= fig.bbox.height, frame
                assert not any(box.overlaps(other) for other in static), frame
                assert not any(box.overlaps(other)
                               for others in boxes[k + 1:]
                               for other in others), frame


# --------------------------------------------------------------------------
# Third pass: what a source review of the planner found (memory, a leak,
# pickled figures, a plan gone stale), and edge inputs

def _table_bytes(tracks):
    return sum(value.nbytes for value in tracks.values()
               if isinstance(value, np.ndarray))


def test_more_labels_than_are_arranged_store_nothing_per_frame():
    """Above 300 labels the labels stay at their usual places. The table
    had a row for every frame all the same (58 bytes a label and frame:
    13.9 MB for 400 labels on 600 frames, more than 10 GB for 10000
    labels on 18000) and every frame was run to fill it. It has one
    row, whatever the number of frames, and working it out again
    allocates next to nothing."""
    import tracemalloc
    rng = np.random.default_rng(3)
    names = [f'S{i}' for i in range(400)]
    sizes = {}
    for duration in (2, 20):
        anim = _quiet(rng.normal(size=(400, 3)), 'o', labels=names,
                      animate='spin', duration=duration, frame_rate=30)
        ax = _labeled_axes(anim.figure)[0]
        tracks = label_tracks(ax)
        assert tracks['n_frames'] == anim.n_frames == 30 * duration
        assert tracks['offsets'].shape == (1, 400, 2)
        sizes[duration] = _table_bytes(tracks)
        tracemalloc.start()
        backend.solve_label_tracks(ax)
        _now, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 2_000_000, peak
        # every frame shows every label at its usual place
        home = backend._label_center_offsets(
            np.zeros(400, dtype=int),
            np.array([backend._label_size_pt(label)
                      for label in _labels(ax)]))
        for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
            anim.draw_frame(frame)
            assert np.allclose([label.xyann for label in _labels(ax)],
                               home, atol=1e-4)
        plt.close(anim.figure)
    assert sizes[2] == sizes[20] <= 400 * 28


def test_a_long_animation_is_planned_at_a_bounded_number_of_frames():
    """30 labels on 300 frames are more labels x frames than are planned
    one by one (`_LABEL_TRACK_BUDGET`): 120 evenly spaced frames are
    planned, the table is no larger than 28 bytes a label and planned
    frame however long the animation, and between two planned frames a
    label is drawn between its offsets on the two (the same picture
    whichever frames were drawn before). Planning every frame took time
    in proportion to the frames: 30 CPU seconds for 60 labels on 600
    frames, against 1.6 on 60."""
    anim = _quiet([CLOUD[:15], CLOUD[15:]], 'o',
                  labels=[NAMES[:15], NAMES[15:]], animate='spin',
                  duration=10, frame_rate=30)
    assert anim.n_frames == 300
    ax = _labeled_axes(anim.figure)[0]
    tracks = label_tracks(ax)
    keys = tracks['frames']
    assert len(keys) == backend._LABEL_TRACK_BUDGET // 30 == 120
    assert keys[0] == 0 and keys[-1] == 299
    assert tracks['offsets'].shape == (120, 30, 2)
    assert tracks['offsets'].dtype == np.float32
    assert _table_bytes(tracks) <= 28 * 30 * 120 + keys.nbytes
    labels = _labels(ax)
    # a planned frame shows its row; a frame between two is between them
    low, high = int(keys[40]), int(keys[41])
    assert high - low >= 2
    seen = {}
    for frame in (high, low + 1, low, low + 1):
        anim.draw_frame(frame)
        seen.setdefault(frame, []).append(
            np.array([label.xyann for label in labels]))
    assert np.allclose(seen[low][0], tracks['offsets'][40], atol=1e-4)
    assert np.allclose(seen[high][0], tracks['offsets'][41], atol=1e-4)
    assert np.array_equal(seen[low + 1][0], seen[low + 1][1])
    share = 1.0 / (high - low)
    assert np.allclose(
        seen[low + 1][0],
        (1 - share) * tracks['offsets'][40] + share * tracks['offsets'][41],
        atol=1e-3)
    turns, _step = _wobbles(anim)
    assert turns == 0
    # and the labels are still kept apart on the frames between
    pairs = []
    for frame in range(0, 300, 7):
        renderer = _show(anim, frame)
        boxes = [label.get_bbox_patch().get_window_extent(renderer)
                 for label in labels]
        pairs.append(sum(1 for one, other in itertools.combinations(boxes, 2)
                         if one.overlaps(other)))
    assert np.mean(pairs) <= 2.0, pairs


def test_closed_figures_with_labels_are_freed():
    """The labels' state held the animation's frame function, which
    refers to the animation and its axes, as the value of a dictionary
    keyed weakly by that axes: the key never died, and every closed
    animated figure with labels stayed allocated with its table. (And
    the last static figure with labels was kept by a module global.)"""
    import gc
    import weakref
    gc.collect()
    before = len(backend._LABEL_STATES)
    alive = []
    for make in (_spin8, _panels, _window2d):
        for _again in range(2):
            anim = make(duration=1, frame_rate=10)
            anim.draw_frame(3)
            anim.figure.canvas.draw()
            alive.append(weakref.ref(anim.figure))
            alive.extend(weakref.ref(ax) for ax in anim.figure.axes)
            plt.close(anim.figure)
            del anim
    for _again in range(2):
        fig = _quiet(CLOUD[:8], 'o', labels=NAMES[:8])
        fig.canvas.draw()
        alive.append(weakref.ref(fig))
        plt.close(fig)
        del fig
    gc.collect()
    gc.collect()
    assert [ref() for ref in alive if ref() is not None] == []
    assert len(backend._LABEL_STATES) == before


def test_an_animation_outlives_nothing_it_needs():
    """The frame function is held weakly, by the labels' state; the
    animation keeps it, so the plan can still be made again while the
    animation lives (a resized figure), garbage collection or not."""
    import gc
    anim = _spin8()
    gc.collect()
    ax = _labeled_axes(anim.figure)[0]
    first = label_tracks(ax)
    anim.figure.set_size_inches(9, 7)
    anim.draw_frame(5)
    assert label_tracks(ax) is not first
    assert label_tracks(ax)['offsets'].shape == first['offsets'].shape


@pytest.mark.parametrize('how', ['pickle', 'deepcopy'])
@pytest.mark.parametrize('kind', ['animated', 'static'])
def test_a_copied_or_unpickled_figure_draws(kind, how, tmp_path):
    """`pickle.loads(pickle.dumps(anim.figure)).canvas.draw()` raised
    ``TypeError: cannot unpack non-iterable NoneType object``: a planned
    label has no named place, and the copy has no plan. A figure read
    back from a pickle, or copied, has no animation to follow: its
    labels stay exactly where the frame it was copied at had them, it
    draws, and it saves. A static figure's labels are arranged as
    before."""
    import copy
    import pickle
    if kind == 'animated':
        anim = _spin12_dressed()
        fig = anim.figure
        anim.draw_frame(17)
    else:
        fig = _quiet(CLOUD[:12], 'o', labels=NAMES[:12])
    fig.canvas.draw()
    was = [(label.get_text(), tuple(np.round(label.xyann, 4)),
            label.get_ha(), label.get_va())
           for label in _labels(_labeled_axes(fig)[0])]
    new = (pickle.loads(pickle.dumps(fig)) if how == 'pickle'
           else copy.deepcopy(fig))
    new.canvas.draw()
    new.savefig(tmp_path / 'restored.png')
    new.canvas.draw()
    assert (tmp_path / 'restored.png').stat().st_size > 5000
    ax = _labeled_axes(new)[0]
    assert label_tracks(ax) is None
    now = [(label.get_text(), tuple(np.round(label.xyann, 4)),
            label.get_ha(), label.get_va()) for label in _labels(ax)]
    assert now == was
    renderer = new._get_renderer()
    boxes = [label.get_bbox_patch().get_window_extent(renderer)
             for label in _labels(ax)]
    assert not any(one.overlaps(other)
                   for one, other in itertools.combinations(boxes, 2))
    plt.close(new)


def test_a_text_added_to_the_figure_is_planned_for():
    """The plan was made again only for another figure size, axes box or
    set of labels: a ``fig.text`` put on top of a label afterwards was
    not kept clear on later frames. What the labels are kept off is
    part of what the plan was made for; the plan is made again once,
    and not again while the figure stays as it is."""
    anim = _spin8()
    fig = anim.figure
    ax = _labeled_axes(fig)[0]
    renderer = _show(anim, 3)
    label = _labels(ax)[0]
    box = label.get_bbox_patch().get_window_extent(renderer)
    first = label_tracks(ax)
    note = fig.text(0.5 * (box.x0 + box.x1) / fig.bbox.width,
                    0.5 * (box.y0 + box.y1) / fig.bbox.height,
                    'A NOTE ADDED LATER', ha='center', va='center')
    renderer = _show(anim, 3)
    second = label_tracks(ax)
    assert second is not first
    for frame in range(anim.n_frames):
        renderer = _show(anim, frame)
        there = note.get_window_extent(renderer)
        assert not any(
            one.get_bbox_patch().get_window_extent(renderer).overlaps(there)
            for one in _labels(ax)), frame
    assert label_tracks(ax) is second       # (planned once)
    _turns, _step = _wobbles(anim)
    assert _turns == 0


@pytest.mark.parametrize('change', ['fontsize', 'text', 'long text'])
def test_a_label_made_larger_is_planned_for(change):
    """`set_fontsize` or `set_text` on a label after the plan was made
    left the old offsets in place, so the larger labels lay on each
    other. The size of every label is part of what the plan was made
    for. (Eight names of 24 characters, each a quarter of the canvas
    wide, are planned with no overlap under matplotlib 3.10; under
    3.11's wider text one glide passes over a resting label on one
    frame of 60, the bound here.)"""
    anim = _spin8()
    ax = _labeled_axes(anim.figure)[0]
    _show(anim, 2)
    first = label_tracks(ax)
    for label in _labels(ax):
        if change == 'fontsize':
            label.set_fontsize(15)
        elif change == 'text':
            label.set_text(label.get_text() + ' (b)')
        else:
            label.set_text(label.get_text() + ', a longer name')
    _show(anim, 3)
    second = label_tracks(ax)
    assert second is not first
    pairs = _pairs_per_frame(anim)
    if change == 'long text':
        assert pairs.max() <= 1 and (pairs > 0).sum() <= 1
    else:
        assert pairs.max() == 0
    assert label_tracks(ax) is second
    assert _census(anim)['off_canvas'] == 0


def test_a_figure_that_changes_every_frame_is_not_planned_every_frame():
    """A caller who rewrites a figure text on every frame (a counter)
    changes what the labels are kept off on every frame; the plan is
    made again for the first two changes in a row and then waits until
    the figure stops changing, when it is made once more."""
    anim = _spin8()
    fig = anim.figure
    ax = _labeled_axes(fig)[0]
    counter = fig.text(0.02, 0.02, 'frame 0', ha='left', va='bottom')
    plans = {id(label_tracks(ax))}
    kept = [label_tracks(ax)]
    for frame in range(20):
        counter.set_text('frame ' + '9' * (frame + 1))
        anim.draw_frame(frame)
        kept.append(label_tracks(ax))
        plans.add(id(kept[-1]))
    assert len(plans) <= 3
    for frame in range(20, 24):             # (the text now stays)
        anim.draw_frame(frame)
        kept.append(label_tracks(ax))
    final = kept[-1]
    assert len({id(tracks) for tracks in kept}) <= 4
    anim.draw_frame(24)
    assert label_tracks(ax) is final
    renderer = _show(anim, 30)
    there = counter.get_window_extent(renderer)
    assert not any(
        one.get_bbox_patch().get_window_extent(renderer).overlaps(there)
        for one in _labels(ax))


def _nan_cloud():
    data = CLOUD[:12].copy()
    data[3] = np.nan
    return data


EDGE_INPUTS = {
    'no labels': lambda: _quiet(
        CLOUD[:12], 'o', animate='spin', duration=1, frame_rate=10),
    'one label': lambda: _quiet(
        CLOUD[:12], 'o', labels=['only'] + [None] * 11, animate='spin',
        duration=1, frame_rate=10),
    'labels with None entries': lambda: _quiet(
        CLOUD[:12], 'o',
        labels=[name if k % 2 else None
                for k, name in enumerate(NAMES[:12])],
        animate='spin', duration=1, frame_rate=10),
    'one frame': lambda: _quiet(
        CLOUD[:12], 'o', labels=NAMES[:12], animate='spin', duration=0.1,
        frame_rate=10),
    'two frames': lambda: _quiet(
        CLOUD[:12], 'o', labels=NAMES[:12], animate='spin', duration=0.2,
        frame_rate=10),
    'NaN point, spin': lambda: _quiet(
        _nan_cloud(), 'o', labels=NAMES[:12], animate='spin', duration=1,
        frame_rate=10),
    'NaN point, window': lambda: _quiet(
        _nan_cloud(), 'o', labels=NAMES[:12], animate='window', duration=1,
        frame_rate=10),
    'panels with unequal label counts': lambda: _quiet(
        [CLOUD[:12], CLOUD[:5], CLOUD[:1], CLOUD[:9]], 'o', panels=(2, 2),
        labels=[NAMES[:12], NAMES[:5], None, NAMES[:9]], animate='spin',
        duration=2, frame_rate=15),
}
EDGE_FRAMES = {'one frame': 1, 'two frames': 2,
               'panels with unequal label counts': 30}
EDGE_LABELS = {'no labels': [], 'one label': [1],
               'labels with None entries': [6],
               'panels with unequal label counts': [12, 5, 9]}


@pytest.mark.parametrize('name', list(EDGE_INPUTS))
def test_edge_inputs_are_planned_and_drawn(name, tmp_path):
    """No labels, one label, None among the labels, an animation of one
    frame and of two, a NaN point with a label (hypertools fills the
    point in, so its label is shown), panels with different
    numbers of labels (one with none): each is planned, every frame
    draws with every shown label on the canvas and off the others, and
    the animation saves."""
    anim = EDGE_INPUTS[name]()
    fig = anim.figure
    assert anim.n_frames == EDGE_FRAMES.get(name, 10)
    axes = _labeled_axes(fig)
    counts = [len(_labels(ax)) for ax in axes]
    assert counts == EDGE_LABELS.get(name, [12])
    for ax, count in zip(axes, counts):
        tracks = label_tracks(ax)
        assert tracks['offsets'].shape == (anim.n_frames, count, 2)
    census = _census(anim)
    assert census['off_canvas'] == 0 and census['on_static'] == 0
    # (a grid has less room, and how much depends on the width of the
    # text: its bound is `test_a_grids_labels_overlap_on_few_frames`)
    # (measured for this grid: at most 2 pairs on 5 frames of 30 under
    # matplotlib 3.10, 3 pairs on 22 frames under 3.11's wider text)
    pairs = _pairs_per_frame(anim)
    assert pairs.max() <= (3 if name.startswith('panels') else 0)
    anim.save(tmp_path / 'edge.gif', writer=PillowWriter(fps=10))
    assert (tmp_path / 'edge.gif').stat().st_size > 1000


def test_saving_an_animation_twice_gives_the_same_frames(tmp_path):
    """Two saves of one animation hold the same frames, pixel for pixel
    (the labels' places belong to the animation and the frame, not to
    what was drawn before)."""
    anim = _spin12_dressed(duration=1, frame_rate=10)
    frames = []
    for name in ('one.gif', 'two.gif'):
        anim.save(tmp_path / name, writer=PillowWriter(fps=10))
        with Image.open(tmp_path / name) as saved:
            assert saved.n_frames == anim.n_frames
            every = []
            for index in range(saved.n_frames):
                saved.seek(index)
                every.append(np.asarray(saved.convert('RGB')).copy())
        frames.append(np.stack(every))
    assert np.array_equal(frames[0], frames[1])
    assert (frames[0][0] != frames[0][5]).any()
