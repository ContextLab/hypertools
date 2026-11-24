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
