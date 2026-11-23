"""1.2.1 review round 1: defects two reviews found in the GH #299 animation
arguments and the GH #297 layout fixes.

A. Observation labels are placed STABLY: a label stays where it is until
   its box is covered, so labels do not hop between places from one
   frame of an animation to the next.
B. Continuous ``hue=`` in an animation: markers of a line-and-marker
   format on tail and trail, serial trails that are empty when they
   should be, the dataset's own opacity, mixed formats.
C. Arguments: ``tail_linewidth=`` warnings, values inside
   ``head_marker=``/``caption_kwargs=``/``suptitle_kwargs=``, static
   plotly, ``hold=`` in a saved gif, the cube's margin, a suptitle over a
   caption that grows.

Real figures and real frames, rendered headless; saved files are read
back. Nothing is mocked, and no assertion depends on a font's absolute
metrics: boxes are measured where they were drawn and compared with each
other.
"""
import itertools
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.collections import PathCollection               # noqa: E402
from mpl_toolkits.mplot3d import proj3d                         # noqa: E402
from PIL import Image                                           # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plot import CAPTION_GID, SUPTITLE_GID      # noqa: E402


def _walk(seed, rows=40, cols=3):
    return np.random.default_rng(seed).normal(size=(rows, cols)).cumsum(0)


X, Y = _walk(0), _walk(1)
HUE = np.linspace(0.0, 1.0, len(X))
CLOUD = np.random.default_rng(3).standard_normal((30, 3))
NAMES = [f'Subject {i}' for i in range(len(CLOUD))]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


# --------------------------------------------------------------------------
# A. stable label placement
# --------------------------------------------------------------------------

def _labels(fig):
    return [entry[0] for entry in fig.axes[0]._hyp_labels_and_points]


def _label_history(anim):
    """Per frame: each label's offset (None while it is hidden), and the
    number of pairs of visible label boxes that overlap."""
    fig = anim.figure
    labels = _labels(fig)
    places, overlaps = [], []
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        places.append([tuple(label.xyann) if label.get_visible() else None
                       for label in labels])
        boxes = [label.get_bbox_patch().get_window_extent(renderer)
                 for label in labels if label.get_visible()]
        overlaps.append(sum(1 for one, other in itertools.combinations(
            boxes, 2) if one.overlaps(other)))
    return places, overlaps


def _switches(places):
    """How many times a label changed place between two consecutive
    frames it was visible on, and how many of those changes went BACK to
    a place the label had left within the three frames before."""
    switches = returns = 0
    for track in zip(*places):
        changed = [k for k in range(1, len(track))
                   if track[k] is not None and track[k - 1] is not None
                   and track[k] != track[k - 1]]
        switches += len(changed)
        for k in changed:
            if any(track[j] == track[k] and track[j] is not None
                   for j in range(max(0, k - 4), k - 1)):
                returns += 1
    return switches, returns


def test_spinning_labels_stay_put_with_few_labels():
    """Eight labels over one turn in 60 frames: the first draft of 1.2.1
    made 71 changes of place (labels were placed afresh on every draw).
    A label now moves only when it is covered: under one change every
    two frames, never two frames running back and forth, and no box on
    another."""
    anim = hyp.plot(CLOUD[:8], 'o', labels=NAMES[:8], animate='spin',
                    duration=4, frame_rate=15, show=False)
    places, overlaps = _label_history(anim)
    switches, returns = _switches(places)
    assert anim.n_frames == 60
    assert switches <= 30, switches
    assert returns == 0
    assert max(overlaps) == 0


def test_spinning_labels_stay_put_with_many_labels():
    """Thirty labels: about 13 of them changed place on EVERY frame (777
    changes in 60 frames). Now well under a tenth of the labels change
    on an average frame, and a label almost never goes straight back."""
    anim = hyp.plot(CLOUD, 'o', labels=NAMES, animate='spin', duration=4,
                    frame_rate=15, show=False)
    places, _overlaps = _label_history(anim)
    switches, returns = _switches(places)
    per_frame = switches / (anim.n_frames - 1)
    assert per_frame <= 0.1 * len(NAMES), per_frame
    assert returns <= 0.05 * switches + 2, (returns, switches)


@pytest.mark.parametrize('ndims', [2, 3])
def test_window_animation_labels_stay_put(ndims):
    """A sliding window: 233 changes in 60 frames before (3-D)."""
    anim = hyp.plot(CLOUD[:, :ndims], 'o', labels=NAMES, animate='window',
                    duration=4, frame_rate=15, show=False)
    places, overlaps = _label_history(anim)
    switches, returns = _switches(places)
    assert switches <= 60, switches
    assert returns <= 2, returns
    assert np.mean(overlaps) <= 0.25


def test_a_label_keeps_its_place_while_it_is_clear():
    """Hysteresis: a label that was moved does not go back when its old
    place becomes free, and a label that is clear is not touched."""
    anim = hyp.plot(CLOUD[:8], 'o', labels=NAMES[:8], animate='spin',
                    duration=4, frame_rate=15, show=False)
    fig = anim.figure
    labels = _labels(fig)
    anim.draw_frame(0)
    fig.canvas.draw()
    for frame in range(1, anim.n_frames):
        before = [tuple(label.xyann) for label in labels]
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        after = [tuple(label.xyann) for label in labels]
        for k, label in enumerate(labels):
            if before[k] == after[k]:
                continue
            # the label moved: at its OLD place, in this frame, its box
            # would be on an earlier label's
            now = label.get_bbox_patch().get_window_extent(renderer)
            width, height = now.width, now.height
            anchor = fig.axes[0].transData.transform(label.xy)
            scale = renderer.points_to_pixels(1.0)
            # (the frame reaches `pad` beyond the text on every side)
            pad = label.get_bbox_patch().get_boxstyle().pad \
                * label.get_size() * scale
            dx, dy = before[k][0] * scale, before[k][1] * scale
            x0 = anchor[0] + dx + (pad - width if dx < 0 else
                                   -0.5 * width if dx == 0 else -pad)
            y0 = anchor[1] + dy + (pad - height if dy < 0 else
                                   -0.5 * height if dy == 0 else -pad)
            # (...or it had been moved there and that place has left the
            # canvas)
            covered = (x0 < 2.0 or y0 < 2.0
                       or x0 + width > fig.bbox.width - 2.0
                       or y0 + height > fig.bbox.height - 2.0)
            for other in labels[:k]:
                box = other.get_bbox_patch().get_window_extent(renderer)
                if (min(x0 + width, box.x1) - max(x0, box.x0) > -2.0
                        and min(y0 + height, box.y1)
                        - max(y0, box.y0) > -2.0):
                    covered = True
            assert covered, (frame, label.get_text(), before[k], after[k])


def test_an_animation_places_its_labels_the_same_on_every_pass():
    """Starting over (a loop, a save after a preview) places the labels
    from the start again."""
    anim = hyp.plot(CLOUD[:12], 'o', labels=NAMES[:12], animate='spin',
                    duration=2, frame_rate=10, show=False)
    first, _ = _label_history(anim)
    again, _ = _label_history(anim)
    assert first == again


def test_static_labels_start_from_their_usual_place():
    """A static figure's labels are placed on the finished figure, not
    left where a draw made during its layout put them: the label that
    comes first is where it has always been, and two draws agree."""
    fig = hyp.plot(CLOUD, 'o', labels=NAMES, show=False)
    fig.canvas.draw()
    labels = _labels(fig)
    assert tuple(labels[0].xyann) == (-20.0, 20.0)
    first = [tuple(label.xyann) for label in labels]
    fig.canvas.draw()
    assert [tuple(label.xyann) for label in labels] == first


def test_hand_rotation_moves_only_covered_labels():
    """Turning a static 3-D plot (what a mouse drag does): between two
    close views almost every label keeps its place."""
    fig = hyp.plot(CLOUD[:12], 'o', labels=NAMES[:12], show=False)
    ax = fig.axes[0]
    fig.canvas.draw()
    labels = _labels(fig)
    changes = 0
    for step in range(1, 31):
        before = [tuple(label.xyann) for label in labels]
        ax.view_init(elev=ax.elev, azim=ax.azim + 3.0)
        fig.canvas.draw()
        changes += sum(one != other for one, other in zip(
            before, [tuple(label.xyann) for label in labels]))
    assert changes <= 30, changes           # 12 labels x 30 views = 360


def test_label_arrows_point_at_their_points_after_the_cube_is_shrunk():
    """A static 3-D plot whose inside legend makes the cube smaller
    (GH #297): the labels' anchors were left at the projection of the
    cube's first size, up to 9 px from their points."""
    data = [_walk(seed, rows=60) for seed in range(5)]
    names = [f'p{i}' if i % 20 == 0 else None for i in range(300)]
    fig = hyp.plot(data, '-', labels=names, size=(5, 5), show=False,
                   legend=[f'A long name for dataset {k}' for k in range(5)],
                   legend_kwargs={'loc': 'upper left', 'fontsize': 8})
    ax = fig.axes[0]
    reference = plt.figure().add_subplot(projection='3d').get_box_aspect()
    assert ax.get_box_aspect()[0] < 0.97 * reference[0]     # it was shrunk
    fig.canvas.draw()
    for label, x, y, z in ax._hyp_labels_and_points:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        np.testing.assert_allclose(
            ax.transData.transform(label.xy),
            ax.transData.transform((x2, y2)), atol=0.5)


# --------------------------------------------------------------------------
# B. continuous hue in an animation
# --------------------------------------------------------------------------

def _roles(ax):
    return {coll._hyp_trace_role: coll for coll in ax.collections
            if getattr(coll, '_hyp_trace_role', None)}


def _marker_scatters(ax):
    return {coll._hyp_markers_of: coll for coll in ax.collections
            if getattr(coll, '_hyp_markers_of', None)}


def _n_points(coll, ndims):
    if not coll.get_visible():
        return 0
    if ndims == 3:
        return len(np.asarray(coll._offsets3d[0]))
    return len(coll.get_offsets())


@pytest.mark.parametrize('ndims', [2, 3])
def test_line_marker_format_with_hue_draws_markers_on_tail_and_trail(ndims):
    """``'-o'`` + ``hue=`` + ``chemtrails=``: the animated plot drew the
    colored line and no marker at all, where the static plot draws both."""
    anim = hyp.plot(X[:, :ndims], '-o', hue=HUE, animate=True,
                    chemtrails=True, tail_duration=1, duration=6,
                    frame_rate=10, show=False)
    anim.draw_frame(35)
    anim.figure.canvas.draw()
    ax = anim.figure.axes[0]
    dots = _marker_scatters(ax)
    assert set(dots) == {(0, 'head'), (0, 'trail')}
    assert all(isinstance(coll, PathCollection) for coll in dots.values())
    head, trail = _n_points(dots[0, 'head'], ndims), _n_points(
        dots[0, 'trail'], ndims)
    # one marker per revealed observation, none twice
    assert head >= 2 and trail >= 2
    assert 20 <= head + trail <= 26                    # row 35/60 of 40
    # ...each in its observation's color: several distinct colors
    colors = np.asarray(dots[0, 'trail'].get_facecolors())
    assert len(np.unique(colors[:, :3].round(3), axis=0)) >= trail - 1
    # the trail's markers are as faint as the trail's line
    assert dots[0, 'trail'].get_alpha() == pytest.approx(0.3)
    assert _roles(ax)['trail'].get_alpha() == pytest.approx(0.3)
    # and they are really drawn: ink in the colors of the palette
    # disappears when the scatters are hidden
    def ink():
        anim.figure.canvas.draw()
        return int((np.asarray(
            anim.figure.canvas.buffer_rgba())[..., :3] < 250).any(-1).sum())
    with_markers = ink()
    for coll in dots.values():
        coll.set_visible(False)
    assert ink() < with_markers


def test_line_marker_format_with_hue_in_an_animated_grid():
    anim = hyp.plot([X, Y], '-o', hue=[HUE, HUE], panels=True,
                    animate=True, chemtrails=True, tail_duration=1,
                    duration=6, frame_rate=10, show=False)
    anim.draw_frame(35)
    anim.figure.canvas.draw()
    for ax in anim.figure.axes[:2]:
        dots = _marker_scatters(ax)
        assert len(dots) == 2
        assert all(_n_points(coll, 3) >= 2 for coll in dots.values())


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('flag', ['chemtrails', 'precog', 'bullettime'])
def test_serial_marker_trails_with_hue_draw_every_frame(ndims, flag):
    """``'o'`` + ``hue=`` + ``animate='serial'`` + a trail flag raised
    AttributeError (`set_segments` on a scatter) on the first frame."""
    anim = hyp.plot([X[:, :ndims], Y[:, :ndims]], 'o', hue=[HUE, HUE],
                    animate='serial', duration=2, frame_rate=10,
                    show=False, **{flag: True})
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        anim.figure.canvas.draw()


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt', ['-o', '-', 'o'])
def test_serial_hue_trail_is_empty_where_the_plain_trail_is(ndims, fmt):
    """A serial trail with ``hue=`` kept the geometry of the frame drawn
    before: after a middle frame, frame 0 and the last frame (where every
    trail is cleared) still showed it."""
    anim = hyp.plot([X[:, :ndims], Y[:, :ndims]], fmt, hue=[HUE, HUE],
                    animate='serial', chemtrails=True, duration=2,
                    frame_rate=10, show=False)
    ax = anim.figure.axes[0]
    trails = [coll for coll in ax.collections
              if getattr(coll, '_hyp_trace_role', None) == 'trail']
    assert len(trails) == 2

    def drawn(coll):
        if isinstance(coll, PathCollection):
            return _n_points(coll, ndims)
        return len(coll._segments3d if ndims == 3 else coll.get_segments())

    def markers():
        return [_n_points(coll, ndims)
                for coll in _marker_scatters(ax).values()
                if coll._hyp_markers_of[1] == 'trail']

    anim.draw_frame(5)
    assert drawn(trails[0]) > 0
    for frame in (0, anim.n_frames - 1):
        anim.draw_frame(frame)
        assert [drawn(coll) for coll in trails] == [0, 0], frame
        assert not any(markers()), frame


@pytest.mark.parametrize('fmt', ['-', 'o', '-o'])
def test_hue_trail_opacity_is_trail_alpha_times_the_dataset_alpha(fmt):
    """``alpha=0.2, trail_alpha=0.5``: the trail was drawn at 0.5 (more
    opaque than its head) instead of 0.1, and an animated hue-colored
    LINE ignored ``alpha=`` altogether."""
    anim = hyp.plot(X, fmt, hue=HUE, animate=True, chemtrails=True,
                    alpha=0.2, trail_alpha=0.5, duration=2, frame_rate=10,
                    show=False)
    anim.draw_frame(15)
    ax = anim.figure.axes[0]
    roles = _roles(ax)
    assert roles['head'].get_alpha() == pytest.approx(0.2)
    assert roles['trail'].get_alpha() == pytest.approx(0.1)
    for coll in _marker_scatters(ax).values():
        want = 0.2 if coll._hyp_markers_of[1] == 'head' else 0.1
        assert coll.get_alpha() == pytest.approx(want)
    # the same call without hue=
    plain = hyp.plot(X, fmt, animate=True, chemtrails=True, alpha=0.2,
                     trail_alpha=0.5, duration=2, frame_rate=10, show=False)
    assert [line.get_alpha() for line in plain.figure.axes[0].lines[:2]] \
        == pytest.approx([0.2, 0.1])


@pytest.mark.parametrize('ndims', [2, 3])
def test_mixed_formats_with_hue_animate_each_dataset_as_asked(ndims):
    """``['-', 'o']`` + ``hue=``: the line dataset was drawn as markers
    as soon as one dataset had no line."""
    anim = hyp.plot([X[:, :ndims], Y[:, :ndims]], ['-', 'o'],
                    hue=[HUE, HUE], animate=True, chemtrails=True,
                    tail_duration=1, duration=4, frame_rate=10, show=False)
    anim.draw_frame(25)
    kinds = {(coll._hyp_trace_index, coll._hyp_trace_role):
             isinstance(coll, PathCollection)
             for coll in anim.figure.axes[0].collections
             if getattr(coll, '_hyp_trace_role', None)}
    assert kinds == {(0, 'head'): False, (0, 'trail'): False,
                     (1, 'head'): True, (1, 'trail'): True}


def test_mixed_formats_with_hue_static():
    fig = hyp.plot([X, Y], ['-', 's'], hue=[HUE, HUE], show=False)
    ax = fig.axes[0]
    lines = [coll for coll in ax.collections
             if getattr(coll, '_hyp_trace_index', None) is not None]
    scatters = [coll for coll in ax.collections
                if isinstance(coll, PathCollection)]
    assert [coll._hyp_trace_index for coll in lines] == [0]
    assert len(scatters) == 1
    assert len(np.asarray(scatters[0]._offsets3d[0])) == len(Y)
    # the dataset's own glyph, a square (four corners), not the default
    square = matplotlib.markers.MarkerStyle('s')
    np.testing.assert_allclose(
        scatters[0].get_paths()[0].vertices,
        square.get_path().transformed(square.get_transform()).vertices)


@pytest.mark.parametrize('hue', [None, 'continuous'])
def test_head_marker_fades_with_its_dataset(hue):
    """`dataset_fade=` + `head_marker=`: with a continuous ``hue=`` the
    finished dataset's line faded and its head marker stayed opaque."""
    anim = hyp.plot([X, Y], '-', hue=None if hue is None else [HUE, HUE],
                    animate='serial', head_marker='o', duration=2,
                    frame_rate=10, dataset_fade={'floor': 0.1, 'decay': 0.5},
                    show=False)
    anim.draw_frame(anim.n_frames - 2)
    ax = anim.figure.axes[0]
    markers = sorted((line for line in ax.lines
                      if getattr(line, '_hyp_trace_role', None)
                      == 'head_marker'), key=lambda m: m._hyp_trace_index)
    assert [marker.get_alpha() for marker in markers] \
        == pytest.approx([0.55, 1.0])
    if hue is not None:
        heads = sorted((coll for coll in ax.collections
                        if getattr(coll, '_hyp_trace_role', None) == 'head'),
                       key=lambda c: c._hyp_trace_index)
        assert [coll.get_alpha() for coll in heads] \
            == pytest.approx([0.55, 1.0])


def test_dataset_fade_reaches_the_right_trail_with_a_flag_list():
    """``chemtrails=[False, True]``: the one trail belongs to dataset 1,
    and was faded as dataset 0's (it is the first trail)."""
    anim = hyp.plot([X, Y], '-', animate='serial', duration=2,
                    frame_rate=10, chemtrails=[False, True],
                    dataset_fade={'floor': 0.1, 'decay': 0.5}, show=False)
    anim.draw_frame(anim.n_frames - 4)           # dataset 1 being revealed
    ax = anim.figure.axes[0]
    trail = [line for line in ax.lines
             if getattr(line, '_hyp_trace_role', None) == 'trail']
    assert [line._hyp_trace_index for line in trail] == [1]
    assert [line.get_alpha() for line in ax.lines[:2]] \
        == pytest.approx([0.55, 1.0])
    assert trail[0].get_alpha() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# C. arguments
# --------------------------------------------------------------------------

def test_tail_linewidth_on_a_markers_only_format_warns():
    with pytest.warns(UserWarning, match='tail_linewidth=.*no dataset.*line'):
        hyp.plot(X, 'o', animate=True, tail_linewidth=5, duration=1,
                 frame_rate=10, show=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # one dataset with a line is enough
        hyp.plot([X, Y], ['-', 'o'], animate=True, tail_linewidth=5,
                 duration=1, frame_rate=10, show=False)


@pytest.mark.parametrize('ndims', [2, 3])
def test_tail_linewidth_does_not_widen_a_spinning_or_static_line(ndims):
    """``animate='spin'`` warned that `tail_linewidth=` is ignored and
    drew the whole line at that width."""
    if ndims == 3:
        with pytest.warns(UserWarning, match='ignoring tail_linewidth'):
            anim = hyp.plot(X, '-', animate='spin', linewidth=1,
                            tail_linewidth=10, duration=1, frame_rate=10,
                            show=False)
        anim.draw_frame(3)
        assert anim.figure.axes[0].lines[0].get_linewidth() == 1.0
    with pytest.warns(UserWarning, match='tail_linewidth only affects'):
        fig = hyp.plot(X[:, :ndims], '-', linewidth=1, tail_linewidth=10,
                       show=False)
    assert fig.axes[0].lines[0].get_linewidth() == 1.0


BAD_HEAD_MARKERS = [
    ({'size': -3}, "head_marker\\['size'\\]"),
    ({'size': 0}, "head_marker\\['size'\\]"),
    ({'size': float('nan')}, "head_marker\\['size'\\]"),
    ({'size': 'big'}, "head_marker\\['size'\\]"),
    ({'markersize': -1}, "head_marker\\['markersize'\\]"),
    ({'color': 'notacolor'}, "head_marker\\['color'\\]"),
    ({'edgecolor': 'nope'}, "head_marker\\['edgecolor'\\]"),
    ({'alpha': 2}, "head_marker\\['alpha'\\]"),
    ({'alpha': 'half'}, "head_marker\\['alpha'\\]"),
    ({'edgewidth': -1}, "head_marker\\['edgewidth'\\]"),
    ({'fillstyle': 'striped'}, "head_marker\\['fillstyle'\\]"),
    ({'zorder': 'top'}, "head_marker\\['zorder'\\]"),
]


@pytest.mark.parametrize('spec, match', BAD_HEAD_MARKERS)
def test_head_marker_values_are_checked_by_name(spec, match):
    with pytest.raises(ValueError, match=match):
        hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                 frame_rate=10, show=False)


def test_good_head_marker_values_still_draw():
    anim = hyp.plot(X, '-', animate=True, duration=1, frame_rate=10,
                    head_marker={'marker': 'D', 'size': 9.5, 'color': 'k',
                                 'edgecolor': (1, 1, 1), 'edgewidth': 0,
                                 'alpha': 1, 'fillstyle': 'full',
                                 'zorder': 9}, show=False)
    marker = [line for line in anim.figure.axes[0].lines
              if getattr(line, '_hyp_trace_role', None) == 'head_marker'][0]
    assert marker.get_markersize() == 9.5 and marker.get_marker() == 'D'


BAD_TEXT_STYLES = [
    ({'size': -1}, 'size'), ({'size': 0}, 'size'),
    ({'size': float('nan')}, 'size'), ({'size': 'huge'}, 'size'),
    ({'fontsize': -2}, 'fontsize'), ({'weight': 'heavyish'}, 'weight'),
    ({'weight': 2000}, 'weight'), ({'style': 'slanted'}, 'style'),
    ({'family': 7}, 'family'), ({'color': 'notacolor'}, 'color'),
    ({'alpha': 3}, 'alpha'), ({'linespacing': 0}, 'linespacing'),
]


@pytest.mark.parametrize('style, key', BAD_TEXT_STYLES)
@pytest.mark.parametrize('name', ['caption_kwargs', 'suptitle_kwargs'])
def test_figure_text_values_are_checked_by_name(name, style, key):
    text = name.split('_')[0]
    with pytest.raises(ValueError, match=f"{name}\\['{key}'\\] must be"):
        hyp.plot(X, '-', show=False, **{text: 'Some text', name: style})


@pytest.mark.parametrize('style, key', [
    ({'size': -3}, 'size'), ({'size': 'huge'}, 'size'),
    ({'color': 'notacolor'}, 'color'), ({'alpha': 3}, 'alpha')])
def test_title_kwargs_values_are_checked_by_name(style, key):
    with pytest.raises(ValueError, match=f"title_kwargs\\['{key}'\\] must be"):
        hyp.plot(X, '-', title='A title', title_kwargs=style, show=False)


def test_good_text_styles_still_draw():
    fig = hyp.plot(X, '-', show=False, title='A title',
                   title_kwargs={'size': 'x-large', 'weight': 600},
                   caption='A caption', suptitle='A heading',
                   caption_kwargs={'size': 9, 'weight': 'bold',
                                   'style': 'italic', 'alpha': 0.5,
                                   'color': '#444444', 'linespacing': 1.1,
                                   'family': ['DejaVu Sans']},
                   suptitle_kwargs={'size': 'large', 'color': (0, 0, 0)})
    caption = fig.findobj(lambda a: a.get_gid() == CAPTION_GID)[0]
    assert caption.get_fontsize() == 9 and caption.get_alpha() == 0.5


def test_suptitle_bytes_are_decoded_like_a_caption():
    fig = hyp.plot(X, '-', suptitle=b'In bytes', caption=b'Also bytes',
                   show=False)
    assert fig.findobj(lambda a: a.get_gid() == SUPTITLE_GID)[
        0].get_text() == 'In bytes'
    assert fig.findobj(lambda a: a.get_gid() == CAPTION_GID)[
        0].get_text() == 'Also bytes'


def test_static_plotly_ignores_the_animation_styling_with_a_warning():
    """As `chemtrails=` on a static plot: a warning, on either backend.
    An ANIMATED plotly plot raises (the docstring said every plotly plot
    does)."""
    pytest.importorskip('plotly')
    for kwargs in ({'head_marker': 'o'}, {'tail_linewidth': 3}):
        name = next(iter(kwargs))
        with pytest.warns(UserWarning, match=f'{name} only affects ANIMATED'):
            hyp.plot(X, '-', backend='plotly', show=False, **kwargs)
        with pytest.raises(NotImplementedError, match=name):
            hyp.plot(X, '-', backend='plotly', animate=True, show=False,
                     **kwargs)
    with pytest.warns(UserWarning, match='chemtrails only affect ANIMATED'):
        hyp.plot(X, '-', backend='plotly', chemtrails=True, show=False)
    with pytest.warns(UserWarning,
                      match='head_marker/tail_linewidth only affect '):
        hyp.plot(X, '-', head_marker='o', tail_linewidth=2, show=False)


def test_hold_in_a_gif_keeps_the_time_and_merges_identical_frames(tmp_path):
    """What the docstring says of a saved gif: the playing time is
    ``duration + hold``; Pillow stores a run of identical frames as one
    frame with the run's duration, so the file has fewer frames."""
    path = tmp_path / 'held.gif'
    anim = hyp.plot(X, '-', animate=True, rotations=0, duration=1,
                    frame_rate=10, hold=1, show=False, save_path=str(path))
    assert anim.n_frames == 20
    with Image.open(path) as gif:
        durations = []
        for k in range(gif.n_frames):
            gif.seek(k)
            durations.append(gif.info['duration'])
    assert sum(durations) == 2000
    assert len(durations) == 10
    assert durations[-1] == 1100 and set(durations[:-1]) == {100}


def _hull_gap(points, box):
    """Distance in px between the convex outline of `points` and the
    rectangle `box` (0 when they touch or overlap)."""
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    hull = points[ConvexHull(points).vertices]
    rect = np.array([[box.x0, box.y0], [box.x1, box.y0],
                     [box.x1, box.y1], [box.x0, box.y1]])
    if Path(hull).contains_points(rect).any() \
            or Path(rect).contains_points(hull).any():
        return 0.0

    def point_segment(p, a, b):
        d = b - a
        t = np.clip((p - a) @ d / (d @ d), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * d)))

    best = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        for j in range(4):
            c, d = rect[j], rect[(j + 1) % 4]
            best = min(best, point_segment(a, c, d), point_segment(b, c, d),
                       point_segment(c, a, b), point_segment(d, a, b))
    return best


@pytest.mark.parametrize('size', [(6.4, 4.8), (8, 8)])
def test_cube_keeps_its_stated_margin_from_the_legend_on_every_frame(size):
    """The changelog's number, measured with the real renderer on every
    frame: the cube's outline stays about 2% of the canvas's SHORTER side
    from the legend (at least 1.9%), and that far inside the canvas."""
    t = np.linspace(0, 4 * np.pi, 80)
    a = np.c_[np.cos(t), np.sin(t), t / 6]
    b = np.c_[np.cos(t) * .5 + 2, np.sin(t) * .5, -t / 6]
    anim = hyp.plot([a, b], '-', animate=True, legend=['A', 'B'], size=size,
                    duration=3, frame_rate=20, show=False)
    fig = anim.figure
    ax = fig.axes[0]
    short = min(fig.bbox.width, fig.bbox.height)
    gaps, edges = [], []
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        corners = np.array(list(itertools.product(
            ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))
        xs, ys, _ = proj3d.proj_transform(*corners.T, ax.get_proj())
        px = ax.transData.transform(np.column_stack([xs, ys]))
        gaps.append(_hull_gap(
            px, ax.get_legend().get_window_extent(renderer)))
        edges.append(min(px[:, 0].min(), fig.bbox.width - px[:, 0].max(),
                         px[:, 1].min(), fig.bbox.height - px[:, 1].max()))
    assert min(gaps) >= 0.019 * short, min(gaps) / short
    assert min(gaps) <= 0.03 * short                # ...and no more room
    assert min(edges) >= 0.019 * short, min(edges) / short


def _extent(fig, gid):
    return fig.findobj(lambda a: a.get_gid() == gid)[0].get_window_extent(
        fig.canvas.get_renderer())


@pytest.mark.parametrize('kwargs', [{}, {'ndims': 2}, {'panels': True}],
                         ids=['3d', '2d', 'grid'])
def test_suptitle_clears_a_top_caption_that_grows(kwargs):
    """The suptitle was placed against frame 0's one-line caption and
    the four-line caption of the later frames ran into it."""
    def caption(ctx):
        return 'short' if ctx.frame == 0 else 'one\ntwo\nthree\nfour'

    data = [X, Y] if kwargs.get('panels') else X
    anim = hyp.plot(data, '-', animate=True, caption=caption,
                    caption_kwargs={'position': 'top'}, suptitle='Heading',
                    duration=2, frame_rate=10, show=False, **kwargs)
    fig = anim.figure
    heights = set()
    for frame in (0, 1, 10, anim.n_frames - 1):
        anim.draw_frame(frame)
        fig.canvas.draw()
        caption_box = _extent(fig, CAPTION_GID)
        title_box = _extent(fig, SUPTITLE_GID)
        heights.add(round(caption_box.height))
        assert caption_box.y1 < title_box.y0, frame
        assert title_box.y1 <= fig.bbox.height
    assert len(heights) == 2                         # it really did grow
