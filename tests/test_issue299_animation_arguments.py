"""GH #299: trails for line+marker formats, and the animation arguments
that came with it.

A. ``hyp.plot(x, '-o', animate=True, chemtrails=True)`` drew no chemtrail:
   the trail artists were only created for a PURE line format. Every
   format now gets its trail, drawn as its head is.
B. ``head_marker=``, ``tail_linewidth=``, ``trail_alpha=``, ``hold=``,
   ``caption_kwargs=`` and ``suptitle=``/``suptitle_kwargs=``.

Real figures and real frames, rendered headless; saved files are read
back. Nothing is mocked, and no assertion depends on a font's absolute
metrics: text is measured where it was drawn and compared with what is
around it.
"""
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from PIL import Image

import hypertools as hyp
from hypertools.plot.plot import CAPTION_GID, SUPTITLE_GID

HAS_FFMPEG = (shutil.which('ffmpeg') is not None
              and shutil.which('ffprobe') is not None)


def _walk(seed, rows=60, cols=3):
    return np.random.default_rng(seed).normal(size=(rows, cols)).cumsum(0)


X = _walk(0)
DATA = [_walk(seed) for seed in range(4)]
ANIM = dict(animate=True, tail_duration=0.5, duration=2, frame_rate=10,
            show=False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _plot(data=None, fmt='-', **kwargs):
    call = dict(ANIM)
    call.update(kwargs)
    return hyp.plot(X if data is None else data, fmt, **call)


def _lines(anim, axes=0):
    return list(anim.figure.axes[axes].lines)


def _trails(anim, axes=0):
    """The trail artists of one axes: the lines drawn below full alpha
    that are not head markers."""
    return [line for line in _lines(anim, axes)
            if line.get_alpha() is not None
            and getattr(line, '_hyp_trace_role', None) != 'head_marker']


def _heads(anim, axes=0):
    return [line for line in _lines(anim, axes)
            if line.get_alpha() is None
            and getattr(line, '_hyp_trace_role', None) != 'head_marker']


def _head_markers(anim, axes=0):
    return [line for line in _lines(anim, axes)
            if getattr(line, '_hyp_trace_role', None) == 'head_marker']


def _n_vertices(line):
    return len(line.get_xdata())


def _rendered(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _extent(fig, artist):
    """`artist`'s drawn box in FIGURE fractions, from a real draw."""
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.transformed(fig.transFigure.inverted())


def _figure_text(fig, gid):
    found = fig.findobj(lambda a: a.get_gid() == gid)
    assert len(found) == 1, f'{len(found)} texts with gid {gid!r}'
    return found[0]


def _content_top(fig, skip=()):
    """The highest point (figure fraction) of everything drawn except the
    texts in `skip`."""
    hidden = [(artist, artist.get_visible()) for artist in skip]
    for artist, _ in hidden:
        artist.set_visible(False)
    try:
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        return box.y1 / fig.get_size_inches()[1]
    finally:
        for artist, visible in hidden:
            artist.set_visible(visible)


def _content_bottom(fig, skip=()):
    hidden = [(artist, artist.get_visible()) for artist in skip]
    for artist, _ in hidden:
        artist.set_visible(False)
    try:
        fig.canvas.draw()
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        return box.y0 / fig.get_size_inches()[1]
    finally:
        for artist, visible in hidden:
            artist.set_visible(visible)


# ---------------------------------------------------------------------------
# A. the bug: a line+marker format gets its trail
# ---------------------------------------------------------------------------

LINE_MARKER_FORMATS = ['-o', 'o-', '.-', '--s', ':^', '-.d']


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt', LINE_MARKER_FORMATS)
def test_line_marker_format_draws_its_chemtrail(fmt, ndims):
    reference = _plot(fmt='-', chemtrails=True, ndims=ndims).draw_frame(15)
    anim = _plot(fmt=fmt, chemtrails=True, ndims=ndims).draw_frame(15)
    (head,), (trail,) = _heads(anim), _trails(anim)
    (ref_head,), (ref_trail,) = _heads(reference), _trails(reference)
    # the trail is the head's own format, faded
    assert trail.get_linestyle() == head.get_linestyle() != 'None'
    assert trail.get_marker() == head.get_marker() != 'None'
    assert trail.get_alpha() == pytest.approx(0.3)
    assert to_rgba(trail.get_color()) == to_rgba(head.get_color())
    # ...over exactly the rows the pure line's trail covers
    assert _n_vertices(trail) == _n_vertices(ref_trail) > 2
    assert _n_vertices(head) == _n_vertices(ref_head)
    np.testing.assert_allclose(trail.get_xdata(), ref_trail.get_xdata())
    # its markers sit on observations, as the head's do: far fewer than
    # the smoothed curve has vertices
    assert 0 < len(trail.get_markevery()) < _n_vertices(trail) / 2
    assert trail.get_label().startswith('_')            # no legend entry


@pytest.mark.parametrize('flag', ['precog', 'bullettime'])
def test_line_marker_format_draws_precog_and_bullettime(flag):
    reference = _plot(fmt='-', **{flag: True}).draw_frame(8)
    anim = _plot(fmt='-o', **{flag: True}).draw_frame(8)
    (trail,), (ref_trail,) = _trails(anim), _trails(reference)
    assert trail.get_marker() == 'o' and trail.get_linestyle() == '-'
    assert _n_vertices(trail) == _n_vertices(ref_trail) > 2


def test_issue_snippet_reports_two_artists_for_both_formats():
    """The issue's own reproduction."""
    seen = {}
    for fmt in ('-', '-o'):
        anim = _plot(fmt=fmt, chemtrails=True).draw_frame(15)
        seen[fmt] = [(line.get_linestyle(), line.get_marker(),
                      line.get_alpha()) for line in _lines(anim)]
    assert seen['-'] == [('-', 'None', None), ('-', 'None', 0.3)]
    assert seen['-o'] == [('-', 'o', None), ('-', 'o', 0.3)]


def test_per_dataset_formats_each_get_their_trail():
    anim = _plot(DATA[:3], fmt=['-o', '-', 'o'], chemtrails=True,
                 ndims=2).draw_frame(15)
    trails = _trails(anim)
    assert [(t.get_linestyle(), t.get_marker()) for t in trails] == [
        ('-', 'o'), ('-', 'None'), ('None', 'o')]
    assert all(_n_vertices(t) > 2 for t in trails)


def test_per_dataset_flags_only_trail_the_flagged_dataset():
    anim = _plot(DATA[:2], fmt='-o', chemtrails=[False, True]).draw_frame(15)
    (trail,) = _trails(anim)
    assert to_rgba(trail.get_color()) == to_rgba(_heads(anim)[1].get_color())


def test_markers_only_format_draws_a_trail_of_faded_markers():
    """A markers-only trail is its faded markers, as the plotly backend
    draws it -- on its own, not only beside a pure line dataset -- so no
    trail flag is ever dropped without a word."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        anim = _plot(fmt='o', chemtrails=True).draw_frame(15)
    (head,), (trail,) = _heads(anim), _trails(anim)
    assert trail.get_linestyle() == 'None' and trail.get_marker() == 'o'
    assert trail.get_alpha() == pytest.approx(0.3)
    # raw rows: the past the head window has left behind
    assert _n_vertices(trail) > _n_vertices(head) > 0


def test_line_marker_trail_reaches_the_pixels():
    """What the figure shows: the frame with the chemtrail has ink where
    the frame without one has none, in the dataset's color, faded."""
    plain = _plot(fmt='-o', ndims=2, markersize=4).draw_frame(15)
    trailed = _plot(fmt='-o', ndims=2, markersize=4,
                    chemtrails=True).draw_frame(15)
    a, b = _rendered(plain.figure), _rendered(trailed.figure)
    assert a.shape == b.shape
    changed = np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max(-1) > 8
    assert changed.sum() > 200
    # the added ink is paler than the head's own color (alpha 0.3 on white)
    head_rgb = np.array(to_rgba(_heads(trailed)[0].get_color())[:3]) * 255
    added = b[changed][:, :3].astype(float)
    assert added.mean() > head_rgb.mean() + 20


def test_serial_reveal_trails_a_line_marker_format():
    anim = _plot(DATA[:2], fmt='-o', animate='serial',
                 chemtrails=True).draw_frame(5)
    trails = _trails(anim)
    assert len(trails) == 2
    # the dataset being revealed traces its past; the other has not begun
    assert _n_vertices(trails[0]) > 2 and _n_vertices(trails[1]) == 0


def test_hue_colored_line_marker_format_gets_a_trail_collection():
    anim = _plot(fmt='-o', chemtrails=True, hue=np.arange(len(X)),
                 ndims=2).draw_frame(15)
    roles = {}
    for coll in anim.figure.axes[0].collections:
        role = getattr(coll, '_hyp_trace_role', None)
        if role is not None:
            roles[role] = coll
    assert set(roles) == {'head', 'trail'}
    assert len(roles['trail'].get_segments()) > 2
    assert roles['trail'].get_alpha() == pytest.approx(0.3)


def test_animated_panel_grid_trails_a_line_marker_format():
    anim = hyp.plot(DATA, '-o', panels=(2, 2), chemtrails=True, **ANIM)
    anim.draw_frame(15)
    for index in range(4):
        (trail,) = _trails(anim, index)
        assert trail.get_marker() == 'o' and trail.get_linestyle() == '-'
        assert _n_vertices(trail) > 2


@pytest.mark.parametrize('fmt', ['-o', 'o-', '--s', 'o'])
def test_plotly_backend_trails_every_format(fmt):
    fig = _plot(fmt=fmt, chemtrails=True, backend='plotly')
    frame = fig.frames[15]
    head, trail = frame.data
    assert len(trail.x) > len(head.x) > 0
    # the trail trace itself: the head's mode, at 0.3 opacity
    trace = fig.data[frame.traces[1]]
    assert trace.mode == fig.data[frame.traces[0]].mode
    assert trace.opacity == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# B1. head_marker=
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ndims', [2, 3])
def test_head_marker_sits_on_the_reveal_head_every_frame(ndims):
    anim = _plot(DATA[:2], chemtrails=True, head_marker='o', ndims=ndims)
    for frame in (0, 7, 19):
        anim.draw_frame(frame)
        heads, markers = _heads(anim), _head_markers(anim)
        assert len(markers) == 2
        for head, marker in zip(heads, markers):
            assert marker.get_linestyle() == 'None'
            assert marker.get_marker() == 'o'
            assert _n_vertices(marker) == 1
            assert marker.get_xdata()[0] == pytest.approx(head.get_xdata()[-1])
            assert marker.get_ydata()[0] == pytest.approx(head.get_ydata()[-1])
            if ndims == 3:
                assert marker.get_data_3d()[2][0] == pytest.approx(
                    head.get_data_3d()[2][-1])
            # the dataset's color, and never a legend entry
            assert to_rgba(marker.get_color()) == to_rgba(head.get_color())
            assert marker.get_label().startswith('_')


def test_head_marker_true_is_a_circle_and_none_draws_nothing():
    assert _head_markers(_plot(head_marker=True))[0].get_marker() == 'o'
    assert _head_markers(_plot(head_marker=None)) == []
    assert _head_markers(_plot(head_marker=False)) == []
    assert _head_markers(_plot()) == []


def test_head_marker_dict_styles_it():
    anim = _plot(ndims=2, head_marker={'marker': 'D', 'size': 13,
                                       'color': 'k', 'edgecolor': 'w',
                                       'edgewidth': 2}).draw_frame(10)
    (marker,) = _head_markers(anim)
    assert marker.get_marker() == 'D'
    assert marker.get_markersize() == 13
    assert to_rgba(marker.get_color()) == to_rgba('k')
    assert to_rgba(marker.get_markeredgecolor()) == to_rgba('w')
    assert marker.get_markeredgewidth() == 2


def test_head_marker_size_follows_the_dataset_and_the_tail_width():
    default = _head_markers(_plot(head_marker='o'))[0].get_markersize()
    assert default == pytest.approx(plt.rcParams['lines.markersize'])
    sized = _head_markers(_plot(head_marker='o', markersize=11))[0]
    assert sized.get_markersize() == 11
    # never lost under a thick tail
    thick = _head_markers(_plot(head_marker='o', tail_linewidth=8))[0]
    assert thick.get_markersize() >= 2 * 8


def test_head_marker_reaches_the_pixels():
    """A black diamond at the reveal head: black ink at the head's own
    pixel, where the same frame without the marker has none."""
    kwargs = dict(ndims=2, linewidth=1)
    plain = _plot(**kwargs).draw_frame(10)
    marked = _plot(head_marker={'marker': 'D', 'size': 14, 'color': 'k'},
                   **kwargs).draw_frame(10)
    head = _heads(marked)[0]
    ax = marked.figure.axes[0]
    image = _rendered(marked.figure)
    px, py = ax.transData.transform(
        (head.get_xdata()[-1], head.get_ydata()[-1]))
    row, col = int(round(image.shape[0] - py)), int(round(px))
    assert image[row, col, :3].max() < 60                 # black
    assert _rendered(plain.figure)[row, col, :3].max() > 60


def test_head_marker_follows_the_hue_color_at_the_head():
    hue = np.arange(len(X))
    anim = _plot(hue=hue, head_marker='o', ndims=2)
    colors = []
    for frame in (2, 18):
        anim.draw_frame(frame)
        colors.append(to_rgba(_head_markers(anim)[0].get_color()))
    assert colors[0] != colors[1]
    pinned = _plot(hue=hue, ndims=2, head_marker={'color': 'k'})
    pinned.draw_frame(18)
    assert to_rgba(_head_markers(pinned)[0].get_color()) == to_rgba('k')


def test_head_marker_in_a_serial_reveal():
    anim = _plot(DATA[:2], animate='serial', head_marker='o').draw_frame(4)
    first, second = _head_markers(anim)
    assert _n_vertices(first) == 1          # being revealed
    assert _n_vertices(second) == 0         # not started: nothing to mark


def test_head_marker_in_an_animated_panel_grid():
    anim = hyp.plot(DATA, '-', panels=(2, 2), head_marker='*', **ANIM)
    anim.draw_frame(9)
    for index in range(4):
        (marker,) = _head_markers(anim, index)
        (head,) = _heads(anim, index)
        assert marker.get_marker() == '*'
        assert marker.get_xdata()[0] == pytest.approx(head.get_xdata()[-1])


@pytest.mark.parametrize('bad', ['not-a-marker', {'marker': 'o', 'colour': 'k'},
                                 {'marker': 'None'}, ''])
def test_head_marker_refuses_what_it_cannot_draw(bad):
    with pytest.raises(ValueError, match='head_marker'):
        _plot(head_marker=bad)


@pytest.mark.parametrize('style', ['spin', 'morph'])
def test_head_marker_says_so_where_there_is_no_reveal_head(style):
    with pytest.warns(UserWarning, match='head_marker'):
        anim = hyp.plot(DATA[:2], 'o' if style == 'morph' else '-',
                        animate=style, duration=2, frame_rate=10,
                        show=False, head_marker='o')
    assert _head_markers(anim) == []


def test_animation_styling_says_so_on_a_static_plot():
    with pytest.warns(UserWarning,
                      match='head_marker/tail_linewidth/trail_alpha only '
                            'affect ANIMATED plots'):
        fig = hyp.plot(X, '-', show=False, head_marker='o',
                       tail_linewidth=3, trail_alpha=0.5)
    assert len(fig.axes[0].lines) == 1


# ---------------------------------------------------------------------------
# B2. tail_linewidth=
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ndims', [2, 3])
def test_tail_linewidth_thickens_the_tail_not_the_trail(ndims):
    anim = _plot(chemtrails=True, linewidth=1.5, tail_linewidth=5,
                 ndims=ndims).draw_frame(15)
    (head,), (trail,) = _heads(anim), _trails(anim)
    assert head.get_linewidth() == 5
    assert trail.get_linewidth() == 1.5


def test_tail_linewidth_default_leaves_both_at_linewidth():
    anim = _plot(chemtrails=True, linewidth=2.5).draw_frame(15)
    assert _heads(anim)[0].get_linewidth() == 2.5
    assert _trails(anim)[0].get_linewidth() == 2.5


def test_tail_linewidth_reaches_the_pixels():
    thin = _plot(ndims=2, chemtrails=True, linewidth=1).draw_frame(15)
    thick = _plot(ndims=2, chemtrails=True, linewidth=1,
                  tail_linewidth=7).draw_frame(15)

    def ink(anim):
        image = _rendered(anim.figure)
        return int((image[..., :3].min(-1) < 200).sum())
    assert ink(thick) > 1.5 * ink(thin)


def test_tail_linewidth_in_a_serial_reveal_is_the_comet_head_only():
    anim = _plot(DATA[:2], animate='serial', chemtrails=True, linewidth=1,
                 tail_linewidth=6)
    anim.draw_frame(4)                    # dataset 0 is being revealed
    assert _heads(anim)[0].get_linewidth() == 6
    anim.draw_frame(19)                   # both finished: drawn in full
    assert [h.get_linewidth() for h in _heads(anim)] == [1, 1]


def test_tail_linewidth_with_a_continuous_hue():
    anim = _plot(hue=np.arange(len(X)), chemtrails=True, linewidth=1.5,
                 tail_linewidth=6, ndims=2).draw_frame(15)
    widths = {getattr(c, '_hyp_trace_role', None): c.get_linewidth()[0]
              for c in anim.figure.axes[0].collections}
    assert widths['head'] == 6 and widths['trail'] == 1.5


@pytest.mark.parametrize('bad', [0, -1, 'thick', True, float('nan')])
def test_tail_linewidth_must_be_a_positive_number(bad):
    with pytest.raises(ValueError, match='tail_linewidth'):
        _plot(tail_linewidth=bad)


# ---------------------------------------------------------------------------
# B3. trail_alpha=
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('flag', ['chemtrails', 'precog', 'bullettime'])
def test_trail_alpha_sets_the_trail_opacity(flag):
    anim = _plot(trail_alpha=0.65, **{flag: True}).draw_frame(8)
    assert _trails(anim)[0].get_alpha() == pytest.approx(0.65)


def test_trail_alpha_multiplies_the_dataset_alpha():
    anim = _plot(chemtrails=True, alpha=0.5, trail_alpha=0.8).draw_frame(8)
    trail = [line for line in _lines(anim)
             if line.get_alpha() == pytest.approx(0.4)]
    assert len(trail) == 1
    default = _plot(chemtrails=True, alpha=0.5).draw_frame(8)
    assert any(line.get_alpha() == pytest.approx(0.15)
               for line in _lines(default))


def test_trail_alpha_with_a_continuous_hue():
    anim = _plot(hue=np.arange(len(X)), chemtrails=True, trail_alpha=0.7,
                 ndims=2).draw_frame(15)
    (trail,) = [c for c in anim.figure.axes[0].collections
                if getattr(c, '_hyp_trace_role', None) == 'trail']
    assert trail.get_alpha() == pytest.approx(0.7)


def test_trail_alpha_reaches_the_pixels():
    def ink(alpha):
        anim = _plot(ndims=2, chemtrails=True, linewidth=3,
                     trail_alpha=alpha, color='k').draw_frame(19)
        image = _rendered(anim.figure)[..., :3].astype(int).max(-1)
        # (black pixels, pixels of black at 20% on white)
        return int((image < 60).sum()), int((abs(image - 204) < 12).sum())
    opaque, faint = ink(1.0), ink(0.2)
    # an opaque trail is black where the faint one is light gray
    assert opaque[0] > faint[0] + 1500
    assert faint[1] > opaque[1] + 1500


def test_trail_alpha_on_plotly():
    fig = _plot(chemtrails=True, trail_alpha=0.65, backend='plotly')
    trace = fig.data[fig.frames[8].traces[1]]
    assert trace.opacity == pytest.approx(0.65)
    default = _plot(chemtrails=True, backend='plotly')
    assert default.data[default.frames[8].traces[1]].opacity == \
        pytest.approx(0.3)


@pytest.mark.parametrize('bad', [-0.1, 1.5, 'faint', True])
def test_trail_alpha_must_be_between_0_and_1(bad):
    with pytest.raises(ValueError, match='trail_alpha'):
        _plot(chemtrails=True, trail_alpha=bad)


def test_trail_alpha_says_so_without_a_trail():
    with pytest.warns(UserWarning, match='trail_alpha'):
        _plot(trail_alpha=0.5)


# ---------------------------------------------------------------------------
# B4. hold=
# ---------------------------------------------------------------------------

def test_hold_appends_frames_that_repeat_the_last_one():
    plain = _plot(chemtrails=True)
    held = _plot(chemtrails=True, hold=1.5)
    assert plain.n_frames == 20
    assert held.n_frames == 20 + 15
    last = _rendered(held.draw_frame(19).figure)
    for frame in (20, 27, 34):
        assert np.array_equal(_rendered(held.draw_frame(frame).figure), last)
    with pytest.raises(IndexError):
        held.draw_frame(35)


def test_hold_zero_is_the_animation_it_always_was():
    assert _plot(hold=0).n_frames == _plot().n_frames == 20
    # a hold shorter than half a frame rounds to no frames
    assert _plot(hold=0.04).n_frames == 20


def test_hold_keeps_the_frame_context_on_the_last_frame(tmp_path):
    """Per-frame texts and callbacks see the LAST index during the hold:
    read from a real save, which is what drives every frame."""
    seen = []
    anim = _plot(ndims=2, hold=0.5,
                 title=lambda ctx: f'title {ctx.frame}',
                 caption=lambda ctx: f'caption {ctx.frame} {ctx.progress}',
                 on_frame=lambda ctx: seen.append(
                     (ctx.frame, ctx.n_frames, ctx.progress)))
    anim.save(str(tmp_path / 'held.gif'))
    drawn = [entry for entry in seen]
    assert len(drawn) >= 25
    assert drawn[-5:] == [(19, 20, 1.0)] * 5
    assert drawn[-6] == (19, 20, 1.0) and drawn[-7][0] == 18
    assert anim.figure.axes[0].get_title() == 'title 19'
    assert _figure_text(anim.figure, CAPTION_GID).get_text() == \
        'caption 19 1.0'


def test_hold_is_in_the_saved_gif(tmp_path):
    def playback(path):
        with Image.open(path) as im:
            total = 0
            for index in range(getattr(im, 'n_frames', 1)):
                im.seek(index)
                total += im.info.get('duration', 0)
        return total / 1000.0
    plain, held = str(tmp_path / 'plain.gif'), str(tmp_path / 'held.gif')
    _plot(ndims=2, save_path=plain)
    _plot(ndims=2, save_path=held, hold=1.5)
    assert playback(plain) == pytest.approx(2.0, abs=0.15)
    assert playback(held) == pytest.approx(3.5, abs=0.15)


def _video_frames(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_frames', '-show_entries', 'stream=nb_read_frames',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split(',')[0])


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg/ffprobe not installed')
def test_hold_is_in_the_saved_mp4(tmp_path):
    plain, held = tmp_path / 'plain.mp4', tmp_path / 'held.mp4'
    _plot(save_path=str(plain), rotations=1)
    _plot(save_path=str(held), rotations=1, hold=1)
    assert _video_frames(plain) == 20
    assert _video_frames(held) == 30


def test_hold_stops_the_camera_with_rotations():
    anim = _plot(rotations=2, hold=1)
    ax = anim.figure.axes[0]
    anim.draw_frame(19)
    last = ax.azim
    anim.draw_frame(25)
    assert ax.azim == pytest.approx(last)
    anim.draw_frame(10)
    assert ax.azim != pytest.approx(last)


def test_hold_adds_to_duration_at_the_frame_rate():
    anim = hyp.plot(X, '-', animate=True, duration=3, frame_rate=8,
                    hold=2, show=False)
    assert anim.n_frames == 3 * 8 + 2 * 8


def test_hold_composes_with_a_looping_morph():
    clouds = [np.random.default_rng(s).normal(size=(40, 3)) for s in range(3)]
    call = dict(animate='morph', loop=True, duration=2, frame_rate=10,
                show=False)
    plain = hyp.plot(clouds, 'o', **call)
    held = hyp.plot(clouds, 'o', hold=1, **call)
    assert held.n_frames == plain.n_frames + 10
    assert held.n_segments == plain.n_segments
    last = _rendered(held.draw_frame(plain.n_frames - 1).figure)
    assert np.array_equal(
        _rendered(held.draw_frame(held.n_frames - 1).figure), last)


@pytest.mark.parametrize('style', ['spin', 'serial', 'window'])
def test_hold_on_every_style(style):
    anim = hyp.plot(DATA[:2], '-', animate=style, duration=2, frame_rate=10,
                    hold=0.5, show=False)
    assert anim.n_frames == 25
    anim.draw_frame(24)


def test_hold_inside_the_animate_dict():
    anim = hyp.plot(X, '-', show=False, animate={
        'style': 'parallel', 'duration': 2, 'frame_rate': 10, 'hold': 1,
        'head_marker': 'o', 'tail_linewidth': 3, 'chemtrails': True,
        'trail_alpha': 0.5})
    assert anim.n_frames == 30
    anim.draw_frame(29)
    assert len(_head_markers(anim)) == 1
    assert _heads(anim)[0].get_linewidth() == 3
    assert _trails(anim)[0].get_alpha() == pytest.approx(0.5)


def test_hold_on_an_animated_panel_grid(tmp_path):
    seen = []
    anim = hyp.plot(DATA, '-', panels=(2, 2), hold=0.5,
                    caption=lambda ctx: f'frame {ctx.frame + 1}',
                    on_frame=lambda ctx: seen.append(ctx.frame), **ANIM)
    assert anim.n_frames == 25
    anim.save(str(tmp_path / 'grid.gif'))
    assert seen[-6:] == [19] * 6
    assert _figure_text(anim.figure, CAPTION_GID).get_text() == 'frame 20'


def test_hold_drawn_extent_and_sampling_still_work():
    anim = _plot(hold=1, title=lambda ctx: f'frame {ctx.frame}',
                 caption=lambda ctx: f'frame {ctx.frame}')
    box = anim.drawn_extent(frames=[0, 29])
    assert 0 <= box.x0 < box.x1 <= 1


@pytest.mark.parametrize('bad', [-1, 'long', True, float('inf')])
def test_hold_must_be_seconds(bad):
    with pytest.raises(ValueError, match='hold'):
        _plot(hold=bad)


def test_hold_requires_an_animation():
    with pytest.raises(ValueError, match='hold= requires an animated plot'):
        hyp.plot(X, '-', show=False, hold=1)
    with pytest.raises(ValueError, match='hold= requires an animated plot'):
        hyp.plot(DATA, '-', panels=(2, 2), show=False, hold=1)


@pytest.mark.parametrize('kwargs', [dict(head_marker='o'),
                                    dict(tail_linewidth=3), dict(hold=1)])
def test_frame_driven_arguments_are_refused_on_plotly(kwargs):
    (name,) = kwargs
    with pytest.raises(NotImplementedError,
                       match=f'{name}= is matplotlib-only'):
        _plot(backend='plotly', **kwargs)


# ---------------------------------------------------------------------------
# B5. caption_kwargs=
# ---------------------------------------------------------------------------

def _inside_canvas(box, slack=1e-6):
    return (box.x0 >= -slack and box.x1 <= 1 + slack
            and box.y0 >= -slack and box.y1 <= 1 + slack)


CAPTION = 'a caption for the whole figure'


@pytest.mark.parametrize('size', [6, 12, 28, 44])
@pytest.mark.parametrize('animated', [False, True])
def test_caption_of_any_size_is_measured_into_the_canvas(size, animated):
    call = dict(caption=CAPTION, caption_kwargs={'size': size}, show=False)
    if animated:
        call.update(animate=True, duration=1, frame_rate=8)
    result = hyp.plot(X, '-', **call)
    fig = result.figure if animated else result
    caption = _figure_text(fig, CAPTION_GID)
    assert caption.get_fontsize() == size
    box = _extent(fig, caption)
    assert _inside_canvas(box)
    # below everything else the figure draws
    assert box.y1 <= _content_bottom(fig, skip=[caption]) + 1e-6


def test_larger_caption_gets_more_room():
    heights = []
    for size in (8, 30):
        fig = hyp.plot(X, '-', animate=True, duration=1, frame_rate=8,
                       show=False, caption=CAPTION,
                       caption_kwargs={'size': size}).figure
        heights.append(fig.get_size_inches()[1])
    assert heights[1] > heights[0]


def test_caption_style_is_applied():
    fig = hyp.plot(X, '-', show=False, caption=CAPTION, caption_kwargs={
        'color': 'crimson', 'weight': 'bold', 'style': 'italic',
        'alpha': 0.5, 'family': 'monospace'})
    caption = _figure_text(fig, CAPTION_GID)
    assert to_rgba(caption.get_color()) == to_rgba('crimson')
    assert caption.get_fontweight() == 'bold'
    assert caption.get_fontstyle() == 'italic'
    assert caption.get_alpha() == 0.5
    assert caption.get_fontfamily() == ['monospace']


@pytest.mark.parametrize('animated', [False, True])
def test_caption_at_the_top_sits_above_everything(animated):
    call = dict(title='the axes title', caption=CAPTION, show=False,
                caption_kwargs={'position': 'top', 'size': 20})
    if animated:
        call.update(animate=True, duration=1, frame_rate=8)
    result = hyp.plot(X, '-', **call)
    fig = result.figure if animated else result
    caption = _figure_text(fig, CAPTION_GID)
    box = _extent(fig, caption)
    assert _inside_canvas(box)
    assert box.y0 >= _content_top(fig, skip=[caption]) - 1e-6
    title = _extent(fig, fig.axes[0].title)
    assert box.y0 >= title.y1


def test_caption_alignment():
    boxes = {}
    for loc in ('left', 'center', 'right'):
        fig = hyp.plot(X, '-', show=False, caption='short',
                       caption_kwargs={'loc': loc})
        boxes[loc] = _extent(fig, _figure_text(fig, CAPTION_GID))
        assert _inside_canvas(boxes[loc])
    assert boxes['left'].x0 < 0.05 and boxes['left'].x1 < 0.5
    assert boxes['right'].x1 > 0.95 and boxes['right'].x0 > 0.5
    assert (boxes['center'].x0 + boxes['center'].x1) / 2 == \
        pytest.approx(0.5, abs=0.01)


def test_long_styled_caption_wraps_inside_the_canvas():
    fig = hyp.plot(X, '-', show=False, caption='word ' * 60,
                   caption_kwargs={'size': 22, 'loc': 'left'})
    caption = _figure_text(fig, CAPTION_GID)
    assert caption.get_text().count('\n') >= 2
    assert _inside_canvas(_extent(fig, caption))


def test_per_frame_caption_at_the_top_of_an_animation():
    anim = hyp.plot(X, '-', animate=True, duration=1, frame_rate=8,
                    show=False, title=lambda ctx: f'title {ctx.frame}',
                    caption=lambda ctx: f'caption {ctx.frame}',
                    caption_kwargs={'position': 'top', 'size': 18})
    caption = _figure_text(anim.figure, CAPTION_GID)
    for frame in (0, 4, 7):
        anim.draw_frame(frame)
        assert caption.get_text() == f'caption {frame}'
        box = _extent(anim.figure, caption)
        assert _inside_canvas(box)
        assert box.y0 >= _extent(anim.figure, anim.figure.axes[0].title).y1


@pytest.mark.parametrize('bad, error', [
    ({'colour': 'red'}, ValueError),
    ({'position': 'middle'}, ValueError),
    ({'loc': 'top'}, ValueError),
    ('big', TypeError),
])
def test_caption_kwargs_are_validated(bad, error):
    with pytest.raises(error, match='caption_kwargs'):
        hyp.plot(X, '-', show=False, caption=CAPTION, caption_kwargs=bad)


def test_caption_kwargs_without_a_caption_are_refused():
    with pytest.raises(ValueError, match='no caption= was given'):
        hyp.plot(X, '-', show=False, caption_kwargs={'size': 12})
    with pytest.raises(ValueError, match='no caption= was given'):
        hyp.plot(DATA, '-', panels=(2, 2), show=False,
                 caption_kwargs={'size': 12})


# ---------------------------------------------------------------------------
# B6. suptitle=: a figure-level title
# ---------------------------------------------------------------------------

PANEL_TITLES = ['one', 'two', 'three', 'four']
SUPTITLE = 'Four walks, one figure'


def _grid(animated, **kwargs):
    call = dict(panels=(2, 2), title=PANEL_TITLES, show=False,
                legend=['a', 'b', 'c', 'd'], suptitle=SUPTITLE)
    if animated:
        call.update(animate=True, duration=1, frame_rate=8, chemtrails=True)
    call.update(kwargs)
    result = hyp.plot(DATA, '-', **call)
    return result, (result.figure if animated else result)


def test_single_string_title_still_names_every_panel():
    """`title=` on a grid is per panel (documented), which is why the
    figure-level title is its own argument."""
    fig = hyp.plot(DATA, '-', panels=(2, 2), title='same', show=False)
    assert [ax.get_title() for ax in fig.axes[:4]] == ['same'] * 4
    assert fig.findobj(lambda a: a.get_gid() == SUPTITLE_GID) == []


@pytest.mark.parametrize('animated', [False, True])
@pytest.mark.parametrize('style', [None, {'size': 30, 'color': 'navy'}])
def test_grid_suptitle_is_above_everything_and_inside_the_canvas(
        animated, style):
    result, fig = _grid(animated, suptitle_kwargs=style)
    suptitle = _figure_text(fig, SUPTITLE_GID)
    assert suptitle.get_text() == SUPTITLE
    frames = (0, 4, 7) if animated else (None,)
    for frame in frames:
        if frame is not None:
            result.draw_frame(frame)
        box = _extent(fig, suptitle)
        assert _inside_canvas(box)
        # clear of every panel title...
        for ax in fig.axes[:4]:
            assert ax.get_title() in PANEL_TITLES
            assert box.y0 >= _extent(fig, ax.title).y1
        # ...of the shared legend, and of everything else drawn
        for legend in list(fig.legends) + [
                ax.get_legend() for ax in fig.axes if ax.get_legend()]:
            assert box.y0 >= _extent(fig, legend).y1
        assert box.y0 >= _content_top(fig, skip=[suptitle]) - 1e-6
    if style:
        assert suptitle.get_fontsize() == 30
        assert to_rgba(suptitle.get_color()) == to_rgba('navy')


def test_grid_suptitle_adds_its_own_room():
    """Nothing already laid out is squeezed: the figure grows by the
    title's strip, and a larger title grows it more."""
    _, bare = _grid(False, suptitle=None)
    _, titled = _grid(False)
    _, large = _grid(False, suptitle_kwargs={'size': 40})
    heights = [f.get_size_inches()[1] for f in (bare, titled, large)]
    assert heights[0] < heights[1] < heights[2]
    widths = [f.get_size_inches()[0] for f in (bare, titled, large)]
    assert widths[0] == pytest.approx(widths[1]) == pytest.approx(widths[2])
    for before, after in zip(bare.axes[:4], titled.axes[:4]):
        b, a = before.get_position(), after.get_position()
        assert b.height * heights[0] == pytest.approx(a.height * heights[1])
        assert b.width == pytest.approx(a.width)


@pytest.mark.parametrize('animated', [False, True])
def test_grid_suptitle_with_a_colorbar(animated):
    hue = [np.arange(len(d)) for d in DATA]
    result, fig = _grid(animated, legend=None, hue=hue,
                        colorbar={'label': 'time'})
    suptitle = _figure_text(fig, SUPTITLE_GID)
    box = _extent(fig, suptitle)
    assert _inside_canvas(box)
    assert len(fig.axes) > 4                       # the colorbar's axes
    for ax in fig.axes:
        assert box.y0 >= _extent(fig, ax).y1 - 1e-6
    assert box.y0 >= _content_top(fig, skip=[suptitle]) - 1e-6


@pytest.mark.parametrize('animated', [False, True])
def test_suptitle_sits_above_a_top_caption_and_a_bottom_one_stays_put(
        animated):
    result, fig = _grid(animated, caption=CAPTION,
                        caption_kwargs={'position': 'top'})
    suptitle = _figure_text(fig, SUPTITLE_GID)
    caption = _figure_text(fig, CAPTION_GID)
    assert _extent(fig, suptitle).y0 >= _extent(fig, caption).y1
    assert _extent(fig, caption).y0 >= _content_top(
        fig, skip=[suptitle, caption]) - 1e-6

    result, fig = _grid(animated, caption=CAPTION,
                        caption_kwargs={'size': 20})
    suptitle = _figure_text(fig, SUPTITLE_GID)
    caption = _figure_text(fig, CAPTION_GID)
    box = _extent(fig, caption)
    assert _inside_canvas(box) and _inside_canvas(_extent(fig, suptitle))
    assert box.y1 <= _content_bottom(fig, skip=[caption]) + 1e-6


def test_animated_grid_suptitle_clears_per_frame_panel_titles():
    """A per-frame panel title is taller on some frames than on frame 0:
    the figure title's room is measured over the sampled frames."""
    def title(ctx):
        return 'short' if ctx.frame < 4 else 'two\nlines'
    anim = hyp.plot(DATA, '-', panels=(2, 2), title=title, ndims=2,
                    suptitle=SUPTITLE, animate=True, duration=1,
                    frame_rate=8, show=False)
    suptitle = _figure_text(anim.figure, SUPTITLE_GID)
    for frame in (0, 7):
        anim.draw_frame(frame)
        box = _extent(anim.figure, suptitle)
        assert _inside_canvas(box)
        for ax in anim.figure.axes[:4]:
            assert box.y0 >= _extent(anim.figure, ax.title).y1


@pytest.mark.parametrize('animated', [False, True])
def test_suptitle_on_a_single_plot(animated):
    call = dict(title='the axes title', suptitle=SUPTITLE, show=False)
    if animated:
        call.update(animate=True, duration=1, frame_rate=8)
    result = hyp.plot(X, '-', **call)
    fig = result.figure if animated else result
    suptitle = _figure_text(fig, SUPTITLE_GID)
    box = _extent(fig, suptitle)
    assert _inside_canvas(box)
    assert box.y0 >= _extent(fig, fig.axes[0].title).y1
    assert fig.axes[0].get_title() == 'the axes title'


def test_suptitle_survives_a_saved_still(tmp_path):
    """Read back from the saved PNG: the strip above the panels has ink
    that the same grid without a title does not have."""
    paths = []
    for name, suptitle in (('bare.png', None), ('titled.png', SUPTITLE)):
        paths.append(str(tmp_path / name))
        hyp.plot(DATA, '-', panels=(2, 2), show=False, suptitle=suptitle,
                 suptitle_kwargs=({'color': 'k', 'size': 24}
                                  if suptitle else None),
                 save_path=paths[-1])
    bare, titled = (np.asarray(Image.open(p).convert('RGB')) for p in paths)
    extra = titled.shape[0] - bare.shape[0]
    assert extra > 0
    strip = titled[:extra]
    assert (strip.max(-1) < 80).sum() > 100           # the title's ink
    # the panels below are what they were (to within the antialiasing of
    # a strip that is not a whole number of pixels tall)
    below = titled[extra:].astype(int)
    assert (np.abs(below - bare.astype(int)).max(-1) > 128).mean() < 0.01
    assert (strip.max(-1) < 80).sum() > 20 * (
        bare[:extra].max(-1) < 80).sum()


def test_suptitle_validation():
    with pytest.raises(TypeError, match='suptitle= must be a string'):
        hyp.plot(X, '-', show=False, suptitle=['a', 'b'])
    with pytest.raises(ValueError, match='no suptitle= was given'):
        hyp.plot(X, '-', show=False, suptitle_kwargs={'size': 20})
    with pytest.raises(ValueError, match='suptitle_kwargs'):
        hyp.plot(X, '-', show=False, suptitle='t',
                 suptitle_kwargs={'position': 'bottom'})
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='caller-supplied ax='):
        hyp.plot(X, '-', show=False, suptitle='t', ax=ax, ndims=2)
    with pytest.raises(NotImplementedError,
                       match='suptitle= is matplotlib-only'):
        hyp.plot(X, '-', show=False, suptitle='t', backend='plotly')
    with pytest.raises(NotImplementedError,
                       match='suptitle= is matplotlib-only'):
        hyp.plot(DATA, '-', panels=(2, 2), show=False, suptitle='t',
                 backend='plotly')
    # an empty title is no title
    fig = hyp.plot(X, '-', show=False, suptitle='  ')
    assert fig.findobj(lambda a: a.get_gid() == SUPTITLE_GID) == []
