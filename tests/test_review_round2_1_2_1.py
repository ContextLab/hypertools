"""1.2.1 review round 2: what a second pair of reviews found in the GH #299
animation arguments and the GH #297 layout fixes, and what was found next
to each fix.

A. Values inside the styling dicts (``title_kwargs=``,
   ``caption_kwargs=``, ``suptitle_kwargs=``, ``head_marker=``):
   matplotlib decides what a property accepts.
B. Observation labels: the labels of every call into one axes, an axes
   that was cleared, one arrangement at every dpi and in every file
   format, a 3-D plot turned with the mouse, no work on a redraw, no
   draw to place an animation's labels.
C. A measurement leaves the figure its own canvas.
D. Warnings about ignored styling in ``panels=`` grids; per-dataset trail
   lists with a categorical ``hue=``; ``labels=`` with ``animate='morph'``.
E. Continuous ``hue=`` markers wear the dataset's marker styling, and a
   trace's second artist follows the one a callback is handed.
F. A per-frame caption or title is never clipped, whichever frame it is
   tallest on.

Real figures and real frames, rendered headless. Nothing is mocked (two
tests COUNT the draws matplotlib makes, through a wrapper that calls the
real method), and no assertion depends on a font's absolute metrics.
"""
import io
import os
import subprocess
import sys
import textwrap
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import matplotlib.text as mtext                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.backend_bases import MouseButton, MouseEvent    # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg     # noqa: E402
from matplotlib.collections import PathCollection               # noqa: E402
from matplotlib.figure import Figure                            # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.matplotlib_backend import (                # noqa: E402
    LABEL_OFFSET_PT, _LabelDeclutter, forget_label_tracks, label_tracks)
from hypertools.plot.plot import CAPTION_GID, SUPTITLE_GID      # noqa: E402


def _walk(seed, rows=40, cols=3):
    return np.random.default_rng(seed).normal(size=(rows, cols)).cumsum(0)


X, Y, Z = _walk(0), _walk(1), _walk(2)
HUE = np.linspace(0.0, 1.0, len(X))
HOME = (-LABEL_OFFSET_PT, LABEL_OFFSET_PT)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _figure_of(result):
    return getattr(result, 'figure', result)


# --------------------------------------------------------------------------
# A. values inside the styling dicts: matplotlib decides
# --------------------------------------------------------------------------

#: values matplotlib accepts and 1.2.0 drew, which the first 1.2.1 check
#: refused
ACCEPTED_TEXT_STYLES = [
    {'size': None}, {'alpha': None}, {'size': '12'}, {'weight': '700'},
    {'alpha': True}, {'family': None}, {'weight': None}, {'style': None},
    {'size': 'large'}, {'size': np.float32(13)}, {'weight': 550},
    {'family': ['serif', 'DejaVu Sans']}, {'color': ('red', 0.5)},
    {'color': '0.5'}, {'size': 0.5},
]


@pytest.mark.parametrize('style', ACCEPTED_TEXT_STYLES)
@pytest.mark.parametrize('animate', [False, True])
def test_title_kwargs_take_what_matplotlib_takes(style, animate):
    extra = dict(animate=True, duration=1, frame_rate=5) if animate else {}
    result = hyp.plot(X, '-', title='A title', title_kwargs=style,
                      show=False, **extra)
    fig = _figure_of(result)
    fig.canvas.draw()
    assert fig.axes[0].get_title() == 'A title'


@pytest.mark.parametrize('style', ACCEPTED_TEXT_STYLES + [{'color': None}])
@pytest.mark.parametrize('name', ['caption_kwargs', 'suptitle_kwargs'])
def test_figure_text_kwargs_take_what_matplotlib_takes(name, style):
    text = name.split('_')[0]
    fig = hyp.plot(X, '-', show=False, **{text: 'Some text', name: style})
    fig.canvas.draw()
    gid = CAPTION_GID if text == 'caption' else SUPTITLE_GID
    assert fig.findobj(lambda a: a.get_gid() == gid)[0].get_text() \
        == 'Some text'


def test_a_size_given_as_a_string_is_that_size():
    fig = hyp.plot(X, '-', show=False, title='T', title_kwargs={'size': '17'},
                   caption='C', caption_kwargs={'size': '9', 'weight': '700'})
    assert fig.axes[0].title.get_fontsize() == 17
    caption = fig.findobj(lambda a: a.get_gid() == CAPTION_GID)[0]
    assert caption.get_fontsize() == 9 and caption.get_fontweight() == 700


def test_none_in_a_figure_text_dict_is_the_default():
    plain = hyp.plot(X, '-', show=False, caption='C', suptitle='S')
    styled = hyp.plot(X, '-', show=False, caption='C', suptitle='S',
                      caption_kwargs={'size': None, 'color': None,
                                      'weight': None, 'linespacing': None,
                                      'loc': None},
                      suptitle_kwargs={'size': None, 'family': None})
    for gid in (CAPTION_GID, SUPTITLE_GID):
        one, other = (fig.findobj(lambda a: a.get_gid() == gid)[0]
                      for fig in (plain, styled))
        assert one.get_fontsize() == other.get_fontsize()
        assert one.get_color() == other.get_color()
        assert one.get_fontweight() == other.get_fontweight()
        assert one.get_position() == other.get_position()


def test_title_kwargs_agree_with_matplotlib_over_many_values():
    """Whatever matplotlib's own `Text` takes for a property, the title
    takes; whatever it refuses is refused with the argument and key
    named. The only values refused beyond matplotlib's are sizes and
    line spacings that are not positive and finite."""
    nan = float('nan')
    values = {
        'size': [None, 12, '12', '12.5', 'x-small', 'smaller', True, 0.5,
                 40, 'huge', 'bogus', [12]],
        'weight': [None, 'bold', 700, '700', 0, 1000, 1001, -1, True,
                   'bogus', nan, 'roman'],
        'style': [None, 'italic', 'oblique', 'bogus', 3],
        'family': [None, 'serif', ['serif'], ('serif',)],
        'color': ['red', 'C1', '#ff000080', (1, 0, 0, .5), ('red', .5),
                  '0.5', 'bogus', None, 0.5, (1, 2, 3)],
        'alpha': [None, 0, 1, .5, True, False, '0.5', 2, -1, nan],
        'variant': [None, 'small-caps', 'bogus'],
        'stretch': [None, 'condensed', 500, 'bogus', 1001],
        'rotation': [None, 30, 'vertical', 'bogus'],
        'backgroundcolor': ['yellow', 'bogus'],
        'ha': ['left', 'bogus'], 'va': ['top', 'bogus'],
        'wrap': [True, False],
    }
    full = {'size': 'fontsize', 'weight': 'fontweight', 'style': 'fontstyle',
            'family': 'fontfamily', 'variant': 'fontvariant',
            'stretch': 'fontstretch'}
    checked = 0
    for key, candidates in values.items():
        for value in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mtext.Text().update({full.get(key, key): value})
                accepted = True
            except Exception:
                accepted = False
            if accepted:
                fig = hyp.plot(X, '-', title='A title',
                               title_kwargs={key: value}, show=False)
                fig.canvas.draw()
                assert fig.axes[0].get_title() == 'A title', (key, value)
            else:
                with pytest.raises(ValueError, match=(
                        f"title_kwargs\\['{key}'\\] must be")):
                    hyp.plot(X, '-', title='A title',
                             title_kwargs={key: value}, show=False)
            plt.close('all')
            checked += 1
    assert checked > 60


@pytest.mark.parametrize('style, key', [
    ({'size': -3}, 'size'), ({'size': 0}, 'size'),
    ({'size': float('nan')}, 'size'), ({'size': '-3'}, 'size'),
    ({'size': float('inf')}, 'size'),
    ({'size': '0'}, 'size'), ({'size': False}, 'size'),
    ({'linespacing': 0}, 'linespacing'), ({'color': 'auto'}, 'color'),
    ({'family': 7}, 'family')])
@pytest.mark.parametrize('name', ['title_kwargs', 'caption_kwargs',
                                  'suptitle_kwargs'])
def test_what_draws_nothing_readable_is_still_refused_by_name(
        name, style, key):
    text = name.split('_')[0]
    with pytest.raises(ValueError, match=f"{name}\\['{key}'\\] must be"):
        hyp.plot(X, '-', show=False, **{text: 'Some text', name: style})


def _head_marker(anim):
    return [line for line in anim.figure.axes[0].lines
            if getattr(line, '_hyp_trace_role', None) == 'head_marker']


@pytest.mark.parametrize('spec', [
    {'color': None}, {'size': None}, {'alpha': None}, {'edgecolor': None},
    {'edgewidth': None}, {'fillstyle': None}, {'zorder': None},
    {'marker': None}, {'edgecolor': 'auto'}, {'alpha': True},
    {'size': '9'}])
def test_head_marker_dict_takes_what_matplotlib_takes(spec):
    anim = hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                    frame_rate=10, show=False)
    anim.draw_frame(4)
    (marker,) = _head_marker(anim)
    assert marker.get_marker() == 'o'
    line = anim.figure.axes[0].lines[0]
    if spec.get('color', 0) is None:
        # None is "the default": the dataset's color, not the next color
        # of the axes' cycle
        assert matplotlib.colors.same_color(marker.get_color(),
                                            line.get_color())
    if spec.get('size') == '9':
        assert marker.get_markersize() == 9


@pytest.mark.parametrize('spec', [{'marker': True}, True, np.True_])
def test_head_marker_true_is_a_circle_wherever_it_is_written(spec):
    """`{'marker': True}` drew matplotlib's marker number 1, a tick."""
    anim = hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                    frame_rate=10, show=False)
    (marker,) = _head_marker(anim)
    assert marker.get_marker() == 'o'


@pytest.mark.parametrize('spec', [{'marker': False}, False, np.False_])
def test_head_marker_false_is_no_marker_wherever_it_is_written(spec):
    anim = hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                    frame_rate=10, show=False)
    assert _head_marker(anim) == []


TRIANGLE = np.array([[0, 1], [-1, -1], [1, -1]])


@pytest.mark.parametrize('spec', [
    TRIANGLE, {'marker': TRIANGLE}, TRIANGLE.tolist(), (5, 1), '$A$'],
    ids=['array', 'array in dict', 'list', 'tuple', 'mathtext'])
def test_head_marker_takes_every_kind_of_matplotlib_marker(spec):
    """A marker given as an array of vertices failed with "needs a
    matplotlib marker" (the truth value of an array)."""
    anim = hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                    frame_rate=10, show=False)
    anim.draw_frame(5)
    anim.figure.canvas.draw()
    (marker,) = _head_marker(anim)
    assert len(marker.get_xdata()) == 1


@pytest.mark.parametrize('spec, match', [
    ({'edgecolor': 'bogus'}, "head_marker\\['edgecolor'\\]"),
    ({'color': 'auto'}, "head_marker\\['color'\\]"),
    ({'size': '-3'}, "head_marker\\['size'\\]"),
    (np.array([1.0, 2.0, 3.0]), 'head_marker= needs a matplotlib marker'),
    ('', 'head_marker= needs a matplotlib marker')])
def test_bad_head_markers_are_refused_by_name(spec, match):
    with pytest.raises(ValueError, match=match):
        hyp.plot(X, '-', animate=True, head_marker=spec, duration=1,
                 frame_rate=10, show=False)


# --------------------------------------------------------------------------
# B. observation labels
# --------------------------------------------------------------------------

def _entries(ax):
    return [entry for entry in ax._hyp_labels_and_points
            if entry[0].axes is ax]


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


def _places(labels):
    return [tuple(np.round(label.xyann, 3)) for label in labels]


def _two_calls(ndims):
    rng = np.random.default_rng(0)
    first = rng.normal(size=(6, ndims))
    second = np.array([[0, 0], [.02, .01], [.04, 0], [.01, .03],
                       [3, 3], [-3, -3.]])
    if ndims == 3:
        second = np.column_stack([second, rng.normal(size=6) * .01])
    return first, second


@pytest.mark.parametrize('ndims', [2, 3])
def test_two_labeled_calls_into_one_axes_are_kept_off_each_other(ndims):
    """The second call's list replaced the first's on the axes, so only
    the last call's labels were ever moved."""
    first, second = _two_calls(ndims)
    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    hyp.plot(first, 'o', labels=[f'first {i}' for i in range(6)], ax=ax,
             show=False)
    hyp.plot(second, 'o', labels=[f'second {i}' for i in range(6)], ax=ax,
             show=False)
    texts = [entry[0].get_text() for entry in _entries(ax)]
    assert texts == [f'first {i}' for i in range(6)] + [
        f'second {i}' for i in range(6)]
    labels = [entry[0] for entry in _entries(ax)]
    assert _overlapping_pairs(fig, labels) == 0
    # ...as one call with the second call's data arranges them
    single = hyp.plot(second, 'o', labels=[f'second {i}' for i in range(6)],
                      show=False)
    assert _overlapping_pairs(
        single, [entry[0] for entry in _entries(single.axes[0])]) == 0


def test_two_labeled_calls_each_answer_their_own_mouse_events():
    """Each call still hides and restores ITS labels on a mouse press."""
    first, second = _two_calls(3)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    hyp.plot(first, 'o', labels=list('abcdef'), ax=ax, show=False)
    hyp.plot(second, 'o', labels=list('uvwxyz'), ax=ax, show=False)
    fig.canvas.draw()
    labels = [entry[0] for entry in _entries(ax)]
    x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2
    fig.canvas.callbacks.process('button_press_event', MouseEvent(
        'button_press_event', fig.canvas, x, y, button=MouseButton.LEFT))
    assert not any(label.get_visible() for label in labels)
    fig.canvas.callbacks.process('button_release_event', MouseEvent(
        'button_release_event', fig.canvas, x, y, button=MouseButton.LEFT))
    assert all(label.get_visible() for label in labels)


def _crowded(n=30, ndims=3, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, ndims))
    crowd = slice(n // 3, 2 * n // 3)
    x[crowd] = x[n // 3] + rng.normal(
        scale=.03, size=(len(x[crowd]), ndims))
    return x, [f'label {i}' for i in range(n)]


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_cleared_axes_declutters_its_new_labels(ndims):
    """`ax.clear()` removed the declutter artist from the axes and left
    the attribute that said it was there, so it was never put back."""
    x, names = _crowded(ndims=ndims)
    fig, ax = plt.subplots(
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    hyp.plot(x, 'o', labels=names, ax=ax, show=False)
    fig.canvas.draw()
    ax.clear()
    hyp.plot(x[::-1], 'o', labels=names, ax=ax, show=False)
    watchers = [child for child in ax.get_children()
                if isinstance(child, _LabelDeclutter)]
    assert len(watchers) == 1
    assert len(_entries(ax)) == len(names) == len(ax.texts)
    fig.canvas.draw()
    labels = [entry[0] for entry in _entries(ax)]
    assert sum(place != HOME for place in _places(labels)) >= 3
    # the same arrangement as the same call into an axes never used
    fresh_fig, fresh_ax = plt.subplots(
        subplot_kw={'projection': '3d'} if ndims == 3 else {})
    hyp.plot(x[::-1], 'o', labels=names, ax=fresh_ax, show=False)
    fresh_fig.canvas.draw()
    assert _places(labels) == _places(
        [entry[0] for entry in _entries(fresh_ax)])


def test_an_axes_reused_after_an_animation_places_labels_again():
    """An animation's hold on its labels (since review round 5 its
    table, `label_tracks`) stayed on the axes after `ax.clear()`: a
    static plot's labels were never moved."""
    x, names = _crowded()
    anim = hyp.plot(x, 'o', labels=names, animate='spin', duration=1,
                    frame_rate=5, show=False)
    fig = anim.figure
    ax = fig.axes[0]
    anim.draw_frame(2)
    assert label_tracks(ax) is not None
    ax.clear()
    hyp.plot(x, 'o', labels=names, ax=ax, show=False)
    assert label_tracks(ax) is None
    fig.canvas.draw()
    labels = [entry[0] for entry in _entries(ax)]
    assert len(labels) == len(names)
    assert sum(place != HOME for place in _places(labels)) >= 3


def test_labels_added_to_an_animations_axes_join_its_table():
    """Without a clear, the animation's labels are still there: the new
    ones are placed with them, in the animation's one table (rewritten
    in review round 5 from the test that all of them hold one place), so
    every frame shows all 24 the same way whenever it is drawn."""
    x, names = _crowded(n=12)
    anim = hyp.plot(x, 'o', labels=names, animate='spin', duration=1,
                    frame_rate=5, show=False)
    ax = anim.figure.axes[0]
    hyp.plot(x + 0.01, 'o', labels=[name.upper() for name in names], ax=ax,
             show=False)
    tracks = label_tracks(ax)
    labels = [entry[0] for entry in _entries(ax)]
    assert len(labels) == 24
    assert [ref() for ref in tracks['labels']] == labels
    home = tracks['places'][0].min()
    assert (tracks['places'] != home).any(axis=0).sum() >= 3
    seen = {}
    for frame in (3, 1, 4, 1, 3):
        anim.draw_frame(frame)
        anim.figure.canvas.draw()
        assert seen.setdefault(frame, _places(labels)) == _places(labels)
    assert label_tracks(ax) is tracks


@pytest.mark.parametrize('ndims', [2, 3])
def test_the_arrangement_is_the_same_at_every_dpi_and_in_every_format(
        ndims, tmp_path):
    """The placement was made from the renderer's pixels: 10 of 30 labels
    sat elsewhere at dpi 101 than at dpi 100, and a PDF differed again."""
    x, names = _crowded(ndims=ndims)

    def arrangement(save):
        fig = hyp.plot(x, 'o', labels=names, show=False)
        save(fig)
        places = _places([entry[0] for entry in _entries(fig.axes[0])])
        plt.close(fig)
        return places

    reference = arrangement(lambda fig: fig.canvas.draw())
    assert sum(place != HOME for place in reference) >= 3
    for dpi in (72, 100, 101, 120, 150, 300):
        assert arrangement(lambda fig: fig.savefig(
            io.BytesIO(), format='png', dpi=dpi)) == reference, dpi
    for fmt in ('pdf', 'svg'):
        assert arrangement(lambda fig: fig.savefig(
            tmp_path / f'labels.{fmt}')) == reference, fmt
    # the screen first, then files of the same figure
    fig = hyp.plot(x, 'o', labels=names, show=False)
    labels = [entry[0] for entry in _entries(fig.axes[0])]
    fig.canvas.draw()
    for dpi in (300, 72):
        fig.savefig(io.BytesIO(), format='png', dpi=dpi)
        assert _places(labels) == reference
    fig.savefig(io.BytesIO(), format='png', bbox_inches='tight')
    assert _places(labels) == reference


def test_a_resized_figure_may_be_arranged_again_and_stays_on_the_canvas():
    x, names = _crowded(ndims=2)
    fig = hyp.plot(x, 'o', labels=names, show=False)
    labels = [entry[0] for entry in _entries(fig.axes[0])]
    fig.canvas.draw()
    before = _overlapping_pairs(fig, labels)
    fig.set_size_inches(12, 9)
    assert _overlapping_pairs(fig, labels) <= before


def _drag(fig, ax, steps=3, dx=0.3):
    """Press, drag and release the left mouse button on `ax`, with real
    mouse events, drawing as a window would."""
    x, y = ax.bbox.x0 + ax.bbox.width / 2, ax.bbox.y0 + ax.bbox.height / 2

    def send(name, at):
        fig.canvas.callbacks.process(name, MouseEvent(
            name, fig.canvas, at, y, button=MouseButton.LEFT))

    send('button_press_event', x)
    during = []
    for step in range(1, steps + 1):
        send('motion_notify_event', x + dx * step)
        fig.canvas.draw()
        during.append([entry[0].get_visible() for entry in _entries(ax)])
    send('button_release_event', x + dx * steps)
    fig.canvas.draw()
    return during


def test_turning_a_plot_with_the_mouse_keeps_the_labels_where_they_were():
    """A press hides the labels; the next draw sent every hidden label
    back to its usual place, so on release the placement started over
    and labels that were clear of each other changed place."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 3))
    x[4:9] = x[4] + rng.normal(scale=.03, size=(5, 3))
    fig = hyp.plot(x, 'o', labels=[f'label {i}' for i in range(12)],
                   show=False)
    ax = fig.axes[0]
    fig.canvas.draw()
    labels = [entry[0] for entry in _entries(ax)]
    before = _places(labels)
    assert sum(place != HOME for place in before) >= 3
    azim = ax.azim
    during = _drag(fig, ax)
    assert ax.azim != azim                        # the plot really turned
    assert not any(any(shown) for shown in during)   # hidden while dragged
    assert all(label.get_visible() for label in labels)
    # a turn of less than a degree: no label's box was covered, so none
    # of them has a reason to move
    assert _places(labels) == before


def test_a_redraw_of_an_unchanged_figure_places_nothing_again():
    x, names = _crowded(n=60)
    fig = hyp.plot(x, 'o', labels=names, show=False)
    ax = fig.axes[0]
    labels = [entry[0] for entry in _entries(ax)]
    # `plot()` leaves the labels at their usual places: they are
    # arranged for the finished figure, on its first draw
    assert all(place == HOME for place in _places(labels))
    fig.canvas.draw()
    answer = ax._hyp_label_answer
    assert answer['settled']
    places = _places(labels)
    assert sum(place != HOME for place in places) >= 3
    for _ in range(2):
        fig.canvas.draw()
        assert ax._hyp_label_answer is answer      # not worked out again
        assert _places(labels) == places
    # the same call again: the same arrangement
    again = hyp.plot(x, 'o', labels=names, show=False)
    again.canvas.draw()
    assert _places([entry[0] for entry in _entries(again.axes[0])]) == places


def test_labels_saved_by_plot_itself_are_arranged(tmp_path):
    """`save_path=` writes the finished figure: its labels are not left
    at their usual places because `plot()` is still running."""
    x, names = _crowded(ndims=2)
    fig = hyp.plot(x, 'o', labels=names, show=False,
                   save_path=str(tmp_path / 'labels.png'))
    labels = [entry[0] for entry in _entries(fig.axes[0])]
    assert sum(place != HOME for place in _places(labels)) >= 3


@pytest.fixture
def count_draws(monkeypatch):
    """Count the draws of whole figures; every draw is really made."""
    drawn = []
    real_draw = Figure.draw

    def counting_draw(self, renderer, *args, **kwargs):
        drawn.append(self)
        return real_draw(self, renderer, *args, **kwargs)

    monkeypatch.setattr(Figure, 'draw', counting_draw)
    return drawn


def test_placing_an_animations_labels_draws_nothing(count_draws):
    """Placing an animation's labels rendered the whole figure (and
    replaced the figure's canvas doing it). The table of review round 5
    (`solve_label_tracks`) is worked out without a draw too."""
    names = [f'a{i}' if i % 5 == 0 else None for i in range(len(X))]
    counts = {}
    for key, labels in (('plain', None), ('labeled', names)):
        del count_draws[:]
        hyp.plot(X, '-', animate=True, labels=labels, duration=1,
                 frame_rate=10, show=False)
        counts[key] = len(count_draws)
    # the one draw `labels=` adds is the one that creates the labels
    assert counts['labeled'] - counts['plain'] <= 1


def test_an_animated_grid_places_its_labels_without_a_draw_per_panel(
        count_draws):
    """A grid of N labeled panels rendered the whole grid N times to
    place the labels."""
    data = [X, Y, Z, X + 1]
    names = [f'a{i}' if i % 5 == 0 else None for i in range(len(X))]
    counts = {}
    for key, labels in (('plain', None), ('labeled', [names] * 4)):
        del count_draws[:]
        anim = hyp.plot(data, '-', animate=True, panels=(2, 2),
                        labels=labels, duration=1, frame_rate=10, show=False)
        counts[key] = len(count_draws)
    # (each panel's labels are created with one draw, as in a static grid)
    assert counts['labeled'] - counts['plain'] <= len(data)
    for ax in anim.figure.axes[:4]:
        assert label_tracks(ax) is not None


def test_an_animations_labels_are_placed_as_a_draw_would_place_them():
    """The placement made without a draw is the one a draw of the same
    state makes (rewritten in review round 5: the table's first row for
    an animation that does not loop, placed frame by frame on its own,
    against a real draw of frame 0 with the labels arranged as a static
    figure's are)."""
    from hypertools.plot.matplotlib_backend import (
        _LABEL_CANDIDATES, _label_place, _send_label_home,
        solve_label_tracks)
    x, names = _crowded()
    anim = hyp.plot(x, 'o', labels=names, animate='spin', duration=1,
                    frame_rate=5, rotations=0.4, show=False)
    ax = anim.figure.axes[0]
    labels = [entry[0] for entry in _entries(ax)]
    tracks = solve_label_tracks(ax, _independent=True)
    tabled = [_LABEL_CANDIDATES[k] for k in tracks['places'][0]]
    assert sum(place != _LABEL_CANDIDATES[tracks['places'][0].min()]
               for place in tabled) >= 3
    anim.draw_frame(0)
    # the same state, placed by a real draw
    forget_label_tracks(ax)
    for label in labels:
        label._hyp_place = None
        _send_label_home(label)
    ax._hyp_label_answer = None
    anim.figure.canvas.draw()
    assert [_label_place(label) for label in labels] == tabled


# --------------------------------------------------------------------------
# C. a measurement leaves the figure its own canvas
# --------------------------------------------------------------------------

class _WindowCanvas(FigureCanvasAgg):
    """A canvas of its own class, standing in for a GUI window's."""


CANVAS_CASES = {
    'legend': dict(legend=['a']),
    'legend outside': dict(legend=['a'], legend_kwargs={
        'loc': 'center left', 'bbox_to_anchor': (1.02, 0.5)}),
    'colorbar': dict(hue=HUE, colorbar=True),
    'labels': dict(labels=[str(i) for i in range(len(X))]),
    'title': dict(title='A title'),
}


@pytest.mark.parametrize('case', CANVAS_CASES)
@pytest.mark.parametrize('ndims', [2, 3])
def test_plot_leaves_a_callers_figure_on_its_own_canvas(case, ndims):
    """A measurement draw attached a new Agg canvas and left it there: in
    a window (macosx, QtAgg, TkAgg) the figure then updated a canvas
    nobody saw."""
    fig = Figure()
    canvas = _WindowCanvas(fig)
    ax = fig.add_subplot(projection='3d' if ndims == 3 else None)
    hyp.plot(X[:, :ndims], '-', ax=ax, show=False, **CANVAS_CASES[case])
    assert fig.canvas is canvas


@pytest.mark.parametrize('kwargs', [
    dict(legend=['a', 'b']), dict(panels=(1, 2)),
    dict(animate=True, duration=1, frame_rate=10,
         labels=[[str(i) for i in range(len(X))]] * 2),
    dict(animate=True, duration=1, frame_rate=10, panels=(1, 2),
         labels=[[str(i) for i in range(len(X))]] * 2)],
    ids=['legend', 'panels', 'animated labels', 'animated labeled panels'])
def test_a_shown_figure_keeps_the_canvas_of_its_manager(kwargs):
    """`show=True` leaves the figure with pyplot: its canvas is still the
    one its figure manager (the window) owns."""
    result = hyp.plot([X, Y], '-', show=True, **kwargs)
    fig = _figure_of(result)
    assert fig.canvas.manager is not None
    assert fig.canvas.manager.canvas is fig.canvas
    assert plt.fignum_exists(fig.number)


def test_a_measurement_keeps_a_scaled_figures_original_dpi():
    """A new canvas overwrites `_original_dpi` with the current dpi,
    which on a high-resolution display is the scaled one."""
    fig = Figure(dpi=100)
    canvas = _WindowCanvas(fig)
    ax = fig.add_subplot()
    canvas._set_device_pixel_ratio(2)
    assert fig.dpi == 200 and fig._original_dpi == 100
    from hypertools.plot.plot import _measurement_renderer
    hyp.plot(X[:, :2], '-', ax=ax, legend=['a'], show=False)
    renderer = _measurement_renderer(fig)
    assert renderer is not canvas.get_renderer()
    assert fig.canvas is canvas
    assert fig.dpi == 200 and fig._original_dpi == 100


def test_the_canvas_of_a_real_window_backend_survives_plot():
    """The same on a GUI backend, where one can be loaded (macOS)."""
    script = textwrap.dedent('''
        import warnings
        import matplotlib
        matplotlib.use('macosx')
        import matplotlib.pyplot as plt
        from matplotlib._pylab_helpers import Gcf
        import numpy as np
        import hypertools as hyp
        warnings.simplefilter('ignore')
        plt.ioff()
        x = np.random.default_rng(0).normal(size=(20, 3)).cumsum(0)
        names = [str(i) for i in range(20)]
        for kwargs in (dict(labels=names), dict(legend=['a']),
                       dict(animate=True, labels=names),
                       dict(animate=True, labels=names, ndims=2),
                       dict(animate=True, caption='A caption')):
            result = hyp.plot(x, '-', show=True, **kwargs)
            fig = getattr(result, 'figure', result)
            (manager,) = Gcf.get_all_fig_managers()
            assert type(fig.canvas).__name__ == 'FigureCanvasMac', kwargs
            assert manager.canvas is fig.canvas, kwargs
            plt.close('all')
        print('CANVAS KEPT')
    ''')
    probe = subprocess.run(
        [sys.executable, '-c',
         "import matplotlib; matplotlib.use('macosx'); "
         "import matplotlib.pyplot as plt; plt.figure()"],
        capture_output=True, text=True, timeout=120)
    if probe.returncode != 0:
        pytest.skip('no macosx backend here: ' + probe.stderr[-200:])
    done = subprocess.run([sys.executable, '-c', script], capture_output=True,
                          text=True, timeout=300, env=dict(os.environ))
    assert done.returncode == 0, done.stderr[-2000:]
    assert 'CANVAS KEPT' in done.stdout


# --------------------------------------------------------------------------
# D. warnings in grids, per-dataset lists with hue, labels with morph
# --------------------------------------------------------------------------

def _warnings_of(*args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = hyp.plot(*args, show=False, **kwargs)
    return result, [str(w.message) for w in caught]


ANIMATED = dict(animate=True, duration=1, frame_rate=8)


@pytest.mark.parametrize('panels', [None, (1, 2)])
def test_tail_linewidth_is_not_called_ignored_when_a_panel_draws_a_line(
        panels):
    extra = {'panels': panels} if panels else {}
    anim, messages = _warnings_of([X, Y], ['-', 'o'], tail_linewidth=4,
                                  linewidth=1, **ANIMATED, **extra)
    assert not [m for m in messages if 'tail_linewidth' in m]
    anim.draw_frame(4)
    widths = [line.get_linewidth() for ax in anim.figure.axes
              for line in ax.lines
              if line.get_linestyle() == '-' and len(line.get_xdata())]
    assert 4.0 in widths


@pytest.mark.parametrize('panels', [None, (1, 2)])
@pytest.mark.parametrize('panel_fit', [None, 'independent'])
def test_trail_alpha_is_not_called_ignored_when_a_panel_draws_a_trail(
        panels, panel_fit):
    if panels is None and panel_fit is not None:
        pytest.skip('panel_fit= belongs to panels=')
    extra = {'panels': panels} if panels else {}
    if panel_fit:
        extra['panel_fit'] = panel_fit
    _anim, messages = _warnings_of(
        [X, Y], '-', trail_alpha=0.6, chemtrails=[True, False],
        **ANIMATED, **extra)
    assert not [m for m in messages if 'trail_alpha' in m]


@pytest.mark.parametrize('kwargs, fmt, text', [
    (dict(tail_linewidth=4), ['o', 'o'], 'tail_linewidth= sets the width'),
    (dict(trail_alpha=0.6), '-', 'trail_alpha= sets the opacity'),
    (dict(head_marker='o', animate='spin'), '-', 'no reveal head'),
    (dict(head_marker='o', animate=False), '-', 'only affects ANIMATED')])
@pytest.mark.parametrize('panels', [None, (1, 2)])
def test_a_true_warning_is_given_once_for_the_whole_figure(
        panels, kwargs, fmt, text):
    extra = {'panels': panels} if panels else {}
    call = dict(ANIMATED, **kwargs)
    _result, messages = _warnings_of([X, Y], fmt, **call, **extra)
    assert len([m for m in messages if text in m]) == 1, messages


CATEGORIES = [['a'] * 20 + ['b'] * 20, ['a'] * 40, ['c'] * 10 + ['a'] * 30]


def _trail_datasets(anim):
    """Which drawn traces have a trail artist."""
    return sorted({line._hyp_trace_index for line in anim.figure.axes[0].lines
                   if getattr(line, '_hyp_trace_role', None) == 'trail'})


@pytest.mark.parametrize('fmt', ['-', '-o'])
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('flag', ['chemtrails', 'precog', 'bullettime'])
def test_a_per_dataset_trail_list_is_matched_to_the_callers_datasets(
        fmt, ndims, flag):
    """A categorical hue splits three datasets into five runs, and the
    three-entry list was refused: "3 entries but there are 5 datasets"."""
    anim = hyp.plot([X, Y, Z], fmt, hue=CATEGORIES, ndims=ndims,
                    show=False, **ANIMATED, **{flag: [False, True, True]})
    anim.draw_frame(5)
    # the runs: dataset 0 gives runs 0 and 1, dataset 1 run 2, dataset 2
    # runs 3 and 4; the trails are the runs of datasets 1 and 2
    assert _trail_datasets(anim) == [2, 3, 4]


def test_a_per_dataset_trail_list_with_cluster_and_on_plotly():
    anim = hyp.plot([X, Y, Z], '-', cluster='KMeans', n_clusters=2,
                    random_state=0, chemtrails=[False, False, True],
                    show=False, **ANIMATED)
    owners = set(_trail_datasets(anim))
    assert owners and len(owners) < len(anim.figure.axes[0].lines)
    pytest.importorskip('plotly')
    fig = hyp.plot([X, Y, Z], '-', hue=CATEGORIES, backend='plotly',
                   chemtrails=[False, True, True], show=False, **ANIMATED)
    assert len(fig.frames) > 1


def test_a_trail_list_of_the_wrong_length_counts_the_callers_datasets():
    with pytest.raises(ValueError, match=(
            'chemtrails list has 2 entries but there are 3 datasets')):
        hyp.plot([X, Y, Z], '-', hue=CATEGORIES, chemtrails=[True, False],
                 show=False, **ANIMATED)
    with pytest.raises(ValueError, match='one trace per category'):
        hyp.plot([X, Y, Z], 'o', hue=CATEGORIES, chemtrails=[True, False],
                 show=False, **ANIMATED)


def test_a_per_dataset_surface_list_is_matched_to_the_callers_datasets():
    """The same list-against-runs mismatch, for `surface=`."""
    fig = hyp.plot([X, Y, Z], '-', hue=CATEGORIES,
                   surface=[False, True, True], show=False)
    assert fig.axes


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('backend', ['matplotlib', 'plotly'])
def test_labels_with_morph_say_that_they_are_not_drawn(ndims, backend):
    """matplotlib hid every label without a word; plotly drew them in 2-D
    at the first positions while the cloud moved away."""
    if backend == 'plotly':
        pytest.importorskip('plotly')
    clouds = [np.random.default_rng(seed).normal(size=(12, 3))
              for seed in (0, 1)]
    names = [[f'{which}{i}' for i in range(12)] for which in 'ab']
    result, messages = _warnings_of(
        clouds, 'o', animate='morph', labels=names, ndims=ndims,
        duration=2, frame_rate=8, backend=backend)
    assert len([m for m in messages
                if "animate='morph'" in m and 'ignoring labels=' in m]) == 1
    if backend == 'plotly':
        assert not result.layout.annotations
        assert not getattr(result.layout.scene, 'annotations', None)
    else:
        for frame in (0, result.n_frames // 2, result.n_frames - 1):
            result.draw_frame(frame)
            assert not [text for text in result.figure.axes[0].texts
                        if isinstance(text, mtext.Annotation)
                        and text.get_visible()]
    # an all-None list asks for nothing, so there is nothing to say
    _result, messages = _warnings_of(
        clouds, 'o', animate='morph', labels=[[None] * 12] * 2, ndims=ndims,
        duration=2, frame_rate=8, backend=backend)
    assert not [m for m in messages if 'ignoring labels=' in m]


def test_a_seeded_clustering_is_the_same_picture_every_time():
    """`cluster='KMeans'` groups differently on each call unless
    `random_state=` is given, as documented for every random stage."""
    both = np.vstack([X, Y + 5])

    def pixels(**kwargs):
        fig = hyp.plot(both, 'o', cluster='KMeans', n_clusters=3,
                       show=False, **kwargs)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        plt.close(fig)
        return buffer.getvalue()

    assert pixels(random_state=0) == pixels(random_state=0)
    assert 'random_state' in hyp.plot.__doc__.split('cluster :')[1].split(
        'n_clusters :')[0]


# --------------------------------------------------------------------------
# E. hue markers: the dataset's marker styling, and one fate per trace
# --------------------------------------------------------------------------

HOLLOW = dict(markerfacecolor='none', markeredgecolor='k',
              markeredgewidth=3, markersize=12)


def _scatters(ax):
    return [coll for coll in ax.collections
            if isinstance(coll, PathCollection)]


def _assert_hollow_black(scatter):
    assert len(scatter.get_facecolor()) == 0              # no fill
    edge = np.atleast_2d(scatter.get_edgecolor())
    assert np.allclose(edge[:, :3], 0.0)                  # black edges
    assert np.allclose(scatter.get_linewidths(), 3)
    assert np.allclose(scatter.get_sizes(), 144.0)


@pytest.mark.parametrize('fmt', ['-o', 'o'])
@pytest.mark.parametrize('ndims', [2, 3])
def test_hue_markers_wear_the_marker_styling_static_and_animated(
        fmt, ndims):
    """Hollow markers with a black outline were drawn as filled dots with
    default edges, by the static plot and by the animation."""
    x = X[:, :ndims]
    fig = hyp.plot(x, fmt, hue=HUE, show=False, **HOLLOW)
    (static,) = _scatters(fig.axes[0])
    _assert_hollow_black(static)
    anim = hyp.plot(x, fmt, hue=HUE, chemtrails=True, tail_duration=0.3,
                    show=False, **ANIMATED, **HOLLOW)
    anim.draw_frame(anim.n_frames - 2)
    anim.figure.canvas.draw()
    shown = [scatter for scatter in _scatters(anim.figure.axes[0])
             if scatter.get_visible()]
    assert len(shown) == 2                                # tail and trail
    for scatter in shown:
        _assert_hollow_black(scatter)


@pytest.mark.parametrize('animated', [False, True])
def test_hue_colors_the_edge_of_a_marker_with_its_own_fill(animated):
    """As on a `Line2D`: `markerfacecolor=` names the fill, and the edge
    keeps the trace's color, which with `hue=` is each observation's."""
    kwargs = dict(ANIMATED) if animated else {}
    result = hyp.plot(X, 'o', hue=HUE, mfc='w', mew=2, ms=10, show=False,
                      **kwargs)
    if animated:
        result.draw_frame(result.n_frames - 1)
    ax = _figure_of(result).axes[0]
    scatter = [s for s in _scatters(ax) if s.get_visible()][0]
    assert np.allclose(np.atleast_2d(scatter.get_facecolor())[:, :3], 1.0)
    edges = np.atleast_2d(scatter.get_edgecolor())
    assert len(np.unique(edges[:, :3].round(3), axis=0)) >= 3
    assert np.allclose(scatter.get_linewidths(), 2)
    # (`ms=` is matplotlib's short name for `markersize=`: the static
    # scatter read the long name only)
    assert np.allclose(scatter.get_sizes(), 100.0)


def test_plain_hue_markers_are_drawn_as_before():
    fig = hyp.plot(X, 'o', hue=HUE, show=False)
    (scatter,) = _scatters(fig.axes[0])
    faces, edges = scatter.get_facecolor(), scatter.get_edgecolor()
    assert len(faces) > 30 and np.allclose(faces, edges)
    assert len(np.unique(faces[:, :3].round(3), axis=0)) > 30


def _trace_and_markers(ax, role='head'):
    line = [coll for coll in ax.collections
            if getattr(coll, '_hyp_trace_role', None) == role][0]
    return line, line._hyp_markers


@pytest.mark.parametrize('ndims', [2, 3])
def test_hiding_a_trace_in_a_callback_hides_its_markers_and_head_marker(
        ndims):
    """`ctx.artists[0]` is the colored line; its markers are a second
    artist the callback never saw, and they stayed on screen."""
    anim = hyp.plot(
        X[:, :ndims], '-o', hue=HUE, head_marker='D', show=False,
        on_frame=lambda ctx: ctx.artists[0].set_visible(ctx.frame % 2 == 0),
        **ANIMATED)
    ax = anim.figure.axes[0]
    line, dots = _trace_and_markers(ax)
    (head_marker,) = _head_marker(anim)
    for frame in (4, 5, 6):
        anim.draw_frame(frame)
        anim.figure.canvas.draw()
        shown = frame % 2 == 0
        assert line.get_visible() is shown
        assert dots.get_visible() is shown
        assert head_marker.get_visible() is shown


def test_fading_a_trace_in_a_callback_fades_its_markers():
    anim = hyp.plot(
        X, '-o', hue=HUE, show=False,
        on_frame=lambda ctx: ctx.artists[0].set_alpha(0.25), **ANIMATED)
    line, dots = _trace_and_markers(anim.figure.axes[0])
    anim.draw_frame(5)
    anim.figure.canvas.draw()
    assert line.get_alpha() == 0.25 and dots.get_alpha() == 0.25


def test_a_head_marker_is_hidden_with_its_line_without_hue():
    anim = hyp.plot(
        X, '-', head_marker='o', show=False,
        on_frame=lambda ctx: ctx.artists[0].set_visible(False), **ANIMATED)
    (head_marker,) = _head_marker(anim)
    anim.draw_frame(5)
    anim.figure.canvas.draw()
    assert not head_marker.get_visible()


def test_markers_hidden_by_the_caller_stay_hidden():
    """Following goes one way: a follower hidden for another reason is
    not shown again because its line is visible."""
    anim = hyp.plot(X, '-o', hue=HUE, show=False, **ANIMATED)
    line, dots = _trace_and_markers(anim.figure.axes[0])
    anim.draw_frame(5)
    dots.set_visible(False)
    anim.figure.canvas.draw()
    assert line.get_visible() and not dots.get_visible()


def test_an_animated_figure_still_pickles():
    """The followers are kept by an artist of a module-level class, not
    by closures on the artists."""
    import pickle
    # (only bytes this test has just written itself are loaded back)
    anim = hyp.plot(X, '-o', hue=HUE, head_marker='o', show=False,
                    labels=[str(i) if i % 9 == 0 else None
                            for i in range(len(X))], **ANIMATED)
    anim.draw_frame(3)
    anim.figure.canvas.draw()
    assert pickle.loads(pickle.dumps(anim.figure)).axes


# --------------------------------------------------------------------------
# F. a per-frame caption or title is never clipped
# --------------------------------------------------------------------------

def _tall_on_frame_five(ctx):
    return ('tall\n' * 6 + 'end') if ctx.frame == 5 else 'short'


def _clipped(anim):
    """The (frame, text) of every caption, suptitle or title that leaves
    the canvas on some frame."""
    fig, out = anim.figure, []
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in fig.texts if t.get_text().strip()] + [
            ax.title for ax in fig.axes if ax.title.get_text().strip()]
        for text in texts:
            box = text.get_window_extent(renderer)
            if (box.y0 < -0.5 or box.y1 > fig.bbox.height + 0.5
                    or box.x0 < -0.5 or box.x1 > fig.bbox.width + 0.5):
                out.append((frame, text.get_text()[:12]))
    return out


FRAME_TEXT_CASES = {
    'caption': dict(caption=_tall_on_frame_five),
    'caption 2-D': dict(caption=_tall_on_frame_five, ndims=2),
    'caption at the top': dict(caption=_tall_on_frame_five,
                               caption_kwargs={'position': 'top'}),
    'caption, large': dict(caption=_tall_on_frame_five,
                           caption_kwargs={'size': 18}),
    'caption, serial': dict(caption=_tall_on_frame_five, animate='serial'),
    'caption, held': dict(caption=_tall_on_frame_five, hold=0.5),
    'caption in a grid': dict(caption=_tall_on_frame_five, panels=(1, 2)),
    'title': dict(title=_tall_on_frame_five),
    'title 2-D': dict(title=_tall_on_frame_five, ndims=2),
    'title under a suptitle': dict(title=_tall_on_frame_five,
                                   suptitle='A heading'),
    'titles in a grid': dict(title=_tall_on_frame_five, panels=(1, 2)),
}


@pytest.mark.parametrize('case', FRAME_TEXT_CASES)
def test_a_text_that_is_tallest_on_any_frame_stays_on_the_canvas(case):
    """The room was sized from the first, middle and last frames: a text
    that was tallest on another frame ran off the canvas."""
    kwargs = dict(animate=True, duration=2, frame_rate=8)
    kwargs.update(FRAME_TEXT_CASES[case])
    anim = hyp.plot([X, Y], '-', show=False, **kwargs)
    assert anim.n_frames > 8                # frame 5 is not a sampled one
    assert _clipped(anim) == []


def test_a_tall_title_frame_clears_the_suptitle_and_a_top_caption():
    anim = hyp.plot([X, Y], '-', animate=True, duration=2, frame_rate=8,
                    title=_tall_on_frame_five, suptitle='A heading',
                    caption='A caption', caption_kwargs={'position': 'top'},
                    show=False)
    fig = anim.figure
    anim.draw_frame(5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title = fig.axes[0].title.get_window_extent(renderer)
    for gid in (SUPTITLE_GID, CAPTION_GID):
        text = fig.findobj(lambda a: a.get_gid() == gid)[0]
        assert text.get_window_extent(renderer).y0 >= title.y1 - 0.5


def test_a_per_frame_caption_is_asked_about_every_frame_once_to_size_it():
    """What the room costs: one call of the caption per frame before the
    animation is handed back (and nothing is rendered for it)."""
    asked = []

    def caption(ctx):
        asked.append(ctx.frame)
        return f'frame {ctx.frame}'

    anim = hyp.plot(X, '-', animate=True, duration=2, frame_rate=8,
                    caption=caption, show=False)
    assert sorted(set(asked)) == list(range(anim.n_frames))
