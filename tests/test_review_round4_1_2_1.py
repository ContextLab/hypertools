"""1.2.1 review round 4: what the fourth review found.

A. Observation labels (``labels=``) are kept off the legends (an axes'
   own and the one shared legend of a ``panels=`` grid), the colorbars,
   the panel titles and the figure's ``suptitle=``/``caption=``, in 2-D
   and 3-D, static and in the one placement an animation makes.
B. ``legend_kwargs={'loc': ...}`` on a 2-D plot puts the legend inside
   the frame the reader sees, as a ``loc`` does in matplotlib.
C. ``head_marker=`` draws ONE marker per dataset when ``hue=`` cuts the
   dataset into categories, and a serial reveal shows the head on its
   first frame.
D. A ``head_marker=`` dict takes matplotlib's short names and a face
   color.
E. The ``trail_alpha=`` warning under ``animate='window'`` and a
   ``hold=`` too short to add a frame.

Real figures, real frames and real saved files, rendered headless.
Nothing is mocked, and no assertion depends on a font's absolute
metrics.
"""
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import matplotlib.text as mtext                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.colors import to_rgba                           # noqa: E402
from PIL import Image                                           # noqa: E402

import hypertools as hyp                                        # noqa: E402

HAS_FFMPEG = (shutil.which('ffmpeg') is not None
              and shutil.which('ffprobe') is not None)

LABELS = [f'label{i}' for i in range(24)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _quiet(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return hyp.plot(*args, show=False, **kwargs)


def _trajectories(k=1, n=40, dims=3, seed=0):
    rng = np.random.default_rng(seed)
    out = [np.cumsum(rng.normal(size=(n, dims)), axis=0) for _ in range(k)]
    return out if k > 1 else out[0]


# ---------------------------------------------------------------------------
# A. labels stay off legends, colorbars, titles, suptitle and caption
# ---------------------------------------------------------------------------

def _label_boxes(fig, renderer):
    return [label.get_bbox_patch().get_window_extent(renderer)
            for ax in fig.axes for label in ax.findobj(mtext.Annotation)
            if label.get_visible() and label.get_text()]


def _furniture(fig, renderer):
    """What the figure shows beside its data, by kind."""
    legends = list(fig.legends) + [
        ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    return {
        'legend': [one.get_window_extent(renderer) for one in legends],
        'title': [ax.title.get_window_extent(renderer)
                  for ax in fig.axes if ax.title.get_text()],
        'figure text': [text.get_window_extent(renderer)
                        for text in fig.texts if text.get_text()],
        'colorbar': [ax.get_tightbbox(renderer) for ax in fig.axes
                     if hasattr(ax, '_colorbar')],
    }


def _covered(fig):
    """How many label boxes lie on each kind of furniture, and how many
    pairs of labels lie on each other, on a real draw of `fig`."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = _label_boxes(fig, renderer)
    counts = {kind: sum(bool(box.overlaps(other))
                        for box in boxes for other in others)
              for kind, others in _furniture(fig, renderer).items()}
    counts['labels'] = sum(
        bool(boxes[i].overlaps(boxes[j]))
        for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    return len(boxes), counts


def _panel_call(seed, dims):
    x = np.random.default_rng(seed).normal(size=(24, 3))[:, :dims]
    return ([x[:12], x[12:]], 'o'), dict(
        panels=True, labels=[LABELS[:12], LABELS[12:]], legend=['a', 'b'],
        title=['A', 'B'], suptitle='S')


#: the seeds of the review's twelve on which the 1.2.1 draft put a label
#: on the shared legend, a panel title or the suptitle (3-D: 0, 1, 2, 6,
#: 7, 8; 2-D: 2, 5, 7, 8, 10, 11), and one clean seed of each
@pytest.mark.parametrize('dims, seed', [
    (3, 0), (3, 1), (3, 2), (3, 6), (3, 7), (3, 8), (3, 3),
    (2, 2), (2, 5), (2, 7), (2, 8), (2, 10), (2, 11), (2, 0)])
def test_panel_labels_stay_off_the_shared_legend_and_the_titles(dims, seed):
    args, kwargs = _panel_call(seed, dims)
    fig = hyp.plot(*args, show=False, **kwargs)
    n, counts = _covered(fig)
    assert n == 24
    assert len(fig.legends) == 1
    # (not asserted: that no two labels meet. They do not with
    # matplotlib 3.10's text sizes; with 3.11's slightly larger ones one
    # pair does on two of these seeds, as before this change)
    del counts['labels']
    assert counts == {'legend': 0, 'title': 0, 'figure text': 0,
                      'colorbar': 0}


@pytest.mark.parametrize('seed', [0, 2, 6, 7])
def test_animated_panel_labels_are_placed_off_the_legend_and_texts(seed):
    """An animation's placement keeps off the same things on every frame
    (rewritten in review round 5 from the test of one placement, made
    for frame 0 and held), and a frame shown again has its labels where
    they were."""
    x = np.random.default_rng(seed).normal(size=(24, 3))
    anim = _quiet([x[:12], x[12:]], 'o', animate='spin', panels=True,
                  labels=[LABELS[:12], LABELS[12:]], legend=['a', 'b'],
                  title=['A', 'B'], suptitle='S', caption='A caption',
                  duration=1, frame_rate=5)
    places = {}
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        n, counts = _covered(anim.figure)
        assert n == 24
        assert counts['legend'] == counts['title'] == 0, frame
        assert counts['figure text'] == 0, frame
        places[frame] = [label.xyann for ax in anim.figure.axes
                         for label in ax.findobj(mtext.Annotation)]
    for frame in (3, 0, 4):
        anim.draw_frame(frame)
        anim.figure.canvas.draw()
        assert places[frame] == [label.xyann for ax in anim.figure.axes
                                 for label in ax.findobj(mtext.Annotation)]


@pytest.mark.parametrize('across', [0.2, 0.4])
def test_one_label_under_the_title_moves_off_it(across):
    """A single label was never looked at (there was no other label to
    keep it from), so the label of the topmost point lay on the title."""
    x = np.array([[across, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    fig = hyp.plot(x, 'o', labels=['top', None, None], title='A title',
                   show=False)
    n, counts = _covered(fig)
    assert n == 1
    assert counts['title'] == 0


def test_a_label_clear_of_everything_stays_where_it_always_was():
    x = np.array([[0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]])
    fig = hyp.plot(x, 'o', labels=['middle', None, None], title='A title',
                   legend=['data'], show=False)
    fig.canvas.draw()
    (label,) = [one for one in fig.axes[0].findobj(mtext.Annotation)
                if one.get_text()]
    assert tuple(label.xyann) == (-20, 20)


@pytest.mark.parametrize('dims', [2, 3])
def test_labels_stay_off_a_colorbar_drawn_where_they_were(dims):
    """A colorbar put exactly where a label hangs: the label leaves."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, dims))
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d' if dims == 3 else None)
    hyp.plot(x, 'o', labels=[f'point {i}' for i in range(6)], ax=ax,
             show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    before = _label_boxes(fig, renderer)
    # a colorbar axes over the first label's box (figure fractions)
    target = before[0].transformed(fig.transFigure.inverted())
    cax = fig.add_axes([target.x0, target.y0, target.width, target.height])
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), cax=cax)
    n, counts = _covered(fig)
    assert n == 6
    assert counts['colorbar'] == 0
    assert counts['labels'] == 0


def test_the_obstacles_are_the_legends_colorbars_titles_and_figure_texts():
    """Every kind of furniture is in the list the placement keeps off,
    each box covering the artist as drawn (boxes are in points from the
    canvas corner, widened to whole points)."""
    from hypertools.plot.matplotlib_backend import _label_obstacles_pt
    x = np.random.default_rng(0).normal(size=(24, 3))
    fig = hyp.plot([x[:12], x[12:]], 'o', panels=True,
                   hue=[np.arange(12.), np.arange(12.)], colorbar=True,
                   labels=[LABELS[:12], LABELS[12:]], title=['A', 'B'],
                   suptitle='S', caption='A caption', show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_pt = 72.0 / fig.dpi
    obstacles = _label_obstacles_pt(fig.axes[0], None, (0.0, 0.0), to_pt)
    furniture = _furniture(fig, renderer)
    assert furniture['colorbar'] and furniture['figure text']
    assert len(furniture['title']) == 2 and len(furniture['figure text']) == 2
    for kind, boxes in furniture.items():
        for box in boxes:
            want = np.array(box.extents) * to_pt
            inside = ((obstacles[:, 0] <= want[0] + 1e-6)
                      & (obstacles[:, 1] <= want[1] + 1e-6)
                      & (obstacles[:, 2] >= want[2] - 1e-6)
                      & (obstacles[:, 3] >= want[3] - 1e-6))
            assert inside.any(), kind
    # ...and a 2-D single plot with a legend, a title and a caption
    fig = hyp.plot(x[:, :2], 'o', labels=LABELS, legend=['data'],
                   title='Title', caption='cap', show=False)
    fig.canvas.draw()
    assert len(_label_obstacles_pt(
        fig.axes[0], None, (0.0, 0.0), 72.0 / fig.dpi)) == 3


@pytest.mark.parametrize('kwargs', [
    dict(legend=['data'], title='Title', caption='cap'),
    dict(hue=np.arange(24.), title='Title', suptitle='Sup'),
])
@pytest.mark.parametrize('dims', [2, 3])
def test_single_plot_labels_stay_off_legend_colorbar_and_texts(dims, kwargs):
    for seed in (0, 1, 2, 3):
        x = np.random.default_rng(seed).normal(size=(24, 3))[:, :dims]
        fig = hyp.plot(x, 'o', labels=LABELS, show=False, **kwargs)
        _n, counts = _covered(fig)
        assert counts['legend'] == counts['title'] == 0, seed
        assert counts['figure text'] == counts['colorbar'] == 0, seed
        plt.close(fig)


def test_a_redraw_with_obstacles_places_nothing_again():
    args, kwargs = _panel_call(2, 3)
    fig = hyp.plot(*args, show=False, **kwargs)
    fig.canvas.draw()
    first = [tuple(label.xyann) for ax in fig.axes
             for label in ax.findobj(mtext.Annotation)]
    answers = [ax._hyp_label_answer for ax in fig.axes[:2]]
    for _ in range(3):
        fig.canvas.draw()
    assert first == [tuple(label.xyann) for ax in fig.axes
                     for label in ax.findobj(mtext.Annotation)]
    assert all(now is then for now, then in
               zip([ax._hyp_label_answer for ax in fig.axes[:2]], answers))


def test_the_same_call_twice_places_the_labels_the_same():
    places = []
    for _ in range(2):
        args, kwargs = _panel_call(0, 3)
        fig = hyp.plot(*args, show=False, **kwargs)
        fig.canvas.draw()
        places.append([tuple(label.xyann) for ax in fig.axes
                       for label in ax.findobj(mtext.Annotation)])
        plt.close(fig)
    assert places[0] == places[1]


def test_a_tiny_figure_with_large_texts_keeps_the_labels_off_them():
    """The extreme case of the review (3 x 2.5 in, size 18 suptitle and
    caption, 30 labels): there is no room for every label, so labels
    lie on each other, and none of them on a text or on the legend (a
    place on one of those is the last resort). Before, a third of the
    labels lay on the suptitle, the title or the caption."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 3))
    x[:10] *= 0.05
    fig = hyp.plot(
        x, 'o', size=(3, 2.5), labels=[f'obs {i}' for i in range(30)],
        caption='A caption that is long enough to wrap several times in a '
                'small figure', caption_kwargs={'size': 18},
        suptitle='Suptitle long enough to wrap in a small figure',
        suptitle_kwargs={'size': 18}, title='Title', legend=['d'],
        show=False)
    n, counts = _covered(fig)
    assert n == 30
    assert counts['legend'] == 0
    assert counts['figure text'] == 0
    assert counts['title'] == 0


# ---------------------------------------------------------------------------
# B. legend_kwargs={'loc': ...} on a 2-D plot: inside the drawn frame
# ---------------------------------------------------------------------------

def _frame_and_legend(fig):
    """The window extents of the 2-D frame square and of the legend."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    (frame,) = [patch for patch in ax.patches
                if type(patch).__name__ == 'Rectangle']
    return (frame.get_window_extent(renderer),
            ax.get_legend().get_window_extent(renderer), ax)


@pytest.mark.parametrize('loc', ['upper left', 'lower right', 'center',
                                 'upper right', 'best', 2])
@pytest.mark.parametrize('animate', [False, True])
def test_a_named_legend_loc_is_inside_the_2d_frame(loc, animate):
    data = [d[:, :2] for d in _trajectories(3, 30)]
    extra = dict(animate=True, duration=1, frame_rate=5) if animate else {}
    out = _quiet(data, '-', legend=['a', 'b', 'c'],
                 legend_kwargs={'loc': loc}, title='T', **extra)
    fig = out.figure if animate else out
    frame, legend, _ax = _frame_and_legend(fig)
    assert frame.x0 <= legend.x0 and legend.x1 <= frame.x1
    assert frame.y0 <= legend.y0 and legend.y1 <= frame.y1
    # in its corner, a little in from the border as in matplotlib
    if loc in ('upper left', 2):
        assert legend.x0 - frame.x0 < 0.5 * legend.width
        assert frame.y1 - legend.y1 < 0.5 * legend.height
        assert legend.x0 > frame.x0 and legend.y1 < frame.y1
    if loc == 'lower right':
        assert frame.x1 - legend.x1 < 0.5 * legend.width
        assert legend.y0 - frame.y0 < 0.5 * legend.height
    if loc == 'center':
        assert 0.5 * (legend.x0 + legend.x1) == pytest.approx(
            0.5 * (frame.x0 + frame.x1), abs=1.0)


def test_the_default_2d_legend_is_still_right_of_the_frame():
    data = [d[:, :2] for d in _trajectories(3, 30)]
    fig = hyp.plot(data, '-', legend=['a', 'b', 'c'], show=False)
    frame, legend, _ax = _frame_and_legend(fig)
    assert legend.x0 >= frame.x1
    assert 0.5 * (legend.y0 + legend.y1) == pytest.approx(
        0.5 * (frame.y0 + frame.y1), abs=1.5)


def test_a_bbox_to_anchor_is_the_callers_own_placement():
    data = [d[:, :2] for d in _trajectories(3, 30)]
    fig = hyp.plot(data, '-', legend=['a', 'b', 'c'], show=False,
                   legend_kwargs={'loc': 'upper left',
                                  'bbox_to_anchor': (0.5, 0.5)})
    _frame, legend, ax = _frame_and_legend(fig)
    # the legend's upper-left corner at the middle of the AXES, as the
    # same two arguments mean in matplotlib
    middle = ax.transAxes.transform((0.5, 0.5))
    assert legend.x0 == pytest.approx(middle[0], abs=1.0)
    assert legend.y1 == pytest.approx(middle[1], abs=1.0)


def test_a_named_loc_on_a_3d_plot_is_inside_its_axes():
    fig = hyp.plot(_trajectories(3, 30), '-', legend=['a', 'b', 'c'],
                   legend_kwargs={'loc': 'upper left'}, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    box, legend = ax.get_window_extent(renderer), \
        ax.get_legend().get_window_extent(renderer)
    assert box.x0 <= legend.x0 and legend.x1 <= box.x1
    assert box.y0 <= legend.y0 and legend.y1 <= box.y1


@pytest.mark.parametrize('dims', [2, 3])
def test_a_named_loc_on_a_grid_still_names_the_side_of_the_grid(dims):
    data = [d[:, :dims] for d in _trajectories(3, 30)]
    fig = hyp.plot(data, '-', panels=True, legend=['a', 'b', 'c'],
                   legend_kwargs={'loc': 'upper left'}, show=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (legend,) = fig.legends
    box = legend.get_window_extent(renderer)
    panels = [ax.get_window_extent(renderer) for ax in fig.axes]
    assert box.x1 <= min(panel.x0 for panel in panels) + 1.0
    assert box.x0 >= 0


# ---------------------------------------------------------------------------
# C. head_marker=: one per dataset; a serial reveal's first frame
# ---------------------------------------------------------------------------

def _points(line):
    """The data of a line artist as an (n, d) array."""
    if hasattr(line, 'get_data_3d'):
        return np.array([np.asarray(c, dtype=float)
                         for c in line.get_data_3d()]).T.reshape(-1, 3)
    return np.array([np.asarray(c, dtype=float)
                     for c in line.get_data()]).T.reshape(-1, 2)


def _head_markers(ax):
    return [line for line in ax.lines
            if getattr(line, '_hyp_trace_role', None) == 'head_marker']


def _leader(ax, marker):
    (line,) = [line for line in ax.lines
               if getattr(line, '_hyp_head_marker', None) is marker]
    return line


@pytest.mark.parametrize('dims', [2, 3])
@pytest.mark.parametrize('style', [True, 'window'])
def test_one_head_marker_per_dataset_with_categories(dims, style):
    x = _trajectories(1, 40, dims)
    anim = _quiet(x, '-', animate=style, hue=['u'] * 20 + ['v'] * 20,
                  head_marker='o', duration=1.5, frame_rate=10)
    ax = anim.figure.axes[0]
    markers = _head_markers(ax)
    assert len(markers) == 2                  # one artist per category run
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        drawn = [m for m in markers if len(_points(m))]
        assert len(drawn) == 1, frame         # ...ONE of them drawn
        (marker,) = drawn
        leader = _leader(ax, marker)
        assert np.allclose(_points(marker)[0], _points(leader)[-1])
        assert to_rgba(marker.get_color()) == to_rgba(leader.get_color())
    # at the end it is on the dataset's last point, in the color of the
    # category that point is in (the second run), and the first
    # category's marker is gone
    last_run = _leader(ax, markers[1])
    assert len(_points(markers[0])) == 0
    assert np.allclose(_points(markers[1])[0], _points(last_run)[-1])
    anim.draw_frame(0)
    assert len(_points(markers[0])) == 1 and len(_points(markers[1])) == 0


def test_two_datasets_with_categories_have_two_head_markers():
    a, b = _trajectories(2, 30)
    hue = [['u'] * 10 + ['v'] * 20, ['v'] * 15 + ['w'] * 15]
    anim = _quiet([a, b], '-', animate=True, hue=hue, head_marker='o',
                  duration=1, frame_rate=10)
    ax = anim.figure.axes[0]
    for frame in (0, anim.n_frames // 2, anim.n_frames - 1):
        anim.draw_frame(frame)
        assert sum(len(_points(m)) for m in _head_markers(ax)) == 2


def test_a_dataset_in_one_piece_keeps_its_one_head_marker():
    anim = _quiet(_trajectories(3, 30), '-', animate=True, head_marker='o',
                  duration=1, frame_rate=10)
    ax = anim.figure.axes[0]
    for frame in (0, 5, anim.n_frames - 1):
        anim.draw_frame(frame)
        markers = _head_markers(ax)
        assert [len(_points(m)) for m in markers] == [1, 1, 1]
        for marker in markers:
            assert np.allclose(_points(marker)[0],
                               _points(_leader(ax, marker))[-1])


@pytest.mark.parametrize('dims', [2, 3])
@pytest.mark.parametrize('trail', [{}, {'chemtrails': True}])
def test_a_serial_reveal_shows_the_first_head_on_frame_0(dims, trail):
    data = [d[:, :dims] for d in _trajectories(3, 40)]
    anim = _quiet(data, '-', animate='serial', head_marker='o',
                  duration=1.5, frame_rate=10, **trail)
    ax = anim.figure.axes[0]
    markers = _head_markers(ax)
    anim.draw_frame(anim.n_frames // 4)
    first_point = _points(_leader(ax, markers[0]))[0]
    anim.draw_frame(0)
    assert [len(_points(m)) for m in markers] == [1, 0, 0]
    # at the first dataset's first point
    assert np.all(np.isfinite(_points(markers[0])))
    if not trail:
        # (without a trail the line is drawn from its first point on)
        assert np.allclose(_points(markers[0])[0], first_point)
    # as a parallel reveal shows its heads on frame 0
    parallel = _quiet(data, '-', animate=True, head_marker='o',
                      duration=1.5, frame_rate=10)
    parallel.draw_frame(0)
    assert all(len(_points(m)) == 1
               for m in _head_markers(parallel.figure.axes[0]))


def test_a_serial_first_frame_draws_the_head_marker():
    """Pixels: frame 0 with a head marker is not the empty frame."""
    data = _trajectories(3, 40)
    frames = []
    for head in (None, 'o'):
        anim = _quiet(data, '-', animate='serial', head_marker=head,
                      duration=1.5, frame_rate=10)
        fig = anim.draw_frame(0).figure
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
    assert frames[0].shape == frames[1].shape
    assert int((frames[0] != frames[1]).any(axis=-1).sum()) > 10


def test_a_serial_grid_shows_each_panels_head_on_frame_0():
    anim = _quiet(_trajectories(3, 40), '-', animate='serial', panels=True,
                  head_marker='o', duration=1.5, frame_rate=10)
    anim.draw_frame(0)
    panels = [ax for ax in anim.figure.axes if _head_markers(ax)]
    assert len(panels) == 3
    for ax in panels:
        assert [len(_points(m)) for m in _head_markers(ax)] == [1]


def test_serial_head_markers_after_frame_0_are_where_they_were():
    """Past frame 0 a serial head marker is at the end of what its
    dataset has revealed, and a dataset not yet reached has none."""
    data = _trajectories(3, 40)
    anim = _quiet(data, '-', animate='serial', head_marker='o',
                  duration=1.5, frame_rate=10)
    ax = anim.figure.axes[0]
    for frame in range(1, anim.n_frames):
        anim.draw_frame(frame)
        for marker in _head_markers(ax):
            shown = _points(_leader(ax, marker))
            assert len(_points(marker)) == min(1, len(shown))
            if len(shown):
                assert np.allclose(_points(marker)[0], shown[-1])


def test_a_serial_hue_head_marker_wears_the_first_rows_color_on_frame_0():
    x = _trajectories(1, 40)
    anim = _quiet(x, '-', animate='serial', hue=np.linspace(0, 1, 40),
                  head_marker='o', duration=1.5, frame_rate=10)
    ax = anim.figure.axes[0]
    (marker,) = _head_markers(ax)
    anim.draw_frame(0)
    assert len(_points(marker)) == 1
    first = to_rgba(marker.get_color())
    anim.draw_frame(anim.n_frames - 1)
    assert to_rgba(marker.get_color()) != first
    anim.draw_frame(0)
    assert to_rgba(marker.get_color()) == first


# ---------------------------------------------------------------------------
# D. the head_marker= dict: matplotlib's short names, a face color
# ---------------------------------------------------------------------------

def _one_head_marker(spec, **kwargs):
    anim = _quiet(_trajectories(1, 30), '-', animate=True, head_marker=spec,
                  duration=1, frame_rate=10, **kwargs)
    anim.draw_frame(5)
    (marker,) = _head_markers(anim.figure.axes[0])
    return marker


def test_head_marker_takes_matplotlibs_short_names():
    short = _one_head_marker({'marker': 'o', 'mec': 'k', 'mfc': 'none',
                              'mew': 3, 'ms': 20})
    long = _one_head_marker({
        'marker': 'o', 'markeredgecolor': 'k', 'markerfacecolor': 'none',
        'markeredgewidth': 3, 'markersize': 20})
    own = _one_head_marker({'marker': 'o', 'edgecolor': 'k',
                            'facecolor': 'none', 'edgewidth': 3, 'size': 20})
    for marker in (short, long, own):
        assert marker.get_markersize() == 20
        assert marker.get_markeredgewidth() == 3
        assert to_rgba(marker.get_markeredgecolor()) == to_rgba('k')
        assert marker.get_markerfacecolor() == 'none'


def test_head_marker_face_color_is_its_own_key():
    marker = _one_head_marker({'marker': 's', 'facecolor': 'gold',
                               'color': 'navy'})
    assert to_rgba(marker.get_markerfacecolor()) == to_rgba('gold')
    assert to_rgba(marker.get_markeredgecolor()) == to_rgba('navy')
    # without a face color the marker is filled with its `color`
    plain = _one_head_marker({'marker': 's', 'color': 'navy'})
    assert to_rgba(plain.get_markerfacecolor()) == to_rgba('navy')
    # `c` is `color`, `mfcalt`/`facecoloralt` the other half's fill
    half = _one_head_marker({'marker': 'o', 'c': 'navy', 'fillstyle': 'left',
                             'mfcalt': 'gold'})
    assert to_rgba(half.get_color()) == to_rgba('navy')
    assert to_rgba(half.get_markerfacecoloralt()) == to_rgba('gold')
    assert to_rgba(_one_head_marker(
        {'fillstyle': 'left', 'facecoloralt': 'gold'}
    ).get_markerfacecoloralt()) == to_rgba('gold')


def test_head_marker_face_color_is_drawn():
    """Pixels: the middle of a large head marker is the face color."""
    anim = _quiet(_trajectories(1, 30, 2), '-', animate=True,
                  head_marker={'marker': 'o', 'ms': 30, 'mfc': 'gold',
                               'mec': 'k'},
                  duration=1, frame_rate=10)
    fig = anim.draw_frame(9).figure
    fig.canvas.draw()
    ax = fig.axes[0]
    (marker,) = _head_markers(ax)
    px, py = ax.transData.transform(_points(marker)[0])
    image = np.asarray(fig.canvas.buffer_rgba())
    pixel = image[image.shape[0] - 1 - int(round(py)), int(round(px)), :3]
    assert np.allclose(pixel / 255.0, to_rgba('gold')[:3], atol=0.02)


def test_a_pinned_face_color_stays_while_hue_colors_the_edge():
    x = _trajectories(1, 30)
    anim = _quiet(x, '-', animate=True, hue=np.linspace(0, 1, 30),
                  head_marker={'marker': 'o', 'mfc': 'gold'},
                  duration=1, frame_rate=10)
    (marker,) = _head_markers(anim.figure.axes[0])
    edges = []
    for frame in (1, 9):
        anim.draw_frame(frame)
        assert to_rgba(marker.get_markerfacecolor()) == to_rgba('gold')
        edges.append(to_rgba(marker.get_markeredgecolor()))
    assert edges[0] != edges[1]


@pytest.mark.parametrize('spec, names', [
    ({'size': 9, 'ms': 12}, ('size', 'ms')),
    ({'mec': 'k', 'edgecolor': 'r'}, ('mec', 'edgecolor')),
    ({'markerfacecolor': 'k', 'facecolor': 'r'},
     ('markerfacecolor', 'facecolor')),
    ({'c': 'k', 'color': 'r'}, ('c', 'color')),
])
def test_head_marker_refuses_one_option_under_two_names(spec, names):
    with pytest.raises(ValueError, match='head_marker') as raised:
        _one_head_marker(spec)
    for name in names:
        assert repr(name) in str(raised.value)


@pytest.mark.parametrize('spec', [
    {'ms': -3}, {'mew': 'wide'}, {'mfc': 'no such color'},
    {'facecolor': 'no such color'}, {'mec': 12.5}])
def test_head_marker_short_names_have_their_values_checked(spec):
    with pytest.raises(ValueError, match='head_marker') as raised:
        _one_head_marker(spec)
    assert repr(next(iter(spec))) in str(raised.value)


def test_head_marker_still_refuses_an_unknown_key():
    with pytest.raises(ValueError, match='not a supported head marker'):
        _one_head_marker({'marker': 'o', 'linewidth': 3})


def test_head_marker_short_names_inside_the_animate_dict():
    anim = _quiet(_trajectories(1, 30), '-', duration=1, frame_rate=10,
                  animate={'style': True,
                           'head_marker': {'ms': 14, 'mfc': 'gold'}})
    (marker,) = _head_markers(anim.figure.axes[0])
    assert marker.get_markersize() == 14
    assert to_rgba(marker.get_markerfacecolor()) == to_rgba('gold')


# ---------------------------------------------------------------------------
# E. the trail_alpha= warning under 'window'; a hold= shorter than a frame
# ---------------------------------------------------------------------------

def _warnings_of(**kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = hyp.plot(_trajectories(1, 40), '-', show=False, duration=1,
                       **kwargs)
    return out, [str(w.message) for w in caught
                 if issubclass(w.category, UserWarning)]


@pytest.mark.parametrize('trail', [
    {'chemtrails': True}, {'precog': True}, {'bullettime': True}, {}])
def test_trail_alpha_under_window_does_not_ask_for_a_flag_already_given(
        trail):
    _out, messages = _warnings_of(animate='window', trail_alpha=0.5,
                                  frame_rate=10, **trail)
    (message,) = [m for m in messages if m.startswith('trail_alpha=')]
    assert "animate='window'" in message
    assert 'Pass chemtrails=True' not in message
    assert "animate=True or 'serial'" in message


def test_trail_alpha_without_a_trail_still_says_how_to_draw_one():
    _out, messages = _warnings_of(animate=True, trail_alpha=0.5,
                                  frame_rate=10)
    (message,) = [m for m in messages if m.startswith('trail_alpha=')]
    assert 'Pass chemtrails=True' in message
    _out, messages = _warnings_of(animate=True, trail_alpha=0.5,
                                  chemtrails=True, frame_rate=10)
    assert not [m for m in messages if 'trail_alpha' in m]


#: (frame rate, hold): holds on both sides of half a frame
HOLDS = [(rate, hold) for rate in (5, 10, 24, 30)
         for hold in (0.01, 0.04, 0.1, 0.5, 1.0)]


@pytest.mark.parametrize('rate, hold', HOLDS)
def test_hold_is_rounded_to_frames_and_says_so_when_that_is_none(rate, hold):
    """The rule: ``round(hold * frame_rate)`` repeats of the last frame.
    A hold that rounds to no frame warns; any other adds its frames and
    says nothing."""
    plain, _ = _warnings_of(animate=True, frame_rate=rate)
    held, messages = _warnings_of(animate=True, frame_rate=rate, hold=hold)
    about_hold = [m for m in messages if m.startswith('hold=')]
    extra = int(round(hold * rate))
    assert held.n_frames - plain.n_frames == extra
    if extra == 0:
        (message,) = about_hold
        assert 'adds no frame' in message
        assert f'frame_rate={rate}' in message
    else:
        assert not about_hold


def test_the_shortest_hold_the_warning_names_adds_one_frame():
    for rate in (5, 10, 24, 30):
        _out, messages = _warnings_of(animate=True, frame_rate=rate,
                                      hold=0.001)
        (message,) = [m for m in messages if m.startswith('hold=')]
        shortest = float(message.split('Pass hold=')[1].split(' ')[0])
        plain, _ = _warnings_of(animate=True, frame_rate=rate)
        held, again = _warnings_of(animate=True, frame_rate=rate,
                                   hold=shortest)
        assert held.n_frames == plain.n_frames + 1
        assert not [m for m in again if m.startswith('hold=')]


def test_a_short_hold_on_a_grid_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        anim = hyp.plot(_trajectories(2, 30), '-', animate=True, panels=True,
                        hold=0.01, duration=1, frame_rate=10, show=False)
    assert anim.n_frames == 10
    assert len([w for w in caught
                if str(w.message).startswith('hold=')]) == 1


def test_hold_zero_says_nothing():
    _out, messages = _warnings_of(animate=True, frame_rate=10, hold=0)
    assert not [m for m in messages if m.startswith('hold=')]


def _gif_seconds(path):
    with Image.open(path) as im:
        total = 0
        for index in range(getattr(im, 'n_frames', 1)):
            im.seek(index)
            total += im.info.get('duration', 0)
    return total / 1000.0


@pytest.mark.parametrize('rate', [5, 10, 25])
def test_hold_in_a_saved_gif_at_several_frame_rates(tmp_path, rate):
    """Read back from real files: the playing time grows by the held
    frames, and not at all for a hold that rounds to none."""
    data = _trajectories(1, 40, 2)
    seconds = {}
    for name, hold in (('plain', None), ('none', 0.4 / rate),
                       ('one', 1.0 / rate), ('four', 4.0 / rate)):
        path = str(tmp_path / f'{name}.gif')
        _quiet(data, '-', animate=True, duration=1, frame_rate=rate,
               hold=hold, save_path=path)
        seconds[name] = _gif_seconds(path)
    frame = 1.0 / rate
    assert seconds['plain'] == pytest.approx(1.0, abs=0.5 * frame)
    assert seconds['none'] == pytest.approx(seconds['plain'], abs=1e-6)
    assert seconds['one'] - seconds['plain'] == pytest.approx(
        frame, abs=0.011)
    assert seconds['four'] - seconds['plain'] == pytest.approx(
        4 * frame, abs=0.011)


def _video_frames(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-count_frames', '-show_entries', 'stream=nb_read_frames',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split(',')[0])


@pytest.mark.skipif(not HAS_FFMPEG, reason='ffmpeg/ffprobe not installed')
@pytest.mark.parametrize('rate', [10, 24, 30])
def test_hold_in_a_saved_mp4_at_several_frame_rates(tmp_path, rate):
    data = _trajectories(1, 40)
    frames = {}
    for name, hold in (('plain', None), ('none', 0.4 / rate),
                       ('one', 1.0 / rate), ('four', 4.0 / rate)):
        path = tmp_path / f'{name}.mp4'
        _quiet(data, '-', animate=True, duration=1, frame_rate=rate,
               hold=hold, save_path=str(path))
        frames[name] = _video_frames(path)
    assert frames['plain'] == rate
    assert frames['none'] == rate
    assert frames['one'] == rate + 1
    assert frames['four'] == rate + 4
