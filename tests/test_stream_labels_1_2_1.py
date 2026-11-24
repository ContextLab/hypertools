"""`labels=` on streaming plots (1.2.1, owner decision 2026-11-23).

A stream's labels are placed causally, once per emitted frame, by the
placement rule of every other hypertools label. Everything here runs a
real generator through `hyp.plot`, saves real frames, and reads the
labels back from the figure as each frame left it: the generator that
feeds the stream looks at the figure before it yields its next sample
(with ``stream_chunk=1`` that is the state of the frame just saved).

Nothing is compared against an absolute font metric: boxes are the ones
matplotlib drew, and bounds are stated in the library's own units.
"""
import gc
import warnings

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from mpl_toolkits.mplot3d import proj3d
from PIL import Image

import hypertools as hyp
from hypertools.plot import matplotlib_backend as mb

matplotlib.use('Agg')

IGNORED = 'no streaming implementation'


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close('all')
    yield
    plt.close('all')


# --------------------------------------------------------------- helpers

def _points(n, d, seed=0, centers=3, spread=0.06):
    """`n` samples of `d` features inside the unit box: the first eight
    span it (so the display box fitted on them holds every later
    sample), the others arrive near one of a few centers, next to each
    other."""
    rng = np.random.default_rng(seed)
    head = rng.uniform(-1.0, 1.0, size=(8, d))
    head[0], head[1] = -1.0, 1.0
    mids = rng.uniform(-0.45, 0.45, size=(centers, d))
    rest = (mids[rng.integers(0, centers, size=max(n - 8, 0))]
            + rng.normal(0.0, spread, size=(max(n - 8, 0), d)))
    return np.vstack([head, rest])[:n]


def _stream_labels(ax):
    return [text for text in ax.texts if getattr(text, '_hyp_stream', False)]


def _shown_points(ax):
    """Where the streamed line's vertices are, in the axes' 2-D data
    coordinates (projected with the camera for a 3-D axes)."""
    line = ax.lines[0]
    if hasattr(ax, 'get_proj'):
        x, y, z = (np.asarray(v, dtype=float) for v in line.get_data_3d())
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        return np.column_stack([x2, y2])
    return np.column_stack([np.asarray(v, dtype=float)
                            for v in line.get_data()])


def _snapshot(ax, consumed):
    """What the frame just drawn shows: per label (by sample index) its
    text, its anchor, its offset and the box matplotlib drew."""
    fig = ax.get_figure()
    points = _shown_points(ax)
    labels = {}
    for label in _stream_labels(ax):
        labels[label._hyp_global_idx] = {
            'text': label.get_text(),
            'xy': np.array(label.xy, dtype=float),
            'offset': np.array(label.xyann, dtype=float),
            'box': np.array(label.get_bbox_patch().get_window_extent()
                            .extents, dtype=float),
        }
    title = ax.title.get_window_extent().extents \
        if ax.title.get_text() else None
    return {'consumed': consumed, 'first': consumed - len(points),
            'points': points, 'labels': labels, 'title': title,
            'canvas': (fig.bbox.width, fig.bbox.height)}


def _live_figures():
    gc.collect()
    # `type()`, not `isinstance()`: the scan meets every tracked object,
    # weak proxies of collected objects among them, and `isinstance`
    # asks a proxy for its referent's class, which raises ReferenceError
    # once the referent is gone.
    return [thing for thing in gc.get_objects()
            if issubclass(type(thing), matplotlib.figure.Figure)]


def _recorded(samples, frames, start, ax=None, poke=None):
    """Yield `samples`; before each one from `start` on, record the
    figure as the last frame left it. `poke(k, ax)` may change the
    figure (turn the camera, draw it again) after the record is taken."""
    known = {id(figure) for figure in _live_figures()} if ax is None else ()
    axes = ax
    for k, row in enumerate(samples):
        if k >= start:
            if axes is None:
                # the figure `hyp.plot` made for this stream (it is not
                # one of pyplot's while the stream runs)
                (figure,) = [figure for figure in _live_figures()
                             if id(figure) not in known and figure.axes
                             and figure.axes[0].lines]
                axes = figure.axes[0]
            frames.append(_snapshot(axes, k))
            if poke is not None:
                poke(k, axes)
        yield row


def _run(samples, tmp_path, name='s.gif', ax=None, poke=None, init=8,
         **kwargs):
    frames = []
    kwargs.setdefault('frame_rate', 30)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_recorded(samples, frames, init + 1, ax, poke),
                       kwargs.pop('fmt', 'o'), stream_init=init,
                       stream_chunk=kwargs.pop('stream_chunk', 1),
                       show=False,
                       save_path=str(tmp_path / name),
                       **({'ax': ax} if ax is not None else {}), **kwargs)
    if fig.stream_info['error'] is not None:
        raise fig.stream_info['error']
    return fig, frames, caught


def _overlap(a, b, slack=0.5):
    return (min(a[2], b[2]) - max(a[0], b[0]) > slack
            and min(a[3], b[3]) - max(a[1], b[1]) > slack)


def _pairs(frame):
    boxes = [one['box'] for _i, one in sorted(frame['labels'].items())]
    return sum(_overlap(boxes[i], boxes[j])
               for i in range(len(boxes)) for j in range(i))


def _moves(frames):
    """Per label, its nonzero offset steps from one frame to the next."""
    steps = {}
    for before, after in zip(frames, frames[1:]):
        for index, one in after['labels'].items():
            if index in before['labels']:
                step = one['offset'] - before['labels'][index]['offset']
                if np.hypot(*step) > 1e-9:
                    steps.setdefault(index, []).append(step)
    return steps


def _turn_backs(frames):
    return sum(float(np.dot(a, b)) < 0.0
               for steps in _moves(frames).values()
               for a, b in zip(steps, steps[1:]))


def _largest_jump(frames):
    return max([float(np.hypot(*step)) for steps in _moves(frames).values()
                for step in steps] or [0.0])


def _every_fifth(index, sample):
    return f'p{index}' if index % 5 == 0 else None


# --------------------------------------------------- the forms of labels=

FORMS = {
    'callable': lambda names: (lambda index, sample: names[index]),
    'list': list,
    'tuple': tuple,
    'array': lambda names: np.array(names, dtype=object),
    'series': lambda names: pd.Series(names, dtype=object),
    'iterator': iter,
    'generator': lambda names: (name for name in names),
}


@pytest.mark.parametrize('form', sorted(FORMS))
@pytest.mark.parametrize('ndims', [2, 3])
def test_every_form_of_labels_draws_the_same_labels(form, ndims):
    data = _points(30, ndims)
    names = [f'n{i}' if i % 4 == 0 else None for i in range(30)]
    fig = hyp.plot(iter(data), 'o', stream_init=10, stream_chunk=5,
                   ndims=ndims, labels=FORMS[form](names), show=False)
    assert fig.stream_info['labels'] == names
    drawn = {label._hyp_global_idx: label.get_text()
             for label in _stream_labels(fig.axes[0])}
    assert drawn == {i: name for i, name in enumerate(names)
                     if name is not None}


def test_a_callable_gets_the_index_and_the_sample_as_yielded():
    rows = [{'name': f'row {i}', 'a': float(i % 7), 'b': float(i % 5),
             'c': float(i % 3)} for i in range(24)]
    seen = []

    def label(index, sample):
        seen.append((index, sample))
        return sample['name'] if index in (0, 11, 23) else None

    fig = hyp.plot(iter(rows), 'o', stream_init=8, stream_chunk=4, ndims=2,
                   labels=label, show=False)
    assert [index for index, _ in seen] == list(range(24))
    assert all(sample is rows[index] for index, sample in seen)
    assert sorted(label.get_text()
                  for label in _stream_labels(fig.axes[0])) \
        == ['row 0', 'row 11', 'row 23']


def test_an_endless_label_iterator_is_consumed_in_step_with_the_samples():
    pulled = []

    def names():
        k = 0
        while True:
            pulled.append(k)
            yield f'#{k}'
            k += 1

    fig = hyp.plot(iter(_points(20, 2)), 'o', stream_init=8, stream_chunk=3,
                   ndims=2, labels=names(), show=False)
    assert pulled == list(range(20))
    assert fig.stream_info['labels'] == [f'#{k}' for k in range(20)]


@pytest.mark.parametrize('ndims', [2, 3])
def test_labels_shorter_than_the_stream_label_the_first_samples(ndims):
    fig = hyp.plot(iter(_points(25, ndims)), 'o', stream_init=10,
                   stream_chunk=5, ndims=ndims, labels=['a', None, 'c'],
                   show=False)
    assert fig.stream_info['labels'] == ['a', None, 'c'] + [None] * 22
    assert {label._hyp_global_idx: label.get_text()
            for label in _stream_labels(fig.axes[0])} == {0: 'a', 2: 'c'}


@pytest.mark.parametrize('labels', [
    [None] * 25, lambda index, sample: None, [], ['', '  ', float('nan')],
], ids=['all None', 'callable returning None', 'empty list',
        'blank and NaN'])
def test_no_label_is_drawn_for_none(labels):
    fig = hyp.plot(iter(_points(25, 3)), 'o', stream_init=10,
                   stream_chunk=5, labels=labels, show=False)
    assert fig.stream_info['labels'] == [None] * 25
    assert not fig.axes[0].texts
    assert fig.stream_info['n_samples'] == 25


def test_without_labels_nothing_is_recorded_or_drawn():
    fig = hyp.plot(iter(_points(25, 3)), 'o', stream_init=10,
                   stream_chunk=5, show=False)
    assert fig.stream_info['labels'] is None
    assert not fig.axes[0].texts


@pytest.mark.parametrize('labels', ['name', 5, True],
                         ids=['a bare string', 'a number', 'True'])
def test_an_unusable_labels_raises_before_a_sample_is_consumed(labels):
    consumed = []

    def source():
        for row in _points(20, 3):
            consumed.append(1)
            yield row

    with pytest.raises(TypeError, match='labels='):
        hyp.plot(source(), 'o', stream_init=10, labels=labels, show=False)
    assert not consumed


# ------------------------------------------------ the labels on the frames

@pytest.mark.parametrize('ndims', [2, 3])
def test_twelve_labels_follow_their_points_and_stay_clear(ndims, tmp_path):
    """Samples arrive next to each other (three small clusters), every
    fifth one labeled: 12 labels by the end.

    Every frame: each label is anchored at its own sample's drawn
    position; no label lies on the title or off the canvas. Over the
    stream: no label turns back, and no two labels overlap on any frame
    (measured: 0 pairs on every frame, 2-D and 3-D, matplotlib 3.10 and
    3.11; a new label takes a free place and the others stay)."""
    data = _points(60, ndims, seed=1)
    fig, frames, caught = _run(data, tmp_path, ndims=ndims,
                               labels=_every_fifth,
                               title='Twelve labels on a stream')
    assert not [w for w in caught if IGNORED in str(w.message)]
    assert len(frames) == 60 - 9
    assert len(frames[-1]['labels']) == 12
    width, height = frames[0]['canvas']
    for frame in frames:
        assert frame['first'] == 0
        for index, one in frame['labels'].items():
            assert one['text'] == f'p{index}'
            np.testing.assert_allclose(
                one['xy'], frame['points'][index - frame['first']],
                atol=1e-9)
            box = one['box']
            assert box[0] >= -0.5 and box[1] >= -0.5
            assert box[2] <= width + 0.5 and box[3] <= height + 0.5
            assert not _overlap(box, frame['title'])
        assert sorted(frame['labels']) == [
            i for i in range(frame['consumed']) if i % 5 == 0]
    assert _turn_backs(frames) == 0
    assert max(_pairs(frame) for frame in frames) == 0
    # ... nor do two boxes touch: a new label is given clear space round
    # it (`_LABEL_CLEARANCE_PT`, less the half point a box is rounded by)
    clear = (mb._LABEL_CLEARANCE_PT - 0.5) * fig.dpi / 72.0
    boxes = [one['box'] for _i, one in sorted(frames[-1]['labels'].items())]
    for i, a in enumerate(boxes):
        for b in boxes[:i]:
            apart = max(a[0] - b[2], b[0] - a[2], a[1] - b[3], b[1] - a[3])
            assert apart >= clear
    # a label that is there stays where it is while new ones arrive
    assert _largest_jump(frames) == 0.0


def test_labels_follow_their_points_when_the_view_turns(tmp_path):
    """The camera is turned twice while the stream runs (as a reader
    turning the plot would). On every frame after it each label is
    still anchored at its point's projection for the camera of that
    frame; the labels that have to change place glide (no jump larger
    than a glide's largest step) and none turns back."""
    data = _points(70, 3, seed=2)

    def turn(k, ax):
        if k in (30, 50):
            ax.view_init(elev=ax.elev + 12, azim=ax.azim + 55)

    fig, frames, _ = _run(data, tmp_path, ndims=3, labels=_every_fifth,
                          title='Turned while streaming', poke=turn)
    width, height = frames[0]['canvas']
    for frame in frames:
        for index, one in frame['labels'].items():
            np.testing.assert_allclose(
                one['xy'], frame['points'][index], atol=1e-9)
            box = one['box']
            assert box[0] >= -0.5 and box[1] >= -0.5
            assert box[2] <= width + 0.5 and box[3] <= height + 0.5
            assert not _overlap(box, frame['title'])
    moves = _moves(frames)
    assert moves, 'the turn was meant to make some label change place'
    assert _turn_backs(frames) == 0
    # a glide takes `ease` updates (smoothstep: its largest step is 1.5
    # times the mean one), over at most the way from one side of the
    # point to the other at the furthest reach
    ease = max(mb._STREAM_LABEL_EASE_MIN,
               round(mb._STREAM_LABEL_EASE_SECONDS * 30))
    sizes = np.array([mb._label_size_pt(label)
                      for label in _stream_labels(fig.axes[0])])
    furthest = 2 * (max(mb._LABEL_REACHES_PT)
                    + 0.5 * float(np.hypot(*sizes.max(axis=0))))
    assert _largest_jump(frames) <= 1.5 * furthest / ease
    # every change of place is spread over a glide's worth of frames
    assert min(len(steps) for steps in moves.values()) >= ease - 2
    # A turn throws the points onto each other at once, and the future
    # was not known: on the frame of the turn labels overlap (measured:
    # 4 pairs after the second turn, 0 after the first), and they are
    # clear again within one glide. No pair on any other frame, and
    # never more pairs than the turn itself made.
    pairs = {frame['consumed']: _pairs(frame) for frame in frames}
    for turned in (31, 51):
        after = [pairs[k] for k in range(turned, turned + ease + 1)]
        assert max(after) == after[0] <= 6
        assert after[-1] == 0
    assert not [k for k, count in pairs.items() if count
                and not (31 <= k <= 31 + ease or 51 <= k <= 51 + ease)]


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_label_leaves_with_its_point_under_stream_window(ndims, tmp_path):
    data = _points(60, ndims, seed=3)
    fig, frames, _ = _run(data, tmp_path, ndims=ndims, stream_window=15,
                          labels=lambda index, sample: (
                              f'p{index}' if index % 3 == 0 else None),
                          fmt='-')
    ax = fig.axes[0]
    for frame in frames:
        shown = [i for i in range(frame['first'], frame['consumed'])
                 if i % 3 == 0]
        assert len(frame['points']) == min(15, frame['consumed'])
        assert sorted(frame['labels']) == shown
        for index, one in frame['labels'].items():
            np.testing.assert_allclose(
                one['xy'], frame['points'][index - frame['first']],
                atol=1e-9)
    # the returned figure holds the last window's labels and no others
    assert sorted(label._hyp_global_idx for label in _stream_labels(ax)) \
        == [i for i in range(45, 60) if i % 3 == 0]
    assert len(ax.texts) == 5
    assert len(ax._hyp_labels_and_points) == 5
    # ... while every consumed sample's label is kept in stream_info
    assert fig.stream_info['labels'] == [
        f'p{i}' if i % 3 == 0 else None for i in range(60)]
    assert _turn_backs(frames) == 0
    assert max(_pairs(frame) for frame in frames) == 0


@pytest.mark.parametrize('ndims,window', [(2, None), (3, None), (3, 20)])
def test_the_saved_frames_are_the_same_run_to_run(ndims, window, tmp_path):
    saved = []
    for run in ('a', 'b'):
        plt.close('all')
        path = tmp_path / f'{run}.gif'
        hyp.plot(iter(_points(50, ndims, seed=4)), '-o', stream_init=10,
                 stream_chunk=2, ndims=ndims, stream_window=window,
                 labels=lambda index, sample: (
                     f'p{index}' if index % 4 == 0 else None),
                 title='Twice the same', save_path=str(path), show=False)
        with Image.open(path) as gif:
            frames = []
            for k in range(gif.n_frames):
                gif.seek(k)
                frames.append((np.array(gif.convert('RGB')),
                               gif.info.get('duration')))
        saved.append((path.read_bytes(), frames))
    (bytes_a, frames_a), (bytes_b, frames_b) = saved
    assert len(frames_a) == len(frames_b) > 10
    for (image_a, time_a), (image_b, time_b) in zip(frames_a, frames_b):
        assert time_a == time_b
        assert np.array_equal(image_a, image_b)
    assert bytes_a == bytes_b


def test_the_labels_are_stepped_per_frame_and_not_per_draw(tmp_path):
    """Drawing the figure more often (a GUI repaints whenever it likes)
    changes nothing: the labels of every frame are where they are
    without the extra draws."""
    data = _points(60, 3, seed=5)

    def turn(k, ax):
        if k == 25:
            ax.view_init(elev=ax.elev + 10, azim=ax.azim + 60)

    def turn_and_draw(k, ax):
        turn(k, ax)
        for _ in range(k % 3):
            ax.get_figure().canvas.draw()

    _, plain, _ = _run(data, tmp_path, 'plain.gif', ndims=3,
                       labels=_every_fifth, poke=turn)
    plt.close('all')
    _, drawn, _ = _run(data, tmp_path, 'drawn.gif', ndims=3,
                       labels=_every_fifth, poke=turn_and_draw)
    assert len(plain) == len(drawn)
    assert _moves(plain), 'the turn was meant to make some label move'
    for one, other in zip(plain, drawn):
        assert sorted(one['labels']) == sorted(other['labels'])
        for index in one['labels']:
            np.testing.assert_array_equal(one['labels'][index]['offset'],
                                          other['labels'][index]['offset'])
    assert (tmp_path / 'plain.gif').read_bytes() \
        == (tmp_path / 'drawn.gif').read_bytes()


# ------------------------------------------------------------ edge inputs

@pytest.mark.parametrize('ndims', [2, 3])
def test_a_stream_that_ends_during_stream_init_is_labeled(ndims, tmp_path):
    path = tmp_path / 'short.gif'
    fig = hyp.plot(iter(_points(12, ndims)), 'o', stream_init=100,
                   ndims=ndims, labels=lambda i, s: f'p{i}' if i % 4 == 0
                   else None, save_path=str(path), show=False)
    assert fig.stream_info['n_samples'] == 12
    assert sorted(label.get_text()
                  for label in _stream_labels(fig.axes[0])) \
        == ['p0', 'p4', 'p8']
    assert path.stat().st_size > 0


def test_stream_max_smaller_than_stream_init_labels_only_those_samples():
    asked = []

    def label(index, sample):
        asked.append(index)
        return f'p{index}'

    fig = hyp.plot(iter(_points(40, 3)), 'o', stream_init=30, stream_max=6,
                   labels=label, show=False)
    assert asked == list(range(6))
    assert fig.stream_info['n_samples'] == 6
    assert fig.stream_info['labels'] == [f'p{i}' for i in range(6)]
    assert len(_stream_labels(fig.axes[0])) == 6


@pytest.mark.parametrize('projection', [None, '3d'])
def test_labels_go_into_the_axes_the_caller_supplied(projection, tmp_path):
    ndims = 3 if projection else 2
    canvas = plt.figure(figsize=(10, 5))
    other = canvas.add_subplot(1, 2, 1)
    other.plot([0, 1], [0, 1])
    ax = canvas.add_subplot(1, 2, 2, projection=projection)
    fig, frames, _ = _run(_points(40, ndims, seed=6), tmp_path, ax=ax,
                          ndims=ndims, labels=_every_fifth)
    assert fig is canvas
    assert not other.texts
    assert len(_stream_labels(ax)) == 8
    width, height = frames[0]['canvas']
    left_panel = other.get_window_extent().extents
    for frame in frames:
        for index, one in frame['labels'].items():
            np.testing.assert_allclose(one['xy'], frame['points'][index],
                                       atol=1e-9)
            box = one['box']
            assert box[0] >= -0.5 and box[1] >= -0.5
            assert box[2] <= width + 0.5 and box[3] <= height + 0.5
            # the next panel is an obstacle, as in a static figure
            assert not _overlap(box, left_panel)
    assert max(_pairs(frame) for frame in frames) == 0
    assert _turn_backs(frames) == 0


def test_stream_labels_keep_off_a_static_plot_s_labels_in_the_same_axes(
        tmp_path):
    canvas, ax = plt.subplots(figsize=(7, 6))
    still = _points(40, 2, seed=7)[8:12]
    hyp.plot(still, 'o', ax=ax, labels=['still a', 'still b', 'still c',
                                        'still d'], show=False)
    canvas.canvas.draw()
    static = [text for text in ax.texts]
    assert len(static) == 4
    fig, frames, _ = _run(_points(40, 2, seed=7), tmp_path, ax=ax, ndims=2,
                          labels=_every_fifth)
    assert len(_stream_labels(ax)) == 8
    assert all(text.axes is ax for text in static)
    canvas.canvas.draw()
    theirs = [text.get_bbox_patch().get_window_extent().extents
              for text in static]
    ours = [text.get_bbox_patch().get_window_extent().extents
            for text in _stream_labels(ax)]
    assert not any(_overlap(a, b) for a in ours for b in theirs)


def test_label_alpha_is_honored_on_a_stream():
    fig = hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                   labels=['a', 'b'], label_alpha=0.9, show=False)
    assert [label.get_bbox_patch().get_alpha()
            for label in _stream_labels(fig.axes[0])] == [0.9, 0.9]
    with pytest.raises(ValueError, match='label_alpha'):
        hyp.plot(iter(_points(20, 3)), 'o', stream_init=10, labels=['a'],
                 label_alpha=3, show=False)


def test_a_labels_callable_that_raises():
    def early(index, sample):
        raise KeyError('no such field')

    with pytest.raises(KeyError):
        hyp.plot(iter(_points(30, 3)), 'o', stream_init=10, labels=early,
                 show=False)

    def late(index, sample):
        if index >= 22:
            raise KeyError('no such field')
        return f'p{index}' if index % 5 == 0 else None

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(iter(_points(40, 3)), 'o', stream_init=10,
                       stream_chunk=5, labels=late, show=False)
    info = fig.stream_info
    assert isinstance(info['error'], KeyError)
    # the samples of the chunk BEFORE the one that failed are kept
    assert info['n_samples'] == 22 == len(info['labels'])
    assert len(info['data'][0]) == len(info['xform_data'][0]) == 22
    assert sorted(label.get_text()
                  for label in _stream_labels(fig.axes[0])) \
        == ['p0', 'p10', 'p15', 'p20', 'p5']


def _in_step(fig, count):
    """`count` samples are kept, and label k is on sample k."""
    info = fig.stream_info
    assert info['n_samples'] == count
    assert info['labels'] == [f'p{k}' for k in range(count)]
    assert len(info['data'][0]) == len(info['xform_data'][0]) == count
    np.testing.assert_array_equal(info['data'][0][:, 0], np.arange(count))
    ax = fig.axes[0]
    points = _shown_points(ax)
    assert len(points) == count
    labels = _stream_labels(ax)
    assert [label._hyp_global_idx for label in labels] == list(range(count))
    for label in labels:
        assert label.get_text() == f'p{label._hyp_global_idx}'
        np.testing.assert_allclose(label.xy, points[label._hyp_global_idx],
                                   atol=1e-9)


def _row(k, d=2):
    return [float(k), float(k % 2), float(k % 3)][:d]


def _counted(count=12, d=2):
    return iter([_row(k, d) for k in range(count)])


@pytest.mark.parametrize('ndims', [2, 3])
def test_a_label_error_in_the_middle_of_a_chunk_keeps_the_samples_before_it(
        ndims):
    """Review of 237e9111: the labels of a chunk were built in one go
    before anything was kept, so a callable that raised for sample 4
    (chunk 3..5) lost sample 3 and its label too."""
    def label(index, sample):
        if index == 4:
            raise ValueError('no label for sample 4')
        return f'p{index}'

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(_counted(d=ndims), 'o', reduce=None, ndims=ndims,
                       stream_init=3, stream_chunk=3, labels=label,
                       show=False)
    assert isinstance(fig.stream_info['error'], ValueError)
    assert fig.stream_info['truncated'] is True
    _in_step(fig, 4)


def test_a_label_iterator_that_raises_keeps_the_samples_before_it():
    def names():
        for k in range(7):
            yield f'p{k}'
        raise LookupError('the names ran dry')

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(_counted(), 'o', reduce=None, ndims=2, stream_init=3,
                       stream_chunk=3, labels=names(), show=False)
    assert isinstance(fig.stream_info['error'], LookupError)
    _in_step(fig, 7)


def test_a_label_error_inside_stream_init_keeps_the_samples_before_it():
    """Like a bad sample inside the head: the models are fitted on the
    samples before it and the figure is returned with them."""
    def label(index, sample):
        if index == 5:
            raise ValueError('no label for sample 5')
        return f'p{index}'

    with pytest.warns(RuntimeWarning,
                      match='while consuming the first stream_init'):
        fig = hyp.plot(_counted(), 'o', reduce=None, ndims=2, stream_init=8,
                       stream_chunk=3, labels=label, show=False)
    assert isinstance(fig.stream_info['error'], ValueError)
    _in_step(fig, 5)


def test_a_source_that_dies_mid_chunk_leaves_the_labels_in_step():
    asked = []

    def source():
        for k in range(8):
            yield _row(k)
        raise ConnectionError('the source went away')

    def label(index, sample):
        asked.append((index, sample[0]))
        return f'p{index}'

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(source(), 'o', reduce=None, ndims=2, stream_init=3,
                       stream_chunk=3, labels=label, show=False)
    assert isinstance(fig.stream_info['error'], ConnectionError)
    # samples 6 and 7 came before the source died, in an unfinished chunk
    _in_step(fig, 8)
    assert asked == [(k, float(k)) for k in range(8)]


def test_a_bad_sample_mid_chunk_keeps_the_samples_and_labels_before_it():
    rows = [_row(k) for k in range(12)]
    rows[7] = ['not', 'a number']
    pulled = []

    def names():
        k = 0
        while True:
            pulled.append(k)
            yield f'p{k}'
            k += 1

    with pytest.warns(RuntimeWarning, match='streaming stopped early'):
        fig = hyp.plot(iter(rows), 'o', reduce=None, ndims=2, stream_init=3,
                       stream_chunk=3, labels=names(), show=False)
    assert isinstance(fig.stream_info['error'], ValueError)
    _in_step(fig, 7)
    # no label was pulled for the sample that could not be read
    assert pulled == list(range(7))


def test_a_one_feature_stream_that_ends_in_the_head_says_labels_are_not_drawn():
    with pytest.warns(UserWarning, match='labels= is not drawn'):
        fig = hyp.plot(iter([[float(v)] for v in range(6)]), stream_init=10,
                       labels=['a', 'b'], show=False)
    assert fig.stream_info['n_samples'] == 6


def test_more_labels_than_are_arranged_warns_once():
    count = mb._STREAM_LABEL_MAX + 12
    data = np.vstack([_points(8, 2), np.random.default_rng(8).uniform(
        -1, 1, size=(count + 8, 2))])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(iter(data), '.', stream_init=count, stream_chunk=8,
                       ndims=2, labels=lambda i, s: str(i), show=False)
    said = [w for w in caught if 'kept off each other' in str(w.message)]
    assert len(said) == 1
    assert str(mb._STREAM_LABEL_MAX) in str(said[0].message)
    labels = _stream_labels(fig.axes[0])
    assert len(labels) == len(data)
    # the labels beyond the arranged ones hang at the usual place
    home = [mb._place_center_offset(mb._LABEL_HOME, mb._label_size_pt(label))
            for label in labels[mb._STREAM_LABEL_MAX:]]
    assert [tuple(label.xyann) for label in labels[mb._STREAM_LABEL_MAX:]] \
        == home


def test_a_line_and_marker_stream_keeps_no_stale_head_markers():
    """Found while adding stream labels: '-o' drew the head twice, as a
    line with its markers off and a markers-only copy that was never
    updated. The streamed samples had no marker, and the head's markers
    stayed on screen after `stream_window` had scrolled them away."""
    for ndims in (2, 3):
        fig = hyp.plot(iter(_points(40, ndims)), '-o', stream_init=10,
                       stream_chunk=5, stream_window=12, ndims=ndims,
                       markersize=5, show=False)
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        assert len(_shown_points(ax)) == 12
        line = ax.lines[0]
        assert line.get_marker() == 'o' and line.get_linestyle() == '-'
        assert line.get_markevery() is None      # a marker on every vertex
        assert line.get_markersize() == 5
    # a marker-only and a line-only format are drawn as before
    for fmt, marker, style in (('o', 'o', 'None'), ('-', 'None', '-')):
        fig = hyp.plot(iter(_points(40, 3)), fmt, stream_init=10,
                       stream_chunk=5, show=False)
        (line,) = fig.axes[0].lines
        assert (line.get_marker(), line.get_linestyle()) == (marker, style)
        assert line.get_markevery() is None


def test_a_stream_into_a_callers_axes_updates_its_own_line():
    """The streamed line was looked up as the first line of the figure's
    first axes: with a line already there (in another panel, or in the
    same axes) the stream updated that one."""
    canvas = plt.figure(figsize=(10, 5))
    other = canvas.add_subplot(1, 2, 1)
    (theirs,) = other.plot([0.0, 1.0], [0.0, 1.0])
    ax = canvas.add_subplot(1, 2, 2)
    (mine,) = ax.plot([-1.0, 1.0], [0.5, 0.5])
    fig = hyp.plot(iter(_points(30, 2)), '-', stream_init=10,
                   stream_chunk=5, ndims=2, ax=ax, show=False)
    assert fig is canvas
    assert [list(v) for v in theirs.get_data()] == [[0.0, 1.0], [0.0, 1.0]]
    assert [list(v) for v in mine.get_data()] == [[-1.0, 1.0], [0.5, 0.5]]
    streamed = [line for line in ax.lines if line is not mine]
    assert len(streamed) == 1
    assert len(streamed[0].get_xdata()) == 30


# ------------------------------------------------------------ the warnings

def test_labels_is_no_longer_named_as_ignored():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                 labels=['a', 'b'], label_alpha=0.7, show=False)
    assert not [w for w in caught if IGNORED in str(w.message)]

    with pytest.warns(UserWarning, match=IGNORED) as caught:
        hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                 labels=['a', 'b'], legend=['a stream'], show=False)
    message = next(str(w.message) for w in caught
                   if IGNORED in str(w.message))
    ignored, honored = message.split('Parameters honored for streams:')
    assert 'legend' in ignored and 'labels' not in ignored
    assert 'labels' in honored and 'label_alpha' in honored


@pytest.mark.parametrize('name,value', [
    ('legend', ['a stream']), ('caption', 'A caption'),
    ('suptitle', 'A suptitle'), ('head_marker', 'o'),
    ('tail_linewidth', 4), ('trail_alpha', 0.5), ('label_anchor', 'last'),
])
def test_the_other_arguments_still_warn_as_ignored(name, value):
    with pytest.warns(UserWarning, match=IGNORED) as caught:
        fig = hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                       labels=['a', 'b'], show=False, **{name: value})
    message = next(str(w.message) for w in caught
                   if IGNORED in str(w.message))
    assert name in message.split('Parameters honored for streams:')[0]
    assert len(_stream_labels(fig.axes[0])) == 2


def test_hold_on_a_stream_still_raises():
    """`hold=` is checked before a stream is recognized and asks for an
    animated plot, as it did before streams had labels (not changed
    here: streaming `hold=` is not built)."""
    with pytest.raises(ValueError, match='hold= requires an animated plot'):
        hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                 labels=['a', 'b'], hold=1, show=False)


def test_a_plotly_request_is_ignored_and_the_labels_are_drawn():
    with pytest.warns(UserWarning, match=IGNORED) as caught:
        fig = hyp.plot(iter(_points(20, 3)), 'o', stream_init=10,
                       labels=['a', 'b'], backend='plotly', show=False)
    message = next(str(w.message) for w in caught
                   if IGNORED in str(w.message))
    assert 'backend' in message.split('Parameters honored')[0]
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(_stream_labels(fig.axes[0])) == 2
