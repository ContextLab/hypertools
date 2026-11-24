"""`legend=`, `caption=` and `suptitle=` on streaming plots (1.2.1, owner
decision 2026-11-24: "Add the three now").

They are placed once, by the static plot of the first `stream_init`
samples, and stay for the whole stream. Everything here runs a real
generator through `hyp.plot` and reads the figure as each saved frame
left it (the generator that feeds the stream looks at the figure before
it yields its next sample).
"""
import gc
import warnings

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import proj3d

import hypertools as hyp

IGNORED = 'no streaming implementation'
CAPTION = 'A caption placed once, under the plot, for the whole stream.'
SUPTITLE = 'Suptitle of the stream'


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close('all')
    yield
    plt.close('all')


def _points(n, d, seed=0):
    """`n` samples of `d` features in the unit box; the first eight span
    it, so the display box fitted on them holds every later sample."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-1.0, 1.0, size=(n, d))
    data[0], data[1] = -1.0, 1.0
    return data


def _live_figures():
    gc.collect()
    return [thing for thing in gc.get_objects()
            # `type()`, not `isinstance()`: a weak proxy of a collected
            # object met in the scan raises ReferenceError when asked
            # for its referent's class.
            if issubclass(type(thing), matplotlib.figure.Figure)]


def _legend(ax):
    fig = ax.get_figure()
    return ax.get_legend() or (fig.legends[0] if fig.legends else None)


def _vertices_px(ax):
    """The streamed line's vertices in display pixels."""
    line = ax.lines[0]
    if hasattr(ax, 'get_proj'):
        x, y, z = (np.asarray(v, dtype=float) for v in line.get_data_3d())
        x, y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    else:
        x, y = (np.asarray(v, dtype=float) for v in line.get_data())
    return ax.transData.transform(np.column_stack([x, y]))


def _snapshot(ax, consumed):
    fig = ax.get_figure()
    legend = _legend(ax)
    return {
        'consumed': consumed,
        'canvas': (fig.bbox.width, fig.bbox.height),
        'vertices': _vertices_px(ax),
        'legend': None if legend is None
        else np.array(legend.get_window_extent().extents),
        'title': np.array(ax.title.get_window_extent().extents)
        if ax.title.get_text() else None,
        'texts': {text.get_text(): np.array(
            text.get_window_extent().extents) for text in fig.texts},
        'labels': {label.get_text(): np.array(
            label.get_bbox_patch().get_window_extent().extents)
            for label in ax.texts if getattr(label, '_hyp_stream', False)},
    }


def _recorded(samples, frames, start):
    known = {id(figure) for figure in _live_figures()}
    axes = None
    for k, row in enumerate(samples):
        if k >= start:
            if axes is None:
                (figure,) = [figure for figure in _live_figures()
                             if id(figure) not in known and figure.axes
                             and figure.axes[0].lines]
                axes = figure.axes[0]
            frames.append(_snapshot(axes, k))
        yield row


def _run(samples, tmp_path, name='s.gif', **kwargs):
    frames = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(_recorded(samples, frames, 9), kwargs.pop('fmt', '-o'),
                       stream_init=8, stream_chunk=1, show=False,
                       save_path=str(tmp_path / name), **kwargs)
    if fig.stream_info['error'] is not None:
        raise fig.stream_info['error']
    return fig, frames, caught


def _overlap(a, b, slack=0.5):
    return (min(a[2], b[2]) - max(a[0], b[0]) > slack
            and min(a[3], b[3]) - max(a[1], b[1]) > slack)


def _inside(box, canvas, slack=0.5):
    return (box[0] >= -slack and box[1] >= -slack
            and box[2] <= canvas[0] + slack and box[3] <= canvas[1] + slack)


@pytest.mark.parametrize('ndims', [2, 3])
def test_legend_caption_and_suptitle_are_drawn_once_and_stay_clear(
        ndims, tmp_path):
    """All three with a title and labels, on every frame up to the last:
    each is inside the canvas, none lies on another or on the title, no
    vertex of the streamed line is under the legend, no label lies on
    any of them, and none of them moves from frame to frame."""
    data = _points(60, ndims, seed=1)
    fig, frames, caught = _run(
        data, tmp_path, ndims=ndims, title='A stream with fixed texts',
        suptitle=SUPTITLE, caption=CAPTION, legend=['the stream'],
        labels=lambda i, s: f'p{i}' if i % 5 == 0 else None)
    assert not [w for w in caught if IGNORED in str(w.message)]
    assert len(frames) == 60 - 9
    first = frames[0]
    assert set(first['texts']) == {CAPTION, SUPTITLE}
    assert first['legend'] is not None
    for frame in frames:
        canvas = frame['canvas']
        fixed = {'legend': frame['legend'], 'title': frame['title'],
                 **frame['texts']}
        for name, box in fixed.items():
            assert _inside(box, canvas), name
            np.testing.assert_array_equal(
                box, {'legend': first['legend'], 'title': first['title'],
                      **first['texts']}[name])
        names = sorted(fixed)
        for i, a in enumerate(names):
            for b in names[:i]:
                assert not _overlap(fixed[a], fixed[b]), (a, b)
        # the stream's content keeps off the legend, late frames included
        legend = frame['legend']
        points = frame['vertices']
        assert len(points) == frame['consumed']
        under = ((points[:, 0] > legend[0]) & (points[:, 0] < legend[2])
                 & (points[:, 1] > legend[1]) & (points[:, 1] < legend[3]))
        assert not under.any()
        # the labels keep off all of it and stay inside the canvas
        for text, box in frame['labels'].items():
            assert _inside(box, canvas), text
            for name, other in fixed.items():
                assert not _overlap(box, other), (text, name)
    assert len(frames[-1]['labels']) == 12
    # the caption is under the plot and the suptitle above the title
    assert first['texts'][CAPTION][3] < first['vertices'][:, 1].min()
    assert first['texts'][SUPTITLE][1] >= first['title'][3] - 0.5


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('fmt', ['-', '-o', 'o'])
def test_the_legend_key_matches_the_streamed_line(ndims, fmt):
    fig = hyp.plot(iter(_points(30, ndims)), fmt, stream_init=10,
                   stream_chunk=5, ndims=ndims, legend=['the stream'],
                   markersize=5, linewidth=2, show=False)
    ax = fig.axes[0]
    legend = _legend(ax)
    assert [text.get_text() for text in legend.get_texts()] == ['the stream']
    (key,), line = legend.legend_handles, ax.lines[0]
    assert key.get_marker() == line.get_marker()
    assert key.get_linestyle() == line.get_linestyle()
    assert matplotlib.colors.same_color(key.get_color(), line.get_color())
    assert len(_vertices_px(ax)) == 30


def test_the_styling_arguments_are_honored():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(iter(_points(30, 3)), '-o', stream_init=10,
                       stream_chunk=5, legend=['a'],
                       legend_kwargs={'title': 'Key'}, legend_colors=['k'],
                       caption='cap', caption_kwargs={'color': 'gray'},
                       suptitle='sup', suptitle_kwargs={'size': 18},
                       show=False)
    assert not [w for w in caught if IGNORED in str(w.message)]
    legend = _legend(fig.axes[0])
    assert legend.get_title().get_text() == 'Key'
    assert matplotlib.colors.same_color(
        legend.legend_handles[0].get_color(), 'k')
    texts = {text.get_text(): text for text in fig.texts}
    assert texts['cap'].get_color() == 'gray'
    assert texts['sup'].get_size() == 18


@pytest.mark.parametrize('name,value', [
    ('legend', ['a stream']), ('caption', 'A caption'),
    ('suptitle', 'A suptitle')])
def test_each_is_honored_alone_and_no_longer_named_as_ignored(name, value):
    for ndims in (2, 3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fig = hyp.plot(iter(_points(30, ndims)), '-', stream_init=10,
                           stream_chunk=5, ndims=ndims, show=False,
                           **{name: value})
        assert not [w for w in caught if IGNORED in str(w.message)]
        if name == 'legend':
            assert [t.get_text() for t in _legend(fig.axes[0]).get_texts()] \
                == value
        else:
            assert [text.get_text() for text in fig.texts] == [value]
        plt.close('all')
    with pytest.warns(UserWarning, match=IGNORED) as caught:
        hyp.plot(iter(_points(30, 3)), '-', stream_init=10, show=False,
                 head_marker='o', **{name: value})
    message = next(str(w.message) for w in caught
                   if IGNORED in str(w.message))
    ignored, honored = message.split('Parameters honored for streams:')
    assert 'head_marker' in ignored and name not in ignored
    assert name in honored


@pytest.mark.parametrize('name,value', [
    ('head_marker', 'o'), ('tail_linewidth', 4), ('trail_alpha', 0.5)])
def test_the_animation_only_arguments_still_warn(name, value):
    with pytest.warns(UserWarning, match=IGNORED) as caught:
        hyp.plot(iter(_points(30, 3)), '-', stream_init=10, show=False,
                 legend=['a'], **{name: value})
    message = next(str(w.message) for w in caught
                   if IGNORED in str(w.message))
    assert name in message.split('Parameters honored for streams:')[0]


def test_a_per_frame_caption_raises_before_a_sample_is_consumed():
    """(A '{index}' pattern is refused earlier, as on any data without
    a row index: a stream has none.)"""
    def caption(ctx):
        return 'frame'

    consumed = []

    def source():
        for row in _points(30, 3):
            consumed.append(1)
            yield row

    with pytest.raises(TypeError, match='caption= on a streaming plot'):
        hyp.plot(source(), '-', stream_init=10, caption=caption, show=False)
    assert not consumed


@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('kwargs,message', [
    ({'legend': ['a', 'b']}, 'legend= was given as a list of length 2'),
    ({'legend': ['a'], 'legend_colors': ['r', 'b']},
     'legend_colors has 2 entries but the legend has 1'),
    ({'legend': True, 'legend_colors': ['r', 'b']},
     'legend_colors has 2 entries but the legend has 1'),
])
def test_a_wrong_entry_count_raises_before_a_sample_is_consumed(
        ndims, kwargs, message):
    """A stream is one dataset, so a legend or a color list of another
    length is refused with the message of the non-stream path, and the
    stream is left whole for the corrected call."""
    consumed = []

    def source():
        for row in _points(30, ndims):
            consumed.append(1)
            yield row

    stream = source()
    with pytest.raises(ValueError, match=message):
        hyp.plot(stream, '-', stream_init=10, ndims=ndims, show=False,
                 **kwargs)
    assert not consumed
    fig = hyp.plot(stream, '-', stream_init=10, stream_chunk=5, ndims=ndims,
                   legend=['a'], legend_colors=['r'], show=False)
    assert len(consumed) == 30
    assert len(_vertices_px(fig.axes[0])) == 30


def test_label_color_pairs_define_the_stream_legend_outright():
    fig = hyp.plot(iter(_points(30, 3)), '-', stream_init=10, stream_chunk=5,
                   legend=['a'], legend_colors=[('x', 'r'), ('y', 'b')],
                   show=False)
    assert [t.get_text() for t in _legend(fig.axes[0]).get_texts()] \
        == ['x', 'y']


@pytest.mark.parametrize('name', ['caption', 'suptitle'])
def test_caption_and_suptitle_refuse_a_callers_axes_as_on_any_plot(name):
    """They resize the figure to make their room, so the non-stream path
    refuses them with `ax=`; a stream does the same, with the same
    message, before a sample is consumed."""
    consumed = []

    def source():
        for row in _points(30, 2):
            consumed.append(1)
            yield row

    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=f'{name}= is drawn') as stream:
        hyp.plot(source(), '-', stream_init=10, ndims=2, ax=ax, show=False,
                 **{name: 'text'})
    assert not consumed
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=f'{name}= is drawn') as static:
        hyp.plot(_points(30, 2), '-', ax=ax, show=False, **{name: 'text'})
    assert str(stream.value) == str(static.value)


def test_a_legend_goes_into_the_axes_the_caller_supplied():
    canvas, ax = plt.subplots()
    fig = hyp.plot(iter(_points(30, 2)), '-o', stream_init=10,
                   stream_chunk=5, ndims=2, ax=ax, legend=['the stream'],
                   show=False)
    assert fig is canvas
    assert [t.get_text() for t in _legend(ax).get_texts()] == ['the stream']


@pytest.mark.parametrize('ndims', [2, 3])
def test_the_saved_frames_are_the_same_run_to_run(ndims, tmp_path):
    saved = []
    for run in ('a', 'b'):
        plt.close('all')
        path = tmp_path / f'{run}.gif'
        hyp.plot(iter(_points(40, ndims, seed=4)), '-o', stream_init=10,
                 stream_chunk=2, ndims=ndims, legend=['the stream'],
                 caption=CAPTION, suptitle=SUPTITLE, title='Twice the same',
                 labels=lambda i, s: f'p{i}' if i % 4 == 0 else None,
                 save_path=str(path), show=False)
        saved.append(path.read_bytes())
    assert saved[0] == saved[1] and len(saved[0]) > 1000
