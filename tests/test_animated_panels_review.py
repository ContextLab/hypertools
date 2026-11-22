"""Review round 1 of animated panel grids and `caption=` (GH #295).

Every test here reproduces a defect two reviewers and the full suite found
in the first cut of the feature, on real figures and real frames rendered
headless (nothing is mocked or patched): `animate={...}` dictionaries in a
grid, the contexts a caption callable receives, figures left registered
with pyplot after a failed grid call, a grid caption's `{index}` against
its panels' titles, captions wider than the figure, an empty caption,
per-frame panel titles clipped off the canvas, and a warning only the grid
raised.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.plot import CAPTION_GID


def _walk(seed, rows=20, cols=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


X = _walk(0)
DATA = [_walk(seed) for seed in range(4)]
DATES = pd.date_range('2024-01-01', periods=20, freq='D')
FRAMES = [pd.DataFrame(d, index=DATES) for d in DATA]
ANIM = dict(animate=True, duration=1, frame_rate=4, show=False)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def _caption(fig):
    found = fig.findobj(lambda a: a.get_gid() == CAPTION_GID)
    assert len(found) == 1
    return found[0]


def _extent(fig, artist):
    """`artist`'s drawn box as figure fractions ``(x0, y0, x1, y1)``."""
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    width, height = fig.canvas.get_width_height()
    return (box.x0 / width, box.y0 / height, box.x1 / width,
            box.y1 / height)


# --------------------------------------------------------------------------
# C1 / S1: animate={...} in a grid
# --------------------------------------------------------------------------

def test_animate_dict_per_dataset_values_are_split_per_panel():
    """A per-dataset list inside `animate={...}` is narrowed per panel,
    exactly as the same list passed as the flat kwarg is."""
    in_dict = hyp.plot(
        [X, X], panels=True, duration=1, frame_rate=4, show=False,
        animate={'style': True, 'chemtrails': [True, False]})
    flat = hyp.plot([X, X], panels=True, chemtrails=[True, False], **ANIM)
    assert in_dict.n_frames == flat.n_frames == 4
    in_dict.draw_frame(3)
    flat.draw_frame(3)
    for ax_dict, ax_flat in zip(in_dict.figure.axes, flat.figure.axes):
        assert len(ax_dict.lines) == len(ax_flat.lines)
    # the first panel draws a trail the second does not
    counts = [len(ax.lines) for ax in in_dict.figure.axes[:2]]
    assert counts[0] > counts[1]


def test_animate_dict_with_a_false_style_is_a_static_grid():
    fig = hyp.plot([X, X], panels=True, animate={'style': False},
                   show=False)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_animate_dict_settings_reach_every_panel():
    anim = hyp.plot([X, X], panels=True, show=False,
                    animate={'style': 'spin', 'duration': 1,
                             'frame_rate': 6})
    assert anim.n_frames == 6


def test_animate_dict_conflicting_with_a_flat_kwarg_raises_in_a_grid():
    with pytest.raises(ValueError, match='only one place'):
        hyp.plot([X, X], panels=True, show=False, duration=3,
                 animate={'style': True, 'duration': 1})


# --------------------------------------------------------------------------
# C2: the context a caption callable receives
# --------------------------------------------------------------------------

def test_animated_caption_callable_gets_the_real_first_frame_context():
    """The context a caption is first resolved with (at construction) is
    the one frame 0 delivers to an `on_frame=` callback."""
    seen, delivered = [], []

    def caption(ctx):
        seen.append(ctx)
        return f'{ctx.window_bounds[0]} {len(ctx.artists)}'

    anim = hyp.plot(X, caption=caption, on_frame=delivered.append, **ANIM)
    first = seen[0]
    assert first.frame == 0 and first.n_frames == anim.n_frames
    assert first.window_bounds is not None and len(first.artists) >= 1
    assert first.figure is anim.figure
    delivered.clear()
    anim.draw_frame(0)
    assert first.window_bounds == delivered[0].window_bounds
    assert first.revealed_counts == delivered[0].revealed_counts
    assert len(first.artists) == len(delivered[0].artists)
    assert first.style == delivered[0].style


def test_animated_3d_title_callable_gets_the_real_first_frame_context():
    seen = []

    def title(ctx):
        seen.append(ctx)
        return f'{ctx.window_bounds[0][1]}'

    hyp.plot(X, title=title, **ANIM)
    assert seen[0].frame == 0
    assert seen[0].window_bounds is not None and seen[0].artists


def test_static_grid_caption_callable_gets_the_finished_grid():
    seen = []

    def caption(ctx):
        seen.append(ctx)
        return f'{len(ctx.datasets[0])} rows'

    fig = hyp.plot(DATA, panels=(2, 2), caption=caption, show=False)
    assert _caption(fig).get_text() == '20 rows'
    ctx = seen[0]
    assert ctx.frame is None and ctx.n_frames is None
    assert ctx.progress == 1.0
    assert ctx.figure is fig
    assert list(ctx.axes) == fig.axes[:4]
    assert len(ctx.datasets) == 4
    assert all(d.shape == (20, 3) for d in ctx.datasets)
    assert ctx.revealed_counts == (20, 20, 20, 20)
    assert ctx.window_bounds == ((0, 20),) * 4
    assert len(ctx.panels) == 4
    for panel, ax in zip(ctx.panels, fig.axes):
        assert panel.axes is ax
        assert len(panel.datasets) == 1 and panel.revealed_counts == (20,)
        assert panel.artists and all(
            artist.axes is ax for artist in panel.artists)
    assert len(ctx.artists) == sum(len(p.artists) for p in ctx.panels)


def test_static_single_plot_caption_callable_gets_the_drawn_artists():
    seen = []
    fig = hyp.plot(X, show=False,
                   caption=lambda ctx: seen.append(ctx) or 'done')
    assert seen[0].figure is fig and seen[0].axes is fig.axes[0]
    assert seen[0].artists
    assert all(a.axes is fig.axes[0] for a in seen[0].artists)


def test_animated_grid_caption_first_context_is_frame_zero_of_the_grid():
    seen = []

    def caption(ctx):
        seen.append(ctx)
        return str(ctx.frame)

    anim = hyp.plot(DATA, panels=(1, 4), caption=caption, **ANIM)
    assert _caption(anim.figure).get_text() == '0'
    assert all(len(ctx.panels) == 4 and len(ctx.datasets) == 4
               and ctx.artists for ctx in seen)


# --------------------------------------------------------------------------
# C3: a failed grid call leaves no figure behind
# --------------------------------------------------------------------------

def _raises(ctx):
    raise RuntimeError('caption failed')


FAILING_GRID_CALLS = {
    'morph, one dataset per panel': dict(animate='morph'),
    'caption returns a number': dict(animate=True, caption=lambda ctx: 3),
    'caption raises': dict(animate=True, caption=_raises),
    'static caption raises': dict(caption=_raises),
    'static caption returns a number': dict(caption=lambda ctx: 3),
    'unknown forecaster, animated': dict(animate=True, predict='Nope'),
    'unknown forecaster, static': dict(predict='Nope'),
    'morph with a forecast': dict(animate='morph', predict='Kalman'),
    'title raises, animated': dict(animate=True, title=_raises),
    'title raises, static': dict(title=_raises),
    'on_frame not callable': dict(animate=True, on_frame=3),
    'unknown style': dict(animate='nope'),
    'unknown reducer': dict(reduce='NoSuchReducer'),
    'bad format string': dict(fmt='not-a-format'),
    'per-dataset list too long': dict(animate=True,
                                      chemtrails=[True, False, True]),
    'bad save extension, animated': dict(animate=True,
                                         save_path='grid.notaformat'),
    'pattern without an index': dict(animate=True, caption='{index}'),
    'pattern title without an index': dict(title='{index}'),
    'bad panel_fit': dict(panel_fit='nope'),
    'companion with a grid': dict(animate=True, companion=np.arange(20.0)),
    'explore with a grid': dict(explore=True),
    'legend of the wrong length': dict(legend=['a', 'b', 'c']),
    'hue of the wrong length': dict(hue=[0, 1, 2]),
    'animated plotly grid': dict(animate=True, backend='plotly'),
    'plotly caption': dict(caption='x', backend='plotly'),
}


@pytest.mark.parametrize('show', [True, False], ids=['show', 'no-show'])
@pytest.mark.parametrize('name', sorted(FAILING_GRID_CALLS))
def test_failed_grid_call_leaves_no_figure_registered(name, show, tmp_path,
                                                      monkeypatch):
    """After ANY exception from a `panels=` call, pyplot's registered
    figures are what they were before the call."""
    monkeypatch.chdir(tmp_path)
    kwargs = dict(FAILING_GRID_CALLS[name])
    if kwargs.get('animate'):
        kwargs.setdefault('duration', 1)
        kwargs.setdefault('frame_rate', 4)
    keep = plt.figure()
    before = plt.get_fignums()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(Exception):
            hyp.plot([X, X], panels=True, show=show, **kwargs)
    assert plt.get_fignums() == before
    assert plt.fignum_exists(keep.number)


def test_failed_grid_call_with_axis_labels_too_big_for_size_leaks_nothing():
    frames = [pd.DataFrame(d, columns=list('xyz')) for d in DATA]
    before = plt.get_fignums()
    with pytest.raises(ValueError, match='size='):
        hyp.plot(frames, panels=(1, 4), size=[4, 1], **ANIM)
    assert plt.get_fignums() == before


def test_successful_grid_call_with_show_keeps_its_figure_registered():
    before = plt.get_fignums()
    fig = hyp.plot([X, X], panels=True)
    assert plt.get_fignums() == before + [fig.number]


# --------------------------------------------------------------------------
# C4: a grid caption's {index} is the panels' reveal head
# --------------------------------------------------------------------------

@pytest.mark.parametrize('style', [
    dict(animate=True),
    dict(animate=True, order='serial'),
    dict(animate='serial'),
    dict(animate='spin'),
    dict(animate='window'),
    dict(animate='parallel', chemtrails=True),
], ids=['parallel', 'order-serial', 'serial', 'spin', 'window', 'trails'])
def test_grid_caption_index_matches_the_panel_titles(style):
    anim = hyp.plot(FRAMES[:2], 'o', reduce=None, panels=True,
                    duration=1, frame_rate=6, show=False,
                    title='{index:%Y-%m-%d}', caption='{index:%Y-%m-%d}',
                    **style)
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        titles = {ax.get_title() for ax in anim.figure.axes[:2]}
        assert titles == {_caption(anim.figure).get_text()}, frame


def test_grid_caption_index_follows_the_first_indexed_panel():
    """Panels whose indexes differ: the caption reads the first dataset
    that carries an index, at the reveal head of the panel drawing it."""
    plain = DATA[0]
    short = pd.DataFrame(DATA[1][:10],
                         index=pd.date_range('1999-06-01', periods=10))
    long = FRAMES[2]
    anim = hyp.plot([plain, short, long], reduce=None, panels=(1, 3),
                    duration=1, frame_rate=6, show=False, animate=True,
                    title=['plain', '{index:%Y-%m-%d}', '{index:%Y-%m-%d}'],
                    caption='{index:%Y-%m-%d}')
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        text = _caption(anim.figure).get_text()
        assert text.startswith('1999-06')
        assert text == anim.figure.axes[1].get_title(), frame


def test_reducer_grid_caption_index_reads_the_first_panel():
    anim = hyp.plot(FRAMES[:2], reduce=['PCA', 'IncrementalPCA'], ndims=2,
                    panels=True, duration=1, frame_rate=6, show=False,
                    animate=True, order='serial', caption='{index:%m-%d}',
                    title=['{index:%m-%d}', 'second'])
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        assert (_caption(anim.figure).get_text()
                == anim.figure.axes[0].get_title())


def test_static_grid_caption_index_is_the_last_row():
    fig = hyp.plot(FRAMES[:2], reduce=None, panels=True, show=False,
                   caption='{index:%Y-%m-%d}')
    assert _caption(fig).get_text() == '2024-01-20'


# --------------------------------------------------------------------------
# C5: a caption wider than the figure is wrapped
# --------------------------------------------------------------------------

SENTENCE = ('A caption sentence that is far too long for one line of a '
            'small figure, so that it has to be wrapped to the width of '
            'the figure it is drawn under, and stay there. ')


def _assert_caption_on_canvas(fig):
    x0, y0, x1, y1 = _extent(fig, _caption(fig))
    assert 0.0 <= x0 < x1 <= 1.0, (x0, x1)
    assert 0.0 <= y0 < y1 <= 1.0, (y0, y1)


def _assert_caption_below_axes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = _caption(fig).get_window_extent(renderer).y1
    for ax in fig.axes:
        if ax.get_visible():
            assert ax.get_window_extent(renderer).y0 >= top - 1.0


@pytest.mark.parametrize('text', ['W' * 200, SENTENCE * 2,
                                  'first line\n' + SENTENCE],
                         ids=['one-word', 'sentence', 'own-breaks'])
@pytest.mark.parametrize('call', [
    dict(x=X), dict(x=X, ndims=2), dict(x=DATA, panels=(2, 2)),
    dict(x=DATA, panels=(1, 4), legend=True),
    dict(x=X, **ANIM), dict(x=DATA, panels=(2, 2), **ANIM),
], ids=['static-3d', 'static-2d', 'static-grid', 'static-grid-legend',
        'animated', 'animated-grid'])
def test_long_caption_is_wrapped_onto_the_canvas(call, text):
    call = dict(call, show=False)
    result = hyp.plot(call.pop('x'), caption=text, **call)
    fig = getattr(result, 'figure', result)
    caption = _caption(fig)
    assert '\n' in caption.get_text()
    # wrapping only adds line breaks
    assert (caption.get_text().replace('\n', '').replace(' ', '')
            == text.replace('\n', '').replace(' ', ''))
    _assert_caption_on_canvas(fig)
    _assert_caption_below_axes(fig)


def test_short_caption_is_not_wrapped():
    fig = hyp.plot(X, caption='a short caption', show=False)
    assert _caption(fig).get_text() == 'a short caption'


@pytest.mark.parametrize('grid', [False, True], ids=['single', 'grid'])
def test_caption_that_grows_over_the_frames_stays_on_the_canvas(grid):
    """A per-frame caption longer in later frames is wrapped per frame,
    and its room is sized for the first, middle and last frames."""
    def caption(ctx):
        return SENTENCE * (1 + ctx.frame)

    data = DATA if grid else X
    extra = dict(panels=(2, 2)) if grid else {}
    anim = hyp.plot(data, caption=caption, duration=1, frame_rate=5,
                    animate=True, show=False, **extra)
    lines = []
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        lines.append(_caption(anim.figure).get_text().count('\n') + 1)
        x0, y0, x1, y1 = _extent(anim.figure, _caption(anim.figure))
        assert 0.0 <= x0 < x1 <= 1.0, frame
        assert 0.0 <= y0 < y1 <= 1.0, frame
        _assert_caption_below_axes(anim.figure)
    assert lines == sorted(lines) and lines[-1] > lines[0]


def test_wrapped_caption_room_grows_with_its_lines():
    one = hyp.plot(X, caption='short', show=False, **{
        k: v for k, v in ANIM.items() if k != 'show'})
    many = hyp.plot(X, caption=SENTENCE * 3, show=False, **{
        k: v for k, v in ANIM.items() if k != 'show'})
    assert (many.figure.get_size_inches()[1]
            > one.figure.get_size_inches()[1] + 0.3)
    assert many.figure.get_size_inches()[0] == pytest.approx(
        one.figure.get_size_inches()[0])


def test_mathtext_caption_is_left_whole():
    text = r'$\alpha + \beta = \gamma$ and so on'
    fig = hyp.plot(X, caption=text, show=False)
    assert _caption(fig).get_text() == text


# --------------------------------------------------------------------------
# T3: an empty caption adds nothing
# --------------------------------------------------------------------------

@pytest.mark.parametrize('call', [
    dict(x=X), dict(x=X, **ANIM), dict(x=DATA, panels=(1, 4)),
    dict(x=DATA, panels=(1, 4), **ANIM),
], ids=['static', 'animated', 'static-grid', 'animated-grid'])
@pytest.mark.parametrize('empty', ['', '   '], ids=['empty', 'blank'])
def test_empty_caption_adds_nothing(call, empty):
    call = dict(call, show=False)
    data = call.pop('x')
    with_caption = hyp.plot(data, caption=empty, **call)
    without = hyp.plot(data, **call)
    fig_a = getattr(with_caption, 'figure', with_caption)
    fig_b = getattr(without, 'figure', without)
    assert not fig_a.findobj(lambda a: a.get_gid() == CAPTION_GID)
    assert tuple(fig_a.get_size_inches()) == tuple(fig_b.get_size_inches())


# --------------------------------------------------------------------------
# T1: per-frame titles of an animated grid stay on the canvas
# --------------------------------------------------------------------------

def _per_frame_titles(kind, n):
    if kind == 'callable':
        return lambda ctx: f'frame {ctx.frame}'
    if kind == 'pattern':
        return '{index:%Y-%m-%d}'
    return [(lambda ctx, i=i: f'panel {i} frame {ctx.frame}')
            for i in range(n)]


@pytest.mark.parametrize('extras', [
    {}, dict(legend=True, caption='a caption'),
    dict(hue=[np.arange(20)] * 4, colorbar=True),
], ids=['bare', 'legend-caption', 'colorbar'])
@pytest.mark.parametrize('kind', ['callable', 'pattern', 'list'])
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape', [(1, 4), (2, 2), (3, 1)])
def test_per_frame_titles_stay_inside_an_animated_grid(shape, ndims, kind,
                                                       extras):
    n = shape[0] * shape[1]
    extras = dict(extras)
    if 'hue' in extras:
        extras['hue'] = extras['hue'][:n]
    anim = hyp.plot(FRAMES[:n], reduce=None if ndims == 3 else 'PCA',
                    ndims=ndims, panels=shape, title=_per_frame_titles(kind, n),
                    duration=1, frame_rate=3, animate=True, show=False,
                    **extras)
    fig = anim.figure
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        panels = fig.axes[:n]
        boxes = [ax.get_window_extent(renderer) for ax in panels]
        for i, ax in enumerate(panels):
            assert ax.get_title(), (frame, i)
            title = ax.title.get_window_extent(renderer)
            # on the canvas
            assert title.y1 <= height + 0.5 and title.y0 >= 0, (frame, i)
            assert title.x0 >= -0.5 and title.x1 <= width + 0.5, (frame, i)
            # ...and clear of every OTHER panel's box and x-axis label
            for j, other in enumerate(panels):
                if j == i:
                    continue
                assert not title.overlaps(boxes[j]), (frame, i, j)
                label = other.xaxis.label
                if label.get_text():
                    assert not title.overlaps(
                        label.get_window_extent(renderer)), (frame, i, j)


# --------------------------------------------------------------------------
# T4: a grid warns only where the single-axes call does
# --------------------------------------------------------------------------

TWO_COLUMNS = [d[:, :2] for d in DATA]


@pytest.mark.parametrize('extra', [
    dict(panels=(2, 2)), dict(panels=(2, 2), panel_fit='independent'),
    dict(panels=(1, 4), animate=True, duration=1, frame_rate=4),
    dict(panels=True, reduce=['PCA', 'IncrementalPCA']),
    dict(return_model=True), dict(n_clusters=2), dict(manip='ZScore'),
], ids=['grid', 'independent', 'animated', 'reducers', 'return_model',
        'clusters', 'manip'])
def test_two_column_data_draws_without_an_ndims_warning(extra):
    """The single-axes call of 2-column data is silent about `ndims`,
    which the caller never passed: so is every other form of the call."""
    with warnings.catch_warnings(record=True) as single:
        warnings.simplefilter('always')
        hyp.plot(TWO_COLUMNS, show=False)
    assert not [w for w in single if 'ndims' in str(w.message)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hyp.plot(TWO_COLUMNS, show=False, **extra)
    assert not [str(w.message) for w in caught
                if 'ndims' in str(w.message)]


def test_an_explicit_reduce_to_too_many_dimensions_still_warns():
    """The public `hyp.reduce`/`hyp.analyze` warning is untouched."""
    with pytest.warns(UserWarning, match='no reduction was performed'):
        hyp.reduce(TWO_COLUMNS, reduce='PCA', ndims=3)
    with pytest.warns(UserWarning, match='no reduction was performed'):
        hyp.analyze(TWO_COLUMNS, reduce='PCA', ndims=3, return_model=True)


# --------------------------------------------------------------------------
# Round 2 (Codex 9): a per-frame title whose HEIGHT changes over the frames
# --------------------------------------------------------------------------

TALL = '\n'.join(['a title line'] * 8)
GROWING = {
    # the frame on which the title is eight lines tall (one line elsewhere)
    'first': lambda ctx: TALL if ctx.frame == 0 else 'short',
    'middle': lambda ctx: (TALL if ctx.frame == ctx.n_frames // 2
                           else 'short'),
    'last': lambda ctx: (TALL if ctx.frame == ctx.n_frames - 1
                         else 'short'),
    'after-first': lambda ctx: 'short' if ctx.frame == 0 else TALL,
}


def _assert_titles_on_canvas(anim, n):
    """On every frame, every one of the first `n` axes' titles is inside
    the canvas and clear of every other panel's box; returns the most
    lines a title had."""
    fig = anim.figure
    most = 0
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        panels = fig.axes[:n]
        boxes = [ax.get_window_extent(renderer) for ax in panels]
        for i, ax in enumerate(panels):
            assert ax.get_title(), (frame, i)
            most = max(most, ax.get_title().count('\n') + 1)
            title = ax.title.get_window_extent(renderer)
            assert title.y1 <= height + 0.5 and title.y0 >= 0, (frame, i)
            assert title.x0 >= -0.5 and title.x1 <= width + 0.5, (frame, i)
            for j in range(len(panels)):
                if j != i:
                    assert not title.overlaps(boxes[j]), (frame, i, j)
    return most


@pytest.mark.parametrize('when', sorted(GROWING))
@pytest.mark.parametrize('ndims', [2, 3])
@pytest.mark.parametrize('shape', [(1, 2), (2, 2), (3, 1)])
def test_a_grid_title_that_grows_after_frame_0_stays_on_the_canvas(
        shape, ndims, when):
    """The room was sized for frame 0's title alone: one that became
    eight lines on a later frame ran 114 px off the top of a 320 px
    canvas. It is sized for the tallest of the first, middle and last
    frames' titles, as a caption's room is."""
    n = shape[0] * shape[1]
    anim = hyp.plot([d[:, :ndims] for d in DATA[:n]], reduce=None,
                    ndims=ndims, panels=shape, title=GROWING[when],
                    backend='matplotlib', **ANIM)
    # a still of the figure shows frame 0's title
    expected = TALL if when == 'first' else 'short'
    assert [ax.get_title() for ax in anim.figure.axes[:n]] == [expected] * n
    assert _assert_titles_on_canvas(anim, n) == 8


@pytest.mark.parametrize('when', sorted(GROWING))
@pytest.mark.parametrize('ndims', [2, 3])
def test_a_single_axes_title_that_grows_after_frame_0_stays_on_the_canvas(
        ndims, when):
    """The same callable on one axes: a 2-D animation reserved no title
    room at all beyond the default margin, and a 3-D one frame 0's."""
    anim = hyp.plot([d[:, :ndims] for d in DATA[:2]], reduce=None,
                    ndims=ndims, title=GROWING[when], backend='matplotlib',
                    **ANIM)
    assert _assert_titles_on_canvas(anim, 1) == 8


def test_a_tall_fixed_title_of_a_2d_animation_stays_on_the_canvas():
    """Eight fixed lines above an animated 2-D plot left the canvas too
    (the default margin holds about three); the figure grows upward by
    what is missing and the axes keeps its size."""
    plain = hyp.plot(DATA[0][:, :2], reduce=None, ndims=2, title='one line',
                     **ANIM)
    tall = hyp.plot(DATA[0][:, :2], reduce=None, ndims=2, title=TALL, **ANIM)
    # a title that fits changes nothing
    assert tuple(plain.figure.get_size_inches()) == (6.4, 4.8)
    assert _assert_titles_on_canvas(tall, 1) == 8
    plain_w, plain_h = plain.figure.get_size_inches()
    tall_w, tall_h = tall.figure.get_size_inches()
    assert tall_w == plain_w and tall_h > plain_h
    a = plain.figure.axes[0].get_position()
    b = tall.figure.axes[0].get_position()
    assert np.isclose(a.height * plain_h, b.height * tall_h)
    assert np.isclose(a.y0 * plain_h, b.y0 * tall_h)


def test_a_tall_per_segment_title_of_a_2d_animation_stays_on_the_canvas():
    anim = hyp.plot([d[:, :2] for d in DATA[:2]], reduce=None, ndims=2,
                    order='serial', title=['short', TALL], **ANIM)
    assert _assert_titles_on_canvas(anim, 1) == 8
