"""`caption=`: a figure-level caption, updated every frame (GH #295).

One line of text centered under the figure -- a single plot or a whole
`panels=` grid -- given as a plain string, a ``'{index...}'`` pattern, or a
callable taking the frame's `FrameContext` (the forms a per-frame `title=`
takes).

Real figures and real frames, rendered headless: caption text is read back
off the returned figure after `HyperAnimation.draw_frame`, and positions
are measured from a real draw. Nothing is mocked, and no assertion depends
on a font's absolute metrics.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.animation_context import FrameContext
from hypertools.plot.plot import CAPTION_GID
from hypertools.plot.plot import subplots as hyp_subplots


def _walk(seed, rows=40, cols=5):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(rows, cols)), axis=0)


DATA = [_walk(seed) for seed in range(4)]
DATES = pd.date_range('2024-01-01', periods=40, freq='D')
FRAMES = [pd.DataFrame(d, index=DATES) for d in DATA]


def _counter(ctx):
    return f'timepoint {ctx.frame + 1} of {ctx.n_frames}'


def _single(data=None, **kwargs):
    call = dict(animate=True, duration=1, frame_rate=8, show=False)
    call.update(kwargs)
    return hyp.plot(DATA[0] if data is None else data, **call)


def _grid(data=None, **kwargs):
    call = dict(panels=(1, 4), animate=True, duration=1, frame_rate=8,
                show=False)
    call.update(kwargs)
    return hyp.plot(DATA if data is None else data, **call)


def _caption(fig):
    """The figure's caption Text, found the documented way (by gid)."""
    found = fig.findobj(lambda a: a.get_gid() == CAPTION_GID)
    assert len(found) == 1, f'{len(found)} captions on the figure'
    return found[0]


def _text_at(anim, frame):
    anim.draw_frame(frame)
    return _caption(anim.figure).get_text()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


#: the three places an animated caption is drawn
_MAKERS = {
    'single-3d': lambda **kw: _single(**kw),
    'single-2d': lambda **kw: _single(ndims=2, **kw),
    'grid': lambda **kw: _grid(**kw),
}


# --- text per frame -----------------------------------------------------

@pytest.mark.parametrize('where', sorted(_MAKERS))
def test_a_plain_string_is_drawn_on_every_frame(where):
    anim = _MAKERS[where](caption='a fixed caption')
    assert [_text_at(anim, f) for f in (0, 3, anim.n_frames - 1)] == (
        ['a fixed caption'] * 3)


@pytest.mark.parametrize('where', sorted(_MAKERS))
def test_a_callable_is_resolved_for_every_frame(where):
    anim = _MAKERS[where](caption=_counter)
    assert _text_at(anim, 0) == 'timepoint 1 of 8'
    assert _text_at(anim, 4) == 'timepoint 5 of 8'
    assert _text_at(anim, 7) == 'timepoint 8 of 8'
    # frames are order-independent: going back redraws the earlier text
    assert _text_at(anim, 2) == 'timepoint 3 of 8'


def test_a_still_of_a_fresh_animation_already_shows_frame_zero():
    anim = _single(caption=_counter)
    assert _caption(anim.figure).get_text() == 'timepoint 1 of 8'
    anim = _grid(caption=_counter)
    assert _caption(anim.figure).get_text() == 'timepoint 1 of 8'


@pytest.mark.parametrize('maker, data', [
    (_single, FRAMES[0]),
    (_grid, FRAMES),
], ids=['single', 'grid'])
def test_an_index_pattern_reads_the_row_under_the_reveal_head(maker, data):
    anim = maker(data, caption='{index:%Y-%m-%d}')
    assert _text_at(anim, 0) == '2024-01-01'
    assert _text_at(anim, anim.n_frames - 1) == '2024-02-09'
    middle = pd.Timestamp(_text_at(anim, anim.n_frames // 2))
    assert DATES[0] < middle < DATES[-1]


def test_a_pattern_caption_matches_the_same_pattern_title():
    anim = _single(FRAMES[0], ndims=2, title='{index:%d %b}',
                   caption='{index:%d %b}')
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        assert (_caption(anim.figure).get_text()
                == anim.figure.axes[0].get_title())


def test_a_grid_callable_gets_the_figure_level_context():
    seen = []

    def caption(ctx):
        seen.append(ctx)
        return f'{len(ctx.panels)} panels, frame {ctx.frame}'

    anim = _grid(caption=caption)
    assert _text_at(anim, 5) == '4 panels, frame 5'
    ctx = seen[-1]
    assert isinstance(ctx, FrameContext)
    assert isinstance(ctx.axes, tuple) and len(ctx.axes) == 4
    assert ctx.figure is anim.figure


def test_a_single_axes_callable_gets_the_ordinary_context():
    seen = []

    def caption(ctx):
        seen.append(ctx)
        return f'{ctx.progress:.2f}'

    anim = _single(caption=caption)
    assert _text_at(anim, anim.n_frames - 1) == '1.00'
    assert seen[-1].axes is anim.figure.axes[0]
    assert seen[-1].revealed_counts is not None
    assert seen[-1].panels == ()


def test_the_caption_is_updated_before_on_frame_runs():
    """A user callback sees the CURRENT frame's caption, as it sees the
    current frame's artists."""
    seen = []
    anim = _single(caption=_counter)
    anim.on_frame(lambda ctx: seen.append(
        _caption(ctx.figure).get_text()))
    anim.draw_frame(6)
    assert seen == ['timepoint 7 of 8']


# --- placement ----------------------------------------------------------

def _boxes(fig):
    """(caption box, [box of everything else drawn]) after a real draw."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    caption = _caption(fig).get_window_extent(renderer)
    others = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        others.append(ax.get_window_extent(renderer))
        if ax.get_title():
            others.append(ax.title.get_window_extent(renderer))
        if ax.get_legend() is not None:
            others.append(ax.get_legend().get_window_extent(renderer))
    return caption, others


_PLACEMENTS = {
    'single-3d': lambda: _single(caption=_counter, title='a title',
                                 legend=['a']),
    'single-2d': lambda: _single(ndims=2, caption=_counter, title='a title',
                                 legend=['a']),
    'single-colorbar': lambda: _single(
        caption=_counter, hue=np.arange(40.0), colorbar=True),
    'single-companion': lambda: _single(
        caption=_counter, companion={'data': np.arange(40.0)}),
    'grid-row': lambda: _grid(caption=_counter, title=list('abcd'),
                              legend=list('abcd')),
    'grid-2x2': lambda: _grid(panels=(2, 2), caption=_counter,
                              title=list('abcd')),
    'grid-2d': lambda: _grid(ndims=2, panels=(2, 2), caption=_counter,
                             title=list('abcd')),
    'grid-colorbar': lambda: _grid(
        DATA[:3], panels=(1, 3), caption=_counter, colorbar=True,
        hue=[np.arange(40.0)] * 3),
    'grid-sized': lambda: _grid(caption=_counter, size=(11, 3)),
    'grid-two-lines': lambda: _grid(caption='first line\nsecond line',
                                    title=list('abcd')),
    'static': lambda: hyp.plot(DATA[0], caption='static', legend=['a'],
                               title='t', show=False),
    'static-grid': lambda: hyp.plot(DATA, panels=(2, 2), caption='static',
                                    legend=list('abcd'), show=False),
}


@pytest.mark.parametrize('case', sorted(_PLACEMENTS))
def test_the_caption_is_inside_the_canvas_and_clear_of_everything(case):
    made = _PLACEMENTS[case]()
    fig = made.figure if hasattr(made, 'draw_frame') else made
    frames = ([0, made.n_frames // 2, made.n_frames - 1]
              if hasattr(made, 'draw_frame') else [None])
    for frame in frames:
        if frame is not None:
            made.draw_frame(frame)
        caption, others = _boxes(fig)
        canvas = fig.bbox
        assert caption.width > 0 and caption.height > 0
        assert caption.x0 >= canvas.x0 and caption.x1 <= canvas.x1, case
        assert caption.y0 >= canvas.y0 and caption.y1 <= canvas.y1, case
        for other in others:
            assert not caption.overlaps(other), (case, frame)
            # under everything, not merely beside it
            assert caption.y1 <= other.y0 + 0.5, (case, frame)
        # centered on the canvas
        middle = 0.5 * (caption.x0 + caption.x1)
        assert middle == pytest.approx(0.5 * canvas.width, abs=1.0)


def test_the_caption_stays_clear_of_a_rotating_cube_on_every_frame():
    import itertools
    from mpl_toolkits.mplot3d import proj3d
    anim = _grid(caption=_counter, duration=3, frame_rate=10, elev=40)
    fig = anim.figure
    axes = [ax for ax in fig.axes if ax.get_visible()]
    for frame in range(anim.n_frames):
        anim.draw_frame(frame)
        fig.canvas.draw()
        top = _caption(fig).get_window_extent(fig.canvas.get_renderer()).y1
        for ax in axes:
            corners = np.array(list(itertools.product(
                ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))
            x, y, _ = proj3d.proj_transform(
                corners[:, 0], corners[:, 1], corners[:, 2], ax.get_proj())
            lowest = ax.transData.transform(np.column_stack([x, y]))[:, 1]
            assert lowest.min() >= top, frame


def test_a_full_canvas_plot_grows_downward_and_the_plot_keeps_its_size():
    """An animated 3-D plot fills its canvas, so the caption's room is
    ADDED below it: the axes keeps its size in inches and its place
    relative to the top."""
    plain = _single()
    captioned = _single(caption='under the plot')
    w0, h0 = plain.figure.get_size_inches()
    w1, h1 = captioned.figure.get_size_inches()
    assert w1 == pytest.approx(w0)
    assert h1 > h0
    box0 = plain.figure.axes[0].get_position()
    box1 = captioned.figure.axes[0].get_position()
    assert box1.width * w1 == pytest.approx(box0.width * w0)
    assert box1.height * h1 == pytest.approx(box0.height * h0)
    # same distance from the top edge
    assert (1 - box1.y1) * h1 == pytest.approx((1 - box0.y1) * h0)


def test_a_plot_with_blank_room_below_is_not_resized():
    """Where the figure already has blank room under the plot (a 2-D
    animation's margin), the caption goes there and nothing moves."""
    plain = _single(ndims=2)
    captioned = _single(ndims=2, caption='fits in the margin')
    assert tuple(captioned.figure.get_size_inches()) == pytest.approx(
        tuple(plain.figure.get_size_inches()))
    assert (captioned.figure.axes[0].get_position().bounds
            == pytest.approx(plain.figure.axes[0].get_position().bounds))


def test_more_lines_get_more_room():
    one = _single(caption='one line')
    three = _single(caption='one\ntwo\nthree')
    assert (three.figure.get_size_inches()[1]
            > one.figure.get_size_inches()[1])
    caption, _ = _boxes(three.figure)
    assert caption.y0 >= 0


def test_no_caption_leaves_the_figure_exactly_as_it_was():
    anim = _single()
    assert not anim.figure.findobj(lambda a: a.get_gid() == CAPTION_GID)
    assert tuple(anim.figure.get_size_inches()) == pytest.approx((6.4, 4.8))
    fig = hyp.plot(DATA, panels=(2, 2), show=False)
    assert not fig.texts


def test_the_caption_uses_the_plots_font():
    anim = _single(caption='styled', title='a title')
    caption = _caption(anim.figure)
    title = anim.figure.axes[0].title
    assert caption.get_fontname() == title.get_fontname()
    # ...and can be restyled through the documented handle
    caption.set_color('gray')
    anim.draw_frame(3)
    assert _caption(anim.figure).get_color() == 'gray'


# --- static plots -------------------------------------------------------

def test_a_static_plot_draws_its_caption_once():
    fig = hyp.plot(DATA[0], caption='a static caption', show=False)
    assert _caption(fig).get_text() == 'a static caption'


def test_a_static_pattern_shows_the_last_row():
    fig = hyp.plot(FRAMES[0], caption='{index:%Y-%m-%d}', show=False)
    assert _caption(fig).get_text() == '2024-02-09'


def test_a_static_callable_gets_the_finished_figures_context():
    seen = []

    def caption(ctx):
        seen.append(ctx)
        return 'done'

    fig = hyp.plot(DATA[0], caption=caption, show=False)
    assert _caption(fig).get_text() == 'done'
    assert len(seen) == 1
    assert seen[0].frame is None and seen[0].n_frames is None
    assert seen[0].progress == 1.0
    assert seen[0].revealed_counts == (len(seen[0].datasets[0]),)


def test_a_static_grid_draws_its_caption_under_every_panel():
    fig = hyp.plot(FRAMES, panels=(1, 4), caption='{index:%Y-%m-%d}',
                   show=False)
    assert _caption(fig).get_text() == '2024-02-09'
    caption, others = _boxes(fig)
    assert all(caption.y1 <= other.y0 + 0.5 for other in others)


def test_a_static_caption_is_saved_with_the_figure(tmp_path):
    """A static 2-D plot fills its canvas down to the tick labels, so the
    caption's room is added below: the saved image is as wide, taller, and
    the caption's own box holds ink."""
    from PIL import Image
    plain, captioned = tmp_path / 'plain.png', tmp_path / 'captioned.png'
    hyp.plot(DATA[0], ndims=2, save_path=str(plain), show=False)
    fig = hyp.plot(DATA[0], ndims=2, save_path=str(captioned), show=False,
                   caption='saved with the figure')
    with Image.open(plain) as a, Image.open(captioned) as b:
        assert a.size[0] == b.size[0]
        assert b.size[1] > a.size[1]
        image = np.asarray(b.convert('L')).astype(int)
    assert image.shape == tuple(
        int(v) for v in (fig.bbox.height, fig.bbox.width))
    fig.canvas.draw()
    box = _caption(fig).get_window_extent(fig.canvas.get_renderer())
    height = image.shape[0]
    rows = slice(int(height - box.y1), int(height - box.y0) + 1)
    cols = slice(int(box.x0), int(box.x1) + 1)
    background = image[-1, 0]
    # the caption's box holds ink, in the bottom fifth of the image
    assert (np.abs(image[rows, cols] - background) > 80).any()
    assert rows.start > 0.8 * height


# --- construction does not start the animation ----------------------------

@pytest.mark.parametrize('where', sorted(_MAKERS))
def test_adding_a_caption_does_not_fire_on_frame(where):
    """Placing the caption measures the figure on a real draw; that draw
    must not run a frame through the user's callback."""
    seen = []
    anim = _MAKERS[where](caption=_counter, on_frame=seen.append,
                          legend=True)
    assert seen == []
    anim.draw_frame(2)
    assert [ctx.frame for ctx in seen] == [2]


def test_a_saved_animation_carries_the_changing_caption(tmp_path):
    from PIL import Image
    out = tmp_path / 'captioned.gif'
    anim = _single(ndims=2, caption=_counter, save_path=str(out),
                   duration=1, frame_rate=6)
    with Image.open(out) as im:
        assert im.n_frames == anim.n_frames == 6
        frames = []
        for k in (0, 5):
            im.seek(k)
            frames.append(np.asarray(im.convert('L')).astype(int))
    fig = anim.figure
    fig.canvas.draw()
    box = _caption(fig).get_window_extent(fig.canvas.get_renderer())
    height = frames[0].shape[0]
    rows = slice(int(height - box.y1) - 2, int(height - box.y0) + 2)
    cols = slice(int(box.x0) - 2, int(box.x1) + 2)
    # the caption's own pixels differ between the first and last frame
    assert (np.abs(frames[0][rows, cols] - frames[1][rows, cols]) > 40).any()


# --- refusals -----------------------------------------------------------

@pytest.mark.parametrize('value', [5, 3.5, ['a', 'b'], ('a',), {'a': 1}])
def test_a_caption_of_the_wrong_type_is_refused(value):
    with pytest.raises(TypeError, match='caption= must be a string'):
        _single(caption=value)
    with pytest.raises(TypeError, match='caption= must be a string'):
        _grid(caption=value)


def test_a_pattern_without_an_index_is_refused_before_the_pipeline_runs():
    with pytest.raises(ValueError, match="caption='{index}' is a per-frame"):
        _single(caption='{index}')
    with pytest.raises(ValueError, match="caption='{index}' is a per-frame"):
        _grid(caption='{index}')


def test_a_pattern_the_index_cannot_format_names_the_caption():
    with pytest.raises(ValueError, match='caption=.*could not be formatted'):
        _single(FRAMES[0], caption='{index:%q%!}{other}')


@pytest.mark.parametrize('maker', [_single, _grid], ids=['single', 'grid'])
def test_a_callable_must_return_a_string(maker):
    before = set(plt.get_fignums())
    with pytest.raises(TypeError, match='caption= callable must return a str'):
        maker(caption=lambda ctx: 3)
    # the abandoned figure is not left registered with pyplot
    assert set(plt.get_fignums()) == before


@pytest.mark.parametrize('extra', [
    dict(animate=True, duration=1, frame_rate=8),
    dict(),
    dict(panels=(1, 4)),
], ids=['animated', 'static', 'static-grid'])
def test_the_plotly_backend_is_refused_like_companion(extra):
    data = DATA if 'panels' in extra else DATA[0]
    with pytest.raises(NotImplementedError,
                       match="backend='matplotlib'") as err:
        hyp.plot(data, caption='c', backend='plotly', show=False, **extra)
    assert 'caption= is matplotlib-only' in str(err.value)


def test_a_caller_supplied_ax_is_refused():
    fig, axes = hyp_subplots(1, 2)
    with pytest.raises(ValueError, match='caption= is drawn under the whole'):
        hyp.plot(DATA[0], ax=axes[0], caption='c', show=False)
    assert not fig.texts


def test_caption_is_a_named_parameter_defaulting_to_none():
    import inspect
    parameter = inspect.signature(hyp.plot).parameters['caption']
    assert parameter.default is None
