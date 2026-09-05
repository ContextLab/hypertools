"""`title_kwargs=`, `title_color=`, and `font=` reaching per-segment titles.

GH #285. Before this, the only way to size/weight/colour a title was to let
it inherit rcParams (or wrap the whole call in `plt.rc_context`), and a
per-segment `title=` list was re-set on EVERY animation frame with no styling
at all -- so three launch examples carried an `on_frame=` callback whose only
job was to restyle the title after hypertools reset it. `resolve_font` only
ever sets a family, and `_make_title_updater` passed no `fontproperties=` at
all, so a resolved `font=` reached the static title and never a segment one.

Real figures and real animation frames; artists are read back off the axes.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


def _datasets(n=3, rows=25, cols=5, seed=1):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


def _drive(anim, frame):
    """Run one animation frame through the real per-frame machinery."""
    anim.draw_frame(frame)


# --- static titles ------------------------------------------------------

def test_title_kwargs_sets_size_weight_and_colour():
    fig = hyp.plot(_datasets(2), title='styled', reduce='PCA', show=False,
                   title_kwargs={'size': 22, 'weight': 'bold',
                                 'color': '#E4572E'})
    title = fig.axes[0].title
    assert title.get_fontsize() == 22
    assert title.get_fontweight() == 'bold'
    assert matplotlib.colors.to_hex(title.get_color()) == '#e4572e'


def test_title_kwargs_long_names_work_too():
    fig = hyp.plot(_datasets(2), title='styled', reduce='PCA', show=False,
                   title_kwargs={'fontsize': 18, 'fontweight': 'bold'})
    assert fig.axes[0].title.get_fontsize() == 18


def test_title_kwargs_y_repositions_the_title():
    plain = hyp.plot(_datasets(2), title='t', reduce='PCA', show=False)
    moved = hyp.plot(_datasets(2), title='t', reduce='PCA', show=False,
                     title_kwargs={'y': 0.5})
    # Axes3D scales the requested y by 0.92 (its own title offset), so the
    # assertion is on the RELATIVE move, not the literal number.
    assert (moved.axes[0].title.get_position()[1]
            < plain.axes[0].title.get_position()[1])
    assert moved.axes[0].title.get_position()[1] == pytest.approx(0.92 * 0.5)


def test_title_kwargs_overrides_the_resolved_font_size():
    """`font=` resolves to a FontProperties carrying only a family, so its
    size would otherwise win; `title_kwargs` must be applied after it."""
    fig = hyp.plot(_datasets(2), title='t', reduce='PCA', show=False,
                   font='DejaVu Sans', title_kwargs={'size': 31})
    assert fig.axes[0].title.get_fontsize() == 31
    assert 'DejaVu Sans' in fig.axes[0].title.get_fontname()


def test_title_color_scalar_is_shorthand_for_title_kwargs_color():
    fig = hyp.plot(_datasets(2), title='t', reduce='PCA', show=False,
                   title_color='#123456')
    assert matplotlib.colors.to_hex(fig.axes[0].title.get_color()) == '#123456'


def test_title_color_rgb_tuple_is_one_colour_not_three():
    fig = hyp.plot(_datasets(2), title='t', reduce='PCA', show=False,
                   title_color=(0.1, 0.2, 0.3))
    np.testing.assert_allclose(
        matplotlib.colors.to_rgb(fig.axes[0].title.get_color()),
        (0.1, 0.2, 0.3), atol=1e-6)


def test_untouched_title_matches_the_default_exactly():
    """Backwards compatibility: with none of the new kwargs the title
    artist must be indistinguishable from the pre-GH-#285 one."""
    fig = hyp.plot(_datasets(2), title='plain', reduce='PCA', show=False)
    styled = hyp.plot(_datasets(2), title='plain', reduce='PCA', show=False,
                      title_kwargs={})
    plain_title, styled_title = fig.axes[0].title, styled.axes[0].title
    assert plain_title.get_text() == 'plain'
    # an EMPTY title_kwargs must not perturb a single property
    for getter in ('get_fontsize', 'get_fontweight', 'get_color',
                   'get_fontname', 'get_position', 'get_ha', 'get_va'):
        assert getattr(plain_title, getter)() == getattr(styled_title,
                                                         getter)()


# --- validation ---------------------------------------------------------

def test_unknown_title_kwarg_raises_naming_the_kwarg():
    with pytest.raises(ValueError, match='title_kwargs'):
        hyp.plot(_datasets(2), title='t', title_kwargs={'sizee': 20},
                 show=False)


def test_title_kwargs_must_be_a_dict():
    with pytest.raises(TypeError, match='title_kwargs'):
        hyp.plot(_datasets(2), title='t', title_kwargs=20, show=False)


def test_title_color_list_without_a_title_list_raises():
    with pytest.raises(ValueError, match='per segment'):
        hyp.plot(_datasets(3), title='one title',
                 title_color=['r', 'g', 'b'], show=False)


def test_title_color_list_length_must_match_the_title_list():
    with pytest.raises(ValueError, match='title_color has 2 entries'):
        hyp.plot(_datasets(3), title=['a', 'b', 'c'], animate='serial',
                 title_color=['r', 'g'], show=False)


# --- per-segment titles (the bug) ---------------------------------------

def test_font_now_reaches_per_segment_titles():
    """The bug: `_make_title_updater` set the title with NO
    `fontproperties=`, so a resolved `font=` styled the static title and
    silently never the per-segment ones."""
    anim = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                    animate='serial', reduce='PCA', show=False,
                    duration=3, frame_rate=4, font='DejaVu Sans')
    ax = anim.figure.axes[0]
    _drive(anim, 1)
    assert ax.get_title() != ''
    assert 'DejaVu Sans' in ax.title.get_fontname()


def test_title_kwargs_are_reapplied_on_every_segment_frame():
    anim = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                    animate='serial', reduce='PCA', show=False,
                    duration=3, frame_rate=4,
                    title_kwargs={'size': 26, 'weight': 'bold'})
    ax = anim.figure.axes[0]
    seen = set()
    for frame in range(0, 12):
        _drive(anim, frame)
        if ax.get_title():
            seen.add(ax.get_title())
            assert ax.title.get_fontsize() == 26
            assert ax.title.get_fontweight() == 'bold'
    assert len(seen) >= 2, 'the per-segment titles never changed'


def test_title_color_list_tints_each_segment():
    colours = ['#ff0000', '#00ff00', '#0000ff']
    anim = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                    animate='serial', reduce='PCA', show=False,
                    duration=3, frame_rate=6, title_color=colours)
    ax = anim.figure.axes[0]
    by_title = {}
    for frame in range(0, 18):
        _drive(anim, frame)
        if ax.get_title():
            by_title[ax.get_title()] = matplotlib.colors.to_hex(
                ax.title.get_color())
    assert by_title.get('a') == '#ff0000'
    assert by_title.get('c') == '#0000ff'


def test_title_color_callable_receives_the_frame_context():
    seen = []

    def tint(ctx):
        seen.append(ctx.current_index)
        return '#333333'

    anim = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                    animate='serial', reduce='PCA', show=False,
                    duration=2, frame_rate=4, title_color=tint)
    ax = anim.figure.axes[0]
    _drive(anim, 3)
    assert seen, 'the callable never fired'
    assert all(i is None or isinstance(i, (int, np.integer)) for i in seen)
    assert matplotlib.colors.to_hex(ax.title.get_color()) == '#333333'


def test_unstyled_segment_titles_are_unchanged():
    """No new kwargs -> the updater makes the same bare `set_title` call."""
    anim = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                    animate='serial', reduce='PCA', show=False,
                    duration=3, frame_rate=4)
    ax = anim.figure.axes[0]
    _drive(anim, 1)
    default_size = matplotlib.rcParams['axes.titlesize']
    expected = (default_size if isinstance(default_size, (int, float))
                else matplotlib.font_manager.font_scalings[default_size]
                * matplotlib.rcParams['font.size'])
    assert ax.title.get_fontsize() == pytest.approx(expected)


def test_title_kwargs_size_grows_the_animated_3d_title_margin():
    """A bigger title needs a bigger reserved top margin, or it renders
    off-canvas (the whole reason `animate_market_sectors` wrapped its call
    in `plt.rc_context({'axes.titlesize': ...})`)."""
    small = hyp.plot(_datasets(2, rows=20), title='t', animate=True,
                     reduce='PCA', show=False, duration=1, frame_rate=2)
    big = hyp.plot(_datasets(2, rows=20), title='t', animate=True,
                   reduce='PCA', show=False, duration=1, frame_rate=2,
                   title_kwargs={'size': 40})
    assert big.figure.get_size_inches()[1] > small.figure.get_size_inches()[1]


# --- plotly parity ------------------------------------------------------

def test_title_kwargs_map_onto_plotly_layout_title():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2, rows=20), title='t', reduce='PCA',
                   backend='plotly', show=False,
                   title_kwargs={'size': 24, 'color': '#E4572E',
                                 'weight': 'bold'})
    font = fig.layout.title.font
    assert font.color == '#E4572E'
    assert font.weight == 'bold'
    assert font.size > 24            # points -> pixels


def test_unsupported_title_kwargs_warn_naming_plotly():
    pytest.importorskip('plotly')
    with pytest.warns(UserWarning, match="plotly.*title_kwargs"):
        hyp.plot(_datasets(2, rows=20), title='t', reduce='PCA',
                 backend='plotly', show=False,
                 title_kwargs={'pad': 20})


def test_callable_title_color_is_refused_under_plotly():
    pytest.importorskip('plotly')
    with pytest.raises(NotImplementedError, match='plotly'):
        hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                 animate='serial', reduce='PCA', backend='plotly',
                 show=False, duration=2, title_color=lambda ctx: 'red')


def test_plotly_segment_titles_carry_the_style_every_frame():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(3, rows=20), title=['a', 'b', 'c'],
                   animate='serial', reduce='PCA', backend='plotly',
                   show=False, duration=2, frame_rate=4,
                   title_kwargs={'size': 24, 'color': '#E4572E'})
    titled = [f for f in fig.frames
              if f.layout is not None and f.layout.title is not None
              and f.layout.title.text]
    assert titled, 'no frame carried a title'
    for frame in titled:
        assert frame.layout.title.font.color == '#E4572E'
