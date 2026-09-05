"""`title_wrap=` (GH #285).

`examples/animate_conversation.py` runs `textwrap.fill` over its per-turn
titles before handing them to `hyp.plot`. `title_wrap=N` does it in the
library, on both the scalar title and every entry of a per-segment list, with
the line break each backend actually uses (a newline for matplotlib,
``<br>`` for plotly).
"""
import textwrap

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import hypertools as hyp


LONG = ("the quick brown fox jumps over the lazy dog and keeps on running "
        "well past the edge of the canvas")


def _datasets(n=3, rows=20, cols=5, seed=9):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(rows, cols)), axis=0) for _ in range(n)]


# --- matplotlib ---------------------------------------------------------

def test_title_wrap_matches_textwrap_fill():
    fig = hyp.plot(_datasets(2), title=LONG, title_wrap=30, reduce='PCA',
                   show=False)
    assert fig.axes[0].get_title() == textwrap.fill(LONG, 30)


def test_title_wrap_actually_breaks_lines():
    fig = hyp.plot(_datasets(2), title=LONG, title_wrap=20, reduce='PCA',
                   show=False)
    assert fig.axes[0].get_title().count('\n') >= 4


def test_narrower_wrap_makes_more_lines():
    wide = hyp.plot(_datasets(2), title=LONG, title_wrap=60, reduce='PCA',
                    show=False).axes[0].get_title()
    narrow = hyp.plot(_datasets(2), title=LONG, title_wrap=20, reduce='PCA',
                      show=False).axes[0].get_title()
    assert narrow.count('\n') > wide.count('\n')


def test_no_wrap_leaves_the_title_alone():
    fig = hyp.plot(_datasets(2), title=LONG, reduce='PCA', show=False)
    assert fig.axes[0].get_title() == LONG
    assert '\n' not in fig.axes[0].get_title()


def test_short_title_is_unchanged_by_a_generous_wrap():
    fig = hyp.plot(_datasets(2), title='short', title_wrap=80, reduce='PCA',
                   show=False)
    assert fig.axes[0].get_title() == 'short'


def test_title_wrap_applies_to_every_segment_of_a_title_list():
    titles = [LONG, 'a short one', LONG.upper()]
    anim = hyp.plot(_datasets(3, rows=20), title=titles, animate='serial',
                    title_wrap=25, reduce='PCA', show=False, duration=3,
                    frame_rate=4)
    ax = anim.figure.axes[0]
    seen = set()
    for frame in range(12):
        anim.draw_frame(frame)
        if ax.get_title():
            seen.add(ax.get_title())
    wrapped = {textwrap.fill(t, 25) for t in titles}
    assert seen <= wrapped
    assert any('\n' in t for t in seen)


def test_wrapped_title_composes_with_title_kwargs():
    fig = hyp.plot(_datasets(2), title=LONG, title_wrap=25, reduce='PCA',
                   show=False, title_kwargs={'size': 18, 'color': '#00ff00'})
    title = fig.axes[0].title
    assert '\n' in title.get_text()
    assert title.get_fontsize() == 18
    assert matplotlib.colors.to_hex(title.get_color()) == '#00ff00'


# --- validation ---------------------------------------------------------

def test_non_integer_title_wrap_raises():
    with pytest.raises(TypeError, match='title_wrap'):
        hyp.plot(_datasets(2), title=LONG, title_wrap='30', show=False)


def test_bool_title_wrap_raises():
    with pytest.raises(TypeError, match='title_wrap'):
        hyp.plot(_datasets(2), title=LONG, title_wrap=True, show=False)


def test_zero_title_wrap_raises():
    with pytest.raises(ValueError, match='at least 1'):
        hyp.plot(_datasets(2), title=LONG, title_wrap=0, show=False)


def test_title_wrap_without_a_title_is_harmless():
    fig = hyp.plot(_datasets(2), title_wrap=20, reduce='PCA', show=False)
    assert fig.axes[0].get_title() == ''


# --- plotly parity ------------------------------------------------------

def test_plotly_wraps_with_br():
    pytest.importorskip('plotly')
    fig = hyp.plot(_datasets(2), title=LONG, title_wrap=30, reduce='PCA',
                   backend='plotly', show=False)
    text = fig.layout.title.text
    assert '<br>' in text
    assert '\n' not in text
    assert text == textwrap.fill(LONG, 30).replace('\n', '<br>')


def test_plotly_segment_titles_wrap_too():
    pytest.importorskip('plotly')
    titles = [LONG, 'short', LONG.upper()]
    fig = hyp.plot(_datasets(3, rows=20), title=titles, animate='serial',
                   title_wrap=25, reduce='PCA', backend='plotly',
                   show=False, duration=2, frame_rate=4)
    texts = {f.layout.title.text for f in fig.frames
             if f.layout is not None and f.layout.title is not None
             and f.layout.title.text}
    assert texts
    assert any('<br>' in t for t in texts)
