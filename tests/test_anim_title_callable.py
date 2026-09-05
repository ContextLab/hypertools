"""A callable `title=` and a `title='{index:...}'` DatetimeIndex pattern
(GH #285).

`animate_market_sectors.py` and `animate_weather_decades.py` both format a
date into the title from the row calendar on every frame, from an
`on_frame=` hook -- which also had to re-apply the title's styling, because
the library's own setter resets it. These tests pin the library doing both.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402


def monthly_frame(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 3)),
                        index=pd.date_range('2020-01-31', periods=n,
                                            freq='ME'))


def titles_of(anim):
    out = []
    for i in range(anim.n_frames):
        anim.draw_frame(i)
        out.append(anim.figure.axes[0].get_title())
    return out


class TestCallableTitle:

    def test_the_callable_runs_once_per_frame_with_that_frame_s_context(self):
        anim = hyp.plot([np.random.default_rng(0).normal(size=(20, 3))],
                        animate=True, duration=2, frame_rate=6,
                        title=lambda ctx: f'frame {ctx.frame} '
                                          f'({ctx.progress:.2f})',
                        show=False)
        try:
            got = titles_of(anim)
            assert got == [f'frame {i} ({i / (anim.n_frames - 1):.2f})'
                           for i in range(anim.n_frames)]
        finally:
            plt.close(anim.figure)

    def test_title_kwargs_style_a_callable_title(self):
        """The bug this closes: the library's title setter reset size and
        weight every frame, so three launch examples carried an `on_frame`
        whose only job was to put them back."""
        anim = hyp.plot([np.random.default_rng(1).normal(size=(20, 3))],
                        animate=True, duration=1, frame_rate=4,
                        title=lambda ctx: f'f{ctx.frame}',
                        title_kwargs={'size': 21, 'weight': 'bold',
                                      'color': '#B03060'},
                        show=False)
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                text = anim.figure.axes[0].title
                assert text.get_text() == f'f{i}'
                assert text.get_fontsize() == 21
                assert text.get_fontweight() == 'bold'
                assert text.get_color() == '#B03060'
        finally:
            plt.close(anim.figure)

    def test_static_plot_gets_a_finished_figure_context(self):
        """Documented contract: frame/n_frames None, progress 1.0,
        revealed_counts full, figure/axes None."""
        seen = {}

        def title(ctx):
            seen.update(frame=ctx.frame, n_frames=ctx.n_frames,
                        progress=ctx.progress, counts=ctx.revealed_counts,
                        figure=ctx.figure, axes=ctx.axes,
                        n_datasets=len(ctx.datasets))
            return 'static'

        data = [np.random.default_rng(2).normal(size=(15, 3)),
                np.random.default_rng(3).normal(size=(9, 3))]
        fig = hyp.plot(data, title=title, show=False)
        try:
            assert fig.axes[0].get_title() == 'static'
            assert seen['frame'] is None and seen['n_frames'] is None
            assert seen['progress'] == 1.0
            assert seen['counts'] == (15, 9)
            assert seen['figure'] is None and seen['axes'] is None
            assert seen['n_datasets'] == 2
        finally:
            plt.close(fig)

    def test_plotly_titles_every_frame_too(self):
        fig = hyp.plot([np.random.default_rng(4).normal(size=(20, 3))],
                       animate=True, duration=2, frame_rate=6,
                       backend='plotly', show=False,
                       title=lambda ctx: f'frame {ctx.frame}')
        got = [f.layout.title.text for f in fig.frames]
        assert got == [f'frame {i}' for i in range(len(fig.frames))]


class TestIndexPattern:

    def test_pattern_formats_the_index_under_the_head(self):
        df = monthly_frame()
        anim = hyp.plot(df, animate=True, duration=2, frame_rate=6,
                        title='{index:%B %Y}', show=False)
        try:
            got = titles_of(anim)
            # the head row is round(progress * (n_rows - 1)) -- exactly the
            # arithmetic animate_weather_decades.py spelled out by hand
            expected = [
                df.index[round(i / (anim.n_frames - 1) * (len(df) - 1))]
                .strftime('%B %Y') for i in range(anim.n_frames)]
            assert got == expected
            assert got[0] == 'January 2020'
            assert got[-1] == 'December 2021'
        finally:
            plt.close(anim.figure)

    def test_pattern_mixes_with_literal_text(self):
        df = monthly_frame(n=12)
        anim = hyp.plot(df, animate=True, duration=1, frame_rate=4,
                        title='Sales through {index:%Y-%m}', show=False)
        try:
            assert titles_of(anim)[-1] == 'Sales through 2020-12'
        finally:
            plt.close(anim.figure)

    def test_serial_pattern_reads_the_revealing_dataset(self):
        """On a serial reveal the head is the last revealed row of the
        dataset currently being revealed, not the clip's progress."""
        df = monthly_frame(n=12)
        anim = hyp.plot([df, df], animate='serial', duration=2,
                        frame_rate=8, title='{index:%m}', show=False)
        try:
            got = titles_of(anim)
            assert got[0] == '01'
            assert got[-1] == '12'
            # the FIRST dataset's reveal already sweeps the whole calendar,
            # so the month resets when the second dataset starts
            assert '01' in got[1:]
        finally:
            plt.close(anim.figure)

    def test_static_pattern_shows_the_last_index_value(self):
        df = monthly_frame()
        fig = hyp.plot(df, title='{index:%B %Y}', show=False)
        try:
            assert fig.axes[0].get_title() == 'December 2021'
        finally:
            plt.close(fig)

    def test_pattern_without_an_index_raises_before_the_pipeline(self):
        with pytest.raises(ValueError, match=r'no row index'):
            hyp.plot(np.random.default_rng(5).normal(size=(20, 3)),
                     animate=True, duration=1, title='{index:%Y}',
                     show=False)

    def test_braces_that_are_not_index_stay_literal(self):
        fig = hyp.plot(np.random.default_rng(6).normal(size=(10, 3)),
                       title='f(x) = {a, b}', show=False)
        try:
            assert fig.axes[0].get_title() == 'f(x) = {a, b}'
        finally:
            plt.close(fig)

    def test_plotly_pattern_matches_matplotlib(self):
        df = monthly_frame()
        anim = hyp.plot(df, animate=True, duration=2, frame_rate=6,
                        title='{index:%B %Y}', show=False)
        try:
            expected = titles_of(anim)
        finally:
            plt.close(anim.figure)
        fig = hyp.plot(df, animate=True, duration=2, frame_rate=6,
                       backend='plotly', title='{index:%B %Y}', show=False)
        assert [f.layout.title.text for f in fig.frames] == expected


class TestAnimated3DTitleMarginCoversDynamicTitles:
    """A dynamic title used to reserve NO top margin (the gate only looked
    at the static and per-segment forms), so it rendered off-canvas."""

    def test_a_callable_title_is_inside_the_canvas(self):
        anim = hyp.plot([np.random.default_rng(7).normal(size=(20, 3))],
                        animate=True, duration=1, frame_rate=4,
                        title=lambda ctx: 'a dynamic title', show=False)
        try:
            anim.draw_frame(0)
            anim.figure.canvas.draw()
            box = anim.figure.axes[0].title.get_window_extent()
            height = (anim.figure.get_size_inches()[1] * anim.figure.dpi)
            assert box.y0 >= 0
            assert box.y1 <= height
        finally:
            plt.close(anim.figure)
