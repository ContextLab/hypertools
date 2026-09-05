"""`FrameContext.progress` and populated `revealed_counts`/`window_bounds`
on the non-serial animation paths (GH #285).

Before this, `frac = ctx.frame / (n_frames - 1)` was recomputed by hand in
both time-indexed launch examples, and `revealed_counts` was ``None`` on
every path except the serial ones -- so a parallel reveal published no way
to say where its head was. Every assertion here is made against a REAL
render (matplotlib frames actually drawn, plotly frames actually built).
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402


def trajectories(n_sets=3, n_rows=24, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(n_rows, 3)), axis=0) + 3.0 * i
            for i in range(n_sets)]


def run(anim):
    """Draw every frame and return the contexts the hook recorded."""
    seen = []
    anim.on_frame(seen.append)
    for i in range(anim.n_frames):
        anim.draw_frame(i)
    return seen


class TestProgress:
    """`progress` is 0 on the first frame, 1 on the last, monotone between,
    and equals the hand-rolled expression it replaces."""

    @pytest.mark.parametrize('style', [True, 'parallel', 'serial', 'spin',
                                       'window', 'morph'])
    def test_progress_spans_zero_to_one_on_every_style(self, style):
        fmt = '.' if style == 'morph' else '-'
        anim = hyp.plot(trajectories(), fmt, animate=style, duration=2,
                        frame_rate=6, show=False)
        try:
            seen = run(anim)
            got = [c.progress for c in seen]
            assert got[0] == 0.0
            assert got[-1] == 1.0
            assert all(b >= a for a, b in zip(got, got[1:]))
            # exactly the expression animate_weather_decades.py and
            # animate_market_sectors.py spelled out by hand
            assert got == [c.frame / max(1, c.n_frames - 1) for c in seen]
        finally:
            plt.close(anim.figure)

    def test_progress_survives_a_one_frame_animation(self):
        anim = hyp.plot(trajectories(), animate=True, duration=0.1,
                        frame_rate=5, show=False)
        try:
            seen = run(anim)
            assert anim.n_frames == 1
            # the same `/ max(1, n - 1)` guard the serial reveal uses
            assert seen[0].progress == 0.0
        finally:
            plt.close(anim.figure)

    def test_progress_matches_plotly(self):
        """The two backends publish the same number for the same call."""
        mpl = hyp.plot(trajectories(), animate=True, duration=2,
                       frame_rate=6, show=False)
        try:
            mpl_progress = [c.progress for c in run(mpl)]
        finally:
            plt.close(mpl.figure)
        seen = []
        hyp.plot(trajectories(), animate=True, duration=2, frame_rate=6,
                 backend='plotly', show=False, on_frame=seen.append)
        assert [c.progress for c in seen] == mpl_progress


class TestRevealedCounts:
    """`revealed_counts` (and `window_bounds`) now describe what each
    non-serial style actually drew."""

    def test_parallel_counts_track_the_drawn_head(self):
        """The published window IS the slice the artist holds.

        Checked INSIDE the hook, never afterwards: matplotlib mutates the
        same artists every frame (`FrameContext`'s artist-lifetime table),
        so a saved context's artists hold the LAST frame's data.
        antialias=False so the drawn vertices are the raw rows rather than
        a denser smooth curve through them.
        """
        checked = []

        def check(ctx):
            assert ctx.revealed_counts is not None
            assert len(ctx.revealed_counts) == len(ctx.datasets)
            for i, (start, end) in enumerate(ctx.window_bounds):
                assert end == ctx.revealed_counts[i]
                drawn = np.asarray(ctx.artists[i].get_data_3d()).T
                assert drawn.shape[0] == end - start
                assert np.allclose(drawn, ctx.datasets[i][start:end])
            checked.append(ctx.revealed_counts)

        anim = hyp.plot(trajectories(), animate=True, duration=2,
                        frame_rate=8, antialias=False, show=False,
                        on_frame=check)
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
            assert len(checked) == anim.n_frames
            assert checked[0] == (1,) * 3
            assert checked[-1] == (anim.n_frames,) * 3
        finally:
            plt.close(anim.figure)

    def test_window_mode_reports_a_moving_window(self):
        # focused=0.5 s at 10 fps is a 5-frame window over a 30-frame clip,
        # so the window really does slide off the beginning
        anim = hyp.plot(trajectories(n_rows=40), animate='window',
                        duration=3, frame_rate=10, focused=0.5,
                        antialias=False, show=False)
        try:
            seen = run(anim)
            starts = [c.window_bounds[0][0] for c in seen]
            assert starts[0] == 0
            # a sliding window leaves the beginning behind: revealed_counts
            # alone cannot say that, which is why window_bounds exists
            assert max(starts) > 0
            for ctx in seen:
                for i, (start, end) in enumerate(ctx.window_bounds):
                    assert 0 <= start <= end
                    assert end == ctx.revealed_counts[i]
        finally:
            plt.close(anim.figure)

    def test_spin_reports_every_row_on_every_frame(self):
        anim = hyp.plot(trajectories(), animate='spin', duration=2,
                        frame_rate=6, antialias=False, show=False)
        try:
            seen = run(anim)
            full = tuple(len(d) for d in seen[0].datasets)
            assert all(c.revealed_counts == full for c in seen)
            assert all(c.window_bounds == tuple((0, n) for n in full)
                       for c in seen)
            # and that really is what is drawn (matplotlib re-delivers the
            # same artists every frame, so reading one afterwards is
            # reading the LAST frame -- fine here, since 'spin' draws every
            # row on every frame)
            drawn = np.asarray(seen[-1].artists[0].get_data_3d()).T
            assert drawn.shape[0] == full[0]
        finally:
            plt.close(anim.figure)

    def test_serial_counts_are_cumulative_and_unchanged(self):
        anim = hyp.plot(trajectories(), animate='serial', duration=2,
                        frame_rate=8, show=False)
        try:
            for ctx in run(anim):
                assert ctx.window_bounds == tuple(
                    (0, c) for c in ctx.revealed_counts)
        finally:
            plt.close(anim.figure)

    def test_morph_keeps_reporting_none(self):
        """A morph interpolates whole clouds; there is no row reveal to
        count, and None must not be normalized into an empty tuple."""
        anim = hyp.plot(trajectories(n_rows=20), '.', animate='morph',
                        duration=2, frame_rate=6, show=False)
        try:
            seen = run(anim)
            assert all(c.revealed_counts is None for c in seen)
            assert all(c.window_bounds is None for c in seen)
        finally:
            plt.close(anim.figure)

    @pytest.mark.parametrize('style', [True, 'spin', 'window'])
    def test_plotly_publishes_the_same_counts(self, style):
        mpl = hyp.plot(trajectories(), animate=style, duration=2,
                       frame_rate=6, show=False)
        try:
            expected = [(c.revealed_counts, c.window_bounds)
                        for c in run(mpl)]
        finally:
            plt.close(mpl.figure)
        seen = []
        hyp.plot(trajectories(), animate=style, duration=2, frame_rate=6,
                 backend='plotly', show=False, on_frame=seen.append)
        assert [(c.revealed_counts, c.window_bounds) for c in seen] == expected


class TestContextStaysImmutable:
    """The new fields keep `FrameContext`'s public container guarantee."""

    def test_window_bounds_is_a_tuple_of_int_pairs(self):
        anim = hyp.plot(trajectories(), animate=True, duration=1,
                        frame_rate=4, show=False)
        try:
            ctx = run(anim)[-1]
            assert isinstance(ctx.window_bounds, tuple)
            assert all(isinstance(b, tuple) and len(b) == 2
                       and all(isinstance(v, int) for v in b)
                       for b in ctx.window_bounds)
        finally:
            plt.close(anim.figure)
