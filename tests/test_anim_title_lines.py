"""Multi-line title reservation for animated 3-D plots (GH #285, bug).

`_animated_3d_title_line_height_in` probed ONE line ('Xygj'), so
``title='two\\nlines'`` reserved the same margin a one-line title did and the
second line ran off the top of the canvas. That is what
`animate_conversation.py`'s `make_room_for_title` worked around.

Single-line reservations must stay byte-identical: the launch clips depend
on the figure heights they already have.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plot import (                              # noqa: E402
    _animated_3d_title_line_height_in, _title_line_count)


def clouds(n_sets=2, n_rows=20, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(n_rows, 3)), axis=0) + 3.0 * i
            for i in range(n_sets)]


def animate(title, **kwargs):
    return hyp.plot(clouds(), animate=True, duration=1, frame_rate=4,
                    title=title, show=False, **kwargs)


def title_pixel_box(anim):
    anim.draw_frame(0)
    anim.figure.canvas.draw()
    return anim.figure.axes[0].title.get_window_extent()


class TestLineCount:

    @pytest.mark.parametrize('title, expected', [
        (None, 1),
        ('one line', 1),
        ('two\nlines', 2),
        ('a\nb\nc', 3),
        (['one', 'two\nlines'], 2),
        (['a', 'b'], 1),
    ])
    def test_counts_the_tallest_title(self, title, expected):
        assert _title_line_count(title) == expected

    def test_counts_across_several_titles(self):
        assert _title_line_count('one', ['a\nb\nc'], None) == 3


class TestProbe:

    def test_two_lines_measure_more_than_twice_a_line_s_ink(self):
        """The probe must include the LEADING between lines, which is why
        it repeats the probe text instead of multiplying one line's
        height."""
        fig, ax = plt.subplots()
        try:
            one = _animated_3d_title_line_height_in(ax, n_lines=1)
            two = _animated_3d_title_line_height_in(ax, n_lines=2)
            three = _animated_3d_title_line_height_in(ax, n_lines=3)
            assert two > one
            assert three > two
            # each extra line adds roughly one line height plus leading;
            # the exact increment depends on the font engine (matplotlib
            # 3.10 and 3.11 hint the probe differently), so compare the two
            # increments loosely rather than to a constant
            assert (three - two) == pytest.approx(two - one, rel=0.3)
        finally:
            plt.close(fig)

    def test_default_is_the_single_line_probe(self):
        fig, ax = plt.subplots()
        try:
            assert _animated_3d_title_line_height_in(ax) == \
                _animated_3d_title_line_height_in(ax, n_lines=1)
        finally:
            plt.close(fig)


class TestReservation:

    def test_a_two_line_title_grows_the_figure_by_one_line(self):
        one, two = animate('one'), animate('one\ntwo')
        try:
            h1 = one.figure.get_size_inches()[1]
            h2 = two.figure.get_size_inches()[1]
            # probed against the REAL animated axes: the probe copies that
            # axes' resolved title font, and hypertools draws inside its
            # own font rc_context, so a plain `plt.subplots()` axes would
            # measure a different family
            axes = one.figure.axes[0]
            line = (_animated_3d_title_line_height_in(axes, n_lines=2)
                    - _animated_3d_title_line_height_in(axes, n_lines=1))
            assert line > 0
            assert h2 - h1 == pytest.approx(line, rel=1e-6)
        finally:
            plt.close(one.figure)
            plt.close(two.figure)

    def test_the_second_line_stays_on_the_canvas(self):
        """The bug itself: measured in PIXELS, against the canvas."""
        anim = animate('first line\nsecond line')
        try:
            box = title_pixel_box(anim)
            height = anim.figure.get_size_inches()[1] * anim.figure.dpi
            assert box.y0 >= 0
            assert box.y1 <= height
            # and the title really is two lines tall
            one = animate('first line')
            try:
                assert box.height > title_pixel_box(one).height * 1.5
            finally:
                plt.close(one.figure)
        finally:
            plt.close(anim.figure)

    def test_a_three_line_title_also_fits(self):
        anim = animate('a\nb\nc')
        try:
            box = title_pixel_box(anim)
            height = anim.figure.get_size_inches()[1] * anim.figure.dpi
            assert box.y0 >= 0 and box.y1 <= height
        finally:
            plt.close(anim.figure)

    def test_title_wrap_multi_line_titles_fit_too(self):
        """`title_wrap=` inserts the newlines itself, so the reservation
        has to see them -- it runs after wrapping."""
        anim = animate('a rather long title that has to wrap somewhere',
                       title_wrap=18)
        try:
            anim.draw_frame(0)
            assert '\n' in anim.figure.axes[0].get_title()
            box = title_pixel_box(anim)
            height = anim.figure.get_size_inches()[1] * anim.figure.dpi
            assert box.y1 <= height
        finally:
            plt.close(anim.figure)

    def test_per_segment_multi_line_titles_fit(self):
        anim = hyp.plot(clouds(3), animate='serial', duration=2,
                        frame_rate=6, title=['one', 'a\nb', 'three'],
                        show=False)
        try:
            height = anim.figure.get_size_inches()[1] * anim.figure.dpi
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                anim.figure.canvas.draw()
                assert anim.figure.axes[0].title.get_window_extent().y1 \
                    <= height
        finally:
            plt.close(anim.figure)

    def test_single_line_figure_height_is_unchanged(self):
        """Identical to the pre-fix reservation, so no launch clip moves:
        the 4.8 in default plus ONE measured title line and the 0.08 in
        pad (5.04 in under matplotlib 3.10's font metrics, 5.107 in under
        3.11's; the probe, not a constant, is the reference)."""
        anim = animate('one line')
        try:
            # probe the animation's OWN axes, so the title's resolved font
            # (the bundled stack, 12 pt) is what gets measured
            one_line = _animated_3d_title_line_height_in(anim.figure.axes[0],
                                                         n_lines=1)
            assert anim.figure.get_size_inches()[1] == pytest.approx(
                4.8 + one_line + 0.08, abs=1e-6)
        finally:
            plt.close(anim.figure)

    def test_titleless_animation_keeps_the_full_canvas(self):
        anim = hyp.plot(clouds(), animate=True, duration=1, frame_rate=4,
                        show=False)
        try:
            assert anim.figure.get_size_inches()[1] == pytest.approx(4.8)
            # the axes still spans the full HEIGHT of the canvas: nothing
            # was reserved above it (its width is set by the 3-D aspect)
            pos = anim.figure.axes[0].get_position()
            assert (pos.y0, pos.height) == pytest.approx((0.0, 1.0))
        finally:
            plt.close(anim.figure)
