"""`HyperAnimation.drawn_extent(frames=None)` (GH #285).

`examples/animate_painting_embeddings.py` measures the union bounding box
of everything drawn over its orbit, from rendered pixels, to place prose and
thumbnail columns beside the spinning cube. Its helper was already generic;
this is that helper, on the animation.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.transforms import Bbox                          # noqa: E402

import hypertools as hyp                                        # noqa: E402


def spinning(n_rows=30, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    data = [np.cumsum(rng.normal(size=(n_rows, 3)), axis=0)]
    return hyp.plot(data, animate='spin', duration=2, frame_rate=8,
                    show=False, **kwargs)


def painting_style_extent(anim, frames):
    """The measurement `animate_painting_embeddings.py` makes by hand, so
    the library's answer can be checked against the code it replaces."""
    fig = anim.figure
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for i in frames:
        anim.draw_frame(i)
        fig.canvas.draw()
        dark = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(-1) < 250
        rows, cols = np.nonzero(dark)
        n_rows, n_cols = dark.shape
        lo = np.minimum(lo, [cols.min() / n_cols,
                             1 - (rows.max() + 1) / n_rows])
        hi = np.maximum(hi, [(cols.max() + 1) / n_cols,
                             1 - rows.min() / n_rows])
    return lo, hi


class TestDrawnExtent:

    def test_matches_the_hand_written_helper_it_replaces(self):
        anim = spinning()
        try:
            frames = range(0, anim.n_frames, 5)
            lo, hi = painting_style_extent(anim, frames)
            box = anim.drawn_extent(frames=frames)
            assert isinstance(box, Bbox)
            assert np.allclose(box.p0, lo)
            assert np.allclose(box.p1, hi)
        finally:
            plt.close(anim.figure)

    def test_default_samples_twelve_frames_across_the_clip(self):
        """The union over a sample must contain the extent of any single
        sampled frame -- and, for a spin, be strictly larger than one
        frame's, since the projected box changes with the camera angle."""
        anim = spinning(n_rows=60)
        try:
            union = anim.drawn_extent()
            single = anim.drawn_extent(frames=[0])
            assert union.x0 <= single.x0 and union.y0 <= single.y0
            assert union.x1 >= single.x1 and union.y1 >= single.y1
            assert union.width > single.width or union.height > single.height
        finally:
            plt.close(anim.figure)

    def test_int_frames_is_a_sample_count(self):
        anim = spinning()
        try:
            few = anim.drawn_extent(frames=3)
            many = anim.drawn_extent(frames=anim.n_frames)
            # more samples can only grow the union
            assert many.x0 <= few.x0 + 1e-12
            assert many.x1 >= few.x1 - 1e-12
        finally:
            plt.close(anim.figure)

    def test_coordinates_are_figure_fractions_with_y_up(self):
        """A title sits at the TOP of the figure, so adding one must raise
        the union's upper y edge, not its lower one."""
        plain = spinning()
        titled = spinning(title='a title up here')
        try:
            a, b = plain.drawn_extent(frames=[0]), titled.drawn_extent(
                frames=[0])
            assert 0.0 <= a.x0 < a.x1 <= 1.0
            assert 0.0 <= a.y0 < a.y1 <= 1.0
            assert b.y1 > a.y1
        finally:
            plt.close(plain.figure)
            plt.close(titled.figure)

    def test_a_bigger_zoom_draws_a_bigger_box(self):
        small = spinning(zoom=0.5)
        big = spinning(zoom=1.4)
        try:
            assert (big.drawn_extent(frames=[0]).width
                    > small.drawn_extent(frames=[0]).width)
        finally:
            plt.close(small.figure)
            plt.close(big.figure)

    def test_empty_frames_raises(self):
        anim = spinning()
        try:
            with pytest.raises(ValueError, match=r'at least one frame'):
                anim.drawn_extent(frames=[])
        finally:
            plt.close(anim.figure)

    def test_out_of_range_frame_raises_the_usual_index_error(self):
        anim = spinning()
        try:
            with pytest.raises(IndexError):
                anim.drawn_extent(frames=[anim.n_frames])
        finally:
            plt.close(anim.figure)

    def test_measuring_does_not_change_what_is_drawn(self):
        """Frames are idempotent by contract, so re-drawing a frame after a
        measurement gives back exactly the same pixels."""
        anim = spinning()
        try:
            anim.draw_frame(2)
            anim.figure.canvas.draw()
            before = np.asarray(anim.figure.canvas.buffer_rgba()).copy()
            anim.drawn_extent()
            anim.draw_frame(2)
            anim.figure.canvas.draw()
            after = np.asarray(anim.figure.canvas.buffer_rgba())
            assert np.array_equal(before, after)
        finally:
            plt.close(anim.figure)
