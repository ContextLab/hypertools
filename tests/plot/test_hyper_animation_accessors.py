"""n_frames / n_segments / draw_frame -- the supported way to inspect and
drive an animation, replacing reaches into FuncAnimation internals."""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402

import hypertools as hyp                                        # noqa: E402


def _data(n=60, d=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).cumsum(axis=0)


def test_n_frames_matches_the_requested_rate_and_duration():
    anim = hyp.plot(_data(), '-', animate=True, duration=4, frame_rate=10,
                    show=False)
    assert anim.n_frames == 40


def test_n_frames_is_never_zero_for_a_sub_frame_request():
    """`max(1, ...)`: an animation that asks for less than one frame still
    draws one. Pinned because the gate's floor assertion is only meaningful
    if this cannot silently be 0."""
    anim = hyp.plot(_data(), '-', animate=True, duration=0.01, frame_rate=1,
                    show=False)
    assert anim.n_frames == 1


def test_n_frames_survives_being_read_twice():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    assert anim.n_frames == anim.n_frames == 10


def test_n_segments_counts_holds_and_transitions():
    """`n` clouds give `2n - 1` segments: n holds interleaved with n-1
    transitions, ending on a hold. There is NO implicit closing transition
    back to the first cloud -- a caller who wants the loop to close appends
    `clouds[0]` itself, as `examples/animate_morph_zoo.py` does. Measured
    against `morph.segment_frame_counts`: 2 clouds -> 3, 3 -> 5, 5 -> 9."""
    clouds = [_data(40, 3, s) for s in range(3)]
    anim = hyp.plot(clouds, '.', animate='morph', duration=6, frame_rate=5,
                    show=False)
    assert anim.n_segments == 5


def test_n_segments_is_none_for_a_non_morph_animation():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    assert anim.n_segments is None


def test_n_segments_is_set_for_a_2d_morph_too():
    """Two FuncAnimation morph branches exist (3-D and 2-D); a tag on only
    one makes n_segments silently None for half of them."""
    clouds = [_data(40, 2, s) for s in range(3)]
    anim = hyp.plot(clouds, '.', animate='morph', duration=6, frame_rate=5,
                    reduce=None, show=False)
    assert anim.n_segments == 5


def test_draw_frame_renders_the_requested_index():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    ax = anim.figure.axes[0]
    anim.draw_frame(0)
    early = len(np.asarray(ax.lines[0].get_data_3d())[0])
    anim.draw_frame(anim.n_frames - 1)
    late = len(np.asarray(ax.lines[0].get_data_3d())[0])
    assert late > early, 'a later frame must reveal more of the trajectory'


def test_draw_frame_is_idempotent_and_order_independent():
    """The FrameContext contract: callbacks must be deterministic for a
    given frame, so driving out of order must give identical geometry."""
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    ax = anim.figure.axes[0]
    anim.draw_frame(3)
    once = np.asarray(ax.lines[0].get_data_3d()).copy()
    anim.draw_frame(7)
    anim.draw_frame(0)
    anim.draw_frame(3)
    assert np.allclose(np.asarray(ax.lines[0].get_data_3d()), once)


def test_draw_frame_rejects_an_out_of_range_index():
    anim = hyp.plot(_data(), '-', animate=True, duration=2, frame_rate=5,
                    show=False)
    with pytest.raises(IndexError, match='0 and 9'):
        anim.draw_frame(anim.n_frames)
