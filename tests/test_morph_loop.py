"""`loop=True` for `animate='morph'` (GH #285).

`examples/animate_morph_zoo.py` samples its clouds by hand and appends
``sampled[0]`` itself, with the comment: "the loop-closing repeat of the
first cloud has to be the SAME sample, and morph_samples draws a fresh
subset per dataset". `loop=True` is that, done by the library.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot import morph as _morph                     # noqa: E402


def zoo_shapes(n=120, seed=0):
    """The zoo's own shape list, small enough to render in a test.

    Built the way `animate_morph_zoo.assemble` builds it: five parametric
    clouds, each isotropically normalized into the drawn box, the cube
    shrunk so it sits visibly inside.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(-1, 1, n)
    ring = np.sqrt(1 - v ** 2)
    raw = {
        'sphere': np.column_stack([ring * np.cos(u), ring * np.sin(u), v]),
        'cube': rng.uniform(-1, 1, (n, 3)),
        'shell': np.column_stack([np.cos(u), np.sin(u),
                                  rng.uniform(-0.3, 0.3, n)]),
        'cone': np.column_stack([(v + 1) / 2 * np.cos(u),
                                 (v + 1) / 2 * np.sin(u), v]),
        'blob': rng.normal(size=(n, 3)) * 0.4,
    }
    out = []
    for name, points in raw.items():
        pts = np.asarray(hyp.manip(points, model='Normalize',
                                   mode='isotropic', min=-1, max=1),
                         dtype=float)
        out.append(pts * (0.8 if name == 'cube' else 1.0))
    return out, [name.capitalize() for name in raw]


def morph_artist_frames(anim):
    """Every frame's drawn point cloud, as arrays."""
    out = []
    for i in range(anim.n_frames):
        anim.draw_frame(i)
        artist = anim.figure.axes[0].lines[-1]
        out.append(np.asarray(artist.get_data_3d()).T.copy())
    return out


class TestSamplerReproducesTheHandMadeSequence:
    """The substantive claim: `loop=True` builds exactly the sequence the
    zoo hand-appends, when the zoo's own pre-sampling is in play."""

    def test_loop_equals_hand_appending_the_first_cloud(self):
        clouds, _ = zoo_shapes()
        looped, looped_masks = _morph.sample_and_match_clouds(
            clouds, morph_samples=len(clouds[0]), loop=True)
        hand, hand_masks = _morph.sample_and_match_clouds(
            clouds + [clouds[0]], morph_samples=len(clouds[0]))
        assert len(looped) == len(clouds) + 1 == len(hand)
        for a, b in zip(looped, hand):
            assert np.array_equal(a, b)
        for a, b in zip(looped_masks, hand_masks):
            assert np.array_equal(a, b)

    def test_loop_closes_on_the_same_sample_when_the_cap_bites(self):
        """The bug the flag exists for: with a cap that actually
        downsamples, hand-appending draws a FRESH subset for the repeat,
        so the loop point jumps. `loop=True` reuses the sample."""
        rng = np.random.default_rng(0)
        clouds = [rng.normal(size=(200, 3)) + i for i in range(4)]
        looped, _ = _morph.sample_and_match_clouds(clouds, morph_samples=50,
                                                   loop=True)
        hand, _ = _morph.sample_and_match_clouds(clouds + [clouds[0]],
                                                 morph_samples=50)
        # matching reorders rows, so compare the POINT SETS
        assert np.array_equal(np.sort(looped[0], axis=0),
                              np.sort(looped[-1], axis=0))
        assert not np.array_equal(np.sort(hand[0], axis=0),
                                  np.sort(hand[-1], axis=0))

    def test_loop_leaves_the_uncapped_sequence_untouched(self):
        clouds, _ = zoo_shapes(n=60)
        plain, _ = _morph.sample_and_match_clouds(clouds)
        looped, _ = _morph.sample_and_match_clouds(clouds, loop=True)
        assert len(looped) == len(plain) + 1
        for a, b in zip(plain, looped):
            assert np.array_equal(a, b)


class TestLoopThroughPlot:

    def test_segment_count_and_frames(self):
        clouds, titles = zoo_shapes(n=60)
        anim = hyp.plot(clouds, '.', animate='morph', loop=True, duration=3,
                        frame_rate=6, morph_samples=60, title=titles,
                        show=False)
        try:
            # n clouds + the closing repeat -> 2(n + 1) - 1 segments
            assert anim.n_segments == 2 * (len(clouds) + 1) - 1
        finally:
            plt.close(anim.figure)

    def test_per_frame_arrays_match_the_hand_appended_version(self):
        """Frame for frame, `loop=True` draws what the hand-appended
        sequence draws -- up to the single rigid translation the extra
        duplicate dataset introduces into `plot`'s own shared
        mean-centring, which is a property of the WORKAROUND, not of the
        morph. Nothing else differs."""
        clouds, titles = zoo_shapes(n=60)
        looped = hyp.plot(clouds, '.', animate='morph', loop=True,
                          duration=3, frame_rate=6, morph_samples=60,
                          title=titles, show=False)
        hand = hyp.plot(clouds + [clouds[0]], '.', animate='morph',
                        duration=3, frame_rate=6, morph_samples=60,
                        title=titles + [titles[0]], show=False)
        try:
            assert looped.n_frames == hand.n_frames
            a = morph_artist_frames(looped)
            b = morph_artist_frames(hand)
            offsets = [np.median(y - x, axis=0) for x, y in zip(a, b)]
            # ONE offset for the whole animation
            for off in offsets[1:]:
                assert np.allclose(off, offsets[0], atol=1e-12)
            for x, y in zip(a, b):
                assert np.allclose(y - offsets[0], x, atol=1e-9)
        finally:
            plt.close(looped.figure)
            plt.close(hand.figure)

    def test_closing_hold_is_titled_like_the_opening_one(self):
        clouds, titles = zoo_shapes(n=40)
        anim = hyp.plot(clouds, '.', animate='morph', loop=True, duration=4,
                        frame_rate=6, morph_samples=40, title=titles,
                        show=False)
        try:
            seen = []
            for i in range(anim.n_frames):
                anim.draw_frame(i)
                seen.append((anim.figure.axes[0].get_title(),))
            texts = [t for (t,) in seen if t]
            assert texts[0] == titles[0]
            assert texts[-1] == titles[0]
            # every shape is still named exactly once, in order, plus the
            # closing repeat
            ordered = [t for i, t in enumerate(texts)
                       if i == 0 or texts[i - 1] != t]
            assert ordered == titles + [titles[0]]
        finally:
            plt.close(anim.figure)

    def test_closing_hold_draws_the_opening_point_set(self):
        clouds, titles = zoo_shapes(n=40)
        anim = hyp.plot(clouds, '.', animate='morph', loop=True, duration=4,
                        frame_rate=6, morph_samples=40, title=titles,
                        show=False)
        try:
            frames = morph_artist_frames(anim)
            assert np.allclose(np.sort(frames[0], axis=0),
                               np.sort(frames[-1], axis=0), atol=1e-9)
        finally:
            plt.close(anim.figure)

    def test_plotly_builds_the_same_number_of_frames(self):
        clouds, titles = zoo_shapes(n=40)
        anim = hyp.plot(clouds, '.', animate='morph', loop=True, duration=3,
                        frame_rate=6, morph_samples=40, title=titles,
                        show=False)
        try:
            n_mpl = anim.n_frames
        finally:
            plt.close(anim.figure)
        fig = hyp.plot(clouds, '.', animate='morph', loop=True, duration=3,
                       frame_rate=6, morph_samples=40, title=titles,
                       backend='plotly', show=False)
        assert len(fig.frames) == n_mpl

    def test_default_is_unchanged(self):
        clouds, titles = zoo_shapes(n=40)
        anim = hyp.plot(clouds, '.', animate='morph', duration=3,
                        frame_rate=6, morph_samples=40, title=titles,
                        show=False)
        try:
            assert anim.n_segments == 2 * len(clouds) - 1
        finally:
            plt.close(anim.figure)

    @pytest.mark.parametrize('style', [True, 'serial', 'spin', 'window'])
    def test_loop_on_a_non_morph_style_raises(self, style):
        clouds, _ = zoo_shapes(n=20)
        with pytest.raises(ValueError, match=r"loop=True is only supported"):
            hyp.plot(clouds[:2], animate=style, loop=True, duration=1,
                     show=False)
