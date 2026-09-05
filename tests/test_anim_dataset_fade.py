"""`dataset_fade={'floor', 'decay'}`: serial cross-dataset recency fade
(GH #285).

``chemtrails``/``precog``/``bullettime`` fade WITHIN one trajectory; nothing
faded ACROSS already-revealed datasets, so `examples/animate_conversation.py`
carried a 55-line `recency_fade` hook. These tests check the library
reproduces that hook's numbers exactly, on both backends.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np                                              # noqa: E402
import pytest                                                   # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402

import hypertools as hyp                                        # noqa: E402
from hypertools.plot.plot import dataset_fade_alpha              # noqa: E402

FLOOR, DECAY = 0.15, 0.7


def turns(n_sets=4, n_rows=12, seed=0):
    rng = np.random.default_rng(seed)
    return [np.cumsum(rng.normal(size=(n_rows, 3)), axis=0) + 4.0 * i
            for i in range(n_sets)]


def conversation_turn_alpha(i, revealed, current):
    """`animate_conversation.py`'s own `turn_alpha`, verbatim, so the
    library's numbers are checked against the code they replace."""
    if i > current or revealed < 2:
        return 0.0
    if i == current:
        return 1.0
    return FLOOR + (1.0 - FLOOR) * DECAY ** (current - i)


class TestFormula:

    def test_public_helper_is_the_conversation_hook(self):
        for current in range(5):
            for i in range(5):
                for revealed in (0, 1, 2, 30):
                    assert dataset_fade_alpha(i, current, revealed,
                                              FLOOR, DECAY) == \
                        conversation_turn_alpha(i, revealed, current)


class TestMatplotlib:

    def test_alphas_match_the_formula_on_every_frame(self):
        checked = []

        def check(ctx):
            n = len(ctx.revealed_counts)
            heads, trails = ctx.artists[:n], ctx.artists[n:]
            assert len(trails) == n
            for i, (head, trail) in enumerate(zip(heads, trails)):
                want = conversation_turn_alpha(i, ctx.revealed_counts[i],
                                               ctx.current_index)
                assert head.get_alpha() == pytest.approx(want)
                # head and trail take the SAME alpha, not the library's
                # 0.3x trail convention
                assert trail.get_alpha() == pytest.approx(want)
            checked.append(ctx.frame)

        anim = hyp.plot(turns(), animate='serial', chemtrails=True,
                        duration=3, frame_rate=8,
                        dataset_fade={'floor': FLOOR, 'decay': DECAY},
                        show=False, on_frame=check)
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
            assert len(checked) == anim.n_frames
        finally:
            plt.close(anim.figure)

    def test_the_positional_pair_is_the_same_thing(self):
        def alphas_for(spec):
            got = {}
            anim = hyp.plot(turns(), animate='serial', duration=2,
                            frame_rate=6, dataset_fade=spec, show=False,
                            on_frame=lambda c: got.__setitem__(
                                c.frame, [a.get_alpha() for a in c.artists]))
            try:
                for i in range(anim.n_frames):
                    anim.draw_frame(i)
                return dict(got)
            finally:
                plt.close(anim.figure)

        assert alphas_for((FLOOR, DECAY)) == \
            alphas_for({'floor': FLOOR, 'decay': DECAY})

    def test_floor_defaults_to_zero(self):
        got = []
        anim = hyp.plot(turns(n_sets=3), animate='serial', duration=2,
                        frame_rate=6, dataset_fade={'decay': 0.5},
                        show=False,
                        on_frame=lambda c: got.append(
                            (c.current_index,
                             [a.get_alpha() for a in c.artists])))
        try:
            for i in range(anim.n_frames):
                anim.draw_frame(i)
            current, alphas = got[-1]
            assert current == 2
            assert alphas == pytest.approx([0.25, 0.5, 1.0])
        finally:
            plt.close(anim.figure)

    def test_a_user_hook_still_runs_after_and_can_override(self):
        """The fade is an INTERNAL updater, so `on_frame=` sees the faded
        artists and its own assignment wins."""
        anim = hyp.plot(turns(n_sets=3), animate='serial', duration=2,
                        frame_rate=6, dataset_fade=(FLOOR, DECAY),
                        show=False,
                        on_frame=lambda c: c.artists[0].set_alpha(0.99))
        try:
            anim.draw_frame(anim.n_frames - 1)
            assert anim.figure.axes[0].lines[0].get_alpha() == 0.99
        finally:
            plt.close(anim.figure)

    def test_default_leaves_alpha_untouched(self):
        anim = hyp.plot(turns(n_sets=3), animate='serial', duration=2,
                        frame_rate=6, show=False)
        try:
            anim.draw_frame(anim.n_frames - 1)
            assert all(line.get_alpha() is None
                       for line in anim.figure.axes[0].lines[:3])
        finally:
            plt.close(anim.figure)


class TestPlotly:

    def test_per_frame_opacity_matches_the_formula(self):
        seen = []
        hyp.plot(turns(), animate='serial', chemtrails=True, duration=3,
                 frame_rate=8, backend='plotly',
                 dataset_fade={'floor': FLOOR, 'decay': DECAY},
                 show=False, on_frame=seen.append)
        assert seen
        for ctx in seen:
            n = len(ctx.revealed_counts)
            for i, trace in enumerate(ctx.artists):
                dataset = i if i < n else i - n
                want = conversation_turn_alpha(
                    dataset, ctx.revealed_counts[dataset], ctx.current_index)
                assert trace.opacity == pytest.approx(want)


class TestValidation:

    def test_parallel_reveal_is_refused(self):
        with pytest.raises(ValueError, match=r'serial reveal'):
            hyp.plot(turns(), animate=True, duration=1,
                     dataset_fade=(0.1, 0.5), show=False)

    def test_morph_is_refused_even_though_its_order_is_serial(self):
        """`animate='morph'` resolves to order='serial' but interpolates
        whole clouds -- `revealed_counts` is None there, so the fade would
        silently do nothing."""
        with pytest.raises(ValueError, match=r'serial reveal'):
            hyp.plot(turns(), '.', animate='morph', duration=1,
                     dataset_fade=(0.1, 0.5), show=False)

    def test_static_plot_is_refused(self):
        with pytest.raises(ValueError, match=r'requires an animated plot'):
            hyp.plot(turns(), dataset_fade=(0.1, 0.5), show=False)

    def test_decay_is_required(self):
        with pytest.raises(ValueError, match=r"needs a 'decay' entry"):
            hyp.plot(turns(), animate='serial', duration=1,
                     dataset_fade={'floor': 0.2}, show=False)

    @pytest.mark.parametrize('spec, match', [
        ({'floor': 2.0, 'decay': 0.5}, r'floor must be between 0 and 1'),
        ({'floor': 0.1, 'decay': 0.0}, r'decay must be greater than 0'),
        ({'floor': 0.1, 'decay': 1.5}, r'decay must be greater than 0'),
        ({'flor': 0.1, 'decay': 0.5}, r'unknown key'),
    ])
    def test_bad_values_raise(self, spec, match):
        with pytest.raises(ValueError, match=match):
            hyp.plot(turns(), animate='serial', duration=1,
                     dataset_fade=spec, show=False)

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match=r'must be a dict'):
            hyp.plot(turns(), animate='serial', duration=1,
                     dataset_fade=0.5, show=False)
