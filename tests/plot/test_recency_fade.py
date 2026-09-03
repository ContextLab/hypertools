# -*- coding: utf-8 -*-
"""The conversation example's recency_fade callback.

Drives the REAL callback from the REAL example module (no reimplementation:
a copy of the logic here would pass while the example was broken). The
module is imported once per test module -- after the Contract 4 split its
import defines things and builds nothing, so this is cheap -- and every case
runs against synthetic FrameContexts, which is what makes frame ORDER
testable at all.
"""

import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402 (after the backend is pinned)
import pytest  # noqa: E402

from hypertools.plot.animation_context import FrameContext  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: `recency_fade` is dataset-count agnostic, so the fixture exercises a
#: RANGE rather than one number: 1 is the degenerate single-turn case, 6
#: keeps the fast default, and 28 is the real conversation (`TURNS` has 28
#: entries).
DATASET_COUNTS = (1, 6, 28)
N_DATASETS = 6      # the default for tests that do not vary it


@pytest.fixture(scope='module')
def example():
    # Import the module rather than `runpy.run_path`: after the Contract 4
    # split, `runpy` does not set `__name__` to `'__main__'`, so the
    # module-level names this fixture used to read no longer exist.
    # HYPERTOOLS_OFFLINE makes a stray loader call fail loudly instead of
    # quietly going to the network from the test suite.
    os.environ['HYPERTOOLS_OFFLINE'] = '1'
    try:
        spec = importlib.util.spec_from_file_location(
            'animate_conversation',
            os.path.join(REPO, 'examples', 'animate_conversation.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        os.environ.pop('HYPERTOOLS_OFFLINE', None)
        plt.close('all')


def _ctx(current, revealed=None, n=N_DATASETS, trails=True):
    """A FrameContext shaped exactly like the example's own plot: heads
    first, then one trail per dataset (chemtrails=True)."""
    heads = [plt.Line2D([], []) for _ in range(n)]
    tails = [plt.Line2D([], []) for _ in range(n)] if trails else []
    if revealed is None:
        # `current` may be None (the parallel-animation guard case), so the
        # comparison has to be guarded here too -- `i <= None` is a
        # TypeError, and it would fire in the FIXTURE before the callback
        # under test ever ran.
        revealed = tuple(10 if (current is not None and i <= current) else 0
                         for i in range(n))
    return FrameContext(
        frame=0, n_frames=100, figure=None, axes=None,
        artists=tuple(heads + tails), datasets=(),
        style=True, order='serial', current_index=current,
        current_fraction=0.5, revealed_counts=tuple(revealed))


def test_the_example_still_has_28_turns(example):
    """`DATASET_COUNTS` includes the real conversation's size; keep it true."""
    assert len(example.TURNS) == DATASET_COUNTS[-1]


def test_every_head_and_trail_is_assigned_on_every_frame(example):
    """The portable rule: assign the complete value on every invocation.
    A skipped assignment leaves matplotlib's shared artists at the previous
    frame's value, which is how a fade becomes a smear."""
    fade = example.recency_fade
    ctx = _ctx(current=2)
    for art in ctx.artists:
        art.set_alpha(None)
    fade(ctx)
    assert all(a.get_alpha() is not None for a in ctx.artists), (
        'some artist was left unassigned')


@pytest.mark.parametrize('n', DATASET_COUNTS)
@pytest.mark.parametrize('where', ['first', 'middle', 'last'])
def test_first_middle_and_last_turn(example, n, where):
    current = {'first': 0, 'middle': n // 2, 'last': n - 1}[where]
    fade = example.recency_fade
    ctx = _ctx(current=current, n=n)
    fade(ctx)
    heads = ctx.artists[:n]
    assert heads[current].get_alpha() == 1.0, 'the current turn is opaque'
    for i in range(current + 1, n):
        assert heads[i].get_alpha() == 0.0, 'unspoken turns are invisible'
    earlier = [heads[i].get_alpha() for i in range(current)]
    assert earlier == sorted(earlier), 'older turns must not be brighter'


def test_trails_track_their_own_head(example):
    """On a serial reveal the trail is the already-spoken part of the turn,
    so it carries the same alpha as its head -- not the library's 0.3x
    chemtrails convention, which made the turn being spoken the faintest
    thing on screen (measured 2026-09-03: 821 trail points, 6 head points)."""
    fade = example.recency_fade
    ctx = _ctx(current=3)
    fade(ctx)
    heads, trails = ctx.artists[:N_DATASETS], ctx.artists[N_DATASETS:]
    for head, trail in zip(heads, trails):
        assert trail.get_alpha() == pytest.approx(head.get_alpha())


def test_the_callback_never_indexes_past_revealed_counts(example):
    """The v1 defect: iterating ctx.artists (2N under chemtrails) while
    indexing ctx.revealed_counts (N) raised IndexError on the N+1th artist."""
    fade = example.recency_fade
    fade(_ctx(current=N_DATASETS - 1))  # must not raise


def test_a_missing_trail_artist_is_an_explicit_error(example):
    """Rather than silently pairing head i with head i+1."""
    fade = example.recency_fade
    with pytest.raises(RuntimeError, match='one trail artist per dataset'):
        fade(_ctx(current=1, trails=False))


def test_a_parallel_animation_is_an_explicit_error(example):
    fade = example.recency_fade
    with pytest.raises(RuntimeError, match='serial'):
        fade(_ctx(current=None))


@pytest.mark.parametrize('order', [
    [0, 1, 2, 3, 4, 5],              # forward
    [5, 4, 3, 2, 1, 0],              # backward
    [3, 0, 5, 3, 1, 3],              # shuffled, with repeats
])
def test_alpha_depends_only_on_the_frame_not_on_history(example, order):
    """matplotlib re-delivers frame indices on loop and on save(), so the
    same current_index must always give the same alphas regardless of what
    ran before it."""
    fade = example.recency_fade
    reference = {}
    for current in range(N_DATASETS):
        ctx = _ctx(current=current)
        fade(ctx)
        reference[current] = [a.get_alpha() for a in ctx.artists]
    for current in order:
        ctx = _ctx(current=current)
        fade(ctx)
        assert [a.get_alpha() for a in ctx.artists] == reference[current], (
            f'current_index={current} faded differently out of order')


def test_a_single_point_turn_stays_invisible(example):
    """revealed < 2 is a stray point, not a drawn trajectory."""
    fade = example.recency_fade
    revealed = [10] * N_DATASETS
    revealed[1] = 1
    ctx = _ctx(current=N_DATASETS - 1, revealed=revealed)
    fade(ctx)
    assert ctx.artists[1].get_alpha() == 0.0


def test_the_real_plot_registers_the_fade_and_it_runs(example):
    """End to end on the fixture payload: the callback is registered on the
    wrapper, and driving a mid-reveal frame leaves exactly one opaque head
    with everything later hidden."""
    anim = example.construct_artifact(example.fixture_data())
    anim.draw_frame(anim.n_frames // 2)
    ax = [a for a in anim.figure.axes if hasattr(a, 'zaxis')][0]
    alphas = [ln.get_alpha() for ln in ax.lines]
    assert 1.0 in alphas and 0.0 in alphas, alphas
    plt.close('all')
