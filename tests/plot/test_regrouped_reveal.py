"""A regrouped trajectory must animate in source-row order.

Real figures through the public API -- these assert what a viewer sees.
"""
import contextlib
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


@contextlib.contextmanager
def only_warning(match=None):
    """Record and assert EXACTLY what was warned, never `simplefilter('ignore')`.

    The suite's standing gate is zero warnings; a blanket ignore inside a test
    lets a NEW product warning through silently, which is the failure mode the
    gate exists to catch. `match=None` means none at all; a pattern means that
    one and nothing else -- `pytest.warns(...)` alone would assert the named
    warning arrived while letting an unexpected SECOND one pass unnoticed, so
    the assertion is on the whole recorded list.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught
    messages = [str(w.message) for w in caught]
    if match is None:
        assert not messages, messages
    else:
        assert len(messages) == 1 and re.search(match, messages[0]), messages


def no_warnings():
    """No warning at all -- the default for every fixture here."""
    return only_warning(None)


def _walk(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n, 3), 0)


def _run_lengths(fig, ani, frame):
    ani._func(frame, *ani._args)
    return [len(line.get_xdata()) for line in fig.axes[0].lines]


def _animate(data, fmt='-', warns=None, **kwargs):
    # `fmt` is a parameter because the marker-only control below needs 'o':
    # that case takes a DIFFERENT regrouping path (`reshape_data`, grouping
    # globally by category) and is the one this feature must leave alone.
    # `warns` names the ONE warning a fixture is expected to provoke.
    with only_warning(warns):
        return hyp.plot(data, fmt, animate=True, duration=2, frame_rate=6,
                        show=False, **kwargs)


def test_a_regrouped_trajectory_sweeps_ONCE_not_three_times():
    """The defect: all three runs of one dataset used to grow together, so
    the same trajectory animated at times ~0-3, ~10-13 and ~20-23 at once."""
    fig, ani = _animate([_walk()], hue=HUE)
    early = _run_lengths(fig, ani, 3)
    assert early[0] > 0, 'the first run should be under way'
    assert early[1] == 0 and early[2] == 0, (
        f'later runs must not have started: {early}')


def test_a_later_run_starts_only_once_the_previous_one_FINISHES():
    fig, ani = _animate([_walk()], hue=HUE)
    full = _run_lengths(fig, ani, 11)
    for frame in range(12):
        drawn = _run_lengths(fig, ani, frame)
        for r in range(1, 3):
            if drawn[r] > 0:
                assert drawn[r - 1] == full[r - 1], (
                    f'frame {frame}: run {r} started at {drawn[r]} while run '
                    f'{r - 1} was only {drawn[r - 1]}/{full[r - 1]}')


def test_the_final_frame_still_draws_EVERYTHING():
    fig, ani = _animate([_walk()], hue=HUE)
    assert all(n > 0 for n in _run_lengths(fig, ani, 11))


def test_an_UNREGROUPED_animation_is_unchanged_row_for_row():
    """The control. Task 2's projection is the identity without regrouping,
    so this must match the pre-change behaviour exactly -- if it drifts, the
    fix leaked into every animation rather than only the regrouped ones."""
    fig, ani = _animate([_walk()])
    assert [_run_lengths(fig, ani, f)[0] for f in range(12)] == [
        1, 83, 165, 247, 329, 411, 493, 575, 657, 739, 821, 903]


def test_two_datasets_still_advance_together():
    fig, ani = _animate([_walk(), _walk(seed=1)])
    a, b = _run_lengths(fig, ani, 5)
    assert a == b


def test_a_2D_regrouped_animation_sweeps_in_order_too():
    """`update_lines_parallel_2d` is a separate updater with its own copy of
    the window call (matplotlib_backend.py:2080)."""
    rng = np.random.RandomState(2)
    fig, ani = _animate([np.cumsum(rng.randn(30, 2), 0)], hue=HUE)
    early = _run_lengths(fig, ani, 3)
    assert early[1] == 0 and early[2] == 0, early


def test_a_precog_trail_on_an_unreached_run_is_not_ONE_STRAY_POINT():
    """`data[end - 1:]` with `end == 0` is `data[-1:]`. A run the sweep has
    not reached must show its WHOLE future, not its last vertex alone
    (Decision R5).

    The trail artists come from the updater's own return value, which is
    `(head_lines, trail_lines)`. Selecting them by `_hyp_row_window` instead
    -- as a first draft did -- selects the HEADS too, because `_aa_window`
    stamps that attribute on every artist it windows. The heads at frame 0
    are `[1, 0, 0]`: run 0 legitimately shows its single first point, so a
    "no artist has exactly one point" assertion over that mixed list fails on
    correct behaviour and says nothing about precog at all.
    """
    fig, ani = _animate([_walk()], hue=HUE, precog=True)
    # the HEADS at the last frame, not `_run_lengths`, which also returns the
    # trail artists -- and a precog trail at the last frame is correctly ONE
    # point, the head's own final vertex it shares to avoid a gap (F05-008)
    final_heads, _ = ani._func(11, *ani._args)
    whole = [len(h.get_xdata()) for h in final_heads]
    heads, trails = ani._func(0, *ani._args)
    assert trails and all(t is not None for t in trails)
    lengths = [len(t.get_xdata()) for t in trails]
    assert all(n != 1 for n in lengths), lengths
    # stronger, and the actual promise of R5: what lies ahead of the head is
    # the whole of every run the clock has not entered, not a stray vertex
    assert lengths == whole, (lengths, whole)
    assert [len(h.get_xdata()) for h in heads] == [1, 0, 0], (
        'frame 0 must show only the first run, one point in')


def test_a_MARKER_only_hue_plot_is_untouched():
    """Marker regrouping groups globally by category through `reshape_data`,
    with no per-dataset row order to sweep; it must keep today's behaviour
    rather than be handed an ownership describing datasets that do not
    exist."""
    fig, ani = _animate([_walk()], 'o', hue=HUE)
    drawn = _run_lengths(fig, ani, 3)
    assert all(n > 0 for n in drawn), drawn


def _plotly(data, warns=None, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        with only_warning(warns):
            return hyp.plot(data, '-', animate=True, duration=2, frame_rate=6,
                            show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _plotly_counts(fig, frame_index):
    # `0 if d.x is None else len(d.x)`, not `len(d.x or ())`: plotly stores
    # a frame's coordinates as a numpy array, and `array or ()` raises
    # "truth value of an array ... is ambiguous" rather than falling back.
    frame = fig.frames[frame_index]
    drawn = {t: (0 if d.x is None else len(d.x))
             for t, d in zip(frame.traces, frame.data)}
    return [drawn[t] for t in sorted(drawn)]


def test_plotly_reveals_regrouped_runs_in_the_same_order():
    pytest.importorskip('plotly')
    counts = _plotly_counts(_plotly([_walk()], hue=HUE), 3)
    assert counts[1] == 0 and counts[2] == 0, counts


@pytest.mark.parametrize('hue_arg,label,warns', [
    (HUE, 'three runs', None),
    # a one-observation hue category cannot be rendered by a pure line format,
    # and the library says so -- expected here, and identically from BOTH
    # backends, which is itself part of what this test checks
    (['A'] * 29 + ['B'], 'a singleton final run', 'only one observation'),
    (['A'] * 2 + ['B'] * 26 + ['A'] * 2, 'unequal run lengths', None),
    (None, 'unregrouped', None),
])
def test_plotly_and_matplotlib_draw_the_SAME_row_counts(hue_arg, label, warns):
    """Both backends consume `dataset_window_bounds`; a transcription drift
    between them is exactly what the `trails` module exists to prevent. An
    empty/non-empty comparison would miss an off-by-one or a mis-scaled
    window, so compare the exact vertex counts, at every frame."""
    pytest.importorskip('plotly')
    kw = {} if hue_arg is None else {'hue': hue_arg}
    pfig = _plotly([_walk()], warns=warns, **kw)
    mfig, ani = _animate([_walk()], warns=warns, **kw)
    for f in range(12):
        assert _plotly_counts(pfig, f) == _run_lengths(mfig, ani, f), (
            f'{label}, frame {f}')


def test_plotly_matches_matplotlib_at_the_BOUNDARY_and_FINAL_frames():
    """The two frames the projection is most likely to get wrong: the one a
    category boundary lands on, and the last."""
    pytest.importorskip('plotly')
    pfig = _plotly([_walk()], hue=HUE)
    mfig, ani = _animate([_walk()], hue=HUE)
    boundary = next(f for f in range(12)
                    if _run_lengths(mfig, ani, f)[1] > 0)
    for f in (boundary - 1, boundary, boundary + 1, 11):
        assert _plotly_counts(pfig, f) == _run_lengths(mfig, ani, f), f'frame {f}'
