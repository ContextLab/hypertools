"""Pure mapping tests: no figures, no backends, no animation."""
from fractions import Fraction

import pytest

from hypertools.plot.forecast import (DatasetRevealSchedule,
                                      revealed_raw_counts)
from hypertools.plot.ownership import TraceOwnership
from hypertools.plot.trails import (RunWindow, anim_window_bounds,
                                    dataset_window_bounds, run_head_param)


def test_runs_of_one_dataset_partition_its_rows_in_order():
    # one dataset, categories A A A | B B B | A A A -> three runs
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3],
                                       [True, True, False])
    assert own.dataset_by_run == (0, 0, 0)
    assert own.source_rows_by_run == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert own.final_run_by_dataset == (2,)
    assert own.row_count(0) == 9
    assert own.runs_of(0) == (0, 1, 2)
    assert own.run_span(1) == (3, 3)


def test_a_bridged_run_DRAWS_one_more_row_than_it_OWNS():
    """`patch_lines` appends the next run's first observation. The owned rows
    and the drawn geometry are different lengths, and every consumer needs to
    know WHICH it is asking for -- v1 conflated them and mis-timed every
    category boundary by one vertex."""
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3],
                                       [True, True, False])
    assert own.run_span(0) == (0, 3)      # owns rows 0, 1, 2
    assert own.draw_span(0) == 3          # draws parameters 0..3 (row 3 bridged)
    assert own.draw_span(2) == 2          # final run: no bridge, 6..8


def test_two_datasets_keep_separate_row_numbering():
    """Row indices are per DATASET, not global: dataset 1's first row is 0,
    not 6. A global numbering would make `histories[i][rows]` slice the wrong
    rows in the forecast schedule."""
    own = TraceOwnership.from_segments([0, 0, 1], [4, 2, 5],
                                       [True, False, False])
    assert own.source_rows_by_run == ((0, 1, 2, 3), (4, 5), (0, 1, 2, 3, 4))
    assert own.final_run_by_dataset == (1, 2)
    assert own.row_count(0) == 6
    assert own.row_count(1) == 5
    # run 1 is the LAST run of dataset 0, so it is not bridged into dataset 1
    assert own.draw_span(1) == 1


def test_run_holding_finds_the_owner_of_an_original_row():
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3],
                                       [True, True, False])
    assert own.run_holding(0, 0) == 0
    assert own.run_holding(0, 2) == 0
    assert own.run_holding(0, 3) == 1
    assert own.run_holding(0, 8) == 2


def test_identity_ownership_is_one_run_per_dataset():
    """The unregrouped case. Every consumer takes the same code path, so the
    regrouped path cannot rot while the common one stays green."""
    own = TraceOwnership.identity([4, 7])
    assert own.dataset_by_run == (0, 1)
    assert own.source_rows_by_run == (tuple(range(4)), tuple(range(7)))
    assert own.final_run_by_dataset == (0, 1)
    assert own.bridged_by_run == (False, False)
    assert own.draw_span(0) == 3


def test_it_is_frozen_and_hashable():
    """It is memoized on and compared across frames; a mutable one would let
    a caller edit a schedule out from under a cache."""
    own = TraceOwnership.identity([2, 2])
    assert hash(own) == hash(TraceOwnership.identity([2, 2]))
    with pytest.raises(Exception):
        own.dataset_by_run = ()


def test_non_contiguous_runs_for_one_dataset_are_rejected():
    """`segment_by_run` emits runs dataset-major, so 0,1,0 cannot happen --
    but this class is importable on its own and the mapping it would build
    (dataset 0 owning rows 0,1 then 5,6) is silently wrong rather than
    obviously so."""
    with pytest.raises(ValueError, match='consecutive'):
        TraceOwnership.from_segments([0, 1, 0], [2, 2, 2],
                                     [False, False, False])


def test_sparse_or_unordered_dataset_ids_are_rejected():
    """`final_run_by_dataset` is INDEXED by dataset, and `runs_of`/`row_count`
    are called with `range(n_datasets)`. Ids of [1, 2] build a two-entry tuple
    whose entry 0 belongs to dataset 1 -- every later lookup is off by one,
    silently. `segment_by_run` always emits 0..n-1 in order; this class is
    independently callable and already validates its other invariants."""
    with pytest.raises(ValueError, match='start at 0'):
        TraceOwnership.from_segments([1, 1, 2], [2, 2, 2],
                                     [True, False, False])
    # starts at 0, so it reaches the SECOND check: `[1, 0]` would not, and a
    # first draft using it passed for the wrong reason
    with pytest.raises(ValueError, match='in order'):
        TraceOwnership.from_segments([0, 2, 1], [2, 2, 2],
                                     [False, False, False])


def test_a_length_mismatch_says_both_counts():
    with pytest.raises(ValueError, match='3 run'):
        TraceOwnership.from_segments([0, 0, 0], [2, 2], [True, True, False])
    with pytest.raises(ValueError, match='bridge'):
        TraceOwnership.from_segments([0, 0], [2, 2], [True])


def test_an_empty_run_is_rejected():
    """`segment_by_run` never emits one (a run exists because an observation
    started it), and a zero-length run would make `run_holding` ambiguous."""
    with pytest.raises(ValueError, match='no rows'):
        TraceOwnership.from_segments([0, 0], [3, 0], [True, False])


def test_a_dataset_s_LAST_run_cannot_be_bridged():
    """`patch_lines` is given `breaks` at every dataset boundary, so a bridge
    never crosses one -- a caller claiming otherwise would make `draw_span`
    describe geometry that was never drawn."""
    with pytest.raises(ValueError, match='last run'):
        TraceOwnership.from_segments([0, 1], [3, 3], [True, False])


def _one_dataset(lengths, n_frames):
    """Ownership + grid lengths for ONE dataset split into `lengths` runs."""
    bridge = [i < len(lengths) - 1 for i in range(len(lengths))]
    own = TraceOwnership.from_segments([0] * len(lengths), lengths, bridge)
    # `_interp_anim_line` leaves an array with fewer than 2 rows alone
    grids = [n_frames if (L + int(b)) >= 2 else 1
             for L, b in zip(lengths, bridge)]
    return own, grids


def _visible(own, grids, num, n_frames, w, dataset=0):
    """Visible rows DERIVED FROM the run windows -- the single source of
    truth this whole design turns on."""
    wins = dataset_window_bounds(num, n_frames, own, grids, w)
    best = None
    for r in own.runs_of(dataset):
        p = run_head_param(wins[r], own, r)
        if p is not None:
            best = p if best is None else max(best, p)
    if best is None:
        return ()
    return tuple(range(min(own.row_count(dataset), int(best) + 1)))


REGROUPED_CASES = [
    ([10, 10, 10], 12, 12), ([10, 10, 10], 40, 40), ([10, 10, 10], 120, 120),
    ([3, 3, 3], 12, 2), ([4, 1, 4], 20, 20), ([1, 1, 1, 1], 8, 8),
    ([7, 2, 11, 5], 30, 30), ([2, 2], 3, 3), ([5, 5, 5, 5, 5, 5], 60, 60),
    ([20, 10], 24, 24), ([1, 29], 12, 12), ([29, 1], 12, 12),
    ([50, 50], 24, 24), ([2, 26, 2], 12, 12),
    # whole ONE-ROW datasets: `span` is 0 for the dataset, not just for a run,
    # so every invariant below meets the degenerate projection too
    ([1], 6, 2), ([1], 12, 12), ([1, 1], 6, 2), ([1, 5], 10, 3),
]


@pytest.mark.parametrize('n_rows,n_frames,w', [
    (30, 12, 12), (9, 12, 2), (60, 60, 2), (5, 15, 2), (100, 24, 24),
    (2, 12, 12)])
def test_unregrouped_bounds_are_IDENTICAL_to_anim_window_bounds(
        n_rows, n_frames, w):
    """The load-bearing invariant. One run per dataset must project to the
    identity at EVERY frame, or fixing the regrouped case silently shifts
    every animation that was already correct."""
    own = TraceOwnership.identity([n_rows])
    for num in range(n_frames + 2):
        got, = dataset_window_bounds(num, n_frames, own, [n_frames], w)
        start, end, trail_stop = anim_window_bounds(num, n_frames, n_frames, w)
        assert (got.head_start, got.head_end, got.past_stop) == (
            start, end, trail_stop), f'frame {num}'
        # today's precog slice, `data[end - 1:]`, with no chance of `data[-1:]`
        assert got.future_start == max(0, end - 1), f'frame {num}'
        assert got.reached


@pytest.mark.parametrize('n_rows,n_frames,w', [
    (30, 12, 12), (9, 12, 2), (60, 60, 2), (5, 15, 2), (100, 24, 24)])
def test_unregrouped_visible_rows_equal_revealed_raw_counts(
        n_rows, n_frames, w):
    """The reveal schedule reads the same clock `ForecastSchedule.for_parallel`
    already reads, so an unregrouped forecast is bit-for-bit unchanged."""
    own = TraceOwnership.identity([n_rows])
    for num in range(n_frames):
        assert len(_visible(own, [n_frames], num, n_frames, w)) == \
            revealed_raw_counts(n_rows, n_frames, num, n_frames), f'frame {num}'


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_a_run_completes_exactly_when_the_NEXT_one_starts(
        lengths, n_frames, w):
    """The bridge contract. `patch_lines` put the next run's first observation
    on the end of this run's line, so that observation is on screen the moment
    this run finishes. The next run must expose its own copy of it at the SAME
    frame, or the two disagree about one vertex at every category boundary."""
    own, grids = _one_dataset(lengths, n_frames)
    for num in range(n_frames):
        wins = dataset_window_bounds(num, n_frames, own, grids, w)
        for r in range(len(lengths) - 1):
            done = wins[r].head_end == grids[r]
            started = wins[r + 1].head_end > 0
            assert done == started, (
                f'frame {num}: run {r} done={done}, run {r+1} '
                f'started={started}')


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_runs_of_one_dataset_reveal_IN_ORDER_not_together(
        lengths, n_frames, w):
    """The defect this task fixes: every run of one dataset used to grow
    simultaneously, so the trajectory animated in several places at once."""
    own, grids = _one_dataset(lengths, n_frames)
    for num in range(n_frames):
        wins = dataset_window_bounds(num, n_frames, own, grids, w)
        for r in range(1, len(lengths)):
            if wins[r].head_end > 0:
                assert wins[r - 1].head_end == grids[r - 1], (
                    f'frame {num}: run {r} started while run {r-1} was only '
                    f'{wins[r-1].head_end}/{grids[r-1]}')


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_visible_rows_are_a_growing_PREFIX_reaching_every_row(
        lengths, n_frames, w):
    own, grids = _one_dataset(lengths, n_frames)
    prev = -1
    for num in range(n_frames):
        rows = _visible(own, grids, num, n_frames, w)
        assert rows == tuple(range(len(rows))), f'frame {num}: {rows}'
        assert len(rows) >= prev, f'frame {num}: shrank {prev} -> {len(rows)}'
        prev = len(rows)
    assert prev == sum(lengths)


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_a_split_dataset_never_reveals_a_row_EARLY(lengths, n_frames, w):
    """Decision R2. A run's head can only stop on one of its own drawn
    vertices, which do not line up with the unsplit grid, so a regrouped
    reveal lags by up to one run-grid step. The DIRECTION is what matters:
    it must never lead, or a forecast could be fit on an observation the
    renderer has not drawn yet."""
    own, grids = _one_dataset(lengths, n_frames)
    n_rows = sum(lengths)
    whole = TraceOwnership.identity([n_rows])
    whole_grid = [n_frames if n_rows >= 2 else 1]   # same rule as _one_dataset
    # `default=0`: an UNSPLIT one-row dataset has no run grid to slide along,
    # so there is no step to lag by and the bound tightens to exact identity.
    step = max((Fraction(own.draw_span(r), grids[r] - 1)
                for r in range(len(lengths)) if grids[r] >= 2
                and own.draw_span(r) > 0), default=Fraction(0))
    for num in range(n_frames):
        split = len(_visible(own, grids, num, n_frames, w))
        unsplit = len(_visible(whole, whole_grid, num, n_frames, w))
        assert split <= unsplit, f'frame {num}: split LEADS unsplit'
        assert unsplit - split <= int(step) + 1, (
            f'frame {num}: lag {unsplit - split} exceeds one run-grid step '
            f'({float(step):.2f} rows)')


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_the_final_frame_draws_every_run_in_full(lengths, n_frames, w):
    own, grids = _one_dataset(lengths, n_frames)
    last = dataset_window_bounds(n_frames - 1, n_frames, own, grids, w)
    assert [win.head_end for win in last] == grids


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_no_window_ever_has_a_NEGATIVE_future_start(lengths, n_frames, w):
    """`data[end - 1:]` with `end == 0` is `data[-1:]` -- one stray point of a
    not-yet-reached category, sitting on screen from frame 0. Four named
    bounds exist so that slice cannot be written."""
    own, grids = _one_dataset(lengths, n_frames)
    for num in range(n_frames):
        for r, win in enumerate(dataset_window_bounds(
                num, n_frames, own, grids, w)):
            assert win.future_start >= 0, f'frame {num}, run {r}'


def test_an_unreached_run_is_EMPTY_head_EMPTY_past_WHOLE_future():
    """Decision R5. `precog` means the trajectory ahead of the head; a run the
    clock has not entered is entirely ahead of it."""
    own, grids = _one_dataset([5, 5], 20)
    win = dataset_window_bounds(0, 20, own, grids, 2)[1]
    assert win == RunWindow(head_start=0, head_end=0, past_stop=0,
                            future_start=0, reached=False, grid=grids[1])


def test_separate_datasets_still_advance_TOGETHER():
    """'Parallel' keeps its meaning across datasets; only runs WITHIN one
    dataset were ever meant to be sequential."""
    own = TraceOwnership.from_segments([0, 1], [10, 10], [False, False])
    for num in range(20):
        a, b = dataset_window_bounds(num, 20, own, [20, 20], 2)
        assert a == b, f'frame {num}'


def test_a_singleton_FINAL_run_appears_when_the_sweep_REACHES_it():
    """A 1-row unbridged run is not interpolated (`plot.py:4901` leaves arrays
    with fewer than 2 rows alone), so it has no grid to slide along: it is
    all-or-nothing, and 'all' must wait for the head rather than showing from
    frame 0 as it did when each run was paced on its own rows."""
    own, grids = _one_dataset([29, 1], 12)
    assert grids[1] == 1
    assert dataset_window_bounds(0, 12, own, grids, 12)[1].head_end == 0
    assert dataset_window_bounds(11, 12, own, grids, 12)[1].head_end == 1


@pytest.mark.parametrize('n_frames,w', [
    (1, 0), (2, 1), (3, 2), (6, 2), (6, 6), (12, 5), (12, 12)])
def test_a_ONE_ROW_DATASET_is_the_unregrouped_identity_at_EVERY_frame(
        n_frames, w):
    """The test above covers a one-row RUN inside a longer dataset. A whole
    one-row DATASET is the harder degenerate case: `span` is 0 for the dataset
    as well as for the run, so the projection has no extent to project ONTO.
    It must still be the identity, at the first frame and the last."""
    own = TraceOwnership.identity([1])
    for num in range(n_frames):
        got, = dataset_window_bounds(num, n_frames, own, [1], w)
        start, end, trail_stop = anim_window_bounds(num, n_frames, 1, w)
        assert (got.head_start, got.head_end, got.past_stop) == (
            start, end, trail_stop), f'frame {num}'
        assert got.future_start == max(0, end - 1), f'frame {num}'


def test_a_ONE_ROW_dataset_beside_LONGER_ones_keeps_its_own_clock():
    """R1 is per SOURCE DATASET, so a singleton must not be dragged along by a
    longer neighbour's grid, nor drag the neighbour back to its own."""
    own = TraceOwnership.identity([5, 1, 3])
    grids = [12, 1, 12]
    for num in range(12):
        wins = dataset_window_bounds(num, 12, own, grids, 2)
        for r, g in enumerate(grids):
            start, end, trail_stop = anim_window_bounds(num, 12, g, 2)
            assert (wins[r].head_start, wins[r].head_end,
                    wins[r].past_stop) == (start, end, trail_stop), (
                        f'frame {num}, dataset {r}')


def test_a_ONE_ROW_dataset_is_ON_SCREEN_from_frame_0():
    """Deliberate, not incidental. `anim_window_bounds` clamps `end` to at
    least 1 (`trails.py:86`), so a single-point dataset is drawn from the first
    frame -- audited behaviour (F05-012), and NOT the `end == 0` state
    `RunWindow` exists to name. There is therefore no 'before the singleton is
    reached' frame to test: `reached` is True throughout, and a `precog` trail
    on it is its whole (one-point) self. A change that hid it until the end
    would be a regression, so it must fail here rather than pass quietly."""
    own = TraceOwnership.identity([1])
    for num in range(6):
        got, = dataset_window_bounds(num, 6, own, [1], 2)
        assert got.reached and got.head_end == 1, f'frame {num}'
        assert got.future_start == 0, f'frame {num}'


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_reached_means_EXACTLY_that_the_head_is_NON_EMPTY(
        lengths, n_frames, w):
    """`reached` must stay a RESTATEMENT of `head_end`, never a second
    derivation of it. Comparing the head parameter against the run's first row
    also works today, but only because `_param` returns 0 for a dataset with no
    extent and the first run's `first_row` is 0 as well -- a degenerate value
    coinciding with a real boundary. This fails the moment the two disagree
    anywhere, which is what the equivalent-by-accident version could not."""
    own, grids = _one_dataset(lengths, n_frames)
    for num in range(n_frames):
        for r, win in enumerate(dataset_window_bounds(
                num, n_frames, own, grids, w)):
            assert win.reached == (win.head_end > 0), f'frame {num}, run {r}'
            assert win.future_start == max(0, win.head_end - 1)


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_the_schedule_agrees_with_the_RENDERED_windows_at_every_frame(
        lengths, n_frames, w):
    """The cross-invariant. Without it the reveal schedule and the renderer
    can drift while each passes its own tests -- and the symptom would be a
    forecast fit on an observation that is (or is not) on screen."""
    own, grids = _one_dataset(lengths, n_frames)
    sched = DatasetRevealSchedule(own, grids, n_frames, w)
    for frame in range(n_frames):
        assert sched.visible_rows(0, frame) == _visible(
            own, grids, frame, n_frames, w), f'frame {frame}'


@pytest.mark.parametrize('lengths,n_frames,w', REGROUPED_CASES)
def test_the_head_run_is_the_run_DRAWING_the_last_visible_row(
        lengths, n_frames, w):
    own, grids = _one_dataset(lengths, n_frames)
    sched = DatasetRevealSchedule(own, grids, n_frames, w)
    for frame in range(n_frames):
        rows = sched.visible_rows(0, frame)
        run = sched.head_run(0, frame)
        if not rows:
            assert run is None, f'frame {frame}'
        else:
            assert run == own.run_holding(0, rows[-1]), f'frame {frame}'


def test_the_head_run_ADVANCES_at_a_category_boundary():
    """Decision R3 depends on this changing: if `head_run` were constant the
    live forecast could never take the new category's colour."""
    own, grids = _one_dataset([10, 10, 10], 12)
    sched = DatasetRevealSchedule(own, grids, 12, 12)
    assert sorted({sched.head_run(0, f) for f in range(12)}) == [0, 1, 2]


def test_visible_rows_are_a_PREFIX_of_the_dataset():
    """The property the fixed reveal buys us, asserted as an invariant rather
    than assumed: `(0, 3, 6)` -- a sample spanning the whole trajectory -- is
    what the old reveal produced, and it is what a forecast must never see."""
    own, grids = _one_dataset([3, 3, 3], 12)
    sched = DatasetRevealSchedule(own, grids, 12, 2)
    for frame in range(12):
        rows = sched.visible_rows(0, frame)
        assert rows == tuple(range(len(rows))), f'frame {frame}: {rows}'


def test_the_last_frame_sees_the_WHOLE_history():
    own, grids = _one_dataset([3, 3, 3], 12)
    sched = DatasetRevealSchedule(own, grids, 12, 2)
    assert sched.visible_rows(0, 11) == tuple(range(9))


def test_frames_past_the_end_clamp_instead_of_raising():
    """matplotlib renders one frame past the end on a loop or a save."""
    own = TraceOwnership.identity([9])
    sched = DatasetRevealSchedule(own, [12], 12, 2)
    assert sched.visible_rows(0, 99) == sched.visible_rows(0, 11)
    assert sched.visible_rows(0, -5) == sched.visible_rows(0, 0)


def test_an_UNREGROUPED_schedule_matches_revealed_raw_counts():
    """The path an unregrouped animated forecast already takes, so switching
    every animation onto this class changes nothing about them."""
    own = TraceOwnership.identity([30])
    sched = DatasetRevealSchedule(own, [12], 12, 12)
    assert [len(sched.visible_rows(0, f)) for f in range(12)] == [
        revealed_raw_counts(30, 12, f, 12) for f in range(12)]


def test_the_serial_schedule_reveals_datasets_one_at_a_time():
    own = TraceOwnership.from_segments([0, 1], [6, 6], [False, False])
    sched = DatasetRevealSchedule(own, [12, 12], 12, 2, serial=True)
    assert sched.visible_rows(1, 0) == ()
    assert len(sched.visible_rows(0, 11)) == 6
    assert len(sched.visible_rows(1, 11)) == 6
