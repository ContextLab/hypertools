"""Pure mapping tests: no figures, no backends, no animation."""
import pytest

from hypertools.plot.ownership import TraceOwnership


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
