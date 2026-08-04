# Regrouped Reveal + Animated Regrouped Forecasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `hue=`/`cluster=` regrouped trajectory animate in source-row order under the default parallel reveal, then draw per-frame forecasts over regrouped animations (the case that is currently warning-only).

**Architecture:** One new pure module, `hypertools/plot/ownership.py`, records which source dataset and which original rows each drawn run came from. `trails.dataset_window_bounds` uses it to drive every run of one dataset from a SINGLE clock, projected onto each run's own grid — so the head sweeps a trajectory once instead of growing in every run simultaneously. Once the reveal is fixed, a dataset's visible rows are a contiguous prefix again, which is the condition the existing `ForecastSchedule` already assumes; the animated-regrouping refusal at `plot.py:5051` is then replaced by a real overlay.

**Tech Stack:** numpy, scipy (PCHIP, already used), matplotlib, plotly, pytest.

## Why this plan exists (measured, not assumed)

`hyp.plot([x], '-', hue=['A']*10+['B']*10+['A']*10, animate=True, duration=2, frame_rate=6)`
drives one 30-row dataset through `segment_by_run`, producing three runs. Points
drawn per run-line, per frame, on `dev-1.0` at `a062f768`:

| frame | run A (rows 0-9) | run B (rows 10-19) | run A' (rows 20-29) |
|-|-|-|-|
| 0 | 1 | 1 | 1 |
| 3 | 247 | 247 | 247 |
| 6 | 493 | 493 | 493 |
| 11 | 903 | 903 | 903 |

Every run advances at the same rate through its OWN rows, because
`plot._interp_anim_line` resamples each drawn trace onto the same `n_frames`
grid and `trails.anim_window_bounds` then reveals `ceil((num+1) * n_points /
total)` of it. The viewer sees three disconnected segments growing at
trajectory times ~0-3, ~10-13 and ~20-23 at once. The same data without `hue=`
sweeps once (`[247]`), and with `order='serial'` also sweeps correctly
(`[657, 0, 0]` -> `[903, 493, 0]` -> `[903, 903, 329]`), because
`serial_reveal_counts` walks the runs in order.

The forecast consequence, measured with `revealed_raw_counts` over the three
runs: dataset 0's visible source rows are `(0, 3, 6)` at frame 0 and
`(0, 1, 3, 4, 6, 7)` at frame 6 — never a prefix, and spanning the whole
timeline from the first frame. Any "forecast from what has been revealed"
fitted on that already knows the endpoint.

**Decision taken (Jeremy, 2026-08-03):** fix the reveal first, in the same
plan, so forecasts are built on a correct sweep.

## Global Constraints

- Python >= 3.10 (`requires-python = ">=3.10"`).
- Run every command with `.venv/bin/python`. The system python's numpy breaks matplotlib.
- Never use `git stash` in this repo (documented data-loss hazard). Use `git show <ref>:<path>`.
- No mock objects and no mock tests, ever — including as fallbacks. If real functionality cannot be exercised, the test must fail or raise.
- `pytest` runs from the repo root (`pyproject.toml` sets `testpaths = ["tests"]`).
- Both backends must consume the SAME reveal arithmetic. `trails.py` owns it; neither backend may re-derive it. This is a standing rule in the `trails` module docstring, which records a plotly transcription drift that blanked a 5-row dataset for 9 of its 15 frames.
- The full suite must stay at zero warnings: it currently emits no `warnings summary` section at all. A new warning is a failure.
- `cd docs && make html` must build clean under `sphinx -W` (warnings are errors).
- Any behavior change to a released 1.0 API goes in `CHANGELOG.md`.

---

## File Structure

| File | Responsibility |
|-|-|
| `hypertools/plot/ownership.py` (new) | `TraceOwnership`: run -> source dataset, run -> original row indices, dataset -> final run. Pure data; no numpy beyond `asarray`, no plotting imports. |
| `hypertools/plot/trails.py` (modify) | Add `dataset_window_bounds`, the ONE frame->rows mapping for regrouped parallel animation. Existing `anim_window_bounds` is unchanged and becomes its per-dataset clock. |
| `hypertools/plot/matplotlib_backend.py` (modify) | 3-D (`:1185`) and 2-D (`:2079`) parallel updaters call `dataset_window_bounds` when ownership is present. |
| `hypertools/plot/plotly_backend.py` (modify) | Head (`:3445`) and trail (`:3484`) window computations do the same. |
| `hypertools/plot/forecast.py` (modify) | `DatasetRevealSchedule` wrapping `TraceOwnership` + the reveal, exposing `visible_rows(dataset, frame)`; `ForecastSchedule.for_regrouped`. |
| `hypertools/plot/plot.py` (modify) | Build `TraceOwnership` at segmentation; thread it to both backends; replace the animated-regrouping refusal at `:5051-5069` with a real schedule. |
| `tests/plot/test_trace_ownership.py` (new) | Pure mapping tests for `TraceOwnership` and `dataset_window_bounds`. No figures. |
| `tests/plot/test_regrouped_reveal.py` (new) | Real `hyp.plot(...)` animations: sweep order, both backends, both reveal orders. |
| `tests/plot/test_forecast_animated_regrouped.py` (new) | Forecasts over regrouped animations, both backends. |

---

## Contracts this plan establishes

```python
# hypertools/plot/ownership.py
@dataclass(frozen=True)
class TraceOwnership:
    dataset_by_run: tuple          # run -> source dataset index
    source_rows_by_run: tuple      # run -> tuple of ORIGINAL row indices
    final_run_by_dataset: tuple    # dataset -> its last run

    n_runs: int                    # property
    n_datasets: int                # property
    def runs_of(self, dataset) -> tuple: ...
    def row_count(self, dataset) -> int: ...
    def run_span(self, run) -> tuple:            # (first_row, n_rows)
    def run_holding(self, dataset, row) -> int:  # which run owns an original row
    @classmethod
    def from_segments(cls, seg_dataset, seg_lengths) -> 'TraceOwnership': ...
    @classmethod
    def identity(cls, dataset_lengths) -> 'TraceOwnership': ...

# hypertools/plot/trails.py
def dataset_window_bounds(num, total_frames, ownership, grid_lengths,
                          window_frames) -> list:   # one (start, end, trail_stop) per RUN

# hypertools/plot/forecast.py
class DatasetRevealSchedule:
    def __init__(self, ownership, grid_lengths, n_frames, serial=False): ...
    def visible_rows(self, dataset, frame) -> tuple: ...   # ORIGINAL row indices, sorted
    def head_run(self, dataset, frame) -> int: ...         # run holding the last visible row

class ForecastSchedule:
    @classmethod
    def for_regrouped(cls, histories, reveal, model, t, n_frames,
                      min_history=DEFAULT_MIN_HISTORY,
                      slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS): ...
```

## Named decisions

**Decision R1 — the reveal clock is per SOURCE DATASET, not per drawn run.**
Settled by Jeremy 2026-08-03. Runs of one input dataset share one clock; runs
of different datasets still advance together (parallel keeps its meaning).

**Decision R2 — projection is exact for the unregrouped case.** The dataset
clock is computed by calling the existing `anim_window_bounds` with the
dataset's REFERENCE GRID LENGTH (`max` of its runs' grid lengths, which is
`n_frames` for every interpolated line trace), then projected onto each run.
With one run per dataset the projection is the identity, so no unregrouped
animation changes by a single row. Task 2 pins this with a test that sweeps
every frame.

**Decision R3 — a live forecast inherits the style of the run holding the
head, which can change mid-animation.** The forecast visually continues the
head, so it takes that trace's colour; when the reveal crosses a category
boundary the forecast changes colour with it. `forecast_hue=`/
`forecast_cluster=`/`forecast_palette=` override this and are already fixed for
the whole animation (see `docs/animation.rst` and
`resolve_forecast_overrides`' docstring), so a user who wants a constant
forecast colour has three ways to ask for one. Task 6 pins both behaviours.

**Decision R4 — `visible_rows` returns a tuple of original row indices even
though the fixed reveal always makes it a prefix.** The maintainer's requested
interface, and it keeps the memoization key honest: `(dataset,
visible_row_tuple)` stays correct if the reveal ever exposes a non-prefix
again, where `(dataset, count)` would silently collide. Task 5 asserts the
prefix property as a separate invariant, so a regression shows up as a failing
invariant rather than as wrong colours.

---

## Task 1: `TraceOwnership` — the run -> dataset -> original rows mapping

**Files:**
- Create: `hypertools/plot/ownership.py`
- Test: `tests/plot/test_trace_ownership.py`

**Interfaces:**
- Consumes: `segment_by_run`'s `seg_dataset` (list of int, one per run) and the PRE-`patch_lines` run lengths.
- Produces: the `TraceOwnership` API in "Contracts" above.

**Background the implementer needs:** `hypertools/_shared/helpers.py:278`
`segment_by_run(x, hue, labels)` walks each input dataset in order and cuts it
into maximal same-category runs, returning `seg_dataset` (each run's source
dataset). It does NOT return row indices — but it does not need to: within a
dataset the runs partition its rows contiguously, in order, so run lengths are
enough to recover them. Note that `patch_lines` (called immediately after, at
`plot.py:364`) APPENDS the next run's first point to each bridged run, so a
drawn run's array is one row longer than its source span. `TraceOwnership` must
be built from the PRE-`patch_lines` lengths; the bridge row belongs to the next
run.

- [ ] **Step 1: Write the failing tests**

```python
"""Pure mapping tests: no figures, no backends, no animation."""
import pytest

from hypertools.plot.ownership import TraceOwnership


def test_runs_of_one_dataset_partition_its_rows_in_order():
    # one dataset, categories A A A | B B B | A A A -> three runs
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    assert own.dataset_by_run == (0, 0, 0)
    assert own.source_rows_by_run == ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    assert own.final_run_by_dataset == (2,)
    assert own.row_count(0) == 9
    assert own.runs_of(0) == (0, 1, 2)
    assert own.run_span(1) == (3, 3)


def test_two_datasets_keep_separate_row_numbering():
    """Row indices are per DATASET, not global: dataset 1's first row is 0,
    not 9. A global numbering would make `histories[i][:k]` slice the wrong
    rows in the forecast schedule."""
    own = TraceOwnership.from_segments([0, 0, 1], [4, 2, 5])
    assert own.source_rows_by_run == ((0, 1, 2, 3), (4, 5), (0, 1, 2, 3, 4))
    assert own.final_run_by_dataset == (1, 2)
    assert own.row_count(0) == 6
    assert own.row_count(1) == 5


def test_run_holding_finds_the_owner_of_an_original_row():
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
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
        TraceOwnership.from_segments([0, 1, 0], [2, 2, 2])


def test_a_length_mismatch_says_both_counts():
    with pytest.raises(ValueError, match='3 run'):
        TraceOwnership.from_segments([0, 0, 0], [2, 2])


def test_an_empty_run_is_rejected():
    """`segment_by_run` never emits one (a run exists because an observation
    started it), and a zero-length run would make `run_holding` ambiguous."""
    with pytest.raises(ValueError, match='no rows'):
        TraceOwnership.from_segments([0, 0], [3, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hypertools.plot.ownership'`

- [ ] **Step 3: Write the implementation**

Create `hypertools/plot/ownership.py`:

```python
"""Which drawn run came from which source dataset, and from which of its rows.

`hue=`/`cluster=` replace the one-trace-per-dataset correspondence with one
trace per contiguous same-category RUN (`_shared.helpers.segment_by_run`).
Everything that has to reason about a DATASET after that -- the animation's
reveal clock, a forecast's history, which trace a forecast attaches to --
needs the inverse mapping, and needs it in ORIGINAL row indices rather than
run-local ones.

Deliberately pure and free of plotting imports: it is the thing both backends
and the forecast schedule agree on, so it must be testable without drawing
anything.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceOwnership:
    """Run -> dataset -> original rows, for one figure.

    Attributes
    ----------
    dataset_by_run : tuple of int
        Each run's source input-dataset index.
    source_rows_by_run : tuple of tuple of int
        Each run's original row indices WITHIN ITS OWN DATASET (dataset 1's
        first row is 0, not the global stacked offset). Excludes the bridge
        row `patch_lines` appends to a bridged run: that row belongs to the
        run it came from.
    final_run_by_dataset : tuple of int
        Each dataset's LAST run -- the one holding its final observation, and
        so the trace a static forecast continues.
    """

    dataset_by_run: tuple
    source_rows_by_run: tuple
    final_run_by_dataset: tuple

    @property
    def n_runs(self):
        return len(self.dataset_by_run)

    @property
    def n_datasets(self):
        return len(self.final_run_by_dataset)

    def runs_of(self, dataset):
        """Every run this dataset produced, in order."""
        return tuple(r for r, d in enumerate(self.dataset_by_run)
                     if d == dataset)

    def row_count(self, dataset):
        """How many original rows this dataset has."""
        return sum(len(self.source_rows_by_run[r])
                   for r in self.runs_of(dataset))

    def run_span(self, run):
        """``(first_row, n_rows)`` of a run, in its dataset's row numbering."""
        rows = self.source_rows_by_run[run]
        return rows[0], len(rows)

    def run_holding(self, dataset, row):
        """The run owning original `row` of `dataset`."""
        for r in self.runs_of(dataset):
            if row in self.source_rows_by_run[r]:
                return r
        raise ValueError(
            f"dataset {dataset} has no row {row} (it has "
            f"{self.row_count(dataset)}).")

    @classmethod
    def from_segments(cls, seg_dataset, seg_lengths):
        """Build from `segment_by_run`'s `seg_dataset` and the PRE-`patch_lines`
        run lengths.

        `patch_lines` appends the NEXT run's first point to every bridged run,
        so a drawn run's array is one row longer than the span it owns. Passing
        post-bridge lengths here would hand every bridged run one row of its
        neighbour's data -- which is exactly the double-counting that makes a
        forecast history wrong by one observation per category boundary.
        """
        seg_dataset = [int(d) for d in seg_dataset]
        seg_lengths = [int(n) for n in seg_lengths]
        if len(seg_dataset) != len(seg_lengths):
            raise ValueError(
                f"one length per run is needed; got {len(seg_dataset)} run "
                f"dataset(s) and {len(seg_lengths)} length(s).")
        bad = [r for r, n in enumerate(seg_lengths) if n <= 0]
        if bad:
            raise ValueError(
                f"every run holds at least one observation (a run exists "
                f"because an observation started it); run(s) {bad} have no "
                f"rows.")

        seen_order = []
        for d in seg_dataset:
            if d not in seen_order:
                seen_order.append(d)
        for d in seen_order:
            runs = [r for r, dd in enumerate(seg_dataset) if dd == d]
            if runs != list(range(runs[0], runs[-1] + 1)):
                raise ValueError(
                    f"a dataset's runs must be consecutive (segment_by_run "
                    f"emits them dataset by dataset); dataset {d} owns runs "
                    f"{runs}.")

        rows_by_run, next_row = [], {}
        for d, n in zip(seg_dataset, seg_lengths):
            start = next_row.get(d, 0)
            rows_by_run.append(tuple(range(start, start + n)))
            next_row[d] = start + n

        final = {}
        for r, d in enumerate(seg_dataset):
            final[d] = r                       # last write wins
        return cls(tuple(seg_dataset), tuple(rows_by_run),
                   tuple(final[d] for d in sorted(final)))

    @classmethod
    def identity(cls, dataset_lengths):
        """The UNREGROUPED case: one run per dataset, holding all its rows.

        Every consumer takes the same code path whether or not `hue=` split
        anything, so the regrouped path cannot quietly rot while the common
        one stays green.
        """
        lengths = [int(n) for n in dataset_lengths]
        return cls.from_segments(list(range(len(lengths))), lengths)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/ownership.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): TraceOwnership maps drawn runs back to source rows"
```

---

## Task 2: `dataset_window_bounds` — one reveal clock per dataset

**Files:**
- Modify: `hypertools/plot/trails.py` (add after `anim_window_bounds`, which ends at line 94; extend `__all__` at line 21)
- Test: `tests/plot/test_trace_ownership.py` (append)

**Interfaces:**
- Consumes: `TraceOwnership` (Task 1); `anim_window_bounds(num, total_frames, n_points, window_frames) -> (start, end, trail_stop)`.
- Produces: `dataset_window_bounds(num, total_frames, ownership, grid_lengths, window_frames) -> list of (start, end, trail_stop)`, one per RUN, indexed by run.

**The arithmetic, stated once.** For dataset `d`:

1. Its reference grid length is `G_ref = max(grid_lengths[r] for r in runs_of(d))`. Every interpolated line run is resampled to exactly `n_frames` rows by `plot._interp_anim_line`, so all of a dataset's line runs share one value; `max` ignores a 1-point run that was left un-interpolated.
2. Its clock is `anim_window_bounds(num, total_frames, G_ref, window_frames)` — the existing function, unchanged, on the length the dataset WOULD have had unsplit.
3. Each of the three positions is a fraction `u = v / G_ref` of the dataset. For run `r` spanning `(a, L)` of `N = row_count(d)` rows, with its own grid length `G_r`, the local fraction is `f = clip((u * N - a) / L, 0, 1)` and the projected position is `0 if f <= 0 else min(G_r, max(1, ceil(f * G_r)))`.

With one run per dataset (`a = 0`, `L = N`, `G_r = G_ref`) this reduces to
`ceil(v)`, and `v` is already an integer — the identity. That is Decision R2,
and Step 1's first test sweeps every frame to prove it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/plot/test_trace_ownership.py`:

```python
from hypertools.plot.trails import anim_window_bounds, dataset_window_bounds


def test_unregrouped_bounds_are_IDENTICAL_to_anim_window_bounds():
    """The load-bearing invariant. One run per dataset must project to the
    identity at EVERY frame, or fixing the regrouped case silently shifts
    every animation that was already correct."""
    own = TraceOwnership.identity([30, 12])
    grids = [40, 40]
    for num in range(40):
        got = dataset_window_bounds(num, 40, own, grids, window_frames=2)
        want = [anim_window_bounds(num, 40, g, 2) for g in grids]
        assert got == want, f'frame {num}'


def test_runs_of_one_dataset_reveal_IN_ORDER_not_together():
    """The defect this task fixes: three runs of one 9-row dataset used to
    grow simultaneously, so the trajectory animated in three places at once."""
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    grids = [12, 12, 12]
    ends = [[e for _, e, _ in dataset_window_bounds(f, 12, own, grids, 2)]
            for f in range(12)]
    # run 1 must not start before run 0 has finished
    for row in ends:
        assert row[1] == 0 or row[0] == grids[0], row
        assert row[2] == 0 or row[1] == grids[1], row
    assert ends[0][1:] == [0, 0]        # only the first run is moving
    assert ends[-1] == [12, 12, 12]     # everything is revealed at the end


def test_every_run_is_fully_revealed_on_the_final_frame():
    own = TraceOwnership.from_segments([0, 0, 1], [4, 2, 5])
    grids = [20, 20, 20]
    last = dataset_window_bounds(19, 20, own, grids, 2)
    assert [e for _, e, _ in last] == [20, 20, 20]


def test_separate_datasets_still_advance_TOGETHER():
    """"Parallel" keeps its meaning across datasets; only runs WITHIN one
    dataset were ever meant to be sequential."""
    own = TraceOwnership.from_segments([0, 1], [10, 10])
    grids = [20, 20]
    for f in range(20):
        a, b = dataset_window_bounds(f, 20, own, grids, 2)
        assert a == b, f'frame {f}'


def test_a_run_that_has_not_started_reveals_NOTHING():
    """`end == 0` must mean an empty slice. `data[0:1]` would leave a stray
    point of a future category on screen from frame 0 -- visible as a dot
    sitting where the trajectory has not reached yet."""
    own = TraceOwnership.from_segments([0, 0], [5, 5])
    starts, ends, _ = zip(*dataset_window_bounds(0, 20, own, [20, 20], 2))
    assert ends[1] == 0
    assert starts[1] == 0


def test_a_singleton_run_keeps_its_single_row():
    """A 1-point run is not interpolated (`plot.py:4901` leaves arrays with
    fewer than 2 rows alone), so its grid length is 1 while its siblings' is
    n_frames."""
    own = TraceOwnership.from_segments([0, 0, 0], [4, 1, 4])
    bounds = dataset_window_bounds(19, 20, own, [20, 1, 20], 2)
    assert bounds[1][1] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q -k window or singleton or reveal or advance`
Expected: FAIL — `ImportError: cannot import name 'dataset_window_bounds' from 'hypertools.plot.trails'`

- [ ] **Step 3: Write the implementation**

In `hypertools/plot/trails.py`, extend `__all__` (line 21) to
`__all__ = ["broadcast_trail_flag", "anim_window_bounds", "dataset_window_bounds"]`
and add after `anim_window_bounds`:

```python
def _project(v, g_ref, first_row, n_rows, n_dataset_rows, g_run):
    """One position on a dataset's clock -> that position on one run's grid.

    `v` is a row position on the dataset's REFERENCE grid (`g_ref` rows for
    the whole dataset). Converting through the dataset's row fraction rather
    than through row COUNTS keeps the unregrouped case exact: with one run,
    `first_row = 0`, `n_rows = n_dataset_rows` and `g_run = g_ref`, so this
    returns `ceil(v)`, and `v` is already an integer. Quantizing to source
    rows first would double-round -- 76 grid rows became 101 in a 9-row,
    12-frame check.
    """
    if n_rows <= 0 or g_run <= 0:
        return 0
    u = v / max(1, g_ref)
    f = (u * n_dataset_rows - first_row) / n_rows
    if f <= 0:
        return 0
    if f >= 1:
        return g_run
    return min(g_run, max(1, int(np.ceil(f * g_run))))


def dataset_window_bounds(num, total_frames, ownership, grid_lengths,
                          window_frames):
    """`(start, end, trail_stop)` per RUN, from ONE clock per source dataset.

    `hue=`/`cluster=` cut each input dataset into contiguous same-category
    runs, each drawn as its own trace and each resampled onto the same frame
    grid. Pacing every trace with `anim_window_bounds` then advances all of
    one dataset's runs at once, so a single trajectory animates in several
    disjoint time windows simultaneously rather than sweeping once (measured
    on `dev-1.0`: three runs of a 30-row dataset all at 247 points on frame 3
    of 12). Driving them from the dataset's own clock and projecting onto each
    run restores the sweep: earlier runs are complete, one run holds the head,
    later runs are empty.

    `anim_window_bounds` is CALLED here rather than reimplemented, so the
    window rescaling, the F05-001 negative-chemtrails clamp and the F05-008
    precog overlap keep exactly one implementation -- the same reason both
    backends share that function.

    Parameters
    ----------
    num, total_frames : int
        Frame index and count, as for `anim_window_bounds`.
    ownership : hypertools.plot.ownership.TraceOwnership
        Which run came from which dataset, and from which of its rows.
    grid_lengths : sequence of int
        Each RUN's drawn row count (post-interpolation).
    window_frames : int
        The opaque head window's length in frames.

    Returns
    -------
    list of tuple of (int, int, int)
        One `(start, end, trail_stop)` per run, indexed by run. A run whose
        dataset has not reached it yet gets `(0, 0, 0)`: an empty slice, not
        a one-row one, so no point of a future category sits on screen.
    """
    bounds = [None] * ownership.n_runs
    for dataset in range(ownership.n_datasets):
        runs = ownership.runs_of(dataset)
        n_dataset_rows = ownership.row_count(dataset)
        g_ref = max(int(grid_lengths[r]) for r in runs)
        d_start, d_end, d_trail = anim_window_bounds(
            num, total_frames, g_ref, window_frames)
        for r in runs:
            first_row, n_rows = ownership.run_span(r)
            g_run = int(grid_lengths[r])
            args = (g_ref, first_row, n_rows, n_dataset_rows, g_run)
            bounds[r] = (_project(d_start, *args),
                         _project(d_end, *args),
                         _project(d_trail, *args))
    return bounds
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/trails.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): one reveal clock per dataset for regrouped runs"
```

---

## Task 3: Use it in the matplotlib parallel updaters

**Files:**
- Modify: `hypertools/plot/matplotlib_backend.py:1185` (3-D `update_lines_parallel`), `:2079` (2-D `update_lines_parallel_2d`), and `_draw`'s signature at `:431`
- Modify: `hypertools/plot/plot.py` — build the ownership and pass it to `_draw`
- Test: `tests/plot/test_regrouped_reveal.py` (new)

**Interfaces:**
- Consumes: `dataset_window_bounds` (Task 2), `TraceOwnership.from_segments` / `.identity` (Task 1).
- Produces: `_draw(..., ownership=None)`; `plot()` passes a `TraceOwnership` whenever it draws lines, regrouped or not.

**Where the ownership comes from.** `plot.py:361` calls `segment_by_run` inside
`_regroup_categorical_lines`, which currently returns six values. Add
`seg_lengths` (pre-`patch_lines`) as a seventh so `plot()` can build the
ownership; the three call sites that unpack it are `plot.py:4149`, `:4175` and
`:4492`.

- [ ] **Step 1: Write the failing test**

Create `tests/plot/test_regrouped_reveal.py`:

```python
"""A regrouped trajectory must animate in source-row order.

Real figures through the public API -- these assert what a viewer sees.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


def _walk(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n, 3), 0)


def _run_lengths(fig, ani, frame):
    ani._func(frame, *ani._args)
    return [len(line.get_xdata()) for line in fig.axes[0].lines]


def test_a_regrouped_trajectory_sweeps_ONCE_not_three_times():
    """The defect: all three runs of one dataset used to grow together, so
    the same trajectory animated at times ~0-3, ~10-13 and ~20-23 at once."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([_walk()], '-', hue=HUE, animate=True,
                            duration=2, frame_rate=6, show=False)
    early = _run_lengths(fig, ani, 3)
    assert early[0] > 0, 'the first run should be under way'
    assert early[1] == 0 and early[2] == 0, (
        f'later runs must not have started: {early}')


def test_a_later_run_starts_only_once_the_previous_one_FINISHES():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([_walk()], '-', hue=HUE, animate=True,
                            duration=2, frame_rate=6, show=False)
    full = _run_lengths(fig, ani, 11)
    for frame in range(12):
        drawn = _run_lengths(fig, ani, frame)
        for r in range(1, 3):
            if drawn[r] > 0:
                assert drawn[r - 1] == full[r - 1], (
                    f'frame {frame}: run {r} started at {drawn[r]} while run '
                    f'{r - 1} was only {drawn[r - 1]}/{full[r - 1]}')


def test_the_final_frame_still_draws_EVERYTHING():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([_walk()], '-', hue=HUE, animate=True,
                            duration=2, frame_rate=6, show=False)
    assert all(n > 0 for n in _run_lengths(fig, ani, 11))


def test_an_UNREGROUPED_animation_is_unchanged_row_for_row():
    """The control. Task 2's projection is the identity without regrouping,
    so this must match the pre-change behaviour exactly -- if it drifts, the
    fix leaked into every animation rather than only the regrouped ones."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([_walk()], '-', animate=True, duration=2,
                            frame_rate=6, show=False)
    assert [_run_lengths(fig, ani, f)[0] for f in range(12)] == [
        1, 83, 165, 247, 329, 411, 493, 575, 657, 739, 821, 903]


def test_two_datasets_still_advance_together():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([_walk(), _walk(seed=1)], '-', animate=True,
                            duration=2, frame_rate=6, show=False)
    a, b = _run_lengths(fig, ani, 5)
    assert a == b


def test_a_2D_regrouped_animation_sweeps_in_order_too():
    """`update_lines_parallel_2d` is a separate updater with its own copy of
    the window call (matplotlib_backend.py:2079)."""
    rng = np.random.RandomState(2)
    data = np.cumsum(rng.randn(30, 2), 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot([data], '-', hue=HUE, animate=True, duration=2,
                            frame_rate=6, show=False)
    early = _run_lengths(fig, ani, 3)
    assert early[1] == 0 and early[2] == 0, early
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`
Expected: FAIL — the sweep tests report later runs already at 247 points on
frame 3. `test_an_UNREGROUPED_animation_is_unchanged_row_for_row` must PASS
already; if it does not, correct its expected list from the actual output
BEFORE touching any implementation, so it records real pre-change behaviour.

- [ ] **Step 3: Return the run lengths from the regrouping helper**

In `hypertools/plot/plot.py`, `_regroup_categorical_lines` (line 341): capture
the pre-bridge lengths and return them.

```python
    segments, seg_labels, seg_cat, seg_bridge, seg_dataset = segment_by_run(
        xform, hue, labels)
    # BEFORE patch_lines, which appends the next run's first point to every
    # bridged run: TraceOwnership must not be told a run owns its neighbour's
    # first observation (see ownership.TraceOwnership.from_segments).
    seg_lengths = [len(s) for s in segments]
```

and change the return to
`return (segments, seg_labels, run_colors, run_group_labels, seg_dataset,
run_category_names, seg_lengths)`, updating its docstring's "Returns" line to
name `seg_lengths` as "each run's row count before bridging".

Update the three unpack sites (`plot.py:4149`, `:4175`, `:4492`) to bind
`_seg_lengths` as the seventh value.

- [ ] **Step 4: Build the ownership in `plot()`**

In `plot.py`, immediately after the block that sets `_forecast_owner` (it ends
at line 5018), add:

```python
    # Run -> dataset -> original rows, for the animation's reveal clock and
    # (below) the forecast schedule. Built for EVERY line plot, not only
    # regrouped ones, so both take the same code path.
    from .ownership import TraceOwnership
    _ownership = None
    if _seg_ds is not None and _seg_lengths is not None:
        _ownership = TraceOwnership.from_segments(_seg_ds, _seg_lengths)
    elif not isinstance(xform, np.ndarray):
        _ownership = TraceOwnership.identity([len(xi) for xi in raw_xform])
```

Initialise `_seg_lengths = None` beside `_seg_ds = None` at line 3837. Pass
`ownership=_ownership` to `_draw(...)` at the call site that already passes
`forecast_schedule=` (`plot.py:5359`).

- [ ] **Step 5: Consume it in both matplotlib updaters**

Add `ownership=None` to `_draw`'s signature (line 431). Inside, before
`update_lines_parallel` is defined, precompute nothing — the bounds depend on
`num` — and replace the per-dataset call at `:1185`:

```python
            start, end, trail_stop = anim_window_bounds(
                num, total_frames, data.shape[0], tail_duration)
```

with a lookup into a per-frame list computed once at the top of the updater,
just after `total_frames` is set (`:1158`):

```python
        # ONE clock per source dataset: `hue=`/`cluster=` runs of the same
        # dataset must reveal in row order, not all at once (see
        # `trails.dataset_window_bounds`). Without regrouping this returns
        # exactly what `anim_window_bounds` returned before.
        _bounds = dataset_window_bounds(
            num, total_frames, ownership,
            [d.shape[0] for d in data_lines], tail_duration)
```

then inside the loop:

```python
            start, end, trail_stop = _bounds[i]
```

Import `dataset_window_bounds` alongside `anim_window_bounds` at line 41.
Apply the identical change to `update_lines_parallel_2d` at `:2079`, computing
`_bounds` after its `total_frames` line (`:2074`).

Guard both: when `ownership is None` (a plain ndarray input with no per-dataset
list), fall back to the existing per-trace `anim_window_bounds` call, so no
input shape loses its animation.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`
Expected: PASS (6 tests)

- [ ] **Step 7: Run the animation and trail suites for regressions**

Run: `.venv/bin/python -m pytest tests/plot -q -k "anim or trail or hue or cluster"`
Expected: PASS, no new warnings.

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/matplotlib_backend.py \
        tests/plot/test_regrouped_reveal.py
git commit -m "fix(plot): a regrouped trajectory animates in row order"
```

---

## Task 4: Plotly parity for the regrouped reveal

**Files:**
- Modify: `hypertools/plot/plotly_backend.py:3445` (head window), `:3484` (trail window), `plotly_draw`'s signature at `:459`
- Modify: `hypertools/plot/plot.py` — pass `ownership=` to `plotly_draw`
- Test: `tests/plot/test_regrouped_reveal.py` (append)

**Interfaces:**
- Consumes: `dataset_window_bounds`, `TraceOwnership`.
- Produces: `plotly_draw(..., ownership=None)`.

**Background:** plotly builds every frame in a Python loop at figure-build
time; `frame.traces` maps each datum to a trace index, and frames carry
geometry only. The head window at `:3445` and the trail window at `:3484` each
call `anim_window_bounds` with `arr.shape[0]`; both must switch to the shared
per-frame list. The trail block deliberately re-derives its bounds rather than
threading a dict from the head loop — keep that structure, calling
`dataset_window_bounds` a second time with the same arguments.

- [ ] **Step 1: Write the failing test**

Append to `tests/plot/test_regrouped_reveal.py`:

```python
def _plotly_frames(**kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return hyp.plot([_walk()], '-', hue=HUE, animate=True,
                            duration=2, frame_rate=6, show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def test_plotly_reveals_regrouped_runs_in_the_same_order():
    pytest.importorskip('plotly')
    fig = _plotly_frames()
    frame = fig.frames[3]
    drawn = {t: len(d.x or ()) for t, d in zip(frame.traces, frame.data)}
    ordered = [drawn[t] for t in sorted(drawn)]
    assert ordered[1] == 0 and ordered[2] == 0, ordered


def test_plotly_and_matplotlib_reveal_the_same_row_counts():
    """Both backends consume `dataset_window_bounds`; a transcription drift
    between them is exactly what the `trails` module exists to prevent."""
    pytest.importorskip('plotly')
    fig = _plotly_frames()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mfig, ani = hyp.plot([_walk()], '-', hue=HUE, animate=True,
                             duration=2, frame_rate=6, show=False)
    for f in range(12):
        frame = fig.frames[f]
        drawn = {t: len(d.x or ()) for t, d in zip(frame.traces, frame.data)}
        ply = [drawn[t] for t in sorted(drawn)]
        mpl = _run_lengths(mfig, ani, f)
        assert [n > 0 for n in ply] == [n > 0 for n in mpl], f'frame {f}'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q -k plotly`
Expected: FAIL — every run has points on frame 3.

- [ ] **Step 3: Write the implementation**

Add `ownership=None` to `plotly_draw` (`:459`). In the frame loop, before the
per-dataset head loop, compute once per frame `k`:

```python
                # ONE clock per source dataset -- the same call the matplotlib
                # updater makes, so the two backends cannot drift (see the
                # `trails` module docstring for the drift this rule prevents).
                frame_bounds = dataset_window_bounds(
                    k, n_frames, ownership,
                    [a.shape[0] for a in arrays], window_frames)
```

replacing `:3445` with `start, end, _ = frame_bounds[idx]` and `:3484` with
`_, end, trail_stop = frame_bounds[idx]`. Import `dataset_window_bounds` at
line 52. Keep the `ownership is None` fallback to `anim_window_bounds`.

In `plot.py`, pass `ownership=_ownership` at the `plotly_draw(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the plotly suite**

Run: `.venv/bin/python -m pytest tests/plot -q -k plotly`
Expected: PASS, no new warnings.

- [ ] **Step 6: Commit**

```bash
git add hypertools/plot/plotly_backend.py hypertools/plot/plot.py \
        tests/plot/test_regrouped_reveal.py
git commit -m "fix(plot): plotly parity for the regrouped reveal order"
```

---

## Task 5: `DatasetRevealSchedule` — visible rows per dataset per frame

**Files:**
- Modify: `hypertools/plot/forecast.py` (add after `revealed_raw_counts`, which ends at line 175)
- Test: `tests/plot/test_trace_ownership.py` (append)

**Interfaces:**
- Consumes: `TraceOwnership`, `revealed_raw_counts(n_raw, n_grid, num, total_frames)`, `matplotlib_backend.serial_reveal_counts(lengths, num, total_frames)`.
- Produces: `DatasetRevealSchedule(ownership, grid_lengths, n_frames, serial=False)` with `visible_rows(dataset, frame) -> tuple` and `head_run(dataset, frame) -> int`.

**Why this is now simple.** After Task 2, a dataset's runs reveal in row order,
so its visible rows are `range(k)` for one `k` — the same prefix the
unregrouped path always had. `visible_rows` still returns the tuple (Decision
R4) and Step 1 asserts the prefix property as its own invariant.

For `serial`, `serial_reveal_counts` already walks runs in order, so a
dataset's visible count is the sum of its runs' counts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/plot/test_trace_ownership.py`:

```python
from hypertools.plot.forecast import DatasetRevealSchedule


def test_visible_rows_are_a_PREFIX_of_the_dataset():
    """The property the fixed reveal buys us, asserted as an invariant rather
    than assumed: `(0, 3, 6)` -- a sample spanning the whole trajectory -- is
    what the old reveal produced, and it is what a forecast must never see."""
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    sched = DatasetRevealSchedule(own, [12, 12, 12], n_frames=12)
    for frame in range(12):
        rows = sched.visible_rows(0, frame)
        assert rows == tuple(range(len(rows))), f'frame {frame}: {rows}'


def test_visible_rows_never_shrink_as_the_animation_runs():
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    sched = DatasetRevealSchedule(own, [12, 12, 12], n_frames=12)
    counts = [len(sched.visible_rows(0, f)) for f in range(12)]
    assert counts == sorted(counts), counts


def test_the_last_frame_sees_the_WHOLE_history():
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    sched = DatasetRevealSchedule(own, [12, 12, 12], n_frames=12)
    assert sched.visible_rows(0, 11) == tuple(range(9))


def test_frames_past_the_end_clamp_instead_of_raising():
    """matplotlib renders one frame past the end on a loop or a save."""
    own = TraceOwnership.identity([9])
    sched = DatasetRevealSchedule(own, [12], n_frames=12)
    assert sched.visible_rows(0, 99) == sched.visible_rows(0, 11)


def test_head_run_is_the_run_holding_the_last_visible_row():
    own = TraceOwnership.from_segments([0, 0, 0], [3, 3, 3])
    sched = DatasetRevealSchedule(own, [12, 12, 12], n_frames=12)
    assert sched.head_run(0, 11) == 2
    early = sched.head_run(0, 0)
    assert early in (0, None)


def test_a_regrouped_schedule_matches_the_UNREGROUPED_one_row_for_row():
    """Splitting a dataset by category must not change WHEN its observations
    become visible -- only which trace draws them."""
    split = DatasetRevealSchedule(
        TraceOwnership.from_segments([0, 0, 0], [3, 3, 3]),
        [12, 12, 12], n_frames=12)
    whole = DatasetRevealSchedule(
        TraceOwnership.identity([9]), [12], n_frames=12)
    assert ([split.visible_rows(0, f) for f in range(12)]
            == [whole.visible_rows(0, f) for f in range(12)])


def test_the_serial_schedule_reveals_datasets_one_at_a_time():
    own = TraceOwnership.from_segments([0, 1], [6, 6])
    sched = DatasetRevealSchedule(own, [12, 12], n_frames=12, serial=True)
    assert sched.visible_rows(1, 0) == ()
    assert len(sched.visible_rows(0, 11)) == 6
    assert len(sched.visible_rows(1, 11)) == 6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q -k "visible or head_run or serial_schedule"`
Expected: FAIL — `ImportError: cannot import name 'DatasetRevealSchedule'`

- [ ] **Step 3: Write the implementation**

In `hypertools/plot/forecast.py`, after `revealed_raw_counts`:

```python
class DatasetRevealSchedule:
    """Which ORIGINAL rows of each source dataset are on screen at each frame.

    `hue=`/`cluster=` draw one trace per contiguous same-category run, so the
    backends' own reveal is expressed in RUNS. A forecast is fit per DATASET,
    from the observations revealed so far, and needs the inverse: original row
    indices, in temporal order, for a dataset that may be spread over several
    traces.

    Since `trails.dataset_window_bounds` gives a dataset's runs one shared
    clock, the visible set is always a prefix -- the same thing an unregrouped
    animation always saw. `visible_rows` returns the row TUPLE rather than a
    count anyway, so a memoization key built from it stays correct if that
    ever stops being true; two frames could otherwise expose different
    observation sets of equal size and collide.
    """

    def __init__(self, ownership, grid_lengths, n_frames, serial=False):
        self.ownership = ownership
        self.grid_lengths = [int(g) for g in grid_lengths]
        self.n_frames = int(n_frames)
        self.serial = bool(serial)
        self._counts = [
            [self._count(d, f) for d in range(ownership.n_datasets)]
            for f in range(self.n_frames)]

    def _count(self, dataset, frame):
        own = self.ownership
        runs = own.runs_of(dataset)
        if self.serial:
            from .matplotlib_backend import serial_reveal_counts
            grid_counts = serial_reveal_counts(
                list(self.grid_lengths), frame, self.n_frames)
            total = 0
            for r in runs:
                g = self.grid_lengths[r]
                _, n_rows = own.run_span(r)
                shown = min(grid_counts[r], g)
                if g < 2 or n_rows < 2 or shown <= 0:
                    total += min(n_rows, max(0, shown))
                else:
                    pos = (shown - 1) * (n_rows - 1) / (g - 1)
                    total += min(n_rows, int(np.floor(pos)) + 1)
            return min(own.row_count(dataset), total)
        # parallel: one clock over the dataset's reference grid, which is what
        # `dataset_window_bounds` paces its runs from
        g_ref = max(self.grid_lengths[r] for r in runs)
        return revealed_raw_counts(own.row_count(dataset), g_ref,
                                   frame, self.n_frames)

    def visible_rows(self, dataset, frame):
        """Original row indices of `dataset` on screen at `frame`, in order."""
        k = self._counts[min(max(frame, 0), self.n_frames - 1)][dataset]
        return tuple(range(k))

    def head_run(self, dataset, frame):
        """The run holding this dataset's LAST visible row, or `None` when
        nothing of it is on screen yet."""
        rows = self.visible_rows(dataset, frame)
        if not rows:
            return None
        return self.ownership.run_holding(dataset, rows[-1])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS (21 tests)

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/forecast.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): DatasetRevealSchedule maps frames to visible rows"
```

---

## Task 6: Draw the forecast over a regrouped animation

**Files:**
- Modify: `hypertools/plot/forecast.py` — add `ForecastSchedule.for_regrouped`
- Modify: `hypertools/plot/plot.py:5051-5069` — replace the refusal
- Test: `tests/plot/test_forecast_animated_regrouped.py` (new)

**Interfaces:**
- Consumes: `DatasetRevealSchedule` (Task 5); `ForecastSchedule.__init__(histories, counts, model, t, min_history, transform, slow_warning_seconds)`; `resolve_forecast_overrides(...)`.
- Produces: `ForecastSchedule.for_regrouped(histories, reveal, model, t, n_frames, min_history, slow_warning_seconds)`.

**What the refusal currently does** (`plot.py:5051`): when `animate` is set and
`len(analyze_histories) != len(xform)`, it warns, records
`_forecast_draw_reason` and sets `analyze_histories = None`, so no schedule is
built. The fit itself still succeeds and `bundle_forecasts` is untouched, so
`return_model=True` already returns the forecasts with `drawn=False`. After
this task the animated regrouped case draws, so `drawn` becomes True and
`draw_reason` stays `None` — the CONTINUOUS-hue case at `:5002-5018` keeps its
refusal (a `LineCollection` has no per-dataset trace to anchor to at all).

- [ ] **Step 1: Write the failing test**

Create `tests/plot/test_forecast_animated_regrouped.py`:

```python
"""Forecasts over an animation whose data hue=/cluster= regrouped."""
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


def _walks(n=2, rows=30, seed=0):
    rng = np.random.RandomState(seed)
    return [np.cumsum(rng.randn(rows, 3), 0) for _ in range(n)]


def _forecast_artists(fig, role=None):
    out = []
    for line in fig.axes[0].lines:
        r = getattr(line, '_hyp_forecast_role', None)
        if r is not None and (role is None or r == role):
            out.append(line)
    return out


def _animate(**kwargs):
    data = _walks()
    hue = HUE * 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig, ani = hyp.plot(data, '-', hue=hue + hue, predict='Kalman', t=4,
                            animate=True, duration=2, frame_rate=6,
                            show=False, **kwargs)
    return fig, ani, caught


def test_a_regrouped_animation_now_DRAWS_its_forecasts():
    fig, ani, caught = _animate()
    ani._func(11, *ani._args)
    assert _forecast_artists(fig, 'live'), 'no live forecast artist was drawn'


def test_it_no_longer_warns_that_it_cannot_draw_them():
    _, _, caught = _animate()
    refusals = [str(w.message) for w in caught
                if 'no forecast overlay' in str(w.message)]
    assert not refusals, refusals


def test_the_bundle_reports_drawn_True():
    data = _walks()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                       animate=True, duration=2, frame_rate=6,
                       show=False, return_model=True)
    info = out[-1]['predict']
    assert info['drawn'] is True
    assert info['draw_reason'] is None


def test_the_final_frame_forecast_equals_the_STATIC_one():
    """The animation's last frame has the whole history, so its forecast must
    be the one a static plot of the same data draws -- otherwise the animated
    and static paths disagree about the same model on the same rows."""
    data = _walks()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                            animate=True, duration=2, frame_rate=6,
                            show=False)
        sfig = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                        show=False)
    ani._func(11, *ani._args)
    live = sorted(np.asarray(a.get_xdata())[-1]
                  for a in _forecast_artists(fig, 'live'))
    static = sorted(np.asarray(a.get_xdata())[-1]
                    for a in _forecast_artists(sfig, 'static'))
    assert np.allclose(live, static, atol=1e-8)


def test_no_forecast_is_fit_on_an_INVISIBLE_observation():
    """The whole point of an animated forecast. Frame 0 exposes at most the
    first observation, so its forecast cannot match one fit on the full
    history."""
    data = _walks()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                            animate=True, duration=2, frame_rate=6,
                            show=False)
    ani._func(2, *ani._args)
    early = [np.asarray(a.get_xdata())[-1] for a in _forecast_artists(fig, 'live')]
    ani._func(11, *ani._args)
    late = [np.asarray(a.get_xdata())[-1] for a in _forecast_artists(fig, 'live')]
    assert early and late
    assert not np.allclose(sorted(early), sorted(late)), (
        'the early forecast already matched the full-history one, so it saw '
        'observations that were not on screen')


def test_the_live_forecast_takes_the_HEAD_run_colour(monkeypatch):
    """Decision R3: the forecast continues the head, so it wears that run's
    colour and changes with it at a category boundary."""
    data = _walks(n=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot(data, '-', hue=HUE, predict='Kalman', t=4,
                            animate=True, duration=2, frame_rate=6,
                            show=False)
    seen = set()
    for frame in range(12):
        ani._func(frame, *ani._args)
        for a in _forecast_artists(fig, 'live'):
            seen.add(tuple(np.round(matplotlib.colors.to_rgb(
                a.get_color()), 5)))
    assert len(seen) > 1, (
        'the forecast kept one colour across a category boundary')


def test_forecast_cluster_still_holds_ONE_colour_across_frames():
    """The override path is unchanged by Decision R3: an explicit grouping is
    resolved once from the full-history forecasts and fixed for every frame."""
    data = _walks(n=4, rows=20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fig, ani = hyp.plot(data, '-', hue=['A'] * 10 + ['B'] * 10,
                            predict='Kalman', t=4, forecast_cluster='KMeans',
                            forecast_n_clusters=2, animate=True, duration=2,
                            frame_rate=6, show=False)
    seen = {}
    for frame in (0, 4, 9, 11, 2, 11, 7):
        ani._func(frame, *ani._args)
        for a in _forecast_artists(fig, 'live'):
            seen.setdefault(id(a), set()).add(
                tuple(np.round(matplotlib.colors.to_rgb(a.get_color()), 5)))
    assert seen
    assert all(len(v) == 1 for v in seen.values()), seen
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: FAIL — the refusal warning fires and no live artist exists.

- [ ] **Step 3: Add `ForecastSchedule.for_regrouped`**

In `hypertools/plot/forecast.py`, beside `for_parallel` and `for_serial`:

```python
    @classmethod
    def for_regrouped(cls, histories, reveal, model, t, n_frames,
                      min_history=DEFAULT_MIN_HISTORY,
                      slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        """Schedule for an animation whose data `hue=`/`cluster=` regrouped.

        The counts come from a `DatasetRevealSchedule` rather than from the
        drawn traces, because a dataset may now be spread over several of
        them. `visible_rows` is a prefix (the runs of one dataset share a
        reveal clock -- `trails.dataset_window_bounds`), so a row COUNT is a
        faithful summary of it; the length is taken from the tuple rather than
        from a parallel counter so the two cannot disagree.
        """
        counts = [[len(reveal.visible_rows(i, f))
                   for i in range(len(histories))]
                  for f in range(n_frames)]
        return cls(histories, counts, model, t, min_history=min_history,
                   slow_warning_seconds=slow_warning_seconds)
```

- [ ] **Step 4: Replace the refusal in `plot()`**

In `plot.py`, replace the block at `:5051-5069` (the `warnings.warn` that says
"no forecast overlay is drawn") with the schedule build. The CONTINUOUS-hue
refusal above it (`:5002-5018`) is untouched — it already set
`raw_forecasts = None`, so this block is only reached when a per-dataset
mapping exists.

```python
    _reveal = None
    if (raw_forecasts is not None and analyze_histories is not None
            and animate and animate not in ('spin',)
            and _ownership is not None
            and len(analyze_histories) != len(xform)):
        # hue=/cluster= regrouped the data into one trace per RUN. The
        # forecast is still per DATASET: `DatasetRevealSchedule` maps each
        # frame onto the dataset's own visible rows, and the runs of one
        # dataset reveal in row order, so those rows are the same prefix an
        # unregrouped animation would have shown.
        from .forecast import DatasetRevealSchedule
        _reveal = DatasetRevealSchedule(
            _ownership, [xi.shape[0] for xi in xform],
            n_frames=max(2, int(round(frame_rate * duration))),
            serial=(animate == 'serial' or order == 'serial'))
```

and extend the schedule build at `:5078` to use it:

```python
        if _reveal is not None:
            forecast_schedule = ForecastSchedule.for_regrouped(
                analyze_histories, _reveal, model=predict, t=t,
                n_frames=_n_frames, slow_warning_seconds=_slow_secs)
        else:
            forecast_schedule = _builder(
                analyze_histories, _grid_lengths, model=predict, t=t,
                n_frames=_n_frames, slow_warning_seconds=_slow_secs)
```

Thread `_reveal` to `_draw`/`plotly_draw` as `forecast_reveal=` so each frame
can ask `head_run(dataset, frame)` for the trace whose colour the live
forecast inherits (Decision R3); where a resolved override exists for that
dataset (`_forecast_overrides[i]` has a `'color'` key), the override wins, as
it already does on the static path.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: PASS (8 tests)

- [ ] **Step 6: Run the whole forecast suite**

Run: `.venv/bin/python -m pytest tests -q -k "forecast or predict"`
Expected: PASS — at least 362 tests (the count before this plan), plus the new ones.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/forecast.py hypertools/plot/plot.py \
        tests/plot/test_forecast_animated_regrouped.py
git commit -m "feat(plot): draw forecasts over regrouped animations"
```

---

## Task 7: Plotly parity, docs, CHANGELOG, examples

**Files:**
- Modify: `hypertools/plot/plotly_backend.py` — consume `forecast_reveal=` for the live/trail forecast colour
- Modify: `docs/animation.rst`, `CHANGELOG.md`
- Test: `tests/plot/test_forecast_animated_regrouped.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: no new API.

- [ ] **Step 1: Write the failing plotly test**

Append to `tests/plot/test_forecast_animated_regrouped.py`:

```python
def test_plotly_draws_forecasts_over_a_regrouped_animation():
    pytest.importorskip('plotly')
    hyp.set_interactive_backend('plotly')
    try:
        data = _walks()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fig = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                           animate=True, duration=2, frame_rate=6, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    roles = {i: (tr.meta or {}).get('hyp_forecast_role')
             for i, tr in enumerate(fig.data)}
    assert 'live' in roles.values(), 'no live forecast trace was built'
    last = fig.frames[-1]
    drawn = [d for t, d in zip(last.traces, last.data)
             if roles.get(t) == 'live']
    assert drawn and any(len(d.x or ()) for d in drawn)


def test_both_backends_end_the_animation_at_the_same_forecast():
    pytest.importorskip('plotly')
    data = _walks()
    hyp.set_interactive_backend('plotly')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pfig = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                            animate=True, duration=2, frame_rate=6,
                            show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mfig, ani = hyp.plot(data, '-', hue=HUE + HUE, predict='Kalman', t=4,
                             animate=True, duration=2, frame_rate=6,
                             show=False)
    ani._func(11, *ani._args)
    mpl = sorted(float(np.asarray(a.get_xdata())[-1])
                 for a in _forecast_artists(mfig, 'live'))
    roles = {i: (tr.meta or {}).get('hyp_forecast_role')
             for i, tr in enumerate(pfig.data)}
    last = pfig.frames[-1]
    ply = sorted(float(d.x[-1]) for t, d in zip(last.traces, last.data)
                 if roles.get(t) == 'live' and (d.x or ()))
    assert np.allclose(mpl, ply, atol=1e-8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q -k plotly or both_backends`
Expected: FAIL — no live trace in the plotly figure.

- [ ] **Step 3: Implement plotly's side and update the docs**

Mirror Task 6's colour rule in `plotly_backend.py` where the live/trail
forecast traces are built (they already consume `forecast_schedule` and the
resolved override map): take the head run's colour from `forecast_reveal`
when no override supplies one.

In `docs/animation.rst`, replace the paragraph stating that animated forecasts
are unsupported under `hue=`/`cluster=` with:

```rst
Animated forecasts under ``hue=``/``cluster=``
----------------------------------------------

When ``hue=`` or ``cluster=`` splits a trajectory into per-category runs, each
run is drawn as its own trace, but the reveal still follows the **dataset**:
one run finishes before the next begins, so the trajectory sweeps once and
changes colour as it crosses a category boundary. A forecast is fit per
dataset from exactly the observations on screen, so it means the same thing it
does without ``hue=``.

A live forecast inherits the colour of the run holding the head, and therefore
changes colour with it. Pass ``forecast_hue=``, ``forecast_cluster=`` or
``forecast_palette=`` to give the forecasts a grouping of their own; those are
resolved once from the full-history forecasts and stay fixed for every frame.

A **continuous** ``hue=`` still draws no forecast overlay: the data becomes a
single ``LineCollection`` with no per-dataset trace to anchor one to, and
``plot()`` says so rather than dropping it silently.
```

In `CHANGELOG.md`, under the 1.1 section, add:

```markdown
- **A regrouped trajectory now animates in row order.** With `hue=`/`cluster=`,
  each contiguous same-category run is drawn as its own trace, and every run
  used to advance at once -- so one trajectory animated in several disjoint
  time windows simultaneously (three runs of a 30-row dataset were all 27%
  drawn on frame 3 of 12). Runs of one input dataset now share a single reveal
  clock, so the head sweeps the trajectory once and changes colour at each
  category boundary, matching both the un-regrouped and `order='serial'`
  behaviour. Animations without `hue=`/`cluster=` are unchanged row for row.

- **`predict=` now works with `hue=`/`cluster=` on ANIMATED plots.** Previously
  the fit succeeded and the forecasts were returned in the `return_model=True`
  bundle with `drawn=False`, but no overlay was drawn. Each frame's forecast is
  fit from exactly the observations visible for that dataset. A live forecast
  inherits the colour of the run holding the head; `forecast_hue=`/
  `forecast_cluster=`/`forecast_palette=` override that with a grouping fixed
  for the whole animation. A continuous `hue=` still draws no overlay.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Run every check**

```bash
.venv/bin/python -m pytest -q
cd docs && ../.venv/bin/python -m sphinx -W -E -a . _build/html && cd ..
.venv/bin/ruff check hypertools/plot/ownership.py hypertools/plot/trails.py \
    hypertools/plot/forecast.py hypertools/plot/plot.py \
    hypertools/plot/matplotlib_backend.py hypertools/plot/plotly_backend.py
```

Expected: full suite passes with **no `warnings summary` section**; docs build
succeeds under `-W`; ruff reports no MORE findings than on the base commit
(compare by running the same command against a `git worktree` of the base —
never `git stash`).

- [ ] **Step 6: Commit**

```bash
git add hypertools/plot/plotly_backend.py docs/animation.rst CHANGELOG.md \
        tests/plot/test_forecast_animated_regrouped.py
git commit -m "feat(plot): plotly parity + docs for animated regrouped forecasts"
```

---

## Self-Review

**1. Spec coverage.**

| Requirement | Task |
|-|-|
| `TraceOwnership` frozen dataclass with `dataset_by_run`, `source_rows_by_run`, `final_run_by_dataset` | 1 |
| Run-local positions map back to original row indices | 1 (`from_segments`, tested per dataset) |
| `DatasetRevealSchedule.visible_rows(dataset, frame)` | 5 |
| Visible rows sorted into original temporal order | 5 (prefix invariant test) |
| No invisible observation enters a forecast fit | 6 (`test_no_forecast_is_fit_on_an_INVISIBLE_observation`) |
| Memoize on `(dataset, visible_row_tuple)`, not `(dataset, count)` | 5 (R4) + 6 (`for_regrouped` derives the count from the tuple) |
| Equal-sized but different visible subsets stay distinguishable | 2 + 5 — the fixed reveal makes visible sets prefixes, so equal size implies equal set; the prefix property is asserted rather than assumed |
| Final-frame output equals the static full-history forecast | 6 (`test_the_final_frame_forecast_equals_the_STATIC_one`) |
| Both backends consume the same schedule | 4, 7 (cross-backend equality tests) |
| Test the mapping independently before integrating | 1, 2, 5 are pure-mapping tasks with no figures |
| Reveal order fixed before forecasts are built on it | 3, 4 precede 5-7 |

**2. Placeholder scan.** No "TBD"/"add error handling"/"similar to Task N".
Task 3 Step 5 and Task 7 Step 3 describe edits to code whose surrounding lines
are quoted, with the replacement shown.

**3. Type consistency.** `TraceOwnership.from_segments(seg_dataset,
seg_lengths)` is called with those names in Task 3; `run_span` returns
`(first_row, n_rows)` and is unpacked that way in Tasks 2 and 5;
`dataset_window_bounds(...)` returns a list indexed by run and is indexed as
`_bounds[i]` / `frame_bounds[idx]` in Tasks 3 and 4; `visible_rows` returns a
tuple and Task 6 takes `len()` of it.

**Known gap, deliberate.** `test_an_UNREGROUPED_animation_is_unchanged_row_for_row`
hard-codes the point counts measured on `dev-1.0` at `a062f768`. Task 3 Step 2
instructs the implementer to re-measure and correct the list BEFORE changing
any implementation, so it records real pre-change behaviour rather than a
number copied from this plan.
