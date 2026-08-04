# Regrouped Reveal + Animated Regrouped Forecasts Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `hue=`/`cluster=` regrouped LINE trajectory animate in source-row
order under the default parallel reveal, then draw per-frame forecasts over
regrouped animations (the case that is currently warning-only).

**Architecture:** One new pure module, `hypertools/plot/ownership.py`, records
which source dataset and which original rows each drawn run came from, *and*
whether `patch_lines` gave it a bridge vertex. `trails.dataset_window_bounds`
uses it to drive every run of one dataset from a SINGLE clock, returning an
explicit `RunWindow` per run rather than three integers carrying four meanings.
`DatasetRevealSchedule` then derives each dataset's visible rows **from those
same windows**, so the renderer and the forecast history cannot describe
different states. The animated-regrouping refusal at `plot.py:5051-5077` is
replaced by a real overlay.

**Tech Stack:** numpy, scipy (PCHIP, already used), `fractions.Fraction`,
matplotlib, plotly, pytest.

---

## What changed from v1, and why

v1 was reviewed and found not implementable as written. Every finding below is
addressed; the architectural direction (ownership -> dataset clock -> reveal ->
forecast -> both backends) is unchanged.

| v1 defect | v2 resolution |
|-|-|
| **Bridge rows broke "exactly the visible history".** `patch_lines` appends the NEXT run's first observation to a bridged run, so that observation is on screen while `visible_rows` still reported only the preceding run's owned rows. | A run's *drawn* span is `n_rows - 1 + bridged` source-parameter units, and the projection uses that span. Run *r* therefore completes at exactly the parameter where run *r+1* shows its first point — verified simultaneous, every frame, in every case below. `head_run` returns the destination run at that instant. |
| **`head_run` was ambiguous at a boundary.** | Falls out of the above: `visible_rows` is derived from the run windows, so the last visible row and the run drawing it are the same fact. |
| **Tuple-key memoization was promised but converted to a count.** | `ForecastSchedule` now stores explicit row tuples and keys `self._paths` on `(dataset, rows)`, slicing `histories[i][list(rows)]`. `counts` becomes a derived view. Task 6. |
| **`precog` mishandled `end == 0`** (`data[end - 1:]` -> `data[-1:]`). | `RunWindow(head_start, head_end, past_stop, future_start, reached)` — four named bounds instead of three overloaded ones. An unreached run gets `RunWindow(0, 0, 0, 0, reached=False)`: empty head, empty chemtrail, precog covering the whole not-yet-revealed run. |
| **Projected grids included bridge geometry but source spans did not.** | `TraceOwnership` carries `owned rows` *and* `bridged`; `draw_span(run)` is the rendered geometry, `run_span(run)` the owned rows, and each is used where it belongs. |
| **Trail forecast colours were unspecified.** | Decision R3 (v2): a retained forecast of age *k* takes `head_run(dataset, past[k-1])` — the head run at the frame it was *fit*, not the current one. Task 7 pins it with a fan spanning a boundary. |
| **`assert early != late` did not prove exclusion.** | Task 6 compares `schedule.polyline(i, f)` against a direct `forecast_from_history(history[list(visible_rows)], ...)` fit, plus a mutation control showing one extra row changes the answer. |
| **`warnings.simplefilter('ignore')` contradicted the zero-warning gate.** | Every test uses a `_no_warnings()` helper that records and asserts empty, or `pytest.warns` for one named expected warning. No blanket ignores. |
| **`TraceOwnership` assumed dense dataset ids without checking.** | `from_segments` validates dense, zero-based, first-appearance-ordered ids. |
| **Ownership construction was too broad** (`not isinstance(xform, np.ndarray)`). | Built only from `_seg_ds` (line regrouping), or as `identity` when *no* regrouping happened at all and the cardinality matches. Marker/scatter regrouping gets `ownership=None` and the existing behaviour. |
| **Plotly parity compared only empty/non-empty.** | Compares exact per-run vertex counts at every frame, plus boundary/final/singleton/unequal-length cases. |

---

## Why this plan exists (measured, not assumed)

`hyp.plot([x], '-', hue=['A']*10+['B']*10+['A']*10, animate=True, duration=2, frame_rate=6)`
drives one 30-row dataset through `segment_by_run`, producing three runs.
Points drawn per run-line, per frame, on `dev-1.0` at `a062f768`:

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

---

## Global Constraints

- Python >= 3.10 (`requires-python = ">=3.10"`).
- Run every command with `.venv/bin/python`. The system python's numpy breaks matplotlib.
- Never use `git stash` in this repo (documented data-loss hazard). Use `git show <ref>:<path>`.
- No mock objects and no mock tests, ever — including as fallbacks. If real functionality cannot be exercised, the test must fail or raise.
- `pytest` runs from the repo root (`pyproject.toml` sets `testpaths = ["tests"]`).
- Both backends must consume the SAME reveal arithmetic. `trails.py` owns it; neither backend may re-derive it. This is a standing rule in the `trails` module docstring, which records a plotly transcription drift that blanked a 5-row dataset for 9 of its 15 frames.
- The full suite must stay at zero warnings: it currently emits no `warnings summary` section at all. A new warning is a failure. **No test in this plan may call `warnings.simplefilter('ignore')`** — record and assert, or name the expected warning with `pytest.warns`.
- `cd docs && make html` must build clean under `sphinx -W` (warnings are errors).
- Any behavior change to a released 1.0 API goes in `CHANGELOG.md`.
- **Scope:** this feature covers regrouped **line** trajectories (`_fmt_draws_line(fmt)`, the `_regroup_categorical_lines` path). Marker-only categorical regrouping goes through `reshape_data` and groups globally by category, with no per-dataset row ordering to sweep; it keeps today's behaviour and is explicitly out of scope.

---

## File Structure

| File | Responsibility |
|-|-|
| `hypertools/plot/ownership.py` (new) | `TraceOwnership`: run -> source dataset, run -> original row indices, run -> bridged?, dataset -> final run. Pure data; no plotting imports. |
| `hypertools/plot/trails.py` (modify) | Add `RunWindow` and `dataset_window_bounds`, the ONE frame->rows mapping for regrouped parallel animation. Existing `anim_window_bounds` is unchanged and becomes its per-dataset clock. |
| `hypertools/plot/matplotlib_backend.py` (modify) | 3-D (`:1185`) and 2-D (`:2080`) parallel updaters consume `RunWindow`s. |
| `hypertools/plot/plotly_backend.py` (modify) | Head (`:3445`) and trail (`:3484`) window computations do the same. |
| `hypertools/plot/forecast.py` (modify) | `DatasetRevealSchedule` (visible rows, derived from the run windows); `ForecastSchedule` re-keyed on row tuples; `ForecastSchedule.for_regrouped`. |
| `hypertools/plot/plot.py` (modify) | Return `seg_lengths`/`seg_bridge` from `_regroup_categorical_lines`; build `TraceOwnership`; thread it to both backends; replace the refusal at `:5051-5077`; per-frame forecast colour from the head run. |
| `tests/plot/test_trace_ownership.py` (new) | Pure mapping tests: ownership, run windows, reveal schedule, cross-invariants. No figures. |
| `tests/plot/test_regrouped_reveal.py` (new) | Real `hyp.plot(...)` animations: sweep order, both backends, both reveal orders. |
| `tests/plot/test_forecast_animated_regrouped.py` (new) | Forecasts over regrouped animations, both backends. |

---

## Contracts this plan establishes

```python
# hypertools/plot/ownership.py
@dataclass(frozen=True)
class TraceOwnership:
    dataset_by_run: tuple          # run -> source dataset index
    source_rows_by_run: tuple      # run -> tuple of ORIGINAL row indices (OWNED)
    bridged_by_run: tuple          # run -> did patch_lines append the next run's first row?
    final_run_by_dataset: tuple    # dataset -> its last run

    n_runs: int                    # property
    n_datasets: int                # property
    def runs_of(self, dataset) -> tuple: ...
    def row_count(self, dataset) -> int: ...
    def run_span(self, run) -> tuple:    # (first_row, n_owned_rows)
    def draw_span(self, run) -> int:     # n_owned_rows - 1 + bridged: the RENDERED span
    def run_holding(self, dataset, row) -> int:
    @classmethod
    def from_segments(cls, seg_dataset, seg_lengths, seg_bridge) -> 'TraceOwnership': ...
    @classmethod
    def identity(cls, dataset_lengths) -> 'TraceOwnership': ...

# hypertools/plot/trails.py
@dataclass(frozen=True)
class RunWindow:
    head_start: int      # data[head_start:head_end] -- the opaque head
    head_end: int
    past_stop: int       # data[0:past_stop]         -- chemtrails
    future_start: int    # data[future_start:]       -- precog
    reached: bool        # has the dataset's clock entered this run at all?
    grid: int            # the run's drawn row count, so the head's source
                         # parameter can be read back off the window alone

def dataset_window_bounds(num, total_frames, ownership, grid_lengths,
                          window_frames) -> list:   # one RunWindow per RUN

def run_head_param(window, ownership, run) -> Fraction | None
    """Source-parameter position of a run's drawn head, or None if unreached."""

# hypertools/plot/forecast.py
class DatasetRevealSchedule:
    def __init__(self, ownership, grid_lengths, n_frames, window_frames,
                 serial=False): ...
    def visible_rows(self, dataset, frame) -> tuple    # ORIGINAL row indices, sorted
    def head_run(self, dataset, frame) -> int | None   # run drawing the last visible row

class ForecastSchedule:
    def __init__(self, histories, counts=None, model=..., t=..., rows=None,
                 min_history=..., transform=None, slow_warning_seconds=...): ...
    @classmethod
    def for_regrouped(cls, histories, reveal, model, t, n_frames,
                      min_history=DEFAULT_MIN_HISTORY,
                      slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS): ...
```

---

## The arithmetic, stated once (verified before writing)

For dataset `d` with `N` owned rows, spread over runs `r`:

1. **Reference grid.** `G_ref = max(grid_lengths[r] for r in runs_of(d))`. Every
   interpolated line run is resampled to exactly `n_frames` rows by
   `plot._interp_anim_line`, so all of a dataset's line runs share one value;
   `max` ignores a run left un-interpolated because it had fewer than 2 rows.
2. **The dataset's clock.** `(start, end, trail_stop) =
   anim_window_bounds(num, total_frames, G_ref, window_frames)` — the existing
   function, unchanged, on the grid the dataset WOULD have had unsplit.
3. **Source parameters.** Reference grid row `v` sits at source parameter
   `P(v) = v * (N - 1) / (G_ref - 1)`. This is exactly the mapping
   `forecast.revealed_raw_counts` already uses. Take
   `p_head = P(end - 1)`, `p_start = P(start)`, and
   `p_trail = P(trail_stop - 1)` (or `None` when `trail_stop == 0`).
4. **Projection onto a run.** Run `r` owns rows `[a, a + L)` and draws a
   polyline spanning source parameters `[a, a + S]` where `S = draw_span(r) =
   L - 1 + bridged`. A source parameter `p` maps to run-grid index
   `j = (p - a) * (g_r - 1) / S`.
   - `count_from(p)` = `0` if `p < a`; `g_r` if `g_r < 2 or S <= 0` (a run with
     nothing to slide along is all-or-nothing); otherwise
     `min(g_r, floor(min(j, g_r - 1)) + 1)`.
   - `index_from(p)` = `0` if `p <= a`; otherwise `min(g_r, floor(j))`.
   - `head_end = count_from(p_head)`, `head_start = index_from(p_start)`,
     `past_stop = count_from(p_trail)`,
     `future_start = max(0, head_end - 1) if reached else 0`,
     `reached = p_head >= a`.
5. **Exact arithmetic.** All of this is done in `fractions.Fraction`. Floating
   point would make the unregrouped identity below hold only to within rounding,
   and the whole point of step 4 is that it is the identity there.

**Everything else is derived from the `RunWindow`s, never computed twice.** A
dataset's head parameter at a frame is
`max(a_r + P_r(head_end_r - 1))` over its reached runs, and its visible rows are
`range(min(N, floor(that) + 1))`. That is why the renderer and the forecast
history cannot drift: there is one quantity, and both read it.

### Verified properties

Checked with `Fraction` arithmetic against the real `anim_window_bounds` and
`revealed_raw_counts`, over every frame of 13 run-length/frame-count cases
(`[10,10,10]`, `[3,3,3]`, `[4,1,4]`, `[1,1,1,1]`, `[7,2,11,5]`, `[2,2]`,
`[5,5,5,5,5,5]`, `[20,10]`, `[1,29]`, `[29,1]`, `[50,50]`, `[2,26,2]` at 3-120
frames) plus 6 unregrouped cases — **0 failures**:

- **Unregrouped identity.** With one run per dataset, `(head_start, head_end,
  past_stop)` equals `anim_window_bounds(...)` exactly, and `future_start`
  equals today's `end - 1`, at every frame. No existing animation moves.
- **Unregrouped visible rows** equal `revealed_raw_counts(...)` exactly.
- **Bridge simultaneity.** Run `r` reaches `head_end == g_r` on exactly the
  frames where run `r+1` has `head_end > 0`. The shared vertex is one source
  row, visible through both.
- **Ordering.** A later run never has points while an earlier one is unfinished.
- **Prefix + monotone.** A dataset's visible rows are always `range(k)`, and `k`
  never decreases.
- **Final frame.** Every run is fully drawn and every row visible.
- **`future_start >= 0`** in every case (the `data[-1:]` defect cannot recur).
- **Parallel across datasets.** Two unsplit datasets produce identical windows.

The verification script is Task 2 Step 1's test body; it is not a throwaway.

---

## Named decisions

**Decision R1 — the reveal clock is per SOURCE DATASET, not per drawn run.**
Settled by Jeremy 2026-08-03. Runs of one input dataset share one clock; runs
of different datasets still advance together (parallel keeps its meaning).

**Decision R2 — projection is exactly the identity for the unregrouped case,
and lags by at most one run-grid step when regrouped.** Measured: splitting a
dataset into runs makes a row become visible *at or after* the frame it would
have appeared on unsplit — never before — by at most one run-grid step
(`draw_span / (g_run - 1)` source rows; 0-1 rows for `[10,10,10]`/12 frames, 3
rows for the extreme `[1,29]`/12 frames, always < one frame of trajectory).
This is intrinsic: a run's head can only stop at one of its own drawn vertices,
and those do not coincide with the unsplit grid. The direction matters and is
asserted — a forecast can never be fit on an observation the renderer has not
yet drawn.

**Decision R3 (revised) — forecast colour follows the head run at the frame the
forecast was FIT.** A live forecast visually continues the head, so it takes
that run's colour and changes at a category boundary. A *retained*
`forecast_trail=` member of age `k` was fit at frame `past[k-1]`, so it takes
`head_run(dataset, past[k - 1])` — otherwise the whole historical fan repaints
whenever the current head crosses a boundary, which would contradict the
already-shipped promise that a saved animation matches a played one.
`forecast_hue=`/`forecast_cluster=`/`forecast_palette=` override both and are
already fixed for the whole animation.

**Decision R4 — `ForecastSchedule` memoizes on `(dataset, visible_row_tuple)`,
for real.** The row tuple is stored, not summarised to a length; `histories[i]`
is sliced with the tuple. Under the fixed reveal the tuple is always a prefix,
so this is equivalent to the count today — the point is that it stays correct
if the reveal ever stops producing prefixes, which is the failure mode
`(dataset, count)` hides. The prefix property is asserted separately (Task 5)
so a regression shows up as a failing invariant, not as wrong colours.

**Decision R5 — an unreached run's `precog` trail is the WHOLE run.** `precog`
means "the trajectory ahead of the head". A run the clock has not entered is
entirely ahead of it, so `future_start = 0`. The alternative readings — one
stray point (`data[-1:]`, the defect) or nothing at all — either lie about the
data or make a `precog` trail blink into existence at a boundary.

---

## Task 1: `TraceOwnership` — run -> dataset -> original rows (+ bridge)

**Files:**
- Create: `hypertools/plot/ownership.py`
- Test: `tests/plot/test_trace_ownership.py`

**Interfaces:**
- Consumes: `segment_by_run`'s `seg_dataset` and `seg_bridge`, plus the PRE-`patch_lines` run lengths.
- Produces: the `TraceOwnership` API in "Contracts" above.

**Background the implementer needs.** `hypertools/_shared/helpers.py:278`
`segment_by_run(x, hue, labels)` walks each input dataset in order and cuts it
into maximal same-category runs, returning `seg_dataset` (each run's source
dataset) and `seg_bridge` (whether run `i` bridges into run `i+1`). It does NOT
return row indices — but it does not need to: within a dataset the runs
partition its rows contiguously, in order, so run lengths recover them.

`plot._regroup_categorical_lines` (`plot.py:341`) then calls
`patch_lines(segments, breaks={i + 1 for i in range(len(segments) - 1) if not
seg_bridge[i]})`, which appends `x[idx+1][0]` to `x[idx]` for every bridged run
(`_shared/helpers.py:346-379`). So a bridged run's drawn array is **one row
longer than the span it owns**, and that extra vertex is the next run's first
observation. `TraceOwnership` must be built from PRE-`patch_lines` lengths, and
must record `bridged` so `draw_span` can describe the RENDERED geometry
separately from the OWNED rows. Conflating the two is the defect that made v1
unimplementable.

- [ ] **Step 1: Write the failing tests**

Create `tests/plot/test_trace_ownership.py`:

```python
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
needs the inverse mapping, in ORIGINAL row indices rather than run-local ones.

It also needs to distinguish two spans that are easy to conflate and were:

* the rows a run OWNS (`run_span`), which is what a forecast history is
  sliced from, and
* the source parameters a run DRAWS (`draw_span`), which is one longer for a
  bridged run because `patch_lines` appends the next run's first observation
  so the polyline is continuous across a colour change.

Using the owned span to pace the drawn geometry mis-times every category
boundary by one vertex, and leaves an observation on screen that the reveal
schedule says is not visible.

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
        Each run's source input-dataset index. Dense, zero-based, and in
        first-appearance order (validated in `from_segments`), because
        `final_run_by_dataset` is indexed by dataset.
    source_rows_by_run : tuple of tuple of int
        Each run's OWNED original row indices, numbered WITHIN ITS OWN
        DATASET (dataset 1's first row is 0, not the global stacked offset).
        Excludes the bridge row `patch_lines` appends: that row is owned by
        the run it came from and merely duplicated here for rendering.
    bridged_by_run : tuple of bool
        Whether `patch_lines` appended the NEXT run's first observation to
        this run's drawn array.
    final_run_by_dataset : tuple of int
        Each dataset's LAST run -- the one holding its final observation, and
        so the trace a static forecast continues.
    """

    dataset_by_run: tuple
    source_rows_by_run: tuple
    bridged_by_run: tuple
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
        """``(first_row, n_owned_rows)`` in this run's dataset's numbering."""
        rows = self.source_rows_by_run[run]
        return rows[0], len(rows)

    def draw_span(self, run):
        """Source-parameter span of the run's DRAWN polyline.

        ``n_owned_rows - 1`` vertices' worth of trajectory, plus 1 when
        `patch_lines` bridged it -- the drawn line reaches the NEXT run's
        first observation, and the reveal must reach it at the same moment
        that observation becomes visible, or the boundary shows a vertex the
        schedule denies.
        """
        return len(self.source_rows_by_run[run]) - 1 + int(
            self.bridged_by_run[run])

    def run_holding(self, dataset, row):
        """The run OWNING original `row` of `dataset`."""
        for r in self.runs_of(dataset):
            if row in self.source_rows_by_run[r]:
                return r
        raise ValueError(
            f"dataset {dataset} has no row {row} (it has "
            f"{self.row_count(dataset)}).")

    @classmethod
    def from_segments(cls, seg_dataset, seg_lengths, seg_bridge):
        """Build from `segment_by_run`'s outputs and PRE-`patch_lines` lengths.

        `patch_lines` appends the NEXT run's first point to every bridged run,
        so a drawn run's array is one row longer than the span it owns.
        Passing post-bridge lengths here would hand every bridged run one row
        of its neighbour's data -- exactly the double-counting that makes a
        forecast history wrong by one observation per category boundary.
        Pass `seg_bridge` straight through from `segment_by_run` rather than
        re-deriving it from run positions: `plot._regroup_categorical_lines`
        turns it into `patch_lines`' `breaks` set, so it is the only record of
        what was actually bridged.
        """
        seg_dataset = [int(d) for d in seg_dataset]
        seg_lengths = [int(n) for n in seg_lengths]
        seg_bridge = [bool(b) for b in seg_bridge]
        if len(seg_dataset) != len(seg_lengths):
            raise ValueError(
                f"one length per run is needed; got {len(seg_dataset)} run "
                f"dataset(s) and {len(seg_lengths)} length(s).")
        if len(seg_bridge) != len(seg_dataset):
            raise ValueError(
                f"one bridge flag per run is needed; got {len(seg_bridge)} "
                f"for {len(seg_dataset)} run(s).")
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
        # `final_run_by_dataset` is INDEXED by dataset and `runs_of` is called
        # with `range(n_datasets)`, so ids must be dense, zero-based and in
        # order. `segment_by_run` always emits them that way; this class is
        # independently callable, and sparse ids fail silently rather than
        # loudly (entry 0 would describe some other dataset).
        if seen_order and seen_order[0] != 0:
            raise ValueError(
                f"dataset indices must start at 0 and be dense; the first "
                f"one seen is {seen_order[0]}.")
        if seen_order != list(range(len(seen_order))):
            raise ValueError(
                f"dataset indices must appear in order 0, 1, 2, ...; got "
                f"{seen_order}.")
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
        for d, r in final.items():
            if seg_bridge[r]:
                raise ValueError(
                    f"a dataset's last run cannot be bridged (patch_lines is "
                    f"given a break at every dataset boundary); run {r} is "
                    f"the last run of dataset {d} and is marked bridged.")
        return cls(tuple(seg_dataset), tuple(rows_by_run), tuple(seg_bridge),
                   tuple(final[d] for d in sorted(final)))

    @classmethod
    def identity(cls, dataset_lengths):
        """The UNREGROUPED case: one run per dataset, holding all its rows.

        Every consumer takes the same code path whether or not `hue=` split
        anything, so the regrouped path cannot quietly rot while the common
        one stays green.
        """
        lengths = [int(n) for n in dataset_lengths]
        return cls.from_segments(list(range(len(lengths))), lengths,
                                 [False] * len(lengths))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/ownership.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): TraceOwnership maps drawn runs back to source rows"
```

---

## Task 2: `RunWindow` + `dataset_window_bounds` — one reveal clock per dataset

**Files:**
- Modify: `hypertools/plot/trails.py` (add after `anim_window_bounds`, which ends at line 94; extend `__all__` at line 21)
- Test: `tests/plot/test_trace_ownership.py` (append)

**Interfaces:**
- Consumes: `TraceOwnership` (Task 1); `anim_window_bounds(num, total_frames, n_points, window_frames) -> (start, end, trail_stop)`.
- Produces: `RunWindow`; `dataset_window_bounds(num, total_frames, ownership, grid_lengths, window_frames) -> list of RunWindow`, indexed by run; `run_head_param(window, ownership, run) -> Fraction | None`.

Read "The arithmetic, stated once" above before writing anything. The four
named bounds exist because three integers could not carry four meanings: the
old `(start, end, trail_stop)` triple encoded `precog` as `data[end - 1:]`, and
a run that has not started has `end == 0`, which Python reads as `data[-1:]` —
one stray point of a future category sitting on screen from frame 0.

- [ ] **Step 1: Write the failing tests**

Append to `tests/plot/test_trace_ownership.py`:

```python
from fractions import Fraction

from hypertools.plot.forecast import revealed_raw_counts
from hypertools.plot.trails import (RunWindow, anim_window_bounds,
                                    dataset_window_bounds, run_head_param)


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
    step = max(Fraction(own.draw_span(r), grids[r] - 1)
               for r in range(len(lengths)) if grids[r] >= 2
               and own.draw_span(r) > 0)
    for num in range(n_frames):
        split = len(_visible(own, grids, num, n_frames, w))
        unsplit = len(_visible(whole, [n_frames], num, n_frames, w))
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: FAIL — `ImportError: cannot import name 'RunWindow' from 'hypertools.plot.trails'`

- [ ] **Step 3: Write the implementation**

In `hypertools/plot/trails.py`, add `from dataclasses import dataclass` and
`from fractions import Fraction` at the top, extend `__all__` (line 21) to
`["broadcast_trail_flag", "anim_window_bounds", "RunWindow",
"dataset_window_bounds", "run_head_param"]`, and add after
`anim_window_bounds`:

```python
@dataclass(frozen=True)
class RunWindow:
    """What one drawn run shows at one frame.

    Four named bounds rather than the ``(start, end, trail_stop)`` triple
    `anim_window_bounds` returns, because a run that the dataset's clock has
    not reached yet needs a FOURTH state the triple cannot express. The
    historical precog slice is ``data[end - 1:]``; with ``end == 0`` that is
    ``data[-1:]`` -- one point of a not-yet-revealed category sitting on
    screen from frame 0. Naming the future bound separately makes that slice
    unwritable.

    Attributes
    ----------
    head_start, head_end : int
        The opaque head is ``data[head_start:head_end]``. Both 0 for a run the
        clock has not reached; both `g_run` for a run the sliding window has
        moved past.
    past_stop : int
        A chemtrails trail is ``data[0:past_stop]``. 0 until the head window
        actually starts sliding (F05-001).
    future_start : int
        A precog trail is ``data[future_start:]``, sharing the head's last
        vertex so there is no one-segment gap (F05-008). 0 -- the WHOLE run --
        for a run the clock has not reached: all of it is still ahead.
    reached : bool
        Whether the dataset's clock has entered this run at all.
    grid : int
        The run's drawn row count. Carried so `run_head_param` can invert the
        projection from the window ALONE -- the reveal schedule must read the
        head position back off the objects the backends actually sliced with,
        not recompute it from the frame index, or the two can drift.
    """

    head_start: int
    head_end: int
    past_stop: int
    future_start: int
    reached: bool
    grid: int


def _param(idx, g, span):
    """Grid-row index -> source-parameter offset within a run or dataset.

    `plot._interp_anim_line` resamples `n` source rows onto
    ``linspace(0, n - 1, g)`` with exact endpoints, so grid row `idx` sits at
    source parameter ``idx * span / (g - 1)``. Exact rational arithmetic, not
    float: the unregrouped case must project to the IDENTITY, and it does so
    only if the round trip cancels exactly.
    """
    if g < 2 or span <= 0:
        return Fraction(0)
    return Fraction(int(idx) * int(span), int(g) - 1)


def dataset_window_bounds(num, total_frames, ownership, grid_lengths,
                          window_frames):
    """One `RunWindow` per RUN, from ONE clock per source dataset.

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

    The projection goes through the SOURCE PARAMETER, not through row counts,
    and uses each run's DRAWN span (`TraceOwnership.draw_span`, which counts
    the bridge vertex `patch_lines` appended). Both choices are load-bearing:
    quantizing to source rows first double-rounds (76 grid rows became 101 in
    a 9-row, 12-frame check), and using the OWNED span instead of the drawn
    one desynchronizes every category boundary by one vertex, leaving an
    observation on screen that the reveal schedule reports as invisible.

    Parameters
    ----------
    num, total_frames : int
        Frame index and count, as for `anim_window_bounds`.
    ownership : hypertools.plot.ownership.TraceOwnership
        Which run came from which dataset, from which of its rows, and
        whether it carries a bridge vertex.
    grid_lengths : sequence of int
        Each RUN's drawn row count (post-interpolation).
    window_frames : int
        The opaque head window's length in frames.

    Returns
    -------
    list of RunWindow
        Indexed by run.
    """
    windows = [None] * ownership.n_runs
    for dataset in range(ownership.n_datasets):
        runs = ownership.runs_of(dataset)
        n_rows = ownership.row_count(dataset)
        g_ref = max(int(grid_lengths[r]) for r in runs)
        start, end, trail_stop = anim_window_bounds(
            num, total_frames, g_ref, window_frames)
        span_ref = n_rows - 1
        p_head = _param(end - 1, g_ref, span_ref)
        p_start = _param(start, g_ref, span_ref)
        p_trail = (None if trail_stop == 0
                   else _param(trail_stop - 1, g_ref, span_ref))
        for r in runs:
            first_row, _ = ownership.run_span(r)
            span = ownership.draw_span(r)
            g_run = int(grid_lengths[r])

            def count_from(p, _a=first_row, _s=span, _g=g_run):
                # a COUNT of drawn grid rows: `data[0:count]`
                if p is None or p < _a:
                    return 0
                if _g < 2 or _s <= 0:
                    # nothing to slide along (a 1-row unbridged run): the
                    # clock either has reached it or has not
                    return _g
                j = min((p - _a) * (_g - 1) / _s, Fraction(_g - 1))
                return min(_g, int(j) + 1)     # int() floors; j >= 0 here

            def index_from(p, _a=first_row, _s=span, _g=g_run):
                # an INDEX into the drawn grid: `data[index:...]`
                if p <= _a:
                    return 0
                if _g < 2 or _s <= 0:
                    return 0
                return min(_g, int((p - _a) * (_g - 1) / _s))

            head_end = count_from(p_head)
            reached = p_head >= first_row
            windows[r] = RunWindow(
                head_start=index_from(p_start),
                head_end=head_end,
                past_stop=count_from(p_trail),
                future_start=max(0, head_end - 1) if reached else 0,
                reached=reached,
                grid=g_run)
    return windows


def run_head_param(window, ownership, run):
    """Source parameter of a run's DRAWN head, or None if it has none.

    The inverse of the projection in `dataset_window_bounds`, and the reason
    the reveal schedule and the renderer cannot describe different states: a
    dataset's visible rows are read back off the windows that were actually
    produced, never computed a second time from the frame index. Everything
    it needs is on the `RunWindow` and the ownership, so it cannot be handed
    a stale frame number.
    """
    if not window.reached or window.head_end <= 0:
        return None
    first_row, _ = ownership.run_span(run)
    span = ownership.draw_span(run)
    if span <= 0 or window.grid < 2:
        # an all-or-nothing run: its single row IS its head
        return Fraction(first_row)
    return first_row + _param(window.head_end - 1, window.grid, span)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS — 11 from Task 1 plus 12 parametrized groups here.

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/trails.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): one reveal clock per dataset for regrouped runs"
```

---

## Task 3: Use it in the matplotlib parallel updaters

**Files:**
- Modify: `hypertools/plot/matplotlib_backend.py:1185` (3-D `update_lines_parallel`), `:2080` (2-D `update_lines_parallel_2d`), `_draw`'s signature at `:431`
- Modify: `hypertools/plot/plot.py` — return `seg_lengths`/`seg_bridge` from the regrouping helper, build the ownership, pass it to `_draw`
- Test: `tests/plot/test_regrouped_reveal.py` (new)

**Interfaces:**
- Consumes: `dataset_window_bounds`, `RunWindow` (Task 2), `TraceOwnership.from_segments` / `.identity` (Task 1).
- Produces: `_draw(..., ownership=None)`; `plot()` passes a `TraceOwnership` for regrouped LINE plots and for unregrouped ones, and `None` otherwise.

**Where the ownership comes from.** `plot.py:341` `_regroup_categorical_lines`
currently returns six values. It must also return the PRE-`patch_lines` run
lengths and `seg_bridge`. The three call sites that unpack it are
`plot.py:4150`, `:4176` and `:4493`.

**When to build it — and when NOT to.** Only two cases produce a valid mapping:

* `_seg_ds is not None` — the LINE regrouping path (`_regroup_categorical_lines`),
  which is what this feature is for.
* No regrouping happened at all *and* the drawn traces correspond positionally
  to the input datasets.

Marker-only categorical regrouping (`plot.py:4498-4509`) goes through
`reshape_data`, which groups **globally by category** across datasets: it
leaves `_seg_ds` as `None` while changing `len(xform)`, and its traces have no
per-dataset row ordering to sweep. It must get `ownership=None` and today's
behaviour. Testing `not isinstance(xform, np.ndarray)`, as an earlier draft
did, would hand it an `identity` ownership describing datasets that do not
exist.

- [ ] **Step 1: Write the failing test**

Create `tests/plot/test_regrouped_reveal.py`:

```python
"""A regrouped trajectory must animate in source-row order.

Real figures through the public API -- these assert what a viewer sees.
"""
import contextlib
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


@contextlib.contextmanager
def no_warnings():
    """Record and assert, never `simplefilter('ignore')`.

    The suite's standing gate is zero warnings; a blanket ignore inside a test
    lets a NEW product warning through silently, which is the failure mode the
    gate exists to catch.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught
    assert not caught, [str(w.message) for w in caught]


def _walk(n=30, seed=0):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n, 3), 0)


def _run_lengths(fig, ani, frame):
    ani._func(frame, *ani._args)
    return [len(line.get_xdata()) for line in fig.axes[0].lines]


def _animate(data, **kwargs):
    with no_warnings():
        return hyp.plot(data, '-', animate=True, duration=2, frame_rate=6,
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
    (Decision R5)."""
    fig, ani = _animate([_walk()], hue=HUE, precog=True)
    ani._func(0, *ani._args)
    trails = [ln for ln in fig.axes[0].lines
              if getattr(ln, '_hyp_row_window', None) is not None]
    assert trails
    lengths = [len(ln.get_xdata()) for ln in trails]
    assert all(n != 1 for n in lengths), lengths


def test_a_MARKER_only_hue_plot_is_untouched():
    """Marker regrouping groups globally by category through `reshape_data`,
    with no per-dataset row order to sweep; it must keep today's behaviour
    rather than be handed an ownership describing datasets that do not
    exist."""
    fig, ani = _animate([_walk()], 'o', hue=HUE)
    drawn = _run_lengths(fig, ani, 3)
    assert all(n > 0 for n in drawn), drawn
```

- [ ] **Step 2: Run the test to verify it fails — and CORRECT the control first**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`

Expected: the sweep tests FAIL, reporting later runs already at 247 points on
frame 3. `test_an_UNREGROUPED_animation_is_unchanged_row_for_row` must PASS
**already**. If it does not, the hard-coded list in this plan is wrong for your
matplotlib/scipy versions: **re-measure it and correct the list BEFORE touching
any implementation**, so it records real pre-change behaviour rather than a
number copied from this document. The same applies to
`test_a_MARKER_only_hue_plot_is_untouched` and
`test_a_precog_trail_on_an_unreached_run_is_not_ONE_STRAY_POINT` — if either
fails at this point, record what it actually does and say so in the commit
message; do not adjust the implementation to a number you have not seen.

If `no_warnings()` fires on any fixture, record the warning text, decide
whether it is expected, and either fix its cause or switch that one test to
`pytest.warns(UserWarning, match=...)`. Do not add a blanket ignore.

- [ ] **Step 3: Return the run lengths and bridge flags from the regrouping helper**

In `hypertools/plot/plot.py`, `_regroup_categorical_lines` (line 341):

```python
    segments, seg_labels, seg_cat, seg_bridge, seg_dataset = segment_by_run(
        xform, hue, labels)
    # BEFORE patch_lines, which appends the next run's first point to every
    # bridged run: TraceOwnership must not be told a run OWNS its neighbour's
    # first observation -- it needs the owned span and the drawn span kept
    # apart (see ownership.TraceOwnership.draw_span).
    seg_lengths = [len(s) for s in segments]
    breaks = {i + 1 for i in range(len(segments) - 1) if not seg_bridge[i]}
    segments = patch_lines(segments, breaks=breaks, labels=seg_labels)
    # what patch_lines ACTUALLY bridged: `seg_bridge` also marks the final
    # run, which has no successor to bridge into
    run_bridged = [bool(seg_bridge[i]) and i < len(segments) - 1
                   for i in range(len(segments))]
```

Change the return to
`return (segments, seg_labels, run_colors, run_group_labels, seg_dataset,
run_category_names, seg_lengths, run_bridged)` and extend its docstring's
"Returns" sentence to name `seg_lengths` ("each run's row count BEFORE
bridging") and `run_bridged` ("whether `patch_lines` appended the next run's
first row").

Update the three unpack sites (`plot.py:4150`, `:4176`, `:4493`) to bind
`_seg_lengths` and `_seg_bridged` as the seventh and eighth values.

- [ ] **Step 4: Build the ownership in `plot()`**

Initialise `_seg_lengths = None` and `_seg_bridged = None` beside
`_seg_ds = None` at `plot.py:3837`. Then, immediately after the block that sets
`_forecast_owner` (it ends at line 5019), add:

```python
    # Run -> dataset -> original rows, for the animation's reveal clock and
    # (below) the forecast schedule. Built for regrouped LINE plots and for
    # unregrouped ones, so both take the same code path; left None for
    # anything whose drawn traces do not correspond to input datasets.
    from .ownership import TraceOwnership
    _ownership = None
    if _seg_ds is not None and _seg_lengths is not None:
        _ownership = TraceOwnership.from_segments(
            _seg_ds, _seg_lengths, _seg_bridged)
    elif (_hue_regrouped_counts is None and isinstance(xform, list)
            and len(xform) == len(raw_xform)):
        # nothing regrouped: one drawn trace per input dataset. MARKER-only
        # hue regrouping (`reshape_data`, plot.py:4503) also leaves `_seg_ds`
        # None while CHANGING the trace count -- it groups globally by
        # category, so its traces are not datasets at all and must not be
        # described as such.
        _ownership = TraceOwnership.identity([len(xi) for xi in raw_xform])
```

Pass `ownership=_ownership` to `_draw(...)` at the call site that already
passes `forecast_schedule=` (`plot.py:5359`).

- [ ] **Step 5: Consume it in both matplotlib updaters**

Add `ownership=None` to `_draw`'s signature (line 431) and import
`dataset_window_bounds` alongside `anim_window_bounds` at line 41.

In `update_lines_parallel`, just after `total_frames` is set (`:1158`), add:

```python
        # ONE clock per source dataset: `hue=`/`cluster=` runs of the same
        # dataset must reveal in row order, not all at once (see
        # `trails.dataset_window_bounds`). Without regrouping this returns
        # exactly what `anim_window_bounds` returned before, frame for frame.
        _windows = None
        if ownership is not None:
            _windows = dataset_window_bounds(
                num, total_frames, ownership,
                [d.shape[0] for d in data_lines], tail_duration)
```

Replace the per-dataset call at `:1185-1186`:

```python
            if _windows is not None:
                win = _windows[i]
            else:
                _s, _e, _ts = anim_window_bounds(
                    num, total_frames, data.shape[0], tail_duration)
                win = RunWindow(_s, _e, _ts, max(0, _e - 1), True,
                                data.shape[0])
            start, end = win.head_start, win.head_end
```

and rewrite the trail block at `:1191-1202` to use the named bounds:

```python
            if trail is not None:
                ct, pc, bt = chemtrails[i], precog[i], bullettime[i]
                trail_seg = None
                if (pc and ct) or bt:
                    trail_seg = _aa_window(i, 0, n_rows, artist=trail)
                elif ct:
                    trail_seg = _aa_window(i, 0, win.past_stop, artist=trail)
                elif pc:
                    # `win.future_start`, never `end - 1`: a run the dataset's
                    # clock has not reached has `end == 0`, and `data[-1:]`
                    # would put one point of a future category on screen.
                    trail_seg = _aa_window(i, win.future_start, n_rows,
                                           artist=trail)
```

(`n_rows = data.shape[0]` already exists at `:1190`; move it above the trail
block.) Import `RunWindow` too.

Apply the identical change to `update_lines_parallel_2d`, computing `_windows`
after its `total_frames` line and replacing `:2080` and the trail block at
`:2084-2091`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`
Expected: PASS (8 tests)

- [ ] **Step 7: Run the animation and trail suites for regressions**

Run: `.venv/bin/python -m pytest tests/plot -q -k "anim or trail or hue or cluster or precog or chemtrail or bullettime"`
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
- Consumes: `dataset_window_bounds`, `RunWindow`, `TraceOwnership`.
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
def _plotly(data, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        with no_warnings():
            return hyp.plot(data, '-', animate=True, duration=2, frame_rate=6,
                            show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _plotly_counts(fig, frame_index):
    frame = fig.frames[frame_index]
    drawn = {t: len(d.x or ()) for t, d in zip(frame.traces, frame.data)}
    return [drawn[t] for t in sorted(drawn)]


def test_plotly_reveals_regrouped_runs_in_the_same_order():
    pytest.importorskip('plotly')
    counts = _plotly_counts(_plotly([_walk()], hue=HUE), 3)
    assert counts[1] == 0 and counts[2] == 0, counts


@pytest.mark.parametrize('hue_arg,label', [
    (HUE, 'three runs'),
    (['A'] * 29 + ['B'], 'a singleton final run'),
    (['A'] * 2 + ['B'] * 26 + ['A'] * 2, 'unequal run lengths'),
    (None, 'unregrouped'),
])
def test_plotly_and_matplotlib_draw_the_SAME_row_counts(hue_arg, label):
    """Both backends consume `dataset_window_bounds`; a transcription drift
    between them is exactly what the `trails` module exists to prevent. An
    empty/non-empty comparison would miss an off-by-one or a mis-scaled
    window, so compare the exact vertex counts, at every frame."""
    pytest.importorskip('plotly')
    kw = {} if hue_arg is None else {'hue': hue_arg}
    pfig = _plotly([_walk()], **kw)
    mfig, ani = _animate([_walk()], **kw)
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q -k plotly`
Expected: FAIL — every run has points on frame 3.

- [ ] **Step 3: Write the implementation**

Add `ownership=None` to `plotly_draw` (`:459`) and import
`dataset_window_bounds`/`RunWindow` at line 52. In the frame loop, before the
per-dataset head loop, compute once per frame `k`:

```python
                # ONE clock per source dataset -- the same call the matplotlib
                # updater makes, so the two backends cannot drift (see the
                # `trails` module docstring for the drift this rule prevents).
                frame_windows = None
                if ownership is not None:
                    frame_windows = dataset_window_bounds(
                        k, n_frames, ownership,
                        [a.shape[0] for a in arrays], window_frames)
```

Replace `:3445` with `start, end = _win(frame_windows, idx, arrays[idx],
k, n_frames, window_frames)[:2]` — where `_win` is a small module-level helper
returning `frame_windows[idx]` when present and otherwise the
`anim_window_bounds`-derived `RunWindow` fallback, exactly as Task 3 Step 5
does for matplotlib — and replace `:3484` with the `past_stop`/`future_start`
fields of the same object, so plotly's precog trail also stops writing
`data[end - 1:]`.

In `plot.py`, pass `ownership=_ownership` at the `plotly_draw(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_regrouped_reveal.py -q`
Expected: PASS (14 tests)

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

## Task 5: `DatasetRevealSchedule` — visible rows, read off the run windows

**Files:**
- Modify: `hypertools/plot/forecast.py` (add after `revealed_raw_counts`, which ends at line 175)
- Test: `tests/plot/test_trace_ownership.py` (append)

**Interfaces:**
- Consumes: `TraceOwnership`; `dataset_window_bounds` / `run_head_param` (Task 2); `matplotlib_backend.serial_reveal_counts(lengths, num, total_frames)`.
- Produces: `DatasetRevealSchedule(ownership, grid_lengths, n_frames, window_frames, serial=False)` with `visible_rows(dataset, frame) -> tuple` and `head_run(dataset, frame) -> int | None`.

**The one rule that makes this correct.** `visible_rows` is derived from the
`RunWindow`s `dataset_window_bounds` actually produced — never recomputed from
the frame index. That is what makes the cross-invariant a tautology instead of
a coincidence: the renderer and the forecast history read the same quantity.

For `serial`, `serial_reveal_counts` already walks runs in order, so a
dataset's visible count is the sum of its runs' revealed rows.

- [ ] **Step 1: Write the failing tests**

Append to `tests/plot/test_trace_ownership.py`:

```python
from hypertools.plot.forecast import DatasetRevealSchedule


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q -k "schedule or head_run or visible or prefix or clamp"`
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

    The rows are read back off the `RunWindow`s
    `trails.dataset_window_bounds` produces -- the SAME objects the backends
    slice their artists with -- rather than recomputed from the frame index.
    Two derivations of "what is on screen" can drift while each passes its own
    tests; one cannot. The bridge vertex `patch_lines` appends is covered by
    the same rule, because a bridged run's drawn span reaches it: the frame on
    which a run completes is the frame on which its successor's first
    observation becomes visible, through either trace.

    `visible_rows` returns the row TUPLE rather than a count. Under the fixed
    reveal it is always a prefix, so the two carry the same information today;
    the tuple is what `ForecastSchedule` memoizes on, so a future reveal that
    exposed a non-prefix would produce a different cache key rather than a
    silent collision between two equal-sized but different histories.
    """

    def __init__(self, ownership, grid_lengths, n_frames, window_frames,
                 serial=False):
        from .trails import dataset_window_bounds, run_head_param
        self.ownership = ownership
        self.grid_lengths = [int(g) for g in grid_lengths]
        self.n_frames = int(n_frames)
        self.window_frames = int(window_frames)
        self.serial = bool(serial)
        self._rows = []
        for frame in range(self.n_frames):
            if self.serial:
                counts = self._serial_counts(frame)
            else:
                windows = dataset_window_bounds(
                    frame, self.n_frames, ownership, self.grid_lengths,
                    self.window_frames)
                counts = []
                for d in range(ownership.n_datasets):
                    head = None
                    for r in ownership.runs_of(d):
                        p = run_head_param(windows[r], ownership, r)
                        if p is not None:
                            head = p if head is None else max(head, p)
                    counts.append(0 if head is None
                                  else min(ownership.row_count(d),
                                           int(head) + 1))
            self._rows.append([tuple(range(k)) for k in counts])

    def _serial_counts(self, frame):
        """`order='serial'` already sweeps runs in order (`serial_reveal_counts`
        walks the trace list), so a dataset's count is the sum of its runs'."""
        from .matplotlib_backend import serial_reveal_counts
        own = self.ownership
        grid_counts = serial_reveal_counts(
            list(self.grid_lengths), frame, self.n_frames)
        out = []
        for d in range(own.n_datasets):
            total = 0
            for r in own.runs_of(d):
                g = self.grid_lengths[r]
                _, n_rows = own.run_span(r)
                span = own.draw_span(r)
                shown = min(grid_counts[r], g)
                if g < 2 or span <= 0 or shown <= 0:
                    total += min(n_rows, max(0, shown))
                else:
                    pos = (shown - 1) * span / (g - 1)
                    total += min(n_rows, int(np.floor(pos)) + 1)
            out.append(min(own.row_count(d), total))
        return out

    def visible_rows(self, dataset, frame):
        """Original row indices of `dataset` on screen at `frame`, in order."""
        return self._rows[min(max(int(frame), 0), self.n_frames - 1)][dataset]

    def head_run(self, dataset, frame):
        """The run DRAWING this dataset's last visible row, or `None` when
        nothing of it is on screen yet."""
        rows = self.visible_rows(dataset, frame)
        if not rows:
            return None
        return self.ownership.run_holding(dataset, rows[-1])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add hypertools/plot/forecast.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): DatasetRevealSchedule reads visible rows off the windows"
```

---

## Task 6: Memoize `ForecastSchedule` on row tuples, and build one from a reveal

**Files:**
- Modify: `hypertools/plot/forecast.py` — `ForecastSchedule.__init__`, `revealed`, `anchor`, `path`, `stacked_paths`, `to_display`; add `for_regrouped`
- Test: `tests/plot/test_trace_ownership.py` (append)

**Interfaces:**
- Consumes: `DatasetRevealSchedule` (Task 5); `forecast_from_history(history, model, t, min_history=...)`.
- Produces: `ForecastSchedule(histories, counts=None, ..., rows=None)` keyed on `(dataset, rows)`; `ForecastSchedule.for_regrouped(histories, reveal, model, t, n_frames, ...)`.

**What actually changes.** Today `self._paths` is keyed `(dataset, count)` and
the history is `histories[i][:k]`. After this task it is keyed
`(dataset, rows)` — a tuple of original row indices — and the history is
`histories[i][list(rows)]`. `for_parallel`/`for_serial` still pass counts;
`__init__` normalizes a count `k` to `tuple(range(k))`, so the fitted arrays
are identical and the existing schedules are byte-for-byte unchanged. This is
Decision R4 made real rather than asserted: v1 converted the tuple straight
back to `len()` and claimed tuple-key memoization on the strength of the
prefix invariant, which is circular.

- [ ] **Step 1: Write the failing tests**

Append to `tests/plot/test_trace_ownership.py`:

```python
import numpy as np

from hypertools.plot.forecast import ForecastSchedule, forecast_from_history


def _history(rows=12, seed=3):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(rows, 3), 0)


def test_the_schedule_keys_on_ROWS_not_counts():
    """Decision R4. Two frames could expose equal-sized but DIFFERENT
    histories; a count key collides silently, a row key does not."""
    h = _history()
    sched = ForecastSchedule([h], rows=[[(0, 1, 2, 3)], [(4, 5, 6, 7)]],
                             model='Kalman', t=3)
    assert set(sched._paths) == {(0, (0, 1, 2, 3)), (0, (4, 5, 6, 7))}
    assert sched.n_fits == 2


def test_a_count_built_schedule_is_UNCHANGED_by_the_row_key():
    """`for_parallel`/`for_serial` still pass counts; normalizing them to
    `range(k)` must fit exactly the same arrays as `histories[i][:k]` did."""
    h = _history()
    sched = ForecastSchedule([h], counts=[[4], [7], [12]], model='Kalman', t=3)
    for k in (4, 7, 12):
        want = forecast_from_history(h[:k], 'Kalman', 3)
        got = sched._paths[(0, tuple(range(k)))]
        assert (want is None) == (got is None)
        if want is not None:
            assert np.allclose(want, got)


def test_for_regrouped_fits_EXACTLY_the_visible_rows():
    """The direct expected fit the review asked for: not 'the early forecast
    differs from the late one' (many wrong subsets would satisfy that), but
    'the forecast IS the one you get from exactly these rows'."""
    own, grids = _one_dataset([6, 6], 12)
    reveal = DatasetRevealSchedule(own, grids, 12, 12)
    h = _history(rows=12)
    sched = ForecastSchedule.for_regrouped([h], reveal, model='Kalman', t=3,
                                           n_frames=12)
    checked = 0
    for frame in range(12):
        rows = reveal.visible_rows(0, frame)
        want = forecast_from_history(h[list(rows)], 'Kalman', 3)
        got = sched.path(0, frame)
        assert (want is None) == (got is None), f'frame {frame}'
        if want is not None:
            assert np.allclose(want, got), f'frame {frame}'
            checked += 1
    assert checked >= 6, f'only {checked} frames had a real fit'


def test_one_EXTRA_row_changes_the_expected_forecast():
    """The control for the test above. Without it the comparison could be
    passing because the fixture is insensitive to its own history."""
    h = _history(rows=12)
    a = forecast_from_history(h[:6], 'Kalman', 3)
    b = forecast_from_history(h[:7], 'Kalman', 3)
    assert a is not None and b is not None
    assert not np.allclose(a, b)


def test_stacked_paths_still_covers_every_forecast_it_will_draw():
    """It feeds the centre/scale statistics, so a missed entry renders a
    forecast outside the cube with the axes off and nothing to clip it."""
    own, grids = _one_dataset([6, 6], 12)
    reveal = DatasetRevealSchedule(own, grids, 12, 12)
    h = _history(rows=12)
    sched = ForecastSchedule.for_regrouped([h], reveal, model='Kalman', t=3,
                                           n_frames=12)
    stacked = sched.stacked_paths()
    for frame in range(12):
        pts = sched.polyline(0, frame)
        if pts is None:
            continue
        for row in pts:
            assert np.any(np.all(np.isclose(stacked, row), axis=1)), frame
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q -k "keys_on_ROWS or count_built or for_regrouped or EXTRA_row or stacked"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rows'`

- [ ] **Step 3: Re-key `ForecastSchedule`**

In `hypertools/plot/forecast.py`:

1. Change the signature to
   `def __init__(self, histories, counts=None, model=None, t=None, rows=None,
   min_history=DEFAULT_MIN_HISTORY, transform=None,
   slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):` and open with:

```python
        if (counts is None) == (rows is None):
            raise ValueError(
                "pass exactly one of counts= (a revealed ROW COUNT per "
                "dataset per frame) or rows= (the revealed ROW INDICES); got "
                f"{'both' if counts is not None else 'neither'}.")
        # ONE internal representation. A count `k` means "the first k rows",
        # which is what every reveal has produced since the runs of a dataset
        # were given a shared clock -- but the KEY is the row tuple, so two
        # frames exposing equal-sized DIFFERENT histories cannot collide in
        # the cache. That is the whole reason this is not `(dataset, count)`.
        if rows is None:
            rows = [[tuple(range(int(k))) for k in frame] for frame in counts]
        self.rows = [[tuple(int(i) for i in r) for r in frame]
                     for frame in rows]
        self.counts = [[len(r) for r in frame] for frame in self.rows]
```

   and delete the old `self.counts = [list(row) for row in counts]`.

2. Replace the `todo` build and the fit loop's key/slice:

```python
        todo = []
        for frame_rows in self.rows:
            for i, r in enumerate(frame_rows):
                if (i, r) not in self._paths:
                    self._paths[(i, r)] = None
                    todo.append((i, r))
        self._paths.clear()
        ...
        for n_done, (i, r) in enumerate(todo):
            start = time.perf_counter()
            path = forecast_from_history(self.histories[i][list(r)],
                                         self.model, self.t,
                                         min_history=self.min_history)
            ...
            self._paths[(i, r)] = path
```

3. Update the lookups:

```python
    def revealed_rows(self, dataset, frame):
        """The ORIGINAL row indices `dataset` has revealed at `frame`."""
        return self.rows[min(frame, self.n_frames - 1)][dataset]

    def revealed(self, dataset, frame):
        """How many raw analyze-space rows `dataset` has revealed at `frame`."""
        return len(self.revealed_rows(dataset, frame))

    def anchor(self, dataset, frame):
        rows = self.revealed_rows(dataset, frame)
        if not rows:
            return None
        return self.histories[dataset][rows[-1]]

    def path(self, dataset, frame):
        return self._paths[(dataset, self.revealed_rows(dataset, frame))]
```

4. `stacked_paths` iterates `for (i, r), path in self._paths.items()` and
   anchors on `self.histories[i][r[-1]]`, skipping `not r`.

5. `to_display` copies `out.rows = self.rows` alongside `out.counts`.

- [ ] **Step 4: Add `for_regrouped`**

Beside `for_parallel` and `for_serial`:

```python
    @classmethod
    def for_regrouped(cls, histories, reveal, model, t, n_frames,
                      min_history=DEFAULT_MIN_HISTORY,
                      slow_warning_seconds=DEFAULT_SLOW_WARNING_SECONDS):
        """Schedule for an animation whose data `hue=`/`cluster=` regrouped.

        The revealed rows come from a `DatasetRevealSchedule` rather than from
        the drawn traces, because a dataset may now be spread over several of
        them. They are passed through as ROW TUPLES, not summarised to counts:
        the reveal is what defines a frame's history, and the cache key must
        say so.
        """
        rows = [[reveal.visible_rows(i, f) for i in range(len(histories))]
                for f in range(n_frames)]
        return cls(histories, rows=rows, model=model, t=t,
                   min_history=min_history,
                   slow_warning_seconds=slow_warning_seconds)
```

- [ ] **Step 5: Run the forecast suite**

```bash
.venv/bin/python -m pytest tests/plot/test_trace_ownership.py -q
.venv/bin/python -m pytest tests -q -k "forecast or predict"
```
Expected: PASS — at least 362 from the second command (the count before this
plan), plus the new ones. Any failure here means the re-keying changed an
existing schedule; fix the code, never the test.

- [ ] **Step 6: Reconcile the frame-count asymmetry**

`plot.py:5081` builds the schedule with `_n_frames = max(2, round(frame_rate *
duration))` while both backends pace with `total_frames = max(1, round(...))`.
The two differ only when `round(frame_rate * duration) < 2`, and
`anim_window_bounds` is sensitive to `total_frames`, so in that case the
schedule and the renderer are on different clocks. Measure it:

```bash
.venv/bin/python - <<'PY'
import matplotlib; matplotlib.use('Agg')
import numpy as np, warnings, hypertools as hyp
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    fig, ani = hyp.plot([np.cumsum(np.random.RandomState(0).randn(8, 3), 0)],
                        '-', predict='Kalman', t=3, animate=True,
                        duration=1, frame_rate=1, show=False)
print('frames:', ani._save_count if hasattr(ani, '_save_count') else '?')
print('warnings:', [str(w.message) for w in caught])
PY
```

Record what it does. If schedule and renderer disagree, make them agree by
passing the backend's `max(1, ...)` value to both, and add a regression test
for that frame count. If they already agree, say so in the commit message. Do
not leave it undiagnosed.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/forecast.py tests/plot/test_trace_ownership.py
git commit -m "feat(plot): memoize forecasts on revealed ROW TUPLES"
```

---

## Task 7: Draw the forecast over a regrouped animation (matplotlib)

**Files:**
- Modify: `hypertools/plot/plot.py:5051-5077` — replace the refusal
- Modify: `hypertools/plot/plot.py:5498-5624` — per-frame forecast colour from the head run
- Test: `tests/plot/test_forecast_animated_regrouped.py` (new)

**Interfaces:**
- Consumes: `DatasetRevealSchedule`, `ForecastSchedule.for_regrouped`, `resolve_forecast_overrides`, `_forecast_style_from`, `trail_frames`.
- Produces: no new public API.

**What the refusal currently does** (`plot.py:5051`): when `animate` is set and
`len(analyze_histories) != len(xform)`, it warns, records
`_forecast_draw_reason`, and sets both `analyze_histories` and `raw_forecasts`
to `None`, so no schedule is built and no static overlay leaks onto the plotly
backend. The fit itself still succeeds and `bundle_forecasts` is untouched, so
`return_model=True` already returns the forecasts with `drawn=False`. After
this task the animated regrouped case draws, so `drawn` becomes True and
`draw_reason` stays `None`. The CONTINUOUS-hue refusal above it
(`:5002-5019`) is untouched — a `LineCollection` has no per-dataset trace to
anchor to at all.

**The colour rule (Decision R3).** `_forecast_style_from(_src_lines[_i], ...)`
takes artist `i`'s colour at BUILD time. With more traces than datasets that is
the wrong trace, and it is fixed for the animation besides. The live artist's
colour must be set per frame from `reveal.head_run(i, frame)`, and each trail
slot's from `reveal.head_run(i, past[age - 1])` — the head run at the frame
that forecast was FIT, so a boundary crossing does not repaint the history.
A resolved `_forecast_overrides[i]` with a `'color'` key wins over both, as it
already does on the static path.

- [ ] **Step 1: Write the failing test**

Create `tests/plot/test_forecast_animated_regrouped.py`:

```python
"""Forecasts over an animation whose data hue=/cluster= regrouped."""
import contextlib
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

import hypertools as hyp

HUE = ['A'] * 10 + ['B'] * 10 + ['A'] * 10


@contextlib.contextmanager
def no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught
    assert not caught, [str(w.message) for w in caught]


def _walks(n=2, rows=30, seed=0):
    rng = np.random.RandomState(seed)
    return [np.cumsum(rng.randn(rows, 3), 0) for _ in range(n)]


def _artists(fig, role=None):
    return [ln for ln in fig.axes[0].lines
            if getattr(ln, '_hyp_forecast_role', None) is not None
            and (role is None or ln._hyp_forecast_role == role)]


def _rgb(artist):
    return tuple(np.round(matplotlib.colors.to_rgb(artist.get_color()), 5))


def _animate(data=None, hue=None, **kwargs):
    data = _walks() if data is None else data
    hue = (HUE * len(data)) if hue is None else hue
    with no_warnings():
        return hyp.plot(data, '-', hue=hue, predict='Kalman', t=4,
                        animate=True, duration=2, frame_rate=6, show=False,
                        **kwargs)


def test_a_regrouped_animation_now_DRAWS_its_forecasts():
    fig, ani = _animate()
    ani._func(11, *ani._args)
    live = [a for a in _artists(fig, 'live') if a.get_visible()]
    assert live, 'no live forecast artist was drawn'


def test_it_no_longer_warns_that_it_cannot_draw_them():
    """`no_warnings()` already asserts this, but name it: the refusal warning
    disappearing is the observable half of the feature."""
    fig, ani = _animate()
    assert _artists(fig, 'live')


def test_the_bundle_reports_drawn_True():
    with no_warnings():
        out = hyp.plot(_walks(), '-', hue=HUE * 2, predict='Kalman', t=4,
                       animate=True, duration=2, frame_rate=6, show=False,
                       return_model=True)
    info = out[-1]['predict']
    assert info['drawn'] is True
    assert info['draw_reason'] is None


def test_the_final_frame_forecast_equals_the_STATIC_one():
    """The animation's last frame has the whole history, so its forecast must
    be the one a static plot of the same data draws -- otherwise the animated
    and static paths disagree about the same model on the same rows."""
    data = _walks()
    fig, ani = _animate(data)
    with no_warnings():
        sfig = hyp.plot(data, '-', hue=HUE * 2, predict='Kalman', t=4,
                        show=False)
    ani._func(11, *ani._args)
    live = sorted(np.asarray(a.get_xdata())[-1] for a in _artists(fig, 'live')
                  if a.get_visible())
    static = sorted(np.asarray(a.get_xdata())[-1]
                    for a in _artists(sfig, 'static'))
    assert np.allclose(live, static, atol=1e-8)


def test_an_EARLY_forecast_is_not_the_full_history_one():
    """A smoke test only -- the exact 'fit on precisely these rows' assertion
    lives in `test_for_regrouped_fits_EXACTLY_the_visible_rows`, where the
    expected fit can be computed directly. Here it guards the WIRING: that
    `plot()` handed the animation the reveal schedule and not the static
    forecast."""
    fig, ani = _animate()
    ani._func(2, *ani._args)
    early = [np.asarray(a.get_xdata())[-1] for a in _artists(fig, 'live')
             if a.get_visible()]
    ani._func(11, *ani._args)
    late = [np.asarray(a.get_xdata())[-1] for a in _artists(fig, 'live')
            if a.get_visible()]
    assert early and late
    assert not np.allclose(sorted(early), sorted(late))


def test_the_live_forecast_takes_the_HEAD_run_colour():
    """Decision R3: the forecast continues the head, so it wears that run's
    colour and changes with it at a category boundary."""
    fig, ani = _animate(_walks(n=1), hue=HUE)
    seen = set()
    for frame in range(12):
        ani._func(frame, *ani._args)
        seen.update(_rgb(a) for a in _artists(fig, 'live')
                    if a.get_visible())
    assert len(seen) > 1, 'the forecast kept one colour across a boundary'


def test_a_RETAINED_trail_keeps_the_colour_it_was_FIT_with():
    """Decision R3's second half. A fan member drawn before the boundary must
    stay in category A when the live forecast moves to B; repainting the whole
    fan would make a saved animation differ from a played one."""
    fig, ani = _animate(_walks(n=1), hue=HUE, forecast_trail=8)
    boundary = None
    prev = None
    for frame in range(12):
        ani._func(frame, *ani._args)
        live = [a for a in _artists(fig, 'live') if a.get_visible()]
        if not live:
            continue
        now = _rgb(live[0])
        if prev is not None and now != prev:
            boundary = frame
            break
        prev = now
    assert boundary is not None, 'no category boundary was crossed'
    ani._func(boundary, *ani._args)
    fan = sorted((a._hyp_forecast_age, _rgb(a)) for a in _artists(fig, 'trail')
                 if a.get_visible())
    assert fan, 'no retained forecast was drawn at the boundary'
    assert len({c for _, c in fan}) > 1 or fan[0][1] == prev, (
        f'the fan was repainted to the new category: {fan}')


def test_replaying_frames_does_not_MUTATE_the_fan():
    """`save()`/`to_jshtml()` replay from 0 and may deliver frames out of
    order; a colour that depended on frame HISTORY would differ between a
    saved and a played animation."""
    fig, ani = _animate(_walks(n=1), hue=HUE, forecast_trail=8)
    def snapshot(frame):
        ani._func(frame, *ani._args)
        return sorted((a._hyp_forecast_age, _rgb(a),
                       len(a.get_xdata())) for a in _artists(fig, 'trail')
                      if a.get_visible())
    first = snapshot(9)
    for frame in (0, 4, 11, 2, 7):
        snapshot(frame)
    assert snapshot(9) == first


def test_forecast_cluster_still_holds_ONE_colour_across_frames():
    """The override path is unchanged by Decision R3: an explicit grouping is
    resolved once from the full-history forecasts and fixed for every frame."""
    fig, ani = _animate(_walks(n=4, rows=20), hue=['A'] * 10 + ['B'] * 10,
                        forecast_cluster='KMeans', forecast_n_clusters=2)
    seen = {}
    for frame in (0, 4, 9, 11, 2, 11, 7):
        ani._func(frame, *ani._args)
        for a in _artists(fig, 'live'):
            if a.get_visible():
                seen.setdefault(id(a), set()).add(_rgb(a))
    assert seen
    assert all(len(v) == 1 for v in seen.values()), seen


def test_a_CONTINUOUS_hue_still_refuses_and_says_so():
    """The other refusal is untouched: a LineCollection has no per-dataset
    trace to anchor a forecast to at all."""
    data = _walks(n=1)
    with pytest.warns(UserWarning, match='no per-dataset trace'):
        hyp.plot(data, '-', hue=list(range(30)), predict='Kalman', t=4,
                 animate=True, duration=2, frame_rate=6, show=False)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: FAIL — `no_warnings()` catches the refusal warning and no live
artist exists.

- [ ] **Step 3: Replace the refusal in `plot()`**

In `plot.py`, replace the block at `:5051-5077` with:

```python
    _reveal = None
    if (raw_forecasts is not None and analyze_histories is not None
            and animate and animate not in ('spin',)
            and len(analyze_histories) != len(xform)):
        if _ownership is None:
            # no per-dataset mapping at all: keep the refusal rather than
            # guessing which trace a forecast belongs to
            warnings.warn(... unchanged text ...)
            _forecast_draw_reason = ... unchanged ...
            analyze_histories = None
            raw_forecasts = None
        else:
            # hue=/cluster= regrouped the data into one trace per RUN. The
            # forecast is still per DATASET: `DatasetRevealSchedule` maps each
            # frame onto that dataset's own visible rows, read off the SAME
            # `RunWindow`s the backends slice their artists with.
            from .forecast import DatasetRevealSchedule
            _reveal = DatasetRevealSchedule(
                _ownership, [xi.shape[0] for xi in xform],
                n_frames=max(2, int(round(frame_rate * duration))),
                window_frames=int(frame_rate * (focused if focused is not None
                                                else tail_duration)),
                serial=(animate == 'serial' or order == 'serial'))
```

**Implementer: `window_frames` must be the value the BACKEND computes**, at
`matplotlib_backend.py:1972-1976` (`focused` when `style == 'window'` or any
trail flag is set, else `tail_duration`, times `frame_rate`, floored at 1 when
zero). Read that block and mirror it exactly, or extract it into a shared
helper in `trails.py` and call it from both — the latter is preferred, and is
the same "one callee cannot drift from itself" rule the module already states.
Verify with a test that `DatasetRevealSchedule`'s windows equal the ones the
figure actually drew (compare `visible_rows` to per-run artist vertex counts on
a real animation).

Then extend the schedule build at `:5078-5092`:

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

- [ ] **Step 4: Colour the live and trail artists per frame**

Pass `_reveal` and `_ownership` into the artist-building block at `:5498`. Keep
the build-time `_forecast_style_from` call for linestyle/linewidth/alpha, but
add to `_update_forecasts` (`:5571-5624`), inside the per-dataset loop:

```python
                    def _run_colour(dataset, frame):
                        """Decision R3: the colour of the run DRAWING the head
                        at `frame` -- for the live forecast, the current
                        frame; for a retained one, the frame it was FIT at, so
                        a boundary crossing does not repaint the fan."""
                        if _reveal is None or _override_colour[dataset]:
                            return None
                        run = _reveal.head_run(dataset, frame)
                        if run is None or run >= len(_src_lines):
                            return None
                        return _src_lines[run].get_color()
```

with `_override_colour[i]` recording whether `_forecast_overrides[i]` supplied
a `'color'`. Apply it in `_fill`: `colour = _run_colour(i, ctx.frame)` for the
live artist and `_run_colour(i, past[_age - 1])` for trail slot `_age`, calling
`art.set_color(colour)` when it is not `None`. `_src_lines` is already the
snapshot of `ax.lines` taken before any forecast artist was added, so index
`run` is the observed run's artist.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: PASS (11 tests)

- [ ] **Step 6: Run the whole forecast suite**

Run: `.venv/bin/python -m pytest tests -q -k "forecast or predict"`
Expected: PASS — at least 362 (the count before this plan) plus the new ones.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/forecast.py hypertools/plot/plot.py \
        tests/plot/test_forecast_animated_regrouped.py
git commit -m "feat(plot): draw forecasts over regrouped animations"
```

---

## Task 8: Plotly parity, docs, CHANGELOG

**Files:**
- Modify: `hypertools/plot/plotly_backend.py` — the live/trail forecast colour rule
- Modify: `docs/animation.rst`, `CHANGELOG.md`
- Test: `tests/plot/test_forecast_animated_regrouped.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: no new API.

- [ ] **Step 1: Write the failing plotly tests**

Append to `tests/plot/test_forecast_animated_regrouped.py`:

```python
def _plotly(data, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        with no_warnings():
            return hyp.plot(data, '-', predict='Kalman', t=4, animate=True,
                            duration=2, frame_rate=6, show=False, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _roles(fig):
    return {i: (tr.meta or {}).get('hyp_forecast_role')
            for i, tr in enumerate(fig.data)}


def test_plotly_draws_forecasts_over_a_regrouped_animation():
    pytest.importorskip('plotly')
    fig = _plotly(_walks(), hue=HUE * 2)
    roles = _roles(fig)
    assert 'live' in roles.values(), 'no live forecast trace was built'
    last = fig.frames[-1]
    drawn = [d for t, d in zip(last.traces, last.data)
             if roles.get(t) == 'live']
    assert drawn and any(len(d.x or ()) for d in drawn)


def test_both_backends_end_the_animation_at_the_same_forecast():
    pytest.importorskip('plotly')
    data = _walks()
    pfig = _plotly(data, hue=HUE * 2)
    mfig, ani = _animate(data)
    ani._func(11, *ani._args)
    mpl = sorted(float(np.asarray(a.get_xdata())[-1])
                 for a in _artists(mfig, 'live') if a.get_visible())
    roles = _roles(pfig)
    last = pfig.frames[-1]
    ply = sorted(float(d.x[-1]) for t, d in zip(last.traces, last.data)
                 if roles.get(t) == 'live' and (d.x or ()))
    assert np.allclose(mpl, ply, atol=1e-8)


def test_both_backends_agree_at_EVERY_frame_not_just_the_last():
    """A final-frame-only check passes for any schedule that ends correctly,
    including one that reveals at a different rate throughout."""
    pytest.importorskip('plotly')
    data = _walks(n=1)
    pfig = _plotly(data, hue=HUE)
    mfig, ani = _animate(data, hue=HUE)
    roles = _roles(pfig)
    for f in range(12):
        ani._func(f, *ani._args)
        mpl = sorted(float(np.asarray(a.get_xdata())[-1])
                     for a in _artists(mfig, 'live') if a.get_visible())
        frame = pfig.frames[f]
        ply = sorted(float(d.x[-1]) for t, d in zip(frame.traces, frame.data)
                     if roles.get(t) == 'live' and (d.x or ()))
        assert np.allclose(mpl, ply, atol=1e-8), f'frame {f}'


def test_plotly_forecast_colour_follows_the_head_run_too():
    pytest.importorskip('plotly')
    fig = _plotly(_walks(n=1), hue=HUE)
    roles = _roles(fig)
    seen = set()
    for frame in fig.frames:
        for t, d in zip(frame.traces, frame.data):
            if roles.get(t) == 'live' and (d.line or {}).get('color'):
                seen.add(d.line['color'])
    assert len(seen) > 1, f'the plotly forecast kept one colour: {seen}'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q -k "plotly or both_backends"`
Expected: FAIL — no live trace in the plotly figure.

- [ ] **Step 3: Implement plotly's side and update the docs**

Mirror Task 7's colour rule where plotly builds the live/trail forecast traces
(they already consume `forecast_schedule` and the resolved override map): take
the head run's colour from the reveal schedule when no override supplies one,
setting it per FRAME (plotly frames may carry `line.color`, unlike the fixed
observed traces — see `plotly_backend.py:674`, the plotly half of matplotlib's
`_hyp_forecast_role` tag).

In `docs/animation.rst`, replace the paragraph stating that animated forecasts
are unsupported under `hue=`/`cluster=` with:

```rst
Animated forecasts under ``hue=``/``cluster=``
----------------------------------------------

When ``hue=`` or ``cluster=`` splits a trajectory into per-category runs, each
run is drawn as its own trace, but the reveal still follows the **dataset**:
one run finishes as the next begins, so the trajectory sweeps once and changes
colour as it crosses a category boundary. (Before 1.1 every run advanced at
once, so one trajectory animated in several places simultaneously.) A forecast
is fit per dataset from exactly the observations on screen, so it means the
same thing it does without ``hue=``.

A live forecast inherits the colour of the run drawing the head, and therefore
changes colour with it; a retained ``forecast_trail=`` member keeps the colour
it was drawn with, so the fan records the history rather than being repainted.
Pass ``forecast_hue=``, ``forecast_cluster=`` or ``forecast_palette=`` to give
the forecasts a grouping of their own; those are resolved once from the
full-history forecasts and stay fixed for every frame.

Because each run is resampled onto its own frame grid, a regrouped reveal can
lag the un-regrouped one by up to one grid step -- under a single frame of
trajectory, and never early, so a forecast is never fit on an observation that
has not been drawn.

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
  A `precog=` trail on a not-yet-reached run now shows that run's whole future
  rather than a single stray point.

- **`predict=` now works with `hue=`/`cluster=` on ANIMATED plots.** Previously
  the fit succeeded and the forecasts were returned in the `return_model=True`
  bundle with `drawn=False`, but no overlay was drawn. Each frame's forecast is
  fit from exactly the observations visible for that dataset. A live forecast
  inherits the colour of the run drawing the head; a retained
  `forecast_trail=` member keeps the colour it was fit with;
  `forecast_hue=`/`forecast_cluster=`/`forecast_palette=` override both with a
  grouping fixed for the whole animation. A continuous `hue=` still draws no
  overlay.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/plot/test_forecast_animated_regrouped.py -q`
Expected: PASS (15 tests)

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
never `git stash`). If any check fails, fix it and re-run **all** of them: a
fix for one can break another.

- [ ] **Step 6: Commit**

```bash
git add hypertools/plot/plotly_backend.py docs/animation.rst CHANGELOG.md \
        tests/plot/test_forecast_animated_regrouped.py
git commit -m "feat(plot): plotly parity + docs for animated regrouped forecasts"
```

---

## Self-Review

**1. Review coverage.** Every finding from the v2 review maps to a task:

| Review finding | Where |
|-|-|
| Bridge rows break "exactly visible history" | Task 1 (`bridged_by_run`/`draw_span`), Task 2 (`test_a_run_completes_exactly_when_the_NEXT_one_starts`) |
| `head_run` ambiguous at a boundary | Task 5 (`test_the_head_run_is_the_run_DRAWING_the_last_visible_row`) |
| Cross-invariant between windows and visible rows | Task 5 (`test_the_schedule_agrees_with_the_RENDERED_windows_at_every_frame`), enforced structurally by `run_head_param` |
| Tuple-key memoization promised but not implemented | Task 6 (`rows=`, `_paths[(i, rows)]`, `histories[i][list(rows)]`) |
| precog `end == 0` -> `data[-1:]` | Task 2 (`RunWindow.future_start`, R5), Task 3 Step 5, Task 4 Step 3 |
| Projected grids vs source spans | Task 1 `draw_span`, Task 2 arithmetic step 4 |
| Trail forecast colours unspecified | R3, Task 7 (`test_a_RETAINED_trail_keeps_the_colour_it_was_FIT_with`, `test_replaying_frames_does_not_MUTATE_the_fan`) |
| Non-discriminating invisible-observation test | Task 6 (`test_for_regrouped_fits_EXACTLY_the_visible_rows` + mutation control) |
| Blanket warning suppression | Global Constraints + `no_warnings()` in every test module |
| Dense dataset ids unvalidated | Task 1 (`test_sparse_or_unordered_dataset_ids_are_rejected`) |
| Ownership construction too broad | Task 3 Step 4 + `test_a_MARKER_only_hue_plot_is_untouched`; scope stated in Global Constraints |
| Weak plotly parity | Task 4 (exact counts, 4 hue shapes, boundary/final frames), Task 8 (every frame) |

**2. Placeholder scan.** No "TBD" / "add error handling" / "similar to Task N".
Two steps deliberately require the implementer to read and mirror existing code
rather than copy a snippet — Task 7 Step 3's `window_frames` (which must match
`matplotlib_backend.py:1972-1976`) and Task 4 Step 3's `_win` helper — and both
say exactly which lines to read and what to verify afterwards.

**3. Type consistency.** `TraceOwnership.from_segments(seg_dataset,
seg_lengths, seg_bridge)` is called with those names in Task 3; `run_span`
returns `(first_row, n_owned_rows)` and `draw_span` returns an int, used that
way in Tasks 2, 5 and 6; `dataset_window_bounds` returns a list of `RunWindow`
indexed by run and is indexed as `_windows[i]` / `frame_windows[idx]` in Tasks
3 and 4; `DatasetRevealSchedule(ownership, grid_lengths, n_frames,
window_frames, serial=)` is constructed with that signature in Tasks 5, 6 and
7; `visible_rows` returns a tuple and Task 6 passes it to `rows=` unchanged.

**Known gaps, deliberate and flagged.**

- `test_an_UNREGROUPED_animation_is_unchanged_row_for_row` hard-codes point
  counts measured on `dev-1.0` at `a062f768`. Task 3 Step 2 requires the
  implementer to re-measure and correct them BEFORE changing any
  implementation.
- Task 6 Step 6 opens the `max(1, ...)` vs `max(2, ...)` frame-count asymmetry
  between `plot.py:5081` and the backends. It predates this work; it is
  measured and resolved here rather than inherited.
