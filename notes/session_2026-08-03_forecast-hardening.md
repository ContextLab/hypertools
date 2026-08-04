# Session record — 2026-08-03 — forecast override hardening + missing hue labels

Branch `dev-1.0`. Continues `session_2026-08-02_forecast-overrides.md`.

Worked the maintainer's second review of the `forecast_*=` work. Their
verdict was "nearly ready, two small hardening changes"; both are done,
plus one defect found while doing them.

| # | Item | State |
|-|-|-|
| 1 | document fixed full-history clustering for animated `forecast_cluster=` | done |
| 2 | cross-backend tests for stable live/trail cluster colours | done |
| 3 | harden `forecast_fmt=` / `forecast_hue=` validation | done |
| 4 | normalize or reject missing forecast-hue labels | normalized |
| 5 | capture the expected test warnings | done (see below) |
| 6 | focused forecast suite | 182 passed, 0 warnings |
| 7 | full suite + zero-warning docs build | see Verification |
| 8 | mark the static forecast override work complete | complete |
| 9 | `TraceOwnership`/`DatasetRevealSchedule` | NOT started — its own task |
| 10 | animated observed hue/cluster forecasts | unchanged, warning-only |

## Finding 1 — animated clustering was right but unsaid

The implementation already resolved the endpoint groups ONCE, from the
full-history forecasts, and held them for every frame. Nothing said so, and
the public wording ("coloured by where it is predicted to end up") reads
just as naturally as "the CURRENT frame's predicted endpoint". Documented in
three places (docstring, `docs/animation.rst`, CHANGELOG) with the reason:

- cluster labels are arbitrary names for groups, so per-frame reclustering
  would let a forecast change colour whenever a fit nudged its endpoint over
  a boundary
- it would repaint a retained `forecast_trail=` fan whose earlier members
  were drawn under the OLD grouping
- a fixed assignment keeps a saved animation identical to a played one — the
  same reason the schedule itself is precomputed

**The load-bearing test is the control**, again:
`test_the_animations_EARLY_forecasts_would_cluster_DIFFERENTLY`. If a
partly-revealed history and the full one grouped the datasets the same way,
a per-frame implementation would ALSO hold the colours still, and neither
stability test could tell the two policies apart. Verified numerically
before use: at k=10 rows `_converging` groups {0,1}|{2,3} (x still dominates),
at k=40 it groups {0,2}|{1,3} (x has collapsed, y decides).

The stability test drives frames **out of order and repeated** (0, 4, 9, 11,
2, 11, 7, 0, 5) because `save()`/`to_jshtml()` do exactly that — a colour
depending on frame HISTORY rather than frame INDEX fails. Mutation-checked:
recolouring one live artist mid-run is caught, so the assertion has teeth.

On plotly the check is structural: frames carry geometry only, so the test
asserts no frame sets `line.color` on a forecast trace. A frame that did
would be per-frame clustering by another name.

## Finding 2 — four inputs that failed as internal errors

`resolve_forecast_overrides` is internal but independently importable, and
each of these is a plausible typo that named nothing:

- `forecast_fmt=3` → "'int' object is not iterable"
- `forecast_fmt=b'--'` → `list(b'--') == [45, 45]`, two ints silently taken
  as one format per dataset. **Decoded**, since that is the only reading
  that is not nonsense.
- `forecast_fmt=['-', 3]` → reached a different layer on each backend
- `forecast_hue='ab'` → one label per CHARACTER. Silently right by accident
  at two datasets; at any other count, wrong with a length message that
  reads as nonsense.
- `forecast_cluster=` on absent / empty / ragged / non-finite forecasts →
  IndexError, `vstack`'s dimension message, or sklearn's "Input contains
  NaN", none of which mentions a forecast

`forecast_fmt=` is now validated with matplotlib's own `fmt=` parser, so it
provably accepts exactly what `fmt=` accepts. `_process_plot_format` now has
ONE guarded import (in `forecast.py`, which validates with it) rather than
two — the check and the application cannot end up using different parsers.

## The defect found on the way: a missing hue label per NaN

The maintainer's note said to normalize missing `forecast_hue=` labels
"consistent with ordinary categorical hue". Checking what ordinary
categorical hue does showed it does not normalize them either:

    hue = ['a']*4 + [float('nan')]*2 + ['b']*4 + [float('nan')]*2
    -> FOUR categories in four saturated colours

`nan != nan`, so two missing labels are not equal and become two SEPARATE
categories — and since `np.nan` is a singleton while `float('nan')` is a
fresh object each time, WHICH of those happens depends on how the caller
spelled it. The library already has a sentinel for "unlabeled" (`None`,
F02-013: one group, neutral gray, no legend entry, no palette slot), so
every missing spelling (`None`, NaN, `pd.NA`) now normalizes to it — in
`hue=` and in `forecast_hue=`.

Guarded on "some entry is a string" so a purely numeric hue (binned as
continuous values, where non-finite entries are already handled by
`mat2colors` → `NAN_COLOR`) is untouched.

`plot._UNLABELED_HUE_COLOR` is now `colors.NAN_COLOR` itself rather than a
second literal of the same value; the comment already claimed they were the
same gray, and they were one edit away from disagreeing.

## Test warnings

The two `'singleton runs'` warnings are now CAPTURED rather than filtered,
with an assertion that the other three fixtures provoke none and that
nothing else warns at all — so a new warning fails the test instead of
scrolling past behind the expected two. A filter would equally have hidden
the notice disappearing, and the fixture exists to provoke it.

**The joblib physical-core warning does not reproduce here.** It is
`joblib.externals.loky`'s notice when its CPU probe fails, which is a
property of the machine (containers/VMs), not of this suite: the focused
forecast run on this box emits zero warnings. Suppressing it would mean
either a global `UserWarning` filter (which the maintainer ruled out) or
setting `LOKY_MAX_CPU_COUNT` for every test run, which changes real
parallelism to quiet someone else's environment. Left alone, flagged.

## Lint

`ruff check` on every touched file: **161 findings before, 161 after**,
compared against `c8c4f533` in a scratch worktree. All pre-existing, all
from a star import that predates this work.

## Round 2 — the resolver as an independently callable interface

The maintainer's follow-up review accepted the `plot()` path and asked for
three boundary guards on `resolve_forecast_overrides` itself. `plot()` cannot
trigger any of them; a direct caller can, and each failed *quietly or
misleadingly* rather than loudly:

| input | before | now |
|-|-|-|
| forecast count != `n_datasets` | 4 for 3: `IndexError` writing `overrides[3]`. 2 for 3: this resolver's OWN message, "it has 3 point(s) to work with" — a count read off `n_datasets` when 2 endpoints were stacked | `ValueError` naming both counts |
| a raw `(t,)` forecast | `atleast_2d` -> `(1, t)`: the whole trajectory as ONE t-dimensional endpoint, clustered silently | `ValueError` naming the shape |
| a ragged nested list | numpy's "inhomogeneous shape", naming neither kwarg nor forecast | `ValueError` naming forecast `i` |
| an unhashable `forecast_hue=` label | bare `TypeError: unhashable type: 'dict'` from the colour code | `TypeError` naming the kwarg |

Two things the red phase changed about the tests:

- The count test's *first* draft asserted against the wrong evidence. Running
  it showed the 2-for-3 case already raised — but with a message asserting
  three points existed. The docstring now records that, because "it already
  raises" was the tempting and wrong reading.
- The 1-D test used two forecasts and passed for a reason unrelated to
  shape: KMeans' default `n_clusters=3` exceeded the 2 samples. At **four**
  datasets the old code SUCCEEDS, clustering trajectories. The test uses four
  so the guard is what makes it raise.

`hash(v)` rather than `isinstance(v, Hashable)`: the ABC only asks whether
`__hash__` exists, and a tuple holding a list has one that raises when called.

Verified empirically before making the 2-D rule strict: every forecast
reaching the resolver through `plot()` is 2-D, including one-feature input
(`(20, 1)` in -> `(11, 1)` forecast), and at every `reduce=` setting.

## Round 3 — regrouped-reveal plan v2

The maintainer reviewed the plan written at the end of round 2 and found it
**not implementable**: architectural direction right, several low-level
contracts contradicting either the renderer or the plan's own guarantees.
Three were fatal. v2 addresses all of them; the full mapping is the table at
the top of the plan document.

The three that mattered most, and what they turned out to be:

- **Bridge rows.** `patch_lines` appends the NEXT run's first observation to
  every bridged run, so that observation is on screen while `visible_rows`
  reported only the preceding run's owned rows. v1 excluded the bridge from
  ownership (correct) and then paced the DRAWN geometry with the OWNED span
  (wrong), desynchronising every category boundary by one vertex. Fixed by
  carrying `bridged_by_run` and separating `run_span` (owned) from
  `draw_span` (`n_rows - 1 + bridged`, the rendered polyline).
- **Two derivations of "what is on screen".** v1 computed run windows one way
  and visible rows another; they can drift while each passes its own tests.
  v2 makes the windows primary and reads the rows back off them
  (`run_head_param`), so the cross-invariant is structural, not a coincidence.
- **Tuple-key memoization was claimed, not implemented.** v1's `for_regrouped`
  converted the row tuple straight to `len()`, then argued the prefix
  invariant made that equivalent — circular. `ForecastSchedule` now stores
  rows and keys `_paths` on `(dataset, rows)`, slicing
  `histories[i][list(rows)]`; counts are a derived view.

Also fixed: `precog`'s `data[end - 1:]` becoming `data[-1:]` when a run has
`end == 0` (four named bounds on a `RunWindow` instead of three overloaded
integers); retained forecast-trail colours (Decision R3 now uses the head run
at the frame each was FIT, not the current one, or a boundary crossing
repaints the whole fan); ownership built from a too-broad predicate that would
have mis-described marker-only regrouping; blanket `simplefilter('ignore')` in
the test bodies; and weak plotly parity assertions.

### The arithmetic was measured, not reasoned

**v1's projection was wrong and its own self-review had already caught one
version of it.** v2's was verified BEFORE being written into the plan, in
`Fraction` arithmetic against the real `anim_window_bounds` /
`revealed_raw_counts`, over every frame of 13 regrouped cases + 6 unregrouped
ones. Findings:

- floor-quantizing each run's head is exact for the unregrouped case (the
  identity, every frame) and never lets a split dataset reveal a row EARLY
- it lags by at most one run-grid step; `ceil` halves the lag but breaks
  bridge simultaneity in 6 places, so floor wins
- a 1-row unbridged final run has no grid to slide along and needs an
  explicit all-or-nothing rule; without it, two bridge violations
- the "split reveals at the same time as unsplit" claim v1 made is FALSE and
  was removed — what holds is "never early, bounded by one grid step", which
  is the direction that matters (no forecast is fit on an undrawn row)

The plan's own Task 1/Task 2 test bodies were then transcribed verbatim
alongside its implementation code and run: **109 passed**, after one
test-drafting fix (a sparse-dataset-id case tripped the dense check before
the order check, so it passed for the wrong reason — the same failure mode as
round 2's two red-phase corrections).

## Still open (unchanged, by the maintainer's scoping)

- **`TraceOwnership` / `DatasetRevealSchedule`** — the missing mapping is
  `run index -> source dataset -> original source-row indices`, not
  run-count -> dataset-count. A run holding source rows `[0, 3, 7]` at
  revealed count 2 has NOT revealed its dataset's first two observations.
  Memoize on `(dataset, visible_source_row_tuple)`, never `(dataset, count)`:
  two frames can expose different observation sets of the same size.
- **Animated forecasts under regrouping** — blocked on the above. Fit to
  all currently visible observations sorted by source index; a
  contiguous-prefix rule stalls whenever another run exposes later rows.

## Verification

- Focused forecast suite: **182 passed, 0 warnings**
- Full suite: **2976 passed, 13 skipped, 0 failed** (10m45s), and **no
  warnings summary section at all** -- the previous run was 2960 passed
  with 2 warnings, so the +16 tests are exactly the ones added here and the
  two singleton notices are now asserted rather than printed
- Docs: `sphinx -W -E -a` **build succeeded** (warnings are errors)
- Commits: `ed7fe3a3` (validation + missing labels),
  `9c227d55` (animated clustering docs, tests, warning capture)

### Round 2 (resolver boundary guards)

- `tests -k "forecast or predict"`: **362 passed**
- Full suite: **2981 passed, 13 skipped, 0 failed** (10m57s), still **no
  warnings summary section** -- +5 over the round-1 run, exactly the 5 tests
  added (2 parametrized count cases, 1-D, ragged, unhashable)
- Docs: `sphinx -W -E -a` **build succeeded**
- Ruff on both touched files: **1 finding before, 1 after** (the pre-existing
  star-import one), compared against `a2550259`

---

# Round 4 — review of plan v2 + Plan 4 (2026-08-04)

Ten-item review arrived. Items 1-5 = Plan 4, 6 = the singleton edge case,
7 = Plan 3's stale prose, 9 = the Kaleido failure. Items 8 and 10
(implement the regrouped plan, then start Plan 4) are the maintainer's
call and are NOT started.

## Item 6 — the one-row-dataset edge case (`5bbeb50d`)

**The stated trigger does not exist.** The review's fix keys on
`anim_window_bounds` returning `end == 0`, but it clamps
`end = max(1, min(n_points, end))` (`trails.py:86`). Swept every frame of
7488 `(total, grid, window)` combinations: **zero** hits. `end == 0` in
`RunWindow`'s docstring comes from the PROJECTION (`count_from`, for a run
the clock has not reached), never from `anim_window_bounds`. Docstring
corrected to say so.

**The concern is still right in kind.** `reached = p_head >= first_row` was
correct only because `_param` returns 0 for a dataset with no extent and
the first run's `first_row` is also 0 -- a degenerate value coinciding with
a real boundary, i.e. a second derivation of what `head_end` already
carries. Now `reached = head_end > 0`, `future_start = max(0, head_end - 1)`
unconditionally. Measured equal over 1116 windows BEFORE substituting, so
no behaviour changed.

**A one-row dataset was genuinely untested** (only a one-row RUN inside a
longer dataset was) and it turns out to project to the exact unregrouped
identity: 480 comparisons across 5 frame counts x 5 window lengths, alone
and beside 5-row and 3-row neighbours; `DatasetRevealSchedule` handles it
in parallel AND serial (168 states). It is visible from frame 0 because
`end >= 1` always -- audited behaviour (F05-012) -- so there is no
"before the singleton is reached" frame to test. Pinned deliberately.

Four tests added; `REGROUPED_CASES` gained four one-row entries so every
existing invariant meets the degenerate projection. That exposed a harness
gap: the lag bound took `max()` over runs with a grid to slide along, empty
for an unsplit one-row dataset -> `default=0`, which TIGHTENS that case to
exact identity. Re-extracted from the edited plan and run: **160 passed**.

## Item 7 — Plan 3's forecast_cluster= prose (`5bbeb50d`)

`:2540` still said OPEN. It shipped: endpoints, in the drawn space, fixed
across animation frames. Recorded with evidence (`plot.py:2010-2024`,
`:5268-5279`, `animation.rst:480-493`, `CHANGELOG.md:122-127`) and listed
as Plan 3's fifth settled decision in the README.

## Items 1-5 — Plan 4 (`12a14133`)

Four defects of the same classes found while fixing the five reported:

1. Ordering: split blocks physically moved after their rewrite, tasks
   renumbered 1..N, all cross-references repointed. Task 8's 0/0b/0c ARE in
   execution order -- left alone with a note rather than renamed into ten
   dangling references.
2. Stale contracts were in ALL FIVE tables, not just market's FRED: weather
   named `fetch_city_months`/`CITIES`, paintings and conversation had
   `vectors` fields for examples that never hold an embedding, morph had no
   `source`. Task 8 Step 0b's worked weather split -- the one the other four
   copy -- has the same defect and is now labelled as being against the
   CURRENT file.
3. Wrapper returns: Task 5's rewrite did `fig, ani = hyp.plot(...)` then
   `ani.on_frame(...)` -- an AttributeError, the exact regression that
   task's own v3 banner warns about. All five now bind `anim`; each split
   step carries its file's real builder tail ending in `return anim`.
4. `git checkout -- <notebook>` replaced by `execute_tutorial.py --out-dir`
   plus an assertion that `git status --porcelain` prints nothing.
5. Budgets MEASURED, two were unattainable:

   | file | rewrite | split | overhead | was | now |
   |-|-|-|-|-|-|
   | market | 110 | 126 | +16 | 130 | 130 |
   | weather | 56 | 73 | +17 | 77 | 75 |
   | paintings | 112 | 135 | +23 | 133 | **140** |
   | conversation | 88 | 106 | +18 | 105 | **110** |
   | morph | 26 | 43 | +17 | 45 | 45 |

   Method: transcribe each rewrite block to a file, apply the split, measure
   with the plan's own `measure_native_ratio.py` -- validated first by
   reproducing the plan's BEFORE figures exactly (market 191, weather 195,
   morph 26). Weather's "+15 MEASURED" was measured on the file Task 3
   DELETES; an overhead cannot be carried across the rewrite that removes it.

Flagged, not silently fixed: Tasks 2-6 verify with `anim.n_frames` and Task
5 drives `draw_frame(i)`, none of which exist until Task 8 Step 0 (ordering
notes now say Steps 0-2). And Contract 4 says the loader is the only code
that may touch the network, but for conversation and paintings the model
download happens inside `hyp.plot`, i.e. inside `construct_artifact` --
written up as an explicit maintainer call.

## Item 9 — the Kaleido failure (`82c5a72c`)

Passes here (Chrome present). The reported failure was a browser dying
during startup, which exits non-zero and hit the hard `returncode == 0`
assertion -- the hang path was handled, the crash path was not.

Render script now exits 3 for exactly the three browser-lifecycle
exceptions kaleido exports plus plotly's no-Chrome `RuntimeError` (needed
because `plotly/io/_kaleido.py:411` catches `ChromeNotFoundError` and
re-raises it untyped, so it cannot be caught by type through
`fig.write_image`; matched on plotly's own constant). Test skips on that
code alone; every other non-zero exit still fails hard.

Both halves pinned with REAL failures: `/bin/echo` is a real non-browser
binary and kaleido genuinely raises `BrowserFailedError: the browser seemed
to close immediately after starting`; an unwritable output path is a plain
OSError that must NOT be laundered into a skip.

## Open / not started

- **Item 8** — implement the regrouped plan. Gated on the maintainer.
- **Item 10** — begin Plan 4. Explicitly not started.
- **Nothing pushed.** `dev-1.0` is now 103 commits ahead of origin; CI has
  not seen any of this work since 2026-07-24.
- macOS-CI `ConvergenceWarning` from `tests/predict/test_gp.py` (run
  30097502289) still not reproduced locally.

## Full-suite failure found and fixed: LSL "any stream" (`98897ece`)

Post-change full suite: **1 failed, 2982 passed, 13 skipped** (11m42s). The
failure was NOT from this session's changes, and it is not a flake:

`test_lsl_stream_resolves_any_stream` -> `HypertoolsIOError: nothing
received for ~10.0s`. `lsl_stream()` with neither `name=` nor `type=` means
"any stream", and LSL resolution is machine- and subnet-wide. This machine
has a STARSTIM-8 attached publishing four outlets (Accelerometer / Markers
/ Quality / EEG; 3 and 8 channels). The accelerometer resolved FIRST and is
idle. Even had it delivered, `len(sample) == N_CHANNELS` would fail on 3
channels. The test assumed its own outlet was the only one on the machine
-- its fixture docstring anticipated collisions between hypertools tests
but not with real hardware, so the assumption was never stated or checked.

Fixed by covering both machine states rather than skipping either:

- sole outlet -> the original sample assertions, unchanged
- foreign outlets present -> `lsl_stream()` must WARN (naming the first
  match and telling the user to pass `name=`) rather than silently binding;
  whether that stream delivers is a fact about someone else's hardware and
  is not asserted

Plus `test_lsl_stream_by_NAME_is_unaffected_by_foreign_outlets` -- the
escape the warning recommends, which must work on exactly the machines that
emit the warning, and must not warn.

**Useful for future debugging:** liblsl scopes resolution by SessionID, so

    printf '[lab]\nSessionID = hypertools-tests\n' > /tmp/lsl.cfg
    LSLAPICFG=/tmp/lsl.cfg .venv/bin/python -m pytest tests/test_lsl_streaming.py

makes `pylsl.resolve_streams()` return `[]` on a machine with hardware
attached. That is how the sole-outlet branch (the one CI takes) was
actually run here rather than assumed. Recorded in the module docstring.
