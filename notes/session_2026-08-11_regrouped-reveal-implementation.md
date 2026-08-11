# Session 2026-08-11 — implementing the regrouped-reveal plan

Executing `docs/superpowers/plans/2026-08-03-hypertools-1.1-regrouped-reveal-and-forecasts.md`
(v2) task by task, inline (no subagents — session constraint). Branch
`dev-1.0`. Every task ran its own TDD cycle: write the plan's tests, confirm
RED for the stated reason, implement, confirm GREEN, run the regression sweep,
commit.

## Where this stands

| Task | State | Commit |
|-|-|-|
| 1 `TraceOwnership` | done, 11 tests | `c50b5037` |
| 2 `RunWindow` + `dataset_window_bounds` | done, 160 tests in module | `aebe4689` |
| (fix) machine-speed-dependent warning | done | `0a719e24` |
| 3 matplotlib updaters | done | `a701b32c` |
| 4 plotly parity | done (same commit — shares plot.py) | `a701b32c` |
| 5 `DatasetRevealSchedule` | done, 202 in module | `4dd5ec16` |
| 6 `ForecastSchedule` row-tuple keys + `for_regrouped` | done, 207 in module | `4e77d20c` |
| 7 forecast over a regrouped animation (matplotlib) | NOT STARTED | |
| 8 plotly parity, docs, CHANGELOG | NOT STARTED | |

Checkpoint the maintainer asked for — all of `tests/plot` once both backends
consume `RunWindow` — ran GREEN: **553 passed**, zero warnings.

## The re-measurement instruction was discharged

The standing instruction was that Task 3 Step 2's
`test_an_UNREGROUPED_animation_is_unchanged_row_for_row` hard-codes point
counts measured at `a062f768`, and they must be re-measured BEFORE any
implementation change. Done: the test was run against unmodified code and
**passed as written** — `[1, 83, 165, 247, 329, 411, 493, 575, 657, 739, 821,
903]` is correct on this machine's matplotlib/scipy. No correction needed.
The regrouped tests failed at that point exactly as the plan predicted (later
runs already at 247 points on frame 3), and precog showed the live
`data[-1:]` defect as `[1, 1, 1, 903, 903, 903]`.

## Defects found in the plan's own listings (all fixed in the code)

1. **`seg_bridge` length.** Task 3 Step 3's `run_bridged` comprehension indexes
   `seg_bridge[i]` per RUN, but `segment_by_run` documents its length as
   `len(segments) - 1` (one flag per GAP). Python subscripts before it reaches
   the guarding `and`, so the last run raised `IndexError`. Correct form is
   `i < len(seg_bridge) and bool(seg_bridge[i])`.
2. **plotly frame loop location.** Task 4 says to add `ownership` to
   `plotly_draw`, but the frame loop lives in `_add_animation` — a separate
   module-level function. `ownership` had to be threaded one level further or
   every frame raised `NameError`.
3. **Missing docstrings.** `TraceOwnership.n_runs`/`.n_datasets` have none in
   the plan's listing; `tests/test_docstrings.py` (GH #276 gate) rejects that.

## Defects in the plan's TEST listings (fixed in the tests)

4. `_animate(data, **kwargs)` takes no `fmt`, yet the marker-only control
   calls `_animate([_walk()], 'o', hue=HUE)` → `TypeError`.
5. `_plotly_counts` used `len(d.x or ())`; plotly stores frame coordinates as
   numpy arrays, so `array or ()` raises "truth value ... is ambiguous".
6. The precog test selected artists by `_hyp_row_window`, which `_aa_window`
   stamps on HEADS too. Frame-0 heads are `[1, 0, 0]`, so "no artist has
   exactly one point" failed on correct behaviour. Trail artists now come from
   the updater's own return value `(head_lines, trail_lines)`, and the test
   additionally pins the R5 promise (an unreached run's precog covers its
   WHOLE run).
7. My own first draft read forecast artists with `get_xdata()`; a `Line3D`
   keeps vertices in `_verts3d`, so that reports a stale 2-D projection and
   reads as empty at EVERY frame. Use `get_data_3d()`. This briefly looked
   like a forecast regression and was a measurement error.

## Two real product findings, beyond the plan

**A machine-speed-dependent warning (fixed, `0a719e24`).** The suite is held
to zero warnings, but `tests/plot/test_forecast_schedule.py` built schedules
at the DEFAULT `slow_warning_seconds=10.0`, and that notice fires when
`one_fit_seconds * remaining_fits > 10`. The 900-frame memoization test
projects ~9.7 s on this laptop — a machine a hair slower emits a warning and
fails the gate, with nothing reproducible locally. Under `PYTHONDEVMODE=1`
(~25x slower fits) it warned at 21.3 s projected. All nine constructions in
that module now pass `slow_warning_seconds=None`; the warning itself stays
fully pinned by `test_forecast_schedule_warning.py` (`0.0` always warns,
`None` never). Dev-mode sweep of all ten forecast-animating modules
afterwards: 249 passed, zero warnings.

**A frame-count asymmetry (fixed, `4e77d20c`).** Plan Task 6 Step 6 asked for
this to be measured. `plot()` built the forecast schedule with
`max(2, round(frame_rate * duration))` while both backends pace with
`max(1, ...)`. The floor of 2 belongs to the interpolation GRID (PCHIP needs
two samples), not to the clock. They differ at exactly one setting,
`round(frame_rate * duration) == 1`: measured on an 8-row dataset at
`frame_rate=1, duration=1`, the renderer drew its single frame holding all 8
raw rows while the schedule reported 1 row revealed — so that animation showed
the whole trajectory and NO forecast. Frame counts 2 and 12 already agreed.
Regression test is mutation-verified (restoring `max(2, ...)` turns it red).

## Repo-wide lint debt (flagged, NOT fixed — maintainer's call)

`ruff check .` reports **417 pre-existing errors** repo-wide (190 F405, 103
F401, 58 E402, 43 E741, ...). There is no lint job in CI (`.github/workflows/`
holds only `test.yml`), so nothing gates them. I fixed only the ones in files
this work touched and kept new files clean. Cleaning all 417 is a separate
decision.

## Resuming at Task 7

Task 7 (plan lines 2054-2378) draws the forecast over a regrouped animation
in matplotlib — replacing the refusal at `plot.py:5051-5077` (line numbers
have drifted; grep for the refusal warning text). Decision R3 governs colour:
a live forecast takes `head_run` at the CURRENT frame; a retained
`forecast_trail=` member of age k takes `head_run(dataset, past[k-1])` — the
head run at the frame it was FIT — so a saved animation matches a played one.
Task 8 is plotly parity + docs + CHANGELOG.

Nothing pushed. `dev-1.0` is ~115 commits ahead of `origin/dev-1.0`.
