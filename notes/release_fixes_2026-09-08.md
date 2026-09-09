# Release review implementation — 2026-09-08

## Authorization and recovery

The user approved fixes 1–4 and the proposed verification plan. Before edits,
all then-current work was committed as `0a2cc0d230d31d30458e991b4576e45c2793b3ac`
on `fix/1.1-release-review`. This is the rollback checkpoint. No pushes,
merges, tags, release publication, or external messages have been made.

The user clarified that observation times must affect fitting, including
irregular and shuffled samples and different times per dataset. The approved
default interval is the median positive gap between sorted timestamps, with
an explicit override. The user approved documented interpolation for models
that require a regular grid.

## Implemented, still undergoing release validation

1. Manipulator classes, dispatcher, Pipeline and fitted reuse interpret 1D
   numeric arrays/Series/lists consistently as one column of observations.
2. Forecasts retain and sort observation times with their values. GP uses
   actual elapsed times divided by the model interval. Kalman, ARIMA,
   AutoRegressor, Laplace and Chronos use linear interpolation onto a grid
   anchored at the latest observation, entirely within the observed span.
   Constructors accept `step`; models retain their fitted interval on reuse.
   Series plots fit signal columns jointly, then add time coordinates for
   drawing. Static, animated and column-hierarchy forecasts preserve times.
   Early animation prefixes wait for sufficient interpolated history.
   If preprocessing changes row counts without preserving timestamps,
   timestamped forecasting raises and requests explicitly indexed output.
3. Rowwise ZScore/min-max Normalize work on multiple datasets, including
   unequal feature widths and repeated row labels. Held-out reuse restrictions
   remain enforced.
4. MatrixColormap's exact float interpolation respects `set_gamma`.

Code and API/class documentation plus regression tests are local/uncommitted.
Tutorial Markdown for manipulation and animated forecasting has been updated.
The original rollback checkpoint remains HEAD (`0a2cc0d2`). Before starting
the next round of source edits, checkpoint this current working tree too,
as requested by the user.

## Newly discovered backtesting decision

Backtesting previously scored forecast rows against held-out rows by position,
even when their timestamps differed. Sorting before splitting is implemented.
An async clarification is pending: evaluate GP at held-out times and linearly
interpolate discrete-model forecasts to those times, or require exact grid
alignment. Do not implement either scoring policy before the answer arrives.

Concrete reproduction: a one-column frame with index and values
`[0, 1, 2, 4, 7, 8, 12, 20]`, passed to
`hyp.predict(..., model='GaussianProcess', holdout=2, return_forecasts=True)`,
returns forecast times `[9, 10]` and truth times `[12, 20]` yet scores the
rows against each other. This is a release blocker for time-aware backtesting.

## Verification evidence so far

All logs below are outside the checkout in `/tmp`.

- `hypertools-fixes-20260908-focused.log`: 236 passed (manipulators, Pipeline,
  datatype/reuse cases, rowwise lists and gamma).
- `hypertools-fixes-20260908-confirm.log`: 101 passed (forecast timing,
  minimum history, forecast audit and animated regrouping).
- `hypertools-fixes-20260908-hierarchy.log`: 180 passed after extending timed
  forecast scheduling to hierarchical plots.
- `hypertools-fixes-20260908-edge-confirm.log`: 54 passed (actual-time
  forecasts and reuse at a different cadence, duration/period indexes,
  explicit truth times, invalid steps/times, series column hierarchies and
  documentation structure).
- `hypertools-fixes-20260908-doctest-live.log`: all 316 doctests passed,
  zero warnings. The first sandboxed run had seven failures from blocked
  Chrome/network access and a short heading underline; the underline was
  corrected and the live rerun succeeded.
- `hypertools-fixes-20260908-examples.log`: 344 native example checks passed.
- `hypertools-fixes-20260908-tutorials.log` and
  `hypertools-fixes-20260908-tutorials-forecast.log`: manipulation, Pipeline,
  plot, normalization, animated forecasting, hierarchy and projectile
  tutorials executed successfully. Outputs and media remain in the scratch
  copy, not the user's working tree. Stock forecasting needs re-execution
  after the backtest policy is resolved.
- Ruff checks of source/tests/scripts/docs config and `git diff --check`
  passed.
- Full pytest (`hypertools-fixes-20260908-full.log`) finished in 18:17:
  **5855 passed, 5 failed, 19 skipped, 2 deselected**. All 13 packaging tests
  passed. The failures were:
  - One docstring gate: two new AutoRegressor history-floor methods lacked
    docstrings. Corrected in the working tree.
  - Three existing tests of skipped manipulation: normalization of input
    shape broke the `model=None/False` identity-preserving no-op. Corrected
    by validating the input but returning the original object when skipped.
  - One deliberate auto-install test conflicted with the run's
    `HYPERTOOLS_AUTO_INSTALL=0`. With that variable unset, the real temporary
    virtualenv install test passed. Do not change this test to hide the
    environment mismatch.
- `hypertools-fixes-20260908-full-fixes.log`: all 116 tests passed after
  the docstring/no-op corrections, including all manipulation tests, the
  existing final-wave audit tests and the new round-13 tests.
- `hypertools-fixes-20260908-install-check.log`: the isolated real-install
  test passed with its expected environment.
- `hypertools-fixes-20260908-schedule-confirm.log`: 62 passed after making
  animation timing/counts recognize that multiple drawn columns share one
  joint model fit. Forecast values remain identical.
- Clean HTML/gallery (`hypertools-fixes-20260908-html.log`) is still running.
  Tool process session: `27302`; the last reported example was
  `animate_market_sectors.py` (96%). Collect its result before starting a
  duplicate build. The full-suite process `40413` has finished.

The 19 full-suite skips include release-only gates, CI environment assertions,
two platform/font-specific checks and six opt-in native example executions.
The native example executions were separately enabled in the 344-pass run.
A **fresh full run on the final tree** is still required after resolving the
backtest policy; the first full run is useful evidence, not a clean final gate.

Full-suite source snapshot: `/tmp/hypertools-verify-20260908-h5gdx5sw`.
Docs/tutorial source copy: `/tmp/hypertools-docs-verify-20260908`.
The full-suite snapshot predates the later animation cost accounting and
no-op/docstring corrections, three added tests and the documentation
cross-reference/underline fixes. These later edits have focused checks above;
the docs copy includes the documentation cross-reference/underline fixes.
A source hash
manifest for the initial snapshot is in
`/tmp/hypertools-current-verify-manifest.json`.

Use `LOKY_MAX_CPU_COUNT=4` for sandbox tests: macOS physical-core detection is
blocked inside the sandbox and otherwise adds a joblib warning to tests that
require warning-free calls. Use `MPLBACKEND=Agg`, `HYPERTOOLS_AUTO_INSTALL=0`,
`PYTHONDONTWRITEBYTECODE=1`, and `-p no:cacheprovider`. Chrome/network/Jupyter
verification may need sandbox escalation; report environmental failures
separately from source failures.

## Release operations still outstanding

Follow `RELEASE_CHECKLIST.md` after source validation. PR #286 must be green at
the final pushed head, merged, and the release regenerated from that exact
commit. The v1.1.0 draft/tag and published gallery previously pointed to
`96ac8b7f`, not the reviewed PR changes. Changelog release date, gallery
manifest, wheel/sdist provenance, master/tag gates, public documentation,
PyPI/GitHub release and conda-forge follow-up still need final verification.
Do not interpret local test passes as approval to publish the release.

Earlier audit history: `notes/release_audit_2026-09-07_paused.md` and
`notes/session_2026-09-05_release-1.1-review.md`.
