# Release review implementation — 2026-09-08

## Authorization and recovery

The user approved fixes 1–4 and the proposed verification plan. Before edits,
all then-current work was committed as `0a2cc0d230d31d30458e991b4576e45c2793b3ac`
on `fix/1.1-release-review`. This is the original rollback checkpoint.
The user subsequently authorized committing and pushing the verified PR
changes. Merging, tagging and publishing remain separate release operations.

The user clarified that observation times must affect fitting, including
irregular and shuffled samples and different times per dataset. The approved
default interval is the median positive gap between sorted timestamps, with
an explicit override. The user approved documented interpolation for models
that require a regular grid.

## Implemented and locally verified

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

Code and API/class documentation plus regression tests are saved locally.
Tutorial Markdown for manipulation and animated forecasting has been updated.
Before the backtesting edits, the entire working tree was additionally saved
as `2e9669df` (the user approved the proposed backtesting policy and push plan).
Both this checkpoint and the original `0a2cc0d2` remain available.

## Newly discovered backtesting decision

Backtesting previously scored forecast rows against held-out rows by position,
even when their timestamps differed. Sorting before splitting is implemented.
The user approved evaluating GP at held-out times and linearly interpolating
discrete-model forecasts to those times. This is now implemented. The shared
model factory preserves ordinary prediction's constructor/spec rules while
fitting once and generating only the forecast required for scoring.

Models and intervals use training rows only. Discrete models generate a grid
covering all held-out times, selecting exact matches or interpolating between
grid forecasts. The last observed training value anchors times before the
first full forecast step; missing endpoints stay missing. Held-out values
never enter fitting or interpolation. Full-index validation rejects duplicate
timestamps across the split. Categorical/repeated numeric IDs remain
positional even if one split happens to have unique IDs.

Concrete reproduction: a one-column frame with index and values
`[0, 1, 2, 4, 7, 8, 12, 20]`, passed to
`hyp.predict(..., model='GaussianProcess', holdout=2, return_forecasts=True)`,
returns forecast times `[9, 10]` and truth times `[12, 20]` yet scores the
rows against each other before the fix. The corrected path evaluates GP at
`[12, 20]` and the API guide now executes this example as a doctest.

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
- The first clean HTML/gallery (`hypertools-fixes-20260908-html.log`)
  completed successfully, executing all 51 gallery examples.
- Backtest alignment regression run (`/tmp/hypertools-backtest-alignment-regression.log`):
  269 passed. Covers native times, shuffled rows, changed held-out values,
  per-model/per-dataset intervals, exact-grid selections, fractional-step
  interpolation, missing endpoints, positional IDs, and existing forecasting,
  scoring and imputation behavior. Later sorting cleanup and portable test
  path handling are included in the final full validation below.
- The updated stock tutorial executed successfully
  (`/tmp/hypertools-backtest-tutorial.log`). Its new calendar-time example
  checks that every forecast shares the held-out index. Executed outputs were
  copied back after source equality checks; temporary checkout prefixes were
  removed from text outputs. Its existing trading-day comparison deliberately
  remains positional, with updated prose explaining that modeling choice.

The 19 full-suite skips include release-only gates, CI environment assertions,
two platform/font-specific checks and six opt-in native example executions.
The native example executions were separately enabled in the 344-pass run.
A fresh full run and updated documentation validation are being prepared on
the final tree; the first full run is useful evidence, not a clean final gate.

Full-suite source snapshot: `/tmp/hypertools-verify-20260908-h5gdx5sw`.
Docs/tutorial source copy: `/tmp/hypertools-docs-verify-20260908`.
The full-suite snapshot predates the later animation cost accounting and
no-op/docstring corrections, three added tests and the documentation
cross-reference/underline fixes. These later edits have focused checks above;
the docs copy includes the documentation cross-reference/underline fixes.
A source hash
manifest for the initial snapshot is in
`/tmp/hypertools-current-verify-manifest.json`.

## Final verification round (September 8–9)

- `/tmp/hypertools-final-suite.log`: **5883 passed, 19 skipped, 2 deselected**
  in 19:02, including the packaging tests. This clean run includes the
  backtesting fixes but predates two final animation regression cases below.
- `/tmp/hypertools-final-doctest.log`: **323 doctests passed**, zero failures.
- `/tmp/hypertools-final-examples.log`: **344 native example checks passed**.
- Final animation inspection found that repeated references to one model spec
  could share a cached forecast across comparison entries. Each entry now has
  its own cache lifetime, while columns within a dataset still share the joint
  fit. Real fitted-call counting verifies independent fits. Model classes are
  distinguished from fitted instances, and dictionary-wrapped fitted models
  bind to the appropriate dataset before animated forecasting.
- `/tmp/hypertools-final-animation-followup.log`: **204 passed**;
  `/tmp/hypertools-final-animation-specs.log`: **36 passed**, including the
  final class/repeated-spec and multi-dataset fitted-instance cases.
- `/tmp/hypertools-pandas-floor-tests.log`: **101 passed using pandas 2.2.2**,
  including backtesting, timestamped animation and manipulation. The minimum
  pandas version was installed only in a temporary target directory; the main
  development environment remains unchanged.
- The final full run passed: **5885 passed, 19 skipped, 2 deselected,
  213 warnings in 18:30**. It checked the exact source snapshot at
  `/tmp/hypertools-push-verify-srhi4kme`, with hash manifest
  `/tmp/hypertools-push-verify-manifest.json`, log
  `/tmp/hypertools-push-suite.log` and JUnit `/tmp/hypertools-push-suite.xml`.
  Source, tests and documentation hashes match the working tree; only this
  evidence note changed after the snapshot. All 13 packaging tests passed.
  The two opt-in large-data tests (476 MB Drive download and real weights/UMAP)
  were deselected by the normal suite configuration; no pass is claimed for
  those two tests in this round.
- All **25 tutorial notebooks executed successfully**. The last 17 results
  are recorded in `/tmp/hypertools-final-tutorials-results.json` and
  `/tmp/hypertools-final-tutorials.log`; the earlier eight include the updated
  stock tutorial. No committed tutorial contains stored exception outputs or
  `/Users/` paths. The editable installation still points at this checkout.
- `/tmp/hypertools-final-html.log`: clean HTML build succeeded with `-W -E -a`,
  executing **51/51 gallery examples**. The separate standard post-build step
  succeeded (`/tmp/hypertools-final-post-build.log`), adding all 51 notebook
  badges and gallery thumbnail links. The first browser check ran before
  this post-build step and correctly failed the five gallery badge checks;
  the rerun against complete output passed **8/8 browser checks**
  (`/tmp/hypertools-final-browser-complete.log`). Screenshots are in
  `/tmp/hypertools-final-browser-evidence`; the Plotly page was also visually
  inspected. This was an incomplete verification invocation, not a source
  defect. The browser checks validate rendered links; publishing updated
  remote notebooks remains a release operation.
- The docs/tutorial execution snapshots predate only the final animation
  class/cache corrections. Those edge cases have the dedicated real-model
  regressions above and are included in the final full-suite snapshot.
- Final Ruff and whitespace checks passed. All changes are ready for the
  authorized PR push; hosted CI must still validate the pushed commit.

Use `LOKY_MAX_CPU_COUNT=4` for sandbox tests: macOS physical-core detection is
blocked inside the sandbox and otherwise adds a joblib warning to tests that
require warning-free calls. Use `MPLBACKEND=Agg`, `PYTHONDONTWRITEBYTECODE=1`,
and `-p no:cacheprovider`. **Unset `HYPERTOOLS_AUTO_INSTALL` for the full suite**
so the deliberate real auto-install test can run; setting it to `0` is useful
only for tutorial execution. Chrome/network/Jupyter
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
