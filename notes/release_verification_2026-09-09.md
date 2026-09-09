# Release verification follow-up — September 9, 2026

## Authorization and recovery

The user explicitly prohibits publication until manual sign-off. Do not merge
the PR, move the release tag, publish notebooks/assets/docs, or publish to
PyPI/GitHub without that sign-off. Read-only review and verification continue.
The user subsequently approved fixing everything found in this review, adding
the fixes to PR #286, and verifying the tests.

Before this round's edits, the working tree was clean at committed and pushed
`c033e857054d831ead74898ea7e3667f1d0694b1`. That commit is the recovery point.
Earlier backups `0a2cc0d2` and `2e9669df` remain available. No release operation
has been performed in this round.

## Additional finding and correction

**Integer timestamp arithmetic can wrap or lose the intended interval.**
For a DataFrame with unsigned index `[0, 1, 3, 6, 7, 9, 13, 15]`, Kalman,
ARIMA and AutoRegressor rejected valid data with “step exceeds the observed
time span.” Subtracting the last timestamp from earlier unsigned timestamps
wrapped negative elapsed times into enormous positive values. Signed
differences spanning the dtype bounds and forecasts extending beyond the
integer dtype's maximum also overflowed.

The correction computes integer differences/additions using Python integers
before converting elapsed coordinates to floating point. It preserves small
intervals at large epoch offsets and leaves datetime/timedelta arithmetic
unchanged. The interpolation policy and forecast model equations are unchanged.

New real-model regressions compare signed/unsigned and nullable integer clocks,
epochs above the signed 64-bit limit, fitted reuse and held-out backtesting.
Independent boundary assertions check inferred gaps and future timestamps.
Against the previous package snapshot, **18 of the 31 new tests fail**; all
31 pass with the correction. The first attempted before-state invocation used
the root test path, which imported the working package; it was discarded and
rerun with the test copied inside the old isolated snapshot.

## Verification

- `/tmp/hypertools-integer-time-regression.log`: **229 passed**, covering
  all forecasting unit tests, backtest timing, animated timing, and the new
  integer-clock regressions.
- `/tmp/hypertools-integer-time-before.log`: **18 failed, 13 passed** against
  the unmodified `c033e857` package; confirms the new tests catch the defect.
- `/tmp/hypertools-review-bigdata-20260909.log`: **2 passed, 41 deselected**.
  Both tests normally excluded by the `bigdata` marker ran successfully:
  the live 476 MB Google Drive interstitial download and the real weights
  dataset/UMAP story-trajectory workflow. Run against the prior verified
  snapshot; neither workflow uses forecast timestamp arithmetic.
- Full-suite verification of the corrected source: **5916 passed, 19 skipped,
  2 deselected, 297 warnings in 18:16**. The isolated snapshot is recorded by
  `/tmp/hypertools-integer-verify-path`; hashes are in
  `/tmp/hypertools-integer-verify-manifest.json`. Log:
  `/tmp/hypertools-integer-full-suite.log`; JUnit:
  `/tmp/hypertools-integer-full-suite.xml`. Final source, tests and documentation
  hashes match the working tree. This new evidence note was added after the
  snapshot. The two deselected large-data tests passed separately above;
  the 19 skips retain the prior documented release/CI/platform/smoke reasons.
- Minimum pandas 2.2.2 verification: **100 passed**; log:
  `/tmp/hypertools-integer-pandas-floor.log`.
- Updated documentation doctests: **323 passed**, zero failures, build
  succeeded with warnings treated as errors; log:
  `/tmp/hypertools-integer-doctest.log`.
- Explicit release file-content gates: **15 passed, 1 deselected** with
  `HYPERTOOLS_REQUIRE_RELEASE=1`; log:
  `/tmp/hypertools-release-file-gates-20260909.log`. The excluded test ties the
  published gallery to the final release commit and cannot pass against this
  unpublished PR; it remains a required post-approval release operation.
- Ruff and `git diff --check` pass.

The correction and this record are being committed separately and added to
PR #286 as authorized. The new commit needs its own hosted CI result; the
baseline run below must not be treated as CI evidence for the new commit.

## Hosted CI and release provenance

CI run [34308363422](https://github.com/ContextLab/hypertools/actions/runs/34308363422)
tests `c033e857`, not the additional integer-clock fix. All ordinary test runs
on Windows/macOS/Ubuntu Python 3.10–3.13 passed. The extra pandas 3 acceptance
run passed; the Ubuntu 3.12 coverage run is still pending completion. Wheel/
sdist smoke, clean docs, dataset and strict live-source gates passed. The
release-only gate is intentionally skipped on PR branches.

Read-only remote checks confirm that master and the peeled `v1.1.0` tag remain
at `96ac8b7f43c132f6f455ad1be3ffc98e84adead5`. The gallery manifest still lists
51 notebooks with that same older `source_commit`. The GitHub v1.1.0 release
is still a draft, with wheel/sdist assets uploaded September 5. These are
draft provenance, not artifacts of the reviewed PR.

After the fixes and their final CI are green, the remaining operations are in
`RELEASE_CHECKLIST.md`: manual sign-off, merge, finalize the release date,
regenerate the gallery and artifacts from the exact final commit, verify
master/tag release gates, then approved publication and distribution smoke
checks. Do not publish or move references merely because verification passes.
