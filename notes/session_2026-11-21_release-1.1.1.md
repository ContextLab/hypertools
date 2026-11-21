# Session 2026-11-21: issues #290, #291, #292 and the 1.1.1 patch release

Request (Jeremy): explore, verify and fix open issues 290, 291, 292; red-team
with fresh subagents and the Codex CLI; update docs; all tests passing; open a
pull request that updates to v1.1.1; merge once CI is green.

Branch: `fix/1.1.1-issues-290-292` (from master @ 428b998a).

## What the issues were (all three reproduced before fixing)

Reproduction script used: the three snippets from the issue bodies, run in the
repo `.venv` (matplotlib 3.10.8) and in a scratch env with matplotlib 3.11.2.

- **#290** backend warning on every call. Reproduced in both envs.
  `hypertools/plot/backend.py` `plot_wrapper` warned whenever
  `BACKEND_WARNING` was set, with no gate.
- **#291** `plt.tight_layout()` on a caller's figure. Reproduced in both envs
  (second axes moved from `[0.5, 0.295, 0.2, 0.4]`).
- **#292** `findfont: Failed to find font weight normal for Hiragino Kaku
  Gothic ProN, now using 300.` Reproduced ONLY under matplotlib 3.11.2.
  Cause: matplotlib 3.11 registers every face of a `.ttc` collection, so the
  family 'Hiragino Kaku Gothic ProN' now exists (W3=300 and W6=600 faces
  only), and 3.11's `_findfont_cached` logs whenever the best face's
  normalized weight differs from the one asked for. matplotlib 3.10.8 has
  neither the family nor the log line (`grep -c "Failed to find font weight"`
  on its `font_manager.py` gives 0).

## Fixes

- #290: `_needs_interactive_display(plot_kwargs)` (animate/interactive, shown,
  not saved, matplotlib-rendered) gates both the backend switch and the
  warning; `_issue_backend_warning()` issues each message once per session
  (`_BACKEND_WARNINGS_ISSUED`). Inside a `set_interactive_backend` context the
  call's arguments are inspected only when a warning is pending.
- #291: `fig.tight_layout()` (never `plt.`) and only for a figure hypertools
  made: no `ax=`, or a `hyp.subplots` grid (marked with the private figure
  attribute `_hypertools_figure`, also set on
  every figure a `hyp.plot` call without `ax=` creates). `panels=` builds its grid with `subplots()`,
  so it is unchanged. The right-legend fit (which widens the figure and
  switches off its layout engine) is skipped for every caller-built figure.
- #292: `_families_with_a_regular_face()` scores each registered face of a
  family with the font manager's own `score_*` methods and keeps the family
  only if the winner's weight is 400. Applied to the families hypertools adds
  on its own (`_NICE_SANS_FAMILIES`, `_PREFERRED_FAMILIES`); `first=` (explicit
  `font=`) and `extra=` (only covering font) are always kept.

## Decisions I made that Jeremy may want to revisit

- `hyp.subplots` grids keep the automatic `tight_layout` after each panel
  (the issue suggested skipping layout for every `ax=`). Reason: the
  `hyp.subplots` docstring example has no explicit layout call, so skipping
  there would change documented figures in a patch release.
- The backend warning is both gated AND once per session (the issue said
  "and/or").
- Bold (non-regular) text can still log under matplotlib 3.11 for fallback
  families that ship one weight (e.g. 'Arial Unicode MS'). Not removed:
  dropping them costs glyph coverage. Recorded under "Documented limitations"
  in the changelog.
- Two older tests assumed 1.1.0 was the top/only version and were updated
  (not weakened): `tests/test_changelog_1_1.py::test_changelog_has_a_1_1_0_section`
  and `tests/test_animation_guide_docs.py::test_animation_guide_version_claims_match_the_package_version`.

## Red team, round 1 (2026-11-21)

Codex CLI (read-only) and two fresh subagents; no blockers. Acted on:

- Legends on caller figures (subagent A, MAJOR; Codex, MINOR). Measured
  (`scratchpad/leg2.py`, matplotlib 3.11.2, 2-D and 3-D, 1 and 2 panels):
  with the legend fit still widening a caller's figure, the caller's own
  `fig.tight_layout()` left the last legend clipped in every case; with the
  figure left alone, `fig.tight_layout()` gave no clipping and no overlap in
  every case, and `layout='constrained'` likewise. So caller figures are no
  longer fitted at all. COST (Jeremy may want to revisit): a single caller
  axes with `legend=` and no layout now has its legend run off the canvas
  until the caller lays the figure out; 1.1.0 widened the figure unasked.
- `_issue_backend_warning` recorded the message before `warnings.warn`, so
  under an "error" filter the second call passed silently. Now recorded after.
- Axes in a SubFigure plus `legend=` crashed (also on master); fixed by the
  same skip, test added.
- Array-valued `animate=` inside a `set_interactive_backend` context raised
  "truth value ambiguous" instead of plot's own error; `_truthy` added.
- Changelog limitations paragraph: three cases that still log under 3.11
  (bold text; an explicit `font=` with no regular face; text whose only
  covering font has none, e.g. 'Hiragino Sans GB').
- Test fixes: the "whatever is current" test now makes another figure
  current before the call; version parser tolerates pre-release versions.

Not changed, by decision: the font filter is not gated on the matplotlib
version (same families under 3.10 and 3.11); an ignored first warning counts
as issued; a title on a tiny caller figure (3x2 in) can touch the top edge
because the figure is no longer tightened.

## Red team, round 2 (2026-11-21)

Codex CLI again and a third fresh subagent, on the revised design. No
blockers. Acted on:

- A second call into hypertools' own figure (`hyp.plot(more,
  ax=fig.axes[0], legend=...)`) was treated as a caller figure and lost its
  legend fit (subagent C, MAJOR). hypertools now marks every figure it
  creates, not only `hyp.subplots` grids; test added.
- Argument inspection for the warning ran before the `try/finally` that
  restores rcParams, so a rejected `backend=` left `pdf.fonttype` at 42
  (Codex). Moved inside the `try`; test covers in and out of the context.
- A plotly Figure passed as `ax=` with `backend='auto'` still got the
  matplotlib warning (Codex). Any non-matplotlib `ax=` now counts as plotly.
- Wording: the colorbar and `size=` exceptions are stated in the changelog
  and the `ax` docstring; the checklist preface no longer says
  `publish_prebuilt_gallery.py` takes `--ref` (it does not).
- `test_animation_guide_docs.py` accepts an `(unreleased)` heading for a
  claimed version, so it can pass during development of a feature release.

Reviewer C confirmed: all three issue scripts fixed on both matplotlib
versions; own-figure, `panels=` and `hyp.subplots` PNGs pixel-identical to
master; 6 gifs byte-identical; 8 plotly figure JSONs identical.

Flagged, kept: the fixture `light_only_family` registers a real font file
under a made-up family with stated weights (300/600). Reviewer C called it a
test double. It is the only way to exercise the filter on Linux/Windows
runners, which have no installed family of that shape; on macOS with
matplotlib >= 3.11 the real Hiragino family is exercised by the other tests.

## Dependency drift found by the fresh matplotlib 3.11 environment

`master` CI last ran on 2026-10-10. A fresh install on 2026-11-21 pulled
pytest 9.2.0, pylsl 1.18.7, numpy 2.5.5, pandas 3.1.1, plotly 7.1.0, and two
test files failed there with or without this branch's changes:

- `tests/test_figure_review_gaps.py::test_plotly_subplots_grid_starts_without_gutters`:
  pytest 9.2 refuses a nested list of tuples in `pytest.approx`. Compared as
  numpy arrays now; same tolerance.
- `tests/test_lsl_streaming.py` (1 failed, 9 errors): pylsl 1.18.7 bundles
  liblsl 1.18.0.b7, whose `lsl_destroy_outlet` takes a constant 2 s for an
  outlet that answered a resolve query and 7 s for one an inlet connected to
  (0.03 s on pylsl 1.18.2). The fixture's thread holds the only reference
  to the outlet, so its exit includes that destroy, and the fixture joined
  with a 5 s budget. The budget is now 30 s (`_OUTLET_TEARDOWN_WAIT`); the
  assertions are unchanged. Measured on macOS arm64 only. Users on pylsl
  1.18.7 see the same delay when a `SyntheticOutlet` is dropped; that is
  liblsl's and may merit an upstream report (not filed).

## Not done in the pull request (release steps, RELEASE_CHECKLIST.md)

The README images stay pinned to `v1.1.0` and `docs/prebuilt_gallery.json`
stays at 1.1.0 until the release is cut: the `v1.1.1` tag does not exist yet,
so flipping the README in the pull request would break its images. After the
merge, `release-gate` on `master` fails until checklist step 2 onward is done
(README flip, gallery publish from the release commit, tag, PyPI). The PyPI
upload is irreversible and needs Jeremy's go.

## Verification log

See the end of this file; updated as checks finish.
