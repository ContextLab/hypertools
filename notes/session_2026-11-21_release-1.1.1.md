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

### Final local pass, head 59be174c (2026-11-21, 15:55-16:19 EST)

Both suites and the docs build ran at the same time on one machine.

- `.venv` (Python 3.12, matplotlib 3.10.8, pytest 9.0.2): 6845 passed,
  23 skipped, 3 deselected, 1 failed. The failure was
  `tests/test_lsl_streaming.py::test_lsl_stream_resolves_any_stream`
  ("nothing received for ~10.0s"): the test resolves ANY stream on the
  network and the other suite's LSL outlets were live at the same time.
  Rerun alone straight after: `tests/test_lsl_streaming.py` 27 passed,
  1 skipped. The earlier pass of the same suite (head 0e4a924c) was
  6842 passed, 0 failed.
- Fresh env (Python 3.12, matplotlib 3.11.2, pytest 9.2.0, pylsl 1.18.7):
  6822 passed, 33 skipped, 3 deselected, 0 failed.
- `ruff check .`: All checks passed.
- Sphinx `-b html -W`: full build with all 51 gallery examples executed at
  head 0e4a924c, "build succeeded"; incremental rebuild of the changed API
  pages at 59be174c, "build succeeded".

## PR #293, first CI run (head d41e4fe1): 16 green, 2 red

- `test (ubuntu-latest, 3.10)`: 8 failed, 6812 passed. All eight are live
  Yahoo Finance tests on that one runner: "Yahoo Finance returned a non-JSON
  response (HTTP 429) for 'AAPL': JSONDecodeError". The other eleven matrix
  jobs loaded the same tickers and passed. Fixed in the library: a 429 is now
  reported as a rate limit ("Yahoo Finance is rate-limiting requests from
  this network address (HTTP 429, too many requests)"), and
  `tests/_netskip.py` reads the phrase "too many requests" as transient, so
  the live tests skip on it (and still fail under
  `HYPERTOOLS_REQUIRE_LIVE_SOURCES=1`). New test serves a real 429 from a
  local HTTP server.
- `docs/readthedocs.org:hypertools` (PR preview build 36071248): ran 1850 s
  and was ended. `python docs/fetch_prebuilt_gallery.py` found no gallery
  recorded for 1.1.1 (`docs/prebuilt_gallery.json` records 1.1.0), so sphinx
  executed all 51 examples and hit Read the Docs' time limit. This is by the
  repository's design (an existing test,
  `test_fetch_ignores_a_record_that_is_not_a_full_commit_id_for_this_version`,
  pins that a record for another release is ignored) and it stays red for
  any version bump until the gallery for the new version is published from
  the release commit (checklist step 2). Not fixed in the PR; asked Jeremy.

## After the first CI run (2026-11-21 evening)

- Jeremy asked for the 1.1.1 gallery to be published before merging. Built in
  release form from 514a12cb (51 examples, `-W`, succeeded), published to
  `docs-gallery-v1.1.1` (c90dd87e), recorded in `docs/prebuilt_gallery.json`
  (commit 9dec09f0). Final CI on 9dec09f0: 12 test jobs, docs-clean,
  wheel-smoke, dataset-gate, live-source-gate, Read the Docs all green.
- PR #293 merged as 84342161; #290, #291, #292 closed. Not tagged, not on
  PyPI.

## Follow-up: fit the legend/title on caller axes automatically

Jeremy (2026-11-21, after the merge): "All of those grid/figure/legend layout
problems need to be fixed. That must happen automatically for the user when
the 'fix' is clear, rather than requiring them to make an explicit call to
fix strangeness or broken components that this toolbox created."

Branch `fix/1.1.1-caller-figure-fit` (from master 84342161).

Measured on master first (`scratchpad/probe_fit.py`, 18 cases, 2-D and 3-D):
every caller figure without a layout call had the legend off the canvas
(77-156 px) or over the next panel; a 3-D legend stayed clipped even after
the caller's own `fig.tight_layout()` (single axes 25 px, 2x2 grid 22 px).

Fix: `_CallerSlotLocator` (hypertools/plot/plot.py), a matplotlib axes
locator installed by `hyp.plot` on a caller-supplied axes that carries a
legend or title. At draw time it narrows that axes from the right (bisection,
floor 30% of the slot) until the legend ends before the canvas edge or the
next panel's tick labels, and lowers it until the title is on the canvas.
The slot (`get_position(original=True)`), other axes, figure size and layout
engine are untouched. Details:

- answers `get_subplotspec()` so `tight_layout` keeps including the axes
  (a locator without it gets "not compatible with tight_layout");
- defers to a constrained-layout engine (`not engine.adjust_compatible`);
- decides nothing lasting during measuring passes (renderer with draw
  methods switched off);
- no room beside the axes even at the floor: legend drawn inside the axes,
  framed, if it is at most 90% of the axes; larger than the panel: left in
  place with one `UserWarning`;
- skipped when a colorbar of the same axes sits to its right (that pair is
  already inside the axes' slot);
- installed only when there is a legend or title: with a locator,
  `ax.get_position()` reports the last drawn position (matplotlib does not
  apply the aspect on the spot for located axes).

After the fix all 18 probe cases report OK; tests added in
`tests/test_caller_figure_hygiene.py` (65 tests, pass on matplotlib 3.10.8
and 3.11.2).

### Red team of the fit (Codex + one fresh subagent), round 1

No blockers; reworked the locator for all of these (each now has a test):
twins undid the fit (twins get a follower locator); a memoized fit went
stale after `savefig(bbox_inches='tight')`, legend/title edits, neighbor
moves (memo removed: the fit is recomputed on every draw); feeding the
fitted box back into `tight_layout`/engines made panels shrink draw after
draw (while matplotlib's layout code is on the call stack the locator now
answers with the unfitted box, so slots are identical to a figure without
it, draw for draw); "inside" was sticky (now decided afresh on each draw,
legend state saved on the legend and restored when room appears); own/caller
colorbars skipped the fit (now ordinary neighbors); neighbor's axis label
ignored and invisible tick labels counted (only visible, non-empty texts
count); title measured before matplotlib's own repositioning; narrowing
pushed a wide title off the left; tall legend not reported (warning).
Not fixed: an inset axes placed by hand where the legend goes (stated in
the changelog). matplotlib's own tight engine keeps moving 3-D rows for
6+ draws with or without hypertools.

### Red team of the fit, round 2 (Codex + a second fresh subagent)

Round-1 items confirmed fixed by both; hypertools' own figures
pixel-identical to master. New findings, all fixed with a test each:
data limits drifting on `set_aspect('equal', adjustable='datalim')` (view
limits restored around the search); tall legends over the row below not
reported (warning now covers off-canvas top/bottom and overlap with other
panels); redraw cost 10x on a 4x4 grid (one interpolation instead of a
bisection for ordinary axes, title height settled only for the top check:
now 163/230 ms vs 60/69 ms fixed, 2-D/3-D); `tight_layout` after an
inside legend measured the inside state (layout passes put it back
outside); twin made after the plot, and a legend on the twin (followers
installed at draw time; twins' legends counted); caller edits to an
inside legend overwritten (a restyled legend is left to the caller);
layout-pass detection by file name (now by module, plus any
`LayoutEngine.execute`); a never-drawn neighbor axis label read at its
initial spot, so the first draw differed (the axis' own `get_tightbbox`).
Left as is: legend over its own twin's tick labels (same on master);
default 2x3 grids narrow panels to about 40% (legend takes the rest).

### Round 3 (Codex only, confirmation)

All round-2 items confirmed fixed; hypertools' own figures pixel-identical
to master. Four narrow P2 edge cases, fixed with a test each: the legend's
vertical-room check ran before the title fit (now on the final box); a
title on a twin was not fitted (titles of the whole twin family count); a
twin kept following a leader that had been removed from the figure (the
follower takes itself off); recoloring the frame of an inside legend was
not recognized as the caller's edit (frame colors and line width are part
of the compared state). 100 tests in tests/test_caller_figure_hygiene.py
pass on matplotlib 3.10.8 and 3.11.2.

### Final local checks for the fit (2026-11-21, 20:06-20:45 EST)

- Full suite at 49165c09 (.venv, matplotlib 3.10.8): 6904 passed, 24 skipped,
  1 failed: `tests/test_docstrings.py::test_no_undocumented_public_definitions`
  (7 helper functions of the new locators had no docstring). Docstrings
  added; that test and the 24 test files that draw into caller axes then
  pass on the final code (1088 passed, 6 skipped).
- matplotlib 3.11.2 env: docstring gate + caller-figure file, 103 passed.
- Sphinx `-b html -W -E -a` with every gallery example re-executed, at
  49165c09: build succeeded.
- `ruff check`: all checks passed.

## Gallery rebuild and before/after review (2026-11-21, late evening)

Request: rebuild the gallery, review before/after for ALL figures (own review
plus an independent one, ideally Codex), and build a draft release. Working
unattended; the user is on a phone.

- PR #294 (automatic legend/title fit) merged as e3abb840 with all CI green.
- Built the gallery in release form from e3abb840 (51 of 51 examples
  executed, sphinx exit 0, 32 min). Compared it with two baselines using
  a scratch script (pixel statistics on every PNG, every GIF frame, every
  MP4 frame, plotly payload hashes): the published 1.1.0 gallery
  (`docs-gallery-v1.1.0`, 519b4af2) and the pre-fit 1.1.1 gallery (built
  from 514a12cb).
- Result: 89 figures + 51 thumbnails. 87 of 89 byte-identical to the
  pre-fit build; 85 of 89 to 1.1.0. The differences:
  - `plot_colorbar_002.png`: k-means unseeded, cluster colors permuted on
    every build (same points). Not from the code change.
  - `plot_digits_003.png`: UMAP unseeded, new layout every build.
  - `animate_market_sectors_001.gif/.mp4` (vs 1.1.0 only): live Yahoo data,
    one more month (ends October 31, 2026 instead of September 30, 2026).
- No gallery example puts a legend on an axes the example made itself (five
  use `ax=`, with titles only, which already fit), so the fit changing no
  gallery figure is expected. My earlier guess that four examples would
  render differently was wrong.
- Fix: both examples now pass `random_state=0` (the library's own argument;
  passing UMAP's `random_state` through the `reduce` dict works too but
  prints umap's n_jobs warning). Verified: two runs each give identical
  figures, no warnings with `-W error::UserWarning`.
- Release form: README images flipped to `v1.1.1` (13 URLs), tutorial
  install cells already correct (migrator: 76 of 76). NOTE: the migrator
  must run on the `master` branch; on any other branch it retargets the
  tutorials to that branch.
- Plan: commit on local `master` (not pushed to `master`; pushed as
  `release/1.1.1` so nothing on `master` changes before the user's go),
  rebuild the gallery from that commit, repeat the comparison, independent
  review (Codex + fresh subagent), publish the pre-built gallery (keeping
  the old commit c90dd87e reachable on a `docs-gallery-v1.1.1-prev` branch
  because `master`'s docs build fetches it by id), record it, publish the
  notebooks, build wheel + sdist, create a DRAFT GitHub release (no tag).
  Tag, PyPI and the `master` push wait for the user's go.
- Found while running the examples gate: `animate_painting_embeddings.py`
  and `animate_conversation.py` seed UMAP through the `reduce` dict, which
  drew umap's "n_jobs value 1 overridden to 1 by setting random_state"
  warning (not shown on the docs pages, but shown to anyone running the
  example). Fixed in `hypertools/reduce/reduce.py`: the dict form now passes
  `n_jobs=1` when the model is seeded and the caller named no `n_jobs`
  (same as the bare-name form already did). Regression test in
  `tests/test_reduce_third_party_warnings.py`; changelog entry. The first
  gallery build from the release commit was stopped for this and restarted.
- Own review of all 89 figures (contact sheets, 117 tiles: every still,
  middle and last frame of every animation): everything matches 1.1.0
  apart from the three explained cases. One defect that 1.1.0 has too:
  `plot_shapes_zoo_001.png` showed stray x/y/z axis labels beside the
  Biplane panel (the biplane loads as a DataFrame, the other six shapes as
  arrays; column names become axis labels by design). Example now passes
  `xlabel='', ylabel='', zlabel=''`. Second gallery build stopped and
  restarted for this.
- Observed, left as is (faithful rendering of raw data, same as 1.1.0):
  in `plot_datasets_tour_001.png` the penguins and Kaggle-iris panels are
  near-lines because one unnormalized column dominates.
