# Session 2026-09-04: HyperTools 1.1.0 release procedure (steps 1-8, pause before PyPI)

Jeremy's instruction (this session): run RELEASE_CHECKLIST.md through the tag push
(his steps 1-8: date, squash-merge PR #283, checklist step 2 on master, gallery
publish, wheel/sdist build, push master + CI, tag v1.1.0 + CI), then PAUSE for his
comprehensive review before the PyPI upload. He will also review the Bluesky post
(`notes/bluesky-launch/POST.md`, gitignored) and the built gallery; the final
release date is determined by those reviews.

## State found at session start (09:31 EDT)

- HEAD a1fd2a41 == origin/dev-1.0; PR #283 OPEN, MERGEABLE/CLEAN; both CI runs on
  a1fd2a41 success (push 33845356432, pull_request 33845360079).
- UNCOMMITTED: Jeremy's two overnight review fixes from the pre-/clear session:
  1. market: `assemble()` trims month-end levels to the last trading date
     (clip was titled "September 30, 2026" on Sept 4) -> 316 months, May 2000-Aug 2026.
  2. weather: bottom-right y-label "Mean temperature (°C)" (capital M).
  Market notebook re-executed (clip re-copied to notes/bluesky-launch/20_*).
- A background chain from the pre-/clear session was STILL RUNNING at 09:31:
  execute_tutorial.py weather_decades -> cp clip -> full pytest -> sphinx -W build,
  logging to the old scratchpad `label_chain.log` / `docs_build3.log`.
- The market chain's full suite reported **2 failed / 3898 passed** (names lost to
  `tail -1`). Hypothesis: the weather notebook had been regenerated (outputs
  stripped) at 09:14, 4 min before that suite started, and
  tests/test_examples_are_native.py is the only test reading it. The chain's
  second suite run (after weather re-execution) decides this.
- venv: .venv is Python 3.12.10, hypertools 1.1.0 EDITABLE (ok). The shell's bare
  `python` is anaconda 3.9 and broken (numpy 2 ABI) -- always use .venv/bin/python.

## Decisions

- Release date: 2026-09-04 (today; the date the release commit + tag are made).
  If Jeremy's reviews push publication later and he wants the CHANGELOG date to
  match, the release commit, gallery publish and tag must all be redone.

## Progress log

- 09:31 started; waiting on the weather chain.
- 09:36 weather notebook re-executed (6 code cells, 2 with output; clip 5,574,832 B, byte-identical copy in notes/bluesky-launch/21_*). Frames checked by eye: weather final title "August 2013", bottom-right y-label "Mean temperature (°C)"; market final title "August 31, 2026" (was "September 30"). Chain now on the full suite, then the sphinx -W build.
- 09:47 Jeremy: colorbar and lower-right y-label must match; both "Average temperature (°C)" (y-label was "Mean"). Applied to examples/animate_weather_decades.py:289. Killed the superseded chain (its suite/docs results would not include this change); new chain in this session's scratchpad: regenerate -> execute weather -> copy clip -> full suite (`pytest_full.log`, with -rf so failure names survive) -> sphinx -W (`docs_build.log`).
- 10:16 chain done: weather 2/6 cells with output; clip byte-identical in notes/bluesky-launch/21_*; lower-right y-label verified by eye "Average temperature (°C)"; **3900 passed, 18 skipped, 0 failed** (the earlier "2 failed" was the stripped-output weather notebook, as hypothesised); sphinx -W exit 0. Committing + pushing dev-1.0 for CI before the squash-merge.
- 10:45 CI on b989ecf3 green: push 33883027571 + pull_request 33883033243, 16 jobs success each, release-gate skipped (dev-1.0). `gh pr edit --body-file` fails here with a GraphQL "Projects (classic)" error and writes nothing; used `gh api -X PATCH .../pulls/283`. Squash-merged via `gh api -X PUT .../pulls/283/merge` with the head-SHA guard: **master = 576601bf**, tree identical to dev-1.0 (`git diff origin/dev-1.0 master` empty). dev-1.0 NOT deleted.
- 11:25 checklist step 2 on master: migrator re-targeted 78 notebooks (20 tutorials + built gallery) to `%pip install -q "hypertools[interactive]"` and dropped the preview note; readme 8 image URLs -> v1.1.0 tag (0 SHA refs); CHANGELOG `## 1.1.0 (2026-09-04)`; stock_forecasting has no preview prose left. Local gates: `HYPERTOOLS_REQUIRE_RELEASE=1 pytest tests/test_notebook_install_gate.py tests/test_release_readiness_gate.py -k 'not gallery_colab_notebooks_are_published'` 13 passed; ruff clean.
- A `skillnote add` lesson landed in tracked `.claude/CLAUDE.md` mid-merge; stashed as stash@{0} on dev-1.0 ("skillnote gh-pr-edit lesson"); reapply on dev-1.0 after the release. Notes written after the release commit must NOT be committed to master (the gallery gate pins manifest source_commit == HEAD); commit them on dev-1.0.
- 11:26 release commit **19fc16fd** on master (not pushed). Step 3 artifacts built FROM 19fc16fd: `python -m build` clean; twine check PASSED x2; sdist carries CHANGELOG + LICENSE-APACHE-2.0 + THIRD_PARTY_NOTICES + NotoSans + OFL; wheel the same minus CHANGELOG (the 1.0.0 wheel on PyPI is identical in that respect); no auto_examples/_build/notes/mp4 in the sdist; fresh-venv smoke import 1.1.0 OK.
  Digests (upload ONLY these exact files after the tag CI is green):
    2256dab93be4a4d18bd1a86e2be3ae626d4c14d05ef41f74ce423d80132db52e  dist/hypertools-1.1.0-py3-none-any.whl
    9385938d65b4d5bd2fb70730e0921eb2204abe0eb58b3e4c4b66c62cadf2fd81  dist/hypertools-1.1.0.tar.gz
- 12:05 gallery built clean on master (`sphinx -W -E -a`, 0 warnings, 58 notebooks, all PyPI spec) + `docs/post_build.py` (badges -> docs-notebooks/v1.1.0). NOTE: `make html` would use the Homebrew sphinx-build and `python3`; build with `../.venv/bin/python -m sphinx ...` then `.venv/bin/python docs/post_build.py` from the repo root. Published: docs-notebooks e0a3877 -> 518c37e, manifest source_commit 19fc16fd, 58 notebooks. All 14 release gates pass locally with HYPERTOOLS_REQUIRE_RELEASE=1 (gallery gate included). The first master run (33889314350, squash commit 576601bf) had release-gate FAIL on exactly the 5 dev-form checks -- expected by the checklist order; not a bug.
- 12:06 pushed master 19fc16fd (checklist step 4). Waiting for its CI incl. release-gate.
- 12:15 gallery examined: index thumbnails + market page (1:00 video, "316 months ... May 2000 - Aug 2026") render; all 58 example pages carry the Colab badge -> docs-notebooks/v1.1.0; 65 thumbnails. Screenshots sent to Jeremy; local server `python -m http.server 8765` in docs/_build/html (session-lifetime only).
- **RTD does NOT auto-build**: GitHub hook 11883014 (https://readthedocs.org/api/v2/webhook/github/hypertools/) gets HTTP 400 on every delivery: "This webhook doesn't have a secret configured. For security reasons, webhooks without a secret are no longer permitted." No RTD build since 647ce929 (2026-07-24). FIX (Jeremy, needs RTD login): RTD project > Settings > Integrations > GitHub incoming webhook > copy/regenerate the secret, then put it on the GitHub hook (Settings > Webhooks > edit > Secret), redeliver a push; or trigger the `latest` + `v1.1.0` builds by hand from the RTD dashboard (checklist step 6). Until then RTD `latest`/`stable` still serve 1.0.0.
- 12:25 master run 33892561136 on 19fc16fd: release-gate PASSED (first real run), wheel-smoke/dataset-gate/live-source-gate PASSED, but ALL 12 matrix jobs FAILED: tests/test_changelog_1_1.py hard-coded `## 1.1.0 (unreleased)` (2 tests). Fixed the test to match `## 1.1.0 (unreleased|YYYY-MM-DD)` (unique, first version section; both forms verified) -> **master 2dea6c29** (tests-only diff vs 19fc16fd). Gallery manifest republished from 2dea6c29 (docs-notebooks 518c37e -> 47d4a0b; built notebooks unchanged since examples/conf are identical between the two commits). Artifacts REBUILT from 2dea6c29 (sdist ships tests/); digests in artifacts2.log / below. 14 release gates pass locally. Pushed master 2dea6c29 with DOC_GATE_OVERRIDE (tests-only). 19fc16fd digests above are SUPERSEDED.
- 12:45 artifacts rebuilt FROM 2dea6c29: twine PASSED x2, sdist 5 bundled license/font/changelog hits, wheel 4, no stray build products, fresh-venv import 1.1.0. Full local suite on master 2dea6c29: **3900 passed, 18 skipped, 0 failed**.
  CURRENT digests (upload ONLY these exact files, only if the tag points at 2dea6c29):
    7f980e9811cf57b670e134028050961b60ffc2b567402ca952e476b6cb42e792  dist/hypertools-1.1.0-py3-none-any.whl
    7be7fe938ae06b68e039e465beaf617874212bfc05468139db881547db7533b5  dist/hypertools-1.1.0.tar.gz
- 13:45 master run 33896556203 on 2dea6c29: ALL 17 jobs success incl. release-gate + docs-clean (release-form CI green). Tagged **v1.1.0 = 2dea6c29** (annotated, "HyperTools 1.1.0") and pushed; waiting for the tag CI. Then PAUSE for Jeremy's review before the PyPI upload (dist/ digests above are for 2dea6c29).
- 13:55 Jeremy: edited `notes/bluesky-launch/POST.md` (gitignored; his review copy is now the authority -- do not regenerate or overwrite it). Post the thread only after 1.1.0 is on PyPI and the five tutorial URLs resolve (per the POST.md caveats).
- 14:00 Jeremy asked for a Colab feature-tour notebook (all user-facing features, by module; first cell installs 1.1.0). Target `notes/colab/hypertools_1.1_feature_tour.ipynb`; `notes/colab/` added to .gitignore (uncommitted on master; commit that .gitignore line + these notes on dev-1.0 after the release). Authored + executed headlessly by a subagent via scripts/execute_tutorial.py, outputs stripped before hand-off.
- 14:20 feature-tour notebook DONE: `notes/colab/hypertools_1.1_feature_tour.ipynb`, 55 code + 52 markdown cells, executed via scripts/execute_tutorial.py exit 0 (52/55 cells with output; the 3 silent = 2 pip cells + commented Chronos), 30 s wall, outputs stripped, venv still editable. Not covered: kaggle/538/HF/Drive loaders (prose), torch autoencoders, streaming plots, `explore=`, plotly animations.
- **BUG found by the tour (NOT fixed; master is tagged)**: inside Jupyter, `with hyp.set_interactive_backend('TkAgg')` on a machine without Tk raises a raw `ModuleNotFoundError` instead of the documented `HypertoolsBackendError`; `_switch_backend_notebook` (hypertools/plot/backend.py ~L806) only catches `KeyError` from the `%matplotlib` magic; the script path wraps it correctly. Disposition = Jeremy's call: fix on dev-1.0 for 1.1.1, or fix now and redo release commit + gallery publish + tag.
- 14:50 **v1.1.0 tag run 33902255584: ALL 17 jobs success** (release-gate + docs-clean on the tag). Notebook install cell rewritten as `!pip install ...==1.1.0 || pip install ...@v1.1.0`, proven in a fresh venv (PyPI miss -> tag -> import 1.1.0). Jeremy's Colab run had hit "No matching distribution for hypertools==1.1.0" (expected: PyPI upload not done).

## PAUSED HERE (steps 1-8 complete) -- resume = checklist step 6 after Jeremy's sign-off

State: master = v1.1.0 = **2dea6c29** (release commit 19fc16fd + changelog-test fix); origin/master identical; docs-notebooks/v1.1.0 manifest source_commit 2dea6c29; dist/ built from 2dea6c29 with the digests above (7f980e98... whl, 7be7fe93... tar.gz). dev-1.0 NOT deleted. The local docs server (127.0.0.1:8765, docs/_build/html) dies with the session.

Jeremy reviews: built gallery (screenshots sent; local server), Bluesky POST.md (his edited copy), the feature-tour notebook, the release overall.

To finish (step 6+), in order:
1. `git rev-parse v1.1.0^{commit}` == 2dea6c29 and `shasum -a 256 dist/*` == digests above; ONLY then
   `twine upload dist/hypertools-1.1.0.tar.gz dist/hypertools-1.1.0-py3-none-any.whl` (Jeremy's PyPI credentials/2FA).
2. GitHub release for v1.1.0: draft body at `<scratchpad>/release_notes_v1.1.0.md` = CHANGELOG 1.1.0 section + the unpublished 1.0.1 section (regenerate from CHANGELOG.md if the scratchpad is gone): `gh release create v1.1.0 --title "HyperTools 1.1.0" --notes-file ...`.
3. Clean-env `pip install hypertools` -> 1.1.0; README quick-start.
4. RTD: fix the webhook secret (see 12:15 note) or trigger `latest` + `v1.1.0` builds by hand; set default/stable -> v1.1.0; confirm landing page shows story_trajectories.gif and the Colab cells say `%pip install "hypertools[interactive]"`.
5. PyPI page: 8 README images resolve (all 200 on the tag already).
6. conda-forge bot PR (hours after upload): carry any new/raised floors from pyproject vs v1.0.0.
7. Bluesky thread (POST.md) after the five tutorial URLs resolve on RTD.
8. Post-release housekeeping: fix the backend-error bug (14:20 note) on dev-1.0 for 1.1.1; delete the two stale agent worktrees under .claude/worktrees (Jeremy's call); dev-1.0 deletion = Jeremy's call.

## 15:00-16:00 Colab errors in the feature tour -> TOOLBOX bug (upstream matplotlib nbAgg), fixed on dev-1.0

Jeremy ran the tour on Colab (Python 3.13): 3 error cells, all `AttributeError: 'NoneType' object has no attribute 'remove_callback'` from `matplotlib.animation.Animation._stop`, raised by `plt.close` (his section-11 `plt.close('all')`, and IPython's end-of-cell `matplotlib_inline...show(close=True)` after the two text-section plots). Pulled the executed notebook from his Drive to read the tracebacks.
Root cause (traced with a monkeypatched `_stop`): `FigureManagerNbAgg.destroy()` -> `clearup_closed()` fires `close_event` RE-ENTRANTLY (the comm on-close handler runs a nested destroy while the outer callback snapshot is still iterating), so `_stop` runs twice; the second call finds `event_source` None. Reproduced with PLAIN matplotlib (no hypertools) 3.10.8 and 3.11.1 under `%matplotlib nbagg` + FuncAnimation + plt.close. hypertools selects nbAgg in Colab/classic Jupyter, so: every displayed animation made the NEXT static-plot cell fail, and `show=False` animations failed inside plot() at its own plt.close(fig). NOT a notebook bug. (matplotlib #25181 is a different nbagg-close issue -- a deprecation warning.)
Fix (dev-1.0, uncommitted at this line): `hypertools.plot.animate.HyperFuncAnimation` (FuncAnimation whose `_stop` ignores the repeat call), used at all 7 construction sites in matplotlib_backend.py; CHANGELOG bullet under 1.1.0 Bug fixes; tests/test_animation_close_nbagg.py (unit: double `_stop`/double close_event no-op; upstream non-idempotence documented; real ipykernel+nbAgg run of the Colab sequence). Verified: repro notebooks pass in .venv (py3.12, mpl 3.10.8) and venv313 (py3.13, mpl 3.11.1); plain-matplotlib repro still fails (upstream).
Repro scripts: scratchpad make_repro*.py / run_repro.py; executed Colab copy: scratchpad/colab_run.ipynb.
DECISION for Jeremy: (a) 1.1.1 right after 1.1.0, or (b) fold into 1.1.0 = new release commit on master, republish gallery manifest, rebuild dist, MOVE the v1.1.0 tag (nothing on PyPI/conda yet, so moving it is feasible). Recommendation: (b) -- Colab is the main first-contact path and every animation currently poisons the next cell there.
