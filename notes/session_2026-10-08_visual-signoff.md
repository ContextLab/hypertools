# Session 2026-10-08: finishing the 1.1 final review (visual sign-off round)

Branch `fix/1.1-release-review`; PR #286 still OPEN and unmerged. Nothing merged, tagged or published.

## State at session start
- PR head = c433b8f0 (unchanged since 2026-09-12). Local branch had 6 unpushed notes commits
  (incl. c7dea477 by Jeremy's 2026-10-03 session moving notes into .claude/compound lessons).
- RTD: still no build since 2026-07-24 (API v2 confirms). GitHub hook 11883014 -> readthedocs
  legacy URL, no secret; deliveries list empty (retention). Needs RTD admin re-sync. MANUAL.
- CI rerun (attempt 2) of run 34674962598 at c433b8f0 started for dependency drift.

## Done this session
- Headless tour at c433b8f0 with current deps: 241 PASS / 3 SKIP / 0 FAIL; 74 visual cases.
- Built a review artifact (https://claude.ai/artifact/UPKL2S8yGd6VaAp9i5adm5) from the executed
  tour (scratchpad gallery/extract.py; plotly animations -> mp4 via gallery/plotly_anim_mp4.py);
  verdicts stored in its db collection `verdicts` (read with ArtifactData list).
- Jeremy's verdicts: 54 pass / 20 flag. Flags -> actions:
  - plotly animations didn't play in page -> page now plays frame-by-frame mp4s.
  - ANIM-clock one curve, COLOR-luminance shows input not palette, TEXT-* unreadable,
    PANEL-models subplots title "example is wrong" -> tour cases changed (b240258a).
  - ANIM-forecast truth shown from frame 0 -> DECIDED: truth point appears when a forecast reaches it.
  - RED-describe 'average' was pooled stack -> DECIDED: mean of individual curves; pooled kept as
    result['pooled']; integer ticks; describe(show=False) returns fig.
  - PLOT-scale-matplotlib grid -> plotly (no grid) is correct; remove matplotlib grid.
  - GUI-native: no Qt binding locally; remains manual/optional.
- Agent visual pre-review found 2 real plotly bugs (Jeremy had passed both): nested-list leaf
  names 1,2,2,4 and 3-D markers at half size -> fixed a4b146af (full suite 6583/0).
- Decisions: forecast_cluster/hue default colours offset past observed colours (approved);
  raise floors (183de73c); XOM CIK pin + HON note (8d02cb81); describe/t= docs + anim size
  note (approved); axis_scale midrange DEFERRED to 1.2 (open issue); set_autoinstall KEEP.
- _aa_x (plotly 1-D forecast branch) proven unreachable (1-D animation refused at
  plotly_backend.py:1630); 528-call settrace sweep, line 5788 hit 0 times.
- Bluesky clips re-rendered at c433b8f0 (gitignored, notes/bluesky-launch/).

## Pending
- Integrate worktree agents: truth timing + cluster colours; describe + grid + docstrings.
- Regenerate media: market_sectors tutorial mp4 + thumb (XOM data), anything else touched.
- Full verification: pytest, ruff, sphinx -W, doctest, tutorials, headless tour, push, CI, Colab.
- Round-2 visual review of changed cases only (carry forward pass for pixel-identical renders).
- Open GitHub issue for axis_scale midrange (1.2).

## Integration + verification (2026-10-08 afternoon)
- Integrated: a4b146af (plotly nested names + 3-D markers), b240258a/ea7f90a2 (tour cases),
  183de73c (floors), 8d02cb81 (XOM CIK), c1b7adc3/888e231d (describe), 279cd73a (no data-scale grid),
  ddd98ae7 (docstrings), cb0f9a95 (truth reveal), 387e7704 (forecast group colours),
  bdbcebf1 (CI apt --no-install-recommends + retries), 5dda6190 (25 tutorials re-executed),
  5520987b (market thumb).
- Verified at 5520987b: tutorials 25/25; pytest 6633 passed/21 skipped/0 failed; ruff clean;
  example smoke 348 passed; sphinx -W html 0 warnings, 51 gallery; doctest 323/0; thumbs (only
  market_sectors changed). ea7f90a2 changed only scripts/update_feature_tour.py (tour tooling).
- Headless tour at ea7f90a2: 241 PASS / 3 SKIP / 0 FAIL, clean tree, exact commit.
- CI drift rerun at c433b8f0: all 15 test jobs + gates green; docs-clean CANCELLED at 75-min timeout
  because apt fetched 182 MB at 111 kB/s (27 min) -> fixed in bdbcebf1.
- Pushed ea7f90a2; CI run 37825666381 in progress (5520987b run cancelled as superseded).
- Issue #287 opened for axis_scale midrange (1.2).
- Round-2 review page published (same artifact URL, collection verdicts_r2): 29 to review
  (flagged or changed render + HIER-plotly names + DOCS-market-sectors), 46 carried (byte-identical
  renders previously passed). Bluesky 20_market_sectors re-rendered at ea7f90a2.

## Round 2 + GUI + final candidate (2026-10-08 evening)
- Round-2 verdicts: 28/28 pass; GUI-native flagged "need more instruction".
- GUI-native recorded in a real Qt window (PyQt6 in a throwaway uv venv; QTest events; frames grabbed
  from the window): 11 hover labels, azim -60 -> 147, xlim +-1.00 -> +-1.18, q closes, clean exit.
  Found + fixed (ea7a5f4e): tour GUI script used show=False (no window ever opened); explore hover label
  clipped at the left edge (now opens toward axes centre; tests/test_explore_label_placement.py).
  Recorder: scripts/record_gui_native_screencast.py (ruff tidy a5c83938). Screencast on the review page.
- Full suite at ea7a5f4e: 6637 passed / 21 skipped / 0 failed; ruff clean at a5c83938.
- Colab at a5c83938: 241/3/0; console showed Colab's table script throwing 'buttonEl already declared'
  where one cell displays two DataFrames (IMP-score, results summary) -> second table as plain HTML
  (4c762c0c). Colab at 4c762c0c (drive/1MbgX7eFkavZPtdSQ4FoJOsSP6eYJSv3s): 241/3/0, SET-01 commit
  confirmed, no notebook-caused console errors.
- Headless tour at 4c762c0c: 241/3/0, clean tree. CI run 37868951965 at 4c762c0c: success, 17/17.
- PR #286 head = 4c762c0c, OPEN, MERGEABLE. NOT merged. Waiting on Jeremy: GUI verdict, RTD webhook,
  merge sign-off. Packet: notes/final_review_2026-09-11/START_HERE.md (untracked).

## 2026-10-09: docs site (RTD parity) + final candidate d21e0340
- Jeremy confirmed GUI-native ("gui looks good") -> recorded in verdicts_r2. Visual sign-off complete.
- RTD-parity build: from docs/, READTHEDOCS=True READTHEDOCS_GIT_IDENTIFIER=master READTHEDOCS_OUTPUT=$OUT
  sphinx -T -b html -d _build/doctrees . $OUT/html, then python docs/post_build.py.
- post_build bug (4e0dbaeb): looked for _images under $READTHEDOCS_OUTPUT not .../html.
- Browser pass (sampled) found gallery/dark/mobile/duplicate/console defects -> 0ad2a366; docstring
  markup leaks (38 on 8 pages) + _Unset sentinel repr -> 09f6e8b1, test 61b5cda5.
- Jeremy then reported hierarchy.html broken: furo red ERROR box from `.. contents::` (CSS ::before, invisible
  to sphinx -W / HTML scans). Fixed 998581dd. Lesson docs-site-check-every-page + checker (7a240862).
- Every-page check also led to: tutorial setup cell collapsed under title (post_build), one Methods/Attributes
  table per class (numpydoc_show_class_members=False), stubs committed as generated (40ea3170);
  teaser GIFs at native 200px (d21e0340).
- Full suite at 40ea3170: 6655 passed/21 skipped/0 failed; ruff clean. Tour at d21e0340: 241/3/0.
- Cold RTD build at d21e0340: 0 warnings, 171 pages; linkcheck 0 broken; every-page check 137 pages 0
  findings; verify_docs_playwright 8/8.
- Colab at d21e0340 (drive/1Bz_253Z5ZDw7tKUqy0v3xJJgPLccnlAz): 241/3/0, SET-01 commit confirmed. Console (b):
  plotly.py renderer's own `import "…plotly-3.6.0.min"` (rstrip(".js") bug) 403 + MathJax 2.7.5 errors: UPSTREAM.
- CI run 37938862214 at d21e0340: success, 17/17 (16 success + release-gate skipped). PR #286 head = d21e0340, MERGEABLE. NOT merged.

## 2026-10-09 (later): red-team review, sign-off, re-cut
- f5a0af21: changelog links (docs home + release notes), release notes brought up to date. CI 37958047439
  success 17/17; Colab 241/3/0 (tab dropped mid-run; counts from the original Run all).
- Jeremy pasted an independent red-team review of f5a0af21 (notes/release_redteam_2026-10-09/, untracked,
  his session's files). All 5 findings reproduced with his probe, then fixed (rules chosen by Jeremy via
  questions): (1) scoring ranks complete models only, attrs['incomplete'], None verdicts (3121910d);
  (2) impute truth=/mask= aligned by label, mismatches raise (3121910d; type checks through shared helpers
  0585a5ad after tests/test_datatype_gate.py failed in the full suite); (3) describe clones+sweeps an
  unfitted instance, rejects fitted models (33810c6a) and reduce=None (9648e258); (4) lazy_import checks
  installed extras against declared versions, upgrades or reports (b2d35ad7); (5) isotropic docstring.
  Also: sklearn clone(hyp.Pipeline) works (33810c6a; failed in 1.0 too).
- Probe after: labels 3.667 x3; best=SimpleImputer, incomplete=['PPCA']; describe instance == name.
- e95777fb: full suite 6760 passed/21 skipped/0 failed; ruff clean; tour 241/3/0, all 73 figure renders
  byte-identical to the round-2 approved ones; RTD-way docs -W 0 warnings, 137 pages 0 findings;
  verify_optional_install PASS; packaging tests 13 passed; Colab (drive/1O-gcAIgAxVMrXTFboIFoE0mjHaihyFoQ)
  241/3/0, SET-01 confirmed. Colab image ships plotly 5.24.1 (below the 6.1.1 floor); the tour's install
  cell replaced it with 7.1.0 before import, so the new version check did not fire there.
- CI 37972394267 at e95777fb: 15 success + release-gate skipped + 1 FAILURE: test (ubuntu-latest, 3.11),
  tests/test_multibyte.py::test_render_script_exits_NO_BROWSER_when_the_browser_will_not_launch,
  subprocess.TimeoutExpired at 120 s (same test passed in seconds on the other 15 jobs). Kaleido's own
  30 s timeout did not fire. Fixed in the render script (hard deadline -> NO_BROWSER exit) + new test
  with a real silent executable (a0c30a10).
- RTD webhook root cause confirmed by POSTing an unsigned request to the hook URL: HTTP 400 "This webhook
  doesn't have a secret configured ... no longer permitted". Hook 11883014 dates from 2017-02-02, legacy
  URL, no secret. Needs Jeremy's RTD admin: re-sync/re-create the GitHub integration, build latest + v1.1.0.
- SIGN-OFF 2026-10-09 (AskUserQuestion): "Sign off: merge and re-cut", CHANGELOG date 2026-10-09.
  Scope: fast-forward master, checklist steps 2-5 (date, gallery, artifacts, master CI, move tag, tag CI,
  replace draft assets + body). STOP before PyPI upload and before publishing the GitHub release.

## 2026-10-09 (evening): re-cut in progress

- Hosted CI found three more problems after the sign-off, each fixed on the release line:
  macOS 3.10 live Kaggle load (DNS; now `skip_on_transient_network`, cb0c5432); ubuntu 3.11
  render-script test timing out at 120 s (54c89af8 added stage lines + faulthandler; the dump showed
  the script had its verdict and Python was waiting at shutdown on choreographer's non-daemon threads
  after a failed browser launch; 75d6c723 exits with `os._exit` once the verdict is printed).
  Upstream (choreographer) not reported.
- 75d6c723: PR CI 16 green; pushed to `master` as a fast-forward at 23:02 UTC. GitHub marked PR #286
  merged and closed #284 and #285 (neither had an unchecked item).
- First `master` run: `release-gate` FAILED, 4 tests, `ModuleNotFoundError: nbformat`. The job installed
  only pytest and is skipped on PRs, so this was its first real run with the executor tests in the gate
  file. f7462896 installs nbformat + nbclient there and adds a pre-flight checklist item (run the gate
  files in a pytest+nbformat+nbclient venv). `master` run 38005395083 at f7462896: 17/17 green.
- Jeremy re-enabled the RTD webhook; both pushes triggered builds, and all three builds FAILED with
  "Build terminated due to time out" (15-minute limit; ~165 s setup, sphinx needs ~25 min for the 51
  gallery examples; the 1.0 build took 885 s). RTD still serves 1.0; the five launch tutorial URLs 404.
- Jeremy's decisions (2026-10-09): make the RTD build reuse the pre-built gallery; shorten the README's
  1.0/1.1 sections to a pointer at the changelog and add the five launch clips as GIFs ("updated readme
  and gifs look great"; "make sure they have nice smooth frame rates").
- Implementation: `scripts/publish_prebuilt_gallery.py` force-pushes one orphan commit
  (`auto_examples/` + `manifest.json`) to `docs-gallery-v<version>`; `docs/fetch_prebuilt_gallery.py`
  runs in `.readthedocs.yaml` `pre_build` and copies it into `docs/auto_examples`; sphinx-gallery then
  skips every example whose `.py.md5` matches. Measured in a fresh clone with the RTD recipe: 38 s,
  0 warnings, output identical to the cold build except the two download zips and the execution-times
  page. New release gate `test_release_gate_prebuilt_gallery_is_published_for_this_commit`.
- Found on the way: two gallery pages printed `/Users/jmanning/hypertools/examples/...` in a captured
  warning. `relative_warning_paths` (docs/_gallery_log_filter.py, installed from conf.py's
  reset hook) makes it `examples/plot_impute.py:43: ...`; the publisher refuses a gallery that names
  the checkout path.
- README GIFs: `images/tour_*.gif`, made with ffmpeg + gifsicle from the launch mp4s at 15-20 fps
  (5.0 / 2.4 / 3.7 / 3.2 / 3.1 MB). The 10 new README URLs (5 tutorials, 5 images at the v1.1.0 tag)
  404 until RTD builds and the tag moves; re-check after both.
- Still held: the `v1.1.0` tag (at 96ac8b7f), the draft release, PyPI. Order from here: commit, cold
  gallery build from that commit, publish notebooks + pre-built gallery, artifacts, push `master`,
  confirm RTD `latest` builds, tag, tag CI, confirm RTD `stable`, replace draft assets and body. STOP
  before PyPI and before publishing the GitHub release.

## 2026-10-09 (evening): re-cut in progress

- Hosted CI found three more problems after the sign-off, each fixed on the release line:
  macOS 3.10 live Kaggle load (DNS; now `skip_on_transient_network`, cb0c5432); ubuntu 3.11
  render-script test timing out at 120 s (54c89af8 added stage lines + faulthandler; the dump showed
  the script had its verdict and Python was waiting at shutdown on choreographer's non-daemon threads
  after a failed browser launch; 75d6c723 exits with `os._exit` once the verdict is printed).
  Upstream (choreographer) not reported.
- 75d6c723: PR CI 16 green; pushed to `master` as a fast-forward at 23:02 UTC. GitHub marked PR #286
  merged and closed #284 and #285 (neither had an unchecked item).
- First `master` run: `release-gate` FAILED, 4 tests, `ModuleNotFoundError: nbformat`. The job installed
  only pytest and is skipped on PRs, so this was its first real run with the executor tests in the gate
  file. f7462896 installs nbformat + nbclient there and adds a pre-flight checklist item. `master` run
  38005395083 at f7462896: 17/17 green.
- The RTD webhook fires again; both pushes triggered builds, and all three builds FAILED with
  "Build terminated due to time out" (15-minute limit; ~165 s setup, sphinx needs ~25 min for the 51
  gallery examples; the 1.0 build took 885 s). RTD still serves 1.0; the five launch tutorial URLs 404.
- Jeremy's decisions (2026-10-09): make the RTD build reuse the pre-built gallery; shorten the README's
  1.0/1.1 sections to a pointer at the changelog and add the five launch clips as GIFs with smooth
  frame rates ("updated readme and gifs look great"); American spelling in the README.
- Implementation: `scripts/publish_prebuilt_gallery.py` force-pushes one orphan commit
  (`auto_examples/` + `manifest.json`) to `docs-gallery-v<version>`; `docs/fetch_prebuilt_gallery.py`
  runs in `.readthedocs.yaml` `pre_build` and copies it into `docs/auto_examples`; sphinx-gallery then
  skips every example whose `.py.md5` matches. Measured in a fresh clone with the RTD recipe: 38 s,
  0 warnings, output identical to the cold build except the two download zips and the execution-times
  page. New release gate `test_release_gate_prebuilt_gallery_is_published_for_this_commit`.
- Found on the way: two gallery pages printed `/Users/jmanning/hypertools/examples/...` in a captured
  warning. `relative_warning_paths` (docs/_gallery_log_filter.py, installed from conf.py's reset hook)
  makes it `examples/plot_impute.py:43: ...`; the publisher refuses a gallery naming the checkout path.
- README GIFs: `images/tour_*.gif`, made with ffmpeg + gifsicle from the launch mp4s at 15-20 fps
  (5.0 / 2.4 / 3.7 / 3.2 / 3.1 MB). The 10 new README URLs (5 tutorials, 5 images at the v1.1.0 tag)
  404 until RTD builds and the tag moves; re-check after both.
- Still held: the `v1.1.0` tag (at 96ac8b7f), the draft release, PyPI. Order from here: commit, cold
  gallery build from that commit, publish notebooks + pre-built gallery, artifacts, push `master`,
  confirm RTD `latest` builds, tag, tag CI, confirm RTD `stable`, replace draft assets and body. STOP
  before PyPI and before publishing the GitHub release.
- e831cb2b: Read the Docs fix + README tour + American spelling (about 1,200 prose replacements; Jeremy
  chose "User-facing text now"; identifiers, quoted literals, vendored code untouched; io.ipynb and
  pipelines.ipynb re-executed). Opened PR #288 from `release/1.1.0-recut` so CI runs before `master`.
- Commit security review of e831cb2b flagged the fetch step: content from a mutable branch, and symlinks
  followed on copy. Fixed: `unsafe_entries` rejects symlinks/non-regular files and pages that read
  outside the gallery (include, literalinclude, :file:, image, download); the publisher applies the
  same check. Accepted as is: the branch is as trusted as push access to the repository. Suggested to
  Jeremy: a branch protection rule for `docs-gallery-*`.
- A second commit security review (of 3a651b52) said the RST content scanner could be bypassed
  (docutils syntax the regexes miss; includes through unscanned files). Replaced the scanner with a
  commit pin: the publisher writes the orphan commit's id to `docs/prebuilt_gallery.json`, the fetch
  step fetches that exact id (`git fetch <remote> <sha>`, verified on GitHub) and only accepts a
  40-hex id for the current version. Overwriting the branch cannot change what a build reads. Symlink
  check kept. Release order is now: build gallery -> publish_prebuilt_gallery --push -> commit the pin
  -> publish notebooks -> artifacts. Gate checks the pin, the branch head, that only the pin changed
  since the build, and the manifest md5s.
- cef40569 binds the record to the gallery branch head and the manifest. PR #288 CI on it: the three
  Python 3.10 jobs FAILED (my new test imported tomllib, 3.11+); RTD's PR build timed out as expected
  (no record committed yet) but its pre_build step ran and printed "none recorded for 1.1.0".
- Jeremy: "several times now the gallery commit hasn't matched the expected tests ... carefully think
  through the *full* logic and do a test run"; lesson commit-pinned-checks-dry-run-first recorded.
  Also: "install the latest [sphinx-gallery] to match readthedocs" -> local 0.21.0 -> 0.22.1, floor raised
  in docs/doc_requirements.txt. (RTD's other docs packages are newer than the local venv too:
  matplotlib 3.11.2 vs 3.10.8, plotly 7.1.0 vs 6.8.0, numpy 2.4.6 vs 2.3.5; not changed.)
- scripts/rehearse_prebuilt_gallery.sh: end-to-end rehearsal against a local bare remote. Mini run
  (three examples, real sphinx-gallery 0.22.1) PASSED: fetch 3 of 3 current, RTD recipe 36 s, 0 examples
  executed, overwritten branch rejected. Changed tests also run under a real Python 3.10 (uv): 61 passed.

## 2026-10-10 (after midnight EDT)

- Jeremy's process lessons (recorded as user-level compound lessons `commit-pinned-checks-dry-run-first`
  and `shorten-the-loop-before-iterating`): think a commit-pinned check through and rehearse it before
  building it; narrow check first and the full suite once; overlap waits (CI beside local work); track
  how long each step takes. Timing table: notes/final_review_2026-09-11/timings_2026-10-09.md (untracked).
- 553c3922: gallery built (0 warnings, 51 notebooks, 28.6 min). Full rehearsal against a local bare
  remote PASSED (51 of 51 current, RTD recipe 41 s, 0 examples executed, overwritten branch rejected).
- Real sequence: published docs-gallery-v1.1.0 = 3e83fb59, recorded it (fc141fff), pushed the PR branch.
  Read the Docs PR build 35055033 on fc141fff SUCCEEDED in 373 s: fetched 3e83fb59, "51 of 51 examples
  are current", sphinx 141 s, "successfully executed 0 out of 0", 0 warnings; the five launch tutorial
  pages and hierarchy.html return 200 on the PR preview; plot_impute shows the repository-relative
  warning path.
- The sequence then stopped at the release gates: the CHANGELOG heading was dated 2026-10-09 and the
  record commit was made at 00:28 EDT on 2026-10-10. Re-dated to 2026-10-10. That is a tracked change
  beyond the record, so the gallery is rebuilt once more from this commit. With the date corrected (trial,
  uncommitted) all 25 release gates passed, and 24 + 1 pylsl skip in the release-gate job's environment.
- Parallel gallery trial (6 workers, scratch clone, while the serial build ran): 12.9 min, exit 0 with -W.
  animate_weather_decades (696 s) and animate_market_sectors (573 s) bound it. One loky "worker stopped"
  notice. Not adopted for 1.1.0.

## 2026-10-10: release day and the first post-release report

- 12:10 UTC: `v1.1.0` tag moved to cf76a931 (master CI 17/17 green at 08:15; four idle hours because my
  waiter watched the "Dependency Graph" run id instead of the "Tests" run). Tag CI 17/17 green 14:11.
  Each CI run takes ~2 h because the ubuntu-3.12 job runs the suite three times serially.
- Jeremy: "Can you do steps 1--3, and also do the conda forge release? I'll post the bluesky thread".
  RTD `stable` re-synced by redelivering the last master push webhook delivery (RTD ignores a force-moved
  tag); PyPI 1.1.0 uploaded 14:40 (digests match; clean-env wheel smoke OK); GitHub release published
  14:42; conda-forge/hypertools-feedstock#1 opened from the jeremymanning fork, rerendered, merged 14:51,
  package uploaded (CDN index lag; clean conda install not yet confirmed).
- First report after release (Jeremy, Colab, gallery notebook animate_morph_zoo): (1) TypeError
  "Normalize.__init__() got an unexpected keyword argument 'mode'": a runtime opened before the PyPI
  upload had hypertools 1.0.0 and the unfloored `%pip install -q "hypertools[interactive]"` was "already
  satisfied". Deleting the runtime fixed it. (2) the notebook showed no figure: six showcase examples end
  with `fig = anim.figure` inside `if __name__ == '__main__':`, which displays nothing in a notebook.
- Fixed in the PUBLISHED notebooks (docs-notebooks branch, no change to master or the tag): all 51 install
  cells now read `hypertools[interactive]>=1.1.0` (Jeremy: keep it); the six showcases have a final cell
  `anim` (Jeremy chose "Play the animation inline"). Verified on Colab: morph zoo plays a 30 s video,
  rendered in 48 s. Measured inline render with the PyPI package, locally: morph 4 s, forecast 9 s,
  conversation 19 s, paintings 52 s, market 74 s, weather 235 s; none exceeds the embed limit.
- Source fix for the next release is on branch `fix/gallery-install-floor` (NOT master: the release gate
  allows no commit there between releases): conf.py emits the floored install line and appends the
  playback cell after sphinx-gallery writes the notebooks (docs/_gallery_notebooks.py);
  check_release_notebooks.py rejects an unfloored install; tests added. What we missed: no gallery
  notebook was run on Colab against the published package; the checklist now says to.
