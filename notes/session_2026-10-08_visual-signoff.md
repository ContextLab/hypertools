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
