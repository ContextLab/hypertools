# Session 2026-09-11: independent final review of PR #286 candidate

Candidate under review: `fa3e60e5670cabfc1cbbd282e122b9098cc6d1a3` (branch
`fix/1.1-release-review`, PR #286 OPEN, CI 16 green + release-gate skipped).
Guide followed: `notes/notebook_critical_review_2026-09-10/START_HERE_FINAL_REVIEW.md`.

**HOLD: do NOT merge / tag / publish anything until Jeremy signs off.**
Fix issues found along the way (commit + push to the PR branch), then refresh
evidence/CI for the new head.

## Plan / status

- [x] A. Headless re-run of feature tour at HEAD: fresh execution 2026-09-11 15:21-15:24Z, 3m31s, exit 0,
      **241 PASS / 3 SKIP / 0 FAIL**; skips = SOURCE-drive, SOURCE-dropbox (trusted_remote_pickle off), GUI-native (native_gui off). Matches reference.
- [~] Colab: uploaded cleared candidate notebook via Playwright (Colab Pro session is signed in) -> https://colab.research.google.com/drive/1c84Tcr-whDrtDynfGxQ0oC15VajdtETg ; Run all started 15:25Z
- [ ] B. Cold code review of the PR library diff (3 reviewer agents: plot core, plotly/colors/forecast, data/predict/io)
- [ ] C. Docs/claims review (README, CHANGELOG, release notes, tutorials, RELEASE_CHECKLIST)
- [ ] D. Visual review of tour outputs (separated expected / observed / adjudicate agents)
- [ ] E. Fresh-venv Colab-proxy run (install from git at exact SHA, installer cell executed) -- AFTER A (LSL outlets collide across concurrent runs)
- [ ] F. Browser frontend check (local Jupyter + playwright: previews survive Run all, shared viewer open/switch/close)
- [ ] G. Fix findings, re-verify, push, refresh CI + evidence packet

## Findings log

### Fresh Colab run at fa3e60e5 (15:25-15:35Z): 236 PASS / 5 FAIL / 3 SKIP
- STREAM-01/02/03 FAIL `'HyperPlotlyFigure' object has no attribute 'axes'` -> LIBRARY BUG (1.0 too):
  plot_stream's inner head plot followed the plotly render preference (Colab auto). FIXED 2503a543
  (+ test_stream_plot_under_global_plotly_render_backend). Tutorials streaming_data/lsl_streaming/io hit it on Colab.
- ANIM-clock / PLOT-cjk FAIL (tour assumed matplotlib) -> tour fix deeb1f93 (backend='matplotlib').
- RED-describe displayed `None` (describe(show=False) returns fig=None by documented design) -> tour fix deeb1f93.
  QUESTION for Jeremy: plot(show=False) returns a fig but describe(show=False) does not -- change API?
- Font scan skipped xlabel/ylabel/zlabel -> tofu for axis-label-only scripts. FIXED 2503a543 (+ test_axis_labels_join_the_font_gap_scan).

### Data/predict/io reviewer (repro scripts: scratchpad/review_data/p*.py)
- MAJOR predict/time.py:41,116 business-day/monthly data treated as irregular -> interpolated weekends, PeriodIndex->DatetimeIndex, DST dup
- MAJOR regression vs 1.0 predict/common.py:533,time.py:76-83 reuse fitted forecaster across index kinds crashes
- MAJOR io/sources.py:1264 yahoo interval='1h' bars all at midnight -> duplicate index
- MINOR arima.py:224 min-history ignores seasonal_order; tools/normalize.py:43 fitted Normalizer rejects its own 1-D input (also master);
  lazy_import.py:244 stale set_autoinstall handle; sources.py:1722 misleading offline Drive error
- NIT sources.py:1786, time.py triple warnings, npz parquet fallback, scikit-image floor py3.13, load() TypeError text omits polars
- Clean: pickle trust (13 payloads), dataset pins, lazy_import input safety, datatype round-trips, backtest

### Plotly/colors/forecast reviewer (repros: scratchpad/review_plotly/r*.py)
- MAJOR plotly fmt='o-' markers on every smoothed vertex (945 vs 60) (r05,r06); continuous hue + 'o-' in 1-D/2-D plotly draws no markers
- MAJOR cyan shift remains when line/marker alpha differ (hue+alpha+'o-' 3-D) (r13)
- MAJOR ax= palette continuation re-samples -> repeated colors ('hls' 2+2: a2==b1) plot.py:11252/11392 (r14,r37)
- MAJOR plotly panels/subplots: every cell counts as having a legend -> colorbar pushed onto next cell (r26,r27)
- MAJOR 1-D data w/o ndims=1: x in upsampled units but forecast/truth raw steps (24x squash); axis_scale='data' uses y range for x (r31-r34)
- MAJOR forecast_fmt='ro:' marks all smoothed vertices (both backends) (r09)
- MAJOR (plot-core) plot.py:11725 2-col data + animate + predict crashes unpack (r40)
- MINOR second draw into plotly cell deletes title (r15); colors.py:410 0-255 RGB list treated as matrix (r22);
  2-D into 3-D cell cryptic TypeError (r28); ax= bundle colors mismatch (r36); 1-D animation plotly silent vs mpl error (r38)
- NIT plotly_backend.py:2331 mutates caller's traces
### Plot-core reviewer (repros: scratchpad/review_plotcore/r*.py)
- MAJOR F1 forecast/truth overlay style indexing w/ marker+line fmt (also master); F2 dict palettes ignored with markers-only fmt;
  F3 ndims=1 2-col truth read as (x,value)
- MINOR F4 legend_colors+predict "legend has 3" (regression); F5 panels+forecast_trail; F6 panels legend_colors split; F7 transform DF index->NaN->zero fc;
  F8 xlim=(None,date) crash; F9 datetime t= per-dataset steps; F10 truth legend key colour; F11 serial {index} title (DELIBERATE 1.1 -> ask Jeremy);
  F12 0-255 palette (same as plotly r22); F13 ndims=1 date ticks overlap; F14 shuffled index scribble; F15 bare-array transform IndexError; F16 2-col into 2-D ax=; F17 labels arrays
- also: font='Noto Sans' fails in fresh process (verified by me)

### Fix wave (worktrees, launched ~16:30Z)
- W1 predict (D1,D2,D4,D9,F9); W2 io/tools/_shared/colors/fonts (D3,D5-D8,D10-D12, 0-255 palette, Noto Sans);
  W3 plotly_backend (Y1,Y2,Y3,Y5,Y7-plotly,Y8,Y10,r38,NIT,frame_kwargs/zoom/yanchor/'^');
  W4a forecast/truth (F1,F3,Y6,r40,Y7-mpl,F4,F5,F7,F8,F10,F14,F15,F13); W4b palettes/markers/panels (F2,Y4,Y11,F6,F16,F17,marker override, markers= smoothed)
- Shared marker contract: markers only at true observations on both backends incl. forecast_fmt.
- After merge: integrate CHANGELOG bullets, full pytest, ruff, docs build, re-run tour headless + Colab at new pushed SHA.

### Docs/claims reviewer (scratchpad/review_docs/FINDINGS.md, 1555 lines)
- BLOCKER H-B1 5 launch notebooks crash on Colab (plotly auto; anim.on_frame/figure/draw_frame) -> W5
- BLOCKER E-B1 hyp.plot([a,b], hue=['x','y'], labels=['A','B']) crash (per-dataset labels + hue/cluster) -> W4b
- BLOCKER H-B2 market_sectors weights: split-adjusted close x as-reported SEC shares -> W5
- MAJOR A0 Colab video block swallows 6 tutorial cell outputs -> W5; A1 RTD webhook 400, nothing built since 07-24,
  README latest/optional_dependencies.html 404 -> checklist (W6) + JEREMY must re-sync RTD integration
- MAJOR A2 release notes stale, A3 checklist order -> W6; C-F1 hue markers ignore alpha -> W4b+W3; E-M1 marker-only regroup fc -> W4a
- 10 tutorial prose MAJORs + many MINORs -> W5; CHANGELOG/checklist MINOR/NIT -> W6; D-1 plotly window_bounds, D-6 ax=cell -> W3;
  D-2 aligner fit arrays, D-3 dispersion, D-5 PPCA warnings -> W7; A12 set_autoinstall teardown -> W2
- Worktree tool sometimes bases on 96ac8b7f (origin default = master)! Prompts now force reset to 09f56b2c/deeb1f93.

### Jeremy report 2026-09-11 ~16:10Z: plotly hover says "trace 0/1" instead of legend names
- Confirmed: nearly every hoverable plotly data trace is unnamed (no-legend plots, panels, animations, forecasts, series curves);
  with a legend, only the first segment per hue/cluster/hierarchy group is named (ANIM-dict-cluster, HIER). Continuous hue drops legend names.
  Animation playback keeps base names (verified in Chromium via gd._fullData after Plotly.animate).
- Routed to W3 with contract: name every hoverable trace with its legend label (category/dataset/column/model/'truth'),
  duplicates showlegend=False + same legendgroup; legend display still governed by legend=; single unlabeled dataset hides the extra box.

### Visual QA adjudicator (scratchpad/visual_qa_adj/VERDICTS.md): OK 26, KNOWN 7, lib 39 rows (15 bugs), tour 12 rows (9), d 11
- L1 plotly 3-D line width ~0.53x (-> W8 later); L2 3-D density volume stipples cube (-> W8); L3 white label connectors, L4 panel/ax titles DejaVu,
  L5 nested legend colours, L6 cluster legend order (-> W4b); L7 plotly animated legend flicker (-> W3); L8 anim DOWNSAMPLING (helix 46% radius),
  L9 morph no motion + dot size, L11 on_frame title off-canvas, L12 companion blue, L13b stream clamp warning (-> W9 new);
  L10 plotly date axes shifted by viewer TZ (-> W4a); L13 pipeline cluster step dropped (-> W7); L14 docstrings (me, after merge); L15 zoom (W3)
- DESIGN-QUESTION for Jeremy: 'unit' affine puts column means off-centre (data sit in upper half of cube) - centring per-axis midrange would change all figures.
- Tour T1-T9 exact replacements in VERDICTS.md lines 287-316 (me, after merge). d: plotly Play/Pause over date ticks (-> W8).

### Merges
- W6 merged 0d4f0cab (22ae49c7 CHANGELOG, 05f7aeb6 checklist + stale-date release gate, e876a556 release notes). D-12 partial.
  TODO after W4a/W4b/W9 merge, apply plot.py docstring fixes: :5333-5334 axis_scale '(-1.1,1.1)' -> data in [-1,1], frame half-width
  1.125, axes pinned (-1.2375,1.2375); :7356 '[-1.1,1.1] frame box' -> '+/-1.2375 around the unit frame'; :5992-5993 slow-warning timing
  -> 'emitted once fits at two or more history lengths (one of >=10 rows, or the longest available) have been timed'; :1124,:1133
  'title must be a string (or None)' -> 'a string, a callable (ctx -> str), or None'. Also animation_context.py:207-209 window_bounds doc (L14).
- W2 merged 70dc00b3 (9 commits: yahoo intraday tz, Normalizer 1-D, autoinstall teardown + re-entry order, offline errors, npz trust msg,
  scikit-image>=0.25.0 (+docs/doc_requirements.txt), load TypeError polars, 0-255 palette -> ValueError, font='Noto Sans' fresh process).
  DESIGN-QUESTION (Jeremy): set_autoinstall — does ENTERING an older handle count as a new call? Current rule: no (newest call by creation order).
  FOLLOW-UPS: pip install -e .[dev] after all merges (metadata floor); io.ipynb re-exec; core floors (numpy 2.0.0, pandas 2.2.2, scipy 1.13.0,
  matplotlib 3.9.0, sklearn 1.4.2, statsmodels 0.14.0) have no cp313 wheels, pillow 8 none for >=3.10 -> packaging decision for Jeremy.
  CHANGELOG bullets drafted in W2 report (integrate at end).
- Jeremy report ~16:40Z "surface colour doesn't match dots" (plot popped up in browser from agent test runs): ROOT CAUSE global IDW in
  meshutil.vertex_colors_from_points -> washed-out mean colour. FIXED a422a97e (8-NN IDW) + tests; tests/conftest.py PLOTLY_RENDERER=json (headless).
- W5 merged 5c09e1bb (launch/examples pinned matplotlib, market_sectors split-adjusted shares, A0 video block, prose). Re-exec needed:
  market_sectors, weather_decades, painting_embeddings, conversation_shape, morph_shapes_zoo, animate_forecast (regenerated, no outputs),
  io, manip, plot, align, analyze, cluster, pipelines, projectile_kalman, stock_forecasting, streaming_data, lsl_streaming, text.
  Regenerate market_sectors.mp4, sphx_glr_animate_market_sectors_thumb.gif, Bluesky 20_market_sectors clip, plot_sotus render.
  W5 library leftovers (-> W10): flat cluster spec {'model':'KMeans','n_clusters':4,'random_state':0} drops random_state (cluster.py ~111-123);
  names= overrides explicit legend=False. Unverifiable market data: HON 2026-06 SEC count half of 2026-03; XOM SEC history starts 2026.
- W1 merged 46c71700 (calendar-regular forecasting, cross-index reuse, ARIMA seasonal floor, warnings once). DESIGN-QUESTION (Jeremy):
  flat-list datetime t= resolving to different step counts per dataset: code takes max(steps) (2e9669df) + test asserts differing lengths,
  docstrings (plot.py:5586, :6836) say must be equal. Follow-ups: stock_forecasting.ipynb cells 11-12 prose (median gap/weekend interp)
  stale; CHANGELOG 17-21 'one future step is always the median gap' stale.

- W4a merged 69c1ff7d; W7 merged 785dca57 (+ my 2deac692: datatype-gate fix + polars transform= SchemaError fix); W10 merged 83b99d6a.
- Full suite at ~69c1ff7d (fullsuite1.log): 6115 passed, 12 failed = only the 6 regenerated launch notebooks' execution gates (expected until re-exec).
- W11 launched (flat spec keys in reduce/manip/align/impute/Pipeline). W3 batch 2 queued: plotly animated continuous hue (3-D colours don't
  travel with window; 2-D segments static), L1 line width, L2 volume stipple, play/pause over date ticks.
- CHANGELOG drafts accumulating in scratchpad/changelog_drafts.md (W1, W2, W4a, W7, W10, main).

- W3 batch1 merged 4d6dd7ce; W4b merged 57553242 (I resolved 2 plotly_backend conflicts: signature + before_show hook, W3's relocated
  alpha normalization kept); W11 merged cc418fa9; docstring fixes 4669ddba; CHANGELOG integration 27d46868 (43 bullets + 2 corrected claims).
- Full suite 2 at 57553242: 6399 passed, 13 failed = 12 launch-notebook gates + 1 LSL collision with an agent's concurrent run (passes alone).
- Remaining: W9 (animation core) + W3 batch 2 (animated hue, L1, L2, play/pause) -> then CHANGELOG bullets for them, animation_context.py
  window_bounds doc (L14), then scratchpad/verify_pipeline.sh (tutorial re-exec, pytest, sphinx -W html+doctest, thumbs, smoke),
  commit notebooks, set tour REVIEW_COMMIT, headless tour, push, CI, Colab re-run, refresh evidence packet.
- NIT to sweep: predict/plot warnings attributed to library lines (e.g. plot.py:9383 'dataset index is not sorted'). -> FIXED 53c7673a/461d2e33.
- W3 batch 2 merged 0ed1e8f3; W9 merged db017681; CHANGELOG b6308028.
- VERIFY PIPELINE at b6308028 (18:17-19:32Z): 25/25 tutorials re-executed, no failures; pytest 6546 passed/21 skipped/0 failed;
  sphinx html -W 0 warnings (51 gallery examples); doctest 323/0; thumbs regenerated; example smoke 6/6.
- Then: warning attribution + step text fix (merge 25da7cc0, 461d2e33), stock_forecasting prose, projectile/stock re-exec,
  tutorials+thumbs commit b1f857cc. ruff clean. Next: push, CI, tour REVIEW_COMMIT=HEAD headless run, Colab run, evidence packet.
- Pushed e684d7cd. Local full suite there: 6551 passed/21 skipped/0 failed. Headless tour 241/3/0. FRESH COLAB (Pro, via Playwright,
  https://colab.research.google.com/drive/1sFcFvxMUzNSjFnYFBjHlq6T8Twqqi-GD): 241 PASS / 3 SKIP / 0 FAIL, SHA confirmed; frontend checks:
  early previews survive, viewer open/switch/close(0 graph divs)/reopen, playback + stable legend, hover names = legend (gd._fullData),
  PLOT-style-plotly swatches match, date axis TZ-correct. Found: viewer displayed twice + IPython IFrame warning -> fixed e9db5204.
- CI 34639769259 at e684d7cd: 15/16 success (incl. docs-clean, all Windows/macOS), ubuntu-3.12 coverage step still running at 21:18Z.
- Review packet (untracked, like the previous one): notes/final_review_2026-09-11/START_HERE.md (9 decisions for Jeremy).
- Worktrees from this session removed (branches kept, all merged).
- Pushed e9db5204 (tour tooling: single viewer, no IFrame warning; DOC_GATE_OVERRIDE recorded). Headless tour at exact e9db5204:
  241/3/0. FRESH COLAB at e9db5204 (https://colab.research.google.com/drive/1bmQ-UMgWOmT_576hpkpbnL4S-dw2nyNq): 241 PASS / 3 SKIP, commit seen.
- CI 34639769259 at e684d7cd: COMPLETED SUCCESS 16/16 (+ release-gate skipped); per-job totals in notes/final_review_2026-09-11/ci_status.md.
- CI 34648813020 at e9db5204: COMPLETED SUCCESS 17/17 (16 jobs + release-gate skipped) at ~23:20Z. Final candidate fully verified.

### Visual expectation writer extra (code-vs-doc)
- mpl fmt='-o' overrides marker=['o','s'] (backends disagree); markers= with '-' marks all smoothed pts (antialias docstring says true samples);
  plotly ignores frame_kwargs; static plotly applies zoom (doc: animation only); '^' -> diamond in plotly 3-D; legend_kwargs x/y yanchor;
  font='Noto Sans' ValueError in fresh process before bundled fonts registered

### After the candidate: animated forecasts hung back from the drawn head (Jeremy's report, 2026-09-11)
- Report: "for animations with predictions, the predictions should show *from the endpoint in the current frame* not just from
  the last observation."
- Confirmed on BOTH backends. An animation is paced on `_interp_anim_line`'s refined grid, so the drawn head usually sits
  BETWEEN raw observations; `ForecastSchedule.anchor` floors to the last observation at or before it, and `polyline` started
  there. Measured (2-D spiral, display box [-1, 1]): 20 rows/40 frames max gap 0.21, 24/37 frames affected, 2-frame stalls;
  12 rows/90 frames 0.51, 71/81, 8-frame stalls; 8 rows/160 frames 0.90 (45% of the box), 130/137, 23-frame stalls.
  The forecast stood still while the head kept moving, then jumped -- exactly what the report describes.
- Fix: the schedule now carries the DRAWN head. `for_parallel`/`for_serial` read `grids=` (the animation grid arrays plot.py
  already has -- not a second interpolation) at the same `end`/`shown` they already compute; `grid_head()` returns the vertex
  and its fractional raw-row position; `polyline()` replaces vertex 0 with it; `to_display` maps heads with the full affine
  (head POSITIONS are row indices, carried through untouched). Predicted points are untouched, so `t=`, `pin_ramp`'s exact x
  ramp and the vertex count are unchanged. Regrouped (hue=/cluster=) reveals pass no grid and keep the raw-row anchor.
- Covered on 1-D/series, 2-D and 3-D, both backends, plus the `forecast_trail=` fan (each retained forecast starts at the head
  of the frame it was fit at). New test: tests/test_animated_forecast_anchor.py (fails on the old code at frame 3).
- Docs: animation.rst's "re-anchored on the last revealed observation" + "joins ... to within one raw observation rather than
  exactly" were describing the defect as the contract; rewritten, with a versionchanged note. CHANGELOG bullet added.
- NOTE: this moves the branch head off the verified candidate e9db5204 -- CI, the headless tour and the Colab run all need
  re-running before sign-off, and notes/final_review_2026-09-11/ needs its evidence table refreshed.
- STALE ARTIFACTS from the fix: examples/animate_forecast.py animates 60 observations over 180 frames (stride 4, 237 grid
  rows), so its old forecast trailed the drawn head by up to 0.40 = 10.7% of the data diagonal on 134/180 frames. Both
  committed clips therefore show the OLD geometry and must be regenerated: docs/_static/thumbnails/
  sphx_glr_animate_forecast_thumb.gif (sphinx gallery build -> scripts/generate_gallery_thumbs.py) and
  docs/tutorials/animate_forecast.mp4 (scripts/execute_tutorial.py docs/tutorials/animate_forecast.ipynb; the climate
  archive IS cached -- 3 x (60, 6) real regions -- so re-execution will not silently fall back to synthetic data).
  It is the ONLY example combining predict= with animate=; the feature tour has no animated-forecast case.
  generate_gallery_thumbs.py never listed this stem (fixed 9f26b995), so nothing regenerated that thumb.
- TOOLING TRAP, fired TWICE today (existing note n410812797x266): `pytest ... > log 2>&1; tail log` and
  `pytest ...; echo "exit: $?"` both exit with the LAST command's status, so the harness reports exit 0 over a failed or
  never-run suite. One run had also invented a test filename (tests/test_plot_forecast_trail.py does not exist), pytest
  errored "no tests ran", and the notification still said exit 0. Read the summary LINE, never the exit code, or glob the
  paths instead of typing them.
