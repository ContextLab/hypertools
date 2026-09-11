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

### Visual expectation writer extra (code-vs-doc)
- mpl fmt='-o' overrides marker=['o','s'] (backends disagree); markers= with '-' marks all smoothed pts (antialias docstring says true samples);
  plotly ignores frame_kwargs; static plotly applies zoom (doc: animation only); '^' -> diamond in plotly 3-D; legend_kwargs x/y yanchor;
  font='Noto Sans' ValueError in fresh process before bundled fonts registered
