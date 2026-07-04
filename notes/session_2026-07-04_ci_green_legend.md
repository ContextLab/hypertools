# Session 2026-07-04 (round 8): CI-green + legend-clipping + bug hunt

Branch `dev-2.0-refactor`; PR **#271** into `dev-2.0` (NOT master; do not merge).

## What shipped this session

### Reported animation fixes (earlier in session)
- Animated bounding box zoomed out (`_anim_box_zoom` 1.25→1.125 + full-canvas `set_position`); regenerated all matplotlib animations + spin.gif.
- Duplicate animation legend (#207): trails tagged `_nolegend_`.

### Open-issue triage (67 issues) → `notes/issues-to-close-on-merge.md`
28 ADDRESSED + 6 OBSOLETE + 6 fixed-here = 40 to close on merge; 27 stay open.
6 bugs fixed (commit bb12cd89): #259, #223, #146/#190, #148, #214, reduce.py custom-estimator UnboundLocalError.

### CI-green fixes (this round; commits c4abaee1, 4b5048ec, 9433f996)
First CI run on the pushed branch: Windows all-fail (collection), macOS/Ubuntu 3.11+ 3-fail, 3.10 pass.

1. **Windows (data-wrangler#32 filed).** dw 0.5 `core/config.ini`: `homedir = os.getenv('HOME')`; `datadir = os.path.join(%(homedir)s, ...)`. HOME unset on Windows → `os.path.join(None,...)` TypeError at dw import → hypertools import fails → all Windows tests error at collection. **Fix:** `os.environ.setdefault('HOME', os.path.expanduser('~'))` before `import datawrangler` in `core/configurator.py`. Validated locally via `env -u HOME`. Lift when dw#32 lands.

2. **matplotlib 3.11 (Python 3.11+ only; 3.11 dropped py3.10).** `plt.close(fig)` (the #148 show=False fix) RESETS the figure canvas to `FigureCanvasBase` on mpl 3.11 → subsequent `fig.canvas.renderer` / `buffer_rgba()` raise AttributeError. mpl 3.10 (py3.10 jobs) keeps the Agg canvas → passed. **savefig-after-close still works on 3.11**, so users unaffected. **Fixes:** guard renderer in `update_position` (skip reposition if absent); render `test_spin_box_never_clipped` + the #223 tests via an explicit `FigureCanvasAgg(fig)`.

3. **Legend clipping (Jeremy: "many gallery legends cut off").** Wide legends (long labels / many entries) clipped the right edge. **Root cause:** hypertools draws inside a seaborn `rc_context` (plot.py:762) whose font is NARROWER than the DEFAULT font the figure is actually saved under downstream (sphinx-gallery scraper / bare savefig after plot() returns). `_fit_right_legend` measured under seaborn font → thought it fit → clipped in the real output. Also `get_tightbbox`/`get_window_extent` under-report legend text by ~0.15in, and `tight_layout` installs a persistent layout engine that re-runs on save. **Fix:** `_fit_right_legend` now (a) `set_layout_engine('none')`, (b) measures ACTUAL rasterized pixels via a fresh Agg canvas UNDER `matplotlib.rcParamsDefault`, (c) WIDENS the figure (keeping the plot's absolute size/position) until the legend has a pixel margin — instead of shrinking the axes to a floor. Regenerated plot_legend/plot_PPCA/plot_missing_data. Pixel-based regression test `test_legend_not_clipped_in_saved_pixels` (fails on old code).

## Test / CI status
- Local: `341 passed, 6 deselected` (the 6 plotly→kaleido image-export tests hang Chromium in THIS sandbox only — kaleido worked earlier same-session then wedged after chrome pkills; CI runs all 347 fine).
- **GOTCHA:** never `pkill` kaleido/chrome mid-run — it wedges kaleido's pool for the rest of the session. Deselect the 6 kaleido tests locally (`test_animation_export::test_plotly_{gif,spin_gif,mp4,spin_gif_preserves}`, `test_round3::test_{static,animated}_svg_plotly`).

## Follow-ups
- Lift the HOME workaround once data-wrangler#32 is released.
- animate_plotly gallery example keeps its cached artifact (900-frame kaleido gif too slow locally; plotly auto-fits so it never clips).

## Round 2 (same session): remaining CI reds after first fix wave

First fix wave result: macOS 3.10-3.13 ALL green, Ubuntu green except 3.12, Windows runs full suite (HOME fix worked) but fails 1 test.

4. **Windows `test_spin_box_never_clipped`** (`'NoneType' object has no attribute 'start'/'add_callback'`): on Windows the animate backend switch to TkAgg SUCCEEDS (headless Linux/mac fall back to Agg), so the #148 `plt.close()` destroyed the FuncAnimation's real Tk timer; the animation's pending first-draw hook then crashes any later draw of the returned figure. **Fix (fb1cb031):** animated figures (`line_ani is not None`) are EXEMPT from the show=False close — #148's complaint was about static figures; animations need their timer alive.
5. **Ubuntu 3.12 screenshot step** (12/13 "plot_fn produced no matplotlib figures"): `scripts/screenshot_harness.py` discovered figures via `plt.get_fignums()`, emptied by the #148 close. **Fix:** `capture()` prefers the RETURNED figure(s) (`_extract_mpl_figs`: Figure / `(fig, ani)` / return_model dict), falling back to the registry. Verified 13/13 locally.
6. **dw 0.5.1 released** (Jeremy): fixes dw#32 via `os.path.expanduser('~')`; verified import-with-HOME-unset OK. Pins bumped to `pydata-wrangler>=0.5.1` (base + [text]) in 80470fd0; HOME guard kept for 0.5.0 environments. dw#32 closed with verification comment.

Local suite after round 2: **343 passed, 6 deselected** (new legend-pixel regression tests added the +2).
