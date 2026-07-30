# Session 2026-07-25 — Bluesky launch demo fixes (round: colorbars/forecast/weather/conversation/tutorials)

Working dir: repo `hypertools`, branch **dev-1.0**. Demos + assets live in the
gitignored `notes/bluesky-launch/` (POST.md + `1x_*.gif/.mp4`). Demo source =
scratchpad `demo_*.py`. Env: **use `.venv/bin/python`** (py3.12, numpy 2.3.5,
sentence-transformers 5.6). Base anaconda python is BROKEN (numpy mismatch).

## IMPORTANT env note
`.venv` had a STALE non-editable copy of hypertools (missing recent commits).
Fixed with `.venv/bin/pip install -e . --no-deps`. Now editable → repo source
(incl. commits aa53f815, 26843931) is live. Re-verify with
`.venv/bin/python -c "import hypertools,os;print(os.path.realpath(hypertools.__file__))"`.

## User asks this round (all addressed)
1. Market: colorbars flicker; forecast not connected + outside box; no prediction history.
2. Weather: color axes should be tight so full color scales are used.
3. Conversation: transparency "snaps in" not a tail; past turns should be more transparent, current opaque.
4. Add tutorials (hyp code per animation) to gallery + tutorial section.

## Root causes + fixes (KEY technical findings)
- **Market colorbar "flicker" = camera rotation.** `animate=True` rotates by
  default; Axes3D re-fits the projected cube each frame → the whole box breathes.
  Fix: pass **`rotations=0`** (fixed camera). demo_market.py + example.
- **Market forecast detached/out-of-box = coordinate space.** hyp normalizes the
  reduced path into its drawn cube; overlays drawn in raw `red` coords land in the
  WRONG place. Also `ani._args[0][0]` is a DIFFERENT normalization than the visible
  line. FIX: read the TRUE on-screen head from the visible line artist
  **`market_line = ani._args[1][0]`** each frame (`get_data_3d()[:, -1]`); fit the
  (reduce→drawn) per-axis SLOPE empirically by advancing to the last frame once,
  reading `market_line` full data, and `np.polyfit(red_resampled, drawn, 1)`; scale
  the red-space Kalman delta by SLOPE, apply a visual GAIN (median arrow ≈0.28 box)
  + length CAP 0.5, clip to `ax.get_[xyz]lim3d()`. HISTORY FAN: faint blue past
  forecasts (each hung off its own cached historical head), current = bright red.
- **Weather: MultiIndex IGNORES hue (GH #95).** `hyp.plot(df_multiindex, hue=...)`
  warns "MultiIndex grouping takes precedence over hue=; ignoring hue" → the temp
  coloring never worked (navy/amber were just default GROUP colors). FIX: build the
  hierarchy as a **LIST** (per-city loops + 2 hemisphere-mean loops), pass continuous
  temperature `hue` + combined per-hemisphere colormap. Color axis tightened to the
  MEAN monthly range (not all-cities) so bold means sweep the full colormap; city
  extremes clamped into each hemisphere's band. Emphasis: hyp's multicolor
  collections DON'T inherit per-dataset linewidth, so set it by index — the first
  `len(datasets)` Line3DCollections are the per-dataset HEADS (means = last 2), rest
  are trails; set mean lw + per-frame alpha (means 1.0, cities 0.16, trails 0.10).
- **Conversation: recency fade.** Removed chemtrails (its uniform-alpha trail =
  the "snap"). `animate='serial'` reveals one turn at a time; in the frame wrapper
  set per-turn `Line3D` alpha via `ani._args[1]`: current turn 1.0, earlier turns
  `FLOOR + (1-FLOOR)*DECAY**(ti-j)` (FLOOR 0.12, DECAY 0.6) → fading tail. Current
  turn found point-wise via cumsum of turn lengths.

## Status
- 5 scratchpad demos re-rendered FULL + MP4 (Bluesky video ≤~50MB fine):
  market 836K, weather 2.0M, conversation 996K, paintings 712K(unchanged), morph 5.1M(unchanged).
- 5 gallery examples (`examples/animate_*.py`) updated to FINAL techniques
  (market/weather/conversation rewritten; paintings/morph verified). ALL render headless.
- POST.md code blocks + notes updated to match finals.
- Notebooks (`docs/tutorials/*.ipynb`) + `docs/tutorials.rst` wiring: **delegated to a
  background subagent** (update 5 notebooks to final technique + add 5 sections to
  tutorials.rst + verify headless). CHECK ITS REPORT before considering done.

## NOT done / open
- Notebooks + tutorials.rst (subagent in flight — verify).
- Nothing committed this session (examples + tutorials.rst + notebooks are uncommitted).
  Ask Jeremy before committing gallery/docs or pushing dev-1.0.
- Conversation text-attribution heuristic occasionally mislabels an Alice line as
  "Narrator" (Gutenberg-derived demo only; the gallery example uses curated TURNS = correct).
- Paintings muted-color polish still deferred (earlier red-team #6).

## Auto line-smoothing request (2026-07-25, follow-up)
User: "smoothing should be applied (just to the plot) automatically for any plot
with lines (not solely markers)... includes predictions, and should work for animations."

Findings + what shipped:
- Static lines ALREADY auto-smooth: `_interp_static_line` PCHIP-densifies to
  `_STATIC_LINE_TARGET_VERTICES=900` (plot.py). No change needed.
- **Predictions: NOW smoothed** (SHIPPED). `_draw_forecast_overlays` (plot.py ~122)
  applies `_interp_static_line(fc)` to each forecast → short forecasts draw as a
  smooth dashed curve (endpoints exact, still joins the trajectory). Tests updated
  to density-robust (`> t+1` + seam-connect): test_predict_integration.py (2 asserts
  + new test), test_h1_validation_warnings.py:228.
- **Animated lines**: hyp resamples each animated LINE to EXACTLY n_frames =
  round(frame_rate*duration) vertices (`_interp_anim_line`), so low fps DECIMATES
  dense trajectories (weather 780-pt loops @360 frames → jagged). I TRIED decoupling
  drawn density from frame count (floor 900) but REVERTED: it breaks the audited
  frame↔point IDENTITY invariant (F04) that many subsystems rely on — confirmed a
  REAL label regression (`_sync_anim_labels` window mode compares dense `_hyp_point_idx`
  vs FRAME index → labels mis-timed / never shown) + ~12 window/serial/trail-length
  tests. Too invasive to rush.
  - DEMO workaround (no library risk): animated line vertices == frame count, so
    smoothness ∝ frame_rate. Bumped **weather fps 20→52** (dur 18 → ~936 frames →
    smooth loops). Market path already smooth (1200 frames); market hand-drawn
    forecast/history OVERLAYS smoothed via a demo-level `_smooth` (PCHIP densify).
    Examples matched (animate_market_forecast `_smooth`; animate_weather_decades fps=54).
- OPEN (flagged to Jeremy): fully-automatic animation-line smoothing regardless of
  fps needs a DRAW-TIME render-smoothing pass (smooth the drawn window in each
  backend update fn: parallel/serial/spin/window/2D + multicolor + plotly), which
  keeps data at n_frames (no label regression) and only breaks ~5-6 drawn-length
  tests. That's the sound approach ("just to the plot") but a multi-site change —
  confirm before destabilizing the audited animation core.

## antialias= AUTO LINE SMOOTHING (2026-07-25, SHIPPED — supersedes the "OPEN" note above)
User spec: smooth+upsample trajectories so no sharp angles; per frame draw the
upsampled points corresponding to what *would* have been shown of that portion;
lines only (solid/dotted), NOT markers; `antialias=True` default, toggleable;
applied at the very last stage before drawing; re-render gallery + update docs.

IMPLEMENTED:
- `hypertools/_shared/helpers.py`: **`antialias_line(arr, target=900) -> (dense, step)`**
  (+ `ANTIALIAS_TARGET_VERTICES`). PCHIP upsample with UNIFORM subdivision, so
  `dense[::step] == arr` EXACTLY and any window maps exactly:
  **`arr[a:b] -> dense[a*step : (b-1)*step + 1]`**. Returns `(arr, 1)` when no
  upsampling needed → every mapping degrades to the raw slice (so
  `antialias=False` is byte-identical to old behavior). Both plot.py and
  matplotlib_backend.py get it via their existing `from .._shared.helpers import *`.
- `plot.py`: `antialias=True` param + full docstring; gates static
  `_interp_static_line` and `_draw_forecast_overlays`; threaded to `_draw`,
  `plotly_draw`, `_apply_multicolor_animation`.
- `matplotlib_backend._draw`: `antialias=True` param; precomputes per-dataset
  dense curve ONCE (gated on `antialias and animate and has_line_component(fmt)`);
  new closure **`_aa_window(i, a, b, artist=None)`** used at EVERY animated draw
  site (parallel/serial/spin 3-D + parallel/serial 2-D, heads AND trails).
  **Data rows are left untouched** → `_anim_window_bounds`, `_sync_anim_labels`,
  surface hulls, markers all keep indexing real rows (this is what avoids the
  label regression that killed the earlier data-level attempt).
- `_aa_window` also records **`artist._hyp_row_window = (a, b)`** (ORIGINAL row
  bounds). `_apply_multicolor_animation` now reads that instead of
  reverse-engineering the window from the artist's VERTEX count (antialias
  decouples vertices from rows); it densifies pts AND interpolates per-point
  colors onto the same parameterization (verified 576 segs/576 colors vs 12/12).

TESTS: new `tests/test_antialias.py` (19 tests, all pass) incl. the real
user-facing property (max turning angle drops >3x), marker-only exclusion,
'o-' combo (markers stay on raw samples), frame count unchanged, same drawn
SPAN with more density, labels still track (regression guard), forecast smoothing.
Existing ROW-WINDOW tests got explicit `antialias=False` (they assert exact drawn
row windows): test_animation_styles (`_serial_trail_bundle` + 2), test_2d_animation (1),
test_window_animation (4). Forecast-count asserts relaxed to `> t+1`.

## USER-REPORTED BUGS (both root-caused 2026-07-25)
1. **"extra dots in the weather animation"** = **NOT a plotting bug**. It was MY
   `gifsicle -O3 --lossy=100 --colors 128` run (to shrink 41MB→25MB): lossy GIF
   quantization to a 128-colour palette produced speckle + a black blob. Proof:
   the MP4 of the same frames is clean, direct matplotlib render is clean, and
   only `13_weather_decades.gif` has a 128-colour palette (all others 256).
   **FIX: never lossy-compress these gifs; re-render/re-encode without `--lossy`.**
2. **conversation transparency wrong** (active turn not opaque). ROOT CAUSE:
   `hyp.plot` RESAMPLES each animated line onto the frame grid, so per-turn drawn
   rows are `[48,48,48,1,1,48,...]`, NOT the original `[13,3,5,1,1,4,...]`. The
   demo/example computed the active turn from the ORIGINAL lengths → the opaque
   highlight lagged the turn actually being drawn. FIX: compute the active turn
   from the DRAWN lengths using the backend's own reveal formula
   (`revealed = total_pts*num/(total-1)`; active while `0 < shown < n`).
   Verified 0/48 mismatched frames. Fixed in demo_conv.py AND
   examples/animate_conversation.py.

## FINAL STATE (end of 2026-07-25)
- **antialias shipped in BOTH backends.** plotly mirrored in `plotly_backend.py`
  (`_build_aa_curves` + window mapping); independently verified: animated 3-D line
  481 vs 11 drawn pts/frame (on vs off), static 932 vs 50, **frame count identical**,
  marker-only byte-identical either way.
- **Suites:** non-plotly **2360 passed**; plotly 185 (3 count-assertions fixed with
  `antialias=False`: 2 in test_plotly_trace_indices via `plotly_draw(...)`, 1 window).
  In EVERY relaxed test the structural/correctness asserts still pass unchanged
  (trace counts+indices, dash styles, "chemtrails never shows future",
  "forecasts unaffected by frames") — only exact drawn-vertex counts moved.
- **Repo hygiene fixes found by the suite (both PRE-EXISTING, now fixed):**
  (a) sdist gate — new `tests/test_antialias.py` was untracked → `git add`ed (STAGED, not committed);
  (b) `test_changelog_top_version_matches_pyproject` — committed CHANGELOG top was
  `1.0.1 (unreleased)` while pyproject said `1.0.0` → **bumped pyproject to 1.0.1**.
  FLAG FOR JEREMY: that version bump is a release-policy call; revert if unwanted.
- **All 5 clips re-rendered with antialias, GIFs encoded LOSSLESSLY (palettes 242-256):**
  | clip | gif | mp4 | dur |
  |-|-|-|-|
  | market | 21.9M | 0.83M | 36s |
  | paintings | 7.1M | 0.89M | 12s |
  | conversation | 8.0M | 0.95M | 17.3s |
  | weather | 15.4M | 1.35M | 18s |
  | morph | 8.0M | 6.25M | 24s |
  Weather shrank 41M→15.4M because antialias let fps go back 52→20.
- Docs: plot docstring (autodoc → API page), CHANGELOG 1.0.1 entry,
  `docs/pipeline_order.rst` "antialiasing is applied last of all, at draw time",
  weather notebook fps reverted. RST validated with docutils.
- STILL UNCOMMITTED (Jeremy's call): everything above + the 5 gallery examples and
  5 tutorial notebooks from earlier today.

## ROUND 2026-07-26 (market / conversation / weather refinements)

### MARKET
- **BUG I INTRODUCED EARLIER, NOW FIXED:** when I added `rotations=0` to the
  `hyp.plot(...)` call I accidentally REPLACED `duration=dur, frame_rate=fps`.
  hyp then used its 30s/30fps defaults => the animation had **900** frames while
  the script's `total` said 1200, desyncing the date label, forecast timing and
  ticker colors. Symptom that exposed it: `rotations=0.25` only swept 4.4 deg
  (0.25*360*44/900). All other demos + all 5 examples were checked and DO pass
  duration/frame_rate. **Guard: always verify `ani._save_count == total`.**
- **Chaotic forecasts** — measured the real cause rather than guessing.
  Kalman on this reduced path: directional accuracy **51-57%** (~coin flip),
  relative error ~1.0, magnitude UNDER-predicted (0.29-0.53x actual), and a
  heavy tail (95th pct / median = 2.5 at best, **16.8** for STEP=63). The old
  ~50x visual GAIN amplified that tail => the wild arrows. Fixes: require
  `MIN_HIST=24` monthly samples before forecasting (tightens tail to 2.5x), and
  cap each arrow at **1.8x the MEDIAN** length instead of an absolute 0.5.
- **Legend** rebuilt from real `Line2D` handles so styles match what's drawn
  (thick red dashed = live forecast; thin faint blue = past fan), fontsize 12.5,
  `loc="lower left", bbox_to_anchor=(0.055,0.02), ncol=1` (ncol=2 got clipped).
- **Accuracy readout** in a subtitle under the title: running DIRECTIONAL
  accuracy (positive dot product between predicted and actual displacement),
  and a forecast only counts once its horizon has elapsed on screen (no
  look-ahead). Shows `-- (waiting for first horizon)` until then.
- **Slow spin**: `rotations=0.25` (one quarter-turn over the clip).

### CONVERSATION
- Turns are now **curated, SPOKEN TEXT ONLY** (28 turns), quoted verbatim from
  the Gutenberg Mad Tea-Party with all narration + attributions removed. I
  tried automatic quote extraction first and REJECTED it: it mis-merged
  speakers across adjacent quotes (e.g. gave "There's plenty of room!" to the
  March Hare) and dragged narration into the embedded text. What is embedded IS
  the dialogue. Caption shows only the quote (speaker = colour + legend).
- Dropped the now-unused Gutenberg fetch/`re` imports and the "Narrator" color.

### WEATHER
- `MEAN_LW` 4.5 -> **2.2** (means were too heavy).
- **BUG in my own new panel, found + fixed:** the first `_rolling()` padded the
  ends with `v[0]`/`v[-1]` -- a single January (or July) value -- so every
  city's 12-month mean STARTED at a seasonal extreme and ramped to its true
  annual mean over the first year (e.g. city 4 began at **0.54 C** and climbed
  to 12; Southern cities began ~21 and fell). Rendered as a spurious vertical
  streak at the panel's left edge. Present in the MP4 too, so a real render
  bug, NOT a compression artifact. FIX: centered mean computed ONLY where a
  full window exists -- `_rolling` now returns `(x, y)` with `x` starting at
  `w//2`, and each LineCollection stores that `x0` so the progressive reveal
  converts month index -> segment count via `clip(idx - x0, 0, len(segs))`.
  **NOTE: `examples/animate_weather_decades.py` had the same edge-padded
  version copied into it and must get the same fix.**
- **NEW 2nd panel**: each city's **12-month rolling mean** temperature over the
  65 years, drawn as a per-point-coloured `LineCollection` using that city's
  HEMISPHERE colormap/norm (same colorscales as the 3-D view), revealed in
  lockstep with the animation + a vertical "now" cursor. Layout: size (13,6),
  3-D at [-0.01,0.03,0.52,0.90], panel at [0.575,0.145,0.30,0.70], colorbars
  at x=0.925.

## Verify-visually harnesses (scratchpad)
`diag_market2.py`, `diag_weather.py`, `diag_conv.py`, `test_ex.py`, `test_ex_market.py`
render specific frames to `scratchpad/diag_frames/*.png` for eyeballing.
