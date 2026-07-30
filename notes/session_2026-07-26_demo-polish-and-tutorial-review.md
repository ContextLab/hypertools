# Session 2026-07-26 — launch-demo polish, forecast debugging, full tutorial review

Continues `session_2026-07-25_bluesky-demo-fixes.md`. Branch `dev-1.0`. Nothing
committed yet (only `tests/test_antialias.py` staged, required by the sdist gate).

Demo sources live in the scratchpad
(`/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/demo_*.py`);
outputs land in the gitignored `notes/bluesky-launch/`.

## Requests this round

Market: centre title over both panels; diagnose why predictions are poor (via
subagents); forecast and past-forecast colours should match; full company names
in black beside each symbol; drop the grey "arrows amplified" line and centre the
legend on the box. Conversation: kill the stray dots; add a per-frame speaker
label in the speaker's colour; bold the current window inside the caption.
Weather: centre title over both panels; right panel shows multicoloured DAILY
temperatures per city (thin/translucent, mirroring the 3-D city loops) plus
thicker/opaque hemisphere means in the same colours. Plus: red-team the
appearance and the tutorials; re-review ALL tutorials for hand-rolled code that
hypertools already does and for missing newcomer explanations; rewrite POST.md
avoiding AI tells.

## Root causes found (all verified by running code, not asserted)

### 1. Market forecasts: a real anchoring bug in the demo
`hyp.predict(hist, model=..., t=H)` returns `t` rows that are ALL FUTURE steps.
`f[0]` is the FIRST FORECAST STEP, not the last observation. Verified on a
deterministic 3-D ramp (`hist[-1] = [79, -39.5, 19.75]`):

    f            = [[80,-40,20], [81,-40.5,20.25], [82,-41,20.5], [83,-41.5,20.75]]
    f - f[0]     = [[0,0,0], [1,-.5,.25], [2,-1,.5], [3,-1.5,.75]]     <- demo did this
    f - hist[-1] = [[1,-.5,.25], [2,-1,.5], [3,-1.5,.75], [4,-2,1]]    <- correct

The demo's `f - f[0]` discarded one whole step of displacement and forced the
first displacement to zero. FIX: `np.vstack([zeros, f - hist[-1]])` — the
explicit leading zero keeps the drawn arrow attached to the live head while the
arrow now spans the full horizon.

### 2. Market forecasts: over-fitting on short history
Default `lags=5` x 3 dims = 15 free parameters, fit from as few as ~20 monthly
rows early in the clip. Over 293 anchors (corrected delta): worst/median arrow
length 6.3x at default vs 4.5x at `lags=1`.

### 3. Model choice (subagent bake-off, 293 real anchors)
Kalman is a poor fit for this target. Direction correct, corrected delta:
Kalman 46.1% (lags default) / 51.2% (lags=1); Laplace ~65%; ARIMA 53.9%;
Chronos 57.3%; AutoRegressor 43.3%; GP 41.3%. Trivial causal baselines:
persistence 59.7%, mean drift 62.5%, "mean of completed past horizons" 67.0%.
Forecast-then-reduce is WORSE than forecasting the reduced path (Kalman 47.1%).

Honest reading: the reduced market path is close to a random walk with drift
(monthly PC increments have lag-1 autocorrelation of -0.139 / +0.056 / +0.100).
Almost all the achievable skill is the persistent upward drift, not anything
about the current path. With 293 OVERLAPPING horizons the effective n is ~73, so
|z| <= 1 for everything near 50%. Do not over-claim any of these numbers.

### 4. Conversation stray dots: NOT a library bug
`hyp.plot` draws a ONE-ROW dataset as `marker='.', linestyle='None'` (there is
no line through a single point). Curating the turns down to spoken text made 12
of 28 turns six words or fewer, and a fixed 6-word window collapsed each of
those to a single window, hence 12 floating dots. FIX: `word_spans()` shrinks the
window (and step) for short turns so every turn yields >= 2 windows, i.e. a real
path. Verified: min windows per turn is now 3 (2 for the shortest turn).

## Weather right panel
Rebuilt from the 12-month rolling mean to raw DAILY temperature, read back out of
the same cached open-meteo responses (no extra requests): 23,742 days x 12
cities. Cities thin/translucent (lw 0.3, alpha 0.12), hemisphere means opaque
(lw 1.8). Full daily resolution costs ~3.5 min of render across 360 frames,
which is affordable. A white halo under the means was tried and REMOVED: at ~6px
per seasonal cycle the halo fills the band with white instead of outlining a
curve. Known limitation: 65 seasonal cycles in a ~390px panel is a picket-fence
texture; individual cities cannot be followed. That is inherent to the request,
not a bug. Flag to Jeremy if it should become a scrolling recent-years window.

## "Use the built-in" fixes applied to the demos themselves
- `demo_market._smooth()` now calls `hypertools._shared.helpers.antialias_line`,
  the exact routine `plot(antialias=True)` runs, instead of a private PCHIP copy.
- `demo_weather` replaced `np.vstack` + manual z-score + manual per-city slicing
  with `hyp.reduce(mats, reduce='IncrementalPCA', ndims=3, normalize='across')`,
  which stacks, normalizes, fits ONE shared model and splits back per dataset.
  Verified equal to the manual route to 3e-14.
- `demo_conv` dropped its manual `[-1,1]` rescale: `hyp.plot` already centres and
  rescales all datasets with one shared affine (verified, differs by 2e-16).

## Tutorial review (20 notebooks, 4 parallel red-team agents)
Real defects found, beyond missing prose:
- `plot.ipynb` reached into the PRIVATE `hypertools.align.procrustes` instead of
  `plot(align={'model':'Procrustes','kwargs':{'index':1}})`, in a cell whose own
  markdown advertises the public kwarg. Verified equivalent to 4.4e-16.
- `plot.ipynb` `normalize='within'` on a single DataFrame is identical to
  'across', so that cell demonstrated nothing.
- `align.ipynb` used the DEPRECATED `hyp.align(data, align='SRM')` (emits a
  DeprecationWarning); should be `model='SRM'`.
- `lsl_streaming.ipynb` markdown was factually WRONG about `stream_max`: it caps
  samples CONSUMED, not samples retained on screen (that is `stream_window`).
- `wikipedia_embeddings.ipynb` said the Gaussian mixture is fit "to the
  embeddings"; it is fit to the post-UMAP reduced coordinates.
- `hugging_face_embeddings.ipynb` hand-built `legend=category_set` where
  `legend=True` is equivalent (plot derives labels in first-appearance order).
- `conversation_trajectories.ipynb` hand-built a seaborn speaker palette and a
  manual Line2D legend where categorical `hue` + `legend=True` does it.
- `weather_decades.ipynb` / `market_forecast.ipynb` / `painting_embeddings.ipynb`
  / `morph_shapes_zoo.ipynb` carried the same hand-rolled patterns fixed in the
  demos above (manual stack+split, manual rescale, private PCHIP copy, manual
  `rng.choice` downsample redundant with `morph_samples=`).
- `streaming_data.ipynb` and `projectile_kalman.ipynb` came back essentially
  clean.


## Appearance red-team (visual, on the rendered frames)

Two BLOCKERS, both real and both since fixed.

### Conversation frame 0 was a genuine bug in this session's own code
With nothing yet drawn, `current_state(0)` fell through to the "between turns"
branch and returned the LAST window of turn 0, so the very first frame bolded
the end of the line and frame 1 snapped back to the start. Fixed by returning
`(0, 0)` when `done < 0`. Two related fixes landed with it:
- a one-frame speck at every turn boundary (a turn with exactly ONE point drawn
  renders as a lone dot: the same root cause as the original 12 dots, in
  transient form). Turns with `shown < 2` are now hidden.
- the ending had faded to almost nothing (`FLOOR = 0.10`). The whole
  conversation now ramps back up over the final 1.4 s. Measured on the last
  frame: 497 -> 11,206 coloured pixels.
All three are fixed in the demo, `examples/animate_conversation.py` AND
`docs/tutorials/conversation_shape.ipynb`. Note the sync agent had ported the
PRE-fix version, so the example/notebook needed the fix applied separately.

### hypertools' bundled `teapot` dataset is sparse (verify before trusting)
    teapot 1728 rows / 301 unique (0.174)      bunny 35947 / 35947 (1.000)
    cube   30246 / 30034 (0.993)               vase  36022 / 36022 (1.000)
    sphere 30135 / 29891 (0.992)
`morph_samples=N` samples rows without replacement, so the teapot contributes
only ~301 distinct dots and its segment draws sparser than its neighbours
(measured ink ~8,000 vs ~13,500). WORTH A LIBRARY ISSUE; the fix belongs
upstream in the data.

DO NOT DROP THE TEAPOT. It was removed from the zoo in this session and Jeremy
reversed that: keeping a shape in his launch clip is his call, not a defect to
silently fix. Restored in demo_morph.py, examples/animate_morph_zoo.py,
docs/tutorials/morph_shapes_zoo.ipynb and POST.md (back to the bunny-to-teapot
framing). The sparseness is documented in-code as a data note, not used as a
justification for removal. If it should look denser, the option is resampling
the teapot surface up to N points, which has NOT been done because it
manufactures geometry the dataset does not contain.

MORPH TITLES: blank during transitions, shown only on fully-formed frames
(Jeremy's instruction). An earlier version labelled transitions "Bunny -> Cube";
that was reverted. Verified 31% of frames carry no title, which is exactly the
transition segments. `rotations` is
`[0.75] + [0.5, 1.0] * (len(SHAPES) - 1) + [0.5, 0.75]` so the 6 clouds give 11
segments totalling a whole 8.0 turns and the azimuth still wraps at the loop.

### Paintings clip: the stated claim was not true
The side panel advertised colours that appear in ZERO pixels of the plot (the
Scream's orange, the Mona Lisa's gold), because `image_palette()` orders k-means
clusters by SIZE and therefore always picks the muted background tone. Every
cloud came out a near-identical warm grey. Fixed at the selection step so the
panel and the plotted clouds agree.

### Feed-size legibility (applies to every clip)
These figures are 1200-1300 px wide and render around 600 px in a feed, so 9 pt
lands near 4 pt. Colorbar labels, tick labels, axis labels and titles were
bumped in the market and weather demos.

### Accepted, not fixed
- The first frame of every animated clip is near-empty, which makes a weak
  poster frame. Fixing it would mean changing the narrative (all three clips
  deliberately build from nothing). Flagged to Jeremy rather than restructured.
- The weather daily panel is a picket-fence texture; individual cities cannot be
  traced. Inherent to daily resolution at this panel width.

## Market forecast: final numbers actually rendered
293 forecasts, 66% directional accuracy on screen, against 62% for the trivial
"keep drifting as it has" baseline. A forecast cache was added to demo_market.py
(cache/fc_<MODEL>_<...>.npz) because the Laplace precompute is about 7 minutes;
restyling re-renders now load it instantly.


## Market colorbar vs the rotating cube (layout constraint, easy to re-break)
The index colorbar is squeezed between two hard limits. Moving it LEFT to clear
the ticker column ran its edge into the box: the rotating cube's rightmost
projected vertex reaches x = 0.554 of the figure, and the bar had been placed at
0.565, i.e. 12px of clearance at 1200px wide and 6px at feed width. Now at
0.590, with the ticker column at 0.690 and company names at 0.755.

HOW TO MEASURE THIS (two obvious approaches both give the WRONG answer):
- inter-frame variance says "no overlap" because the colorbar is OPAQUE and
  drawn on top, so the occluded cube pixels never vary;
- projecting the cube corners analytically is fragile because mplot3d
  re-positions its own axes, so ax.get_position() is not the rect you passed.
What works: render the bare [-1, 1] box alone, in the same axes rect, across the
clip's real azimuth sweep (elev 10, azim -60 to 25.5 for rotations=0.25), and
measure the black ink. The shipped example was checked the same way and is fine
(110px gap), so this constraint is specific to the demo's two-panel layout.

## State / next steps
- Conversation + weather full renders and the market re-render with the fixed
  forecast still need to finish, then MP4s via ffmpeg + LOSSLESS `gifsicle -O3`
  (never `--lossy`; that caused the earlier "extra dots" false alarm in weather).
- Appearance red-team over the finished clips.
- POST.md rewrite: no em dashes, no "not X but Y", no vacuous prose, no
  over-claiming. Must state the forecast result honestly (see section 3).
- Still open from last session and NOT yet decided by Jeremy: whether to keep
  the `pyproject` bump 1.0.0 -> 1.0.1, and whether to commit the whole set to
  `dev-1.0`.

---

## Round 4 (2026-07-26 evening): Bluesky post refactor

**Ask:** match the real voice of @jeremyrmanning / @contextlab; alt text = short description +
full code or notebook pointer; make the POST about hypertools rather than the pictures; confirm
GIFs fit Bluesky's limits.

### Voice, pulled from 58 real posts (public.api.bsky.app getAuthorFeed, no auth needed)
`--` never em dashes ("Re-sharing from Xitter--", "reviewers-- lots of neat"); bare-domain links;
labeled link stacks ("Paper: ...\nCode/data: ...\n🤗: ..."); `pip install --upgrade <pkg>` inline;
"read on to learn more!" as the thread hook (they never use 🧵); emoji opener ("🚨 New preprint
alert!", "🤠 New release announcement"); we/our; parenthetical asides; emoji sign-off.
Repo uses "color" 2084x vs "colour" 14x -> American spelling throughout.

### Limits, verified against the atproto lexicons (NOT memory -- both had changed)
post text 300 graphemes; **video alt 1000 graphemes** (the binding constraint on "full code");
video file 100 MB (was 50); image/GIF 2 MB (was 1). Old POST.md claimed a 1 MB image cap.

### GIF fitting, measured across fps x width, palette + `gifsicle -O3`
market 420px@6fps 1.75MB · paintings 500px@15fps 1.99MB · conversation 420px@6fps 1.83MB ·
weather 420px@12fps 2.00MB · **morph: NO FIT** (4.99MB at 420px@6fps; dithered dense black dots
are worst-case for GIF). Checked frames visually: text survives the 3x downscale better than
expected, motion is what dies. Video is the answer, and the only option for morph.

### Two errors of mine caught by verifying instead of asserting
1. Read a partial background log and wrote "market GIF: no fit at any size". It fits at 420px@6fps.
2. Published-ready alt text said `hyp.plot(shapes, '.', animate='morph')` was "complete and
   runnable". It **hangs** -- killed at 10 min on `duration=1, frame_rate=2`. Hungarian matching is
   ~O(n^3) and `hyp.load` returns 30135-36022 pts (teapot 1728). `morph_samples=2000` -> 8.2 s.
   => POSSIBLE LIBRARY ISSUE: default morph on the built-in shapes is effectively unusable.

### Verified live
`hyp.reduce(list_of_strings, ndims=3)` -> (8,3) and `hyp.plot(list_of_strings)` -> Figure, so
Post 2's "hand it text" claim is real (the examples use a local `embed()` only for offline runs).

### BLOCKER recorded in POST.md
All 5 tutorial URLs 404 -- the notebooks + examples are untracked, so RTD never built them. They
are already in tutorials.rst:154-186 and the URL pattern is confirmed (plot/reduce.html both 200).
Commit -> let RTD build -> recheck -> post.

**Env note:** repo `.venv/bin/python` (hypertools 1.0.1, numpy 2.3.5). zsh aborts a whole command
on an unmatched glob, which is why an early `ls .venv` probe wrongly reported no venv.

---

## Round 5 (2026-07-26 late): "tutorials must showcase NATIVE hypertools" audit

**Maintainer's verdict:** the examples are mostly custom code wrapped around hypertools. Measured
and confirmed: **6.0% of the 5 launch examples is hyp.* calls; 37.9% is defect (B+C).** Older
tutorials range 2.5%-39%; `analyze.ipynb` never calls `hyp.plot`.

Deliverables in `notes/audit/`: `PLAN.md` (the synthesis), `launch_examples_audit.md`,
`other_tutorials_audit.md`, `native_capability_map.md`, `temperatures_dataset_findings.md`.

### Root cause ranking
R1 **no public per-frame hook** -> 4/5 examples monkeypatch private `FuncAnimation._func`; the
conversation example re-derives hypertools' OWN reveal formula (`matplotlib_backend.py:1316-1318`)
by hand. R2 `predict=` NotImplementedError for every time-progressing animate mode
(`plot.py:2347`). R3 MultiIndex drops continuous `hue=` (loud warning, `plot.py:2682`).
R4 no per-dataset `alpha=`. R5 no per-segment titles. R6 native text bypassed 6x. R7 58 lines of
ffmpeg boilerplate `save_path='x.gif'` replaces. R8 no palette-from-image.

### Things I initially got wrong and corrected from evidence
- Assumed chemtrails-serial was missing. It **works today** (commit 26843931): serial+chemtrails
  = 6 artists vs 3, alpha-.3 trails, and the 3 flags differ (chemtrails 739 / precog 165 /
  bullettime 903 pts). The example just used the wrong call.
- Wrote "hue silently ignored" for MultiIndex; it warns loudly at `plot.py:2682`.
- Missed that MultiIndex's 22 artists = **20 leaf traces + 2 per-level averages** -- the weather
  figure's structure is ALREADY native. Big effort reduction.
- Wrote `labels=` as per-dataset; it is **per OBSERVATION (row)**, flat or nested
  (`plot.py:895-910`). `names=` is the per-dataset one.

### Constraint worth remembering
`names=` RAISES with a categorical `hue=`, and is mutually exclusive with a `legend=` list. The
proposed conversation fix (categorical hue for speakers + per-turn titles) collides with this;
folded into the R5 design.

### New bugs filed in PLAN.md
B1 `linewidth=` ignored for animated continuous-hue plots (measured [0.5,0.5,5.0] -> all 1.5;
static correct). B2 `title=[...]` stringifies onto the figure, no error/warning. B3 `animate='morph'`
never finishes without `morph_samples` (10-min kill; ~O(n^3) Hungarian, 30k pts). B4 teapot 1728
rows / 301 unique.

### 5 open questions for Jeremy (end of PLAN.md)
ordering-API spelling (`order='serial'` vs named combos); weather panel 16/4 vs balanced, absolute
vs anomaly; where time lives in the MultiIndex; forecast scoring in-library or not (I recommend
not); whether Bluesky launch waits for this work or ships on current clips.
