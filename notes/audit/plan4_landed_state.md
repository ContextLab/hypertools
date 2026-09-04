# Plan 4 (1.1 examples & tutorials) — landed-state audit

**Date:** 2026-08-01
**Repo:** `/Users/jmanning/hypertools`, branch `dev-1.0`, HEAD `065c841e`, tree clean
**Plan under audit:** `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md` (v2, 2615 lines)
**Python used for every measurement:** `/Users/jmanning/hypertools/.venv/bin/python`

**Why this audit exists.** Commit `d730a085` ("docs(1.1): document order=, per-dataset alpha=, on_frame, per-segment titles; simplify examples", 2026-08-01 09:46) landed *after* Plan 4 v2 was written and rewrote four of the five example scripts. Plan 4's task bodies still describe the pre-`d730a085` files. Several of its steps are **line-anchored surgical edits**, which are the dangerous kind: applied verbatim they now cut the wrong lines.

**Metric.** All "code lines / native lines" figures below use Plan 4's **own** metric — `scripts/measure_native_ratio.py` as specified in Task 8 Step 1 (plan lines 2081–2183). That file does not exist in the repo yet (Task 8 Step 1 has not been executed; `ls scripts/` confirms), so it was copied verbatim into a scratchpad and run from there. No repo file was created or modified by this audit.

---

## Summary table

| Task | Script | Script status | Notebook status | Plan action needed |
|-|-|-|-|-|
| **2** | `examples/animate_market_forecast.py` | **Partially landed** — `_func`/`_args` per-frame monkeypatch → `anim.on_frame(decorate)`; 2 private uses deliberately retained with inline evidence | **UNCHANGED since `9b94d86f`** — still the full `_orig`/`_wrapped`/`ani._func = _wrapped` cell. **OUT OF SYNC** | **REBASE** (also blocked: `forecast_trail=` does not exist in `plot()`) |
| **3** | `examples/animate_weather_decades.py` | **Partially landed** — `_func` monkeypatch → `anim.on_frame(decorate)`; `fig, ani =` → `anim` + `anim.figure` | **UNCHANGED** — still `_orig = ani._func` / `ani._func = _wrapped`. **OUT OF SYNC** | **REBASE** (light: plan replaces the file wholesale, but its BEFORE numbers and both verification snippets are stale) |
| **4** | `examples/animate_painting_embeddings.py` | **UNTOUCHED** by `d730a085` (last change `4d1d2223`); plan's baseline + line citations verified accurate | **UNCHANGED**, and consistent with its script — **IN SYNC** | **WRITE-AS-IS** (still gated on Task 1: `hypertools.plot.colors.image_palette` does not exist) |
| **5** | `examples/animate_conversation.py` | **Partially landed** — `_func` monkeypatch → `anim.on_frame(decorate)`; `ani._args[0]/[1]` → `ctx.datasets`/`ctx.artists` | **UNCHANGED** — still `lines = ani._args[1]`, `ani._func = _wrapped`. **OUT OF SYNC** | **REBASE** |
| **6** | `examples/animate_morph_zoo.py` | **FULLY LANDED** — Task 6 Step 1 is already done, verbatim, including the prescribed docstring rewrite | **UNCHANGED** — still imports `_morph`, recomputes `morph_schedule`, monkeypatches `ani._func`. **OUT OF SYNC** | **REBASE** — delete Step 1 entirely, fix Step 2's snippet, keep Steps 3–5 |

**One-line verdict:** `d730a085` modernized four *scripts* and left all five *notebooks* alone. Plan 4's Contract 2 ("Script and notebook are one deliverable … land in the same commit", plan line 76) is already violated on disk, and the **notebooks are now the bulk of the remaining work**.

---

## Cross-cutting facts (established once, cited by every section)

### Commit ordering

```
065c841e 2026-08-01 21:54:12 docs(plans): Plan 4 v2 review — fix the 3 defects v2 itself introduced   <- HEAD
21f46d3c 2026-08-01 10:35:14 fix(plot): reserve top margin so 3-D animated titles actually render
d730a085 2026-08-01 09:46:46 docs(1.1): document order=, ... ; simplify examples                      <- the four scripts
9b94d86f (2026-07-30)        fix(docs): execute the five new tutorials; repair make html at the source <- the five notebooks
```

`git log --oneline -- docs/tutorials/<name>.ipynb` returns exactly `9b94d86f` then `4d1d2223` for **all five** notebooks. **No notebook has been touched by `d730a085` or by any commit since.**

### Measured code lines / native ratio (plan's own metric)

| file | plan's BEFORE | pre-`d730a085` (actual) | **current (HEAD)** |
|-|-|-|-|
| `examples/animate_market_forecast.py` | 191 code, 11 native, 5.8% | 191 / 11 / 5.8% | **191 / 11 / 5.8%** |
| `examples/animate_weather_decades.py` | 196 / 11 / 5.6% | 196 / 11 / 5.6% | **195 / 11 / 5.6%** |
| `examples/animate_conversation.py` | 166 / 9 / 5.4% | 166 / 9 / 5.4% | **165 / 9 / 5.5%** |
| `examples/animate_morph_zoo.py` | 40 / 6 / 15.0% | 40 / 6 / 15.0% | **26 / 6 / 23.1%** |
| `examples/animate_painting_embeddings.py` | 146 / 11 / 7.5% | 146 / 11 / 7.5% | **146 / 11 / 7.5%** |
| `docs/tutorials/market_forecast.ipynb` | 192 / 11 / 5.7% | — | **193 / 12 / 6.2%** |
| `docs/tutorials/weather_decades.ipynb` | 206 / 10 / 4.9% | — | **207 / 11 / 5.3%** |
| `docs/tutorials/conversation_shape.ipynb` | 186 / 11 / 5.9% | — | **191 / 12 / 6.3%** |
| `docs/tutorials/morph_shapes_zoo.ipynb` | 45 / 8 / 17.8% | — | **46 / 9 / 19.6%** |
| `docs/tutorials/painting_embeddings.ipynb` | 116 / 10 / 8.6% | — | **121 / 11 / 9.1%** |

**Only morph moved materially** (40 → 26 code lines, 15.0% → 23.1%). Weather and conversation each lost exactly 1 code line; market and paintings are unchanged. The notebook baselines are all off by 1–5 code lines and 1 native line, in every case *understating* the current figure.

### Raw line counts

| file | plan's BEFORE raw | pre-`d730a085` | current |
|-|-|-|-|
| market | 355 | 355 | **376** |
| weather | 333 | 332 | **336** |
| conversation | 316 | 315 | **320** |
| morph | 129 | 128 | **96** |
| paintings | 213 | 212 | **212** |

(Counted with both `wc -l` and `len(text.splitlines())`; all five files end in a newline, so the two agree. The plan's numbers are +1 on four of five, suggesting a different counting convention; the morph and market deltas are real.)

### Notebook execution state (still exactly as the plan's revision note says)

| notebook | code cells | with outputs | with committed traceback |
|-|-|-|-|
| `market_forecast.ipynb` | 7 | **4** | 0 |
| `weather_decades.ipynb` | 7 | **2** | 0 |
| `conversation_shape.ipynb` | 6 | **2** | 0 |
| `morph_shapes_zoo.ipynb` | 6 | **1** | 0 |
| `painting_embeddings.ipynb` | 6 | **2** | 0 |

These match the plan's **Revision note (v2)** line 24 exactly ("measured 2/6, 4/7, 1/6, 2/6, 2/7"). **But the five per-task BEFORE headers were never corrected** — all five still read "0 of N code cells executed" (plan lines 632, 1004, 1199, 1433, 1805), directly contradicting the plan's own revision note, which claims the 0-figure was "Corrected wherever it appears". This is a residual plan defect independent of `d730a085`.

### Library API surface today (bears on whether the plan's prescribed code can even run)

`inspect.signature(hyp.plot)` → 75 parameters. Of the ones Plan 4 leans on:

```
title            -> PRESENT      predict          -> PRESENT
order            -> PRESENT      t                -> PRESENT
alpha            -> PRESENT      forecast_trail   -> ABSENT
simplify         -> PRESENT      colorbar         -> PRESENT
on_frame         -> PRESENT      morph_samples    -> PRESENT
vectorizer/semantic/corpus/labels/hue/palette -> PRESENT
```

`FrameContext` fields: `frame, n_frames, figure, axes, artists, datasets, style, order, current_index, current_fraction, revealed_counts, segment_index, segment_kind` — a **superset** of the three the plan's Task 5 callback needs, so the plan's "Interface check" note is satisfied.

`from hypertools.plot.colors import image_palette` → **ImportError**. Task 1 has not landed.

`HyperAnimation` lives in `hypertools/plot/hyper_animation.py:45` and is a `tuple` subclass:
```python
class HyperAnimation(tuple):
    def __new__(cls, figure, animation, frame_hooks=None):
```
so `fig, ani = result` and `result.figure` both work, but **`anim._func` does not exist** — `_func` is on `result[1]`.

### Plan Task 8's `DEFECT_MARKERS` gate, simulated against the files as they stand

| file | gate result |
|-|-|
| `examples/animate_market_forecast.py` | **FAIL**: `ani._func`, `ani._args`, `hypertools._shared`, `antialias_line` |
| `examples/animate_weather_decades.py` | **FAIL**: `ani._func` *(prose only — see below)* |
| `examples/animate_painting_embeddings.py` | **FAIL**: `SentenceTransformer` |
| `examples/animate_conversation.py` | **FAIL**: `SentenceTransformer`, `ani._func` *(prose only)* |
| `examples/animate_morph_zoo.py` | **PASS** |
| `docs/tutorials/market_forecast.ipynb` | **FAIL**: `ani._func`, `ani._args`, `hypertools._shared`, `antialias_line` |
| `docs/tutorials/weather_decades.ipynb` | **FAIL**: `ani._func` |
| `docs/tutorials/painting_embeddings.ipynb` | **FAIL**: `SentenceTransformer` |
| `docs/tutorials/conversation_shape.ipynb` | **FAIL**: `SentenceTransformer`, `ani._func`, `ani._args` |
| `docs/tutorials/morph_shapes_zoo.ipynb` | **FAIL**: `ani._func`, `from hypertools.plot import morph`, `morph_schedule|frame_to_segment` |

**New gate hazard introduced by `d730a085`.** The gate's `_code_text()` returns the **entire** `.py` file, docstrings included. `d730a085` deliberately added migration-explaining prose that names the removed pattern, e.g. `examples/animate_weather_decades.py:318`:

> ``` ``ani._func`` monkeypatch this replaces) there is no original updater ```

and `examples/animate_conversation.py:281`:

> ``` the pre-1.1 ``ani._func`` monkeypatch this replaces) there is no ```

Neither is a private reach; both are documentation of the fix. As written, Task 8's gate would fail these two files for their *docstrings*. The gate needs to strip Python docstrings (as `measure_native_ratio._code_lines_py` already does) or the prose needs rewording.

**Morph is the only file that already passes both the defect gate and its budget** (26 code lines ≤ the plan's 30-line budget).

---

## Task 2 — Market (`examples/animate_market_forecast.py` + `docs/tutorials/market_forecast.ipynb`)

Plan section: lines 630–1000. `d730a085` changed 39 lines here.

### A. What `d730a085` already accomplished that Task 2 also wanted

| Task 2 wanted | already done at HEAD | evidence |
|-|-|-|
| Remove `_wrapped` + `ani._func = _wrapped` (row 6 of the "What goes" table, `:323-356`) | **Done.** The per-frame monkeypatch is gone; the decorator is a `FrameContext` callback registered on the public hook | `examples/animate_market_forecast.py:340` `def decorate(ctx):` and `:376` `anim.on_frame(decorate)` |
| — | `num = ctx.frame` replaces the `_wrapped(num, *args)` signature | `:346` `    num = ctx.frame` |

### B. What Task 2 still wants that the current file does NOT do

Everything else. Verified by running the current file (`MPLBACKEND=Agg .venv/bin/python`, 6.1 s):

```
market data: 1941 days x 5 series (FRED daily series)
forecasts: 74 drawn, 73 scored; final directional accuracy = 66%
axes: 2 | ax.lines: 19 | distinct lws: [1.1, 2.2, 2.6]
ax.get_title(): ''
fig.texts[0]: 'many markets as one path'
```

- **5 FRED series, not a 24-ticker `(Market, Sector, Ticker)` MultiIndex.** `:66` `FRED_IDS = ['SP500', 'NASDAQCOM', 'DGS10', 'DCOILWTICO', 'VIXCLS']`
- **No native `predict=`/`t=`.** Forecasts are still precomputed by hand: `:170` `f = np.asarray(hyp.predict(hist, model='Kalman', t=HORIZON))`
- **The whole reduce→drawn affine recovery survives.** `:222-223` `SLOPE = np.array([np.polyfit(_red_rs[:, c], full_drawn[:, c], 1)[0] ...`, plus `GAIN` `:238`, `CAP` `:246`, `_scale` `:249`, `BLO`/`BHI` `:224-225`, `_hang` `:296`, `_frame_of` `:228`
- **The 16-slot hand-drawn fan survives.** `:303` `N_HIST = 16` … `:306` `hist_lines = [ax.plot(...)]`
- **Hand-built colorbar survives.** `:315-318` `cax = fig.add_axes([0.82, 0.14, 0.02, 0.66])` / `cbar.set_label('equal-weight index (start = 100)', fontsize=9)` — with `colorbar=False` still on the call (`:191`)
- **Hand-built title survives.** `:320` `title = fig.text(0.40, 0.965, '', ...)`; probe confirms `ax.get_title() == ''`
- **Two private reaches survive, now deliberately** — see C.

### C. Where applying Task 2 verbatim would REGRESS the current file

Task 2 Step 2 says *"Replace `examples/animate_market_forecast.py` entirely"*, so by construction it deletes everything `d730a085` added. Three specific losses:

**C1 — deletes the public-hook migration and re-introduces nothing in its place.** The plan's replacement body simply has no per-frame code (row 6: *"nothing — there is no per-frame work left"*). If Task 2 lands complete, that is fine. If it lands **partially** — which is likely, because `forecast_trail=` does not exist (see C3) — the `anim.on_frame(decorate)` migration at `:376` is the thing most at risk of being reverted.

**C2 — deletes two evidence-bearing rationales that a future reader needs.** `d730a085` did not just migrate; it recorded *why* two private usages must stay, with measurements. Verbatim replacement silently discards both:

```
examples/animate_market_forecast.py:204-213
# This ONE-TIME setup step is the one place this example still reaches into
# matplotlib's private FuncAnimation internals (`ani._args`/`ani._func`),
# deliberately: it needs the fully-revealed, ANTIALIASED on-screen line (this
# is a synchronous "force a render, then read it back" operation, not a
# per-frame callback), and there is no public equivalent -- `ctx.datasets`
# (from `on_frame=`) is the pre-antialiasing array at a coarser resolution
# and fits a measurably different (~2-8%, checked empirically) slope. The
# RECURRING per-frame decoration below has a clean public replacement and
# uses it (`anim.on_frame`); this setup step does not, so it is left alone
# rather than silently changing the fitted forecast geometry.
```

```
examples/animate_market_forecast.py:283-287
    There is no public re-export of it (unlike ``title=``/``on_frame=``, this
    is smoothing, not a per-frame callback, so it is outside plan 1.1's
    scope) -- reimplementing PCHIP antialiasing by hand here would risk
    silently drifting from what ``hyp.plot`` actually draws, so the private
    import stays.
```

This is a **direct contradiction** the rebase must resolve, not paper over. Plan Contract 3 (line 78) states *"After this plan, no example or notebook contains `ani._func`, `ani._args`, `hypertools._shared` …"*. `d730a085` ruled the opposite for these two sites, with evidence. One of the two positions has to be withdrawn explicitly.

**C3 — the prescribed `hyp.plot(...)` call cannot run today.** Plan line 880: `predict='Kalman', t=1, forecast_trail=16,`. `forecast_trail` is **ABSENT** from `plot()`'s 75 parameters (verified via `inspect.signature`). Task 2's Prerequisites row (plan line 135) names "Forecast-animation T5 (`forecast_trail=`)" — that dependency has **not** landed. Task 2 is blocked, not merely stale.

### D. Contract-by-contract check of the CURRENT file

| contract | satisfied? | evidence |
|-|-|-|
| Native `title=` | **NO** | `:320` `title = fig.text(0.40, 0.965, '', ha='center', va='top', fontsize=14,` — probe: `ax.get_title() == ''` |
| Native `colorbar={'label': ...}` | **NO** | `:191` `colorbar=False` + `:317-318` `cbar = fig.colorbar(sm, cax=cax)` / `cbar.set_label(...)` |
| `on_frame` instead of `_func` monkeypatch | **YES** | `:376` `anim.on_frame(decorate)` |
| No private API use | **NO (deliberately)** | `:214` `market_line = ani._args[1][0]`; `:215` `_orig = ani._func`; `:216` `_orig(total - 1, *ani._args)`; `:289` `from hypertools._shared.helpers import antialias_line` |
| Column MultiIndex / `order=` / per-dataset `alpha=` | **NO** | none present; `:191` passes a flat `red` array with `linewidth=2.2` |
| Native `predict=` on the plot call | **NO** | `:170` precomputed `hyp.predict` in a Python loop |

### E. Stale line citations in Task 2's "What goes" table

| plan citation | actual location at HEAD |
|-|-|
| `_frame_of` … `_hang` at `:197-243` | `_frame_of` `:228-230`; `SLOPE` `:222-223`; `GAIN` `:238`; `CAP` `:246`; `_scale` `:249-252`; `BLO`/`BHI` `:224-225`; `_hang` `:296-297` |
| `hist_lines` fan at `:279-296` | `:303-312` |
| `_smooth` + `antialias_line` at `:265-276` | `:277-293` |
| `ScalarMappable` + `fig.colorbar` at `:297-301` | `:314-318` |
| `fig.text(...)` title at `:303-304` | `:320-321` |
| `_wrapped` + `ani._func = _wrapped` at `:199-213`, `:323-356` | **`_wrapped` and `ani._func = _wrapped` NO LONGER EXIST**; `ani._args[1][0]` is at `:214` |

### F. Notebook — `docs/tutorials/market_forecast.ipynb`: **OUT OF SYNC**

7 code cells, 4 with outputs, 0 tracebacks. Cell 12 (163 lines) still carries the pre-`d730a085` script verbatim:

```
market_forecast.ipynb cell 12
  L 16| market_line = ani._args[1][0]
  L 17| _orig = ani._func
  L 18| _orig(total - 1, *ani._args)                              # reveal fully, once
  L 81|     from hypertools._shared.helpers import antialias_line
  L131| def _wrapped(num, *args):
  L132|     result = _orig(num, *args)
  L163| ani._func = _wrapped
```

The script at `:340`/`:376` now reads `def decorate(ctx):` / `anim.on_frame(decorate)`. The notebook's `L131`/`L163` are the exact lines `d730a085` deleted from the script. **The published tutorial teaches the monkeypatch the library now has a public replacement for.**

### G. Task 2 verification snippets

Step 4's probe (plan line 947) uses `ani._func(40, *ani._args)` on `ns['ani']`. This **still works** for market — the current file binds both names (`:191` `anim = hyp.plot(...)`, `:195` `fig, ani = anim`). Confirmed by running it. Market is the only one of the three migrated scripts where this holds.

---

## Task 3 — Weather (`examples/animate_weather_decades.py` + `docs/tutorials/weather_decades.ipynb`)

Plan section: lines 1002–1194. `d730a085` changed 28 lines here.

### A. What `d730a085` already accomplished

| Task 3 wanted | already done | evidence |
|-|-|-|
| No per-frame `_func` callback (plan line 1043: *"and no per-frame callback"*) | **Partially** — the monkeypatch is gone; a public-hook callback remains | `:312` `def decorate(ctx):`, `:336` `anim.on_frame(decorate)`, `:320` `    frame = ctx.frame` |
| — | the return value is now handled as a `HyperAnimation` | `:186` `anim = hyp.plot(datasets, fmt='-', hue=hue, palette=combined,` and `:191` `fig = anim.figure` |

Deleted by `d730a085`: `_orig = ani._func`, `def _wrapped(frame, *args):`, `result = _orig(frame, *args)`, `return result`, `ani._func = _wrapped`.

### B. What Task 3 still wants that the current file does NOT do

Essentially all of it. Task 3 is a **reframe**, not a cleanup: 20 cities as 20 *features* of one path. The current file is still the 6-city, two-hemisphere, two-panel figure. Verified by running it:

```
weather: 6 cities (open-meteo archive)
type(anim): HyperAnimation | fig axes: 4
ax title: ''
fig.texts[0]: ''
```

- **6 cities, not 20.** `:62` `CITIES = {                       # 3 per hemisphere: loops stay followable`
- **Hand-built hemisphere hierarchy survives.** `:141-142` `Nmean_loop = np.mean([city_loops[i] for i in N_idx], axis=0)` / `Smean_loop = ...`
- **Hand-spliced colormap survives.** `:151-153` `combined = LinearSegmentedColormap.from_list('combo', [Ncm(x) for x in np.linspace(0, 1, 128)] + [Scm(x) for x in np.linspace(0, 1, 128)])`
- **Two hand-built `ScalarMappable` colorbars survive.** `:300` `cbN = fig.colorbar(ScalarMappable(Normalize(Nlo, Nhi), Ncm), cax=caxN)`; `:303` `cbS = ...`; with `colorbar=False` on the call (`:187`)
- **The `Line3DCollection` linewidth workaround survives.** `:199-204` `_colls = [c for c in ax.collections if isinstance(c, Line3DCollection)]` … `heads[k].set_linewidth(MEAN_LW)`
- **The whole second daily-temperature panel survives** (`:206-296`, ~90 lines), which Task 3 deletes as class **D**
- **`fig.text` title survives.** `:307` `title = fig.text(0.47, 0.965, '', ha='center', va='top', fontsize=13.5,`; probe: `ax.get_title() == ''`

### C. Where applying Task 3 verbatim would REGRESS the current file

Task 3 Step 1 says *"Replace `examples/animate_weather_decades.py` entirely"*. The one thing lost that `d730a085` added is the public-hook migration and its rationale:

```
examples/animate_weather_decades.py:312-319
def decorate(ctx):
    """Per-frame decoration: bold hemisphere means vs. faint cities (the
    multicolor updater resets per-segment alpha every frame, so it is
    re-applied here), and the 2nd panel's lockstep reveal + "now" cursor.
    Registered below via ``anim.on_frame`` -- by the time this runs,
    hyp.plot() has already drawn the frame, so (unlike the pre-1.1
    ``ani._func`` monkeypatch this replaces) there is no original updater
    to call through to, and nothing to return."""
```

Because Task 3's replacement has **no** callback at all, this is a clean deletion rather than a silent revert — the risk here is **low**, and lower than any other task. Weather is the safest of the four to apply as prescribed.

One genuine loss worth an explicit decision: `d730a085`'s docstring sentence documenting the artist-lifetime gotcha ("*the multicolor updater resets per-segment alpha every frame*") is the only place in `examples/` that records that behaviour. It is also covered by `docs/animation.rst` (added in the same commit), so the loss is recoverable.

### D. Contract-by-contract check of the CURRENT file

| contract | satisfied? | evidence |
|-|-|-|
| Native `title=` | **NO** | `:307` `title = fig.text(0.47, 0.965, '', ...)`; probe `ax.get_title() == ''` |
| Native `colorbar={'label': ...}` | **NO** | `:187` `colorbar=False`; `:300`/`:303` two hand-built colorbars |
| `on_frame` instead of `_func` | **YES** | `:336` `anim.on_frame(decorate)` |
| No private API use | **YES in code**, **NO under the plan's own gate** | no functional private reach; but `:318` contains the literal string `` ``ani._func`` `` inside a docstring, which Task 8's `DEFECT_MARKERS` regex would flag |
| `order=` / per-dataset `alpha=` / per-segment titles | **NO** | none present |
| 20 cities as features, one native call | **NO** | `:62-69` six cities as six datasets |

### E. Stale numbers

Plan line 1004 says *"333 raw lines, **196 code lines**"* and *"`weather_decades.ipynb` — 206 code lines, 10 native (4.9%), 0 of 7 code cells executed"*.
Actual at HEAD: **336 raw, 195 code, 11 native, 5.6%**; notebook **207 code, 11 native, 5.3%, 2 of 7 cells executed**.

### F. Notebook — `docs/tutorials/weather_decades.ipynb`: **OUT OF SYNC**

7 code cells, 2 with outputs. Cell 11 (152 lines) still carries the monkeypatch:

```
weather_decades.ipynb cell 11
  L129| _orig = ani._func
  L132| def _wrapped(frame, *args):
  L133|     result = _orig(frame, *args)
  L152| ani._func = _wrapped
```

vs the script's `:336` `anim.on_frame(decorate)`. The notebook also still opens the plot call as `fig, ani = hyp.plot(...)` (cell 11 L8) where the script now uses `anim = hyp.plot(...)` / `fig = anim.figure` (`:186`/`:191`).

### G. Task 3's verification snippet is now BROKEN

Plan Step 3 (line 1148–1150):

```python
ns = runpy.run_path('examples/animate_weather_decades.py')
fig, ani = ns['fig'], ns['ani']
ani._func(150, *ani._args)
```

The current file **defines no name `ani`**. Verified: `runpy.run_path` namespace contains `anim`, not `ani` (`grep` confirms `:186` `anim = hyp.plot(`, `:191` `fig = anim.figure`; there is no `ani =` anywhere in the file). The snippet raises `KeyError: 'ani'`. Even reaching the object directly fails — `anim._func` raises `AttributeError: 'HyperAnimation' object has no attribute '_func'` (measured), because `HyperAnimation` is a `tuple` subclass and `_func` belongs to element `[1]`. The working spelling today is `ns['anim'][1]._func(150, *ns['anim'][1]._args)`.

---

## Task 4 — Paintings (`examples/animate_painting_embeddings.py` + `docs/tutorials/painting_embeddings.ipynb`)

Plan section: lines 1197–1428.

### Confirmation: genuinely untouched, baseline still accurate

```
$ git log --oneline -- examples/animate_painting_embeddings.py
4d1d2223 docs(examples): add five animated gallery demos and refresh the tutorials
```

Single commit. `d730a085` does not list it (`git show --stat d730a085` names only the other four). Line counts identical pre- and post-`d730a085`: **212 raw / 146 code / 11 native / 7.5%**.

Plan's BEFORE (line 1199): *"213 raw lines, **146 code lines, 11 native (7.5%)**"* — code/native/ratio **exact**; raw is +1 by the plan's convention. Notebook BEFORE *"116 code lines, 10 native (8.6%)"* vs measured **121 / 11 / 9.1%** — off by 5 code / 1 native, same understating direction as the other four notebooks.

### Line citations spot-checked against the file — all accurate

| plan citation | verified at HEAD |
|-|-|
| `PAINTINGS` dict "lines 43-96" | `:96` is the dict's closing `}` — **correct** |
| `embed()` at `:101-111` | `:101` `def embed(texts):` … `:111` `        return vec.fit_transform(texts).toarray().astype(float)` — **correct** |
| k-means + `np.argmax(counts)` + luminance clamp at `:136-144` | `:138` `km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(px)`, `:140` `rgb = km.cluster_centers_[np.argmax(counts)] / 255.0`, `:142-143` the `lum > 0.5` clamp — **correct** |
| `all_windows`/`owners` bookkeeping at `:148-160` | `:153` `all_windows, owners, colors_by_name = [], [], {}` … `:160` `owners = np.array(owners)` — **correct** |
| outlier trim at `:172-179` | `:174-179`, the `np.percentile(dist, 85)` block — **correct** (plan's range includes the two preceding comment lines) |
| `clouds = [red[owners == name] ...]` at `:181-182` | `:181-182` — **correct** |
| `fig.text(...)` title at `:198-201` | `:198-201` — **correct** |

### Remaining blocker, unrelated to `d730a085`

Task 4 depends on Task 1. `from hypertools.plot.colors import image_palette` → **ImportError** at HEAD. Task 4's prescribed body imports it at plan line 1266 (`from hypertools.plot.colors import image_palette`) and calls it at line 1305 (`return tuple(image_palette(dest)[0])`).

### Notebook — `docs/tutorials/painting_embeddings.ipynb`: **IN SYNC**

6 code cells, 2 with outputs. Cell 5 still carries `from sentence_transformers import SentenceTransformer` / `model = SentenceTransformer('all-MiniLM-L6-v2')` — which **matches** its script (`examples/animate_painting_embeddings.py:104-105`, identical two lines). Script and notebook tell the same story; both are pre-1.1, together. This is the only pair not out of sync.

### Verdict: **WRITE-AS-IS**

Task 4 needs no rebase. It needs Task 1 landed first.

---

## Task 5 — Conversation (`examples/animate_conversation.py` + `docs/tutorials/conversation_shape.ipynb`)

Plan section: lines 1431–1800. `d730a085` changed 49 lines here — the largest of the four.

### A. What `d730a085` already accomplished that Task 5 also wanted

| Task 5 wanted ("What goes" table row) | already done | evidence |
|-|-|-|
| `_wrapped` + `ani._func = _wrapped` (`:286-316`, class **C**) → `on_frame=` | **Done** | `:277` `def decorate(ctx):`, `:320` `anim.on_frame(decorate)`, `:293` `    num = ctx.frame` |
| `ani._args[0]`/`[1]` (part of row 5, class **C**) → library-published schedule | **Done, partially.** The private reads are gone; the *derivation* survives, now fed from `FrameContext` | `:283` `    lines = ctx.artists                                     # one Line3D per turn`; `:290` `    drawn_lens = [np.asarray(a).shape[0] for a in ctx.datasets]` |

The three deleted lines were:
```
-lines = ani._args[1]                                       # one Line3D per turn
-drawn_lens = [np.asarray(a).shape[0] for a in ani._args[0]]
-_orig = ani._func
```

### B. What Task 5 still wants that the current file does NOT do

Verified by running the current file (SentenceTransformer + UMAP, ~90 s):

```
CONVERSATION namespace has: ['anim', 'decorate', 'fig', 'speakers', 'trajectories']
frame   0 ax.get_title()=''  fig.texts=["Alice's Mad Tea-Party", 'Alice']
frame  60 ax.get_title()=''  fig.texts=["Alice's Mad Tea-Party", 'Alice']
frame 120 ax.get_title()=''  fig.texts=["Alice's Mad Tea-Party", 'Dormouse']
frame 191 ax.get_title()=''  fig.texts=["Alice's Mad Tea-Party", 'Alice']
legend entries: ['Alice', 'Hatter', 'March Hare', 'Dormouse']
distinct alphas: [0.62, 0.63, 0.64, 0.65, 0.7, 0.79, 1.0]
```

- **`embed()` survives** (row 1). `:88-100`, incl. `:92` `        from sentence_transformers import SentenceTransformer`
- **Manual re-split survives** (row 2). `:145-151` `trajectories, colors, speakers = [], [], []` … `    speakers.append(spk)`
- **`mpatches.Patch` + `fig.legend` survive** (row 3). `:169` `import matplotlib.patches as mpatches`; `:173-176` `fig.legend(handles=[mpatches.Patch(color=SPEAKER_COLOR[s], label=s) ...`
- **`fig.text` title survives** (row 4). `:177-178`; probe confirms `ax.get_title() == ''` on every sampled frame
- **`shown_counts` / `current_state` survive** (row 5) — they were *re-signatured*, not deleted. `:193` `def shown_counts(num, starts, drawn_lens, total_pts):`; `:202` `def current_state(num, starts, drawn_lens, total_pts):`. They still hand-mirror `update_lines_serial`: `:197` `    revealed = total_pts * num / max(1, total - 1)`
- **The speaker artist + `caption_lines` + `set_caption` survive** (row 6). `:180-181`, `:231-248`, `:259-274`
- **`animate='serial'` (not `order='serial'`)**: `:164` `                animate='serial',`
- **Legend order is dict order, not first-appearance.** Measured `['Alice', 'Hatter', 'March Hare', 'Dormouse']`; Task 5 Step 3 (plan line 1763) expects first-appearance `Alice, March Hare, Hatter, Dormouse`. The current legend comes from `SPEAKER_COLOR`'s insertion order (`:44-47`), not from `colors.py`'s category ordering.

### C. Where applying Task 5 verbatim would REGRESS the current file

Task 5 Step 1: *"Keep `SPEAKER_COLOR` and the `TURNS` list verbatim (lines 44-85). Replace everything below."* Lines 44–85 are still exactly `SPEAKER_COLOR` + `TURNS` at HEAD, so the anchor itself is intact — **this is the one line-anchored instruction in Plan 4 that still points at the right text.**

But "replace everything below" deletes the `d730a085` migration:

```
examples/animate_conversation.py:277-283
def decorate(ctx):
    """Per-frame decoration: recency fade across turns, the speaker label,
    and the bolded caption. Registered below via ``anim.on_frame`` -- by the
    time this runs, ``hyp.plot`` has already drawn the frame, so (unlike
    the pre-1.1 ``ani._func`` monkeypatch this replaces) there is no
    original updater to call through to, and nothing to return."""
    lines = ctx.artists                                     # one Line3D per turn
```
```
examples/animate_conversation.py:290
    drawn_lens = [np.asarray(a).shape[0] for a in ctx.datasets]
```
```
examples/animate_conversation.py:320
anim.on_frame(decorate)
```

**Task 5's replacement re-establishes an equivalent** (`ani.on_frame(recency_fade)`, plan line 1564), so this is a *replacement*, not a loss — **provided the whole task lands**. The regression risk is a partial application: reverting `ctx.datasets` back to `ani._args[0]` is exactly what the "What goes" table's row 5 tells an implementer to look for, and that text no longer exists to be deleted.

**Real, non-obvious regression:** the plan's `recency_fade` **drops the finale ramp**. `d730a085` preserved it; Task 5's replacement has no equivalent:

```
examples/animate_conversation.py:186-190
# Over the final stretch the whole conversation is lifted back up, so the clip
# ends on the shape it spent the whole run building rather than on one lit turn
# against near-invisible history.
FINALE = int(1.4 * fps)
FINALE_FLOOR = 0.62
```
```
examples/animate_conversation.py:299-300
    ramp = min(1.0, max(0.0, (num - (total - 1 - FINALE)) / max(1, FINALE)))
    floor = FLOOR + (FINALE_FLOOR - FLOOR) * ramp
```

The plan's `turn_alpha` (plan lines 1507–1520) uses a fixed `FLOOR = 0.10` with no ramp. The measured alphas at frame 191 confirm the finale is live today (`0.62, 0.63, 0.64, 0.65, 0.7, 0.79, 1.0` — floor lifted to ≈0.62, exactly `FINALE_FLOOR`). Under the plan's version the last frame would floor at 0.10. **Plan Contract 7 (line 98) requires this to be named in *Decisions still needed* rather than quietly lost — it currently is not.**

### D. Contract-by-contract check of the CURRENT file

| contract | satisfied? | evidence |
|-|-|-|
| Native per-segment `title=` | **NO** | `:177` `fig.text(0.5, 0.965, "Alice's Mad Tea-Party", ...)`; measured `ax.get_title() == ''` at frames 0/60/120/191 |
| `order='serial'` | **NO** — uses the older `animate='serial'` spelling | `:164` `                animate='serial',` |
| Per-dataset `alpha=` | **NO** — alpha is set artist-by-artist in the callback | `:307` `            ln.set_alpha(0.0)`, `:309` `ln.set_alpha(1.0)`, `:311` `ln.set_alpha(floor + (1.0 - floor) * DECAY ** (ti - j))` |
| `on_frame` | **YES** | `:320` `anim.on_frame(decorate)` |
| No private API | **YES in code** (`ani._args`/`ani._func` gone), **NO under the gate** — `:281` contains `` ``ani._func`` `` in a docstring |
| Native text (`vectorizer=`/`semantic=`/`corpus=`) | **NO** | `:88-100` hand-rolled `embed()`; `:92` `from sentence_transformers import SentenceTransformer` |
| Categorical `hue=` + `legend=True` | **NO** | `:163` `color=colors`; `:173` `fig.legend(handles=[mpatches.Patch(...)])` |
| The `revealed_counts`/`current_index` head-trail split the plan's Fatal fix needs | **N/A today** — the current callback uses `ctx.artists`/`ctx.datasets` and derives its own counts (`:290-294`), so the 2N-vs-N hazard the plan's v1 hit is not present, but neither is the plan's fix | `:283`, `:290-292` |

### E. Stale line citations in Task 5's "What goes" table

| plan citation | actual at HEAD |
|-|-|
| `embed()` `:88-100` | **still correct** |
| manual re-split `:144-151` | `:145-151` — essentially correct |
| `mpatches.Patch` + `fig.legend` `:173-176` | **still correct** |
| `fig.text(...)` title `:177-178` | **still correct** |
| `ani._args[0]`/`[1]`, `drawn_lens`, `starts`, `total_pts`, `shown_counts`, `current_state` `:182-237` | **`ani._args` GONE.** `drawn_lens`/`starts`/`total_pts` moved *into* `decorate` at `:290-292`; `shown_counts` `:193-199`; `current_state` `:202-228` |
| speaker artist + `caption_lines` + `set_caption` `:180-181`, `:240-283` | `:180-181` correct; `caption_lines` `:231-248`; `set_caption` `:259-274` |
| `_wrapped` + `ani._func = _wrapped` `:286-316` | **BOTH GONE.** Replaced by `decorate` `:277-317` + `anim.on_frame(decorate)` `:320` |

### F. Notebook — `docs/tutorials/conversation_shape.ipynb`: **OUT OF SYNC**

6 code cells, 2 with outputs. Cell 9 (144 lines) still carries all three private reads the script no longer has:

```
conversation_shape.ipynb cell 9
  L 23| lines = ani._args[1]                       # one Line3D per turn
  L 25| drawn_lens = [np.asarray(a).shape[0] for a in ani._args[0]]
  L 34| _orig = ani._func
  L116| def _wrapped(num, *args):
  L117|     result = _orig(num, *args)
  L144| ani._func = _wrapped
```

vs the script's `:283` `lines = ctx.artists`, `:290` `drawn_lens = [... for a in ctx.datasets]`, `:320` `anim.on_frame(decorate)`. **L23/L25/L34/L144 are, line for line, the four lines `d730a085` deleted from the script.** This is the sharpest script/notebook divergence of the five.

Cell 5 also still carries `from sentence_transformers import SentenceTransformer` — but so does the script (`:92`), so that half is consistent.

### G. Task 5's verification snippet is now BROKEN

Plan Step 3 (line 1747–1752):

```python
ns = runpy.run_path('examples/animate_conversation.py')
fig, ani = ns['fig'], ns['ani']
...
    ani._func(f, *ani._args)
```

Measured namespace: `['anim', 'decorate', 'fig', 'speakers', 'trajectories']` — **no `ani`**. Raises `KeyError: 'ani'`. Working spelling today: `fig, funcani = ns['anim']` then `funcani._func(f, *funcani._args)`.

Also stale: Step 2a's test module hard-codes `N_DATASETS = 6` (plan line 1604) with a note to confirm against `len(ctx.revealed_counts)`. The current example draws **28** turn datasets (`len(TURNS)` = 28, `:56-85`), so that constant is only meaningful after Task 5's `hue=` reshaping lands.

---

## Task 6 — Morph (`examples/animate_morph_zoo.py` + `docs/tutorials/morph_shapes_zoo.ipynb`)

Plan section: lines 1803–1912. `d730a085` changed 54 lines here — and **completed Task 6 Step 1 in full**.

### A. What `d730a085` already accomplished

Both rows of Task 6's "What goes, and to what" table are **done**:

| plan row | status | evidence |
|-|-|-|
| Delete `from hypertools.plot import morph as _morph` (`:35`) and the `morph_schedule` recomputation with hardcoded `azim0=-60` (`:105-107`) | **DONE** | the import and `frame_counts, _, _ = _morph.morph_schedule(len(clouds), total_frames, rotations, azim0=-60)` are both gone from the file; `grep -n "_morph\|morph_schedule" examples/animate_morph_zoo.py` → no hits |
| Delete `shape_title`, `label`, `_wrapped`, `ani._func = _wrapped` (`:108-128`) → `title=titles` | **DONE** | `examples/animate_morph_zoo.py:96` `                    title=titles)` |

The prescribed plot call (plan lines 1838–1841) and the file's actual call are the same call, modulo argument order:

```
examples/animate_morph_zoo.py:93-96
fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
                    animate='morph', rotations=rotations, morph_samples=N,
                    duration=duration, frame_rate=fps, size=(6, 6), show=False,
                    title=titles)
```

Task 6 Step 1's prescribed **docstring rewrite** is also already applied. Plan lines 1847–1851 prescribe *"The shape names come straight from the library: ``title=`` takes one string per cloud …"*; the file at `:14-19` reads:

```
The **title that tracks the current shape** comes straight from the library:
passing a list of per-shape names as ``title=`` to ``hyp.plot`` is enough. A
morph animation alternates "hold" segments (the camera slowly orbits a
finished shape) with "transition" segments (one shape flowing into the next);
``hyp.plot`` names the shape while holding and shows nothing mid-transition,
so the label never sits over a half-formed cloud.
```

All four "Kept, deliberately" bullets are also intact: the teapot note (`:41-46`), `CUBE_SCALE = 0.8` (`:62`), the closed loop `clouds.append(clouds[0])` (`:78`), `normalize()` (`:34-38`), and the explicit `morph_samples=N` (`:95`).

Budget: **26 code lines ≤ the plan's 30-line budget**; ratio 23.1% (plan's non-gated target was 26%). Defect-marker gate: **PASS** — the only one of the ten files that passes.

### B. What Task 6 still wants that the current file does NOT do

**Nothing in the script.** Only the notebook (Steps 3–5) remains.

### C. Where applying Task 6 verbatim would REGRESS the current file — **HIGHEST RISK IN PLAN 4**

Task 6 Step 1 (plan line 1827) is a **line-anchored surgical edit**:

> Delete `from hypertools.plot import morph as _morph` (line 35) and replace everything from line 94 to the end of the file with:

Against the file as it stands:

- **"line 35"** is now the **docstring of `normalize()`**. There is no `_morph` import to delete; deleting line 35 strips the helper's documentation instead:
  ```
  examples/animate_morph_zoo.py:34-38
  34| def normalize(points):
  35|     """Center a point cloud and scale it into the hypertools [-1, 1] cube."""   <- what "line 35" hits today
  36|     points = np.asarray(points, dtype=float)
  37|     points = points - points.mean(axis=0)
  38|     return points / np.abs(points).max()
  ```
  (The file lost 32 lines in `d730a085`: 128 → 96, so every anchor below the import has shifted.)
- **"line 94 to the end"** lands *inside* the `hyp.plot(...)` call, which now spans `:93-96`:
  ```
  92| # hyp.plot itself during every transition -- no hand-rolled schedule needed.
  93| fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
  94|                     animate='morph', rotations=rotations, morph_samples=N,   <- "line 94"
  95|                     duration=duration, frame_rate=fps, size=(6, 6), show=False,
  96|                     title=titles)
  ```
  Replacing from 94 leaves the orphan fragment `fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,` with an unclosed paren, followed by the plan's own complete call — a **`SyntaxError`**.

Net effect of applying Step 1 verbatim: a stripped `normalize()` docstring **and** a truncated/duplicated plot call that will not parse. **Step 1 must be deleted from the plan, not rebased.**

### D. Task 6 Step 2's verification snippet is itself defective — and would produce a false failure

Plan Step 2 (lines 1858–1874) computes the expected title from:

```python
counts = segment_frame_counts(len(ns['clouds']), total)
```

but the example passes a **non-uniform** per-segment rotation list (`:87` `rotations = [0.75] + [0.5, 1.0] * (len(SHAPES) - 1) + [0.5, 0.75]`), and the real signature is:

```
segment_frame_counts (n_datasets, total_frames, rotations=None)
```

Run both ways against the current file:

```
PLAN snippet (no rotations)      mismatches: 34 [(22, 1, 'Bunny', ''), (38, 1, 'Cube', ''), (39, 1, 'Cube', '')]
WITH the example's rotations     mismatches: 0 []
```

The plan's expected output is `frames checked: 240 | mismatches: 0 []`, and it instructs: *"Any mismatch means the native titles are not tracking `frame_to_segment`'s parity — that is animation-core Task 8's contract, so fix it there, not here."* **Following that instruction would send an implementer to "fix" a library that is already correct.** The library's per-segment titles track the schedule exactly; the probe omits `rotations`. Fix: pass `ns['rotations']` as the third argument.

### E. Contract-by-contract check of the CURRENT file

| contract | satisfied? | evidence |
|-|-|-|
| Native per-segment `title=` | **YES** | `:96` `                    title=titles)`; measured 0/240 title mismatches against `frame_to_segment` parity |
| Removal of private API use | **YES** | no `_morph`, no `morph_schedule`, no `frame_to_segment`, no `ani._func` anywhere in the file; defect gate **PASS** |
| `morph_samples=` explicit and reproducible | **YES** | `:95` `                    animate='morph', rotations=rotations, morph_samples=N,` with the rationale at `:50-57` |
| Teapot kept with its note | **YES** | `:41-46`, e.g. `# NOTE on the teapot: ``hyp.load('teapot')`` returns 1728 rows but only 301` |
| `CUBE_SCALE` kept with its reason | **YES** | `:59-62`, `CUBE_SCALE = 0.8` |
| Closed loop + shared sample kept | **YES** | `:78` `clouds.append(clouds[0])`, `:79` `titles = TITLES + [TITLES[0]]` |
| Hand `normalize()` kept | **YES** | `:34-38`, with the "NOT redundant with hyp.plot" rationale at `:50-53` |
| `order=` / per-dataset `alpha=` / `on_frame` | **N/A** — Task 6 does not ask for them | — |

### F. Stale numbers and citations

Plan line 1805 says *"129 raw lines, **40 code lines, 6 native (15.0%)**"*.
Actual at HEAD: **96 raw, 26 code, 6 native, 23.1%** — the largest baseline drift in Plan 4.
Notebook BEFORE *"45 code lines, 8 native (17.8%), 0 of 6 code cells executed"* vs measured **46 / 9 / 19.6%, 1 of 6 executed**.

All "Kept, deliberately" citations shifted: teapot `:45-50` → `:41-46`; `CUBE_SCALE` `:63-66` → `:59-62`; closed loop `:54-61` → `:50-57`.

### G. Notebook — `docs/tutorials/morph_shapes_zoo.ipynb`: **OUT OF SYNC (worst pair in the set)**

6 code cells, 1 with outputs. The notebook still contains, verbatim, **every single line** `d730a085` deleted from the script:

```
morph_shapes_zoo.ipynb cell 3
   4| from hypertools.plot import morph as _morph
```
```
morph_shapes_zoo.ipynb cell 7
   4| fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
   5|                     animate='morph', rotations=rotations,
   6|                     morph_samples=N, duration=duration,
   7|                     frame_rate=fps, size=(6, 6), show=False)
```
```
morph_shapes_zoo.ipynb cell 9  (24 lines, the whole cell)
   1| total_frames = int(round(fps * duration))
   2| frame_counts, _, _ = _morph.morph_schedule(
   3|     len(clouds), total_frames, rotations, azim0=-60)
   4| label = fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=16,
   5|                  fontweight='bold', color='#1a1a1a')
   8| def shape_title(frame):
   9|     seg, _step, _n = _morph.frame_to_segment(frame_counts, frame)
  12|     return titles[seg // 2] if seg % 2 == 0 else ''
  15| _orig = ani._func
  18| def _wrapped(frame, *args):
  19|     result = _orig(frame, *args)
  20|     label.set_text(shape_title(frame))
  21|     return result
  24| ani._func = _wrapped
```

The script's cell-7 equivalent now ends `size=(6, 6), show=False,` + `title=titles)` and cell 9 has no counterpart at all. The published tutorial page therefore teaches a 24-line private-schedule workaround for a feature the script proves is a single keyword. Task 6 Step 3's instruction — *"The current cell 9 (24 lines of schedule recomputation and `_func` monkeypatching) is **deleted outright**"* — is still exactly right and still entirely undone.

---

## Consolidated rebase actions

**Delete outright (already landed):**
- Task 6 Step 1 in its entirety — script + docstring both match the prescription.
- Task 2 "What goes" row 6's `_wrapped`/`ani._func = _wrapped` half.
- Task 5 "What goes" row 7 (`_wrapped` + `ani._func = _wrapped`) and the `ani._args[0]`/`[1]` half of row 5.
- Task 3's implicit "delete the per-frame callback" — the monkeypatch half is done.

**Fix (plan is wrong against today's repo):**
- Task 6 Step 2: pass `ns['rotations']` to `segment_frame_counts`, or the check reports 34 false mismatches and misdirects the fix into the library.
- Task 3 Step 3 and Task 5 Step 3: `ns['ani']` → `KeyError`; the scripts bind `anim`. Use `fig, funcani = ns['anim']`. (Task 2 Step 4 is unaffected — market binds both.)
- All five BEFORE tables: refresh raw/code/native counts, and correct "0 of N code cells executed" to the measured 4/7, 2/7, 2/6, 2/6, 1/6 the plan's own revision note already established.
- All of Task 2's and most of Task 5's "What goes" line citations.

**Resolve as a decision, not an edit:**
- Task 2 vs `d730a085` on the two retained private usages in market (`ani._args`/`ani._func` one-time setup, `antialias_line` import). `d730a085` retained them with measured evidence; Plan 4 Contract 3 forbids them. One side must yield explicitly.
- Task 5's `recency_fade` silently drops the finale ramp (`FINALE`/`FINALE_FLOOR`, `:189-190`, `:299-300`), which is live today. Contract 7 requires this be named in *Decisions still needed*.
- Task 8's `DEFECT_MARKERS` gate must stop scanning Python docstrings, or `d730a085`'s migration prose in weather `:318` and conversation `:281` will fail the gate for documenting the fix.

**Blocked on other plans (unchanged by this audit):**
- Task 2: `forecast_trail=` is **ABSENT** from `plot()`.
- Task 4: `hypertools.plot.colors.image_palette` does not exist (Task 1).

**Untouched and correct:**
- Task 4 in full. Its baseline, its seven line citations, and its notebook/script consistency all verified accurate.
