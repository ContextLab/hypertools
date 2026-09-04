# Launch-gallery examples audit — hand-rolled code vs. native hypertools

**Date:** 2026-07-26 · **Repo:** `/Users/jmanning/hypertools` · **Branch:** `dev-1.0` (HEAD `26843931`)
**Scope:** the five newest gallery examples + their matching `docs/tutorials/*.ipynb`.

Classification key:

| code | meaning |
|-|-|
| **A** | LEGITIMATELY CUSTOM — data acquisition/prep hypertools does not and should not own |
| **B** | DUPLICATES NATIVE — hypertools already does this; the example should call it |
| **C** | LIBRARY GAP — presentation/animation logic that belongs inside hypertools but does not exist |
| **D** | COSMETIC MATPLOTLIB — side panels, captions, extra colorbars layered on the figure |
| **NATIVE** | actual `hyp.*` calls |

Line counts are non-blank, non-comment-only source lines (module docstrings excluded).

---

## 0. Headline numbers

| file | A | B | C | D | NATIVE | total |
|-|-|-|-|-|-|-|
| `examples/animate_conversation.py` | 61 | 31 | 49 | 40 | 9 | 190 |
| `examples/animate_market_forecast.py` | 61 | 17 | 100 | 14 | 7 | 199 |
| `examples/animate_morph_zoo.py` | 17 | 5 | 16 | 0 | 5 | 43 |
| `examples/animate_painting_embeddings.py` | 97 | 25 | 6 | 13 | 8 | 149 |
| `examples/animate_weather_decades.py` | 72 | 8 | 44 | 70 | 19 | 213 |
| **TOTAL** | **308** | **86** | **215** | **137** | **48** | **794** |

**6.0% of the code in these five "showcase hypertools" examples is hypertools calls.**
**37.9% (B+C = 301 lines) is defect: either a re-implementation of something native, or a
feature the library is missing.**

### Notebook parity — every defect is duplicated

The five `docs/tutorials/*.ipynb` are code-for-code clones of the scripts. Verified by parsing
the `.ipynb` JSON and counting marker occurrences:

| notebook | nb code lines | py code lines | same hand-rolled helpers? | reaches private internals? |
|-|-|-|-|-|
| `conversation_shape.ipynb` | 186 | 220 | `embed`, `word_spans`, `shown_counts`, `current_state`, `caption_lines`, `set_caption`, `_wrapped` — all present | `ani._func` ×2, `ani._args` ×2 |
| `market_forecast.ipynb` | 192 | 242 | `fetch_fred`, `_wrapped` — present | `ani._func` ×2, `ani._args` ×2, `hypertools._shared` ×1 |
| `morph_shapes_zoo.ipynb` | 45 | 66 | `normalize`, `_wrapped` — present | `ani._func` ×2, `_morph.` ×2 |
| `painting_embeddings.ipynb` | 116 | 169 | `embed`, `canvas_color` — present | none |
| `weather_decades.ipynb` | 206 | 245 | `temp_line`, `_wrapped` — present | `ani._func` ×2 |

Fixing a script without fixing its notebook leaves the defect published.

---

## 1. Suspicion-by-suspicion verdicts

### 1.1 The local `embed()` helpers — **CONFIRMED (B), no real blocker**

Both `examples/animate_conversation.py:88-100` and
`examples/animate_painting_embeddings.py:101-111` define a byte-for-byte identical helper:

```python
def embed(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return np.asarray(model.encode(texts, show_progress_bar=False), dtype=float)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1)
        return vec.fit_transform(texts).toarray().astype(float)
```

**What native text support actually exists** (`hypertools/tools/text2mat.py`):

* `text2mat(data, vectorizer='CountVectorizer', semantic='LatentDirichletAllocation',
  corpus='wiki')` — `text2mat.py:187-188`. That is the DEFAULT pipeline: bag-of-words →
  LDA topic model, fit on a hosted `wiki` corpus.
* Three-tier name resolution for `vectorizer=`/`semantic=` strings — `text2mat.py:36-37`:
  *scikit-learn → gensim → Hugging Face*.
* Tier 3 is `_hf_fallback_model(name)` — `text2mat.py:89-130`, inserted at `text2mat.py:184`
  (`registry[name] = _hf_fallback_model(name)  # tier 3: HuggingFace`). Its docstring names the
  exact model the examples hand-roll: *"A Hugging Face sentence-transformers model name/id,
  e.g. `'all-MiniLM-L6-v2'`"* (`text2mat.py:100-101`). It calls
  `datawrangler.zoo.text.apply_text_model` (`text2mat.py:125-127`).
* `TfidfVectorizer` — the exact fallback the helper reaches for — is a tier-1 built-in
  (`text2mat.py:16`).
* `plot()` exposes `vectorizer=`, `semantic=`, `corpus=` directly — `plot.py:570-572`.

**Empirically verified in this repo (`.venv/bin/python`):**

```
text2mat(texts, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None)  -> (8, 384)
hyp.plot(texts, '.', ndims=3, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None) -> Figure
hyp.plot(texts, '.', ndims=3, vectorizer='TfidfVectorizer', semantic=None, corpus=None)  -> Figure
hyp.reduce(list_of_strings, ndims=3)                                        -> (8, 3)
hyp.reduce([[str, str], [str, str, str]], ndims=3)                          -> [(2,3), (3,3)]
```

`(8, 384)` is precisely the `all-MiniLM-L6-v2` embedding the helper produces. **The helper is a
verbatim re-implementation of tier 3 of hypertools' own resolver, including the TF-IDF fallback.**

**Was there a real blocker?** One narrow, genuine one, and it does not justify the helper:

* `hyp.reduce()` has **no** `vectorizer=`/`semantic=`/`corpus=` parameter — signature at
  `hypertools/reduce/reduce.py:24-27`. Verified: `hyp.reduce(texts, vectorizer='all-MiniLM-L6-v2')`
  → `TypeError: reduce() got an unexpected keyword argument 'vectorizer'`. Both examples call
  `hyp.reduce` (not `hyp.plot`) because they need UMAP kwargs, so from *inside `reduce`* they
  could not select the HF model — they could only get the default CountVectorizer+LDA.
* But `hyp.plot(x, reduce={'model':'UMAP', 'kwargs':{...}}, vectorizer='all-MiniLM-L6-v2',
  semantic=None, corpus=None)` does everything both examples need in ONE call, and
  `text2mat(...)` is a one-line public escape hatch if the two-step shape is wanted.

**Verdict: habit, not blocker.** Two consequences:

1. **(B)** Delete both `embed()` helpers; pass `vectorizer='all-MiniLM-L6-v2', semantic=None,
   corpus=None` through `plot()`.
2. **(C, small)** `reduce()`/`analyze()` should accept the same `vectorizer=`/`semantic=`/`corpus=`
   text kwargs `plot()` already documents (`plot.py:570-572`), so text handling is uniform across
   the API. Today it is a `plot()`-only capability.

Downstream, `hyp.reduce` on a **list of lists of strings** returns a list of per-group arrays
(verified: `[(2,3), (3,3)]`), so the manual re-split loops are also avoidable:

* `animate_conversation.py:144-151` — the `for (spk, _text), nw in zip(TURNS, n_wins)` loop that
  slices `red[k:k+nw]` back into per-turn arrays. **(B)**
* `animate_painting_embeddings.py:148-160,181-182` — the `all_windows`/`owners` bookkeeping and
  `clouds = [red[owners == name] for name in PAINTINGS]`. **(B)** The stated reason for embedding
  all windows in one call ("the TF-IDF fallback fits its vocabulary once — embedding per painting
  would give each a different feature dimension", lines 149-152) evaporates entirely once
  `format_data`/`text2mat` owns the vectorization: it fits one vocabulary across the whole input.

Also **(B)**: the hand-built per-speaker legend at `animate_conversation.py:168-175`
(`mpatches.Patch` + `fig.legend`) plus the `SPEAKER_COLOR`→`colors` list. Verified natively:

```
hyp.plot(ds, '-', hue=<per-observation speaker strings>, palette=[...], legend=True)
   -> legend == ['Alice', 'Hatter', 'March Hare']   # one entry per speaker, not per turn
hyp.plot(..., animate='serial', ...) with the same hue -> 6 datasets still drawn
```

The one-entry-per-category behavior is implemented at `plot.py:204-228`
(`_regroup_categorical_lines`: *"gives each category exactly ONE legend entry"*), and run count
equals dataset count here, so `animate='serial'`'s per-turn reveal is unaffected.

And **(B)**: `fig.text(0.5, 0.965, "Alice's Mad Tea-Party", ...)`
(`animate_conversation.py:176-177`) — `plot()` has `title=` (`plot.py:535`, documented
`plot.py:950-951`). Verified: `hyp.plot(ds, title='many markets as one path')` sets the axes
title. Same duplication at `animate_market_forecast.py:303-304`,
`animate_painting_embeddings.py:198-201`, `animate_weather_decades.py:306-307`.

---

### 1.2 Morph example reaching into `_morph.frame_to_segment` — **CONFIRMED (C)**

`examples/animate_morph_zoo.py:35` imports the private module
(`from hypertools.plot import morph as _morph`) and lines 105-116 recompute the schedule:

```python
total_frames = int(round(fps * duration))
frame_counts, _, _ = _morph.morph_schedule(len(clouds), total_frames, rotations, azim0=-60)
...
seg, _step, _n = _morph.frame_to_segment(frame_counts, frame)
return titles[seg // 2] if seg % 2 == 0 else ''
```

Both functions ARE in that module's `__all__` (`hypertools/plot/morph.py:36-50`), but
`hypertools/plot/__init__.py` exports nothing (it only calls `backend._init_backend()`), so
`hypertools.plot.morph` is not part of the public surface any user would find.

The example's own comment documents the fragility this creates
(`animate_morph_zoo.py:100-104`): *"azim0 must equal hyp.plot's default azim (-60): the schedule
is RECOMPUTED here rather than read back off the figure, and its per-frame azimuth track
accumulates from azim0, so another starting angle would return a schedule that no longer tracks
the rendered camera."* An example that must silently re-derive an internal schedule, and hardcode
a default it cannot read back, is the definition of a missing public feature.

**Missing feature:** `animate='morph'` should accept per-segment **names/titles** — e.g.
`hyp.plot(clouds, animate='morph', names=['Bunny','Cube','Sphere','Teapot','Vase','Bunny'])` —
drawing the name during hold segments and blanking it mid-transition (exactly `shape_title`'s
16 lines). `labels=` today is per-OBSERVATION point annotations (`plot.py:895-901`), not per
morph segment, so it cannot express this.

Also in this file:

* **(B)** `def normalize(points)` at lines 38-42 (center + divide by `max(|.|)`).
  `plot()` already mean-centers and rescales into `[-1,1]` — but with **ONE shared pooled affine**
  across all datasets (`plot.py:4040-4051`; `hypertools/_shared/helpers.py:24-42` `center()`
  "stacked together to compute a single pooled mean", `helpers.py:44-69` `scale()` "a single pooled
  min/max"), so a per-cloud normalization genuinely is NOT redundant — the example's comment at
  lines 54-57 is correct. The near-native equivalent is `normalize='within'`
  (`hypertools/tools/normalize.py:175`, modes `{'across','within','row'}` per `normalize.py:86`),
  which z-scores each dataset independently — but per-column z-scoring distorts a point cloud's
  aspect ratio, which `normalize()` here deliberately preserves. **Classified (B) only because the
  intent (per-dataset size equalization before a shared-affine plot) is a normalize mode
  hypertools nearly has**; a `normalize='isotropic'`/`'unit-cube'` mode would close it cleanly.
* **(A)** The hand-sampling at lines 62-83 is justified: `morph_samples=N` exists
  (`plot.py:1512+`) but "draws a fresh subset per dataset", and the loop-closing repeat
  (`clouds.append(clouds[0])`, line 82) requires the SAME sample. The example's comment
  (lines 54-61) states this accurately.

---

### 1.3 Market example's forecast overlay — **CONFIRMED (C), and the native path is hard-blocked**

`plot()` genuinely has `predict=` (`plot.py:550`) and `t=10` (`plot.py:551`), and
`_draw_forecast_overlays(ax, raw_forecasts, antialias=True)` exists at `plot.py:122-165`.

**What the native path already covers** (verified by reading + running):

* One dashed, alpha-0.6 forecast trace per input dataset, in the SAME color as its source line
  (`plot.py:122-124, 154-164`).
* Forecasts computed in the plotted (post normalize→reduce→align) space, so they line up with the
  drawn trajectory (`plot.py:2963-2999`).
* Forecasts pushed through the **same** center/scale transform as the data, and the center/scale
  statistics computed from data **+** forecasts so the arrow cannot render outside the cube
  (`plot.py:4002-4032`).
* Antialiasing of the forecast via the same `_interp_static_line` the library uses everywhere
  (`plot.py:143-150`).
* `predict=` + `animate='spin'` — verified working.

**What it does not cover, and why this example could not use it:**

* `predict=` raises `NotImplementedError` for **every** time-progressing animate mode. Verified:
  `hyp.plot(..., predict='Kalman', animate=True)` →
  `NotImplementedError: predict= is only supported with static plots and with animate='spin'`
  (raised at `plot.py:2347-2354`; the reason is spelled out at `plot.py:2338-2345`: *"appending a
  growing forecast trace is out-of-scope follow-up work"*). This example is `animate=True,
  chemtrails=True` (`animate_market_forecast.py:192-193`), so the native path is **structurally
  unavailable**. The blocker is real.

Everything the example builds to work around that is **(C)**, ~100 lines:

| lines | what | why it is a gap |
|-|-|-|
| 161-185 | Kalman anchors + per-anchor `hyp.predict(hist, model='Kalman', t=HORIZON)` + reduce-space delta + directional hit scoring | a rolling/"live" forecast at each animation step is exactly what `predict=` + `animate=True` would be |
| 199-213 | reads `ani._args[1][0]` (private), force-runs `_orig(total-1, ...)` to reveal fully, then `np.polyfit`s a per-axis (reduce → drawn) scale | pure consequence of the coordinate-space mismatch the library already solves internally at `plot.py:4002-4032` — the example must reverse-engineer plot's own affine |
| 216-243 | `_frame_of`, GAIN, CAP, `_scale` | no native forecast-scaling/legibility control |
| 244-262 | matured-forecast accumulation → running accuracy array | forecast **scoring** is a genuinely new feature |
| 279-296 | 16-slot history-fan line artists + the live dashed line | a "history of forecasts" trail is the forecast analogue of `chemtrails=` |
| 323-356 | `_wrapped` per-frame updater | needs `ani._func` monkeypatching (see §2) |

The module docstring itself concedes the coordinate problem (lines 30-38): *"hyp.plot internally
normalizes the reduced path into its drawn cube, so points in the original reduce space do NOT
line up with what's on screen."*

**(B)** in this file:

* Lines 297-301 — hand-built `ScalarMappable` + `fig.colorbar` + `cbar.set_label(...)`, after
  passing `colorbar=False` at line 191. `colorbar` accepts a dict with a `'label'` key
  (`plot.py:930`, documented `plot.py:951-958`: *"Pass a dict for finer control:
  `{'label': str, 'ticks': [...], 'location': ...}`"*). Verified:
  `hyp.plot(ds, hue=hue, colorbar={'label': 'equal-weight index (start = 100)'})` works.
* Lines 265-276 — `_smooth()` importing `antialias_line` from `hypertools._shared.helpers`
  (private; `helpers.py:419`). The example's own docstring admits why: *"we call it directly here
  because this forecast overlay is hand-drawn matplotlib rather than a plotted dataset."*
  Disappears the moment the overlay is native (`plot.py:143-150` already antialiases forecasts).
* Lines 303-304 — `fig.text` title; `title=` is native.

**(A)** legitimately: `fetch_fred` (70-97), `synthetic_basket` (100-110), the fetch/fallback
dispatch (113-121), log-prices and thinning (129-131). Downloading CSVs from FRED is exactly the
data acquisition hypertools should not own.

---

### 1.4 Weather example's list-of-loops instead of a MultiIndex — **JUSTIFICATION CONFIRMED TRUE**

The docstring claim (`animate_weather_decades.py:19-22`) — *"hyp's MultiIndex expansion draws the
same bold-means/faint-leaves hierarchy automatically, but it colors by group and ignores a
continuous `hue=` (GH #95)"* — **is accurate against current source.**

Source: `plot.py:2678-2684` —

```python
if hue is not None:
    warnings.warn(
        "x has a row MultiIndex (GH #95): MultiIndex grouping "
        "(leaf traces + per-level averages) takes precedence over "
        "hue=; ignoring hue."
    , stacklevel=external_stacklevel())
    hue = None
```

Verified at runtime with a real 2-level-MultiIndex DataFrame + a continuous `hue`:

```
MULTIINDEX+hue warnings: ['x has a row MultiIndex (GH #95): MultiIndex grouping
 (leaf traces + per-level averages) takes precedence over ...']
```

`build_multiindex_styles` then overwrites `color`, `linewidth`, `alpha`, `linestyle` and `label`
wholesale (`plot.py:3051-3058`), and `color=`/`linewidth=` are likewise warned-and-ignored
(`plot.py:3039-3050`). So the example's choice is correct, and **the gap is in hypertools:**

**(C) Missing feature:** MultiIndex expansion should compose with a continuous `hue=` — take the
hierarchy (leaves + per-level means) from the index and the *color* from `hue`, instead of
discarding one of them. That single change would delete the entire hand-built hierarchy
(`animate_weather_decades.py:138-171`, 26 lines: hemisphere index lists, mean loops, mean temps,
the spliced two-half colormap, `enc()`, the `datasets`/`hue`/`lws` assembly).

**Second, independently confirmed library BUG in this file — (C):**

`animate_weather_decades.py:196-203` says *"hyp's multicolor collections don't inherit the
per-dataset linewidth, so set the emphasis ourselves"* and pokes `heads[k].set_linewidth(MEAN_LW)`
on `Line3DCollection` objects fished out of `ax.collections`. **This is a genuine hypertools bug,
verified empirically:**

```
hyp.plot(ds, '-', hue=<continuous>, linewidth=[0.5, 0.5, 5.0], animate=True, chemtrails=True)
   ANIM   multicolor collection linewidths: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]   <-- all rcParams default
   STATIC multicolor collection linewidths: [..., 0.5, 0.5, 5.0]             <-- correct
```

Root cause: `matplotlib_backend.animate_plot3D` **pops** `linewidth` out of each per-dataset
kwargs dict in place —

```python
# matplotlib_backend.py:1602-1604
linewidths = [
    kwargs_list[idx].pop("linewidth", 1)
```
(and the 2-D twin at `matplotlib_backend.py:2197-2199`)

— and `_apply_multicolor_animation` is called *afterwards* (`plot.py:4377-4383`, guarded by
`line_ani is not None`), where its `_linewidth(i)` helper reads the now-absent key and silently
falls back:

```python
# plot.py:5150-5153
def _linewidth(i):
    tkwargs = kwargs_list[i] if i < len(kwargs_list) else {}
    return (tkwargs.get('linewidth')
            or plt.rcParams['lines.linewidth'])
```

The static path (`_apply_multicolor_lines`, `plot.py:5075-5101`) reads the same key *before* any
pop and is correct — which is exactly the asymmetry the measurement shows. **This should be filed
and fixed; `linewidth=` is silently ignored for any animated continuous-hue line plot.**

**(D)** in this file is large and legitimate-as-cosmetic: lines 205-296, the entire second daily
temperature panel (`temp_line`, the `LineCollection`s, axis setup, the two per-hemisphere
colorbars, the "now" cursor) — **70 lines**. Worth noting the two colorbars at 297-303 are **(B)**
in form (`colorbar={'label': ...}` is native) but **not** in substance: one native colorbar cannot
describe two different colormaps, which the example correctly states at lines 183-185.

**(A)**: `fetch_city_months` (74-95), `synthetic_city_months` (98-111), the fetch loop (114-123).
Downloading open-meteo archives is not hypertools' job.

---

## 2. The cross-cutting gap: **there is no public per-frame hook**

**Four of the five examples end with the identical five lines:**

```python
_orig = ani._func
def _wrapped(num, *args):
    result = _orig(num, *args)
    ...
    return result
ani._func = _wrapped
```

* `animate_conversation.py:199, 286-288, 312, 315`
* `animate_market_forecast.py:203, 323-325, 352, 355`
* `animate_morph_zoo.py:119, 122-125, 128`
* `animate_weather_decades.py:309, 312-314, 329, 332`

`_func` and `_args` are **private matplotlib `FuncAnimation` attributes**. `HyperAnimation`
(`hypertools/plot/hyper_animation.py:46-143`) exposes only `.figure`, `.animation`,
`to_html5_video`, `to_jshtml`, `save`, `_repr_html_` — **no frame callback, no per-frame
annotation, no access to the drawn artists**. Three examples additionally read `ani._args[0]` /
`ani._args[1]` to recover the drawn arrays and line artists
(`animate_conversation.py:184, 190`; `animate_market_forecast.py:202`).

**Missing feature (C):** a public per-frame callback, e.g.
`hyp.plot(..., on_frame=lambda ctx: ...)` where `ctx` exposes `frame`, `n_frames`, the drawn
artists, and — critically — the **animation schedule** (which dataset/segment/sample is current).
That single feature absorbs:

* `animate_morph_zoo.py:99-128` (16 lines) — its `_morph` import and title tracking
* `animate_conversation.py:182-237` (49 lines) — `shown_counts`/`current_state` re-derive
  `update_lines_serial`'s reveal formula **by hand**. The example says so explicitly at lines
  206 and 216 (*"mirroring `update_lines_serial`: `revealed = total_points * num /
  (total_frames - 1)`"*), and the real formula is at
  `hypertools/plot/matplotlib_backend.py:1316-1318`:
  ```python
  lengths = [d.shape[0] for d in data_lines]
  total_points = sum(lengths)
  revealed = total_points * num / max(1, total_frames - 1)
  ```
  Two of its comments (lines 218-222, 231-236) document off-by-one bugs the author hit while
  reproducing library-internal truncation semantics. That is a library-internal invariant leaking
  into user code.
* the frame plumbing in `animate_market_forecast.py:197, 216-218` and
  `animate_weather_decades.py:308, 320-322`

**Second missing feature (C): per-dataset opacity.** Verified:

```
hyp.plot(ds, '-', alpha=[0.1, 0.1, 1.0])
  -> TypeError: alpha must be numeric or None, not <class 'list'>
```

`alpha` is a passthrough `**kwargs` value and is explicitly documented as never per-dataset
(`plot.py`, `**kwargs` docstring: *"an extra kwarg's value is NEVER interpreted as 'one entry per
dataset' even if it happens to be a list/tuple"*). This is what forces:

* `animate_weather_decades.py:315-319` — re-applying `set_alpha(1.0 if k in MEAN_IDX else 0.16)`
  to every collection **on every frame**, because the multicolor updater resets colors (and so
  alpha) each frame.
* `animate_conversation.py:292-305` — the whole recency-fade block (`FLOOR`, `DECAY`, `FINALE`,
  `FINALE_FLOOR`, per-line `set_alpha`). `chemtrails`/`precog`/`bullettime` fade *within* one
  trajectory (`plot.py:1443-1484`); nothing fades *across* already-revealed datasets in
  `animate='serial'`. A `serial_fade=`/`recency=` option, or simply accepting `alpha=[...]`
  per dataset, would absorb it.

---

## 3. Per-file inventory

### 3.1 `examples/animate_conversation.py` (315 lines · A=61 B=31 C=49 D=40 NATIVE=9)

| lines | class | note |
|-|-|-|
| 40-42 | A | imports |
| 44-47 | A | `SPEAKER_COLOR` palette dict |
| 49-85 | A | `TURNS` — curated verbatim Gutenberg dialogue. Genuinely custom; the comment at 49-55 documents why automatic extraction was rejected |
| 88-100 | **B** | `embed()` → `vectorizer='all-MiniLM-L6-v2'` / `'TfidfVectorizer'` (`text2mat.py:89-130, 16, 184`) |
| 103-123 | A | `word_spans()` — sliding word windows. Note lines 110-117: `min_wins` exists purely to dodge a *rendering* artifact ("hyp.plot draws a ONE-ROW dataset as a dot"). A `plot()` option to draw a 1-row dataset as a marker-suppressed no-op, or a documented minimum, would remove the workaround |
| 126-131 | A | window/flat assembly |
| 133-142 | NATIVE | `hyp.reduce(vecs, reduce={'model':'UMAP', ...}, ndims=3)` |
| 144-151 | **B** | manual re-split into per-turn arrays — `hyp.reduce` on a list of per-turn window lists returns them already split (verified `[(2,3),(3,3)]`) |
| 153-166 | NATIVE | `hyp.plot(trajectories, animate='serial', ...)` — the actual showcase |
| 168-175 | **B** | `mpatches.Patch` + `fig.legend` per speaker → `hue=<speaker per row>` + `legend=True` (verified: legend `['Alice','Hatter','March Hare']`) |
| 176-177 | **B** | `fig.text` title → `title=` (`plot.py:535, 950`) |
| 179-180 | D | live speaker label artist |
| 182-199 | **C** | `ani._args[0]`/`[1]`, `drawn_lens`, `starts`, `total_pts` — recovering the drawn schedule from private state |
| 202-237 | **C** | `shown_counts`/`current_state` — hand re-derivation of `update_lines_serial` (`matplotlib_backend.py:1316-1318`) |
| 240-257 | D | `caption_lines` word wrapping |
| 260-283 | D | `set_caption` — per-word `TextArea`/`HPacker`/`VPacker` |
| 286-306 | **C** | `_wrapped` + recency fade (no per-dataset `alpha`, no cross-dataset fade mode) |
| 307-311 | D | speaker/caption text updates |
| 312-316 | **C** | `ani._func = _wrapped` |

### 3.2 `examples/animate_market_forecast.py` (355 lines · A=61 B=17 C=100 D=14 NATIVE=7)

| lines | class | note |
|-|-|-|
| 51-67 | A | imports, cache dir, FRED series ids |
| 70-97 | A | `fetch_fred` — HTTP + pandas merge/ffill |
| 100-110 | A | `synthetic_basket` offline fallback |
| 113-121 | A | fetch/fallback dispatch |
| 129-131 | A | thinning, equal-weight index, log prices |
| 133-134 | NATIVE | `hyp.reduce(logp, reduce='IncrementalPCA', ndims=3, manip='Smooth', normalize='across')` — good, idiomatic |
| 136-185 | **C** | rolling Kalman anchors via `hyp.predict` + reduce-space deltas + directional scoring. `predict=`/`t=` exist (`plot.py:550-551`) but raise `NotImplementedError` for `animate=True` (`plot.py:2347-2354`, verified) |
| 186-194 | NATIVE | `hyp.plot(red, hue=idx_level, chemtrails=True, rotations=0.25, animate=True, ...)` |
| 195-196 | D | axes repositioning for the side colorbar |
| 197-262 | **C** | frame math; `ani._args[1][0]`; force-run `_orig(total-1)`; `np.polyfit` recovery of plot's own reduce→drawn affine (which the library computes internally at `plot.py:4002-4032`); GAIN/CAP; running-accuracy array |
| 265-276 | **B** | `_smooth` → private `hypertools._shared.helpers.antialias_line` (`helpers.py:419`); `plot.py:143-150` already antialiases native forecasts |
| 279-296 | **C** | 16-slot forecast-history fan + live dashed line artists — the forecast analogue of `chemtrails=` |
| 297-304 | **B** | hand `ScalarMappable`+`fig.colorbar`+`set_label` → `colorbar={'label': ...}` (`plot.py:930, 951-958`, verified); `fig.text` title → `title=` |
| 305-321 | D | accuracy subtitle, `Line2D` legend, disclaimer caption |
| 323-356 | **C** | `_wrapped` + `ani._func` monkeypatch |

### 3.3 `examples/animate_morph_zoo.py` (128 lines · A=17 B=5 C=16 D=0 NATIVE=5)

| lines | class | note |
|-|-|-|
| 32-35 | A/**C** | imports — line 35 `from hypertools.plot import morph as _morph` is the private reach |
| 38-42 | **B** | `normalize()` — near-equivalent to a `normalize='within'`-style per-dataset mode (`tools/normalize.py:175`, modes at `normalize.py:86`); plot's own rescale is a single pooled affine (`plot.py:4040-4051`, `helpers.py:24-69`), so per-cloud scaling is genuinely needed — the gap is that hypertools has no aspect-preserving per-dataset normalize mode |
| 45-52 | A | shape list + the teapot-duplicates note |
| 62-83 | A | sampling + loop-close. Correctly justified: `morph_samples=` (`plot.py:1512+`) resamples per dataset, so it cannot produce the identical closing sample |
| 85-97 | NATIVE | `hyp.plot(clouds, animate='morph', rotations=..., morph_samples=N, ...)` — an excellent one-call showcase |
| 99-129 | **C** | `morph_schedule`/`frame_to_segment` recomputation, `shape_title`, `_wrapped`, `ani._func`. Needs: per-segment `names=` for `animate='morph'` + a public frame hook |

This is the **best-proportioned** of the five (5 NATIVE lines drive the whole visual), and the only
defect is the missing per-segment title feature.

### 3.4 `examples/animate_painting_embeddings.py` (212 lines · A=97 B=25 C=6 D=13 NATIVE=8)

| lines | class | note |
|-|-|-|
| 28-41 | A | imports + cache/Wikimedia base URL |
| 43-96 | A | `PAINTINGS` — inline descriptions/blurbs/fallback colors |
| 98 | A | `WINDOW, STEP` |
| 101-111 | **B** | `embed()` — identical to the conversation file's; same native replacement |
| 114-117 | A | `windows()` — sliding word windows |
| 120-146 | A | `canvas_color()` — download the real canvas, k-means over pixels, luminance clamp. **Textbook (A)**: hypertools should never own image fetching or dominant-color extraction |
| 148-160 | **B** | `all_windows`/`owners` bookkeeping + single `embed` call — unnecessary once `format_data`/`text2mat` vectorizes (it fits one vocabulary across the whole input) |
| 162-170 | NATIVE | `hyp.reduce(all_vecs, reduce={'model':'UMAP', ...}, ndims=3)` |
| 172-179 | **C** | per-cloud 85th-percentile outlier trim. No `manip` model does this — the manipulator roster is `Normalize`/`ZScore`/`Smooth`/`Resample` (`manip/manip.py:180-200`). A `manip='TrimOutliers'` would absorb it |
| 181-182 | **B** | split by owner — avoidable via list-of-lists-of-strings input |
| 184-193 | NATIVE | `hyp.plot(clouds, '.', color=colors, reduce=None, animate='spin', rotations=2, ...)` |
| 195-197 | D | axes repositioning |
| 198-201 | **B** | `fig.text` title → `title=` |
| 202-213 | D | side panels (names, wrapped blurbs, swatch rectangles) |

This file has the healthiest **A** ratio — most of its bulk really is data. Its defect is
concentrated in `embed()` + the split/regroup bookkeeping that `embed()` forced.

### 3.5 `examples/animate_weather_decades.py` (332 lines · A=72 B=8 C=44 D=70 NATIVE=19)

| lines | class | note |
|-|-|-|
| 44-72 | A | imports, cache, city table, feature list |
| 74-95 | A | `fetch_city_months` — open-meteo HTTP + monthly aggregation |
| 98-111 | A | `synthetic_city_months` offline fallback |
| 114-123 | A | fetch loop |
| 125-145 | NATIVE | length alignment + `hyp.reduce(mats, reduce='IncrementalPCA', ndims=3, normalize='across')` on a LIST — one shared fit, returns per-city arrays. Idiomatic and correct |
| 146-171 | **C** | hand-built hierarchy: hemisphere index lists, mean loops/temps, spliced two-half colormap, `enc()`, `datasets`/`hue`/`lws` assembly. MultiIndex expansion draws exactly this hierarchy (`plot.py:2685`, `multiindex.py:48-91`) but **discards `hue=`** (`plot.py:2678-2684`, verified) |
| 173-190 | NATIVE | `hyp.plot(datasets, hue=hue, palette=combined, linewidth=lws, animate=True, chemtrails=True, manip='Smooth', legend=False, colorbar=False, ...)` |
| 191-203 | **C** | `ax.set_position` (D) + the `Line3DCollection` linewidth workaround — **a confirmed hypertools bug** (`matplotlib_backend.py:1602-1604` pops `linewidth`; `plot.py:5150-5153` then falls back to rcParams; measured `[1.5]*6` for `linewidth=[0.5,0.5,5.0]`) |
| 205-296 | D | the entire second daily-temperature panel (92 raw / 70 code lines) |
| 297-303 | D/**B** | two `ScalarMappable` colorbars — `colorbar={'label': ...}` is native, but one colorbar cannot describe two colormaps, so this stays cosmetic-but-necessary |
| 305-307 | **B** | `fig.text` title → `title=` (the *dynamic* per-frame date part is (C)) |
| 308-333 | **C** | `_wrapped`: per-frame alpha re-application (no per-dataset `alpha`, verified `TypeError`), lockstep panel reveal, `ani._func` monkeypatch |

---

## 4. Prioritized recommendations

**Immediate example fixes (B) — no library change needed, ~86 lines deleted from scripts and again
from notebooks:**

1. Delete both `embed()` helpers; pass `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None`
   through `plot()`.
2. Feed grouped text directly (list of lists of strings) and delete the manual split/regroup loops.
3. Replace all four `fig.text` titles with `title=`.
4. Replace the market example's hand-built colorbar with `colorbar={'label': ...}`.
5. Replace the conversation example's `mpatches` legend with a categorical `hue=` + `legend=True`.

**Library work (C), ranked by how much example code each deletes:**

| # | feature | absorbs |
|-|-|-|
| 1 | **Public per-frame hook** on `HyperAnimation` (frame index, artists, and the animation *schedule*) | ~120 lines across 4 files; ends all `ani._func`/`ani._args`/`_morph` private reaches |
| 2 | **`predict=` with time-progressing animations** (+ a forecast-history trail and a scoring readout) | ~100 lines in the market example; removes the reduce→drawn `polyfit` hack entirely |
| 3 | **MultiIndex expansion that composes with continuous `hue=`** (GH #95) | ~26 lines in the weather example |
| 4 | **Per-dataset `alpha=`** (and/or a cross-dataset recency fade for `animate='serial'`) | ~20 lines across the weather + conversation examples |
| 5 | **Per-segment `names=` for `animate='morph'`** | ~16 lines in the morph example |
| 6 | **`vectorizer=`/`semantic=`/`corpus=` on `reduce()`/`analyze()`** | makes fix #1 above possible from `reduce()`, not just `plot()` |
| 7 | An aspect-preserving per-dataset normalize mode; a `manip='TrimOutliers'` | ~13 lines (morph + painting) |

**Bug to file and fix regardless of the examples:**

> `linewidth=` is silently ignored for **animated** continuous-hue line plots.
> `matplotlib_backend.animate_plot3D` pops `linewidth` from each per-dataset kwargs dict
> (`matplotlib_backend.py:1602-1604`; 2-D twin at `2197-2199`) before
> `_apply_multicolor_animation._linewidth` reads it (`plot.py:5150-5153`), so every collection
> falls back to `plt.rcParams['lines.linewidth']`.
> Measured: `linewidth=[0.5, 0.5, 5.0]` → animated `[1.5, 1.5, 1.5, 1.5, 1.5, 1.5]`;
> static (correct) → `[..., 0.5, 0.5, 5.0]`.

---

## 5. Verification log

Commands run with `/Users/jmanning/hypertools/.venv/bin/python` (matplotlib `Agg`):

```
text2mat(texts, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None)   -> (8, 384)
hyp.plot(texts, '.', ndims=3, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None) -> Figure
hyp.plot(texts, '.', ndims=3, vectorizer='TfidfVectorizer', semantic=None, corpus=None)  -> Figure
hyp.reduce(texts, ndims=3)                                                  -> (8, 3)
hyp.reduce(texts, vectorizer='all-MiniLM-L6-v2')                            -> TypeError (no such kwarg)
hyp.reduce([[s,s],[s,s,s]], ndims=3)                                        -> [(2,3), (3,3)]
hyp.plot([[s,s],[s,s,s]], '.', vectorizer='all-MiniLM-L6-v2', ...)          -> Figure
hyp.plot(ds,'-',hue=<per-obs speakers>,palette=[...],legend=True)           -> legend ['Alice','Hatter','March Hare']
   same + animate='serial'                                                  -> 6 datasets still drawn
hyp.plot(ds,'-',hue=cont,linewidth=[0.5,0.5,5.0],animate=True,chemtrails=True)
                                                                            -> collection lw [1.5]*6  (BUG)
   same, static                                                             -> [..., 0.5, 0.5, 5.0]  (correct)
hyp.plot(ds,'-',alpha=[0.1,0.1,1.0])                                        -> TypeError: alpha must be numeric
hyp.plot(multiindex_df,'-',hue=cont)                                        -> UserWarning "...ignoring hue" (GH #95)
hyp.plot(ds,'-',hue=cont,colorbar={'label':'equal-weight index (start = 100)'}) -> OK
hyp.plot(ds,'-',title='many markets as one path')                           -> axes title set
hyp.plot(ds,'-',predict='Kalman',t=10,animate='spin')                       -> OK
hyp.plot(ds,'-',predict='Kalman',t=10,animate=True)                         -> NotImplementedError
```

Notebook/script parity checked by parsing `.ipynb` JSON and counting helper/private-internal
markers (table in §0).
