# Tutorials-should-showcase-hypertools: audit + library plan

**Date:** 2026-07-26 · **Branch:** dev-1.0 (HEAD `26843931`) · **Verified with** `.venv/bin/python`

You were right. The measurements are worse than the impression.

---

## 1. The verdict, in numbers

### The five launch examples

| file | legit custom (A) | duplicates native (B) | library gap (C) | cosmetic mpl (D) | **hyp.\* calls** | total |
|-|-|-|-|-|-|-|
| `animate_conversation.py` | 61 | 31 | 49 | 40 | **9** | 190 |
| `animate_market_forecast.py` | 61 | 17 | 100 | 14 | **7** | 199 |
| `animate_morph_zoo.py` | 17 | 5 | 16 | 0 | **5** | 43 |
| `animate_painting_embeddings.py` | 97 | 25 | 6 | 13 | **8** | 149 |
| `animate_weather_decades.py` | 72 | 8 | 44 | 70 | **19** | 213 |
| **TOTAL** | **308** | **86** | **215** | **137** | **48** | **794** |

**6.0% of the code in these "showcase hypertools" examples is hypertools calls.**
**37.9% (B+C = 301 lines) is defect** — either a re-implementation of something native, or a
feature the library should own. Every defect is duplicated verbatim in the matching notebook.

### The other 15 tutorials, ranked by share of real `hyp.*` calls

| worst → best | %hyp | headline |
|-|-|-|
| `conversation_trajectories` | 2.5 | hand-rolled HF embed + sentence windows + ffmpeg |
| `projectile_kalman` | 3.2 | 3 hand-drawn mpl figures, 59 lines archive plumbing |
| `stock_forecasting` | 3.7 | hand-rolled backtest, pandas rolling mean, 2x2 mpl grid |
| `wikipedia_embeddings` | 7.1 | hand-rolled HF embed x2 + ffmpeg |
| `lsl_streaming` | 8.2 | 29-line synthetic LSL outlet |
| `hugging_face_embeddings` | 10.0 | hand-rolled HF embed — its own markdown prints the native call |
| `modern_sklearn_dynamics` | 10.2 | hand-rolled delay embedding + ffmpeg |
| `streaming_data` | 11.8 | clean |
| `text` | 13.3 | clean except a hand-rolled chunker |
| `analyze` | 20.0 | **never calls `hyp.plot`**; 5x copy-pasted `sb.heatmap` |
| `align`,`plot`,`normalize`,`reduce`,`cluster` | 29–39 | clean |

---

## 2. Root causes, ranked by how much custom code each one forces

### R1. There is no public per-frame hook — **the single biggest cause**

Four of the five examples end with the same private-attribute monkeypatch:

```python
_orig = ani._func
def _wrapped(num, *args):
    result = _orig(num, *args)
    ...
    return result
ani._func = _wrapped          # matplotlib FuncAnimation private API
```

`HyperAnimation` (`hyper_animation.py:46-143`) exposes only `.figure`, `.animation`,
`to_html5_video`, `to_jshtml`, `save`, `_repr_html_`. No frame callback, no artist access, no way
to ask "which dataset/segment is current?". Three examples also read `ani._args[...]` to recover
the drawn arrays.

Worst consequence: `animate_conversation.py:182-237` **re-derives hypertools' own reveal formula by
hand**, and says so in its comments. The real formula lives at `matplotlib_backend.py:1316-1318`:

```python
revealed = total_points * num / max(1, total_frames - 1)
```

Two of its comments document off-by-one bugs hit while reproducing library-internal truncation.
A library invariant is leaking into user code.

**Absorbs ~120 lines across 4 files.**

### R2. `predict=` is structurally unavailable for time-progressing animations

```
hyp.plot(..., predict='Kalman', animate=True)
-> NotImplementedError: predict= is only supported with static plots and with animate='spin'
```
(raised `plot.py:2347-2354`; reason at `2338-2345`: *"appending a growing forecast trace is
out-of-scope follow-up work"*).

The market example is `animate=True, chemtrails=True`, so the native path was genuinely closed to
it. Everything it builds to compensate — anchors, reduce-space deltas, a `np.polyfit` to recover
plot's own affine, a 16-slot forecast-history fan, scoring — is **~100 lines of gap**.

### R3. MultiIndex expansion drops a continuous `hue=` — but does more than expected otherwise

Built your requested structure from the real paper data (rows indexed by `(Hemisphere, City)`):

| call | line artists | distinct colors |
|-|-|-|
| `hyp.plot(mi)` | 22 | **2** |
| `hyp.plot(mi, hue=<continuous temp>)` | 22 | **2** |

**Good news, and it changes the effort estimate:** those 22 artists are **20 leaf traces (one per
city) + 2 per-level averages (one per hemisphere)**. The MultiIndex path already produces exactly
the structure the weather figure wants — per-city lines with hemisphere means over them. The
warning text names this explicitly: *"MultiIndex grouping (leaf traces + per-level averages) takes
precedence over hue=; ignoring hue."*

So the weather rewrite needs only two things, not a rebuild:
1. a continuous `hue=` that survives MultiIndex grouping;
2. per-dataset `alpha=`/`linewidth=` (R4, B1) so means read bold and city lines read faint.

The hue is ignored **loudly, not silently** — warning at `plot.py:2682`. Colors were byte-identical
to the no-hue run, so the hue had zero effect. GH #95 is CLOSED; this is fidelity inside a shipped
feature, not a missing feature.

### R4. No per-dataset `alpha=`

```
hyp.plot(ds, '-', alpha=[0.1, 0.1, 1.0])  ->  TypeError: alpha must be numeric or None
```
Documented as deliberate (extra kwargs are never per-dataset). Forces the conversation example's
whole recency-fade block and the weather example's per-frame alpha re-application.

### R5. No per-dataset / per-segment titles — and the current behavior is a **bug**

```
hyp.plot(ds, title=['a','b','c'])   ->  axes title renders the literal string "['a', 'b', 'c']"
```
It stringifies the list onto the figure. In a serial animation the title never changes across
frames (verified: 12 frames, 1 distinct title). This is why the morph example imports the private
`_morph.frame_to_segment`.

### R6. Native text support exists and was bypassed anyway

`hyp.reduce(list_of_strings, ndims=3)` → `(8,3)` and `hyp.plot(list_of_strings)` → Figure, both
verified. The native path is `vectorizer='<hf-id>', semantic=None, corpus=None`
(`text2mat.py:89/184/391/404`).

Six notebooks/examples hand-roll `SentenceTransformer.encode()` anyway. **Two of them print the
native one-call form in adjacent markdown and then hand-roll it.** No blocker was found; this is
pure habit.

### R7. ~58 lines of identical ffmpeg mp4→gif `subprocess` boilerplate across 4 notebooks

`save_path='x.gif'` writes the GIF with no ffmpeg (`plot.py:1246-1250`, `animate.py:80-116`) —
already proven inside this same tutorial set by `streaming_data` and `lsl_streaming`.

### R8. No palette-from-image primitive (confirmed absent)

---

## 3. Your five items, answered

### 3.1 Conversation — text machinery, and "chemtrails-serial"

- **Native text: confirmed available, example must switch.** Delete `embed()`; pass
  `vectorizer='all-MiniLM-L6-v2'`. Grouped text can go in as a list of lists.
- **chemtrails-serial already exists.** `animate='serial', chemtrails=True` composes today
  (commit `26843931`). Verified it genuinely renders, not merely that it doesn't raise:

  | call | artists | pts/artist | alphas |
  |-|-|-|-|
  | serial (plain) | 3 | [903, 739, 0] | [1.0, 1.0, 1.0] |
  | serial + chemtrails | 6 | [903, 247, 0, 0, **739**, 0] | [1,1,1, **.3,.3,.3**] |
  | serial + precog | 6 | [903, 247, 0, 0, **165**, 0] | [1,1,1, .3,.3,.3] |
  | serial + bullettime | 6 | [903, 247, 0, 0, **903**, 0] | [1,1,1, .3,.3,.3] |

  So the example simply used the wrong call. **No library work needed for the effect itself.**

- **But your general principle is not satisfied by the current API shape.** `animate=` currently
  mixes two orthogonal things: a *style* (`spin`, `window`, `morph`) and an *ordering*
  (`parallel` vs `serial`). That is why "the serial version of chemtrails" has no name.

  **Recommended:** make ordering its own axis.
  ```python
  hyp.plot(..., animate=True|'spin'|'window'|'morph', chemtrails=True, order='serial')
  ```
  with `animate='serial'` kept as a permanent alias for `animate=True, order='serial'`. Every
  style then has a parallel and a serial analog by construction, which is exactly the rule you
  stated. Full matrix verified as currently *accepted*: all 5 styles x 4 trail flags = 20/20 OK.

  Applying your rule literally exposes exactly **three** asymmetries today:

  | conceptual style | parallel analog | serial analog | gap |
  |-|-|-|-|
  | plain trajectory reveal | `True`/`'parallel'` | `'serial'` | none |
  | trail decorations | `'parallel'` + flag | `'serial'` + flag (**mpl only**) | **plotly has no serial analog** |
  | sliding window | `'window'` | — | **missing** |
  | camera spin | `'spin'` | — | missing; arguably meaningless as defined |
  | point-cloud morph | — | `'morph'` (inherently sequential) | **missing parallel analog** |

  1. **`'window'` has no serial analog** — the cleanest, most defensible gap.
  2. **`'morph'` has no parallel analog** (N clouds morphing at once toward their own targets).
  3. **Backend asymmetry:** identical code gives a trailed animation on matplotlib and a plain
     opaque reveal plus a warning on plotly. This is the only case where the same call means
     different things by backend, and is the highest-value item to close.

### 3.2 Market — native animation + next-day horizon

Needs **R2**. The horizon change (`t=1`, next day) is trivial and independent, but the example
cannot become native until `predict=` works with `animate=True`. Three sub-capabilities are all
absent today: forecast anchored to a moving animated head, retained forecast history, and scoring.

Recommend: implement forecast-follows-head + a `forecast_trail=` (the forecast analogue of
`chemtrails=`). **Scoring should stay out of the library** — it is analysis, not plotting, and
belongs in the tutorial as legitimately custom code.

### 3.3 Morph — per-dataset titles

Needs **R5**. Design: `names=` (already the per-dataset name kwarg) drives an automatic title
during any serial-style animation (`serial`, `window`, `morph`), blank during transitions. That
is exactly the behavior the morph example hand-builds, and it generalizes to every serial style
as you asked. Fix the `title=[...]` stringification bug at the same time.

**Constraint that affects the conversation example too:** `names=` currently **raises** when a
categorical `hue=` is in play (a categorical hue regroups traces, so they are no longer the named
datasets), and `names=` is mutually exclusive with a `legend=` list (`plot.py:3789+`). The
conversation fix in Phase 0 wants a categorical `hue=` for speaker colors *and* per-turn titles,
which collides. Resolving this is part of the R5 design, not a separate item: either per-segment
titles must read from something other than `names=`, or `names=` must survive categorical
regrouping.

### 3.4 Paintings — full text, native embeddings, native palettes, `labels=`

- **Full text: already in the data.** Each painting dict carries both `text` (the full paragraph,
  which is what gets embedded) and a short `blurb` (what the side panel displays). The fix is
  display-side: show `text`.
- **Native embeddings:** R6.
- **Palette from image: genuinely absent.** Proposed `hyp.load('image.jpg')` → palette helper, or
  `palette='image:<path>'`. Note the known defect from the earlier session: the existing
  hand-rolled `image_palette()` orders k-means clusters by size, so it always returns the muted
  background tone — worth fixing properly inside the library rather than in the example.
- **`labels=`:** exists, but note the exact scope — it is **per OBSERVATION (row)**, flat (one per
  row across all datasets) or nested (one sub-list per dataset), rendered via `ax.annotate`
  (`plot.py:895-910`). `labels[idx] is None` skips that point. So the idiomatic way to label each
  painting cloud once is a nested `labels=` with a single non-None entry per cloud and `None`
  elsewhere. `names=` is the *per-dataset* kwarg, but it drives the legend, not an in-scene
  annotation.

### 3.5 Weather — the paper dataset as a MultiIndex

Fetched and inspected (both CSVs HTTP 200):

- 20 cities, monthly, **1645 complete rows spanning 1875–2013**.
- Hemisphere derives from `Lat` sign: **16 Northern / 4 Southern** (Cape_Town, Santiago,
  Sao_Paulo, Sydney). Note the imbalance vs. the current demo's balanced 6/6.
- Each city has both an absolute and an `_anomaly` column.
- The original paper notebook did `hyp.plot(temps, group=years, palette='RdBu_r',
  normalize='across')` with **months as observations and cities as features** — one trajectory
  through "city space". Everything else in that notebook is seaborn.

Blocked by **R3**. Also worth adding `'temperatures'` to `EXAMPLE_DATA` (`io/load.py:25`) so the
tutorial opens with `hyp.load('temperatures')` — currently a one-line dict entry plus hosting;
`hyp.load` already resolves arbitrary URLs.

---

## 4. Bugs found along the way (independent of any rewrite)

| # | bug | evidence |
|-|-|-|
| B1 | `linewidth=` silently ignored for **animated continuous-hue** line plots | `animate_plot3D` pops `linewidth` (`matplotlib_backend.py:1602-1604`, 2-D twin `2197-2199`) before `_apply_multicolor_animation` reads it (`plot.py:5150-5153`). Measured `linewidth=[0.5,0.5,5.0]` → animated all `1.5`; static correct. |
| B2 | `title=[...]` stringifies the list onto the figure | renders literal `"['a', 'b', 'c']"` |
| B3 | `animate='morph'` with default `morph_samples` never finishes on the built-in shapes | killed at 10 min on `duration=1, frame_rate=2`; Hungarian matching ~O(n^3), `hyp.load` returns 30135–36022 pts. `morph_samples=2000` → 8.2 s. Needs a default cap or an upfront warning. |
| B4 | `hyp.load('teapot')` ships 1728 rows but only 301 unique points | from the earlier session; 3 other shipped examples use it |

---

## 5. Proposed work plan

**Phase 0 — free wins, no library change** (deletes ~86 lines from scripts and again from notebooks)
1. Delete all 6 hand-rolled `embed()` helpers → `vectorizer=`.
2. Delete ~58 lines of ffmpeg boilerplate → `save_path='x.gif'`.
3. `fig.text` titles → `title=`; hand-built colorbars → `colorbar={'label': ...}`;
   `mpatches` legends → categorical `hue=` + `legend=True`.
4. Conversation example → `animate='serial', chemtrails=True`.
5. Paintings side panel → show full `text`, use `labels=`.
6. Market horizon → `t=1`.

**Phase 1 — the unblocking feature**
7. **Public per-frame hook** on `HyperAnimation`, exposing frame index, artists, and the animation
   *schedule*. This is the keystone: it retires every `ani._func` / `ani._args` / `_morph` reach.

**Phase 2 — the per-dataset axis**
8. Per-dataset `alpha=` (+ optional cross-dataset recency fade for serial styles).
9. Per-segment titles from `names=` during serial styles; fix B2.
10. `order='serial'` as an orthogonal modifier; `animate='serial'` becomes an alias.

**Phase 3 — the harder features**
11. `predict=` with time-progressing animations + `forecast_trail=`.
12. MultiIndex: split lines at the inner level, group color at the outer level, respect a
    continuous `hue=`.
13. Palette-from-image.
14. `'temperatures'` in `EXAMPLE_DATA`.

**Phase 4 — rewrite and re-verify**
15. Rewrite all 5 examples + 5 notebooks against the new API; re-audit the ratio.
16. Fix the 15 older tutorials (Phase 0 items apply to most of them).
17. Fix B1, B3, B4.

---

## 6. Open questions for you

1. **Ordering API.** Is `order='serial'` the right spelling, or would you rather every combination
   get a name (`'chemtrails-serial'`, `'precog-serial'`, ...)? The former is 1 kwarg and composes
   automatically; the latter matches how you described it.
2. **Weather panel.** All 20 cities (honest, 16/4 imbalance) or a balanced subset? And absolute
   temperatures or anomalies?
3. **Where does time live in the MultiIndex?** Rows `(hemisphere, city)` with time as columns
   gives one point per city and no trajectory. Repeated `(hemisphere, city)` rows ordered by time
   is the only shape that reproduces the current animation. I assume the latter — confirm.
4. **Forecast scoring:** library feature, or leave in the tutorial as legitimate custom code?
   I recommend leaving it out of the library.
5. **Scope/sequencing.** Phases 1–3 are real library work on top of a shipped 1.0. Should this
   target 1.1, and should the Bluesky launch wait for it or go out on the current clips?

---

## Source reports
- `notes/audit/launch_examples_audit.md` — 5 launch examples, line-by-line A/B/C/D
- `notes/audit/other_tutorials_audit.md` — the 15 older tutorials
- `notes/audit/temperatures_dataset_findings.md` — dataset + MultiIndex probes
