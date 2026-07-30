# HyperTools native capability map

Repo: `/Users/jmanning/hypertools`, branch `dev-1.0`, HEAD `26843931`, `hyp.__version__ == 1.0.1`.
All results below were produced by **executing real code** with `.venv/bin/python`
(matplotlib `Agg`, `show=False`), not read off docstrings. Every claim carries a
`file:line` citation into `hypertools/` source.

Verdict key: **SUPPORTED** / **PARTIAL** / **ABSENT**.

---

## 1. Animation style matrix — PARTIAL

### 1.1 Accepted values

`animate=` is validated at `hypertools/plot/plot.py:2201`:

```python
if isinstance(animate, str) and animate not in ('parallel', 'spin',
                                                'serial', 'morph', 'window'):
    raise ValueError(
        f"unknown animate style {animate!r}; valid styles are 'parallel', "
        "'spin', 'serial', 'morph', 'window' (or True/False). ...")
```

So the full set is `False`, `True` (== `'parallel'`), `'parallel'`, `'spin'`,
`'serial'`, `'window'`, `'morph'`, plus:

* a **per-dataset list** — resolved by `_resolve_animate_mode` (`plot.py:453`).
  Entries may only be `'morph'` / `None` / `False`; the list form exists
  **solely** to tag which datasets join a morph sequence (`plot.py:480-512`).
  `'spin'`/`'serial'`/`'window'` **cannot** vary per dataset.
* a **dict "mega-form"** (GH #154) — `plot.py:2119-2181`, e.g.
  `animate={'style': 'spin', 'rotations': 2, 'duration': 15}`; every non-`style`
  key is unpacked onto the flat kwarg of the same name, and specifying the same
  key twice raises.
* **sugar strings** `'chemtrails'` / `'precog'` / `'bullettime'` — `plot.py:2189-2197`
  map to `animate='parallel'` with the matching boolean flag set.

`chemtrails=` / `precog=` / `bullettime=` are **boolean OR per-dataset list of
bool** (GH #127), broadcast at `plot.py:3715-3717` via
`hypertools/plot/trails.py:18` `broadcast_trail_flag`.

### 1.2 The verified matrix

Script: 3 datasets of `np.cumsum(np.random.randn(40,4))`, `reduce='PCA'`,
`duration=1`, `frame_rate=4`. `lines=` is `len(ax.lines)` on the returned figure
(3 = head artists only; 6 = 3 heads + 3 low-opacity trail artists).

**matplotlib backend**

| animate | none | chemtrails | precog | bullettime |
|-|-|-|-|-|
| `False` | Figure, lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 |
| `True` | HyperAnimation, lines=3 | **composes**, lines=6 | **composes**, lines=6 | **composes**, lines=6 |
| `'parallel'` | HyperAnimation, lines=3 | **composes**, lines=6 | **composes**, lines=6 | **composes**, lines=6 |
| `'spin'` | HyperAnimation, lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 |
| `'serial'` | HyperAnimation, lines=3 | **composes**, lines=6 | **composes**, lines=6 | **composes**, lines=6 |
| `'window'` | HyperAnimation, lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 | **ignored** (warns), lines=3 |
| `'morph'` | HyperAnimation, lines=4 | **ignored** (warns), lines=4 | **ignored** (warns), lines=4 | **ignored** (warns), lines=4 |

**plotly backend**

| animate | none | chemtrails | precog | bullettime |
|-|-|-|-|-|
| `False` | Figure | **ignored** (warns) | **ignored** (warns) | **ignored** (warns) |
| `True` | Figure | **composes** | **composes** | **composes** |
| `'parallel'` | Figure | **composes** | **composes** | **composes** |
| `'spin'` | Figure | **ignored** (warns) | **ignored** (warns) | **ignored** (warns) |
| `'serial'` | Figure | **ignored** (warns) | **ignored** (warns) | **ignored** (warns) |
| `'window'` | Figure | **ignored** (warns) | **ignored** (warns) | **ignored** (warns) |
| `'morph'` | Figure | **ignored** (warns) | **ignored** (warns) | **ignored** (warns) |

Nothing is *silently* ignored: every ignored cell emits a `UserWarning`. The
static-plot warning is `plot.py:3731-3737`; the animated-style warning is
`plot.py:3776-3781`. The style→warning gating is `plot.py:3757-3759`:

```python
_trail_ignoring_modes = ("spin", "morph", "window")
if resolve_backend(backend) == "plotly":
    _trail_ignoring_modes = _trail_ignoring_modes + ("serial",)
```

### 1.3 Commit 26843931 ("serial composes with chemtrails/precog/bullettime") — VERIFIED

Not just artist creation — the frames genuinely differ per flag. Driving
`anim.animation._draw_frame(n)` on 3×30-row datasets, `duration=2`,
`frame_rate=5` (lines 0-2 are heads, lines 3-5 are trails at alpha 0.3):

```
== chemtrails n_lines 6
  frame 5  npts [901, 301, 0, 0, 501, 0]   alpha [None,None,None,0.3,0.3,0.3]
== precog n_lines 6
  frame 5  npts [901, 301, 0, 0, 401, 0]   alpha [None,None,None,0.3,0.3,0.3]
== bullettime n_lines 6
  frame 5  npts [901, 301, 0, 0, 901, 0]   alpha [None,None,None,0.3,0.3,0.3]
```

At frame 5 dataset 0 is fully revealed (901 pts), dataset 1 is mid-reveal
(301 pts), and its trail artist (index 4) carries **501 / 401 / 901** points for
chemtrails / precog / bullettime respectively — i.e. revealed-past,
not-yet-revealed-future, and whole-trajectory. Exactly the documented semantics.
Implementation: `hypertools/plot/matplotlib_backend.py:1283` `update_lines_serial`
(3-D) and `:2048` `update_lines_serial_2d`.

Per-dataset lists also compose with serial:

```
serial + chemtrails=[True,False,True]                       -> lines 5   (2 trails)
serial + mixed chemtrails/precog/bullettime one-hot lists   -> lines 6   (3 trails)
serial + chemtrails=True + focused=1.0                      -> lines 6, no warning
animate=['morph','morph',None]                              -> lines 4
```

### 1.4 Why plotly differs

`hypertools/plot/plotly_backend.py:946-949` builds trail traces only for the
parallel styles:

```python
trail_dataset_indices = [
    i for i in range(len(data))
    if chemtrails[i] or precog[i] or bullettime[i]
] if animate in (True, 'parallel') else []
```

There is no plotly analogue of `update_lines_serial`, so `'serial'` on plotly
reveals fully opaque with no trail.

### 1.5 Assessment against "every style should have a parallel (default) and a serial (optional) analog"

Mapping the current design onto the principle:

| conceptual style | parallel analog | serial analog | gap |
|-|-|-|-|
| plain trajectory reveal | `True`/`'parallel'` | `'serial'` | none — the principle is fully met here |
| trail decorations (chemtrails / precog / bullettime) | `animate='parallel', chemtrails=…` | `animate='serial', chemtrails=…` (mpl only) | **plotly has no serial analog** |
| sliding window | `'window'` | — | **missing**: no `window`-serial (a sliding window that advances dataset-by-dataset) |
| camera spin | `'spin'` | — | missing; arguably meaningless as defined (only the camera moves), but "spin each dataset in turn" is a coherent unimplemented idea |
| point-cloud morph | — | `'morph'` (inherently sequential) | **missing parallel analog**: no "all datasets morph simultaneously toward their own targets" |

Concretely, exactly **three** things are missing or asymmetric:

1. **`'window'` has no serial analog.** `'window'` is defined as "bullettime
   minus the precog and chemtrail parts" (`plot.py:1296-1310`) and always runs
   all datasets in lockstep. There is no way to get a sliding window that walks
   through datasets one at a time. This is the cleanest, most defensible gap.
2. **`'morph'` has no parallel analog.** Morph draws a *single* traveling cloud
   through datasets in list order; there is no mode where N clouds morph at once.
3. **Backend asymmetry for `'serial'` + trails.** Identical user code produces a
   trailed animation on matplotlib and a plain opaque reveal (plus a warning) on
   plotly. This is the only case where the *same* call means different things by
   backend, and is the highest-value item to close.

Also worth noting: the trail flags are *flags*, not styles, so they multiply
against styles rather than participating in the parallel/serial duality
themselves. If the maintainer wants the principle applied literally, the natural
refactor is to make "serial-ness" an orthogonal axis (e.g. `serial=True`)
composable with every style, rather than one value inside the `animate=`
enum — today `'serial'` occupies a slot that `'window'`/`'morph'`/`'spin'`
cannot share.

---

## 2. Per-dataset titles / labels / names / legend — PARTIAL (per-frame title: ABSENT)

### 2.1 What each kwarg does

| kwarg | scope | doc | render site |
|-|-|-|-|
| `title=` | **whole figure**, `str` only | `plot.py:950-951` — verbatim: `title : str` / `A title for the plot` | `matplotlib_backend.py:2555-2557` → `ax.set_title(title, ...)` |
| `names=` | **per DATASET** | `plot.py:882-893` | resolved `plot.py:3789+`; turns the legend on, one entry per dataset |
| `labels=` | **per OBSERVATION (row)** — flat (one per row across all datasets) or nested (one sub-list per dataset) | `plot.py:895-910` | mpl `ax.annotate`; plotly `layout.(scene.)annotations` |
| `legend=` | `True` / `str` / list, **one entry per drawn dataset/group** | `plot.py:923-928` | standard legend |

`names=` is mutually exclusive with a `legend=` list, and raises with a
categorical `hue=` (which regroups traces so they are no longer the named
datasets).

### 2.2 Does `title=` accept a list? — **No, and it fails silently**

```
title=str      : OK title='My Title'          lines=3 warns=[]
title=list     : OK title="['A', 'B', 'C']"   lines=3 warns=[]
title=tuple    : OK title="('A', 'B', 'C')"   lines=3 warns=[]
```

A list is neither rejected nor honored — `ax.set_title` stringifies it, so the
literal text `['A', 'B', 'C']` is drawn as the title. **No error, no warning.**
This is the one genuinely silent misbehavior found in this audit.

### 2.3 Per-dataset title that changes frame-by-frame during a serial-style animation — **ABSENT**

Tested by driving `anim.animation._draw_frame(n)` for frames 0-3 and reading
`ax.get_title()` after each:

```
title=list+serial : titles across frames: ["['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']"]
title=list+morph  : titles across frames: ["['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']"]
title=list+window : titles across frames: ["['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']", "['A', 'B', 'C']"]
```

The title is set **once**, outside the frame-update callbacks. None of
`update_lines_serial` / `update_lines_parallel` / `update_lines_spin` /
the morph branch touches the title artist — confirmed by grepping `set_title`
in `matplotlib_backend.py`, whose **only** occurrences are the one-shot
`:2555` / `:2557`.

**Closest existing primitives** a per-frame title would build on:
* `names=` already carries one string per dataset (validated, backend-agnostic) —
  the natural source of per-frame text.
* `_sync_anim_labels` (`matplotlib_backend.py:809`) already updates *per-point*
  text artists per frame and already knows, for serial, which dataset is
  currently revealing (via `_hyp_global_idx` / `revealed`) — that is exactly the
  bookkeeping a frame-synced title needs.

So the feature is absent but the required plumbing (per-dataset strings +
per-frame "who is active" state) already exists in both places.

---

## 3. Native text support — SUPPORTED (including list-of-lists)

### 3.1 Which module

`hypertools/tools/text2mat.py` — signature at `:187-188`:

```python
def text2mat(data, vectorizer='CountVectorizer',
             semantic='LatentDirichletAllocation', corpus='wiki'):
```

Registries at `:14-23`:

```python
vectorizer_models = {'CountVectorizer': CountVectorizer,
                     'TfidfVectorizer': TfidfVectorizer}
texts = {'LatentDirichletAllocation': LatentDirichletAllocation, 'NMF': NMF}
```

### 3.2 Default embedding model

`hyp.plot(['a', 'b'])` uses **CountVectorizer → LatentDirichletAllocation**, with
the semantic model *not* fit on your data but loaded pretrained from the hosted
`'wiki'` corpus. `text2mat.py:290-313`:

```python
if corpus in ('wiki', 'nips', 'sotus',):
    ...
        semantic = load(corpus + '_model')
    ...
    corpus = np.array(load(corpus))
```

`'wiki_model'` is a pickled fitted sklearn `Pipeline` (`io/load.py:43`),
downloaded on first use and cached in `~/hypertools_data` (verified present on
this machine), hash-checked against `_EXAMPLE_DATA_SHA256` (`io/load.py:89`)
before unpickling. **The default path therefore needs network access once**;
subsequent calls are local (measured 0.3 s warm).

### 3.3 Choosing a different model

Three-tier string resolution for both `vectorizer=` and `semantic=`
(GH #198, `text2mat.py:36-50`, `:158` `_resolve_registry_name`,
`:89` `_hf_fallback_model`):

1. **scikit-learn** — `vectorizer`: `'CountVectorizer'`, `'TfidfVectorizer'`;
   `semantic`: `'LatentDirichletAllocation'`, `'NMF'`.
2. **gensim** (needs `pip install "hypertools[gensim]"`) —
   `vectorizer`: `'Word2Vec'`, `'Doc2Vec'`, `'FastText'`;
   `semantic`: `'LdaModel'`, `'LsiModel'`, `'HdpModel'`.
3. **Hugging Face** — any unresolved name is treated as a
   sentence-transformers model id (e.g. `'all-MiniLM-L6-v2'`) via
   data-wrangler's `datawrangler.zoo.text.apply_text_model`
   (`text2mat.py:125`). Needs the `text` extra (`pydata-wrangler[hf]`), which
   `[dev]` does **not** install.

Each also accepts a **dict** (`{'model': ..., 'kwargs': {...}}`; legacy
`{'model', 'params'}` still accepted), a **class**, or a **configured class
instance** (`text2mat.py:198-233`).

`corpus=` accepts `'wiki'`, `'nips'`, `'sotus'`, **your own list/list-of-lists of
strings**, or `None`. `semantic=None` means "skip the semantic stage entirely"
(`text2mat.py:246-253`) — the fastest fully-local configuration.
An unrecognized corpus *string* raises rather than being treated as a
one-document corpus (`text2mat.py:283-290`).

All three are surfaced directly on `plot()` (`plot.py:570-572`:
`vectorizer="CountVectorizer"`, `semantic="LatentDirichletAllocation"`,
`corpus="wiki"`).

### 3.4 Verified runs

Flat list of strings → **one dataset**:

```python
S = ['the cat sat on the mat', 'dogs bark loudly at night', 'machine learning is fun',
     'neural networks learn patterns', 'the mat was red and soft', 'barking dogs rarely bite']
f = hyp.plot(S, show=False, reduce='PCA')                       # defaults
# DEFAULT text path OK: lines= 1 npts [901] time 0.3s
```

Fully-local variant (no corpus download at all):

```python
f = hyp.plot(S, show=False, reduce='PCA', corpus=None, semantic=None)
# FLAT list of str (corpus=None,semantic=None): lines= 1 time 0.2s
```

**List of lists → one dataset per group — SUPPORTED:**

```python
f = hyp.plot([S[:3], S[3:]], show=False, reduce='PCA')
# DEFAULT list-of-lists OK: lines= 2 time 0.0s

f = hyp.plot([S[:3], S[3:]], show=False, reduce='PCA', corpus=None, semantic=None)
# LIST OF LISTS: lines= 2 npts per line= [901, 901]
#   colors= [(0.86, 0.3712, 0.34), (0.34, 0.8288, 0.86)]
```

Two traces, two distinct palette colors — i.e. one dataset per group, exactly as
wanted. Nested-string detection helpers live at `plot.py:4993` `_flatten_nested`,
`:5009` `_iter_leaves`, `:5017` `_contains_string`.

---

## 4. Palette from an image file — ABSENT

Repo-wide grep over package source only:

```
$ grep -rn "PIL|imread|imageio|dominant|swatch|from_image" --include="*.py" hypertools/
hypertools/plot/plotly_backend.py:1699:    from PIL import Image      # frame rendering for animation EXPORT
hypertools/plot/plotly_backend.py:2184:    dominant visual element).  # prose in a comment
hypertools/plot/plot.py:4686:  # colorbar swatches ...             # prose in a comment
hypertools/plot/density.py:78,86:                                    # prose in comments
hypertools/io/streaming.py:37,414:                                   # PIL error-message text (GIF/video writing)
```

Every hit is **figure export** (writing PNG/GIF frames) or comment prose. There
is **no** image-decoding, color-quantization, or dominant-color extraction code
anywhere in `hypertools/`. `hypertools/plot/colors.py` (341 lines) reads no files
at all.

### What `colors.py` does support

`_get_palette` (`colors.py:287-309`) is the resolver; the public wrapper is
`get_palette_colors` (`colors.py:227`), plus `continuous_colormap` (`:250`) and
`_continuous_palette` (`:269`, with cyclic-palette handling for `'hls'`/`'husl'`).
`palette=` accepts (`colors.py:47-53`, `:287-309`):

* a **seaborn/matplotlib palette name** (`str`) → `sns.color_palette(...)`;
* an **explicit list of colors** (hex, names, RGB tuples) — if shorter than
  `n_colors`, the entries are used as anchors with seaborn `blend_palette`
  semantics;
* a **`matplotlib.colors.Colormap`** instance.

`palette=None` raises. Related: `mat2colors` (`colors.py:24`) maps a data matrix
to colors, `colors2groups` (`:178`) inverts colors back to groups, and `hue=` /
`color=` / `colors=` / `color_reduce=` drive per-observation and per-dataset
coloring.

### Closest existing primitive — verified

An arbitrary list of hex colors passed as `palette=` is honored exactly:

```python
pal = ['#ff0000', '#00ff00', '#0000ff']
f = hyp.plot(D, palette=pal, show=False, reduce='PCA')
# custom hex palette -> line colors: ['#ff0000', '#00ff00', '#0000ff']

from hypertools.plot.colors import get_palette_colors
get_palette_colors(pal, 3)
# [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
```

So palette-from-image is a **pure user-space add-on**: extract N dominant colors
yourself (e.g. PIL + k-means), hand the resulting hex list to `palette=`.
Nothing in hypertools needs to change to consume it — only to *produce* it.

---

## 5. Forecasting in `plot()` — PARTIAL

### 5.1 What `predict=` and `t=` do

`plot(..., predict=None, t=10)` (`plot.py:550-551`). Docstring `plot.py:1213-1231`:

> If set, forecasts `t` new rows per input dataset (in the plotted,
> post normalize/reduce/align space) using the specified
> `hypertools.predict` model, e.g. 'Kalman', 'ARIMA', 'GaussianProcess' …
> and overlays one dashed, low-opacity (alpha 0.6) forecast trace per dataset in
> the SAME color as its source line (no separate legend entry). The drawn overlay
> prepends the last observed row so the dashed trace connects to the trajectory
> (`t + 1` drawn vertices); the forecast DATA itself … has exactly `t` rows

`t` (`plot.py:1233-1236`): *"Forecast horizon passed to `predict` (see
`hypertools.predict.common.resolve_t`); ignored unless `predict` is set
(default: 10)."* It accepts `int` **or datetime-like**.

Models: `hypertools/predict/predict.py:39`
`FORECASTERS = [Kalman, GaussianProcess, AutoRegressor, ARIMA, Laplace, Chronos]`,
plus alias `'GP' → 'GaussianProcess'` (`:44`), matched case-insensitively;
also accepts dict/class/instance forms.

Order in the pipeline: forecasting happens **last**, on the already
reduced/aligned/plotted coordinates — not on raw input.

### 5.2 What `_draw_forecast_overlays` renders

`hypertools/plot/plot.py:122-165`. Per source dataset it draws **one**
`ax.plot(...)` line with:

* `linestyle='--'`
* `color=` the **source line's own color** (`src_lines[i].get_color()`, `:151`)
* `alpha=0.6`
* `label='_nolegend_'` (so it never gains a legend entry — it is drawn after the
  legend is built, `:126-129`)
* optionally smoothed by `_interp_static_line` when `antialias=True` (`:149-150`)
* branches for 3-D / 2-D / 1-D (`:153-164`)

It returns the artist list so the `'spin'` path can `set_clip_on(False)` them.

### 5.3 Verified runs (50×4 cumsum, `reduce='PCA'`)

```
predict=ARIMA static   lines= 2 [(932, '-', None), (901, '--', 0.6)] warns []
predict=ARIMA t=1      lines= 2 [(932, '-', None), (900, '--', 0.6)] warns []
predict + animate=True ERROR NotImplementedError predict= is only supported with static plots
                       and with animate='spin' (which just rotates the camera around the static
                       forecast ove…
predict + animate=spin lines= 2 [(1, '-', None), (901, '--', 0.6)] warns []
predict + serial       ERROR NotImplementedError predict= is only supported with static plots …
```

Readings:

* **Static works.** 2 artists: the solid data line and a dashed alpha-0.6
  forecast line in the same color. (Point counts are post-antialias
  interpolation, not raw rows.)
* **`t=1` works** — no error, produces a (shorter) forecast overlay.
* **`animate=True` / `'serial'` raise `NotImplementedError`**, guard at
  `plot.py:2348`. Confirms the docstring (`plot.py:1226-1231`): NOT supported
  with `True`/`'parallel'`/`'serial'`/`'window'`/`'morph'`.
* **`animate='spin'` works** (commit `aa53f815` verified) — but it is the
  *static* overlay with the camera rotating around it. The head line shows 1
  point at frame 0 only because the spin callback had not yet drawn; the
  forecast is precomputed once and never updated per frame.

### 5.4 The three user-facing questions

| capability | verdict | evidence |
|-|-|-|
| forecast drawn **ahead of an animated head**, updating as the animation progresses | **ABSENT** | `NotImplementedError` at `plot.py:2348` for every time-progressing style; `'spin'` renders a *static* overlay (`_draw_forecast_overlays` docstring `plot.py:130-132`: *"Shared by the STATIC path and the `animate='spin'` setup (which only rotates the camera around this same static overlay)"*) |
| **earlier forecasts retained** on screen (accumulating ghost forecasts) | **ABSENT** | `_draw_forecast_overlays` draws exactly **one** artist per dataset (`plot.py:142` `for i, fc in enumerate(raw_forecasts)`), created once; no per-frame forecast recomputation or accumulation anywhere |
| native **forecast scoring** (error/accuracy metric) | **ABSENT** | no `score`/`metric`/`mse`/`rmse`/`r2` function anywhere in `hypertools/predict/` (grepped all of `arima.py`, `autoreg.py`, `chronos.py`, `common.py`, `gp.py`, `kalman.py`, `laplace.py`, `predict.py`) |

**Closest primitive for scoring:** `return_model=True` returns a dict whose
`'predict'` key holds the raw forecast arrays, so a user can score manually:

```python
out = hyp.plot(d, predict='ARIMA', t=5, show=False, reduce='PCA', return_model=True)
list(out.keys())
# ['fig', 'xform_data', 'animation', 'pipeline', 'models', 'predict']
```

---

## 6. MultiIndex DataFrames — SUPPORTED (GH #95 claim confirmed)

`plot()` special-cases a row-MultiIndex DataFrame (`x.index.nlevels >= 2`)
**before** the format_data/analyze/reduce pipeline (`plot.py:616-640`,
implementation in `hypertools/plot/multiindex.py`):

* expand into one **leaf** dataset per unique full index combination;
* run the normal pipeline on the leaves;
* **after** transforming, append one **mean** trajectory per unique
  value-combination of each non-leaf level, deepest → outermost;
* style by depth: `linewidth = 1 + (L - 1 - level_idx)`,
  `alpha = min(1.0, 1/(level_idx + 1) + 0.2)`.

### Verified with a 2-level index (outer `grp`, inner `item`), 4 combos × 10 rows

```python
full_idx = pd.MultiIndex.from_tuples([t for t in idx for _ in range(10)],
                                     names=['grp', 'item'])
df = pd.DataFrame(np.vstack(rows), index=full_idx, columns=[f"f{i}" for i in range(4)])
hyp.plot(df, show=False, reduce='PCA')
```

```
df shape (40, 4) nlevels 2
plain MultiIndex: lines=6
  colors=[(0.86,0.37,0.34), (0.86,0.37,0.34), (0.34,0.83,0.86), (0.34,0.83,0.86),
          (0.86,0.37,0.34), (0.34,0.83,0.86)]
  lw= [1.0, 1.0, 1.0, 1.0, 2.0, 2.0]   alpha= [0.7, 0.7, 0.7, 0.7, 1.0, 1.0]
  warns: []
```

6 traces = 4 leaves + 2 group means. **Colored by group**: lines 0-1 (both `g1`
items) share one color, lines 2-3 (`g2`) share another, and each group mean
(lines 4-5) matches its group. Widths/alphas match the documented formula
exactly (leaves lw 1.0 / alpha 0.7; means lw 2.0 / alpha 1.0).

### Continuous `hue=` **is ignored**, with an explicit warning — CONFIRMED

```python
hyp.plot(df, hue=np.linspace(0, 1, len(df)), show=False, reduce='PCA')
```

```
MultiIndex + continuous hue: lines=6 collections=6
  colors= [(0.86,0.3712,0.34), (0.86,0.3712,0.34), (0.34,0.8288,0.86),
           (0.34,0.8288,0.86), (0.86,0.3712,0.34), (0.34,0.8288,0.86)]
  warns: ['x has a row MultiIndex (GH #95): MultiIndex grouping (leaf traces +
           per-level averages) takes precedence over hue=; ignoring hue.']
```

Colors are byte-identical to the no-hue run — the continuous hue had **zero**
effect. Warning emitted at `hypertools/plot/plot.py:2682`. GH #95 claim verified
in both halves: grouping colors win, and continuous `hue=` is ignored (loudly,
not silently).

---

## 7. `hyp.load()` datasets — SUPPORTED (no weather/temperature dataset)

`hypertools/io/load.py:162`. Returns **raw data** (arrays / DataFrames / lists of
strings / fitted `Pipeline` for `*_model`) — never a `DataGeometry`.

### Built-in example datasets — `EXAMPLE_DATA`, `io/load.py:25-46` (exhaustive, 17 keys)

All hosted remotely (Dropbox for data, Google Drive for the three `*_model`
pickles), SHA-256-pinned at `io/load.py:89` and cached under `~/hypertools_data`.

| name | what it is |
|-|-|
| `weights` | fMRI-style weight matrices (list of arrays) |
| `weights_avg` | averaged version (list of arrays) |
| `weights_sample` | sampled subset (list of arrays) |
| `spiral` | synthetic spiral trajectories (list of arrays) |
| `mushrooms` | UCI mushroom table (DataFrame) |
| `wiki` | Wikipedia article text (object array of document strings) |
| `nips` | NIPS paper text (object array of document strings) |
| `sotus` | 29 State-of-the-Union speeches (flat list of strings) |
| `bunny` | Stanford bunny point cloud (array) |
| `cube` | cube point cloud (array) |
| `dragon` | dragon mesh point cloud (array) |
| `sphere` | sphere point cloud (array) |
| `teapot` | Utah teapot point cloud (array) |
| `vase` | vase point cloud (array) |
| `biplane` | biplane point cloud (DataFrame) |
| `datasaurus` | Datasaurus Dozen, 13 DataFrames with original indexes restored |
| `wiki_model` / `nips_model` / `sotus_model` | pretrained fitted sklearn `Pipeline`s (hash-verified before unpickling) |

### scikit-learn bundled — `SKLEARN_DATASETS`, `io/sources.py:185-192` (exhaustive)

```python
SKLEARN_DATASETS = {
    'iris': 'load_iris',
    'digits': 'load_digits',
    'wine': 'load_wine',
    'breast_cancer': 'load_breast_cancer',
    'diabetes': 'load_diabetes',
    'linnerud': 'load_linnerud',
}
```

Bundled with sklearn (no download). Network-fetched `fetch_*` datasets are
deliberately excluded (`io/sources.py:211-215`).

### Full resolution chain (`io/load.py:179-222`, dispatch `:415-449`)

A string is tried in this order:

1. built-in `EXAMPLE_DATA` name (**always wins**)
2. scikit-learn bundled name (**wins over seaborn** for shared names like `iris`)
3. **seaborn** dataset — *any* name from `seaborn.get_dataset_names()`
   (`'penguins'`, `'tips'`, `'titanic'`, …); network lookup, cached per-process,
   skipped if unreachable
4. `'fivethirtyeight/<slug>'` (explicit prefix)
5. `'kaggle/<owner>/<dataset>'` (explicit prefix, needs `hypertools[kaggle]`)
6. local file (`.geo`/pickle, `.npy`/`.npz`, `.csv`/`.tsv`/`.txt`, `.json`,
   `.parquet`, `.mat`, `.xlsx`/`.xls`, `.gz` variants; extensionless files
   content-sniffed)
7. Hugging Face dataset id (e.g. `'scikit-learn/iris'`; `streaming=True` supported)
8. Google Sheets URL
9. Google Drive URL or bare file id
10. Dropbox URL / shared-link path
11. any other URL (scheme optional)

`load()` also accepts a **list/tuple** of any of the above (`io/load.py:400-406`),
and a path-like.

### Weather / temperature / climate — **NONE built in**

No key in `EXAMPLE_DATA` or `SKLEARN_DATASETS` is weather-, temperature-, or
climate-related. The nearest native routes, all requiring network and none
bundled:

* the **seaborn** tier (step 3) — verified live via `sns.get_dataset_names()`
  (22 names): `anagrams, anscombe, attention, brain_networks, car_crashes,
  diamonds, dots, dowjones, exercise, flights, fmri, geyser, glue, healthexp,
  iris, mpg, penguins, planets, seaice, taxis, tips, titanic`. The only
  climate-adjacent entry is **`seaice`** (Arctic sea-ice extent) — reachable as
  `hyp.load('seaice')`. There is no temperature/weather series. Note this list
  is itself fetched over the network (cached per-process, `io/sources.py:194-197`),
  so it is not enumerable offline and may drift with seaborn-data;
* **`'fivethirtyeight/<slug>'`** (step 4) for any 538 weather dataset;
* **`'kaggle/<owner>/<dataset>'`** (step 5) or a Hugging Face id (step 7);
* a **plain URL / local CSV** (steps 6, 11) — the simplest path for
  NOAA/Berkeley-Earth-style temperature data.

Note: `examples/animate_weather_decades.py` and
`docs/tutorials/weather_decades.ipynb` exist in the working tree (untracked),
so weather demos are being written against externally-sourced data, not a
built-in dataset.

---

## Summary table

| # | area | verdict | headline |
|-|-|-|-|
| 1 | animation style matrix | **PARTIAL** | serial+trails verified on matplotlib (commit 26843931 is real); plotly ignores it. `'window'` has no serial analog, `'morph'` no parallel analog |
| 2 | per-dataset titles | **PARTIAL** | `names=` is per-dataset; `title=` is figure-wide `str` and **silently stringifies a list**; per-frame title during serial/morph/window is **ABSENT** |
| 3 | native text | **SUPPORTED** | `text2mat.py:187`; default CountVectorizer→LDA pretrained on hosted `'wiki'`; sklearn→gensim→HF name tiers; **list-of-lists gives one dataset per group** (verified: 2 traces) |
| 4 | palette from image | **ABSENT** | no image decoding anywhere in `hypertools/`; closest primitive is `palette=['#ff0000', …]`, verified exact |
| 5 | forecasting in plot | **PARTIAL** | static + `'spin'` only; `animate=True`/`'serial'` raise `NotImplementedError` (`plot.py:2348`); no retained forecasts, no scoring |
| 6 | MultiIndex | **SUPPORTED** | 4 leaves + 2 means, colored by group, lw/alpha per depth; continuous `hue=` ignored with a warning (`plot.py:2682`) — GH #95 confirmed |
| 7 | `hyp.load()` | **SUPPORTED** | 17 `EXAMPLE_DATA` + 6 sklearn names + 9 more resolution tiers; **no weather/temperature dataset** — closest is seaborn's `seaice` via the seaborn tier |
