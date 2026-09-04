# Tutorial audit: "does it showcase NATIVE hypertools?"

**Date:** 2026-07-26
**Branch:** dev-1.0
**Scope:** the 15 pre-existing notebooks in `docs/tutorials/` (the five newest --
`conversation_shape`, `market_forecast`, `morph_shapes_zoo`, `painting_embeddings`,
`weather_decades` -- are covered by a separate agent).
**Pass type:** INVENTORY ONLY. No rewrites proposed; findings are "what is
hand-rolled" + "what should replace it".

**Method.** Every notebook was parsed as JSON and every code cell read in full.
Cell indices below are 0-based indices into `nb['cells']` (counting markdown
cells), matching what `json.load(open(nb))['cells'][i]` returns. Every "this is
already native" claim is cited to a line in `hypertools/` source, verified in
this session.

---

## 0. Verified native-API facts used throughout this report

These are the citations that back the per-notebook findings. All were read from
source on dev-1.0 in this session.

| Claim | Source citation |
|-|-|
| `vectorizer='<any HF sentence-transformers id>'` embeds text natively | `hypertools/tools/text2mat.py:89` `_hf_fallback_model(name)`: *"Wrap an unresolved vectorizer=/semantic= string as a pretrained Hugging Face sentence-transformers embedding model -- the third and final tier of the sklearn -> gensim -> HuggingFace name-resolution order (GH #198), built on data-wrangler's existing HF text-embedding support (`datawrangler.zoo.text.apply_text_model`)"*; dispatch at `text2mat.py:184` `registry[name] = _hf_fallback_model(name)  # tier 3: HuggingFace`; transform at `text2mat.py:125-128` `from datawrangler.zoo.text import apply_text_model; embedded = apply_text_model(name, list(X), mode='transform')` |
| `semantic=None` skips the topic model (embedding-only path) | `text2mat.py:391-399`: `if semantic: ... else: tmodel = None` |
| `corpus=None` fits on the passed data instead of a hosted corpus | `text2mat.py:404-407`: `if corpus is None: _fit_models(vmodel, tmodel, data, model_is_fit)` |
| The HF text tier ships as an extra | `pyproject.toml:92` `text = ["pydata-wrangler[hf]>=0.5.1"]` |
| Animated `save_path='*.gif'` needs **no ffmpeg** | `plot.py:1246-1250`: *"ANIMATED matplotlib plots accept .gif, .png/.apng (animated PNG), and .svg (animated vector graphics) with no extra dependencies, plus the video formats .mp4/.mov/.avi/.m4v/.mkv, which -- and ONLY which -- require FFmpeg"*; writer dispatch at `hypertools/plot/animate.py:80-116` (`ext = os.path.splitext(save_path)[1]...`); `HyperAnimation.save` docstring `hyper_animation.py:82-84` |
| `manip='Smooth'` (kernel `boxcar`/`gaussian`/`savgol`) is native | `hypertools/manip/manip.py:32` `MANIPULATORS = [Normalize, ZScore, Smooth, Resample]`; `hypertools/manip/smooth.py:14` `KERNELS = ('savgol', 'gaussian', 'boxcar')`; exposed on plot at `plot.py:1064-1076` (`manip : model spec or None`) |
| `plot(..., predict=..., t=...)` draws the dashed forecast overlay | `plot.py:1213-1231`: *"forecasts `t` new rows per input dataset ... and overlays one dashed, low-opacity (alpha 0.6) forecast trace per dataset in the SAME color as its source line"* |
| `plot(..., ax=ax)` draws into a caller-supplied axes | `plot.py:1611-1612` `ax : matplotlib.Axes / Axis handle to plot the figure`. **Runtime-verified** this session: `hyp.plot([d, d+1], ['-','--'], reduce=None, ndims=2, ax=ax, legend=['a','b'], show=False)` -> `Figure`, 2 lines on the supplied axes |
| `hue=` handles categorical labels, continuous arrays, matrices, and per-dataset nesting; `legend=True` derives group names from `hue` | `plot.py:821-871` (`hue : list, numpy array, pandas Series/Index/Categorical, or 2D matrix`, incl. *"When the data is a list of datasets, `hue` may mirror that nesting -- one hue sub-sequence per dataset"*) |
| `color_reduce=` maps an arbitrary matrix hue to RGB | `plot.py:872-881` |
| `hyp.describe()` scores reduction quality | `hypertools/reduce/describe.py:13-23` |
| Streaming (`stream_init`/`stream_chunk`/`stream_max`/`stream_window`) and `hyp.io.lsl_stream()` are native | `plot.py:1621-1667`; `hypertools/io/lsl.py:36` |
| **No** native text chunking / sentence-windowing exists | grep for chunking helpers in `hypertools/**/*.py` returns only `stream_chunk` and animation "sliding window" hits |
| **No** native time-delay (Takens) embedding exists | grep `takens\|time.delay\|delay_embed` over `hypertools/` -> 0 hits |
| **No** native forecast scoring/backtest utility exists | grep `def score\|mape\|MAE\|backtest` over `hypertools/predict/*.py` -> 0 hits |

---

## 1. Ranking, worst to best

"hyp lines" counts non-blank, non-comment code lines matching `\bhyp\.|\bhypertools\b`;
"install" is the Colab `%pip install` / `find_spec` preamble; "other" is everything
else (some of it legitimate domain setup, some of it the findings below).

| Rank | Notebook | code lines | hyp | install | other | % hyp | verdict |
|-|-|-|-|-|-|-|-|
| 1 (worst) | `conversation_trajectories.ipynb` | 118 | 3 | 6 | 109 | 2.5% | hand-rolled HF embedding + hand-rolled sentence windows + redundant ffmpeg |
| 2 | `projectile_kalman.ipynb` | 156 | 5 | 3 | 148 | 3.2% | 3 hand-rolled matplotlib figures; 59 lines of archive plumbing |
| 3 | `stock_forecasting.ipynb` | 164 | 6 | 6 | 152 | 3.7% | hand-rolled backtest loop, rolling-mean smoothing, 2x2 mpl grid |
| 4 | `wikipedia_embeddings.ipynb` | 98 | 7 | 6 | 85 | 7.1% | hand-rolled HF embedding x2 + redundant ffmpeg |
| 5 | `lsl_streaming.ipynb` | 49 | 4 | 2 | 43 | 8.2% | 29-line synthetic outlet (unavoidable, but nothing else is taught) |
| 6 | `hugging_face_embeddings.ipynb` | 50 | 5 | 2 | 43 | 10.0% | hand-rolled HF embedding, *while its own markdown documents the native call* |
| 7 | `modern_sklearn_dynamics.ipynb` | 49 | 5 | 2 | 42 | 10.2% | hand-rolled delay embedding + redundant ffmpeg |
| 8 | `streaming_data.ipynb` | 34 | 4 | 1 | 29 | 11.8% | clean; the 15-line generator is the point of the tutorial |
| 9 | `text.ipynb` | 90 | 12 | 2 | 76 | 13.3% | hand-rolled Wikipedia fetch + chunker (35 lines) |
| 10 | `analyze.ipynb` | 30 | 6 | 1 | 23 | 20.0% | **never calls `hyp.plot` at all**; 5x seaborn heatmap |
| 11 | `align.ipynb` | 24 | 7 | 1 | 16 | 29.2% | clean |
| 12 | `plot.ipynb` | 71 | 21 | 1 | 49 | 29.6% | clean; the densest hypertools-per-line notebook |
| 13 | `normalize.ipynb` | 23 | 8 | 1 | 14 | 34.8% | clean |
| 14 | `reduce.ipynb` | 19 | 7 | 1 | 11 | 36.8% | clean (but never visualizes) |
| 15 (best) | `cluster.ipynb` | 18 | 7 | 1 | 10 | 38.9% | clean |

---

## 2. Recurring gaps (each appears in 3+ notebooks)

### G1. Hand-rolled `sentence-transformers` instead of `vectorizer=<hf-model-id>`
**Notebooks:** `hugging_face_embeddings` (cell 4), `wikipedia_embeddings` (cells 6, 17),
`conversation_trajectories` (cell 8).

All four sites do the same thing:
```python
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(headlines, show_progress_bar=False)
```
Native replacement (verified, section 0): `hyp.plot(headlines, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None, ...)`.

**This is a documented-but-unused feature.** `hugging_face_embeddings.ipynb` cell 3
(markdown) literally says: *"HyperTools can also do the whole thing in one call:
`hyp.plot(headlines, vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None, ...)`"*
-- and then cell 4 does it by hand anyway. `wikipedia_embeddings.ipynb` cell 5
(markdown) similarly says *"HyperTools can also embed text for you in a single c[all]..."*
before cell 6 hand-rolls it. The tutorials advertise the native path in prose and
demonstrate the non-native path in code.

### G2. `save_path='x.mp4'` + a 14-15 line ffmpeg `subprocess.run` to make a GIF
**Notebooks:** `conversation_trajectories` (cell 15, 14 lines), `hugging_face_embeddings`
(cell 13, 15 lines), `modern_sklearn_dynamics` (cell 13, 15 lines), `wikipedia_embeddings`
(cell 11, 14 lines). ~58 lines total.

Every one is byte-identical boilerplate: `shutil.which('ffmpeg')`, a palettegen/paletteuse
filter string, `subprocess.run([...], check=True, capture_output=True)`, `os.remove(mp4)`.

`save_path='foo.gif'` writes the GIF **directly, with no ffmpeg at all** (`plot.py:1246-1250`,
`animate.py:80-116`). The proof is in the same tutorial set: `streaming_data.ipynb` cell 4
and cell 8, and `lsl_streaming.ipynb` cell 6, all pass `save_path='*.gif'` to `hyp.plot`
and get a GIF. So four notebooks take the ffmpeg-dependent long way round to produce
exactly what three other notebooks get with one kwarg -- and each one prints an
"ffmpeg not found -- skipping GIF conversion" fallback for a dependency hypertools
does not need.

### G3. Hand-rolled matplotlib figures where `hyp.plot(..., ax=)` would do
**Notebooks:** `stock_forecasting` (cell 14, 24 lines: `plt.subplots(2,2)` + 3 `ax.plot`
calls per panel + manual legend/labels), `projectile_kalman` (cell 11, 17 lines: 1x3
panel grid; cell 15, 12 lines: 3-series x-vs-z comparison; cell 6, 8 lines: single
side-view line), `text` (none), `analyze` (cells 8/13/18/23/28: `sb.heatmap` x5).

`hyp.plot` accepts `ax=`, `reduce=None`, `ndims=2`, a per-dataset `fmt` list, and
`legend=[...]` -- runtime-verified in section 0, and the per-dataset-fmt-list pattern is
already demonstrated inside this same tutorial set at `plot.ipynb` cell 32
(`hyp.plot([data1_r, data2_r, missing_data], ['-', '--', '*'], legend=['Full', 'Missing', 'Missing Points'])`).
`projectile_kalman` cell 15 in particular (observed / actual / forecast, 3 series, one
legend, 2D) is a direct match for that shape.

### G4. Continuous/categorical color built by hand
Not a widespread problem -- `hue=` is used correctly nearly everywhere (`text` cells 8/10/17,
`hugging_face_embeddings` cells 6/10/12, `wikipedia_embeddings` cells 18/20,
`conversation_trajectories` cells 10/12/14, `modern_sklearn_dynamics` cell 10,
`plot` cells 26/29, `cluster` cell 21). Only two soft spots:
- `plot.ipynb` cells 26 and 29 duplicate a 3-line `np.array_split`-based hue builder
  verbatim (same 3 lines twice, cells 26 and 29) -- repetition, not a missing feature.
- No notebook in this set exercises `palette=`, `color_reduce=`, or `colorbar=` at all,
  despite all three being documented plot kwargs (`plot.py:807`, `:872`, `:930`).

### G5. Nothing exercises the trail/camera animation kwargs
`chemtrails=`, `precog=`, `bullettime=`, `focused=`, `zoom=`, `animate='window'`,
`animate='morph'`, `density=` on an animation, `surface=` -- none appear in any of the
15 notebooks (`modern_sklearn_dynamics` cell 15 markdown *mentions* `chemtrails=True`
as a "next step" but never runs it; `conversation_trajectories` cell 14 uses `zoom=1.5`).
`wikipedia_embeddings` cell 20 is the only `density=True` use.

---

## 3. Per-notebook findings

### 3.1 `conversation_trajectories.ipynb` -- WORST (2.5% hypertools)
**Teaches:** `hyp.plot(list_of_arrays, hue=..., animate='serial')`.
**hypertools calls:** 3 lines total (cell 12 `hyp.plot`, cell 14 `hyp.plot`, plus the import).
**Hand-rolled:** 109 lines.

| Cell | Lines | What it does | Native replacement / classification |
|-|-|-|-|
| 6 | 38 | `usable_utterances`, `split_sentences` (regex `(?<=[.!?])\s+`), `count_windows`, `substantial` -- corpus filtering + sentence splitting | **MISSING FEATURE (M1).** Corpus filtering is legitimately ConvoKit-specific, but `split_sentences` + `count_windows` are generic text-windowing that a text-aware plotting library plausibly owns. No native equivalent (verified: no chunking helper in `hypertools/`). |
| 8 | 23 | Builds 3-sentence sliding windows, `SentenceTransformer('all-MiniLM-L6-v2').encode(...)`, then manually re-splits the flat embedding matrix back into per-utterance arrays with a running `start` offset, `np.vstack([emb, emb])` to force >=2 rows | **REPLICATES NATIVE (G1)** for the embedding: `vectorizer='all-MiniLM-L6-v2', semantic=None, corpus=None` (`text2mat.py:89,184`). `hyp.plot` accepts a *list of lists of strings* (`plot.py:1606` `corpus : list (or list of lists) of text samples`; `format_data.py:124` `x : numpy array, dataframe, series, string or (mixed, possibly nested) list`), so the flat-encode-then-recarve dance is the manual version of what `format_data` does per dataset. The `vstack` duplication to dodge single-row datasets is a **MISSING FEATURE (M2)**: `hyp.plot` should handle a 1-row dataset in a list without the caller faking a second row. |
| 10 | 7 | Builds `speaker_hue` as a nested list-of-lists | **CORRECT NATIVE USE.** `plot.py:865-868` documents exactly this nesting. Keep. |
| 12, 14 | 4 + 6 | `hyp.plot(..., hue=, legend=True)` and `hyp.plot(..., animate='serial', duration, rotations, frame_rate, zoom, save_path='...mp4')` | **CORRECT NATIVE USE** -- the two good cells in the notebook. |
| 15 | 14 | ffmpeg mp4->gif | **REDUNDANT (G2).** Change cell 14's `save_path` to `.gif` and delete cell 15 entirely. |

**Ratio:** ~3 hypertools lines : ~109 scaffolding lines (~1:36).

### 3.2 `projectile_kalman.ipynb` (3.2%)
**Teaches:** `hyp.impute(model='Kalman')`, `hyp.predict(model='Kalman', t=)`, `hyp.plot(predict=, t=)`.
**hypertools calls:** 5 (cells 9, 13, 17, plus import).

| Cell | Lines | What it does | Classification |
|-|-|-|-|
| 3 | 36 | Download a 6 MB `.7z` from GitHub, cache to `~/.hypertools_cache`, `py7zr` extract | **LEGITIMATE domain setup**, but note `hyp.load()` already owns URL fetching + caching (`hypertools/io/load.py:162`). **MISSING FEATURE (M3):** `hyp.load(<url>)` does not cover archive members (`.7z`/`.zip` -> inner file), so the tutorial hand-rolls a cache dir + extractor that duplicates loader responsibilities. |
| 4 | 23 | Parse SportVU JSON, pull the ball sentinel rows | Legitimate. |
| 6 | 8 | `plt.subplots` + `ax.plot(arc['x_ft'], arc['z_ft'])` side view | **G3.** `hyp.plot(arc[['x_ft','z_ft']], '-o', reduce=None, ndims=2, xlabel=..., ylabel=..., title=...)` (`plot.py:1013` `xlabel, ylabel, zlabel : str or None`; runtime-verified `reduce=None, ndims=2`). |
| 8 | 25 | Injects NaNs: an occlusion slice plus ~10% scattered single-cell dropouts, including a 5-line comment about `.to_numpy()` read-only/Fortran-order semantics and `.flat` views | **MISSING FEATURE (M4).** "Damage a dataset so you can score an imputer" is a canonical demo need for a library that ships `hyp.impute`. The notebook is forced to explain numpy memory-layout gotchas that have nothing to do with hypertools. |
| 9 | 17 | `hyp.impute(damaged, model='Kalman')` (2 lines) then 15 lines of per-axis RMSE bookkeeping | **MISSING FEATURE (M5).** No native imputation/forecast scoring exists (verified: no `score`/`MAE` in `hypertools/predict/` or `impute/`). |
| 11 | 17 | 1x3 matplotlib panel grid: `axvspan`, true vs imputed lines, `scatter` of imputed entries | **G3 (partial).** Per-feature-vs-time panels are not what `hyp.plot` draws, so this is defensible; but the "highlight the imputed entries" overlay is a plausible native affordance. |
| 13 | 9 | `hyp.predict(first30, model='Kalman', t=HORIZON)` (1 line) + 8 lines of MAE bookkeeping | **M5** again. |
| 15 | 12 | 3-series x-vs-z comparison figure by hand | **G3, direct hit.** Same shape as `plot.ipynb` cell 32's `hyp.plot([a, b, c], ['-','--','*'], legend=[...])`. |
| 17 | 3 | `hyp.plot([first30], predict='Kalman', t=HORIZON, reduce=None, fmt='-o', ...)` | **CORRECT NATIVE USE** -- and it makes cell 15 look redundant: cell 15 draws by hand what cell 17 draws in one call (minus the "actual" series). |

**Ratio:** ~5 hypertools lines : ~148 scaffolding lines (~1:30). Roughly 59 of those
148 are data acquisition, ~57 are matplotlib, ~32 are scoring.

### 3.3 `stock_forecasting.ipynb` (3.7%)
**Teaches:** `hyp.predict` model comparison, `return_model=True` reuse, `hyp.plot(predict=, t=)`.

| Cell | Lines | What it does | Classification |
|-|-|-|-|
| 3 | 39 | `fetch_prices()` -- yfinance with a 3-attempt retry loop and CSV-snapshot fallback | **LEGITIMATE** (network robustness), though `hyp.load` already has retry/caching machinery for its own sources. |
| 6 | 23 | Backtest loop: per ticker, split train/holdout, naive baseline, `hyp.predict(...)` (1 line), compute MAE/MAPE per (ticker, model) | **MISSING FEATURE (M5/M6).** The only hypertools content is one `hyp.predict` call inside 22 lines of split/score/tabulate. A native `hyp.predict(..., holdout=)` or scoring helper would invert that ratio. |
| 7, 8, 9 | 5+5+18 | pivot tables, sorting, "did it beat naive" prose logic | **M6** (model-comparison reporting). Pure pandas. |
| 12 | 14 | Builds a 2-column (log_close, smoothed log_volume) frame per ticker using `pandas.rolling(smooth, min_periods=1).mean()` | **REPLICATES NATIVE.** `manip={'model': 'Smooth', 'kwargs': {'kernel': 'boxcar', 'kernel_width': 10}}` is a `hyp.plot` kwarg (`plot.py:1064-1076`; `manip/smooth.py:14` `KERNELS = ('savgol','gaussian','boxcar')`) and runs at the canonical first pipeline stage. The tutorial does the smoothing in pandas and never mentions `manip=`. |
| 12 (last 3 lines) | 3 | `hyp.plot(train_2col, predict='Kalman', t=HOLD, legend=tickers, ndims=2, ...)` | **CORRECT NATIVE USE** -- the notebook's best cell. |
| 14 | 24 | 2x2 `plt.subplots` grid; per panel, 3 `ax.plot` calls (train / held-out / forecast) plus labels, one manual `legend`, `suptitle`, `tight_layout` | **G3.** `hyp.plot([...], [...], ax=ax, reduce=None, ndims=2, legend=[...])` per panel (runtime-verified). Also re-runs `hyp.predict` inside the plotting loop, duplicating cell 6's work -- `plot(predict=...)` would fold both together. |
| 16 | 16 | `return_model=True` reuse demo (2 hypertools lines) + 14 lines of pooling/scoring | **M5** again for the scoring half; the `return_model` demo itself is good. |

**Ratio:** ~6 hypertools lines : ~152 scaffolding lines (~1:25).

### 3.4 `wikipedia_embeddings.ipynb` (7.1%)
**Teaches:** `reduce='UMAP'` dict spec, `cluster='GaussianMixture'`, `animate='spin'`,
`hue=`, `density=True`.

| Cell | Lines | Classification |
|-|-|-|
| 4 | 5 | `hyp.load('wiki')` + `str(doc) for doc in wiki[0].ravel()` -- **minor gap:** the loader hands back an object array the user must unwrap by hand. Same 2-line unwrap appears in `text.ipynb` cells 13 and 15. **MISSING FEATURE (M7):** `hyp.load('wiki')` should be able to return a plain `list[str]`. |
| 6 | 4 | `SentenceTransformer('BAAI/bge-small-en-v1.5').encode(...)` -- **G1.** `vectorizer='BAAI/bge-small-en-v1.5', semantic=None, corpus=None` (`text2mat.py:89`). Cell 5's own markdown says hypertools can do this in one call. |
| 8, 10, 18, 20 | 4+6+4+4 | `hyp.plot(..., reduce={'model':'UMAP',...}, cluster=..., hue=, legend=True, density=True, animate='spin', save_path=)` -- **CORRECT NATIVE USE**, the strongest native content in the whole set. |
| 11 | 14 | ffmpeg mp4->gif -- **REDUNDANT (G2).** |
| 15 | 31 | `wikipediaapi` live fetch loop over 4 keyword groups | **LEGITIMATE** (live-data demo, GH #187), though 31 lines is a lot of preamble for 4 lines of payoff. |
| 17 | 4 | Second hand-rolled `model.encode(...)` -- **G1** again. |

**Ratio:** ~7 hypertools lines : ~85 scaffolding lines (~1:12). Removing G1 (8 lines)
and G2 (14 lines) would take the notebook to ~63 scaffolding lines with no loss of content.

### 3.5 `lsl_streaming.ipynb` (8.2%)
**Teaches:** `hyp.io.lsl_stream()` -> `hyp.plot(stream, stream_init=, stream_chunk=, stream_max=)`.

- Cell 4 (29 lines): builds a real `pylsl.StreamOutlet` on a background daemon thread
  (`sample_for_index`, `start_synthetic_outlet`, thread + `threading.Event` stop flag).
  **MISSING FEATURE (M8).** The cell's own markdown says *"This is the exact pattern used
  by hypertools' own test suite (`tests/test_lsl_streaming.py`)"* -- i.e. hypertools has
  this code internally and makes every user retype it. A `hyp.io.lsl_test_outlet()` (or
  documented test helper) would cut this notebook's scaffolding by 60%. There is no
  outlet-side helper in `hypertools/io/lsl.py` (only the inlet-side `lsl_stream`, line 36).
- Cell 6 (5 lines) and cell 9 (4 lines) are correct native use, including
  `save_path='lsl_stream.gif'` -- **this notebook is one of the three that proves G2**.
- **Ratio:** ~4 hypertools lines : ~43 scaffolding lines (~1:11), essentially all of it
  the synthetic outlet.

### 3.6 `hugging_face_embeddings.ipynb` (10.0%)
**Teaches:** `hue=`, `cluster='GaussianMixture'`, `reduce='UMAP'`, `animate='spin'`.

- Cell 4 (8 lines): `load_dataset(...)` (legitimate) + `SentenceTransformer(...).encode(...)`
  -- **G1, the flagship instance**, because cell 3's markdown states the native
  one-call form verbatim and the code ignores it.
- Cells 6, 8, 10, 12 (2+2+3+3 lines): textbook native usage. Good.
- Cell 13 (15 lines): ffmpeg -- **REDUNDANT (G2)**.
- **Ratio:** ~5 hypertools lines : ~43 scaffolding lines (~1:9); ~23 of those 43 are G1+G2.

### 3.7 `modern_sklearn_dynamics.ipynb` (10.2%)
**Teaches:** `cluster='HDBSCAN'`, `cluster='GaussianMixture'` + `n_clusters`,
continuous `hue`, `animate=True`.

- Cell 8 (7 lines): Lorenz system + `solve_ivp` -- **LEGITIMATE**.
- Cell 10 (6 lines): builds the time-delay embedding by hand,
  `np.column_stack([x[i*tau : i*tau+n] for i in range(dims)])` -- **MISSING FEATURE (M9).**
  A delay/lag embedding is a *manipulator* in the exact sense of `hypertools/manip/`
  (alongside `Smooth`/`Resample`), and the notebook's markdown frames it as a first-class
  technique ("Takens' embedding theorem"). No native equivalent exists (verified: 0 hits
  for `takens|time.delay|delay_embed`).
- Cells 4, 6, 10 (tail), 12: correct native use.
- Cell 13 (15 lines): ffmpeg -- **REDUNDANT (G2)**.
- **Ratio:** ~5 hypertools lines : ~42 scaffolding lines (~1:8); 15 of the 42 are G2.

### 3.8 `streaming_data.ipynb` (11.8%) -- CLEAN
**Teaches:** stream detection, `stream_init/chunk/max/window`, `fig.stream_info`, HF `IterableDataset`.
- Cell 4's 15-line `live_feed()` generator *is* the subject matter (you cannot demo
  streaming without a stream). Cells 4/8/10 use `save_path='*.gif'` natively -- **G2 proof**.
- Only nit: cell 6's 3-line dict-comprehension pretty-printer of `stream_info`.
- **No findings.**

### 3.9 `text.ipynb` (13.3%)
**Teaches:** native text plotting (`vectorizer`/`semantic`/`corpus` defaults), `hyp.load('nips'/'wiki'/'sotus')`, `reduce=`/`cluster=` on text.

- Cell 4 (35 lines): `wiki_text()` MediaWiki-API fetcher with User-Agent + fallback
  snippets, plus a `chunk(s, count)` splitter. The fetcher is **LEGITIMATE**; `chunk()`
  is **MISSING FEATURE (M1)** -- the same generic text-splitting need as
  `conversation_trajectories` cell 6. Two notebooks independently hand-rolling a text
  chunker is the signal.
- Cells 8, 10, 13, 15, 17, 19, 21: excellent native usage -- raw `list[str]` straight into
  `hyp.plot`, `hue=`, `labels=`, `reduce=` dict spec, `cluster='HDBSCAN'`, `corpus='nips'`.
  **This notebook is the reference for what "showcases native functionality" means.**
- Cells 13, 15: the `[str(x) for x in data[0].ravel()]` unwrap (**M7**, see 3.4).
- **Ratio:** ~12 hypertools lines : ~76 scaffolding lines (~1:6), but ~35 of the 76 are
  the one data-fetch cell.

### 3.10 `analyze.ipynb` (20.0%) -- structural gap
**Teaches:** `hyp.analyze(normalize=, reduce=, ndims=, align=)`.

- **`hyp.plot` is never called.** Verified: the only hypertools calls in the notebook are
  `hyp.load` x1 and `hyp.analyze` x4. A pipeline tutorial that shows the result of
  normalize->reduce->align only as `sb.heatmap(x)` never demonstrates why the pipeline
  exists.
- Cells 8, 13, 18, 23, 28 (3 lines each, 15 total) are the *same* copy-pasted seaborn
  loop: `for x in <result>: sb.heatmap(x); plt.show()`. Reduced-to-3D output (cells 18, 23,
  28) is exactly what `hyp.plot(result)` is for.
- **MISSING FEATURE (M10):** no native way to view a data matrix. `hyp.describe()`
  (`reduce/describe.py:13`) covers reduction *quality* but not "show me the matrix", so
  seaborn is a reasonable stand-in for cells 8/13 -- but not for 18/23/28.
- **Ratio:** ~6 hypertools lines : ~23 scaffolding lines, 15 of which are the repeated
  heatmap loop.

### 3.11 `align.ipynb` (29.2%) -- CLEAN
`hyp.load('weights')` -> `hyp.align(data)` / `hyp.align(data, model='SRM')` -> `hyp.plot`.
The only non-hypertools code is `np.mean(data[:18], 0)` group-averaging (3x, 2 lines each),
which is the scientific point being made (align-then-average vs. average-then-align),
not scaffolding. **No findings.**

### 3.12 `plot.ipynb` (29.6%) -- CLEAN, the model tutorial
21 hypertools lines across 20 code cells; every cell is a `hyp.plot` kwarg demo
(`fmt`, `ndims`, `reduce` string + dict, `hue`, `legend`, `labels`, `n_clusters`,
`normalize`, `align` dict spec, `save_path`, text input). Notes:
- Cell 32 (18 lines) is the PPCA/missing-data demo -- synthetic data generation plus
  `hyp.reduce` and `hyp.tools.missing_inds`; legitimate, and it is the reference for the
  per-dataset `fmt`-list + `legend`-list pattern that cells elsewhere hand-roll (G3).
- Cells 26 and 29 repeat an identical 3-line hue builder; minor duplication only.
- Cell 41 (6 lines) uses `hyp.tools.df2mat` to split a DataFrame in half -- fine.
- Cell 47 is a commented-out `save_path` example (never executed).
- **Gap by omission:** no `palette=`, `color_reduce=`, `colorbar=`, `predict=`, `surface=`,
  `density=`, `chemtrails=`, or non-default `animate=` demo in the *main plotting tutorial*.

### 3.13 `normalize.ipynb` (34.8%) -- CLEAN
Synthetic multivariate-normal data (cell 7, 10 lines, legitimate: the tutorial needs two
datasets with deliberately different means/covariances), then `hyp.normalize` x3 +
`hyp.plot` x4. **No findings.**

### 3.14 `reduce.ipynb` (36.8%) -- CLEAN
7 `hyp.reduce` calls, 1 `hyp.load`, 11 lines of `print('shape...')`.
- **Gap by omission:** never calls `hyp.plot`, and never mentions `hyp.describe()`
  (`reduce/describe.py:13-23`, *"Useful for evaluating quality of dimensionality reduced
  plots"*) -- the obvious native companion to a reduction tutorial ("how many dims do I need?").
- No hand-rolled code. **No findings beyond the omission.**

### 3.15 `cluster.ipynb` (38.9%) -- BEST RATIO
`hyp.cluster` x5, `hyp.plot` x3, `hyp.load` x1; the only non-hypertools lines are
`Counter(labels)` and an f-string hue list. Cell 21's
`hue=[f'cluster {label}' for label in labels_10]` is deliberate and explained in cell 22's
markdown (forcing the categorical hue path). **No findings.**

---

## 4. Missing-feature backlog (candidates for "bake it into hypertools")

Ordered by how many notebooks would shrink.

| ID | Proposed capability | Evidence (notebook + cell) | Notes |
|-|-|-|-|
| **M5/M6** | Forecast/imputation **scoring + backtest** (`holdout=`, or a `hyp.score`/`hyp.backtest` returning MAE/MAPE/RMSE per model x dataset) | `stock_forecasting` 6, 7, 8, 9, 16; `projectile_kalman` 9, 13 | ~90 hand-rolled lines across 2 notebooks. Verified absent: no `score`/`MAE`/`backtest` in `hypertools/predict/`. |
| **M1** | Text **chunking / sentence-windowing** helper (fixed-size chunks, sentence windows, overlap) | `text` 4 (`chunk`); `conversation_trajectories` 6, 8 (`split_sentences`, sliding windows) | Two independent hand-rolls. Fits naturally beside `manip`'s `Resample`. Verified absent. |
| **M9** | **Time-delay (Takens) embedding** manipulator | `modern_sklearn_dynamics` 10 | Belongs in `hypertools/manip/` next to `Smooth`/`Resample`. Verified absent (0 grep hits). |
| **M4** | **Damage/missingness simulator** for imputation demos (drop rows / scatter cells, seeded) | `projectile_kalman` 8 | 25 lines including a numpy memory-layout digression. `hyp.tools.missing_inds` exists for *finding* missing data but nothing for creating it. |
| **M8** | **Synthetic LSL outlet** helper (already exists in `tests/test_lsl_streaming.py`) | `lsl_streaming` 4 | 29 of the notebook's 49 code lines. Notebook markdown itself points at the internal test-suite version. |
| **M2** | Accept a **single-row dataset** inside a list without caller-side row duplication | `conversation_trajectories` 8 (`np.vstack([emb, emb])`) | Currently a documented-by-comment workaround. |
| **M7** | `hyp.load('wiki'/'nips')` returning **`list[str]` directly** | `wikipedia_embeddings` 4; `text` 13, 15 | The `[str(x) for x in data[0].ravel()]` unwrap appears 3x. |
| **M3** | `hyp.load(<url>)` support for **archive members** (`.7z`/`.zip` -> inner path) with caching | `projectile_kalman` 3 | 36 lines of download/cache/extract that partially duplicate `hypertools/io/load.py`. |
| **M10** | A native **matrix/heatmap view** (or a documented "use `hyp.plot`" story for the analyze pipeline) | `analyze` 8, 13, 18, 23, 28 | 15 lines of copy-pasted seaborn; the reduced-data cells (18/23/28) should just be `hyp.plot`. |

## 5. Quick-win summary (no new library features required)

These are pure "use the API that already exists" fixes, all verified against source:

1. **Delete 4 ffmpeg cells (~58 lines)** -- `conversation_trajectories` 15,
   `hugging_face_embeddings` 13, `modern_sklearn_dynamics` 13, `wikipedia_embeddings` 11 --
   and change the preceding `save_path='*.mp4'` to `'*.gif'` (`plot.py:1246-1250`).
2. **Replace 4 `SentenceTransformer(...).encode(...)` blocks** with
   `vectorizer='<model-id>', semantic=None, corpus=None` (`text2mat.py:89,184,391,404`) --
   `hugging_face_embeddings` 4, `wikipedia_embeddings` 6 & 17, `conversation_trajectories` 8.
   Two of these notebooks already *document* the native call in adjacent markdown.
3. **Replace the pandas rolling mean** in `stock_forecasting` 12 with
   `manip={'model':'Smooth','kwargs':{'kernel':'boxcar','kernel_width':10}}` (`plot.py:1064`; `manip/smooth.py:14`).
4. **Route the hand-drawn comparison figures through `hyp.plot(..., ax=, reduce=None, ndims=2, legend=[...])`**
   (runtime-verified) -- `stock_forecasting` 14, `projectile_kalman` 6 & 15.
5. **Add `hyp.plot` to `analyze.ipynb`** (cells 18/23/28), and `hyp.describe()` to `reduce.ipynb`.
