# HyperTools 2.0 — Class-Based Refactor Design

- **Status:** Approved design (2026-07-03); ready for implementation planning
- **Branch:** `dev-2.0-refactor` (off `dev-2.0`) → PR **into `dev-2.0`** (never `master`)
- **Related:** `notes/hypertools_2.0_roadmap.md`, memory `hypertools-2-0-modernization`, PR #270

## 1. Goal

Reorganize the working `dev-2.0` codebase into the class-based module structure Jeremy
sketched years ago in the `jeremymanning/hypertools` fork (`jeremy/dev` `dev/` folder and
`jeremy/master` `hypertools/`), **without losing any functionality from either source**.

`dev-2.0` is the **implementation template** — its code works and its ~239-test suite is
green. The fork contributes **organization and simplicity**, not implementations (much of the
fork is broken). The refactor keeps `dev-2.0`'s behavior and re-homes it into the fork's
shape.

Fundamental objectives (Jeremy's words):
1. **Single source of truth** — re-use, don't re-implement.
2. **Readability & maintainability** — a base class per area, a folder per module, one file
   per child class.
3. **Compatibility & performance** across environments — default target is Google Colab, but
   also Kaggle, local Jupyter/`.ipynb`, and base Python.
4. **Better, more comprehensive tests** — many current tests (in both `dev-2.0` and the fork)
   are trivial; replace them with real-call coverage.

Framed as HyperTools **2.0**: a functionality upgrade *and* a codebase overhaul. Deliverable
is one comprehensive PR into `dev-2.0` with concrete evidence the new code works.

**Source-of-truth rule (governs every module):** where a capability exists in `dev-2.0`, its
working code is the trusted template. Where a capability exists **only in the fork** (e.g.
`manip.Smooth`/`Resample`/`ZScore`, and any other fork-only children discovered while porting),
the fork is a **starting point to validate and fix, not a trusted template** — port it, then
prove it with real tests. "ALL functionality from BOTH" means each module port includes an
explicit diff of the fork's children against `dev-2.0` so no fork-only capability is dropped.

## 2. Pivotal decisions (locked with Jeremy)

| # | Decision | Choice |
|-|-|-|
| 1 | datawrangler (`dw`) | **Adopt (hybrid).** Use `dw` for the data-wrangling core (funnel, stack/unstack, format detection, model dispatch, text/HF embeddings). Keep `dev-2.0`'s own plotting, animation, streaming, and coloring — `dw` does not do those. |
| 2 | DataGeometry | **Remove entirely.** Pure fork style: functions return DataFrames/lists; `plot()` returns the figure (+ animation); `return_model=True` threads fitted models out. Geo-like needs met by `return_model` + `io/save.py`. |
| 3 | Public API surface | **Classic names + new module aliases.** Keep `hyp.plot/analyze/reduce/align/cluster/normalize/describe/load` (+ new `save`, `apply_model`) AND expose `hyp.manip`, `hyp.io.*`, submodules. |
| 4 | Docs scope | **Everything in one PR** — library + comprehensive tests + docs/gallery/notebooks migrated to the new API. |
| 5 | Vendored code | **Keep `external/`.** Quarantine third-party, differently-licensed vendored algorithms (brainiak SRM Apache-2.0, PPCA) with license headers; thin adapters live in `align/`/`reduce/`. |
| 6 | Execution strategy | **Approach B — strangler, module-by-module, green at every commit.** Old `tools/` names become shims re-exporting new locations until the final cleanup. |
| 7 | Polars | **Support natively via `dw`.** pandas stays primary/default; polars DataFrames supported through the funnel, with their own real tests. |

## 3. Target module layout

One folder per area; each area = `common.py` base class + one file per child class + a thin
dispatcher. Mirrors `jeremy/master/hypertools/`.

```
hypertools/
├── __init__.py          # classic names + module aliases; lazy __getattr__
├── analyze.py           # hyp.analyze pipeline (normalize→reduce→align convenience)
├── core/
│   ├── model.py         # apply_model — dw-backed, eval-FREE registry   ← tools/apply_model.py
│   ├── configurator.py  # config.ini defaults + apply_defaults          ← _shared/params.py
│   ├── shared.py        # unpack_model, RobustDict                       ← new (from fork)
│   ├── util.py          # data-shape helpers surviving dw                ← _shared/helpers.py (split)
│   ├── exceptions.py    #                                                ← _shared/exceptions.py
│   └── config.ini       # central plot + model defaults                  ← new (from fork)
├── external/            # vendored third-party (quarantined, license headers retained)
│   ├── ppca.py          ← _externals/ppca.py         (pca-magic replacement)
│   └── brainiak.py      # SRM / DetSRM / RSRM        ← _externals/srm.py (Apache-2.0)
├── manip/
│   ├── common.py        # Manipulator base
│   ├── manip.py         # dispatcher (hyp.manip)
│   ├── normalize.py     ← tools/normalize.py (includes dev-2.0's 'zscore' normalize mode)
│   ├── smooth.py        # FORK-ONLY (jeremy/master:manip/smooth.py) — validate/fix, real tests
│   ├── resample.py      # FORK-ONLY (jeremy/master:manip/resample.py) — validate/fix, real tests
│   └── zscore.py        # FORK-ONLY standalone (dev-2.0 has zscore only as a normalize mode)
├── reduce/
│   ├── common.py        # Reducer base
│   ├── reduce.py        ← tools/reduce.py (dispatcher, @dw.decorate.apply_stacked)
│   ├── umap.py          # lazy UMAP hook
│   └── describe.py      ← tools/describe.py (hyp.describe: reduction fidelity)
├── align/
│   ├── common.py        # Aligner base (+ pad / trim_and_pad)
│   ├── align.py         ← tools/align.py (dispatcher)
│   ├── hyperalign.py    # HyperAlign (n_iter, per-pass rescale)
│   ├── procrustes.py    ← tools/procrustes.py (Procrustes child)
│   ├── srm.py           # SharedResponseModel/Deterministic/Robust adapters → external.brainiak
│   └── null.py          # NullAlign
├── cluster/
│   ├── common.py        # Clusterer base — hard labels AND soft mixture proportions
│   └── cluster.py       ← tools/cluster.py (dispatcher)
├── io/
│   ├── load.py          ← tools/load.py    (named datasets / local / legacy .h5)
│   ├── sources.py       ← tools/sources.py (universal loader: HF ids, Drive/Dropbox/URLs)
│   ├── streaming.py     ← tools/streaming.py (DATA side: is_stream/row_to_vector/_fit_stream_models)
│   └── save.py          # NEW — unified png/pdf/svg/html/gif/mp4 (replaces geo.save)
└── plot/
    ├── plot.py               ← plot/plot.py (orchestrator; @dw.decorate.funnel)
    ├── backend.py            ← plot/backend.py (env detect + backend switching)
    ├── matplotlib_backend.py ← plot/draw.py
    ├── plotly_backend.py     ← plot/interactive.py
    ├── colors.py             ← tools/colors.py + helpers.vals2colors/vals2bins (robust coloring)
    └── animate.py            ← plot animation helpers + _shared/animated_svg.py
                               # NOTE: plot_stream renderer moves here from tools/streaming.py
```

**Absorbed into `dw`** (become thin classic-name shims, then documented as dw-backed):
`format_data.py`, `df2mat.py`, `text2mat.py`, `missing_inds.py`, `fill_missing`. These are
exactly `dw.decorate.funnel` / `dw.zoo` / stack-fill territory.

**Deleted:** `datageometry.py`, `_shared/helpers.check_geo`, all geo-coupled paths.

## 4. Core layer & the base-class pattern

### 4.1 datawrangler integration (`core` is where `dw` enters)

Every functional module is decorated with the funnel so it accepts arrays / DataFrames /
polars / text / nested lists uniformly and stacks-applies-unstacks automatically:

- `@dw.decorate.funnel` — format → fill-missing → stack/unstack → list-generalize.
- `@dw.decorate.apply_stacked` — stack list into MultiIndex, apply once, unstack (correct
  shared embedding/clustering across datasets).
- `dw.stack` / `dw.unstack`, `dw.core.update_dict` / `apply_defaults` /
  `get_default_options('config.ini')`.
- `dw.zoo.is_dataframe` / `is_array` / `is_multiindex_dataframe`; `dw.zoo.text.*` for the
  text/HF path.

### 4.2 `core/model.py` — one `apply_model`, eval-FREE

Universal dispatch (string / dict / list-pipeline / callable / object), ported from the fork's
`apply_model` **but** with the fork's `eval()`/`exec()` model loading replaced by explicit
`importlib.import_module(m)` + `getattr` against a whitelisted module list (sklearn submodules
+ `umap` + `external`). `dev-2.0`'s `_build_registry`/`supported_models` already does this
eval-free — keep it, merge in the fork's dict/list-pipeline handling, `return_model`
threading, and `mode` handling (`fit_transform`/`predict`/`predict_proba`/soft-mixture from the
fork's `get_sklearn_method`).

### 4.3 The base-class trio (every area identical shape)

```
<area>/common.py    class <Base>(BaseEstimator): __init__(fitter, transformer, required) / fit / transform / fit_transform
<area>/<child>.py   def fitter(...) / def transformer(...) / class <Child>(<Base>)  # supplies fitter+transformer+defaults
<area>/<area>.py    @dw.decorate.funnel  def <area>(data, model=<default>, **kw):
                        return apply_model(data, unpack_model(model, valid=[...], parent_class=<Base>), **defaults|kw)
```

A base class is a sklearn-compatible wrapper around a `(fitter, transformer, required-params)`
triple. Child classes supply those three plus their config defaults. The dispatcher resolves
the model spec via `unpack_model` (whitelist + `parent_class` check — the eval-free registry)
and hands it to `apply_model`. `Aligner`, `Manipulator`, `Reducer`, `Clusterer` all follow this.

### 4.4 `core/configurator.py` + `config.ini`

Central defaults for model params *and* plot styling, read once via
`dw.core.get_default_options`, applied through an `apply_defaults` decorator. Replaces
`_shared/params.default_params`.

## 5. HuggingFace / text handling

Text/HF embedding is **entirely a `dw` concern** and is a **strict superset** of `dev-2.0`'s
current sklearn-only text path. Any function that can receive text is `@dw.decorate.funnel`-
wrapped; when the input is text, `dw`'s `wrangle_text`/`apply_text_model` embeds it to a matrix
before reduce/align/cluster/plot logic runs.

| Input | `dw` routes to | Matches |
|-|-|-|
| `'CountVectorizer'`/`'TfidfVectorizer'` → `'LatentDirichletAllocation'`/`'NMF'` | sklearn vectorizer → semantic | dev-2.0's current pipeline |
| `'wiki'`/`'nips'`/`'sotus'` | `dw.zoo.text.get_corpus` (pretrained corpora) | dev-2.0's `load(corpus+'_model')` |
| `'all-MiniLM-L6-v2'` (any non-sklearn name) | **sentence-transformers (HF)** | fork's HF path + round-6 ST demo, now first-class |
| dict `{'model','args','kwargs'}` | either, custom params | both branches |

Consequences:
1. **No hypertools-owned text-embedding code.** `text2mat`/`format_data`'s
   vectorizer→semantic→corpus logic becomes a thin classic-name shim mapping
   `vectorizer`/`semantic`/`corpus` onto `dw`'s text-model dict — old API preserved, and
   `hyp.plot(docs, semantic='all-MiniLM-L6-v2')` transformer embeddings come for free.
2. **Optional & lazy.** Transformer embeddings require `dw`'s `hf` extra (torch / transformers
   / sentence-transformers / tokenizers / datasets). hypertools exposes `hypertools[text]` →
   `pydata-wrangler[hf]`. sklearn path needs nothing extra. Satisfies the roadmap's "text
   embeddings optional extra / → optional sentence-transformers".
3. **Resolution ordering guaranteed safe.** hypertools' own model names (`HyperAlign`,
   `IncrementalPCA`, `KMeans`, `Procrustes`…) resolve against each module's `unpack_model`
   whitelist FIRST; only genuine text *data* enters the funnel's text branch. No collision
   between "a model named X" and "a HF embedder named X".

## 6. Functional modules (manip / reduce / align / cluster)

- **manip** — `Manipulator` base + `Normalize` / `Smooth` / `Resample` / `ZScore` children +
  `manip` dispatcher (`search=['sklearn.preprocessing']`). `hyp.normalize` stays a classic
  alias routing into manip, sourced from dev-2.0's `tools/normalize.py` (which also holds the
  `'zscore'` mode). **`Smooth`/`Resample`/`ZScore` are FORK-ONLY** — no standalone dev-2.0
  implementation exists, so port from `jeremy/master:hypertools/manip/`, treat as a starting
  point (fork code may be broken), fix, and cover with real tests. The weights-trajectory
  recipe (gaussian var=300 smooth → SRM n_iter=20 → smooth → UMAP nn=36) is the acceptance
  scenario for `Smooth`.
- **reduce** — `Reducer` base + `reduce` dispatcher (`search=['sklearn.decomposition',
  'sklearn.manifold', 'sklearn.mixture', 'umap', 'ppca']`), `get_n_components` short-circuit
  (no-op / pad when dims already ≤ target), lazy `umap`, `PPCA` via `external.ppca`. `describe`
  (reduction fidelity: `get_corr`/`get_cdist`) lives here, exposed as `hyp.describe`.
- **align** — `Aligner` base (+ `pad`/`trim_and_pad`) + `HyperAlign` (n_iter with per-pass
  rescale fixing procrustes-scaling collapse), `Procrustes`, `SharedResponseModel` /
  `DeterministicSharedResponseModel` / `RobustSharedResponseModel` (adapters →
  `external.brainiak`), `NullAlign` + `align` dispatcher.
- **cluster** — `Clusterer` base + `cluster` dispatcher. Returns **hard labels** for cluster
  models but **soft mixture proportions** for `GaussianMixture` / `BayesianGaussianMixture` /
  `LatentDirichletAllocation` / `NMF` (via `predict_proba`/`transform`). Plot colors blend by
  membership weights through `mat2colors`. `cluster={'model':…, 'n_clusters':k}` single-call
  syntax preserved.

## 7. io — loaders + save

- `load.py` (named datasets / local / legacy `.h5` with clear error), `sources.py` (universal
  loader: HF dataset ids, Drive/Dropbox/URLs, lists→lists) — kept intact.
- `streaming.py` — DATA side only (`is_stream`, `row_to_vector`, `_fit_stream_models`); the
  animated `plot_stream` renderer moves to `plot/`. Streams stay first-class (no flag):
  `hyp.plot(a_stream)` detects and routes. `stream_init`/`stream_chunk`/`stream_max`/
  `stream_window` semantics preserved (box frozen from head, view never twitches).
- **`save.py` (new)** — unified exporter replacing `geo.save()`: `hyp.save(fig, 'out.gif')`
  handles png/pdf/svg/html/gif/mp4 across both backends, operating on the returned
  figure/animation object.

## 8. plot + colors (dev-2.0 work preserved, reorganized)

Keep `dev-2.0`'s dual-backend architecture wholesale — do NOT adopt the fork's plotly-centric
plot. matplotlib default; plotly optional; `backend='auto'` → plotly only on Colab/Kaggle.

- `plot.py` orchestrator, `@dw.decorate.funnel` (accepts arrays/DataFrames/polars/text/nested/
  streams). **Return = pure fork style:** figure (mpl `Figure` or plotly `Figure`) and, for
  animations, the animation object — never a geo. `return_model=True` threads fitted models.
  All kwargs kept: `ax=` multipanel, `zoom`, `tail_duration`, `explore`, chemtrails/precog/
  bullettime trails, frame_rate=30/duration=30/1-rotation animation standard.
- `backend.py` (unchanged), `matplotlib_backend.py` (← draw.py), `plotly_backend.py`
  (← interactive.py), `animate.py` (← animation helpers + animated_svg).
- **`plot/colors.py` — robust coloring (required 2.0 carry):** `mat2colors`/`vals2colors`/
  `colors2groups` intact — categorical, continuous 1-D, mixture proportions / arbitrary
  matrices (2-D→1-D, ≥3-D→RGB), user colors, multicolored lines (2-D segments, 3-D
  streamtubes), colored cluster connections. **Multilevel-index styling** (color by outermost
  level; thickness/opacity decay by depth) rides on `dw` MultiIndex frames here.

**Geo-removal cleanup:** animations return the `Animation`/frames object directly; the
sphinx-gallery scraper's `ani = geo.line_ani` becomes `ani = <returned animation>`;
`check_geo` deleted.

## 9. Public API — `__init__.py`

- Classic top-level (unchanged for users): `plot, analyze, reduce, align, cluster, normalize,
  describe, load` + new `save`, `apply_model`.
- New module surface: `hyp.manip`, `hyp.io` (`io.load`/`io.save`), submodules
  (`hyp.reduce`, `hyp.align`, …) as documented entry points.
- Lazy `__getattr__` at package level so `umap` / `plotly` / `sentence-transformers` /
  `datasets` import only on use — preserve the <1s import target.

## 10. Polars support

`dw` round-trips polars natively (declares `polars>=0.20.0`; "verified working with Polars
backend"). Through `@dw.decorate.funnel`, `hyp.*` accepts polars DataFrames with essentially no
hypertools code. **pandas stays primary/default.** Add real polars tests: polars DataFrame (and
list-of-polars) input across `plot`/`reduce`/`align`/`cluster`/`normalize`/`analyze`, asserting
parity with the pandas path. Document which operations return polars vs coerce to pandas (per
`dw` behavior).

## 11. datawrangler coordination

`dw` is a separate package, not recently updated; a parallel Claude Code instance works on
`/Users/jmanning/data-wrangler`. Workflow:
- When the step-0 probe or any later step hits a `dw` bug or missing/changed API, **file a
  GitHub issue on `ContextLab/data-wrangler`** (public, issues enabled, default branch `main`)
  via `gh issue create -R ContextLab/data-wrangler` — with a minimal repro and the exact
  hypertools call site. Prefer filing over an internal workaround, to keep hypertools' `dw`
  usage clean.
- The parallel instance fixes at the source; pin the `dw` version once fixed.
- Track filed issues in the session notes so nothing is lost across context boundaries.

## 12. Testing strategy (top priority — real calls, no mocks)

Per-module test dirs mirror the tree: `tests/core/`, `tests/manip/`, `tests/reduce/`,
`tests/align/`, `tests/cluster/`, `tests/io/`, `tests/plot/`.

Every public function exercised across the full matrix:
- single array · list of arrays · **pandas DataFrame(s)** · **polars DataFrame(s)** · text
  (sklearn vectorizer AND a real sentence-transformers embedding) · missing data · 2-D/3-D/
  high-D · static + animated · matplotlib + plotly.
- Real dataset loads (HF ids / Drive / Dropbox / URL), real HF streaming (iris stream),
  round-trippable save/load.
- **Screenshot harness:** every plot case saves a PNG to `tests/screenshots/<fn>/<case>.png`;
  CI uploads as artifacts (plotly via kaleido).
- **Backend parity** assertions (mpl vs plotly identical structure — trace/frame counts,
  limits, titles).
- **dw-API probe test** (step 0) asserting every `dw` symbol we use exists at the pinned
  version.
- **Regression tests:** #27 story-trajectory reproducibility (normalization + PPCA-fill
  semantics), #259 (no global rcParams mutation), #264/#265 (numpy≥2 Jupyter animations,
  figures in loops), streaming box-frozen view.
- **Performance regression:** import time (<1s target) + `plot()` wall time on canonical
  datasets.
- Migrate and **de-trivialize** the existing 239+ tests.

No mocks: real API/library/model/file/network calls per Jeremy's testing methodology. Where a
call is expensive, verify once with a real call, then keep a same-syntax test that still
exercises real code paths.

## 13. Execution — Approach B (strangler, one green commit per module)

| # | Step | Gate |
|-|-|-|
| 0 | Install `pydata-wrangler[hf]`, pin, **dw-API probe test**, reconcile CI py-matrix (dw 3.9–3.12 vs our 3.10–3.13; verify 3.13, else cap at 3.12) | probe green |
| 1 | `core` (eval-free `apply_model`, `configurator`+`config.ini`, `unpack_model`, `util`, `exceptions`) | suite green |
| 2 | `external` (brainiak, ppca) ← `_externals/` | green |
| 3 | `manip` (common + normalize/smooth/resample/zscore + dispatcher) | green |
| 4 | `reduce` (common + reduce + describe + ppca ref) | green |
| 5 | `align` (common + hyperalign/procrustes/srm/null + dispatcher) | green |
| 6 | `cluster` (common + cluster; hard labels AND soft mixtures) | green |
| 7 | `io` (load + sources + streaming + save) | green |
| 8 | `plot` + `colors` (backend/mpl/plotly/animate/colors) | screenshots pass |
| 9 | top-level API + aliases (`__init__`, `analyze`) | green |
| 10 | delete `DataGeometry` + retire `tools/` shims | green |
| 11 | docs / gallery / notebooks migrated to new API | doc build clean + **Playwright visual verify** |
| 12 | full verification + PR evidence | all green |

Through steps 1–9 the old `tools/` names remain thin shims re-exporting the new locations, so
the suite never goes dark; new comprehensive tests land with each module. Commit after each
step with a descriptive message; push frequently (branch backup).

## 14. Docs / gallery / notebooks (step 11) — Playwright-verified

Migrate the full Sphinx site, gallery examples, and tutorial notebooks to the new API. A clean
build is **not** sufficient evidence:
- Serve the built HTML locally and drive it with **Playwright** to screenshot rendered gallery
  and tutorial pages; visually confirm plots render, **embedded animations play**, and Colab
  badges are present.
- Apply round-6.5 gotchas: delete sphinx-gallery `.py.md5` files to force re-execution after
  renderer/library changes; `ffprobe` every artifact mp4 (don't trust build success);
  frame-stripped png + capped html for plotly gallery items to avoid multi-minute kaleido
  serialization.
- Colab install cells stay branch-aware (preview branch installs
  `hypertools[interactive] @ git+…@dev-2.0-refactor`).

## 15. Risks & mitigations

| Risk | Mitigation |
|-|-|
| `dw` 0.4.0 API drift from the fork's older `dw` | Step-0 probe catches it before anything builds on it; file data-wrangler issues for gaps |
| `dw` Python floor 3.9–3.12 vs our 3.10–3.13 | Verify `dw` on 3.13 in step 0; cap CI at 3.12 only if it genuinely lags |
| PPCA missing-fill semantics differ from `dw`'s fill | Keep a PPCA-fill path; add #27 story-trajectory reproducibility regression test |
| `polars` transitive via `dw` | Confirm no conflict; hypertools stays pandas-first; explicit polars tests |
| Large docs PR | Round-6.5 gotchas (delete `.md5`s, `ffprobe` artifacts); Playwright visual gate |
| Model-name/text-HF collision | Whitelist-first resolution ordering (§5.3) |

## 16. Branch, PR & evidence

- New branch `dev-2.0-refactor` off `dev-2.0`; PR **into `dev-2.0`** (never `master` — Jeremy
  signs off manually).
- PR evidence comments: test counts (before/after), screenshot matrix, Playwright doc
  screenshots, dev-notebook run, import-time measurement, mpl/plotly parity montages, list of
  filed data-wrangler issues.

## 17. Non-goals / out of scope

- No push to `master`.
- No new custom plotting backend (D3/Three.js/pythreejs — roadmap dead-ends).
- No flair / tensorflow-hub text embeddings (replaced by `dw`'s sentence-transformers path).
- No memoize/caching layer (deliberately removed — stringified keys ignored kwargs).
- Broad Jupyter auto-backend switching beyond Colab/Kaggle deferred (conservative for 2.0).
