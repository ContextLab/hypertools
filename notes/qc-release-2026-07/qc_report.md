# HyperTools 1.0 — PR #272 Release QC Report

185 commits, 946 files (+61,780/-8,439). 53 commits carry review flags.


## Task list (items to address)

These are the items the audit surfaced. Add your own as you work the checklist and notebook, then send the exported notes back and I'll fix them. The **static** commit audit came back clean (no high-risk logic in any of the 185 commits), but **running the verification notebook against the built library surfaced four real code issues** (P0/P1 below) — exactly what this pass was for. Everything else is documentation accuracy.

### P0 · Confirmed code bugs (found by running the verification notebook)

**P0-1 · `hyp.normalize(..., return_model=True).transform(new_data)` crashes.** `hypertools/tools/normalize.py:94` (`Normalizer.transform`) raises `IndexError: tuple index out of range` for **all three** modes (`across`/`within`/`row`) — it indexes `i.shape[1]` on what is a 1-D row. The entire normalize return-model *reuse* path (part of the #227 "return_model on every module" contract) is broken. It slipped through because Task 6's test likely exercised a shape/list form that dodges this line. Fix `Normalizer.transform` to apply the stored fit-time statistics to new data correctly, and add a real 2-D-array reuse test.

**P0-2 · `hyp.manip` rejects the `kwargs`-only dict spec.** `hyp.manip(x, model={'model':'Smooth','kwargs':{...}})` raises `ValueError: unknown model`, but `{'model':'Smooth','args':[],'kwargs':{...}}` (with an explicit `args`) and the list form both work. A canonical dict spec should accept `kwargs` without requiring an `args` key — this is an inconsistency with `reduce`/`cluster` (which check `'args' in x or 'kwargs' in x`). Fix manip's single-spec dict resolution to treat `args` as optional.

### P1 · Rough edges (found by running the notebook)

**P1-1 · `Pipeline.inverse_transform` can't pass through `ZScore`/`Normalize`.** `Pipeline(['ZScore','PCA']).inverse_transform(...)` raises `NotImplementedError` because `ZScore` has no `inverse_transform` — even though it stores the mean/std needed to invert. Consider giving the invertible manipulators (`ZScore`, `Normalize`) an `inverse_transform` so pipeline round-trips work through them.

**P1-2 · `vectorizer='Word2Vec'` breaks with the default `semantic`.** `hyp.plot(docs, vectorizer='Word2Vec')` (no `semantic=`) raises `ValueError: Negative values in data passed to LatentDirichletAllocation` — the default semantic model can't consume the negative embeddings that Word2Vec/FastText produce. Either default `semantic=None` when an embedding vectorizer is selected, or raise a clear, actionable error. (README/notebook examples correctly pass `semantic=None`.)



### A · Documentation accuracy — recommended before release

**A1 · CLAUDE.md is stale (highest leverage — AI tools read it every session).** 7 wrong statements: (1) tests run from repo root / `tests/`, not `hypertools/`; (2) `[dev]` does **not** install all deps (omits `text`, `predict-hf`); (3) DataGeometry is now an internal unpickle-only shell, not the central container; (4) Main-API list omits `manip`/`predict`/`impute`/`save`/`apply_model`/`Pipeline`/`set_interactive_backend`/`io`; (5) "Tools Module" list is wrong — `reduce`/`cluster`/`align`/`load` moved to their own subpackages; new `reduce/ cluster/ align/ manip/ io/ predict/ impute/ core/` unmentioned; (6) `plot/draw.py` is now a shim for `matplotlib_backend.py`; new plot modules unlisted; (7) "Python 3.9+" → floor is 3.10.

**A2 · API docstring inaccuracies (9).** `align` documented as default `'hyper'` in `plot`/`analyze`/`load` but the real default is `None` — the worst one (implies data is hyperaligned by default when it isn't). Also: `plot` `ndims` says "None→3" (is `3`); `zoom` says default 0 (is 1); `normalize` doc self-contradictory; `plot` `cluster` example uses `reduce=`+deprecated `'params'`; `plot` `return_model` Returns bundle omits the `'pipeline'` key; `set_interactive_backend` example calls `geo.plot()` which no longer exists.

**A3 · API docstring incompleteness (6).** `reduce`/`cluster` model-name lists in `plot`/`analyze`/`load`/`describe` omit UMAP, the mixture reducers, and the six autoencoders; `apply_model` documents only the deprecated `{'model','params'}` dict form (not the canonical `args`/`kwargs`).

**A4 · README.** One broken import path: `hypertools.tools.colors.mat2colors` → `hypertools.plot.colors.mat2colors` (line 52). "What's new in 1.0" omits several shipped features worth showcasing: `hyp.Pipeline`, manip chaining, autoencoders (`[torch]`), gensim (`[gensim]`), LSL (`[lsl]`), `predict`/`impute`, and the `window`/2-D animation modes; Requirements names only 2 of 11 extras.

**A5 · Sphinx tutorials (2 build-affecting).** `geo.ipynb` (retired DataGeometry) is still published via `tutorials.rst`; 8 stale 0.x `.rst` tutorials duplicate the new executed `.ipynb`, producing 8 "multiple files found" warnings and risking that Sphinx publishes the retired-API `.rst` instead of the current notebook. Fix: delete the superseded `docs/tutorials/{align,analyze,cluster,geo,normalize,plot,reduce,text}.rst` (+ their `*_files/`) and drop the geo section. Also: `docs/index.rst` title underline too short (1 cosmetic warning).

### B · Code items to confirm (not defects — intentional 1.0 changes worth a second look)

**B1 · SRM alignment semantics changed** (commit `22cab2b49c`). The classic `align='SRM'` path no longer does `n_iter` repeated re-fits (single fit via the new SRM class), dropped two legacy warnings (len-1 list, features>samples), and `align=True` now raises. Confirm SRM alignment quality is still what you expect on real data.

**B2 · pandas 3.0 now permitted** (commit `60b598f294`; `pandas<3` pin lifted, `>=2.2`). A CI acceptance gate pins pandas 3.0 on one job, but the runtime surface is wider than 0.x. Confirm you're comfortable supporting pandas 3.

### C · Optional / housekeeping

**C1 · Add `AGENTS.md`** (absent) — the emerging cross-tool standard; mirror the corrected CLAUDE.md so non-Claude agents get the same guidance.
**C2 · `CONTRIBUTING.md` is stale** — points at a mozsprint milestone and a defunct Gitter channel; no mention of the `pip install -e ".[dev]"` / `pytest` workflow.
**C3 · Cosmetic sphinx warnings** — scipy intersphinx 404 (double slash) and the sphinx-gallery pickle-cache warning; both harmless, optional to silence.
**C4 · `pipeline_order.rst`** calls the story-trajectories walkthrough a "tutorial" but it's a gallery example — reword the cross-reference.



## Documentation audit


### API docstrings

I have completed the audit. Note: I could not write the `api_docstrings.md` file — this is a strict READ-ONLY task and I have no write tools, and my guidance is to return findings directly. Full findings below.

## API Docstring Accuracy Audit — HyperTools 1.0

### INACCURATE / MISLEADING

**hypertools/plot/plot.py:538–544 — `plot`**: `ndims` docstring says "Default is None, which will plot data in 3 dimensions". Signature (line 159) is `ndims=3`. Fix: state "Default is 3."

**hypertools/plot/plot.py:776–777 — `plot`**: `zoom` doc says "(default: 0)". Signature (line 176) is `zoom=1` (confirmed by the animate-dict defaults at line 1193, `'zoom': 1`). Fix: "(default: 1)."

**hypertools/plot/plot.py:546–551 — `plot`**: `align` doc says "(default : 'hyper')". Signature (line 183) is `align=None` → no alignment by default. Misleads users into thinking data is hyperaligned by default. Fix: "default None (no alignment)."

**hypertools/plot/plot.py:491–496 — `plot`**: `normalize` doc contradicts itself — "z-scored across lists (default)" and also "the input data will be returned (default is False)". Real signature default is `normalize=None` (no normalization). Fix: state default None and remove the contradictory "(default is False)"/"(default)" claims.

**hypertools/plot/plot.py:558–560 — `plot`**: `cluster` example reads `reduce={'model' : 'KMeans', 'params' : {'max_iter' : 100}}` — wrong kwarg (`reduce=` instead of `cluster=`) and uses the deprecated `'params'` dict form. Fix: `cluster={'model': 'KMeans', 'kwargs': {'max_iter': 100}}`.

**hypertools/plot/plot.py:1166–1170 — `plot`**: the Returns section lists the `return_model=True` dict as `{'fig','xform_data','animation','models','predict'}`, omitting `'pipeline'`. The actual returned bundle (line 2541–2557) and the `return_model` param doc (line 1143) both include `'pipeline'`. Fix: add `'pipeline'` to the Returns dict.

**hypertools/tools/analyze.py:53–58 — `analyze`**: `align` doc says "(default : 'hyper')". Signature (line 14) is `align=None`. Fix: default None.

**hypertools/io/load.py:203–208 — `load`**: `align` doc says "(default : 'hyper')". Signature (line 49) is `align=None`. Fix: default None.

**hypertools/plot/backend.py:930–950 — `set_interactive_backend`**: examples do `geo = hyp.load('weights_avg')` then `geo.plot(interactive=True)`. In 1.0 `load` returns raw arrays/list (its own docstring: "hypertools 1.0 users never receive a geo"), which has no `.plot()` method — the example would fail. Fix: `data = hyp.load('weights_avg'); hyp.plot(data, interactive=True)`.

### INCOMPLETE

**hypertools/plot/plot.py:528–536 — `plot`**: `reduce` model list stops at MDS; omits UMAP, the mixture reducers (GaussianMixture/BayesianGaussianMixture/LatentDirichletAllocation/NMF), and the six autoencoders — all of which `reduce.reduce` (which `plot` delegates to) documents and resolves. Also shows the deprecated `'params'` dict example.

**hypertools/plot/plot.py:553–564 — `plot`**: `cluster` algorithm list ("...SpectralClustering and HDBSCAN") omits MeanShift, DBSCAN, OPTICS, AffinityPropagation and the mixture models — all accepted by `cluster.cluster` and its registry (CLUSTERERS/MIXTURES).

**hypertools/tools/analyze.py:40–48 — `analyze`**: `reduce` list omits UMAP/mixtures/autoencoders (same as plot).

**hypertools/io/load.py:190–198 — `load`**: `reduce` list omits UMAP/mixtures/autoencoders.

**hypertools/reduce/describe.py:27–37 — `describe`**: `reduce` list omits UMAP/mixtures/autoencoders; the `reduce` default (`'IncrementalPCA'`, line 12) is not stated.

**hypertools/core/model.py:52–54 — `apply_model`**: dict form documented only as `{'model': <str or instance>, 'params': {...}}` (the deprecated legacy shape). Code (`_resolve_model`→`unpack_model`) also accepts the canonical `{'model':..., 'args':[...], 'kwargs':{...}}`. Fix: document the canonical form as primary.

### MINOR

**Multiple files — deprecated `'params'` dict shown as the example form**: plot.py:534, analyze.py:46, describe.py:33, load.py:196 all use `reduce={'model':'PCA','params':{'whiten':True}}`. This still works but emits a `DeprecationWarning`; the canonical `'kwargs'` form should be shown instead.

**hypertools/plot/plot.py:528 — `plot`**: `reduce` parameter's default (`"IncrementalPCA"`) is not stated in its docstring.

### FULLY ACCURATE
`reduce`, `align`, `cluster`, `normalize`, `manip`, `predict`, `impute`, `save`, `Pipeline`/`build_pipeline`, `lsl_stream`, the reducer/aligner/forecaster/imputer registries + `resolve_reducer`, the six autoencoder classes, and the gensim `Word2VecVectorizer` all have signatures, defaults, model-name lists, and return types matching their code. `apply_model`'s params/modes/returns are accurate aside from the dict-form note above.

---
Summary: **Inaccurate 9 / Incomplete 6 / Minor 2.** Worst: `align` documented as "default 'hyper'" in plot, analyze, and load — but the real default is `None` (no alignment), so users believe data is hyperaligned by default when it isn't. (File not written — read-only mode; findings returned inline.)

### README

I've completed the audit. I could not write the findings file: this is a read-only exploration role with no Write/Edit tool and file-creation via Bash is prohibited, so I'm returning the full findings directly (the launching agent reads my output).

FULL AUDIT — README.md vs current codebase (dev-1.0-refactor)

INACCURATE / BROKEN
- Line 52: README calls the new helper `hypertools.tools.colors.mat2colors`. That module does not exist. Actual location is `hypertools.plot.colors.mat2colors` (`import` of the README path raises ModuleNotFoundError). Fix: change `hypertools.tools.colors.mat2colors` → `hypertools.plot.colors.mat2colors`.

VERIFIED CORRECT
- All 5 code snippets run clean under MPLBACKEND=Agg on .venv python: line 192 `plot(...animate=True, hue=...)`; line 201 `plot(...align='hyper')`; line 220 `plot(array,'o',cluster='GaussianMixture',n_clusters=3)`; line 232 `plot([a,b],'.',surface=True)`; line 242 `describe(...reduce='PCA',max_dims=14)`. (describe emits benign "large data"/Agg-noninteractive UserWarnings only.)
- All API named in text exists: apply_model, Pipeline, manip, predict, impute, io.lsl_stream, autoencoders (6 reducers), gensim vectorizers. `align='hyper'` valid.
- All badges + every link (18 URLs) return 200, including master-branch CONTRIBUTING.md, JMLR, arXiv, readthedocs, YouTube, archive.org Kaggle.
- All 8 referenced images exist in images/ (hypercube.png, hypertools.gif, plot.gif, align_before/after.gif, cluster_example.png, surface_example.png, describe_example.png).
- Example files cited (examples/plot_shape_morph.py, animate_surface_morph.py) exist.
- Requirements list matches pyproject (py 3.10–3.13; sklearn/pandas/etc floors). Extras named in README ([interactive], [dev]) match pyproject exactly.
- No stale/retired-API references (DataGeometry, group=, hyp.tools.*) appear in prose; the "retired arguments" bullet correctly describes removed args. Filename is tracked as lowercase `readme.md`, matching pyproject `readme="readme.md"`.

MISSING COVERAGE (present in 1.0, absent from README/What's-new)
- Pipeline class (`hyp.Pipeline`) — not showcased (apply_model bullet mentions "pipeline of models" but not the class).
- manip chaining — not mentioned.
- Autoencoder reducers + `[torch]` extra — not mentioned at all.
- gensim text vectorizers + `[gensim]` extra — not mentioned.
- LSL streaming (`hyp.io.lsl_stream`) + `[lsl]` extra — not mentioned.
- predict/impute (`hyp.predict`, `hyp.impute`) + `[predict]`/`[predict-hf]` extras — not in What's-new.
- Kaggle loader + `[kaggle]` extra; `[io]`/`[density3d]` extras — unnamed in Requirements.
- Animation modes `window` and `serial`/`spin`, and 2D animation — only `morph` is showcased.

MINOR
- Requirements section lists only `[interactive]`/`[dev]`; 8 other real extras go unnamed (ties to missing-coverage above).

---
Severity counts: Broken/Inaccurate 1 | Missing-coverage 8 | Minor 1. Worst: README's `hypertools.tools.colors.mat2colors` (line 52) is a nonexistent import path — actual is `hypertools.plot.colors.mat2colors`. All 5 code snippets, 18 links, and 8 images verified good.

Audit complete. Summary:

Severity counts: Broken/Inaccurate 1 | Missing-coverage 8 | Minor 1.

Worst issue: README line 52 references `hypertools.tools.colors.mat2colors`, a nonexistent import path (ModuleNotFoundError). Actual location is `hypertools.plot.colors.mat2colors`.

All 5 code snippets ran clean, all 18 links returned 200, all 8 images exist, and named extras ([interactive]/[dev]) match pyproject. Main gap is What's-new omits several shipped 1.0 features (Pipeline, autoencoders/[torch], gensim, LSL, predict/impute, window/2D animation).

Note: I could not write the scratchpad file — this read-only role has no Write/Edit tool and Bash file-creation is blocked — so the complete grouped findings are in my message above for you to consume.

The audit is complete and all findings have been delivered above. No further action is needed.

Final summary:
- Broken/Inaccurate: 1 (README line 52 `hypertools.tools.colors.mat2colors` → should be `hypertools.plot.colors.mat2colors`)
- Missing-coverage: 8 (Pipeline, manip chaining, autoencoders/[torch], gensim, LSL, predict/impute, kaggle/io/density3d extras, window+2D animation modes)
- Minor: 1 (Requirements names only 2 of 11 extras)

Everything else verified correct: all 5 code snippets execute, all 18 URLs return 200, all 8 image references resolve, no stale/retired-API references, and named extras match pyproject.

### Sphinx / tutorials

# HyperTools 1.0 Sphinx/RTD Documentation Audit

Branch: dev-1.0-refactor. Build: `sphinx -b html` succeeded, **11 warnings, 0 errors**.

---

## SEVERITY: HIGH

### H1. Retired DataGeometry tutorial still published (content currency)
- `docs/tutorials.rst:56-62` publishes a **"DataGeometry Objects"** section ->
  `tutorials/geo.ipynb`. `geo.ipynb` contains real `DataGeometry` usage (2 hits) —
  a 0.x object that no longer exists in the 1.0 public API
  (`hypertools/__init__.py` exports none of it).
- Fix: drop the DataGeometry section from `tutorials.rst` and remove/retire
  `docs/tutorials/geo.ipynb` (+ `geo.rst`, `geo_files/`), or rewrite as a
  Pipeline/`return_model` tutorial.

### H2. Stale 0.x `.rst` tutorials duplicate the new `.ipynb` tutorials -> 8 ambiguous-doc warnings
- 8 `WARNING: multiple files found for the document "tutorials/X"` for
  align, analyze, cluster, geo, normalize, plot, reduce, text — each has BOTH a
  legacy `tutorials/X.rst` and the new executed `tutorials/X.ipynb`.
- Sphinx resolves the docname nondeterministically, so the **stale 0.x `.rst`**
  (retired API) can be published instead of the current executed notebook. Retired
  API present in the `.rst` copies: `geo.rst` (5 hits, DataGeometry), `text.rst`
  (5), `plot.rst` (1).
- Fix: delete the superseded `docs/tutorials/{align,analyze,cluster,geo,normalize,
  plot,reduce,text}.rst` (and their `*_files/` dirs) now that `.ipynb` versions
  exist. Clears all 8 warnings and the ambiguity.

---

## SEVERITY: LOW (cosmetic / trivial)

### L1. index.rst title underline too short
- `docs/index.rst:7` — `WARNING: Title underline too short.` The `====` under the
  long "**HyperTools**: A python toolbox..." H1 is shorter than the title.
- Fix: extend the `=` underline to >= title length.

### L2. sphinx_gallery_conf unpickleable config-cache warning
- `WARNING: cannot cache unpickleable configuration value: 'sphinx_gallery_conf'`
  — because `image_scrapers`/`first_notebook_cell` hold function objects. Cosmetic,
  pre-existing, harmless (only disables config caching). No action required.

### L3. scipy intersphinx/reference_url 404
- `WARNING: ...docs.scipy.org/doc/scipy/reference//_static/documentation_options.js:
  404`. From `sphinx_gallery_conf['reference_url']` scipy entry (double slash).
  Cosmetic/network. Optional: drop scipy from `reference_url` or fix trailing slash.

### L4. pipeline_order.rst cross-reference imprecision
- `docs/pipeline_order.rst:110-111` says `:doc:`tutorials`` includes "the
  story-trajectories walkthrough" — but story-trajectories is a **gallery example**
  (`examples/plot_story_trajectories.py` -> `auto_examples/`), NOT in `tutorials.rst`.
- `docs/pipeline_order.rst:103` `:ref:`...<examples-index>`` resolves (no build
  warning) but points into the gallery, not tutorials.
- Fix: reword line 110-111 to point at the gallery example, or add the walkthrough
  to tutorials.

---

## AUDIT RESULTS (checklist)

### 1. api.rst vs public exports — NO GAPS
All 16 named public exports present in `docs/api.rst`:
plot, analyze, reduce, align, cluster, normalize, manip, load, describe, predict,
impute, apply_model, save, Pipeline, set_interactive_backend, io.lsl_stream — all
have autosummary entries. Extra (valid) entries: reduce.autoencoders.* (6),
align.procrustes, tools.{text2mat,format_data,missing_inds,df2mat},
tools.gensim_models.* (6). No autosummary import/stub warnings in the build =>
every api.rst entry still resolves. **Coverage gaps: 0. Dead entries: 0.**
Note: `hyp.tools.*` is NOT retired in 1.0 — tools.format_data/missing_inds/df2mat/
text2mat remain public and are intentionally documented; example/tutorial uses of
them (plot_missing_data.py, plot_text.py, plot_gensim_text.py, plot.ipynb) are current.

### 2. pipeline_order.rst vs code — ACCURATE
`build_pipeline` `CANONICAL_ORDER = ('manip','normalize','reduce','align','cluster')`
(hypertools/core/pipeline.py:38) exactly matches the documented geometry order
(manip->normalize->reduce->align->cluster). Doc correctly frames impute (load/format)
before, and plot/animate + predict-overlay after, the pipeline builder's 5 stages.
Rationale (impute-first, manip in native space, normalize before variance/distance
models, reduce-before-align, cluster last, predict overlay) is accurate and consistent
with `analyze == align(reduce(normalize(x)))`. SVG alt-text matches. No action.

### 3. Tutorials — retired-API count
Genuinely-retired API in *published* tutorial notebooks: **1** (`geo.ipynb`,
DataGeometry — see H1). No `group=`, no geo-chaining, no retired `hyp.tools.*` in the
current `.ipynb` set. (Stale `.rst` duplicates carry more retired API — see H2 — but
those are legacy files that should be deleted, not maintained.)
New tutorials all have REAL executed outputs:
- wikipedia_embeddings.ipynb: 12 code cells, 17 executed, 17 outputs ✓
- lsl_streaming.ipynb: 6 code, 8 executed, 8 outputs ✓ (+lsl_stream.gif)
- conversation_trajectories.ipynb: 9 code, executed, 9 outputs ✓
- stock_forecasting.ipynb / projectile_kalman.ipynb: executed w/ outputs ✓
Story-trajectories exists as a gallery example (examples/plot_story_trajectories.py),
not a tutorial (see L4).

### 4. examples/ gallery — no retired API
No `DataGeometry` / `group=` / geo-chaining in examples/*.py. `hyp.tools.*` hits are
current public API in comments/docstrings. New 1.0 gallery scripts:
plot_story_trajectories.py, plot_datasets_tour.py, plot_pipelines_return_model.py,
plot_autoencoders.py, plot_gensim_text.py, plot_impute.py, plot_predict.py,
plot_apply_model.py, plot_procrustes.py, plot_shape_morph.py, plot_surface_morph.py,
animate_* — all use 1.0 dispatcher API.

### 5. conf.py — sane
Extensions fine (autodoc, numpydoc, autosummary, viewcode, sphinx_gallery,
sphinxcontrib.video, nbsphinx). Deliberate/justified settings: numpydoc_class_members_
toctree=False (avoids ~100 stub warnings), nbsphinx_execute='never', furo theme,
plotly scraper fallback, auto_examples/*.ipynb excluded. No misconfig. The
sphinx_gallery_conf pickle warning (L2) is inherent to embedding function objects.

### 6. Build warning summary
Total: **11 warnings, 0 ERROR, build succeeded.**
Actionable: 8x ambiguous-doc (H2) + 1x title underline (L1) = 9.
Cosmetic/pre-existing: sphinx_gallery pickle (L2) + scipy 404 (L3) = 2.


### CLAUDE.md / project meta

I have everything needed. Note: I'm in strict READ-ONLY mode with no Write/Edit tools, so I cannot create `project_meta.md`. Per my operating rules I return the full findings directly below — the complete grouped audit, followed by the requested summary.

---

# Project-Meta Doc Audit — HyperTools 1.0 (dev-1.0-refactor)

## CLAUDE.md — stale / inaccurate statements

**1. Line 12** — "`pytest` - Run all tests from the hypertools/ directory"
Correct: tests live in `tests/` (`[tool.pytest.ini_options] testpaths=["tests"]`); run from the repo root, not `hypertools/`.

**2. Line 17** — "`pip install -e ".[dev]"` … (all deps declared in pyproject.toml)"
Correct: `[dev]` does NOT install all optional deps. It omits the `text` extra (`pydata-wrangler[hf]` — transformers/sentence-transformers) and `predict-hf` (`chronos-forecasting`). Those paths are `importorskip`-guarded, not covered by a plain `[dev]` install.

**3. Lines 28–31 (major)** — "**DataGeometry Class** (`hypertools/datageometry.py`) — Central data container that holds raw data, transformed data, and transformation parameters / Stores matplotlib figure/axes handles and animation objects / Contains normalization, reduction, and alignment model parameters"
Correct: `datageometry.py` still exists but is now an INTERNAL, unpickle-only legacy shell (docstring: "INTERNAL -- not part of the public API"). It exists solely so hosted pre-1.0 example pickles resolve `hypertools.datageometry.DataGeometry`. In 1.0 `plot()` returns a matplotlib `Figure` and `load()` returns raw data — no DataGeometry is ever constructed or returned. This is the most misleading entry.

**4. Lines 33–41 (incomplete)** — "**Main API Functions** (`hypertools/__init__.py`)"
The listed set is missing the new 1.0 public API actually exported in `__init__.py`: `manip()`, `predict()`, `impute()`, `save()`, `apply_model`, `Pipeline`, `set_interactive_backend`, and the `io` submodule. Also, `analyze`/`normalize` are imported from `hypertools.tools`, while `reduce`/`align`/`describe`/`cluster` now come from their own subpackages (not `tools`).

**5. Lines 43–53 (major)** — "**Tools Module** (`hypertools/tools/`)" listing `align.py, reduce.py, normalize.py, cluster.py, format_data.py, text2mat.py, df2mat.py, load.py, missing_inds.py, procrustes.py"
Correct: `tools/` now contains only `align.py, analyze.py, df2mat.py, format_data.py, gensim_models.py, missing_inds.py, normalize.py, text2mat.py`. The 1.0 refactor MOVED: `reduce.py`→`hypertools/reduce/reduce.py`; `cluster.py`→`hypertools/cluster/cluster.py`; `load.py`→`hypertools/io/load.py`; align/procrustes→`hypertools/align/` (`align.py, hyperalign.py, procrustes.py, srm.py, null.py`). So `tools/reduce.py`, `tools/cluster.py`, `tools/load.py`, `tools/procrustes.py` no longer exist. Entirely unmentioned new top-level subpackages: `reduce/`, `cluster/`, `align/`, `manip/`, `io/`, `predict/`, `impute/`, `core/`. (New `tools/gensim_models.py`, `tools/analyze.py` also unlisted.)

**6. Lines 55–58 (major)** — "**Plot Module** … `draw.py` - Low-level drawing functions"
Correct: `plot/draw.py` is now just a 3-line compatibility shim (`from .matplotlib_backend import *`); the real low-level drawing code lives in `plot/matplotlib_backend.py` (~93 KB). The section also omits the many new plot modules: `matplotlib_backend.py, plotly_backend.py, interactive.py, animate.py, colors.py, density.py, surface.py, morph.py, trails.py, meshutil.py, multiindex.py, fonts.py`.

**7. Line 88** — "The codebase maintains compatibility with Python 3.9+"
Correct: `pyproject.toml` sets `requires-python = ">=3.10"` with classifiers 3.10–3.13. Floor is 3.10, not 3.9.

**Minor/incompleteness:** Line 7/overview ("built on top of matplotlib, scikit-learn, and seaborn") omits the new plotly backend and the `pydata-wrangler` core dependency. Lines 60–62 (`_externals/` ppca.py, srm.py) are still accurate, but note SRM also now lives at `align/srm.py` and PPCA at `impute/ppca.py`; and there is a separate top-level `hypertools/external/` dir not mentioned. Data-flow section (64–71) is conceptually OK but omits new `manip`/`predict`/`impute` capabilities.

Count: **7 clearly stale/inaccurate statements** (#1–#7 above), plus ~4 incompleteness items.

## AGENTS.md / AGENT.md
Neither exists (confirmed). Only `CLAUDE.md`. **Recommendation:** add an `AGENTS.md` (the emerging cross-tool standard) — either mirroring the corrected CLAUDE.md or a symlink/pointer — so non-Claude agents get the same guidance. Do NOT create now (read-only); flag for maintainer.

## pyproject.toml
Internally consistent overall. Python floor (3.10) matches classifiers. Extras: `interactive` (plotly, kaleido), `text` (pydata-wrangler[hf]), `predict` (pykalman, statsmodels, skaters), `predict-hf` (chronos-forecasting), `io` (openpyxl), `density3d` (scikit-image), `torch` (torch), `kaggle` (kagglehub), `lsl` (pylsl), `gensim` (gensim), `dev`. All extras map to real code paths (autoencoders, lsl.py, gensim_models.py, chronos.py, density.py, io/load.py, etc.) — no obviously unused extra.
Issues to flag:
- `dev` intentionally omits `text` (pydata-wrangler[hf]) and `predict-hf` (chronos), so a plain `[dev]` install does NOT exercise HF text-embedding or Chronos paths (matches CI + importorskip, but contradicts CLAUDE.md's "all deps").
- `dev` duplicates plotly/kaleido/openpyxl/scikit-image/torch/pykalman/statsmodels/skaters/kagglehub/pylsl/gensim already in named extras (intentional but redundant; could reuse extras).
- No `[project.scripts]`/console entry points defined (only `[project.urls]`) — fine for a library, but "entry points" per the audit request: none exist.

## CONTRIBUTING.md — stale
- Line 22 points to a mozsprint milestone (`milestone/5`) — long-obsolete, pre-1.0 framing.
- Line 45 links a **Gitter** channel (`gitter.im/hypertools/Lobby`) — effectively defunct.
- No mention of the actual dev workflow (`pip install -e ".[dev]"`, `pytest`, extras) — stale relative to 1.0. `RELEASE_NOTES_0.8.1.md` is also an old-version artifact.

## .github/workflows/test.yml
Installs `pip install -e ".[dev,torch]"` — covers torch, gensim, pylsl, kagglehub, scikit-image, openpyxl, plotly, statsmodels/pykalman/skaters, datasets (all in `[dev]`; torch also explicit). It does NOT install `text`/`predict-hf` (HF/Chronos) — those tests importorskip, so OK. `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}` is set on all three pytest steps to authenticate the fivethirtyeight loader's GitHub API calls (538 tests, `tests/test_load_538_kaggle.py`), raising the rate limit 60→5000/hr. pylsl relies on bundled native liblsl (no extra provisioning). Consistent and correct.

---

Summary: **7 stale CLAUDE.md statements** (biggest: DataGeometry is now an internal unpickle-only shell, not the central container; tools/reduce|cluster|load|procrustes moved to reduce/,cluster/,io/,align/ subpackages; plot/draw.py is a shim for matplotlib_backend.py; Python 3.9+ → 3.10+) + ~4 incompleteness gaps. **AGENTS.md: recommended** (absent; don't create yet). **pyproject:** consistent, no unused/undeclared deps; only note = `[dev]` omits text/predict-hf, so CLAUDE.md's "all deps" claim is wrong; no console entry points. CONTRIBUTING.md is stale (mozsprint milestone, Gitter). File not written (read-only mode).


# Commit-by-commit audit



## Commit batch 1

# QC Audit — batch_01 (PR #272, dev-1.0-refactor)

### 69be48a356 — Design spec: HyperTools 2.0 class-based refactor
- **What:** Adds a single design-doc markdown (361 lines) describing the 2.0 class-based reorg: base-class-per-area, remove DataGeometry, adopt datawrangler for the wrangling core, keep dev-2.0 plotting/animation, strangler migration keeping tests green.
- **Files:** `notes/2026-07-03-hypertools-2.0-class-refactor-design.md` (new).
- **Why:** Establishes the architecture plan for the refactor.
- **Review flags:** none — doc only, no code.

### f86f18569d — Plan 1: datawrangler adoption & verification (2.0 refactor step 0)
- **What:** Adds a 420-line TDD plan markdown for declaring the dw dep + text extra, probing dw 0.4.0 API/behavior, reconciling py3.13/CI.
- **Files:** `notes/...-plan-1-datawrangler-adoption.md` (new).
- **Why:** Bite-sized execution plan for step 0.
- **Review flags:** none — doc only.

### bb993fd06a — Plan 1: incorporate verified env facts (.venv, pandas<3 pin, dw#30, dw API)
- **What:** Rewrites the Plan-1 doc (134 ins / 172 del) to fold in recon findings: standardize on .venv (py3.12), pin pandas<3, verified dw text call, 242-pass baseline.
- **Files:** same plan-1 markdown.
- **Why:** Update plan with confirmed environment facts.
- **Review flags:** none — doc only. (Records the 242-pass baseline as a claim to verify later.)

### 9612e8857c — build: add datawrangler dep; pin pandas <3 for dw 0.4.0 (data-wrangler#30)
- **What:** pyproject: adds `pydata-wrangler>=0.4.0` to base deps, adds `text = pydata-wrangler[hf]` optional extra, and changes `pandas>=2.2.0` → `pandas>=2.2.0,<3`.
- **Files:** `pyproject.toml`.
- **Why:** Adopt dw as the wrangling core; pandas ceiling because dw 0.4.0 type detection breaks on pandas 3.0.
- **Review flags:** ⚠️ New hard runtime dependency (`pydata-wrangler`) pulled into base install — supply-chain/transitive-dep surface. ⚠️ `pandas<3` upper pin is a real constraint that can conflict with other user deps; explicitly marked TEMPORARY pending upstream dw#30 (tracked, not yet resolved). `[hf]` extra pulls torch/transformers but is opt-in only.

### a419634218 — test: probe datawrangler API surface (symbol existence)
- **What:** New parametrized test asserting ~17 dw symbols exist (funnel, stack/unstack, wrangle, zoo.text.*, core.*). Also adds empty `tests/core/__init__.py`.
- **Files:** `tests/core/test_dw_probe.py`, `tests/core/__init__.py` (new).
- **Why:** Verification-first gate; missing symbols should become filed dw issues + xfail, not silent green.
- **Review flags:** ⚠️ Tests couple hypertools CI to an external lib's private-ish surface (`datawrangler.core`, `datawrangler.zoo.text`) — will fail loudly on any dw refactor; intentional but brittle. Additive.

### aa0efb3959 — test: probe datawrangler behavior (stack/unstack, funnel, polars, text)
- **What:** Adds 4 behavior tests to the probe: stack/unstack MultiIndex round-trip, funnel over numpy/pandas/list, funnel over polars (importorskip), and `dw.wrangle(docs, model="CountVectorizer")` text embedding.
- **Files:** `tests/core/test_dw_probe.py`.
- **Why:** Prove the real round-trips later plans depend on.
- **Review flags:** none notable — additive tests, real calls. polars path gated by importorskip (won't fail if polars absent).

### 4c60d76563 — ci: run on dev-2.0-refactor; seed datawrangler coordination log (dw#30)
- **What:** Adds `dev-2.0-refactor` to CI push triggers (PR triggers unchanged: still master/dev only). Adds a coordination-log note file. Comment added around the py3.13 matrix row.
- **Files:** `.github/workflows/test.yml`, `notes/datawrangler_coordination.md` (new).
- **Why:** Exercise the refactor branch in CI; document dw coordination + dw#30.
- **Review flags:** ⚠️ CI now builds/imports dw on py3.13 where dw classifiers stop at 3.12 — comment pre-authorizes dropping the 3.13 row if it breaks (potential future coverage gap). Otherwise mechanical.

### 3c6b1f4fd5 — Plan 2: core layer (exceptions/shared/configurator/model) — additive strangler
- **What:** Adds a 542-line plan doc for the core layer.
- **Files:** `notes/...-plan-2-core-layer.md` (new).
- **Why:** Execution plan for Plan 2.
- **Review flags:** none — doc only.

### 268d0a83a7 — refactor(core): move exceptions to hypertools.core; shim _shared
- **What:** Moves the 3 exception classes verbatim into new `hypertools/core/exceptions.py`; `_shared/exceptions.py` becomes a re-export shim. New `core/__init__.py` re-exports them.
- **Files:** `hypertools/core/{__init__,exceptions}.py`, `hypertools/_shared/exceptions.py`, `tests/core/test_core_exceptions.py`.
- **Why:** Begin core layer; keep old import path working (strangler).
- **Review flags:** none — mechanical move; test asserts shim re-exports the SAME class objects (identity), guarding against isinstance breakage.

### 6c364cf09a — feat(core): add RobustDict and eval-free unpack_model
- **What:** New `core/shared.py` with `RobustDict` (dict returning a default instead of KeyError) and `unpack_model` (eval-free model-spec resolver: whitelist name match, parent_class passthrough, recursive dict unpack, unmatched string passthrough, else ValueError).
- **Files:** `hypertools/core/shared.py`, `core/__init__.py`, `tests/core/test_shared.py`.
- **Why:** Eval-free replacement for the fork's string→eval model lookup (security/robustness).
- **Review flags:** ⚠️ `unpack_model` dict branch requires all of `("model","args","kwargs")` present — a `{model,params}` dict falls through to the ValueError path (not handled here; core.model handles that form separately). ⚠️ `RobustDict.__getitem__` silently swallows missing keys — can mask typos in config lookups. Behavior covered by tests. Additive (not yet wired into runtime paths).

### 1ccf632592 — feat(core): central config.ini defaults via datawrangler configurator
- **What:** New `core/config.ini` (plot/reduce/align/cluster default sections) + `core/configurator.py` exposing `get_default_options` (merges dw defaults with hypertools config into a RobustDict) and `apply_defaults`.
- **Files:** `hypertools/core/{config.ini,configurator.py}`, `core/__init__.py`, `tests/core/test_configurator.py`.
- **Why:** Single source of truth for defaults shared with dw.
- **Review flags:** ⚠️ Config values are strings from INI (`ndims`="3", `verbose`="False") — callers must cast; test casts with `int(...)`. Note `verbose = False` as a string is truthy if used directly. ⚠️ Merges over `dw.core.get_default_options()` — hypertools defaults now inherit/override dw's global defaults (coupling). Additive.

### b89ad8db16 — refactor(core): relocate apply_model to core.model; accept fork dict form
- **What:** Moves `apply_model`/`supported_models`/helpers into new `core/model.py`; `tools/apply_model.py` becomes a shim. One behavior change: `_resolve_model` now reads params from `model.get('params', model.get('kwargs', {}))`, so the fork `{model,args,kwargs}` dict form works alongside `{model,params}`. Registry imports switched to `..tools.*`; format_data import made lazy to avoid a new circular import.
- **Files:** `hypertools/core/model.py` (new, 199 lines), `hypertools/tools/apply_model.py` (→shim), `core/__init__.py`, `tests/core/test_model.py`.
- **Why:** Establish core.model as source of truth; support both dict spec forms.
- **Review flags:** ⚠️ Behavior change (additive): `kwargs` key now consumed as model params — verify no caller passed a dict with a `kwargs` key meaning something else. ⚠️ Lazy `format_data` import inside the function is a circular-import workaround (documented) — mildly non-obvious. ⚠️ core.model is now imported at `core/__init__` load time, so importing `hypertools.core` triggers the tools/reduce/cluster import chain lazily only when apply_model runs (registry build is lazy) — OK, but worth confirming no import-time cycle. Tests only cover the two dict forms with PCA; the string/instance/pipeline paths are unchanged moved code.

### a10bb082c0 — Notes: Plans 1-2 complete (dw adoption + core layer); Plan 3 design analysis
- **What:** Adds a 41-line session notes file.
- **Files:** `notes/session_2026-07-03_refactor_plans1-2.md` (new).
- **Why:** Progress log + Plan 3 design analysis.
- **Review flags:** none — doc only.

### 78ce4cde8a — Plan 3: external/ + manip/ (compat-wrapper design)
- **What:** Adds a 541-line plan doc for the external/ and manip/ layers.
- **Files:** `notes/...-plan-3-external-manip.md` (new).
- **Why:** Execution plan for Plan 3.
- **Review flags:** none — doc only.

### 015d75c751 — refactor(external): quarantine vendored ppca+srm in hypertools.external; shim _externals
- **What:** Moves vendored PPCA and SRM/DetSRM (brainiak) code into new `hypertools/external/{ppca,brainiak}.py`; old `_externals/{ppca,srm}.py` become re-export shims. Verified: new files are byte-identical to originals except one added provenance/license comment line at top of each.
- **Files:** `hypertools/external/{__init__,ppca,brainiak}.py` (new), `hypertools/_externals/{ppca,srm}.py` (→shims), `tests/external/{__init__,test_external_move}.py`.
- **Why:** Quarantine third-party code under `external/` with clear licensing (Apache-2.0 SRM, pca-magic PPCA); keep old paths working.
- **Review flags:** none — confirmed pure move (diff = 1 comment line each); test asserts shims re-export the SAME objects. Note: file renamed srm.py→brainiak.py, so grep-by-filename references should be re-checked, but import paths are preserved by shims.

### d1ae8c5085 — feat(core): add get() elementwise list indexer (Resample prereq)
- **What:** Adds `get(value, i)` to `core/shared.py`: returns `value[i]` for a list/tuple in range, else returns `value` unchanged (broadcast semantics). Exported from `core/__init__`.
- **Files:** `hypertools/core/shared.py`, `core/__init__.py`, `tests/core/test_get.py`.
- **Why:** Let manipulators accept either one shared param or a per-dataset list; prereq for Resample.
- **Review flags:** ⚠️ Out-of-range index returns the WHOLE value rather than raising (documented broadcast semantics) — could silently mask an index bug in callers. Additive, tested.



## Commit batch 2

### b7db98a21f — feat(manip): add Manipulator base class
- **What:** Adds sklearn-compatible `Manipulator` base (fit/transform/fit_transform wrapping a fitter/transformer/required-params triple; stores fit dict as attrs).
- **Files:** hypertools/manip/{__init__,common}.py; tests/manip/test_manip_base.py.
- **Why:** Foundation for Normalize/ZScore/Smooth/Resample in the 2.0 manip package.
- **Review flags:** none — additive/mechanical.

### cbafc3d1a2 — feat(manip): add Normalize (min-max) and ZScore manipulators
- **What:** Adds Normalize and ZScore manipulators (fitter/transformer via dw funnel/apply_stacked decorators; axis=1 via transpose recursion).
- **Files:** hypertools/manip/{normalize,zscore}.py, __init__.py; tests/manip/test_normalize_zscore.py.
- **Why:** First concrete manipulators.
- **Review flags:** ⚠️ Ships a latent axis=1 bug (transpose branch self-calls decorated transformer → double-stack KeyError); tests only cover axis=0. Fixed in 536ed32. Redundant attr assignment (set in super() then again in __init__).

### 22def749023 — feat(manip): add Smooth (savgol) and Resample (pchip) + manip dispatcher
- **What:** Adds Smooth (Savitzky-Golay) and Resample (pchip interp) manipulators plus `manip` dispatcher resolving model specs via unpack_model.
- **Files:** hypertools/manip/{smooth,resample,manip}.py, __init__.py; notes/session_2026-07-03…md; tests/manip/test_smooth_resample.py.
- **Review flags:** ⚠️ Notes file documents a known-shipped axis=1 bug (fixed next commit) and an owed Gaussian-smooth mode (TODO/Plan 6). Resample.transformer has bare `except IndexError` re-key via `int(c)` fallback (non-obvious). Same latent axis=1 bug in Smooth.

### 536ed32d35 — fix(manip): axis=1 transpose recursion double-stacked via decorator
- **What:** Fixes axis=1 KeyError in Normalize/ZScore/Smooth by splitting the always-axis0 core into decorated `_transform_stacked` and making the public `transformer` an undecorated dispatcher that transposes raw (pre-stack) data.
- **Files:** hypertools/manip/{normalize,smooth,zscore}.py; tests/manip/test_axis1.py.
- **Why:** apply_stacked re-stacked transposed frames, leaking a synthetic ID level so fitted params couldn't be looked up.
- **Review flags:** ⚠️ Real behavior fix (axis=1 now works). Well-tested (adds axis=1 tests). Resample untouched (no decorator). Low risk.

### 0976879a1c — feat(api): expose hyp.manip; hyp.normalize compat unchanged
- **What:** Exports `manip` at top-level `hypertools` namespace.
- **Files:** hypertools/__init__.py; tests/manip/test_public_api.py.
- **Why:** Public API for new manip dispatcher; legacy hyp.normalize left intact.
- **Review flags:** none — additive/mechanical.

### 60b598f294 — build(deps): upgrade datawrangler 0.4.0->0.5.0, lift pandas<3 pin
- **What:** Bumps pydata-wrangler (+[hf]) to >=0.5.0, removes pandas `<3` ceiling (now `pandas>=2.2.0`), adds a CI acceptance gate pinning pandas 3.0 on ubuntu/py3.12.
- **Files:** pyproject.toml; .github/workflows/test.yml; notes/datawrangler_coordination.md.
- **Why:** dw 0.5.0 fixes pandas-3 type detection (dw#30).
- **Review flags:** ⚠️ Dependency/version change (pandas 3.0 now allowed; runtime surface widens). Notes an unfiled forward-looking Pandas4Warning from dw. Claims 293 tests pass; not independently verified here.

### 8826124f26 — docs(plan): add Plan 4 (reduce+align+cluster) implementation plan
- **What:** Adds 926-line planning doc.
- **Files:** 1 markdown plan (docs/dev notes).
- **Why:** Design record for the reduce/align/cluster migration.
- **Review flags:** none — additive/mechanical (doc only).

### da576ec345 — refactor(reduce): re-home reduce+describe into hypertools.reduce; shim tools
- **What:** Moves reduce.py/describe.py into new hypertools/reduce/ package; old tools/ paths become re-export shims (`from ..reduce… import *`).
- **Files:** hypertools/reduce/{reduce,describe,__init__}.py; hypertools/tools/{reduce,describe}.py (→shim); tests/reduce/test_reduce_module.py.
- **Why:** 2.0 package re-homing; preserves core.model registry import of `models` via shim.
- **Review flags:** ⚠️ Code moved verbatim (test asserts shim `is` new fn). Retains wildcard `from .._shared.helpers import *`. Mechanical.

### 9e985ba2be — feat(align): add Aligner base + pad/trim_and_pad helpers
- **What:** Adds list-of-DataFrames `Aligner` base + pad/trim_and_pad helpers (fit unstacks→trims to common rows→pads to common cols).
- **Files:** hypertools/align/{common,__init__}.py; tests/align/test_align_base.py.
- **Why:** Foundation for align algorithms.
- **Review flags:** none — additive. Note: trim_and_pad uses set-intersection on row labels → row order not deterministic (unordered set→list), could reorder rows; acceptable for algorithms but worth noting.

### c613e2e456 — feat(align): add procrustes fn + Procrustes/NullAlign children; shim tools/procrustes
- **What:** New align/procrustes.py with legacy `procrustes()` fn plus new `align`/`xform`/fitter/transformer + Procrustes and NullAlign classes; tools/procrustes.py becomes shim.
- **Files:** hypertools/align/{procrustes,null}.py, __init__.py; hypertools/tools/procrustes.py (→shim); tests/align/test_procrustes_child.py.
- **Why:** Class-based aligners over shared SVD primitive.
- **Review flags:** ⚠️ Legacy `procrustes()` still uses `np.asmatrix` (deprecated in NumPy 2.0, pending removal). Duplicated align logic (fit-local `fit()` vs module `align()`). Early-return identity when source==target.

### dd235b9014 — feat(align): add HyperAlign (dev-2.0 rescaled hyperalignment algorithm)
- **What:** Adds HyperAlign porting dev-2.0's per-pass-rescaled hyperalignment; accumulates composed per-dataset projections so transform reproduces align() output.
- **Files:** hypertools/align/hyperalign.py, __init__.py; tests/align/test_hyperalign.py.
- **Why:** Fork version omitted rescale, collapsing data toward zero; this preserves scale.
- **Review flags:** ⚠️ Non-obvious numerical algorithm (per-pass Frobenius rescale folded into projections). n_iter=0/n==1 → identity. Well-tested (rotation recovery, scale preservation). Correctness rests on rescale-into-proj math.

### 5be6ca2359 — feat(align): add SRM + DetSRM adapters over external.brainiak (RSRM not carried)
- **What:** Adds SharedResponseModel/DeterministicSharedResponseModel Aligner children wrapping vendored external.brainiak SRM/DetSRM.
- **Files:** hypertools/align/srm.py, __init__.py; tests/align/test_srm.py.
- **Why:** SRM alignment in class API.
- **Review flags:** ⚠️ Intentional feature drop: RSRM (Robust SRM) not carried (documented). Single fit (no n_iter). Test import quirk noted (hyp.align callable shadows submodule attr).

### 22cab2b49c — feat(align): add align dispatcher + classic tools/align compat shim
- **What:** Adds align dispatcher (unpack_model over aligners) and rewrites classic tools/align.py to translate legacy string/dict specs to the new class dispatcher.
- **Files:** hypertools/align/align.py, __init__.py; hypertools/tools/align.py (rewritten→wrapper); tests/align/test_align_dispatcher.py.
- **Why:** Unify classic and class-based align paths.
- **Review flags:** ⚠️ Behavior changes to classic API: SRM path no longer does n_iter repeated re-fits (now single fit via new SRM class); dropped legacy warnings (len-1 list "cannot be aligned", features>samples overfitting warning); `align=True` now raises. Old `_externals.srm` import removed. Results should match hyperalign but SRM semantics changed — flag for regression check.

### 49a87c6693 — refactor(cluster): re-home cluster into hypertools.cluster; shim tools
- **What:** Moves tools/cluster.py verbatim to cluster/cluster.py (format_data import routed via tools.format_data); old path becomes re-export shim.
- **Files:** hypertools/cluster/{cluster,__init__}.py; hypertools/tools/cluster.py (→shim); tests/cluster/test_cluster_module.py.
- **Why:** 2.0 re-homing; keeps core.model registry (models/mixture_models) resolving.
- **Review flags:** none — additive/mechanical (code identical; test asserts shim `is` new fn).

### a24f36b820 — refactor(io): re-home load+sources+streaming into hypertools.io; shim tools
- **What:** Moves load/sources/streaming into new hypertools/io/ package; old tools/ paths become re-export shims. plot_stream temporarily rides in io.streaming.
- **Files:** hypertools/io/{load,sources,streaming,__init__}.py; hypertools/tools/{load,sources,streaming}.py (→shims); tests/io/test_io_module.py.
- **Why:** 2.0 re-homing (Plan 5).
- **Review flags:** ⚠️ Pure move+shim, but load.py contains network fetch (Google Docs/Dropbox example-data URLs, requests) — no logic change, just relocated. plot_stream placement is a documented temporary (Plan 6 TODO).

### cceccd19c9 — docs(plan): add Plan 5 (io) implementation plan
- **What:** Adds 266-line io planning doc.
- **Files:** 1 markdown plan.
- **Why:** Design record for io re-homing.
- **Review flags:** none — additive/mechanical (doc only).



## Commit batch 3

I have audited all 16 commits. I'm in read-only mode without file-writing tools, so I'm returning the full audit content here for the parent to persist.

---

### e23be72416 — docs(plan): Plan 5 save.py = plain pickle
- **What:** Edits Plan 5 planning doc to specify `save.py` as plain pickle instead of `dw.io.save` (which has reversed args + non-pickle format).
- **Files:** `docs/.../plan-5-io.md` (4+/12-).
- **Why:** Correct the plan before implementation.
- **Review flags:** none — doc only.

### f07b1f3e34 — feat(io): add save + expose hyp.save/hyp.io
- **What:** New `io/save.py` with `save(obj, fname)` that pickles any object; wires `hyp.save` and `hyp.io`. Figure-export deferred.
- **Files:** `hypertools/io/save.py` (+22), `__init__.py`, `io/__init__.py`, `tests/io/test_save.py` (+24).
- **Why:** Data serialization surface for 2.0.
- **Review flags:** ⚠️ Uses `pickle` (deserialization is unsafe on untrusted input) — but explicitly documented, same trust model as numpy/pandas pickle; `**kwargs` accepted but ignored. Low risk, additive.

### f0276c7f56 — docs(plan): add Plan 6
- **What:** Adds Plan 6 implementation doc (plot reorg + colors + gaussian Smooth).
- **Files:** `docs/.../plan-6-plot-colors.md` (+337).
- **Why:** Plan for subsequent commits.
- **Review flags:** none — doc only.

### b364cdc0dd — feat(manip): add gaussian mode to Smooth
- **What:** Adds `mode='gaussian'` (default stays `savgol`) to `Smooth` using `gaussian_filter1d(sigma=sqrt(var))`, var default 300; rewires weights script to use `hyp.manip`.
- **Files:** `manip/smooth.py`, `scripts/generate_weights_trajectory.py`, `tests/manip/test_smooth_gaussian.py` (+39).
- **Why:** Reusable gaussian smoothing for the weights recipe.
- **Review flags:** ⚠️ New required params `mode`/`var` threaded through fitter — must exist on all Smooth calls; default preserved so backward-compatible. Good test coverage (scipy parity). Additive.

### dc71d2aa85 — refactor(plot): consolidate coloring into plot.colors
- **What:** Moves `tools/colors.py` → `plot/colors.py`; `tools/colors.py` becomes a `from ..plot.colors import *` shim; repoints `plot.py`.
- **Files:** `plot/colors.py` (+132), `tools/colors.py` (shim), `plot/plot.py`, `tests/plot/test_colors_module.py`.
- **Why:** Single coloring surface.
- **Review flags:** none — mechanical move with identity-preserving shim (test asserts `new is old`).

### 304d1cc285 — refactor(plot): rename draw→matplotlib_backend, interactive→plotly_backend
- **What:** Renames two backend modules; old files become `import *` shims; repoints `plot.py` and a comment.
- **Files:** `matplotlib_backend.py` (+677), `plotly_backend.py` (+666), `draw.py`/`interactive.py` (shims), `plot.py`, test.
- **Why:** Clearer backend naming.
- **Review flags:** none — mechanical rename; shims re-export public + underscore names.

### 787b763950 — refactor(plot): extract animation save helpers into plot.animate
- **What:** Moves `_save_animation`, `_SVGFrameCollector`, `_save_animated_svg` from `plot.py` into new `plot/animate.py`; imports them back.
- **Files:** `plot/animate.py` (+94), `plot/plot.py` (-79), test.
- **Why:** Modularize animation-save logic (closes Plan 6).
- **Review flags:** none — pure code move, imported at top of plot.py.

### 66dfc2eb24 — docs(plan): add Plan 7
- **What:** Adds Plan 7 doc (DataGeometry removal + API finalization).
- **Files:** `docs/.../plan-7-geo-removal.md` (+116).
- **Why:** Plan for geo removal.
- **Review flags:** none — doc only.

### f8b54076b1 — refactor(tools): repoint production imports off shims
- **What:** Repoints 4 production files' imports from `tools.*` re-export shims to real modules (io.load, reduce.reduce, cluster.cluster, core.model, io.streaming). Shims left in place.
- **Files:** `__init__.py`, `core/model.py`, `plot/plot.py`, `tools/text2mat.py`.
- **Why:** Move consumers to canonical module homes.
- **Review flags:** none — pure import-path change, zero behavior change.

### 12a53deffa — refactor(tools): retire 9 re-export shims; repoint tests
- **What:** Deletes 9 pure re-export shims, prunes `tools/__init__.py`, repoints all test imports, fixes 2 out-of-scope shim-dependent imports (datageometry.py, tools/analyze.py) that broke `import hypertools`.
- **Files:** 12 hypertools files, 17 test files.
- **Review flags:** ⚠️ Deletes 4 shim-parity tests (`old is new`) — justified: no old path remains; all real-module behavior tests kept/repointed. Removes public `tools.*` re-export paths (breaking for external code importing `hypertools.tools.load` etc.) but intentional for 2.0. Reports 331 passed.

### a742225210 — feat(align): vendor RSRM + RobustSharedResponseModel adapter
- **What:** Vendors Robust SRM (Turek 2017) into `external/brainiak.py`, adds `RobustSharedResponseModel` Aligner + `RSRM` dispatcher alias; registered in ALIGNERS.
- **Files:** `external/brainiak.py` (+423), `align/srm.py`, `align/align.py`, `align/__init__.py`, `tools/align.py`, `tests/align/test_rsrm.py`, `test_srm.py`.
- **Why:** Restore RSRM aligner.
- **Review flags:** ⚠️ Flips `test_rsrm_not_exported`→`test_rsrm_now_exported` — justified (state genuinely changed). Verified: vendored `RSRM.transform` was modified to return only the shared-response list (not the `(R,S)` tuple), so the shared `transformer` adapter is compatible. Solid.

### 508fbe7478 — feat(plot): return figure/(fig,ani) + return_model bundle; drop DataGeometry return
- **What:** `plot()` now returns bare Figure (static), `(fig, line_ani)` (animated mpl), or `{'fig','xform_data','models'}` when `return_model=True`. Removes the DataGeometry construction and the kwargs/reduce_dict bundling block.
- **Files:** `plot/plot.py` (-129 net logic), 7 test files rewritten.
- **Why:** Plan 7 API finalization — no more geo returns.
- **Review flags:** ⚠️ Major breaking API change to plot()'s return contract. `models` are specs, not fitted instances (documented). Return shape now branches on `line_ani is not None`. Extensive test rewrites (test_plot +137/-... etc.) — reasonable but worth confirming assertions weren't merely loosened.

### 9bbdb9e5cd — feat(io): streaming plots carry fig.stream_info
- **What:** `plot_stream` now consumes a Figure from `hyp_plot` (not a geo), reaches artists via `fig.axes[0].lines`, and stores all outputs in a new `fig.stream_info` dict instead of mutating geo attributes.
- **Files:** `io/streaming.py`, `tests/test_streaming.py` (16 tests rewritten), `test_load_sources.py`.
- **Why:** Align streaming with the new figure-return contract.
- **Review flags:** ⚠️ Behavior/return change (geo→fig). `stream_info` now also carries `data`/`xform_data` (previously separate geo attrs). Test rewrite large; assertion intent preserved (geo.ax→fig.axes[0] etc.).

### 71d3e7542e — chore(scripts): add example-dataset re-host regeneration script
- **What:** Adds `scripts/rehost_example_datasets.py` to extract `.data` from hosted DataGeometry pickles into plain pickles (gitignored `rehost/`).
- **Files:** `.gitignore`, `scripts/rehost_example_datasets.py` (+97).
- **Why:** Prep for geo deletion. (Note: this script is deleted again in commit 15.)
- **Review flags:** none — additive script; not wired into package.

### 9a9353b712 — refactor(geo): hide DataGeometry as internal unpickle-only class; load() returns raw data
- **What:** Trims DataGeometry to a minimal internal unpickle-only class (drops plot/save/transform/get_formatted_data); removes public `hyp.DataGeometry`; `load()` now returns raw data (or analyze() output) instead of a geo; deletes dead `check_geo`, format_data 'geo' branch, and the rehost script.
- **Files:** `datageometry.py` (-~280), `__init__.py`, `io/load.py`, `_shared/helpers.py`, `tools/format_data.py`, `tools/text2mat.py`, `plot/plot.py`, +26 geo tests rewritten (test_geo.py deleted).
- **Why:** Plan 7 Task 7 final geo removal.
- **Review flags:** ⚠️ Big deletion + breaking API (`hyp.DataGeometry` gone, load() return type changed). ⚠️ Minor latent inconsistency: `get_dtype` in helpers.py still returns `'geo'` for DataGeometry instances, but format_data's `'geo'` dispatch branch was deleted — an internal geo reaching format_data would fall through to the raw `else` append. Low impact (geo is unpickle-only; load extracts raw data before use). Reports 318 passed. `plot.py` gains explicit `import copy` (was leaking via `import *`) — good hygiene fix.

### a5a1cf9257 — docs(plan): add Plan 8
- **What:** Adds Plan 8 doc (docs/gallery/notebooks migration + Playwright verify).
- **Files:** `docs/.../plan-8-docs.md` (+106).
- **Why:** Plan for docs migration.
- **Review flags:** none — doc only.

---

batch 03: 16 audited; high-risk: none. Notable behavior/API changes (all intentional per Plans 6-7): 508fbe7478 (plot return contract), 9a9353b712 (DataGeometry removal), 9bbdb9e5cd (streaming return); minor latent flag: get_dtype still returns 'geo' after format_data dropped its geo branch.


## Commit batch 4

# QC audit — PR #272 (HyperTools 1.0) — batch 04

### 310e127d22 — docs(examples): migrate 20 gallery examples to figure-return / raw-load API
- **What:** Mechanically updates 20 `examples/*.py` gallery scripts to the 2.0 API: `hyp.load` returns raw arrays, `hyp.plot(...)` returns a bare Figure (no `DataGeometry`/`.get_data()`/`line_ani`).
- **Files:** examples/{analyze,animate*,chemtrails,explore,plot_*,precog,save_*}.py.
- **Why:** DataGeometry removed in 2.0; examples must use the new return model.
- **Review flags:** none — additive/mechanical doc-source migration (small net -12 lines).

### 3587310b10 — docs: repurpose geo example to the 2.0 return model (DataGeometry removed)
- **What:** Rewrites `examples/plot_geo.py` from the removed `geo` object API to demonstrate the 2.0 Figure return + `return_model=True` dict bundle `{'fig','xform_data','models'}`; saves to a tempdir.
- **Files:** examples/plot_geo.py.
- **Why:** Old example used geo.plot/save/transform/get_data, all removed.
- **Review flags:** none — doc-source only. Minor: title underline length mismatch (RST cosmetic), harmless.

### 1dbe999246 — docs(tutorials): migrate 13 tutorials to 2.0 figure-return API + re-execute
- **What:** Re-executes/updates 13 tutorial notebooks to the 2.0 API; regenerates 3 embedded gif assets. Large (+2946/-841) but notebook JSON + binaries.
- **Files:** docs/tutorials/*.ipynb, *.gif.
- **Why:** Tutorials must run under 2.0 API and show current outputs.
- **Review flags:** none — generated/doc content; skimmed, no source code. Not code-reviewable beyond spot check.

### 3de22f5624 — fix(plot): plotly ',' (and all mpl markers) render as markers; legend placed right on both backends
- **What:** Completes plotly `_MARKER_SYMBOLS` table so every mpl marker char maps to a plotly symbol (previously `, 1 2 3 4 P X | _` fell through to lines-only); adds 3D symbol fallbacks. Adds `_fit_right_legend()` (shrink-axes approach) to keep wide 3D-plot legends on-canvas after tight_layout.
- **Files:** hypertools/plot/plot.py, plotly_backend.py; tests/test_interactive.py, test_plot.py.
- **Review flags:** ⚠️ behavior change (marker rendering + legend geometry). `_fit_right_legend` shrink approach is superseded by commit c4abaee1 (widen approach) later in this batch — verify final behavior. Well-tested.

### a1e6526c05 — docs(examples): add shapes-zoo morphing gallery example
- **What:** New self-contained sphinx-gallery example morphing 7 shapes-zoo point clouds via Hungarian matching (`linear_sum_assignment`) + FuncAnimation with rotating camera; 910 frames.
- **Files:** examples/plot_shape_morph.py (new, 92 lines).
- **Why:** Showcase 2.0 API + shapes zoo.
- **Review flags:** none — additive example. Minor: reaches into `fig.axes[0]`/`get_lines()[0]` internals and manual `set_3d_properties`; example-only, acceptable.

### a8585cb7a7 — fix(plot): exported animation gifs preserve real-time duration (were ~6x too fast)
- **What:** Decouples per-frame delay from frame count: export path now uses `frame_ms = 1000/frame_rate` (and `fps=frame_rate`) over the full frame set instead of `1000*duration/n_frames`, which collapsed playback when frames were subsampled. Regenerates dev demo gifs.
- **Files:** hypertools/plot/animate.py (comment only), plotly_backend.py; tests/test_animation_export.py; dev/*.gif.
- **Why:** Subsampled export played gifs ~6x too fast.
- **Review flags:** ⚠️ behavior change to exported gif/mp4 timing. Sound reasoning, regression tests assert frame count + total duration. Low risk.

### d987a7d757 — docs: rebuild gallery+API under 2.0 API (figure-return, no geo); fix api.rst + stale stubs
- **What:** Large regeneration of `docs/auto_examples/**` (generated sphinx-gallery output) plus real edits to `docs/api.rst` (removed stale geo stubs).
- **Files:** docs/api.rst (real), docs/auto_examples/** (generated, many binaries).
- **Why:** Rebuild docs under 2.0 API.
- **Review flags:** none — mostly generated. Only api.rst is real doc source; change is removal of obsolete stubs, consistent with 2.0.

### a57548a615 — docs: Playwright visual verification of built docs + PR evidence
- **What:** Adds `scripts/verify_docs_playwright.py` (serves built docs over local HTTP, drives headless Chromium, asserts non-blank images/animations + branch-aware Colab links) plus screenshot evidence + PR_EVIDENCE.md.
- **Files:** scripts/verify_docs_playwright.py (new, 355 lines), docs/images/v2.0-docs/*.png.
- **Why:** Visual QC evidence for the docs migration.
- **Review flags:** ⚠️ network/deps — dev tooling requiring Playwright+chromium; local-only HTTP server, not imported by package. Not shipped in library. Low risk.

### 764e6d78a3 — fix: address whole-branch review findings (load dict-arg crash + 4 minor)
- **What:** Fixes 5 review findings: `load` used `any({reduce,ndims,align,normalize})` which crashes on unhashable dict args → generator form; HDBSCAN n_clusters guard now checks resolved model name not raw arg; `return_model` bundle now carries `animation` handle; `update_lines_parallel` uses `elev` param not hardcoded 10; docstring fix.
- **Files:** hypertools/io/load.py, plot/plot.py, plot/matplotlib_backend.py, core/model.py; tests/test_load.py, test_plot.py.
- **Review flags:** ⚠️ real bug fixes (crash paths). Correct. Note: `apply_model` mode='auto' docstring changed to claim predict_proba-first ordering — doc-only, assumes existing code intent; not verified against impl here.

### 346ddc6cb4 — fix(plot): zoom out animated plots slightly so the bounding box is never clipped
- **What:** Adds `_anim_box_zoom` (mpl, 9/(9-zoom)=1.125 vs prior 1.25) and `_anim_zoom_r` (plotly, *1.1) applied only in animation paths; static plots byte-identical. Regenerates dev gifs.
- **Files:** hypertools/plot/matplotlib_backend.py, plotly_backend.py; tests/test_plot.py, test_plotly_trails.py.
- **Why:** User-requested comfort margin (commit notes measurements show no actual clipping pre-fix).
- **Review flags:** ⚠️ behavior change (animation framing). Cosmetic/subjective, static unaffected, tested. Low risk.

### a16cf34fe1 — fix(plot): exclude animation trails from the legend (no duplicate entries)
- **What:** Sets mpl animation trail artists' label to `_nolegend_` so each dataset appears once in the legend (was duplicated: window + tail). Plotly already handled this.
- **Files:** hypertools/plot/matplotlib_backend.py; tests/test_animation_export.py.
- **Review flags:** ⚠️ minor behavior change; well-scoped, tested. Low risk.

### bb12cd89d0 — fix: address 6 bugs surfaced by open-issue triage against dev-2.0
- **What:** (#259) moves `pdf/ps.fonttype=42` from module import-time into `manage_backend` scope so import no longer mutates global rcParams; (#223) `update_position` no longer calls Axes3D-only `get_proj` on 2D + tuple-shape fix; (#146/#190) cluster injects n_clusters only if model signature accepts it + registers MeanShift/DBSCAN/OPTICS/AffinityPropagation; (#148) `show=False` closes fig; (#162) reduce accepts custom class/instance without UnboundLocalError; #214 docstring.
- **Files:** hypertools/cluster/cluster.py, io/load.py, plot/backend.py, matplotlib_backend.py, plot.py, reduce/reduce.py; 3 test files.
- **Review flags:** ⚠️ HIGH-TOUCH — 6 behavior changes across core modules. reduce.py instance-vs-class handling is non-trivial (mutates instance.n_components on row-count fallback). `show=False` now closes figures (behavior change; guarded to skip user-supplied ax + plotly). rcParams-scope move is correct but changes when fonttype applies. Good regression tests; warrants the closest look in this batch.

### fa8a643687 — docs: regenerate animated gallery with the anim zoom-out + legend fix
- **What:** Regenerates animated gallery assets (gifs/mp4/rst/execution-times) reflecting commits 10/11.
- **Files:** docs/auto_examples/** (generated), examples/spin.gif.
- **Why:** Refresh generated outputs.
- **Review flags:** none — generated assets. Note: examples/spin.gif and docs copy grew ~7.9MB→13.2MB (larger binary committed); size-only concern.

### 538db81539 — docs(notes): issue-triage catalog + animation-fix PR evidence
- **What:** Adds `notes/issues-to-close-on-merge.md` (67-issue triage catalog) + before/after PNG evidence.
- **Files:** notes/issues-to-close-on-merge.md, docs/images/v2.0-anim-fix/*.png.
- **Why:** Merge-time issue-closing guide + evidence.
- **Review flags:** none — notes/evidence only, no code.

### c4abaee1e0 — fix(ci): Windows dw import, mpl-3.11 canvas access, and robust legend fit
- **What:** Sets `HOME` via `os.environ.setdefault` before importing datawrangler (Windows lacks HOME → dw crash); guards `update_position` when canvas renderer absent (post-`plt.close`, mpl≥3.11); rewrites `_fit_right_legend` to WIDEN the figure (measuring rasterized pixels under `rcParamsDefault` via fresh Agg canvas) instead of shrinking axes.
- **Files:** hypertools/core/configurator.py, plot/matplotlib_backend.py, plot/plot.py; tests/test_plot.py.
- **Review flags:** ⚠️ non-obvious logic. `_fit_right_legend` rewrite renders a fresh FigureCanvasAgg up to 6x, freezes layout engine, and resizes figure/axes (caps growth 3x) — heavyweight and it mutates figure size on every plot with a legend. `os.environ.setdefault("HOME",...)` mutates process env (only if unset; low risk). Correct fixes but the legend logic is the subtlest code in the batch; watch for interaction with user-set figure sizes.

### 4b5048ecdb — test: pixel-based regression for wide-legend clipping (saved image)
- **What:** Adds a parametrized test saving a figure with long/many legend labels and asserting a >4px right margin in the saved PNG pixels; guards the c4abaee1 widen fix.
- **Files:** tests/test_plot.py.
- **Review flags:** none — test-only, additive.

---
batch 04: 16 audited; high-risk: bb12cd89 (6 cross-module behavior changes), c4abaee1 (subtle legend-widen logic). Both well-tested; no blockers.



## Commit batch 5

I've completed my audit. Since I'm in read-only mode (no write/edit tools available), I'm returning the full `batch_05.md` content directly for the parent agent to persist.

---

### 9433f99666 — docs: regenerate legend gallery examples with the widen-to-fit legend fix
- **What:** Regenerated 6 gallery PNGs (plot_PPCA/legend/missing_data + thumbs) and updated the paired `.rst` output metadata + `sg_execution_times.rst`. No source code.
- **Files:** `docs/auto_examples/*.png`, `*.rst`, `sg_execution_times.rst` (all generated).
- **Why:** Re-render galleries after the widen-to-fit legend fix landed elsewhere.
- **Review flags:** none — generated docs only, no library/test change.

### 0617424250 — docs(notes): session log for CI-green + legend fixes + dw#32 coordination
- **What:** Adds two notes files (session log + 3 lines to coordination note).
- **Files:** `notes/session_2026-07-04_ci_green_legend.md`, `notes/datawrangler_coordination.md`.
- **Why:** Developer session log.
- **Review flags:** none — additive docs.

### fb1cb0316a — fix(ci): keep animated figures open on show=False; harness uses returned fig
- **What:** Two real behavior fixes. (1) `plot.py`: the GH#148 `show=False` auto-close of the figure now also skips ANIMATED figures (`line_ani is None` added to the close guard) so the FuncAnimation timer isn't destroyed on GUI backends (Windows/TkAgg). (2) `screenshot_harness.py`: `capture()` now prefers the RETURNED figure(s) via new `_extract_mpl_figs()` (handles bare Figure, `(fig, ani)` tuple, and `return_model` dict), falling back to the pyplot registry.
- **Files:** `hypertools/plot/plot.py`, `scripts/screenshot_harness.py` (+ regenerated gallery zips, sg_execution_times).
- **Why:** Two remaining CI failures (Windows spin-clip crash; Ubuntu screenshot "no figures" because #148 empties the registry).
- **Review flags:** ⚠️ Behavior change in `plot()` show=False path — narrow and well-reasoned (only affects animated figures, which now stay registered in pyplot). Reasonable but worth confirming it doesn't reintroduce the #148 Jupyter double-display for animations specifically. Harness change is test-infra only.

### 80470fd06d — deps: require pydata-wrangler>=0.5.1 (Windows import fix released upstream)
- **What:** Bumps `pydata-wrangler` base + `[text]` pins `0.5.0`→`0.5.1`; updates the configurator comment to reflect the upstream fix. The `os.environ.setdefault("HOME", ...)` guard is retained.
- **Files:** `pyproject.toml`, `hypertools/core/configurator.py`.
- **Why:** dw 0.5.1 fixes the Windows `HOME`-unset import crash (data-wrangler#32).
- **Review flags:** none — mechanical dep bump; guard kept for 0.5.0 environments (zero-risk).

### 92110f41da — docs(notes): round-2 CI fixes
- **What:** 10 lines appended to the session note.
- **Files:** `notes/session_2026-07-04_ci_green_legend.md`.
- **Why:** Session log.
- **Review flags:** none — additive docs.

### e3530a6c88 — chore: renumber release 2.0 -> 1.0
- **What:** Large mechanical rename. `pyproject` version `2.0.0.dev0`→`1.0.0.dev0`; "2.0"→"1.0" prose across docstrings/comments/notebooks/gallery; `v2.0-*`→`v1.0-*` and `*_v2.0`→`*_v1.0` file renames; `dev-2.0`→`dev-1.0` branch refs. Verified the hypertools/** source diffs (align, model, datageometry, io) are all docstring/comment prose — no logic changes. One user-facing string changed: `"align=True was removed in hypertools 2.0"`→`"1.0"` (consistent).
- **Files:** `pyproject.toml`, many docs/examples/notebooks, `hypertools/**` (docstrings only), some `tests/**` (string refs).
- **Why:** Correct over-numbered release (this is 1.0, not 2.0).
- **Review flags:** ⚠️ Large (>50 files) but mechanical; version literal changes are the only functional bits. Spot-checked source — no code logic touched. Low risk.

### 17e1bf5eab — docs(notes): round-3 log — 2.0 -> 1.0 renumber
- **What:** 9 lines appended to session note.
- **Files:** `notes/session_2026-07-04_ci_green_legend.md`.
- **Why:** Session log.
- **Review flags:** none — additive docs.

### ecacef7efc — plan: hyp.predict + hyp.impute implementation plan (GH #169)
- **What:** New 188-line planning doc.
- **Files:** `docs/superpowers/plans/2026-07-05-hypertools-predict-impute.md`.
- **Why:** Implementation plan for the predict/impute feature.
- **Review flags:** none — additive doc.

### 7db07716a0 — feat(predict): Forecaster base + GH#169 t semantics
- **What:** New `hypertools/predict` package. `Forecaster(BaseEstimator)` wrapping fitter/forecaster callables, fit-one-model-per-dataset, list-in/list-out. `resolve_t()` implements GH#169 horizon semantics: int → step-count via min non-zero index diff; datetime → step count to target, negative ("truncate") if target ≤ last observation. Includes 146-line test file.
- **Files:** `hypertools/predict/common.py`, `__init__.py`, `tests/predict/test_common.py`.
- **Why:** Task 1 of predict/impute plan — the base class + horizon logic.
- **Review flags:** ⚠️ Uses the `assert cond, ValueError('msg')` antipattern throughout (`fit`, `_infer_step`) — the `ValueError(...)` becomes the assertion *message*, it is NOT raised as a ValueError, and all these validations are stripped under `python -O`. Recurs across the whole predict/impute batch. Functionally the messages surface as `AssertionError`; a caller catching `ValueError` won't catch them. Test coverage present.

### a61179b18c — plan(predict): return_model contract
- **What:** 3 lines added to the plan doc.
- **Files:** `docs/superpowers/plans/...predict-impute.md`.
- **Why:** Document the return_model=(result, fitted) reuse contract.
- **Review flags:** none — additive doc.

### 6922ff500b — feat(predict): Kalman, GaussianProcess, AutoRegressor forecasters
- **What:** Three forecasters + tests. Kalman (pykalman EM+filter_update, NaN-tolerant via masked_invalid, lazy import guarded for `[predict]`). GaussianProcess (sklearn GPR over time index, base-install only). AutoRegressor (recursive lagged-feature, registry of 7 sklearn regressors, MultiOutputRegressor fallback for multivariate).
- **Files:** `hypertools/predict/{kalman,gp,autoreg}.py`, matching tests.
- **Why:** Task 2 — concrete forecasters.
- **Review flags:** ⚠️ `assert ... , ValueError(...)` antipattern again (autoreg `_resolve_estimator`, `fitter`). Minor: each `__init__` sets attributes both via `super().__init__(...)` and redundantly re-assigns them after (belt-and-suspenders for sklearn `get_params`/`clone`; harmless but noisy). Lazy import + friendly ImportError is well done. Tests present.

### 6e4a18cd90 — feat(predict): ARIMA + skaters Laplace forecasters
- **What:** ARIMA (statsmodels, per-column univariate, order (1,1,1) default, convergence warnings suppressed *narrowly* around `.fit()` only). Laplace (skaters `laplace(k=t)` factory, feeds full series through the online closure; defensive re-chunking loop if a single call returns < horizon). Both lazy-imported for `[predict]`. Tests included.
- **Files:** `hypertools/predict/{arima,laplace}.py`, tests.
- **Why:** Task 3.
- **Review flags:** ⚠️ Laplace re-feeds forecast *means* back as observations for chunking — a documented approximation; author verified skaters 0.11.0 doesn't actually need it (single call handles k≤100). ARIMA warning suppression is correctly scoped (not global). No blocking issues.

### c45450b0a0 — feat(predict): Chronos forecaster + hyp.predict dispatcher with return_model
- **What:** Chronos forecaster (`amazon/chronos-t5-tiny`, lazy `[predict-hf]`/torch import, per-column median-quantile). `predict.py` dispatcher (dw funnel + `unpack_model`, resolves str/dict-both-forms/class/instance). Extends `Forecaster` with `is_fitted` + `predict_new(data, t)` reuse path via per-child `applier`; Kalman/ARIMA/AutoRegressor/GP add appliers, Laplace/Chronos leave `applier=None` (re-derive). Registers `hyp.predict` in `__init__`. Also fixes a real latent shadow bug: `hyp.predict` (function) shadows the `predict` submodule attr, breaking `import hypertools.predict.X` traversal — test imports switched to `from hypertools.predict import X`.
- **Files:** `hypertools/predict/{predict,chronos,common}.py`, appliers in `{arima,autoreg,gp,kalman,laplace}.py`, `hypertools/__init__.py`, dispatcher/chronos tests, 3 test import fixups.
- **Why:** Task 4 — dispatcher + foundation-model forecaster + reuse contract.
- **Review flags:** ⚠️ The function-shadows-submodule situation is real: `hypertools.predict` the attribute now resolves to the function, not the package. The commit fixes the test imports but downstream/user code doing `hypertools.predict.kalman` will hit the function — a genuine API footgun, though intentional (mirrors existing `manip`). PPCA/predict_new reuse for `applier=None` re-derives params (documented). No test regressions claimed.

### 45f65ad310 — feat(impute): PPCA/sklearn/Kalman imputers + hyp.impute dispatcher (GH #169)
- **What:** New `hypertools/impute` package: `Imputer(BaseEstimator)` (stack list → joint fit → transform → split, same-shape semantics), dispatcher, PPCA/SimpleImputer/KNNImputer/IterativeImputer/Kalman imputers. Reroutes `format_data.fill_missing` through `impute(model='PPCA')`. **Verified byte-compat claim:** the impute PPCA wraps `..external.ppca.PPCA` (the real 211-line vendored class; `_externals.ppca` is a re-export shim) with identical defaults (`tol=1e-4, min_obs=10, d=None`) matching `PPCA.fit`, and the same stack→fit→transform→NaN-row→split sequence.
- **Files:** `hypertools/impute/*` (7 files), `hypertools/tools/format_data.py`, `hypertools/__init__.py`, `tests/impute/*`.
- **Why:** Task 5 — pluggable imputation, default preserves legacy PPCA behavior.
- **Review flags:** ⚠️ `assert ... , ValueError()` antipattern again (common.py, ppca.py). PPCA reuse-on-new-data path is an explicit approximation (zero-fill + fitted rotation, no EM) — documented. Byte-compat verified sound. Good test coverage.

### df68b5a581 — feat(plot,analyze): predict= and impute= integration
- **What:** Wires predict/impute into `plot`/`analyze`/`normalize`/`format_data`. `impute=None` threads through to override the PPCA fill (None = byte-identical legacy). `plot(predict=, t=10)`: forecasts per original dataset in post-analyze space, prepends last observed row (length t+1), mirrors the same center + scale transform onto forecasts, overlays dashed alpha-0.6 `_nolegend_` traces color-matched to source; skips if cluster/hue reshaping breaks 1:1 trace correspondence; plotly parity; `animate=True + predict` raises NotImplementedError. return_model bundle gains `predict` + `models.impute`.
- **Files:** `hypertools/plot/plot.py`, `plotly_backend.py`, `tools/{analyze,normalize,format_data}.py`, `tests/plot/test_predict_integration.py`.
- **Why:** Task 6 — user-facing integration.
- **Review flags:** ⚠️ The center/scale mirror onto forecasts *reimplements* `scale()`'s min/max math inline rather than reusing the function — a drift risk if `scale()` changes (the two must stay in sync). Forecast overlay reads colors from `ax.lines`/`kwargs_list` and is added before the per-point-color artist swap — relies on that swap preserving forecast lines; covered by the artist-count test (2×len). The 1:1 mismatch guard silently drops forecasts under cluster/hue regrouping (documented, reasonable). 158 tests pass per commit msg. Well-guarded overall.

### 912ac841ef — deps: [predict] and [predict-hf] extras; predict deps in dev + doc builds
- **What:** Adds `[predict]` (pykalman/statsmodels/skaters) and `[predict-hf]` (chronos-forecasting) extras; adds the `[predict]` trio to `dev` (CI) and `docs/doc_requirements.txt`. Chronos deliberately kept out of CI (importorskip); yfinance intentionally not a dep.
- **Files:** `pyproject.toml`, `docs/doc_requirements.txt`.
- **Why:** Task 7 — package the optional forecasting deps.
- **Review flags:** ⚠️ New heavy/niche deps: `skaters>=0.11` (Microprediction ecosystem, large transitive tree) and `chronos-forecasting>=2.0` (pulls torch). All opt-in extras, never base — low risk. statsmodels/pykalman added to `dev`, so CI install weight grows. Supply-chain note only.

---

batch 05: 17 audited; high-risk: none. Watch items: fb1cb03 (show=False animated-figure behavior change), c45450b (predict function shadows submodule — API footgun), df68b5a (inlined scale-math duplication); pervasive `assert cond, ValueError()` antipattern across predict/impute (stripped under -O, not raised as ValueError).


## Commit batch 6

No Write tool is available and redirect-based file writing is prohibited in this read-only mode, so I'm returning the complete audit as my final message for the parent agent to persist.

---

### 4d264dfc0f — fix(plot): scale forecasts into the plot frame (2D+3D)
- **What:** Forecast (`predict=`) vertices were center/scaled using stats from observed data only, so forecasts extending past the observed range rendered outside the [-1,1] square/cube frame. Now center-mean and scale min/max are computed from the FULL stacked data (observed + forecasts) and both pass through the same transform. No-forecast path unchanged (still `center()`/`scale()`).
- **Files:** `hypertools/plot/plot.py`, `tests/plot/test_predict_integration.py` (new regression `test_forecast_vertices_stay_inside_frame`).
- **Why:** Forecasts visually escaped the frame (axes off, no clipping).
- **Review flags:** ⚠️ Behavior change to coordinate scaling, but well-scoped: only active when forecasts present, else-branch preserves legacy path exactly. Test asserts containment. Low risk.

### 0de6b216da — docs(gallery): plot_predict + plot_impute examples; predict/impute API sections
- **What:** Adds `examples/plot_predict.py`, `examples/plot_impute.py`, and Predict/Impute autosummary sections to `docs/api.rst`.
- **Files:** `docs/api.rst`, `examples/plot_impute.py`, `examples/plot_predict.py`.
- **Why:** Gallery/API coverage for new predict/impute features.
- **Review flags:** none — additive docs/examples.

### 1081ef6c5e — docs(gallery): plot_predict uses helical data so forecasts are visible
- **What:** Rewrites `plot_predict.py` example from random walks to noisy 5D helices; switches model from Kalman to GaussianProcess so dashed forecast tails are visible.
- **Files:** `examples/plot_predict.py`.
- **Why:** Random-walk Kalman forecasts are near-constant stubs.
- **Review flags:** none — additive/cosmetic doc example.

### 0e313a0a2f — docs(tutorials): stock forecasting tutorial (real yfinance data, honest backtest)
- **What:** New 1097-line notebook: downloads real 2y daily closes (AAPL/MSFT/NVDA/JPM) via yfinance, holds out last 30 days, backtests 5 forecasters vs naive baseline with MAE/MAPE, demos `return_model=True`. Registers in `tutorials.rst`.
- **Files:** `docs/tutorials/stock_forecasting.ipynb`, `docs/tutorials.rst`.
- **Why:** Tutorial for hyp.predict.
- **Review flags:** ⚠️ Network: notebook self-installs `yfinance` + `hypertools[predict]` via `%pip install` (guarded by `find_spec`) and pulls live market data at runtime — expected for a tutorial, not run in CI, but reproducibility depends on external service. No secrets/tokens. Not test/library code.

### 2eda5b9c58 — docs(tutorials): Kalman projectile tutorial (real SportVU jump-shot arc)
- **What:** New 855-line notebook: downloads NBA SportVU tracking archive from a GitHub raw URL, extracts via `py7zr`, demos `hyp.impute(model='Kalman')` and `hyp.predict` with RMSE/MAE vs ground truth, 2D/3D overlays.
- **Files:** `docs/tutorials/projectile_kalman.ipynb`, `docs/tutorials.rst`.
- **Why:** Tutorial for Kalman impute/predict on real data.
- **Review flags:** ⚠️ Network: self-installs `py7zr` + `hypertools[predict]`; downloads a public GitHub-hosted archive (no auth) at runtime. Same tutorial-reproducibility caveat as above. No secrets. Not library/test code.

### 4585987e01 — docs(notes): #169 implemented (hyp.predict + hyp.impute) -- move to fixed list
- **What:** Moves issue #169 in `notes/issues-to-close-on-merge.md` from "leave open" to the fixed/close-on-merge list with implementation summary.
- **Files:** `notes/issues-to-close-on-merge.md` (1 line).
- **Why:** Bookkeeping.
- **Review flags:** none — mechanical notes edit.

### bbb96b468c — docs: build gallery with plot_predict + plot_impute (and 1.0-rename re-executions)
- **What:** Regenerated Sphinx-Gallery output: new plot_predict/plot_impute pages (png/rst/ipynb/zip/md5) plus md5/re-execution churn for a few renamed examples.
- **Files:** all under `docs/auto_examples/` + `docs/sg_execution_times.rst` (generated).
- **Why:** Gallery regen.
- **Review flags:** none — generated docs, not scrutinized per policy.

### 04c10e2578 — fix(predict,format_data): GP forecasts extrapolate trends; align df columns by name (GH #132)
- **What:** (1) GP default kernel `RBF+White` → `DotProduct+RBF+White` so forecasts extrapolate trends instead of mean-reverting. (2) `format_data` now aligns multiple named-column DataFrames BY NAME to the first dataset's column order (warns on reorder, raises ValueError on mismatched sets); RangeIndex/duplicate-column frames keep positional behavior.
- **Files:** `hypertools/predict/gp.py`, `hypertools/tools/format_data.py`, `tests/predict/test_gp.py`, `tests/test_format_data.py`.
- **Why:** Wrong-direction forecasts; silent feature misalignment (GH #132).
- **Review flags:** ⚠️ Two real behavior changes. Column-reorder logic is guarded (only >1 named-column df) and tested for reorder/mismatch/positional-passthrough. New default kernel changes all GP forecast outputs (DotProduct linear trend). Both intentional and covered; medium-low risk.

### 18a6d91f5c — docs: regenerate plot_predict gallery (trend-extrapolating forecasts) + evidence
- **What:** Rebuilt plot_predict gallery images with new GP kernel; added before/after evidence PNGs; moved #132 to fixed list in notes.
- **Files:** `docs/auto_examples/*` (generated), `docs/images/v1.0-predict-fix/*.png`, `notes/issues-to-close-on-merge.md`.
- **Why:** Reflect the kernel fix.
- **Review flags:** none — generated docs + evidence images + notes.

### 8d424e0856 — fix(plot): colors= kwarg was a no-op unless color= also passed (GH #142)
- **What:** Hoists `colors` handling out of the `if color is not None:` guard so `colors=[...]` alone now sets `mpl_kwargs['color']`. `colors` still wins over `color`; conflict warning only fires when both supplied.
- **Files:** `hypertools/plot/plot.py`, `tests/test_plot_colors142.py` (new: mpl static/animated, plotly, conflict), notes, evidence PNGs.
- **Why:** `colors=` silently ignored (GH #142).
- **Review flags:** none — bugfix, additive tests. Behavior change is the intended fix; well-tested.

### 26dec4f459 — fix(plot): linestyles=/markers= aliases were no-ops without singular kwarg; fix color(s) docstring
- **What:** Same fix pattern applied to `linestyles=` and `markers=` (hoisted out of singular-kwarg guards); fixes `color(s)` docstring ("A list of marker types" → "A list of colors").
- **Files:** `hypertools/plot/plot.py`, `tests/test_plot_style_aliases142.py` (new, 142 lines).
- **Why:** Plural style aliases silently ignored.
- **Review flags:** none — consistent bugfix with tests.

### f150750570 — feat(io): complete hyp.load -- Drive large files, Excel, Sheets CSV, remote pickle policy (GH #177)
- **What:** `_fetch_bytes` parses/follows Google Drive large-file virus-scan interstitial (regex-extracts confirm form action+hidden inputs → re-GET); replaces old cookie handler in `load.py`. Adds `.xlsx/.xls` (openpyxl/xlrd). Rewrites Sheets URLs to CSV export. New `trust=False` kwarg: remote unpickle warns; remote `.npy/.npz` use `allow_pickle=False` unless trust. New `[io]` extra + `bigdata` pytest marker (deselected by default).
- **Files:** `hypertools/io/load.py`, `hypertools/io/sources.py`, `pyproject.toml`, `tests/test_load_sources.py`, `tests/data/drive_large_file_interstitial.html`.
- **Why:** Complete remote-load support + a security policy for remote code execution via pickle (GH #177).
- **Review flags:** ⚠️ Security-relevant. Good: `.npy/.npz` pickle blocked unless trust; local files exempt. But note: remote `.pkl/.geo/.p` unpickle still EXECUTES with only a UserWarning (trust=False does not block raw pickles, only warns) — documented as a design choice but a genuine RCE surface. Also, this commit's per-branch `except ValueError: raise` is over-broad (any parse ValueError escapes the "tried in order" digest) — corrected in the very next commit. HTML interstitial parsing is regex-based on Google's markup (fragile to upstream changes). `bigdata` network tests deselected by default.

### 9aff4c5eef — fix(io): scope trust-policy re-raise to HypertoolsTrustError, not all ValueErrors
- **What:** Introduces `HypertoolsTrustError(ValueError)`; `_npy_load`/`_unpack_npz` wrap np.load to convert the allow_pickle ValueError into it; `load_source` branches now `except HypertoolsTrustError: raise` so unrelated ValueErrors (pandas ParserError etc.) fall through to the digest.
- **Files:** `hypertools/io/sources.py`, `pyproject.toml` (comment), `tests/test_load_sources.py`.
- **Why:** Fixes the over-broad catch from commit 12 — malformed remote payloads should join the digest, not escape raw.
- **Review flags:** none — corrective bugfix with new regression test; tightens prior commit. Good.

### 8adedc99c6 — feat(plot): colorbar support for continuous hue and discrete groups (GH #100)
- **What:** New `colorbar=True/dict` for both backends. Continuous hue → ScalarMappable colorbar; discrete groups → BoundaryNorm-segmented with group-name ticks. Eager validation of dict keys/location. mpl widens figure so legend+colorbar don't overlap; plotly uses hidden marker trace. Also fixes pre-existing bug: `legend=True` with categorical string hue showed integer group ids instead of category names (new `hue_category_names`). Adds public `get_palette_colors`/`continuous_colormap` in colors.py.
- **Files:** `hypertools/plot/plot.py` (+310), `hypertools/plot/colors.py`, `hypertools/plot/plotly_backend.py`, `tests/test_colorbar.py` (283 lines), notes, evidence PNGs.
- **Why:** GH #100 colorbar feature.
- **Review flags:** ⚠️ Large additive feature but includes a real behavior change: categorical-hue `legend=True` now shows category names, not ints (a fix, but changes existing legend output). Raises ValueError when colorbar requested with no mapping. Broad test coverage. Medium-low risk.

### be1e44d1f3 — feat(plot): mesh utilities for smooth convex-hull surfaces (GH #109)
- **What:** New leaf geometry module `meshutil.py` (470 lines): convex-hull smoothing 2D/3D via scipy ConvexHull, subdivision/Taubin smoothing. Pure geometry, imports only numpy/scipy; no dependency on plot layer.
- **Files:** `hypertools/plot/meshutil.py`, `tests/test_meshutil.py` (238 lines).
- **Why:** Geometry primitives for surface feature.
- **Review flags:** none — additive, self-contained, tested, no network/deps beyond scipy.

### 4ef198c176 — feat(plot): smooth convex-hull surfaces with lighting, dict API (GH #109)
- **What:** New `surface=` kwarg: per-dataset smooth lit convex-hull overlay (2D Catmull-Rom fill, 3D shaded hull), both backends, static+animated. Dict API (alpha/color/lighting/smoothing/pre_inflate/keep_points) broadcasts per-dataset, validated eagerly with ValueError on unknown keys. New `surface.py` built on meshutil.
- **Files:** `hypertools/plot/matplotlib_backend.py`, `hypertools/plot/plotly_backend.py`, `hypertools/plot/plot.py`, `hypertools/plot/surface.py`, `tests/test_surface.py` (206 lines), notes, evidence PNGs.
- **Why:** GH #109 surface rendering.
- **Review flags:** ⚠️ Large additive feature touching both backends. New kwarg defaults off, so existing plots unaffected. Eager validation + broad tests. Low risk (opt-in).

---

batch 06: 16 audited; high-risk: none. Watch: f150750570 (remote .pkl unpickle warns-not-blocks = RCE surface; over-broad ValueError catch fixed in 9aff4c5eef); behavior changes in 4d264dfc0f, 04c10e2578, 8adedc99c6 all tested/intentional.


## Commit batch 7

All 16 commits reviewed. Here is the audit (content for `batch_07.md`). Note: I operate read-only and have no write tool, so I'm returning the report inline for the parent to persist.

### 30dac24100 — fix(plotly): trail frame updates tracked by explicit trace indices
- **What:** `plotly_draw` now records `trail_trace_start` (actual `fig.data` index where trail traces begin) and `_add_animation` addresses trail traces by `range(trail_trace_start, +n_trail_traces)` instead of assuming contiguity after data traces.
- **Files:** hypertools/plot/plotly_backend.py; tests/test_plotly_trace_indices.py (new, 127 lines).
- **Why:** With `predict=` present, forecast traces are appended between data and trail traces, so the old contiguous range overwrote forecast traces and left real trails empty.
- **Review flags:** none — targeted correctness fix, new dedicated test.

### 7766a5e041 — docs(examples): surface-morph demo (GH #109)
- **What:** New standalone example + evidence PNG.
- **Files:** examples/animate_surface_morph.py (new, 167 lines); docs image PNG.
- **Why:** Showcase hull surfaces via a hand-rolled morph animation.
- **Review flags:** none — additive doc/example.

### f8e8bfee0d — feat(plot): KDE density shading, off by default (GH #108, #191)
- **What:** New `density=True/dict` kwarg; new density.py (validate/broadcast mirroring surface.py); 2D imshow/heatmap, 3D marching-cubes iso-surfaces (mpl) / go.Volume (plotly); animated plots compute density once as static bg. Fail-fast on 1D. New `[density3d]` extra (scikit-image).
- **Files:** hypertools/plot/density.py (new), plot.py, matplotlib_backend.py, plotly_backend.py, pyproject.toml, docs/doc_requirements.txt, tests/test_density.py (369 lines).
- **Why:** Requested density visualization feature.
- **Review flags:** ⚠️ deps — adds optional `scikit-image>=0.22.0` (opt-in extra + dev + docs); graceful scatter-fog fallback + UserWarning when absent, exercised via subprocess import-blocker test. New public API surface but off by default; additive.

### c805b268dd — fix(density): wire levels knob, validate option values (GH #108)
- **What:** Adds `_validate_density_values` (alpha in (0,1], grid int≥8, levels int[1,10], per_group bool) and `resolve_iso_fracs_alphas`; `levels` now wired into both 3D backends (mpl Poly3DCollection per level, plotly surface_count=5*levels). Refactors duplicated color-resolution into `_resolve_dataset_colors` shared by surface/density.
- **Files:** density.py, matplotlib_backend.py, plot.py, plotly_backend.py, tests/test_density.py (+70).
- **Why:** `levels` was previously documented but ignored for plotly; missing value validation.
- **Review flags:** none — `levels=3` reproduces original hand-tuned constants exactly (verified in code); validation is additive; refactor is behavior-preserving.

### 15e975311f — feat(animate): per-dataset chemtrails/precog/bullettime (GH #127)
- **What:** Trail flags now accept bool or list/tuple of bool (one per final dataset) via new trails.py `broadcast_trail_flag`; backends build trail artists/traces only for flagged datasets. `animate` mode stays global.
- **Files:** hypertools/plot/trails.py (new), plot.py, matplotlib_backend.py, plotly_backend.py, tests/test_animation_styles.py.
- **Why:** Previously a list value was silently truthy-broadcast to all datasets.
- **Review flags:** ⚠️ behavior — a list `chemtrails=[...]` previously broadcast-as-truthy now honored per-dataset (intended fix); length mismatch now raises ValueError. Well tested.

### 85f263de6a — feat(plot): MultiIndex DataFrames (GH #95)
- **What:** Row-MultiIndex DataFrame (nlevels≥2) expanded to leaf datasets before pipeline; post-transform per-level mean trajectories appended with color/linewidth/alpha/linestyle/label overrides. cluster/n_clusters raise; hue superseded w/ warning; color/colors/linewidth ignored w/ warning. plotly showlegend now honors `_nolegend_`. `cluster` branch changed to `elif`.
- **Files:** hypertools/plot/multiindex.py (new, 217 lines), plot.py (+129), plotly_backend.py (+1), tests/test_multiindex.py (422 lines).
- **Why:** Structured MultiIndex plotting feature.
- **Review flags:** ⚠️ behavior/logic — substantial new branch in plot.py that overrides user color/linewidth/legend for MultiIndex input; converts `if cluster` to `elif` (MultiIndex now precedes cluster/hue chain). Unequal-length groups averaged over shortest prefix (documented, warned). Extensive tests; risk contained to MultiIndex-input path.

### 188b63a544 — fix(multiindex): warn on list-bypass, error on predict= combo, dedupe warnings (GH #95)
- **What:** Warns when a list contains MultiIndex DataFrame(s) (expansion only for bare single DF); raises ValueError on `predict=` + MultiIndex; aggregates unequal-length warnings into one per call.
- **Files:** multiindex.py, plot.py, tests/test_multiindex.py (+69).
- **Why:** Close edge-case gaps in the new MultiIndex feature.
- **Review flags:** ⚠️ behavior — `predict=` + MultiIndex now errors (was undefined/broken). Guardrail, tested.

### db9edfe8a8 — fix(surface): plotly opaque blend + occlusion fixes; tiny-cloud rescale (GH #109)
- **What:** Bakes alpha into mesh color (opacity=1.0) to dodge plotly translucent depth-sort bug; NaN-hides markers a surface encloses; trims mesh faces inside another dataset's mesh; new `points_enclosed`; `_rescale_for_containment` grow-only rescale so smoothed hull recovers ≥96% point containment, warns if not.
- **Files:** meshutil.py (+99), plotly_backend.py (+104), tests/test_meshutil.py, tests/test_surface.py.
- **Why:** Verified upstream plotly.js WebGL rendering defects + smooth_hull_3d under-inflation for sparse clouds.
- **Review flags:** ⚠️ logic — nearest-by-direction proxy in rescale is heuristic (author acknowledges); guarded by containment check + warning. Rendering-only; tested.

### 925e0b4012 — fix(surface): contain meshes within axes cube; clean plotly seams (GH #109)
- **What:** New `surface_cube_scale()` sizes axes cube to actual mesh extent (not hard-coded 1); both backends widen limits/scene ranges; plotly Mesh3d emits both winding orders w/ flatshading; mesh-mesh trim made priority-based (cut only against lower-indexed datasets). Example updated to use helper.
- **Files:** surface.py, matplotlib_backend.py, plotly_backend.py, examples/animate_surface_morph.py, tests/test_surface.py.
- **Why:** Meshes bulge past cube after pre_inflate/smoothing/rescale; plotly per-triangle back-face culling punches holes.
- **Review flags:** none material — layout/rendering fixes; author notes deep-overlap seam reduced not eliminated. Tested (containment + priority-trim regressions).

### fd064aa52b — fix(surface): plotly surfaces use precomputed vertex shading
- **What:** Replaces flatshading-per-face (which darkened reversed doubled faces) with per-vertex Blinn-Phong (`blinn_phong_vertex_colors`/`vertex_normals`), passed as `vertexcolor` with plotly lighting forced to identity; recomputed per animation frame. `roughness`/`fresnel` now silently ignored.
- **Files:** meshutil.py (+154), plot.py (docstring), plotly_backend.py (+187), surface.py, tests.
- **Why:** Prior double-winding fix produced dark jagged patches.
- **Review flags:** ⚠️ behavior — `lighting` dict `roughness`/`fresnel` keys now no-ops (documented); per-frame recompute adds animation cost. Rendering-only; tested.

### fcd8f96749 — fix(colorbar,multiindex): top-level colorbar segments; legend fitting all layouts (GH #100, #95)
- **What:** Filters `_nolegend_` traces out of colorbar segments (raises if none labeled); reorders layout so colorbar added before legend fit; replaces rasterized-pixel measurement with `get_tightbbox`-based `_tight_right_edge_in` (one-shot true-extent); legend fitting now runs for animated plots too; max_iter 6→3.
- **Files:** plot.py only (+192/−85); tests/test_colorbar.py (+87), tests/test_multiindex.py (+33).
- **Why:** MultiIndex leaked leaf traces onto colorbar; legend clipped in 3D/animated/colorbar layouts; rasterize approach under-reported overflow.
- **Review flags:** ⚠️ logic — meaningful rewrite of figure-sizing heuristics (measurement method + iteration count changed); could shift exact saved figure dimensions. Layout-only, no data impact; covered by new colorbar tests.

### 4fd9e65791 — fix(density): auto-boost 3D shell/volume opacity for small-in-scene clusters (GH #108)
- **What:** New `bbox_extent`, `density_alpha_boost` (clamp((scene/dataset)**2,1,6)), `resolve_plotly_volume_params` (widen KDE pad/isomin/opacityscale as boost ramps); wired into mpl iso-surface/fog alphas and plotly Volume. No-op (boost≈1) for scene-filling dataset.
- **Files:** density.py (+121), matplotlib_backend.py, plotly_backend.py, tests/test_density.py (+188), evidence PNGs.
- **Why:** Jointly-scaled separated clusters rendered density invisibly.
- **Review flags:** none material — additive multiplier, verified no-op for single-dataset case; empirically-tuned constants documented; tested.

### a5be2400e0 — fix(animate): warn + skip trail styles under spin/serial (GH #127)
- **What:** `_wants_trail` forced False for spin/serial in mpl; plot.py emits one UserWarning naming mode/flags/dataset indices. plotly already skipped (only builds trails for True/parallel).
- **Files:** matplotlib_backend.py (+13), plot.py (+43), tests/test_animation_styles.py.
- **Why:** Trails silently no-op'd (frozen invisible stubs) under spin/serial.
- **Review flags:** none — purely informational warning + skip; tested.

### 3d320c4254 — docs(examples): gallery examples + readme bullets
- **What:** Five new example scripts (surface/density/colorbar/multiindex/trails-mix) + readme feature bullets.
- **Files:** examples/*.py (new), readme.md (+23).
- **Why:** Gallery coverage for new features.
- **Review flags:** none — additive docs.

### 61bf92a24e — docs: regenerate gallery; fix morph figure registration for sphinx-gallery
- **What:** Generated auto_examples output (codeobj/ipynb/rst/zip/gif/mp4/png). Real source change: two morph examples now pass their own `ax=` so the figure stays pyplot-registered for sphinx-gallery capture.
- **Files:** docs/auto_examples/** (generated, skimmed); example source tweaks.
- **Why:** `show=False` (GH #148) deregistered figures, breaking animation capture; stale md5 masked it.
- **Review flags:** none — generated docs + small doc-source fix; not shipped code.

### 3861ca6e61 — docs: impute/predict api stubs + seven-features plan + session notes
- **What:** Two `.rst` autofunction stubs; a planning markdown + session notes.
- **Files:** docs/hypertools.impute.rst, docs/hypertools.predict.rst, .claude/.../plans/2026-07-05-seven-features.md.
- **Why:** Match module-stub doc convention; track plan.
- **Review flags:** none — additive docs (stubs carry a UTF-8 BOM + no trailing newline, matching existing convention; cosmetic only).

---

batch 07: 16 audited; high-risk: none. Heaviest logic (scrutinized, well-tested): 85f263de (MultiIndex plot.py branch), fcd8f967 (colorbar/legend layout rewrite), db9edfe8/925e0b40 (surface rescale/containment heuristics). One dep added: optional scikit-image ([density3d] extra, graceful fallback).


## Commit batch 8

All commits reviewed. I'm in read-only mode with no Write tool, so I'm returning the audit content directly (parent agent reads this output). Here is `batch_08.md`:

---

### eaf7fcaa20 — notes: move #108/#191 to fixed list, drop duplicate #109 leave-open row
- **What:** Removes 3 lines from `notes/issues-to-close-on-merge.md` bookkeeping.
- **Files:** `notes/issues-to-close-on-merge.md`.
- **Why:** Track which GH issues close on merge; dedupe #109.
- **Review flags:** none — notes-only/mechanical.

### c0eff29a73 — fix(surface): tight hulls — no blanket inflation, bounded containment grow
- **What:** Replaces flawed nearest-by-angle vertex proxy in `_rescale_for_containment` with exact ray-vs-convex-hull exit distance (halfspace/`ConvexHull.equations`). `pre_inflate` default 1.15→1.0; new hard `_RESCALE_CAP=3.0` (was unbounded `ratios.max()`); containment target 0.96→0.99; grow-only, top-k (99th pct) scale so outliers can't force huge grow.
- **Files:** `hypertools/plot/meshutil.py`, `surface.py`, `plot.py`, 2 regenerated PNGs, tests.
- **Why:** Maintainer reported surface-morph "explosion"; math was genuinely wrong (proxy under/over-estimated reach). New approach matches `points_enclosed`'s Delaunay test exactly. Sound.
- **Review flags:** ⚠️ Behavior change: default surface fit tighter (pre_inflate 1.0), containment threshold raised, warning wording changed. ⚠️ Doc bug: `plot.py` docstring says post-hoc grow is bounded to "at most 10%" while actual cap is 3.0 (300%) — contradicts the code and the commit's own rationale (later example docs were corrected in d71da01, but this plot.py line was the introduced inconsistency). Logic itself is correct and well-tested.

### d39a4ef146 — fix(tests): version-robust Poly3DCollection vertex extraction
- **What:** Test helper tries `_vec`/`_faces`(+`_invalid_vertices` mask)/`_segments3d` to read verts across matplotlib 3.10/3.11+.
- **Files:** `tests/test_surface.py`.
- **Why:** 9 CI jobs failed AttributeError on mpl 3.11 private-attr rename.
- **Review flags:** ⚠️ Relies on matplotlib private attributes (fragile by nature, but test-only and defensively multi-path). Otherwise none — test-only.

### c8af7b589b — feat(meshutil): hull-hugging smoothing — pull smoothed verts back to hull
- **What:** New `_ray_exit_distance` helper (factors out M1 halfspace math, shared) and `_pull_back_to_hull`: interior verts pulled toward original hull along centroid ray after each Taubin pass; new `hull_blend=0.85` param, ramped across rounds + light touch-up + 2nd half-strength pull-back. Tighter test bounds.
- **Files:** `hypertools/plot/meshutil.py`, tests.
- **Why:** M1's uniform regrow ballooned already-tight faces; pull-back hugs data by construction. Grow-only rescale kept as safety net.
- **Review flags:** ⚠️ Behavior change to default surface geometry (new pull-back path + new param). Additive/backward-compatible signature. Adds a per-round `ConvexHull(points)` call (up-front, once). Well-reasoned; tests honestly document unmet adversarial-cube targets.

### b52b5b8bf6 — feat(animate): morph animation style — Hungarian point-cloud morphs
- **What:** New `hypertools/plot/morph.py` (243 lines: sample+Hungarian match via `scipy.optimize.linear_sum_assignment`, smoothstep ease, hold/morph schedule, per-segment azimuths). `animate='morph'` in both backends + `plot.py` dispatch; per-dataset list tagging; per-segment `rotations` list; `surface=True` recomputes hull per frame. Trail styles warned+ignored.
- **Files:** `morph.py`(new), `matplotlib_backend.py`, `plotly_backend.py`, `plot.py`, `tests/test_morph_animation.py`(455 new), 2 PNGs, plan doc.
- **Why:** Maintainer request (Task M2). Lifts hand-rolled example into reusable module.
- **Review flags:** ⚠️ Large new feature (~1260 lines) touching both backends + core dispatch — highest surface-area commit in batch. Input validation present (ValueError on <2 clouds/tags, bad list entries, length mismatch). Hungarian is O(n³) but sampling capped at 1000. Deterministic seed=0. Well-tested; no obvious logic errors.

### e1df0103f9 — docs(examples): shape morphs use first-class animate='morph'
- **What:** Rewrites `examples/plot_shape_morph.py` & `animate_surface_morph.py` to delegate to library-native morph instead of hand-rolled FuncAnimation/matching; README bullet; regenerated evidence PNGs.
- **Files:** 2 example scripts (net -117 lines), `readme.md`, 2 PNGs.
- **Why:** Dogfood the new API; remove duplicated logic.
- **Review flags:** none — examples/docs, net code reduction.

### b6074f749b — fix(morph): size axes cube from union-hull mesh
- **What:** Sizes surface axes cube once up-front from meshes built with the exact Hungarian-sampled arrays + a union mesh, instead of from full original-order clouds (smooth_hull_3d isn't row-order-invariant for coplanar cubes; mid-morph points can exceed either endpoint hull).
- **Files:** `matplotlib_backend.py`, `plotly_backend.py`, tests(+110), PNG.
- **Why:** Surfaces escaped the axes box mid-morph. Correct root-cause reasoning.
- **Review flags:** ⚠️ Bugfix to sizing logic in both backends; adds up-front mesh builds. Note: this commit's approach (build full-cloud mesh for sizing) is the OOM risk that d71da01 immediately walks back — so treat as superseded intermediate. Tested.

### d71da01138 — fix(morph): size surface box from sampled/union meshes only; correct docs; earlier validation
- **What:** Self-review fix of the series: (1) drops redundant full-cloud `ConvexHull` per morph dataset (documented OOM/hang risk) — sizes from sampled+union only; (2) corrects false "10% overshoot" docstring in example to real `_RESCALE_CAP=3.0` behavior; (3) moves rotations-list + <2-dataset validation earlier, documents why pipeline-dependent checks stay late.
- **Files:** `matplotlib_backend.py`, `plot.py`, `plotly_backend.py`, `examples/animate_surface_morph.py`, tests(+63).
- **Why:** Code review of c0eff29..b6074f74 found real perf/doc/validation issues.
- **Review flags:** ⚠️ Perf-sensitive fix (removes potential OOM). Fixes the overshoot doc lie in the example — but the analogous `plot.py` "at most 10%" line from c0eff29 appears NOT addressed here (see c0eff29 flag). Good defensive work otherwise.

### f3e044bf9c — fix(morph): untagged datasets render fully in mpl mixed-tag morph
- **What:** In mpl `animate=['morph',None,'morph']`, untagged backdrop lines were left initialized at `dat[0:1]` (1 point) since `update_morph` never touches them; now drawn with full data once before animation. Mirrors plotly.
- **Files:** `matplotlib_backend.py` (+18), tests (+43).
- **Why:** Backdrop datasets silently rendered as a single point, contradicting docstring. Genuine bug.
- **Review flags:** none beyond the fixed bug — additive, backend-parity, tested.

### 987fb5e7ea — docs: regenerate gallery — morph demos as videos; hull-tightness evidence
- **What:** Regenerated sphinx-gallery auto_examples (rst/ipynb/py/codeobj/zip/gif/mp4) for the two morph demos + a new evidence PNG.
- **Files:** `docs/auto_examples/*` (generated), 1 source PNG.
- **Why:** Reflect new morph API/videos in built gallery.
- **Review flags:** none — generated docs (not scrutinized per policy); no hypertools/tests changes.

### 488b929853 — notes: round-3 session log
- **What:** Appends 6 lines to session notes.
- **Files:** `notes/2026-07-05-seven-features-session.md`.
- **Why:** Session bookkeeping.
- **Review flags:** none — notes-only.

### e7becfe614 — fix(tests): widen meshutil runtime guard for slow CI (0.5s→3s)
- **What:** Loosens a perf-assertion upper bound; measured 0.5006s on windows-3.11 tripped the old 0.5s guard.
- **Files:** `tests/test_meshutil.py` (+4/-2).
- **Why:** Flaky timing on slow CI runners.
- **Review flags:** ⚠️ Weakens a runtime-perf test guard 6x — legitimate for CI variance, but a genuinely slow regression would now pass. Test-only.

### 108c08bb0b — fix(morph): constant rotation speed — segment duration ∝ rotation count
- **What:** List-`rotations` frame counts now allocated proportional to per-segment rotation count via largest-remainder (Hamilton) method, `ZERO_ROTATION_FLOOR=0.1`, min 2 frames/segment — constant deg/frame. New `morph.morph_schedule()` computes counts/azimuths once; both backends call it (no drift). New evidence script.
- **Files:** `morph.py`(+134), `matplotlib_backend.py`, `plotly_backend.py`, `plot.py`, `scripts/generate_morph_schedule_evidence.py`(new), tests(+145), 2 PNGs.
- **Why:** Even split made high-rotation segments spin faster. Scalar rotations unchanged.
- **Review flags:** ⚠️ Timing/allocation behavior change for list-rotations morphs. Backends refactored onto shared `morph_schedule` (reduces divergence risk). Well-tested; algorithm sound.

### ec4d5fa361 — fix(plotly): marker sizes calibrated to matplotlib; more transparent volumetric
- **What:** Fixes 3 compounding marker-size bugs: `PT_TO_PX` 96/72→100/72 (real dpi=100); `_DOT_MARKER_SCALE=0.5` for `.`/`,`; `_SCATTER3D_SIZE_FACTOR=1.776` (empirical Scatter3d vs Scatter diameter); morph default markersize 6.0→1.5pt. Volume shading: `MAX_VOLUME_OPACITY` 0.95→0.75, base opacity `min(2a,0.4)`, opacityscale retuned. Empirical parity scripts+tests.
- **Files:** `plotly_backend.py`(+126), `density.py`, 4 scripts, tests(+226), 6 PNGs.
- **Why:** Plotly dots ~8x fatter than mpl; volume glow too heavy. Maintainer R2.
- **Review flags:** ⚠️ Visual behavior change to all plotly 3D markers + volume rendering. `1.776` empirical magic constant (documented, R²~1.0, verified via real renders, no mocks). Parity within 20% — good but a tolerance, not exact. Well-evidenced.

### 0eac4183c9 — fix(surface): remove orphaned roughness/fresnel lighting keys, add lightdir
- **What:** Unifies `_MPL`/`_PLOTLY_LIGHTING_KEYS` into one `_LIGHTING_KEYS`; deletes dead `plotly_lighting_kwargs`/`_PLOTLY_LIGHTING_DEFAULTS`. `roughness`/`fresnel` (accepted but no-op, no callers) now raise ValueError. New validated `lightdir` (finite non-zero 3-vector). Rewrites lighting docstring. 12 new tests.
- **Files:** `surface.py`, `plot.py`, `scripts/generate_surface_controls_evidence.py`(new), tests(+163), 2 PNGs.
- **Why:** Maintainer R3: confirm lighting is parameter-controllable. Gap analysis found roughness/fresnel silently ignored.
- **Review flags:** ⚠️ Breaking API change: `surface['lighting']['roughness'/'fresnel']` previously silently accepted, now raises ValueError. Low real impact (they never did anything), but any existing user code passing them will now error. Kept `_MPL_LIGHTING_KEYS` back-compat alias. Otherwise clean, well-tested.

### 188d301c60 — docs: regenerate gallery — constant-speed morph videos + recalibrated plotly markers
- **What:** Regenerated sphinx auto_examples (rst/ipynb/py/zip/gif/mp4/png/html, execution times) reflecting 108c08b + ec4d5fa.
- **Files:** `docs/auto_examples/*` (generated).
- **Why:** Keep built gallery in sync.
- **Review flags:** none — generated docs; no hypertools/tests changes.

---

batch 08: 16 audited; high-risk: none. Watch items: c0eff29 (plot.py "at most 10%" doc contradicts 3.0 cap), 0eac418 (roughness/fresnel now raise — minor breaking API), b52b5b8 (large new dual-backend feature). All well-tested, sound logic.


## Commit batch 9

### 4f686898a9 — docs: refresh spin.gif build artifact from gallery re-execution
- **What:** Replaces the committed `examples/spin.gif` binary (13.2MB -> 6.8MB) with a re-rendered artifact. No code.
- **Files:** `examples/spin.gif`.
- **Why:** Keep a checked-in gallery artifact in sync with current rendering.
- **Review flags:** none — regenerated binary artifact only.

### b8c4bbe2bc — test(morph): add bbox-margin regression guard for all animate styles (D1)
- **What:** Adds a real-render margin regression test (`test_animation_margins.py`, 193 lines) + 2 evidence PNGs. No source change — the investigation concluded no clipping regression existed (the two apparent "regressions" were a legend box and a plotly Play button confound).
- **Files:** `tests/test_animation_margins.py`, `docs/images/v1.0-seven-features/bbox_margin_{before,after}.png`.
- **Why:** Lock in scale-invariant zoom/margin behavior across every animate style.
- **Review flags:** none — additive test + evidence. Note: `before`/`after` PNGs are identical size (28560 bytes each) which fits the "no fix, guard only" narrative.

### b461feed77 — fix(animate): render correctly at any save dpi
- **What:** Adds `_make_save_dpi_safe`, which monkeypatches the returned `line_ani.save` to null out `fig.canvas.manager` before `real_save` and restore it after, pre-empting a matplotlib `MovieWriter._adjust_frame_size` codec-detection quirk that resized a live OS window to odd pixel dims (corrupting low-dpi gif thumbnails).
- **Files:** `hypertools/plot/matplotlib_backend.py` (+70), `tests/test_animation_margins.py` (+150).
- **Why:** sphinx-gallery re-saves anims at low dpi; the manager resize sheared/cropped thumbnails.
- **Review flags:** ⚠️ Wraps/overrides an animation's `.save` method (monkeypatch on instance) and temporarily mutates `canvas.manager`. Restore is in a `finally`, so exceptions are safe. Behavior is narrow and well-reasoned, but it does silently replace a public method on the returned object — worth awareness. Root-cause analysis is thorough and matches matplotlib internals.

### a3df5cdc14 — feat(morph): full-sample morphs — duplicate to largest dataset, hide duplicates at holds
- **What:** Real semantic change to morph animations. `sample_and_match_clouds` now pads each cloud UP to the largest dataset's size by random (seeded) duplication instead of shrinking to the smallest; returns new `dup_masks`. New `morph_visible_mask` hides a dataset's duplicate rows on its own hold frames (even seg_idx) so alpha compositing matches a plain plot, while morph frames show all points. `morph_samples` becomes an optional pre-cap (default None = uncapped). Adds `MORPH_SURFACE_SIZING_MARGIN=1.2` applied to surface morph cube_scale. Both backends + both gallery examples updated; +289 lines of tests.
- **Files:** `hypertools/plot/morph.py` (+165), `plot.py`, `matplotlib_backend.py`, `plotly_backend.py`, `tests/test_morph_animation.py`, 2 example .py.
- **Why:** Maintainer request — stop discarding real data points in morphs.
- **Review flags:** ⚠️ Behavior change to a default (`morph_samples=None` now means UNCAPPED, previously capped at min(smallest,1000)). Docstring itself warns the uncapped default can be slow / memory-heavy for large datasets due to O(n^3) Hungarian matching — a genuine performance regression risk for large-cloud morphs that previously auto-capped at 1000. `MORPH_SURFACE_SIZING_MARGIN=1.2` is an empirically-tuned magic constant (~9% observed worst case, 20% buffer). Logic (dup masks permuted alongside sampled, hull-invariance to duplicates) is sound and well-tested.

### 9aa5c85a7e — docs: regenerate all animated gallery examples
- **What:** Regenerated auto_examples media (gifs/mp4s/html/zip/rst) + regenerated example .py sources reflecting the dpi + full-sample-morph changes. Bulk generated-doc regen.
- **Files:** `docs/auto_examples/**` (media + rst + ipynb + py). All generated/build artifacts.
- **Why:** Sync gallery to code changes in prior commits.
- **Review flags:** none — generated-doc regeneration. The embedded `docs/auto_examples/*.py` mirror the real `examples/` sources (skimmed, consistent with a3df5cd).

### 3d5d79a905 — fix(tests): CI-robust manager-restore ordering + explicit Agg canvases (mpl 3.11)
- **What:** Small test hardening — reorders manager restore and forces explicit Agg canvases for mpl 3.11 CI robustness.
- **Files:** `tests/test_animation_margins.py` (+4/-1), `tests/test_morph_animation.py` (+8).
- **Why:** Avoid CI flakiness with the manager-nulling trick under newer matplotlib.
- **Review flags:** none — test-only robustness change.

### d8e37c03bd — test(animate): add chemtrails/PCHIP-overshoot regression guard (no source fix needed)
- **What:** Adds `TestChemtrailsOvershootMargins` (+106 lines) proving PCHIP interpolation can't overshoot data range and plot.py rescales to [-1,1] post-interp. No source change.
- **Files:** `tests/test_animation_margins.py`.
- **Why:** Disprove an overshoot hypothesis and lock in the invariant.
- **Review flags:** none — additive test. References a gitignored task report (`.superpowers/sdd/task-D3-report.md`) — not in repo, informational only.

### 06dfbffad9 — fix(animate): 3D scene artists unclipped in all animation paths (axes-box slicing)
- **What:** Real fix: adds `set_clip_on(False)` to every animated 3-D artist (cube wireframe, data/trail lines, morph artist, density & surface Poly3DCollections) so Axes3D's shrunk-square viewport doesn't slice wide projections. Static path deliberately left clipped (documented: unclipping there perturbed tight_layout bbox). Adds `TestAxesBoxNoClipping`.
- **Files:** `hypertools/plot/matplotlib_backend.py`, `tests/test_animation_margins.py` (+? ).
- **Why:** Maintainer-reported chemtrails cube/trails cut off at ~3s frame.
- **Review flags:** ⚠️ Behavior change to rendering (disables clipping). Density/surface colls are shared with the static path, so unclipping affects static too — author notes this is intentional/protective and static path never stretches so no visible change. Reasoning is careful and empirically verified; low risk but touches shared draw code.

### 1487435b0d — docs: regenerate animated gallery media (axes-box slicing fix)
- **What:** Regenerated gallery gifs/mp4s/rst after the unclip fix. Note media dropped from stale 2x (1280x960) to correct 640x480 (Retina side effect, documented as orthogonal).
- **Files:** `docs/auto_examples/**` (media + rst).
- **Why:** Reflect the clip fix in committed media.
- **Review flags:** none — generated-doc regeneration.

### 91d3fabcf8 — docs: evidence images rebuilt from CURRENT media at worst-angle frames
- **What:** Rebuilds two evidence PNGs from current media at measured worst-margin angled frames (previous ones used intermediate/near-head-on frames).
- **Files:** `docs/images/v1.0-seven-features/{chemtrails_clip_after,gif_thumb_after}.png`.
- **Why:** Make evidence images accurate to the shipped fix.
- **Review flags:** none — evidence image regeneration.

### 224b56f0c7 — feat(plot): multibyte character support — auto font detection + font= kwarg (matplotlib)
- **What:** New `hypertools/plot/fonts.py` (`find_covering_font`/`resolve_font`, cmap coverage scan via FT2Font, preferred-family ordering, caches). Threads `font=` kwarg through `plot()` -> `_draw` and colorbar; applies FontProperties to labels/legend/title/colorbar. ASCII-only path preserved byte-identically (keeps `family="serif"` for labels when no font resolved). +327 lines of tests.
- **Files:** `hypertools/plot/fonts.py` (new, 227), `plot.py`, `matplotlib_backend.py`, `tests/test_multibyte.py`, `notes/issues-to-close-on-merge.md`.
- **Why:** GH #205 — CJK/multibyte labels rendered as tofu.
- **Review flags:** ⚠️ New feature touching every text surface; broad `except Exception` in `_font_covers` (intentional, caches failures). No new deps (FT2Font ships with mpl). Font scan cost mitigated by module-level caches. ASCII no-op path carefully preserved. Low risk, well-tested.

### dfb25292eb — feat(plot): multibyte support for the plotly backend + CI font provisioning
- **What:** Threads resolved font family into plotly `layout.font.family` (with CJK fallback chain), fixes title hardcoding its own family. CI (ubuntu) now installs `fonts-noto-cjk`, runs `fc-cache -f`, and force-rebuilds mpl font cache; adds committed subprocess helper `scripts/render_multibyte_plotly.py` (subprocess isolation to survive kaleido/Chromium hangs). +186 test lines.
- **Files:** `.github/workflows/test.yml`, `plot.py`, `plotly_backend.py`, `scripts/render_multibyte_plotly.py` (new), `tests/test_multibyte.py`.
- **Why:** GH #205 plotly parity + make CI actually exercise CJK (previously would silently skip).
- **Review flags:** ⚠️ Network/deps in CI: adds `apt-get install fonts-noto-cjk`. Plotly can only take a family name, not embed a font file — static kaleido export still depends on the exporting machine's fonts (documented honestly). Reasonable.

### 26568257cc — feat(plotly): labels= point annotations (parity with matplotlib)
- **What:** New `_build_point_annotations` gives plotly the previously-missing `labels=` point annotations, mirroring matplotlib `annotate_plot` semantics (np.vstack, itertools.chain flatten, skip None, IndexError on short list, ignore extras, 2D/3D only). Rendered as layout.(scene.)annotations, drawn unconditionally incl. animate. Replaces the F2 placeholder test; adds `tests/test_plotly_labels.py` (7 parity cases).
- **Files:** `plotly_backend.py` (+105), `plot.py`, `tests/test_plotly_labels.py` (new), `tests/test_multibyte.py`, `scripts/render_multibyte_plotly.py`.
- **Why:** GH #205 F3 — close the last backend gap.
- **Review flags:** none — additive parity feature, mirrors documented matplotlib semantics, directly tested against matplotlib anchor points.

### 28fb7e78c1 — fix(tests): use Unicode noncharacters for the no-covering-font path
- **What:** Test-only: swaps PUA codepoint U+10FFFD (genuinely covered by pan-Unicode fonts on CI) for two noncharacters (U+FDD0, U+10FFFE) to deterministically exercise the no-covering-font warning path.
- **Files:** `tests/test_multibyte.py` (+8/-5).
- **Why:** Test was passing/failing based on installed fonts.
- **Review flags:** none — test correctness fix. Pairs with the source-side placeholder-exclusion in the next commit.

### 3148856390 — fix(fonts): exclude universal-fallback/placeholder fonts from coverage detection
- **What:** `_font_covers` now rejects any font claiming to cover Unicode noncharacters (U+FDD0, U+10FFFE) — catches placeholder fonts (e.g. macOS LastResort, format-13 cmap maps everything to a box) that would otherwise be selected and render tofu.
- **Files:** `hypertools/plot/fonts.py` (+17/-4).
- **Why:** Found via CI: all 9 platforms had such a font; the exact GH #205 symptom.
- **Review flags:** none — targeted, correct heuristic (noncharacters are never in legitimate cmaps). Small extra per-font cost (2 probe lookups).

### b87e635625 — fix(colorbar): discrete segments read in legend order (GH #100 follow-up)
- **What:** Vertical discrete colorbars now render first group at TOP to match legend order: mpl `cbar.ax.invert_yaxis()`; plotly reverses colorscale + reverses tickvals for orientation 'v'. Horizontal/continuous untouched. Adds evidence script + colorbar/multiindex tests.
- **Files:** `plot.py` (+16), `plotly_backend.py` (+19), `scripts/generate_colorbar_evidence.py`, `tests/test_colorbar.py`, `tests/test_multiindex.py`.
- **Why:** Discrete vertical colorbar read bottom-up, reverse of legend.
- **Review flags:** ⚠️ Minor edge case: in the plotly discrete branch, when a user supplies custom `colorbar_info['ticks']`, the colorscale is reversed for orientation='v' but that custom `tickvals` list is NOT reversed (only the default-tick branch reverses tickvals). Custom ticks on a vertical discrete colorbar could therefore mis-align labels vs. flipped segments. Likely a rare/unused path, but a real latent inconsistency worth confirming coverage.



## Commit batch 10

I have audited all 16 commits. Per my read-only constraints and reporting guidelines, I cannot create files, so I am returning the complete audit content directly below (ready to be saved as `batch_10.md`).

---

### bf33a54f55 — test: procrustes non-default-parameter coverage + weak-assert audit fixes (GH #199, #244)
- **What:** Test-only. Adds parametrized coverage for Procrustes `scaling`/`reflection`/`oblique`/`oblique_rcond`/`reduction` params; strengthens weak `format_data`/`reduce`/`text2mat` assertions with real shape/value/invariant checks; fixes GH #244 tests that called `isinstance(...)` with no `assert` (always passed).
- **Files:** tests/align/test_procrustes_child.py, tests/test_procrustes.py, tests/test_format_data.py, tests/test_reduce.py, tests/test_text2mat.py.
- **Why:** Close coverage gaps and repair tautological/no-op assertions found in audit.
- **Review flags:** none — additive test hardening, no source changes.

### 029ed07cf7 — fix: use isinstance for type checks throughout (GH #209)
- **What:** Replaces `type(x) is list/dict` identity checks with `isinstance` across align/cluster/plot/reduce/manip/_shared. Fixes trim_and_pad crash on list subclasses. Adds regression tests (list subclass, OrderedDict cluster spec).
- **Files:** hypertools/_shared/helpers.py, align/{common,hyperalign,procrustes,srm}.py, cluster/cluster.py, manip/resample.py, plot/{plot,matplotlib_backend}.py, reduce/describe.py; tests/test_isinstance_209.py.
- **Why:** Subclasses of built-ins were crashing or mis-dispatched.
- **Review flags:** ⚠️ Minor behavior widening — subclasses now accepted where previously rejected. Correct direction, well-tested; `assert isinstance(...)` still stripped under `-O` but that pre-existed.

### 2b76d4e126 — fix(plot): smooth marker+line styles like line-only styles (GH #141)
- **What:** Interpolation gate changed from `is_line` to new `has_line_component`, so `'o-'` gets the same smoothing as `'-'`. Static mpl backend splits combo styles into a smoothed line artist + markers at raw pre-interpolation points (`raw_xform` threaded through center/scale/nan_to_num using the same stats). Animated/plotly get line smoothing only (documented follow-up).
- **Files:** _shared/helpers.py (has_line_component, split_marker_line_fmt), plot/matplotlib_backend.py, plot/plot.py; tests/test_gh141_marker_line_smoothing.py.
- **Why:** `'o-'` drew unsmoothed straight segments while `'-'` smoothed identical data.
- **Review flags:** ⚠️ Real behavior change to rendered output for marker+line styles (intended). Combo styles now emit 2 artists (one is `_nolegend_`) — could surprise code counting `get_lines()`. Thorough real-render tests. Backend inconsistency (mpl vs plotly/animated) acknowledged in docstring.

### 966abe3d29 — feat(plot): resample= kwarg wiring hyp.manip Resample into plot pipeline (GH #94)
- **What:** New `resample=N` kwarg PCHIP-resamples each dataset via `hyp.manip` Resample right after format_data, before analyze. Early validation: False/None disables; non-bool int >=2 required else ValueError.
- **Files:** hypertools/plot/plot.py; tests/test_gh94_resample_kwarg.py.
- **Why:** Expose existing Resample manipulator in plot pipeline.
- **Review flags:** none — additive, defaults preserve behavior, validated (bool rejected), values verified equal to hyp.manip.

### 715f05581f — feat(plot): arbitrary mpl kwargs passthrough + strict list-length validation (GH #206)
- **What:** `plot(**kwargs)` passes arbitrary mpl kwargs verbatim to every dataset (merged after named kwargs; named/internal wins; extras never per-dataset broadcast). Plotly warns on unmappable extras. `parse_kwargs` now raises ValueError on mismatched-length list instead of silently degrading to None.
- **Files:** hypertools/_shared/helpers.py (parse_kwargs), hypertools/plot/plot.py; tests/test_gh206_extra_kwargs.py.
- **Why:** Support extra styling; stop silent color/marker drops.
- **Review flags:** ⚠️ Behavior change: `parse_kwargs` raising affects ALL callers of color=/marker=/linestyle=/etc. — user code that previously passed a wrong-length list and silently got no styling will now raise. Intended hardening but a compat break for pre-1.0 scripts. New `**kwargs` also means previously-rejected typo'd kwargs now flow to matplotlib (surfacing mpl's own error instead of a hypertools TypeError).

### 4e1a09a289 — notes: mark issue triage as executed
- **What:** Adds a "SUPERSEDED/EXECUTED" banner to notes/issues-to-close-on-merge.md recording audit outcome.
- **Files:** notes/issues-to-close-on-merge.md.
- **Why:** Record that the triage was carried out.
- **Review flags:** none — docs only.

### 48e9d8fbb1 — notes: issue-audit session log
- **What:** Appends a completed-audit summary section to a session log note.
- **Files:** notes/2026-07-05-seven-features-session.md.
- **Why:** Session record.
- **Review flags:** none — docs only.

### b11285b11a — plan: round 17 — address all 20 non-deferred open issues
- **What:** Adds a round-17 plan doc and scope note.
- **Files:** notes/.../plans/2026-07-07-all-open-issues.md, notes/2026-07-07-round17-scope.md.
- **Why:** Plan the migration work implemented in following commits.
- **Review flags:** none — docs/plan only.

### b8ac8fe0c4 — feat(core): add Pipeline class + build_pipeline; unpack_model legacy dict support
- **What:** New hyp.Pipeline (sklearn-style fit/transform chaining), build_pipeline assembling cross-module stages in CANONICAL_ORDER, `_AlignedStep` wrapper for aligners, `_CallableStep` re-run adapters. `unpack_model` gains legacy `{'model','params'}` support (DeprecationWarning) and passes unmatched instances through unchanged.
- **Files:** hypertools/core/pipeline.py (new), core/shared.py, __init__.py, core/__init__.py; tests/test_pipeline.py.
- **Why:** Task 1 of 1.0 API unification (#138/#153/#227).
- **Review flags:** ⚠️ Behavior change: `unpack_model` no longer raises on unmatched non-type instances (loosens validation for manip/align/impute/predict dispatchers) — this is a bug, fixed in the very next commit. `_AlignedStep.transform` mutates `aligner.data` in place (try/finally guarded) — a hack, removed in commit 15. `_CallableStep` "reuse = re-run" is genuinely refit, not real reuse (documented interim).

### 1847c0f381 — fix(core): restore ValueError for wrong-type instances in unpack_model
- **What:** Passthrough of unmatched instances now only when `parent_class is None`; otherwise raises ValueError as before. Skips Pipeline docstring doctest; documents sklearn.clone incompatibility.
- **Files:** hypertools/core/pipeline.py, core/shared.py; tests/core/test_shared.py, tests/test_pipeline.py.
- **Why:** Fixes the validation-loosening regression from b8ac8fe0.
- **Review flags:** ⚠️ Correctness fix for a regression introduced one commit earlier — batch is self-consistent, but confirms b8ac8fe0 shipped a real (short-lived) validation gap. Now tested.

### 33fac24d62 — feat(reduce): migrate reduce/ to the 1.0 pattern + mixture models (#174)
- **What:** New reduce/common.py Reducer base + REDUCERS registry (+ lazy UMAP); reduce() gains return_model= and cross-module manip/normalize/align/cluster kwargs (via build_pipeline). Mixture models (GMM/BGMM/LDA/NMF) become valid reduce= specs returning membership proportions, reusing cluster.py's extracted mixture_proportions/normalize_membership_rows. Canonical + legacy dict specs supported. reduce_list now returns (list, fitted).
- **Files:** hypertools/reduce/common.py (new), reduce/reduce.py, cluster/cluster.py; tests/test_reduce_migration.py.
- **Why:** Task 2 unification + soft-clustering reducers.
- **Review flags:** ⚠️ Large surface change to core reduce path; every legacy call form + return shape must stay byte-identical (commit claims 996 passed). reduce/common.py imports from cluster.cluster at module load (import-cycle sensitivity; repointed in commit 14). Legacy `{'model','params'}` now emits DeprecationWarning (was silent). Mostly additive; single-stage default path preserved.

### 1acd585e0f — fix(reduce): warn on ndims mismatch for fitted-Reducer reuse; test bare-class mixture path
- **What:** Fitted-Reducer reuse branch now emits the same "Unequal values passed to dims and n_components" warning as the constructed path when ndims conflicts with fit-time n_components. Adds tests incl. bare-class GaussianMixture.
- **Files:** hypertools/reduce/reduce.py; tests/test_reduce_migration.py.
- **Why:** Task 2 review follow-up (consistency).
- **Review flags:** none — small, additive warning + tests.

### ab8b6d238b — feat(cluster): migrate cluster/ to the 1.0 pattern (round17 Task 3)
- **What:** New cluster/common.py Clusterer base + CLUSTERERS/MIXTURES registries (old dicts re-exported as models/mixture_models). cluster() gains return_model= and cross-module kwargs; full spec grammar; fitted-Clusterer reuse (predict-less hard clusterers raise NotImplementedError on reuse). core/model.py `_build_registry` composed from REDUCERS+CLUSTERERS; `_resolve_model` delegates to unpack_model. `apply_model(list, return_model=True)` now returns a fitted Pipeline. plot.py internal cluster call switched to canonical `{'model','kwargs'}`.
- **Files:** cluster/{cluster,common}.py, core/model.py, plot/plot.py; tests/test_{apply_model,cluster_migration}.py.
- **Why:** Task 3 unification (#138/#153/#174).
- **Review flags:** ⚠️ Documented behavior change: `apply_model(model=[...], return_model=True)` returns a Pipeline instead of a list of models (tests adapted; could affect external callers). Fitted hard-clusterer reuse raises NotImplementedError for predict-less models — new, deliberate. Large refactor of core dispatch; n_clusters injection logic re-homed but preserved.

### 0c471ba25d — refactor(cluster): point reduce's mixture imports at cluster.common; modernize dict-spec error message
- **What:** reduce/common.py imports mixture_proportions/normalize_membership_rows/MIXTURES directly from cluster.common instead of via cluster.py re-export chain; dict-spec error message names canonical 'kwargs' key.
- **Files:** hypertools/cluster/cluster.py, reduce/common.py.
- **Why:** Reduce import indirection/cycle risk; message accuracy.
- **Review flags:** none — mechanical import retarget + message wording.

### 084b5a61a9 — feat(align): export the 1.0 align dispatcher; fix Aligner.transform to apply to new data (GH #227)
- **What:** hyp.align now exports the class-based dispatcher (new signature `align(data, model='HyperAlign', return_model=, manip/normalize/reduce/ndims/cluster=, format_data=, **kwargs)`), with legacy string aliases, deprecated `align=` kwarg alias, canonical+legacy dict specs, array output. `Aligner.transform(new_data)` now validates shape vs fit-time and applies fitted alignment to new data (was: ignored arg, replayed fit data); adds `is_fitted`. Removes now-redundant `_AlignedStep` from pipeline. tools/align.py becomes a thin shim.
- **Files:** __init__.py, align/align.py, align/common.py, core/pipeline.py, tools/align.py; tests/test_align_migration.py.
- **Why:** Task unification + genuine out-of-sample align (#227).
- **Review flags:** ⚠️ Public API surface change: `hyp.align`'s primary kwarg renamed `align=`→`model=` (old name kept as deprecated alias; positional still works). `Aligner.transform` semantics changed (now transforms passed data) — a real behavior change, but the old "ignore arg" behavior was effectively broken. `fit_transform` now calls `transform(data)` (equivalent). `is_fitted` keys off `self.data is not None`. 28 new tests incl. no-refit poison-pill. Well-covered but the highest-blast-radius commit in the batch.

### 2b8377e521 — fix(align): SRM family transform() reused fit-time index (GH #227)
- **What:** srm.py transformer builds output DataFrame index from incoming data's own index instead of fit-time `kwargs['indices']`, fixing crashes/mislabeling on `.transform(new_data)`. Also numpydoc underline fix.
- **Files:** hypertools/align/srm.py, align/align.py; tests/test_align_migration.py.
- **Why:** SRM out-of-sample transform was broken (depends on commit 15's transform fix).
- **Review flags:** ⚠️ Correctness behavior change for SRM/DetSRM/RSRM transform output indexing (fixes a genuine bug; fit-time path unchanged). Tested.

---

batch 10: 16 audited; high-risk: none (all tested/documented). Watch behavior changes: 715f0558 (parse_kwargs now raises), ab8b6d23 (apply_model returns Pipeline), 084b5a61 (hyp.align model= rename + transform semantics). b8ac8fe0's unpack_model validation gap was fixed in 1847c0f3.


## Commit batch 11

I have reviewed all 16 commits. Since I operate in read-only mode (no file-write tools), here is the audit content for `batch_11.md`, returned directly:

### b4fd5276c5 — fix(align): coerce raw-array new_data to DataFrame(s) in Aligner.transform
- **What:** Runs `new_data` through `dw.wrangle` before shape-validation/`unstack` in `Aligner.transform`, so `model.transform(raw_array)` no longer crashes with "Unsupported datatype: list".
- **Files:** hypertools/align/common.py (+9), tests/test_align_migration.py (+40).
- **Why:** Direct `.transform()` calls bypass the funnel coercion `align()` applies before fit; affects HyperAlign/Procrustes/SRM.
- **Review flags:** none — additive guard, preserves index/shape and existing error messages.

### 77749d65d5 — feat(manip): manip list-chaining via Pipeline + return_model + Smooth kernel=
- **What:** `manip(data, model=[...])` chains via Pipeline; adds `return_model`; routes already-fitted Manipulator/Pipeline through `.transform`. Fixes `Manipulator.transform(*_)` which ignored its arg and always replayed fit-time data (silently broke Pipeline/fitted reuse). Adds Smooth `kernel=` (savgol/gaussian/boxcar).
- **Files:** hypertools/manip/common.py, manip.py, smooth.py; tests/test_manip_chaining.py (+267).
- **Why:** GH #274/#153 cross-module chaining; enable genuine fit-once reuse.
- **Review flags:** ⚠️ Real logic fix to `Manipulator.transform` (behavior change: previously fitted-instance reuse was broken). `kernel='savgol'` default here can't distinguish explicit vs default — a known gap fixed in the very next commit. Well-tested.

### 783dd8b2d6 — fix(manip): Resample.transform uses new data's values; Smooth kernel= sentinel; drop dead imports
- **What:** `Resample.transform(B)` now rebuilds pchip interpolators from B's own values/x-index instead of replaying fit-time A's resampled values. Smooth `kernel` default changed to `None` sentinel so explicit `kernel='savgol'` always beats legacy `mode='gaussian'`. Removes unused imports.
- **Files:** hypertools/manip/resample.py, smooth.py, manip.py; tests (+89).
- **Why:** Fixes 4 round17 review findings on 77749d65 (silent wrong-data replay bug + sentinel ambiguity).
- **Review flags:** ⚠️ Genuine correctness fix (fitted Resample previously returned wrong values for new data). fit_transform on fit-time data unchanged. Regression tests added.

### b49e70c273 — feat(core,tools,plot): cross-module kwargs on analyze/normalize/plot + pipeline= reuse
- **What:** Adds `manip=`/`cluster=`/`pipeline=`/`return_model=` to analyze/normalize/plot. Replaces `_CallableStep` with `_DispatchStep` in build_pipeline — fixes a real bug where every stage silently refit on each `.transform()` (e.g. reduce='PCA' fit a new basis each call). New `Normalizer` fit/transform wrapper (byte-identical z-scoring verified). impute/predict/plot-cluster now emit the DeprecationWarning they previously swallowed; plot cluster canonical-dict `kwargs` no longer dropped.
- **Files:** core/pipeline.py, tools/analyze.py, tools/normalize.py, plot/plot.py, impute/impute.py, predict/predict.py; tests/test_cross_module_kwargs.py (+301).
- **Why:** GH #138/#227 — genuine fit-once-reusable pipelines.
- **Review flags:** ⚠️ Broad behavior change touching 6 dispatchers; new DeprecationWarnings surface. Legacy analyze path kept byte-identical (gated on no new kwargs + return_model=False). Normalizer refactor verified equivalent. Minor: plot's return_model bundle refits build_pipeline on `raw` a second time (perf, acknowledged next commit).

### db211e46ca — fix(plot,tools): thread n_clusters into bundle pipeline; honor impute on build_pipeline path
- **What:** FINDING 1 (HIGH): plot bundle pipeline pre-resolves cluster spec with `n_clusters` (was always defaulting to KMeans n_clusters=3, mismatching the figure). FINDING 2 (MED): analyze/normalize build_pipeline path pre-applies format_data so `impute=` is honored (was silently always PPCA).
- **Files:** plot/plot.py (+23), tools/analyze.py (+12), tools/normalize.py (+11); tests (+62).
- **Why:** Fixes two review findings on b49e70c2.
- **Review flags:** ⚠️ Real correctness fixes. Minor: pre-format then stage's normalize re-runs format_data (idempotent no-op on already-clean data) — acceptable double-compute, documented.

### b513bca74c — feat(plot): label_alpha=, xlabel=/ylabel=/zlabel=, animate= dict form
- **What:** Adds `label_alpha`, axis labels (both backends, 2D/3D, static+animated), and `animate=` dict form unpacked to flat kwargs with conflict/unknown-key validation. mpl label rendering hides ticks/spines/panes individually instead of `set_axis_off()` (which would drop label artists), byte-identical when no label requested.
- **Files:** plot/plot.py (+167), matplotlib_backend.py (+70), plotly_backend.py (+70); tests/test_plot_kwargs_round17.py (+238); 5 PNGs.
- **Why:** GH #103/#154.
- **Review flags:** ⚠️ Uses private matplotlib APIs (`ax._axis_map`, `_axis3don`, `_axis.pane/.line`) — fragile across mpl versions. Otherwise additive, well-validated, byte-identical default paths.

### ad347ccb02 — feat(plot): animate='window' + focused= + duration= verification
- **What:** Adds `animate='window'` (opaque sliding window, no trail) and `focused=` (in-focus window length, defaults to tail_duration → byte-identical when omitted) on both backends. Trail flags warned-ignored for 'window'.
- **Files:** plot/plot.py (+93), matplotlib_backend.py, plotly_backend.py; tests/test_window_animation.py (+387); 2 PNGs.
- **Why:** round17 #8, GH #275.
- **Review flags:** none — additive; focused defaults preserve existing behavior; mpl frame-count var renamed cleanly, both backends resolve identically.

### e02e7eb5bc — feat(plot): 2D animations, both backends
- **What:** Every animate style except 'spin' now works for ndims=2 (fixed viewport). 'spin' now raises ValueError in 2D (was crash/silent-misbehave). rotations=/zoom= warned-ignored in 2D. morph extended to 2D; 1D still raises. surface= per-frame hull stays 3D-only (no-op in 2D).
- **Files:** plot/plot.py (+92), matplotlib_backend.py (+280, new animate_plot2D), plotly_backend.py (+121); tests/test_2d_animation.py (+298), test_morph_animation.py, test_plot.py.
- **Why:** round17 #9, GH #123.
- **Review flags:** ⚠️ Behavior change: 2D `animate` previously raised, now succeeds; 'spin' 2D raises ValueError. Two existing "raises" tests redefined to assert real behavior — verified NOT weakened (added 1D-still-raises test).

### 29f225fa63 — feat(reduce): add torch-backed autoencoder reducers
- **What:** New 880-line autoencoders.py with 6 sklearn-style reducers (shallow/deep/sparse/conv/sequence/variational), shared base + training loop, device resolution, random_state reproducibility, internal standardization. Registered lazily in `resolve_reducer` (mirrors UMAP), so torch not needed to import hypertools.
- **Files:** reduce/autoencoders.py (new), reduce/common.py, reduce/reduce.py, pyproject.toml, .github/workflows/test.yml; tests/test_autoencoders.py (+248).
- **Why:** GH #162.
- **Review flags:** ⚠️ New heavyweight dependency `torch>=2.0` (optional `[torch]` extra, also added to `[dev]` and CI). Lazy import + friendly ImportError done correctly. Large new module — relies on tests for numerical correctness.

### 8dad36ab28 — feat(io): add scikit-learn and seaborn named datasets to hyp.load
- **What:** Inserts sklearn (6 bundled load_* only, no fetch_*) and seaborn resolvers into the load() chain after built-in names, before local-file. Built-in wins; sklearn beats seaborn.
- **Files:** io/load.py, io/sources.py; tests/test_load_sklearn_seaborn.py (+87).
- **Why:** GH #273.
- **Review flags:** ⚠️ Precedence change: a same-named local file (e.g. `iris`) is now shadowed by sklearn/seaborn. ⚠️ Network: `seaborn.get_dataset_names()`/`load_dataset` hit network (cached, broad `except Exception` → skip). Shadowing documented in next commit.

### 161b870240 — docs(io): warn that sklearn/seaborn names shadow same-named local files
- **What:** Docstring-only note documenting the local-file shadowing and how to force a local path (use an extension or a `/`).
- **Files:** io/load.py (+5/-1).
- **Why:** MEDIUM review follow-up on Task 11.
- **Review flags:** none — docstring only.

### cf432049fb — feat(io): add fivethirtyeight/ and kaggle/ explicit-prefix loaders
- **What:** `load('fivethirtyeight/<slug>')` lists folder via GitHub contents API (per-process cached) and loads CSVs from raw.githubusercontent; `load('kaggle/<owner>/<dataset>')` anonymous download via kagglehub. Single CSV→DataFrame, multiple→dict. Explicit prefixes raise immediately on failure (no fall-through).
- **Files:** io/load.py, io/sources.py (+180), pyproject.toml (`[kaggle]` extra); tests/test_load_538_kaggle.py (+91).
- **Why:** GH #116.
- **Review flags:** ⚠️ Network I/O (GitHub API 60/hr unauthenticated limit; kagglehub download). New optional dep `kagglehub>=0.3` (also in `[dev]`). Bug fixed in next commit re: test skip scoping + key collisions.

### 3158605699 — fix(io): scope kagglehub skip to kaggle tests, dedupe kaggle table keys, wrap 538 API errors
- **What:** Moves `importorskip('kagglehub')` from module scope into the 2 kaggle tests (was silently skipping all 6); adds `_table_file_keys()` (unique stems, path-relative fallback on collision) so same-stem files in different dirs don't overwrite; wraps non-404 GitHub HTTP errors in HypertoolsIOError with a 403 rate-limit message.
- **Files:** io/sources.py (+45), tests/test_load_538_kaggle.py, tests/test_load_sources.py (+30).
- **Why:** Review findings on cf432049.
- **Review flags:** ⚠️ Real bug fixes (silent test-skip + data-overwrite). 403/rate-limit branch reviewed by inspection, not live test (documented, reasonable — avoids burning quota).

### ac9421a884 — feat(io): add hyp.io.lsl_stream for LSL input
- **What:** New `lsl_stream()` resolves a live Lab Streaming Layer stream (by name/type, else any) and returns an iterator of per-sample vectors compatible with existing streaming machinery. New `[lsl]` extra (pylsl), also in `[dev]`.
- **Files:** io/lsl.py (new, +114), io/__init__.py, pyproject.toml, .github/workflows/test.yml; tests/test_lsl_streaming.py (+262, real in-process outlet).
- **Why:** GH #130.
- **Review flags:** ⚠️ New optional dep pylsl (wheels bundle native liblsl). Unbounded `pull_sample()` could block forever + no inlet cleanup — fixed in next commit. Minor: docstring references `hypertools.core.exceptions` but code imports `.._shared.exceptions`.

### f01df42cbf — fix(io): bounded per-sample timeout + inlet cleanup for lsl_stream
- **What:** Bounded per-sample pull timeout raises HypertoolsIOError after ~timeout seconds of consecutive silence (stalled source no longer hangs consumer); inlet closed via try/finally on generator exit.
- **Files:** io/lsl.py (+29), tests/test_lsl_streaming.py (+41, real stalled-outlet test).
- **Why:** Task 13 review follow-ups.
- **Review flags:** none — robustness fix, real tests, no mocks.

### 5f42880154 — feat(text2mat): gensim vectorizer=/semantic= wrappers, sklearn->gensim->HF parse order
- **What:** New gensim_models.py (713 lines): sklearn-API wrappers for Word2Vec/Doc2Vec/FastText and LdaModel/LsiModel/HdpModel. text2mat string resolution now sklearn→gensim→HuggingFace (via datawrangler). Fixes `semantic=None` silently defaulting to LDA (now truly skips semantic stage). Dict specs accept canonical `'kwargs'` (legacy `'params'` fallback).
- **Files:** tools/gensim_models.py (new), tools/text2mat.py (+215), pyproject.toml (`[gensim]` extra); tests/test_gensim_text.py (+337); 2 PNGs.
- **Why:** GH #198.
- **Review flags:** ⚠️ Behavior change: `semantic=None` now skips the semantic stage instead of running LDA (justified bug fix, but changes output for anyone who passed None). New optional dep gensim (also `[dev]`). HF fallback tier means any unknown vectorizer/semantic string is now treated as a HF model id rather than erroring — a silent semantics change worth noting. `default_params(...) or {}` guards added. Verified `if semantic:` downstream correctly skips on None.

batch 11: 16 audited; high-risk: none (several medium — 77749d65/783dd8/b49e70c2 logic fixes, 8dad36/cf4320 network+precedence, 29f225/ac9421/5f4288 new deps torch/pylsl/gensim; all gated, tested).


## Commit batch 12

### 2fef66c12a — test(gensim): cover all-OOV zero-vector fallback; document Doc2Vec infer_vector rng caveat
- **What:** Adds one regression test asserting an all-OOV document produces a zero vector (and a mixed doc does not); appends a docstring note to Doc2VecVectorizer that repeated transform() on one fitted instance is not bitwise-identical (gensim infer_vector advances model rng).
- **Files:** hypertools/tools/gensim_models.py (docstring only), tests/test_gensim_text.py (+20).
- **Why:** Task 14 review follow-ups (MEDIUM/LOW).
- **Review flags:** none — additive test + docstring, no logic change.

### 991381779 — feat(evidence): pieman story-trajectories demo end-to-end + jumps evidence (round17 Task 15, GH #275/#274)
- **What:** New evidence script (scripts/round17_evidence/story_trajectories.py, 336 lines) running the GH #275 snippet on full 36-subject 'weights' data, saving mp4 + 3 ffmpeg-extracted frames, and computing real metrics (pre/post-align inter-subject corr, turning angle, max inter-frame jump). New tests exercise the same manip/reduce/align/plot paths on a fast synthetic proxy, plus one opt-in @pytest.mark.bigdata real-data test.
- **Files:** scripts/round17_evidence/story_trajectories.py, tests/test_story_trajectories.py (+200), 7 binary assets incl. a 6.8MB mp4.
- **Why:** Generate reproducible acceptance evidence for #275/#274.
- **Review flags:** ⚠️ Commits a 6.8MB mp4 into the repo (bloat). ⚠️ Tests/script reach into matplotlib private internals (ani._save_count, ani._args index positions) — fragile to matplotlib version bumps; frame-count assert is guarded with <=1/<=2 tolerance. ⚠️ Uses subprocess ffmpeg (script only, not tests) — external tool dependency. No production code touched.

### d43649df9 — fix(evidence): story-trajectories early frame -- require fully-populated window
- **What:** Regenerates the 3 story_frame PNGs after changing candidate-frame search to start at window_frames (so segments are fully populated). Per the parent commit's script logic; only binary images change here.
- **Files:** 3 PNGs (binary only).
- **Why:** Avoid a degenerate/sliver early-frame screenshot.
- **Review flags:** none — regenerated evidence images only; no code in this commit.

### ab3973286 — fix(io,ci): authenticate fivethirtyeight GitHub API calls to avoid CI rate-limit
- **What:** Adds _github_api_headers() sending a Bearer token from GITHUB_TOKEN/GH_TOKEN when present; the 538 folder-listing call now uses it. CI test.yml passes secrets.GITHUB_TOKEN to the three pytest steps. Unauthenticated local use unchanged.
- **Files:** hypertools/io/sources.py, .github/workflows/test.yml.
- **Why:** ~12 concurrent CI jobs exhausted the 60/hr anonymous api.github.com quota (403).
- **Review flags:** ⚠️ Minor behavior change (adds auth header when env token present) — correct and gated; token read from env only, not logged. Low risk.

### 357d1f97c — docs: add numpydoc docstrings to every public def/class outside _externals/ (GH #276)
- **What:** Adds docstrings to 131 public defs/classes across 31 source files; adds tests/test_docstrings.py, an AST gate asserting no undocumented public defs outside _externals/. Only non-docstring diff is 3 signature reformats (space after comma in interp_array/interp_array_list/parse_args).
- **Files:** 31 hypertools/** files (docstrings), tests/test_docstrings.py (new, +98).
- **Why:** Documentation coverage (#276).
- **Review flags:** none — verified docstring-only (sole removals are 3 whitespace-only signature reformats); no behavior change.

### 6c3a3733d — docs: comprehensive Sphinx docs pass (GH #278 #159 #153 ... )
- **What:** 82-file docs pass: api.rst coverage, pipeline_order.rst + committed SVG, 5 new sphinx-gallery examples, 2 tutorial notebooks, conf.py toctree flag. Only real source change is hypertools/plot/backend.py (78 lines) — pure RST reformatting of set_interactive_backend's docstring (Markdown fences -> RST literal blocks, fixed duplicate "3." -> "4." numbering).
- **Files:** docs/**, examples/*.py, docs/conf.py, scripts/round17_evidence/pipeline_order_diagram.py, hypertools/plot/backend.py (docstring only).
- **Why:** Full 1.0 docs buildout.
- **Review flags:** none — backend.py change is docstring-only, no logic. Generated docs/auto_examples not scrutinized (regen). Large but mechanical.

### 3bb956dde — docs(examples): make datasets-tour resilient to network hiccups at gallery-build time
- **What:** Refactors examples/plot_datasets_tour.py to load each of the 4 sources independently in a try/except, skip failures with a printed note, and draw the panel grid from whichever succeeded (dynamic subplot layout, unused panels hidden).
- **Files:** examples/plot_datasets_tour.py.
- **Why:** A transient outage shouldn't abort the readthedocs gallery build (Task 17 LOW).
- **Review flags:** ⚠️ Broad `except Exception` — acceptable/intended for build resilience. Example script only, no library code.

### 777b58e53 — docs(readme): regenerate all media with 1.0 code, fix badges + stale text (GH #277)
- **What:** New re-runnable scripts/round17_evidence/readme_media.py (+212) regenerating all README media; regenerates 6 images + 1 new surface PNG (images/ footprint 10.8MB->6.9MB); updates readme.md API samples (hue=/animate=/cluster=/hyp.describe), adds Surfaces section, swaps dead Travis/Gitter/mybinder badges for GH Actions/RTD/PyPI, fixes stale 2017-era text.
- **Files:** readme.md (+58/-22), scripts/round17_evidence/readme_media.py (new), 7 binary media.
- **Why:** README media/text stale vs 1.0 (#277).
- **Review flags:** ⚠️ README code samples (surface=True, cluster='GaussianMixture', hyp.describe) are doc claims not verified against runtime here; commit message calls n_clusters= "retired" yet the new Cluster sample still uses n_clusters=3 (message imprecision, sample itself plausible). Doc-only, low risk.

### 8c40499ac — fix(io): fetch 538 CSVs via authenticated GitHub API to avoid raw.githubusercontent 429
- **What:** Adds _fetch_538_csv(): when a token is present, downloads each CSV via the authenticated GitHub contents API (Accept: application/vnd.github.raw, 5000/hr); without a token falls back to raw.githubusercontent.com as before. fivethirtyeight_dataset() now calls it.
- **Files:** hypertools/io/sources.py.
- **Why:** Follow-up to ab3973286 — CSV downloads still hit anon raw.githubusercontent 429 in concurrent CI.
- **Review flags:** ⚠️ Real behavior change on the authenticated path (different URL/host + raise_for_status); anon fallback preserved. Well-scoped, verified both paths per message. Low risk.

batch 12: 9 audited; high-risk: none
