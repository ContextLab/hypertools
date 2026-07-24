# HyperTools 1.0: All Non-Deferred Open Issues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all 20 non-deferred open ContextLab issues (#103 #116 #123 #130 #138 #153 #154 #159 #161 #162 #174 #187 #198 #227 #273 #274 #275 #276 #277 #278) with a fully consistent user-facing API, evidence-backed verification, current docs, and green CI on PR #272.

**Architecture:** Complete the migration of ALL modules to the 1.0 pattern already used by `manip`/`predict`/`impute`: one folder per module, main dispatcher file + helper files, base class + registry list + `core.shared.unpack_model` + `dw.decorate.funnel`. A single public `Pipeline` class unifies list-of-steps chaining, cross-module kwargs, and `return_model` everywhere.

**Tech Stack:** Python 3.9+, numpy/pandas/sklearn/datawrangler, matplotlib + plotly backends, torch (NEW optional extra), gensim (NEW optional extra), pylsl (NEW optional extra), kagglehub (NEW optional extra), sphinx/furo/sphinx-gallery docs.

## Global Constraints

- Branch `dev-1.0-refactor` only; PR #272 stays open — NEVER merge, never touch master.
- Env: `/Users/jmanning/hypertools/.venv/bin/python`, `MPLBACKEND=Agg`. Local pytest deselects the 6 kaleido-deadlock tests (`test_animation_export.py::test_plotly_gif_export`, `::test_plotly_spin_gif_export`, `::test_plotly_mp4_export`, `::test_plotly_spin_gif_preserves_realtime_duration`, `test_round3.py::test_static_svg_plotly`, `::test_animated_svg_plotly`); they run in CI. Never pkill kaleido.
- No mocks, ever. Tests use real function calls, real (tiny) models, real files, real streams. If a real test can't run, it fails — it does not skip silently (CI guard-test pattern from #205 fonts).
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Backward compatibility: every currently-passing test (947) keeps passing unless a task explicitly says a behavior is redefined. Legacy `{'model','params'}` dict specs keep working everywhere (with DeprecationWarning), so existing examples don't break.
- Every graphical change ships PNG evidence to `docs/images/v1.0-round17/` rendered from final media, verified by a FRESH subagent viewing the file.
- All new/changed public functions get numpydoc docstrings in the same task (not deferred to the docstring sweep).
- When adding pip deps: add to `pyproject.toml` (as optional extras where noted) + `docs/doc_requirements.txt` if docs need them + CI workflow if tests need them.

---

## The API Design Language (audit outcome — binding for all tasks)

The audit found two coexisting generations plus a third resolver in `core/model.py`. The 1.0 language (below) is the NEW pattern, applied uniformly:

**1. Module layout:** one folder per module: `hypertools/<mod>/<mod>.py` (dispatcher) + helper files (one class/concern each) + `common.py` (base class).

**2. Model specs.** Every place a model/operation is named accepts exactly these forms:
- `str` — registry name, eval-free lookup (`core.shared.unpack_model`)
- class or sklearn-like instance (duck-typed `fit_transform`/`transform`)
- `dict {'model': <any form>, 'args': [...], 'kwargs': {...}}` — canonical dict
- `dict {'model': ..., 'params': {...}}` — LEGACY, accepted with `DeprecationWarning`, mapped to `kwargs`
- `list` of any of the above — a chained pipeline (fit/transform sequentially)
- an already-fitted model or `Pipeline` — applied via `.transform()`; never refit

**3. Every transform dispatcher** (`manip`, `reduce`, `align`, `cluster`, `normalize`, `analyze`, `impute`, `predict`) supports `return_model=False` kwarg. `return_model=True` returns `(result, model)`; `model` is the single fitted wrapper when one step ran, a fitted `hyp.Pipeline` when multiple steps ran.

**4. Cross-module kwargs (#138).** Every transform dispatcher accepts the other stages as kwargs — `manip=`, `normalize=`, `reduce=`, `ndims=`, `align=`, `cluster=` — executed in the canonical order below, with the function's own operation slotted at its stage. `hyp.plot`/`hyp.analyze` accept all of them plus `pipeline=` (a fitted Pipeline reapplied to new data; mutually exclusive with the stage kwargs → `ValueError`).

**5. Canonical pipeline order (#153)** — the single documented order, rendered as a flowchart in docs:

```
load/format (impute happens here) → manip → normalize → reduce → align → cluster(hue) → plot/animate → predict overlays
```

Rationale (documented verbatim in docs): impute must precede everything (models need complete data); manip (smooth/resample/zscore) is per-dataset preprocessing in native space; normalize standardizes feature scales; reduce projects to the target dimensionality (before align, preserving 0.x semantics — `analyze()` has always been `align(reduce(normalize(x)))` — and keeping alignment tractable for high-dim data); align rotates the reduced datasets into a shared space; cluster labels the final space for coloring. A manip LIST may interleave any stages explicitly (e.g. `['Smooth', {'model':'HyperAlign',...}, 'UMAP']`) — explicit lists override the canonical order; standalone kwargs follow it.

**6. Naming:** `model=` is the spec kwarg on single-purpose dispatchers (manip/predict/impute keep it); `reduce=`/`align=`/`cluster=`/`normalize=` remain the stage kwargs everywhere (both as each module's own first kwarg and as cross-module kwargs). No new abbreviations; kwargs are full words.

**7. #154 resolution (documented, not sweeping):** flat kwargs remain the 1.0 direction (Jeremy's own doubt on the mega-dict idea + massive churn). Additively: `animate=` gains dict-spec support `{'style': 'window', 'duration': 30, 'focused': 4, ...}` mirroring the model-spec grammar (`style` plays the role of `model`; remaining keys are the flat animation kwargs; conflict with an explicitly-set flat kwarg → `ValueError`). New `xlabel=`/`ylabel=`/`zlabel=` kwargs (both backends). `style=`/`labels=` mega-dicts are explicitly rejected — recorded in the issue comment and docs.

---

## Workstream 1 — Core unification (#227 #161 #138 #153 #174 #274)

### Task 1: Unified spec resolution + `Pipeline` class

**Files:** Create `hypertools/core/pipeline.py`; Modify `hypertools/core/shared.py` (unpack_model), `hypertools/core/__init__.py`, `hypertools/__init__.py` (export `Pipeline`); Test `tests/test_pipeline.py`.

**Interfaces (produces):**
- `unpack_model(m, valid, parent_class)` additionally: legacy `{'model','params'}` → DeprecationWarning + treated as kwargs; passes through fitted instances unchanged.
- `class Pipeline` (sklearn `BaseEstimator`): `Pipeline(steps)` where steps = list of `(name:str, model)` tuples or bare specs (auto-named); methods `fit(data)`, `transform(data)`, `fit_transform(data)`, `inverse_transform(data)` (best-effort reverse through steps that implement it, else `NotImplementedError` naming the blocking step), `named_steps` dict, `__repr__` listing steps. Operates on hypertools data (array | list of arrays), preserving list structure per stage semantics.
- `build_pipeline(manip=None, normalize=None, reduce=None, ndims=None, align=None, cluster=None, order=CANONICAL_ORDER)` → `Pipeline` assembling given stages in canonical order (helper used by every dispatcher in Tasks 2-6). Stage resolvers imported lazily to avoid cycles.
- Fitted-`Pipeline`-as-spec: any dispatcher receiving a fitted Pipeline applies `.transform`.
- Align steps inside a Pipeline validate on `.transform(new)`: same number of datasets and same column count as fit-time, else `ValueError` with the fit-time shape in the message (Jeremy's #227 constraint).

Tests (real data): chain fit/transform round-trips; refit-vs-reuse distinction; inverse_transform through PCA; align shape validation error; legacy params deprecation warning fired exactly once.

### Task 2: Migrate `reduce/` to the 1.0 pattern (+ mixture models, #174)

**Files:** Modify `hypertools/reduce/reduce.py`; Create `hypertools/reduce/common.py` (`Reducer` base); Modify `hypertools/reduce/describe.py` (keep working); Test `tests/test_reduce_migration.py`.

- `Reducer` base (mirrors `Manipulator`): wraps any sklearn reducer; registry `REDUCERS` covering the current `models` dict names + `'UMAP'` (lazy import preserved) + mixture models (`GaussianMixture`, `BayesianGaussianMixture`, `LatentDirichletAllocation`, `NMF`) whose transform returns membership proportions (reuse the cluster module's soft-membership funnel so `hyp.reduce(x, reduce='GaussianMixture', ndims=3)` yields (n,3) proportions — this closes #174).
- Signature: `reduce(x, reduce='IncrementalPCA', ndims=None, return_model=False, manip=None, normalize=None, align=None, cluster=None, internal=False, format_data=True)`. Cross-kwargs assemble via `build_pipeline`. All legacy call forms keep working (string, class, instance, `{'model','params'}`).
- `hyp.describe` unchanged behavior; adapt internals if needed.

### Task 3: Migrate `cluster/` to the 1.0 pattern

**Files:** Modify `hypertools/cluster/cluster.py`; Create `hypertools/cluster/common.py` (`Clusterer` base); Test `tests/test_cluster_migration.py`.

- `Clusterer` base + `CLUSTERERS` registry (hard) and `MIXTURES` (soft) replacing the two dicts; `n_clusters=` convenience preserved (injected only when in the model signature, as today).
- Signature: `cluster(x, cluster='KMeans', n_clusters=3, return_model=False, manip=None, normalize=None, reduce=None, ndims=None, align=None, format_data=True)` (#138's original ask verbatim).
- `core/model.py::_build_registry` updated to consume the new registries; `apply_model`'s own `_resolve_model` DELEGATES to `unpack_model` + registries (kills the third resolver; `apply_model` keeps its public signature and list-pipeline behavior, now returning `hyp.Pipeline` for lists when `return_model=True`).

### Task 4: Export the 1.0 `align` dispatcher

**Files:** Modify `hypertools/__init__.py` (`align` → `hypertools/align/align.py`), `hypertools/align/align.py`, `hypertools/align/common.py`; keep `tools/align.py` as thin shim calling the new one; Test `tests/test_align_migration.py`.

- New exported signature: `align(data, model='HyperAlign', return_model=False, manip=None, normalize=None, reduce=None, ndims=None, cluster=None, format_data=True, **kwargs)`.
- Legacy compat: `align='hyper'` → `'HyperAlign'`, `'SRM'` → `'SharedResponseModel'` (whatever the registry names are), `n_iter=` passthrough; `hyp.plot(..., align='hyper')` and every existing test/example keeps working.
- `Aligner.transform(new_data)` for held-out data with the #227 shape validation; `fit`/`transform` split verified by aligning half of each dataset and transforming the other half (real numeric assertion: transformed halves are closer across datasets than unaligned, e.g. mean pairwise correlation increases).

### Task 5: `manip` chaining + interleaved stages (#274 #153) + Gaussian smoothing

**Files:** Modify `hypertools/manip/manip.py`, `hypertools/manip/smooth.py`; Test `tests/test_manip_chaining.py`.

- `manip(data, model=[...])` chains via `Pipeline` — Jeremy's exact #274/#275 spec must run:
  `manip = [{'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}}, {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}}, 'ZScore']`.
- Step resolution order inside manip lists: MANIPULATORS → REDUCERS → ALIGNERS → CLUSTERERS → normalize strings — so `hyp.manip(data, model=['Smooth', 'UMAP'])` works (#153: "add reduce/align/cluster to manip").
- `manip` gains `return_model=False` + legacy `{'model','params'}` acceptance (currently the one dispatcher missing it).
- `Smooth` gains `kernel='savgol'` default + `'gaussian'` + `'boxcar'` options (`scipy.ndimage.gaussian_filter1d` / uniform_filter1d), needed for #274's jumps comparison.

### Task 6: Cross-module kwargs on `analyze`/`normalize`/`plot` + `pipeline=` (#138 #227)

**Files:** Modify `hypertools/tools/analyze.py`, `hypertools/tools/normalize.py`, `hypertools/plot/plot.py`; Test `tests/test_cross_module_kwargs.py`.

- `analyze(data, manip=None, normalize=None, reduce=None, ndims=None, align=None, cluster=None, pipeline=None, return_model=False, internal=False, impute=None)` — implemented ON `build_pipeline`; legacy positional/kwarg behavior identical when only old kwargs used. `cluster=` appends cluster labels? NO — analyze returns transformed data; with `cluster=` set it returns `(data, labels)`? Keep it simple and consistent: `analyze` with `cluster=` returns the transformed data with cluster labels available via `return_model` (fitted pipeline's cluster step); document this.
- `normalize(x, normalize='across', return_model=False, ...)` — same semantics, adds return_model (fitted scaler params), stays z-score.
- `plot` gains `manip=` (runs at canonical position — the existing `resample=` kwarg becomes sugar for a Resample manip step, unchanged behavior) and `pipeline=` (fitted Pipeline; mutually exclusive with manip/normalize/reduce/align/cluster kwargs → ValueError). `plot(..., return_model=True)` bundle gains `'pipeline'` key (fitted Pipeline covering all stages that ran).
- END-TO-END test: fit pipeline on dataset A via `hyp.analyze(A, manip='Smooth', reduce='PCA', align='HyperAlign', return_model=True)`, apply to structurally-identical dataset B via `hyp.plot(B, pipeline=p)` — asserts no refit (fitted params unchanged) and correct output shapes.

---

## Workstream 2 — Plot & animation (#103 #154 #123 #275)

### Task 7: `label_alpha=` (#103) + `xlabel`/`ylabel`/`zlabel` + `animate=` dict (#154)

**Files:** Modify `hypertools/plot/plot.py`, `hypertools/plot/matplotlib_backend.py` (annotate_plot ~466/483), `hypertools/plot/plotly_backend.py` (_build_point_annotations); Test `tests/test_plot_kwargs_round17.py`.

- `label_alpha=None` (default → current 0.5), resolve-once like `font=`; mpl bbox alpha + plotly rgba alpha.
- `xlabel=None, ylabel=None, zlabel=None` — both backends, static + animated (mpl `set_xlabel` etc.; plotly `scene.xaxis.title`/layout axis titles; 2D too).
- `animate={'style': 'spin', 'rotations': 2, ...}` dict form unpacked into the flat kwargs before the pipeline runs; `ValueError` on conflict with explicitly-passed flat kwarg; `'style'` key required in dict form.
- Tests assert the rendered artifacts (extract mpl annotation bbox alpha; plotly annotation bgcolor; axis label text in both backends; dict-animate produces identical frames to flat-kwarg call, compared pixel-wise).

### Task 8: `animate='window'` + `focused=` + `duration` semantics (#275)

**Files:** Modify `hypertools/plot/plot.py`, `hypertools/plot/matplotlib_backend.py`, `hypertools/plot/plotly_backend.py`; Test `tests/test_window_animation.py`.

- `animate='window'`: sliding fully-opaque window of the trajectory, NOTHING outside the window drawn (bullettime minus the faded full-trajectory backdrop and minus precog/chemtrails). Works with rotations, both backends.
- `focused=None` kwarg: length (in seconds of data-time, consistent with `tail_duration`'s existing unit) of the in-focus window for `window`/`chemtrails`/`precog`/`bullettime` (today's hardcoded/tail_duration-derived focus length becomes this kwarg's default); ignored (no error) for spin/parallel/morph — document.
- `duration=` (exists, default 30): verify it controls wall-clock animation duration for ALL animate styles including 'window' and morph (frame count = duration × frame_rate); fix any style that ignores it; regression-test actual frame counts and per-style timing.

### Task 9: 2D animations, both backends (#123)

**Files:** Modify `hypertools/plot/matplotlib_backend.py` (dispatch_animate ~966, shape asserts ~1390/1418), `hypertools/plot/plotly_backend.py` (~267/358/822), `hypertools/plot/plot.py` (~1447/1580); Test `tests/test_2d_animation.py`.

- All animate styles (parallel/spin→pan-free static camera/serial/window/chemtrails/precog/bullettime/morph) work for `ndims=2`; spin for 2D = slow rotation of the 2D plane is meaningless → 'spin' on 2D raises a clear `ValueError` suggesting other styles; everything else animates with fixed viewport.
- Tests: frame counts, artist data at first/mid/last frames for a 2D serial + window + morph animation, both backends; save a GIF and assert non-identical frames.

---

## Workstream 3 — Autoencoders (#162)

### Task 10: `hypertools/reduce/autoencoders.py` (torch optional extra)

**Files:** Create `hypertools/reduce/autoencoders.py`; Modify `hypertools/reduce/reduce.py` (registry), `pyproject.toml` (`[project.optional-dependencies] torch = ["torch>=2.0"]`, add to `dev` extra), `.github/workflows/test.yml` (install extra); Test `tests/test_autoencoders.py`.

- Six `Reducer`-compatible classes: `Autoencoder` (shallow), `DeepAutoencoder`, `SparseAutoencoder` (L1 activation penalty), `ConvolutionalAutoencoder` (1-D convs over feature axis for tabular/timeseries), `SequenceAutoencoder` (seq2seq GRU, encodes per-timepoint latents), `VariationalAutoencoder`. Shared torch `nn.Module` scaffolding + training loop in the same file (`_fit_torch_model`): Adam, MSE (+KL for VAE, +L1 for sparse), `epochs=100, batch_size=64, lr=1e-3, hidden_dims=None (sensible geometric defaults), device='auto', random_state=None` kwargs; deterministic when random_state set.
- sklearn-like: `fit/transform/fit_transform/inverse_transform` (decoder!), `n_components` (wired to `ndims=`). Registered in `REDUCERS` via lazy import; importing hypertools WITHOUT torch works; using an AE name without torch raises `ImportError` naming `pip install "hypertools[torch]"`.
- Real tests (no mocks): tiny nets, tiny synthetic data (e.g. 200×10 low-rank + noise), few epochs, CPU: reconstruction MSE decreases vs untrained; latent dim == ndims; inverse_transform shape round-trip; VAE latent ~N(0,1) sanity (mean |μ| below loose bound); works through `hyp.reduce(x, reduce='Autoencoder', ndims=3)` and `hyp.plot(x, reduce={'model': 'VariationalAutoencoder', 'kwargs': {'epochs': 30}})`.

---

## Workstream 4 — IO (#273 #116 #130)

### Task 11: sklearn + seaborn named datasets (#273)

**Files:** Modify `hypertools/io/sources.py`, `hypertools/io/load.py`; Test `tests/test_load_sklearn_seaborn.py`.

- Resolution order becomes: built-in names → **sklearn** (`sklearn.datasets.load_*` names: `'iris'`, `'digits'`, `'wine'`, `'breast_cancer'`, `'diabetes'`, ...) → **seaborn** (`seaborn.load_dataset` names, via its `get_dataset_names()`) → local file → HF → Sheets/Drive/Dropbox → URL. sklearn returns `(data as DataFrame with feature names, target appended as 'target' column)`; seaborn returns its DataFrame unchanged.
- Real tests: `hyp.load('iris')` shape/columns; `hyp.load('penguins')` (seaborn); a built-in name that shadows nothing still resolves first; unknown name error message lists the tried resolvers.

### Task 12: fivethirtyeight + kaggle loaders (#116)

**Files:** Modify `hypertools/io/sources.py`; `pyproject.toml` (`kaggle = ["kagglehub"]` extra + dev); Test `tests/test_load_538_kaggle.py`.

- `hyp.load('fivethirtyeight/<slug>')` → raw.githubusercontent.com/fivethirtyeight/data/master/<slug>/ — list the folder via the GitHub API, load the CSV (single CSV → DataFrame; multiple → dict of DataFrames). `hyp.load('kaggle/<owner>/<dataset>')` → `kagglehub.dataset_download` (anonymous works for public datasets), then load contained CSV(s) same way. Both are explicit prefixes — NO index maintained (Jeremy's directive).
- Real network tests (small datasets, e.g. `fivethirtyeight/bechdel`): correct type + non-empty; kaggle test uses a tiny public dataset; kaggle test marked to fail-with-message (not skip) if kagglehub missing in CI env where it's installed.

### Task 13: LSL streaming via pylsl (#130)

**Files:** Modify `hypertools/io/streaming.py`; `pyproject.toml` (`lsl = ["pylsl"]` extra + dev); CI workflow (install `liblsl` on ubuntu, extra on all OS — verify availability per-OS; where the native lib can't install, the test must be excluded EXPLICITLY in the workflow with a comment, not skipped silently); Test `tests/test_lsl_streaming.py`.

- `hypertools.io.lsl_stream(name=None, type=None, timeout=10.0)` → iterator of samples compatible with the existing `is_stream()`/`row_to_vector()` machinery, so `hyp.plot(hyp.io.lsl_stream(type='EEG'), stream_init=..., ...)` just works; `pylsl.resolve_byprop` under the hood; ImportError message names `pip install "hypertools[lsl]"`.
- Real test: spin up an in-process `pylsl.StreamOutlet` on a background thread pushing deterministic synthetic samples, consume via `lsl_stream()`, assert the received vectors match; end-to-end `plot_stream` smoke with `show=False`.

---

## Workstream 5 — Text: gensim (#198)

### Task 14: gensim wrappers with sklearn → gensim → HF parse order

**Files:** Modify `hypertools/tools/text2mat.py`; Create `hypertools/tools/gensim_models.py` (wrappers); `pyproject.toml` (`gensim = ["gensim>=4"]` extra + dev); Test `tests/test_gensim_text.py`.

- sklearn-API wrappers (fit/transform/fit_transform) for gensim `Word2Vec`, `Doc2Vec`, `FastText` (vectorizer-stage: doc vector = trained embedding average / Doc2Vec inferred vector) and `LdaModel`, `LsiModel`, `HdpModel` (semantic-stage over a BoW corpus built internally).
- Name resolution for `vectorizer=`/`semantic=`: try sklearn registry → gensim wrapper registry → HF/data-wrangler fallback (existing) — exactly Jeremy's order; dict/instance specs work per the unified grammar.
- Real tests: train Word2Vec + LDA on a ~50-sentence synthetic corpus; assert output matrix shapes; `hyp.plot(docs, vectorizer='Word2Vec', semantic=None, ...)` end-to-end; parse-order test: a name existing in sklearn is NOT taken from gensim.

---

## Workstream 6 — Story trajectories (#275) + #274 re-verification

### Task 15: pieman demo end-to-end + jumps evidence

**Files:** Create `scripts/round17_evidence/story_trajectories.py` (evidence generator; committed under scripts/); Test `tests/test_story_trajectories.py`.

- Jeremy's exact snippet must run and save an animation:
  ```python
  data = hyp.load('weights')
  manip = [{'model': 'Smooth', 'args': [], 'kwargs': {'kernel_width': 25}},
           {'model': 'Resample', 'args': [], 'kwargs': {'n_samples': 1000}},
           'ZScore']
  hyp.plot(data, manip=manip, align={'model': 'HyperAlign', 'args': [], 'kwargs': {'n_iter': 10}},
           animate='window', reduce='UMAP', duration=30, focused=4)
  ```
- Acceptance (Jeremy's words): "lines moving together, loosely aligned" + "interesting (not just straight-line) paths". Quantify: post-align mean pairwise inter-subject trajectory correlation ≫ pre-align; path curvature non-trivial. Screenshot frames at 3 angles/timepoints from the FINAL saved media at measured worst-case frames; fresh-subagent visual verification.
- #274: side-by-side renders — no-manip (jumpy baseline) vs chained-manip (smooth) vs `Smooth(kernel='gaussian')` — same seed/view; per-step numeric velocity/discontinuity stats (max inter-frame jump distance) demonstrating the fix; posted to both issues.

---

## Workstream 7 — Documentation & evidence (#276 #278 #159 #187 #277 #153-docs)

### Task 16: Docstring sweep (#276)

**Files:** All flagged files (`manip/*`, `align/*`, `predict/*`, `_shared/helpers.py`, `external/*`, `reduce/describe.py`, + anything new); Test `tests/test_docstrings.py`.

- numpydoc docstrings for every public (non-underscore) def/class package-wide (excluding `_externals/`). Enforcement test: AST scan asserts ZERO missing public docstrings outside `_externals/` — a real regression gate, not a count snapshot.

### Task 17: Sphinx docs current + comprehensive (#278 #159 #153-flowchart)

**Files:** Modify `docs/api.rst` (add `manip`, `save`, `set_interactive_backend`, `Pipeline`, autoencoder/gensim/LSL entries), delete `docs/hypertools.tools.align.rst` + `docs/hypertools.tools.normalize.rst`; Create `docs/pipeline_order.rst` + flowchart (graphviz/mermaid rendered to SVG committed to docs/_static/); update `docs/tutorials.rst`.

- New/updated tutorials (executed notebooks, real outputs): (a) story trajectories with full background — dataset from Simony et al. 2016 Nature Communications (ncomms12141), PieMan audio story (The Moth), HTFA k=100 hub extraction, what the animation shows; (b) Wikipedia embeddings (#187): install `Wikipedia-API` in-notebook, pick keyword set, fetch articles, plot point clouds colored by keyword, volumetric density overlay (extends the existing `wikipedia_embeddings` tutorial); (c) datasets tour: sklearn/seaborn/538/kaggle loading (#116/#273); (d) LSL streaming demo (#130, synthetic outlet so it executes anywhere); (e) pipelines & `return_model` (fit-once/reuse, Pipeline round-trip, #227/#161); (f) autoencoders (#162); (g) gensim text models (#198). Every notebook actually executed; committed with outputs.
- `sphinx-build -b html` completes with zero NEW warnings; `toc.not_included` warnings for the two orphans gone.

### Task 18: README refresh + regenerated media (#277)

**Files:** Modify `README.md`, regenerate `images/*` assets (same filenames + new ones), `docs/index.rst` (gif reference).

- Regenerate every embedded image/GIF with current 1.0 code via a committed `scripts/round17_evidence/readme_media.py`: hero animation (story-trajectories window animation), plot/align before-after/cluster/describe examples, plus at least one plotly render and one hull-surface or mixture-model example (new-in-1.0 per the issue). Replace defunct Travis/Gitter/mybinder badges with GitHub Actions + readthedocs badges. Update stale 2017/2018 text. Fresh-subagent visual verification of every image.

### Task 19: Evidence roll-up, issue comments, PR update, CI

- Per-issue evidence comments (code + pasted output, screenshots where graphical) on all 20 issues; close each as completed once verified (matching the audit round's convention Jeremy endorsed); PR #272 body + summary comment updated with the round's evidence; effort labels removed on closed issues.
- Full local suite + linters + `make html` green → push → all 12 CI jobs green (watch to completion). Any failure fixed at the code level (never by weakening tests).

---

## Execution order & dependencies

1 → (2,3 parallel-safe conceptually but SDD runs serially) → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19. Tasks 7-14 depend only on Task 1-6's contracts; Task 15 needs 5,6,8; docs tasks need everything.

## Self-review notes

- Spec coverage: all 20 issues map to tasks (checked; #161 covered by Tasks 1/4/6 evidence + tutorial (e); #154 by Task 7 + design-language §7; #153's flowchart by Task 17).
- Type consistency: `Pipeline`, `build_pipeline`, `Reducer`/`Clusterer`/`Aligner` names used consistently across tasks.
- The one intentionally-redefined behavior: `apply_model`'s internal resolver delegates to `unpack_model` (Task 3) — public signature and outputs unchanged; suite must stay green.
