# HyperTools 2.0 Roadmap (branch: dev-2.0)

> **STATUS (2026-07-02):** Phases 0-2 implemented and verified (169 tests,
> 44/44 screenshot matrix, notebook executed clean); PR opened to master for
> Jeremy's review. Remaining: stack/unstack apply_model core, DataGeometry
> transform chain (#227/#236), gallery regeneration, animation bug burn-down
> (#264/#265), deprecated-kwarg retirement.

Synthesized 2026-07-01 from: (a) the `jeremy/dev` refactor (97 commits), (b) the backend
experiment branches (`threejs-backend`, `d3-threejs-backend`, `matplotlib-backend-revert`),
(c) a dependency/CI/test audit of master v0.8.2, and (d) the open GitHub issue list.

## Strategy

**Modernize in place.** The old refactor was a parallel rewrite in a `dev/` folder that
drifted 88 commits behind master with no tests. Instead: evolve `hypertools/` directly on
`dev-2.0`, keeping the existing public API and the 129-test suite green at every commit.
No push to master; final integration only via PR after Jeremy's manual sign-off.

## What we take from the old refactor (verified against jeremy/dev)

| Idea | Source | Notes |
|-|-|-|
| Unified `apply_model` dispatch (string/callable/list-pipeline/dict) | dev/core/model.py | Reimplement WITHOUT `eval()` — explicit registry + whitelist (`module_checker` idea) |
| Composable decorator chain: format → fill-missing → stack/unstack → list-generalize | dev/decorate.py | The heart of the design; lets every fn accept arrays/DataFrames/text/lists |
| `Aligner` base class (fit/transform + required-params) | dev/align/common.py | Strongest module; port with hyperalign bugfixes (Procustes typo, projection accumulation) |
| Complete Procrustes impl | dev/align/procrustes.py | Port directly |
| Vendored SRM/DetSRM/RSRM (brainiak, Apache-2.0) | dev/external/brainiak.py | Replaces brainiak dep |
| Standalone PPCA | dev/ppca.py | Replaces unmaintained pca-magic |
| Config-file defaults + `apply_defaults` decorator | dev/core/config.ini | Central place for plot styling + model defaults |
| Forward+inverse transform chain ("HyperData") | dev notebook (never built) | Becomes DataGeometry 2.0: replay/invert transforms, addresses #227, #236, #224 |
| `vals2colors`, `snap` (linear_sum_assignment), sliding-window helpers | dev notebook | Port into plot/color + animation modules |

**Dead ends (do NOT carry):** flair/tensorflow-hub text embeddings (→ optional
sentence-transformers), modin, holoviews abstraction layer, Google-Docs corpus downloads,
`eval()`-based model importing, pythreejs (widget rendering fundamentally broken — see
fork notes/visualization_backend_research_2025-06-14.md), D3+Three.js custom backend
(14-week plan, never started — maintenance trap).

## Data shapes & color system — first-class 2.0 features (per Jeremy, 2026-07-02)

These revamp ideas are CRITICAL carries, not nice-to-haves:

1. **Multilevel-index support** (fork #14, #16): nested lists coerce to MultiIndex
   DataFrames; color determined by the OUTERMOST index level; line thickness and opacity
   decrease per deeper level (summary → detail rendering). This is the general mechanism
   that subsumes today's list-of-arrays handling.
2. **Stack/unstack as the universal implementation strategy** (fork #15; dev/decorate.py
   `stack_handler`): every model-applying function stacks the list of DataFrames into one
   MultiIndex frame, applies the model ONCE (correct shared embedding/clustering across
   datasets), then inverts the stack to restore the input structure. `stack=True/False`
   exposed to users; `return_model` threads the fitted model out. Modern datawrangler
   provides `dw.stack`/`dw.unstack`/`dw.decorate.apply_unstacked` equivalents.
3. **Robust coloring** (revamp notebook `mat2colors`/`vals2colors`; fork #11, #24, #32):
   one colorization pathway accepting group labels (categorical), continuous 1D values
   (binned through a palette), mixture proportions / arbitrary matrices (2D → reduce to
   1D; ≥3D → reduce to 3D, treat as RGB or map through palette), or user-specified
   colors. Includes multicolored lines (2D segment-based, 3D streamtube-style), colored
   connections between cluster blocks (#32), and must be TESTED against lists of
   DataFrames from day 1 (fmt-string crash, #24). Notebook draft impls are buggy
   (np.hist/np.reduce typos, dead code) — treat as specs, not code.
4. **Mixture models as first-class alternative to discrete clustering** (fork #10, #23):
   `cluster()` returns hard labels for cluster models but SOFT mixture proportions for
   `GaussianMixture`, `BayesianGaussianMixture`, `LatentDirichletAllocation`, `NMF`
   (via predict_proba/transform). Plot colors blend by membership weights through
   mat2colors. Line-mode rendering with soft assignments must be tested (#23 regression).

## Plotting architecture (the central design decision)

- **matplotlib stays the default backend** — publication-quality, universal.
- **plotly as the optional interactive backend** — best-in-class in Colab/Kaggle (where
  hypertools is used most); fixes #235 (Colab animation) and #191 (interactivity) for free.
- `hyp.plot(..., backend='auto' | 'matplotlib' | 'plotly')`: auto = matplotlib except when
  an interactive notebook frontend (Colab/Kaggle/Jupyter widget) is detected.
- **`HyperToolsFigure` wrapper** (from d3-threejs planning docs): backend-agnostic result
  object with `.show()`, `.save()` (png/pdf/svg/html/gif/mp4), access to underlying
  fig/ax or plotly Figure, and the fitted geo (reduce/align/cluster models + data).
- **Shared style layer**: hypertools' signature aesthetic (palettes, 3D camera angle, clean
  grid, no tick labels) defined once as data, rendered by both backends — the
  matplotlib-revert branch flagged aesthetic parity as the main unresolved gap.
- Styling applied per-figure via rc_context — never mutate global rcParams at import (#259).
- Animations: matplotlib `FuncAnimation` (port — old branch's animate.py was plotly-only)
  AND plotly frames; fixes #265 (numpy≥2 Jupyter animations), #264 (figures in loops).

## Modernization checklist (from master audit)

1. **Packaging**: pyproject.toml (PEP 621); delete .travis.yml; drop importlib_metadata
   marker; classifiers → py3.10–3.13.
2. **Deps**: drop pca-magic (vendor dev/ppca.py), drop hdbscan (→ sklearn.cluster.HDBSCAN),
   umap-learn optional + lazy, deepdish removed (legacy .h5 loader with clear error),
   text embeddings optional extra.
3. **Performance**: lazy imports via module `__getattr__` (import currently 5.1s warm,
   ~3s = umap/numba; target < 1s); fix double-reduction in plot.py (analyze() then
   reducer() again); replace str()-keyed memoize with bounded shape+bytes-hash cache.
4. **Correctness**: remove blanket SyntaxWarning filter; fix format-string linestyle
   parsing (documented in fork notes/technical_findings_commit_da5ae16.md); retire
   long-deprecated kwargs (model/model_params/group/ndims/align/normalize legacy forms).
5. **CI**: py3.10–3.13 matrix; setup-python@v5, cache@v4, codecov@v4; add doc build +
   linter + screenshot-artifact upload jobs.

## Testing (the top priority per Jeremy)

- **Every public function tested with real calls** (no mocks) across: single array, list of
  arrays, DataFrame(s), text input, missing data, 2D/3D/high-D, static + animated,
  matplotlib + plotly backends.
- **Screenshot harness**: every plot-producing test saves a PNG to
  `tests/screenshots/<function>/<case>.png`; CI uploads as artifacts; visual review
  checklist in the dev notebook. (Plotly figures rendered via kaleido.)
- **Dev notebook** `dev/hypertools_2.0_dev.ipynb`: one section per public function,
  exercising the full use-case matrix, for interactive verification in Jupyter/Colab.
- Performance regression checks: import time + plot() wall time on canonical datasets.

## Phases

- **Phase 0 — infra (first)**: pyproject.toml, CI bump, lazy imports, screenshot harness
  + dev notebook scaffold, baseline screenshots of v0.8.2 behavior for comparison.
- **Phase 1 — core**: registry-based apply_model, decorator chain, DataGeometry 2.0
  (transform chain w/ inverses), port PPCA/SRM/procrustes/Aligner.
- **Phase 2 — plotting**: style layer, HyperToolsFigure, matplotlib renderer (default),
  plotly renderer, backend='auto', animations on both.
- **Phase 3 — issue burn-down**: #265, #264, #259, #235, #236, #227, #223, #217, #211,
  #206, #205, #193, #190...
- **Phase 4 — docs**: rebuild Sphinx site, regenerate gallery (incl. interactive plotly
  examples), update README/tutorials for 2.0 API, migration guide.

## Design decisions mined from the fork's issue tracker (jeremymanning/hypertools #1-34, incl. comments)

- **Backend lesson (the big one):** the refactor adopted plotly as PRIMARY backend (#1, #2)
  and its hardest, never-fixed bugs were plotly rendering issues: 3D camera mis-centering
  (#25, reopened after a claimed fix), GIF exports losing background (#33) and freezing
  camera position (#34). This validates 2.0's reversal: matplotlib default, plotly optional.
- **Caching removed deliberately** (#3, #4): stringified cache keys ignored kwargs/nested
  dicts → stale results with same data + different args. 2.0: no memoize (matches audit).
- **API hygiene decisions to keep** (#4, #5): single canonical form per argument (no
  `color`/`colors` aliases; `align='hyper'` not `align=True`); DataFrame-centric I/O
  (arrays coerced to DataFrames; list-of-DataFrames as the universal internal form).
- **DataGeometry tension:** fork decided to REMOVE DataGeometry; but ContextLab users ask
  for geo-like access (#227, #236, #224). Resolution: `HyperToolsFigure` result object
  carries data + fitted models + transform chain — the useful parts of geo without the
  legacy class.
- **Normalization semantics are load-bearing** (#27): the old pipeline normalized by
  default; the refactor didn't, and the "story trajectories" demo became irreproducible.
  2.0 must document defaults precisely and add a reproducibility test for that pipeline.
- **Known-broken patterns to test from day 1:** fmt-string/color handling on lists of
  DataFrames crashed (#24); cluster+line mode rendering (#23); smooth/resample+UMAP
  trajectory "jumps" (#26); multilevel-index styling (color by outer level, thickness/
  opacity by depth, #14) was designed but never finished.
- **load/save design** (#12, #13): extensible per-filetype loader (named datasets, URLs,
  local files, images/text/audio, sklearn/seaborn datasets) with round-trippable save.
  Good 2.0 blueprint; implement incrementally.

## Decisions confirmed by Jeremy (2026-07-02)

- `backend='auto'` policy **APPROVED**: matplotlib default everywhere EXCEPT Colab/Kaggle
  (→ plotly). Conservative for 2.0; revisit broader Jupyter auto-switching later.
- Multilevel indices, stack/unstack strategy, robust coloring, and mixture-model soft
  clustering are REQUIRED carries (see "Data shapes & color system" section above).
