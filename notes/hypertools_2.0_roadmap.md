# HyperTools 2.0 Roadmap (branch: dev-2.0)

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

## Open decision for Jeremy

- `backend='auto'` policy: recommend matplotlib default everywhere EXCEPT Colab/Kaggle
  (→ plotly). Alternative: interactive whenever in any Jupyter frontend. Affects what
  millions of existing notebook users see after upgrading. Recommendation: conservative
  (Colab/Kaggle only) for 2.0, revisit later.
