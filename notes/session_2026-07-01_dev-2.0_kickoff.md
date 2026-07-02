# Session 2026-07-01: hypertools 2.0 kickoff (dev-2.0 branch)

## Goal (user request)
1. Diagnose/clean up "10k+ uncommitted changes" in local repo. ✅ DONE
2. Explore the old refactor attempt (jeremymanning/hypertools fork) in detail.
3. Start a new dev branch integrating the refactor's *ideas* into a correct, modernized toolbox.

## Hard constraints from user
- **NEVER push to or touch master directly** — mission-critical repo; user must manually sign off on the PR.
- Not committed to plotly; it's attractive for interactivity in Colab/Kaggle (where hypertools is used most).
- Ideal: **preserve matplotlib backend**, add the good parts of the refactor.
- Update all docs; modernize to current library versions; drop unmaintained deps; optimize bottlenecks ("snappy").
- **EVERY function meticulously tested + screenshotted** across many use cases; create a dev notebook for testing.

## Findings so far

### "10k+ uncommitted changes" — RESOLVED
Working tree was actually clean. The count came from two untracked dirs: `.venv/` (545MB virtualenv, Python 3.12.10) and `.omc/` (12KB tooling state). Fixed by adding `.venv/`, `venv/`, `.omc/`, `__pycache__/` to .gitignore (commit c8257b6 on dev-2.0).

### Repo/branch state
- `origin` = ContextLab/hypertools, master at 125a09b (v0.8.2).
- `origin/dev` is fully merged into master (0 ahead / 18 behind) — nothing to salvage.
- Added remote `jeremy` = jeremymanning/hypertools fork. Branches:
  - `jeremy/dev` — the big refactor: 97 commits ahead / 88 behind master (merge-base 948050a). Rewrite lives in `dev/` folder: modular packages (align/, cluster/, core/, manip/, reduce/, plot/, external/), sklearn-like model API, config.ini/configurator, datawrangler integration, HyperData class concept. Plotting used **holoviews** (not plotly!). Heavy deps: flair, modin, holoviews. Last commits = align module draft "not tested or debugged".
  - `jeremy/threejs-backend`, `jeremy/d3-threejs-backend` — Three.js/D3 backend experiments + research docs.
  - `jeremy/matplotlib-backend-revert` — documents why backend experiment was reverted (technical findings docs).
  - `jeremy/streaming` — zmq-based streaming plots, keyboard listener.
  - `jeremy/text-features` — format_data/transform fixes, ndims bug fixes.
- Deleted stale local branch `fix/colab-import-error` (already merged).
- Created local branch **`dev-2.0`** from master — all 2.0 work goes here.

### Performance
- `import hypertools`: 11.6s cold / **5.1s warm**. ~3s is umap→pynndescent (numba JIT) eagerly imported via `hypertools.tools.reduce`; seaborn ~0.7s. Fix: lazy imports (PEP 562 module `__getattr__`).

### Open GitHub issues worth addressing in 2.0 (selection)
- #265 animations broken with numpy≥2 in Jupyter; #235 animate=True broken in Colab
- #264 multiple figures in a loop; #259 import mutates matplotlib rcParams (side effects!)
- #251/#244/#199 better tests; #236 extract cluster/reduced data from plot result
- #227 sklearn-pipeline-style DataGeometry; #217 return procrustes projection matrix
- #212 pip install dependency pain; #193 geo iterable/indexable; #191 interactive backend request (ipyvolume)

## Completed this session
- [x] All 4 subagent reports collected (jeremy/dev design, backend branches, master audit, fork issue tracker #1-34 incl. comments). Full synthesis in **notes/hypertools_2.0_roadmap.md** — read that first when resuming.
- [x] Roadmap/architecture doc committed on dev-2.0.
- [x] Screenshot harness (scripts/screenshot_harness.py) + baseline generator (scripts/generate_baseline_screenshots.py): 13/13 cases pass on v0.8.2, PNGs in tests/screenshots/baseline_v0.8.2/ (gitignored), 2 spot-checked visually — correct.
- [x] Dev notebook: dev/hypertools_2.0_dev.ipynb (valid nbformat, 20 cells, one section per public function + use-case matrix).

## Jeremy's confirmations (2026-07-02)
- backend='auto' policy approved (plotly only on Colab/Kaggle; matplotlib elsewhere).
- REQUIRED carries from revamp: multilevel-index support, stack/unstack implementation strategy, robust coloring (mat2colors/vals2colors), mixture models as soft-clustering alternative. Detailed specs added to roadmap ("Data shapes & color system" section) + dev notebook section 4.

## Key facts to remember when resuming
- Branch: **dev-2.0** (all work here; NEVER touch master — PR only after Jeremy signs off).
- Fork remote added as `jeremy`; refactor code = `jeremy/dev` branch `dev/` folder; backend lessons in `jeremy/matplotlib-backend-revert` notes/.
- Fork issues verdict: plotly-as-primary caused the worst unfixed bugs (GIF camera #34, GIF background #33, 3D camera centering #25) → matplotlib default + plotly optional is the confirmed 2.0 architecture.
- `describe(show=False)` skips figure creation entirely (API inconsistency, 2.0 cleanup item).
- Import perf: 5.1s warm (umap/pynndescent ~3s) — lazy imports are Phase 0.
- Agent-report gotcha (harness quirk this session): subagent final messages arrived as stubs; recover with jq over the task .output transcript (assistant text entries).

## Implementation status (updated 2026-07-02, second work block)
All implemented on dev-2.0, tests green at every commit (169 passing, up from 136):
- **Phase 0 DONE**: pyproject.toml (v2.0.0.dev0, py3.10+), CI matrix 3.10-3.13 + action bumps + screenshot artifacts, readthedocs py3.11, memoize REMOVED everywhere (user requirement), lazy imports (import 5.1s → 1.46s), sklearn HDBSCAN swap (external hdbscan + SyntaxWarning filter dropped), double-format fix in plot, rc_context styling fix (#259, verified rcParams untouched).
- **Phase 1 DONE**: mixture models (GaussianMixture/BayesianGM/LDA/NMF) return (n,k) proportions from cluster(); hypertools/tools/colors.py (mat2colors/colors2groups); plot() supports mixture cluster= (blended colors), matrix-valued hue, continuous hue; nested-list input with multilevel styling (outer-group color, depth-scaled linewidth/alpha; text lists excluded).
- **Phase 2 DONE**: hypertools/plot/interactive.py plotly backend (2D/3D, fmt→mode, camera conversion, no-ticks aesthetic, sliding-window + spin animations w/ play controls); plot(backend='auto'|'matplotlib'|'plotly'); auto = plotly ONLY on Colab/Kaggle (approved policy). kaleido export wired into screenshot harness.
- **Verification: 44/44 screenshot cases pass** (scripts/generate_verification_screenshots.py) covering plot/reduce/align/normalize/cluster/analyze/describe/format_data/load/text + plotly backend cases; INDEX.md manifest generated. Spot-checked visually: correct.
- README updated (What's new in 2.0, requirements, extras).

## Remaining before PR
- [ ] Execute dev notebook end-to-end (in progress), commit executed copy.
- [ ] Commit curated verification screenshots (docs/images/v2.0-verification/) for PR evidence.
- [ ] Final full-suite run + import-time check.
- [ ] Push dev-2.0 to origin; open PR to master (user will review/sign off; DO NOT MERGE).

## Original Phase 0 plan (for reference)
- [ ] pyproject.toml (PEP 621) + delete .travis.yml + CI bump (py3.10-3.13, setup-python@v5, cache@v4, codecov@v4).
- [ ] Lazy imports (module __getattr__) — target `import hypertools` < 1s; verify with -X importtime.
- [ ] Fix double-reduction in plot.py; remove str-keyed memoize (fork issue #3 confirmed it returns stale results).
- [ ] Swap hdbscan→sklearn.cluster.HDBSCAN; vendor dev/ppca.py to replace pca-magic.
- [ ] Run FULL test suite (129 tests) before/after each change; keep green.
- Then Phase 1 (core: apply_model registry, decorator chain, transform-chain result object), Phase 2 (plotting: style layer, HyperToolsFigure, plotly backend, backend='auto'), Phase 3 (issue burn-down), Phase 4 (docs). Details in roadmap.
