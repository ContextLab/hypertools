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

## Pending / next steps
- [ ] Collect 3 subagent reports: jeremy/dev design analysis, backend-branch lessons, master audit (deps/CI/tests/bottlenecks).
- [ ] Write ROADMAP + architecture doc on dev-2.0.
- [ ] Scaffold modernized package (keep matplotlib default backend + optional plotly interactive backend; lazy imports; modern deps; pyproject.toml).
- [ ] Dev/testing notebook + screenshot-based verification harness for EVERY function.
- [ ] Update docs.
- [ ] PR to master only after user sign-off.
