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
- [x] Execute dev notebook end-to-end (8/8 code cells, 0 errors) — dev/hypertools_2.0_dev_executed.ipynb committed. Executing it caught a REAL bug: backend.py's mpl.use fallback only caught ImportError, but matplotlib>=3.9 raises ValueError for missing ipympl (likely the Colab #235 root cause) — fixed.
- [x] Committed verification screenshots to docs/images/v2.0-verification/ (1.2MB, 44 PNGs + INDEX.md).
- [x] Final checks: 169/169 tests, 13/13 baselines, import 1.5s, README updated.
- [x] Sphinx docs build succeeded (use .venv/bin sphinx-build; GIF thumbnail post-processing ok).
- [x] Pushed dev-2.0; PR #270 opened: https://github.com/ContextLab/hypertools/pull/270 (awaiting Jeremy sign-off; DO NOT MERGE). CI matrix (3 OS x py3.10-3.13) running.

## PR body
Saved at scratchpad pr_body.md (session-local); recreate from this file's summary + roadmap if lost.

## Original Phase 0 plan (for reference)
- [ ] pyproject.toml (PEP 621) + delete .travis.yml + CI bump (py3.10-3.13, setup-python@v5, cache@v4, codecov@v4).
- [ ] Lazy imports (module __getattr__) — target `import hypertools` < 1s; verify with -X importtime.
- [ ] Fix double-reduction in plot.py; remove str-keyed memoize (fork issue #3 confirmed it returns stale results).
- [ ] Swap hdbscan→sklearn.cluster.HDBSCAN; vendor dev/ppca.py to replace pca-magic.
- [ ] Run FULL test suite (129 tests) before/after each change; keep green.
- Then Phase 1 (core: apply_model registry, decorator chain, transform-chain result object), Phase 2 (plotting: style layer, HyperToolsFigure, plotly backend, backend='auto'), Phase 3 (issue burn-down), Phase 4 (docs). Details in roadmap.

## Final CI status (2026-07-02 ~01:10 EDT)
- PR #270 full matrix GREEN: 24/24 checks pass (3 OS x py3.10-3.13, both push + pull_request runs).
- One transient failure fixed en route: GitHub's windows/py3.13 runner ships broken Tcl/Tk (TkAgg imports but window creation raises _tkinter.TclError). Fixed in 6e6330e: manage_backend retries the plot once on the original backend after an interactive-backend TclError.
- Session complete. Awaiting Jeremy's PR review/sign-off. NEXT SESSION: address review comments; then remaining roadmap items (stack/unstack apply_model, DataGeometry transform chain #227/#236, gallery regen, #264/#265).

## Third work block (2026-07-02, after Jeremy's PR feedback)
Feedback: (a) backends must match visually (styles/sizing/colors), (b) many features unscreenshotted, (c) formerly-deferred items all IN scope. All addressed:
- **Backend parity**: plotly renderer rewritten to mirror matplotlib exactly — black wireframe cube (3D)/square (2D) via traces/shapes, axes hidden, unit range, camera from elev/azim (r=2.5), pt→px conversion (1.5pt lines, 6pt markers), full fmt support (markers+dashes, 3D symbol fallback since plotly Scatter3d only supports 8 symbols). Parity montage generator: scripts/generate_parity_screenshots.py (22 side-by-side cases; cluster fits precomputed once — refit permutes component colors, not backend skew).
- **is_line() bug found+fixed**: '' in Line2D.markers made it False for ALL fmt strings → line interpolation silently disabled on modern matplotlib; also parse linestyles before marker chars ('-.'). Restoring interpolation exposed label-index bug → labels now re-mapped onto interpolated trajectories (_expand_labels).
- **Multicolored lines**: continuous/matrix hue + line fmt = per-segment continuous coloring. mpl: Line3DCollection/LineCollection replacing line artists; plotly: per-point line colors (3D) / segment traces (2D). Excluded for animate (warns).
- **Deprecated kwargs RETIRED**: plot(group/model/model_params), reduce(model/model_params/normalize/align), align(method/normalize/ndims/align=True→ValueError), cluster(ndims). Old saved geos replay via translation (group→hue) with warning in DataGeometry.plot.
- **apply_model core**: hypertools/tools/apply_model.py — stack→fit once→unstack; specs: name/dict/instance/pipeline-list; modes auto/fit_transform/fit_predict/predict_proba; return_model; stack=False per-dataset; whitelist registry (NO eval). Public as hyp.apply_model. 12 tests.
- **Verification matrix expanded 44→75 cases** (all features BOTH backends incl. multicolor/nested/mixtures/animations) — 75/75 pass. Parity 22/22.
- **Regression tests**: #264 (loop staleness — memoize was root cause), #265 (numpy2 animate, exact repro from issue), #259 (rcParams). 
- **Gallery**: 5 new examples (interactive_backend, mixture_models, multicolored_lines, nested_lists, apply_model) all execute clean; apply_model added to docs/api.rst.
- Notebook re-executed 0 errors. README updated. Tests: 185+ green.

## Final status after third work block (2026-07-02 ~09:00 EDT)
- Evidence push bf2899c: CI fully green again (24/24 checks, 3 OS x py3.10-3.13).
- PR #270 body rewritten with full scope; evidence comment posted:
  https://github.com/ContextLab/hypertools/pull/270#issuecomment-4865614702
- Everything Jeremy flagged is done: backend parity (22/22 montages, docs/images/v2.0-parity), full feature screenshot coverage (75/75, docs/images/v2.0-verification), apply_model core, gallery+sphinx, animation bugs #264/#265 regression-tested, deprecated kwargs retired.
- Awaiting Jeremy's review/sign-off. DO NOT MERGE.

## Fourth work block (2026-07-02, review round 2)
Jeremy's feedback: mixtures not showing multi-class membership; title/aspect/sizing mismatch across backends; animation "doesn't appear to work" + need gif/apng/mp4 export; gallery thumbnails old + verify plotly animations. All addressed (commit b77945f):
- **Animation export**: _save_animation (plot.py) picks writer by extension (.gif Pillow / .png+.apng animated PNG / .mp4 ffmpeg); plotly _export_animation_file renders frames via kaleido + assembles (PIL/ffmpeg), controls excluded from exports (must set layout.updatemenus = (), NOT update_layout([])); plotly n_frames now scales with duration (15/s, clamped 10-90) — export tests 14min → <5min. 7 tests w/ real files. Samples in docs/images/v2.0-animations/ + INDEX.
- **Overlapping mixtures**: all mixture demos use 1.5-sd-separated blobs (make_overlapping_clusters in both scripts, examples, notebook); test asserts >15% genuinely soft assignments. Verified visually: blended boundary colors on both backends.
- **Parity round 2**: title centered/black/16px matching mpl; default 640x480; 2D frame fills canvas (dropped scaleanchor); 3D aspectmode manual 4:4:3 (mpl default box aspect) — test updated accordingly; camera r=1.95 for size match.
- **Gallery**: plotly_sg_scraper in docs/conf.py (renderer sphinx_gallery_png); animate_plotly example; animated thumbnail sphx_glr_animate_plotly_thumb.gif generated + registered in post_build.py. Verified: plotly PNG in built gallery html.
- **Notebook**: animations display inline via to_jshtml + plotly frames; gif export cell; re-executed 0 errors (resources path dev/).
- Suite: 192 passing (185 + 7 animation-export). 22/22 parity, 75/75 verification regenerated. Awaiting CI + posting round-2 evidence comment.

## Round 2 final status (2026-07-02 ~11:15 EDT)
- CI fully green on 0264b3a: 24/24 checks. Round-2 evidence comment posted:
  https://github.com/ContextLab/hypertools/pull/270#issuecomment-4867433316
- Extra hardening shipped during CI stabilization: dataset downloads validate content + retry w/ backoff (fe3bb8f, 0264b3a); CI shares one cross-OS cache of ~/hypertools_data so 24 jobs stop hammering Google Drive (root cause of intermittent text-test failures).
- Totals: 193 tests, 75/75 verification, 22/22 parity, 4 animation GIF exports committed, plotly in sphinx gallery w/ animated thumbnail.
- Awaiting Jeremy's review. DO NOT MERGE.
