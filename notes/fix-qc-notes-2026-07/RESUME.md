# RESUME — HyperTools 1.0 QC-notes fix effort

**Session suspended 2026-07-08. Read this to resume.**

## The task (verbatim intent)
Drive comprehensive fixes from Jeremy's QC verification-notes notebook. Rules:
- Don't take notes at face value: where he says "looks good", re-verify numerically
  on REAL datasets (dozens+ obs) with manual calculations; where he flags an issue,
  treat it as a PATTERN/search signal → find related issues, other affected modules,
  edge cases. Verify numerically AND with screenshots for plotting.
- Work on a NEW branch: **`fix/qc-notes-2026-07`** (created off `dev-1.0-refactor`
  @ 724ad5f0). Currently checked out. No source changes yet — triage only.
- For EVERY fix, spawn a subagent to independently red-team the fix.
- **/goal: open a NEW PR into `dev-1.0-refactor`** with per-issue comments carrying
  DIRECT EVIDENCE (screenshots, copy-pasted outputs, numeric tables): how each issue
  was tested, what was found, broader implications/interpretation of the deeper
  cause, how it was fixed, how the fix was verified.
- Standing constraints: NEVER merge; NEVER touch master; use
  `/Users/jmanning/hypertools/.venv/bin/python` + `MPLBACKEND=Agg`; no mocks; harden
  (don't weaken) tests; commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context)
  <noreply@anthropic.com>`.

## Source of issues
`notes/qc-release-2026-07/hypertools_1.0_verification_notes.ipynb` on
`origin/qc/release-audit-2026-07`. Full extraction:
`<scratchpad>/notes_extracted.txt`. Master catalog:
`<scratchpad>/ISSUE_CATALOG.md`.
(`<scratchpad>` = /private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad)

## Confirmed reproductions (local, real data)
- Animations return a static `matplotlib.figure.Figure` (no `.to_html5_video()`) for
  spin/window/morph/parallel → Jeremy's "pure failure". `animate='chemtrails'` also
  returns a static Figure (renders as window). Per-point `labels=` show every frame.
- `hyp.cluster(new, cluster=fitted_pipeline, reduce=, manip=)` →
  `AttributeError: 'Pipeline' object has no attribute 'labels_'` @ cluster/common.py:196.
- Top-level `hyp.describe/predict/impute/reduce/cluster` ALL already exist & work
  (`__init__.py:7-17` are direct internal imports). Jeremy's "wire in hyp.X" notes =
  Colab pip-cache artifact. STILL must re-verify each numerically for the PR evidence.

## TRIAGE FINDINGS (4 of 6 complete; 2 pending — see below)

### D. Dependencies / packaging  (COMPLETE)
- `pykalman` already in `[predict]` (pyproject L57) + `[dev]` (L109); imported
  predict/kalman.py:22, impute/kalman.py:26. Colab failed only because install was
  `[interactive,torch,gensim]`. It IS installed in `.venv` (v0.11.2).
- Genuinely UNDECLARED external imports: `deepdish` (io/load.py:361, legacy hdf5
  load, guarded, unmaintained — check numpy-2 compat); `PIL` (plotly_backend.py:1022,
  transitive via mpl, low risk); `IPython` (backend.py:420,642, transitive). `datasets`
  is [dev]/[text]-only but sources.py:568 error suggests bare `pip install datasets`.
- Kaleido/Plotly: `[interactive]=["plotly>=5.20.0","kaleido>=0.2.1"]`. Unbounded
  kaleido installs 1.3.0 which needs plotly>=6.1.1 → `write_image` breaks on Colab's
  plotly 5.24.1. write_image/to_image used plotly_backend.py:953,991,1056,1070.
- RECOMMENDED pyproject changes:
  1. `[interactive] = ["plotly>=6.1.1", "kaleido>=1.0"]` (+ mirror in `[dev]`) — matches
     validated dev venv (plotly 6.8.0 + kaleido 1.3.0).
  2. pykalman: to satisfy Jeremy's "add as a dependency", promote pure-python
     `pykalman>=0.11` (+ `statsmodels>=0.14` for ARIMA) to core `dependencies` so
     Kalman/ARIMA predict work out of the box; leave `skaters` in `[predict]`.
     (Decision point: confirm with Jeremy whether statsmodels belongs in core.)
  3. deepdish: add `legacy = ["deepdish>=0.3.7"]`, cite in load.py error (verify
     numpy-2 import first).
  4. Fix sources.py datasets error to suggest `hypertools[text]`.
- Per CLAUDE.md, `[dev]` intentionally omits `[text]`/`[predict-hf]`.

### C. Fitted-model reuse  (COMPLETE)
Reuse matrix (real 40x6, X vs Y=X+5):
| Case | Result |
|-|-|
| reduce(Y, reduce=fitted_Reducer) | OK |
| cluster(Y, cluster=bare fitted sklearn KMeans) | OK |
| cluster(Y, cluster=fitted_Clusterer) single-stage | OK |
| cluster(Y, cluster=fitted_Pipeline, reduce=,manip=) | CRASH labels_ @ common.py:196 |
| align(Ylist, model=fitted_Aligner) | OK |
| manip(Y, model=fitted_Manipulator) | OK |
| normalize(Y, ... return_model) + .transform | OK (P0-1 fine) |
| cluster(Y, cluster=fitted_Clusterer, reduce=,manip=) | CRASH 'Clusterer' has no 'fit' @195 (double-wrap) |
| reduce(Ylist, reduce=fitted_Reducer, align='HyperAlign') | OK |
| reduce(Y, reduce=fitted_Pipeline, ...) | CRASH ValueError @ reduce/reduce.py:189 |
| align(Ylist, model=fitted_Pipeline, reduce=) | CRASH 'unknown model: Pipeline' @ core/shared.py:110 |
| manip(Y, model=fitted_Pipeline) | OK |
- ROOT CAUSE: `return_model=True` is polymorphic — bare wrapper when 1 stage ran, a
  `Pipeline` when cross-module stages ran. Only manip (manip.py:121-123) + normalize
  handle a Pipeline-as-spec. reduce/cluster/align don't; `_resolve_cluster_spec`
  (cluster.py:89-93) re-wraps ANY non-str/non-class (incl. a fitted Clusterer or a
  Pipeline) in a fresh UNfitted Clusterer → `list(model.labels_)` on a Pipeline crashes.
- FIX LOCATION (recommended): dispatcher spec-resolution layer of reduce/cluster/align,
  mirroring manip: (1) add `isinstance(spec, Pipeline)` reuse branch (→ spec.transform
  if fitted); (2) make resolvers idempotent — pass an already-fitted wrapper through
  unwrapped instead of re-wrapping. Optional shared `is_reusable_fitted(spec)` in
  core/shared.py. `unpack_model`/`_DispatchStep` need no change.
- Repro script: `<scratchpad>/triage/reuse_matrix.py`.

### E. names= / labels / legend / double-display  (COMPLETE)
- Current semantics (all in plot/plot.py): `labels=` (sig:151, doc:377-385) = PER-POINT
  text, drawn by add_labels (matplotlib_backend.py:499-621) / _build_point_annotations
  (plotly_backend.py:207-270). `legend=` (sig:152) = PER-DATASET legend; resolved
  :2058-2073 — `legend=True`+no hue → `['1'..'N']` (:2071); +hue → category names.
  Sets `mpl_kwargs["label"]`. `hue=` (:150) = per-obs color. NO per-dataset name concept.
- Issue-1 repro: Smooth-kernel plot legend shows `['1','2','3','4']` while
  `['raw','savgol','gaussian','boxcar']` land as POINT annotations. Screenshot
  `<scratchpad>/triage/names_current.png`.
- names= THREADING: add `names=None` to signature (:151-152) + `_font_texts`(:1360) +
  docstring; NO format_data (per-dataset not per-point); funnel at legend-resolution
  :2058-2073 → set `legend=list(names)`; both backends' existing legend code renders it
  (matplotlib_backend.py:2069-2074; plotly `_trace_name` :486/:1743-1749). Validate
  `len(names)==n_datasets`; guard hue/cluster/MultiIndex regroup. Cleaner: thread names=
  explicitly into `_draw`(:2358)/`plotly_draw`(:2305)/`_trace_name` so it's independent
  of the `legend` bool.
- DOUBLE-DISPLAY root cause: plot() returns bare Figure (:2572) → Jupyter auto-reprs
  (path 1). Plotly ALSO calls `fig.show()` on show=True (plotly_backend.py:955-967) →
  path 2, firing before mpl flush_figures so plotly renders FIRST. `plt.close(fig)`
  deregistration currently only for show=False (:2490-2492). REMEDY: one display path —
  gate/drop plotly internal fig.show(), extend plt.close deregistration to show=True;
  documented "figure without display" recipe = `hyp.plot(..., show=False)`.

### F. describe styling + top-level API  (COMPLETE)
- describe (reduce/describe.py): seaborn lineplot built :120-138, NO despine → closed
  box. FIX: `sns.despine(ax=ax, top=True, right=True)` before plt.show() :138. Also
  hardcoded matplotlib (import :7, no `backend` kwarg :12, never calls resolve_backend).
  Plotly support needs: `backend='auto'` kwarg → resolve_backend (plot/plotly_backend.py
  :164), keep seaborn+despine for mpl, add plotly go.Figure Scatter-per-trace path
  (plotly has no top/right spines). Return stays a dict. Screenshot describe_current.png.
- Top-level API: all 10 public names ARE the internals (`hyp.X is internal` = True).
  12/14 calls PASS. TWO NEW FLAGS:
  1. `predict(model='GP')` FAILS ValueError — registered name is `GaussianProcess`
     (predict/gp.py:108); 'GP' alias missing. ADD alias.
  2. `cluster('GaussianMixture', n_clusters=2)` returns (100,2) float PROBABILITY
     matrix, not hard labels like KMeans (100,) int — behavioral inconsistency; ties
     into hue-blending (B1). Confirm intended (soft clustering) + document.
- Repro/verify: `<scratchpad>/triage/verify.py`.

### B. Hue / color / surface / soft-cluster  (COMPLETE)
- B1 (matrix/proportion hue → no blend). ROOT CAUSE `plot/colors.py:66`:
  `weights = m - np.min(m, axis=1, keepdims=True)` before renormalizing. For k=2 this
  turns every non-tie row into a pure palette vertex (hard argmax); k>2 distorts blend.
  Evidence: real GaussianMixture (160,2) props → mat2colors yields only 2 unique colors;
  ~50/50 row `[0.466,0.534]` → `[0.34,0.83,0.86]`, identical to a 10/90 point. Backend-
  independent (plotly same). Plumbing correct (plot.py:2138-2141→_multicolor_line_colors
  →mat2colors; apply 2424-2429; plotly 491-560) — only color math wrong. FIX: for
  non-negative rows summing >0, use rows directly as weights (L1-normalize only); reserve
  min-shift for signed matrices. Screenshot hue_matrix_scatter_mpl.png.
- B2 (surface ignores hue). ROOT CAUSE: `_resolve_dataset_colors()` (plot.py:2216-2241)
  sources surface_colors from palette cycle (2232) → _resolve_surface_color/
  _surface_base_rgb (matplotlib_backend.py:40-47, plotly_backend.py:1138-1147); hue never
  enters. Points/lines DO get hue; only the hull mesh uses palette. FIX: derive hull color
  from dataset mean hue color when spec color is None. Screenshot hue_surface_mpl.png.
- B3 (arbitrary matrix + color_reduce). NOT IMPLEMENTED — `color_reduce` exists nowhere;
  (n,k>3) hue accepted (no k cap, plot.py:1860) but blended over k palette colors, never
  reduced to RGB. DESIGN: add `color_reduce=` kwarg; in matrix branch (plot.py:1860-1878),
  when k>3 or color_reduce given, `hyp.reduce(hue, reduce=color_reduce or 'IncrementalPCA',
  ndims=3)`, min-max each col to [0,1] as (r,g,b), used verbatim (add "already-RGB" branch
  in mat2colors); k≤3/proportions stay on palette blend after B1 fix.
- RELATED: colorbar+matrix hue intentionally raises ValueError (plot.py:2616-2623) — matrix
  hue still has no colorbar post-fix. `set_interactive_backend('plotly')` does NOT switch
  renderer for these paths; must use `plot(backend='plotly')`. Repros: hue_repro*.py.

### A. Animation system  (COMPLETE)
- A1. ROOT CAUSE `plot.py:2569-2570`: animated matplotlib path returns tuple
  `(Figure, FuncAnimation)` (or a bare Figure depending on path) → caller's
  `anim.to_html5_video()` fails ('Figure' has no to_html5_video). Need a single, documented
  return object for animations exposing to_html5_video()/save without breaking the static
  "plot() returns a Figure" contract.
- A2. ROOT CAUSE `matplotlib_backend.py:1896`: animate-style whitelist EXCLUDES
  chemtrails/precog/bullettime and does no unknown-string validation → falls through to a
  silent static/window plot. (Confirmed chemtrails returns static Figure.)
- A3. ROOT CAUSE `matplotlib_backend.py:499-557` + `plot.py:3038-3039`: per-point labels
  drawn once, never frame-synced; `_expand_labels` clobbers labels on down-sample. Labels
  must be shown only when their datapoint is visible in the current frame.
- Artifacts: `<scratchpad>/triage/anim_spin.gif`, repro scripts + frame PNGs.

## NEXT STEPS (in order)
1. TRIAGE COMPLETE (all 6). Synthesize full fix plan grouped by subsystem (consider
   writing-plans skill). Reports were recovered INLINE via
   `<scratchpad>/extract_report.py <task_output_file>` (harness blocks subagent .md writes).
3. Implement fixes subsystem-by-subsystem; RED-TEAM each with an independent subagent.
   Likely design decisions to confirm with Jeremy: animation return-object shape;
   statsmodels-in-core; `names=` independent-vs-legend threading; GaussianMixture
   labels-vs-proportions default; color_reduce API.
4. Verify: full pytest (local deselects 6 kaleido-deadlock tests — see prior QC notes),
   numeric checks, screenshots. Harden tests for every fix.
5. Open PR into dev-1.0-refactor with evidence-rich per-issue comments.

## Useful recovery commands
- Extract a triage report: `<venv>/python <scratchpad>/extract_report.py <task.output>`
- Notes notebook: `git show origin/qc/release-audit-2026-07:notes/qc-release-2026-07/hypertools_1.0_verification_notes.ipynb`
