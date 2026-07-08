# HyperTools 1.0 — QC-notes fix effort

Branch: `fix/qc-notes-2026-07` (off `dev-1.0-refactor` @ 724ad5f0).
Goal: open a NEW PR into `dev-1.0-refactor` with evidence-rich comments per issue.
Constraints: real data (dozens+ obs), numeric + screenshot verification, red-team
subagent per fix. NEVER merge; NEVER touch master. Co-Authored-By: Claude Opus 4.8 (1M context).

Source of issues: `notes/qc-release-2026-07/hypertools_1.0_verification_notes.ipynb`
(on origin/qc/release-audit-2026-07). Extracted to scratchpad/notes_extracted.txt.

## Issue catalog (verbatim signal → expanded scope)

### A. Animation system  [triage: report_animation.md]
- A1. `hyp.plot(..., animate='spin', show=False)` returns matplotlib **Figure**, no
  `.to_html5_video()` → "pure failure" for spin/window/morph/parallel. CONFIRMED.
- A2. `animate='chemtrails'` renders as 'window'. CONFIRMED returns static Figure.
- A3. Per-point `labels=` show in EVERY animation frame; should show only when the
  datapoint is visible.
- Pattern: whole animation return-contract; all styles; 2D vs 3D; save/gif; plotly.

### B. Hue / color / surface / soft-cluster  [triage: report_hue_color.md]
- B1. (n,2) GaussianMixture proportion matrix as `hue=` does NOT blend per-point
  colors (0.5/0.5 points not blended). Likely argmax hard-assignment.
- B2. `surface=True` ignores `hue=`.
- B3. Feature: arbitrary (n,k) matrix hue → reduce to 3 dims (r,g,b) via
  IncrementalPCA by default; `color_reduce=` to customize the reduction.
- Pattern: matrix-hue in line vs scatter, mpl vs plotly, colorbar interaction.

### C. Fitted-model reuse  [triage: report_model_reuse.md]
- C1. `hyp.cluster(new, cluster=fitted_pipeline, reduce=, manip=)` →
  `AttributeError: 'Pipeline' object has no attribute 'labels_'` at
  cluster/common.py:196. CONFIRMED.
- Pattern: reuse across reduce/cluster/align/manip/normalize, bare estimator vs
  hyp Pipeline (from return_model=True), + cross-module kwargs.

### D. Dependencies / packaging  [triage: report_deps.md]
- D1. `pykalman` missing → predict(model='Kalman') ModuleNotFoundError. Add dep.
- D2. Plotly 5.24 vs Kaleido 1.3 conflict → static image export broken. Pin.
- Pattern: every undeclared external import across predict/impute/reduce/align.

### E. names= + labels/legend + display  [triage: report_names_display.md]
- E1. New `names=` kwarg: per-DATASET names, distinct from per-point `labels` and
  `group`. Current `labels=[...]` on list-of-datasets behaves "strangely".
- E2. Double-display in Jupyter (returned Figure shown twice); plotly shows before
  matplotlib. Want figure object access without double display.

### F. describe styling + top-level API  [triage: report_describe_api.md]
- F1. describe() plot: despine top/right; also support plotly backend.
- F2. Verify hyp.describe/predict/impute/reduce/cluster/manip/normalize/align/
  analyze/apply_model all callable at top level (already exported; confirm numeric).
  (Jeremy's "wire in" notes appear to be pip-cache artifacts — CONFIRM.)

## Verified-good (Jeremy said OK) — still re-verify numerically per instructions
- Core plot (sec 1), reduce (sec 2), autoencoders (sec 3), align (sec 4),
  Pipeline (sec 8), cross-module reduce(align=) (sec 9), text/Word2Vec (sec 13),
  loaders incl kaggle (sec 12). normalize reuse P0-1 (sec 7a).

## Status ledger
- 2026-07-08: branch created; 6 triage agents dispatched (anim, hue, reuse, deps,
  names/display, describe/api). Awaiting reports.
