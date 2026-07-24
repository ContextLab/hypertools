# Session 2026-07-17: zero-warnings sweep (release-1.0 audit)

Branch: audit/release-1.0-2026-07. Goal: full pytest run ends with ZERO
warnings without weakening any assertion. Baseline: 2331 passed, 154 warnings.

## Production fixes (animation GC — REAL gaps in the wave-6d fix)

The 29 "Animation was deleted without rendering anything" instances were NOT
attributable to the tests pytest blamed: the FuncAnimation is held in a
reference cycle by its own canvas callbacks, so it dies at the NEXT cyclic-gc
pass, and the warning lands on whatever test runs then. Two paths bypassed
HyperAnimation.__del__'s silencing entirely:

1. `return_model=True` bundles: plot.py returns the RAW FuncAnimation in
   `bundle['animation']` (asserted raw by
   tests/test_hyper_animation.py::test_return_model_bundle_animation_is_raw_animation),
   so the wrapper's __del__ never runs. FIX: plot.py now calls
   `hyper_animation.mark_draw_started(line_ani)` before returning the bundle.
2. save_path failure path: when `_save_animation` raises (bad extension), the
   exception propagates BEFORE the HyperAnimation is constructed; the
   abandoned animation warned at next gc. FIX: the except-branch in plot.py's
   save block marks the animation before re-raising.

New helper `mark_draw_started()` in hypertools/plot/hyper_animation.py
(refactored __del__ to use it; matplotlib only reads `_draw_was_started` in
`Animation.__del__`, and itself force-sets it in `save()`, so no rendering
side effects). Test-first: two new tests in tests/test_hyper_animation.py
(`test_return_model_bundle_animation_gc_does_not_warn`,
`test_failed_save_path_animation_gc_does_not_warn`) — both red before fix,
green after.

## Test upgrades (pytest.warns / catch_warnings asserts) — deliberate hypertools warnings
- align row-trim: tests/align/test_align_base.py, test_final_wave_audit_fixes
  (catch_warnings, both duplicated-row-index AND trim asserted)
- align= kwarg + 'hyper' alias + params-dict deprecations: test_rsrm,
  test_isinstance_209 (x2), test_round3 (3 warnings one call),
  test_align_migration (2, conditional on param), test_analyze_audit_fixes,
  test_plot (3), test_cluster, test_cluster_migration, test_load,
  core/test_model, test_apply_model (4), test_reduce (TSNE, UMAP)
- PPCA missing-data/cannot-fill: test_format_data, impute/
  test_format_data_integration (2), test_format_data_f08_fixes (nested),
  test_input_coercion_hardening (scattered-nan), test_manip_impute_hardening
- impute no-observed-values: test_impute_audit_fixes (warns+raises),
  test_manip_impute_hardening (parametrized)
- smooth kernel width: test_manip_impute_hardening (warns+raises)
- scalar hue: test_input_coercion_hardening (parametrized),
  test_plot_audit_b2 (warns+raises TypeError)
- Unequal dims/n_components: test_plot_audit_b1, test_plot_f08_fixes (2nd
  call), test_reduce (UMAP nested with params-dict dep)
- streaming outside-display-box: test_streaming (11), test_io_audit_streaming
  (2, one nested with streaming-stopped-early), test_d1_code_residue,
  test_load_sources (HF stream), test_lsl_streaming (ramp data => guaranteed)
- remote unpickle trust notice (confirmed intentional at sources.py:1124,
  remote=True without trust): test_load_sources drive + dropbox tests
- single-observation reduce notice: test_load_sources::test_plot_accepts_source_strings
- mixed text/numeric: test_format_data (mixed_list, force_align)
- reduce-dispatch notices: test_reduce_dispatch_hardening (2, one conditional)
- window trail flags: test_window_animation

## Narrow filterwarnings marks (upstream, per-test, message+category scoped)
- Pandas4Warning "The copy keyword is deprecated" (datawrangler pd.concat):
  13 tests across core/test_dw_probe, manip/test_axis1,
  manip/test_normalize_zscore, test_fitted_model_reuse,
  test_manip_audit_fixes, test_manip_chaining, test_pipeline
- sklearn GP ConvergenceWarning (noise-level bound): 29 tests across
  predict/test_gp (6), predict/test_dispatcher (9),
  plot/test_predict_integration (1), test_edge_cases_hardening (3),
  test_final_wave_audit_fixes (2), test_predict_audit_fixes (9)
  + lbfgs variant on 2 of those
- FastICA / SpectralEmbedding / UMAP n_jobs / NMF max_iter /
  IterativeImputer early stopping: one test each

FINAL full-suite result (MPLBACKEND=Agg .venv/bin/python -m pytest -q):
`2333 passed, 4 skipped, 2 deselected in 1017.69s (0:16:57)` — NO warnings
summary block at all (baseline was 2331 passed + 154 warnings; +2 = the two
new animation-GC regression tests). Nothing committed — working tree only.
