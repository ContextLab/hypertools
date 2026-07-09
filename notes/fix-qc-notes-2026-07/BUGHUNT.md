# Release-hardening bug hunt (2026-07-09) — /goal: fix ALL errors

8 parallel read-only hunt agents + 2 known issues. Confirmed, reproduced bugs below.
Fix each: verify (screenshot/numeric) → independent red-team → PR #280 evidence comment.
NEVER merge; base dev-1.0-refactor.

## KNOWN (pre-flagged)
- K1 [DONE-fix pending commit] plotly backend + animation crash: set_interactive_backend('plotly')
  + animate -> HypertoolsBackendError (tried to switch mpl to 'plotly'). FIXED: render-backend
  preference (PREFERRED_RENDER_BACKEND) in backend.py; set_interactive_backend('plotly'/'matplotlib')
  routes there; resolve_backend consults it; plot_wrapper skips mpl switch for plotly. Also fixes the
  deeper "set_interactive_backend('plotly') doesn't switch renderer" inconsistency. VERIFIED (6 checks).
- K2 [TODO] margin test: wide/flat chemtrails cube corner projects off-canvas (y~-312 on 640px) frame 0.
  test_animation_margins.py::...wide_chemtrails_cube_corners. Real clip vs too-strict test? (anim agent investigating)

## CORE / PIPELINE / ANALYZE
- C1 [HIGH] analyze(cluster=...) returns cluster LABELS not transformed data (contradicts analyze.py
  docstring L70-78). Cross-module reduce/normalize/align(cluster=) same (last pipeline step=cluster -> labels).
- C2 [HIGH] normalize=False + other stage + return_model -> Pipeline.transform crashes NotFittedError.
  build_pipeline treats False as real spec; normalize False returns (x,None) -> _DispatchStep._fitted=None.
- C3 [MED-HIGH] Pipeline/manip list w/ hard-clusterers (DBSCAN/Agglomerative/OPTICS/MeanShift) crash
  (no fit_predict fallback @ pipeline.py:295); KMeans-in-Pipeline returns distance matrix not labels (silent).
- C4 [MED] apply_model(model='KMeans', ndims=3) crashes (n_components forced on KMeans @ model.py:181-183);
  string path vs instance path inconsistent (instance guards hasattr n_components).

## REDUCE / CLUSTER / ALIGN / DESCRIBE
- R1 [HIGH] hyp.align crashes by DEFAULT on datasets w/ different column counts (format_data.py:179
  vstack NaN-check). Aligners support it (format_data=False works). Canonical hyperalignment use case.
- R2 [MED-HIGH] hyp.reduce returns a LIST for a single array when ndims None or >= n_features
  (reduce.py:228-229). ndims=3/reduce=None return bare ndarray. Inconsistent return type.
- R3 [MED] describe(max_dims > n_features) crashes (cdist on list; downstream of R2).
- R4 [LOW] describe(show=True) crashes when component range empty (max_dims=2 -> range(2,2)).
- R5 [minor] FeatureAgglomeration returns per-FEATURE labels (wrong axis, breaks plot coloring).
- R6 [minor] reduce(ndims=0) silently returns (n,0) instead of error.
- R7 [minor] align silently trims longer dataset on mismatched ROW counts ([50,40]->[40,40]) no warning.

## MANIP / NORMALIZE / PREDICT / IMPUTE
- M1 [CRASH] impute all-missing column (SimpleImputer/KNN/Iterative) IndexError @ sklearn_imputers.py:25
  (_splice mask mismatch; sklearn drops all-NaN col). Kalman/PPCA handle it.
- M2 [WRONG-NUMBER] manip ZScore/Normalize divide-by-zero on constant col -> whole col NaN, no warning
  (zscore.py:69, normalize.py:83). tools.normalize._zscore_column guards this -> inconsistency.
- M3 [WRONG-SHAPE] impute PPCA on rank-deficient data returns FEWER columns (violates "preserves shape").
- M4 [UX] manip() lacks cross-module stage kwargs (reduce/align/cluster -> constructor TypeError).
  normalize()/analyze() have them.
- M5 [UX] Smooth kernel_width edge cases surface raw scipy errors (window_length/polyorder).
- M6 [minor] normalize(format_data=False): single NaN poisons column mean/std.

## PLOT MATPLOTLIB
- P1 [HIGH] hue length never validated -> silent truncation (renders subset, no warning) OR IndexError
  (helpers.py:270 reshape_data zip; plot.py:2049). Other per-obs kwargs validate.
- P2 [MED-HIGH] 1-D ndarray crashes: hyp.plot(np.random.rand(50)) IndexError @ helpers.py:389 (data[0][0]).
- P3 [MED] flat list of numbers crashes though advertised: hyp.plot([1.,2.,3.]) (format_data.py:91 maps
  get_type over each element; list_num branch dead).
- P4 [MED] NaN inconsistent: isolated NaN interpolated silently; whole-NaN-row crashes IncrementalPCA.
- P5 [LOW] all-NaN/inf -> ugly crashes (zero-size reduction / infinity).

## PENDING agent reports: io/text (a28bc3f4), plotly (afd46db7), animation (a89e41d2), cross-cutting (a62bb1b9)
