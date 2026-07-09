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

## CROSS-CUTTING ROBUSTNESS
- X1 == R2 (reduce list/ndarray return flip).
- X2 [CRYPTIC] list-of-numeric-lists crashes reduce/normalize/cluster/describe (format_data.py:166 .ndim on list).
- X3 == P5 (all-NaN -> zero-size reduction @ _externals/ppca.py:65).
- X4 [ERROR-QUALITY] align() unknown model name -> AttributeError not clear ValueError (align.py:215).
- X5 == P3 (flat list of numbers rejected though "supported"; Series/tuple too).
- X6 [CRYPTIC] string ndims -> TypeError (reduce.py:228). No "ndims must be int".
- X7 [LOW] predict(t=-3) silent empty (0,n); predict(one_row) cryptic.
- X8 [LOW] normalize propagates inf silently.
- X9 [reproducibility] no top-level random_state (dict-kwargs workaround works). LOW.
- CLEAN: no hypertools-own numpy2/pandas3 Future/Deprecation warnings; no hangs; 5000x500 <1s.

## ANIMATION DEEP-DIVE
- A1 [MED] HyperAnimation.save('x.svg'/'x.png'/'x.apng') crashes (delegates to mpl Animation.save,
  bypasses _save_animation ext dispatcher). gif/mp4 + save_path='x.svg' WORK. Asymmetric.
- A2 [MED] duration=0 / frame_rate=0 -> ZeroDivisionError (plot.py:2264/2277, helpers.py:139).
- A3 [LOW] duration<0 -> cryptic ValueError.
- A4 [LOW/extreme] zoom=3 clips cube (cosmetic, user extreme value).
- K2 RESOLVED = FALSE POSITIVE: margin test helper _cube_corner_pixels ignores set_box_aspect(1.125)
  view scaling -> phantom off-canvas corners. Actual render never clipped (118px margin/30 frames;
  frames in hunt/A_wide_chemtrails_f0.png). FIX THE TEST helper, not source.

## IO / TEXT
- I1 [HIGH-silent-wrong] canonical reduce={'model':...,'kwargs':{...}} + ndims DROPS the reduction:
  hyp.load('iris', reduce={'model':'PCA','kwargs':{'whiten':True}}, ndims=2) -> (150,5) not (150,2).
  Bare-string + legacy 'params' forms correct. reduce.py canonical branch pre-builds instance, ignores
  n_components=ndims. Hits reduce/analyze/load. THE documented dict form.
- I2 [MED] streaming plot ignores canonical reduce kwargs (streaming.py:133 reads only 'params').
- I3 [MED] text2mat(flat list of strings) -> mostly empty (uses len(string) char count). Not via plot.
- I4 [LOW] text2mat ragged list-of-lists crashes (vstack). Not via plot.
- I5 [LOW] text2mat([]) crashes.
- I6 [LOW-doc] load() docstring: 'weights' "list of 2 arrays" but returns 36 (meant weights_avg?).
- I7 [minor] corrupt .parquet raw ArrowInvalid not wrapped.
- IO CLEAN: all local formats round-trip; save/load sniff; network loads correct shapes; LSL no-hang;
  legacy .geo friendly; gensim all vectorizers/semantic; unicode.

## FIX BATCHES (prioritize silent-wrong > crash > cryptic > UX)
B1 reduce dispatch: I1 (dict+ndims), R2/X1 (return type), R3, X6 (str ndims), R6 (ndims=0).
B2 input coercion + error quality: P2 (1D), P3/X5 (flat list), X2 (list-of-lists), P1 (hue len),
   X4 (align unknown model), P4/P5/X3 (NaN/inf clear errors).
B3 manip/normalize/impute: M1 (all-missing col), M2 (constant col NaN), M3 (PPCA shape), M4 (manip
   stage kwargs), M5 (Smooth kw errors), M6.
B4 pipeline/analyze/apply_model: C1 (analyze cluster= labels), C2 (normalize=False), C3 (hard-clusterer),
   C4 (apply_model ndims).
B5 align: R1 (mismatched cols), R7 (mismatched rows warn).
B6 animation: A1 (save formats), A2/A3 (duration/frame_rate<=0), K2 (fix margin test helper).
B7 text/streaming/doc: I2, I3, I4, I5, I6, I7.
B8 misc: X7 (predict t<0), X8 (inf), X9 (random_state - defer/decide).
Each batch: fix -> verify (numeric/screenshot) -> independent red-team subagent -> commit -> PR evidence.
Plotly agent (afd46db7) still running; fold its findings into B2/plotly parity.
