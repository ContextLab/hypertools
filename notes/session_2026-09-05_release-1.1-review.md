# Session 2026-09-05 (late): 1.1.0 release review against v1.0.0

Branch: `fix/1.1-release-review` (PR #286 -> master; PR body is a full prior
review report; it "Closes #284, #285" on merge). Started at b3eafe5c.

## Done this session
- Ultrareview (cloud) of PR #286: ONE nit -- `from .common import Forecaster/Imputer`
  inside the scoring loops of `hypertools/predict/backtest.py` and
  `hypertools/impute/backtest.py`. Fixed (hoisted to module scope) in f116dd7e;
  `tests/test_predict_backtest.py tests/test_impute_backtest.py` 58 passed; ruff clean.
- Issues #284 and #285 confirmed OPEN; every checkbox ticked except #284 D:
  "CI green; v1.1.0 moved; tag CI green; draft release updated" (unchecked --
  it depends on the PR merge + release re-cut; see PR body's "disposition").

## In flight (9 read-only background verifiers, dispatched ~23:55)
1. #284 A+B (examples de-dup, native rewrites)
2. #284 C+D (tutorials, gate, release state) + "found while fixing"
3. #285 Bugs/Plotting/Animation items
4. #285 Text/Data/Forecasting/Adjacent/Convert-now items
5. Documentation review (CHANGELOG vs API diff, docstrings, docs/*.rst, links, sphinx -W build, notebooks)
6. Code review: hypertools/plot/ diff v1.0.0..HEAD
7. Code review: io/tools/manip/_shared diff
8. Code review: predict/impute/align/reduce/cluster/core diff
9. Packaging/CI/tests/scripts review + wheel smoke test

## Next
- Collect findings -> fix every confirmed defect (dispatch fixers; tests real, no mocks)
- Re-run ALL checks after fixes (full pytest, ruff, sphinx -W)
- Commit, push branch, wait for PR CI, then: Jeremy merges PR #286; issues close on merge
  (or close manually with a closing comment if the PR is merged without the keywords).
- After merge: release re-cut from the merge commit (gallery manifest, wheel/sdist,
  move v1.1.0 tag, draft release assets) per RELEASE_CHECKLIST.md.

## Findings log (to fix once all verifiers report)
### From verifier 1 (#284 A+B): all VERIFIED except one PARTIAL
- P1 gate: tests/test_examples_are_native.py BUDGETS (:100-140) only scans 6 launch scripts + 6 notebooks, so the new DEFECT_MARKERS never scan plot_digits.py etc.; no `ax=` marker / PRIVATE_API_EXCEPTIONS={} (:186) -> the "per-file allowlist for deliberate ax= demos" does not exist.
- The `find_spec\(` marker would flag the legitimate Colab-install guard in 8 tutorials if widened.
- docs/tutorials/analyze.ipynb:338 + plot.ipynb:930 still present 'hyper'/'HyperAlign' as interchangeable ('hyper' is a DeprecationWarning alias, align.py:52-54).
- docs/tutorials/manip.ipynb never mentions Normalize(mode='isotropic').
- docs/conf.py:459 stale comment citing chemtrails/precog.
- conversation_shape/painting_embeddings install cells still `%pip install -q sentence-transformers` (redundant, harmless).
### From verifier 2 (#284 C+D): VERIFIED except
- align.ipynb "local image": the alignment illustration was DROPPED (73874cc8), not localized; docs/tutorials/img/alignment.png is unreferenced.
- Stored notebook outputs leak /Users/jmanning paths: plot.ipynb code cell 20 stderr (format_data.py:495 UserWarning), projectile_kalman.ipynb cell 1 (~/.hypertools_cache), conversation_trajectories.ipynb cell 2 (~/.convokit).
- EXPECTED_VISIBLE_OUTPUTS gate (test_examples_are_native.py:1130-1151) covers only 5 launch notebooks + animate_forecast; the 8 rebuilt tutorials (hierarchy/io/pipelines/manip/plot/align/analyze/reduce) have no output-cell entry.
- Release state: master == v1.1.0 == 96ac8b7f (local+remote), master CI run 33967006516 all 17 green, tag run green, draft release exists (wheel+sdist). PR #286 runs on 569a3089/b3eafe5c were in_progress; f116dd7e unpushed.
- Test count at HEAD: 4904 collected (issue text says 4118).
### From verifier 9 (packaging/CI/tests): wheel+sdist clean, fresh-venv smoke OK, ruff clean, no secrets
- HIGH (process): tests/test_release_readiness_gate.py:409 fails with HYPERTOOLS_REQUIRE_RELEASE=1 on HEAD (manifest source_commit 96ac8b7f != HEAD). Expected until the re-cut: republish gallery from the final commit AND move the pushed v1.1.0 tag. RELEASE_CHECKLIST.md covers neither (line 5 says the tag does not exist yet).
- MED: RELEASE_CHECKLIST.md stale (lines 5, 14-21, 39-43 describe dev-1.0->master PR; "3700+ passed"); CHANGELOG 1.1.0 date 2026-09-04 predates post-tag commits; HYPERTOOLS_EXAMPLE_SMOKE=1 gate (test_examples_are_native.py:1251) is in no CI job and not on the checklist.
- MED (no-mocks rule): tests/predict/test_common.py:179 monkeypatch spy on _infer_step; tests/predict/test_predict_multiindex.py:202-203 wraps group_columns/group_rows_for_forecast; tests/test_names_display.py:96-175 replaces pio.show/go.Figure.show with counting lambdas and IPython.get_ipython with _FakeShell (new since 1.0).
- LOW: git diff --check fails: trailing whitespace in docs/_static/pipeline_order.svg, docs/hypertools.FrameContext.rst, docs/hypertools.io.LSLStream.rst, docs/superpowers/plans/2026-07-28-...md:163, notes/audit/review_plan3_v3.md:117, notes/audit/review_plan4_v2.md:61.
- LOW: scripts/generate_baseline_screenshots.py:9 references notes/hypertools_1.0_roadmap.md (missing); /tmp defaults in generate_marker_parity_evidence.py:32,39 and audit_gallery_backends.py:94.
- INFO: ipympl and numba declared core deps but never imported (numba pinned for umap on purpose).
### From verifier 3 (#285 bugs/plotting/animation): ALL VERIFIED (367 focused tests pass)
- Doc nit: hyp.plot `font=` docstring says nothing about weights / the bundled NotoSans-Bold face (fonts.py:58-65 + CHANGELOG do).
- Doc nit: HyperAnimation.drawn_extent documented only on the class (hyper_animation.py:140), not referenced from the hyp.plot docstring.
- Pre-existing quirk: low-level hypertools.tools.text2mat.text2mat(list_of_3_strings, ...) returns [(3,d),(0,d),(0,d)] for CountVectorizer and HF vectorizers (hyp.plot wraps correctly). Investigate.
### From verifier 8 (predict/impute/align/core): numerics + ownership + validation all clean; findings:
- LOW: predict/backtest.py:91-113 resolve_metrics accepts metrics=['mae','mae'] -> TypeError deep in build_scores (:276-278) for predict(holdout=) and impute(truth=). Fix: ValueError naming the duplicate.
- LOW: align/align.py:163-178 _compute_score: hyp.align([a(50,4), b(40,4)], return_score=True) raises ValueError telling the user to run hyp.align; plain hyp.align works. Fix: score the trimmed before-data (`raw` already equal shape); document.
- COSMETIC: predict/backtest.py:332-333 `holdout=True, t=0` message says "got True" (should name t).
- NOTE: backtest.py:264 `unscored` warning uses default stacklevel (others use external_stacklevel()).
### Own check: hypertools.tools.text2mat (PUBLIC: docs/api.rst:202, tools/__init__ __all__) with a flat list of N strings returns [(N,d),(0,d),(0,d)...]: _transform splits at len(str) (character counts). Pre-existing in v1.0.0 (same _transform). Fix: treat a flat list of str as ONE dataset (wrap) + test.
### From verifier 4 (#285 text/data/forecast/adjacent): VERIFIED (272 focused tests pass); findings:
- BUG (new in 1.1; 1.0 raised NotImplementedError for animate+predict): hyp.plot(hyp.load('random_walk', n_samples=100, n_features=3, random_state=0), predict='ARIMA', t=10, animate=True, show=False) -> IndexError from statsmodels via plot/forecast.py:390 -> predict/arima.py:111 (first revealed history has 2 rows, DEFAULT_MIN_HISTORY=2; ARIMA order (1,1,1) needs >=3). Existing test dodges it with duration=1, frame_rate=4.
- hyp.load('lorenz', streaming=True) silently returns the full array (streaming= documented HF-only); `dim=` TypeError. Proposal's streaming generator not implemented (fine) but the silent ignore should warn/raise.
- `baseline=` is not a kwarg on predict(holdout=) (naive row is always present, per CHANGELOG): passing it -> TypeError from the model ctor. Acceptable API; note in the issue closure.
- HypertoolsOfflineError lives only at hypertools.io.sources: not exported from hyp / hyp.io, not in docs/api.rst (the other three exceptions are).
- text.ipynb code cell 5 (issue "cell 6") hue half not converted (hue=hue, labels=hue over dog+cat+bball) -- issue said convert the hue half to legend=.
- stack signature: stack(frames, names=None, level_names=None, aggregate=None) (issue spelled level names as names=).
- synthetic_outlet returns SyntheticOutlet (.stop/.thread/.closed), not the proposed tuple. Fine.
### From plot sub-reviewer (animation ctx/morph/fade/companion): all reproduced
- MED: dataset_fade= (and any on_frame artist mutation) silent no-op with continuous hue= on matplotlib: plot.py:1969-1993 _make_dataset_fade_updater; record(artists=...) sites matplotlib_backend.py:1609/1808/2543/2610 hold the HIDDEN Line2D/Line3D heads (_apply_multicolor_animation plot.py:11034 draws LineCollections). Pixels identical with/without fade. Plotly correct. Fix: expose the LineCollections in ctx.artists (and fade them) or raise.
- MED: loop=True + per-segment rotations= list can never validate: plot.py:8229 checks len vs sum(morph_tags) (3 clouds -> 5) but matplotlib_backend.py:2450-2453 re-validates on the looped list (-> 7). Both 5 and 7 entries rejected.
- MED: companion= head jumps backwards under order='serial' with several datasets: plot.py:1841-1847 _title_head_row rescales the CURRENT dataset's reveal onto the panel's rows (head x 0->20->10->0->39). Same rule drives {index} titles.
- LOW: window_bounds.start always 0 on serial reveals when a trail flag moved the head (matplotlib_backend.py:1816, 2617 hardcode (0,c)); docstring animation_context.py:198-205 promises >0.
- LOW: raw internal errors: companion=lambda -> TypeError 'function' not iterable (plot.py:2021); companion={'data':..,'smooth':'a'} -> int() ValueError (plot.py:2080); dataset_fade=('a','b') -> float() ValueError (plot.py:1929).
- LOW: raising on_frame during .save('x.gif') surfaces as IndexError from _RealTimePillowWriter.finish (animate.py:96).
- LOW: title= callable raising inside plot() -> orphaned 'Animation was deleted without rendering anything' warning on gc.
### From plot sub-reviewer (titles/fonts/labels): all reproduced
- MED: title_wrap ignored for dynamic (callable/pattern) titles under animation (plot.py:9139-9145 applies str(_fn(ctx)) unwrapped); docstring 3628 promises wrapping.
- MED: title_wrap destroys explicit newlines (plot.py:1506-1519 textwrap.wrap default replace_whitespace): 'a\nb' -> 'a b'.
- MED: plotly renders '\n' titles on one line (plotly_backend.py:1846 passes raw text); docstring 3560 promises identical rendering. title_wrap does convert to <br>; explicit newlines are not.
- LOW: nested-TUPLE labels= drawn as literal tuple text (validator plot.py:775 accepts tuples; flatteners matplotlib_backend.py:1222,1235 / plotly_backend.py:323 test only list).
- LOW: static '{index:%B %Y}' title with no index drawn verbatim (plot.py:1798-1801); docstring 3574 promises ValueError.
- DOC: plot.py:3633 stale ("animated 3-D title still reserves top margin for ONE line") -- matplotlib reserves per line now; plotly stays margin.t=40 (undocumented parity gap; 3609 "size sizes the probe" untrue on plotly).
- LOW: dynamic 3-D title that grows lines after frame 0 clips (plot.py:9817-9827 measures frame 0 only).
- LOW list: callable returning None/int drawn as 'None'/'42' (9140); serial list [1,None] -> 'None' (866); title_color='blue' silently loses to title_kwargs color (5799); label_anchor='bogus' with labels=None accepted (2385); labels='only' str counts chars; pattern format errors surface as bare KeyError/ValueError never naming title=.
### From plot sub-reviewer (panels/axis_scale/series/truth/predict): all reproduced
- HIGH: predict='ARIMA' + time-progressing animate= crashes (plot.py:8835 -> forecast.py:390 -> arima.py:111; 2-row history). Also predict=['Kalman','ARIMA'], animate=True. (Same as verifier 4's bug.)
- MED: t= datetime-like never works in plot() (plot.py:7005 hands _predictor bare ndarrays): ValueError "got t=Timestamp on a RangeIndex"; docstring says int or datetime-like.
- MED: panels= + truth= impossible: shared probe (plot.py:2588) sets predict=None but keeps truth; panel_fit='independent' never narrows truth list.
- MED: panels= shared + ndims=1 loses DataFrame index/column names; 3-col frames with reduce=None raise "zlabel= is not supported for 2-D data" (plot.py:2633-2641).
- MED: panels= + nested per-dataset hue=[arr30,arr40]: shared -> "hue has 1 entry but 30 observations" (plot.py:2612-2615); independent -> hue/labels not narrowed (plot.py:2603-2610).
- LOW: panels= + ndims>3: mpl ValueError ndims must be 1,2,3 (plot.py:2192); plotly scene/scatter error (plot.py:2721). Docstring says ndims>3 draws in 3-D.
- LOW: panel path skips _normalize_save_path: save_path='~/x.png' FileNotFoundError (plot.py:2665); pathlib.Path under plotly AttributeError (plot.py:2755).
- LOW: ndims=1 multi-column fmt=['-',':','--'] rejected (plot.py:6551 checks len(raw) before series expansion).
- LOW: return_model in series mode: predict['forecasts'] is per-COLUMN [x,value] arrays; docstring says one per input dataset matching hyp.predict.
- LOW: xlim on a date axis (ndims=1): floats = day numbers on mpl vs epoch-ms on plotly; strings work on plotly, TypeError on mpl. Document.
- LOW: forecast_hue=['g1','g2'] with predict=['Kalman','ARIMA'] on 2 datasets -> "got 2 for 4 forecast(s)"; docstring unclear.
- LOW: ylabel auto-set to 'dataset 1' when default reduce collapses a named 3-col frame (plot.py:6864-6866).
- Unconfirmed: hyp.subplots() 3-D axes accept plot(ndims=2, ax=) silently; TimedeltaIndex under ndims=1 draws raw ns.
### From verifier 7 (io/tools/manip): all reproduced
- HIGH: offline=True is not offline: load.py:652 calls seaborn_dataset(dataset) for every non-builtin string BEFORE load_source looks at the cache; sources.py:264-288 sns.get_dataset_names() urlopen with no timeout, cache stays None on failure (:283) so it retries every call. Blackholed proxy: 75 s block; proxy log shows CONNECT raw.githubusercontent.com on every offline=True call. Fix: honour offline/cache before any network probe (check URL-cache first; skip seaborn probe for URLs; timeout + remember failure).
- MED: yahoo: bars dated one day early east of UTC (sources.py:1134 normalises UTC timestamp; BHP.AX raw 23:00 UTC). Fix: add meta['gmtoffset'] before normalize().
- MED: synthetic random_state=np.random.RandomState(0) crashes numpy-native datasets (sources.py:542 .bit_generator); docstrings list RandomState.
- MED: synthetic 'blobs'/'moons'/... with Generator/SeedSequence and n_datasets==1 -> sklearn InvalidParameterError (sources.py:819 passes through); works for n_datasets=2 (ints derived).
- LOW: text2mat.py:383 isinstance(semantic, str) guard: gensim vectorizer + dict-spec semantic {'model':'NMF',...} bypasses skip/warn -> raw sklearn NMF ValueError.
- LOW: reusing one SeedSequence for n_datasets>1 not reproducible (sources.py:830 spawns children, mutating it). Document or derive without mutating.
- LOW: n_datasets=2.7 silently truncated (sources.py:808).
- LOW: text_windows.py:39 rejects np.int64 size=.
- Unconfirmed/doc: Smooth(center=False) default min_periods inside hyp.plot(manip=) gives the misleading all-NaN-rows PPCA error; Manipulator.fit() returns None (sklearn convention self); sec dedupe across units lossy; wikipedia lang= unescaped into host; load() docstring sec kwargs omit taxonomy/unit/dedupe and "cache/offline cover steps 9-13" but HF (step 9) is not cached.
### From plot sub-reviewer (hue/palette/legend/colors): all reproduced
- HIGH (regression vs 1.0): hyp.plot([a,b,c], palette=['red','blue']) with no hue raises "palette= supplies 2 color(s) but 3 are required" (plot.py:10325 _build_colors_info -> dataset_colors -> colors.py:715); v1.0.0 cycled red/blue/red on both backends. Fix: cycle.
- MED: palette=[] (or ()/np.array([])) with categorical hue escapes as bare StopIteration (plot.py:539 -> sns.color_palette([], n)).
- MED: per-dataset palette list + categorical hue / n_clusters: wrong count in the error (plot.py:536 _seaborn_palette_arg(palette, len(drawn)): says 2 datasets when 3 passed, 26 with n_clusters=3); the docstring's per-entry {category: color} dict form is unusable.
- MED (pre-existing): NaN in continuous hue poisons vmin/vmax (plot.py:10180 np.min over non-finite) -> bundle['colors'] vmin/vmax NaN, colorbar spans -0.1..0.1; docstring says NaNs excluded.
- LOW: legend_kwargs={'fontsize':N} ignored whenever font= set (matplotlib_backend.py:311-317 prop=font beats fontsize).
- LOW: bundle['colors']['categories'] not RGB for blend kind with legend_colors (plot.py:10318).
- LOW: nested hue with one mismatched sub-list -> misleading "hue has 3 entries but 36 observations" (plot.py:7794).
- LOW doc: hue docstring (~3340) says integer ids legend-labeled in sorted order; line fmt draws first-appearance order, fmt='o' sorts.
