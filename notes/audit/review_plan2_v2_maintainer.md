# Maintainer review of the MultiIndex plan (v2) — 2026-07-28

Verbatim. 24 findings; findings 1-5 and 17-24 called required before implementation.

---

The v2 plan is materially better and correctly addresses the first review. However, it is not yet implementation-ready. The largest
  remaining issues are the FinalTraces integration, misuse of the v1.0 return contract, package layering, datetime forecasting, and the near-
  total absence of tutorial/example updates.

  ## Blocking findings

  ### 1. FinalTraces duplicates existing mean construction

  The plan says FinalTraces.from_hierarchy() will append all hierarchy means:

  > “building leaves … followed by one mean per non-leaf level”

  But build_multiindex_styles() currently does two jobs:

  1. appends those same mean arrays;
  2. constructs their styles.

  See hypertools/plot/multiindex.py:91, especially the mean construction at lines 197–220.

  Later tasks still say to call build_multiindex_styles() after constructing FinalTraces. Unless that function is refactored, this will
  either:

  - append every mean twice, or
  - fail because it receives final traces where it expects leaves only.

  The plan needs an explicit ownership split. A clean design would be:

  expand/group input
    -> analyze leaf arrays
    -> build_hierarchy_traces(leaves, metadata, auxiliary values)
    -> build_hierarchy_styles(final trace metadata)
    -> forecasts/rendering

  Recommended changes:

  - Move mean construction out of build_multiindex_styles().
  - Rename/refactor it into a style-only function accepting FinalTraces.level_idx and keys.
  - Ensure only one helper owns unequal-length truncation and warning generation.
  - Add a regression test asserting that every expected mean appears exactly once.

  This is the most immediate implementation blocker.

  ### 2. The plan uses a nonexistent return_data=True API

  Several proposed tests call:

  hyp.plot(..., return_data=True)

  There is no return_data parameter. The v1.0 redesign deliberately exposes transformed data through return_model=True; see the signature at
  hypertools/plot/plot.py:517 and the return bundle at hypertools/plot/plot.py:4516.

  Because plot() accepts **kwargs, return_data=True may not even fail cleanly—it could leak into backend arguments.

  Every such test should use:

  out = hyp.plot(..., return_model=True, show=False)

  The plan should not introduce return_data unless it deliberately proposes a new public API, documents it, validates it, and updates all
  relevant tests.

  ### 3. Redefining xform_data as “every drawn trace” conflicts with v1.0 semantics

  The plan’s contract says:

  > One final-trace list governs ordering for xform_data, hue, forecasts, styles and artists.

  But v1.0 defines xform_data as analyzed output from the canonical pipeline, captured immediately after analyze():

  - capture: hypertools/plot/plot.py:2827
  - documentation: hypertools/plot/plot.py:1920

  Hierarchy means are derived presentation traces created afterward in reduced/plotted space. Putting them into xform_data creates several
  problems:

  - The returned fitted pipeline cannot reproduce those derived means by calling .transform().
  - xform_data no longer represents only pipeline output.
  - Hierarchical and ordinary plots acquire different interpretations of the same bundle field.
  - Existing code may assume one xform_data entry per analyzed input dataset.
  - The plan also builds means after the final display-dimensionality reduction, while xform_data is currently captured before that display-
    only adjustment.

  A better contract is probably:

  {
      "xform_data": analyzed_leaf_data,
      "trace_data": final_drawn_trace_data,
      "trace_metadata": ...,
      "predict": {
          "forecasts": final_trace_forecasts,
          ...
      }
  }

  Or keep trace_data internal and document that hierarchical forecast arrays correspond to drawn traces rather than xform_data. Whichever
  design is chosen must be reconciled with the established promise that bundled forecasts match hyp.predict(xform_data, ...).

  ### 4. Shared hierarchy code should not live under hypertools.plot

  The proposed hypertools/plot/hierarchy.py is described as axis-agnostic infrastructure used by both plot and predict. Task 5 then imports
  plotting internals from the prediction package:

  from ..plot.hierarchy import ...

  That cuts against the v1.0 redesign, which split the old toolbox into focused packages with shared machinery under core/ and _shared/; see
  the architecture section in CHANGELOG.md:58.

  Prediction should not depend on plotting.

  Recommended placement:

  - hypertools/core/hierarchy.py if it is genuine cross-dispatch infrastructure; or
  - a small shared grouping module under hypertools/_shared/;
  - keep rendering-specific final trace/style objects under hypertools/plot/.

  In particular, group_rows_for_forecast() has nothing to do with plotting and should not be located in hypertools.plot.

  ### 5. Row forecasting loses its time index

  group_rows_for_forecast() currently proposes:

  groups.append(sub.reset_index(drop=True))

  That discards the innermost time/date level. It breaks the existing hyp.predict contract for datetime-like horizons, where the data must
  retain a datetime index; see hypertools/predict/predict.py:262.

  The proposed datetime test only checks grouping shape. It never calls:

  hyp.predict(row_multiindex_df, t=<future Timestamp>)

  Recommended implementation:

  - group by the outer levels;
  - drop only those grouping levels;
  - retain the innermost level as the resulting DataFrame index;
  - validate that it is ordered and unique as required by the forecaster.

  Add tests for:

  - an integer horizon on (Sector, day);
  - a future Timestamp horizon on (Sector, date);
  - an at-or-before-last timestamp truncation;
  - unsorted time levels;
  - duplicate times within a group.

  ### 6. Missing-value labels can silently drop hierarchy groups

  Both proposed grouping functions use pandas groupby() without dropna=False:

  df.T.groupby(level=group_levels, sort=False)
  df.groupby(level=group_levels, sort=False)

  Pandas normally drops NA grouping keys. A column or row whose hierarchy label is missing can disappear silently.

  The proposed “NaN columns” test sets a data value to NaN; it does not put NaN in a MultiIndex level and therefore does not test this.

  Recommendation:

  - use dropna=False, if its MultiIndex behavior is reliable on all supported pandas versions; or
  - use explicit positional first-appearance grouping.
  - Add tests with missing values in outer and intermediate hierarchy labels.

  ## Public API and compatibility concerns

  ### 7. “Additive only” conflicts with dual-axis rejection

  A frame with both row and column MultiIndexes currently follows the row-MultiIndex path while its column hierarchy is ignored. The plan
  changes this to a ValueError.

  That is arguably the right behavior, but it is not additive. It turns previously accepted input into rejected input.

  The plan should call this an intentional compatibility change and include it under a “Changed behavior” or “Breaking/validation changes”
  changelog heading.

  ### 8. Row plotting and row forecasting have surprising divergent semantics

  The revised plan correctly recognizes that:

  - row plotting groups by the full tuple;
  - row forecasting drops the innermost level as time.

  But this divergence is likely to confuse users. A (Sector, day) frame would produce:

  - 2 × T one-row plotting leaves;
  - two forecasting groups.

  That means the same frame is “first-class” in both functions but interpreted very differently.

  This may be unavoidable for backward compatibility, but it needs highly visible documentation—not only docstrings. A small comparison table
  would help:

   Axis and consumer                  Innermost level means    Grouping
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   Row MultiIndex, plot               part of leaf identity    full tuple
  ─────────────────────────────────  ───────────────────────  ──────────────────
   Row MultiIndex, predict            time/observation         all outer levels
  ─────────────────────────────────  ───────────────────────  ──────────────────
   Column MultiIndex, plot/predict    feature name             all outer levels

  Also consider whether new row-forecast behavior should require the innermost level to be demonstrably time-like or monotonic. Otherwise an
  arbitrary row MultiIndex such as (country, participant) will silently treat participant as time.

  ### 9. Constructed/fitted model semantics remain incomplete

  The plan says each group is fitted independently by deep-copying a Forecaster instance. That only guarantees independent fitting when the
  supplied instance is unfitted.

  A deep copy of an already-fitted model remains fitted, so each recursive call takes predict_new() at hypertools/predict/predict.py:216.
  That may be desirable because v1.0 explicitly promises fitted-model reuse, but it contradicts the blanket statement that every hierarchy
  group is “fitted independently.”

  The contract should distinguish:

  - name/class/dict/unfitted instance: fit one independent model per group;
  - fitted instance: reuse learned parameters independently on each group via cloned fitted instances.

  The proposed “shared unfitted instance” test does not actually pass a shared instance; it passes 'Kalman'. It should construct one:

  shared = Kalman(...)
  assert not shared.is_fitted
  forecasts, models = hyp.predict(df, model=shared, return_model=True)
  assert not shared.is_fitted

  Also test the fitted-instance path explicitly.

  ### 10. Hierarchical DataFrames inside lists remain undefined

  Both plot and predict accept lists of datasets. The plan handles only a bare DataFrame.

  Current row-MultiIndex plotting already warns when such a frame appears in a list. Column-MultiIndex frames would need equivalent handling.
  Prediction would silently funnel or flatten them unless explicitly checked.

  The plan should either:

  - support hierarchical DataFrames inside lists with flattened final ordering; or
  - reject/warn clearly and document bare-frame-only support.

  ### 11. Two-level column hierarchies need explicit styling semantics

  For columns (Group, Feature), meta["n_levels"] == 1. That produces one leaf per group and no aggregate mean level.

  Under the existing formulas those leaves have:

  - linewidth 1;
  - alpha 1;
  - no legend label, unless the styling logic is changed.

  That may yield multiple completely unlabeled traces, which is probably not desirable. The plan should specify whether a one-level grouping
  hierarchy treats each leaf as top-level and gives it a legend label, or faithfully preserves the existing “only top-level means receive
  labels” rule even though no mean exists.

  Add an end-to-end two-level column test covering trace count, color, opacity, linewidth, and legend.

  ## Hue design gaps

  ### 12. The three hue forms are ambiguous

  The plan accepts:

  1. flat length T;
  2. nested one vector per leaf;
  3. flat length equal to total drawn observations.

  For a one-leaf hierarchy, or other cases where T == total drawn observations, forms 1 and 3 are indistinguishable. More importantly, form 3
  includes values for derived mean traces—something users cannot naturally know until expansion has occurred.

  Form 3 is not really “existing behavior unchanged”: before this feature, there was no public final hierarchy trace list users could target.

  Recommendation:

  - Support input-relative forms only: shared row-wise or one vector per leaf.
  - Treat final-trace flattened values as an advanced/internal form only if there is a compelling use case.
  - If retained, define precedence and exact ordering in the public documentation.

  ### 13. Hue interpolation and mean construction need a defined order

  Unequal-length row-MultiIndex leaves are truncated to their overlapping prefix when means are computed. Hue arrays must follow exactly the
  same truncation and grouping logic. A separate implementation inside FinalTraces risks data/hue length drift.

  The plan should explicitly guarantee that data and auxiliary arrays are co-truncated by the same operation, followed by assert_consistent()
  checks for every trace.

  ### 14. Continuous hue plus forecasts is not fully specified

  The observed trajectory becomes a LineCollection with per-segment colors, while forecasts are drawn by _draw_forecast_overlays() as
  ordinary dashed lines. Forecast overlays currently inherit source colors positionally. For a continuously colored source there is no single
  obvious source color.

  The plan tests only that dashed forecasts exist. It needs to define whether a forecast:

  - uses the final observed hue color;
  - receives forecast hue values;
  - uses a fixed forecast color;
  - is multicolored according to extrapolated hue.

  This applies to both Matplotlib and Plotly.

  ## Plotly and animation

  ### 15. Task 7 is not actionable enough

  This is too open-ended:

  > implement whatever the plotly backend needs … or, if parity is deferred, raise NotImplementedError

  An implementation plan should decide which behavior is required. “Pass or deliberately fail” allows a major backend gap to be decided
  during implementation without product direction.

  The v1.0 redesign presents Matplotlib and Plotly as supported peer backends. Unless the release explicitly permits otherwise, parity should
  be required.

  The Plotly test is also too weak. It only checks a base column hierarchy. It needs coverage for:

  - exact leaf/mean trace count and order;
  - width/opacity/legend styling;
  - continuous price hue;
  - colorbar;
  - predict=;
  - hierarchy plus animated prediction, subject to the prerequisite forecast-animation plan;
  - return_model=True bundle correspondence;
  - backend state restoration through the existing fixture or helper rather than manual global mutation where possible.

  ### 16. The animation test relies on private internals

  The proposed test directly calls:

  ani._func(4, *ani._args)

  Existing tests may do this, but the prerequisite plan introduces on_frame/FrameHooks. If this work is meant to build on that redesign,
  tests should preferentially exercise the new stable hook or public animation interface.

  At minimum the plan should reconcile its test strategy with those prerequisite contracts.

  ## Documentation audit

  The documentation work is insufficient for the stated goal.

  ### 17. The market tutorial is not updated at all

  The goal says the market example should become almost entirely native HyperTools using:

  - sectors as leaves;
  - market aggregate;
  - price hue;
  - per-trace forecasts.

  But the current docs/tutorials/market_forecast.ipynb still:

  - downloads five unrelated FRED series;
  - reduces them into one market path;
  - hand-computes repeated forecasts;
  - manually converts reduce-space deltas to drawn coordinates;
  - manually creates forecast artists;
  - monkey-patches ani._func;
  - manually creates the colorbar and legend.

  It neither creates a column MultiIndex nor demonstrates any proposed feature.

  The plan must include a dedicated tutorial rewrite task. Otherwise the feature is implemented but its flagship use case remains old and
  hand-built.

  ### 18. The gallery example must also be rewritten

  examples/animate_market_forecast.py mirrors the old tutorial and contains the same manual forecast machinery.

  Because Sphinx-Gallery builds this source independently, updating only the notebook would leave conflicting documentation. The example
  should be rewritten in the same task and kept semantically aligned with the notebook.

  The plan should include:

  - Yahoo Finance input, as required by its own global constraint;
  - a cached/offline deterministic fallback;
  - construction of (Market, Sector, Ticker) columns;
  - sector mean price vectors for nested hue;
  - native hyp.plot(..., hue=..., predict=..., animate=...);
  - only example-layer additions still intentionally outside the API, such as accuracy scoring or a custom sector legend.

  ### 19. Yahoo Finance is promised but no docs task uses it

  The plan says Yahoo Finance is a maintainer decision. Existing market assets use FRED. No task replaces FRED with Yahoo or adds yfinance
  handling.

  This is a direct plan inconsistency.

  Also verify packaging policy: if yfinance is only a docs/example dependency, it should be represented in the docs/dev environment rather
  than the library runtime dependency set. The tutorial should degrade gracefully when it is unavailable.

  ### 20. Tutorial navigation and introductory docs need updates

  docs/tutorials.rst already includes the market notebook, but its current title and surrounding description reflect the old “one moving
  path” tutorial.

  The plan should update:

  - the section title and synopsis in tutorials.rst;
  - the feature list in docs/index.rst, which currently says only “DataFrames (including MultiIndex)” without distinguishing row and column
    semantics;

  - potentially the Plot and Predict sections of the API landing page with a short hierarchy note and link to the market tutorial.

  ### 21. pipeline_order.rst needs hierarchy placement

  The new feature adds operations outside the current pipeline diagram:

  - hierarchy expansion before format/analyze;
  - hierarchy mean construction after reduce/align;
  - hue propagation onto final traces;
  - forecasting after final trace construction.

  docs/pipeline_order.rst:19 currently shows:

  load/format -> manip -> normalize -> reduce -> align
  -> cluster (hue) -> plot/animate -> predict overlays

  The plan should update its prose and likely its SVG to explain hierarchy-specific ordering. Without that, the new architecture conflicts
  with the canonical-pipeline documentation.

  A hierarchy side branch would be clearer than pretending mean construction is an ordinary pipeline stage.

  ### 22. The API reference relies entirely on docstrings

  The autosummary pages will pick up expanded plot() and predict() docstrings, which is helpful but not sufficient for these nontrivial
  semantics.

  At minimum add a short “Hierarchical DataFrames” guide or section covering:

  - row versus column semantics;
  - plot versus predict divergence;
  - hue forms;
  - mean trace construction;
  - ragged limitation;
  - dual-axis rejection;
  - return shapes;
  - fitted model behavior;
  - backend parity.

  Then link it from api.rst, tutorials.rst, the relevant docstrings, and the market tutorial.

  ### 23. Changelog placement is wrong for the current tree

  The current changelog has 1.0.1 (unreleased), not 1.1.0. The plan says “add under 1.1.0” without first creating that release section.

  Add an explicit step to create:

  ## 1.1.0 (unreleased)

  It should describe:

  - new features;
  - validation/compatibility changes, including dual-axis rejection;
  - documented limitations;
  - any Plotly limitation if parity is not delivered.

  ### 24. Documentation validation needs more than make html

  Given notebook and gallery changes, the plan should run the repository’s existing publication checks, not only a Sphinx build. Relevant
  tests include:

  - tests/test_release_notebook_check.py
  - tests/test_publish_gallery_notebooks.py
  - tests/test_docs_thumbnails.py
  - tests/test_notebook_install_gate.py
  - tests/test_colab_install_cell.py

  It should also regenerate or validate the market gallery thumbnail if the visual changes. The existing docs have screenshot/parity and
  gallery auditing infrastructure that should be used.

  ## Test defects in the revised plan

  A few proposed tests still do not prove their names:

  - test_mean_trace_forecast_comes_from_the_mean_trajectory calculates direct but never compares it to the bundled or drawn forecast.
  - test_return_model_bundle_has_one_entry_per_trace checks only xform_data length, not models or forecast count.
  - test_nan_columns_do_not_silently_drop_a_group tests NaN data, not NaN hierarchy labels.
  - test_a_shared_unfitted_instance_is_not_mutated_across_groups does not pass an instance.
  - test_price_hue_actually_maps_low_to_high only checks that two colors differ; it does not check monotonic palette mapping or correct
    normalization.

  - The row-MultiIndex regression test asserts >= 6, which is too weak to pin the established exact leaf/mean count.
  - The Plotly test accepts >= 3, allowing duplicates and extra incorrect traces.
  - Task counts such as “11 passed” and “12 passed” should be rechecked against the actual number of test functions.

  ## Recommended revised task structure

  I would restructure v3 as follows:

  1. Shared hierarchy grouping in core/_shared, including time-index preservation and NA-label behavior.
  2. Refactor existing mean construction into one authoritative final-trace builder.
  3. Add a style-only hierarchy function; prevent duplicate means.
  4. Define the return bundle contract, preserving xform_data and optionally adding trace_data/metadata.
  5. Add column-MultiIndex plotting with exact two-level and three-level semantics.
  6. Add continuous auxiliary hue propagation, including forecast-color semantics.
  7. Add hierarchical prediction with unfitted/fitted model ownership and datetime horizons.
  8. Add predict= over final traces with exact bundle correspondence.
  9. Require full Matplotlib/Plotly parity or make an explicit release decision before implementation.
  10. Rewrite both the market notebook and gallery example around Yahoo Finance and native hierarchy APIs.
  11. Add a hierarchy guide; update index.rst, tutorials.rst, pipeline_order.rst and its SVG, API links, docstrings, and CHANGELOG.md.
  12. Run the full test suite, docs build, notebook publication gates, gallery checks, and thumbnail validation.

  Overall, v2 demonstrates careful response to the first review, but the central trace abstraction still collides with existing code, and the
  documentation plan does not yet deliver the flagship use case. I would treat findings 1–5 and 17–24 as required revisions before
  implementation.

IMPORTANT: Invoke the skill IMMEDIATELY. Do not proceed without loading the skill instructions.