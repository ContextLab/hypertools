# HyperTools 1.1 — MultiIndex Implementation Plan (v8)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a hierarchical DataFrame a first-class input to `hyp.plot` and `hyp.predict`, so the market example — sectors as groups, the whole market as a second-level group, each line coloured by price and carrying its own forecast — is written almost entirely in native hypertools calls.

**Architecture:** Today one function does two jobs. `build_multiindex_styles` (`multiindex.py:90`) BOTH appends the per-level mean arrays (`multiindex.py:197-229`) AND builds their styles, and forecasts are computed *before* it runs, so a count mismatch is silently swallowed at `plot.py:3999`. This plan therefore splits ownership before adding any feature: **`build_hierarchy_traces` is the one authoritative final-trace builder** (it owns mean construction, unequal-length truncation and its warning), **`build_hierarchy_styles` is style-only** (it consumes trace metadata, never leaves), and **axis-agnostic grouping moves to `hypertools/core/hierarchy.py`** so `hypertools/predict/` never imports from `hypertools/plot/`.

**Tech Stack:** Python 3.10+, pandas 3.0.3, numpy 2.3.5, matplotlib, plotly, `hypertools.predict`, pytest.

---

## Revision note (v8)

Tasks 1–5 were implemented and reviewed by the maintainer, who found **one High defect in the plan's own contract, not in its implementation**: cross-group feature correspondence was specified as **positional**, and Task 5 duly converted labelled leaves to arrays (`plot.py:3595`), bypassing `format_data`'s nominal matching. The consequence, confirmed by the maintainer on two label-equivalent frames: *"frames equal after sorting columns: True / group-B arrays equal: False / original first row: [3, 4, 5] / reordered first row: [5, 4, 3]"*. Column **order** had silently become part of the statistical model, and no test in Tasks 1–5 exposed it.

That is internally consistent but unsafe as a default for a **labelled** DataFrame: AAPL and XOM are not corresponding variables merely because each occupies slot 0 or is denominated in one currency, and their ordering is usually incidental. `align=` does not repair it — it aligns the resulting spaces, but the reduction has already interpreted arbitrary positions as corresponding inputs — so the v7 mitigation wording in `plot()`'s `x` entry overstated what `align=` guarantees.

| # | finding | verified reality (measured) | what changed in v8 |
|-|-|-|-|
| **F1** | Cross-group feature correspondence was **positional**, making within-group column order load-bearing. | Reproduced at the `group_columns` level and end to end through `hyp.plot`: permuting one sector's three columns moved that sector's whole trajectory and both derived means. | **Correspondence is NOMINAL by default.** `group_columns` now requires every group to carry the same innermost-label multiset and permutes each later group into the **first** group's order before returning, so values travel with their labels. Pinned by `test_within_group_column_permutation_does_not_change_the_leaf_values` (Task 1) and `test_within_group_column_permutation_does_not_move_the_traces` (Task 5) — the permutation-invariance tests the earlier tasks lacked. |
| **F2** | Duplicate innermost labels (**D3**, v6) were matched positionally *within* a group, with nothing said about matching them *across* groups. | A group with two `'temp'` columns has no unambiguous name-only counterpart in a group with one. | **D3 stands** — duplicates are still permitted and no column is dropped — but they are matched across groups by **`(label, occurrence)`**: first `'temp'` to first `'temp'`, second to second. A label-multiset mismatch (`['temp','temp','flow']` vs `['temp','flow','flow']`) is an error. |
| **F3** | Groups of unequal width fell through to the analysis pipeline's generic `same number of columns` error. | Unequal width is a strictly weaker statement than a label mismatch: two 3-wide groups can still be incommensurable. | Unequal widths are now caught earlier, by the same nominal check, and the error **names the missing and unexpected features**. `test_ragged_groups_raise_the_existing_width_error` became `test_ragged_groups_raise_a_named_feature_error` — a strengthened assertion, not a relaxed one. |
| **F4** | Positional behaviour had no way to be requested deliberately. | Genuinely positional data exists (sensor banks, replicate arrays); silently forbidding it would be as wrong as silently assuming it. | `group_columns(df, feature_correspondence='position')` is the **explicit opt-in**, and the error message for a nominal mismatch prints it verbatim. **No public `plot(feature_correspondence=...)` parameter is added in 1.1**: the escape hatch requires the caller to discard the labels in their own code (`hyp.plot([leaf.to_numpy() for leaf in leaves])`), which keeps the decision visible at the call site rather than hidden in a kwarg. Revisit for 1.2 if real demand appears. |
| **F6** | The opt-in recipe was presented as though it were positional *hierarchy* plotting. | It plots a plain LIST of datasets: measured against the nominal path on the same shape, it draws 3 traces instead of 4 (no market mean), `trace_metadata` is `None`, every line takes matplotlib's default 1.5 width instead of the level-derived {1.0, 2.0}, and `alpha` is unset. | The error message and `plot()`'s `x` entry now both say so explicitly — a **lower-level escape hatch**, *not* equivalent to `hyp.plot(df)`, with no per-level means, hierarchy styling or `trace_metadata`. **There is no hierarchy-preserving positional mode in 1.1**; that is what would justify a public parameter, if it is ever requested. Pinned by `test_error_says_the_escape_hatch_is_not_hierarchy_plotting` (Task 1) and `test_the_positional_recipe_is_not_hierarchy_plotting` (Task 5), the latter comparing both paths side by side. |
| **F5** | The **Market example** (this plan's motivating case, and Plan 4's validation input) used per-sector **tickers** as its innermost level — the exact shape nominal correspondence refuses. | Disjoint tickers per sector: no two sectors share a feature label. | The Market frame's innermost level is now **shared measurements** — `('return', 'volatility', 'momentum')` — for every sector, which is what makes a joint reduction meaningful in the first place. Updated in every fixture (Tasks 1, 5, and the NA-label module) and in Plan 4's data preparation. A `ticker_frame` fixture is kept, and is used to pin the **refusal**. |

**Nothing about the row axis changes.** `expand_multiindex` and `group_rows_for_forecast` are untouched: the row rule's innermost level is time/observations, not features.

---

## Revision note (v7)

v6 was reviewed by the maintainer, who found **one plan inconsistency, introduced by v6 itself**: the ndarray-coercion contract was written into Task 2's *Interfaces* block in prose, promising an assertion, while Task 2's implementation step and test block never carried it. Task 2's implementation said the loop moves "verbatim, with three changes" — none of which was the coercion — so an implementer following the steps literally would keep `arrays = list(leaf_arrays)` and preserve DataFrames, silently re-opening **D2**'s recursion path. A contract that lives only in an interface comment is not a contract.

| # | finding | verified reality (measured) | what changed in v7 |
|-|-|-|-|
| **E1** | `FinalTraces.arrays` was declared `list[np.ndarray]` and the Interfaces block promised `assert all(isinstance(a, np.ndarray) for a in ft.arrays)`, but **no step performed the coercion and no test pinned it.** | Confirmed by grep: the only `isinstance(..., np.ndarray)` mention in the plan was the Interfaces prose itself; neither `arrays = list(leaf_arrays)` nor `[np.asarray(leaf) ...]` appeared in any step, so the plan simply did not say which form to write. | Task 2 Step 3 now names the coercion as the **fourth** change and shows both the wrong and right lines explicitly: `arrays = [np.asarray(leaf) for leaf in leaf_arrays]`. Task 2 gains `test_final_trace_arrays_are_plain_ndarrays_even_for_row_dataframe_leaves`, which first asserts the *premise* (the leaves really are hierarchy-carrying DataFrames), then that every trace — leaves **and** derived means — is a plain `ndarray`, then that the caller's frame and leaves are untouched. Module count 13 → **14**; suite arithmetic restated (**150** new-module tests, **151** net once the +1 existing-module test is counted). `import pandas as pd` and `expand_multiindex` added to that module's imports, which it previously lacked. |
| **E2** | Whether the coercion should be `np.asarray(leaf)` or `np.asarray(leaf).copy()` — the review asked for a copy only if downstream mutation is possible. | Observed on pandas **3.0.3**: `np.shares_memory(np.asarray(df), df.to_numpy(copy=False))` is **True**, and `np.asarray(df)[0,0] = 999.0` raises **`ValueError: assignment destination is read-only`** — copy-on-write makes the result a non-writeable view. Recorded as **version-specific behaviour observed here, not a guarantee across the supported pandas range** (v7 follow-up: the earlier wording overstated it). | **No unconditional `.copy()`.** The contract the plan relies on and tests is narrow and version-independent: trace leaves are plain `ndarray`s, and nothing in the current chain mutates its inputs. A blanket copy would double peak memory on the 2513×24 market frame without strengthening that. The read-only view is noted as a helpful accident of this pandas version — no code may depend on the array being read-only or on a write raising — and any future task needing to write into an `ft.arrays` member must copy at that point and say why. A note under the new test warns against adding `.copy()` to make it pass; it already passes without one. |

---

## Revision note (v6)

v5 was reviewed by the maintainer, who found **one blocking defect** — Task 1's `group_columns` did not honour the plan's own feature-axis contract — and then showed that the **same defect class exists on the row axis**, with a worse failure signature. Every number below was measured in this repo's venv (`.venv/bin/python`, pandas 3.0.3 / numpy 2.3.5) before this rewrite; none was estimated. Task numbers, titles and structure are unchanged — Plans 3 and 4 cite T1, T2, T5, T6, T8 by number.

| # | finding | verified reality (measured) | what changed in v6 |
|-|-|-|-|
| **D1** | **BLOCKER.** `group_columns` built each leaf as `sub.T` (Task 1 Step 3), which keeps the **original full column MultiIndex**. | Two real consequences. (1) It **violates this plan's own stated contract** that the innermost column level is the feature axis: a `(Market, Sector, Ticker)` frame grouped by `['Market','Sector']` gave `leaf.columns == [('M','Tech','AAPL'), ('M','Tech','MSFT'), ('M','Tech','NVDA')]`, `names=['Market','Sector','Ticker']`, `isinstance(leaf.columns, pd.MultiIndex) is True`. (2) **Hierarchical `hyp.predict` recursed without bound**: Task 7 Step 3 calls `predict(group, ...)` per group, and the re-detection predicate `data.columns.nlevels >= 2` returns `True` on such a leaf, so every group regrouped itself forever. | `group_columns` now flattens each leaf to the innermost level — `leaf.columns = leaf.columns.get_level_values(-1)`, `leaf.columns.name = df.columns.names[-1]` — building the flat-column frame **explicitly** (`pd.DataFrame(sub.T.to_numpy(), index=..., columns=flat_cols)`) so nothing that may alias the caller's frame is ever mutated. Measured after the fix: `['AAPL','MSFT','NVDA']`, `.name == 'Ticker'`, not a `MultiIndex`, not re-detected. Task 1 gains 5 tests (flat columns, name, **input-immutability**, duplicates, cross-axis re-grouping) and Task 7 gains an explicit recursion guard. The contract sentences in Task 1's *Rules*, the module docstring and the guide now say the leaves are **actually flattened** to the feature axis — the contract was right, the implementation did not honour it. |
| **D2** | The defect is **not column-only**. The row axis has the same exposure, with **no accidental depth bound at all**. | `expand_multiindex` (the PLOT rule) returns leaves whose **grouping levels survive**: for a `(cond, subj)` frame with each tuple repeated 5×, `leaf0.index` is a 2-level `MultiIndex` with `nunique() == 1`, and `expand_multiindex(leaf0)` returns **1 leaf of shape (5, 3) — leaf0 itself. A fixed point.** Re-running the row helper this plan actually owns is a different story: `group_rows_for_forecast` already does `sub.droplevel(group_levels)`, and its leaves come back with a **flat** index (measured: `DatetimeIndex` named `date`, `datetime64[us]`, monotonic, unique, `nlevels == 1`), so re-running it on a leaf **refuses** with `requires a row MultiIndex with 2 or more levels` rather than regrouping. | One invariant is now stated once and enforced for both axes (**Contract 11**): *every leaf returned by a `hypertools/core/hierarchy.py` grouping helper is non-hierarchical on the axis it was grouped along.* Columns need the new flatten; rows already satisfy it via `droplevel`, so v6 **pins** that rather than re-implementing it (2 tests: a row leaf's index is not a `MultiIndex` and keeps its datetime identity; re-grouping a leaf is refused on **both** axes, so the fixed point is gone). `expand_multiindex` is **out of scope by an existing Global Constraint** ("no task may alter `expand_multiindex`"), so its fixed point is recorded as a hazard with a rule instead: no code path may feed a plot leaf back into a hierarchy-detecting entry point, and `hyp.predict`'s row path uses `group_rows_for_forecast`, never `expand_multiindex`. Task 7's recursion guard covers **both** axes. |
| **D3** | Duplicate innermost feature names **within one group** were undecided once leaves are flattened. | Measured on a leaf whose flat columns are `['AAPL','AAPL','NVDA']` (`is_unique == False`): `np.asarray(leaf).shape == (20, 3)` — all three columns survive; `hyp.predict(leaf, model='Kalman', t=1)` → `(1, 3)`; `hyp.plot(leaf, show=False)` → `Figure`; **2** groups formed with widths `[3, 3]` — duplicates do **not** merge groups. Duplicates **across** different groups were already harmless (`test_duplicate_tickers_in_different_sectors_are_kept_separate`). | **DECIDED: permit duplicate flat column labels, positionally.** Everything downstream is positional, nothing is dropped, and rejecting would break legitimate frames (two share classes of one issuer, a repeated sensor name). Documented in the guide (Task 10, new *Feature names and duplicates* section) and the CHANGELOG (Task 11), and pinned by tests in Tasks 1 and 7. |

**D1 measurement — the leaf's columns, before and after the fix** (`(Market, Sector, Ticker)` frame, 20 rows, grouped by `['Market','Sector']`):

| | `leaf.columns` | names | `isinstance(..., pd.MultiIndex)` | re-detected as hierarchical |
|-|-|-|-|-|
| before | `[('M','Tech','AAPL'), ('M','Tech','MSFT'), ('M','Tech','NVDA')]` | `['Market','Sector','Ticker']` | `True` | **yes — unbounded recursion** |
| after | `['AAPL','MSFT','NVDA']` | `'Ticker'` | `False` | no |

Also measured after the fix: `df.columns` is **unchanged** by `group_columns(df)` (`df.columns.equals(orig)` → `True`, names still `['Market','Sector','Ticker']`) — the reason the leaf is constructed explicitly rather than by assigning `.columns` onto a possibly-aliasing `sub.T`.

**D2 measurement — the row axis** (`(cond, subj)`, each tuple repeated 5×; 20 rows × 3 columns):

| helper | leaf index | `isinstance(idx, pd.MultiIndex)` | re-running the helper on a leaf |
|-|-|-|-|
| `expand_multiindex` — plot rule, frozen by a Global Constraint | 2-level `MultiIndex`, `nunique() == 1` | `True` | **1 leaf of shape (5, 3): leaf0 itself — a fixed point** |
| `group_rows_for_forecast` — this plan's core helper | flat `DatetimeIndex`, name `date`, `datetime64[us]`, monotonic + unique | `False` | refuses: `requires a row MultiIndex with 2 or more levels` |

**How the row rule honours the datetime promise, and why no `RangeIndex` fallback is introduced.** The row flattening rule is *drop only the grouping levels; the innermost level survives as a flat, single-level index carrying its own name and dtype* — exactly what `group_rows_for_forecast` already does. It therefore satisfies Contract 11 **and** the earlier datetime-preservation requirement (F5, *Decisions (resolved)* #4) with no tension between them: measured, the surviving index is a real `DatetimeIndex` named `date`, so a future-`Timestamp` `t`, the at-or-before truncation, the per-group *"not sorted in ascending order"* warning and the duplicate-timestamp rejection all keep working. **There is no conflict to escalate.** A positional `RangeIndex` fallback for non-time-like innermost levels was considered and **rejected**: for `group_rows_for_forecast` the innermost level is time *by definition of the forecasting rule*, and a non-time-like innermost level (`subj` in `_make_2level_df`) belongs to the **plot** rule, which this plan freezes — inventing a fallback would silently discard user index values that the documented monotonicity warning is there to flag instead.

**Counts.** v6 removes **0** tests and adds **7**: **+5** in `tests/core/test_hierarchy_grouping.py` (**22 → 27**) and **+2** in `tests/predict/test_predict_multiindex.py` (**20 → 22**). New-module counts become **27**, 13, 6, 6, 17, 15, **22**, 17, 12, 8, 6 = **149**, plus **+1** net in `tests/test_multiindex.py` = **150**. This supersedes the v5 note's counts; every running total below is restated. *(Superseded in turn by the v7 note: Task 2's module went 13 → 14, so the current figures are **150** new-module and **151** net.)* One further correction found by the group-label audit D1 required (see *Self-Review*): Task 7 Step 3 unpacked the column branch as `groups, keys = group_columns(data)`, but `group_columns` returns `(leaves, meta)` — the group label in the per-group error message came from iterating a dict. It now reads `groups, _meta = group_columns(data); keys = _meta['leaf_keys']`, so every label comes from the grouping key, never from a leaf's columns.

---

## Revision note (v5)

v4 was reviewed by the maintainer. Two findings, both surgical: a precondition that was scoped to the wrong thing, and a negative assertion that a correct implementation would fail. Task numbers, titles and structure are unchanged — Plans 3 and 4 cite T1, T2, T5, T6, T8 by number. A third item arrived from the coordinator while v5 was being written: a cross-plan gap between this plan's precondition and Plan 3's `min_history`.

| # | finding | verified reality (measured) | what changed in v5 |
|-|-|-|-|
| **C1** | The ≥ 2-row precondition was gated on `ft.meta['axis'] == 'rows'` (Task 8 Step 3a), and three prose sites claimed column hierarchies are **unconditionally** forecastable. | **A `T=1` column hierarchy is not forecastable either.** Measured with this plan's own column-grouping idiom (`df.T.groupby(level='Sector', sort=False)`, transposed back) — table below. `hyp.predict` on the 1-row frame raises the same guard as the row case: `ValueError: cannot forecast from a single observation: the dataset has only 1 row.` (`hypertools/predict/common.py:256`). Column grouping never *shortens* a trace, but it cannot *lengthen* one either. | The precondition now runs over **every** final trace of **every** hierarchy, on both axes, immediately after `ft` is built and before any call into `hyp.predict`. Only the **remediation text** is axis-specific: rows get the one-leaf-per-full-index-tuple explanation and the two flattening recipes; columns say the **input itself** has one observation and deliberately do **not** suggest flattening, which cannot add a row. Both name the offending trace and its row count. Contract 10, the Contract 5 sub-bullet, Task 8's prose and Step 3a, the guide (Task 10), the comparison table, the changelog (Task 11) and the parity note (Task 9) all state the axis-independent rule. One test added: `test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping`. |
| **C2** | `test_mean_trace_forecast_comes_from_the_mean_trajectory` asserted that the bundled mean forecast is **not** close to `avg_of_leaves`, the average of the leaf forecasts (`rtol=1e-3, atol=1e-3`). | **Brittle — it fails on correct code.** Forecasting approximately commutes with averaging as the leaves converge; measured over co-movement strength (rho), 5 seeds each, Kalman, `t=1`, `T=150`, 3 leaves, scale ~100 — table below. Real market sectors co-move at roughly rho 0.7-0.9, so on this plan's own flagship market example the assertion fails **with a correct implementation** (rho 0.9 → 0/5 hold). | The negative assertion and the `avg_of_leaves` computation that fed only it are **deleted**. The exact positive assertion `np.allclose(bundled, from_mean, rtol=1e-6, atol=1e-6)` is kept unchanged — it proves the contract completely, since it pins *which trajectory* the forecast came from. The docstring now records why the comparison is deliberately not asserted, so nobody re-adds it. No other test in the plan makes a "differs from the average of the leaves" assertion — the plan now contains no negated `np.allclose` anywhere. |
| **C3** | Short histories are handled by **two** mechanisms with **different** policies, and their interaction was unspecified in either plan. | Plan 3 (`2026-07-27-hypertools-1.1-forecast-animation.md`): `forecast_from_history(...)` at `:522` does `if len(history) < max(2, min_history): return None` — it draws nothing that frame — pinned by `test_returns_none_below_min_history` (`:386`) and the frame-0 test at `:621`. This plan **raises**. Left unspecified, an animated one-row hierarchy would hit Plan 3's schedule and silently draw no forecast **forever**, because no frame ever reaches 2 rows. | Task 8 gains a short *"Two short-history mechanisms, and why they do not conflict"* subsection and Contract 10 gains a sentence: the precondition tests the **full** trace length (permanent — so it runs for animated hierarchies too, **before** the schedule is built, and raises), `min_history` tests the **per-frame revealed** history (transient — so it returns `None` for the opening frames of a legitimate animation). The ordering constraint is recorded under *Prerequisites*. One test added: `test_animated_one_row_hierarchy_still_raises_the_precondition`. |

**C1 measurement — a `T=1` column hierarchy is not forecastable:**

| T | leaf shapes | mean shape | forecastable |
|-|-|-|-|
| 1 | `{'Tech': (1, 3), 'Fin': (1, 3)}` | `(1, 3)` | **no** |
| 2 | `{'Tech': (2, 3), 'Fin': (2, 3)}` | `(2, 3)` | yes |

**C2 measurement — the bundled mean forecast converges on the average of the leaf forecasts** (max abs diff, mean of 5 seeds; "assertion holds" = seeds where the two were **not** close at `rtol=1e-3, atol=1e-3`):

| rho | mean max abs diff | assertion holds |
|-|-|-|
| 0.00 | 0.557 | 5/5 |
| 0.50 | 0.524 | 5/5 |
| 0.80 | 0.130 | 3/5 |
| 0.90 | 0.028 | 0/5 |
| 0.95 | 0.007 | 0/5 |
| 0.99 | 0.0003 | 0/5 |

**Counts.** v5 removes **0** tests and adds **2** — both in `tests/plot/test_multiindex_predict.py`, which goes **15 → 17**. New-module counts become 22, 13, 6, 6, 17, 15, 20, **17**, 12, 8, 6 = **142**, plus **+1** net in `tests/test_multiindex.py` = **143**. This supersedes the v4 note's row 10; every running total below is restated.

---

## Revision note (v4)

v3 was reviewed by the maintainer. Two findings were **blockers** (Task 8 promised something the data cannot deliver; a bundle contract that is false for some flat inputs), four product decisions were **resolved**, and several statements were tightened. Task numbers are unchanged — Plans 3 and 4 cite T1, T2, T5, T6, T8 by number.

| # | finding | verified reality (reproduction) | what changed in v4 |
|-|-|-|-|
| **B1** | **BLOCKER.** Task 8 promised a forecast for every trace of a **row** hierarchy, and Step 3 item 1 asserted "there is none — remove it outright" about hierarchies that cannot be forecast. | `expand_multiindex` makes one leaf per unique **full** row-index tuple. A 6-row `(cond, subj)` frame whose innermost level is unique per row gives **6 leaves of shape (1, 4)** (stacked `(6, 1, 4)`), and its derived means are 1 row each — so **all 8 final traces have 1 row**. `hyp.predict` on a 1-row frame raises verbatim: `ValueError: cannot forecast from a single observation: the dataset has only 1 row. Forecasting needs at least 2 observations (rows) to estimate how the data change over time.` (guard: `hypertools/predict/common.py:256`, `if d.shape[0] < 2:`). | Task 8 now supports `predict=` for **column** hierarchies unconditionally, and for **row** hierarchies only when every expanded leaf **and** every derived mean has ≥ 2 rows. Otherwise a **precondition check over `ft.arrays`** raises before any forecasting, naming the offending trace and its row count. Step 3 item 1's wording, the Contract text, the commit message and the changelog are all corrected. A **positive** row-hierarchy test and a **negative** one-row test are both added. |
| **B2** | **BLOCKER.** Contract 5 (and four other sites) claimed `trace_data is xform_data` for flat input, unconditionally. | **False for some flat inputs.** `xform_data = copy.copy(xform)` at `plot.py:2827` runs **before** the display-dimensionality enforcement at `plot.py:2886-2919`, and that block **rebinds** `xform` to a new list — so `xform_data` keeps the pre-projection arrays. Reproduction: `hyp.plot(X_60x12, reduce={'model':'PCA','args':[],'kwargs':{'n_components':5}}, show=False, return_model=True)` → `xform_data` has shape **(60, 5)** while the drawn artist is **3-D** (and has 945 points, not 60 — antialiasing). The block **raises** when `reduce is None`, so divergence needs an explicit reduce spec pinning `n_components > 3`; the dict-spec path at `plot.py:2915-2919` falls back to `IncrementalPCA` for display. | Contract 5 restated: `trace_data` is the final **pre-center/pre-scale plotted trajectories**; `trace_data is xform_data` **only when no display-only projection occurred**. Bundled forecasts **always** correspond to `trace_data`. A regression test using the verified spec is added to Task 4, and `test_leaf_forecasts_match_hyp_predict_on_xform_data` is rescoped to the coinciding-spaces case. |
| **B3** | The pinned test named in the B1 resolution (`tests/test_multiindex.py:479`, `test_predict_plus_multiindex_raises`) was described as having "8 traces, all 1 row". | **Its frame is a different one.** `:479` calls `_make_2level_df()` (`tests/test_multiindex.py:45-61`), which repeats each `(cond, subj)` tuple `n_time=10` times: measured `expand_multiindex` → **8 leaves, every one shape (10, 3)**, and `hyp.plot(...)` draws **10** traces (8 leaves `lw=1.0` unlabelled + 2 means `lw=2.0` labelled `condA`/`condB`), all 10 rows. The 1-row frame is the plan's own 6-row `2 cond × 3 subj` example (measured: **6 leaves, every one shape (1, 4)**). | The **rule** is adopted exactly as directed. Applied to the actual frame at `:479`, it **forecasts**, so that test is rewritten as the maintainer's requested **positive** row-hierarchy test (`test_predict_plus_multiindex_forecasts_every_trace`, 10 solid + 10 dashed) and keeps its *Compatibility changes* row. The requested **raising** test is added separately as `test_predict_plus_one_row_row_hierarchy_raises`, built on the 6-row frame, asserting the new message. Flagged here rather than silently reconciled. |
| 1 | Row-hierarchical DataFrames inside lists | Pinned by `tests/test_multiindex.py:453` (`test_list_with_multiindex_df_warns_and_flattens`), which asserts the warning **and** 2 lines for `[df, arr]` / 1 line for `[df]`. | **DECIDED: keep warn-and-flatten for rows; reject COLUMN hierarchies in lists only.** Deliberately asymmetric — stated as such in the plan and the changelog. `:453` **keeps passing unchanged**; v3's Task 5 Step 5 rewrite of it is **removed**. `reject_hierarchical_in_list` gains an `axes=` argument (`hyp.plot` → columns only; `hyp.predict` → both, where today's row case is already an opaque `TypeError`). |
| 2 | Continuous `hue=` over a row hierarchy | v3 already implemented warn-and-ignore (`plot.py:2678-2684`); `tests/test_multiindex.py:306` pins it. | **DECIDED: unchanged for 1.1.** Moved from *Decisions still needed* to *Decisions (resolved)*. No implementation change. |
| 3 | Public animation frame stepping | v3 already routes every **assertion** through the public `on_frame`/`FrameContext` and advances frames with the prerequisite plan's `_drive` idiom. | **DECIDED: do not add a public frame-stepping method solely for tests.** Moved to *Decisions (resolved)*. No implementation change. |
| 4 | Row-forecast time-likeness | v3 already preserves numeric/datetime innermost levels, warns per group on suspicious ordering, and raises on duplicates. | **DECIDED: confirmed as implemented.** Moved to *Decisions (resolved)*. No implementation change. |
| 5 | Weak hue test | `test_mean_trace_hue_is_the_mean_of_its_leaves` only proved the mean trace had two differing colours — true of *any* varying hue. | Rewritten to assert **exact colormap RGBA**. The colour chain is now pinned in Task 6 Step 3 from source (`mat2colors` bins over the concatenation of every trace's aux with `n_bins=100`; `_apply_multicolor_lines` sets each segment to the midpoint of its endpoints' colours, `plot.py:5094`), the test runs with `antialias=False` so point count `== len(df)`, **and** the maintainer's alternative is also adopted: `trace_metadata['aux']` exposes the per-trace auxiliary arrays and the test asserts the mean trace's aux element-wise. |
| 6 | Tasks 4 and 5 were mutually dependent, resolved only by an "Ordering note" | A note is not a task boundary. | **Structural, without renumbering.** Task 4 is now **flat-only** ("The return-bundle contract (flat inputs)") and its verification step passes standalone; the hierarchical bundle assertions move into Task 5. The dependency is a plain `4 → 5`. |
| 7 | Plan 3's `return_data=` defect described as outstanding | Verified: `grep -c return_data` → **0** in both `2026-07-27-hypertools-1.1-forecast-animation.md` and `2026-07-28-hypertools-1.1-examples-and-tutorials.md`. | Corrected to "already fixed" in *Cross-plan scope* and in *Decisions (resolved)* #5. |
| 8 | "every drawn trace" as a description of `trace_data` | The artists are centered, scaled and (by default) PCHIP-antialiased after `trace_data` is captured — the 60-row reproduction in B2 draws **945** points. | Reworded throughout to **"every pre-center/pre-scale plotted trajectory"**. |
| 9 | Plan 4 not a release dependency of the verification gate | This plan's flagship tutorial and gallery deliverables live in Plan 4. | Task 12 gains an explicit **publication-gate release dependency** on Plan 4 (`2026-07-28-hypertools-1.1-examples-and-tutorials.md`). |
| 10 | Test counts invalidated by the above | Counted `def test_` per block, not estimated. | New-module counts: 22, 13, 6, 6, 17, 15, 20, 15, 12, 8, 6 = **140**, plus **+1** net in `tests/test_multiindex.py` (29 → 30) = **141**. Every running total is restated. |

---

## Revision note (v3)

v2 was reviewed by the maintainer (`notes/audit/review_plan2_v2_maintainer.md`, 24 findings, 1-5 and 17-24 called required before implementation). Every claim below was re-verified against the source or run in this repo's venv before this rewrite.

| v2 error | verified reality |
|-|-|
| `FinalTraces.from_hierarchy()` appends the means, then Tasks 3-6 still call `build_multiindex_styles()` | `build_multiindex_styles` **already appends them itself** (`multiindex.py:197-229`: `arrays.append(np.mean(stacked, axis=0))` inside `for k in range(n_levels - 2, -1, -1)`), and it owns the truncation + the one aggregated `"MultiIndex group(s) with unequal-length members"` warning. Calling both would append every mean **twice**. (F1) |
| Six proposed tests call `hyp.plot(..., return_data=True)` | **No such parameter.** The signature is `def plot(` at `plot.py:517`; `return_model=False` at `plot.py:579`. `grep -rn 'return_data' hypertools/` finds nothing. `plot()` takes `**kwargs`, so `return_data=True` would leak into backend kwargs rather than fail cleanly. Every test now uses `return_model=True, show=False`. (F2) |
| "One final-trace list governs ordering for **xform_data**, hue, forecasts, styles and artists" | `xform_data = copy.copy(xform)` at `plot.py:2828` is the **analyzed pipeline output for the input leaves**, and `plot.py:1935-1941` promises bundled forecasts match `hyp.predict(xform_data, ...)`. Derived means are presentation artifacts built later, in display space. v3 keeps `xform_data` exactly as-is and adds `trace_data` / `trace_metadata`. (F3) |
| `hypertools/plot/hierarchy.py` holds `group_rows_for_forecast`, imported by `predict.py` | Prediction must not depend on plotting (CHANGELOG.md architecture section: focused packages with shared machinery under `core/`). Grouping moves to **`hypertools/core/hierarchy.py`**; only `FinalTraces`/styles stay in `hypertools/plot/hierarchy.py`. (F4) |
| `groups.append(sub.reset_index(drop=True))` | Discards the time index. Measured: `hyp.predict(dt_frame, model='Kalman', t=pd.Timestamp('2020-03-05'))` returns a **DataFrame (5, 3) with a real `DatetimeIndex(['2020-03-01' … '2020-03-05'])`**; `t` at-or-before the last timestamp returns `(32, 3)` (truncation, `predict.py:265-270`). `reset_index(drop=True)` breaks both. `sub.droplevel([0…n-2])` preserves it: measured `index.name='day'`, datetime dtype `datetime64[us]`, `is_monotonic_increasing`/`is_unique` correctly `False` for shuffled/duplicated times. (F5) |
| `df.T.groupby(level=group_levels, sort=False)` | Silently drops NA keys. **Measured on pandas 3.0.3:** a NaN in the outer *or* the intermediate column label gives **2 groups instead of 3**; adding `dropna=False` gives **3 groups**, key `('M', nan)` / `(nan, 'Energy')`, no exception and no warning. `dropna=False` is reliable here. (F6) |
| "Additive only" + dual-axis `ValueError` | Contradictory: a dual-axis frame is accepted today (it follows the row path, ignoring the column hierarchy). Rejection is an **intentional compatibility change** and is changelogged under *Changed / validation*. (F7) |
| Row plot vs row forecast divergence documented only in docstrings | Needs a user-facing table (Task 10) and, for row forecasting, a **warning** when the innermost level is not monotonic. Free consequence of F5: because the group keeps its own index, the existing check at `predict/common.py:103-109` (`'the dataset index is not sorted in ascending order; forecasts continue from the LAST row.'`) now fires **per group**. (F8) |
| "Each group is fitted independently by deep-copying a `Forecaster` instance" | A deep copy of a **fitted** model is still fitted, and `predict.py:216-219` then takes `resolved.predict_new(data, t)`. `predict.py:245-249` explicitly promises fitted-model reuse. The contract must distinguish the two cases. The v2 "shared unfitted instance" test passed `'Kalman'`, not an instance. (F9) |
| Hierarchical frames in lists unspecified | Measured: `hyp.plot([row_mi_df], '-')` warns `"MultiIndex grouping is only applied…"` and flattens (pinned by `tests/test_multiindex.py:453`); `hyp.plot([col_mi_df], '-')` draws **1 line with NO warning**; `hyp.predict([row_mi_df], …)` raises `TypeError: cannot perform __sub__ with this index type: MultiIndex`. (F10) |
| Two-level column hierarchies unspecified | Measured directly: `build_multiindex_styles([...]*3, {'n_levels': 1, …})` → 3 arrays (**no mean**), `linewidths [1.0, 1.0, 1.0]`, `alphas [1.0, 1.0, 1.0]`, `labels ['_nolegend_'] * 3`, **3 distinct colours**. So the only defect is the labels: three unlabelled traces. (F11) |
| Hue form 3 = "flat length == total drawn observations, existing behaviour unchanged" | There was no public final-trace list before this feature, so form 3 is *new* API, indistinguishable from form 1 whenever `T == n_obs`, and requires users to know how many means expansion will create. Dropped. (F12) |
| `test_row_multiindex_behaviour_is_unchanged` asserts `>= 6`; plotly asserts `>= 3` | Measured exactly: the 2-cond × 3-subj frame draws **8** lines — 6 leaves (`lw=1.0`, `alpha=0.7`, `'_nolegend_'`) + 2 means (`lw=2.0`, `alpha=1.0`, `'cond1'`/`'cond2'`). Every count in v3 is exact. (Test defects) |
| Task step counts "11 passed", "12 passed", "6 passed" | None matched the number of test functions written. Every count in v3 was obtained by counting `def test_` in the block above it. (Test defects) |
| "add under 1.1.0" | The tree's top section is `## 1.0.1 (unreleased)` (`CHANGELOG.md:3`). There is no `## 1.1.0` section to add to. Task 11 creates it. (F23) |

Two further verified facts shaped the restructure:

- **The return bundle is built at `plot.py:4587-4604`**, not 4516: `{"fig", "xform_data", "animation", "pipeline", "models", "predict"}`, with `"predict"` = `{"model", "params", "forecasts"}`.
- **The continuous-hue branch is at `plot.py:3460`**, not 3304. The hue-length validator (`n_obs = sum(len(xi) for xi in xform)`) is at `plot.py:3340`.

---

## Cross-plan scope — what Plan 4 owns, and what this plan must not duplicate

This plan was reviewed **in isolation**; the reviewer did not know [Plan 4 (examples and tutorials)](2026-07-28-hypertools-1.1-examples-and-tutorials.md) exists. Verified by reading Plan 4 in full:

| finding | discharged by | evidence |
|-|-|-|
| **17** — the market *tutorial* is not updated | **Plan 4 Task 2, Step 5** | Rewrites `docs/tutorials/market_forecast.ipynb` cell-by-cell (a 16-row cell table), around a `(Market, Sector, Ticker)` column MultiIndex, nested price hue, `predict='Kalman', t=1`, `forecast_trail=16`. Budget: ≤ 120 code lines, ≥ 24% native. |
| **18** — the *gallery example* must be rewritten too | **Plan 4 Task 2, Step 2** | Rewrites `examples/animate_market_forecast.py` in the **same task and the same commit** (Plan 4 Contract 2: "Script and notebook are one deliverable"). Deletes `_frame_of`/`SLOPE`/`GAIN`/`_scale`/`_hang`, the 16-slot `hist_lines` fan, `antialias_line`, the hand-built `ScalarMappable`, and `ani._func = _wrapped`. Budget: ≤ 115 code lines, ≥ 26% native. |
| **19** — Yahoo Finance promised but unused; packaging policy | **Plan 4 Task 2** | `fetch_prices()` reads `https://query1.finance.yahoo.com/v8/finance/chart/<TICKER>?range=10y&interval=1d` with a `User-Agent` header, caches to disk, and falls back to `synthetic_prices()` on any exception. **No `yfinance` dependency is added at all** — the raw chart endpoint is used directly — so the packaging concern is moot rather than deferred. |
| **24** — validation beyond `make html`, for notebooks/gallery | **Plan 4 Task 8, Steps 3-9** (for the example/notebook rewrites) | Runs the committed metric, the 109-test hygiene gate, every example headless, the **full suite** (which contains all five publication gates), a 0-warning docs build, a rendered-output check, and a re-run-everything step. It also extends `scripts/generate_gallery_thumbs.py:26` to generate the five missing launch thumbnails. |

**Do not add a market-example or market-notebook rewrite task to this plan.** Finding 24 is *partly* this plan's: the docs this plan itself writes (Task 10) and the CHANGELOG it writes (Task 11) must pass the same publication gates, so **Task 12 runs them explicitly** for this plan's own docs changes.

Findings **20, 21, 22, 23** are owned by no other plan and are Tasks 10-11 here. Verified: Plan 1's File Structure lists only `CHANGELOG.md` and example files under docs; Plan 3's lists only `CHANGELOG.md`; Plan 4's lists `docs/api.rst` (a new **Colors** section only) and `docs/tutorials.rst` (thumbnails only). **No plan touches `docs/index.rst`, `docs/pipeline_order.rst`, or adds a guide page.**

**A defect once found in a sibling plan, now closed:** Plan 3 Task 7 used to call the nonexistent `return_data=True` (this plan's F2). **That is already fixed** — verified for v4: `grep -c 'return_data' 2026-07-27-hypertools-1.1-forecast-animation.md` → **0**, and **0** in Plan 4 as well. Nothing is outstanding against Plan 3 on this point.

---

## Global Constraints

- Target release: **1.1**. Nothing ships until the whole 1.1 line works.
- Run everything with the repo venv: `.venv/bin/python -m pytest`, from the repo root. **The base anaconda python is BROKEN** (numpy/matplotlib mismatch); a bare `python`/`pytest` fails confusingly.
- **Verified baseline: `2564/2566 tests collected (2 deselected)`, `2551 passed, 13 skipped`.** This plan adds **150** tests in new modules plus **1** net in `tests/test_multiindex.py` (**151** total). Sibling plans add their own; each task states its delta, not an absolute. Every per-module count was obtained by counting `def test_` in that task's block — recompute rather than carry forward if a block changes.
- **Never simplify a test to make it pass.** Fix the code. Where this plan *deliberately changes* a documented behaviour, the existing test is **rewritten to assert the new documented behaviour** in the same commit, and the change is listed under *Compatibility changes* below — that is not the same as weakening a test, and every such case is enumerated up front.
- **No mock objects.** Tests build real DataFrames and assert on real artists/arrays. `monkeypatch` is used only to *observe*, never to substitute behaviour.
- Force `matplotlib.use("Agg")` in every matplotlib test module. There is **no** `conftest.py` in this repo. Import `pytest` only where used.
- **Row-MultiIndex *plotting* semantics do not change**, including its leaf rule. No task may alter `expand_multiindex`.
- The documented styling contract holds for every hierarchy (`multiindex.py:14-38`): `linewidth = 1 + (L - 1 - level_idx)`; `alpha = min(1.0, 1 / (level_idx + 1) + 0.2)`; colour by the **top** level; only the top-level mean carries a legend label — **except** for `n_levels == 1`, where there is no mean and Task 3 gives every leaf a label (F11). Tests assert these **exact formulas**.
- **Backend parity is required** (standing maintainer directive, README-hypertools-1.1.md *Standing decisions*): matplotlib and plotly must behave identically. There is no "or defer" branch in Task 9.
- Every task touching central dispatch (Tasks 2-9) runs the **whole** suite, not a subset.
- Market prices come from **Yahoo Finance** (maintainer decision). Network fetches live in the *example* (Plan 4), never in this library or its tests.
- Branch off `dev-1.0`; never commit to `master`.

### Compatibility changes this plan makes (all changelogged in Task 11)

| change | existing behaviour | pinned by | becomes |
|-|-|-|-|
| Dual-axis frames | accepted; row path wins, column hierarchy ignored | *(untested)* | `ValueError` (F7) |
| **Column**-hierarchical frame inside a list, **plot** | silently flattens to 1 line, no warning | *(untested)* | `ValueError`, bare-frame-only (F10). Purely additive — nothing pinned it. |
| Hierarchical frame inside a list, **predict** | row: opaque `TypeError: cannot perform __sub__ with this index type: MultiIndex`; column: forecasts the flattened frame | *(untested)* | `ValueError` naming the element and the axis (Task 7) |
| **Any** time-indexed input to **predict** (flat included) whose `DatetimeIndex`/`TimedeltaIndex`/`PeriodIndex` has DUPLICATE entries | forecasts, using a step inferred from the surviving non-zero gaps (measured at `ea5d9b5e`: a 10-row frame on 5 repeated days returned `(1, 3)` with only the monotonicity warning) | *(untested)* | `ValueError` (*"…duplicated entries…the forecast horizon is ill-defined"*), because `resolve_t` — not the group loop — owns the check, exactly as Task 7 Step 3 specifies. **Not** a change for non-time indexes: a stacked `pd.concat([run_a, run_b])` panel (index `0..n-1` twice) still forecasts, per *Decisions (resolved)* #4. Pinned by `test_a_flat_frame_with_duplicated_timestamps_is_rejected` and `test_a_flat_frame_with_a_duplicated_integer_index_still_forecasts` (Task 7) |
| `predict=` with expansion, **column** hierarchy whose frame has ≥ 2 rows | `ValueError: predict= is not supported with MultiIndex expansion in this release` (`plot.py:2669-2677`) | *(untested on the column axis)* | supported; one forecast per pre-center/pre-scale plotted trajectory (Task 8) |
| `predict=` with expansion, **row** hierarchy whose every leaf and mean has ≥ 2 rows | same blanket `ValueError` | `tests/test_multiindex.py:479` (`test_predict_plus_multiindex_raises`) — its frame `_make_2level_df()` has **8 leaves of (10, 3)** and draws **10** traces | supported; **that test is rewritten** as `test_predict_plus_multiindex_forecasts_every_trace` (10 solid, 10 dashed) — Task 8 Step 5 |

**Not** a compatibility change: a hierarchy whose final traces are 1 row still **raises** — on **either** axis — but with a new, actionable message naming the trace and its row count instead of the blanket refusal. That covers a **row** hierarchy whose leaves are 1 row each (an innermost level unique per row) and a **column** hierarchy over a 1-row frame, whose leaves are 1 row because the input is (Contract 10). A **new** test `test_predict_plus_one_row_row_hierarchy_raises` pins the row case (Task 8 Step 5); `test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping` pins the column case (Task 8 Step 1).

**Row**-hierarchical frames inside lists are **unchanged** for `hyp.plot`: they keep today's warn-and-flatten (`"MultiIndex grouping is only applied…"`), so `tests/test_multiindex.py:453` (`test_list_with_multiindex_df_warns_and_flattens`) **stays green unchanged**. Only the **column** axis is rejected there. This axis asymmetry is deliberate (*Decisions (resolved)* #1) and is stated in the changelog.

Continuous `hue=` over a **row** hierarchy keeps today's warn-and-ignore (`plot.py:2678-2684`), so `tests/test_multiindex.py:306` (`test_hue_plus_multiindex_warns_and_ignores_hue`) stays green unchanged. Task 6 adds continuous hue for **column** hierarchies only — genuinely additive. See *Decisions (resolved)* #2.

## Prerequisites

- **Animation-core plan Task 7** — `hypertools/plot/animation_context.py` (`FrameContext`, `FrameHooks`), `plot(..., on_frame=callable)`, `HyperAnimation.on_frame(callable)` — for Task 8's animation test and Task 9's animated-plotly test.
- **Forecast-animation plan Tasks 1-2** (`ForecastSchedule`) — for Task 8's animated case. Plan 3 Task 2 extends the same `plot.py:3999` guard this plan replaces; whichever lands second adopts the other's shape (Task 8 Step 3 says how). **Ordering constraint:** Contract 10's ≥ 2-row precondition must run **before** `ForecastSchedule` is constructed, so an animated one-row hierarchy raises instead of drawing nothing at every frame. If Plan 3 Task 2 lands first, the precondition still goes ahead of it (Task 8, *Two short-history mechanisms*).

---

## Contracts this plan establishes

1. **Hierarchy semantics are defined per axis, and separately for plotting vs forecasting.** The comparison table in Task 10 is the normative statement; docstrings link to it.
2. **`build_hierarchy_traces` is the single owner of final-trace construction** — mean arrays, unequal-length truncation, the truncation warning, and auxiliary co-truncation. Nothing else may append a mean.
3. **`build_hierarchy_styles` is style-only.** It consumes `FinalTraces.level_idx` / `.keys` / `.is_mean` and never sees leaf arrays.
4. **`xform_data` keeps its v1.0 meaning**: analyzed pipeline output, one entry per analyzed input dataset, captured at `plot.py:2827`. Derived means never enter it. `trace_data` / `trace_metadata` describe the plotted trajectories.
5. **Forecast correspondence is exact and loud.** For every pre-center/pre-scale plotted trajectory *i*, `bundle['predict']['forecasts'][i] == hyp.predict(bundle['trace_data'][i], model, t)`.
   - `xform_data` — the canonical **analysed pipeline output**, unchanged from v1.0.
   - `trace_data` — the final **pre-center/pre-scale plotted trajectories** (leaves, then per-level means, for a hierarchy).
   - `trace_data is xform_data` **only when no display-only projection occurred.** When a `reduce=` spec pins more than three components, the display-dimensionality enforcement at `plot.py:2886-2919` rebinds `xform` to a **new** list *after* `xform_data = copy.copy(xform)` at `plot.py:2827`, and the two diverge (verified: `reduce={'model':'PCA','args':[],'kwargs':{'n_components':5}}` on a `(60, 12)` array gives `xform_data` of shape `(60, 5)` while the drawn artist is 3-D).
   - **Bundled forecasts always correspond to `trace_data`.** They correspond to `xform_data` only when the two spaces coincide — which is the common case, so `plot.py:1935-1941`'s promise holds whenever it was true before.
   - For hierarchical input, the first `len(xform_data)` entries are the leaf forecasts and the remainder are the mean-trace forecasts. Any length mismatch raises instead of silently nulling.
   - **`predict=` is not defined for every hierarchy.** Every final trace needs ≥ 2 rows, on **either** axis: column hierarchies qualify whenever the input has at least two rows, and a row hierarchy qualifies only when every expanded leaf and every derived mean has ≥ 2 rows (Contract 10).
6. **Auxiliary per-trace values** (hue today, anything else later) are co-truncated with the data by the same operation, then checked by `assert_consistent()`.
7. **Hierarchical `hyp.predict`** has its own grouping rule, its own documented return shape, and explicit unfitted/fitted model ownership — independent of plot expansion.
8. **Dual-axis hierarchies are rejected** in 1.1, on both entry points. **Hierarchical frames inside lists** are rejected **asymmetrically** (*Decisions (resolved)* #1): `hyp.plot` rejects **column** hierarchies only and leaves row hierarchies on today's warn-and-flatten path; `hyp.predict` rejects **both** axes, where today's row behaviour is already an opaque `TypeError`.
9. **`hypertools/predict/` never imports from `hypertools/plot/`.** A test asserts it.
10. **A hierarchy is forecastable only when every final trace has at least 2 rows — on either axis.** `hyp.predict` cannot forecast a one-row trace (`predict/common.py:256`), so `plot(..., predict=...)` runs a **precondition check over every entry of `ft.arrays`** — leaves *and* derived means, for **both** axes — immediately after `ft` is built and **before** any call into `hyp.predict`. Both messages name the offending trace and its row count; only the **remediation** is axis-specific. *(Amended by the follow-up commit — see Task 8 correction 14. The remediation has a third, axis-INDEPENDENT case: the check runs on post-pipeline arrays, so when a row-count-changing analysis stage shortened the trace, the message names that stage instead of the grouping.)* **Row:** `expand_multiindex` makes one leaf per unique **full** row-index tuple, so a frame whose innermost level is unique per row yields one-row traces; the message explains that rule and offers `df.reset_index(drop=True)` or moving the grouping to the columns. **Column:** every group keeps all `len(df)` rows, so a one-row trace means the **input itself** has a single observation; the message says so and does **not** suggest flattening, which cannot add a row. Measured with this plan's own column-grouping idiom: a `T=1` frame gives leaf shapes `(1, 3)` and a mean of `(1, 3)` — not forecastable — while `T=2` gives `(2, 3)` throughout and forecasts. The precondition tests the **full** trace length, a permanent property of the data, so it runs for animated hierarchies too — **before** Plan 3's `ForecastSchedule` is built — and is not the same thing as that plan's per-frame `min_history` (Task 8, *Two short-history mechanisms*).
11. **Every leaf returned by a `hypertools/core/hierarchy.py` grouping helper is non-hierarchical on the axis it was grouped along** (*Revision note (v6)* **D1/D2**). This is what makes hierarchy handling terminate: both entry points detect a hierarchy by `nlevels >= 2` on an axis, and `hyp.predict` recurses into each group, so a leaf that still carries the grouping levels is re-detected and regrouped forever. **Columns:** `group_columns` flattens each leaf's columns to the innermost (feature) level, keeping that level's name — measured, `[('M','Tech','AAPL'), …]` becomes `['AAPL','MSFT','NVDA']` with `.name == 'Ticker'`. **Rows:** `group_rows_for_forecast` drops only the grouping levels, so the innermost level survives as a **flat, single-level** index with its own name and dtype (measured: a `DatetimeIndex` named `date`) — satisfying this contract *and* the datetime-preservation requirement (F5) at once, with no `RangeIndex` fallback. Re-running either helper on one of its own leaves is refused (`requires a … MultiIndex with 2 or more levels`), so there is no fixed point. Flattening **never mutates the caller's frame**, and duplicate flat feature labels are **permitted positionally** (*Revision note (v6)* **D3**). The plot-side `expand_multiindex` is exempt only because a Global Constraint freezes it — its leaves *are* a measured fixed point (`expand_multiindex(leaf0) is` structurally `leaf0`), so **no code path may feed a plot leaf back into a hierarchy-detecting entry point**; `hyp.predict`'s row path uses `group_rows_for_forecast` instead.

---

## File Structure

| file | responsibility | change |
|-|-|-|
| `hypertools/core/hierarchy.py` | **new** — axis-agnostic grouping + rejections, importable by predict and plot alike | create |
| `hypertools/plot/hierarchy.py` | **new** — `FinalTraces`, `build_hierarchy_traces`, `build_hierarchy_styles` (rendering only) | create |
| `hypertools/plot/multiindex.py` | `build_multiindex_styles` becomes a thin compatibility wrapper over the new pair; module docstring gains the column rule | modify |
| `hypertools/plot/plot.py` | dispatch, hue classification, forecast ordering, bundle keys, docstrings | modify |
| `hypertools/plot/plotly_backend.py` | parity for hierarchy + hue + forecast colour | modify |
| `hypertools/predict/predict.py` | hierarchical forecasting + model ownership | modify |
| `docs/hierarchy.rst` | **new** — the "Hierarchical DataFrames" guide | create |
| `docs/index.rst`, `docs/api.rst`, `docs/tutorials.rst`, `docs/pipeline_order.rst` | register + link the guide; split row/column in the feature list; hierarchy side branch | modify |
| `docs/_static/pipeline_order.svg`, `scripts/round17_evidence/pipeline_order_diagram.py` | regenerate the diagram with the hierarchy branch | modify |
| `CHANGELOG.md` | create `## 1.1.0 (unreleased)`; Added / Changed-validation / Limitations | modify |
| `tests/core/test_hierarchy_grouping.py` | grouping rules, leaf flatness (Contract 11), NA labels, time-index preservation, rejections (27) | create |
| `tests/plot/test_hierarchy_traces.py` | one-owner trace building, truncation, aux co-truncation (13) | create |
| `tests/plot/test_hierarchy_styles.py` | style-only styling; the one-level legend rule (6) | create |
| `tests/plot/test_hierarchy_bundle.py` | the return-bundle contract, **flat inputs only** (6) | create |
| `tests/plot/test_column_multiindex.py` | column expansion end-to-end, 2- and 3-level, **+ the hierarchical bundle** (17) | create |
| `tests/plot/test_multiindex_hue.py` | hue forms through a hierarchy (15) | create |
| `tests/predict/test_predict_multiindex.py` | hierarchical predict, model ownership, datetime horizons, the recursion guard (22) | create |
| `tests/plot/test_multiindex_predict.py` | `predict=` with expansion, bundle correspondence, the ≥ 2-row precondition on both axes (17) | create |
| `tests/plot/test_multiindex_plotly.py` | full backend parity (12) | create |
| `tests/test_docs_hierarchy_guide.py` | the guide exists, is linked, and covers its sections (8) | create |
| `tests/test_changelog_1_1.py` | the 1.1.0 section and its headings (6) | create |
| `tests/test_multiindex.py` | rewrite the **1** test pinning behaviour this plan deliberately changes (`:479`), and **add 1** for the one-row row hierarchy. `:453` and `:306` are **untouched** (29 → 30) | modify |

---

## Task 1: Shared grouping in `hypertools/core/hierarchy.py`

> **Executed, then amended by *Revision note (v8)*.** `group_columns` gained
> the `feature_correspondence` parameter and nominal cross-group matching
> (`_feature_keys` / `_match_features_by_name`), and `meta` gained
> `'feature_correspondence'`. The `market_frame` fixture below now names
> shared measurements; a `ticker_frame` fixture was added to pin the
> refusal of disjoint labels. The as-shipped tests are the authority:
> `tests/core/test_hierarchy_grouping.py`.

Define grouping once, for both axes and both consumers, **outside** `hypertools/plot/` so `hypertools/predict/` can import it without depending on plotting (F4). `expand_multiindex` is untouched.

**Rules (these are the contract):**
- **Column hierarchy (plot and predict):** the innermost column level is the FEATURE axis; every level above it is the grouping hierarchy. `(Market, Sector, Measure)` groups by `(Market, Sector)`, and **each leaf is actually flattened onto that feature axis** — its columns become the innermost level's values, carrying that level's name (`['return','volatility','momentum']`, `.name == 'Measure'`). Keeping the full tuples would leave the leaf hierarchical, which is exactly the v5 defect (*Revision note (v6)* **D1**).
- **Row hierarchy (plot):** unchanged — one leaf per unique full index tuple (`expand_multiindex`, `multiindex.py:76-81`). Note its leaves keep the full row MultiIndex and are a measured **fixed point**; they must never be fed back into a hierarchy-detecting entry point (**D2**).
- **Row hierarchy (predict):** the innermost level is the TIME/observation axis; group by every level above it, **dropping only those grouping levels so the innermost survives as the group's own FLAT (single-level) index**, keeping its name and dtype (F5). That single rule satisfies both the datetime-preservation requirement and the flatness invariant; there is no `RangeIndex` fallback.
- **Leaf flatness is an invariant, not an accident (Contract 11):** every leaf returned by a helper here is non-hierarchical on the axis it was grouped along, on **both** axes, and re-running the helper on a leaf is refused rather than nesting. Grouping **never mutates the caller's frame**.
- **Feature correspondence across groups is NOMINAL** (*Revision note (v8)* **F1**) — every group must carry the same innermost-label multiset, and later groups are permuted into the first group's order before analysis, so within-group column order is not part of the model. Groups with different labels (including groups of unequal width) are refused by name; `feature_correspondence='position'` is the explicit opt-in.
- **Duplicate innermost feature labels are permitted** — nothing downstream is name-addressed, and measured, all such columns survive, forecast and plot (**D3**) — and are matched across groups by `(label, occurrence)` (**F2**).
- **NA hierarchy labels never drop a group** — `dropna=False` everywhere (F6).
- **Dual-axis frames** and **hierarchical frames inside lists** are rejected (F7, F8, F10).

**Files:** Create `hypertools/core/hierarchy.py`; Test `tests/core/test_hierarchy_grouping.py`

**Interfaces:**
- Produces `group_columns(df)` → `(leaves, meta)`; `group_rows_for_forecast(df)` → `(groups, keys)`; `reject_dual_axis(df)` → `None`; `reject_hierarchical_in_list(x, caller, axes='columns'|'both')` → `None`; `is_hierarchical(obj, axes=)` → `bool`.
- **Every returned leaf/group satisfies Contract 11**: flat columns (innermost level, named) for `group_columns`, a flat index (innermost level, named, original dtype) for `group_rows_for_forecast`. Callers may therefore recurse into a leaf without re-detecting a hierarchy. Group labels come from `meta['leaf_keys']` / `keys` — **never** from a leaf's columns or index.
- `meta` matches `expand_multiindex`'s contract (`n_levels`, `leaf_keys`, `level_names`) plus `axis`, so the style layer consumes either without branching.
- Consumed by Tasks 5, 7, 8, 9.

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_hierarchy_grouping.py
"""Axis-agnostic hierarchy grouping (1.1, plan 2 Task 1).

Three DIFFERENT rules live here deliberately; conflating them was the main
defect in v1 of this plan. Every expectation below was measured on the
pandas 3.0.3 / numpy 2.3.5 in this repo's venv before it was written.
"""
import numpy as np
import pandas as pd
import pytest

from hypertools.core.hierarchy import (group_columns, group_rows_for_forecast,
                                       reject_dual_axis,
                                       reject_hierarchical_in_list)


def market_frame(T=120, seed=0):
    """rows = trading days, columns = (Market, Sector, Measure).

    The innermost level names SHARED measurements, not per-sector tickers:
    feature correspondence across groups is NOMINAL (Revision note (v8) F5).
    """
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], range(T)],
                                     names=['Sector', 'day'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


def dated_row_frame(T=30):
    days = pd.date_range('2020-01-01', periods=T)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], days],
                                     names=['Sector', 'date'])
    return pd.DataFrame(np.arange(2 * T * 3).reshape(2 * T, 3) * 1.0, index=idx)


# --- column rule: group by every level ABOVE the innermost -------------------

def test_columns_group_by_every_level_above_the_innermost():
    leaves, meta = group_columns(market_frame())
    assert len(leaves) == 3
    assert [k[-1] for k in meta['leaf_keys']] == ['Tech', 'Financials', 'Energy']
    assert all(leaf.shape == (120, 3) for leaf in leaves)


def test_column_leaves_are_flattened_to_the_innermost_feature_level():
    """Contract 11. `sub.T` alone keeps the FULL column MultiIndex, which
    (a) contradicts the feature-axis rule and (b) makes hyp.predict recurse
    without bound (Revision note (v6) D1). Measured before the fix:
    [('Market','Tech','AAPL'), ...], names ['Market','Sector','Ticker']."""
    leaves, _ = group_columns(market_frame())
    assert all(not isinstance(leaf.columns, pd.MultiIndex) for leaf in leaves)
    assert leaves[0].columns.tolist() == ['AAPL', 'MSFT', 'NVDA']
    assert leaves[0].columns.name == 'Ticker'
    assert all(leaf.columns.nlevels == 1 for leaf in leaves)


def test_group_columns_does_not_mutate_the_callers_frame():
    """`df.T` / `sub.T` may return a VIEW depending on the pandas version and
    copy-on-write state, so the leaf is built explicitly rather than by
    assigning `.columns` onto a possibly-aliasing transpose. An input-mutation
    bug here would be silent."""
    df = market_frame()
    before = df.columns.copy()
    before_values = df.to_numpy(copy=True)
    group_columns(df)
    assert df.columns.equals(before)
    assert list(df.columns.names) == ['Market', 'Sector', 'Ticker']
    assert isinstance(df.columns, pd.MultiIndex)
    assert np.array_equal(df.to_numpy(), before_values)


def test_column_meta_matches_the_expand_multiindex_contract():
    _, meta = group_columns(market_frame())
    assert set(meta) >= {'leaf_keys', 'level_names', 'n_levels', 'axis'}
    assert meta['level_names'] == ['Market', 'Sector']
    assert meta['n_levels'] == 2
    assert meta['axis'] == 'columns'


def test_column_groups_keep_first_appearance_order_when_unsorted():
    df = market_frame()
    shuffled = df.iloc[:, [8, 0, 4, 1, 5, 2, 6, 3, 7]]
    _, meta = group_columns(shuffled)
    assert meta['leaf_keys'][0] == ('Market', 'Energy')


def test_single_level_columns_are_rejected():
    df = pd.DataFrame(np.zeros((10, 3)), columns=['a', 'b', 'c'])
    with pytest.raises(ValueError, match='2 or more'):
        group_columns(df)


def test_duplicate_tickers_in_different_sectors_are_kept_separate():
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', 'X'), ('M', 'Tech', 'Y'),
         ('M', 'Energy', 'X'), ('M', 'Energy', 'Y')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.zeros((10, 4)), columns=cols)
    leaves, _ = group_columns(df)
    assert len(leaves) == 2 and all(leaf.shape[1] == 2 for leaf in leaves)


def test_duplicate_innermost_feature_names_are_kept_positionally():
    """DECIDED (Revision note (v6) D3): duplicates WITHIN a group are
    permitted. Flattening can collide two tickers (two share classes, a
    repeated sensor); everything downstream is positional, so nothing is
    dropped and rejecting would break legitimate frames. Measured: widths
    [3, 3], np.asarray -> (20, 3), predict -> (1, 3), plot -> Figure."""
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', 'AAPL'), ('M', 'Tech', 'AAPL'), ('M', 'Tech', 'NVDA'),
         ('M', 'Fin', 'JPM'), ('M', 'Fin', 'GS'), ('M', 'Fin', 'BAC')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 6)),
                      columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 2, 'duplicate names must not merge groups'
    assert [leaf.shape[1] for leaf in leaves] == [3, 3]
    assert leaves[0].columns.tolist() == ['AAPL', 'AAPL', 'NVDA']
    assert not leaves[0].columns.is_unique
    assert np.asarray(leaves[0]).shape == (20, 3), 'no column was dropped'
    assert np.allclose(np.asarray(leaves[0]), df.to_numpy()[:, :3])
    assert meta['leaf_keys'] == [('M', 'Tech'), ('M', 'Fin')]


def test_unnamed_levels_are_tolerated():
    cols = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'z')])
    df = pd.DataFrame(np.zeros((5, 3)), columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 2
    assert meta['level_names'] == [None]


def test_two_level_columns_give_one_leaf_per_group_and_no_mean_level():
    """(Group, Feature) -> n_levels == 1, so the style layer must NOT expect
    an aggregate mean (see Task 3)."""
    cols = pd.MultiIndex.from_tuples(
        [('A', 'f0'), ('A', 'f1'), ('B', 'f0'), ('B', 'f1'),
         ('C', 'f0'), ('C', 'f1')], names=['Group', 'Feature'])
    df = pd.DataFrame(np.zeros((10, 6)), columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert meta['n_levels'] == 1
    assert meta['leaf_keys'] == [('A',), ('B',), ('C',)]


def test_nan_in_an_outer_column_label_does_not_drop_the_group():
    """Measured: the pandas default drops it (3 groups -> 2)."""
    df = market_frame(T=20)
    tuples = [(np.nan if s == 'Energy' else m, s, t)
              for m, s, t in df.columns]
    df.columns = pd.MultiIndex.from_tuples(tuples,
                                           names=['Market', 'Sector', 'Ticker'])
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert any(isinstance(k[0], float) and np.isnan(k[0])
               for k in meta['leaf_keys'])


def test_nan_in_an_intermediate_column_label_does_not_drop_the_group():
    df = market_frame(T=20)
    tuples = [(m, np.nan if s == 'Energy' else s, t) for m, s, t in df.columns]
    df.columns = pd.MultiIndex.from_tuples(tuples,
                                           names=['Market', 'Sector', 'Ticker'])
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert any(isinstance(k[1], float) and np.isnan(k[1])
               for k in meta['leaf_keys'])


# --- forecasting rule: innermost level is TIME, and it SURVIVES --------------

def test_row_forecast_grouping_treats_the_innermost_level_as_time():
    groups, keys = group_rows_for_forecast(row_frame())
    assert len(groups) == 2, 'one group per sector, NOT one per (sector, day)'
    assert [k[0] for k in keys] == ['Tech', 'Energy']
    assert all(g.shape == (60, 3) for g in groups)


def test_row_forecast_grouping_preserves_the_time_index():
    """reset_index(drop=True) would discard this; hyp.predict needs it."""
    groups, _ = group_rows_for_forecast(row_frame())
    assert groups[0].index.name == 'day'
    assert list(groups[0].index[:5]) == [0, 1, 2, 3, 4]
    assert groups[0].index.nlevels == 1


def test_row_forecast_grouping_preserves_a_datetime_index():
    groups, _ = group_rows_for_forecast(dated_row_frame())
    idx = groups[0].index
    assert idx.name == 'date'
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx[0] == pd.Timestamp('2020-01-01')
    assert idx.is_monotonic_increasing and idx.is_unique


def test_row_groups_are_flat_and_keep_their_datetime_identity():
    """Contract 11 on the ROW axis, together with the F5 promise. `droplevel`
    delivers both at once: the grouping levels go, the innermost level stays
    as a FLAT index with its own name and dtype. Measured: DatetimeIndex,
    name 'date', datetime64[us], monotonic, unique, nlevels == 1."""
    groups, _ = group_rows_for_forecast(dated_row_frame())
    for g in groups:
        assert not isinstance(g.index, pd.MultiIndex)
        assert g.index.nlevels == 1
        assert isinstance(g.index, pd.DatetimeIndex)
        assert g.index.name == 'date'
        assert g.index.is_monotonic_increasing and g.index.is_unique


def test_regrouping_a_leaf_is_refused_on_both_axes():
    """The fixed point is gone. A leaf that still carried its grouping levels
    would be re-detected as hierarchical and regrouped forever -- measured on
    the plot rule, `expand_multiindex(leaf0)` returns leaf0 itself. Neither
    core helper does that: each refuses, because there is nothing left to
    group. This is the property hyp.predict's recursion relies on."""
    col_leaf = group_columns(market_frame())[0][0]
    assert col_leaf.columns.nlevels == 1
    with pytest.raises(ValueError, match='2 or more'):
        group_columns(col_leaf)

    row_leaf = group_rows_for_forecast(row_frame())[0][0]
    assert row_leaf.index.nlevels == 1
    with pytest.raises(ValueError, match='2 or more'):
        group_rows_for_forecast(row_leaf)


def test_row_forecast_grouping_differs_from_plot_expansion():
    """Documented divergence: plot leaves are full tuples, forecast groups
    drop the innermost (time) level."""
    from hypertools.plot.multiindex import expand_multiindex
    df = row_frame()
    plot_leaves, _ = expand_multiindex(df)
    forecast_groups, _ = group_rows_for_forecast(df)
    assert len(plot_leaves) == 120
    assert len(forecast_groups) == 2


def test_three_level_row_forecast_grouping():
    idx = pd.MultiIndex.from_product([['M'], ['Tech', 'Energy'], range(30)],
                                     names=['Market', 'Sector', 'day'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, keys = group_rows_for_forecast(df)
    assert len(groups) == 2 and keys[0] == ('M', 'Tech')
    assert groups[0].index.name == 'day'


def test_nan_in_an_outer_row_label_does_not_drop_the_group():
    df = row_frame(T=10)
    tuples = [(np.nan if s == 'Energy' else s, d) for s, d in df.index]
    df.index = pd.MultiIndex.from_tuples(tuples, names=['Sector', 'day'])
    groups, _ = group_rows_for_forecast(df)
    assert len(groups) == 2


def test_unsorted_times_are_detectable_on_the_returned_index():
    """Preserving the index is what makes predict/common.py:103-109's
    'not sorted in ascending order' warning fire per group (F8)."""
    days = pd.date_range('2020-01-01', periods=30)
    perm = np.random.default_rng(0).permutation(30)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 30 + ['Energy'] * 30, list(days[perm]) + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, _ = group_rows_for_forecast(df)
    assert not groups[0].index.is_monotonic_increasing
    assert groups[1].index.is_monotonic_increasing


def test_duplicate_times_are_detectable_on_the_returned_index():
    days = pd.date_range('2020-01-01', periods=30)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 30 + ['Energy'] * 30, list(days[:15]) * 2 + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, _ = group_rows_for_forecast(df)
    assert not groups[0].index.is_unique
    assert groups[1].index.is_unique


# --- rejections -------------------------------------------------------------

def test_dual_axis_frames_are_rejected():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(5)])
    cols = pd.MultiIndex.from_tuples([('M', 'Tech'), ('M', 'Energy')])
    df = pd.DataFrame(np.zeros((10, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        reject_dual_axis(df)


def test_single_axis_frames_pass_the_dual_axis_check():
    reject_dual_axis(market_frame())
    reject_dual_axis(row_frame())


def test_column_hierarchical_frame_in_a_list_is_rejected():
    """Both callers reject a COLUMN hierarchy nested in a list."""
    with pytest.raises(ValueError, match='element 1'):
        reject_hierarchical_in_list([np.zeros((5, 3)), market_frame()],
                                    caller='hyp.plot', axes='columns')
    with pytest.raises(ValueError, match='hyp.predict'):
        reject_hierarchical_in_list([market_frame()], caller='hyp.predict',
                                    axes='both')


def test_row_hierarchical_frame_in_a_list_is_rejected_for_predict_only():
    """The deliberate asymmetry (Decisions (resolved) #1): `hyp.plot` keeps
    today's warn-and-flatten for the ROW axis, pinned by
    tests/test_multiindex.py:453, so the check must let it through; for
    `hyp.predict` today's behaviour is an opaque pandas TypeError, so
    rejecting it is additive."""
    reject_hierarchical_in_list([row_frame()], caller='hyp.plot',
                                axes='columns')          # must NOT raise
    with pytest.raises(ValueError, match='hyp.predict'):
        reject_hierarchical_in_list([row_frame()], caller='hyp.predict',
                                    axes='both')


def test_flat_frames_in_a_list_pass():
    reject_hierarchical_in_list(
        [np.zeros((5, 3)), pd.DataFrame(np.zeros((5, 3)))], caller='hyp.plot',
        axes='columns')
    reject_hierarchical_in_list(market_frame(), caller='hyp.plot',
                                axes='columns')
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/core/test_hierarchy_grouping.py -v`
Expected: collection FAILS with `ModuleNotFoundError: No module named 'hypertools.core.hierarchy'`.

- [ ] **Step 3: Implement**

```python
# hypertools/core/hierarchy.py
#!/usr/bin/env python
"""Axis-agnostic hierarchy grouping, shared by `hyp.plot` and `hyp.predict`.

This module lives under `core/` rather than `plot/` on purpose: grouping a
frame by its outer index levels is not a rendering concern, and
`hypertools.predict` must never import from `hypertools.plot` (the 1.0
package split put shared machinery under `core/`).

Three DIFFERENT rules exist deliberately:

- `group_columns`            -- COLUMN hierarchy, plot AND predict: the
                                innermost level is the FEATURE axis,
                                everything above it groups.
- `group_rows_for_forecast`  -- ROW hierarchy, predict only: the innermost
                                level is the TIME axis, everything above it
                                groups, and the innermost level SURVIVES as
                                each group's index.
- ROW hierarchy, plot        -- keeps its historical rule (one leaf per
                                unique full index tuple) in
                                `hypertools.plot.multiindex.expand_multiindex`,
                                untouched.

ONE INVARIANT spans both helpers here: every leaf they return is
NON-HIERARCHICAL on the axis it was grouped along -- flat columns for
`group_columns`, a flat index for `group_rows_for_forecast`. Hierarchy is
detected by `nlevels >= 2`, and `hyp.predict` recurses into each group, so a
leaf that still carried its grouping levels would be regrouped forever. Note
`expand_multiindex` deliberately does NOT satisfy this (its leaves keep the
full row MultiIndex and re-expand to themselves -- a fixed point), which is
why its leaves must never be fed back into a hierarchy-detecting entry point.
Grouping never mutates the caller's frame.

See docs/hierarchy.rst for the user-facing comparison table.
"""

import numpy as np
import pandas as pd


def is_hierarchical(obj, axes='both'):
    """True when `obj` is a DataFrame carrying a MultiIndex on `axes`."""
    if not isinstance(obj, pd.DataFrame):
        return False
    if axes == 'rows':
        return obj.index.nlevels >= 2
    if axes == 'columns':
        return obj.columns.nlevels >= 2
    return obj.index.nlevels >= 2 or obj.columns.nlevels >= 2


def reject_dual_axis(df):
    """Refuse frames carrying a hierarchy on BOTH axes.

    Which hierarchy should win is genuinely ambiguous. Before 1.1 such a
    frame followed the ROW path and its column hierarchy was silently
    ignored; 1.1 declines to guess. This is an intentional compatibility
    change (see CHANGELOG 1.1.0, "Changed / validation").
    """
    if (isinstance(df, pd.DataFrame)
            and df.index.nlevels >= 2 and df.columns.nlevels >= 2):
        raise ValueError(
            "x has both a row and a column MultiIndex. hypertools 1.1 does "
            "not define which hierarchy takes precedence. Flatten one axis "
            "(e.g. df.reset_index(drop=True), or "
            "df.columns = df.columns.map('_'.join)) and try again.")


def reject_hierarchical_in_list(x, caller, axes='columns'):
    """Refuse a hierarchical DataFrame nested inside a list/tuple.

    Hierarchy expansion is defined for a BARE frame only: the hierarchy
    determines the whole trace/group list, which cannot be reconciled with
    a caller-supplied list of datasets.

    `axes` is DELIBERATELY asymmetric between the two callers (CHANGELOG
    1.1.0, "Changed / validation"):

    - ``'columns'`` (``hyp.plot``): reject a COLUMN hierarchy only. Before
      1.1 it flattened to a single line, silently -- nothing pinned it, so
      rejecting it is purely additive. A ROW hierarchy in a list keeps its
      documented warn-and-flatten path ("MultiIndex grouping is only
      applied..."), which `tests/test_multiindex.py:453` pins and 1.1 does
      not change.
    - ``'both'`` (``hyp.predict``): reject either axis. There is nothing to
      preserve: a row-hierarchical frame in a list raises `TypeError:
      cannot perform __sub__ with this index type: MultiIndex` deep inside
      pandas today, and a column-hierarchical one silently forecasts the
      flattened frame.
    """
    if not isinstance(x, (list, tuple)):
        return
    if axes not in ('columns', 'both'):
        raise ValueError(f"axes= must be 'columns' or 'both'; got {axes!r}")
    for i, element in enumerate(x):
        if not isinstance(element, pd.DataFrame):
            continue
        row_hier = element.index.nlevels >= 2
        col_hier = element.columns.nlevels >= 2
        if col_hier or (row_hier and axes == 'both'):
            axis = ('row' if row_hier else 'column')
            raise ValueError(
                f"{caller} received a list whose element {i} is a DataFrame "
                f"with a {axis} MultiIndex. Hierarchy expansion is defined "
                "for a BARE DataFrame only, because the hierarchy determines "
                "the entire group list. Pass the frame on its own "
                f"({caller}(df, ...)), or flatten it first "
                "(df.reset_index(drop=True), or "
                "df.columns = df.columns.map('_'.join)).")


def group_columns(df):
    """Group a column-hierarchical frame into one leaf per group.

    The innermost column level is the FEATURE axis; every level above it is
    the grouping hierarchy. Returns ``(leaves, meta)`` with `meta` shaped
    exactly like `expand_multiindex`'s (plus ``'axis'``), so the style layer
    consumes either without branching.

    Each leaf is FLATTENED onto the feature axis (Contract 11): its columns
    are the innermost level's values, carrying that level's name. Keeping the
    caller's full tuples would leave the leaf hierarchical, so `hyp.predict`'s
    per-group recursion would re-detect it and regroup without bound.

    Duplicate flattened labels are permitted and are matched ACROSS GROUPS
    by (label, occurrence) --
    two share classes of one issuer, or a repeated sensor name, are legitimate
    inputs and nothing downstream is name-addressed. Group labels come from
    `meta['leaf_keys']`, never from a leaf's columns.
    """
    reject_dual_axis(df)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError(
            "group_columns requires a column MultiIndex with 2 or more "
            f"levels; got {df.columns.nlevels} level(s).")

    group_levels = list(range(df.columns.nlevels - 1))
    feature_name = df.columns.names[-1]
    leaves, leaf_keys = [], []
    # `df.groupby(..., axis=1)` was REMOVED in pandas 3 (TypeError on the
    # 3.0.3 in this venv), so group the transpose and transpose each group
    # back. sort=False preserves first-appearance order, which `leaf_keys`
    # and the palette both depend on. dropna=False keeps groups whose
    # hierarchy LABEL is missing -- measured on pandas 3.0.3, the default
    # silently turns 3 groups into 2.
    for key, sub in df.T.groupby(level=group_levels, sort=False, dropna=False):
        # COPY FIRST, then flatten. `sub.T` may be a VIEW onto the caller's
        # frame depending on the pandas version and copy-on-write state, so
        # assigning `.columns` to it directly risks silently rewriting the
        # input's columns. (Equivalent and also acceptable: build it outright,
        # `pd.DataFrame(sub.T.to_numpy(), index=sub.columns, columns=flat)`.)
        leaf = sub.T.copy()
        # Contract 11: flatten onto the FEATURE axis. Without this the leaf
        # keeps the full ('Market', 'Sector', 'Ticker') tuples, contradicting
        # the feature-axis rule and making hyp.predict recurse without bound
        # (Revision note (v6) D1). Duplicates in the flattened labels are
        # fine -- see the docstring.
        leaf.columns = leaf.columns.get_level_values(-1)
        leaf.columns.name = feature_name
        leaves.append(leaf)
        leaf_keys.append(key if isinstance(key, tuple) else (key,))

    return leaves, {
        'n_levels': len(group_levels),
        'leaf_keys': leaf_keys,
        'level_names': list(df.columns.names[:-1]),
        'axis': 'columns',
    }


def group_rows_for_forecast(df):
    """Group a row-hierarchical frame for forecasting, KEEPING the time axis.

    The innermost row level is the TIME/observation axis, so grouping uses
    every level above it: a ``(Sector, day)`` index yields one group per
    SECTOR, each a full time series **still indexed by ``day``**. Only the
    grouping levels are dropped -- `reset_index(drop=True)` would discard
    the timestamps `hyp.predict` needs for a datetime-like `t` (see
    `hypertools/predict/common.py`).

    This is intentionally NOT `expand_multiindex`, whose plotting rule is one
    leaf per unique full tuple -- that rule makes every ``(Sector, day)``
    pair its own one-row leaf, which cannot be forecast, and whose leaves
    keep the full row MultiIndex (measured: re-expanding one returns itself).

    `droplevel` is what makes this helper satisfy Contract 11 on the row axis
    AND keep the datetime promise at the same time: only the grouping levels
    go, so the surviving index is FLAT and still carries its own name and
    dtype. No `RangeIndex` fallback is used -- for forecasting the innermost
    level IS the time axis, and replacing a non-monotonic or duplicated one
    with positions would hide exactly what the warning/rejection below is for.
    """
    reject_dual_axis(df)
    if df.index.nlevels < 2:
        raise ValueError(
            "group_rows_for_forecast requires a row MultiIndex with 2 or "
            f"more levels; got {df.index.nlevels} level(s).")

    group_levels = list(range(df.index.nlevels - 1))
    groups, keys = [], []
    for key, sub in df.groupby(level=group_levels, sort=False, dropna=False):
        # droplevel, NOT reset_index: the innermost level survives as this
        # group's index, carrying its name and dtype (verified: a datetime
        # innermost level comes back as a DatetimeIndex named 'date').
        groups.append(sub.droplevel(group_levels))
        keys.append(key if isinstance(key, tuple) else (key,))
    return groups, keys
```

- [ ] **Step 4: Run and confirm pass**

Run: `.venv/bin/python -m pytest tests/core/test_hierarchy_grouping.py -v`
Expected: **27 passed.**

- [ ] **Step 5: Assert the layering rule holds**

Run:
```bash
.venv/bin/python -c "
import ast, pathlib
bad = []
for p in pathlib.Path('hypertools/predict').rglob('*.py'):
    src = p.read_text()
    if '.plot' in src and 'hypertools.plot' in src or 'from ..plot' in src:
        bad.append(str(p))
print('predict -> plot imports:', bad)
assert not bad, bad
"
```
Expected: `predict -> plot imports: []`.

- [ ] **Step 6: Commit**

```bash
git add hypertools/core/hierarchy.py tests/core/test_hierarchy_grouping.py
git commit -m "feat(core): axis-agnostic hierarchy grouping shared by plot and predict"
```

---

## Task 2: One authoritative final-trace builder

`build_multiindex_styles` currently does two jobs (F1): it appends the per-level mean arrays (`multiindex.py:197-229`) **and** builds their styles. Task 2 moves job 1 into `build_hierarchy_traces`, the **only** place a mean is ever constructed, unequal lengths are ever truncated, and the truncation warning is ever emitted. `build_multiindex_styles` survives as a thin compatibility wrapper so the five direct-import tests in `tests/test_multiindex.py` stay green unchanged.

**Files:** Create `hypertools/plot/hierarchy.py`; Modify `hypertools/plot/multiindex.py`; Test `tests/plot/test_hierarchy_traces.py`

**Interfaces:**
- Produces `FinalTraces` — `arrays: list[np.ndarray]`, `keys: list[tuple]`, `level_idx: list[int]`, `is_mean: list[bool]`, `aux: list | None`, `meta: dict`, and `assert_consistent(**named_sequences)` raising `ValueError` naming any sequence whose length differs.
- **`arrays` elements are `np.ndarray`, never `DataFrame` — this is what makes Contract 11's "never fed back into a hierarchy-detecting entry point" rule hold *by construction* rather than by discipline.** `expand_multiindex`'s row leaves are DataFrames carrying the full row MultiIndex and are a measured fixed point (**D2**); if one reached `hyp.predict` via `ft.arrays` it would be re-detected as hierarchical and regrouped forever. `build_hierarchy_traces` therefore coerces every leaf with `np.asarray(...)` on the way in, and Task 8 forecasts over `ft.arrays` — so no index or column labels survive to be re-detected. Pin it: `assert all(isinstance(a, np.ndarray) for a in ft.arrays)`, and assert the same for the row-hierarchy path specifically, where the input leaves *are* DataFrames.
- Produces `build_hierarchy_traces(leaf_arrays, meta, aux=None)` → `FinalTraces`.
- `level_idx[i]` is the depth used by the documented style formulas: `L - 1` for a leaf, `k` for a mean over levels `0..k`. Leaves come first (in `leaf_keys` order), then means deepest-first, so the top-level mean is last — the order `tests/test_multiindex.py` already pins.
- Consumed by Tasks 3, 4, 5, 6, 8, 9.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_hierarchy_traces.py
"""`build_hierarchy_traces` is the ONE owner of final-trace construction.

Before 1.1, `build_multiindex_styles` (multiindex.py:197-229) both appended
the per-level means and styled them. Any second builder would append every
mean twice -- `test_every_expected_mean_appears_exactly_once` is the
regression test for exactly that.
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

from hypertools.plot.hierarchy import FinalTraces, build_hierarchy_traces
from hypertools.plot.multiindex import build_multiindex_styles, expand_multiindex

COL_META = {'n_levels': 2, 'axis': 'columns', 'level_names': ['Market', 'Sector'],
            'leaf_keys': [('M', 'Tech'), ('M', 'Fin'), ('M', 'Energy')]}
ROW_META = {'n_levels': 3, 'axis': 'rows', 'level_names': ['grp', 'cond', 'subj'],
            'leaf_keys': [('X', 'A', 'S0'), ('X', 'A', 'S1'),
                          ('X', 'B', 'S0'), ('X', 'B', 'S1'),
                          ('Y', 'A', 'S0'), ('Y', 'A', 'S1'),
                          ('Y', 'B', 'S0'), ('Y', 'B', 'S1')]}


def _leaves(n, rows=5, cols=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(rows, cols)) for _ in range(n)]


def test_leaves_come_first_then_means_shallowest_last():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    assert ft.is_mean == [False, False, False, True]
    assert ft.level_idx == [1, 1, 1, 0]
    assert ft.keys[-1] == ('M',)


def test_every_expected_mean_appears_exactly_once():
    """THE F1 regression test. A 3-level tree has 4 (grp, cond) means and 2
    grp means; each must be built once and only once."""
    ft = build_hierarchy_traces(_leaves(8), ROW_META)
    assert len(ft.arrays) == 8 + 4 + 2
    mean_keys = [k for k, m in zip(ft.keys, ft.is_mean) if m]
    assert mean_keys == [('X', 'A'), ('X', 'B'), ('Y', 'A'), ('Y', 'B'),
                         ('X',), ('Y',)]
    assert len(mean_keys) == len(set(mean_keys))


def test_two_level_hierarchy_builds_no_mean():
    """n_levels == 1: one leaf per group, and NO aggregate level exists."""
    meta = {'n_levels': 1, 'axis': 'columns', 'level_names': ['Group'],
            'leaf_keys': [('A',), ('B',), ('C',)]}
    ft = build_hierarchy_traces(_leaves(3), meta)
    assert len(ft.arrays) == 3
    assert ft.is_mean == [False, False, False]
    assert ft.level_idx == [0, 0, 0]


def test_means_equal_numpy_mean_of_their_members():
    leaves = _leaves(3)
    ft = build_hierarchy_traces(leaves, COL_META)
    assert np.array_equal(ft.arrays[3], np.mean(np.stack(leaves), axis=0))


def test_unequal_length_members_are_truncated_to_the_overlap():
    leaves = _leaves(3)
    leaves[2] = leaves[2][:3]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ft = build_hierarchy_traces(leaves, COL_META)
    assert ft.arrays[3].shape == (3, 2)
    expected = np.mean(np.stack([leaf[:3] for leaf in leaves]), axis=0)
    assert np.array_equal(ft.arrays[3], expected)


def test_unequal_length_warning_is_emitted_exactly_once():
    """One underlying issue, one warning -- the dedup multiindex.py:189-196
    already documents, now owned here."""
    leaves = _leaves(8)
    leaves[0] = leaves[0][:3]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_hierarchy_traces(leaves, ROW_META)
    unequal = [w for w in caught if 'unequal-length' in str(w.message)]
    assert len(unequal) == 1


def test_aux_arrays_are_co_truncated_with_the_data():
    """Contract 6: hue must never drift out of step with its trace."""
    leaves = _leaves(3)
    leaves[2] = leaves[2][:3]
    aux = [np.arange(5.0), np.arange(5.0), np.arange(3.0)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ft = build_hierarchy_traces(leaves, COL_META, aux=aux)
    for arr, a in zip(ft.arrays, ft.aux):
        assert len(arr) == len(a)
    assert len(ft.aux[3]) == 3


def test_mean_aux_is_the_mean_of_its_members_aux():
    aux = [np.array([1.0, 2, 3, 4, 5]), np.array([3.0, 4, 5, 6, 7]),
           np.array([5.0, 6, 7, 8, 9])]
    ft = build_hierarchy_traces(_leaves(3), COL_META, aux=aux)
    assert np.allclose(ft.aux[3], np.mean(np.stack(aux), axis=0))


def test_aux_is_none_when_no_aux_is_supplied():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    assert ft.aux is None


def test_assert_consistent_names_the_offending_sequence():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    with pytest.raises(ValueError, match='forecasts'):
        ft.assert_consistent(forecasts=[None])


def test_assert_consistent_reports_both_lengths():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    with pytest.raises(ValueError, match='4.*1|1.*4'):
        ft.assert_consistent(forecasts=[None])


def test_assert_consistent_passes_on_matching_lengths():
    ft = build_hierarchy_traces(_leaves(3), COL_META)
    ft.assert_consistent(forecasts=[None] * 4, hues=[None] * 4)


def test_the_legacy_wrapper_appends_each_mean_exactly_once():
    """`build_multiindex_styles` keeps its (arrays, style) contract, but now
    delegates: no second mean-construction path exists."""
    leaves = _leaves(8)
    arrays, style = build_multiindex_styles(leaves, ROW_META)
    assert len(arrays) == 8 + 4 + 2 == len(style['linewidths'])
    assert style['linewidths'] == [1.0] * 8 + [2.0] * 4 + [3.0] * 2


def test_final_trace_arrays_are_plain_ndarrays_even_for_row_dataframe_leaves():
    """Contract 11 must hold BY CONSTRUCTION, not by discipline.

    `expand_multiindex` hands back real `DataFrame`s whose INDEX is still the
    full row MultiIndex, and those are a measured fixed point (D2): re-expanding
    one returns exactly itself. If such a leaf survived into `ft.arrays` it would
    reach `hyp.predict` in Task 8, be re-detected as hierarchical, and regroup
    forever. `build_hierarchy_traces` therefore coerces with `np.asarray`, which
    drops the index and column labels so there is nothing left to re-detect.

    This test exists because an earlier revision stated the ndarray contract in
    prose only -- the implementation still said `arrays = list(leaf_arrays)`,
    which preserves DataFrames, and nothing pinned it.
    """
    idx = pd.MultiIndex.from_tuples(
        [(c, s) for c in ('condA', 'condB') for s in ('S0', 'S1')
         for _ in range(5)], names=['cond', 'subj'])
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)),
                         index=idx, columns=['x', 'y', 'z'])

    leaves, meta = expand_multiindex(frame)
    # the premise: the leaves really are hierarchy-carrying DataFrames
    assert all(isinstance(leaf, pd.DataFrame) for leaf in leaves)
    assert all(isinstance(leaf.index, pd.MultiIndex) for leaf in leaves)

    ft = build_hierarchy_traces(leaves, meta)

    # every trace -- leaves AND derived means -- is a plain ndarray
    assert all(isinstance(arr, np.ndarray) for arr in ft.arrays)
    assert not any(isinstance(arr, pd.DataFrame) for arr in ft.arrays)
    assert all(not isinstance(arr, np.ndarray) or arr.ndim == 2
               for arr in ft.arrays)
    # the means specifically (they are built here, not passed in)
    assert all(isinstance(ft.arrays[i], np.ndarray)
               for i, m in enumerate(ft.is_mean) if m)

    # and the caller's frames are untouched
    assert isinstance(frame.index, pd.MultiIndex)
    assert frame.columns.tolist() == ['x', 'y', 'z']
    assert all(isinstance(leaf, pd.DataFrame) for leaf in leaves)
    assert all(isinstance(leaf.index, pd.MultiIndex) for leaf in leaves)
```

> **On `.copy()`:** deliberately not used — see Step 3. The tested contract is that trace leaves are
> plain `ndarray`s and that nothing here mutates its inputs; that holds on every supported pandas.
> (Separately, on pandas 3.0.3 `np.asarray(df)` happens to return a read-only view, so a stray write
> would raise — version-specific behaviour, not something to depend on.) Do not add `.copy()` to make
> this test pass; it already passes without one.

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_hierarchy_traces.py -v`
Expected: collection FAILS with `ModuleNotFoundError: No module named 'hypertools.plot.hierarchy'`.

- [ ] **Step 3: Implement `hypertools/plot/hierarchy.py`**

Create the module with:

```python
#!/usr/bin/env python
"""Final-trace construction for hierarchical plots.

`build_hierarchy_traces` is the SINGLE owner of:
  * per-level mean construction,
  * truncation of unequal-length members to their overlapping prefix,
  * the one aggregated unequal-length warning,
  * co-truncation of any auxiliary per-observation values (hue).

Nothing else in hypertools may append a mean trace. Grouping lives in
`hypertools.core.hierarchy`; styling lives in `build_hierarchy_styles`
below, which consumes this module's METADATA and never sees leaf arrays.
"""
```

`FinalTraces` is a dataclass with fields `arrays`, `keys`, `level_idx`, `is_mean`, `aux`, `meta`, plus:

```python
    def assert_consistent(self, **named_sequences):
        """Raise naming any sequence whose length != len(self.arrays)."""
        n = len(self.arrays)
        for name, seq in named_sequences.items():
            if seq is not None and len(seq) != n:
                raise ValueError(
                    f"hierarchy trace/{name} mismatch: {n} traces but "
                    f"{len(seq)} {name}. Every per-trace sequence must be "
                    "built from the same FinalTraces "
                    "(hypertools/plot/hierarchy.py).")
```

`build_hierarchy_traces(leaf_arrays, meta, aux=None)` moves the loop currently at `multiindex.py:197-229` verbatim, with **four** changes: it appends to `keys`/`level_idx`/`is_mean` instead of calling `_append_style`; it applies the **same** `min_len` slice to `aux` members as to data members (Contract 6); it **coerces every leaf to a plain `np.ndarray` on the way in** (below); and it returns a `FinalTraces`. Leaves get `level_idx = n_levels - 1`, `is_mean = False`, `keys = meta['leaf_keys']`; means get `level_idx = k`, `is_mean = True`, `keys = prefix`.

**The coercion is the fourth change, and it is load-bearing — do not carry the old line across.** The existing loop begins by preserving whatever it was handed:

```python
arrays = list(leaf_arrays)          # WRONG here: preserves DataFrames
```

Replace it with:

```python
arrays = [np.asarray(leaf) for leaf in leaf_arrays]
```

This is what makes Contract 11 hold **by construction** rather than by discipline. `expand_multiindex`'s row leaves are `DataFrame`s carrying the full row MultiIndex and are a measured fixed point (**D2**); if one survived into `ft.arrays` it would reach `hyp.predict` in Task 8 and be re-detected as hierarchical, regrouping forever. `np.asarray` drops the index and column labels, so there is nothing left to re-detect.

Use `np.asarray(leaf)`, **not** an unconditional `np.asarray(leaf).copy()`. The contract this plan actually relies on — and tests — is narrow: **final trace leaves are plain `ndarray`s, and nothing in the current processing chain mutates its inputs.** That holds on every supported pandas.

An *observation* under the installed pandas (**3.0.3**), which is why no defensive copy is added:

```
np.shares_memory(np.asarray(df), df.to_numpy(copy=False))  ->  True
np.asarray(df)[0, 0] = 999.0  ->  ValueError: assignment destination is read-only
```

Under this version copy-on-write makes the result a non-writeable view, so an accidental in-place write **raises loudly** rather than silently corrupting the caller. Treat that as version-specific behaviour observed here, **not** a guarantee across the supported pandas range — do not write code that depends on the array being read-only, and do not rely on a write raising. An unconditional `.copy()` would also double peak memory on the 2513×24 market frame for no gain against the actual contract.

**What this means in practice:** on pandas 3.0.3, `ft.arrays` members derived from DataFrame leaves are read-only. Every consumer in this plan (mean construction via `np.mean`, `min_len` slicing, styling) only reads or allocates, so writeability never arises. *Any* future task that needs to write into an `ft.arrays` member must copy at that point and say why, rather than depending on whichever behaviour the installed pandas happens to give.

- [ ] **Step 4: Reduce `build_multiindex_styles` to a wrapper**

Replace its body (`multiindex.py:90-241`) with a delegation that keeps its exact `(arrays, style)` return contract:

```python
def build_multiindex_styles(leaf_arrays, meta, palette='hls', linestyle=None,
                            linestyles=None):
    """Deprecated internal shim: build traces, then style them.

    Kept because `tests/test_multiindex.py` imports it directly and pins its
    `(arrays, style)` contract. New code calls `build_hierarchy_traces` and
    `build_hierarchy_styles` separately -- see hypertools/plot/hierarchy.py.
    """
    from .hierarchy import build_hierarchy_styles, build_hierarchy_traces
    ft = build_hierarchy_traces(leaf_arrays, meta)
    return ft.arrays, build_hierarchy_styles(ft, palette=palette,
                                             linestyle=linestyle,
                                             linestyles=linestyles)
```

Keep the leaf-count validation (`test_build_styles_mismatched_leaf_count_raises`) in `build_hierarchy_traces`, where the leaves are.

- [ ] **Step 5: Run and confirm pass**

Run: `.venv/bin/python -m pytest tests/plot/test_hierarchy_traces.py tests/test_multiindex.py -v`
Expected: **14 passed** in the new module; **29 passed** in `tests/test_multiindex.py`, unchanged.

- [ ] **Step 6: Run the WHOLE suite (central dispatch changed)**

Run: `.venv/bin/python -m pytest -q`
Expected: baseline + 41 (Task 1's 27 + this task's 14).

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/hierarchy.py hypertools/plot/multiindex.py \
        tests/plot/test_hierarchy_traces.py
git commit -m "refactor(plot): build_hierarchy_traces owns mean construction and truncation"
```

---

## Task 3: Style-only hierarchy styling, with the two-level legend rule

`build_hierarchy_styles(trace_meta)` receives a `FinalTraces` and **never sees a leaf array**, so it cannot append anything (F1). It also fixes the `n_levels == 1` case: measured today, a `(Group, Feature)` hierarchy yields `labels ['_nolegend_'] * 3` — three completely unlabelled traces (F11). With no mean to carry the legend, **each leaf is the top level and gets its own label**.

**Files:** Modify `hypertools/plot/hierarchy.py`; Test `tests/plot/test_hierarchy_styles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_hierarchy_styles.py
"""Styling consumes trace METADATA only, and never builds a trace.

Formulas (multiindex.py:14-38): linewidth = 1 + (L - 1 - level_idx);
alpha = min(1.0, 1 / (level_idx + 1) + 0.2); colour by the TOP level; only
the top-level mean is labelled -- EXCEPT when n_levels == 1, where no mean
exists and every leaf is itself top-level.
"""
import matplotlib
matplotlib.use("Agg")

import inspect

import numpy as np
import pytest

from hypertools.plot.hierarchy import build_hierarchy_styles, build_hierarchy_traces

COL_META = {'n_levels': 2, 'axis': 'columns', 'level_names': ['Market', 'Sector'],
            'leaf_keys': [('M', 'Tech'), ('M', 'Fin'), ('M', 'Energy')]}
ROW_META = {'n_levels': 3, 'axis': 'rows', 'level_names': ['grp', 'cond', 'subj'],
            'leaf_keys': [('X', 'A', 'S0'), ('X', 'A', 'S1'),
                          ('X', 'B', 'S0'), ('X', 'B', 'S1'),
                          ('Y', 'A', 'S0'), ('Y', 'A', 'S1'),
                          ('Y', 'B', 'S0'), ('Y', 'B', 'S1')]}
ONE_META = {'n_levels': 1, 'axis': 'columns', 'level_names': ['Group'],
            'leaf_keys': [('A',), ('B',), ('C',)]}


def _traces(meta, n):
    return build_hierarchy_traces([np.zeros((5, 2))] * n, meta)


def test_style_formulas_match_the_documented_contract_two_levels():
    style = build_hierarchy_styles(_traces(COL_META, 3))
    assert style['linewidths'] == [1.0, 1.0, 1.0, 2.0]
    assert style['alphas'] == pytest.approx([0.7, 0.7, 0.7, 1.0])


def test_style_formulas_match_the_documented_contract_three_levels():
    style = build_hierarchy_styles(_traces(ROW_META, 8))
    assert style['linewidths'] == [1.0] * 8 + [2.0] * 4 + [3.0] * 2
    assert style['alphas'] == pytest.approx(
        [1 / 3 + 0.2] * 8 + [0.7] * 4 + [1.0] * 2)


def test_only_the_top_level_mean_is_labelled_when_means_exist():
    style = build_hierarchy_styles(_traces(ROW_META, 8))
    assert style['labels'] == ['_nolegend_'] * 12 + ['X', 'Y']


def test_one_level_hierarchy_labels_every_leaf():
    """F11: with no mean, three unlabelled traces was the bug."""
    style = build_hierarchy_styles(_traces(ONE_META, 3))
    assert style['labels'] == ['A', 'B', 'C']
    assert style['linewidths'] == [1.0, 1.0, 1.0]
    assert style['alphas'] == pytest.approx([1.0, 1.0, 1.0])


def test_one_level_hierarchy_gives_each_leaf_its_own_colour():
    style = build_hierarchy_styles(_traces(ONE_META, 3))
    assert len(set(style['colors'])) == 3
    assert style['unique_top'] == ['A', 'B', 'C']


def test_styles_take_metadata_not_leaf_arrays():
    """Structural guarantee for F1: the styler has no leaves to average."""
    params = list(inspect.signature(build_hierarchy_styles).parameters)
    assert params[0] == 'traces'
    assert 'leaf_arrays' not in params
    ft = _traces(COL_META, 3)
    ft.arrays = None                      # styling must not need them
    assert build_hierarchy_styles(ft)['linewidths'] == [1.0, 1.0, 1.0, 2.0]
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_hierarchy_styles.py -v`
Expected: `ImportError: cannot import name 'build_hierarchy_styles'`.

- [ ] **Step 3: Implement**

Add to `hypertools/plot/hierarchy.py`:

```python
def build_hierarchy_styles(traces, palette='hls', linestyle=None,
                           linestyles=None):
    """Per-trace color/linewidth/alpha/linestyle/label from trace METADATA.

    Consumes a `FinalTraces`' `keys`, `level_idx` and `is_mean` -- never its
    arrays -- so it structurally cannot construct or append a trace.

    Returns the same dict `build_multiindex_styles` always returned:
    ``{'colors', 'linewidths', 'alphas', 'labels', 'linestyles', 'n_top',
    'unique_top'}``.

    Label rule. Only the TOP-level mean (`level_idx == 0` and `is_mean`)
    carries a legend label. When ``meta['n_levels'] == 1`` there IS no mean:
    each leaf is itself a top-level group, so each carries its own label.
    Without this, a two-level (Group, Feature) column hierarchy drew several
    completely unlabelled traces.
    """
    L = traces.meta['n_levels']
    ...
    labels = []
    for key, level_idx, is_mean in zip(traces.keys, traces.level_idx,
                                       traces.is_mean):
        top_level = (level_idx == 0) and (is_mean or L == 1)
        labels.append(str(key[0]) if top_level else '_nolegend_')
```

Colour, linewidth, alpha, linestyle cycling and `unique_top`/`n_top` move across from `multiindex.py` unchanged (they already key off the top-level value and `level_idx`).

- [ ] **Step 4: Run and confirm pass**

Run: `.venv/bin/python -m pytest tests/plot/test_hierarchy_styles.py tests/plot/test_hierarchy_traces.py tests/test_multiindex.py -v`
Expected: **6 passed**, **14 passed**, **29 passed**.

- [ ] **Step 5: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. Expected: baseline + 47 (27 + 14 + 6).

- [ ] **Step 6: Document** the `n_levels == 1` label rule in the `multiindex.py` module docstring (`multiindex.py:14-38`), beside the existing 2- and 3-level worked examples.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/hierarchy.py hypertools/plot/multiindex.py \
        tests/plot/test_hierarchy_styles.py
git commit -m "refactor(plot): style-only hierarchy styling; label every leaf of a one-level hierarchy"
```

---

## Task 4: The return-bundle contract (flat inputs)

**Scope: FLAT inputs only.** This task adds the two new bundle keys and proves them on non-hierarchical input, so its verification step **passes standalone**. Every hierarchical bundle assertion lives in Task 5 (v3 had them here, with an "ordering note" admitting Tasks 4 and 5 were mutually dependent; a note is not a task boundary, so the dependency is now a plain `4 → 5`).

`xform_data` keeps its v1.0 meaning (F3). `plot.py:2827` captures it immediately after `analyze()`, and `plot.py:1935-1941` promises each bundled forecast "has exactly `t` rows, matching what `hyp.predict(xform_data, model=..., t=t)` returns". Derived means are built later, in display space, from data the returned `pipeline` cannot reproduce with `.transform()` — so they must not enter `xform_data`.

**The contract (documented in `plot()`'s `return_model` entry, `plot.py:1920-1941`):**

```python
{
    'fig': ...,
    'xform_data': [...],      # UNCHANGED: analyzed output, one entry per analysed INPUT dataset
    'trace_data': [...],      # NEW: the final PRE-CENTER/PRE-SCALE plotted trajectories,
                              #      in drawn order (for a hierarchy: leaves, then means)
    'trace_metadata': {...},  # NEW: {'keys', 'level_idx', 'is_mean', 'axis', 'level_names', 'aux'}
    'animation': ..., 'pipeline': ..., 'models': {...},
    'predict': {'model': ..., 'params': {'t': t}, 'forecasts': [...]},
}
```

**Reconciling with the documented promise (Contract 5).** For every pre-center/pre-scale plotted trajectory *i*, `forecasts[i] == hyp.predict(trace_data[i], model, t)`. The two keys mean different things:

- **`xform_data`** — the canonical analysed pipeline output, unchanged from v1.0.
- **`trace_data`** — the final **pre-center/pre-scale plotted trajectories**. The artists drawn on the axes are *not* these arrays: they are centered, scaled and (unless `antialias=False`) PCHIP-upsampled afterwards, which is why "drawn trace" is the wrong word for `trace_data`.

`trace_data is xform_data` **only when no display-only projection occurred** — the usual case, which is why `plot.py:1935-1941` keeps holding wherever it held before. It is **not** universal, and the counterexample is a *flat* input: `xform_data = copy.copy(xform)` at `plot.py:2827` runs **before** the display-dimensionality enforcement at `plot.py:2886-2919`, and that block **rebinds** `xform` to a new list, leaving `xform_data` holding the pre-projection arrays. Verified: `hyp.plot(X_60x12, reduce={'model': 'PCA', 'args': [], 'kwargs': {'n_components': 5}}, show=False, return_model=True)` gives `xform_data[0].shape == (60, 5)` while the drawn artist is 3-D (with 945 points, from antialiasing). The block **raises** when `reduce is None`, so the divergence needs an explicit reduce spec pinning more than three components; the dict-spec path at `plot.py:2915-2919` falls back to `IncrementalPCA` for the display projection.

**Bundled forecasts always correspond to `trace_data`.** They correspond to `xform_data` only when the two spaces coincide. `trace_metadata` is `None` for non-hierarchical input.

**Files:** Modify `hypertools/plot/plot.py:1920-1941`, `:2827`, `:2886-2919`, `:4587-4604`; Test `tests/plot/test_hierarchy_bundle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_hierarchy_bundle.py
"""What `return_model=True` returns, and what it must NOT redefine.

FLAT inputs only -- the hierarchical bundle assertions live in
tests/plot/test_column_multiindex.py (Task 5), so this module passes
standalone the moment Task 4's keys exist.

There is no `return_data=` parameter (verified: `def plot(` at plot.py:517,
`return_model=False` at plot.py:579, no `return_data` anywhere in
hypertools/). Every test here uses `return_model=True, show=False`.
"""
import matplotlib
matplotlib.use("Agg")

import inspect

import numpy as np

import hypertools as hyp

# A reduce spec pinning MORE than three components, so the display-only
# projection at plot.py:2886-2919 actually runs and `xform_data` (captured
# at plot.py:2827, BEFORE it) keeps the 5-D arrays. Verified in this repo:
# xform_data[0].shape == (60, 5) while the drawn artist is 3-D.
FIVE_D = {'model': 'PCA', 'args': [], 'kwargs': {'n_components': 5}}


def flat_data(n=2, T=40, k=12, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(T, k)) for _ in range(n)]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def test_no_return_data_parameter_exists():
    """v2 of this plan invented one; plot() takes **kwargs, so passing it
    would silently leak into backend kwargs instead of failing."""
    assert 'return_data' not in inspect.signature(hyp.plot).parameters


def test_bundle_keys_are_stable():
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert set(out) == {'fig', 'xform_data', 'trace_data', 'trace_metadata',
                        'animation', 'pipeline', 'models', 'predict'}


def test_flat_input_bundle_is_unchanged():
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert len(out['xform_data']) == 2
    assert out['trace_metadata'] is None


def test_flat_input_trace_data_is_xform_data_when_no_display_projection():
    """The COMMON case -- and the only one in which the two keys may be the
    same object. Contract 5 makes this conditional, not universal."""
    out = hyp.plot(flat_data(), '-', return_model=True, show=False)
    assert out['trace_data'] is out['xform_data']


def test_display_projection_makes_trace_data_diverge_from_xform_data():
    """Contract 5, the counterexample -- on FLAT input.

    `xform_data = copy.copy(xform)` (plot.py:2827) happens BEFORE the
    display-dimensionality enforcement (plot.py:2886-2919), which REBINDS
    `xform` to a new list. So a reduce spec pinning 5 components leaves
    `xform_data` 5-D while the plotted trajectory is 3-D.
    """
    out = hyp.plot(flat_data(n=1), '-', reduce=FIVE_D, return_model=True,
                   show=False)
    assert np.asarray(out['xform_data'][0]).shape == (40, 5)
    assert np.asarray(out['trace_data'][0]).shape[1] == 3
    assert out['trace_data'] is not out['xform_data']
    assert np.asarray(_ax(out['fig']).lines[0].get_data_3d()).shape[0] == 3


def test_bundled_forecasts_correspond_to_trace_data_not_xform_data():
    """Contract 5's headline: forecasts follow `trace_data`, always."""
    out = hyp.plot(flat_data(n=1), '-', reduce=FIVE_D, predict='Kalman', t=2,
                   return_model=True, show=False)
    forecast = np.asarray(out['predict']['forecasts'][0], dtype=float)
    from_trace = np.asarray(
        hyp.predict(np.asarray(out['trace_data'][0]), model='Kalman', t=2),
        dtype=float)
    assert forecast.shape[1] == 3
    assert np.allclose(forecast, from_trace, rtol=1e-6, atol=1e-6)
    from_xform = np.asarray(
        hyp.predict(np.asarray(out['xform_data'][0]), model='Kalman', t=2),
        dtype=float)
    assert from_xform.shape[1] == 5, 'the two spaces genuinely differ here'
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_hierarchy_bundle.py -v`
Expected: `test_no_return_data_parameter_exists` PASSES (it already holds); the other **5** FAIL with `KeyError: 'trace_data'`.

- [ ] **Step 3: Implement**

At `plot.py:4587-4604`, add the two keys:

```python
            "trace_data": trace_data,
            "trace_metadata": trace_metadata,
```

Initialise both beside `xform_data` (`plot.py:2827`) as `trace_data = xform_data` / `trace_metadata = None`, then **re-point `trace_data` at the rebound `xform`** at the end of the display-dimensionality block (`plot.py:2886-2919`), so a display-only projection is reflected in `trace_data` and not in `xform_data`. Task 5 sets both keys again where the hierarchy is resolved (`trace_data = ft.arrays`, `trace_metadata = {...}`).

The rule, in one line: **`trace_data` is whatever the plotted trajectories are at the last point before centering/scaling; `xform_data` is never reassigned after `plot.py:2827`.**

- [ ] **Step 4: Document** the bundle in `plot()`'s `return_model` entry (`plot.py:1920-1941`), stating the contract precisely:

> *"`xform_data` is the analysed pipeline output, one entry per analysed input dataset. `trace_data` is the final pre-center/pre-scale plotted trajectories — for a hierarchical input, the leaves followed by the per-level means. The two are the same object only when no display-only projection occurred; if a `reduce=` spec pins more than three components, `xform_data` keeps that many while `trace_data` is projected to the plotted dimensionality. Bundled forecasts always correspond to `trace_data`, so `forecasts[i]` matches `hyp.predict(trace_data[i], model=..., t=t)` for every i; they match `hyp.predict(xform_data, ...)` element-wise only when the two spaces coincide."*

- [ ] **Step 5: Run and confirm pass** — `.venv/bin/python -m pytest tests/plot/test_hierarchy_bundle.py -v` → **6 passed**, standalone, with no dependency on Task 5.

- [ ] **Step 6: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. Expected: baseline + 53 (47 + 6).

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_hierarchy_bundle.py
git commit -m "feat(plot): bundle exposes trace_data/trace_metadata; xform_data keeps its v1.0 meaning"
```

---

## Task 5: Column MultiIndex end-to-end in `plot()`

> **Executed, then amended by *Revision note (v8)*.** Correspondence is
> nominal, so `test_ragged_groups_raise_the_existing_width_error` became
> `test_ragged_groups_raise_a_named_feature_error`, and permutation-
> invariance and disjoint-label-refusal tests were added. The as-shipped
> tests are the authority: `tests/plot/test_column_multiindex.py`.

**Files:** Modify `hypertools/plot/plot.py:2659-2685`; Test `tests/plot/test_column_multiindex.py`

**Interfaces:** Consumes `group_columns`, `reject_dual_axis`, `reject_hierarchical_in_list`, `build_hierarchy_traces`, `build_hierarchy_styles`.

**Documented modelling rule (must appear in the docstring).** Joint reduction stacks every group, so the groups' features must genuinely correspond. Since v8 that correspondence is established BY NAME, not by position: `group_columns` requires the same innermost-label multiset in every group and permutes later groups into the first group's order, so Tech's `return` is stacked with Energy's `return`. Groups with disjoint labels (one ticker per sector) are refused, naming the missing and unexpected features. `align=` does NOT substitute for this — it aligns the resulting spaces, but by then the reduction has already interpreted whatever inputs it was given as corresponding.

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_column_multiindex.py
"""Column-hierarchy expansion end to end, INCLUDING its return bundle.

Rule: the innermost column level is the FEATURE axis; every level above it
groups. (Market, Sector, Ticker) -> 3 sector leaves + 1 market mean.
Every count below is exact -- measured, not bounded.

The hierarchical half of the Task 4 bundle contract lives here rather than
in tests/plot/test_hierarchy_bundle.py: the bundle needs the column path,
and the column path's assertions read the bundle, so putting them in one
task makes each task's verification step pass standalone (4 -> 5).
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def market_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def two_level_frame(T=60, seed=0):
    """(Group, Feature): n_levels == 1, one leaf per group, NO mean."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples(
        [(g, f) for g in ('A', 'B', 'C') for f in ('f0', 'f1', 'f2')],
        names=['Group', 'Feature'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0), columns=cols)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _labelled(ax):
    return [ln for ln in ax.lines
            if ln.get_label() and not ln.get_label().startswith('_')]


def test_draws_one_trace_per_sector_plus_a_market_mean():
    fig = hyp.plot(market_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_three_level_column_hierarchy_exact_styles():
    """linewidth = 1 + (L-1-level_idx), alpha = min(1, 1/(level_idx+1)+0.2),
    with L = n_levels = 2 for a (Market, Sector, Ticker) frame."""
    fig = hyp.plot(market_frame(), '-', show=False)
    lines = _ax(fig).lines
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == \
        [1.0, 1.0, 1.0, 2.0]
    assert [ln.get_alpha() for ln in lines] == pytest.approx(
        [0.7, 0.7, 0.7, 1.0])
    assert [ln.get_label() for ln in lines] == \
        ['_nolegend_'] * 3 + ['Market']
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in lines}
    assert len(colours) == 1, 'colour comes from the single top level'


def test_two_level_column_hierarchy_draws_one_trace_per_group():
    fig = hyp.plot(two_level_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 3


def test_two_level_column_hierarchy_labels_every_trace():
    """F11: with no mean, unlabelled traces would leave an empty legend."""
    fig = hyp.plot(two_level_frame(), '-', show=False)
    assert [ln.get_label() for ln in _ax(fig).lines] == ['A', 'B', 'C']
    assert len(_labelled(_ax(fig))) == 3


def test_two_level_column_hierarchy_colours_widths_and_opacities():
    fig = hyp.plot(two_level_frame(), '-', show=False)
    lines = _ax(fig).lines
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == [1.0] * 3
    assert [ln.get_alpha() for ln in lines] == pytest.approx([1.0] * 3)
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in lines}
    assert len(colours) == 3, 'each leaf is its own top-level group'


def test_ragged_groups_raise_the_existing_width_error():
    """Expansion accepts ragged groups; the analysis pipeline does not
    (plot.py:2750-2751)."""
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', t) for t in ('AAPL', 'MSFT', 'NVDA', 'ORCL')]
        + [('M', 'Energy', t) for t in ('XOM', 'CVX')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(rng.normal(size=(60, 6)), columns=cols)
    with pytest.raises(ValueError, match='same number of columns'):
        hyp.plot(df, '-', show=False)


def test_dual_axis_frame_is_rejected_by_plot():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(30)])
    cols = pd.MultiIndex.from_tuples([('M', 'Tech'), ('M', 'Energy')])
    df = pd.DataFrame(np.zeros((60, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        hyp.plot(df, '-', show=False)


def test_nan_hierarchy_label_does_not_silently_drop_a_group():
    """A NaN LABEL, not a NaN value. Measured: the pandas default gives 2
    groups instead of 3 (see tests/core/test_hierarchy_grouping.py)."""
    df = market_frame()
    df.columns = pd.MultiIndex.from_tuples(
        [(m, np.nan if s == 'Energy' else s, t) for m, s, t in df.columns],
        names=['Market', 'Sector', 'Ticker'])
    fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_nan_data_values_still_plot():
    df = market_frame()
    df.iloc[:, 0] = np.nan
    fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_row_multiindex_behaviour_is_unchanged():
    """Exact, not `>= 6`: 6 leaves (lw 1.0, alpha 0.7, unlabelled) + 2 means
    (lw 2.0, alpha 1.0, labelled) -- measured on dev-1.0."""
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 4)), index=idx)
    fig = hyp.plot(df, '-', show=False)
    lines = _ax(fig).lines
    assert len(lines) == 8
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == \
        [1.0] * 6 + [2.0] * 2
    assert [ln.get_alpha() for ln in lines] == pytest.approx(
        [0.7] * 6 + [1.0] * 2)
    assert [ln.get_label() for ln in lines] == \
        ['_nolegend_'] * 6 + ['cond1', 'cond2']


def test_datetime_row_index_with_column_hierarchy():
    df = market_frame(T=60)
    df.index = pd.date_range('2020-01-01', periods=60)
    fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_column_hierarchy_inside_a_list_is_rejected():
    """Before 1.1 this silently flattened to ONE line, with no warning."""
    with pytest.raises(ValueError, match='element 0'):
        hyp.plot([market_frame()], '-', show=False)


def test_colorbar_shows_one_segment_per_top_level_group():
    """The GH #100/#95 invariant, now on the column axis: one segment per
    top-level value, never '_nolegend_'."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig = hyp.plot(two_level_frame(), '-', colorbar=True, show=False)
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert '_nolegend_' not in labels
    assert [lbl for lbl in labels if lbl] == ['A', 'B', 'C']


# --- the hierarchical half of the Task 4 bundle contract --------------------

def test_xform_data_holds_only_analysed_leaves():
    """3 sectors in, 3 analysed leaves out -- the market mean is NOT one."""
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['xform_data']) == 3


def test_trace_data_holds_every_plotted_trajectory():
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert np.allclose(np.asarray(out['trace_data'][3]),
                       np.mean(np.stack([np.asarray(a)
                                         for a in out['trace_data'][:3]]),
                               axis=0))


def test_trace_data_length_matches_the_drawn_artists():
    """One artist per plotted trajectory. The artists' VERTEX counts differ
    (centering, scaling and antialiasing come after `trace_data`); only the
    counts of traces are compared."""
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['trace_data']) == len(_ax(out['fig']).lines) == 4


def test_trace_metadata_describes_every_trace():
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    meta = out['trace_metadata']
    assert meta['axis'] == 'columns'
    assert meta['level_names'] == ['Market', 'Sector']
    assert meta['is_mean'] == [False, False, False, True]
    assert meta['level_idx'] == [1, 1, 1, 0]
    assert meta['keys'][-1] == ('Market',)
    assert meta['aux'] is None, 'no hue was passed'
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_column_multiindex.py -v`
Expected: the sector/mean/two-level/colorbar tests FAIL (1 trace drawn, columns flattened); the dual-axis and column-in-a-list tests FAIL (no rejection); the four bundle tests FAIL (one flattened trace, `trace_metadata is None`); `test_row_multiindex_behaviour_is_unchanged` and `test_ragged_groups_raise_the_existing_width_error` PASS already.

- [ ] **Step 3: Dispatch on the column hierarchy**

At `plot.py:2659`, before the row check, add the two rejections, then the column branch:

```python
    reject_hierarchical_in_list(x, caller='hyp.plot', axes='columns')
    reject_dual_axis(x)

    _multiindex_meta = None
    if isinstance(x, pd.DataFrame) and x.index.nlevels >= 2:
        ...                                    # unchanged row path
    elif isinstance(x, pd.DataFrame) and x.columns.nlevels >= 2:
        if cluster is not None or n_clusters is not None:
            raise ValueError(...)              # same message, 'column-MultiIndex'
        x, _multiindex_meta = group_columns(x)
```

`hue` is **not** squelched on this branch (Task 6 honours it). Then replace the `build_multiindex_styles` call at `plot.py:3051` with the split pair, and record the bundle keys from Task 4:

```python
        ft = build_hierarchy_traces(xform, _multiindex_meta, aux=_hue_per_leaf)
        _mi_style = build_hierarchy_styles(ft, palette=palette,
                                           linestyle=linestyle,
                                           linestyles=linestyles)
        xform, trace_data = ft.arrays, ft.arrays
        trace_metadata = {'keys': ft.keys, 'level_idx': ft.level_idx,
                          'is_mean': ft.is_mean, 'axis': ft.meta['axis'],
                          'level_names': ft.meta['level_names'],
                          'aux': ft.aux}          # None when no hue (Task 6)
```

Recompute `pre_interp_lengths` (`plot.py:2999`) from `ft.arrays`. Because `trace_data` is assigned here — after the display-dimensionality block — the hierarchical path automatically satisfies Contract 5: `trace_data` is the pre-center/pre-scale trajectory list, and `xform_data` still holds the analysed leaves captured at `plot.py:2827`.

- [ ] **Step 4: Run and confirm pass**

Run: `.venv/bin/python -m pytest tests/plot/test_column_multiindex.py tests/plot/test_hierarchy_bundle.py -v`
Expected: **17 passed**, **6 passed** (Task 4's module must still be green — it was green before this task too).

- [ ] **Step 5: Confirm the one pinned test this task does NOT change**

`tests/test_multiindex.py:453` (`test_list_with_multiindex_df_warns_and_flattens`) pins **row**-hierarchical frames in lists, and 1.1 keeps that behaviour (*Decisions (resolved)* #1). It must still pass **unchanged**:

```bash
.venv/bin/python -m pytest \
    tests/test_multiindex.py::test_list_with_multiindex_df_warns_and_flattens -v
```
Expected: **1 passed**, with no edit to the file. If it fails, `reject_hierarchical_in_list` was called with `axes='both'` on the plot path — fix the call, not the test.

- [ ] **Step 6: Document** the column rule, the nominal-correspondence rule and its named-feature error (including the `feature_correspondence='position'` opt-in), the **column-only** list rejection (and that row hierarchies in lists still warn and flatten) and the dual-axis rejection in `plot()`'s `x` entry (`plot.py:616-625`) and the `multiindex.py` module docstring, each linking to `docs/hierarchy.rst` (Task 10).

- [ ] **Step 7: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. Expected: baseline + 70 (27 + 14 + 6 + 6 + 17).

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/multiindex.py \
        tests/plot/test_column_multiindex.py tests/test_multiindex.py
git commit -m "feat(plot): column MultiIndex frames expand into per-group traces"
```

---

## Task 6: Continuous hue as a per-trace auxiliary value

> **EXECUTED 2026-08-15, then amended by the corrections below.** Four things
> in Step 1 as written did not survive contact with the code, and one defect
> outside this task's scope had to be fixed for its tests to be satisfiable.
> Measured, not projected:
>
> 1. **`_collections(ax)` as written counts SIX collections too many.** The
>    3-D bounding cube is drawn as six `Line3DCollection` wireframe faces
>    (`matplotlib_backend.py`, `_draw_cube`), so "the `LineCollection`s on the
>    axes" is not the list of data artists — measured, a no-hue hierarchy plot
>    already has 6 of them before any hue is involved.
>    `_apply_multicolor_lines` now tags each collection it creates with
>    `_hyp_trace_index` (the same device as the existing `_hyp_forecast_role`
>    tag on forecast lines), and the helper filters and orders on that tag.
>    Task 8 needs the same handle to find a trace's final colour.
> 2. **`get_segments()` is empty before a draw** on a `Line3DCollection`: it
>    returns the PROJECTED 2-D segments, and nothing has projected them yet.
>    `test_hue_and_data_are_co_truncated` calls `fig.canvas.draw()` first, and
>    asserts the segment count is non-zero so it cannot pass vacuously.
> 3. **`test_colorbar_renders_for_a_continuous_hue_over_a_hierarchy` passed
>    BEFORE the implementation** — `colorbar=True` produced a second axes even
>    while the hue was being warned about and dropped, so `len(fig.axes) == 2`
>    proved nothing. It now also asserts the colorbar's limits equal the range
>    of the concatenated aux (leaves AND means), which is the actual claim.
> 4. **A 2-measure fixture is not 3-D.** The NA-label case below uses three
>    measures so `_ax` finds a `zaxis`.
>
> **Defect found and fixed here, not in this task's scope as written:**
> `_apply_multicolor_lines` never read `alpha` from its per-trace kwargs, so a
> continuous hue silently rendered fully opaque however `alpha=` was spelled
> — the artists carrying the alpha are the very `Line2D`s it removes. That is
> also the channel a hierarchy's level-derived alphas need, so
> `test_hierarchy_still_sets_exact_alphas_under_a_continuous_hue` is
> unsatisfiable without fixing it. Two regression tests for the plain
> (non-hierarchy) case were added to `tests/plot/test_per_dataset_alpha.py`.

The blocker is structural: the MultiIndex branch (`plot.py:3038-3060`) wins the `if/elif` chain outright — its own comment says it *"always wins the cluster/hue/nested_groups chain below"* — so the continuous-colour path at `plot.py:3460` never runs. Hue must be classified **before** the branch and carried through `FinalTraces.aux`, with the hierarchy contributing width/alpha/label only.

**Scope.** Continuous hue over a **column** hierarchy only. Over a **row** hierarchy, `plot.py:2678-2684`'s warn-and-ignore is unchanged (Global Constraints: row plotting semantics do not change; pinned by `tests/test_multiindex.py:306`). See *Decisions (resolved)* #2.

**Accepted hue forms — input-relative only (F12):**
1. **Flat, length T** (T = `len(df)` rows): shared row-wise values, broadcast to every leaf.
2. **Nested, one sequence per leaf**, each length T: per-leaf values. A mean trace takes the **element-wise mean of its contributing leaves' hue vectors**, computed by `build_hierarchy_traces` in the same pass that averages the data.

A flat array whose length equals the *total drawn observations* is **rejected**, not silently reinterpreted: before 1.1 there was no public final-trace list to target, the form is indistinguishable from form 1 whenever `T == n_obs`, and it would require the user to predict how many means expansion will create.

Categorical hue keeps deferring to the grouping (it regroups traces so they are no longer the named leaves) and keeps warning.

**Forecast colour (F14).** A forecast overlay under a continuous hue takes **the final observed hue colour of its source trace** — the last RGBA of that trace's colour array. Simple, visually coherent (the dashed line continues from the head's colour), and expressible on both backends. Tested on both in Tasks 8 and 9.

**Files:** Modify `hypertools/plot/plot.py`, `hypertools/plot/hierarchy.py`; Test `tests/plot/test_multiindex_hue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_multiindex_hue.py
"""Continuous hue through a COLUMN hierarchy.

Two accepted forms, both input-relative: flat length-T (broadcast), or one
sequence per leaf. A flat array sized to the TOTAL DRAWN observations is
rejected -- it is new API, not "existing behaviour", and it would require
the caller to know how many mean traces expansion creates.
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

import hypertools as hyp


def market_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def sector_prices(df):
    """The scalar the market example colours by: each sector's mean price."""
    return [df['Market'][s].mean(axis=1).to_numpy()
            for s in ('Tech', 'Financials', 'Energy')]


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def test_flat_hue_is_broadcast_to_every_trace():
    df = market_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    assert not [w for w in caught if 'ignoring hue' in str(w.message)]
    assert len(_collections(_ax(fig))) == 4


def test_nested_hue_supplies_one_vector_per_leaf():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=sector_prices(df), show=False)
    assert len(_collections(_ax(fig))) == 4


def test_mean_trace_hue_is_the_mean_of_its_leaves():
    """The mean trace's colours are the EXACT colormap RGBA of the
    element-wise mean of its leaves' hue -- not merely "two colours differ",
    which any varying hue would satisfy.

    BOTH available checks are asserted, deliberately:

    1. **Exact RGBA.** The colour chain is pinned in Task 6 Step 3 and
       reproduced here from the same public helpers: `mat2colors` bins the
       CONCATENATION of every trace's aux (`_multicolor_line_colors`:
       "Colors are mapped over the CONCATENATED hue values so the scale is
       shared across datasets") with the default `n_bins=100`, and
       `_apply_multicolor_lines` (plot.py:5094) gives each SEGMENT the
       midpoint of its two endpoints' colours. `antialias=False` keeps the
       point count at `len(df)`, so no interpolation intervenes and the
       arithmetic is exact.
    2. **The bundle's auxiliary array.** `trace_metadata['aux']` exposes the
       per-trace auxiliary values after the same co-truncation the data
       gets (Contract 6), so the mean-of-leaves rule is asserted on the
       numbers themselves and not only through the colormap.
    """
    from hypertools.plot.colors import mat2colors

    df = market_frame()
    hues = sector_prices(df)
    out = hyp.plot(df, '-', hue=hues, palette='viridis', antialias=False,
                   return_model=True, show=False)

    # (2) the numbers
    aux = out['trace_metadata']['aux']
    assert len(aux) == 4
    expected_hue = np.mean(np.stack([np.asarray(h, dtype=float)
                                     for h in hues]), axis=0)
    assert np.allclose(np.asarray(aux[-1], dtype=float), expected_hue)

    # (1) the exact RGBA those numbers must produce
    concatenated = np.concatenate([np.asarray(a, dtype=float) for a in aux])
    point_colors = mat2colors(concatenated, palette='viridis')
    start = sum(len(a) for a in aux[:-1])
    mean_points = point_colors[start:start + len(expected_hue)]
    expected_segments = (mean_points[:-1] + mean_points[1:]) / 2.0

    mean_colours = np.asarray(_collections(_ax(out['fig']))[-1].get_colors())
    assert len(mean_colours) == len(expected_segments)
    assert np.allclose(mean_colours[:, :3], expected_segments, atol=1e-6)


def test_hierarchy_still_sets_exact_widths_under_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    widths = [float(np.atleast_1d(c.get_linewidth())[0])
              for c in _collections(_ax(fig))]
    assert widths == pytest.approx([1.0, 1.0, 1.0, 2.0])


def test_hierarchy_still_sets_exact_alphas_under_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)), show=False)
    alphas = [np.asarray(c.get_colors())[:, 3].max()
              for c in _collections(_ax(fig))]
    assert alphas == pytest.approx([0.7, 0.7, 0.7, 1.0])


def test_nested_hue_with_wrong_leaf_count_raises():
    df = market_frame()
    with pytest.raises(ValueError, match='one hue sequence per'):
        hyp.plot(df, '-', hue=sector_prices(df)[:2], show=False)


def test_nested_hue_with_unequal_lengths_raises():
    df = market_frame()
    bad = sector_prices(df)
    bad[1] = bad[1][:-5]
    with pytest.raises(ValueError, match='length'):
        hyp.plot(df, '-', hue=bad, show=False)


def test_flat_hue_of_wrong_length_raises():
    df = market_frame()
    with pytest.raises(ValueError, match='hue'):
        hyp.plot(df, '-', hue=np.linspace(0, 1, 7), show=False)


def test_flat_hue_of_total_drawn_length_is_rejected():
    """F12: 4 traces x 120 rows = 480 is NOT an accepted form."""
    df = market_frame()
    with pytest.raises(ValueError, match='120 row'):
        hyp.plot(df, '-', hue=np.linspace(0, 1, 480), show=False)


def test_categorical_hue_still_defers_to_the_grouping():
    df = market_frame()
    labels = np.array(['a', 'b'] * (len(df) // 2))
    with pytest.warns(UserWarning, match='hue'):
        hyp.plot(df, '-', hue=labels, show=False)


def test_row_hierarchy_hue_is_still_warned_and_ignored():
    """Row plotting semantics are unchanged in 1.1 (Global Constraints)."""
    idx = pd.MultiIndex.from_tuples(
        [('c1', s) for s in range(3)] + [('c2', s) for s in range(3)],
        names=['cond', 'subj'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 4)), index=idx)
    with pytest.warns(UserWarning, match='ignoring hue'):
        fig = hyp.plot(df, '-', hue=list(range(6)), show=False)
    assert len(_ax(fig).lines) == 8


def test_no_hue_keeps_the_documented_group_colour():
    fig = hyp.plot(market_frame(), '-', show=False)
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in _ax(fig).lines}
    assert len(colours) == 1


def test_colorbar_renders_for_a_continuous_hue_over_a_hierarchy():
    df = market_frame()
    fig = hyp.plot(df, '-', hue=np.linspace(0, 1, len(df)),
                   colorbar=True, show=False)
    assert len(fig.axes) == 2


def test_price_hue_maps_monotonically_through_the_palette():
    """Not merely 'two colours differ': a monotone hue under a sequential
    palette must give monotone luminance along the trace."""
    df = market_frame()
    hues = [np.linspace(100.0, 200.0, len(df)) for _ in range(3)]
    fig = hyp.plot(df, '-', hue=hues, palette='viridis', show=False)
    rgb = np.asarray(_collections(_ax(fig))[0].get_colors())[:, :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    assert np.all(np.diff(lum) > -1e-6), 'viridis luminance must rise with hue'
    assert lum[-1] > lum[0] + 0.2


def test_hue_and_data_are_co_truncated():
    """Contract 6: one truncation operation, applied to both."""
    df = market_frame()
    hues = sector_prices(df)
    fig = hyp.plot(df, '-', hue=hues, show=False)
    for coll, line_len in zip(_collections(_ax(fig)),
                              [len(df)] * 4):
        segs = coll.get_segments()
        assert len(np.asarray(coll.get_colors())) == len(segs)
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_multiindex_hue.py -v`
Expected: every column-hierarchy test FAILS — after Task 5, a column-hierarchical frame draws 4 plain `Line2D`s and `_collections()` is empty. `test_row_hierarchy_hue_is_still_warned_and_ignored` and `test_no_hue_keeps_the_documented_group_colour` PASS already.

- [ ] **Step 3: Restructure the hue pipeline**

1. **Classify hue early.** Before the MultiIndex branch (`plot.py:3038`), determine continuous vs categorical vs matrix using the existing classifier logic, and — when the hierarchy is a **column** hierarchy and hue is continuous — normalise it to one vector per leaf:
   - length `len(df)` → broadcast (same array object per leaf);
   - a sequence of `len(leaf_keys)` sequences, each length `len(df)` → use as given;
   - anything else → `ValueError` naming the expected shapes, e.g. *"hue must be a flat sequence of 120 rows, or one sequence per leaf (3 sequences of 120); got 480 values."*
2. **Hand it to the trace builder** as `build_hierarchy_traces(xform, meta, aux=hue_per_leaf)`, so mean hue and mean data are averaged (and truncated) by the same operation. Then `ft.assert_consistent(aux=ft.aux)`.
3. **Run continuous colour preparation over `ft.aux`** without entering categorical regrouping. **Do not invent a normalisation** — reuse the existing chain, which this plan pins so tests can reproduce it exactly:
   - `_multicolor_line_colors` (`plot.py:5023-5072`) concatenates every trace's aux and maps the whole concatenation at once — *"Colors are mapped over the CONCATENATED hue values so the scale is shared across datasets"* — so the scale spans leaves **and** means;
   - `mat2colors` (`colors.py:24`) takes the 1-D path: `edges = np.linspace(min(finite), max(finite), n_bins + 1)`, `ranks = clip(digitize(fvals, edges) - 1, 0, n_bins - 1)`, colour `= _continuous_palette(palette, n_bins)[rank]`, with the default `n_bins=100`;
   - `_apply_multicolor_lines` (`plot.py:5094`) sets each **segment** to `(ci[:-1] + ci[1:]) / 2.0`, the midpoint of its endpoints' colours;
   - with `antialias=False` no re-interpolation happens, so a trace of `len(df)` rows yields `len(df) - 1` segments. Task 6's `test_mean_trace_hue_is_the_mean_of_its_leaves` asserts exactly these RGBA values.
4. **Expose the auxiliary values in the bundle.** Add `'aux'` to `trace_metadata` (Task 4/5): the per-trace auxiliary arrays as `build_hierarchy_traces` produced them — one per trace, in trace order, co-truncated with the data — or `None` when no continuous hue was given. This is what lets a test assert the mean-of-leaves rule on the numbers rather than only through the colormap.
5. **Apply hierarchical `linewidth`/`alpha`/`label` after colour preparation**, dropping `_mi_style['colors']` on this path only. *(Implemented: `mpl_kwargs['color']` is simply not set under a continuous hue rather than set and then overwritten — a flat colour the collections replace is dead state a later reader would take for the real colours. `legend` also stays whatever the caller passed instead of being bound to the hierarchy labels, matching the existing continuous-hue rule that a colorbar, not a legend, is the key.)*
   **The hue is validated against the INPUT frame's rows, so the analysis pipeline can invalidate it between validation and use** (`manip='Resample'`, a smoother that trims edges). A per-leaf length check runs immediately before `build_hierarchy_traces` and raises naming the stage, rather than letting the aux arrays silently describe different observations than the trace does.
6. Record `ft.aux[i][-1]`'s resolved RGBA per trace as `_forecast_colors`, for Task 8.
7. Categorical hue and row-hierarchy hue: unchanged (*Decisions (resolved)* #2).

- [x] **Step 4: Run and confirm pass** — `.venv/bin/python -m pytest tests/plot/test_multiindex_hue.py -v` → **20 passed**, not the 15 written above. The extra 5 are the adversarial-matrix cases owed against this task: NA hierarchy labels under a hue (parametrized over `np.nan`/`None`/`pd.NA`, so the NA-aware grouping key cannot silently regress into one group per leaf), duplicate innermost feature names reaching the hue path, and the aux co-truncation rule asserted directly on `build_hierarchy_traces`. That last one is a UNIT test on purpose: a column hierarchy is a column slice of one frame, so its groups cannot have unequal-length members, and a `plot()`-level test of Contract 6 would be unreachable rather than merely awkward.

- [x] **Step 5: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. **Measured: 3331 → 3353** (+22: 20 here, plus the 2 continuous-hue alpha regressions in `tests/plot/test_per_dataset_alpha.py`). Note the plan's "baseline + 85 (70 + 15)" arithmetic is a DELTA carried forward from earlier tasks; per Global Constraints, recompute rather than carry it.

- [ ] **Step 6: Document** the two accepted hue forms, the mean-trace derivation rule, the rejected total-observations form, the forecast-colour rule, and the row-hierarchy exception in `plot()`'s `hue` entry, linking to `docs/hierarchy.rst`.

- [ ] **Step 7: Commit**

```bash
git add hypertools/plot/plot.py hypertools/plot/hierarchy.py \
        tests/plot/test_multiindex_hue.py
git commit -m "feat(plot): continuous hue propagates through column hierarchies as a per-trace value"
```

---

## Task 7: Hierarchical `hyp.predict` with explicit model ownership

> **EXECUTED 2026-08-16, then amended by the corrections below.** Every
> "measured" claim in this section's preamble reproduced exactly at
> `ea5d9b5e` (column input → one `(1, 6)` frame; row input and a
> row-hierarchical frame in a list → `TypeError: cannot perform __sub__ with
> this index type: MultiIndex`), and the extracted 22-test module ran
> **21 failed, 1 passed** before any implementation. Corrections, measured:
>
> 1. **The duplicate-timestamp check cannot be unconditional.** Step 3's
>    "add the `index.is_unique` check beside the existing monotonicity
>    check" fires BEFORE `_infer_step` (`predict/common.py:44-46`), whose
>    *"all observations share one timestamp"* message
>    `tests/test_predict_audit_fixes.py:183-187` pins for an all-identical
>    `DatetimeIndex`. An unconditional raise there changes that message and
>    fails that test. Implemented as a BRANCH instead: a non-unique
>    `DatetimeIndex` with `nunique() == 1` re-raises the existing wording;
>    everything else gets the new duplicate message. The audit test is
>    unchanged.
> 2. **The duplicate check is a GLOBAL compatibility change**, not a
>    hierarchy-only one — `resolve_t` runs for flat inputs too. Measured at
>    `ea5d9b5e`: `hyp.predict(df_with_duplicated_DatetimeIndex, t=1)`
>    succeeded (returning `(1, 3)` with only the monotonicity warning); it
>    now raises. The *Compatibility changes* table (L194-200) does not list
>    this — **Task 11 must add it to the CHANGELOG.**
> 3. **`test_unsorted_times_warn_naming_the_group` did not test its own
>    name.** `match='not sorted in ascending order'` is satisfied by the
>    UNPREFIXED warning `predict/common.py` has raised since 1.0, so it
>    verified nothing about F8's group prefix; it failed only because the
>    surrounding call raised `TypeError`. STRENGTHENED to assert exactly one
>    such warning and that it starts with `"group ('Tech',): "` (verified to
>    fail when the prefix is removed from the implementation).
> 4. **`test_flat_frame_return_type_is_unchanged` PASSED before the
>    implementation.** It is a legitimate no-regression guard rather than a
>    broken test, and is kept as written — but it asserts nothing about this
>    feature.
> 5. **`test_duplicate_times_raise_naming_the_group` leaked an unasserted
>    warning.** Its Tech group is also non-monotonic, so it warns on the way
>    to the error. STRENGTHENED with an enclosing `pytest.warns`, which also
>    pins that per-group warnings are re-emitted BEFORE the re-raise —
>    `warnings.catch_warnings(record=True)` SUPPRESSES whatever is not
>    re-emitted, so this fails loudly instead of silently swallowing them.
> 6. **Three tests ADDED (22 → 25)**, each covering a rule this section
>    states but nothing pinned: a class/dict spec fits one independent model
>    per group (only the `str` branch was exercised); a per-group warning
>    keeps its CATEGORY through the re-emission (verified to fail when
>    `w.category` is dropped — a `DeprecationWarning` silently became a
>    `UserWarning`); and grouping does not mutate the caller's frame
>    (Contract 11) end-to-end through `hyp.predict`.
> 7. **Step 5's guard command names a file that does not exist.**
>    `tests/test_predict.py` — the predict suite is `tests/predict` plus
>    `tests/test_predict_audit_fixes.py` (146 passed).
> 8. **Placement, left open by Step 3:** the block runs AFTER
>    `_normalize_data(data)`. A frame with a column MultiIndex but zero
>    columns would otherwise group into ZERO leaves and return an empty LIST
>    of forecasts instead of the clear "no observations" error.
> 9. **Line-number drift** (each verified): `predict/common.py:103-109` is
>    actually 104-110; `predict/common.py:256` is 254; `predict.py:271-275`
>    is 270-274. `predict.py:216-219` and `:245-249` are exact.
> 10. **Known gap, deliberately not widened here:** Step 3 wraps only
>    `ValueError`, so a per-group `TypeError` (unknown kwargs) or
>    `NotFittedError` still escapes without a group name.
>
> **AMENDED 2026-08-16 after a three-reviewer audit of that commit.** Five
> defects, each reproduced before it was fixed:
>
> 11. **A dict spec holding an INSTANCE broke all three ownership
>     promises.** Step 3's `isinstance(model, (str, dict, type))` passthrough
>     assumed a dict spec is stateless, but `{'model': <instance>}` is an
>     accepted form (`predict.py:149-166`; the same shape
>     `tests/test_pipeline_analyze_hardening.py:92` uses for `cluster=`).
>     Measured: the caller's `Kalman()` came back `is_fitted == True`, both
>     groups shared ONE model object, and group 2's forecast did not match an
>     independently fitted Energy forecast. Fixed by dropping `dict` from the
>     passthrough — deep-copying a stateless spec is behaviourally identical
>     — and `test_a_class_or_dict_spec_fits_one_model_per_group` gained the
>     instance-in-a-dict case (the only dict form where this is observable).
> 12. **Per-group warnings were lost whenever a group failed with anything
>     other than `ValueError`.** Step 3's re-emission loop sat AFTER the
>     `with warnings.catch_warnings(record=True)` block, so correction 10's
>     "known gap" silently extended to warnings: measured with
>     `{'model': 'Kalman', 'params': {'bogus': 1}}` (warns, then raises
>     `TypeError`), the FLAT path emitted the DeprecationWarning and the
>     hierarchical path emitted nothing. Fixed by re-emitting in a `finally`.
>     Correction 10 itself stands: non-`ValueError`s still carry no group
>     name. New test:
>     `test_warnings_survive_a_group_that_fails_with_a_non_valueerror`.
> 13. **The duplicate check was too broad by dtype** (correction 2's global
>     scope was right; its *reach* was not). `index.is_unique` was tested for
>     EVERY index dtype, so `pd.concat([run_a, run_b])` — index `0..n-1`
>     twice — began raising where 1.0 forecast fine, contradicting
>     *Decisions (resolved)* #4's "legitimate integer-indexed panels are not
>     rejected" and arguing about a "time axis" that a positional index does
>     not have. Scoped to `DatetimeIndex`/`TimedeltaIndex`/`PeriodIndex`; a
>     datetime-like `t` on a non-time index already raises separately.
> 14. **The remaining (time-indexed) compatibility change is now in the
>     plan, not only in Task 11's CHANGELOG** — *Compatibility changes*
>     (L194-201) gained a row, with both sides of the scope pinned by
>     `test_a_flat_frame_with_duplicated_timestamps_is_rejected` and
>     `test_a_flat_frame_with_a_duplicated_integer_index_still_forecasts`.
>     `test_duplicate_times_raise_naming_the_group` was strengthened from
>     `match="Tech"` (satisfied by ANY ValueError naming the group — it could
>     not detect the over-breadth above) to the message's substance.
> 15. **`_infer_step`'s all-identical-timestamp branch had become dead
>     code**: correction 1 copied its message into `resolve_t`, so
>     `tests/test_predict_audit_fixes.py:186` pinned a duplicated string
>     literal (measured with a spy: `_infer_step` was never called). The
>     degenerate `DatetimeIndex` case is now handed to `_infer_step`, which
>     owns the message; `test_all_identical_timestamps_message_comes_from_
>     live_infer_step` pins that.
> 16. **Not changed, recorded instead:** the column path calls
>     `group_columns(data)` with its default `feature_correspondence='name'`,
>     a rule that exists for PLOTTING. Consequences, both measured: groups
>     naming different features (or of different widths) raise instead of
>     forecasting, and groups sharing labels in a different order come back
>     permuted into the first group's order. That is what Step 3's code
>     prescribes, and *Decisions (resolved)* #6 cites Task 7's
>     `test_duplicate_innermost_names_forecast_by_occurrence` as pinning the
>     nominal `(label, occurrence)` match through `hyp.predict`, so the
>     behaviour is left alone and both consequences are now documented in
>     `predict()`'s docstring. Whether independent per-group forecasts should
>     require cross-group correspondence at all is a maintainer decision.

`hyp.predict(row_multiindex_df)` raises `TypeError: cannot perform __sub__ with this index type: MultiIndex` today (measured); a column-hierarchical frame silently flattens all tickers into one series (measured: `(1, 6)` for a 6-ticker frame).

**Return contract (must be documented):**
- Flat input: unchanged — `forecast`, or `(forecast, model)` when `return_model=True`.
- Hierarchical input: `[f0, f1, ...]`, or `([f0, f1, ...], [m0, m1, ...])` when `return_model=True` — parallel sequences, mirroring the flat shape rather than a list of pairs.

**Model ownership (F9).** Two cases, and they are different on purpose:

| `model=` | behaviour |
|-|-|
| a name (`'Kalman'`), a class, a dict spec, or an **unfitted** instance | **one independent model is fitted per group.** An unfitted instance is `copy.deepcopy`-ed per group, so the caller's object is never fitted and later groups never fall onto `predict_new`. |
| an **already-fitted** instance | **its learned parameters are reused** on each group, via an independent deep copy per group (`predict.py:216-219` → `resolved.predict_new(data, t)`). This is the reuse `predict.py:245-249` promises; it is not "fitted independently". |

Either way the caller's instance is never mutated, and `return_model=True` returns one distinct object per group.

**Time index (F5).** Row groups keep the innermost level as their index, so `t` may be a future `Timestamp` and each group's forecast comes back with a real `DatetimeIndex`. Because the index survives, `predict/common.py:103-109`'s *"the dataset index is not sorted in ascending order"* warning fires **per group** — that is the monotonicity warning F8 asks for, with a group name prepended.

**Files:** Modify `hypertools/predict/predict.py`; Test `tests/predict/test_predict_multiindex.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/predict/test_predict_multiindex.py
"""Hierarchical forecasting: grouping, return shape, model ownership.

Column hierarchy -> one forecast per group (innermost level = features).
Row hierarchy    -> one forecast per group of the OUTER levels, with the
                    innermost level kept as each group's time index.
"""
import copy

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.predict.kalman import Kalman


def col_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 6)).cumsum(axis=0) + 100.0, columns=cols)


def row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], range(T)],
                                     names=['Sector', 'day'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


def dated_row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range('2020-01-01', periods=T)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], days],
                                     names=['Sector', 'date'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


# --- grouping and shape -----------------------------------------------------

def test_column_hierarchy_forecasts_each_group():
    out = hyp.predict(col_frame(), model='Kalman', t=1)
    assert isinstance(out, list) and len(out) == 2
    assert all(np.asarray(g).shape == (1, 3) for g in out)


def test_row_hierarchy_groups_by_every_level_above_time():
    """(Sector, day) -> 2 sector forecasts, NOT 2*T one-row leaves."""
    out = hyp.predict(row_frame(), model='Kalman', t=2)
    assert isinstance(out, list) and len(out) == 2
    assert all(np.asarray(g).shape == (2, 3) for g in out)


def test_row_hierarchy_forecast_keeps_a_datetime_index():
    out = hyp.predict(dated_row_frame(), model='Kalman', t=3)
    for g in out:
        assert isinstance(g.index, pd.DatetimeIndex)
        assert g.index[0] > pd.Timestamp('2020-02-29')


def test_future_timestamp_horizon_on_a_row_hierarchy():
    """reset_index(drop=True) would make this impossible."""
    out = hyp.predict(dated_row_frame(), model='Kalman',
                      t=pd.Timestamp('2020-03-05'))
    assert len(out) == 2
    for g in out:
        assert g.index[-1] == pd.Timestamp('2020-03-05')
        assert np.asarray(g).shape[1] == 3


def test_at_or_before_timestamp_truncates_each_group():
    out = hyp.predict(dated_row_frame(), model='Kalman',
                      t=pd.Timestamp('2020-02-01'))
    assert len(out) == 2
    for g in out:
        assert g.index[-1] <= pd.Timestamp('2020-02-01')
        assert len(g) == 32


def test_unsorted_times_warn_naming_the_group():
    days = pd.date_range('2020-01-01', periods=40)
    perm = np.random.default_rng(0).permutation(40)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 40 + ['Energy'] * 40, list(days[perm]) + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(80, 3)).cumsum(0),
                      index=idx)
    with pytest.warns(UserWarning, match='not sorted in ascending order'):
        hyp.predict(df, model='Kalman', t=1)


def test_duplicate_times_raise_naming_the_group():
    days = pd.date_range('2020-01-01', periods=20)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 40 + ['Energy'] * 40,
         list(days) * 2 + list(pd.date_range('2020-01-01', periods=40))],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(80, 3)).cumsum(0),
                      index=idx)
    with pytest.raises(ValueError, match="Tech"):
        hyp.predict(df, model='Kalman', t=1)


def test_group_forecast_matches_forecasting_that_group_alone():
    df = col_frame()
    grouped = hyp.predict(df, model='Kalman', t=1)
    alone = hyp.predict(df['Market']['Tech'], model='Kalman', t=1)
    assert np.allclose(np.asarray(grouped[0]), np.asarray(alone),
                       rtol=1e-6, atol=1e-6)


def test_grouped_leaves_are_non_hierarchical_so_the_recursion_terminates(
        monkeypatch):
    """The recursion guard, made OBSERVABLE rather than inferred (Revision
    note (v6) D1/D2). `predict()` recurses with `predict(group, ...)`, so a
    leaf still carrying its grouping levels is re-detected by the same
    `nlevels >= 2` predicate and regrouped without bound -- measured on v5's
    `sub.T` leaves. Both core helpers are WRAPPED here: they still run (this
    OBSERVES, it does not substitute), recording each leaf's axis index and
    their own call counts. Patching `hypertools.core.hierarchy` rather than
    the predict module is deliberate -- `predict()` imports them inside the
    function, so the name is looked up on the source module at call time.
    A test that merely 'does not hang' would not be adequate: the counts and
    the leaf indices are asserted explicitly.
    """
    import hypertools.core.hierarchy as hier

    real_columns, real_rows = hier.group_columns, hier.group_rows_for_forecast
    col_calls, row_calls, seen_cols, seen_rows = [], [], [], []

    def observing_columns(df):
        leaves, meta = real_columns(df)
        col_calls.append(df.columns.nlevels)
        seen_cols.extend(leaf.columns for leaf in leaves)
        return leaves, meta

    def observing_rows(df):
        groups, keys = real_rows(df)
        row_calls.append(df.index.nlevels)
        seen_rows.extend(group.index for group in groups)
        return groups, keys

    monkeypatch.setattr(hier, 'group_columns', observing_columns)
    monkeypatch.setattr(hier, 'group_rows_for_forecast', observing_rows)

    col_out = hyp.predict(col_frame(), model='Kalman', t=1)
    row_out = hyp.predict(row_frame(), model='Kalman', t=2)

    assert len(col_calls) == 1, \
        f'group_columns ran {len(col_calls)}x: the leaves were regrouped'
    assert len(row_calls) == 1, \
        f'group_rows_for_forecast ran {len(row_calls)}x: leaves regrouped'
    assert len(seen_cols) == 2 and len(seen_rows) == 2
    assert all(not isinstance(cols, pd.MultiIndex) for cols in seen_cols)
    assert all(not isinstance(idx, pd.MultiIndex) for idx in seen_rows)
    assert len(col_out) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in col_out)
    assert len(row_out) == 2
    assert all(np.asarray(f).shape == (2, 3) for f in row_out)


def test_duplicate_innermost_names_forecast_by_occurrence():
    """DECIDED (Revision note (v6) D3, refined by (v8) F2): flattening onto
    the feature axis can collide two innermost labels WITHIN a group; they
    are kept, and matched across groups by (label, occurrence). Measured:
    2 groups, widths [3, 3], one (1, 3) forecast each -- nothing merged,
    nothing dropped."""
    cols = pd.MultiIndex.from_tuples(
        [('Rig', 'North', s) for s in ('temp', 'temp', 'flow')]
        + [('Rig', 'South', s) for s in ('temp', 'temp', 'flow')],
        names=['Rig', 'Well', 'Sensor'])
    df = pd.DataFrame(
        np.random.default_rng(0).normal(size=(120, 6)).cumsum(axis=0) + 100.0,
        columns=cols)
    out = hyp.predict(df, model='Kalman', t=1)
    assert len(out) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in out)


def test_flat_frame_return_type_is_unchanged():
    flat = pd.DataFrame(np.random.default_rng(0).normal(size=(80, 4)).cumsum(0))
    out = hyp.predict(flat, model='Kalman', t=3)
    assert not isinstance(out, list)
    assert np.asarray(out).shape == (3, 4)


def test_horizon_is_respected_per_group():
    out = hyp.predict(col_frame(), model='Kalman', t=5)
    assert all(np.asarray(g).shape[0] == 5 for g in out)


def test_groups_come_back_in_input_order():
    df = col_frame()
    out = hyp.predict(df, model='Kalman', t=1)
    alone = hyp.predict(df['Market']['Tech'], model='Kalman', t=1)
    assert np.allclose(np.asarray(out[0]), np.asarray(alone))


# --- model ownership --------------------------------------------------------

def test_return_model_yields_parallel_sequences():
    forecasts, models = hyp.predict(col_frame(), model='Kalman', t=1,
                                    return_model=True)
    assert len(forecasts) == len(models) == 2
    assert all(np.asarray(f).shape == (1, 3) for f in forecasts)


def test_returned_models_are_distinct_objects_per_group():
    _, models = hyp.predict(col_frame(), model='Kalman', t=1, return_model=True)
    assert models[0] is not models[1]


def test_an_unfitted_instance_is_not_mutated_across_groups():
    """A REAL instance, not the string 'Kalman'. Fitting group 0 must not
    push group 1 onto the predict_new path, nor fit the caller's object."""
    shared = Kalman()
    assert not shared.is_fitted
    forecasts, models = hyp.predict(col_frame(), model=shared, t=1,
                                    return_model=True)
    assert not shared.is_fitted, 'the caller\'s instance was mutated'
    assert models[0] is not shared and models[1] is not shared
    independent = hyp.predict(col_frame()['Market']['Energy'],
                              model='Kalman', t=1)
    assert np.allclose(np.asarray(forecasts[1]), np.asarray(independent),
                       rtol=1e-6, atol=1e-6)


def test_a_fitted_instance_is_reused_not_refitted():
    """predict.py:245-249 promises fitted-model REUSE; hierarchical input
    must honour it, applying the same learned parameters to each group."""
    df = col_frame()
    _, fitted = hyp.predict(df['Market']['Tech'], model='Kalman', t=1,
                            return_model=True)
    assert fitted.is_fitted
    grouped = hyp.predict(df, model=fitted, t=1)
    reference = copy.deepcopy(fitted)
    per_group = [np.asarray(hyp.predict(df['Market'][s],
                                        model=copy.deepcopy(reference), t=1))
                 for s in ('Tech', 'Energy')]
    for got, want in zip(grouped, per_group):
        assert np.allclose(np.asarray(got), want, rtol=1e-6, atol=1e-6)


def test_a_fitted_instance_is_not_mutated_across_groups():
    df = col_frame()
    _, fitted = hyp.predict(df['Market']['Tech'], model='Kalman', t=1,
                            return_model=True)
    before = copy.deepcopy(fitted)
    _, models = hyp.predict(df, model=fitted, t=1, return_model=True)
    assert models[0] is not fitted and models[1] is not fitted
    assert np.allclose(np.asarray(hyp.predict(df['Market']['Tech'],
                                              model=before, t=1)),
                       np.asarray(hyp.predict(df['Market']['Tech'],
                                              model=fitted, t=1)))


def test_returned_models_can_be_reused_on_new_data():
    df = col_frame()
    _, models = hyp.predict(df, model='Kalman', t=1, return_model=True)
    again = hyp.predict(df['Market']['Tech'], model=models[0], t=1)
    assert np.asarray(again).shape == (1, 3)


# --- rejections -------------------------------------------------------------

def test_dual_axis_frame_is_rejected():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(30)])
    cols = pd.MultiIndex.from_tuples([('M', 'T'), ('M', 'E')])
    df = pd.DataFrame(np.zeros((60, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        hyp.predict(df, model='Kalman', t=1)


def test_hierarchical_frame_in_a_list_is_rejected():
    """Today this raises TypeError deep inside pandas; 1.1 says why."""
    with pytest.raises(ValueError, match='hyp.predict'):
        hyp.predict([row_frame()], model='Kalman', t=1)


def test_group_with_too_little_history_raises_naming_the_group():
    idx = pd.MultiIndex.from_tuples([('Tech', 0), ('Energy', 0)],
                                    names=['Sector', 'day'])
    df = pd.DataFrame(np.zeros((2, 3)), index=idx)
    with pytest.raises(ValueError, match='Tech|Energy'):
        hyp.predict(df, model='Kalman', t=1)
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/predict/test_predict_multiindex.py -v`
Expected: column tests FAIL (a single `(1, 6)` frame comes back); row tests FAIL with `TypeError: cannot perform __sub__ with this index type: MultiIndex`.

- [ ] **Step 3: Implement**

At the top of `predict()`, before model dispatch:

```python
    from ..core.hierarchy import (group_columns, group_rows_for_forecast,
                                  reject_dual_axis, reject_hierarchical_in_list)
    reject_hierarchical_in_list(data, caller='hyp.predict', axes='both')
    if isinstance(data, pd.DataFrame) and (data.index.nlevels >= 2
                                           or data.columns.nlevels >= 2):
        reject_dual_axis(data)
        if data.columns.nlevels >= 2:
            # `group_columns` returns (leaves, META); the group LABELS live in
            # meta['leaf_keys']. Unpacking it as `keys` would label each group
            # with a dict key ('n_levels', 'leaf_keys', ...) in the messages
            # below. Labels come from the grouping key, NEVER from a leaf's
            # columns -- flattening (Contract 11) removes the tuples anyway.
            groups, _meta = group_columns(data)
            keys = _meta['leaf_keys']
        else:
            groups, keys = group_rows_for_forecast(data)

        # TERMINATION (Contract 11): every group above is flat on the axis it
        # was grouped along -- flat columns, or the innermost level as a flat
        # index -- so the recursive `predict(group, ...)` below cannot
        # re-detect it as hierarchical. Before v6 the column leaves kept the
        # caller's full column MultiIndex and this recursion did not
        # terminate. Never route `expand_multiindex` leaves through here:
        # those keep their row MultiIndex and re-expand to themselves
        # (Revision note (v6) D2).
        forecasts, models = [], []
        for key, group in zip(keys, groups):
            # OWNERSHIP: an UNFITTED instance is deep-copied so each group
            # fits independently and the caller's object is never fitted; a
            # FITTED instance is deep-copied so each group reuses the same
            # learned parameters via predict_new (predict.py:216-219)
            # without the groups sharing mutable state. `dict` is NOT in the
            # passthrough (correction 11): `{'model': <instance>}` is an
            # accepted spec form, so a dict can carry fitted state.
            group_model = (model if isinstance(model, (str, type))
                           else copy.deepcopy(model))
            try:
                result = predict(group, model=group_model, t=t,
                                 return_model=True, **kwargs)
            except ValueError as err:
                raise ValueError(f"group {key}: {err}") from err
            forecasts.append(result[0])
            models.append(result[1])
        return (forecasts, models) if return_model else forecasts
```

Wrap the per-group call in `warnings.catch_warnings(record=True)` and re-emit each captured warning with `f"group {key}: {msg}"` **from a `finally`** (correction 12 — the loop must run even when the group fails with a non-`ValueError`, or `catch_warnings` swallows what it recorded), so `predict/common.py:103-109`'s monotonicity warning names its group (F8). Duplicate timestamps within a group raise a `ValueError` naming the group — add the `index.is_unique` check beside the existing monotonicity check in `predict/common.py`, **guarded on a time-like index** (`DatetimeIndex`/`TimedeltaIndex`/`PeriodIndex`; correction 13) so duplicated positional indexes keep 1.0's behaviour.

- [x] **Step 4: Run and confirm pass** — `.venv/bin/python -m pytest tests/predict/test_predict_multiindex.py -v` → **22 passed** *(EXECUTED: **25 passed**, see correction 6 above)*.

- [x] **Step 5: Guard the predict suite** — ~~`.venv/bin/python -m pytest tests/predict tests/test_predict.py -q`~~ *(`tests/test_predict.py` does not exist)*: `.venv/bin/python -m pytest tests/predict tests/test_predict_audit_fixes.py -q` → **146 passed**.

- [x] **Step 6: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. Expected: baseline + 107 (85 + 22) *(EXECUTED: measured baseline 3353 + 25 = **3378 passed**; the "85" was a carry-forward, recomputed per Global Constraints)*.

- [ ] **Step 7: Document** the grouping rule, the leaf-flatness invariant that makes the per-group recursion terminate (Contract 11), the return shape, the unfitted-vs-fitted ownership table, and the datetime-horizon behaviour in `predict()`'s docstring (beside `model=`, `predict.py:224-250`, and `return_model`, `predict.py:271-275`), linking to `docs/hierarchy.rst`.

- [ ] **Step 8: Commit**

```bash
git add hypertools/predict/predict.py hypertools/predict/common.py \
        tests/predict/test_predict_multiindex.py
git commit -m "feat(predict): hierarchical forecasting with explicit unfitted/fitted model ownership"
```

---

## Task 8: `predict=` over the final traces, with exact bundle correspondence

> **EXECUTED (2026-08-16), with amendments.** Implemented as described; the corrections below were measured against the tree at `c5fb889c`, not assumed.
>
> 1. **Every `plot.py:NNNN` line number in this task is stale by roughly +1000** (Tasks 5-7 grew the file). `:2669-2677` → `:3677-3718`; `:2963-2999` → `:4076-4106`; `:3999` → `:5345`; `:4000-4002` → `:5363-5387`. Cite by symbol, not by line.
> 2. **There are TWO refusal guards, not one.** Step 3.1 says "delete the blanket refusal at `plot.py:2669-2677`" (singular), but the ROW guard (`plot.py:3677-3685`, remediation `df.reset_index(drop=True)`) and the COLUMN guard (`plot.py:3709-3718`, remediation `df.columns = df.columns.map('_'.join)`) are separate sites with different text. Both were replaced; deleting only one would leave the other axis refusing, contradicting this task's own Compatibility table.
> 3. **Step 3.2 was applied CONDITIONALLY, not as a blanket move.** The forecast computation's placement before the cluster/hue chain is load-bearing: `_forecast_owner` (`plot.py:5345-5387`), `TraceOwnership` and `DatasetRevealSchedule` are all defined in terms of INPUT DATASETS. Moving it unconditionally past the chain would compute per-RUN forecasts for `hue=`/`cluster=` input and break `tests/plot/test_forecast_with_hue.py`, `test_forecast_animated_regrouped.py` and `test_regrouped_reveal.py`. It was factored into a local `_compute_forecasts(datasets)` closure, called on `xform` when `_multiindex_meta is None` and on `_ft.arrays` inside the hierarchy branch otherwise.
> 4. **Step 3.3 was already done** at `plot.py:4293` (`pre_interp_lengths` is recomputed from `_ft.arrays` with the comment "the means were appended after the earlier pass"), landed by Task 4/5. No edit needed.
> 5. **Step 3.4 needed an `_ft = None` initialiser.** `_ft` was assigned only inside `if _multiindex_meta is not None:`, so referencing it at the guard site raised `NameError` on every other input. Also: that guard is no longer the "silent null-out" this step describes — since the regrouped-reveal work it warns and records `_forecast_draw_reason`. It was left intact as the `elif` arm (per this step's own "extend rather than replace" caveat) with the `assert_consistent` call as the new `if _ft is not None:` arm.
> 6. **Step 3.5's `_forecast_colors` does not exist.** Task 6 Step 3 item 6 prescribed it, but Task 6 as EXECUTED implemented F14 differently: `_apply_multicolor_lines` (`plot.py:7110-7123`) recolours each kept forecast line to `line_colors[_hyp_forecast_dataset][-1]`. `grep -rn _forecast_colors hypertools/` is empty. Since `line_colors` is built from `xform` (= `_ft.arrays`) and `_hyp_forecast_dataset` is the forecast's index into `raw_forecasts` (also per final trace), the indices line up once forecasts become one-per-final-trace, so Step 3.5 needed no code. ~~VERIFIED by `test_forecast_takes_the_final_observed_hue_colour`, which fails without it.~~ **That verification claim was FALSE as first written — see correction 12.** It is true after the follow-up commit.
> 7. **The prescribed `_solid`/`_dashed` test helpers cannot work.** `_forecast_style_from` (`plot.py:222-234`) makes a forecast INHERIT its source line's linestyle, so under this module's `fmt='-'` every artist is solid and `_dashed(ax)` returns `[]`; 9 of the 17 tests plus Step 5's "10 solid + 10 dashed" depended on them. They were rewritten as `_observed`/`_forecasts`, splitting on `_hyp_forecast_role` — the repo's own idiom (`tests/plot/test_predict_integration.py:20-31`, whose docstring already says "a solid dataset's forecast is solid").
> 8. **`test_forecast_takes_the_final_observed_hue_colour` filtered the wrong collections.** `[c for c in ax.collections if isinstance(c, LineCollection)]` is 10 long here: the first SIX are the `_draw_cube` wireframe faces, only the last four are traces, so the prescribed `zip` paired cube faces with forecasts. Replaced with a `_trace_collections` helper filtering and ordering on `_hyp_trace_index` (Task 6 revision note #1's own trap, never propagated into this task).
> 9. **Two tests were strengthened before implementing** (per "if a prescribed test passes before you implement anything, it does not test the feature"): `test_leaf_forecasts_match_hyp_predict_on_xform_data_when_spaces_coincide` gained explicit `len(...) == 3 / == 4` assertions, because its `zip` truncates at 3 and could not tell a 3-trace figure from a 4-trace one; `test_mean_trace_forecast_comes_from_the_mean_trajectory` gained `assert trace_metadata['is_mean'][-1] is True`, so it cannot pass by comparing a leaf against itself. Measured pre-implementation: 16 failed, 1 passed (`test_one_row_row_hierarchy_still_plots_without_predict`, as Step 2 predicts).
> 10. **Step 6's arithmetic does not reconcile** ("baseline + 125 (107 + 17 + 1)" is a carry-forward, as in Task 6). Measured absolutely instead: `.venv/bin/python -m pytest -q` → **3402 passed, 13 skipped, 2 deselected**, no warnings summary. That is +18 over the `c5fb889c` baseline (17 new tests in `tests/plot/test_multiindex_predict.py` + 1 net in `tests/test_multiindex.py`); the implied baseline of 3384 is itself 6 above Task 7 Step 6's recorded 3378, because Task 7 landed a second commit (`c5fb889c`) after writing that note. `tests/test_multiindex.py` has 30 `def test_` functions as this task predicts (pytest reports 33 collected, the difference being parametrisation).
>
>     One transient failure worth recording: `tests/test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist` fails while the new test module is UNTRACKED ("1 untracked file(s) leaked into the sdist"). That is the guard working as designed, not a defect — it passes as soon as the file is `git add`ed, and the 3402-passing run above was taken with it staged.
> 11. **Step 7's `docs/hierarchy.rst` row could not be written**: that file does not exist yet (Task 10 creates it). Deferred to Task 10. The `x`, `predict=` and `return_model=` docstring entries were all updated here.
>
> **Follow-up commit (2026-08-16), from a review of the above.** Three defects, each reproduced before being fixed:
>
> 12. **Step 3.5's verification did not verify.** The plan's `hue=[np.linspace(0, 1, len(df)) for _ in range(3)]` gives every leaf the SAME ramp, so all three leaves *and* their derived mean end at hue 1.0 and share one final colour — measured `[0.9744 0.9036 0.1302]` four times. `test_forecast_takes_the_final_observed_hue_colour`'s `zip(colls, forecasts)` therefore compared one colour against itself four times, and every permutation of the pairing passed. Mutation proof: an off-by-one on `_apply_multicolor_lines`' `_hyp_forecast_dataset` lookup left all 17 tests passing (the same mutation IS caught for flat input, by `test_forecast_with_hue.py::test_a_continuous_hue_colours_each_forecast_from_ITS_OWN_dataset`). Fixed by giving the leaves ramps ending at 0.0 / 0.25 / 1.0, so the mean ends at 0.4167 and all four end colours differ (closest pair 0.164 per channel, 8x the 0.02 match tolerance); both sides are now keyed on `_hyp_forecast_dataset`/`_hyp_trace_index` rather than list position, and a pairwise-distinctness guard keeps the ramps from silently degenerating again. The mutation now fails the test. **Step 3.5 needed no code, and that conclusion still holds — but only now is it measured.**
>
> 13. **The animated hierarchy path was asserted by COUNTS only.** `test_predict_with_hierarchy_and_animation_via_on_frame` checked `len(seen) == 8`, `len(seen[-1].datasets) == 4` and `len(_forecasts(ax)) == 4`, and never touched a coordinate — so `analyze_histories`, the one input the animated schedule re-forecasts from, was unverified. Since `predict=` + MultiIndex raised before this task, no pre-existing test covered it either. Mutation proof: rotating `analyze_histories` by one after the hierarchy branch's `_compute_forecasts(_ft.arrays)` left all 17 tests passing while every animated forecast detached from its trace (forecast-start-to-trace-end gaps 0.0 → 0.641 / 0.512 / 0.264 / 0.345). The test now asserts anchoring on COORDINATES, as the static path already did, and is parametrised over BOTH axes (the row axis reaches the code through a different expansion rule; 8 traces, all gaps 0.0). The mutation now fails both parametrisations.
>
> 14. **Contract 10's message blamed the grouping for what the PIPELINE did.** The precondition necessarily runs on `_ft.arrays`, i.e. AFTER manip/normalize/reduce/align, but both axis messages were emitted unconditionally. Measured: `hyp.plot(market_frame(T=30), '-', predict='Kalman', t=2, manip={'model': 'Resample', 'kwargs': {'n_samples': 1}})` said *"every group keeps all 1 of the frame's rows — the input itself has only one observation … pass a frame with more rows"* about a **30-row** frame; the row axis said the innermost level *"is unique per row"* of a frame whose innermost level repeats 10x. Both remedies were inert for those inputs. The repo's own idiom for exactly this was 25 lines above, in the sibling hue-length check (*"the analysis pipeline changed the row count before plotting … drop the row-count-changing stage"*), and had not been applied. Fixed by capturing each leaf's PRE-pipeline row count at expansion time (`_mi_input_rows`) and branching on `_rows != _input_rows` — the pipeline is named when it changed the count, and the grouping/input wording is reserved for when it did not. A mean has no direct pre-pipeline length, but a mean is the average over its members' overlapping prefix, so once every leaf clears >= 2 rows no mean can be short; `min` over all leaves is the (unreachable, conservative) stand-in. The `x` docstring's claim that the requirement *"can only fail when the INPUT frame has fewer than 2 rows"* was corrected on both axes. Two new tests: `test_a_row_count_changing_stage_is_blamed_instead_of_the_grouping` (both axes) and `test_a_genuinely_short_input_still_blames_the_input_not_the_pipeline`, the latter pinning that the new branch did not simply relabel every failure.

`plot()` refuses this today (`plot.py:2669-2677`), and its error message states the cause: *"forecasts are computed one-per-leaf before the per-level mean traces are appended, so the leaf count no longer matches the final trace count."* Task 2 supplies the ordering; this task computes forecasts over the final trace list and proves the bundle stays consistent (Contract 5).

### What is supported, and what still raises

Forecasting is defined per **trace**, so it inherits `hyp.predict`'s minimum: a trace needs at least 2 rows (`predict/common.py:256`, `if d.shape[0] < 2:`, raising *"cannot forecast from a single observation: the dataset has only 1 row. Forecasting needs at least 2 observations (rows) to estimate how the data change over time."*).

- **Column hierarchies: supported whenever the input has at least two rows.** Every leaf keeps all `len(df)` rows, and so does every derived mean — only the feature axis is grouped, so grouping never *shortens* a trace. It cannot *lengthen* one either. Measured with this plan's grouping idiom (`df.T.groupby(level='Sector', sort=False)`, transposed back): `T=1` gives leaf shapes `{'Tech': (1, 3), 'Fin': (1, 3)}` and a mean of `(1, 3)` — **not** forecastable; `T=2` gives `(2, 3)` throughout — forecastable. `hyp.predict` on that 1-row frame raises the same `predict/common.py:256` guard.
- **Row hierarchies: supported only when the shape allows it.** `expand_multiindex` makes one leaf per unique **full** row-index tuple, so a frame whose innermost level is unique per row produces **one-row leaves** — and one-row means. Measured: a 6-row `(cond, subj)` frame with 4 feature columns gives 6 leaves of shape `(1, 4)`, so all 8 of its final traces have 1 row and none of them can be forecast. A frame whose innermost level **repeats** (several timepoints per `(cond, subj)` pair) gives multi-row leaves and forecasts normally: measured, `tests/test_multiindex.py`'s `_make_2level_df()` gives **8 leaves of shape (10, 3)** and draws **10** traces.

**The check is a precondition over `ft.arrays`, not a bubbled-up `predict` error, and it runs for EVERY hierarchy.** Every final trace — leaves *and* derived means, on **both** axes — is checked immediately after `ft` is built and *before* any forecasting, so the message is about the data. Both messages name the offending trace and its row count; only the remediation differs by axis.

**Row hierarchy** — the traces are short because expansion split the frame, so there are two ways out:

```
plot(..., predict=...) needs at least 2 rows per trace, but trace 0
('cond1', 0) has 1 row. Row-MultiIndex expansion draws one trace per
unique FULL index tuple, so a frame whose innermost index level is
unique per row yields one-row traces (and one-row per-level means).
Either drop the hierarchy so the frame is one trajectory
(df.reset_index(drop=True)), or move the grouping to the COLUMN axis,
where every group keeps all of the frame's rows.
```

**Column hierarchy** — grouping never shortens a trace, so a short trace means the **input itself** is short. Flattening would not add a row, and is deliberately not suggested:

```
plot(..., predict=...) needs at least 2 rows per trace, but trace 0
('Market', 'Tech') has 1 row. A column MultiIndex groups FEATURES, so
every group keeps all 1 of the frame's rows -- the input itself has
only one observation. Forecasting needs at least 2 observations (rows)
to estimate how the data change over time; pass a frame with more rows.
```

### Two short-history mechanisms, and why they do not conflict

Short histories are handled at **two** levels, with deliberately different policies:

1. **This plan's precondition tests the FULL trace length** — a *permanent* property of the data. It therefore runs for animated hierarchies too, **before** Plan 3's `ForecastSchedule` is built, and **raises**. A hierarchy whose traces are one row can never produce a forecast at any frame, so raising is correct and silence would be misleading.
2. **Plan 3's `min_history` tests the PER-FRAME revealed history** — a *transient* property. `forecast_from_history(...)` (`2026-07-27-hypertools-1.1-forecast-animation.md:522`) does `if len(history) < max(2, min_history): return None`, so the opening frames of a legitimate animation simply show no forecast yet (pinned there by `test_returns_none_below_min_history` at `:386` and the frame-0 test at `:621`).

These do not conflict: an animated hierarchy with long traces **passes** the precondition, and `min_history` still suppresses forecasts in its opening frames. Without the ordering, a one-row hierarchy would reach the schedule and draw nothing forever, because no frame ever reaches 2 rows. `test_animated_one_row_hierarchy_still_raises_the_precondition` pins the ordering.

**Files:** Modify `hypertools/plot/plot.py:2669-2677`, `:2963-2999`, `:3999`; Test `tests/plot/test_multiindex_predict.py`, `tests/test_multiindex.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/plot/test_multiindex_predict.py
"""One forecast per pre-center/pre-scale plotted trajectory, and a bundle
that proves it.

Contract 5: forecasts[i] == hyp.predict(trace_data[i], model, t) for every
i. plot.py:3999 used to null raw_forecasts on any count mismatch, so a
missing forecast was invisible; here a mismatch raises.

Contract 10: EVERY final trace of EVERY hierarchy -- leaf or derived mean,
row axis or column axis -- needs >= 2 rows. All three sides are tested: a
repeating innermost level forecasts; a unique-per-row one raises a message
about the row-expansion rule; a T=1 column frame raises a message about the
INPUT, since column grouping never shortens a trace and flattening cannot
lengthen one. The precondition also runs before the animation schedule is
built, so an animated one-row hierarchy raises rather than silently drawing
no forecast at every frame.
"""
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection

import hypertools as hyp


def market_frame(T=150, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Financials', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def multirow_row_frame(n_time=10, seed=0):
    """A ROW hierarchy whose innermost level REPEATS, so each leaf keeps
    `n_time` rows: 2 conds x 3 subjs x n_time timepoints.

    Verified construction: `expand_multiindex` yields 6 leaves of shape
    (n_time, 3), and the plot draws 8 traces (6 leaves + 2 top-level means),
    every one of them n_time rows -- so every trace clears the >= 2-row
    precondition."""
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for ci, cond in enumerate(['cond1', 'cond2']):
        for si in range(3):
            rows.append(rng.standard_normal((n_time, 3)).cumsum(axis=0)
                        + ci * 5.0)
            tuples.extend([(cond, f'S{si}')] * n_time)
    idx = pd.MultiIndex.from_tuples(tuples, names=['cond', 'subj'])
    return pd.DataFrame(np.vstack(rows), index=idx, columns=['x', 'y', 'z'])


def one_row_row_frame(seed=0):
    """A ROW hierarchy whose innermost level is UNIQUE PER ROW, so every
    leaf is 1 row. Verified: 6 leaves of shape (1, 4), 8 traces, all 1 row."""
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    return pd.DataFrame(np.random.default_rng(seed).normal(size=(6, 4)),
                        index=idx)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _solid(ax):
    return [ln for ln in ax.lines if ln.get_linestyle() in ('-', 'solid')]


def _dashed(ax):
    return [ln for ln in ax.lines if ln.get_linestyle() not in ('-', 'solid')]


def test_every_plotted_trajectory_gets_its_own_forecast():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    ax = _ax(fig)
    assert len(_solid(ax)) == 4 and len(_dashed(ax)) == 4


def test_bundle_forecasts_correspond_to_trace_data():
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert len(out['predict']['forecasts']) == 4
    assert len(_dashed(_ax(out['fig']))) == 4


def test_each_bundled_forecast_equals_hyp_predict_on_its_trace():
    """Contract 5, asserted numerically for EVERY trace including the mean."""
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=2,
                   return_model=True, show=False)
    for trace, forecast in zip(out['trace_data'], out['predict']['forecasts']):
        direct = np.asarray(hyp.predict(np.asarray(trace), model='Kalman',
                                        t=2), dtype=float)
        assert np.allclose(np.asarray(forecast, dtype=float), direct,
                           rtol=1e-6, atol=1e-6)


def test_leaf_forecasts_match_hyp_predict_on_xform_data_when_spaces_coincide():
    """The v1.0 promise (plot.py:1935-1941) still holds for the leaves --
    but ONLY where the analysed space and the plotted space are the same.

    Contract 5 makes that conditional: a `reduce=` spec pinning more than
    three components leaves `xform_data` in the higher-dimensional space
    while `trace_data` is projected for display, and then this comparison is
    meaningless. The guard below is the condition, asserted rather than
    assumed; `tests/plot/test_hierarchy_bundle.py` covers the diverging
    case, where forecasts follow `trace_data`.
    """
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=2,
                   return_model=True, show=False)
    leaves = out['xform_data']
    assert all(np.asarray(x).shape == np.asarray(tr).shape
               for x, tr in zip(leaves, out['trace_data'])), \
        'this assertion is only valid when the two spaces coincide'
    direct = hyp.predict([np.asarray(x) for x in leaves], model='Kalman', t=2)
    for got, want in zip(out['predict']['forecasts'][:len(leaves)], direct):
        assert np.allclose(np.asarray(got, dtype=float),
                           np.asarray(want, dtype=float),
                           rtol=1e-6, atol=1e-6)


def test_forecasts_are_not_silently_dropped():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=3, show=False)
    assert len(_dashed(_ax(fig))) == 4, 'forecasts vanished instead of raising'


def test_mean_trace_forecast_comes_from_the_mean_trajectory():
    """A mean trace is forecast from its OWN averaged trajectory, proven by
    exact equality with `hyp.predict(mean_traj)` -- which pins precisely
    which trajectory the bundled forecast came from, so the contract is
    proven completely.

    Comparing against the average of the LEAF forecasts is deliberately NOT
    asserted: forecasting approximately commutes with averaging as the
    leaves co-move, so a correct implementation fails such an assertion on
    exactly the data this plan targets. Measured (Kalman, t=1, T=150, 3
    leaves, scale ~100, 5 seeds per rho), the deleted assertion -- that the
    bundled forecast is NOT close to the average of the leaf forecasts at
    rtol=1e-3, atol=1e-3 -- held 5/5 at rho=0.0 (mean max abs diff 0.557)
    and 5/5 at rho=0.5 (0.524), but only 3/5 at rho=0.8 (0.130) and 0/5 at
    rho=0.9 (0.028), 0.95 (0.007) and 0.99 (0.0003). Real market sectors
    co-move at roughly rho 0.7-0.9. Do not re-add it.
    """
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    mean_traj = np.asarray(out['trace_data'][-1])
    from_mean = np.asarray(hyp.predict(mean_traj, model='Kalman', t=1),
                           dtype=float)
    bundled = np.asarray(out['predict']['forecasts'][-1], dtype=float)
    assert np.allclose(bundled, from_mean, rtol=1e-6, atol=1e-6)


def test_forecasts_anchor_on_their_own_trace():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    ax = _ax(fig)
    for line, fc in zip(_solid(ax), _dashed(ax)):
        drawn = np.array(line.get_data_3d())
        assert np.allclose(np.array(fc.get_data_3d())[:, 0], drawn[:, -1],
                           atol=1e-6)


def test_forecasts_stay_inside_the_axes_limits():
    fig = hyp.plot(market_frame(), '-', predict='Kalman', t=5, show=False)
    ax = _ax(fig)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    for fc in _dashed(ax):
        pts = np.array(fc.get_data_3d())
        assert (pts.min(axis=1) >= lims[:, 0] - 1e-6).all()
        assert (pts.max(axis=1) <= lims[:, 1] + 1e-6).all()


def test_predict_composes_with_a_continuous_hue():
    df = market_frame()
    fig = hyp.plot(df, '-', predict='Kalman', t=1,
                   hue=np.linspace(0, 1, len(df)), show=False)
    assert len(_dashed(_ax(fig))) == 4


def test_forecast_takes_the_final_observed_hue_colour():
    """F14, matplotlib half. A forecast under a continuous hue continues in
    the colour its source trace ended on."""
    df = market_frame()
    fig = hyp.plot(df, '-', predict='Kalman', t=1, palette='viridis',
                   hue=[np.linspace(0, 1, len(df)) for _ in range(3)],
                   show=False)
    ax = _ax(fig)
    colls = [c for c in ax.collections if isinstance(c, LineCollection)]
    for coll, fc in zip(colls, _dashed(ax)):
        last = np.asarray(coll.get_colors())[-1][:3]
        assert np.allclose(matplotlib.colors.to_rgb(fc.get_color()), last,
                           atol=0.02)


def test_predict_with_hierarchy_and_animation_via_on_frame():
    """Assertions go through the PUBLIC on_frame hook (animation-core Task
    7), not through ani._args. Frames are driven with the same `_drive`
    helper that plan's own test module defines."""
    seen = []
    fig, ani = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                        animate=True, duration=2, frame_rate=4,
                        on_frame=seen.append, show=False)
    for f in range(8):
        ani._func(f, *ani._args)          # harness only; never asserted on
    assert len(seen) == 8
    assert len(seen[-1].datasets) == 4
    assert len(_dashed(_ax(fig))) == 4


def test_return_model_bundle_has_one_model_and_forecast_per_trace():
    out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                   return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert len(out['predict']['forecasts']) == 4
    assert out['predict']['params'] == {'t': 1}
    assert out['predict']['model'] == 'Kalman'
    assert len(out['trace_metadata']['keys']) == 4


# --- Contract 10: the >= 2-row precondition, on BOTH axes -------------------

def test_row_hierarchy_with_multi_row_leaves_forecasts_every_trace():
    """Row hierarchies DO forecast when the shape allows it.

    The construction is verified before the counts are asserted, so a
    silently-degenerate frame fails here rather than passing vacuously."""
    from hypertools.plot.multiindex import expand_multiindex

    df = multirow_row_frame(n_time=10)
    leaves, _ = expand_multiindex(df)
    assert len(leaves) == 6
    assert all(np.asarray(leaf).shape[0] == 10 for leaf in leaves), \
        'the frame must have MULTI-row leaves for this test to mean anything'

    out = hyp.plot(df, '-', predict='Kalman', t=2, return_model=True,
                   show=False)
    assert all(np.asarray(tr).shape[0] >= 2 for tr in out['trace_data'])
    assert len(out['trace_data']) == 8
    assert len(out['predict']['forecasts']) == 8
    ax = _ax(out['fig'])
    assert len(_solid(ax)) == 8 and len(_dashed(ax)) == 8


def test_row_hierarchy_with_one_row_leaves_raises_naming_the_trace():
    """Contract 10's other side. The message must be about the DATA -- it
    names the trace and its row count and explains the leaf rule -- not a
    bubbled-up `predict` internal error, so it is raised as a precondition
    over `ft.arrays` before any forecasting."""
    from hypertools.plot.multiindex import expand_multiindex

    df = one_row_row_frame()
    leaves, _ = expand_multiindex(df)
    assert all(np.asarray(leaf).shape[0] == 1 for leaf in leaves)

    with pytest.raises(ValueError) as excinfo:
        hyp.plot(df, '-', predict='Kalman', t=1, show=False)
    message = str(excinfo.value)
    assert 'at least 2 rows per trace' in message
    assert '1 row' in message
    assert 'unique FULL index tuple' in message
    assert 'reset_index(drop=True)' in message
    assert 'COLUMN axis' in message
    assert 'cannot forecast from a single observation' not in message, \
        'the bubbled-up predict error must not be what the user sees'


def test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping():
    """The precondition is NOT row-specific.

    A column hierarchy never shortens a trace -- every group keeps all
    `len(df)` rows -- but it cannot lengthen one either. Measured with this
    plan's grouping idiom: T=1 gives leaf shapes {'Tech': (1, 3),
    'Fin': (1, 3)} and a mean of (1, 3), NOT forecastable; T=2 gives
    (2, 3) throughout, forecastable. So the check must run on this axis
    too, and its message must be about the INPUT having one observation --
    flattening the hierarchy cannot add a row, so it is not offered.

    A 1-row frame also warns from `reduce` ('Cannot reduce a single
    observation (row) of data ...', reduce.py:455). That is orthogonal to
    what is asserted here, so it is tolerated rather than allowed to fail
    the test.
    """
    df = market_frame(T=1)
    assert len(df) == 1

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='Cannot reduce a single observation')
        with pytest.raises(ValueError) as excinfo:
            hyp.plot(df, '-', predict='Kalman', t=1, show=False)

    message = str(excinfo.value)
    assert 'at least 2 rows per trace' in message
    assert '1 row' in message
    assert 'one observation' in message
    assert 'reset_index(drop=True)' not in message, \
        'flattening a COLUMN hierarchy cannot add a row -- do not offer it'
    assert 'cannot forecast from a single observation' not in message, \
        'the bubbled-up predict error must not be what the user sees'


def test_animated_one_row_hierarchy_still_raises_the_precondition():
    """The precondition runs BEFORE the forecast schedule is built.

    Plan 3's `min_history` returns None for a frame whose revealed history
    is too short -- correct for the opening frames of a real animation. But
    a one-row hierarchy never reaches 2 rows at ANY frame, so deferring to
    that path would draw no forecast forever, silently. The full-trace
    precondition is a permanent property of the data, so it raises here
    exactly as it does for a static plot.
    """
    with pytest.raises(ValueError, match='at least 2 rows per trace'):
        hyp.plot(one_row_row_frame(), '-', predict='Kalman', t=1,
                 animate=True, duration=2, frame_rate=4, show=False)


def test_one_row_row_hierarchy_still_plots_without_predict():
    """The precondition is scoped to `predict=`; plotting is untouched."""
    fig = hyp.plot(one_row_row_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 8
```

- [ ] **Step 2: Run and confirm failure**

Run: `.venv/bin/python -m pytest tests/plot/test_multiindex_predict.py -v`
Expected: every test FAILS with `ValueError: predict= is not supported with MultiIndex expansion in this release`, except `test_one_row_row_hierarchy_still_plots_without_predict`, which PASSES already (it does not pass `predict=`). Note that the three precondition tests — `test_row_hierarchy_with_one_row_leaves_raises_naming_the_trace`, `test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping` and `test_animated_one_row_hierarchy_still_raises_the_precondition` — also fail at this point: today's blanket refusal raises, but with the *old* message.

- [ ] **Step 3: Implement**

1. Delete the blanket refusal at `plot.py:2669-2677`. It is **replaced, not simply removed**: `predict=` works for either axis, but only when the shape allows it — every final trace needs ≥ 2 rows (Contract 10) — so a narrower, later check takes its place, item 3a below.
2. Move the forecast computation (`plot.py:2963-2988`) to **after** `FinalTraces` is built, iterating `ft.arrays` so leaves and means are forecast alike; a mean is forecast from its own averaged trajectory.
3. Recompute `pre_interp_lengths` (`plot.py:2999`) from `ft.arrays`.
   - **3a. The ≥ 2-row precondition, for every hierarchy.** Immediately after `ft` is built and **before** any call into `hyp.predict` — and, when `animate=` is set, **before** the `ForecastSchedule` is constructed — when `predict is not None`, scan **every** entry of `ft.arrays` for the first with fewer than 2 rows and raise, interpolating that trace's position, its key (`ft.keys[i]`) and its row count. Scanning `ft.arrays` covers derived means as well as leaves. **The check is not gated on the axis**: a `T=1` column frame is measurably unforecastable too (leaves `(1, 3)`, mean `(1, 3)`), so gating on `'rows'` would let it fall through to `predict/common.py:256`'s internal shape error. `ft.meta['axis']` selects only the **remediation sentence** of the message: `'rows'` gets the one-trace-per-full-index-tuple explanation plus `df.reset_index(drop=True)` / move-to-columns; `'columns'` says the input itself has a single observation and offers neither, because flattening cannot add a row. Both variants are quoted at the top of this task. Raising here — rather than letting the forecaster fail — is what makes the message about the user's data instead of an internal shape error.
4. Replace the silent null-out at `plot.py:3999` with, for hierarchical input, `ft.assert_consistent(raw_forecasts=raw_forecasts, bundle_forecasts=bundle_forecasts)`. Keep the existing silent guard for the **non**-hierarchical hue/cluster-regrouping case that `plot.py:4000-4002`'s comment describes — that is a different situation (regrouping by category), it is pinned by Plan 3's `test_hue_regrouping_drops_forecasts_exactly_like_the_static_path`, and README-hypertools-1.1.md's open decision *"Silent forecast drop under `hue=`/`cluster=`"* leaves it as status quo (cited by name, not number — these get renumbered). **If Plan 3 Task 2 lands first, extend its version of the guard rather than replacing it.**
5. Under a continuous hue, give each forecast overlay `_forecast_colors[i]` from Task 6 Step 3.5.

- [ ] **Step 4: Run and confirm pass** — `.venv/bin/python -m pytest tests/plot/test_multiindex_predict.py -v` → **17 passed**.

- [ ] **Step 5: Rewrite the test pinning the old refusal, and add its counterpart**

Both edits are in `tests/test_multiindex.py`, and together they take it from **29** to **30** tests.

1. **Rewrite** `test_predict_plus_multiindex_raises` (`:479`) as `test_predict_plus_multiindex_forecasts_every_trace`. Its frame is `_make_2level_df()` (`tests/test_multiindex.py:45-61`), which repeats each `(cond, subj)` tuple `n_time=10` times — measured: `expand_multiindex` gives **8 leaves, every one shape (10, 3)**, and `hyp.plot(df, show=False)` draws **10** traces (8 leaves `lw=1.0`, 2 means `lw=2.0`, labelled `condA`/`condB`). Every trace clears the ≥ 2-row precondition, so this frame now **forecasts**: assert 10 solid lines and 10 dashed forecasts. Tabulated under *Compatibility changes*. Keep the docstring's explanation of the leaf rule, updated.

   > This corrects a factual error in the v3→v4 review (v4 revision note **B3**), which described `:479`'s frame as having "8 traces, all 1 row". That describes the plan's own 6-row `2 cond × 3 subj` example (measured: 6 leaves of shape `(1, 4)`), not `_make_2level_df()`. The **rule** is applied exactly as directed; only the claim about which frame this test uses is corrected.

2. **Add** `test_predict_plus_one_row_row_hierarchy_raises`, built on the 6-row `2 cond × 3 subj` frame whose innermost level is unique per row (measured: 6 leaves of shape `(1, 4)`, 8 traces, all 1 row). Assert `pytest.raises(ValueError)` and the **new** message text — `'at least 2 rows per trace'`, `'unique FULL index tuple'`, `'reset_index(drop=True)'` — and assert the old blanket wording `'not supported with MultiIndex expansion'` is **gone**.

- [ ] **Step 6: Run the WHOLE suite** — `.venv/bin/python -m pytest -q`. Expected: baseline + 125 (107 + 17 new-module tests + 1 net in `tests/test_multiindex.py`). `tests/test_multiindex.py` now reports **30**.

- [ ] **Step 7: Document** the per-trace forecasting rule, the mean-trajectory choice, **and Contract 10's ≥ 2-row requirement — both axes, with its two error messages** — in the `predict=` docstring, replacing the "not supported" note, and in `plot()`'s `return_model` entry (Task 4). The `docs/hierarchy.rst` comparison table (Task 10) gains the same row.

- [ ] **Step 8: Commit**

```bash
git add hypertools/plot/plot.py tests/plot/test_multiindex_predict.py \
        tests/test_multiindex.py
git commit -m "feat(plot): predict= over column hierarchies and multi-row row hierarchies"
```

---

## Task 9: Full matplotlib/plotly parity

**Parity is required, not optional** (standing maintainer directive; README-hypertools-1.1.md *Standing decisions*: *"Plotly and matplotlib must behave identically. Where a capability cannot cross the browser boundary it raises, naming the backend; it never silently degrades."*). Every capability in Tasks 5-8 is Python-side geometry and styling — none of it needs a per-frame Python callback — so there is no case here that cannot cross the boundary. **There is no "or defer" branch.**

The existing `test_plot_2level_plotly_parity` (`tests/test_multiindex.py:336`) already pins the row-hierarchy shape exactly: **11 traces** = 10 data + 1 cube wireframe, per-trace `line.width`, alpha baked into an `rgba()` string, `name`/`showlegend` only on the two top-level means. This task holds the column path to the same standard.

**Two rejections need no per-backend work and get none here:** dual-axis frames, and Contract 10's ≥ 2-row precondition. Both live in `plot()` before any backend is chosen, and the precondition is **axis-independent** — it scans every final trace of every hierarchy, on both axes, differing only in the remediation sentence — so plotly inherits identical behaviour by construction rather than by parallel implementation. `test_dual_axis_frame_is_rejected_on_plotly` proves the pattern; the precondition's coverage stays in `tests/plot/test_multiindex_predict.py` (three tests: row, column, animated). What plotly **does** need is everything downstream of the check — one dashed trace per plotted trajectory, on either axis, in the final observed colour.

**Files:** Modify `hypertools/plot/plotly_backend.py`; Test `tests/plot/test_multiindex_plotly.py`

> **EXECUTED 2026-08-16 in `c9b91293` + `a309f49e`, plus the Task 7-8 triage
> commit `b48c2848`.** The steps below were ticked retroactively during Task 12,
> which found this task had shipped with every box unticked and no EXECUTED
> note; the numbers are the ones MEASURED in those commit bodies, attributed to
> their shas, not re-measured here. What the task got wrong, in brief (the
> commit messages carry the full measurements):
>
> 1. **The prescribed test block was 9 failed / 3 passed and could not have
>    passed as written** (`c9b91293`): `_data_traces` filtered `name != 'cube'`
>    but the cube trace is UNNAMED; `t.line.dash` selects every line because
>    plotly spells solid `dash='solid'`; the F14 colour test compared an
>    `rgba(...)` slice against an `rgb(...)` slice, which can never be equal,
>    and gave both leaves the same hue so a mis-pairing passed;
>    `[1.0, 1.0, 2.0]` are matplotlib POINTS while plotly's `line.width` is
>    pixels; `len(t.line.color) == len(df)` ignores `antialias=True`; and
>    `test_colorbar_renders_on_plotly` was VACUOUS (plotly instantiates a
>    `ColorBar` on every trace — `marker.showscale` is the discriminator).
> 2. **Step 3's `_forecast_colors` does not exist** — Task 6 implemented F14 as
>    `_apply_multicolor_lines` anchoring on `line_colors[dataset][-1]`, so the
>    plotly side needed a new `_hue_anchor_color`, not a consumer of a symbol
>    the plan invented.
> 3. **Step 4's "12 passed" became 14** (`c9b91293`), then **23** after the
>    review commit `a309f49e` added the alpha, marker-forecast, 1-D-seam and
>    row-hierarchy-on-plotly cases.
> 4. **Step 5's "baseline + 137 (125 + 12)" is a carry-forward** and does not
>    reconcile; measured absolutely instead: **3420** at `c9b91293`, **3429**
>    at `a309f49e`, **3443** at `b48c2848`.

- [x] **Step 1: Write the failing test** *(EXECUTED at `c9b91293`: written with the six corrections in note 1; 14 tests, all failing before the change.)*

```python
# tests/plot/test_multiindex_plotly.py
"""Backend parity for hierarchies: exact counts, styles, hue, forecasts.

The matplotlib expectations these mirror live in test_column_multiindex.py,
test_multiindex_hue.py and test_multiindex_predict.py. `>= 3` was the v2
assertion; every count here is exact, because a duplicated or extra trace
is precisely the failure mode parity work introduces.
"""
import numpy as np
import pandas as pd
import pytest

import hypertools as hyp

pytest.importorskip('plotly')


def market_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    tuples = [('Market', sector, m)
              for sector in ('Tech', 'Energy')
              for m in ('return', 'volatility', 'momentum')]
    cols = pd.MultiIndex.from_tuples(tuples,
                                     names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 6)).cumsum(axis=0) + 100.0, columns=cols)


def two_level_frame(T=40, seed=0):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples(
        [(g, f) for g in ('A', 'B', 'C') for f in ('f0', 'f1', 'f2')],
        names=['Group', 'Feature'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0), columns=cols)


def _data_traces(fig):
    """Data traces only -- the 3-D cube wireframe is not one of them."""
    return [t for t in fig.data
            if t.type in ('scatter3d', 'scatter')
            and getattr(t, 'name', None) != 'cube']


def _rgb(rgba):
    return rgba.rsplit(',', 1)[0]


def _alpha(rgba):
    return float(rgba.rstrip(')').rsplit(',', 1)[1])


def _plot(*args, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(*args, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def test_three_level_column_hierarchy_exact_trace_count_and_order():
    fig = _plot(market_frame(), '-', show=False)
    traces = _data_traces(fig)
    assert len(traces) == 3, '2 sector leaves + 1 market mean'
    assert traces[-1].name == 'Market'


def test_plotly_widths_match_the_documented_formula():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    assert [t.line.width for t in traces] == pytest.approx([1.0, 1.0, 2.0])


def test_plotly_opacities_match_the_documented_formula():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    assert [_alpha(t.line.color) for t in traces] == pytest.approx(
        [0.7, 0.7, 1.0])


def test_plotly_legend_labels_only_the_top_level_mean():
    traces = _data_traces(_plot(market_frame(), '-', show=False))
    assert [t.showlegend for t in traces] == [False, False, True]
    assert len(set(_rgb(t.line.color) for t in traces)) == 1


def test_two_level_column_hierarchy_labels_every_trace():
    traces = _data_traces(_plot(two_level_frame(), '-', show=False))
    assert len(traces) == 3
    assert [t.name for t in traces] == ['A', 'B', 'C']
    assert [t.showlegend for t in traces] == [True, True, True]
    assert len(set(_rgb(t.line.color) for t in traces)) == 3


def test_continuous_price_hue_renders_per_point_colours():
    df = market_frame()
    hues = [df['Market'][s].mean(axis=1).to_numpy() for s in ('Tech', 'Energy')]
    traces = _data_traces(_plot(df, '-', hue=hues, palette='viridis',
                                show=False))
    assert len(traces) == 3
    per_point = [t for t in traces
                 if not isinstance(t.line.color, str)
                 and t.line.color is not None]
    assert len(per_point) == 3
    assert all(len(t.line.color) == len(df) for t in per_point)


def test_colorbar_renders_on_plotly():
    df = market_frame()
    fig = _plot(df, '-', hue=np.linspace(0, 1, len(df)), colorbar=True,
                show=False)
    has_bar = [t for t in fig.data
               if getattr(getattr(t, 'marker', None), 'colorbar', None)
               is not None
               or getattr(getattr(t, 'line', None), 'colorbar', None)
               is not None]
    assert has_bar, 'expected a colorbar-bearing trace'


def test_predict_draws_one_dashed_trace_per_drawn_trace():
    fig = _plot(market_frame(), '-', predict='Kalman', t=1, show=False)
    traces = _data_traces(fig)
    dashed = [t for t in traces if getattr(t.line, 'dash', None)]
    assert len(dashed) == 3
    assert len(traces) == 6


def test_plotly_forecast_takes_the_final_observed_hue_colour():
    """F14, plotly half -- the same rule as matplotlib."""
    df = market_frame()
    hues = [np.linspace(0.0, 1.0, len(df)) for _ in range(2)]
    traces = _data_traces(_plot(df, '-', hue=hues, palette='viridis',
                                predict='Kalman', t=1, show=False))
    observed = [t for t in traces if not getattr(t.line, 'dash', None)]
    dashed = [t for t in traces if getattr(t.line, 'dash', None)]
    for obs, fc in zip(observed, dashed):
        assert _rgb(str(fc.line.color)) == _rgb(str(obs.line.color[-1]))


def test_return_model_bundle_matches_the_matplotlib_bundle():
    hyp.set_interactive_backend('plotly')
    try:
        out = hyp.plot(market_frame(), '-', predict='Kalman', t=1,
                       return_model=True, show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    assert len(out['xform_data']) == 2
    assert len(out['trace_data']) == 3
    assert len(out['predict']['forecasts']) == 3
    assert out['trace_metadata']['is_mean'] == [False, False, True]


def test_hierarchy_with_animated_prediction():
    """Prerequisite: forecast-animation Tasks 1-2 (the precomputed
    schedule) and its Task 6 (plotly parity)."""
    fig = _plot(market_frame(), '-', predict='Kalman', t=1, animate=True,
                duration=2, frame_rate=4, show=False)
    assert getattr(fig, 'frames', None), 'expected plotly animation frames'
    first = _data_traces(fig)
    assert len(first) == 6


def test_dual_axis_frame_is_rejected_on_plotly():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(20)])
    cols = pd.MultiIndex.from_tuples([('M', 'T'), ('M', 'E')])
    df = pd.DataFrame(np.zeros((40, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        _plot(df, '-', show=False)
```

- [x] **Step 2: Run and confirm failure** *(EXECUTED at `c9b91293`: all 14 failed before the change; red/green re-proved by mutation afterwards — disabling `anchor_color` fails exactly the two forecast-hue tests, restoring truncation in `_to_plotly_color` fails exactly the three colour tests.)*

Run: `.venv/bin/python -m pytest tests/plot/test_multiindex_plotly.py -v`
Expected: the column-hierarchy, hue, forecast and bundle tests FAIL; the dual-axis test passes (rejection is backend-independent, in `plot()`).

> `_plot()` restores the backend in a `finally`, the pattern `tests/test_multiindex.py:336` and `tests/test_backend_state_safety.py` already use. If a shared fixture exists by the time this runs, prefer it.

- [x] **Step 3: Implement** *(EXECUTED at `c9b91293`, extended at `a309f49e`. Measured first: the geometry and styling ALREADY crossed correctly; what did not was the continuous-hue forecast colour, the two disagreeing colour serializers, the missing "which traces are DATA" handle (now `meta['hyp_trace_index']`), and — from the review — per-trace alpha under a hue, the marker-only forecast anchor, and matplotlib's 1-D forecast x-offset.)*

Whatever `plotly_backend.py` needs to consume `FinalTraces` + the style dict + `_forecast_colors`: per-trace `line.width`, alpha baked into the `rgba()` string, `name`/`showlegend` from the labels, a per-point colour list for a continuous hue, and one dashed trace per plotted trajectory with the final observed colour. **No `NotImplementedError` is acceptable here** — every one of these is Python-side data preparation that plotly already supports (the row path proves it).

- [x] **Step 4: Run and confirm pass** — ~~**12 passed**~~ *(EXECUTED: **14 passed** at `c9b91293`, **23 passed** at `a309f49e`, **63 passed** across the focused set at `b48c2848`.)*

- [x] **Step 5: Run the WHOLE suite** — ~~baseline + 137 (125 + 12)~~ *(EXECUTED, measured absolutely per Global Constraints: **3420** at `c9b91293`, **3429** at `a309f49e`, **3443** at `b48c2848`, each 13 skipped / 2 deselected with no warnings summary; ruff parity empty both ways and the `-W -E -a` docs build succeeded at each.)*

- [x] **Step 6: Commit** *(EXECUTED: `c9b91293`, then `a309f49e` for the nine review findings, then `b48c2848` for the Tasks 7-8 Minor triage.)*

```bash
git add hypertools/plot/plotly_backend.py tests/plot/test_multiindex_plotly.py
git commit -m "feat(plot): plotly parity for hierarchies, hue and per-trace forecasts"
```

---

## Task 10: The "Hierarchical DataFrames" guide, and the docs that point at it

Docstrings alone are not sufficient for these semantics (F22), the row-plot/row-forecast divergence needs a **visible** comparison table (F8), `docs/tutorials.rst:148` still describes the market notebook as *"A market as one moving path"* (F20), `docs/index.rst:35-36` says only *"Pandas DataFrames (including MultiIndex)"* without distinguishing the axes (F20), and `docs/pipeline_order.rst` documents a linear pipeline with no room for hierarchy expansion or mean construction (F21).

**Files:** Create `docs/hierarchy.rst`; Modify `docs/index.rst`, `docs/api.rst`, `docs/tutorials.rst`, `docs/pipeline_order.rst`, `scripts/round17_evidence/pipeline_order_diagram.py`, `docs/_static/pipeline_order.svg`; Test `tests/test_docs_hierarchy_guide.py`

> **EXECUTED 2026-08-16 in `f2a7a2b1`, corrected by `cdae7096`.** Backfilled
> during Task 12, which found this task had shipped with no EXECUTED note and
> no ticked step; the numbers below are the ones MEASURED in those two commit
> bodies, attributed to their shas, not re-measured here. Three places the task
> was wrong about the tree, and two later corrections:
>
> 1. **Step 4's market-section retitle could not be written.** It asks
>    `docs/tutorials.rst` to describe the market notebook as a hierarchy, but
>    `docs/tutorials/market_forecast.ipynb` contains **0** MultiIndex
>    constructions — that rewrite is Plan 4 Task 2 Step 5, which has not
>    landed. Writing the synopsis would have made `tutorials.rst` describe a
>    notebook that does not exist. The plan's `assert 'one moving path' not in
>    tut` was **replaced, not dropped**, by a two-way guard: while the notebook
>    is flat the old title is REQUIRED, and the moment it gains a MultiIndex
>    the test fails naming Plan 4 Task 2 and the synopsis to write. The
>    hierarchy link went to the WEATHER section instead, which genuinely is the
>    bold-means/faint-leaves tutorial.
> 2. **`.. doctest::` needs `sphinx.ext.doctest`, absent from `docs/conf.py`** —
>    without it the `-W` build failed with 22 "Unknown directive type" errors.
> 3. **The plan's own `test_api_rst_links_the_guide` could not fail.**
>    `':doc:`hierarchy`' in api or 'hierarchy' in api` makes the first clause
>    dead, and `count('hierarchy') >= 2` is met by any two mentions anywhere.
>    Replaced with a test requiring a real `:doc:` link inside BOTH the Predict
>    and Plot sections; mutation-verified.
> 4. **Two prose claims were measured false while writing** and corrected
>    before shipping: the reason `xform_data`/`trace_data` are distinct lists
>    (an `n_levels == 1` hierarchy has no means, equal lengths, equal contents,
>    and they are *still* distinct), and the forecast-colour-at-the-anchor
>    claim, which was verified with disjoint per-leaf hues rather than asserted.
> 5. **`cdae7096` then found four more**, two of them in this guide: the
>    Limitations section gave the ROW remedy for a refusal that fires on BOTH
>    axes (`reset_index(drop=True)` does not clear a column MultiIndex), and
>    "a datetime innermost level comes back as a DatetimeIndex … with its name
>    intact" was false of the object shown (`forecasts[0].index.name is None`;
>    the name survives on the GROUP, and `hyp.predict` builds a fresh unnamed
>    horizon index on FLAT input too). The guide grew 126 → **138** executed
>    examples.

- [x] **Step 1: Write the failing test** *(EXECUTED at `f2a7a2b1`: written with the corrections in notes 1 and 3.)*

```python
# tests/test_docs_hierarchy_guide.py
"""The hierarchy guide exists, is reachable, and covers its subject.

A guide that is written but never linked is not documentation. These
assertions are structural, not prose-quality: they pin the sections the
maintainer review named (F22) and the links that make the page reachable.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        return handle.read()


REQUIRED_SECTIONS = [
    'Row versus column semantics',
    'Plotting versus forecasting',
    'Hue over a hierarchy',
    'Mean trace construction',
    'Limitations',
    'Dual-axis and list inputs',
    'Return shapes',
    'Fitted model behaviour',
    'Backend parity',
    'Feature names and duplicates',
]


def test_the_guide_page_exists():
    assert os.path.exists(os.path.join(REPO, 'docs/hierarchy.rst'))


def test_the_guide_is_in_the_index_toctree():
    index = _read('docs/index.rst')
    toctree = index.split('.. toctree::')[1]
    assert re.search(r'^\s+hierarchy\s*$', toctree, re.M)


def test_the_guide_covers_every_required_section():
    guide = _read('docs/hierarchy.rst')
    missing = [s for s in REQUIRED_SECTIONS if s not in guide]
    assert not missing, f'hierarchy.rst is missing sections: {missing}'


def test_the_comparison_table_is_in_the_guide():
    """The row-plot vs row-forecast divergence, visibly -- including the
    shape rule that decides whether `plot(..., predict=)` works at all
    (Contract 10)."""
    guide = _read('docs/hierarchy.rst')
    assert 'Row MultiIndex, plot' in guide
    assert 'Row MultiIndex, predict' in guide
    assert 'Column MultiIndex' in guide
    assert 'full tuple' in guide
    assert 'at least 2 rows' in guide


def test_api_rst_links_the_guide():
    api = _read('docs/api.rst')
    assert ':doc:`hierarchy`' in api or 'hierarchy' in api
    assert api.count('hierarchy') >= 2, 'link it from Plot AND Predict'


def test_tutorials_rst_links_the_guide_and_drops_one_moving_path():
    tut = _read('docs/tutorials.rst')
    assert 'one moving path' not in tut
    assert 'hierarchy' in tut


def test_index_rst_distinguishes_row_and_column_semantics():
    index = _read('docs/index.rst')
    assert 'row MultiIndex' in index and 'column MultiIndex' in index


def test_pipeline_order_documents_the_hierarchy_branch():
    po = _read('docs/pipeline_order.rst')
    assert 'hierarchy' in po.lower()
    assert 'expansion' in po.lower()
    assert 'mean trace' in po.lower()
```

- [x] **Step 2: Run and confirm failure** *(EXECUTED at `f2a7a2b1`. Note the prescribed reason is half wrong: `docs/hierarchy.rst` genuinely did not exist, but `'one moving path'` is the title the notebook still legitimately carries — see note 1 — so the guard was inverted rather than satisfied.)*

Run: `.venv/bin/python -m pytest tests/test_docs_hierarchy_guide.py -v`
Expected: 8 failed — `docs/hierarchy.rst` does not exist, and `'one moving path'` **is** currently in `docs/tutorials.rst:148`.

- [x] **Step 3: Write `docs/hierarchy.rst`** *(EXECUTED at `f2a7a2b1`: ten sections, **126** executed doctests — **138** after `cdae7096` — every claim measured against the code at `b48c2848` before it was written, and `test_every_doctest_in_the_guide_runs` mutation-proven (changing the linewidth formula, or disabling nominal correspondence, each fail it).)*

Sections, in order, each with a runnable example:

1. **Row versus column semantics.** What each axis means, with the comparison table below. State that a column group's leaf is **flattened onto the feature axis** — its columns become the innermost level's values, keeping that level's name (`(Market, Sector, Measure)` → `['return', 'volatility', 'momentum']`, named `Measure`) — and that a row forecast group keeps the innermost level as its **flat** index, with its name and dtype intact. Grouping never modifies the frame you passed in.
2. **Plotting versus forecasting** — why they diverge, what a `(Sector, day)` frame does in each, and **when `plot(..., predict=)` is defined**: whenever every plotted trace has at least 2 rows, on either axis (Contract 10). For a **column** hierarchy that means whenever the frame itself has at least 2 rows, since every group keeps all of them; for a **row** hierarchy it means every leaf *and* every derived mean, because row expansion draws one trace per unique full index tuple. Show a frame that qualifies (several timepoints per `(cond, subj)`) and both that do not — an innermost level unique per row, and a `T=1` column frame — with the error text of each, noting that the column message does not offer flattening because flattening cannot add a row.
3. **Hue over a hierarchy** — the two accepted forms, the rejected total-observations form, mean-trace hue, the row-hierarchy exception, forecast colour.
4. **Mean trace construction** — leaves first then means deepest-last; the exact style formulas; the `n_levels == 1` label rule.
5. **Limitations** — groups whose innermost labels differ (ragged groups included) are refused by name, since correspondence is nominal; unequal-length row groups are truncated to their overlap, with a warning.
6. **Dual-axis and list inputs** — dual-axis frames rejected on both entry points; hierarchical frames in lists rejected **asymmetrically** (`hyp.plot` rejects a column hierarchy and still warns-and-flattens a row one; `hyp.predict` rejects either axis), with the exact errors, the reason for the asymmetry and the flattening recipes.
7. **Return shapes** — the `return_model=True` bundle; `xform_data` (analysed pipeline output) vs `trace_data` (the final pre-center/pre-scale plotted trajectories), **the same object only when no display-only projection occurred**, with forecasts always corresponding to `trace_data`; and `hyp.predict`'s parallel sequences.
8. **Fitted model behaviour** — the unfitted/fitted ownership table from Task 7.
9. **Backend parity** — matplotlib and plotly draw the same hierarchy; nothing degrades silently.
10. **Feature names, correspondence and duplicates** — the innermost labels are **feature identities**: every group must carry the same ones, and later groups are permuted into the first group's order before analysis, so reordering a group's columns changes nothing. Groups with different labels — one ticker per sector, say — raise `ValueError` naming the missing and unexpected features; make the innermost level shared measurements (`return`, `volatility`), or, if slot *i* really is the same feature everywhere, opt in deliberately with `group_columns(df, feature_correspondence='position')` and pass the resulting arrays — noting that this is a **lower-level escape hatch, not positional hierarchy plotting**: it draws a plain list of datasets, so the per-level means, the hierarchy styling and `trace_metadata` are all absent. There is no hierarchy-preserving positional mode in 1.1. Note `align=` does **not** substitute for correspondence. Flattening can also leave two identical labels inside one group (two share classes of one issuer, a repeated sensor). That is **permitted**: every column survives, the group is not split or merged, and both plotting and forecasting work normally (measured: widths `[3, 3]`, `np.asarray` → `(20, 3)`, `hyp.predict` → `(1, 3)`, `hyp.plot` → a `Figure`); such labels are matched across groups by `(label, occurrence)`, so each group needs the same *number* of them. Duplicates **across** different groups were always fine and stay fine — `('M','Tech','X')` and `('M','Energy','X')` remain two separate leaves.

The comparison table is normative (reproduce the reviewer's, as an RST `list-table`):

| Axis and consumer | Innermost level means | Grouping | `plot(..., predict=)` |
|-|-|-|-|
| Row MultiIndex, plot | part of leaf identity | full tuple | only when every leaf and mean has at least 2 rows |
| Row MultiIndex, predict | time/observation | all outer levels | n/a — `hyp.predict` groups by the outer levels instead |
| Column MultiIndex, plot/predict | feature name | all outer levels | whenever the frame has at least 2 rows — every group keeps all of them |

- [x] **Step 4: Register and link it** *(EXECUTED at `f2a7a2b1`, except the market-section retitle — see note 1; the link went to the weather section, which is genuinely the hierarchy tutorial. Verified in the built HTML at Task 12 Step 3: `hierarchy.html` is referenced **14x** from `index.html`, **3x** from `api.html`, **2x** from `tutorials.html`.)*

- `docs/index.rst:41-48` — add `hierarchy` to the toctree, after `pipeline_order`.
- `docs/index.rst:35-36` — replace *"Pandas DataFrames (including MultiIndex)"* with a formulation naming both axes and linking the guide, e.g. *"Pandas DataFrames, including hierarchical frames — a **row MultiIndex** groups observations into leaf trajectories, a **column MultiIndex** groups features into per-group trajectories (see :doc:`hierarchy`)"*.
- `docs/api.rst` — a one-paragraph hierarchy note plus `see :doc:`hierarchy`` in **both** the Plot section (`api.rst:108-116`) and the Predict section (`api.rst:100-107`).
- `docs/tutorials.rst:148-149` — retitle the market section away from *"one moving path"* to the hierarchy framing, and add a synopsis sentence naming sectors-as-leaves, the market mean, price hue and per-trace forecasts, with `:doc:`../hierarchy`` — **coordinate with Plan 4 Task 8 Step 6**, which adds the thumbnail to the same section; whichever lands second keeps both edits.
- Every docstring changed in Tasks 4-8 gains `See docs/hierarchy.rst`.

- [x] **Step 5: Add the hierarchy side branch to `pipeline_order.rst` and regenerate the SVG** *(EXECUTED at `f2a7a2b1`. SVG regenerated **59 106 → 79 580 bytes**; two rendering defects fixed rather than shipped — one branch arrow drew backwards (both are now built explicitly, verified by rendering to PNG and looking at it), and the `predict (overlay)` box had its right edge clipped by `xlim`, a defect this script had been shipping since round17. Placement verified against `plot.py`, not asserted: expansion at `:3766`/`:3789` runs before `format_data` (`:3867`) and `analyze` (`:3950`); mean construction at `:4459` runs after the display reduce (`:4083`), which is why means reach `trace_data` and never `xform_data`.)*

The new operations sit **outside** the linear chain, so document them as a side branch rather than pretending mean construction is an ordinary stage:

```
    load/format (impute happens here)
      -> [hierarchy expansion, if x is a hierarchical DataFrame]
      -> manip -> normalize -> reduce -> align
      -> cluster (hue)
           \-> [hierarchy: mean trace construction + hue propagation]
      -> plot/animate
      -> predict overlays (one per plotted trajectory, when the
                           shape allows it -- see Contract 10)
```

Add a *"Where hierarchy expansion fits"* subsection explaining: expansion happens **before** format/analyze (so every leaf goes through the identical canonical pipeline); mean traces are built **after** reduce/align, in the plotted space, which is why they are `trace_data` and not `xform_data`; hue is co-propagated at that point; forecasting runs last, over the final traces.

Then update `scripts/round17_evidence/pipeline_order_diagram.py` (its `STAGES` list, and `OUT = docs/_static/pipeline_order.svg`, `plt.savefig(OUT, format='svg', bbox_inches='tight')`) to draw the side branch, and regenerate:

```bash
.venv/bin/python scripts/round17_evidence/pipeline_order_diagram.py
ls -la docs/_static/pipeline_order.svg
```
Expected: the SVG is rewritten (its mtime changes; it was 59 106 bytes before). Update the `:alt:` text at `docs/pipeline_order.rst:18-20` to mention the hierarchy branch — an unchanged alt text describing a diagram that changed is a documentation defect.

- [x] **Step 6: Run and confirm pass** — ~~**8 passed**~~ *(EXECUTED at `f2a7a2b1`: **12 passed** — the extra four come from note 3's replacement api-link test and the tutorials two-way guard. Full suite **3455 passed** at that commit.)*

- [x] **Step 7: Build the docs to the RTD-parity standard** *(EXECUTED at `f2a7a2b1`: `sphinx -b html -W -E -a` **succeeded with 0 warnings**, which also proves Step 4's toctree registration; re-run and re-confirmed at Task 12 Step 3.)*

Run: `cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html 2>&1 | tail -30`
Expected: build succeeds with **0 warnings** (the bar the 1.0 release gate enforces). A new page not in any toctree is a Sphinx warning, so this also proves Step 4 landed.

- [x] **Step 8: Commit** *(EXECUTED: `f2a7a2b1`, with the guide's four measured corrections following in `cdae7096`.)*

```bash
git add docs/hierarchy.rst docs/index.rst docs/api.rst docs/tutorials.rst \
        docs/pipeline_order.rst docs/_static/pipeline_order.svg \
        scripts/round17_evidence/pipeline_order_diagram.py \
        hypertools/plot/plot.py hypertools/predict/predict.py \
        tests/test_docs_hierarchy_guide.py
git commit -m "docs(1.1): hierarchical DataFrames guide; pipeline-order hierarchy branch"
```

---

## Task 11: CHANGELOG — create the 1.1.0 section

The tree's top section is `## 1.0.1 (unreleased)` (`CHANGELOG.md:3`); there is no `## 1.1.0` to add to (F23). This task creates it, and records the **compatibility changes** (F7) as first-class entries rather than burying them under new features.

**Files:** Modify `CHANGELOG.md`; Test `tests/test_changelog_1_1.py`

> **EXECUTED 2026-08-16, with amendments.** The section was written from
> `git log 59405545..HEAD` and the commits' own EXECUTED notes, not from the
> prose block below, because several things shipped differently from it.
> Everything asserted was measured at `f2a7a2b1`. Corrections:
>
> 1. **The prose block's *Documented limitations* still described POSITIONAL
>    feature correspondence** ("joint reduction stacks every group, so feature
>    position *i* is treated as…", a half-edited sentence). `c5662249` made
>    correspondence NOMINAL, permuting later groups into the first group's
>    order and refusing disjoint labels by name. The shipped section says so,
>    and records that `feature_correspondence='position'` is not a positional
>    hierarchy mode: measured (`442285af`), passing its arrays to `hyp.plot`
>    gives 3 traces instead of 4, `trace_metadata` `None`, and matplotlib's
>    default width on every line.
> 2. **The FOURTH compatibility change was missing.** Task 7's own note
>    flagged it for this task: `resolve_t` owns the duplicate-index check, so
>    it runs for FLAT inputs too. Reproduced here in both directions — at
>    `ea5d9b5e`, `hyp.predict(flat 5-row frame on a DatetimeIndex with one
>    repeated day, model='Kalman', t=1)` returned `(1, 3)`; at `f2a7a2b1` the
>    same call raises `ValueError: the dataset index has 1 duplicated entry
>    …`. It is now a first-class entry under *Changed / validation*, pinned by
>    a documentation test AND an executed one.
> 3. **`## 1.0.0 (unreleased)` was stale** (`CHANGELOG.md:460` before this
>    commit): 1.0.0 shipped to master on 2026-07-24 and `dev-1.0` never picked
>    up the release-time flip. `git show master:CHANGELOG.md` carries
>    `## 1.0.0 (2026-07-24)` and the two sections are otherwise byte-identical
>    (verified by diff), so the heading was corrected here rather than left
>    claiming a shipped release is unreleased.
> 4. **`pyproject.toml` had to move to 1.1.0**, which this task did not
>    anticipate. `tests/test_release_readiness_gate.py::test_changelog_top_
>    version_matches_pyproject` requires the FIRST `## X.Y.Z (...)` heading to
>    equal the project version, so creating a 1.1.0 section on top of a tree
>    declaring 1.0.1 fails an existing gate. Bumping is also what semver
>    requires: this section rejects four previously-accepted inputs, which
>    cannot ship in a patch release. **Follow-up for the maintainer** (not
>    done here, it belongs to Plan 1/3's deliverables): ~10 sites still say
>    "since 1.0.1"/"pre-1.0.1" (`hypertools/plot/forecast.py:61,64`,
>    `plot.py:190,1904,2841`, `plotly_backend.py:2694`, plus test docstrings),
>    and `tests/test_animation_guide_docs.py::test_animation_guide_version_
>    claims_match_the_package_version` forbids "new in 1.1"/"As of 1.1" in
>    `docs/animation.rst` on the premise that those features are 1.0.1. Both
>    assertions still pass, but the premise no longer holds once 1.0.1 is
>    never published.
> 5. **Two landed fixes were undocumented anywhere** and were added to the
>    1.0.1 section's *Bug fixes*: `d6a2ccdb` (Kalman forecasts diverging from
>    a near-saturated fit) and `bf17bb7d` (the singleton-hue warning naming
>    matplotlib's `'_nolegend_'` sentinel instead of the caller's category).
>    Every other `feat`/`fix` commit since `aab82600` was checked against the
>    file and is represented.
> 6. **Three of the six prescribed tests could not detect what they claim.**
>    `_section()` bounded a section at the next `\n## `, which does not match
>    `\n### `, so `_section(text, '### Changed / validation')` swallowed
>    *Documented limitations* and every "Changed says X" assertion could be
>    satisfied by text under Limitations; it now stops at the next
>    same-or-higher heading. `assert 'list' in changed.lower()` is satisfied
>    by "listed"; it asserts the real claim now. And nothing executed anything
>    — `test_the_documented_duplicate_time_rejection_actually_happens` runs
>    `hyp.predict` so the entry cannot drift from the code (it is the one test
>    that passes before the CHANGELOG edit, because it tests the shipped code;
>    mutation-proven: disabling the `resolve_t` branch fails it). 6 tests
>    became **9**.

- [x] **Step 1: Write the failing test** *(EXECUTED: written with the three strengthenings in correction 6, and 3 tests added — the duplicate-time entry, its executed counterpart, and the stale `## 1.0.0 (unreleased)` heading — for **9**.)*

```python
# tests/test_changelog_1_1.py
"""The 1.1.0 section exists, is on top, and records the behaviour changes.

A validation change that only appears under "New features" is a change
users will meet as a crash. `## 1.1.0 (unreleased)` did not exist when this
plan was written -- the top section was `## 1.0.1 (unreleased)`.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _changelog():
    with open(os.path.join(REPO, 'CHANGELOG.md'), encoding='utf-8') as handle:
        return handle.read()


def _section(text, heading):
    parts = text.split(heading, 1)
    assert len(parts) == 2, f'missing heading {heading!r}'
    return re.split(r'\n## ', parts[1])[0]


def test_changelog_has_a_1_1_0_unreleased_section():
    assert '## 1.1.0 (unreleased)' in _changelog()


def test_1_1_0_precedes_1_0_1():
    text = _changelog()
    assert text.index('## 1.1.0') < text.index('## 1.0.1')


def test_the_section_has_added_changed_and_limitations_headings():
    section = _section(_changelog(), '## 1.1.0 (unreleased)')
    for heading in ('### Added', '### Changed / validation',
                    '### Documented limitations'):
        assert heading in section, f'missing {heading}'


def test_changed_validation_documents_dual_axis_rejection():
    changed = _section(_changelog(), '### Changed / validation')
    assert 'both a row and a column MultiIndex' in changed


def test_changed_validation_documents_list_and_predict_changes():
    changed = _section(_changelog(), '### Changed / validation')
    assert 'list' in changed.lower()
    assert 'predict=' in changed


def test_added_documents_every_new_capability():
    added = _section(_changelog(), '### Added')
    for phrase in ('column MultiIndex', 'hue', 'hyp.predict', 'trace_data',
                   'plotly'):
        assert phrase in added, f'missing {phrase!r}'
```

- [x] **Step 2: Run and confirm failure** *(EXECUTED: **8 failed, 1 passed**. The one that passes is `test_the_documented_duplicate_time_rejection_actually_happens`, which tests the SHIPPED code rather than the file; mutation-proven instead — disabling the `resolve_t` duplicate branch fails it.)*

Run: `.venv/bin/python -m pytest tests/test_changelog_1_1.py -v`
Expected: 6 failed — `## 1.1.0 (unreleased)` is absent.

- [x] **Step 3: Write the section** *(EXECUTED: written from the commits, not from the block below — see corrections 1, 2, 3 and 5. `### Bug fixes` was added as a fourth heading for six defects the hierarchy work uncovered that also affect FLAT input.)*

Insert directly below `# Changelog` (`CHANGELOG.md:1`), above `## 1.0.1 (unreleased)`:

```markdown
## 1.1.0 (unreleased)

### Added

- **Column MultiIndex frames expand into one trace per group.** The
  innermost column level is the feature axis; every level above it groups,
  so a `(Market, Sector, Ticker)` frame draws one trajectory per sector plus
  a heavier market-mean trajectory. Widths, opacities, colours and legend
  labels follow the same documented formulas as row expansion. A two-level
  `(Group, Feature)` hierarchy has no aggregate mean, so **every leaf is
  treated as top-level and carries its own legend label** (previously such
  traces would all have been unlabelled). Each group's leaf is flattened onto
  the feature axis — its columns become the innermost level's values, keeping
  that level's name — and the frame you passed in is never modified.
  **Feature correspondence across groups is by name**: every group must
  carry the same innermost labels, later groups are permuted into the first
  group's order, and mismatches (unequal widths included) are refused by
  name. **Duplicate feature names inside a group are permitted**, matched
  across groups by `(label, occurrence)`: no column is dropped and no group
  is merged.
- **Continuous `hue=` propagates through a column hierarchy** as a per-trace
  value: a flat sequence is broadcast to every leaf, or pass one sequence
  per leaf. A mean trace takes the element-wise mean of its leaves' hue, and
  hue is truncated by the same operation that truncates the data. A forecast
  overlay takes the final observed hue colour of its source trace.
  Categorical hue still defers to the grouping.
- **`hyp.predict` accepts hierarchical frames.** Column hierarchies group by
  every level above the innermost (feature) level; row hierarchies group by
  every level above the innermost (time) level and **keep that level as each
  group's flat index**, with its name and dtype intact, so a datetime-like `t`
  works per group. `return_model=True` returns parallel forecast/model
  sequences.
- **`predict=` works with hierarchies**, forecasting every plotted
  trajectory including per-level means; a mean is forecast from its own
  averaged trajectory, not from the average of its leaves' forecasts.
  Every plotted trace needs at least 2 rows, on **either** axis, and
  `plot()` says so directly instead of failing deep inside the forecaster.
  For a **column** hierarchy that holds whenever the frame itself has at
  least 2 rows, since every group keeps all of them. For a **row**
  hierarchy every expanded leaf and every derived mean must clear it;
  because row expansion draws one trace per unique full index tuple, a frame
  whose innermost index level is unique per row yields one-row traces.
- **`return_model=True` now also returns `trace_data` and `trace_metadata`**
  describing every plotted trajectory. `trace_data` holds the final
  pre-center/pre-scale trajectories — the drawn artists are centered, scaled
  and (by default) antialiased copies of them. `xform_data` is unchanged: it
  remains the analysed pipeline output for the input datasets. The two are
  the same object only when no display-only projection occurred; a `reduce=`
  spec pinning more than three components makes them differ. **Bundled
  forecasts always correspond to `trace_data`.**
- **Full plotly parity** for all of the above.
- **New guide:** *Hierarchical DataFrames* (`docs/hierarchy.rst`), covering
  row versus column semantics, the plot/predict divergence, hue forms, mean
  construction, limitations, return shapes and fitted-model behaviour.

### Changed / validation

These turn previously-accepted input into rejected input. Each was
previously ambiguous or silently lossy.

- **Frames carrying a MultiIndex on BOTH axes are now rejected** with a clear
  error. Before 1.1 such a frame followed the row path and its column
  hierarchy was silently ignored.
- **A COLUMN-hierarchical DataFrame nested inside a list is now rejected**
  by `hyp.plot`; before 1.1 it was flattened to a single line, silently.
  `hyp.predict` rejects a hierarchical frame in a list on **either** axis,
  where it previously raised `TypeError: cannot perform __sub__ with this
  index type: MultiIndex` from deep inside pandas. **This is deliberately
  asymmetric:** a ROW-hierarchical frame inside a list passed to `hyp.plot`
  keeps its documented warn-and-flatten behaviour, unchanged in 1.1.
  Hierarchy expansion is defined for a bare frame only.
- **`predict=` with a MultiIndex frame no longer raises blanketly.** It
  previously raised `ValueError: predict= is not supported with MultiIndex
  expansion in this release` for every hierarchy; it now forecasts every
  plotted trajectory. A hierarchy whose traces are shorter than 2 rows still
  raises — on either axis, since a forecast needs at least two observations
  — but the message now names the offending trace and its row count, and
  explains the cause: the one-trace-per-index-tuple rule for a row
  hierarchy, or a single-observation input for a column one.
- **Hierarchy groups whose label is missing (NaN) are no longer dropped.**
  Grouping uses `dropna=False`, so a group with a NaN level label is kept
  and drawn rather than silently disappearing.

### Documented limitations

- Ragged groups (unequal feature counts per group) are rejected by the
  analysis pipeline, which requires equal per-dataset widths.
- Unequal-length row groups are averaged over their overlapping prefix, with
  one aggregated warning.
- Joint reduction stacks every group, so feature position *i* is treated as
  established by NAME, so groups with disjoint innermost labels are refused
  rather than silently stacked; `feature_correspondence='position'` on
  `group_columns` is the deliberate opt-in, and passing its arrays to
  `hyp.plot` gives a plain list of datasets — no means, no hierarchy
  styling, no `trace_metadata`.
- Continuous `hue=` over a **row** hierarchy is still warned-and-ignored;
  only column hierarchies honour it in 1.1.
- Duplicate innermost feature names inside one group are **kept** rather than
  rejected or de-duplicated, and matched across groups by
  `(label, occurrence)`: all such columns are plotted and forecast. Rename
  the innermost level first if you need them distinguishable in a legend.
- `predict=` needs at least 2 rows per plotted trace, on **either** axis.
  Over a **row** hierarchy this is the binding constraint: expansion draws
  one trace per unique full index tuple, so a frame whose innermost index
  level is unique per row cannot be forecast; flatten it
  (`df.reset_index(drop=True)`) or move the grouping to the column axis.
  Over a **column** hierarchy every group keeps all of the frame's rows, so
  it bites only when the frame itself has a single row — and flattening
  cannot help, so the error does not suggest it.
```

- [x] **Step 4: Run and confirm pass** — ~~**6 passed**~~ *(EXECUTED: **9 passed**.)*

- [x] **Step 5: Commit**

```bash
git add CHANGELOG.md tests/test_changelog_1_1.py
git commit -m "docs(1.1): CHANGELOG 1.1.0 section incl. validation/compatibility changes"
```

---

## Task 12: Verification — the full suite, the docs build, and the publication gates

`make html` alone is not enough (F24). This plan changes `docs/`, `CHANGELOG.md` and a committed SVG, all of which the repo's existing release/publication machinery inspects.

> **EXECUTED 2026-08-16 at `cdae7096`.** All six gates run in ONE tree, nothing
> re-run because nothing needed fixing. What this task's own text got wrong:
>
> 1. **Tasks 9 and 10 shipped with no EXECUTED note and not one ticked step**,
>    unlike Tasks 6, 7, 8 and 11, each of which wrote its corrections back.
>    Their commits (`c9b91293`, `a309f49e`, `b48c2848`, `f2a7a2b1`) do record
>    measured gates in their bodies, but nothing reached the plan, so at HEAD
>    the file still read as though plotly parity and the guide had not started
>    — the exact failure mode this plan warns about elsewhere (*"a rewrite step
>    and its split step are edited separately"*). Backfilled here from those
>    commit bodies, each number attributed to its sha: that is **commit
>    evidence, not a re-measurement**, and it is labelled as such. (Steps in
>    Tasks 1-5 are also still unticked; they are left alone — this task only
>    fixes the two tasks that recorded *nothing*.)
> 2. **Step 1's "+151" does not reconcile**, as the plan's own note warns.
>    Measured absolutely: **3465 passed** vs the **3406** baseline at
>    `5c2f29e9` = **+59** for Tasks 9-11. Against the plan's true base
>    `59405545` (3331) the plan's whole span is **+134**, not +151 — the
>    per-block counts are `def test_` counts, and several blocks shipped with
>    more tests than prescribed (Task 7: 22 → 25; Task 11: 6 → 9) while
>    parametrisation moves the collected total independently.
> 3. **Step 2's "40 passed" is a `def test_` count, not a collected count.**
>    Measured **47 passed, 2 skipped** (49 collected). The 13+8+4+4+11 = 40
>    functions are correct; parametrisation adds 9, and the 2 skips are the
>    release-gated cases in `tests/test_notebook_install_gate.py:125,144`
>    (`HYPERTOOLS_REQUIRE_RELEASE=1`), which are *supposed* to skip off a
>    master/tag build. A green run here is 47/2, not 40/0.
> 4. **Step 4's layering command has a precedence bug.** `'.plot' in src and
>    'hypertools.plot' in src or 'from ..plot' in src` binds as
>    `(A and B) or C`, so a bare `from ..plot import x` is the only two-token
>    form it reliably catches. Re-run as written (`[]`) *and* with a regex over
>    every `import`/`from` line in `hypertools/predict/` (also `[]`), so the
>    empty result is not an artifact of the operator precedence.
> 5. **Step 2a's dependency is UNMET and this is the finding, not a failure.**
>    Plan 4 has not landed: `grep -c 'MultiIndex.from_tuples'
>    docs/tutorials/market_forecast.ipynb` → **0**, and
>    `examples/animate_market_forecast.py` has **0** `MultiIndex` references.
>    Per this step's own instruction, recorded rather than tagged: **1.1 is
>    not releasable from this tree.**

- [x] **Step 1: Run the FULL suite** — ~~baseline + **151**~~ *(EXECUTED: `.venv/bin/python -m pytest -q` → **3465 passed, 13 skipped, 2 deselected in 705.10s**, zero failures, zero errors, no "warnings summary" section. That is **+59** over the 3406 baseline at `5c2f29e9`; see correction 2 for why +151 does not reconcile.)*

> If any block's contents drift during implementation, do not carry these numbers forward — recompute each one by counting `def test_` in the block, and reconcile the total before Step 6.

- [x] **Step 2: Run the publication gates explicitly** — *(EXECUTED: **47 passed, 2 skipped in 1.07s**, not the prescribed 40 — see correction 3.)*

```bash
.venv/bin/python -m pytest tests/test_release_notebook_check.py \
    tests/test_publish_gallery_notebooks.py tests/test_docs_thumbnails.py \
    tests/test_notebook_install_gate.py tests/test_colab_install_cell.py -v
```
Expected: **13 + 8 + 4 + 4 + 11 = 40 passed.** These are the gates the reviewer named; they are inside the full suite too, but running them explicitly makes a docs-side regression obvious rather than buried in 2 700 lines of output.

> Plan 4 Task 8 owns the notebook/gallery side of this (executed outputs, the five launch thumbnails, the native-ratio gate). This step covers **this plan's** docs changes only.

- [x] **Step 2a: Confirm the Plan 4 release dependency** — *(EXECUTED: **UNMET**. `docs/tutorials/market_forecast.ipynb` exists but contains **0** `MultiIndex.from_tuples`; `examples/animate_market_forecast.py` exists but contains **0** `MultiIndex` references. Plan 4 Task 2 has not landed, so the last three rows of the checklist below are RED and **this plan is not releasable from this tree** — recorded here, as this step instructs, instead of tagging.)*

**[Plan 4 — examples and tutorials](2026-07-28-hypertools-1.1-examples-and-tutorials.md) is an explicit release dependency of this plan.** This plan deliberately does *not* rewrite the market example or the market tutorial (see *Cross-plan scope*): those are the flagship demonstration of everything Tasks 5-9 add, and 1.1 must not ship the capability without them. **Neither plan is releasable alone.**

Publication-gate checklist — all must be true in the same tree before 1.1 is tagged:

| gate | owner | check | measured 2026-08-16 @ `cdae7096` |
|-|-|-|-|
| Full suite green (~~this plan's 151~~ **+59 over `5c2f29e9`**) | this plan | Step 1 | **GREEN** — 3465 passed, 13 skipped, 2 deselected, 0 failed |
| Five publication gates green | this plan | Step 2 | **GREEN** — 47 passed, 2 release-gated skips |
| 0-warning docs build, `hierarchy.rst` built and linked | this plan | Step 3 | **GREEN** — build succeeded, 0 warnings; linked from index/api/tutorials |
| `predict → plot` layering clean | this plan | Step 4 | **GREEN** — `[]` |
| Exactly one mean-construction site | this plan | Step 5 | **GREEN** — `plot/hierarchy.py:171` only |
| `docs/tutorials/market_forecast.ipynb` rewritten around the column hierarchy, executed, ≤ 120 code lines, ≥ 24% native | **Plan 4** Task 2 Step 5 | Plan 4 Task 8 | **RED** — 0 `MultiIndex.from_tuples` |
| `examples/animate_market_forecast.py` rewritten in the same commit, ≤ 115 code lines, ≥ 26% native, runs headless | **Plan 4** Task 2 Step 2 | Plan 4 Task 8 | **RED** — 0 `MultiIndex` references |
| Five launch thumbnails generated; gallery/notebook gates green | **Plan 4** Task 8 Steps 3-9 | Plan 4 Task 8 | gates green, thumbnails owned by Plan 4 |

```bash
# Plan 4's gate, run from this tree, after both plans have landed:
.venv/bin/python -m pytest tests/test_docs_thumbnails.py \
    tests/test_release_notebook_check.py -q
test -f docs/tutorials/market_forecast.ipynb && \
    grep -c 'MultiIndex.from_tuples' docs/tutorials/market_forecast.ipynb
```
Expected: green, and a non-zero count — the tutorial genuinely uses a column hierarchy. **If Plan 4 has not landed, this plan is not releasable**; record that here rather than tagging.

> **RECORDED, not tagged (2026-08-16):** the count is **0**. This plan's own
> five gates are green in this tree; Plan 4's three are not. 1.1 stays untagged
> until Plan 4 Task 2 and Task 8 land in the same tree and this step is re-run.

- [x] **Step 3: Build the docs to RTD parity** — *(EXECUTED: `MPLBACKEND=Agg .venv/bin/python -m sphinx -b html -W -E -a docs /tmp/docsbuild` → **build succeeded**, 0 warnings. `hierarchy.html` built; `grep -c 'hierarchy.html'` → **14** in `index.html`, **3** in `api.html`, **2** in `tutorials.html`. Built to `/tmp/docsbuild` rather than `docs/_build/html` so the check leaves `git status --short` empty; the builder and flags are identical.)*

```bash
cd docs && MPLBACKEND=Agg ../.venv/bin/python -m sphinx -b html -W -E -a . _build/html 2>&1 | tail -30
```
Expected: **0 warnings**. Then confirm the new page rendered and is linked:

```bash
test -f docs/_build/html/hierarchy.html && echo "guide built"
grep -c 'hierarchy.html' docs/_build/html/index.html
grep -c 'hierarchy.html' docs/_build/html/api.html
```
Expected: `guide built`, and a non-zero count from both.

- [x] **Step 4: Confirm the layering rule still holds** — *(EXECUTED: `predict -> plot imports: []`, and `[]` again from a regex over every `import`/`from` line under `hypertools/predict/` — see correction 4 on why the prescribed one-liner alone is not sufficient evidence.)*

Run the `predict -> plot` import check from Task 1 Step 5 again. Expected: `[]`.

- [x] **Step 5: Confirm no mean is built twice** — *(EXECUTED: exactly one line, `hypertools/plot/hierarchy.py:171: arrays.append(np.mean(stacked, axis=0))`.)*

```bash
.venv/bin/python -c "
import subprocess
hits = subprocess.run(['grep','-rn','np.mean(stacked','hypertools/'],
                      capture_output=True, text=True).stdout.strip()
print(hits)
assert hits.count(chr(10)) == 0, 'more than one mean-construction site'
"
```
Expected: exactly one line, in `hypertools/plot/hierarchy.py`.

- [x] **Step 6: Re-run everything after any fix** — *(EXECUTED: **nothing needed fixing**, so nothing was re-run. Steps 1-5 all ran against the same unmodified tree at `cdae7096` with `git status --short` empty throughout; the only edits in this task are to this plan file and the session note, neither of which is imported, built or asserted by any gate — verified: the two test files mentioning a plan path, `tests/test_meshutil.py:52` and `tests/plot/test_multiindex_predict.py:19`, cite it in prose comments only, and `docs/superpowers/` is not a sphinx source. Two gates beyond this task's list were also run: `tests/plot tests/core tests/predict` → **952 passed**, and ruff set-difference parity vs `59405545` → **empty in both directions, 141 keys each side**.)*

Per the repo rule (*"repeat **all** checks if any changes were made to fix any of the checks"*): if Steps 2-5 changed anything, re-run Steps 1, 2, 2a, 3, 4 and 5 in order and confirm all six are green **in the same tree**.

- [x] **Step 7: Commit**

```bash
git add -A
git commit -m "test(1.1): full-suite, publication-gate and docs verification for hierarchy support"
```

---

## Decisions (resolved)

All four open product decisions were **resolved by the maintainer at the v3 → v4 review** (#1-#4); #5 is a cross-plan check closed at the same time, and #6 is the v6 duplicate-name decision. Each entry records what was decided, what the plan implements, and — where the decision reversed v3 — what changed. **#7 is the one exception and is OPEN**: it was raised by the Task 9 review, it is a conflict between two rules the plan already states, and it is pinned by a test rather than answered here.

1. **Row-hierarchical DataFrames inside lists — RESOLVED: keep warn-and-flatten; reject COLUMN hierarchies in lists only.**
   - **Implemented.** `reject_hierarchical_in_list(x, caller, axes=)` takes `axes='columns'` from `hyp.plot` and `axes='both'` from `hyp.predict` (Task 1 Step 3; call sites in Task 5 Step 3 and Task 7 Step 3). `tests/test_multiindex.py:453` (`test_list_with_multiindex_df_warns_and_flattens`) **keeps passing unchanged**, and Task 5 Step 5 now *verifies* that rather than rewriting it.
   - **This reverses v3**, which rejected both axes and rewrote that test. Rejecting a column hierarchy in a list is purely additive (measured: it flattens to one line with no warning today); rejecting a row one would have broken a pinned, documented behaviour for no functional gain in 1.1.
   - **The asymmetry is deliberate and is stated as such** in *Global Constraints*, in Contract 8, in `reject_hierarchical_in_list`'s docstring, in `plot()`'s `x` entry (Task 5 Step 6) and in the CHANGELOG (Task 11). For `hyp.predict` there is nothing to preserve on either axis — today a row-hierarchical frame in a list raises an opaque pandas `TypeError` — so it rejects both.

2. **Continuous `hue=` over a *row* hierarchy — RESOLVED: unchanged for 1.1.**
   - **Implemented, unchanged from v3.** Task 6 scopes continuous hue to **column** hierarchies; row hierarchies keep `plot.py:2678-2684`'s warn-and-ignore, so `tests/test_multiindex.py:306` (`test_hue_plus_multiindex_warns_and_ignores_hue`) stays green and `test_row_hierarchy_hue_is_still_warned_and_ignored` guards it going forward. Genuinely additive; no code change from v3.
   - Recorded under *Documented limitations* in the CHANGELOG so users meet it as documentation rather than as a surprise.

3. **Public animation frame stepping — RESOLVED: do not add one solely for tests.**
   - **Implemented, unchanged from v3.** Every **assertion** goes through the public `on_frame`/`FrameContext` hook from animation-core Task 7 (Task 8's `test_predict_with_hierarchy_and_animation_via_on_frame`); frames are *advanced* with the same `_drive` idiom the prerequisite plan's own test module establishes, marked in-code as harness-only. No new public API in 1.1.

4. **Row-forecast time-likeness — RESOLVED: preserve, warn, reject duplicates.**
   - **Implemented, unchanged from v3.** `group_rows_for_forecast` uses `sub.droplevel(group_levels)`, so a numeric or datetime innermost level survives as each group's own index (measured: `index.name='day'`, `datetime64[us]`, `is_monotonic_increasing`/`is_unique` correctly `False` for shuffled/duplicated times). A non-monotonic innermost level **warns** per group via `predict/common.py:103-109`, prefixed with the group name (Task 7 Step 3); duplicate timestamps **raise**, because they make the horizon ill-defined. Legitimate integer-indexed panels are not rejected.

5. **Cross-plan `return_data=` — RESOLVED: already fixed.** Plan 3 no longer calls the nonexistent `return_data=True`. Verified for v4: `grep -c 'return_data'` → **0** in `2026-07-27-hypertools-1.1-forecast-animation.md` and **0** in `2026-07-28-hypertools-1.1-examples-and-tutorials.md`. Nothing outstanding.

6. **Duplicate innermost feature names inside one group — RESOLVED (v6): permit them; matched across groups by `(label, occurrence)` since v8.**
   - Flattening a leaf onto the feature axis (Contract 11) can collide two innermost labels. Measured on a group whose flat columns are `['AAPL','AAPL','NVDA']` (`is_unique == False`): `np.asarray` → `(20, 3)` (nothing dropped), `hyp.predict` → `(1, 3)`, `hyp.plot` → a `Figure`, and **2** groups form with widths `[3, 3]` — duplicates do not merge groups. Duplicates **across** groups were already harmless (`test_duplicate_tickers_in_different_sectors_are_kept_separate`).
   - **Permitting is the evidence-backed choice:** nothing downstream is name-addressed, and rejecting would break legitimate frames (two share classes of one issuer, a repeated sensor name). Since v8 the *cross-group* match is by `(label, occurrence)`, so a group with two `'temp'` columns needs a counterpart with two. Documented in `docs/hierarchy.rst` §10 and the CHANGELOG, pinned by `test_duplicate_innermost_feature_names_are_kept_positionally` and `test_duplicate_labels_match_by_occurrence` (Task 1) and `test_duplicate_innermost_names_forecast_by_occurrence` (Task 7).

7. **F14 under `animate=` — OPEN, and deliberately so: F14 is implemented for the STATIC overlay only, and the animated behaviour is pinned on both backends until the owner rules.**
   - **The plan contains two rules that cannot both hold for an animated hue forecast.** F14 (Task 6): *"a forecast overlay under a continuous hue takes the final observed hue colour of its source trace — the last RGBA of that trace's colour array"*, a **fixed** colour. *Decision R3* (forecast-animation plan, already implemented here — `plot._update_forecasts._run_colour`, `plotly_backend`'s `forecast_frame_colors`): *"the colour a live/retained forecast wears is the HEAD RUN's, which changes from frame to frame"*, a **per-frame** colour, and under a continuous hue the head run's own line artist carries the per-dataset **palette** colour (`plot.py`, "if 'color' not in mpl_kwargs" for plotly; the hidden single-colour artist `_apply_multicolor_animation` replaces, on matplotlib). Extending F14 to animation means deciding which of "the trace's final colour" and "the colour under the head *right now*" wins — a product call, not a defect.
   - **Measured, at Task 9 (`market_frame`, `hue=[linspace(0,1,60), linspace(9,10,60)]`, `palette='viridis'`, `predict='Kalman'`, `t=1`).** Static, both backends: `{0: (72,38,119), 1: (248,230,33), 2: (30,157,137)}` — the hue tails, F14 satisfied. Animated, **both backends identically**: `{0: (59,82,139), 1: (33,145,140), 2: (94,201,98)}` — `sns.color_palette('viridis', 3)`, R3 satisfied. **Task 9's parity contract is therefore met**: the divergence is between the static and animated *paths*, not between the two *backends*, and no backend degrades silently.
   - **Pinned, not left uncovered:** `test_animated_forecast_hue_colour_is_the_SAME_on_both_backends` (Task 9's module) asserts the two backends agree **and** asserts the animated colour still differs from the static one, so the day F14 is extended to animations the test fails and names this decision instead of quietly agreeing with whatever landed.

**Two blockers were also resolved at the same review**; both are implemented, not deferred. See the *Revision note (v4)* table for the reproductions.

- **B1** — `predict=` over a hierarchy is supported only when every leaf and mean has ≥ 2 rows (Contract 10, Task 8). **v5 generalised this to both axes** — the v4 form was gated on the row axis, and a `T=1` column hierarchy is measurably unforecastable too (*Revision note (v5)* **C1**).
- **B2** — `trace_data is xform_data` is conditional, not universal (Contract 5, Task 4).

---

## Self-Review

**Every one of the 24 maintainer findings, mapped to what closes it.**

| # | finding | closed by |
|-|-|-|
| 1 | `FinalTraces` duplicates existing mean construction | **Tasks 2 + 3.** `build_hierarchy_traces` is the sole owner (mean construction, truncation, the one warning); `build_hierarchy_styles` takes `traces` and is proven not to need `arrays` (`test_styles_take_metadata_not_leaf_arrays` sets `ft.arrays = None`). `build_multiindex_styles` becomes a wrapper, so `tests/test_multiindex.py`'s 29 tests stay green. `test_every_expected_mean_appears_exactly_once` and Task 12 Step 5's one-site grep are the regressions. |
| 2 | nonexistent `return_data=True` | **Task 4.** Every test uses `return_model=True, show=False`; `test_no_return_data_parameter_exists` asserts the parameter does not exist. Verified: `def plot(` at `plot.py:517`, `return_model=False` at `:579`, no `return_data` in `hypertools/`. |
| 3 | redefining `xform_data` | **Task 4.** `xform_data` keeps `plot.py:2827`'s meaning; `trace_data`/`trace_metadata` are new. The promise at `plot.py:1935-1941` is reconciled in Contract 5 and asserted three ways — `test_each_bundled_forecast_equals_hyp_predict_on_its_trace`, `test_leaf_forecasts_match_hyp_predict_on_xform_data_when_spaces_coincide` (which now *asserts* the coincidence rather than assuming it) and `test_bundled_forecasts_correspond_to_trace_data_not_xform_data`. `trace_data is xform_data` **only when no display-only projection occurred** — see v4 revision note **B2**. |
| 4 | shared code under `hypertools.plot` | **Task 1.** Grouping lives in `hypertools/core/hierarchy.py`; `hypertools/plot/hierarchy.py` holds only `FinalTraces`/styles. Task 1 Step 5 and Task 12 Step 4 assert `predict → plot` imports are empty. |
| 5 | row forecasting loses its time index | **Tasks 1 + 7.** `sub.droplevel(group_levels)` keeps the innermost level, as a FLAT single-level index (measured: `index.name='day'`, `DatetimeIndex` preserved, `nlevels == 1`) — which is also how the row axis satisfies Contract 11 (*Revision note (v6)* **D2**). All five reviewer-requested tests exist: integer horizon (`test_row_hierarchy_groups_by_every_level_above_time`), **future `Timestamp`** (`test_future_timestamp_horizon_on_a_row_hierarchy`), at-or-before truncation, unsorted times, duplicate times. |
| 6 | NA labels silently drop groups | **Tasks 1 + 5.** `dropna=False` everywhere, verified reliable on pandas 3.0.3 (measured: 2 groups → 3). Tests put NaN in **outer** and **intermediate** hierarchy **labels**, on both axes — not in data values. |
| 7 | "additive only" vs dual-axis rejection | **Global Constraints** now carries a *Compatibility changes* table; **Task 11** puts it under `### Changed / validation`, pinned by `test_changed_validation_documents_dual_axis_rejection`. |
| 8 | divergent row plot/forecast semantics | **Tasks 7 + 10.** The comparison table is in `docs/hierarchy.rst` (user-facing, pinned by `test_the_comparison_table_is_in_the_guide`), not only in docstrings. Non-monotonic innermost levels **warn** per group (free consequence of F5 — `predict/common.py:103-109` now sees each group's own index). Strictness settled in *Decisions (resolved)* #4. |
| 9 | fitted/unfitted model semantics | **Task 7.** An ownership table distinguishes name/class/dict/**unfitted** (fit independently) from **fitted** (reuse via cloned fitted instances). `test_an_unfitted_instance_is_not_mutated_across_groups` constructs a real `Kalman()` (verified importable, `is_fitted == False`) and asserts the caller's object stays unfitted; `test_a_fitted_instance_is_reused_not_refitted` and `test_a_fitted_instance_is_not_mutated_across_groups` cover the fitted path. |
| 10 | hierarchical frames inside lists | **Tasks 1, 5, 7.** `reject_hierarchical_in_list(x, caller, axes=)` in both `plot` (`axes='columns'`) and `predict` (`axes='both'`), bare-frame-only documented in `docs/hierarchy.rst` §6. Tested on both axes and both entry points. The collision with the pinned row test is **resolved** in *Decisions (resolved)* #1: rows keep warn-and-flatten, so `tests/test_multiindex.py:453` passes unchanged and Task 5 Step 5 verifies that. |
| 11 | two-level column hierarchies | **Tasks 1, 3, 5, 9.** Measured today: `labels ['_nolegend_'] * 3`. Now every leaf of a one-level hierarchy is top-level and labelled. The end-to-end test covers **trace count, colour, opacity, linewidth and legend** on matplotlib (`test_two_level_column_hierarchy_*`, 3 tests) and on plotly (`test_two_level_column_hierarchy_labels_every_trace`). |
| 12 | ambiguous hue forms | **Task 6.** Input-relative only: flat length-T (broadcast) or one sequence per leaf. Form 3 dropped and actively rejected — `test_flat_hue_of_total_drawn_length_is_rejected` passes 480 values for a 120-row frame. |
| 13 | hue/mean ordering and drift | **Tasks 2 + 6.** `build_hierarchy_traces(..., aux=)` applies the **same** `min_len` slice to data and aux in one operation, then `assert_consistent`. `test_aux_arrays_are_co_truncated_with_the_data`, `test_mean_aux_is_the_mean_of_its_members_aux`, `test_hue_and_data_are_co_truncated`. |
| 14 | continuous hue + forecasts unspecified | **Task 6** defines it (final observed hue colour of the source trace); **Task 8** tests it on matplotlib (`test_forecast_takes_the_final_observed_hue_colour`), **Task 9** on plotly (`test_plotly_forecast_takes_the_final_observed_hue_colour`). |
| 15 | Task 7 not actionable; weak plotly test | **Task 9.** Parity is required; the "or defer" branch is deleted, and the task states why no case here needs the browser boundary. The reviewer's full list is covered: exact count/order, width/opacity/legend, continuous price hue, colorbar, `predict=`, animated prediction, `return_model=True` correspondence, backend restored in a `finally`. `>= 3` is gone — every count is exact. |
| 16 | animation test uses private internals | **Task 8.** Assertions go through the public `on_frame`/`FrameContext` (animation-core Task 7). The residual `_drive` idiom, and why it exists, is reconciled explicitly in *Decisions (resolved)* #3. |
| 17 | market tutorial not updated | **Plan 4 Task 2 Step 5** (verified by reading it). Not duplicated here. |
| 18 | gallery example not rewritten | **Plan 4 Task 2 Step 2**, same task and commit as the notebook (its Contract 2). Not duplicated here. |
| 19 | Yahoo Finance promised but unused; packaging | **Plan 4 Task 2** uses the Yahoo chart endpoint with a disk cache and a synthetic fallback, and adds **no** `yfinance` dependency, so the packaging question is answered rather than deferred. |
| 20 | tutorial navigation and intro docs | **Task 10 Step 4.** `docs/tutorials.rst:148` loses *"one moving path"*; `docs/index.rst:35-36` distinguishes row and column semantics; `docs/api.rst` gets a hierarchy note in **both** Plot and Predict. Pinned by 3 tests. |
| 21 | `pipeline_order.rst` needs hierarchy placement | **Task 10 Step 5.** A **side branch**, as the reviewer preferred, in the prose block *and* in the regenerated `docs/_static/pipeline_order.svg` (via `scripts/round17_evidence/pipeline_order_diagram.py`), with the `:alt:` text updated. Pinned by `test_pipeline_order_documents_the_hierarchy_branch`. |
| 22 | API reference relies on docstrings | **Task 10.** `docs/hierarchy.rst` covers all nine subjects the reviewer listed, is registered in the `index.rst` toctree, and is linked from `api.rst`, `tutorials.rst`, the docstrings and (via Plan 4) the market tutorial. `test_the_guide_covers_every_required_section` pins the list. |
| 23 | changelog placement | **Task 11** creates `## 1.1.0 (unreleased)` above `## 1.0.1 (unreleased)` (verified: `CHANGELOG.md:3`), with Added / Changed-validation / Documented-limitations headings. 6 tests. |
| 24 | validation beyond `make html` | **Task 12** (this plan's docs) + **Plan 4 Task 8** (the notebooks and thumbnails). All five named gates are run explicitly with their exact counts (13+8+4+4+11 = 40, counted from the files), plus a 0-warning build and a rendered-link check. |

**Test defects, individually fixed.** `test_mean_trace_forecast_comes_from_the_mean_trajectory` compares the bundled forecast to `hyp.predict(mean_trajectory)` exactly. (v4 also asserted it *differs* from the average of the leaf forecasts; **v5 deletes that** — measured, it fails on correct code once the leaves co-move, see *Revision note (v5)* **C2**.) `test_return_model_bundle_has_one_entry_per_trace` became `test_return_model_bundle_has_one_model_and_forecast_per_trace`, checking forecasts, metadata and params. `test_nan_columns_do_not_silently_drop_a_group` became `test_nan_hierarchy_label_does_not_silently_drop_a_group` (a NaN **label**), with a separate `test_nan_data_values_still_plot`. `test_a_shared_unfitted_instance_is_not_mutated_across_groups` constructs a real `Kalman()`. `test_price_hue_actually_maps_low_to_high` became `test_price_hue_maps_monotonically_through_the_palette`, asserting monotone luminance rather than "two colours differ". `>= 6` became `== 8` with per-artist linewidth/alpha/label assertions (measured). `>= 3` became exact plotly counts.

**v4 test-defect fixes.** `test_mean_trace_hue_is_the_mean_of_its_leaves` no longer settles for "two colours differ": it reproduces the pinned colour chain (`mat2colors` over the concatenated aux with `n_bins=100`, then the segment-midpoint rule at `plot.py:5094`) under `antialias=False` and asserts **exact RGBA**, *and* asserts the mean-of-leaves rule directly on `trace_metadata['aux']`. `test_leaf_forecasts_match_hyp_predict_on_xform_data` became `..._when_spaces_coincide` and now **asserts** the precondition it used to assume. `test_flat_input_trace_data_is_xform_data` became `..._when_no_display_projection`, with `test_display_projection_makes_trace_data_diverge_from_xform_data` as its counterexample. `test_every_drawn_trace_gets_its_own_forecast` became `test_every_plotted_trajectory_gets_its_own_forecast`.

**v6 defect fix, and what the audit it triggered found.** `group_columns` returned `sub.T`, which keeps the caller's **full** column MultiIndex — breaking this plan's own feature-axis contract and making `hyp.predict`'s per-group recursion non-terminating (*Revision note (v6)* **D1**). The leaves are now flattened to the innermost level, copied before the columns are replaced so the caller's frame cannot be mutated, and the invariant is stated once for both axes as **Contract 11**. The maintainer's follow-up showed the row axis is the same defect class with a worse signature — `expand_multiindex`'s leaves are a measured **fixed point** — so v6 records that hazard, pins the row helper's already-flat `droplevel` behaviour, and reconciles it explicitly with the F5 datetime promise (no conflict; no `RangeIndex` fallback). **Group-label audit (requested with the fix): nothing derives a group name, legend label or `trace_metadata` entry from a leaf's columns.** Plot-side labels come from `ft.keys` ← `meta['leaf_keys']` (Tasks 2, 5), colorbar/legend text from the same keys (Task 3), and `trace_metadata['keys']` likewise. **One real defect surfaced by that audit and fixed here:** Task 7 Step 3 unpacked `groups, keys = group_columns(data)`, but `group_columns` returns `(leaves, meta)` — the per-group error/warning label would have been a *dict key* (`'n_levels'`, `'leaf_keys'`, …). It now reads `groups, _meta = group_columns(data); keys = _meta['leaf_keys']`.

Every "N passed" was obtained by counting `def test_` in the block above it: 27, **14**, 6, 6, 17, 15, 22, 17, 12, 8, 6 = **150** (Task 2's module went 13 → 14 in v7).

**Placeholders.** None. Every step carries runnable code or an exact command with its expected output.

**Task dependencies.** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12. The v3 mutual dependency between Tasks 4 and 5 is **gone**: Task 4 is flat-only and its verification step passes standalone; the hierarchical bundle assertions live in Task 5, which needs Task 4's keys but is not needed by it. Task 7 depends only on Task 1, so it can run in parallel with 2-6. Tasks 8 and 9 need animation-core Task 7 and forecast-animation Tasks 1-2. **Task 12 additionally gates on Plan 4** (Step 2a) — this plan is not releasable without it.

**Suite arithmetic.** Baseline `2564 collected (2 deselected)`, `2551 passed, 13 skipped`. This plan adds **150** tests in new modules, rewrites **1** existing test in place and adds **1** to `tests/test_multiindex.py` (29 → 30), for a delta of **151**. Expected end state, this plan alone: **2715 collected, 2702 passed, 13 skipped.** Sibling plans add their own.

**Remaining risk.** Four places:

1. **Task 6 is the largest single change** and the likeliest to disturb existing figures: the continuous-vs-categorical branch must keep categorical hue deferring to the grouping and keep row-hierarchy hue warning, or existing figures silently change colour. `test_categorical_hue_still_defers_to_the_grouping`, `test_row_hierarchy_hue_is_still_warned_and_ignored` and the full-suite run in Step 5 are the guards; if the task grows beyond one reviewable diff, split it into "classify hue early" and "propagate hue through FinalTraces".
2. **Tasks 8 and 9 both touch `plot.py:3999`, and so does forecast-animation Task 2.** Whichever lands second must extend rather than replace — Task 8 Step 3.4 says so explicitly, and the distinction (hierarchical mismatch = raise; hue/cluster regrouping = status-quo silent, per the README's open decision named **"Silent forecast drop under `hue=`/`cluster=`"**) is stated so the merge is mechanical.
3. **`tests/test_multiindex.py:479` is now the ONLY passing test this plan rewrites.** *Decisions (resolved)* #1 removed the other one (`:453` keeps warn-and-flatten), so the compatibility surface is a single line: `predict=` over a hierarchy stops raising blanketly. It is tabulated up front, changelogged, and Task 5 Step 5 actively verifies that `:453` was *not* touched.
4. **Contract 10 is a shape rule users will meet as an error, on either axis.** A row hierarchy whose innermost level is unique per row cannot be forecast, and neither can any hierarchy over a 1-row frame; both are properties of the data rather than limitations the user can configure away. The mitigations are all in the message and the docs: the error names the trace and its row count, and then explains the cause the axis actually has — the one-trace-per-index-tuple rule plus `df.reset_index(drop=True)` / a column hierarchy for rows, or a single-observation input for columns, where flattening is deliberately **not** offered because it cannot add a row. `docs/hierarchy.rst`'s comparison table and the CHANGELOG's *Documented limitations* say the same thing before the user hits it. `test_row_hierarchy_with_one_row_leaves_raises_naming_the_trace` and `test_one_row_column_hierarchy_raises_about_the_input_not_the_grouping` assert the messages rather than merely the exception type, `test_animated_one_row_hierarchy_still_raises_the_precondition` pins that the check precedes the forecast schedule, and `test_row_hierarchy_with_multi_row_leaves_forecasts_every_trace` verifies its own frame's leaf shapes before asserting counts — so no side can pass vacuously.
