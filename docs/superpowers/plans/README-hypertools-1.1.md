# HyperTools 1.1 — plan set

Entry point for reviewing the 1.1 work. Four plans, written to be executed in dependency order.

**Why 1.1 exists.** The five newest gallery examples were measured against their own source:
**6.0% of their code is hypertools calls; 37.9% is defect** — either re-implementing something the
library already does, or hand-rolling something the library should own. The 15 older tutorials run
2.5%–39% hypertools; `analyze.ipynb` never calls `hyp.plot` at all. Rather than paper over that in
the examples, 1.1 adds the missing library features and then rewrites the examples against them.

Nothing ships until the whole line works; the Bluesky announcement waits.

---

## The plans

| # | plan | what it delivers |
|-|-|-|
| 1 | [animation core](2026-07-26-hypertools-1.1-animation-core.md) | `order='parallel'\|'serial'`, per-dataset `alpha=`, public `on_frame` hook + `FrameHooks`, per-segment titles, plotly serial parity, 3 bug fixes |
| 2 | [MultiIndex](2026-07-28-hypertools-1.1-multiindex.md) | shared grouping in `core/hierarchy.py`, one authoritative final-trace builder, column-hierarchy expansion, continuous `hue=` through a hierarchy, hierarchical `hyp.predict`, `predict=` with expansion, plotly parity, the hierarchy guide + CHANGELOG 1.1.0 |
| 3 | [forecast animation](2026-07-27-hypertools-1.1-forecast-animation.md) | precomputed forecast schedule, `predict=` with time-progressing animations, `forecast_trail=`, plotly parity |
| 4 | [examples and tutorials](2026-07-28-hypertools-1.1-examples-and-tutorials.md) | native palette-from-image, then the 5 launch examples + 15 older tutorials rewritten against the new API |

### Dependency order

```
Plan 1 (animation core)
  ├─ Task 5  order=            ─┐
  └─ Task 7  FrameHooks        ─┴─> Plan 3 (forecast animation)
                                      └─ Tasks 1-2 ─┐
Plan 2 (MultiIndex, 12 tasks) ────────────────────  ┴─> Plan 2 Tasks 8-9
                                                          │
                          Plans 1 + 2 + 3 ────────────────┴─> Plan 4
```

Plan 1 is the keystone: its `FrameHooks` registry is what lets Plans 2-4 stop monkeypatching
matplotlib's private `FuncAnimation._func`, which four of the five examples do today.

---

## How these plans were built

Every plan was written, then **adversarially reviewed against the source**, then rewritten. That
was not ceremony — each review found defects that would have failed on first execution:

| plan | review | outcome |
|-|-|-|
| 2 | maintainer | 9 findings; 3 tasks rested on false assumptions → **v2** |
| 2 | [maintainer, round 2](../../../notes/audit/review_plan2_v2_maintainer.md) | 24 findings (1-5, 17-24 blocking): duplicated mean construction, a nonexistent `return_data=` API, `xform_data` redefined, predict→plot layering, a discarded time index → **v3** |
| 2 | [maintainer, round 3](../../../notes/audit/review_plan2_v3_maintainer.md) | 2 blockers + 5 corrections, and 4 open decisions resolved: row hierarchies cannot always be forecast, `trace_data is xform_data` is not universal → **v4** |
| 2 | [maintainer, round 4](../../../notes/audit/review_plan2_v4_maintainer.md) | 1 edge case + 1 brittle assertion: the ≥2-row precondition was gated to the row axis, and a negative assertion failed on co-moving leaves → **v5** |
| 1 | maintainer, round 5 (decisions) | all 4 open decisions resolved; investigating one of them disproved the plan's own "plotly cannot do this" premise → **v3** |
| 1 | [review](../../../notes/audit/review_plan1_animation_core.md) | 9 critical findings → **v2** |
| 3 | [review](../../../notes/audit/review_plan3_forecast_animation.md) | 8 defects, 4 fatal → **v2** |

The recurring defect was writing tests that *looked* correct without tracing whether they could
actually pass. Three examples, all caught by review rather than by reading:

- A test asserting a grouping helper returns 2 groups where the source provably returns 2×T.
- A morph-title rule keyed off `current_fraction`, when holds and transitions **both** sweep 0→1 —
  so the rule blanked titles mid-hold and named them mid-transition, and its test passed anyway.
- A forecast horizon silently redefined ~15×, because animations animate the antialiased array
  (60 raw rows → 904 drawn rows), making `t=3` forecast 0.20 real samples.

Each v2 carries a "Revision note" table listing every v1 error against verified reality; Plan 2's
v3 carries the equivalent for its 24 second-round findings.

**Cross-plan defects found and fixed** (both were the same class of drift — a plan citing
something that no longer, or never did, exist):

- Plan 3 called `hyp.plot(..., return_data=True)` in Task 7. No such parameter exists (`def plot(`
  at `plot.py:517`, `return_model=False` at `:579`). Fixed: those tests now use `return_model=True`.
- Plan 1's rewrite inserted a new Task 4, and Plan 2's v3 renumbered to 12 tasks. Every sibling
  citation was re-pointed: Plan 3 → animation-core Tasks 5/7; Plan 4 → MultiIndex T1/T2/T5/T6/T8.
- Plan 3 depended on `_register_frame_callback`, which Plan 1 v2 never ships; it now uses the
  `FrameHooks` interface that does exist.

- Plan 2 cited *"README Decisions still open #5"* for the silent forecast drop, but #5 was the
  row-in-list decision — the drop was #9. Caught while renumbering this README, which would
  otherwise have made the stale citation accidentally correct. Now cited **by name**.

When any plan is renumbered, re-check every sibling citation — this bit four times. Cite
decisions by name, not by number; numbers move.

**A reviewer's claim is evidence, not proof.** The round-3 review's blocking finding was right in
principle but named the wrong frame: it read `tests/test_multiindex.py:479` as producing "six
one-row leaves and two one-row means". Measured, `_make_2level_df()` repeats each `(cond, subj)`
tuple `n_time=10` times, giving **8 leaves of shape (10, 3)** and 10 drawn traces — that frame
forecasts fine. The one-row case is the plan's own 6-row `2 cond × 3 subj` example (measured: 6
leaves of shape (1, 4)). The rule was adopted exactly as directed; only the example was corrected.

---

## Decisions still open

Each is implemented one way so the plans stay runnable end to end; switching is a documented
one-line change. **These want a maintainer call before execution.**

> These are **deliberately unnumbered**. A numbered list here went stale the moment decisions were
> resolved and items renumbered, and a sibling plan cited a number that had since moved. Cite these
> by name.

**Plans 1 and 2 have none left** — Plan 2's four were resolved in the round-3 review, Plan 1's four
in the round-5 exchange. Both sets are recorded under *Standing decisions*.

### Plan 3

- **Silent forecast drop under `hue=`/`cluster=`** (`plot.py:3999`) — keep silent, warn, or raise?
  Implemented as status quo (silent) and pinned with a test.
- **Throttling beyond memoization.** Memoization caps a 900-frame animation at ≤177 fits instead
  of 2700 (~10 s vs ~146 s), but a 500-row history is ~440 ms per fit, so long series stay slow.
  A `forecast_every=` default would be a product decision.
- **`min_history`** — a 2-row history draws a degenerate flat stub in the opening frames.
- **A finished dataset's forecast under `order='serial'`** — freeze (implemented), fade, or hide.

### Plan 4

- **Where `image_palette` is exported.** Implemented as `hypertools.plot.colors.image_palette`
  plus the declarative `palette='image:<path>'` spelling. The alternative is a top-level
  `hyp.image_palette`, which grows the curated `__all__` in a minor release.
- **The paintings outlier trim.** Dropped, because a single `hyp.plot` over raw text leaves no
  gap between reduce and plot to trim in. Restoring it needs either `vectorizer=` on `reduce()`
  or a `manip='TrimOutliers'` — and no 1.1 plan currently owns either.
- **The morph example's 5-line `normalize()` helper.** Kept, because it is genuinely not
    redundant: `plot()` applies one shared pooled affine, and `normalize='within'` z-scores per
    column, which distorts a point cloud's aspect ratio. An aspect-preserving
    `normalize='isotropic'` mode would delete it, but again no 1.1 plan owns it.

---

## Standing decisions already made

- **`t` is in raw samples**, not interpolated ones.
- **Forecast bounds:** static data (including *animated* static data) precomputes all forecasts,
  fits the reduction on **real data only** then applies it to the forecasts, and derives the box
  from data + forecasts together — so no clamping. Streaming keeps the existing head-frozen box and
  clamp (`streaming.py:382-401`). Verified: `analyze(raw, ...)` at `plot.py:2803` already fits on
  real data only, so this matches shipped behaviour.
- **Plotly and matplotlib must behave identically.** Where a capability genuinely cannot cross the
  browser boundary it raises, naming the backend; it never silently degrades. **As of the round-5
  exchange there is no such capability in 1.1** — `on_frame=` was the sole claimed exception and
  turned out not to be one (below).
- **`on_frame=` works on BOTH backends, as a purity contract rather than a timing contract.**
  The earlier claim that plotly has no Python per-frame loop was wrong: `_add_animation`
  (`plotly_backend.py:2517`) builds every frame in a Python loop, appending `go.Frame` at
  **four** sites: `:2729` (spin), `:2819` (morph), `:2865` (serial), `:2975` (the `else:`
  parallel/window branch). What plotly lacks is a loop during *playback*. **All four must be
  patched** — `:2975` serves the default `animate=True` parallel style, so patching only the first
  three would ship an `on_frame` that never fires for the most common animation. Plan 1's Task 7
  Step 6a is the authority here and applies one identical block to all four. So the hook is called once
  per frame by each backend's natural loop — matplotlib at render time (`FuncAnimation`,
  `matplotlib_backend.py:1935`, possibly re-called per frame index across loops/saves), plotly once
  per frame at build. **`on_frame` must therefore be a pure function of its `FrameContext`;**
  output parity is the tested guarantee. Verified safe: all four examples that monkeypatch
  `FuncAnimation._func` compute per-frame content purely from the frame index.
- **`morph_samples` above the tractability cap is governed by `simplify=`** (new in 1.1, default
  `True`). Below the cap it does nothing. Above it, `simplify=True` downsamples **silently, with no
  warning**; `simplify=False` raises with a message naming `simplify=True`. This narrows the
  `morph.py:17-24` no-point-dropped guarantee, which must be updated in source to match.
- **Animated continuous-hue default linewidth 1.5 → 1.0**, so hue and no-hue animations agree.
  A visible change to existing animated hue figures, changelogged as such.
- **`order='serial'` with `spin`/`window`** warns-and-ignores, matching `plot.py:3760-3781`.
- **Forecast scoring stays out of the library** — it is analysis, not plotting, and belongs in the
  tutorial as legitimately custom code.
- **Column MultiIndex rule:** innermost level is the feature axis; every level above it groups.
- **`xform_data` keeps its v1.0 meaning** (analysed pipeline output for the input datasets). The
  pre-center/pre-scale plotted trajectories — leaves plus derived per-level means — are exposed
  separately as `trace_data` / `trace_metadata`, and forecasts always correspond to `trace_data`.
  `trace_data is xform_data` **only when no display-only projection occurred**: `xform_data` is
  captured at `plot.py:2827`, *before* the display-dimensionality enforcement at `:2886-2919`
  rebinds `xform`. Measured counterexample on a **flat** input — `reduce={'model':'PCA','kwargs':
  {'n_components':5}}` gives `xform_data` shape (60, 5) while the artist is 3-D.
- **`predict=` is not defined for every hierarchy — on either axis.** A hierarchy qualifies only
  when **every** final trace (leaves *and* derived means) has ≥ 2 rows, because `hyp.predict`
  refuses a one-row trace (`predict/common.py:256`). A precondition over the final traces raises
  before any forecasting; only the remediation text is axis-specific. **Row:** `expand_multiindex`
  makes one leaf per unique *full* index tuple, so a frame whose innermost level is unique per row
  yields one-row traces — flatten, or move the grouping to the columns. **Column:** every group has
  `len(df)` rows, so the input itself has only one observation — measured, a `T=1` column hierarchy
  gives leaves `(1, 3)` and a mean `(1, 3)`.
- **Short histories are handled by two mechanisms with different policies, and that is correct.**
  The precondition above tests **full trace length** — a *permanent* property — so it runs for
  animated hierarchies too, before the forecast schedule is built, and **raises**. Plan 3's
  `min_history` tests the **per-frame revealed history** — a *transient* property — and returns
  `None`, so the opening frames of a legitimate animation simply show no forecast yet. They do not
  conflict: a long-trace animated hierarchy passes the precondition while `min_history` still
  suppresses its opening frames.
- **Hierarchical frames inside lists: rejected on the *column* axis only.** Rows keep today's
  warn-and-flatten, so `tests/test_multiindex.py:453` passes unchanged. Deliberately asymmetric.
- **Continuous `hue=` over a row hierarchy** stays today's warn-and-ignore for 1.1.
- **No public single-frame render API** is added merely to serve tests.
- **Row-forecast time-likeness:** numeric/datetime innermost levels are preserved, suspicious
  ordering warns, duplicate timestamps raise.
- **Shared hierarchy grouping lives in `hypertools/core/hierarchy.py`.** `hypertools/predict/` never
  imports from `hypertools/plot/`; only `FinalTraces`/styles are rendering code.
- **Dual-axis frames are rejected** in 1.1 — a compatibility change, changelogged under
  *Changed / validation*. (Frames nested inside lists: column axis only — see above.)
- **Market data comes from Yahoo Finance.** Verified: 24/24 tickers, 2513 trading days
  (2016-07-28 → 2026-07-28), 6 sectors × 4 tickers, equal feature widths (required by
  `plot.py:2750-2751`).

---

## Supporting evidence

| document | what it establishes |
|-|-|
| [PLAN.md](../../../notes/audit/PLAN.md) | the original audit + synthesis |
| [launch_examples_audit.md](../../../notes/audit/launch_examples_audit.md) | line-by-line A/B/C/D classification of the 5 examples |
| [other_tutorials_audit.md](../../../notes/audit/other_tutorials_audit.md) | the 15 older tutorials, ranked |
| [native_capability_map.md](../../../notes/audit/native_capability_map.md) | what the library actually supports today |
| [temperatures_dataset_findings.md](../../../notes/audit/temperatures_dataset_findings.md) | the paper dataset + MultiIndex probes |
| [hierarchical_example_ideas.md](../../../notes/audit/hierarchical_example_ideas.md) | 10 candidate hierarchical datasets, ranked |
| [climate_hierarchy_feasibility.md](../../../notes/audit/climate_hierarchy_feasibility.md) | why climate cannot show loop + drift together |

### Two findings that closed off directions

- **Climate cannot be the hierarchical example.** Smoothing tightens the seasonal loop but never
  lifts the warming drift: absolute drift stays flat (0.218–0.470) at every kernel while the loop
  diameter collapses 4.27 → 0.83. At shipped smoothing the warming is 0.070% of variance against
  the loop's 37.5%, and decade centroids are strictly monotone in **0 of 6** cities at every
  setting. A physical limit, not a code limit. Weather instead became the paper-style figure —
  essentially one native call.
- **The market became the hierarchical showcase** because sectors-within-market is a real
  two-level hierarchy over data that already carries a multi-dimensional measurement per timestep.

---

## Verified baseline

`2564 collected`, **2551 passed, 13 skipped**. Every task ends by re-running the full suite; the
pass count may only grow. Use `.venv/bin/python -m pytest` — the base anaconda python is broken
(numpy/matplotlib mismatch).
