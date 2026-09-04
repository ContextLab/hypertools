# Maintainer review — MultiIndex plan v3 (round 3)

Received 2026-07-28. Verbatim, followed by the verification I ran against each blocking claim.
Outcome: **v4**. Two blockers fixed, five corrections applied, four open decisions resolved.

---

## Review (verbatim)

I've reviewed the plan again. The v3 revision is dramatically stronger and addresses almost every
prior finding with concrete contracts, tests, and documentation work. I'd call it close—but two
technical blockers remain.

### Blocking issues

**1. Row-MultiIndex predict= still cannot work under preserved plotting semantics**

The plan preserves row plotting as one leaf per full index tuple. Its example (cond, subj) frame
therefore produces six one-row leaves and two one-row means.

Task 8 then rewrites the existing refusal test to expect forecasts for all eight traces. But
hyp.predict() rejects a one-row trajectory:

    ValueError: cannot forecast from a single observation

I verified this directly in the repository venv.

Therefore these contracts cannot all hold simultaneously:

- preserve row plotting expansion;
- forecast every drawn row-hierarchy trace;
- require at least two observations for forecasting.

The cleanest 1.1 decision is:

- support plot(..., predict=...) for column hierarchies;
- retain a clear error for row hierarchies whose final traces have fewer than two observations;
- possibly support row hierarchies only when every expanded leaf and derived mean has at least two
  rows.

Task 8 must not promise eight forecasts for the existing (cond, subj) regression frame.

**2. trace_data is xform_data is not universally true for flat inputs**

xform_data is captured before the display-dimensionality enforcement:

- capture: hypertools/plot/plot.py:2827
- possible later display projection: hypertools/plot/plot.py:2887

A reducer configured to return more than three dimensions can produce:

    xform_data: analyzed 5-D output
    trace_data: final 3-D plotted-space trajectories

Forecast overlays are currently computed from the latter. Therefore v3's statements that
`trace_data is xform_data` for every flat input, and that forecasts always match
hyp.predict(xform_data), are not universally correct.

Revise the contract to:

- xform_data: canonical analyzed pipeline output, unchanged;
- trace_data: final pre-center/pre-scale plotted trajectories;
- trace_data is xform_data only when no display-only projection occurred;
- forecasts always correspond to trace_data;
- forecasts correspond to xform_data only when the two spaces coincide.

Add a regression test using a reducer/spec configured above three components so the display
projection actually runs.

### Decisions that should be resolved before execution

The plan is runnable because it marks defaults, but four decisions are still product decisions:

- rejecting row-hierarchical DataFrames inside lists;
- whether continuous hue should remain ignored for row hierarchies;
- whether to add public animation frame stepping;
- how strictly row prediction validates that the innermost level is time-like.

The first two materially affect public behavior. I would resolve them before implementation rather
than letting "implemented" defaults become accidental decisions.

My preferences:

- Keep the existing row-in-list warning for 1.1; reject column hierarchies in lists. It is
  asymmetric but avoids breaking pinned row behavior.
- Keep row-hierarchy hue unchanged for 1.1.
- Do not add public frame stepping solely for tests.
- Preserve numeric/datetime innermost levels, warn for suspicious ordering, and reject duplicates.

### Smaller corrections

- Task 4 and Task 5 are mutually dependent. Combine them into one atomic task/commit or split Task
  4 into a flat bundle-extension task whose tests genuinely pass before Task 5.
- test_mean_trace_hue_is_the_mean_of_its_leaves still only proves that the mean trace has varying
  colors. It does not prove those colors came from the element-wise mean hue. Expose normalized
  auxiliary data in trace metadata for testing, or compare exact expected colormap RGBA values.
- "Every drawn trace" should be described as "every pre-center/pre-scale plotted trajectory"; the
  actual artists are later centered, scaled, and possibly antialiased.
- Fix the known return_data=True defect in Plan 3 before treating it as an executable prerequisite.
- Make Plan 4 an explicit release dependency in the final verification gate, since this plan relies
  on it for the flagship tutorial and gallery deliverables.

### Overall assessment

v3 successfully fixes: duplicate mean construction; package-layering violations; nonexistent
return_data; NA-label dropping; datetime-index loss; model ownership ambiguity; two-level legend
behavior; Plotly parity ambiguity; missing hierarchy guide and pipeline documentation; changelog
placement; tutorial/gallery scope through Plan 4.

Once the row-hierarchy forecasting contradiction and the trace_data/display-projection contract are
corrected, the plan will be implementation-ready.

---

## Verification

Both blockers reproduce. One supporting detail does not.

### Blocker 1 — principle CONFIRMED, example CORRECTED

The rule is right. `expand_multiindex` makes one leaf per unique **full** row-index tuple, so a
frame whose innermost level is unique per row yields one-row traces, and those cannot be forecast:

```
2 cond x 3 subj, 6 rows, 4 cols  ->  6 leaves, every one shape (1, 4)
hyp.predict(1-row frame)         ->  ValueError: cannot forecast from a single observation:
                                     the dataset has only 1 row.       [predict/common.py:256]
```

But the named regression frame is **not** that frame. `tests/test_multiindex.py:479` calls
`_make_2level_df()` (`:45-61`), which repeats each `(cond, subj)` tuple `n_time=10` times:

```
_make_2level_df()  ->  80 rows, 8 unique index tuples
                       8 leaves, every one shape (10, 3)   -> min 10 rows -> FORECASTABLE
                       10 drawn traces (8 leaves + 2 condition means)
```

So that frame forecasts fine and stays a permissive compatibility change; it became v4's *positive*
row-hierarchy test. The raising test was added separately on the 6-row frame. v3's actual defect
there was a smaller one the review did not name: it promised **8** forecasts where the frame draws
**10** traces.

The rule was adopted exactly as directed, in its third (most permissive) form: column hierarchies
always qualify; row hierarchies qualify only when every leaf and derived mean has >= 2 rows,
enforced as a precondition over the final traces so the error names the offending trace instead of
bubbling up a `predict` internal.

### Blocker 2 — CONFIRMED

```
hyp.plot(X_60x12, reduce={'model':'PCA','args':[],'kwargs':{'n_components':5}},
         show=False, return_model=True)
  xform_data shape  (60, 5)
  drawn artist      3-D, 945 points   (945 not 60: antialiasing)
```

Mechanism: `xform_data = copy.copy(xform)` (`plot.py:2827`) is a *shallow* list copy, so it shares
array objects with `xform`. The display block then **rebinds** `xform = reducer(...)`
(`:2886-2919`) to a new list — rebinding does not touch the alias. `trace_data is xform_data`
therefore holds by accident on every path that skips the projection, which is nearly all of them.

Narrowing worth recording: the divergence needs an explicit `reduce=` spec above three components.
`plot.py:2887` **raises** when `reduce is None`, so no default-path call can reach it. That is why
it survived the suite.

### Corrections adopted

All five. Task 4 is now flat-only and passes standalone (`4 -> 5`, not `4 <-> 5`); the hue test
asserts exact colormap RGBA plus `trace_metadata['aux']`; "every drawn trace" became "every
pre-center/pre-scale plotted trajectory"; Plan 4 is an explicit release dependency in Task 12.
Plan 3's `return_data=` was already fixed at the end of the previous session (verified: 0
occurrences) — it was a clean prerequisite when this review arrived.

### Decisions

All four resolved as preferred and moved to *Standing decisions* in the README. Only the first
changes implementation: v3 had rejected hierarchies on **both** axes, which required rewriting the
pinned `tests/test_multiindex.py:453`. v4 rejects the **column** axis only, so that test passes
untouched. `reject_hierarchical_in_list` gains an `axes=` argument (`hyp.plot` -> columns;
`hyp.predict` -> both). The asymmetry is stated in both the plan and the changelog.

Task numbering was deliberately held at 12 throughout, since renumbering has broken sibling
citations repeatedly. See [[the README]](../../docs/superpowers/plans/README-hypertools-1.1.md)
*Cross-plan defects*.
