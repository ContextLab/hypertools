# Maintainer review — MultiIndex plan v4 (round 4)

Received 2026-07-28. Verbatim, followed by the verification I ran. Outcome: **v5**.
Verdict: *"essentially implementation-ready, with one edge-case correction still needed."*

---

## Review (verbatim)

The v4 revision addresses both previous blockers and nearly all remaining concerns. It is now
essentially implementation-ready, with one edge-case correction still needed.

### Remaining issue

The plan incorrectly says column hierarchies always qualify for predict=:

> "Column hierarchies: always supported."

That is only true when the input DataFrame has at least two rows. A column-hierarchical frame with
len(df) == 1 produces one-row leaves and one-row means, which hyp.predict() cannot forecast for the
same reason as a one-row row hierarchy.

Task 8 currently applies its proactive length check only when:

    ft.meta['axis'] == 'rows'

and explicitly skips column hierarchies. That would allow a one-row column hierarchy to fail later
with the generic prediction error.

Recommended correction:

- Apply the len(trace) >= 2 precondition to every hierarchy.
- Customize only the remediation text by axis:
  - row hierarchy: explain full-tuple expansion and suggest flattening or moving grouping to
    columns;
  - column hierarchy: say the input itself has only one observation and requires at least two rows.
- Replace "column hierarchies always qualify" with "column hierarchies qualify whenever the input
  has at least two rows."
- Add a T=1 column-hierarchy test.
- Update Contract 10, the hierarchy guide, changelog limitation, and Plotly coverage accordingly.

### Minor test robustness concern

This assertion may be brittle:

    assert not np.allclose(bundled, avg_of_leaves, ...)

The intended contract is that the mean trajectory is forecast directly. That is already proven by:

    bundled == hyp.predict(mean_traj)

For some models or datasets, forecasting may commute with averaging, making the negative assertion
false even when the implementation is correct. I would remove it or use a deliberately nonlinear
forecaster/dataset with a separately verified non-commuting result.

### What v4 successfully resolves

one-row row-hierarchy forecasting with an explicit precondition; positive row-hierarchy forecasting
when leaves contain multiple rows; trace_data versus xform_data under display-only projection;
flat-only Task 4 followed by hierarchical Task 5; exact auxiliary hue verification; preservation of
pinned row-in-list behavior; resolved row-hue, animation, and time-index decisions; the Plan 3
return_data defect; Plan 4 as an explicit release dependency; tutorial, gallery, Yahoo Finance,
docs, pipeline diagram, changelog, Plotly, and publication-gate coverage.

I independently confirmed that return_data no longer appears in either prerequisite plan.

Once the one-row column-hierarchy case is folded into the generic trace-length precondition, I
would consider the plan ready to execute.

---

## Verification

Both findings confirmed. The second is stronger than "may be brittle" — it fails deterministically
on the plan's own flagship example.

### Finding 1 — CONFIRMED

Measured with the plan's own column-grouping idiom (`df.T.groupby(level='Sector', sort=False)`,
then transpose — pandas 3 removed `groupby(axis=1)`):

```
T=1: leaf shapes {'Tech': (1, 3), 'Fin': (1, 3)}, mean (1, 3)  -> NOT forecastable
T=2: leaf shapes {'Tech': (2, 3), 'Fin': (2, 3)}, mean (2, 3)  -> forecastable
```

and `hyp.predict(1-row frame, model='Kalman', t=1)` raises the same
`ValueError: cannot forecast from a single observation` (`predict/common.py:256`).

The three offending prose sites were at lines 133, 2056 and 2853 of v4; the gate was at 2349.
All fixed. Note the axis asymmetry the correction preserves: for a **column** hierarchy every group
has `len(df)` rows, so the rule collapses to `len(df) >= 2`; for a **row** hierarchy leaves can have
different lengths, so it must be checked per trace. The axis-independent `len(trace) >= 2` rule
handles both.

### Finding 2 — CONFIRMED, and quantified

The concern was that forecasting might commute with averaging. It does, progressively, as the
leaves co-move. Kalman, t=1, T=150, 3 leaves, scale ~100, 5 seeds per rho; rho = how strongly the
leaves share a common component:

| rho | mean max abs diff | assertion holds |
|-|-|-|
| 0.00 | 0.557 | 5/5 |
| 0.50 | 0.524 | 5/5 |
| 0.80 | 0.130 | 3/5 |
| 0.90 | 0.028 | 0/5 |
| 0.95 | 0.007 | 0/5 |
| 0.99 | 0.0003 | 0/5 |

An earlier check on *independent* random walks passed 13/13, which is exactly why this needed
measuring rather than reasoning: independence is the best case, and the assertion's margin there is
only 4-9x the tolerance. Real market sectors co-move at roughly rho 0.7-0.9, so on the plan's
flagship market example a **correct** implementation fails this assertion.

Deleted, per the recommendation. The positive assertion
`np.allclose(bundled, hyp.predict(mean_traj), rtol=1e-6, atol=1e-6)` proves the contract completely
— it pins precisely which trajectory the bundled forecast came from. The full measurement is
recorded in the surviving test's docstring so nobody re-adds the negative one.

### Cross-plan gap found while generalizing the precondition (C3)

Not raised in the review; found by checking Plan 3 once the precondition became axis-independent.

Plan 3 solves the same shape problem with the **opposite** policy:
`forecast_from_history(...)` returns `None` when `len(history) < max(2, min_history)`
(`2026-07-27-hypertools-1.1-forecast-animation.md:289`), pinned by
`test_returns_none_below_min_history` (`:153`) and a test whose docstring reads *"Frame 0 reveals 1
raw row; min_history=2 refuses to fit it"* (`:389`).

Both policies are right, for different reasons, but their interaction was unspecified in either
plan: an animated hierarchy whose traces are one row would hit Plan 3's schedule and silently draw
no forecast forever, because no frame ever reaches 2 rows.

Resolved as a two-level rule, now stated in both the plan and the README:

1. **The precondition tests full trace length** — a *permanent* property — so it runs for animated
   hierarchies too, **before** the forecast schedule is built, and raises.
2. **`min_history` tests the per-frame revealed history** — a *transient* property — and returns
   `None`, so the opening frames of a legitimate animation simply show no forecast yet.

They do not conflict: a long-trace animated hierarchy passes the precondition while `min_history`
still suppresses its opening frames. A test pins that the precondition is not bypassed when
`animate=` is set.

### Counts

v5 removes 0 tests and adds 2, both in `tests/plot/test_multiindex_predict.py` (15 -> 17).
New-module totals 22, 13, 6, 6, 17, 15, 20, 17, 12, 8, 6 = **142**, plus +1 net in
`tests/test_multiindex.py` = **143**. Sum verified independently.
