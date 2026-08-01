# Session 2026-08-01 — Plan 1 (animation core) executed end to end

Branch `dev-1.0`. Nothing pushed. Started from `0f999c77` (round-13 fixes), which is the last commit
of `notes/session_2026-07-31_plan1-v4.4-review-round13.md`.

**All 9 tasks complete, every one reviewed, plus a whole-branch review and its fixes.**
Suite **2554 → 2740**. Sphinx `-W` clean, 0 warnings.

## Task-by-task

| task | commit(s) | suite | verdict |
|-|-|-|-|
| 1 reject non-string `title=` | `7c859581` | 2563 ✓ | SPEC PASS / APPROVED |
| 2 `linewidth=` in animated hue | `84abc289` | 2568 ✓ | SPEC PASS / APPROVED |
| 3 `simplify=` morph tractability | `01ecb94c` | 2579 ✓ | SPEC PASS / APPROVED |
| 4 plotly serial × trail parity | `ad2fbeee` | 2589 ✓ | SPEC PASS / APPROVED, bonus fix NECESSARY |
| 5 `order=` orthogonal to `animate=` | `db02c64e` | 2609 ✓ | SPEC PASS / APPROVED, deviation JUSTIFIED |
| 6 per-dataset `alpha=` | `76c4f27f` `8d089c23` `dc61f412` | 2631 | APPROVED **after 2 fix passes** |
| 7 public `on_frame` + `FrameContext` | `1dd5a6f0` `da4b2033` | 2680 ✓ | SPEC PASS, 3 contracts VERIFIED |
| 8 per-segment serial titles | `1e75165e` `ccbb28c3` | 2698 | APPROVED after 1 fix pass |
| 9 CHANGELOG + animation guide | `d730a085` | 2717 ✓ | done; surfaced the 3-D title defect |
| — 3-D animated titles render | `21f46d3c` | **2728** | fix for Task 9's finding |

Plus `a03bad01` / `60cdf9b5` (the `patch_lines` label crash) and the count re-derivations.

## The headline: paper review and execution find different defects

Plan 1 had **thirteen** maintainer review rounds before a line was written. Executing it still found
that **the plan's own verbatim code was defective in four of the nine tasks**:

- **Task 5** — `order = _resolve_order(animate, order)` rebinds the public parameter, which the
  streaming diff loop reads as user-set on *every* call → a false *"order has no streaming
  implementation"* warning on every streaming call. Also, its three independent `if`s would have
  **silently clobbered the morph branches**; needed `if/elif/else`.
- **Task 6** — the prescribed write-site validated `alpha` against the FINAL post-reshape dataset
  count, which raises under hue-driven run segmentation.
- **Task 1** — step ORDER was wrong: "run the full suite, then `git add`" fails the sdist gate on the
  new test file. Cost a 9-minute run before it was fixed in Global Constraints.
- **Task 2** — every `plot.py` line citation in the plan is pre-Task-1 and drifts as each task edits
  that one file. Now stated as a constraint: locate by symbol, cite by symbol.

Reading a plan carefully thirteen times did not surface any of these. Running it surfaced all four.

## Review caught a regression that would have shipped

Task 6's `alpha=` validated eagerly, before the code knew an internal branch would override it. Two
calls that **succeeded silently in 1.0** began raising:

    hyp.plot(mi_df, '-', alpha=[0.1, 0.2, 0.3])   # 3 values, 6 leaves
    hyp.plot(mi_df, '-', alpha=['a', 'b', 'c'])

That breaks *"no existing call may change meaning"*. The reviewer proved it with an old-vs-new
worktree diff rather than asserting it.

**Then the first fix was incomplete and the re-review caught that too** — it closed the two cited
calls but not the class: the lookahead snapshotted `hue` at the early write site while
`animate='morph'` nulls `hue` later. Fix 2 moved the lookahead after `hue` is finalised, and the
second re-review traced **all 8 values the lookahead reads** to confirm none can still go stale.
**Re-reviewing a fix is not optional** — this is the evidence.

## Three defects found by refusing "out of scope" / "cosmetic"

1. **Task 4's implementer** deferred a precog parity gap as "out of scope". Diagnosis found it was
   one of **three** confirmed parity defects sharing **one** architectural cause (below).
2. **Task 8's implementer** called a plotly margin change "cosmetic only, no test affected". The
   reviewer rendered real PNGs through kaleido: the hold-frame title was **visibly clipped at the
   canvas edge**, in 100% of the feature's plotly uses. Fixed in `ccbb28c3`.
3. **Task 9's implementer** flagged 3-D animated titles as "out of this task's file scope". Same
   class as (2), other backend: `ax.set_position([0, 0, 1, 1])` left no margin, so per-segment titles
   were computed correctly and never rendered. Fixed in `21f46d3c`.

"No test affected" is a statement about coverage, not about correctness.

## A test that could not fail on the bug it documented

The `patch_lines` crash fix shipped a test asserting `pytest.approx(30, abs=3)` / `approx(60, abs=3)`
for label frame placement. The **pre-fix values were 29 and 59** — inside the tolerance — so it
passed with and without the fix. Tightened to exact equality (`60cdf9b5`). Four sibling tests did
fail pre-fix, so the bug was covered; but a test that documents a defect it cannot catch is worse
than no test, because it reads as coverage.

## Suite arithmetic held throughout, because it was maintained

Every task landed on its predicted total exactly. That only worked because the predictions were
re-derived **three times** when fix passes added tests (+5, +7, +5, +11 at various points) and once
at the start when round 13's own tests moved the baseline 2551 → 2554. Left alone, the plan's drift
detection would have fired falsely at every checkpoint — and a false drift everywhere is
indistinguishable from a real one anywhere. Each re-derivation recomputed the chain from the
per-task additions rather than hand-editing numbers, with every substitution refusing to apply
unless it matched exactly once.

## STILL OPEN — needs Jeremy

### 1. Three plotly/matplotlib window-parity defects (a RULE CONFLICT)

All confirmed, all **shipped in 1.0**, all one architectural gap: plotly computes ONE `end`/`window`
per frame from `max_len` and clamps everyone into it, instead of per-dataset like
`trails.anim_window_bounds` (then `matplotlib_backend.trails.anim_window_bounds`). Full diagnosis:
`.superpowers/sdd/2026-07-26-hypertools-1.1-animation-core/plotly-window-parity-diagnosis.md`.

| | defect | measured |
|-|-|-|
| A | `start = max(0, end - window)` omits mpl's `-1` | head counts differ by 1pt every steady-state frame, **with no trail flags at all** |
| B | one shared `window`; mpl rescales per-dataset | 5-row marker-only + 15-row line → **0 points for 9 of 15 frames (60% of its own animation)** vs mpl's live 2-point window |
| C | plotly `end = max(2, …)` vs mpl `max(1, …)` | frame 0 only: mpl 12pts, plotly 11 |

One patch fixes all three (verified empirically across four sweeps). **Blast radius: 2 tests of
2602**, both from A, neither pinning a documented contract.

**The conflict:** the plan says *"Additive only: existing `animate=` behavior must not change"*; the
standing contract says *"Plotly and matplotlib must behave identically."* Parity is reachable only by
changing shipped plotly output. Diagnosis recommends fixing all three and landing it as **Fixed**,
not **Changed**. **Not resolved unilaterally.**

### 2. Deferred Minor
`plot()` gained `simplify=` / `order=` / `alpha=` / `on_frame=` as positional-or-keyword parameters in
a 50+ parameter signature. Pre-existing property of every prior addition; no call site goes
positional that deep. Making `plot()` keyword-only past some point is a BREAKING change — **2.0
material, not 1.1**.

## Also still open from earlier rounds

- **Plans 3 and 4 carry unaddressed FATAL findings** (`notes/audit/review_plan3_v2_recheck.md`,
  `notes/audit/review_plan4_examples_and_tutorials.md`). Neither is implementation-ready.
- **Restart for OMC 4.15.7** — the loaded 4.2.15 hook emitted false "Command failed" notices all
  session, every one on a call that succeeded.
- 44 pre-existing pyflakes findings across `tests/`; no linter in CI.

## Final whole-branch review — and why it was worth running

The nine per-task reviews all passed. The whole-branch review over all 21 commits then returned
**CHANGES NEEDED** with 3 Important findings, every one a **cross-task interaction** that no
single-task review could have seen:

1. **Partial-tag morph mislabelled clouds.** Task 8's per-segment `title=` and Task 7's
   `current_index` indexed a morph by SEGMENT POSITION (`seg_idx // 2`) and never consulted
   `morph_tags` — while Task 3's simplify guard *did*. Of three new morph-tag consumers, **one got it
   right and two drifted**. Repro: `plot([a,b,c], animate=[None,'morph','morph'], title=['a','b','c'])`
   morphs datasets 1→2 but titles the holds `'a','b'`, leaving `'c'` unreachable. Both backends.
2. **`FrameContext.datasets` diverged by backend** — plotly recorded the raw input, matplotlib the
   morph-sampled arrays, falsifying the field's own docstring *and* `animation.rst`.
3. **Plotly trail traces dropped `alpha=`** — hardcoded `0.3` where matplotlib uses `alpha * 0.3`.
   Only reachable because Task 6 added the per-dataset list form.

**A second toothless test.** Finding 2's existing parity test compared dataset shapes but pinned
`morph_samples=50` on 20-row data, so it could never fail on the divergence it nominally covered.
Repaired, and the repair was **confirmed red at the pre-fix commit**.

All six findings fixed in `f6084c7d` (+12 tests). Re-review verdict: **READY TO MERGE**, with
Important 1's class confirmed closed by enumerating all five segment→dataset consumers repo-wide and
running four partial-tag shapes the original repro never covered — **3 of 5 failed at the pre-fix
commit**, so the fix is real rather than coincidental.

## Final state

- **Suite: `2740 passed, 13 skipped, 2 deselected`** (from 2554 at session start).
- **Sphinx `-W -E -a`: exit 0, 0 warnings.**
- Packaging + release + notebook gates: 21 passed, 6 skipped.
- Live-source gate: 41 passed in BOTH default and strict modes.
- `git diff --check` clean.
- 23 commits ahead of `origin/dev-1.0`. **Nothing pushed.**

The pattern worth carrying forward: **per-task review and whole-branch review catch different
defect classes, and neither substitutes for the other.** Nine clean per-task reviews still left three
Important cross-task defects. Equally, a fix is not done when it is written — Task 6's first fix and
this branch's findings both needed a re-review to confirm the CLASS was closed, not just the cited
case.
