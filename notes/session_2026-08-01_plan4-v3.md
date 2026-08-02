# Session 2026-08-01 (part 3) — Plan 3 closure audit + Plan 4 v3

Branch `dev-1.0`, base commit `065c841e`, clean worktree. **Nothing pushed — 50 commits ahead of
`origin/dev-1.0`.**

Maintainer's recommended sequence (their words, abbreviated):
1. Run and record the Plan 3 closure audit. 2. Mark Plan 3 implementable if the extracted corrected
blocks pass. 3. Create Plan 4 v3 before touching implementation. 4. Fix the metric and notebook
execution contracts first. 5. Retarget image palette through both resolver paths, dynamic category
count. 6. Rebase Task 6 on the already-landed morph script. 7. Replace network-running tests with
fixture-driven artifact construction tests. 8. Sweep citations, recalculate every count and budget.
9. Adversarially execute Plan 4 v3's prescribed code in a disposable worktree. 10. Implement Plan 3
before Plan 4.

## Findings established BEFORE the plan edits (all measured, not assumed)

### 1. The staleness is FOUR tasks, not one — bigger than the review found

The review flagged "Task 6 is stale because half already landed". Measured: commit `d730a085`
(2026-08-01 09:46, "docs(1.1): document order=, per-dataset alpha=, on_frame, per-segment titles;
simplify examples") rewrote **four** of the five example scripts:

| task | script | touched by d730a085 | lines changed |
|-|-|-|-|
| 2 | `animate_market_forecast.py` | YES | 39 |
| 3 | `animate_weather_decades.py` | YES | 28 |
| 4 | `animate_painting_embeddings.py` | **NO** (last: `4d1d2223`) | — |
| 5 | `animate_conversation.py` | YES | 49 |
| 6 | `animate_morph_zoo.py` | YES | 54 |

So Tasks 2, 3, 5 and 6 all need rebasing onto landed code, and only Task 4 is a clean rewrite.
Following the plan verbatim would overwrite newer code with the plan's older prescribed text.

`d730a085` also touched `hypertools/plot/animation_context.py` — Plan 3 Task 0's target file.

### 2. A reviewer Med finding is MOSTLY WRONG — Task 7's assertions have teeth

The review listed as cleanup: *"Already-satisfied older-tutorial assertions that cannot detect
regressions introduced by this plan."* Measured against the real committed notebooks today:

| notebook | `'SentenceTransformer' not in text` | `'ffmpeg' not in text` |
|-|-|-|
| conversation_trajectories | **FAILS** | **FAILS** |
| hugging_face_embeddings | **FAILS** | **FAILS** |
| wikipedia_embeddings | **FAILS** | **FAILS** |
| modern_sklearn_dynamics | passes | **FAILS** |
| stock_forecasting | passes | passes |
| projectile_kalman | passes | passes |

and both single-notebook assertions are red too: `analyze.ipynb` does **not** contain `hyp.plot`
(False), `reduce.ipynb` does **not** contain `hyp.describe` (False).

So 4 of 6 parametrized notebooks fail at least one assertion, and both scalar assertions fail.
These tests are RED today and Task 7 is what turns them green — correct TDD structure, not dead
weight. The reviewer's finding survives only as a **Low**: `stock_forecasting` and
`projectile_kalman` are already-satisfied non-red cases and should be *labelled as controls*
(the same treatment the 5 non-red tests in `tests/test_backend_window_parity.py` got), not
redesigned.

Lesson repeated from the azimuth incident: verify the reviewer, not just the plan.

### 3. Confirmed dead assertion (reviewer right)

`assert 'morph' in str(ns.get('ANIMATE', 'morph'))` — `ANIMATE` does **not** exist in
`examples/animate_morph_zoo.py` (grep: symbol absent; top-level names are `SHAPES`, `TITLES`, `N`,
`CUBE_SCALE`, `rng`, `clouds`, `titles`, `rotations`). With the name absent the `.get` default
makes the expression `'morph' in 'morph'` → True. Proven by direct execution: `True`. The
assertion cannot fail under any circumstances.

### 4. Global Constraints baseline is stale (measured)

- Plan says **"Verified baseline: `2564 collected`"**. Actual today: **`2782/2784 tests collected
  (2 deselected)`**.
- Plan says *"The working tree is not clean, and that is expected"* + *"as of 2026-07-28 15:48
  Plans 1-3 are being implemented concurrently"* + *"the five launch examples and their notebooks
  are untouched"*. All three are false now: tree is clean at `065c841e`, and four of the five
  examples were rewritten (finding 1).

### 5. Plan contradicts itself on the notebook gate (reviewer right)

The gate code at the end of the plan carries a comment saying *"'every code cell' is not a
reachable target"*, while the revision-note table (line 23) still promises *"Exact: every code cell
must carry outputs"* and line 36 repeats *"Task 8's gate is now exact (every code cell, no
committed tracebacks)"*. Prose and code disagree inside one document.

## Status

Seven parallel audits dispatched (reports land in `notes/audit/`):
`plan3_closure_audit.md`, `plan4_metric_remeasure.md`, `plan4_landed_state.md`,
`plan4_image_palette.md`, `plan4_notebook_gate.md`, `plan4_network_decoupling.md`,
`plan4_citations_and_ci.md`.
</content>
</invoke>
