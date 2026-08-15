# Session 2026-08-15 — implementing Plan 2 (MultiIndex)

Executing `docs/superpowers/plans/2026-07-28-hypertools-1.1-multiindex.md`
(v7) task by task, inline (no subagents — session constraint). Branch
`dev-1.0`, base `59405545`. Same TDD cycle per task as the regrouped-reveal
run: write the plan's tests, confirm RED for the stated reason, implement,
confirm GREEN, regression sweep, commit.

## Why this plan, and why now

The maintainer's standing recommendation was "freeze Plan 4 and begin
implementing the regrouped-reveal plan; once its focused tests and
tests/plot pass, proceed to Plan 4 Task 2." The regrouped-reveal plan is
**done** (see `session_2026-08-11_regrouped-reveal-implementation.md`) and
its gate is green, but **Plan 4 Task 2 is still blocked** — measured against
the branch:

| Task 2 dependency | state |
|-|-|
| Regrouped-reveal T1/T2/T3/T5/T6/T7 | landed, green |
| Animation-core T1 (`title=`) | landed (`_validate_title`, plot.py:584) |
| Forecast-animation T3/T4/T5 | landed (`forecast_trail` in plot/forecast/plotly) |
| **MultiIndex T1, T2, T5, T6, T8** | **not started** |

Evidence for the last row: `hypertools/core/hierarchy.py` did not exist and
`group_columns` appeared nowhere; `expand_multiindex` raises `"requires a
row MultiIndex"` (no column-MultiIndex support at all, which is what the
Market example passes); and `plot.py:3952-3953` states in-code that under a
MultiIndex `hue=` is "already squelched (with a warning) above" — exactly
what T6 exists to fix. Plan 4's own dependency table predicts this:
"Without MultiIndex T6 the hue is discarded."

Jeremy chose "Execute Plan 2 (MultiIndex)" when asked.

## Where this stands

| Task | State | Commit |
|-|-|-|
| 1 `hypertools/core/hierarchy.py` grouping | done, 27 tests | `420ef60c` |
| 2 `build_hierarchy_traces` (one trace owner) | done, 14 tests | `1f6e4d6d` |
| 3 style-only `build_hierarchy_styles` | done, 6 tests | `7a415d28` |
| 4 `trace_data`/`trace_metadata` bundle keys | done, 6 tests | `86db0842` |
| (fix) NA hierarchy labels — review finding 1 | done, 14 tests, mutation-verified | `af77f09e` |
| 5 column MultiIndex end-to-end in `plot()` | done, 17 tests | `5b21e3c6` |
| 6 continuous hue as per-trace aux | not started | — |
| 7 hierarchical `hyp.predict` | not started | — |
| 8 `predict=` over final traces | not started | — |
| 9 matplotlib/plotly parity | not started | — |
| 10 `docs/hierarchy.rst` guide | not started | — |
| 11 CHANGELOG 1.1.0 section | not started | — |
| 12 verification | not started | — |

## Defects found in the plan's own listings (fixed in the code)

1. **Task 1's implementation imports `numpy as np` and never uses it**
   (ruff F401). Dropped — new files stay lint-clean.
2. **Task 2's test module imports `FinalTraces` and never uses it** (F401).
   Rather than delete the import, added `assert isinstance(ft, FinalTraces)`
   to the first test — that pins the return type the Interfaces block
   promises ("Produces `build_hierarchy_traces(...)` → `FinalTraces`"), so
   the import earns its place. A strengthening, not a weakening.
3. **Task 2 Step 4's wrapper calls `build_hierarchy_styles`, which the plan
   does not create until Task 3.** Taken literally, Task 2 Step 5's
   "29 passed in tests/test_multiindex.py, unchanged" is unreachable — the
   wrapper would `ImportError`. Resolved by porting the styling loop into
   `hierarchy.py` during Task 2 with **today's** label rule
   (`level_idx == 0 and is_mean`), leaving Task 3 to add the `n_levels == 1`
   rule (`or L == 1`) with its own RED/GREEN cycle. Neither task's tests
   were changed to accommodate this.
4. **Removing the loop from `multiindex.py` orphaned three imports**
   (`warnings`, `numpy`, `get_palette_colors`) — F401 ×3. Removed; the
   module now imports nothing.

## Review round 1 (maintainer, after Task 4) — findings and disposition

**Finding 1, HIGH: missing hierarchy labels produced duplicate means and
styles. CONFIRMED, fixed, mutation-verified.** Reproduced end-to-end, not
just from hand-built metadata:

| spelling in a MultiIndex level | stored as | same object across group keys? | groups built |
|-|-|-|-|
| `np.nan` | `float('nan')` | **no** | 2 duplicate `(nan,)` means |
| `None` | `float('nan')` | **no** | same |
| `pd.NA` | `float('nan')` | **no** | same |
| `pd.NaT` | `NaTType` | yes (singleton) | correct, by luck |

`groupby` mints a separate nan object per group key, so `NaN != NaN` split
one group into one group per leaf: `unique_top` came back `[nan, nan]`,
`n_top == 2`, and the legend carried two `'nan'` entries. The `dropna=False`
promise in `core/hierarchy.py` was kept at the grouping layer and then
broken downstream.

The **row** axis was correct at HEAD, but only by accident:
`expand_multiindex` reads keys off `df.index`, which returns the same nan
object each time. The fix is axis-independent so that stops being luck.

Fix: one internal NA-aware `_canonical_label`/`_canonical_key` in
`core/hierarchy.py` (so `predict` can reach it later without importing
`plot`), used for prefix grouping, top-level uniqueness and top-level style
lookup. Original values are preserved in `FinalTraces.keys`, `unique_top`
and legend labels — the sentinel is never user-visible.

Mutation-verified both halves: reverting prefix canonicalisation → 9 failed;
reverting top-level canonicalisation → 8 failed, including a
`KeyError: <object object>` that proves index construction and lookup are
coupled and cannot be canonicalised independently. Restored → 14 passed.

**Finding 2, HIGH while HEAD is consumable: Task 4's docstring documents a
hierarchical bundle Task 5 has not implemented. ACCEPTED.** Resolution per
the reviewer's first option: complete Task 5 immediately rather than
narrowing the docstring.

**Finding 3, MEDIUM: `README-hypertools-1.1.md` materially stale.
CONFIRMED, fixed.** It still called continuous hue the refusal example, said
animated regrouped forecasts were refused, and described the regrouped-reveal
plan as "not yet implemented" — all three false since `59405545`. Added an
execution-status table and corrected the refusal text to marker-only
categorical regrouping.

**Finding 4, MEDIUM: this note was untracked and behind HEAD. CONFIRMED,
fixed** — refreshed and committed.

**Testing assessment.** The reviewer could not certify the 3281-passed
full-suite figure from their environment: their run stopped at 994 passed on
a Kaleido/Chrome-for-Testing startup failure in a pre-existing real-PNG
serial-title test, plus the known local joblib physical-core warning. Noted
as an environment difference, not a disputed result — the figure here is
reproducible in this venv and is re-measured after every task.

## Task 5's unanticipated collision (resolved, worth remembering)

`format_data` matches DataFrame features **by column name** across datasets
(GH #132). Column-hierarchy leaves have different names per group by
construction, so the market frame was rejected outright: *"dataset 1 is
missing ['AAPL','MSFT','NVDA'] and has unexpected ['BAC','GS','JPM']"*.

The plan had already settled the question in its *Documented modelling
assumption* — correspondence across groups is **positional**, because the
joint reduction stacks every group and treats position *i* as commensurable.
So `plot()` passes `leaf.to_numpy()` into the pipeline. `group_columns`'
flattened, NAMED leaves are unchanged, so Task 7's predict path still gets
them; and ragged groups reach the existing equal-width check
(`plot.py:3650`) instead of a name-mismatch error.

## Ruff counting: use ruff's own summary, not a grep

Earlier parity checks used `grep -c '^[A-Z][0-9]'`, which UNDERCOUNTS. The
reliable figure is ruff's own `Found N errors.` line, and the reliable
comparison is a **set difference** against a base worktree keyed on
`(file, code, message)` with line numbers stripped. That method caught a real
slip in Task 5: an added `np.asarray` registered as a NEW `F405`, because
`plot.py` takes `np` from a star import and every `np.` use is already
flagged (~190 of them). `leaf.to_numpy()` avoids the reference. Parity is
**353 at base `59405545`, 353 now**.

## Task 6 design notes (NEXT — the piece Plan 4 Task 2 needs most)

**The structural blocker.** `plot()`'s hue handling is an `elif` arm of the
same chain the hierarchy branch wins (`if _multiindex_meta is not None:` at
`plot.py:4081`, hue at `:4536`), so a hierarchy never reaches the continuous
colour path at all. Hue must be classified BEFORE the chain and carried
through `FinalTraces.aux`.

Key call shapes, measured:

- `multicolor_hue` is a FLAT array of length `sum(len(trace))`, split by
  `pre_interp_lengths` — so for a hierarchy it is
  `np.concatenate(ft.aux)` with `pre_interp_lengths = [len(a) for a in
  ft.arrays]` (already recomputed in the branch as of Task 5).
- `_multicolor_line_colors(multicolor_hue, pre_interp_lengths, xform,
  palette, is_rgb=...)` at `:5115`; `_apply_multicolor_lines` at `:5966`
  gives each SEGMENT the midpoint of its endpoints' colours.
- Setting `multicolor_hue` requires `hue = None` afterwards so the
  categorical regroup does not also run.
- On this path `_mi_style['colors']` must be DROPPED (colour comes from the
  hue), while `linewidths`/`alphas`/`labels` still apply.

**Accepted hue forms (input-relative only, F12):** flat length `len(df)`
(broadcast to every leaf), or one sequence per leaf each of length
`len(df)`. A flat array sized to the TOTAL DRAWN observations is REJECTED —
it is new API, indistinguishable from form 1 when `T == n_obs`, and would
require the caller to predict how many means expansion creates.

**Task 5 leaves two temporary guards for Task 6/8 to lift:** the column
branch currently warns-and-ignores `hue=`, and raises on `predict=`. Both
are deliberate intermediate states, not oversights.

## Reviewer's adversarial matrix (to fold into Tasks 6/8/9)

Requested in review round 1, item 4. Covered so far / still owed:

| case | state |
|-|-|
| missing labels at every hierarchy level | covered (`test_hierarchy_na_labels.py`); still owed WITH hue |
| duplicate flattened feature names | covered in grouping; still owed end-to-end through `plot()` |
| unequal trace lengths + auxiliary hue | Task 6 (`aux` co-truncation) |
| one-row traces, animated precondition ordering | Task 8 |
| row vs column hierarchy | covered |
| matplotlib and plotly | Task 9 |
| static / parallel / serial / window animation | Task 9 |
| inherited vs explicit forecast grouping | Task 8 |

## Plan counting discrepancy (not a defect, but the arithmetic is off)

The plan expects `tests/test_multiindex.py` to report **29 passed**. It is
unchanged by this work and reports **32 passed**: 29 `def test_` plus two
`@pytest.mark.parametrize` decorators that expand to 3 extra cases
(`test_multiindex_trail_alpha_no_collision[bullettime-3]` etc.). The plan's
Global Constraint anticipates exactly this — "every per-module count was
obtained by counting `def test_` … recompute rather than carry forward" — so
every downstream "baseline + N" figure in the plan is a `def test_` count,
not a collected-test count. Recomputing per task rather than carrying the
plan's numbers.

## Standing constraints in force

- `.venv/bin/python` is mandatory (system numpy breaks matplotlib).
- **Never `git stash`** in this repo (documented data-loss hazard) — use
  `git show <ref>:<path>` or a worktree.
- No subagents, no workflows (session constraint).
- Zero-warning suite; ruff parity against the base commit; docs build under
  `sphinx -W`.

## Left for the maintainer (carried over, unchanged)

- **Nothing is pushed.** `dev-1.0` is ~120 commits ahead of
  `origin/dev-1.0`; CI has not seen any of it since 2026-07-24.
- `CHANGELOG.md:460` still says `## 1.0.0 (unreleased)` on this branch
  although 1.0.0 shipped on master 2026-07-24 — `dev-1.0` never picked up
  master's release-time flips. Plan 2 Task 11 creates the `## 1.1.0
  (unreleased)` section; the stale 1.0.0 heading is separate and predates
  this work.
- 417 pre-existing repo-wide ruff findings, ungated (no lint job in CI).
