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
| (fix) nominal feature correspondence — review finding 1 | done, 14 tests | `c5662249` |
| 6 continuous hue as per-trace aux | done, 22 tests (20 focused + 2 alpha regressions) | `ea5d9b5e` |
| 7 hierarchical `hyp.predict` | done, 25 tests (plan said 22) | `c51d274d`, fix `c5fb889c` |
| 8 `predict=` over final traces | done, 17 tests | `5238d6bc`, fix `5c2f29e9` |
| 9 matplotlib/plotly parity | done, 14 → 23 tests (plan said 12) | `c9b91293`, review `a309f49e`, triage `b48c2848` |
| 10 `docs/hierarchy.rst` guide | done, 12 tests (plan said 8), 138 executed doctests | `f2a7a2b1`, fixes `cdae7096` |
| 11 CHANGELOG 1.1.0 section | done, 9 tests (plan said 6) | `b0076f8f`, fixes `cdae7096` |
| 12 verification | done, no new tests — all six gates green | `3a4ce8e0` |
| (fix) honour `legend=`, false `names=` error, plotly sentinel | done, 21 tests | `bb4ad30c` |
| (fix) true duplicate-mismatch message, aux validation, warning blame | done | `e52dd861` |
| (fix) re-appliable bundled pipeline; group-order caveat pinned | done, 23 tests | `6f07c213` |
| (fix) adversarial sweep: colorbar under `legend=False`, `legend=` containers, caller-supplied `pipeline=` | done, 37 tests, all mutation-verified | `5b15d3dc` |

**Plan 2 is COMPLETE.** It is **not releasable**, and that is a Plan 4
dependency, not a defect here: see *Task 12 as EXECUTED* below and
*Adversarial sweep and fixes (2026-08-16)*.

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
assumption* — correspondence across groups is **positional** — so `plot()`
passed `leaf.to_numpy()` into the pipeline.

**That resolution was wrong, and the maintainer's review after Task 5 caught
it (see the section below).** Deferring to the plan's own documented
assumption was the right instinct for a collision discovered mid-task, but
the assumption itself had never been stress-tested: it silently made column
ORDER part of the statistical model. The current behaviour is nominal —
`group_columns` matches feature labels across groups and permutes later
groups into the first group's order — and `plot()` still passes
`leaf.to_numpy()`, but now position MEANS name by the time it does.

**The lesson worth keeping:** "the plan already decided this" is a reason to
follow a decision, not evidence that the decision is sound. A modelling
assumption that no test can distinguish from its opposite is not settled —
it is unexamined. The permutation-invariance test that would have exposed
this took four lines and did not exist in Tasks 1–5.

## Review round 2 (maintainer, after Task 5) — nominal correspondence

**Finding (High): positional matching silently makes column order part of
the statistical model.** Reproduced by the maintainer on two label-
equivalent frames — *"frames equal after sorting columns: True / group-B
arrays equal: False / original first row: [3, 4, 5] / reordered first row:
[5, 4, 3]"* — and reproduced again here at both the `group_columns` level
and end to end through `hyp.plot`. **ACCEPTED in full.**

Decision, as directed: **nominal by default, positional only by explicit
opt-in.**

- `group_columns` requires every group to carry the same innermost-label
  multiset and permutes each later group into the FIRST group's order, so
  values travel with their labels (`_match_features_by_name`).
- Duplicates are matched across groups by `(label, occurrence)`
  (`_feature_keys`), so v6's D3 decision survives intact — no column is
  dropped, no group merged — while `['temp','temp','flow']` vs
  `['temp','flow','flow']` is now correctly an error.
- Missing feature LABELS are matched NA-aware, reusing `_canonical_label`
  from the round-1 fix. Without it every group after the first would report
  a missing feature, because `NaN != NaN`.
- Unequal group widths no longer fall through to the pipeline's generic
  `same number of columns` message; they are a label mismatch, and the
  error NAMES the missing and unexpected features.
- `group_columns(df, feature_correspondence='position')` is the opt-in, and
  the mismatch error prints it verbatim along with the two-line recipe.

**No public `plot(feature_correspondence=...)` parameter in 1.1** — my
recommendation, flagged to the maintainer. The escape hatch requires the
caller to discard the labels in their own code
(`hyp.plot([leaf.to_numpy() for leaf in leaves])`), which keeps the choice
visible at the call site instead of hidden in a kwarg, and avoids growing
`plot()`'s signature for a case with no demonstrated demand. Revisit for
1.2 if real demand appears.

**Fixture consequence, which is the interesting part.** Every Market
fixture in this plan — and Plan 4's whole Market example — used per-sector
TICKERS as the innermost level, which is precisely the shape now refused.
That is the rule working: four dollar-denominated closes per sector do not
become corresponding variables by being written in the same slot. All
fixtures now use shared measurements (`return`, `volatility`, `momentum`)
and a `ticker_frame` fixture was added to pin the REFUSAL. Plan 4 carries a
v4 revision note specifying the same change to its data preparation, with
two items flagged for the maintainer (the measure definitions, and whether
sector measurement-space still tells the example's story).

## Review round 3 (maintainer, after the nominal fix) — escape hatch + Market

**Finding: the positional escape hatch is not hierarchy-equivalent. ACCEPTED.**
`hyp.plot([leaf.to_numpy() for leaf in leaves])` plots a plain LIST, so it
loses the per-level means, the hierarchy linewidth/alpha/legend styling,
`trace_metadata`, and later the hierarchical hue and forecast behaviour.
Measured side by side on the same shape: 3 traces vs 4, `trace_metadata`
`None` vs populated, matplotlib's default 1.5 width for every line vs the
level-derived {1.0, 2.0}, `alpha` unset vs {0.7, 1.0}. Both the error
message and `plot()`'s `x` entry now say so explicitly. **There is no
hierarchy-preserving positional mode in 1.1** — that is what would justify
a public parameter, if it is ever asked for.

### The Market example, rebuilt in regime space (Plan 4 v4)

All decisions came from the maintainer; everything below was MEASURED on
2026-08-15 against live data, not projected.

- **Adjusted** closes (`indicators.adjclose`), equal-weighted constituents.
  `return` = mean constituent daily log return; `volatility` = trailing
  20-session sd; `momentum` = trailing 60-session sum.
- **2514** adjusted closes (2016-08-15 → 2026-08-14) → **2454 × 18** after
  dropping the 60 leading rows. `range=10y` SLIDES, so v3's "2513" was never
  a constant — the verify step must re-measure rather than assert.
- **Centred smoothing removed.** Verified the claim rather than taking it on
  trust: `uniform_filter1d(size=11)` spreads an impulse at t=50 over 45..55
  with **45% of its weight landing before t=50** — five sessions of
  lookahead into the history the forecast is fit on.
- **`normalize=None, reduce=None, ndims=3`**, and the payoff is checkable:
  `trace_data[0]` equals the sector's own leaf EXACTLY, so the panel's
  `return` component and the picture's first coordinate are the same number.
- **210 fits in 11.1 s**; whole script 13 s cached; static plot 0.3 s.
- **The forecast has no directional edge**: 52% at 210 fits, **48.7% at 700**
  (SE 1.9 pp). Before accepting that I measured the alternatives —
  volatility-change direction 49.9%, momentum-change 51.6%, both at 700
  fits. There is nothing with more signal to switch to, so the example
  states ~50% plainly. **Flagged to the maintainer as the one open item**:
  shipping a gallery example whose headline number is "no better than
  chance" is honest, but it is a narrative call.
- Budget: the rewrite measures **116** code lines (metric validated by
  reproducing v3's 191 for the file on disk). The **+16 split overhead is
  INHERITED, not re-measured** → projected 132, budget 130 → **135**. That
  projection is flagged in the plan as the one unmeasured number.
- `hue=` and `predict=` over a column hierarchy are still guarded (Tasks 6
  and 8), so those two kwargs are the ONLY unexercised part; with them
  removed the prescribed script runs end to end against live data.

> Superseded in places by round 4 below: the 48.7%/700-fit figure and its SE
> are replaced by a 50.0%/600-fit figure with no interval, and the 116-line /
> 132-projected / 135-budget numbers are replaced by 124 / 140 / 145.

## Review round 4 (maintainer, on the Plan 4 v4 note) — executability

Verdict on the open narrative item: **ship the chance-level result.** Four
findings, all accepted; everything below measured 2026-08-15 on live data.

1. **Fatal to executing Task 2 — the loader/builder split still named
   deleted symbols.** Step 3 prescribed `fetch_prices` / `synthetic_prices`
   / `data.prices` / `sector_index`, all deleted by the v4 rewrite in the
   step immediately above it. Rewritten against the real names, with the
   `Market(regimes, closes, source)` payload, `load_market`, `fixture_data`
   and `construct_artifact` spelled out. Task 8's fixture table said
   `Market(prices, source)`; fixed. **The same staleness is a recurring
   failure mode in this plan** — v2's names survived into v3's split, v3's
   into v4's — because a rewrite step and its split step are edited
   separately. Both now carry an explicit "these names are the rewrite's"
   note listing the dead ones.

2. **The hue was not equal-weighted.** `closes[sector].mean(axis=1)` averages
   price LEVELS, so a constituent's weight is its share price. Measured
   inside Technology: **IBM 0.477, MSFT 0.240, ORCL 0.166, AAPL 0.117** — in
   a figure whose panel says constituents are equally weighted. Not
   cosmetic: Financials ends at 599 equal-weight vs 657 price-level, largest
   gap 98 index points. Now `100·exp(cumsum(r) − cumsum(r)[0])` on a shared
   `sector_returns()` helper feeding measurements, hue and score alike —
   the maintainer's "better still" suggestion, taken, so equal weight cannot
   be derived three ways. Verified: 6 sequences × 2454, each starting at
   exactly 100.0.

3. **Three quantities, not one.** Pooled **52%** over the six sectors' 180
   fits, cross-sector mean trace **57%** of 30, per sector 37–63% (n=30
   each, explicitly not a ranking). *Deviation from the review, stated:* the
   maintainer said "pooled across all 210 fits"; I pooled **180**, the six
   sectors only. The mean trace is computed from those same six histories,
   so including it counts the same data twice. The panel and the printed
   line label all three.

4. **The SE is gone.** Re-measured the large-sample check with the SAME
   definition as the printed statistic — 100 anchors × 6 sectors = 600 fits
   — giving **50.0%** (31 s), replacing the earlier 48.7%/700. No interval
   is quoted anywhere: overlapping 60-session windows, shared anchor dates
   and correlated sectors make these non-independent trials, so a binomial
   SE would overstate the precision.

**Also found while checking, not in the review.** v4 raised the market script
budget 130 → 135 in prose but left `SCRIPT_BUDGETS` at **130** — the enforced
number contradicted the plan and would have failed
`test_file_is_within_its_size_budget`. Both are now **145**, and Step 0c's
measured table carries a note that market's row is no longer a measurement.

**Budget re-measured:** the rewrite is **124** code lines (116 + 8 for the
shared helper and the three labelled quantities). +16 split overhead still
INHERITED → projected **140**, script budget **145**, notebook **150**. The
payload gaining a third field makes +16 a floor, which the plan now says.
Whole script 13.7 s cached.

**Recorded for later, at the maintainer's request:** the ship-it decision was
made against measurements with no rendered figure in existence. The plan
carries a REVISIT block in both *Revision note (v4)* and the *Decisions still
needed* entry: once Task 2 executes, re-open the question with the demo in
front of you — is a chance-level headline honest and educational in context,
or does it read as broken? Check panel legibility at real figure size and
whether the title carries the story alone.

**Method note worth keeping:** the plan's "docstring-aware metric" is
PHYSICAL non-blank/non-comment/non-docstring lines, not logical statements
— an AST logical-statement counter gives 159 where the plan says 191. The
way to be sure is to reproduce a number the plan already records before
trusting a counter on new code.

## Ruff counting: use ruff's own summary, not a grep

Earlier parity checks used `grep -c '^[A-Z][0-9]'`, which UNDERCOUNTS. The
reliable figure is ruff's own `Found N errors.` line, and the reliable
comparison is a **set difference** against a base worktree keyed on
`(file, code, message)` with line numbers stripped. That method caught a real
slip in Task 5: an added `np.asarray` registered as a NEW `F405`, because
`plot.py` takes `np` from a star import and every `np.` use is already
flagged (~190 of them). `leaf.to_numpy()` avoids the reference. Parity is
**353 at base `59405545`, 353 now**.

## Task 6 as EXECUTED (2026-08-15) — what the design notes below missed

The design below was right about the structure and wrong about four
observable details. Everything here is measured.

- **`ax.collections` is not a list of data artists.** The 3-D bounding cube
  is SIX `Line3DCollection` wireframe faces (`matplotlib_backend._draw_cube`),
  so the plan's `_collections()` helper counted 6 too many — a no-hue
  hierarchy plot already has them. Fixed by TAGGING: `_apply_multicolor_lines`
  sets `coll._hyp_trace_index = i`, mirroring the existing
  `_hyp_forecast_role` tag on forecast lines, and the test helper filters and
  sorts on it. Task 8 needs the same handle for forecast colours.
- **`Line3DCollection.get_segments()` returns `[]` until a draw** — it
  projects the private 3-D segments. The co-truncation test calls
  `fig.canvas.draw()` first and asserts a non-zero segment count so it cannot
  pass vacuously.
- **One of the plan's own tests passed BEFORE the implementation.**
  `colorbar=True` produced a second axes even while the hue was being dropped,
  so `len(fig.axes) == 2` proved nothing. Strengthened to assert the
  colorbar's limits equal the concatenated aux range.
- **A defect outside the task made one of its tests unsatisfiable.**
  `_apply_multicolor_lines` never read `alpha` from its per-trace kwargs, so
  EVERY continuous-hue plot rendered fully opaque however `alpha=` was
  spelled — the artists carrying the alpha are the `Line2D`s it removes. The
  hierarchy's level-derived alphas need that same channel. Fixed, with two
  regressions for the plain non-hierarchy case in
  `tests/plot/test_per_dataset_alpha.py`.

Also added, unprompted by the plan: the hue is validated against the INPUT
frame's rows, so a row-count-changing pipeline stage (`manip='Resample'`)
could invalidate it between validation and use. A per-leaf length check
before `build_hierarchy_traces` raises naming the stage.

Adversarial-matrix items discharged here: NA hierarchy labels WITH hue
(parametrized over `np.nan`/`None`/`pd.NA`), duplicate innermost feature
names end-to-end through `plot()`, and aux co-truncation under unequal-length
members. The last is a UNIT test on `build_hierarchy_traces` on purpose: a
column hierarchy slices ONE frame, so unequal member lengths are unreachable
through `plot()`, and a `plot()`-level test would be vacuous rather than
merely awkward.

Measured: focused module **20 passed**; suite **3331 → 3353**; ruff set
difference vs `59405545` **empty both directions** (141 = 141 unique
`(file, code, message)` keys; the raw count moves 353 → 373 because 20 more
lines use `np.` under the pre-existing `import *`, which is the same key
repeated, not a new finding).

## Task 6 design notes (as written BEFORE execution — kept for the record)

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
are deliberate intermediate states, not oversights. *(The hue guard is
LIFTED as of Task 6 — a continuous hue is carried through, and only a
categorical one still warns and defers. The `predict=` guard remains, for
Task 8.)*

## Reviewer's adversarial matrix (to fold into Tasks 6/8/9)

Requested in review round 1, item 4. Covered so far / still owed:

| case | state |
|-|-|
| missing labels at every hierarchy level | **covered, incl. WITH hue** (`test_multiindex_hue.py`, parametrized over `np.nan`/`None`/`pd.NA`) |
| duplicate flattened feature names | **covered end-to-end** through `plot()` under a hue (`test_multiindex_hue.py`) |
| unequal trace lengths + auxiliary hue | **covered** as a unit test on `build_hierarchy_traces` — unreachable through a column `plot()`, which slices one frame |
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

## Tasks 7-12 as EXECUTED by the ultracode run (2026-08-16)

Tasks 7-11 were implemented by dispatched subagents, one per task, each
under the same rule: **run every test the plan prescribes BEFORE
implementing; if it passes already it does not test the feature, so
strengthen it until it fails and report that.** That rule paid for itself in
every single task. The per-task detail lives in the commit bodies and in the
plan's own EXECUTED notes; what follows is the pattern.

### The plan's prescribed tests were defective in every task

| task | what the prescribed tests got wrong |
|-|-|
| 7 | `test_flat_frame_return_type_is_unchanged` PASSED before the feature existed (it is a no-regression guard, not coverage); `test_unsorted_times_warn_naming_the_group` did not test its own claim; one test leaked an unasserted warning. 22 prescribed → **25** shipped. |
| 8 | the prescribed `_solid`/`_dashed` helpers **cannot work at all** — `_forecast_style_from` makes a forecast INHERIT its source linestyle, so under `fmt='-'` `_dashed(ax)` is always `[]`, and 9 of 17 tests depended on them; one test's `zip` paired **cube wireframe faces** with forecasts; two more passed before implementation and were strengthened first. |
| 9 | the block was **9 failed / 3 passed and could not have passed as written**: `_data_traces` filtered `name != 'cube'` but the cube trace is unnamed; `t.line.dash` selects every line (plotly spells solid `dash='solid'`); the F14 colour test compared an `rgba()` slice against an `rgb()` slice, which can never be equal; points vs pixels; `antialias=True` ignored; and `test_colorbar_renders_on_plotly` was **vacuous** (plotly instantiates a `ColorBar` on every trace). |
| 10 | `test_api_rst_links_the_guide`'s `X or Y` made its real clause dead, and `count('hierarchy') >= 2` was met by any two mentions anywhere. |
| 11 | 3 of 6 could not detect what they claim: `_section()` bounded a section at `\n## `, which cannot match `\n### `, so every "Changed / validation says X" assertion could be satisfied by text under *Documented limitations*; `assert 'list' in changed.lower()` is satisfied by "listed"; and nothing executed anything. 6 → **9**. |

Two tests were proven inert by **mutation** rather than argument, which is
the part worth keeping: Task 8's `test_forecast_takes_the_final_observed_hue_colour`
gave every leaf the SAME hue ramp, so all four traces ended in one colour and
`zip(colls, forecasts)` compared a colour against itself four times — an
off-by-one in `_hyp_forecast_dataset` left all 17 tests green. And Task 8's
animated path was asserted by COUNTS only: rotating `analyze_histories` by one
detached every animated forecast from its trace (gaps 0.0 → 0.641/0.512/…)
with the whole module still passing.

### Four defects found outside the plan's scope, fixed anyway

1. **Contract 10's message blamed the grouping for what the PIPELINE did**
   (`5c2f29e9`): a 30-row frame under `manip='Resample', n_samples=1` was told
   *"the input itself has only one observation … pass a frame with more rows"*.
   Fixed by capturing each leaf's PRE-pipeline row count.
2. **Under a continuous `hue=`, plotly discarded per-trace alpha entirely**
   (`a309f49e`) — hierarchy level alphas and plain `alpha=` both rendered fully
   opaque, on the backend the maintainer requires to be identical.
3. **With `ndims=1`, matplotlib drew the `predict=` overlay at x = 0..t**
   (`a309f49e`), painting every forecast back over the START of the plot. The
   seam VALUE was right, which is how it survived; only x was wrong.
4. **14 sites still dated behaviours to a release that was never published**
   (`cdae7096`), and a test actively FORBADE correcting them.

One reported finding was **declined with evidence** rather than "fixed":
`cluster=` does not make the backends draw different coordinates —
`Line3D.get_xdata()` returns PROJECTED coordinates once a figure is drawn, and
`cluster=` triggers a draw. Read through `get_data_3d()` the vertex multisets
are equal. A fix for a non-defect is a regression.

### Task 12 as EXECUTED — every gate MEASURED at `cdae7096`

| gate | result |
|-|-|
| full suite `.venv/bin/python -m pytest -q` | **3465 passed, 13 skipped, 2 deselected in 705.10s** — 0 failed, 0 errors, **no "warnings summary" section** |
| delta vs the 3406 baseline at `5c2f29e9` | **+59** (Tasks 9-11); vs the plan's true base `59405545` (3331) the whole plan is **+134**, not the plan's projected +151 |
| five publication gates | **47 passed, 2 skipped** (the skips are the release-gated cases at `test_notebook_install_gate.py:125,144`) |
| `tests/plot tests/core tests/predict` | **952 passed** |
| docs `sphinx -b html -W -E -a` | **build succeeded**, 0 warnings; `hierarchy.html` built and linked 14x/3x/2x from index/api/tutorials |
| `predict → plot` layering | **`[]`** |
| exactly one mean-construction site | **`hypertools/plot/hierarchy.py:171`**, only |
| ruff set-difference vs `59405545` | **empty in both directions**, 141 keys each side |
| `git status --short` / `git worktree list` | empty / main checkout only |

Nothing needed fixing, so Step 6's "re-run everything after any fix" did not
fire; all six gates ran against one unmodified tree.

**Three plan-text corrections Task 12 itself needed:**

1. **Tasks 9 and 10 shipped with no EXECUTED note and not one ticked step.**
   Their commits record measured gates in their bodies, but nothing reached
   the plan, so at HEAD the file read as though plotly parity and the guide
   had not started. Backfilled from those commit bodies, each number
   attributed to its sha and labelled as commit evidence, not
   re-measurement. **This is the same failure mode round 4 already named** —
   "a rewrite step and its split step are edited separately" — recurring
   one level up, in the plan's own bookkeeping.
2. **Step 2's "40 passed" is a `def test_` count, not a collected count.**
   13+8+4+4+11 = 40 functions is right; parametrisation makes it 49
   collected, so a green run is **47 passed / 2 skipped**. Same class of
   error as the *Plan counting discrepancy* section above.
3. **Step 4's layering one-liner has an operator-precedence bug**:
   `'.plot' in src and 'hypertools.plot' in src or 'from ..plot' in src`
   binds as `(A and B) or C`. It was re-run as written AND with a regex over
   every `import`/`from` line under `hypertools/predict/` — both `[]`, so the
   empty result is not an artifact of the precedence.

### The one thing that is RED, and it is not this plan's

**Plan 4 has not landed, so 1.1 is not releasable from this tree.** Measured:
`docs/tutorials/market_forecast.ipynb` exists but contains **0**
`MultiIndex.from_tuples`, and `examples/animate_market_forecast.py` contains
**0** `MultiIndex` references. Task 12 Step 2a's own instruction is to record
that rather than tag, which is what was done — in the plan's checklist table
and here. The flagship demonstration of everything Tasks 5-9 add still shows
the flat market.

## Adversarial sweep and fixes (2026-08-16)

After the three fix commits (`bb4ad30c`, `e52dd861`, `6f07c213`) landed, a
completeness critic and an adversarial user swept the finished branch. Four
Critical/Important findings were routed here. **All four reproduced**, so
none was declined; two of them are the same defect reported twice.

### What was fixed (`5b15d3dc`)

1. **`colorbar=True` + `legend=False` under a hierarchy lost the group
   names AND the leaf filter.** A *regression* introduced by `bb4ad30c`,
   never released. That commit made the hierarchy's
   `legend = _mi_style["labels"]` install conditional on
   `legend is not False` — correct for the legend, but `_build_colorbar_info`
   reads the colorbar's group names off that SAME list, and relies on its
   `'_nolegend_'` entries to collapse leaves and intermediate means down to
   the top-level groups. With `legend=False` it got `None` and fell through
   to `labels = [i + 1 for i in range(n_groups)]`.

   Measured at `6f07c213`, 3-level column frame (US/EU x tech/fin x a,b,c),
   matplotlib colorbar tick labels: default `['US', 'EU']`,
   `legend=False` `['1' ... '6']`. Same on plotly (`marker.colorbar.ticktext`)
   and under `animate=True`. On a ROW hierarchy the COUNT is the loud part:
   the reporter's 3-level frame (G1/G2 x s1/s2 x 10 rows = 40 leaves + 4
   subject means + 2 group means) went from **2 segments to 46**. The
   regression test here uses the 2-level version of the same shape (4 leaves
   + 2 group means, so 2 -> 6), which fails identically and runs in
   milliseconds.

   Fixed by giving the colorbar its own `_mi_colorbar_labels`, set
   unconditionally in the hierarchy branch and threaded into
   `_build_colorbar_info` as `hierarchy_labels=`, ranked below the user's
   `legend` list and above `hue_group_labels`. `legend=False -> legend = None`
   still happens, so nothing leaks back into the drawn legend (asserted).
   The colorbar is the colour key for the drawn groups, not a second legend.

2. **`legend=` as an ndarray/Series/Index mislabelled every trace** —
   *pre-existing*, not a regression (the reporter states it reproduces
   identically at `3a4ce8e0`; I measured it at `6f07c213` only), but at the
   exact site the fix pass edited. All three are accepted by the type check and by
   `bb4ad30c`'s new `_legend_user_list`, but the per-trace length check and
   the label assignment tested `list`/`tuple` only, so the whole container
   became EACH artist's label: two traces both named `['a' 'b']`, plus two
   matplotlib "Passing label as a length 2 sequence" warnings. The hierarchy
   path handled all four containers correctly, so the two paths disagreed
   about the same accepted input.

   Fixing it turned up one more case the reviewers did not report and I
   measured myself: a **tuple** legend labels the traces correctly but misses
   `_build_colorbar_info`'s narrower `isinstance(legend, list)` test, so
   `legend=('A', 'B'), colorbar=True` drew a colorbar reading `['1', '2']`.

   Fixed by normalising every accepted container to a plain list where it is
   type-checked. A 0-d array (`np.array('a')`) is not iterable, so it becomes
   ONE label — the same thing the existing `isinstance(legend, str)` wrap does
   with `legend='a'`, which then reports the length mismatch instead of
   silently broadcasting one label over every trace (measured before:
   `['a', 'a']`).

3. **A caller-supplied `pipeline=` was bundled without `input_hierarchy`.**
   `6f07c213` recorded the column grouping only on the pipeline `plot()`
   builds for itself. `plot.py`'s `if pipeline is not None: bundle_pipeline =
   pipeline` handed the caller's object back untouched, so
   `bundle['pipeline'].transform(df)` still raised the exact pre-1.1.0
   scikit-learn error the `return_model` docstring says it no longer raises:
   measured `ValueError: X has 15 features, but IncrementalPCA is expecting 5
   features as input`.

   Fixed by hoisting `_bundle_hierarchy` above the branch and recording it on
   the passed-in pipeline too — **in place**, because the bundle hands back
   that same object by design (the docstring says so and
   `tests/test_cross_module_kwargs.py:194` asserts the identity). Two guards
   make that safe, both commented at the site: an `input_hierarchy` the
   caller's pipeline ALREADY carries belongs to its own fit and is left
   alone, and the recorded `n_features` is not a guess — the `analyze(raw,
   pipeline=pipeline)` call that drew the figure already pushed those same
   groups through every fitted step, so a width disagreement would have
   raised there first.

### Nothing was declined

All four reproduced on the first attempt with the reporters' own repro
shapes. Findings 1 and 2 in the routed list are the same defect (column
frame vs. 30x12 frame), fixed once.

### Tests

37 new tests, every one proven red-then-green by MUTATION (break the fix,
watch the test fail, restore):

| file | tests | mutation result |
|-|-|-|
| `tests/plot/test_colorbar_group_names.py` (new) | 7 | 6 failed with the fix broken; the 7th pins the paths that were already right and correctly passes both ways |
| `tests/plot/test_legend_containers.py` (new) | 27 | 10 failed with the fix broken (the `list`/`tuple` params pass both ways, which is the point) |
| `tests/test_hierarchy_group_order_and_pipeline.py` (appended) | 3 | 1 failed with the fix broken; the other two pin the "leave an existing record alone" and "record nothing off-hierarchy" branches |

Also documented: `CHANGELOG.md` (one amended bullet, two new bug-fix
bullets) and `docs/hierarchy.rst` (the `legend=False`/colorbar sentence, plus
a new executed doctest for the caller-supplied pipeline).

### Every gate MEASURED on the committed tree

| gate | result |
|-|-|
| focused (3 files) | **60 passed** in 3.26s |
| neighbours `tests/plot tests/core tests/predict` | **1033 passed** in 173.16s |
| full suite `.venv/bin/python -m pytest -q` | **3572 passed, 13 skipped, 2 deselected in 717.88s** — 0 failed, 0 errors, **no "warnings summary" section** (`grep -c 'warnings summary'` → 0) |
| ruff parity vs `59405545` | **141 keys each side, `comm` empty in BOTH directions**; raw count 380 (see the ruff note below) |
| docs `sphinx -b html -W -E -a` | **build succeeded**, zero warnings (`-W` makes any warning an error) |
| `git status --short` / `git worktree list` | clean / main checkout only |

**The full suite caught a real thing on the first run** and is worth
remembering: `tests/test_packaging_artifacts.py::
test_sdist_contains_only_tracked_files_plus_allowlist` FAILED with *"2
untracked file(s) leaked into the sdist"* because the two new test files were
not yet `git add`ed. Not a flake and not a test to work around — the gate is
doing its job. `git add` first, then run the suite.

### OPEN ITEMS — the maintainer must decide these

None of these is a defect in this branch's implementation; each is a
product decision that this work surfaced and did not have standing to make.

1. **`hyp.predict`'s duplicate-timestamp rejection is GLOBAL, and that is
   wider than the plan's Compatibility table.** Verified 2026-08-16 on an
   input with **no MultiIndex on either axis** (`df.index.nlevels == 1`,
   `df.columns.nlevels == 1`, a 5-row frame with one repeated
   `DatetimeIndex` entry): `hyp.predict(df, model='Kalman', t=2)` raises
   *"the dataset index has 1 duplicated entry ... so the forecast horizon is
   ill-defined"*. `CHANGELOG.md:12` already warns that this one "is not
   hierarchy-specific and reaches flat `hyp.predict` callers", so the fact is
   recorded — the DECISION (accept the wider blast radius in a minor release,
   or narrow the check to hierarchical inputs) is not made.
2. **F14 vs Decision R3: animated forecasts under a continuous hue wear the
   PALETTE colour, not the hue anchor, on BOTH backends.** The static overlay
   takes the trace's final observed hue colour; the animated one takes the
   colour of the run drawing the head. The backends agree frame for frame —
   it is static vs. animated that disagree. Recorded in *Documented
   limitations* as "an open product decision rather than a defect"; it is
   still open.
3. **`_to_plotly_color` now ROUNDS instead of truncating.** The two plotly
   colour helpers disagreed (one truncated each channel, one rounded), so
   `rgb(219,95,87)` on matplotlib came out `rgb(219,94,86)` on plotly and an
   anchored forecast could not equal the colour it was copied from. Both
   round now — which is a **user-visible change to the colour strings in
   exported HTML**, up to 1/255 per channel. Correct, but it moves bytes in
   anything a user has saved and diffed.
4. **Plan 4 Task 2 is still unlanded, so 1.1 is NOT releasable.** Nothing in
   the tree demonstrates a column hierarchy end to end:
   `docs/tutorials/market_forecast.ipynb` contains **0**
   `MultiIndex.from_tuples`, and `examples/animate_market_forecast.py`
   contains **0** `MultiIndex` references (re-confirmed 2026-08-16). The
   flagship demonstration of everything Tasks 5-9 add still shows the flat
   market.
5. **In-place mutation of a caller's `Pipeline`.** Fix 3 above writes
   `input_hierarchy` onto the object the caller passed to `pipeline=`. This
   follows from two promises that were already in the docstring (the bundle
   returns that same object; that object re-applies to a hierarchical frame),
   and it matches what the auto-built path already records — but it IS a
   visible side effect on an argument, now documented in both the `pipeline=`
   and `return_model` docstring entries. Worth an explicit nod.
6. **The reducer/pipeline agent's own `concerns` were not available to this
   agent.** I had no channel to that agent's structured output, so if it
   raised concerns beyond the three findings routed to me, they are NOT
   captured here and must be collected from its report before sign-off.

## Standing constraints in force

- `.venv/bin/python` is mandatory (system numpy breaks matplotlib).
- **Never `git stash`** in this repo (documented data-loss hazard) — use
  `git show <ref>:<path>` or a worktree.
- ~~No subagents, no workflows (session constraint).~~ **Lifted on 2026-08-16:**
  Tasks 7-12 were run as one dispatched subagent per task (the "ultracode"
  run). Tasks 1-6 were inline, which is why the note above them reads that way.
- Zero-warning suite; ruff parity against the base commit; docs build under
  `sphinx -W`.

## Left for the maintainer (carried over, unchanged)

- **Nothing is pushed.** `dev-1.0` is **140** commits ahead of
  `origin/dev-1.0` (measured 2026-08-16, was 128 on 2026-08-15); CI has not
  seen any of it since 2026-07-24.
- ~~`CHANGELOG.md:460` still says `## 1.0.0 (unreleased)`~~ **FIXED** in
  Task 11 (`b0076f8f`): the heading is now `## 1.0.0 (2026-07-24)`, verified
  byte-identical to `git show master:CHANGELOG.md` apart from the date. The
  same commit created `## 1.1.0 (unreleased)` and moved `pyproject.toml`
  1.0.1 → 1.1.0, which an existing gate
  (`test_changelog_top_version_matches_pyproject`) forced.
- Pre-existing ruff findings, ungated (no lint job in CI). **State the scope
  when quoting the number**: `ruff check` over the whole repo → **441**;
  over the code this work touches, `ruff check hypertools tests` → **380**
  (was 377 at `cdae7096`, 353 on 2026-08-15). **The raw count moving is
  expected and is not a regression** — `plot.py` takes `np` and `pd` from a
  star import, so every added `np.`/`pd.` line repeats a pre-existing
  `F405`; the +3 at this commit is exactly the `np.ndarray`/`pd.Series`/
  `pd.Index` mentions in the new `legend=` container normalisation. The gate
  is the SET difference of `(file, code, message)` keys against base
  `59405545`, which is **empty in both directions, 141 keys each side**,
  re-measured on the committed tree of the adversarial-sweep fixes.
