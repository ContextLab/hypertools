# Session notes — HyperTools 1.1 planning (2026-07-26 → 2026-07-28)

**Outcome:** four implementation plans + an index README, all reviewed and rewritten. No library
code changed, nothing committed. Entry point: `docs/superpowers/plans/README-hypertools-1.1.md`.

## What triggered it

Jeremy's verdict on the launch examples: "aside from the non-hypertools panels this should all be
baked into the toolbox and achievable with just a few lines of toolbox calls." Measured and
confirmed: **6.0% of the five launch examples is hypertools calls; 37.9% is defect.** The 15 older
tutorials run 2.5%–39%; `analyze.ipynb` never calls `hyp.plot`.

## The plans

| # | file | size |
|-|-|-|
| 1 | `2026-07-26-hypertools-1.1-animation-core.md` | 9 tasks / 68 steps |
| 2 | `2026-07-28-hypertools-1.1-multiindex.md` | **v5** — 12 tasks / 85 steps |
| 3 | `2026-07-27-hypertools-1.1-forecast-animation.md` | 8 tasks / 55 steps |
| 4 | `2026-07-28-hypertools-1.1-examples-and-tutorials.md` | 8 tasks / 55 steps |

## THE LESSON OF THIS SESSION

**Every plan I wrote first-draft contained tests that could not pass.** All three were caught by
adversarial review against source, never by re-reading. The failure mode: writing a test that
*looks* right without tracing the code path it exercises.

- Plan 2 v1: asserted a grouping helper returns 2 groups; `expand_multiindex` (`multiindex.py:76-81`)
  builds one leaf per unique FULL index tuple → 2×T one-row leaves.
- Plan 1 v1: keyed morph-title blanking off `current_fraction`, but holds AND transitions both
  sweep 0→1. The rule blanked mid-hold and named mid-transition — and its own test passed anyway
  (blank fraction 12/24, inside the asserted window). Correct discriminator: `seg_idx % 2`.
- Plan 3 v1: silently redefined `t` ~15x. Animations animate the ANTIALIASED array (60 raw rows →
  904 drawn), so `t=3` forecast 0.20 real samples.

**Process that worked:** write → adversarial subagent review against source (run code, don't read)
→ rewrite with a "Revision note (v2)" table of each error vs verified reality.

## Cross-plan integration bugs (found late, would have blocked execution)

1. Plan 1's rewrite inserted a new Task 4 (plotly parity), shifting every later task by one. Plans
   2 and 3 still cited `order=` as Task 4 (now 5) and the hook as Task 6 (now 7). FIXED.
2. Plan 3 depended on `_register_frame_callback(line_ani, fn)` — **which Plan 1 v2 never ships**.
   It provides `FrameHooks` (`.callbacks`, `.record()`, `.dispatch()`). FIXED.

3. Plan 2 cited "README *Decisions still open* #5" for the silent forecast drop — but #5 was the
   row-in-list decision; the drop was #9. Caught while renumbering the README, which would
   otherwise have made the stale citation accidentally correct. FIXED (now cited by name).

Lesson: when sibling plans are rewritten independently, re-verify every cross-reference.
**Cite decisions by NAME, not by number — numbers move.** This class has bitten four times.

## Round 3 (v4) — the reviewer was right in principle, wrong in the example

Two blockers, both reproduced before acting:

- **Row hierarchies can't always forecast.** `expand_multiindex` = one leaf per unique FULL index
  tuple, so a frame whose innermost level is unique per row yields ONE-ROW traces, and
  `predict/common.py:256` refuses those. Measured: 2 cond x 3 subj -> 6 leaves of shape (1, 4).
  Resolved as: column hierarchies always qualify; row hierarchies only when every leaf AND derived
  mean has >= 2 rows, checked as a precondition over the final traces.
- **`trace_data is xform_data` is NOT universal.** `xform_data = copy.copy(xform)` (`plot.py:2827`)
  is a SHALLOW copy; the display block then REBINDS `xform` (`:2886-2919`), which the alias never
  sees. Measured on a FLAT input: `reduce={'model':'PCA','kwargs':{'n_components':5}}` ->
  `xform_data` (60, 5) vs a 3-D artist. Needs an explicit spec >3 components; `:2887` RAISES when
  `reduce is None`, which is why the suite never caught it.

**BUT** the review's named frame was wrong, and I verified rather than accepting it:
`tests/test_multiindex.py:479` uses `_make_2level_df()`, which repeats each `(cond, subj)` tuple
`n_time=10` times -> **8 leaves of shape (10, 3)**, 10 drawn traces, all forecastable. So that test
stays a permissive change and became the POSITIVE test; the raising test was added on the 6-row
frame. v3's real defect there was smaller and unnamed: it promised 8 forecasts where the frame
draws 10 traces.

Lesson: a reviewer's claim is evidence, not proof. Reproduce the specific example, not just the
principle — the principle can be right while the instance is wrong.

Four open decisions were also resolved (row-in-list = reject COLUMN axis only, so
`tests/test_multiindex.py:453` passes untouched; row hue unchanged; no public frame stepping; row
time-likeness warns/rejects). **11 decisions remain open, none in Plan 2.**

## Round 4 (v5) — an edge case, and an assertion that punishes correct code

Reviewer verdict on v4: "essentially implementation-ready, with one edge-case correction still
needed." Both findings confirmed by measurement.

- **The >=2-row precondition was gated to the row axis.** A `T=1` COLUMN hierarchy also yields
  one-row traces: measured, leaves `(1,3)` and mean `(1,3)`. Now axis-independent — `len(trace)>=2`
  over every final trace on both axes, with only the REMEDIATION text differing (row: full-tuple
  expansion, flatten or move grouping to columns; column: the input itself has one observation).
  Note the asymmetry: for columns every group has `len(df)` rows so the rule collapses to
  `len(df)>=2`; for rows leaves can differ in length, so it must be per-trace.
- **`assert not np.allclose(bundled, avg_of_leaves)` fails on CORRECT code.** Forecasting
  approximately commutes with averaging as leaves co-move. Measured (Kalman, t=1, T=150, 3 leaves,
  scale ~100, 5 seeds per rho): holds 5/5 at rho=0.0 (diff 0.557) and 0.5 (0.524), 3/5 at rho=0.8
  (0.130), **0/5 at rho>=0.9** (0.028 / 0.007 / 0.0003). Real market sectors co-move at rho~0.7-0.9
  — so it fails on the plan's own flagship example. DELETED; the positive assertion
  `bundled == hyp.predict(mean_traj)` at rtol=1e-6 already proves the contract. The measurement is
  in the surviving test's docstring so nobody re-adds it.

  An earlier probe on INDEPENDENT random walks passed 13/13 — which is exactly why this needed
  measuring rather than reasoning. Independence is the best case, and even there the margin was
  only 4-9x the tolerance.

**Cross-plan gap found while generalizing (not in the review).** Plan 3 solves the same shape
problem with the OPPOSITE policy: `forecast_from_history` RETURNS NONE below `min_history=2`
(`forecast-animation:289`), vs Plan 2 RAISING. Both right, but the combination was unspecified: an
animated one-row-trace hierarchy would silently draw nothing forever. Resolved as a two-level rule
now in both plans and the README:
- **precondition = FULL trace length (permanent)** -> runs even when animated, before the schedule
  is built, RAISES;
- **`min_history` = per-frame revealed history (transient)** -> returns None so opening frames of a
  legitimate animation just show no forecast yet.
No conflict: a long-trace animated hierarchy passes the precondition while `min_history` still
suppresses its opening frames.

Lesson: when a rule is generalized on one axis, check whether a SIBLING plan solves the same
problem with a different policy. The policies can both be right and still combine badly.

## Round 5 — Plan 1's four decisions resolved; one of them disproved the plan's own premise

Jeremy resolved all four (2026-07-29). **Plans 1 and 2 now have ZERO open decisions; 7 remain
(4 in Plan 3, 3 in Plan 4).** Plan 1 -> v3, verified independently (9 tasks / 70 steps, 187 KB).

1. **`morph_samples` -> a new `simplify=` flag**, replacing the v2 raise. Below the cap: no-op.
   Above + `simplify=True` (DEFAULT): silently downsample, **NO warning** (explicit). Above +
   `simplify=False`: raise, naming the cost and suggesting `simplify=True`. Contract 7 rewritten
   as conditional; `hypertools/plot/morph.py:17-24`'s in-source guarantee is now a tracked
   deliverable to update (a stale in-source guarantee is the defect class these reviews keep
   catching). Interpretation flagged in-plan: "print a message and then raise" = ONE raise carrying
   the message, not a print() + raise.
2. **`on_frame=` ships on BOTH backends; the `NotImplementedError` is DELETED.** Jeremy asked me to
   explain the limitation instead of accepting it, and the premise turned out FALSE:
   - `_add_animation` (`plotly_backend.py:2517`) builds every frame in a **Python loop at build
     time** (`frames.append(go.Frame(...))` at :2729 spin, :2819 morph, :2865 serial, **:2975
     parallel/window** — FOUR sites; this line originally listed only three, corrected 2026-07-30).
     Plotly's missing loop is at PLAYBACK, not build.
   - matplotlib fires at RENDER time (`FuncAnimation`, blit=False, `matplotlib_backend.py:1935`);
     lazily when displayed, eagerly when saved (`animate.py:116`).
   - So schedules CANNOT be identical, but OUTPUT can. Contract: **`on_frame` must be a pure
     function of its `FrameContext`**; output parity is the tested guarantee
     (`test_on_frame_output_parity_across_backends`).
   - Verified safe: all 4 examples that monkeypatch `_func` compute per-frame content purely from
     the frame index (`shape_title(frame)`; 0 live-state reads in conversation/weather; market's
     2 reads are one-time SLOPE/BLO/BHI setup constants).
   - **Backend parity now has NO exception in 1.1.**
3. **Animated continuous-hue linewidth 1.5 -> 1.0.** Shipped; changelogged as a visible change.
4. `order='serial'` with spin/window: warn-and-ignore, unchanged.

**STRUCTURAL FIX to the citation-drift class (bit 4x):** the README's open decisions are no longer
NUMBERED. They are named bullets with an explicit "cite by name" note. Renumbering can no longer
invalidate a sibling citation.

Lesson: when a plan says a thing is impossible, check it. "Explain the limitation exactly" turned
a documented permanent exception into a deleted one.

## Round 6 (v6) — leaves were never flattened; unbounded recursion on BOTH axes

Jeremy found that `group_columns` built leaves as `sub.T`, which KEEPS the original full column
MultiIndex. Verified:

```
('M','Tech') -> leaf.columns = [('M','Tech','AAPL'), ('M','Tech','MSFT'), ('M','Tech','NVDA')]
                names = ['Market','Sector','Ticker']   isinstance(..., MultiIndex) == True
```

Two consequences: (1) it violated the plan's OWN contract that the innermost column level is the
feature axis; (2) hierarchical `hyp.predict` recurses per group, so a leaf still carrying its
grouping levels is re-detected and regrouped forever.

**I then checked the ROW axis — same defect, worse signature.** `expand_multiindex` leaves are
DataFrames whose INDEX is still a MultiIndex with ONE unique tuple:

```
expand_multiindex(leaf0) -> 1 leaf, shape identical to leaf0    FIXED POINT
```

A row leaf re-expands to EXACTLY ITSELF — no accidental depth bound at all, whereas the column case
at least regroups into something structurally different.

**The row fix is NOT the column fix by symmetry** (the trap). For columns the innermost level IS
the feature axis, so `get_level_values(-1)` is right. For rows the innermost level is not
necessarily time — in `_make_2level_df` it is `subj`, repeated, with time implicit in row position
— and the plan already carried a datetime-index-preservation requirement from round 2. Resolved by
splitting the row rule by PURPOSE:
- **row (plot):** unchanged, leaves keep the full MultiIndex (frozen by a Global Constraint) and
  must never be fed back into a hierarchy-detecting entry point;
- **row (predict):** group by every level above the innermost, dropping ONLY those grouping levels
  so the innermost survives as a FLAT single-level index with its name and dtype. No RangeIndex
  fallback. That single rule satisfies the datetime promise and the flatness invariant at once.

New **Contract 11**: *every leaf returned by a `core/hierarchy.py` grouping helper is
non-hierarchical on the axis it was grouped along*, on both axes, and re-running a helper on a leaf
is refused rather than nesting.

**Seam I closed myself after verifying v6:** `FinalTraces.arrays` was typed only as `list`. Since
`expand_multiindex`'s row leaves ARE DataFrames, an untyped `arrays` left the "never fed back" rule
resting on discipline. Now `list[np.ndarray]`, with `build_hierarchy_traces` coercing via
`np.asarray` — so the invariant holds BY CONSTRUCTION, and no index or column labels survive to be
re-detected.

**Duplicate innermost feature names: PERMITTED positionally**, decided on measurement not
preference. `['AAPL','AAPL','NVDA']` -> `np.asarray` (20,3), `hyp.predict` (1,3), `hyp.plot` Figure;
groups do not merge (2 groups, widths [3,3]). Nothing downstream is name-addressed.

Lesson: a defect found on one axis is a hypothesis about the other. Check the sibling — but do not
fix it by symmetry; check what the OTHER axis's innermost level actually means first.

## Round 7 (v7) — MY OWN edit created the next defect

When I closed the `ft.arrays` seam in round 6 I typed the ndarray contract into Task 2's
*Interfaces* block — declaring `arrays: list[np.ndarray]` and promising
`assert all(isinstance(a, np.ndarray) for a in ft.arrays)` — but never wired it into Task 2's
implementation step or tests. Task 2 Step 3 still said the loop moves "verbatim, with three
changes", none of which was the coercion. An implementer following the steps literally would keep
`arrays = list(leaf_arrays)`, preserve DataFrames, and silently re-open the D2 recursion path.

Jeremy caught it. Fixed: Step 3 now names the coercion as the FOURTH change and shows both the
wrong and right lines; a new test asserts the PREMISE (the leaves really are hierarchy-carrying
DataFrames) before asserting every trace — leaves and means — is a plain ndarray, plus that the
caller's frame is untouched. Module 13 -> 14, suite 149 -> 150. That module was also missing
`import pandas as pd` and `expand_multiindex`.

**`.copy()` question settled by measurement, and my first rationale was wrong.** I had argued no
copy was needed because "downstream never writes in place". The real reason is better: on pandas
3.0.3 `np.asarray(df)` SHARES memory but is READ-ONLY —
`np.shares_memory(...) is True`, and a write raises
`ValueError: assignment destination is read-only`. So copy-on-write already protects the caller, and
an accidental write fails LOUDLY rather than corrupting — stronger than a defensive copy, which
would hide the bug. Consequence now recorded: `ft.arrays` members from DataFrame leaves are
read-only; any future task needing to write must copy there and say why.

**LESSON (the important one): a contract that lives only in an interface comment is not a
contract.** It needs (a) an implementation step that names the line to write, and (b) a test that
fails if it isn't. I wrote (a) as prose and skipped (b), which is precisely the defect class this
whole review series keeps finding in my work — stating a guarantee somewhere other than where it is
enforced. Also: verifying a rationale beats asserting one; the measurement replaced my reasoning
with a better answer.

## Design decisions Jeremy made

- `order='parallel'` default, `order='serial'` optional; `animate='serial'` stays a permanent alias.
- **`t` is in raw samples.**
- Forecast bounds: static data (incl. animated-static) precomputes all forecasts, fits the
  reduction on REAL DATA ONLY then applies it to forecasts, box from data+forecasts → no clamp.
  Streaming keeps the head-frozen box + clamp (`streaming.py:382-401`). Verified `plot.py:2803`
  already fits on real data only.
- **Plotly and matplotlib must behave identically**; serial via per-segment alpha like
  chemtrails/precog/bullettime.
- Forecast scoring stays OUT of the library (tutorial-side analysis).
- Column MultiIndex rule: innermost level = features, outer levels = grouping hierarchy.
- Market = the hierarchical showcase (sectors → market). Weather = the paper figure.
- Release 1.1; announcement waits for the whole line.

## Killed directions (do not re-litigate)

- **Climate as the hierarchical example — NOT RESCUABLE.** Smoothing tightens the loop but never
  lifts the drift: absolute drift flat (0.218–0.470) at every kernel while loop diameter collapses
  4.27 → 0.83; warming is 0.070% of variance vs the loop's 37.5%; decade centroids strictly
  monotone in 0/6 cities at every setting. Physical limit. See `climate_hierarchy_feasibility.md`.
- **Seasonal-profile weather shape** (rows=years, cols=(hemi,city,month)): gives per-city
  trajectories but collapses the seasonal loop into axes, so only the trend survives. Rejected.
- **CDC flu / Beijing air quality / PhysioNet gait** as the hierarchical example — superseded once
  the market took that role. Renders and rankings in `hierarchical_example_ideas.md`.

## Verified facts worth keeping

- Sector panel: 24/24 tickers via `query1.finance.yahoo.com/v8/finance/chart/<T>?range=10y`
  (User-Agent required), 2513 days 2016-07-28 → 2026-07-28, 6 sectors × 4, EQUAL widths (required
  by `plot.py:2745`). `yfinance` 1.5.1 installed.
- Weather paper figure is ~one native call: `hyp.plot(temps, hue=avg, palette='RdBu_r',
  normalize='across', manip='Smooth', animate=True, chemtrails=True, colorbar=True)` → 516 distinct
  colours + colorbar.
- Native text confirmed: `hyp.reduce(list_of_strings, ndims=3)` → (8,3); `hyp.plot(list_of_strings)`
  → Figure. Path: `vectorizer='<hf-id>', semantic=None, corpus=None` (`text2mat.py:89/184/391/404`).
- `save_path='x.gif'` replaces ~58 lines of ffmpeg boilerplate across 4 notebooks.
- `image_palette()` bug reproduced: 90%-beige/10%-red synthetic → returns beige `[0.784,0.769,0.737]`;
  ordering by population × chroma returns red.
- pandas **3.0.3** installed, `<3` ceiling lifted → `df.groupby(axis=1)` raises TypeError (REMOVED,
  not deprecated). Use `df.T.groupby(level=..., sort=False)` + transpose back.
- Baseline: **2564 collected, 2551 passed, 13 skipped**. Use `.venv/bin/python` — base anaconda is
  broken (numpy/matplotlib mismatch).

## OPEN — needs Jeremy before execution

**Superseded as of round 9 (2026-07-30). This section previously listed eleven decisions and two
claims that later turned out wrong; corrected below. Do not resurrect the old text.**

Eight of the eleven are **resolved** — all eight are recorded in the README's *Standing decisions*.
The two that had been called out as most important both closed, and one of them closed because the
claim behind it was false:

1. ~~**`morph_samples` above threshold: raise vs cap-with-warning**~~ → **RESOLVED**: new public
   `simplify=` flag. No-op below the tractability cap; above it, `simplify=True` (default) silently
   downsamples with *no* warning, `simplify=False` raises and names `simplify=True` in the message.
   `morph.py:17-24`'s no-point-dropped guarantee becomes conditional and its docstring must be
   updated as part of Plan 1 (Contract 7).
2. ~~**`on_frame=` on plotly is genuinely unreachable**~~ → **THIS CLAIM WAS WRONG.** plotly has a
   build-time Python frame loop: `_add_animation` at `plotly_backend.py:2517`, appending
   `go.Frame(**frame_kwargs)` at **four** sites — :2729, :2819, :2865, **:2975** (re-verified
   2026-07-30; earlier notes in this file listed only the first three). What it lacks is a
   *playback* loop, which is a different thing. `on_frame` is therefore reachable on both backends
   as a purity contract, the `NotImplementedError` is **deleted rather than relocated**, and
   `test_on_frame_output_parity_across_backends(style, order)` pins it. The parity directive has no
   exception in 1.1.

   Separately, streaming is matplotlib-only regardless, but **not** by a backend-specific warning.
   `streaming.py:263-265` is only the *docstring* describing the behavior. The real mechanism is a
   signature-driven aggregate check in `plot()` at `plot.py:2551-2581`: `_stream_forwarded` is the
   allow-list of parameters streaming honors, every other formal parameter whose value differs from
   its default is collected into `_stream_dropped` (plus all of `**kwargs`), and one `UserWarning`
   names the whole set. `backend` **is** a formal `plot()` parameter and is **not** in
   `_stream_forwarded` (both verified 2026-07-30), so a `backend=` request is reported through that
   general path — there is no code anywhere that warns about `backend` and streaming specifically.

**Seven decisions genuinely remain**, none in Plans 1 or 2 — four in Plan 3 (silent forecast drop
under `hue=`/`cluster=`; throttling beyond memoization; the `min_history` degenerate stub;
finished-dataset forecast under `order='serial'`) and three in Plan 4 (where `image_palette` is
exported; the paintings outlier trim; the morph example's `normalize()` helper). They are listed by
**name, deliberately unnumbered**, in the README — cite them by name, never by number. Four rounds
of citation drift in this plan set all traced to numbered references going stale under renumbering.

Both earlier-session items are **closed, committed, not pushed**: the `pyproject` 1.0.0 → 1.0.1 bump
shipped with the antialias work in `74c50b39`, and the notebooks/examples question resolved as
*commit them* in `4d1d2223` (5 `examples/animate_*.py`, 5 new notebooks, 14 re-executed tutorials).
The Bluesky launch clips are **not** committed — `notes/bluesky-launch/` is gitignored and
`git check-ignore` confirms `POST.md` is ignored; tracked files under that path = 0.

---

## Round 9 (2026-07-30) — Plan 1 review; animation-core v3 → v4

Four maintainer blockers, **all four confirmed by reproduction** before any edit. Three further
defects surfaced from the same audit that the review did not name.

| # | blocker | verification |
|-|-|-|
| B1 | no animation guide is planned | The ONLY `docs/` string in the whole plan was `git add CHANGELOG.md docs/ examples/` — staging a directory no step wrote to. Task 9 was *titled* "CHANGELOG, docs, and example cleanup" with zero doc steps. Self-contradictory in the other direction too: Step 4's 0-warning gate means an **unlinked** new `.rst` would have FAILED the build |
| B2 | callback contract misstated as "purity" | Confirmed in adjacent lines: the `plot()` docstring said *"must be a pure function"* and the example on the NEXT line called `ctx.axes.set_title(...)` |
| B3 | `FrameContext` public but unexported | Confirmed. Also found the enforcement: `tests/test_codeorg_licensing_audit_fixes.py:295-300` is a **hardcoded literal set**, so `__all__` and that literal must change atomically or the suite goes red |
| B4 | Plan 1 / Plan 4 example ownership | Confirmed and **mutual** — Plan 4 has a task per example (T2/T3/T5/T6) over the same four files, 52 mentions |

**Found by the audit, not the review:**

- **E1 — every 1.1 plan's docs-verification step could not run.** All six `cd docs && make clean &&
  make html` fail: `make` runs the `sphinx-build` **console script**, whose `sys.path[0]` is the
  venv `bin/`, so `docs/conf.py:367`'s `from _gallery_log_filter import install` raises
  `ModuleNotFoundError`. Reproduced twice. `python -m sphinx` works because `-m` puts the CWD
  (`docs/`) on `sys.path`. Worse: `make html` omits `-W`, so it never enforced the zero-warning
  gate it claimed to. **CI truth** (`.github/workflows/test.yml:283-291`):
  `cd docs && python -m sphinx -b html -W -E -a . _build/html`. All six replaced across all 4 plans.
- **E2** — animation-core's *Decisions* list was numbered with 3 numeric back-references. De-numbered.
- **E3** — Plan 4 still stated the pre-`simplify=` morph behaviour ("raise"); default now caps silently.

**The sixth citation-drift instance, and the worst one.** The README's own drift log said Plan 2's
`"Decisions still open #5"` citation was *"now cited by name"*. It wasn't — a second occurrence
survived at `2026-07-28-hypertools-1.1-multiindex.md:3565` for eleven days, carrying the same wrong
number (the drop was #9). **Lesson recorded in the README: a partial fix plus a note claiming
"fixed" is worse than no fix, because the note stops anyone re-checking.** Verify a citation fix by
grepping the *pattern* across every plan, never by fixing the instance you were shown.

**Two of my own claims were wrong and were caught by verifying them:**
1. I wrote "zero of the four examples accumulate — no `+=`" into the plan. A grep found `+=` in two.
   Both turned out benign — `animate_market_forecast.py:255` is **module-level precompute** building
   an `ACC` array (wrapper at `:323` only does `ACC[min(num, total-1)]`), and
   `animate_conversation.py:254` is a loop-local in a deterministic helper. Claim narrowed to "inside
   the per-frame wrapper" and the **precompute-then-index** idiom is now taught in the guide.
2. I wrote a test using `ani = hyp.plot(...)`. `HyperAnimation` is a `(figure, animation)` **tuple
   subclass** (`hyper_animation.py:45`), so that binds the tuple. Fixed to `fig, ani = ...`.

**Arithmetic:** Task 7 24 → **27**, Task 9 0 → **15**. Total 102 → **120**; final **2,671 passed /
13 skipped** vs the 2,551 baseline. Checkpoints from Task 7 on: 2643, 2656, 2671.

**Also this round (maintainer request):** the "welcome" animation. The **README was already**
`story_trajectories.gif` (`README.md:10`); only `docs/index.rst:9` still used `hypertools.gif`.
Switched, with alt text describing the hyperaligned story trajectories.
`plot_story_trajectories.py` is the storytelling-hyperalignment demo (36 subjects, Simony et al.
2016 "PieMan"). `hypertools.gif` is now referenced only by generator scripts; left in tree.

## Round 10 (2026-07-30) — "nothing is out of scope": findings I had dismissed

Jeremy, CRITICAL: *"there's nothing out of scope — ALL findings must be addressed and treated as
relevant. failed checks mean: dispatch subagents to diagnose and fix. incidental findings mean:
dispatch subagents to diagnose and fix."* Saved to memory as `nothing-is-out-of-scope`. I had
dismissed three items in one session; every one is now diagnosed.

| finding | how I dismissed it | resolution |
|-|-|-|
| 2 background tasks died, exit 144, 0-byte output (`bku8h5xu7` "Block until final test suite completes", `b4ibj9igv` "Wait for tests and paintings re-render") | *"aren't from this work"* — they predate a context compaction | subagent dispatched to trace what they gated and fix anything half-finished, esp. `animate_painting_embeddings.py` + its notebook |
| `docs/sg_execution_times.rst` stale — tracked, says "53 files", repo has **58** examples | *"build noise, CI regenerates it"* | **UNTRACKED + gitignored.** Correct per the repo's own rule: `docs/auto_examples/` is already ignored (`.gitignore:35`) and CI asserts generated gallery output must not be tracked *because "its presence would MASK execution failures"* (`test.yml:268-272`). Verified safe: **no** `.rst` or `conf.py` references it, so no toctree breakage |
| `images/hypertools.gif` orphaned | *"harmless, left in tree"* | **KEEP — on evidence.** It is the documented OUTPUT TARGET of `scripts/round17_evidence/readme_media.py:63` and the stated visual reference in `generate_story_trajectories.py:48,85` and `generate_weights_trajectory.py:2`. Deleting it would dangle a generator's contract |

**Widened by the scan (the original framing was too narrow).** `hypertools.gif` was not the only
orphan — a full-repo reference scan found **six** unreferenced images, and it is the *least*
orphaned of them. These five have **zero references anywhere in the repo**:
`demo_density.png`, `demo_multicolored.png`, `demo_plotly.png`, `demo_predict.png`, `hypercube.pdf`.
**Not deleted, and deliberately so:** this repo serves `images/` over
`raw.githubusercontent.com/...` URLs (README does exactly that), so third parties — old issues, PRs,
posts — may hot-link them, and deletion breaks those silently and invisibly to us. Needs Jeremy's
call, since only he knows whether they are dead. Recorded rather than left ambiguous.

**A fourth finding, self-caught: the PostToolUse verifier cries wolf.** The
*"Edit operation failed"* / *"Command failed"* messages I had been calling spurious are real, with a
precise root cause: `~/.claude/plugins/.../oh-my-claudecode/4.2.15/scripts/post-tool-verifier.mjs`
decides failure by **substring-matching the tool's output text** — `detectWriteFailure` (`:209-220`)
tests `/error/i` and `/failed/i`, `detectBashFailure` (`:138-152`) adds `/cannot/i`, `/abort/i`.
No exit code, no error flag. So an edit whose *content* discusses failure modes ("raises
`NameError`") is reported as a failed edit, and a successful `grep` printing text containing "fail"
becomes "Command failed". **This is the same pathology Jeremy just corrected in me:** a check that
always cries wolf trains the reader to ignore it, so a genuine failure passes unnoticed. Installed
OMC is 4.2.15; the local marketplace copy is byte-identical, so there is no newer local copy. Fixes
both touch Jeremy's GLOBAL config (`omc update` to 4.15.7, or override the hook), so they need his
call — patching the plugin cache in place would be erased by the next update.

### Round 10b — what the "nothing is out of scope" sweep actually turned up

Chasing the dismissed findings surfaced **three real defects**, two of them mine:

1. **I committed five UNEXECUTED tutorials** (`4d1d2223`). `docs/conf.py:115` sets
   `nbsphinx_execute = 'never'`, so Read the Docs does **not** run notebooks — committed outputs are
   the only thing that renders. Measured across all 20 tutorials: 15 carry executed outputs; the
   only 5 that do not are exactly the 5 I added. On RTD they would have shown bare code with no
   figures and no animations. This is what the dead background task *"Wait for tests and paintings
   re-render"* was gating.
2. **`tests/test_plot_audit_b2.py::test_nonfont_file_rejected_at_resolve_time` passed for the wrong
   reason everywhere except this laptop.** It passed the hardcoded path
   `/Users/jmanning/hypertools/README.md` to `resolve_font`, which branches on `os.path.exists`
   (`fonts.py:401-406`): where the file exists you get *"exists but is not a loadable font file"*
   (the branch the test names); where it does not you get *"is not a recognized installed font"* — a
   different branch. `match='font='` matched **both**, so on CI the test never exercised its own
   subject. Rewritten to use `tmp_path` and to assert the specific message; verified it now FAILS
   against a non-existent path, which the old form did not.
3. **Personal-path leakage is repo-wide, not new.** `/Users/jmanning` appears in **23 tracked
   files**. Scrubbed the 4 pre-existing tutorials (`plot`, `projectile_kalman` ×3,
   `streaming_data`, `conversation_trajectories`) — output cells only, source cells were clean,
   6 → 0, byte-level replace so the diff is 6 lines and nbformat is untouched. The 13 plan/spec docs
   and 7 notes files still carry it; cosmetic only (the username is already public via git
   authorship on a public repo), listed here so it is tracked rather than forgotten.

**Two process lessons, both about checks that report success they did not verify:**
- I wrote `git add ...` and `git commit ...` as SEPARATE statements, so a failed `git add` did not
  stop the commit — producing `2f26b711`, whose message described a `.gitignore` change the commit
  did not contain. Amended to `6baee557`. **Always `&&`-chain staging to committing.**
- A background-task notification reported **"exit code 0"** for a `make html` that actually died
  with `make: *** [html] Error 2`. The trailing `| tail` meant the reported status was the pipe's,
  not the command's. **Never trust a notification's exit code for a piped command.**
