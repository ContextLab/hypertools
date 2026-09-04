# Plan citation sweep — 2026-08-01

Precision pass over stale citations in three 1.1 plan documents, dispatched against
`dev-1.0` HEAD `eaf5b23d`. Every correction below was verified against the real file
before being applied (not copied from the prior audits, which were themselves stale
after commit `4877287c` moved code and shifted line numbers). Commit: `fcba79ba`.

Files:
- A = `docs/superpowers/plans/2026-07-27-hypertools-1.1-forecast-animation.md`
- B = `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md`
- C = `docs/superpowers/plans/2026-07-28-hypertools-1.1-multiindex.md`

Audits read first: `notes/audit/review_plan3_v2_recheck.md`,
`notes/audit/review_plan4_examples_and_tutorials.md`.

---

## In A (forecast-animation)

**1. Task 1 "6 named" tests — FIXED.**
Counted `tests/plot/test_forecast_core.py`'s Step 1 block (A:373-447): 7 `def test_`
functions, one of them (`test_displacement_is_anchored_on_the_last_observation`)
parametrized over 2 models. 6 non-parametrized + 2 parametrized instances = 8
collected, matching the "8 passed" total already stated. Fixed "6 named" → "7 named"
at A:537.

**2. Task 5 Step 2 "10 items" — FIXED.**
Counted `tests/plot/test_forecast_trail.py`'s Step 1 block (A:1658-1798): 9 `def test_`
functions, one (`test_invalid_forecast_trail_raises`) parametrized over 3 values. 8×1 +
1×3 = 11 collected, matching Step 5's own "11 passed (8 named tests + one parametrized
over 3 values)". Fixed "10 items" → "11 items" at A:1803.

**3. Self-Review "T4 wrong pass counts" row — FIXED.**
Re-derived every figure in the row against each task's own re-derived Step count:
- Task 1 → 8: already correct (Task 1 Step 4: "8 passed").
- Task 2 → 12: WRONG. Task 2 Step 4 (A:939) explicitly says "**14 passed**" and explains
  why (v3 added `test_display_paths_are_displacements_not_positions`, the 14th test, on
  top of v2's 13). Counted the Step 1 block directly: 14 `def test_` functions, none
  parametrized. Fixed 12 → 14.
- Task 3 → 9 (+17 in the edited integration file): the "+17" was WRONG — Task 3 Step 7
  (A:1150-1161) already recomputes this to **15** in its own table (18 measured − 5
  removed + 2 added = 15) — this is the exact fix the v2-recheck audit's High-severity
  finding asked for, already applied at Step 7 but never propagated to this summary row.
  Fixed +17 → +15.
- Task 4 → 21 cumulative: WRONG. Task 4 Step 6 (A:1599) says "**27 passed** (9 from Task
  3 + 18 here)"; counted the Step 1 block: 16 `def test_` functions, 2 parametrized (×2
  each) = 14×1 + 2×2 = 18, confirming 9+18=27. Fixed 21 → 27.
- Task 5 → 11: already correct (Task 5 Step 5: "11 passed").
- Task 6 → 8: WRONG — this is the OLD v2 figure. Task 6 Step 6 (A:2128) says "**18
  passed** — 6 unparametrized tests + 3 parametrized over the 4 entries of STYLES", and
  explicitly notes "v2 expected 8 here, when every test ran animate=True only." Counted
  the block: 9 `def test_` functions, 3 parametrized over 4 STYLES entries each = 6×1 +
  3×4 = 18. Fixed 8 → 18.
- Task 7 → 25 cumulative: WRONG (also the old v2 figure: 9+12+4=25). Task 7 Step 4
  (A:2242) says "**31 passed** (9 from Task 3 + 18 from Task 4 + 4 here)"; counted 4 new
  `def test_` functions in Task 7's own block, none parametrized. Fixed 25 → 31.

**4. Task 4 "Files:" list — FIXED.**
Task 4 (A:1178-1642) prescribes code edits only in `plot.py` (Steps 3, 4, 5 are all
`plot.py:NNNN`-cited insertions, including the live-forecast-artist creation and
`_update_forecasts` closure). No code block in Task 4 touches `matplotlib_backend.py`.
Removed the stale "Modify: `hypertools/plot/matplotlib_backend.py` (the live forecast
artists)" bullet and reworded the `plot.py` bullet to name what it actually covers.

**5. Task 3 Step 2 "morph tests FAIL" claim — FIXED.**
Extracted Task 3 Step 1's full 9-test block verbatim and ran it against HEAD
(`.venv/bin/python -m pytest`): **7 failed, 2 passed**. The 2 passes are exactly
`test_scalar_morph_still_refuses_predict` and `test_list_form_morph_still_refuses_predict`
— both already pass today because the shipped refusal message interpolates
`animate={animate!r}`, which already contains the substring "morph" for both
`animate='morph'` and `animate=['morph','morph']`, so `pytest.raises(..., match='morph')`
matches without any code change. Only the two static/spin tests fail today, and only on
the missing `_hyp_forecast_role` tag. Rewrote A:1061 to state this precisely, naming
which tests pass/fail and why, with the measured "7 failed, 2 passed" count.

**6. Full `hypertools/plot/*.py` citation sweep — FIXED (51 of 53 wrong).**
Extracted every `file.py:NNN` citation into `plot.py`, `matplotlib_backend.py`,
`plotly_backend.py`, `trails.py` — 53 unique citation values across 96 total occurrences
(including markdown-backtick-wrapped prose, plain-text references inside Python
docstrings/comments embedded in prescribed code blocks, `hypertools/plot/plot.py:NNN`
full-path citations, and 4 bare `` `:NNNN` `` shorthand continuations of an adjacent
citation, plus one bare "line NNNN" prose reference). Each was checked by opening the
real file at the cited location and confirming/deriving the line number that actually
matches what the prose describes (by function/statement text search, not a constant
offset — drift ranged from 0 lines to ~590 lines depending on where in the file the
content lives, since the plan predates a long chain of commits, not just `4877287c`).

Result: **51 wrong, 2 already correct** (`matplotlib_backend.py:1185` and
`plotly_backend.py:465` both still land exactly on the cited statement).

Every plot.py citation was wrong (37 of 37 unique values); 11 of 12 plotly_backend.py
citations were wrong; 1 of 2 matplotlib_backend.py citations (the other, `:1185`, was
already correct); both trails.py citations were wrong. Representative corrections (full
list in the commit diff):
- The `predict=`+animate refusal: `plot.py:2338-2354`/`:2346-2354` → `:2740-2756`.
- `_resolve_animate_mode` call: `plot.py:3653` → `:4158` (and "~1300 lines after" the
  refusal → "~1400 lines", since 4158−2740=1418).
- `_draw_forecast_overlays`: `plot.py:122-165`/`:140-165` → `:137-180` (function
  unchanged in content/length, just relocated).
- The hue/cluster-regroup guard nulling `raw_forecasts`: `plot.py:3999` → `:4552`.
- The centre/scale block `DisplayTransform` reproduces: `plot.py:4018-4031` → `:4569-4582`.
- `plotly_draw` call site: `plot.py:4181-4230` → `:4771-4813`.
- Static overlay draw+gate: `plot.py:4339`/`:4339-4341`/`:4339-4350` → `:4907`/
  `:4907-4909`/`:4907-4918` (also fixed a bare prose "Task 3 gates line 4339" → "4907").
- `_add_animation` signature: `plotly_backend.py:2517-2529` → `:2580-2593`.
- `trails.anim_window_bounds`: `trails.py:24-89` → `:24-94`; `trails.py:80-81` → `:85-86`.
- The two paired bare shorthand citations (`` `1888-1890` `` beside
  `matplotlib_backend.py:1785`, and `` `:1955` `` beside `plot.py:1920-1941`) were folded
  into single corrected citations (`matplotlib_backend.py:1941-1943`;
  `hypertools/plot/plot.py:2271-2295`) since both real targets turned out to be single
  contiguous blocks, not two separate locations.

---

## In B (examples-and-tutorials)

**7. "MultiIndex T4" for continuous hue, 4 places — FIXED (3 of the 4).**
Verified against Plan C's task headings: Task 6 is "Continuous hue as a per-trace
auxiliary value"; Task 4 is "The return-bundle contract (flat inputs)" — unrelated to
hue. Fixed the 3 standalone "MultiIndex T4" (hue) references:
- Prereqs Task 2 row (B:135): "Without MultiIndex T4 the hue is discarded" → T6.
- Self-Review Market row (B:2559 orig): "(MultiIndex T4 form 2)" → T6.
- Remaining risk #1 (B:2581 orig): "If MultiIndex T4 (`hue` through a hierarchy) slips" → T6.
The 4th "T4" occurrence is the one inside the "T1/T2/T3/T4/T6" list — that one is fixed
as item 8 below (it needed T5/T8 substituted, not a plain T4→T6 swap).

**8. Self-Review vs. Prereqs task-list mismatch — FIXED.**
Self-Review's "Prerequisites, per task" row (B:2556 orig) said "MultiIndex
T1/T2/T3/T4/T6"; the actual Prereqs table row (B:135) itemises T1 (`group_columns`), T2
(final-trace builder), T5 (column MultiIndex in `plot()`), T6 (hue as a per-trace
auxiliary value), T8 (`predict=` over final traces) — i.e. T1/T2/T5/T6/T8. Cross-checked
against Plan C's task list (T1=hierarchy.py, T2=final-trace builder, T5=column
MultiIndex in plot(), T6=continuous hue, T8=predict= over final traces): the detailed
Prereqs-table list is correct and internally consistent; the compressed Self-Review list
was wrong. Fixed "T1/T2/T3/T4/T6" → "T1/T2/T5/T6/T8".

**9. Baseline notebook table (~B:56-68) — COULD NOT VERIFY, left unchanged (flagged).**
Confirmed `scripts/measure_native_ratio.py` does **not** exist in the repo (`ls` →
No such file or directory) — it is created by this same plan's Task 8 Step 1, so it
cannot exist yet on a plan document. Per the task's own explicit fallback instruction
("otherwise report that you could not verify and leave them, flagging it"), the five
`.ipynb` baseline rows were left exactly as they were. Re-implementing the plan's
"logical-statement" native-ratio metric by hand (rather than via the plan's own
not-yet-written script) risked introducing a second, possibly-divergent measurement, so
it was not attempted. **Flag for whoever runs Task 8: re-measure these five rows with
`scripts/measure_native_ratio.py` once it exists, per the plan's Verification-note
methodology.**

**10. Tasks 3-6 "Execute and measure" cell counts — FIXED (4 of 4).**
Counted each cell table directly:
- Weather (B:1165-1176): code cells at rows 0,3,5,7,9 = 5 code cells; Expected said
  "4/4 code cells (cells 3,5,7,9)" — undercounted the denominator by omitting cell 0.
  Fixed to "4/5" and noted cell 0 (the Colab install cell) produces no output.
- Paintings (B:1397-1410): code cells at 0,3,5,7,9,11 = 6; Expected said "5/5". Fixed to "5/6".
- Conversation (B:1764-1777): code cells at 0,3,5,7,9,11 = 6; Expected said "5/5". Fixed to "5/6".
- Morph (B:1878-1889): code cells at 0,3,5,7,9 = 5; Expected said "4/4". Fixed to "4/5".
- Market (B:961-978): code cells at 0,3,5,7,9,11,13,15 = 8; Expected "7/8" — already
  correct, and is the convention the other four now match (cell 0 counted in the
  denominator, not expected to produce output itself).

**11. Task 7 Step 2 grep-gate dependency — FIXED (noted, not gated).**
Verified: `grep -l SentenceTransformer examples/*.py` on the **current, real**
`examples/` directory matches both `examples/animate_conversation.py` (line 92-93) and
`examples/animate_painting_embeddings.py` (line 104-105) today. Since Task 7 Step 2's
gate globs `examples/*.py` (not just the notebooks it directly touches), it cannot
report clean until Task 4 (Paintings) and Task 5 (Conversation) have rewritten those two
files to drop `SentenceTransformer` in favour of `vectorizer=`. Added an explicit
dependency note directly beside the grep command (B, after the `grep -l ...` code
fence) rather than changing the Prereqs table's "*(none)*"/parallel claim, which remains
true with respect to Plans 1-3 (the dependency is intra-plan, on this plan's own Tasks
4-5, not on Plans 1-3).

**12. "Decisions still needed" UNNUMBERED header vs. numbered items 4/5/6 — FIXED.**
The header (B:2511) explicitly states "**These entries are deliberately UNNUMBERED —
cite them by name**," with a rationale about citation drift under reordering — a
deliberate, well-justified design rule. Items 1-3 were already bullets; items 4
("The conversation caption"), 5 ("How the five launch tutorials get a visible figure"),
6 ("Whether the market example should report a disappointing number") were numbered,
contradicting the rule. Converted all three to bullets to match. (Self-Review's "Six
items in *Decisions still needed*" claim was checked and remains correct: 6 total.)

**13. Task 7 baseline "the five clean ones" naming seven — FIXED.**
B:1914 named 7 notebooks (`align`, `plot`, `normalize`, `cluster`, `streaming_data`,
`text`, `lsl_streaming`) after saying "the five clean ones." Cross-checked against the
File Structure table, which lists exactly 8 older tutorials Task 7 *does* touch
(`conversation_trajectories`, `hugging_face_embeddings`, `wikipedia_embeddings`,
`modern_sklearn_dynamics`, `stock_forecasting`, `projectile_kalman`, `analyze`,
`reduce`); 15 total − 8 touched = 7 untouched, confirming the *list* of 7 names is
correct and only the word "five" was wrong. Fixed "five" → "seven".

**14. Task 5 citations into `examples/animate_conversation.py` — FIXED (3 of 3).**
Opened the real file at the cited ranges:
- `mpatches.Patch` + `fig.legend` cited `:168-175` — the real `fig.legend(...)` call
  (using `mpatches.Patch` inline) is lines 173-176. Fixed to `:173-176`.
- `fig.text(...)` title cited `:176-177` — the real call is lines 177-178. Fixed to `:177-178`.
- The speaker text artist cited `:179-180` — line 179 is the explanatory comment; the
  actual `speaker = fig.text(...)` statement is lines 180-181. Fixed to `:180-181`.

**Bonus fix (verified, not one of the 16 items):** Task 1 Step 5's "Expected: 17 passed"
for `tests/plot/test_image_palette.py` — counted 16 real `def test_` functions in the
Step 1 block (none parametrized), matching this plan's own "Revision note (v2)" table,
which explicitly says Task 1 should read 16 (not 17). Fixed 17 → 16 at that one step.
Left the Self-Review's "Suite arithmetic" paragraph (still says Task 1 adds 17, Task 8
adds 109, total +126) **unmodified** — fixing it correctly requires re-deriving Task 8's
full breakdown, which was not in scope and not independently verified.

---

## In C (multiindex)

**15. `test_forecast_dropped_under_hue_regrouping` — FIXED.**
No such test exists in Plan A. The real name (Plan A Task 4, `test_hue_regrouping_drops_forecasts_exactly_like_the_static_path`,
confirmed at A:1449 pre-edit / unchanged by this session's Plan-A edits) was substituted
at C's one occurrence (Task 8 Step 3, item 4 of the numbered list).

**16. Plan-3 line citations `:289`/`:153`/`:389` — FIXED, RE-DERIVED (not copied from the audit).**
The prior audit's suggested fix (`:291`/`:155`/`:390`) was explicitly not used, per the
task's instruction, since Plan A has been heavily edited since that audit ran (including
this session's own item-6 sweep above). Re-derived against Plan A's real, final,
post-edit state:
- `forecast_from_history`'s `if len(history) < max(2, min_history): return None` check
  is at Plan A `:522` (not `:289`, not the audit's `:291`).
- `test_returns_none_below_min_history` is at Plan A `:386` (not `:153`, not `:155`).
- The "frame-0 test" is `test_early_frames_have_no_forecast` at Plan A `:621` — confirmed
  by its own docstring ("Frame 0 reveals 1 raw row; min_history=2 refuses to fit it.")
  and its `sched.path(0, 0) is None` assertion (not `:389`, not `:390`).
Both occurrences in C (the C3 review-table row, and the Task 8 short-history-mechanisms
prose paragraph) were fixed identically.

Note: C also independently cites `plot.py:3999` and `plot.py:4000-4002` directly (Task 8
Step 3, item 4) — these were **not** touched, since items 15-16 are C's complete assigned
scope and this plot.py citation is a separate, unrequested fix (for reference: per item
6's verified mapping, `plot.py:3999` is now `plot.py:4552` in the real file — flagged
here in case a future pass wants to extend the sweep to Plan C's own plot.py citations,
which were out of scope for this session).

---

## Verification

- `git diff --stat`: 3 files changed, 95 insertions(+), 94 deletions(-).
- Final grep sweep of every `(hypertools/plot/)?(plot|matplotlib_backend|plotly_backend|trails)\.py:[0-9]+(-[0-9]+)?`
  occurrence in Plan A confirmed zero remaining stale values (spot-checked all 29
  distinctive 4-digit old citation numbers individually — zero hits).
- Committed as `fcba79ba` on `dev-1.0`.

## Summary counts

- **FIXED:** 14 of 16 items (all except 9, which is COULD-NOT-VERIFY by design).
  Item 6 alone covers 51 individually-verified citation corrections across 96 occurrences.
- **ALREADY-CORRECT** (checked and left alone): Market's "7/8" cell count (part of
  item 10's verification); `matplotlib_backend.py:1185` and `plotly_backend.py:465`
  (part of item 6); the Self-Review "Six items" count in Plan B (checked while fixing
  item 12).
- **COULD-NOT-VERIFY:** 1 item (9 — the five notebook baseline rows; the plan's own
  measurement script does not exist yet).
- **Bonus fixes** (verified but outside the 16 assigned items): Plan A's Task 1 test
  count (17→16 passed) and a bare "line 4339" prose reference (→4907), both directly
  evidenced by work already done for the assigned items.
- **Flagged, not fixed** (explicitly out of scope): Plan B's "Suite arithmetic"
  paragraph (Task 8's 109/+126 breakdown was not independently re-derived); Plan C's own
  `plot.py:3999`/`:4000-4002` citations (item 6's fix was scoped to Plan A only).
