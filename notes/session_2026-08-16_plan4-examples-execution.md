# Plan 4 (examples and tutorials) — execution record

Started 2026-08-16, on `dev-1.0`, after the maintainer's review round 7 directed
"begin Plan 4". Plan: `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md` (v4).

**Procedure chosen, explicitly** (the plan offers two and forbids half-doing
both): **Task 8 Steps 0–2 FIRST**, so Tasks 1–7 have somewhere to record their
measurements, and Task 8 Step 3's reds are the to-do list. The alternative —
Task 8 last — was rejected because every task's measure step writes into
`SCRIPT_BUDGETS` / `EXPECTED_VISIBLE_OUTPUTS`, which do not exist until Step 2.

## Order of work

| # | what | state |
|-|-|-|
| T8 S0 | `HyperAnimation.n_frames` / `.n_segments` / `.draw_frame` | **DONE** |
| T8 S0b | loader/builder split contract (text; Tasks 2–6 apply it) | read, not code |
| T8 S0c | budget renegotiation | already done 2026-08-04; verified, not repeated |
| T8 S1 | `scripts/measure_native_ratio.py` | **DONE** |
| T8 S2 | `tests/test_examples_are_native.py` (the gate) | **DONE — deliberately RED** |
| T1 | native palette-from-image | **DONE** |
| T2 | Market — the MultiIndex showcase | **IN PROGRESS** — library work done and committed; the artifact itself is NOT settled (see below) |
| T3–T7 | weather, paintings, conversation, morph, 15 older tutorials | pending |
| T8 S3–S7 | run the gate, re-measure, headless run, notebook figures, full suite | pending |

## Status, 2026-08-18/19 (maintainer review round 8)

Task 2 is partly executed. Everything that is a LIBRARY fix has landed as its
own commit; the Market artifact deliberately has not, because the evidence now
says its forecast framing is wrong and the representation is still under review.

| commit | unit |
|-|-|
| `3fe1c6a5` | tag animated multicolor collections with their trace |
| `6da372b4` | matrix hue through a column hierarchy, contract pinned + documented |
| `9615c382` | forecast projection sampled at two DISTINCT history lengths |
| `9d630c9f` | study: horizon-aware window baseline; rule applied in code |
| `fd46b7db` | correct the stale Market budget claim (gate stays red) |

**Deliberately uncommitted:** `examples/animate_market_forecast.py`,
`docs/tutorials/market_forecast.{ipynb,gif}`, `scripts/execute_tutorial.py`.
The staged GIF/notebook publish an artifact whose own supporting study now
says it should carry no forecast claim; committing them would ship two
contradictory conclusions in one tree.

**The editorial decision, and the evidence for it.** Applied in code, the
preregistered rule passes three specifications, all `drawdown` at h=1 — so
the earlier flat claim that "nothing passes" was too broad and is retracted
in the study notes. But the drawdown audit those notes pre-committed to
kills all three: a parameter-free "predict full recovery" rule beats the
models in 10 of 12 cells. **No specification earns a forecast claim at the
horizon the example draws.** Market should become the hierarchy showcase
with no forecast-skill claim; the prediction story moves to data with real
temporal structure. Awaiting the maintainer's sign-off before the rewrite.

## Review round 9 (2026-08-19)

| commit | unit |
|-|-|
| `d41fc265` | explicit `import numpy as np` in plot.py — ruff 186 → 28, zero net-new keys |
| `dfc8e7db` | `hue_mode=` — mixture vs RGB said outright instead of inferred from width |
| `1d220643` | acceptance rule hardened (positive + exact block set) + study made self-contained |

**`hue_mode=` exists because the previous commit broke its own use case.**
Mixture weights are one palette colour per column, one column per leaf —
but the routing rule sent any matrix wider than 3 columns to the RGB
reducer, so at most three leaves could ever get a primary. Measured on the
Market candidate (6 leaves, 4-colour palette): `palette=` was ignored and
leaves intended blue drew pure red. The width rule is now the DEFAULT, not
the only rule; flipping the default would silently repaint existing figures.

**The Market artifact was discarded**, not committed. Its script, notebook
and GIF were copied to a SESSION-SCOPED scratchpad —
`/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-.../scratchpad/discarded_market_artifact/`
— which is outside the repository and does not survive the session. Treat
the artifact as GONE: it is not recoverable from this note, and nothing
depends on recovering it (its representation was rejected, so what matters
is the measurements, which are in
`notes/market_representation_study_2026-08-17.md`). The earlier version of
this paragraph wrote that path relative, as `scratchpad/…`, which resolves
to nothing from the repo root and read as a promise of recoverability the
tree could not keep.

The prototypes ARE preserved, because a visual decision cannot be reviewed
from prose: `notes/evidence/plan4-market-prototypes/` holds both PNGs and
the script that makes them (3 s, live data with a synthetic fallback, and
it re-asserts the shared-normalization property on the way past).

Discarding the artifact broke the committed study, which imported the
example's universe and fetcher — fixed by lifting both in, because evidence
must not depend on the artifact it judges.

**Two library limits measured while prototyping the replacement**, both
relevant to any 2-D Market composition:

* `hyp.plot` draws in a NORMALIZED unit box (`xlim == [-1.1, 1.1]` whatever
  the data) and removes ticks by design. `xlabel=`/`ylabel=` are native and
  do render, so an axis can say what it IS but never what it equals.
* each call normalizes ITS OWN inputs, so N calls give N private scales.
  Small multiples are only comparable if every panel is passed the same
  data and differentiated by hue — verified: all six panels then share one
  limit tuple exactly.

**Prototype verdict:** the single fixed 2-D panel still tangles and its
labels collide (`PROTO_A2_single_panel.png`). The small multiples are
legible (`PROTO_B2_small_multiples.png`) and are the composition to take
forward, pending the maintainer's approval. No GIF was rendered.

## Review round 10 (2026-08-19) — RESUME HERE

The maintainer's round 10 found no code defects. Both High findings were
about **process integrity**: the plan, the gate, the recorded measurements
and the actual files had drifted apart when the Market rewrite was
discarded after its measurements had already been written down. Both were
verified by measurement and corrected in this session:

| finding | verified as | disposition |
|-|-|-|
| recorded gate state ≠ repository gate state | **confirmed exactly** — 191/145 and 187/150, not the recorded 148 | budget comment corrected; `EXPECTED_VISIBLE_OUTPUTS['market_forecast']` REMOVED (it described a discarded notebook and was failing against `{2, 4, 5, 6}`); reds table re-measured at 42 |
| scratchpad evidence unavailable | **half right, and the half that matters is right** — the assets exist, but at an absolute session-scoped path outside the repo, which no reader can reach | prototypes + generator now TRACKED at `notes/evidence/plan4-market-prototypes/`; the discarded artifact is declared gone rather than implied recoverable |

### What is NOT done — the resume list

The maintainer's sequence, steps 1–3 complete:

1. ~~correct the session record's Market measurements and red-count~~ **done**
2. ~~correct the gate so it describes HEAD~~ **done**
3. ~~recover and track the prototype PNGs + generation script~~ **done**
4. **Formally revise Plan 4's Market and prediction requirements** — NEXT.
   The plan still prescribes a Market forecast story. This note recording
   that the study rejected it is an execution-note deviation, not a revised
   specification; without the revision, Task 2 could be called complete
   while failing the plan that defines completion. The revision must state:
   the forecast claim was rejected by the preregistered study plus the
   post-hoc recovery audit; the replacement's acceptance criteria; WHICH
   example carries the prediction story instead (maintainer's suggestions:
   seasonal weather, motion, sensor data); and any budget change if the new
   composition changes the loader/builder split.
5. review `PROTO_B2_small_multiples.png` at documentation display width —
   not full-resolution PNG size, which is not the size that decides it
6. approve a static composition BEFORE building any animation
7. implement the new Market script, no scoring claims
8. generate + execute the notebook, then record its REAL output cells in
   `EXPECTED_VISIBLE_OUTPUTS` (it is empty again, deliberately)
9. re-measure budgets and update the reds table ONCE, after the design settles
10. Tasks 3–7, then the publication and full-suite gates

Still open from round 9's sequence: harden `scripts/execute_tutorial.py`
(argparse, create the output directory, reject output collisions) — it is
currently DELETED from the tree and lives only in the session scratchpad,
so it must be rewritten, not restored.

### Checks that were re-run after these edits

`tests/test_examples_are_native.py` (42 red, unchanged and every red
accounted for above), the sdist tracked-files guard (5 passed — the newly
tracked evidence files are the thing that guard exists to catch), and
`ruff check` on both touched files, clean.

## The reds: **42** (re-measured 2026-08-19)

`tests/test_examples_are_native.py` collects **139**: **92 pass, 42 fail,
5 skip**.

**This supersedes a "39 → 31" table that was wrong in both directions.**
Those 31 were counted against the Market rewrite, which was then discarded;
the tree went back to the un-rewritten example and notebook, and the count
went back UP. It landed at 42 rather than the original 39 because two of
the gate's own records had been updated to describe the rewrite and were
not reverted with it — the budget comment (145 "measured", then 148) and
`EXPECTED_VISIBLE_OUTPUTS['market_forecast'] = {3, 5, 6, 7, 8}`. The second
is the one that mattered: the committed notebook emits on `{2, 4, 5, 6}`,
so that test failed by comparing HEAD against a file that no longer
existed. Both are corrected; the market entry is REMOVED rather than
re-measured, since the committed notebook is itself due for replacement and
re-measuring would only pin a second doomed artifact.

| test | n | note |
|-|-|-|
| `test_no_defect_marker_in_the_launch_examples` | 18 | Tasks 2–7; market contributes 6 again now that the un-rewritten file is back |
| `test_file_is_within_its_size_budget` | 7 | market **191/145**, weather 195/75, paintings 146/140, conversation 165/110, market.ipynb **187/150**, weather.ipynb 194/80, conversation_shape.ipynb 176/115 |
| `test_the_right_cells_carry_visible_output` | 5 | `EXPECTED_VISIBLE_OUTPUTS` is empty again — all five say "no measured index set recorded" |
| `test_examples_produce_their_stated_artifact` | 5 | Step 0b's split, Tasks 2–6 |
| `test_older_tutorials_dropped_their_hand_rolled_helpers` | 4 | Task 7 |
| `test_analyze_tutorial_actually_plots`, `test_reduce_tutorial_mentions_describe` | 2 | Task 7 |
| `test_every_allowlisted_reach_is_still_present_and_still_explained` | 1 | conversation's `ani._args`; deliberate, Task 4's to-do — NOT to be silenced by re-adding a dead entry |

Every one of the 42 is an un-rewritten example. **Nothing in this table is
Task 2's own any more**: with the rewrite discarded, Task 2 has produced
library commits and evidence, and no artifact. Regenerate with
`.venv/bin/python -m pytest tests/test_examples_are_native.py -q --tb=no -rf`.

## The original 39 deliberate reds (historical — superseded by the table above)

`tests/test_examples_are_native.py` collects **139**: 93 pass, **39 fail**, 7 skip
(2 `PRIVATE_API_EXCEPTIONS` + 5 opt-in smoke). Every failure is an un-rewritten
example, i.e. the to-do list Tasks 1–7 burn down. Recorded here so a NEW failure
is distinguishable from an expected one — a red suite that hides a regression is
worse than no gate:

| test | n | why |
|-|-|-|
| `test_file_is_within_its_size_budget` | 7 | market 191/145, weather 195/75, paintings 146/140, conversation 165/110, market.ipynb 187/150, weather.ipynb 194/80, conversation_shape.ipynb 176/115 |
| `test_no_defect_marker_in_the_launch_examples` | 16 | `ani._func`×5, `SentenceTransformer`×4, `antialias_line`×2, `ani._args`×2, `hypertools._shared`, `from hypertools.plot import morph`, `morph_schedule` |
| `test_examples_produce_their_stated_artifact` | 5 | no `__main__` guard yet — Step 0b's split is Tasks 2–6's work |
| `test_the_right_cells_carry_visible_output` | 5 | `EXPECTED_VISIBLE_OUTPUTS` ships empty by design |
| `test_older_tutorials_dropped_their_hand_rolled_helpers` | 4 | Task 7 |
| `test_analyze_tutorial_actually_plots`, `test_reduce_tutorial_mentions_describe` | 2 | Task 7 |

The exact node-id list is in the commit that added them; regenerate with
`.venv/bin/python -m pytest tests/test_examples_are_native.py -q | grep ^FAILED`.

## Deviations from the plan's prescribed code, and why

1. **`measure_native_ratio.py` used a bare `open(path).read()`** in both
   `_code_lines_py` and `_code_lines_nb`. The gate measures 30 files per run, so
   under `-W error` that is a `ResourceWarning` per file and the budget checks
   become errors; on a non-refcounting interpreter it is a real descriptor leak.
   Both are now `with` blocks. The measurement is unchanged — market still reads
   `code=191 native=11 ratio=5.8%`, reproducing the plan's BEFORE table exactly.
2. **Two unused top-level imports** in the plan's Step 2 block (`importlib.util`,
   `json`) — both are re-imported inside the function bodies that use them, so
   kept verbatim they are new ruff `F401`s. Removed; no body touched.
3. **Every `plot.py`/`colors.py` line number in Task 1 was stale**, as the plan
   itself predicts (it is sequenced after Plans 1–3). Located by symbol instead.
   The plan's *content* claims all held, including "five raw-seaborn call sites,
   one of them `sns.set_palette`".

## Verified, not assumed

- `HyperAnimation` accessors: 9 tests, and **both** morph tag sites are
  mutation-verified — deleting the 2-D tag alone fails
  `test_n_segments_is_set_for_a_2d_morph_too`, which is exactly the
  silently-`None`-for-half-of-morphs failure the plan warns about.
- Task 1's largest-cluster regression really reproduces: restoring the old
  `score = frac` ordering fails 5 tests, and `pal[0]` comes back as the beige
  background `(0.784, 0.769, 0.737)` instead of the vivid `(0.863, 0.078, 0.078)`.
- Live market data is reachable for Task 2: **2514 sessions, 2016-08-15 →
  2026-08-14**, `adjclose` present — matching the plan's 2026-08-15 measurement
  (the `range=10y` window slides, so this count is run-date dependent by design).

## Full-suite state at the checkpoint

Before the gate landed: **3589 passed, 13 skipped, 2 deselected, 1 failed** — the
one failure was `test_sdist_contains_only_tracked_files_plus_allowlist`, caused by
the new test files being untracked at collection time. It passes once they are
`git add`ed, which is the guard working as designed.
