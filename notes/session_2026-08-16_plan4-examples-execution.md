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
| T2 | Market — the MultiIndex showcase | next; the release blocker |
| T3–T7 | weather, paintings, conversation, morph, 15 older tutorials | pending |
| T8 S3–S7 | run the gate, re-measure, headless run, notebook figures, full suite | pending |

## The 39 deliberate reds

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
