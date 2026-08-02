VERDICT: NOT IMPLEMENTABLE

# Plan 4 v2 (examples and tutorials) — adversarial re-review

Dispatched 2026-08-01 against `dev-1.0` HEAD `4ecb3b6d`.
Method, per instruction: **nothing was reasoned about — every prescribed block was extracted
from the plan by line range, parsed with `ast`, and executed.** Real `FrameContext`, real
`matplotlib.lines.Line2D`, real `hyp.plot` animations, real notebook execution via `nbclient`
with a kernel registered from this repo's venv, and Task 1's implementation applied in a
throwaway `git worktree` (removed; repo tree untouched).

v2 fixes one of its two Fatals outright. The other Fatal was *renamed*, not removed: v1 gated an
unsatisfiable **ratio floor**, v2 gates an unsatisfiable **line budget**. Four new Fatals surfaced
only on execution.

---

## Per prior finding

**Prior finding 1 (Fatal — Task 8 `BUDGETS` ratio floor unsatisfiable): PARTIAL.**
The ratio floor is genuinely gone (`grep` for `min_ratio|MIN_RATIO|ratio_floor|meets_its_native_ratio|FLOORS`
in the extracted `tests/test_examples_are_native.py` → **NONE**), and `test_native_ratio_is_reported`
correctly only reports. But the replacement gate `test_file_is_within_its_size_budget` is itself
unsatisfiable against the plan's own prescribed code — see **New F1/F2**. The failure mode is
identical to v1's: a threshold picked before the code existed, which the code cannot meet.

**Prior finding 2 (Fatal — `recency_fade` IndexError on `ctx.artists` vs `ctx.revealed_counts`): FIXED.**
Verified three independent ways.
1. The real backend ordering matches the plan's premise: `matplotlib_backend.py:1423` and `:2168`
   both build `artists=list(lines) + [t for t in trail_lines if t is not None]` alongside
   `revealed_counts=_counts` (`:1426`, `:2171`), and `_counts = serial_reveal_counts(lengths, ...)`
   (`:1343`, `:2125`) is one entry per dataset — the same `lengths` that indexes `lines`. Heads-then-trails
   is correct, and `len(lines) == len(_counts)`.
2. Exhaustive drive of the extracted `turn_alpha`+`recency_fade` over every
   `(n_datasets, current_index)` for n=1..29: **ALL OK** — no `IndexError`, and every artist assigned
   on every frame.
3. End-to-end on a real 28-turn `order='serial', chemtrails=True` animation with `recency_fade`
   registered as a real `on_frame` hook: 66 frames including repeats and out-of-order — **errors: NONE**.

   The `hue=` reshaping worry is also disposed of: measured `len(ctx.artists) = 56`,
   `len(ctx.revealed_counts) = 28`, `artists == 2*counts` → True, for 28 turns. `revealed_counts`
   is authoritative exactly as the plan asserts, and the 28 turns survive categorical `hue=` because
   `_regroup_categorical_lines` (`plot.py:219`) merges *contiguous runs* and the dialogue has **28
   turns / 28 contiguous runs** (no two consecutive turns share a speaker).

   **But its test does not pass — see New F5.**

**Prior finding 3 (High — "MultiIndex T4" for continuous hue ×4; Self-Review list mismatch): FIXED.**
`grep "MultiIndex T4"` → zero hits. Prereqs row (`plan:135`) and Self-Review (`:2558`) now both read
T1/T2/T5/T6/T8, and `:2561`/`:2583` read T6.

**Prior finding 4 (High — "all five launch notebooks ship ZERO executed outputs" is false): FIXED.**
v2 states 2/6, 4/7, 1/6, 2/6, 2/7. Re-measured today: `conversation_shape` **2/6**,
`market_forecast` **4/7**, `morph_shapes_zoo` **1/6**, `painting_embeddings` **2/6**,
`weather_decades` **2/7** — all five exact.

**Prior finding 5 (High — all five notebook baseline rows wrong): NOT FIXED.**
The citation sweep declined this as its item 9 ("COULD NOT VERIFY"). Re-measured with the plan's own
extracted `measure_native_ratio.py`:

| notebook | plan `:64-68` | measured | 
|-|-|-|
| conversation_shape | 186 / 11 / 5.9% | **191 / 12 / 6.3%** |
| market_forecast | 192 / 11 / 5.7% | **193 / 12 / 6.2%** |
| morph_shapes_zoo | 45 / 8 / 17.8% | **46 / 9 / 19.6%** |
| painting_embeddings | 116 / 10 / 8.6% | **121 / 11 / 9.1%** |
| weather_decades | 206 / 10 / 4.9% | **207 / 11 / 5.3%** |

Reproduces the prior audit's numbers exactly. Five of five still wrong.

**Prior finding 6 (Med — `docs/conf.py:115` → `:131`): FIXED.** `sed -n 131p docs/conf.py` →
`nbsphinx_execute = 'never'`.

**Prior finding 7 (Med — Task 1 "17 passed" should be 16): PARTIAL.**
Steps say 16 (`:556`, `:613`); Self-Review still says 17 twice (`:2559`, `:2579`). And the real
number is neither — measured **14 passed, 2 failed** (New F4).

**Prior finding 8 (Med — Tasks 3–6 "Execute and measure" cell counts impossible): FIXED, but now
self-contradictory.** The counts are corrected to 7/8, 4/5, 5/6, 5/6, 4/5 — each explicitly exempting
cell 0. Task 8's gate requires **every** code cell. Both are wrong anyway (New F3).

**Prior finding 9 (Med — Task 7 Step 2 grep gate dependency): FIXED.** Explicit dependency note at
`plan:1965`.

**Prior finding 10 (Med — "Decisions still needed" numbering): FIXED.** All six are bullets.

**Prior finding 11 (Low — "the five clean ones" naming seven): FIXED.** `plan:1914` reads
"The seven clean ones".

**Prior finding 12 (Low — Task 5 citations into `animate_conversation.py`): FIXED.** Verified in the
real file: `:173-176` = `fig.legend(handles=[mpatches.Patch(...)...])`; `:177-178` = the
`fig.text(0.5, 0.965, "Alice's Mad Tea-Party", ...)` title; `:180-181` = `speaker = fig.text(...)`.

**Prior finding 13 (Low — fixed `IMAGE_PALETTE_N=6` breaks a >6-category hue): NOT FIXED.**
No change in v2; and moot in the worse sense, because the categorical path never reaches
`colors.py:332` at all (New F4).

---

## New findings

### `Fatal | Task 8 Step 2 BUDGETS + Task 5 budget | test_file_is_within_its_size_budget cannot pass on the plan's own prescribed code`

Extracted every prescribed rewrite and measured it with the plan's own extracted script.

| candidate | measured code lines | budget | verdict |
|-|-|-|-|
| market (plan `:729-921`, whole file) | 109 | ≤115 | OK |
| weather (plan `:1019-1128`, whole file) | 56 | ≤62 | OK |
| paintings (plan `:1223-1350` + real 54-line `PAINTINGS`) | 111 | ≤118 | OK |
| **conversation** (real `:1-85` + plan `:1458-1564`) | **88** | ≤72 | **EXCEEDS by 16** |
| morph (real `:1-93` + plan `:1825-1836`) | 28 | ≤30 | OK |

Best case for conversation — no module docstring at all, a single `import hypertools as hyp`, nothing
but what the plan mandates — is **87**, still 15 over:

- `SPEAKER_COLOR` + `TURNS`, which Task 5 Step 1 says to keep **verbatim** (`:44-85`) = **34** code lines
  (the plan says "the `TURNS` list alone is 29 lines");
- the prescribed tail (`plan:1458-1564`) = **52** code lines;
- one import = 1.

v2 *added* `turn_alpha` as a separate documented function to fix the v1 Fatal, which is why this is
worse than the 74 the first review measured. Contract 6 says the budget must be renegotiated in the
plan rather than the assertion weakened — so this is a plan edit, not an implementation detail, and
"106 passed" is impossible until it is made.

### `Fatal | scripts/measure_native_ratio.py _code_lines_nb | the metric is not comparable between .py and .ipynb, so every notebook budget is unachievable`

`_code_lines_py` strips docstrings; `_code_lines_nb` has no docstring handling at all and counts every
docstring line as code. Proven on byte-identical source:

```
IDENTICAL source measured two ways:
  as .py   : (3, 2)
  as .ipynb: (11, 2)
```

The notebook budgets are set only 4–8 lines above their script counterparts (120/115, 66/62, 110/118,
76/72, 34/30), i.e. calibrated as if the two were comparable. Built from the plan's own cell tables
(Colab cell + the prescribed code + `HTML(ani.to_jshtml())`), all five blow up:

```
notebook                    code  budget  verdict
nb_market.ipynb              153     120  *** EXCEEDS by 33 ***
nb_weather.ipynb              91      66  *** EXCEEDS by 25 ***
nb_paintings.ipynb           150     110  *** EXCEEDS by 40 ***
nb_conversation.ipynb        115      76  *** EXCEEDS by 39 ***
nb_morph.ipynb                53      34  *** EXCEEDS by 19 ***
```

Even granting the most generous reading — every docstring manually stripped out of every notebook,
which contradicts the plan's own cell tables (cell 9 of the conversation notebook is specified as
`recency_fade` + `ani.on_frame(...)`, and `recency_fade`'s docstring is 13 lines) — notebook code is
script code + 4, so **paintings (115 > 110)** and **conversation (91 > 76)** still fail. This same
asymmetry is why the plan's baseline table compares `.py` and `.ipynb` rows that are not on the same
scale.

### `Fatal | Task 8 test_every_launch_notebook_ships_executed_outputs | requires every code cell to carry outputs; imports-only and assignment-only cells never can`

Built a notebook matching the plan's **own** weather cell table (Task 3 Step 4: code at cells 0,3,5,7,9)
and executed it with `nbclient` against a kernel registered from this repo's venv:

```
code cell 0: 2 outputs ['stream', 'stream']
code cell 1: 0 outputs []          <- plan cell 3: imports + CACHE + BASE
code cell 2: 1 outputs ['stream']
code cell 3: 0 outputs []          <- plan cell 7: `fig, ani = hyp.plot(...)`
code cell 4: 1 outputs ['execute_result']

executed 3/5   unexecuted=[1, 3]
plan's Task 3 Step 5 expects: 4/5
Task 8 gate requires: 5/5 -> FAILS
```

An imports-and-constants cell and an assignment cell emit no output no matter how many times the
notebook is re-executed. Applying the same reading to the other four cell tables: market can reach at
most 4 of 8 (cells 3, 7, 9, 13 are imports / a comprehension / `fig, ani = ...` / a `fig.text` loop)
against a gate of 8/8 and a stated expectation of 7/8; paintings at most 3 of 6 against 6/6 and a
stated 5/6; morph at most 2 of 5 against 5/5 and a stated 4/5.

Two separate defects here, and the plan does **not** say the gate is meant to fail until re-execution —
Task 8 Step 3 says "Expected: **106 passed**":
1. the gate (all cells) contradicts the per-task expectations (cell 0 explicitly exempt: "cell 0's
   Colab install cell produces none");
2. both are unreachable regardless.

For the record the gate does fail today, for the ordinary reason: `market_forecast` 4/7,
`weather_decades` 2/7, `painting_embeddings` 2/6, `conversation_shape` 2/6, `morph_shapes_zoo` 1/6 →
**WOULD FAIL**. `test_no_launch_notebook_committed_an_error_output` passes today (no committed tracebacks).

### `Fatal | Task 1 Step 4 | "ONE interception point serves every palette consumer" is false; Task 1 measures 14 passed / 2 failed, not 16`

Applied Task 1 Step 3 (`plan:402-519`) and Step 4 verbatim to `hypertools/plot/colors.py` in a
throwaway worktree and ran the extracted `tests/plot/test_image_palette.py` (`plan:197-388`):

```
FAILED tests/plot/test_image_palette.py::test_palette_string_colours_a_categorical_hue
FAILED tests/plot/test_image_palette.py::test_plotly_backend_accepts_an_image_palette
2 failed, 14 passed in 2.14s
```

Both failures are the categorical path, both with seaborn's own
`ValueError: 'image:/.../painting.png' is not a valid palette name`. Traceback origin:

```
File ".../hypertools/plot/plot.py", line 4825, in plot
File ".../seaborn/rcmod.py", line 526, in set_palette
File ".../seaborn/palettes.py", line 237, in color_palette
```

`plot.py:4825` calls `sns.set_palette(palette=_seaborn_palette_arg(palette, len(xform)), ...)` and never
touches `colors._get_palette`. There are six such un-intercepted seaborn palette call sites outside
`colors.py`: `plot.py:208`, `:4118`, `:4657`, `:4767`, `:4825`, and `_shared/helpers.py:116`.

This invalidates the plan's stated API justification (`plan:177`: "one interception makes an image
palette work on every path — categorical hue, continuous hue, matrix hue, the matplotlib colorbar and
the plotly colorbar — with no per-call-site change") and its Self-Review claim of "one interception
point at `colors.py:305-306` … justified against the four consumers it automatically serves".

**Concrete fix:** intercept in `_seaborn_palette_arg` (`plot.py:113-124`) as well — it already exists
for exactly this purpose (its docstring: "`palette` in a form seaborn's `color_palette`/`set_palette`
accept", and it already special-cases a `Colormap` instance by routing through `get_palette_colors`).
Add the `IMAGE_PALETTE_PREFIX` branch there, returning `[tuple(c) for c in image_palette(path,
n_colors=IMAGE_PALETTE_N)]`. Then re-check `plot.py:4118`/`:4657`/`:4767` for the plotly path.

The other 14 tests pass, including the regression that is the point of the task
(`test_a_vivid_minority_colour_beats_the_muted_background`), so the `frac × chroma` ordering rule,
the achromatic fallback, the dedup, the `k = min(n_colors, n_unique)` cap and the error messages are
all sound.

### `High | Task 5 Step 2a _ctx helper | test_a_parallel_animation_is_an_explicit_error cannot pass; Task 5 is 11 passed / 1 failed, not 12`

The ID count is right — `ast` expansion of the extracted block gives exactly **12** (8 `def test_`,
two parametrized ×3), matching the plan's derivation table. But running it gives:

```
1 failed, 11 passed in 8.01s
```

```
    with pytest.raises(RuntimeError, match='serial'):
>       fade(_ctx(current=None))
tests/plot/test_recency_fade.py:40: in _ctx
    revealed = tuple(10 if i <= current else 0 for i in range(n))
E   TypeError: '<=' not supported between instances of 'int' and 'NoneType'
```

The helper crashes while building the argument, before `recency_fade` is ever called, so
`pytest.raises(RuntimeError)` never sees the guard. The guard itself is correct — passing a
hand-built `FrameContext` with `current_index=None` directly does raise
`RuntimeError: recency_fade needs a serial reveal: ...`.

**Fix (one line):** `revealed = tuple(10 if (current is not None and i <= current) else 0 for i in range(n))`.

**Test quality otherwise: good.** Mutation-tested against four deliberately broken callbacks; every
test fails on the defect it documents:

| mutant | result |
|-|-|
| v1's defect (iterate `ctx.artists`, index `revealed_counts[i]`) | 12 failed |
| skipped assignment for unspoken turns (the "smear") | 5 failed, 7 passed — caught by `test_every_head_and_trail_is_assigned_on_every_frame`, `test_first_middle_and_last_turn`, `test_trails_track_their_own_head` |
| stateful alpha (depends on call count) | 4 failed — caught by all 3 `test_alpha_depends_only_on_the_frame_not_on_history` IDs |
| head/trail mispaired by one | 2 failed — caught by `test_trails_track_their_own_head` |

The `importorskip` guards are correct: `sentence_transformers` and `umap` are both importable in this
venv, both `importorskip` calls precede the `from hypertools.plot.animation_context import FrameContext`
import, and `plt.Line2D` resolves (`hasattr(plt,'Line2D')` → True). `FrameContext` accepts the helper's
11 keyword arguments (`segment_index`/`segment_kind` default to `None`).

### `High | Task 6 (all of it) + Global Constraints plan:124 | Task 6 has already landed on HEAD; its BEFORE numbers and every citation are stale, and its notebook half is missing`

`examples/animate_morph_zoo.py` is **96 raw lines**, not the 129 the plan states, and its tail already is
what Task 6 Step 1 prescribes:

```python
fig, ani = hyp.plot(clouds, fmt='.', color='k', markersize=1.6,
                    animate='morph', rotations=rotations, morph_samples=N,
                    duration=duration, frame_rate=fps, size=(6, 6), show=False,
                    title=titles)
```

Commit `d730a085 docs(1.1): document order=, per-dataset alpha=, on_frame, per-segment titles;
simplify examples` did it. A defect-marker scan confirms `examples/animate_morph_zoo.py` is **CLEAN**
while `docs/tutorials/morph_shapes_zoo.ipynb` still carries `ani._func`, `from hypertools.plot import
morph` and `morph_schedule` — i.e. Contract 2's script/notebook lockstep is **already broken on HEAD**,
which is the exact condition Task 8's gate is meant to prevent.

Consequences:
- Task 6's BEFORE block (129 raw / 40 code / 6 native / 15.0% / A=17 B=5 C=16 D=0 NATIVE=5) is wrong;
  measured **26 code / 6 native / 23.1%**.
- Every Task 6 line citation is stale: "Delete `from hypertools.plot import morph as _morph` (line 35)"
  — line 35 is now `"""Center a point cloud and scale it into the hypertools [-1, 1] cube."""` inside
  `normalize()`, and the import is already gone. "Replace everything from line 94 to the end" — the file
  ends at 96 and lines 90-96 are already the prescribed call. Likewise `:105-107`, `:108-128`, `:45-50`,
  `:54-61`, `:63-66`, `:14-22`.
- The *Decisions* entry citing the `normalize()` helper at `:38-42` is wrong; it is at **`:34-38`**.
- Global Constraints `plan:124` — "the five launch examples and their notebooks are untouched" — is
  false as of today.
- Raw line counts for the other four also drifted: conversation **320** (plan says 316), market **376**
  (355), paintings **212** (213), weather **336** (333).

Task 6 needs rewriting as "the script is done; do the notebook", or deleting.

### `High | throughout | every plot.py citation in this plan is stale — the 2026-08-01 sweep's item 6 was scoped to Plan A only`

The sweep note says so explicitly ("Full `hypertools/plot/*.py` citation sweep" appears under
"## In A (forecast-animation)", and found 37 of 37 unique `plot.py` values wrong there). Plan B's were
never checked. Spot-checked eleven; **all eleven are wrong**:

| plan claims | line actually contains | true location |
|-|-|-|
| `plot.py:2750-2751` equal per-dataset widths check | `"predict= is only supported with static plots and with "` | **`:3152-3153`** |
| `plot.py:807` `palette` docstring entry | `focused=None,` | **`:1066`** |
| `plot.py:930` `colorbar=` | MultiIndex prose | **`:1189`** |
| `plot.py:950` `title=` | "automatically propagated to each dataset's runs" | **`:1209`** |
| `plot.py:895-910` per-observation `labels=` | `lw=3, alpha=1.0.` | **`:1154`** |
| `plot.py:1013` `xlabel`/`ylabel`/`zlabel` | GH #206 extra-kwarg prose | `:832` (signature) |
| `plot.py:1064` `manip=` | "resulting drawn-trace count" | **`:1333`** |
| `plot.py:1246` GIF writer | coverage-gap prose | `:1515`/`:1523` |
| `plot.py:2347-2354` the `predict=` `NotImplementedError` | `:2347` is **blank** | **`:2740-2756`** |
| `plot.py:2678-2684` "hue is discarded" | matplotlib save-time IndexError comment | — |
| `plot.py:4040-4051` pooled affine | `hue = group_by_category(hue)` | — |

`plot.py:2750-2751` is the sharpest case: the plan's own *Verification note* claims to have re-derived
it ("the **check** is `plot.py:2750-2751` … Cite 2750-2751"), and it now lands on the forecast refusal
string — a different claim entirely, and one Plan A separately cites correctly at `:2740-2756`.

Partially right: `plot.py:143-150` falls inside `_draw_forecast_overlays` (`:137-180`);
`plot.py:3039-3050` brackets the linewidth-warn block (`:3050` = `# unless we warn here.`);
`plot.py:204-228` brackets `_regroup_categorical_lines` (`:219`).

**Everything outside `plot.py` checks out**, which is worth stating plainly:
`colors.py:24/105/227/250/269/287/305/306/323-331/332` all land exactly;
`text2mat.py:184/391/404`; `smooth.py:14/232`; `animate.py:84`; `normalize.py:86/175`;
`_shared/helpers.py:24`; `morph.py:36`; `describe.py:13`; `docs/api.rst:108-116`;
`scripts/generate_gallery_thumbs.py:26` (`MPL_ANIMS = ['animate', ...]`, six stems);
`docs/conf.py:131`; and `docs/tutorials.rst:144` really does carry the
`.. image:: _static/thumbnails/sphx_glr_plot_story_trajectories_thumb.gif` pattern Task 8 Step 6 copies.

### `High | Task 8 test_examples_produce_their_stated_artifact | violates the plan's own Contract 4 by putting network fetches and model downloads in the library test suite`

Contract 4 (`plan:92`): "**Network fetches live in examples, wrapped in a fallback, never in a library
test.**" `test_examples_produce_their_stated_artifact` does `runpy.run_path(f'examples/{stem}.py')` for
all five. Three of those five contain a live `urllib.request.urlopen` (weather, market, paintings — one
each), and two more resolve `all-MiniLM-L6-v2` through `sentence_transformers` (a ~90 MB download on a
cold cache) plus a UMAP fit. Market additionally runs the 210-Kalman-fit accuracy loop the plan itself
budgets at ~7 s. Timed one today: `runpy.run_path('examples/animate_weather_decades.py')` = **2.22 s
real**, and it printed `weather: 6 cities (open-meteo archive)` — a real network fetch, from inside what
would be a `pytest` run. `tests/plot/test_recency_fade.py` executes the conversation example a second
time via its own module-scoped fixture.

Either the contract or the test has to give. The contract is the better rule; the semantic checks can be
kept by importing the example's *functions* and driving them on synthetic data, or by marking these five
IDs with a network/slow marker that the default run deselects.

### `Med | Task 8 Step 2 import note | from scripts.measure_native_ratio import measure fails under bare pytest, which is what CI runs, and aborts collection of the whole suite`

The plan says "pytest inserts the rootdir on `sys.path` under the default `rootdir`-based import mode".
It does not. What makes the prescribed command work is `python -m`, which puts the cwd on `sys.path`:

```
$ .venv/bin/python -m pytest tests/test_examples_are_native.py --collect-only -q
106 tests collected in 0.01s

$ .venv/bin/pytest tests/test_examples_are_native.py --collect-only -q
E   ModuleNotFoundError: No module named 'scripts'
!!!! Interrupted: 1 error during collection !!!!
no tests collected, 1 error in 0.05s
```

`.github/workflows/test.yml:148` runs `pytest -v --tb=short`, `:159` runs `pytest -q`, `:167` runs
`pytest --cov=...` — all bare. So this goes green locally on every command the plan prescribes and takes
the **entire suite** down in CI. The plan's fallback ("add an empty `scripts/__init__.py`") does not fix
it either: with import mode `prepend` and no `tests/__init__.py`, pytest inserts `<repo>/tests`, not
`<repo>`. The robust fix is `pythonpath = ["."]` in `pyproject.toml`'s `[tool.pytest.ini_options]`.

### `Med | plan:2579, :2559, :2567 vs :25, :556, :1719, :2408, :2478 | three mutually inconsistent sets of test counts`

- Revision note `:25`: **16**, **106**, **+135**, "Task 5 now contributes 13"; `:22` says "13 tests".
- Steps: `:556`/`:613` = 16; `:1719` = **12** (its own table sums to 12); `:2408` = 106; `:2478` = **+134**
  ("Task 1's 16, Task 5's 12 and Task 8's 106" — 16+12+106 = 134, internally consistent).
- Self-Review `:2559` = "17 real tests"; `:2567` = "a 109-test gate"; `:2579` = "Task 1 adds **17** …
  Task 8 adds **109** (10 budget + 10 ratio + …) … Total **+126**" — still counting the ratio gate v2
  deleted.

Independently derived: Task 5 = **12** IDs (`ast`), Task 8 = **106** IDs (`pytest --collect-only`,
breakdown below), Task 1 = **16** IDs. So `+134` at `:2478` is the arithmetically correct figure and
`:25`'s "13"/"+135" and the whole Self-Review paragraph are wrong. Measured pass counts are lower still
(14 + 11 + <106).

**Task 8's 106 is exactly right**, and its derivation table matches the collected IDs one for one:

```
  80  test_no_defect_marker_in_the_launch_examples      (8 markers x 10 files)
  10  test_file_is_within_its_size_budget
   6  test_older_tutorials_dropped_their_hand_rolled_helpers
   5  test_examples_produce_their_stated_artifact
   1  test_native_ratio_is_reported
   1  test_analyze_tutorial_actually_plots
   1  test_reduce_tutorial_mentions_describe
   1  test_every_launch_notebook_ships_executed_outputs
   1  test_no_launch_notebook_committed_an_error_output
 106  total   (BUDGETS=10, DEFECT_MARKERS=8, STATED_ARTIFACT=5)
```

### `Med | plan:60 | the morph .py baseline row is wrong (the other four .py rows are right)`

Measured today with the plan's own script: conversation 165/9/5.5% (plan 166/9/5.4%),
market 191/11/5.8% (exact), **morph 26/6/23.1% (plan 40/6/15.0%)**, paintings 146/11/7.5% (exact),
weather 195/11/5.6% (plan 196/11/5.6%). The five-script total is therefore **723**, not 739, so the
plan's headline "48 of 739 code lines … 6.5%" — repeated in the Goal, the Verification note, and the
`test_examples_are_native.py` docstring — should be 48/723 = 6.6%. Consequence of F6.

### `Med | Task 8 STATED_ARTIFACT / test_examples_produce_their_stated_artifact | the morph assertion is vacuous and the market assertion is unverifiable`

`assert 'morph' in str(ns.get('ANIMATE', 'morph'))` — `ANIMATE` does not exist in
`examples/animate_morph_zoo.py` (`grep` → no match), and Task 6's prescribed rewrite passes
`animate='morph'` inline, so it never will. The expression is `'morph' in 'morph'` → **always True**.
This test cannot fail on the thing it documents. Use `assert "animate='morph'" in _read(f'examples/{stem}.py')`
or inspect `ani` for the morph schedule.

The market `predicts` assertion (`ax = ns['fig'].axes[0]`; `any(ln.get_linestyle() in ('--',':')
for ln in ax.lines)`) has two soft spots. `fig.axes[0]` happens to be the `Axes3D` when a colorbar is
present (verified: `axes[0]: Axes3D has zaxis=True`, `axes[1]: Axes`), so that part holds. But forecast
overlays are `linestyle='--'` only via `_draw_forecast_overlays` (`plot.py:137`, `:170`/`:174`/`:178`),
which the static path calls behind `if raw_forecasts is not None` (`plot.py:4907-4909`); whether
`animate=True` + `predict=` populates `ax.lines` at construction time — the test drives no frame —
is Plan 3 Task 4's business and cannot be checked today, because `predict=` + `animate=True` still
raises `NotImplementedError: predict= is only supported with static plots and with animate='spin'`.
Flagging rather than asserting: this assertion may silently pass or fail for reasons unrelated to the
example.

`ns['ani']._save_count` does work — verified `_save_count = 10` on a real `fig, ani = hyp.plot(...)`
unpack (`ani` is the raw `FuncAnimation`).

### `Med | Task 8 Step 1 | the git stash baseline-verification recipe cannot work in task order`

`git stash && ... measure examples/animate_conversation.py && git stash pop`, expecting
`code= 166 native= 9 ratio= 5.4%`. Task 8 depends on Tasks 1–7, and Task 5 has already **committed**
the rewritten conversation example, so `git stash` cannot restore the baseline — it would measure the
rewrite. Doing Step 1 first instead (as Task 2 Step 6 suggests) means the tree may be clean, in which
case `git stash` saves nothing and `git stash pop` exits 1 with "No stash entries found"; and per Global
Constraints the tree is expected *not* to be clean, so the stash would also sweep up Plans 1–3's
in-flight edits. Use `git show <base>:examples/animate_conversation.py` instead. (The expected value is
also stale — measured **165 / 9 / 5.5%** today.)

### `Low | plan:1604, :1733 | N_DATASETS = 6 contradicts the example's 28 turns`

`_ctx`'s docstring says the contexts are "shaped exactly like the example's own plot", and `plan:1733`
says "`N_DATASETS` must equal the number of FINAL drawn datasets the example produces". Measured:
**28**. The 6 appears to be carried over from the 6-dataset probe in the Verification note (`plan:49`).
Harmless in practice — the callback is generic and my n=1..29 sweep passes — but nothing in the suite
detects the mismatch, so the instruction is unenforceable. Either set it to 28 or derive it from the
fixture.

### `Low | plan:2586-2588 | stray tool-call scaffolding committed at EOF`

The plan file's last two lines are literally `</content>` and `</invoke>`:

```
0002020    h   t   .  \n   <   /   c   o   n   t   e   n   t   >  \n   <
0002040    /   i   n   v   o   k   e   >  \n
```

### `Low | Task 8 test_older_tutorials_dropped_their_hand_rolled_helpers | 2 of its 6 IDs cannot fail`

Measured today: `stock_forecasting` and `projectile_kalman` contain neither `SentenceTransformer` nor
`ffmpeg`, and Task 7 Steps 1–2 do not touch either for those markers. `modern_sklearn_dynamics`'s
`SentenceTransformer` half is likewise already true. Not wrong, just non-discriminating.

### `Low | plan:2583 | Remaining risk #1 still names the deleted test`

"so `test_file_meets_its_native_ratio_floor` would not catch it" — that test no longer exists in v2.
(The extracted test file itself is clean of every `min_ratio`/floor reference; this is prose only.)

---

## Verification performed

Interpreter: `/Users/jmanning/hypertools/.venv/bin/python` (3.12.10, matplotlib 3.10.8, numpy 2.3.5).
Scratch: `/private/tmp/claude-501/.../scratchpad/p4v2` (outside the repo). Worktree created for Task 1
and removed (`git worktree list` → only `/Users/jmanning/hypertools 4ecb3b6d [dev-1.0]`);
`git status --porcelain` → empty. **No repo file was modified.**

| # | command | measured output |
|-|-|-|
| 1 | `git log --oneline -1` | `4ecb3b6d notes(audit): plan-citation-sweep-2026-08-01 summary` |
| 2 | `ls hypertools/plot/animation_context.py`; `grep -n "class FrameContext"` | exists, `:25` — Plan 1 T7 has landed; 13 fields incl. `segment_index`/`segment_kind` |
| 3 | `grep -n "artists=\|revealed_counts\|_counts" hypertools/plot/matplotlib_backend.py` | serial recorders at `:1421-1426` / `:2166-2171`: `artists=list(lines) + [t for t in trail_lines if t is not None]`, `revealed_counts=_counts`; `_counts = serial_reveal_counts(lengths,...)` at `:1343`/`:2125` |
| 4 | extracted `plan:1507-1561` + `FLOOR, DECAY = 0.10, 0.45`; exhaustive drive n=1..29 × current=0..n-1 vs real `FrameContext`+`Line2D` | `sweep n=1..29 x current=0..n-1 : ALL OK`; guards: missing-trail → `RuntimeError: expected one trail artist per dataset, got 0 trails for 6 datasets`; `revealed_counts=None` → `RuntimeError: recency_fade needs a serial reveal...`; **`current_index=None` via `_ctx` → `TypeError`**; purity over shuffled/repeated frames → `PURE`; n=6 cur=3 heads `[0.182, 0.2823, 0.505, 1.0, 0.0, 0.0]`, trails `[0.0546, 0.0847, 0.1515, 0.3, 0.0, 0.0]` |
| 5 | real `hyp.plot(28 turns, hue=nested categorical, order='serial', chemtrails=True, title=[28])` | `len(ctx.artists)=56`, `len(revealed_counts)=28`, `artists == 2*counts: True`, `current_index=14`, legend `['Alice','March Hare','Hatter','Dormouse']`, title `'Alice "line 14"'` |
| 6 | same plot with `on_frame=recency_fade`, 66 frames incl. repeats/out-of-order | `errors: NONE`; 20 distinct alphas; 56 `Line3D` artists |
| 7 | `ast` expansion of `plan:1581-1712` | 8 `def test_`, TOTAL **12** IDs |
| 8 | `pytest tests/plot/test_recency_fade.py -v` against a stand-in example holding the plan's callback | **`1 failed, 11 passed`** — `test_a_parallel_animation_is_an_explicit_error`, `TypeError: '<=' not supported between instances of 'int' and 'NoneType'` at `_ctx` line 40 |
| 9 | four mutants (v1 defect / skipped assignment / stateful / mispaired trails) | `12 failed` · `5 failed, 7 passed` · `4 failed, 8 passed` · `2 failed, 10 passed` — each caught by the test that documents it |
| 10 | `ast.parse` of `plan:2077-2184` and `plan:2197-2399` | both AST OK; `min_ratio`/`MIN_RATIO`/`ratio_floor`/`meets_its_native_ratio`/`FLOORS` → **NONE** |
| 11 | `pytest tests/test_examples_are_native.py --collect-only -q` | **`106 tests collected`**; breakdown 80/10/6/5/1/1/1/1/1 |
| 12 | `.venv/bin/pytest ... --collect-only` (bare, as CI runs) | `ModuleNotFoundError: No module named 'scripts'` … `Interrupted: 1 error during collection` |
| 13 | `grep -n pytest .github/workflows/test.yml` | `:148 pytest -v --tb=short`, `:159 pytest -q`, `:167 pytest --cov=...` — all bare |
| 14 | notebook-output census of the five launch notebooks | 4/7, 2/7, 2/6, 2/6, 1/6; committed errors: none → gate `WOULD FAIL` |
| 15 | `nbclient` execution of a notebook built from the plan's own weather cell table, venv kernel | `executed 3/5  unexecuted=[1, 3]` (imports cell, `fig, ani =` cell) vs plan 4/5 vs gate 5/5 |
| 16 | extracted `measure_native_ratio.py` on the current 5 examples + 5 notebooks | conversation 165/9/5.5, market 191/11/5.8, **morph 26/6/23.1**, paintings 146/11/7.5, weather 195/11/5.6; notebooks 191/12/6.3, 193/12/6.2, 46/9/19.6, 121/11/9.1, 207/11/5.3 |
| 17 | same script on the plan's own prescribed rewrites | market 109, weather 56, paintings 111, **conversation 88** (best case 87) vs budgets 115/62/118/**72**/30; morph 28 |
| 18 | `measure()` on byte-identical source saved as `.py` and `.ipynb` | `(3, 2)` vs `(11, 2)` — `_code_lines_nb` never strips docstrings |
| 19 | notebooks built from the plan's cell tables | market 153/120, weather 91/66, paintings 150/110, conversation 115/76, morph 53/34 — all exceed |
| 20 | Task 1 Steps 3+4 applied in worktree; `pytest tests/plot/test_image_palette.py` | **`2 failed, 14 passed`**; both categorical, both `ValueError: 'image:...' is not a valid palette name` |
| 21 | traceback origin of that failure | `hypertools/plot/plot.py:4825` → `seaborn/rcmod.py:526 set_palette` → `palettes.py:237`; six un-intercepted call sites: `plot.py:208/4118/4657/4767/4825`, `_shared/helpers.py:116`; natural fix point `_seaborn_palette_arg` at `plot.py:113-124` |
| 22 | defect-marker scan of the 5 examples + 5 notebooks | `animate_morph_zoo.py` **CLEAN**; `morph_shapes_zoo.ipynb` still `ani._func, import morph, morph_schedule`; market `ani._func, ani._args, _shared, antialias_line`; weather `ani._func`; paintings `ST`; conversation `ST, ani._func` |
| 23 | `wc -l examples/animate_*.py`; `git log -- examples/animate_morph_zoo.py` | 320/376/**96**/212/336 vs plan's 316/355/**129**/213/333; `d730a085 docs(1.1): … per-segment titles; simplify examples` |
| 24 | citation spot-checks (≥25 values) | `plot.py` 807/895/930/950/1013/1064/1246/2347/2678/2750-2751/4040 all wrong (true: 1066/1154/1189/1209/832/1333/1515/2740-2756/—/3152-3153/—); `colors.py` 24/105/227/250/269/287/305/306/323-331/332 all correct; `text2mat.py:184/391/404`, `smooth.py:14/232`, `animate.py:84`, `normalize.py:86/175`, `helpers.py:24`, `morph.py:36`, `describe.py:13`, `api.rst:108-116`, `generate_gallery_thumbs.py:26`, `conf.py:131`, `tutorials.rst:144` all correct; morph `normalize()` is `:34-38`, not `:38-42` |
| 25 | `ANIMATE` in the morph example; `_save_count`; `fig.axes[0]` identity | `ANIMATE`: no match → assertion vacuous; `_save_count = 10`; `axes[0]` is `Axes3D`, `axes[1]` the colorbar |
| 26 | `hyp.plot(..., predict='Kalman', animate=True)` today | `NotImplementedError: predict= is only supported with static plots and with animate='spin' ...` |
| 27 | `runpy.run_path('examples/animate_weather_decades.py')` timed | `real 2.22`, printed `weather: 6 cities (open-meteo archive)` — live network from inside a would-be test |
| 28 | `grep -c urllib.request.urlopen` on the three fetching examples | 1 each (weather, market, paintings) |
| 29 | Task 7 gate state today | `stock_forecasting` / `projectile_kalman`: ST=False ffmpeg=False (assertions non-discriminating); `modern_sklearn_dynamics`: ST=False ffmpeg=True |
| 30 | `od -c` of the plan's last bytes | `</content>\n</invoke>\n` |

---

## What has to change before this is implementable

1. **Raise `examples/animate_conversation.py`'s budget to ≥ 90** (measured best case 87) in the plan,
   per Contract 6 — or cut the prescribed tail.
2. **Give `_code_lines_nb` the same docstring stripping `_code_lines_py` has**, then re-derive all five
   notebook budgets. Without this the `.py` and `.ipynb` rows of every table in the plan are on
   different scales.
3. **Replace `test_every_launch_notebook_ships_executed_outputs`** with something achievable — e.g.
   "every code cell that *can* produce output does", or a floor per notebook — and reconcile it with the
   per-task 7/8, 4/5, 5/6, 5/6, 4/5 expectations.
4. **Extend Task 1 Step 4 to `_seaborn_palette_arg` (`plot.py:113-124`)** and re-check the plotly path;
   correct "16 passed" and the "one interception point" rationale.
5. **Fix `_ctx`**: `10 if (current is not None and i <= current) else 0`.
6. **Rewrite Task 6** — its script half is already on HEAD (`d730a085`); only
   `docs/tutorials/morph_shapes_zoo.ipynb` remains. Fix Global Constraints `plan:124`.
7. **Sweep this plan's `plot.py` citations** the way the 2026-08-01 sweep did Plan A's.
8. **Resolve Contract 4 vs `test_examples_produce_their_stated_artifact`.**
9. **Add `pythonpath = ["."]`** to `[tool.pytest.ini_options]` so bare `pytest` (CI) can import `scripts`.
10. Housekeeping: Self-Review suite arithmetic (17/109/+126 → 16/12/106/+134); revision note's "13
    tests"/"+135"; the morph baseline row and the 739 total; the vacuous `ANIMATE` assertion; the
    `git stash` recipe; `N_DATASETS`; the `test_file_meets_its_native_ratio_floor` reference; the
    `</content></invoke>` tail.
