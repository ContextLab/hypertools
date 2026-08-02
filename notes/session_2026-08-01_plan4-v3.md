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

### 6. Test-count arithmetic: ONE off-by-one, and the plan contradicts itself

Derived with an AST counter that expands `@pytest.mark.parametrize`, including
`sorted(DICT.items())` and module-level `pytest.param` lists. **The counter was itself wrong first**
— it scored `sorted(DEFECT_MARKERS.items())` as 1 case instead of 8, giving Task 8 = 36. Fixed by
unwrapping `.items()/.keys()/.values()` inside the `sorted(...)` call, then validated against a
hand-built known-answer fixture (expected 17, got 17) before any number below was trusted.

| block | derived | plan claims | |
|-|-|-|-|
| Task 1 | 16 | 16 (L25, L613) | ✓ |
| Task 5 | **12** | **13** (L25 revision note) | ✗ off by one |
| Task 5 | **12** | **12** (L1724 step text) | ✓ — the plan disagrees with *itself* |
| Task 8 | 106 | 106 (L25) | ✓ |
| total delta | **134** | **+135** (L25) | ✗ propagates the same off-by-one |

Cross-check: 3 python blocks define 16 + 8 + 9 = 33 `def test_`; the file has 35 occurrences of
`def test_`, the other 2 being prose mentions at L613 and L1724. No test block hides under a
non-python fence. The reviewer independently derived 12 / 106 / +134 — agreement.

Fix: the step-level number (12) is right; the revision note (13, +135) is wrong.

### 7. The `git stash` recipe is a data-loss hazard, not just broken

Plan L2195: `git stash && .venv/bin/python scripts/measure_native_ratio.py <file> && git stash pop`.

Demonstrated in a scratch repo with a clean tree — the exact state at `065c841e`:

```
--- stashes before recipe: 1        (a PRE-EXISTING unrelated stash)
git stash  ->  "No local changes to save"  (exit 0, saves nothing)
git stash pop  ->  POP SUCCEEDED, Dropped refs/stash@{0}
--- after recipe the tree contains: f.txt  other.txt   <-- other.txt is the unrelated work
--- stashes remaining: 0
```

With nothing to stash, `pop` restores and then **drops a different, pre-existing stash**. The
recipe silently applies and destroys unrelated work. Upgrade this from Low to a real hazard.

Replacement (maintainer's suggestion, verified in the real repo): `git show <base>:<path>`.
It is read-only, needs no clean tree, and mutates nothing — `git status --porcelain` count is 0
before and after. `git show 4d1d2223:examples/animate_conversation.py` → 315 lines vs 320 in the
tree today, i.e. it reads the true BEFORE state that the stash recipe was groping for.

### 8. The prescribed replacement gate repeats the flaw it replaces

The maintainer prescribed, as the fix for the unattainable "every code cell has outputs" gate:

```python
assert all(cell["execution_count"] is not None for cell in code_cells)
```

Measured: **this fails on all five notebooks**, because code cell 0 of every one is the Colab
install cell and is deliberately never executed:

```
market_forecast  cell 0  execution_count=None
    # Install hypertools (dev-1.0 preview) -- run this first on Colab.
    %pip install -q "hypertools[interactive] @ git+https://github.com/ContextLab/hypertools.git@dev-1.0"
```

`scripts/add_colab_install_cell.py` injects it on purpose and re-targets it per branch. Executing
it locally would pip-install during a docs build. So the prescribed gate is *another* universal
quantifier over a set containing a deliberately-exempt member — the same class of error as v2's
"every code cell", just relocated from outputs to execution_count. Measured index sets:
`exec_idx` is `[1..6]` or `[1..5]` for all five; never includes 0.

**Better gate — derive the rule from cell content instead of recording a constant.** A recorded
index set risks rubber-stamping whatever the notebook happens to do. A rule keyed on what a cell
*calls* can be written before the artifact exists and still has teeth:

> every code cell whose source calls a rendering API (`hyp.plot(`, `plt.show(`, `display(`,
> `plt.imshow(`) must carry a visible output.

Measured against today's committed notebooks — it is RED on 4 of 5:

| notebook | code cells | rendering cell | has output? |
|-|-|-|-|
| market_forecast | 7 | 5 | yes — **passes** (control) |
| weather_decades | 7 | 5 | **no — FAILS** |
| painting_embeddings | 6 | 4 | **no — FAILS** |
| conversation_shape | 6 | 4 | **no — FAILS** |
| morph_shapes_zoo | 6 | 3 | **no — FAILS** |

It needs no measured constant, cannot be satisfied by a stray `print()` in the wrong cell, and goes
green exactly when the notebooks are properly executed. `market_forecast` is the non-red control
and should be labelled as one.

Resulting four-part gate for v3: (a) exempt the install cell **by content, not index**; (b) every
remaining code cell has non-null `execution_count`; (c) no cell carries an `output_type == "error"`;
(d) the derived render-implies-output rule. An exact index-set pin may be added on top as
drift-detection, but it is not the load-bearing assertion.

### 9. Landed-state audit (`notes/audit/plan4_landed_state.md`) — independently re-verified

| task | script | notebook | action |
|-|-|-|-|
| 2 market | partially landed | out of sync | REBASE + **BLOCKED** |
| 3 weather | partially landed | out of sync | REBASE |
| 4 paintings | untouched, baseline accurate | in sync | WRITE-AS-IS (gated on Task 1) |
| 5 conversation | partially landed | out of sync | REBASE |
| 6 morph | **fully landed** — Step 1 already done verbatim | out of sync | REBASE: delete Step 1 |

Four claims re-checked by hand rather than accepted:

1. **`forecast_trail=` is ABSENT** from `plot()`'s 75 parameters (`inspect.signature`). It comes from
   Plan 3 Task 5. So Task 2 is **blocked, not merely stale** — its prescribed `hyp.plot(...)` call
   cannot run today. This independently confirms the maintainer's ordering: Plan 3 must land first.
2. `HyperAnimation` is a `tuple` subclass and has no `_func`; the old monkeypatch reached
   `result[1]`.
3. In weather and conversation, `ani._func` now appears **only inside docstrings** (`:318`, `:281`)
   explaining the migration away from it. Real code is gone.
4. All five notebooks still carry the private reaches their scripts dropped:
   market `[ani._func, ani._args, hypertools._shared]`, weather `[ani._func]`,
   conversation `[ani._func, ani._args, SentenceTransformer]`,
   morph `[ani._func, from hypertools.plot import morph]`, paintings `[SentenceTransformer]`.
   **The notebooks now teach the private-API approach the scripts just abandoned.**

### 10. The metric bug and the gate bug are ONE bug in two places

`measure_native_ratio._code_lines_nb` doesn't strip docstrings (finding: the .py/.ipynb asymmetry),
and Task 8's `_code_text()` doesn't either — so the `DEFECT_MARKERS` gate reads docstrings as code
and would fail weather and conversation **for their own documentation**, since `d730a085` wrote
prose naming `ani._func` while explaining its removal. Same missing docstring-strip, two call sites.
Fix once, as a shared callee both use — the `anim_window_bounds` lesson: a shared callee cannot
drift from itself.

### 11. NEEDS A RULING — Contract 3 vs. `d730a085`

Plan 4 Contract 3 (L78): *"After this plan, no example or notebook contains `ani._func`,
`ani._args`, `hypertools._shared` …"*. But `d730a085` deliberately **kept** two private usages in
the market example and recorded why, with measurements:

> `:204-213` — "the one place this example still reaches into matplotlib's private FuncAnimation
> internals (`ani._args`/`ani._func`), deliberately: it needs the fully-revealed, ANTIALIASED
> on-screen line … `ctx.datasets` is the pre-antialiasing array at a coarser resolution and fits a
> measurably different (~2-8%, checked empirically) slope."

> `:283-287` — "There is no public re-export of it … reimplementing PCHIP antialiasing by hand here
> would risk silently drifting from what `hyp.plot` actually draws, so the private import stays."

One of the two positions must be withdrawn explicitly; the rebase cannot paper over it.

**My recommendation — the `d730a085` position governs and Contract 3 narrows.** It is newer, it
carries measurement, and Contract 3's real purpose (examples must not *teach* private API as the way
to do things) is not served by banning a one-time setup step that has no public equivalent and says
so inline. Proposed replacement contract, which keeps teeth: *no private API where a public
equivalent exists; any remaining private use must appear in an explicit
`PRIVATE_API_EXCEPTIONS = {(path, marker): reason}` allowlist.* Any unlisted private reach still
fails the gate, so new ones cannot creep in, and each retained one is reviewed rather than assumed.
Proceeding on this assumption; flagging it for the maintainer.

### 12. Metric audit (`notes/audit/plan4_metric_remeasure.md`)

The docstring-strip fix is safe: `_code_lines_py` results are **identical line-for-line** on all five
scripts after refactoring the shared state machine out. Notebook counts move where docstrings exist:
market 193→187, weather 207→194, conversation 191→176; paintings and morph unchanged.

**All five script budgets are ATTAINABLE** by the plan's own prescribed content — market 109≤115,
weather 56≤62, paintings 111≤118, conversation 88≤90, morph 26≤30 (morph is already under budget
today at 26).

**Two notebook budgets are not, and the reason is simpler than docstrings** — they are set *below
their own script budgets*:

| task | script ≤ | notebook ≤ | headroom |
|-|-|-|-|
| market | 115 | 120 | +5 |
| weather | 62 | 66 | +4 |
| paintings | 118 | **110** | **−8 — impossible** |
| conversation | 90 | **76** | **−14 — impossible** |
| morph | 30 | 34 | +4 |

A notebook contains its script's code plus an install cell plus a display cell. A notebook budget
below the script budget cannot be met by any correct notebook, whatever the metric does.

**Fix — derive the notebook budget instead of writing it down.**
`notebook_budget = script_budget + NOTEBOOK_OVERHEAD`, with `NOTEBOOK_OVERHEAD = 5` measured
(max cell-0 code lines across the five = 3, plus a 2-line `from IPython.display import HTML` +
`HTML(ani.to_jshtml())` cell). Then every notebook budget is computed from the one number per task
that is actually chosen, it can never again be set below its script's, and it self-updates. Checked
against the prescribed content — all ten now pass, and still tightly:

| task | derived nb ≤ | prescribed | headroom |
|-|-|-|-|
| market | 120 | 113 | 7 |
| weather | 67 | 60 | 7 |
| paintings | 123 | 116 | 7 |
| conversation | 95 | 93 | **2** |
| morph | 35 | 30 | 5 |

This is not weakening a contract to fit the code (Contract 6): two of the ten were unsatisfiable by
construction, and the replacement is a derivation, not a fitted number.

### 13. The removed ratio floor is still promised in ten places

The revision note records that the per-file native-ratio floor was **deleted** as one of v1's two
Fatals, and Task 8's module docstring confirms ratio is "REPORTED, not gated". But ten lines still
state it as a budget: **634, 990, 1006, 1186, 1201, 1420, 1435, 1792, 1807, 1904** — e.g. L634
*"script ≤ 115 code lines, ≥ 26% native"*. The plan promises floors its own gate no longer enforces.
L1435 is doubly stale: it also still says *"script ≤ 72"* where the enforced `BUDGETS` dict says 90.

### 14. "Corrected wherever it appears" — it wasn't

Revision-note L24 claims the false *"all five launch notebooks ship ZERO executed outputs"* was
"Corrected wherever it appears." All five per-task BEFORE headers still say it — **632, 1004, 1199,
1433, 1805**, each reading "0 of N code cells executed". Measured reality is 4/7, 2/7, 2/6, 2/6, 1/6.

### 15. The plan already knew about the install cell — the gate just didn't

Four "Expected:" lines say it outright, e.g. L1186: *"`4/5 code cells produced output` (cells 3, 5,
7, 9; **cell 0's Colab install cell produces none**)"*, and likewise at L1420, L1792, L1904. So the
exemption that breaks the prescribed `execution_count` gate (finding 8) was **documented in this
same file four times** and simply never made it into the assertion. Same shape as the plotly
Play-button defect: the rule was written down in the file and violated in the file.

### 16. NEW FATAL — `fig, ani = hyp.plot(...)` throws the public API away

`hyp.plot(..., animate=...)` returns a `HyperAnimation`, a **2-tuple subclass** of
`(figure, animation)`. Unpacking it binds `ani` to element `[1]` — the raw
`matplotlib.animation.FuncAnimation` — and discards the wrapper that carries `.on_frame()`.
Reproduced directly:

```
whole result : HyperAnimation   | has on_frame: True
unpacked ani : FuncAnimation    | has on_frame: False
on_frame on unpacked: AttributeError -> 'FuncAnimation' object has no attribute 'on_frame'
on_frame on whole result: OK, returns self: True
```

Plan 4 Task 5's prescribed notebook does exactly this: cell 3 does `fig, ani = hyp.plot(...)`, then
cell 4 calls `ani.on_frame(recency_fade)` → **`AttributeError`**, and `nbclient` halts there, so
cell 5 never runs. No "N of 6" prediction can be derived for `conversation_shape` from the plan's
text, because the notebook does not execute at all.

Worse: the **already-landed** `examples/animate_conversation.py` avoids this by binding
`anim = hyp.plot(...)` with no unpacking. So applying Task 5 verbatim would not merely fail to add
titles — it would **regress working code into a crash**.

**Blast radius measured, and it is confined to the plans.** Scanned every `.py`/`.rst`/`.ipynb`/`.md`
under `docs/`, `examples/`, `hypertools/`, plus README and CHANGELOG, for "unpack, then call a
wrapper-only method". Hits: only `2026-07-28-...examples-and-tutorials.md` (Plan 4) and
`2026-07-26-...animation-core.md` (Plan 1). The shipped library and docs are correct —
`plot.py:1865` uses `anim = hyp.plot(...)`; `plot.py:2304` accurately documents that the tuple
unpacking works; `docs/animation.rst` unpacks only where it passes `on_frame=` as a **kwarg**
(`:122`, `:326`) and uses the un-unpacked `anim` before calling `.on_frame()` (`:230`, `:248`).
The defect is in the plan text, not the product.

Note `_save_count` survives unpacking (the raw `FuncAnimation` has it), so the `STATED_ARTIFACT`
test would have passed — for the wrong reason.

### 17. CORRECTION to my finding 8 — "render implies output" is weaker than I said

I claimed a cell calling a rendering API must carry visible output, and that the rule was red on 4
of 5 notebooks. The arithmetic was right; **the interpretation was wrong**. Measured MIME types
across all five notebooks: there is **no `image/png` and no `text/html` output anywhere**. The
outputs that exist are `stream` (i.e. `print`) plus, in paintings and conversation, tqdm
progress-bar widgets from `sentence_transformers`. So market's render cell "passes" my rule purely
because it also prints — the rule measures *did this cell print*, not *did a figure render*.

The real convention (commit `9b94d86f`) is a **companion GIF**: the last code cell calls
`ani.save('<stem>.gif', fps=fps)` and the final **markdown** cell embeds it — matching how
`conversation_trajectories`, `streaming_data` and `wikipedia_embeddings` already ship. So the
"intended rendered artifact" assertion must check *the GIF exists and the markdown reference
resolves*, not that a cell emitted an image. That also catches a real oddity the audit found:
`morph_shapes_zoo.ipynb` embeds `morph_zoo.gif`, not `morph_shapes_zoo.gif`.

Keep the derived rule as a cheap sanity check if it earns its place, but the load-bearing artifact
assertion is the GIF-plus-reference one.

### 18. Image palette: "ONE interception point" is FALSE, and the real score is 0/6

`notes/audit/plan4_image_palette.md`. Ten consumers *do* route through `_get_palette` as the plan
says — but there is a whole second family it never mentions: `_seaborn_palette_arg`
(`plot.py:113`) plus **five raw seaborn call sites**, of which `sns.set_palette` (`plot.py:4825`)
runs on **every matplotlib plot call, unconditionally**.

The plan's own test file reports only 2 failures, which flatters it. Run as real `hyp.plot` calls,
the maintainer's six scenarios score **0/6** — every one dies on
`ValueError: 'image:…' is not a valid palette name`. The plan's suite misses four because its
"continuous hue" case calls `continuous_colormap()` directly instead of `hyp.plot()`, and its
"missing file" case asserts an error anyway.

**A third defect, in the plan's own test.** `test_palette_string_colours_a_categorical_hue` can
never pass, against any implementation: it harvests colours from `ax.collections`, but a `fmt='.'`
plot draws `Line2D` into `ax.lines`. The only collections on a 3-D axes are pane/grid artists with
**empty** facecolor arrays, so the filter empties the list and `np.vstack([])` raises. Measured: the
implementation was producing exactly the right colours (vivid `[0.863 0.078 0.078]` first); the
assertion was looking in the wrong place.

**Fix, with measurements.** Route through both interception points, and make the count dynamic via
the `continuous` flag `_get_palette` already carries — categorical/matrix extract exactly
`n_colors` (removing the 6-category cap); continuous keeps 6 anchors and lets the module's existing
short-list blending build the gradient. `IMAGE_PALETTE_N` survives, demoted from "the count" to
"the continuous-anchor count". For images with fewer distinct colours than requested: **deterministic
interpolation via `sns.blend_palette`**, not cycling — cycling would give two categories the same
colour, the exact ambiguity `_get_palette` already refuses for short user lists (`colors.py:332-335`),
and blending is already this module's answer to too-few anchors (`colors.py:323-331`). A
single-colour image raises, naming the file and three fixes. Verified bit-identical across 5 repeats;
most-salient anchor stays first at n = 2, 5, 9, 12; k-means costs 10-15 ms so no caching is needed.

Result: **0/6 → 6/6**, checked on rendered pixels not just objects (a 9-category PNG contains 9
distinguishable colours). Full suite **2788 passed, 13 skipped, 0 failed** — which reconciles exactly
with my baseline: 2784 collected + 19 new tests = 2803, and 2769 + 19 = 2788 passing.

Incidental find worth keeping: a first `-x` run failed
`test_packaging_artifacts.py::test_sdist_contains_only_tracked_files_plus_allowlist` because the new
test file was untracked. That is the guard working, not a false positive — but Task 1's Step 7 needs
a line saying to `git add` the new test before running the suite.

## Status

Seven parallel audits dispatched (reports land in `notes/audit/`):
`plan3_closure_audit.md`, `plan4_metric_remeasure.md`, `plan4_landed_state.md`,
`plan4_image_palette.md`, `plan4_notebook_gate.md`, `plan4_network_decoupling.md`,
`plan4_citations_and_ci.md`.
</content>
</invoke>
