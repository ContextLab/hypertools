# Adversarial review — Plan 4 v3 (`2026-07-28-hypertools-1.1-examples-and-tutorials.md`)

# VERDICT: **NOT IMPLEMENTABLE**

Reviewed at `09de97a9` in a disposable worktree (`git worktree add /tmp/p4v3_audit 09de97a9`), removed afterwards.
Every prescribed code block that could be extracted was extracted verbatim, applied to the real library, and run
with `/Users/jmanning/hypertools/.venv/bin/python`. Nothing below is reasoned about; each finding carries the
command and its real output.

**Findings: 4 Fatal · 5 High · 8 Medium · 6 Low.**

The plan's *library* work is sound — Task 1's 19 tests and Task 8 Step 0's 8 tests all pass against the real
codebase with no regressions (2813 passed / 13 skipped). The plan is blocked on Task 8's **gate module**, which
imports a symbol no step defines, calls two functions no step defines, and states a green result its own source
cannot produce.

---

# FATAL

## F1. `strip_docstrings` is never defined — 90 of the gate's 124 tests raise `ImportError`

Task 8 Step 2's `_code_text()` (plan L2819) does `from scripts.measure_native_ratio import strip_docstrings`.
Step 1's prescribed `measure_native_ratio.py` (plan L2552–2659) defines only `_code_lines_py`, `_code_lines_nb`,
`_depth_delta`, `measure`.

```
$ grep -n "^def " /tmp/p4v3_audit/scripts/measure_native_ratio.py
30:def _code_lines_py(path)
49:def _code_lines_nb(path)
62:def _depth_delta(line)
84:def measure(path)
```

```
$ .venv/bin/python -c "<import the prescribed module>; print(hasattr(m,'strip_docstrings'))"
has strip_docstrings? False
module names: ['HYP', '_code_lines_nb', '_code_lines_py', '_depth_delta', 'json', 'measure', 're', 'sys']
```

Real failure, prescribed source, real repo:

```
$ .venv/bin/python -m pytest tests/test_examples_are_native.py::test_a_docstring_naming_a_removed_pattern_is_not_a_defect -q
>       from scripts.measure_native_ratio import strip_docstrings
E       ImportError: cannot import name 'strip_docstrings' from 'scripts.measure_native_ratio'
        (/private/tmp/p4v3_audit/scripts/measure_native_ratio.py)
tests/test_examples_are_native.py:148: ImportError
```

Blast radius — every test routed through `_code_text`: `test_a_docstring_naming_a_removed_pattern_is_not_a_defect` (1),
`test_no_defect_marker_in_the_launch_examples` (80), `test_no_example_or_notebook_unpacks_then_uses_the_wrapper` (1),
`test_older_tutorials_dropped_their_hand_rolled_helpers` (6), `test_analyze_tutorial_actually_plots` (1),
`test_reduce_tutorial_mentions_describe` (1) = **90 of 124**.

Measured with the gate as written: `106 failed, 16 passed, 2 skipped`.
After I supplied an AST-based `strip_docstrings` shim: `40 failed, 82 passed, 2 skipped`.

## F2. The v2 Fatal that v3 claims to have fixed is **not fixed** — the prescribed metric still drifts

v3's revision note (L21): *"`_code_lines_nb` and `_code_lines_py` now share ONE docstring-stripping callee (a shared
callee cannot drift from itself). Re-measured: market 193→187, weather 207→194, conversation 191→176."*

The prescribed `_code_lines_py` strips docstrings inline; the prescribed `_code_lines_nb` skips only blanks and
comments. They share nothing. Identical source, both forms:

```
$ .venv/bin/python  # prescribed measure() on one module written as .py and as a 1-cell .ipynb
measure(.py)    = (3, 2)
measure(.ipynb) = (10, 2)
```

That is exactly the v2 defect ("identical source measured `(3,2)` vs `(11,2)`"), still present.

Decisive proof that the note's numbers were taken against a script that was never written into the plan — the
prescribed source reproduces the **pre-fix** numbers, and the **post-fix** numbers are what you get by applying
`_code_lines_py`'s heuristic to notebook code:

```
v3 note claims: market 193->187, weather 207->194, conversation 191->176
  market_forecast        prescribed= 193   py-heuristic= 187   claimed=187
  weather_decades        prescribed= 207   py-heuristic= 194   claimed=194
  conversation_shape     prescribed= 191   py-heuristic= 176   claimed=176
```

Consequence, and it is structural: notebook budgets are derived `script + NOTEBOOK_OVERHEAD` (5), with 2–7 lines of
headroom. The script side strips docstrings, the notebook side does not, so any notebook that inlines a documented
helper is over-counted against a budget that assumed it would not be. The class of unsatisfiable-budget failure v3
says it made "structurally impossible" is reintroduced by the metric.

## F3. `construct_artifact` and `fixture_data` are never defined by any step

`test_examples_produce_their_stated_artifact` (5 IDs) does
`anim = module.construct_artifact(module.fixture_data())` (plan L3036).

```
$ grep -c "def construct_artifact" <plan>   -> 0
$ grep -c "def fixture_data"       <plan>   -> 0
$ grep -c "fixture_data"           <plan>   -> 1     (the call site itself)
$ grep -rl "construct_artifact\|fixture_data" examples/ hypertools/ scripts/ tests/   -> (nothing)
```

```
$ .venv/bin/python -m pytest tests/test_examples_are_native.py::test_examples_produce_their_stated_artifact -q
E  AttributeError: module 'animate_conversation'     has no attribute 'construct_artifact'
E  AttributeError: module 'animate_market_forecast'  has no attribute 'construct_artifact'
   (… all five)
```

Contract 4 (L150) and the test docstring (L3010) both *describe* the `construct_artifact(data)` boundary, but no
step in Tasks 2–6 prescribes writing either function into any example. The plan's own "Placeholders: None" claim
(L3385) is false here.

## F4. Task 8 Step 3's stated result (`122 passed`) cannot be produced by the plan's own source

`EXPECTED_VISIBLE_OUTPUTS = {}` is prescribed empty (L3110–3112), and
`test_the_right_cells_carry_visible_output` calls `pytest.fail()` for any stem not in it. Five guaranteed failures:

```
$ .venv/bin/python -m pytest tests/test_examples_are_native.py::test_the_right_cells_carry_visible_output -q
FFFFF
E  Failed: market_forecast: no measured index set recorded. Execute the notebook and paste the measured set
   into EXPECTED_VISIBLE_OUTPUTS -- do not guess it ahead of the artifact (v2 guessed five and got all five wrong)
5 failed
```

The failure message is excellent and actionable (see Confirmed fixes #8). The defect is that Step 3 lists those
5 IDs inside a table totalling 124 and then states "**124 collected — 122 passed, 2 skipped**". Best case is
117 passed / 5 failed / 2 skipped. The only instruction to populate the dict lives in Tasks 2–6 (L1238, L1447,
L1690, L2088, L2220) — which execute *before* Task 8 Step 2 creates the file they are told to edit.

---

# HIGH

## H1. `_import_example_without_fetching` will fetch — its premise is false and `HYPERTOOLS_OFFLINE` is read by nothing

The helper's docstring (L2966–2972): *"Import must be side-effect-free: the example's `if __name__ == '__main__':`
guard runs the loaders, the module body only defines them. Setting HYPERTOOLS_OFFLINE makes any fetcher raise…"*

Both halves are false today.

```
$ grep -c "__main__" examples/animate_*.py
examples/animate_MDS.py:0          examples/animate_morph_zoo.py:0
examples/animate_conversation.py:0 examples/animate_painting_embeddings.py:0
examples/animate_market_forecast.py:0  examples/animate_trails_mix.py:0
examples/animate_plotly.py:0       examples/animate_surface_morph.py:0
examples/animate_spin.py:0         examples/animate_weather_decades.py:0
```

Loaders run at module scope: `examples/animate_morph_zoo.py:74` (`clouds = [load(name) for name in SHAPES]`),
`examples/animate_market_forecast.py:113` (`fetched = fetch_fred(FRED_IDS, START, END)`).

```
$ grep -rn "HYPERTOOLS_OFFLINE" examples/ hypertools/ scripts/ tests/   -> (nothing)
$ grep -c "HYPERTOOLS_OFFLINE" <plan>                                   -> 3   (all three inside the gate helper)
```

Nothing reads the variable, so setting it changes nothing. No task adds a `__main__` guard or offline handling.
As written, `spec.loader.exec_module(module)` in the **default** pytest suite downloads Dropbox shape files, FRED
CSVs and HuggingFace models — precisely what Contract 4 and this test's own docstring say was fixed. Proof the
network is genuinely reached at module scope (cold cache, sockets refused):

```
$ HOME=/tmp/p4v3_home PYTHONPATH=<socket-blocker> MPLBACKEND=Agg .venv/bin/python examples/animate_morph_zoo.py
[BLOCKED-DNS #1] ('www.dropbox.com', 443)   … #2 #3 #4
hypertools.core.exceptions.HypertoolsIOError: Failed to download 'bunny' dataset
exit=1
```

## H2. Contract 8's gate misses the idiom Contract 8 itself blesses, and the three accessors Step 0 just added

Regex under test: `^\s*\w+\s*,\s*(\w+)\s*=\s*(?:hyp|hypertools)\.plot\(` + `\b{name}\.on_frame\s*\(`.

```
A. THE BUG (should flag)                     -> unpacked=['ani']  FLAGGED=['ani']   ✓
B. market idiom (should NOT flag)            -> unpacked=[]       FLAGGED=[]        ✓
C. anim = hyp.plot(); fig, ani = anim;
   ani.on_frame(cb)   (IS the same bug)      -> unpacked=[]       FLAGGED=[]        ✗ MISSED
D. fig, ani = hyp.plot(); ani.draw_frame(0)  -> unpacked=['ani']  FLAGGED=[]        ✗ MISSED
E. fig, ani = hyp.plot(); ani.n_frames       -> unpacked=['ani']  FLAGGED=[]        ✗ MISSED
F. res = hyp.plot(); ani = res[1]; ani.on_frame(cb) -> FLAGGED=[]                   ✗ MISSED
G. fig, ani = ht.plot(); ani.on_frame(cb)    -> FLAGGED=[]                          ✗ MISSED
```

The two cases the review brief asked about are correct (A flags, B does not). But **case C is the same
`AttributeError`**, and it is one line away from the idiom Contract 8 recommends
(`examples/animate_market_forecast.py:191` `anim = hyp.plot(...)`, `:195` `fig, ani = anim`). Confirmed on the
real object:

```
$ anim = hyp.plot(d,'-',animate=True,...); fig, ani = anim
on_frame   -> ABSENT on the unpacked name
n_frames   -> ABSENT     n_segments -> ABSENT     draw_frame -> ABSENT
_save_count survives unpack -> True
```

Cases D/E matter specifically because **this revision added `n_frames`, `n_segments` and `draw_frame` as
`HyperAnimation`-only members** (Step 0) and then wrote a gate that greps for `.on_frame(` alone. v3 widened the
trap and left the guard where v2 had it.

## H3. Task 8 Step 7 still states v2's numbers, contradicting three other places in the same plan

| plan location | Task 1 | Task 5 | Task 8 | total |
|-|-|-|-|-|
| Revision note L29 | 19 | 12 | 132 | **+163** |
| Step 5 L789 / Step 7 L846 | 19 | — | — | — |
| Step 3 L3230 | — | — | 132 | — |
| **Task 8 Step 7 L3290** | **16** | 12 | **106** | **+134** |
| Self-Review L3371/3379/3391 | **17** | — | **109** | **+126** |

Step 7 is the *final full-suite gate*. An implementer following it would see a correct run and conclude the suite
is wrong. Verified real counts (all three by real collection, not arithmetic):

```
$ .venv/bin/python -m pytest tests/plot/test_image_palette.py -q          -> 19 passed
$ .venv/bin/python -m pytest tests/test_examples_are_native.py --collect-only -q -> 124 tests collected
$ .venv/bin/python -m pytest tests/plot/test_hyper_animation_accessors.py -q     -> 8 passed
$ .venv/bin/python -m pytest tests/plot/test_recency_fade.py --collect-only -q   -> 12 tests collected
```

So 19 / 12 / 132 (= 124 + 8) and +163 are the correct figures; L3290 and the Self-Review are stale.

## H4. `test_every_allowlisted_reach_is_still_present_and_still_explained` — the "still explained" half is near-vacuous

The assertion greps the **whole file** for two words: `assert 'deliberately' in raw or 'no public' in raw`.
Both allowlist entries target the same file, so one occurrence anywhere satisfies both. Demonstrated by mutation:

```
MUT1 (every real rationale destroyed; one unrelated
      "# colors chosen deliberately for print" prepended):  PASS
MUT3 (both words removed):                                  FAIL: unexplained
```

So it is not a pure tautology (it *can* fail), but it does not test what its docstring claims — "the source must
explain itself where a reader will find it". There is no proximity check, no per-entry check, and no requirement
that the two reaches have two rationales.

The first assertion is weak in the mirror direction: it greps `_read(path)` (raw) while the ban test greps
`_code_text(path)` (stripped). A dead allowlist entry therefore survives on a docstring mention alone —
`examples/animate_market_forecast.py:34` mentions `ani._args` in prose:

```
MUT2 (all live ani._args reaches removed from code; only the :34 docstring mention left):
     -> ani._args still present in raw? True   (entry NOT reported as dead weight)
```

## H5. The prescribed `n_segments` docstring is factually wrong, and contradicts its own test

Implementation docstring (plan L2512–2515): *"`n` clouds give `2n` segments -- one hold and one transition each,
**the closing transition back to the first cloud included**."*
Test docstring (plan L2443–2447): *"`n` clouds give `2n - 1` segments … There is NO implicit closing transition."*

Measured — the test is right, the shipped docstring is wrong:

```
n=2: len(frame_counts)=3   2n-1=3   2n=4
n=3: 5   n=4: 7   n=5: 9   n=6: 11   n=7: 13
2 clouds -> n_segments=3   3 -> 5   5 -> 9   6 -> 11
```

Per the repo rule that docs travel with code, this ships a wrong public docstring. No test asserts it, so nothing
catches it.

---

# MEDIUM

## M1. `test_no_notebook_budget_is_below_its_own_scripts` cannot fail

`BUDGETS = [(p, n) …] + [(NOTEBOOKS[p], n + NOTEBOOK_OVERHEAD) …]` with `NOTEBOOK_OVERHEAD = 5`. The test asserts
`limits[nb] >= limits[script]`, i.e. `n + 5 >= n`. True for every `n`, unconditionally. The plan itself says the
derivation makes "the error class **structurally impossible**" (L22) — which is precisely why the assertion is
inert. It is the same defect class flagged in v2 (`_save_count >= 1`), in a new location. Keep it only if
`NOTEBOOK_OVERHEAD` is ever allowed to be negative; otherwise it is a comment wearing a test's clothes.

## M2. The equal-width citation the *Verification note* corrects is itself stale

Plan L62: *"The comment block starts at `plot.py:2748-2756`; the **check** is `plot.py:3152-3153`
(`_widths = [ri.shape[1] for ri in raw]` / `if len(set(_widths)) > 1:`). **Cite 2750-2751**."*

```
$ sed -n '3152,3153p' hypertools/plot/plot.py
        if resample:
            from ..manip.manip import manip as _manip
$ grep -n "_widths = \[ri.shape\[1\]" hypertools/plot/plot.py
3164:        _widths = [ri.shape[1] for ri in raw]
3165:        if len(set(_widths)) > 1:
$ sed -n '2748,2756p' hypertools/plot/plot.py
        resolved_focused = focused
    …
    # predict= + animate: a forecast is a FIXED overlay, …
```

Real location is `3164-3165`. `2748-2756` is now `focused`/predict-animate prose, and the plan cites it *twice*
more (Prerequisites table L235: "the call raises `NotImplementedError` (`plot.py:2748-2756`)"; L62).

## M3. `test_each_notebook_ships_its_rendered_artifact` passes today — the catch it advertises does not exist

Docstring: *"…which also catches morph_shapes_zoo.ipynb embedding `morph_zoo.gif` rather than
`morph_shapes_zoo.gif`."*

```
market_forecast      refs=['market_forecast.gif']       exists=[True]
weather_decades      refs=['weather_decades.gif']       exists=[True]
painting_embeddings  refs=['painting_embeddings.gif']   exists=[True]
conversation_shape   refs=['conversation_shape.gif']    exists=[True]
morph_shapes_zoo     refs=['morph_zoo.gif']             exists=[True]
$ pytest …::test_each_notebook_ships_its_rendered_artifact -q  -> 5 passed
```

The reference *does* resolve; the mismatch is a naming inconsistency, not a broken link, and this test cannot
detect it. Fine as a regression control — but it is listed as one of the three notebook gates and does not gate
anything today.

## M4. `NOTEBOOK_OVERHEAD = 5`'s "2-line display cell" is not measured

Claim (L166, and the constant's comment L2711–2714): *"the largest install cell across the five is 3 code lines,
plus a 2-line display cell (`from IPython.display import HTML` + `HTML(ani.to_jshtml())`)."*

Install half verified (paintings and conversation are 3; the rest 2). Display half is not present anywhere:

```
$ grep -l "to_jshtml\|IPython.display" docs/tutorials/{market_forecast,weather_decades,painting_embeddings,conversation_shape,morph_shapes_zoo}.ipynb
NONE
```

So 3 of the 5 lines are measured and 2 are assumed about notebooks that do not exist yet. With headroom of 2 lines
on conversation, that assumption is load-bearing.

## M5. The "Measured baseline" table contradicts Task 6 and overstates the headline total

L85: `examples/animate_morph_zoo.py` **40** code lines. L2110: *"script ≤ 30 code lines — already met, at 26."*
Real, with the plan's own prescribed script:

```
$ .venv/bin/python scripts/measure_native_ratio.py examples/animate_*.py
animate_conversation.py        code= 165 native=  9  (plan: 166/9)
animate_market_forecast.py     code= 191 native= 11  (plan: 191/11)  ✓
animate_morph_zoo.py           code=  26 native=  6  (plan:  40/6)   ✗
animate_painting_embeddings.py code= 146 native= 11  (plan: 146/11)  ✓
animate_weather_decades.py     code= 195 native= 11  (plan: 196/11)
```

Total is 723, not 739, so the headline "48 of 739 code lines" / 6.5% is 48/723 = 6.6%. The table is labelled
"the numbers every task below is held to" with no note that it predates `d730a085`.

## M6. The opt-in smoke test is promised and never created

`HYPERTOOLS_EXAMPLE_SMOKE=1` appears exactly once in the plan (L3015, inside a docstring): *"The whole-example run
survives as an opt-in smoke test (`HYPERTOOLS_EXAMPLE_SMOKE=1`), never in the default suite."* No step writes it.
Combined with F3, the whole-example coverage v2 had via `runpy` is removed and nothing replaces it.

## M7. `_hyp_morph_segments` is tagged in "the morph branch" — there are two

L2535: *"tagged … in `matplotlib_backend`'s morph branch"* (singular). There are two `FuncAnimation` morph
constructions: 3-D at `hypertools/plot/matplotlib_backend.py:2036` (`update_morph`) and 2-D at `:2447`
(`update_morph_2d`). I patched both to make Step 0's tests green. Tagging only the 3-D one leaves
`anim.n_segments is None` for every 2-D morph, and `test_n_segments_is_none_for_a_non_morph_animation` would then
pass on a 2-D morph too — a silently wrong answer, not a failure.

## M8. Task 1 is declared dependency-free but its Step 6 requires a heading only Plan 1 creates

Prerequisites table L234: *"**Task 1** … *(none)* … Can start immediately, in parallel with Plans 1–3."*
Step 6 L830: *"In `CHANGELOG.md`, under the `## 1.1.0 (unreleased)` → `### Added` heading created by the
animation-core plan."*

```
$ grep -n "1.1.0\|### Added" CHANGELOG.md   -> (nothing)
```

Task 1 cannot complete Step 6 standalone.

---

# LOW

## L1. morph's offline exit code is 1, not 17

Contract 4's table (L144): *"**HARD FAILS — `HypertoolsIOError`, exit 17**"*. Reproduced with a cold cache and
refused sockets: 4 blocked events to `www.dropbox.com`, `HypertoolsIOError: Failed to download 'bunny' dataset`,
**exit=1**. Everything except the exit code is exactly right.

## L2. `test_every_launch_notebook_ships_executed_outputs` is cited twice and does not exist

L3349 and L3397 both name it as the guard that makes skipping notebook execution a test failure. The prescribed
test is `test_every_launch_notebook_ran_every_cell_it_should` (L3116). Stale symbol.

## L3. `test_native_ratio_is_reported`'s only assertion is near-vacuous

`assert code > 0, f'{path}: no code lines found -- moved or renamed?'`. A *missing* file raises `FileNotFoundError`
inside `measure()` before the assert is reached, so the message's stated purpose is unreachable; for a present
file, `code > 0` holds for anything non-empty. The plan is honest that the ratio is reported, not gated — noted for
completeness only.

## L4. The stated baseline is stale (informational — the plan says to re-measure)

Plan L191: `2782/2784 tests collected (2 deselected)` at `065c841e`.
```
$ .venv/bin/python -m pytest --collect-only -q   ->  2799/2801 tests collected (2 deselected)
```
The plan explicitly instructs re-measuring at start, so this is correct process, wrong number.

## L5. `scripts/__init__.py` does not exist yet

The import note (L3200–3204) is accurate: `from scripts.measure_native_ratio import measure` resolves today as a
PEP 420 namespace package, verified by real collection (124 collected, no `ModuleNotFoundError`). The note's
instruction to add the file anyway is sound and should not be dropped.

## L6. Not verified — the zero-warning docs build

Task 1 Step 8 and Task 8 Step 8 (`sphinx -b html -W -E -a`, 0 warnings) were **not run** — the gallery build is
long and the tree already diverges from the plan's end state. Stated explicitly rather than guessed. In particular
I did not verify that the three new `.. autofunction::` directives in a `docs/api.rst` section that otherwise uses
`autosummary`/`:toctree:` build warning-free.

---

# Prior-revision fixes CONFIRMED genuinely fixed

| # | claim | evidence |
|-|-|-|
| 1 | Task 1's palette work is real and correct | 19/19 pass against the real library: `pytest tests/plot/test_image_palette.py -q` → `19 passed in 2.27s`. Existing colour tests unaffected: `tests/test_colors.py tests/plot/test_colors_module.py tests/test_colorbar.py` → `36 passed`. |
| 2 | **Two** interception points are genuinely required (v2's "ONE point" was false) | Reverting only patch 3 (`_seaborn_palette_arg`) and re-running: `5 failed, 14 passed`, every failure `ValueError: 'image:…' is not a valid palette name` from `seaborn/palettes.py:237`. With both patches: 19/19. The dynamic colour count and `blend_palette` interpolation both work — the 9-category test yields 9 distinct colours, the two-tone image yields 5 distinct interpolated colours led by the vivid anchor, and a single-colour image raises `single dominant color`. |
| 3 | Step 0's accessors work; `n_segments` really is `2n-1` | `pytest tests/plot/test_hyper_animation_accessors.py -v` → `8 passed`. `morph_schedule` for n=2..7 → 3/5/7/9/11/13 = `2n-1`. `_hyp_morph_segments` is tagged where the plan says (beside `sum(frame_counts)` in `matplotlib_backend`'s morph branch) — see M7 for the second branch. |
| 4 | Task 8 gate collects exactly 124, with 2 real allowlist skips | `pytest tests/test_examples_are_native.py --collect-only -q` → `124 tests collected`. Run shows `2 skipped` (the `ani\._args` / `hypertools\._shared` pairs on `animate_market_forecast.py`), each printing its recorded reason. |
| 5 | Task 5 = 12, not 13 | `pytest tests/plot/test_recency_fade.py --collect-only -q` → `12 tests collected` (8 defs, 2 parametrized ×3). All 12 fail today with `KeyError: 'recency_fade'` — correct red state for an unimplemented rewrite. |
| 6 | Task 1 + Step 0 land cleanly on the real suite | `pytest -q` (gate + recency_fade deselected) → **`2813 passed, 13 skipped, 138 deselected in 558.71s`**. Baseline 2799 + 19 + 8 = 2826 = 2813 + 13. **Zero regressions.** |
| 7 | The Contract 8 unpack regex does what the brief asked about | Case A (`fig, ani = hyp.plot(...)` then `ani.on_frame(...)`) **is** flagged; the market idiom (`anim = hyp.plot(...)` then `fig, ani = anim`) is **not** flagged. Both verified by running the regex. (Gaps: H2.) |
| 8 | `test_the_right_cells_carry_visible_output` fails loudly and actionably for all five | `FFFFF`, message names the notebook, says exactly what to do, and cites why guessing is banned. Best failure message in the plan. (But see F4 — it makes Step 3's stated total impossible.) |
| 9 | The corrected executed-output counts are exact | market 4/7, weather 2/7, paintings 2/6, conversation 2/6, morph 1/6 — measured, matches L31/L61 exactly (v1's "0 for all five" was indeed wrong). |
| 10 | "9 of the 20 notebooks ship the install cell executed" is exact | 20 notebooks, all 20 have an install cell, exactly 9 executed. `_is_install_cell` detecting by content rather than index is correct. |
| 11 | The coverage-vs-control annotation is exact | `test_older_tutorials_dropped_their_hand_rolled_helpers`: `stock_forecasting` PASSED, `projectile_kalman` PASSED (controls); the other four FAILED; `modern_sklearn_dynamics` has `ffmpeg=True, SentenceTransformer=False` — fails on ffmpeg only, exactly as annotated. |
| 12 | Network-coupling measurements are real | morph (cold cache): 4 blocked → `www.dropbox.com`, `HypertoolsIOError`, hard fail. market (cold cache + cold `TMPDIR`): exactly **1** blocked → `fred.stlouisfed.org`, degrades to `synthetic basket (offline fallback)`, exit 0. Both match the table. |
| 13 | The 11-segment morph analysis is exact | `SHAPES` = 5 (`:47`), `clouds.append(clouds[0])` (`:78`) → 6 clouds → 11 segments (measured). `rotations = [0.75] + [0.5, 1.0]*(len(SHAPES)-1) + [0.5, 0.75]` = 11 entries. The stale inline comment "for the 5 clouds = 9 segments" (`:81`) is present exactly as the plan describes. |
| 14 | v2's two tautologies are genuinely gone | No `_save_count` and no `'morph' in str(ns.get('ANIMATE','morph'))` anywhere in v3's gate; replaced by `n_frames >= min_frames` floors and `n_segments == 11`. Both would discriminate — once F3 supplies `construct_artifact`. |
| 15 | The `git stash` data-loss hazard is gone | No `git stash` anywhere in v3 except the one place it survives: **Task 8 Step 1's verification command still uses it** — `git stash && … measure_native_ratio.py … && git stash pop` (L2665). Global Constraints L192 bans exactly this. *(Reported as a residual, not a confirmed fix.)* |
| 16 | Line citations spot-checked | `docs/conf.py:131` = `nbsphinx_execute = 'never'` ✓; `colors.py:305-306` ✓; `plot.py:113` = `def _seaborn_palette_arg` ✓; `plot.py:4825-4826` = `sns.set_palette(... _seaborn_palette_arg ...)` ✓; `plot.py:1066` = the `palette :` docstring entry ✓; `scripts/generate_gallery_thumbs.py:26` = `MPL_ANIMS = [...]` ✓; `docs/api.rst:108-116` = the Plot section ✓; `pyproject.toml:54` = `"seaborn>=0.13.0",` ✓; the five raw-seaborn call sites are at 208 / 4118 / 4657 / 4767 / 4825 ✓. Only the equal-width citation is stale (M2). |

---

## Note on #15 — a v2 hazard that survives into v3

Global Constraints L192 states the rule emphatically: *"Reading a file's BEFORE state: use `git show <base>:<path>`,
never `git stash`. … That is a **data-loss hazard**, demonstrated end-to-end."* Task 8 Step 1 (L2664–2667) then
prescribes:

```bash
git stash && .venv/bin/python scripts/measure_native_ratio.py examples/animate_conversation.py && git stash pop
```

This is the exact command the plan bans, on the exact hazard the plan documents, in the same plan. I did **not**
execute it (it would drop a stash in the host repo). Its expected output is also stale: it says
`code= 166 native= 9`; the real file measures `code= 165 native= 9`.

## Minimum work to make this plan implementable

1. Write `strip_docstrings` into Task 8 Step 1 and make **both** `_code_lines_py` and `_code_lines_nb` call it
   (F1 + F2). Re-derive every notebook budget afterwards — the current derivations assume a stripping notebook
   counter.
2. Add `fixture_data()` and `construct_artifact(data)` to the prescribed source of all five examples in Tasks 2–6,
   with an `if __name__ == '__main__':` guard, and implement the `HYPERTOOLS_OFFLINE` contract in each fetcher
   (F3 + H1).
3. Move "record the measured index set into `EXPECTED_VISIBLE_OUTPUTS`" after Task 8 Step 2, and restate Step 3's
   expectation as `117 passed / 5 failed / 2 skipped` until Tasks 2–6 populate it (F4).
4. Extend the Contract 8 regex to chained unpacks and to `draw_frame` / `n_frames` / `n_segments` (H2).
5. Reconcile Step 7 and the Self-Review with 19 / 12 / 132 / +163 (H3).
6. Fix the `n_segments` docstring to `2n - 1` (H5) and tag both morph branches (M7).
7. Replace Task 8 Step 1's `git stash` command with `git show <base>:<path>` (#15).
