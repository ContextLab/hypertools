# Adversarial RE-review — Plan 4 v3 after the 23-finding fix pass

# VERDICT: **NOT IMPLEMENTABLE** (but much closer — the gate module now runs)

Reviewed at `1810746b` (`docs(plans): Plan 4 v3 - fix L1-L4`) in a disposable worktree
(`git worktree add /tmp/p4v3_recheck HEAD`), removed afterwards. Every prescribed code block was
**extracted verbatim by line range and executed** with `/Users/jmanning/hypertools/.venv/bin/python`.
Nothing below is reasoned about; each finding carries the command and its real output.

**Findings: 3 Fatal · 3 High · 7 Medium · 6 Low.**
**Prior findings: 13 CONFIRMED FIXED · 8 PARTIAL · 1 NOT FIXED · 1 rejection UPHELD.**

The fix pass is real work, not paper. F1, F2, H1, H4, H5, M1, M3, M4, M6, M7, L1, L2, L3 are
**genuinely fixed and I ran the code that proves it**: the gate now collects 138 with zero
`ImportError`, identical source measures identically as `.py` and `.ipynb`, all five re-measured
notebook numbers reproduce to the digit, the Contract-8 guard catches all eight cases the prior
review named, the allowlist rationale check now fails when the rationale is moved away, and the
suite counts 19 / 12 / 147 are each confirmed by real collection.

**What still blocks it is a single structural gap and one new defect in the fix itself:**

1. **F3 was answered with a definition, not an implementation.** Step 0b defines
   `construct_artifact` / `fixture_data` / `load_*` / the `__main__` guard and says *"Tasks 2–6 each
   WRITE them"* — but those four symbols appear **nowhere in Tasks 2, 3, 4, 5 or 6**. An implementer
   following those tasks verbatim still produces five example files without them, and
   `test_examples_produce_their_stated_artifact` still fails on all five, exactly as it does today.
2. **Step 0c mandates a budget renegotiation that was never performed**, so weather's budget is now
   *provably* unsatisfiable (56 prescribed + 15 measured split overhead = 71 against `≤ 62`). That is
   the same unsatisfiable-budget class the plan says it made "structurally impossible".
3. **The shared `strip_docstrings` callee that the F1/F2 fix introduced is itself wrong.** Any line
   whose stripped form *begins* with `"""` or `'''` flips it into docstring mode — including the
   closing quote of an ordinary multi-line string. Everything after is silently dropped. Demonstrated
   on 7 real files in this repo; `tests/test_backend_state_safety.py` loses **123 of its 195 code
   lines**. Both the size budget and the defect-marker ban go blind in that region.

---

# What was verified GREEN (commands and real output)

| claim | command | real output |
|-|-|-|
| Gate collects 138 | `pytest tests/test_examples_are_native.py --collect-only -q` | `138 tests collected in 0.06s` |
| Gate runs with no `ImportError` (F1) | `pytest tests/test_examples_are_native.py -q` | `40 failed, 91 passed, 7 skipped in 0.46s` |
| 7 skips = 2 allowlist + 5 opt-in smoke (M6) | same run | 7 skipped, confirmed by ID |
| Task 1 = 19 | Task 1 Steps 3–4 patches applied, then `pytest tests/plot/test_image_palette.py -q` | `19 passed in 2.19s` |
| Task 5 = 12 | `pytest tests/plot/test_recency_fade.py --collect-only -q` | `12 tests collected in 8.30s` |
| Step 0 = 9 | Step 0 accessors + both morph tags applied, `pytest tests/plot/test_hyper_animation_accessors.py -q` | `9 passed in 1.93s` |
| Task 8 = 147 | 138 + 9 | exact |
| Suite delta +178 | 19 + 12 + 147 | exact |
| No colour regressions from Task 1 | `pytest tests/test_colors.py tests/plot/test_colors_module.py tests/test_colorbar.py -q` | `36 passed in 3.11s` |
| No library regressions from Step 0 | `pytest -q -k "morph or animation or hyper_anim"` | `10 failed, 415 passed, 1 skipped` — **all 10 failures are the un-landed gate module**, zero library failures |

Every row of Step 3's 138-ID derivation table matches real collection:

```
$ pytest tests/test_examples_are_native.py --collect-only -q | sed 's/\[.*//' | sort | uniq -c
  80 test_no_defect_marker_in_the_launch_examples
  10 test_no_example_or_notebook_unpacks_then_uses_the_wrapper
  10 test_file_is_within_its_size_budget
   6 test_older_tutorials_dropped_their_hand_rolled_helpers
   5 test_the_right_cells_carry_visible_output
   5 test_examples_produce_their_stated_artifact
   5 test_example_runs_end_to_end
   5 test_every_launch_notebook_ran_every_cell_it_should
   5 test_each_notebook_ships_its_rendered_artifact
   1 x 6 singletons
```

**The metric fix (F2) is verified end to end.**

```
$ # identical source written as .py and as a 1-cell .ipynb
measure(.py)            = (3, 2)
measure(.ipynb)         = (3, 2)      EQUAL: True
measure(2-cell .ipynb)  = (3, 2)      (v2/v3-pre-fix gave (10,2) / (11,2))

$ # _code_lines_py invariance across the refactor (old = the 09de97a9 inline version)
animate_conversation.py           old= 165 new= 165 identical=True
animate_market_forecast.py        old= 191 new= 191 identical=True
animate_morph_zoo.py              old=  26 new=  26 identical=True
animate_painting_embeddings.py    old= 146 new= 146 identical=True
animate_weather_decades.py        old= 195 new= 195 identical=True
ALL FIVE identical: True    hypertools/*.py files differing: 0 (of 220)

$ .venv/bin/python scripts/measure_native_ratio.py docs/tutorials/*.ipynb
market_forecast.ipynb       code= 187 native= 11   (plan: 187) OK
weather_decades.ipynb       code= 194 native= 11   (plan: 194) OK
painting_embeddings.ipynb   code= 121 native= 11   (plan: 121) OK
conversation_shape.ipynb    code= 176 native= 11   (plan: 176) OK
morph_shapes_zoo.ipynb      code=  46 native=  9   (plan:  46) OK

$ # and the pre-fix notebook counter reproduces the note's BEFORE numbers exactly
market 193   weather 207   conversation 191     (revision note L21: 193->187, 207->194, 191->176)
```

**H5 / M7 verified on the real object** (Step 0 implementation applied, both morph branches tagged):

```
n=2: n_segments=3   n=3: 5   n=4: 7   n=5: 9   n=6: 11   n=7: 13      2n-1 in every case
6 clouds (5 shapes + closing repeat) -> 11    STATED_ARTIFACT morph=11 correct
tests/plot/test_hyper_animation_accessors.py -> 9 passed (incl. the 2-D morph case)
```

**H1 verified**: the gate no longer fetches. The whole 138-ID run finishes in **0.46 s** and each
of the five artifact IDs fails at the pre-exec assertion:

```
E  AssertionError: examples/animate_conversation.py has no __main__ guard, so importing it
   would run its loaders and hit the network (Step 0b). Do the loader / construct_artifact
   split before enabling this gate.
   tests/test_examples_are_native.py:482: AssertionError
```

---

# FATAL

## N1. F3 is answered by a step no task performs — `construct_artifact` / `fixture_data` are still never written

Step 0b (L2572) states the contract in bold: *"This step defines two functions per example.
**Tasks 2–6 each WRITE them**; nothing else in the plan defines them."* Tasks 2–6 do not.

```
$ grep -n "construct_artifact\|fixture_data\|load_weather\|load_market\|load_paintings\|
          load_shapes\|embed_turns\|__main__" <plan>
152   Contract 4 (prose)
955   scripts/execute_tutorial.py            <- Task 2 Step 1, not an example
2576-2665   Task 8 Step 0b                   <- the definition
2799  scripts/measure_native_ratio.py __main__
3274-3363   Task 8's gate                    <- the call sites
```

Line 955 is the only `__main__` between L860 and L2400 (the entire span of Tasks 2–7) and it is
inside `scripts/execute_tutorial.py`. **Zero occurrences in Task 2, 3, 4, 5 or 6.** Their prescribed
"replace the file entirely" source blocks contain no `NamedTuple`, no `load_*`, no
`construct_artifact`, no `fixture_data`, no guard.

Step 0b's worked example is weather only, and its bodies are ellipses:

```python
def load_weather(cities=CITIES):
    """The two existing fetch loops, now named."""
    ...
def construct_artifact(data):
    """Everything from `min_len = ...` to `anim.on_frame(decorate)`, verbatim, ..."""
    ...
```

Consequences, all measured:

- `test_examples_produce_their_stated_artifact` (5 IDs) fails today and still fails after Tasks 2–6.
- `_import_example_without_fetching`'s `HYPERTOOLS_OFFLINE` contract is prescribed only in Step 0b
  prose; nothing in Tasks 2–6 adds it to a fetcher (`grep -rn HYPERTOOLS_OFFLINE examples/
  hypertools/ scripts/ tests/` → still nothing).
- The **paintings 1.7 KB fixture thumbnail** that Step 0b's table requires is never created,
  committed, or `git add`ed anywhere in the plan.
- The Self-Review's *"**Placeholders.** None. Every step carries runnable code"* (L3758) is false
  for exactly this material.

## N2. Step 0c mandates a budget renegotiation that was never applied — weather is provably unsatisfiable

Step 0c (L2669): *"Measured on weather: **+15 code lines** (195 → 210) … So each of Tasks 2–6
raises its script budget by its own measured split overhead … Record each measured overhead in
`SCRIPT_BUDGETS`."*

Neither happened. The prescribed `SCRIPT_BUDGETS` (L2882–2891) still holds the pre-split numbers,
and every task's AFTER line still quotes them:

| example | AFTER line | `SCRIPT_BUDGETS` | prescribed content | + weather's measured split | verdict |
|-|-|-|-|-|-|
| weather (L1265) | `≤ 62` | 62 | **56** | 56 + 15 = **71** | **71 > 62 — FAILS** |
| conversation (L1718) | `≤ 90` | 90 | 88 (2 lines headroom) | 88 + ≥10 | FAILS |
| morph (L2110) | `≤ 30` | 30 | 26 (4 lines headroom) | 26 + ≥8 | FAILS |
| market (L874 region) | `≤ 115` | 115 | 109 | 109 + ≥10 | FAILS |
| paintings (L1472) | `≤ 118` | 118 | 111 | 111 + ≥10 (2 fetch sites) | borderline/FAILS |

The split's cost does not shrink with a shorter body — the `NamedTuple` (6), two `def` lines,
`load_*` scaffolding and the 4-line `__main__` guard are fixed overhead, which is why Step 0c
measured them as a *constant* +15 on weather.

Step 3 then predicts **`126 passed, 5 failed, 7 skipped`** with all 10
`test_file_is_within_its_size_budget` IDs green and all 5
`test_examples_produce_their_stated_artifact` IDs green. Neither is reachable. This is the
"unsatisfiable budget" Fatal from v2, reintroduced by v3's own remedy for F3, and it is the exact
error class Contract 6 says *"a budget is renegotiated **in the plan**, never weakened in the test."*

## N3. NEW — `strip_docstrings`, the shared callee the F1/F2 fix introduced, silently swallows code

`strip_docstrings` enters docstring mode on **any** line whose stripped form starts with a triple
quote and does not also end with one — which includes the closing quote of an ordinary multi-line
string, and a triple-quoted continuation line. Everything after is dropped until the next line
containing a triple quote.

```python
import hypertools as hyp

TEXT = """
a multi-line string constant
"""                                # <- stripped == '"""', len == 3 -> in_doc = True

ani._func = evil                   # a REAL defect marker, after the string closed
anim = hyp.plot(TEXT)
```

```
$ list(strip_docstrings(SRC.splitlines()))
    ['import hypertools as hyp', 'TEXT = """', 'a multi-line string constant']

ani._func visible to the ban scan?  False
hyp.plot  visible to the metric?    False
code lines counted: 3   (real code lines = 8)
```

Not hypothetical. Compared against an AST-accurate implementation of the plan's own CODE-line
definition across 334 real `.py` files in this repo:

```
tests/test_backend_state_safety.py    strip=  72  ast-truth= 195  delta= -123
tests/test_density.py                 strip= 304  ast-truth= 475  delta= -171
tests/test_backend_headless.py        strip=  38  ast-truth=  85  delta=  -47
tests/test_format_data_f08_fixes.py   strip= 183  ast-truth= 214  delta=  -31
tests/test_gensim_text.py             strip= 232  ast-truth= 242  delta=  -10
tests/test_lsl_streaming.py           strip= 168  ast-truth= 178  delta=  -10
tests/test_autoencoders.py            strip= 160  ast-truth= 165  delta=   -5
files disagreeing: 7 of 334
```

The exact trigger, located in a real file:

```
$ tests/test_backend_headless.py:48      "    ''').format("     <- ENTERS doc-mode
$ context:  script = textwrap.dedent('''  ...  ''').format(
```

None of the **ten currently gated files** trigger it, so the gate is green today — but
`scripts/measure_native_ratio.py` is committed as a general-purpose tool (`measure_native_ratio.py
<any file>`), all three counters route through this one function by design, and the five examples
are about to be rewritten. A file that trips it gets an under-counted size budget **and** an
invisible defect-marker scan simultaneously. The function's own docstring claims it *"Drops … any
bare triple-quoted **docstring** block"*; it actually drops any block delimited by lines *starting*
with a triple quote, docstring or not.

---

# HIGH

## N4. The "shared callee cannot drift from itself" claim holds for the callee and fails at the call sites

`_code_lines_nb` resets the state machine per cell, with a comment insisting this is required:

```python
# Reset per cell: a bare docstring cannot span a cell boundary (each
# cell is parsed and executed independently), so carrying in_doc /
# delim across cells would be wrong, not merely unnecessary.
```

`_code_text` — the *other* consumer of the same callee — concatenates every code cell first and
calls `strip_docstrings` **once**, so it does exactly what `_code_lines_nb` calls wrong:

```
notebook: cell 1 = an unclosed bare """ note;  cell 2 = import + ani._func = evil + hyp.plot(d)

_code_text(nb)                    -> ''
defect marker visible to the ban scan?  False
measure(nb)                       -> (5, 2)      <- sees cell 2 fine
```

So on the same file `test_file_is_within_its_size_budget` counts 5 code lines while
`test_no_defect_marker_in_the_launch_examples` sees an empty file and passes unconditionally.
That is the F2 defect class — two counters disagreeing on identical input — relocated, in the very
code written to eliminate it. `_code_text`'s docstring (L2979-2982) asserts the opposite.

## N5. The Contract 8 guard is silently vacuous on any file `ast.parse` rejects, and nothing asserts it parsed

`_unpacked_wrapper_uses` returns `[]` on `SyntaxError`; `_docstring_lines` returns `set()`.
`_parsable_code` handles only lines whose first non-space character is `%` or `!`. Four of seven
realistic notebook idioms defeat it:

```
magic at cell start (%matplotlib inline)   parses            CAUGHT
magic MID-cell                             parses            CAUGHT
shell escape mid-cell (!pip)               parses            CAUGHT
cell magic %%bash with a shell body        SYNTAX ERROR      *** SILENTLY PASSED ***
indented magic inside a block (    %time)  SYNTAX ERROR      *** SILENTLY PASSED ***
help suffix (hyp.plot?)                    SYNTAX ERROR      *** SILENTLY PASSED ***
genuinely invalid python (py2 print)       SYNTAX ERROR      *** SILENTLY PASSED ***
```

The indented-magic case is the realistic one: commenting out `    %time foo()` leaves
`for i in range(3):` with an empty body. All ten gated files parse today, so this is latent — but
"an assertion that cannot fail" is the defect class this plan has now shipped four revisions
running, and there is no `assert it parsed` anywhere.

## N6. `_unpacked_wrapper_uses` false-positives on `ax.plot` / `df.plot`, and the trigger already exists in a gated file

`_is_plot_call` matches **any** attribute call named `plot`, deliberately ("matched on the attribute
name, so an import alias cannot evade it"). matplotlib's and pandas' `.plot` are the collateral:

```
FP1  ln, = ax.plot(x, y);  print(ln.figure)          -> [('ln','figure')]   FALSE POSITIVE
FP2  fig, ax = df.plot(subplots=True); ax.figure     -> [('ax','figure')]   FALSE POSITIVE
FP3  def a(): fig, ani = ax.plot(x)
     def b(): ani = hyp.plot(d); ani.on_frame(cb)    -> [('ani','on_frame')] FALSE POSITIVE (no scoping)
```

`Line2D.figure` and `Axes.figure` are real public matplotlib attributes, and `WRAPPER_ONLY`
includes `figure` and `animation`. The tuple-unpack-from-`ax.plot` form is **already in the repo, in
a file this gate covers**:

```
examples/animate_market_forecast.py:310   fc_line, = ax.plot([], [], [], '--', color=FC_COLOR, ...)
examples/animate_market_forecast.py:306   hist_lines = [ax.plot([], [], [], '-', ...
```

`fc_line` therefore already enters `unpacked`; the test passes only because nothing reads
`fc_line.figure` today. One `fc_line.figure` in the rewrite produces a failure whose message
(*"comes from unpacking a hyp.plot() result, so it is a raw FuncAnimation"*) is factually wrong, and
Global Constraints L196 forbids relaxing the test to clear it.

Twelve evasions also remain (see Low N12); the two that matter are plain-language plausible:

```
EV14  ani = hyp.plot(d).animation ; ani.on_frame(cb)     -> []  MISSED
EV15  a = hyp.plot(d); b = a; fig, ani = b; ani.on_frame -> []  MISSED
```

EV14 is arguably the *most* likely real form of the bug after direct unpacking, because `.animation`
is the documented property that hands back the raw `FuncAnimation`.

---

# MEDIUM

## N7. H3 only partly fixed — the Self-Review still states v1's number

Step 7 (L3661) and *Suite arithmetic* (L3764) are now correct and I confirmed all three components by
real collection (19 / 12 / 138+9=147 / +178). But:

```
L3752  "Task 8: a committed metric, a 109-test gate, a per-file re-measure, ..."
```

109 was v1's figure. It contradicts 138 stated twice elsewhere in the same document.

## N8. The banned `git stash` command survives, with a stale expected output

Global Constraints L194 bans it in bold and demonstrates the data loss. Task 8 Step 1 (L2810) still
prescribes it:

```bash
git stash && .venv/bin/python scripts/measure_native_ratio.py examples/animate_conversation.py && git stash pop
```
```
Expected on the untouched file: code= 166 native=   9 ratio=  5.4%
Real (measured with the plan's own prescribed script): code= 165 native= 9 ratio= 5.5%
```

Both halves wrong: the command is the one the plan forbids, and its expected output is stale. (I did
not execute it; it would drop a stash in the host repo.) `git stash list` also survives at L967.
This was residual #15 and item 7 of the prior review's "minimum work" list.

## N9. M5 only partly fixed — the baseline table is still labelled authoritative and is stale in 8 of 10 rows

The Goal line was corrected to 723 / 6.6%. The table at L81–95, headed *"the numbers every task
below is held to"*, was not:

```
                                    table   real (prescribed script)
examples/animate_conversation.py      166    165
examples/animate_morph_zoo.py          40     26
examples/animate_weather_decades.py   196    195
five scripts total                    739    723
conversation_shape.ipynb              186    176
market_forecast.ipynb                 192    187
morph_shapes_zoo.ipynb                 45     46
painting_embeddings.ipynb             116    121
weather_decades.ipynb                 206    194
```

Only market (191) and paintings (146) still hold. There is no note that the table predates
`d730a085` or the metric fix — the correction lives 700 lines away, in the Goal paragraph.

## N10. Commit steps omit files the plan's own Global Constraints say will break the suite

Global Constraint L195 (*"New test files must be `git add`ed before running the full suite"*) is real
— I reproduced it:

```
$ pytest tests/test_packaging_artifacts.py -q
FAILED test_sdist_contains_only_tracked_files_plus_allowlist
1 failed, 12 passed
```

Task 8 Step 10's `git add` (L3686) lists five paths and omits:
`tests/plot/test_hyper_animation_accessors.py` (Step 0 creates it),
`hypertools/plot/hyper_animation.py` and `hypertools/plot/matplotlib_backend.py` (Step 0's
implementation), and `scripts/__init__.py` (the Import note says *"Add it anyway, in the same
commit"*). Task 5's commit (L2095) adds only the example and notebook — not
`tests/plot/test_recency_fade.py`. Task 1's Step 7 does get this right (L848 calls it out
explicitly), which shows the omission elsewhere is an oversight, not a policy.

## N11. The plan contradicts itself on the suite baseline, and both numbers are stale

```
L193   "Baseline, re-measured 2026-08-02: 2799/2801 tests collected (2 deselected)"
L3764  "re-measured 2026-08-02 at 2782/2784 collected"
$ .venv/bin/python -m pytest --collect-only -q   ->  2819/2821 tests collected (2 deselected)
```

Same stated date, two different numbers, neither current. The plan does instruct re-measuring at
start, so this is process-correct and number-wrong — but a self-contradiction inside one document is
not.

## N12. Twelve Contract-8 evasions remain

Measured with the prescribed `_unpacked_wrapper_uses`. `MISSED` = should flag, returned `[]`:

```
EV1  from hypertools import plot as p; fig, ani = p(d)          MISSED
EV2  fig, ani = getattr(hyp, 'plot')(d)                         MISSED
EV3  d2 = {'a': hyp.plot(d)}; fig, ani = d2['a']                MISSED
EV4  L = [hyp.plot(d)]; fig, ani = L[0]                         MISSED
EV6  fig, ani = (anim := hyp.plot(d))                           MISSED
EV7  if (anim := hyp.plot(d)): fig, ani = anim                  MISSED
EV8  fig, *rest = hyp.plot(d); rest[0].on_frame(cb)             MISSED
EV10 anim = hyp.plot(d); ani: object = anim[1]                  MISSED  (AnnAssign)
EV11 for fig, ani in [hyp.plot(d)]:                             MISSED
EV12 with hyp.plot(d) as (fig, ani):                            MISSED
EV14 ani = hyp.plot(d).animation                                MISSED
EV15 a = hyp.plot(d); b = a; fig, ani = b                       MISSED  (alias chain)
```

Correctly handled: A–H (all eight the prior review named, including the chained `anim`/`fig, ani =
anim`/`ani.on_frame` case and `draw_frame`/`n_frames`), plus `*front, ani = hyp.plot(d)` (EV9),
`a = b = hyp.plot(d)` (EV13), and unpacking inside a function body (EV5). The docstring's *"Two
passes so the result does not depend on source order"* is verified (case H flags); the two passes
do **not** propagate through a Name→Name alias, which is what EV15 exposes.

## N12b. Every `plot.py:` citation in this plan is already going stale, mid-review

Plan 3's Tasks 2–3 landed on `dev-1.0` while this review ran (`e1aa1144`, +50 lines to `plot.py`),
moving the equal-feature-width check from `3152` to `3165`. Plan 4 Task 2 (L1040) writes
`plot.py:3152-3153` into the shipped example source as an inline comment. Plan 4 is sequenced
*after* Plans 1–3, so this is structural, not bad luck: every `plot.py:`/`colors.py:` line number in
this plan must be re-derived at implementation time. Detail in the M2 section below.

---

# LOW

## N13. A NEW stale test symbol, in the table where the old one was fixed

L2 is genuinely fixed — `test_every_launch_notebook_ships_executed_outputs` has zero occurrences.
But Step 3's derivation table row 1 now reads:

```
L3580  | `test_no_notebook_budget_is_below_its_own_scripts` | 1 |
```

The prescribed test (L2909) is `test_notebook_budgets_are_derived_not_written_down`. The old name is
the one M1 asked to be *replaced*; it survives in the table that counts it.

## N14. The residue of M2's row is still wrong

The M2 rejection is correct (see below), but the *rest* of L64 was not re-checked:

```
plan L64:  "The comment block starts at plot.py:2748-2756 ... Cite 2750-2751."
$ sed -n '2748,2756p' hypertools/plot/plot.py
    if predict is not None and animate and animate != "spin":
        raise NotImplementedError("predict= is only supported with static plots ...")
```

`2748-2756` is the predict/animate refusal, not a comment block about equal feature widths, and
`2750-2751` are two f-string fragments inside it. (L237 cites the same range correctly, for the
`NotImplementedError`.)

## N15. `RATIONALE_WINDOW` is asymmetric

`lines[max(0, i - RATIONALE_WINDOW) : i + RATIONALE_WINDOW]` gives 15 lines before the reach and 14
after. Harmless, but the docstring says "within `RATIONALE_WINDOW` lines".

## N16. `_docstring_lines` fails open on a parse error, and then blames a docstring

With the file made unparseable, `_docstring_lines` returns `set()`, so the module docstring's
`ani._args[1][0]` mention counts as a live reach:

```
PARSE-1 file made unparseable  -> FAIL: examples/animate_market_forecast.py:34 reaches
        'ani\._args' with no rationale within 15 lines
```

It fails rather than passes, which is the right direction — but the message names line 34 (prose in
the Coordinate note) instead of saying the file does not parse, and the dead-entry half of the test
is satisfied by that prose alone in this state.

## N17. `hyp_animation.py:67` / `:72`

Filename typo (`hyper_animation.py`), and both numbers land on the `@property` decorator rather than
the `def` (`figure` is at `:68`, `animation` at `:73`). Cosmetic; the properties are where the plan
says.

## N18. `plot.py:3037-3043` cited for a warning it does not contain

Task 2's prescribed comment says *"no `linewidth=` is passed (it would be warned and ignored,
`plot.py:3037-3043`)"*. That range is a comment block about MultiIndex mean-trajectory expansion; it
mentions per-dataset `linewidth` overrides but states no warn-and-ignore behaviour.

---

# M2: the rejection is CORRECT, and its stated mechanism is verified

The plan rejected the prior review's M2 and gave a mechanism. Both check out on a **clean** checkout
(`git status --porcelain` empty for tracked files, no plan patches applied):

```
$ sed -n '3152,3153p' hypertools/plot/plot.py
        _widths = [ri.shape[1] for ri in raw]
        if len(set(_widths)) > 1:
$ grep -n "_widths = \[ri.shape\[1\]" hypertools/plot/plot.py
3152:        _widths = [ri.shape[1] for ri in raw]
$ sed -n '3164,3165p' hypertools/plot/plot.py
                return False
            _text_hint = (
```

`3152-3153` **is** the equal-feature-width check; `3164-3165` is unrelated. The plan's diagnosis of
*why* the prior reviewer got 3164 is also exactly right — I reproduced it:

```
$ # apply Task 1 Steps 3-4 patches (colors.py + plot.py _seaborn_palette_arg), then:
$ grep -n "_widths = \[ri.shape\[1\]" hypertools/plot/plot.py
3164:        _widths = [ri.shape[1] for ri in raw]
```

Task 1's patches insert exactly 12 lines above that point. The plan's rule — *"A line number verified
in a patched tree is not verified"* — is correct and should stay.

**However, the citation went stale during this review.** Plan 3's Tasks 2–3 landed on `dev-1.0` while
I was working (`e1aa1144`, then `fcaf13c8`), adding 50 lines to `plot.py`:

```
$ git log --oneline -4
fcaf13c8 test(plot): Task 4 tests, with two vacuous ones fixed before implementing
e1aa1144 feat(plot): ForecastSchedule + narrow the predict= refusal (Plan 3 Tasks 2-3)
1810746b docs(plans): Plan 4 v3 - fix L1-L4          <- the commit reviewed
$ grep -n "_widths = \[ri.shape\[1\]" hypertools/plot/plot.py     # at fcaf13c8
3165:        _widths = [ri.shape[1] for ri in raw]
$ sed -n '3152,3153p' hypertools/plot/plot.py                     # at fcaf13c8
        # on whatever row count resample= (or the original data) leaves it.
        if resample:
```

So `3152-3153` is correct at `1810746b` (the rejection stands) and **wrong at `fcaf13c8`**. Plan 4
Task 2 hard-codes it as an inline source comment (L1040: *"the analysis pipeline requires
(hypertools/plot/plot.py:3152-3153)"*), which will ship a wrong citation into
`examples/animate_market_forecast.py`. Since Plan 4 runs *after* Plans 1–3 by design, every
`plot.py:` line citation in this plan must be re-derived at implementation time, not trusted.

Every other v3 citation I checked
on the clean tree resolves: `matplotlib_backend.py:2036/2039/2448/2451`, `plot.py:113/1066/1154/
219/3080/4825-4826`, `colors.py:105/305-306`, `docs/conf.py:131`, `animate.py:84`, `smooth.py:232`,
`scripts/generate_gallery_thumbs.py:26`, `animate_market_forecast.py:113/191/195`,
`animate_morph_zoo.py:47/74/78/81`, `animate_painting_embeddings.py:138-140`, and
`sphinx_gallery/gen_rst.py:1271-1280` (the fake `__main__` module — sphinx-gallery 0.21.0).

---

# The 23 prior findings

| # | prior finding | status | evidence |
|-|-|-|-|
| F1 | `strip_docstrings` never defined — 90 ImportErrors | **CONFIRMED FIXED** | Defined at L2707 and imported by `_code_text`. Real run: `40 failed, 91 passed, 7 skipped`, zero `ImportError`. |
| F2 | metric drifts .py vs .ipynb | **CONFIRMED FIXED** (new defect inside the fix: N3, N4) | `measure(.py)==measure(.ipynb)==(3,2)`; `_code_lines_py` byte-identical on all 5 scripts and 0/334 repo files; 187/194/121/176/46 reproduce exactly; pre-fix counter gives 193/207/191 as the note claims. |
| F3 | `construct_artifact` / `fixture_data` undefined | **NOT FIXED** *(defined, never written)* | Step 0b defines them; `grep` shows zero occurrences in Tasks 2–6. All 5 IDs still fail. → **N1** |
| F4 | Step 3's stated result impossible | **PARTIAL** | Restated to `138 collected — 126 passed, 5 failed, 7 skipped`; arithmetic is now self-consistent and every ID count matches real collection. Still unreachable because of N1/N2. |
| H1 | `_import_example_without_fetching` will fetch | **CONFIRMED FIXED** | Asserts the guard **before** `exec_module`; whole 138-ID run takes 0.46 s with zero network. (The `HYPERTOOLS_OFFLINE` half is prescribed only in Step 0b — folded into N1.) |
| H2 | Contract 8 guard misses the blessed idiom + the 3 new accessors | **PARTIAL** | AST rewrite catches **all 8** named cases (A–H) incl. the chained `anim`→`fig, ani = anim`→`ani.on_frame`, `draw_frame`, `n_frames`, `res[1]`, `ht.plot`, and reverse source order. 12 evasions remain (N12) and 3 false-positive classes introduced (N6). |
| H3 | Step 7 states v2's numbers | **PARTIAL** | Step 7 + Suite arithmetic now 19/12/147/+178, all three confirmed by real collection. Self-Review L3752 still says "109-test gate" (N7); baselines contradict (N11). |
| H4 | "still explained" is near-vacuous | **CONFIRMED FIXED** | Proximity enforced: rationale at file top → **FAIL**; 16 lines above → **FAIL**; 10 lines above → still FAILs on the *other* two reaches (per-hit checking works). Dead-entry detection now works with the docstring mention retained: deleting the 3 live reaches → *"no longer contains 'ani\\._args'; drop the PRIVATE_API_EXCEPTIONS entry"*. |
| H5 | `n_segments` docstring says 2n | **CONFIRMED FIXED** | Docstring now says `2n - 1`; measured n=2..7 → 3/5/7/9/11/13. |
| M1 | budget test cannot fail | **CONFIRMED FIXED** | Equality assertion. Hand-writing `painting_embeddings.ipynb: 110` (the exact v2 defect) → `FAIL: budgeted at 110, but the derivation says 118 + 5 = 123`. |
| M2 | `plot.py:3152-3153` stale | **REJECTION UPHELD** | Clean checkout: `3152` **is** `_widths = [...]`. Applying Task 1's patches moves it to `3164` — the prior reviewer's number, reproduced. Residual: L64's `2748-2756` / "Cite 2750-2751" is wrong (N14). |
| M3 | artifact test's advertised catch doesn't exist | **CONFIRMED FIXED** | Docstring rewritten: *"It PASSES today, on all five — it is a CONTROL, not coverage … the stem mismatch is a naming inconsistency, not a broken link."* Verified: 5 passed. |
| M4 | the 2-line display cell is unmeasured | **CONFIRMED FIXED** | Constant's comment now labels it *"NOT MEASURED — a design decision"*, records the `grep` that returns nothing, and flags it LOAD-BEARING with conversation's 2-line headroom. |
| M5 | baseline table contradicts Task 6 / overstates the total | **PARTIAL** | Goal line corrected to 723 / 6.6%. The L81–95 table itself is untouched and wrong in 8 of 10 rows, still headed "the numbers every task below is held to" (N9). |
| M6 | opt-in smoke test never created | **CONFIRMED FIXED** | `test_example_runs_end_to_end` exists (L3522–3550), 5 IDs collected, all skipped without `HYPERTOOLS_EXAMPLE_SMOKE`. |
| M7 | only one morph branch tagged | **CONFIRMED FIXED** | L2550 names both (`:2036` 3-D, `:2448` 2-D) in bold and adds `test_n_segments_is_set_for_a_2d_morph_too`. Both cited lines verified; tagging both → 9 passed. |
| M8 | Task 1 Step 6 needs a heading only Plan 1 creates | **CONFIRMED FIXED** | Prerequisites L236 now records the `grep` returning nothing and offers *"Either land animation-core's CHANGELOG step first, or have Step 6 create the heading if it is missing."* CHANGELOG still lacks it — as documented. |
| L1 | morph exit code 17 vs 1 | **CONFIRMED FIXED** | Contract 4 table L146 now reads *"HARD FAILS — `HypertoolsIOError`, exit 1"*. |
| L2 | `test_every_launch_notebook_ships_executed_outputs` doesn't exist | **CONFIRMED FIXED** *(new instance introduced)* | Zero occurrences; both citations now name `test_every_launch_notebook_ran_every_cell_it_should`. A different stale symbol appeared in Step 3's table (N13). |
| L3 | `test_native_ratio_is_reported` near-vacuous | **CONFIRMED FIXED** | Now asserts `os.path.exists(full)` first, with a comment explaining the message was unreachable, and the `code > 0` message was reworded to "parsed to zero code lines". |
| L4 | stale suite baseline | **PARTIAL** | L193 updated to 2799/2801 with the arithmetic reconciliation; L3764 still says 2782/2784; real is 2819/2821 (N11). |
| L5 | `scripts/__init__.py` doesn't exist | **CONFIRMED FIXED** *(as noted)* | Import note retained and the hedge resolved; I reproduced the PEP 420 resolution (138 collected, no `ModuleNotFoundError`). But Step 10 never `git add`s the file (N10). |
| L6 | zero-warning docs build unverified | **N/A** | Was the prior reviewer's own non-verification, not a plan defect. **I did not run the docs build either** — stated rather than guessed. |
| #15 | `git stash` residual in Step 1 | **NOT FIXED** | Still at L2810, plus `git stash list` at L967, plus a stale expected output (`166`/`5.4%` vs real `165`/`5.5%`). → **N8** |

---

# Minimum work to make this plan implementable

1. **Write the split into Tasks 2–6.** Each task's prescribed source must contain the `NamedTuple`,
   `load_*()`, `fixture_data()`, `construct_artifact(data)`, the `HYPERTOOLS_OFFLINE` guard line in
   every fetcher, and the `if __name__ == '__main__':` driver — not a pointer to Step 0b. Add the
   paintings 1.7 KB fixture to a task and to its `git add` (N1).
2. **Perform Step 0c's renegotiation.** Measure the split overhead per example, raise each AFTER line
   *and* the prescribed `SCRIPT_BUDGETS`, then restate Step 3's expected result. Weather at `≤ 62`
   is provably unsatisfiable (N2).
3. **Fix `strip_docstrings`.** Use `ast`/`tokenize` for `.py`, or at minimum require the opening
   triple quote to be the first token of a statement. Re-run the 10 measurements afterwards — the
   budgets assume the current counts (N3).
4. **Make `_code_text` reset per cell**, exactly as `_code_lines_nb` does, or move the notebook
   concatenation behind one shared helper (N4).
5. **Assert the file parsed** in `_parsable_code` / `_unpacked_wrapper_uses` / `_docstring_lines`
   rather than returning empty, and extend `_parsable_code` to `%%` cell magics and `?` suffixes
   (N5).
6. **Restrict `_is_plot_call`** to hypertools-rooted calls (module alias resolved from the imports)
   so `ax.plot` / `df.plot` cannot false-positive, and add the `.animation` re-bind and the Name→Name
   alias chain to the tracked forms (N6, N12).
7. Sweep the residuals: `109-test gate` (N7), the `git stash` block and its stale output (N8), the
   L81–95 baseline table (N9), the commit-step `git add` lists (N10), the two baseline figures (N11),
   the Step 3 table's test name (N13), and L64's `2748-2756` (N14).
