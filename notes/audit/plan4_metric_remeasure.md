# Plan 4 metric audit: fixing `_code_lines_nb` and re-measuring everything that depended on it

**Date:** 2026-08-01. **Repo:** `/Users/jmanning/hypertools`, branch `dev-1.0`, HEAD `065c841e` (clean).
**Python:** `/Users/jmanning/hypertools/.venv/bin/python` throughout.
**Plan under audit:** `docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md` ("Plan 4"),
Task 8 Step 1 (script starts at plan line 2082).

**Scope note:** this is a metric-correctness audit and re-measurement only. The plan file itself was
**not** edited, and `scripts/measure_native_ratio.py` was **not** created in the repo — all work below
lives under the scratch directory
`/private/tmp/claude-501/-Users-jmanning-hypertools/7e6531b3-066a-4ce2-b1f6-7c07c5e87b15/scratchpad/`.

**Prior art:** this exact defect, with the exact same reproduction numbers, was already found and
recorded in `/Users/jmanning/hypertools/notes/audit/review_plan4_v2.md` (`Fatal | scripts/measure_native_ratio.py
_code_lines_nb`, lines 127–156, evidence-table row 18). That review reasoned about the fix qualitatively
("most generous reading ... notebook code is script code + 4") because it had no fixed tool to run. This
document builds the actual fix and re-measures everything with it, which is the natural continuation of
that finding rather than duplicate work. Every number below was independently produced by the tool built
in this session; agreement with the prior review (noted inline) is cross-validation, not a citation
standing in for a measurement.

---

## 1. The defect

`scripts/measure_native_ratio.py` (prescribed verbatim in the plan, plan lines 2082–2189) measures
`(code_lines, native_lines)` for a `.py` file with `_code_lines_py`, and for a `.ipynb` file with
`_code_lines_nb`. Both are supposed to implement the same CODE-line definition ("non-blank, not
comment-only, not part of a bare docstring" — the module docstring, plan line 2087), but they are two
independently written functions:

- `_code_lines_py` runs a real docstring-stripping state machine (`in_doc`/`delim`).
- `_code_lines_nb` only drops blank and comment-only lines — it has **no** docstring handling at all, so
  every physical line of a bare triple-quoted docstring inside a code cell is counted as CODE.

Since `measure()` dispatches on file extension (plan lines 2165–2168: `.ipynb` → `_code_lines_nb`, else →
`_code_lines_py`), identical source measures differently purely because of how it's stored on disk.

The verbatim extracted script (byte-for-byte from plan lines 2082–2189) is saved at
`.../scratchpad/measure_native_ratio_original.py`.

## 2. Reproducing the defect

Built an 11-line source with a 6-line bare docstring followed by 3 real code lines (one of them a
`hyp.plot` call), saved identically as `.py` and as one Jupyter code cell:

```python
"""
line 1
line 2
line 3
line 4
line 5
line 6
"""
x = 1
import hypertools as hyp
hyp.plot(x)
```

Files: `.../scratchpad/repro_docstring.py` and `.../scratchpad/repro_docstring.ipynb` (one code cell,
`source` = the same 11 lines).

Running the **original** (unfixed) `measure()` on both:

```
repro_docstring.py                                       code=   3 native=   2 ratio= 66.7%
repro_docstring.ipynb                                    code=  11 native=   2 ratio= 18.2%
```

**Identical source: `(code=3, native=2)` as `.py` vs. `(code=11, native=2)` as `.ipynb`.** This
reproduces, exactly, the `(3, 2)` vs. `(11, 2)` numbers cited in `review_plan4_v2.md:132–135` — the
8 extra "code" lines are precisely the 6 docstring-body lines plus its 2 delimiter lines, all of which
`_code_lines_py` correctly discards and `_code_lines_nb` does not.

## 3. The fix

Factored the docstring-stripping state machine out of `_code_lines_py` into a shared generator,
`_strip_docstrings(lines)`, and made **both** counters call it. `_code_lines_nb` now resets the
`in_doc`/`delim` state once per code cell (not once per notebook) — this is correct, not just
convenient: a triple-quoted string literal cannot span a Jupyter cell boundary, because each cell is
parsed and executed as an independent unit, so there is no possibility of a docstring that opens in one
cell and closes in the next. A shared callee cannot drift from itself, which is the actual fix — the two
functions no longer contain two copies of the same logic that can silently diverge again.

Full fixed source (`.../scratchpad/measure_native_ratio_fixed.py`):

```python
r"""Measure how much of an example or tutorial is a hypertools call.

Definitions (these are the contract Task 8 of the 1.1 examples plan gates on):

CODE line    -- non-blank, not comment-only, not part of a bare docstring.
LOGICAL stmt -- consecutive code lines joined while bracket depth > 0, or
                while a line ends in a backslash. A continuation line belongs
                to the statement it continues, so a 10-line ``hyp.plot(...)``
                call counts as 10 native lines rather than 1. This is the
                whole point: the metric must reward a big native call.
NATIVE       -- every code line of a logical statement whose text matches
                ``\bhyp\.|\bhypertools\b``.

Measured against the 2026-07-26 audit's independent NATIVE-line
classification, this metric gives 48/739 = 6.5% for the five launch scripts
where the audit reported 6.0% -- i.e. the two agree.

    .venv/bin/python scripts/measure_native_ratio.py examples/animate_*.py
    .venv/bin/python scripts/measure_native_ratio.py docs/tutorials/*.ipynb

FIX (2026-08-01, plan4 metric audit): ``_code_lines_py`` and ``_code_lines_nb``
used to strip bare docstrings with two INDEPENDENT copies of the same state
machine, and the ``.ipynb`` copy was never written -- it only dropped blank
and comment-only lines, so a bare triple-quoted docstring inside a code cell
counted every one of its lines as CODE. Identical source measured differently
depending on whether it was stored as ``.py`` or ``.ipynb`` (reproduced:
(code=3, native=2) vs (code=11, native=2) for the same 11-line source with a
6-line docstring). Both counters now call the SAME ``_strip_docstrings``
generator, so they cannot drift from each other again. The state machine is
reset per code cell for notebooks (matching per-file reset for scripts),
which is correct: a triple-quoted string cannot span a Jupyter cell boundary,
since each cell is independently parsed/executed.
"""

import json
import re
import sys

HYP = re.compile(r'\bhyp\.|\bhypertools\b')


def _strip_docstrings(lines):
    """Yield the CODE lines from an iterable of source lines.

    Drops blank lines, comment-only lines, and any bare ``\"\"\"``/``'''``
    docstring block (a state machine that treats a stripped line starting
    with a triple quote as the open/close of a docstring). This is the ONE
    place that logic lives -- shared by both the ``.py`` and ``.ipynb``
    counters below, so they cannot independently drift out of sync the way
    ``_code_lines_py``/``_code_lines_nb`` did before this fix.
    """
    in_doc, delim = False, None
    for line in lines:
        stripped = line.strip()
        if in_doc:
            if delim in stripped:
                in_doc = False
            continue
        if stripped.startswith(('"""', "'''")):
            delim = stripped[:3]
            if not (len(stripped) > 3 and stripped.endswith(delim)):
                in_doc = True
            continue
        if not stripped or stripped.startswith('#'):
            continue
        yield line


def _code_lines_py(path):
    return list(_strip_docstrings(
        open(path, encoding='utf-8').read().splitlines()))


def _code_lines_nb(path):
    out = []
    for cell in json.load(open(path, encoding='utf-8'))['cells']:
        if cell.get('cell_type') != 'code':
            continue
        # Reset per cell: a bare docstring cannot span a cell boundary (each
        # cell is parsed/executed independently), so carrying `in_doc`/`delim`
        # state across cells would be wrong, not just unnecessary.
        out.extend(_strip_docstrings(
            line.rstrip('\n') for line in cell['source']))
    return out


def _depth_delta(line):
    depth, quote, i = 0, None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == '\\':
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in '"\'':
            quote = ch
        elif ch == '#':
            break
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        i += 1
    return depth


def measure(path):
    """Return ``(code_lines, native_lines)`` for one .py or .ipynb file."""
    lines = _code_lines_nb(path) if str(path).endswith('.ipynb') \
        else _code_lines_py(path)
    statements, current, depth = [], [], 0
    for line in lines:
        current.append(line)
        depth += _depth_delta(line)
        if depth <= 0 and not line.rstrip().endswith('\\'):
            statements.append(current)
            current, depth = [], 0
    if current:
        statements.append(current)
    total = sum(len(s) for s in statements)
    native = sum(len(s) for s in statements
                 if HYP.search('\n'.join(s)))
    return total, native


if __name__ == '__main__':
    for target in sys.argv[1:]:
        code, native = measure(target)
        pct = 100.0 * native / code if code else 0.0
        print(f'{target:56s} code={code:4d} native={native:4d} '
              f'ratio={pct:5.1f}%')
```

**One-line summary of the fix:** both counters now call one shared `_strip_docstrings(lines)` generator
(reset per code cell for notebooks) instead of `_code_lines_py` having a state machine that
`_code_lines_nb` lacked entirely.

## 4. Proof

**4a. Fix closes the repro gap.** Fixed `measure()` on the same two repro files:

```
repro_docstring.py                                       code=   3 native=   2 ratio= 66.7%
repro_docstring.ipynb                                    code=   3 native=   2 ratio= 66.7%
```

`.py` and `.ipynb` now measure **identically** for identical source.

**4b. Refactor safety — `.py` results unchanged.** Ran the ORIGINAL `_code_lines_py` and the FIXED
(shared-callee) `_code_lines_py` on the five real, on-disk example scripts and diffed the returned line
lists (not just their lengths):

| script | orig `_code_lines_py` count | fixed `_code_lines_py` count | identical line-for-line |
|-|-|-|-|
| `examples/animate_market_forecast.py` | 191 | 191 | True |
| `examples/animate_weather_decades.py` | 195 | 195 | True |
| `examples/animate_painting_embeddings.py` | 146 | 146 | True |
| `examples/animate_conversation.py` | 165 | 165 | True |
| `examples/animate_morph_zoo.py` | 26 | 26 | True |

All five identical. The refactor does not change `.py` behavior — only `.ipynb` behavior changes (and
only for notebooks whose code cells actually contain a bare docstring).

**4c. Does the defect matter on the real files?** Ran original vs. fixed `measure()` on the five current
notebooks:

| notebook | orig (code, native) | fixed (code, native) | defect changed this file? |
|-|-|-|-|
| `docs/tutorials/market_forecast.ipynb` | (193, 12) | (187, 11) | **yes** |
| `docs/tutorials/weather_decades.ipynb` | (207, 11) | (194, 11) | **yes** |
| `docs/tutorials/painting_embeddings.ipynb` | (121, 11) | (121, 11) | no |
| `docs/tutorials/conversation_shape.ipynb` | (191, 12) | (176, 11) | **yes** |
| `docs/tutorials/morph_shapes_zoo.ipynb` | (46, 9) | (46, 9) | no |

Not hypothetical: 3 of the 5 real current launch notebooks contain a bare docstring inside a code cell
today and were measured wrong by the unfixed tool (market by 6 lines, weather by 13, conversation by 15).

---

## 5. Re-measurement with the fixed metric

### 5a/5b. The five current notebooks and the five current example scripts

All numbers below are `measure()` (fixed) run directly against the real files on disk right now.

| file | code | native | ratio |
|-|-|-|-|
| `docs/tutorials/market_forecast.ipynb` | 187 | 11 | 5.9% |
| `docs/tutorials/weather_decades.ipynb` | 194 | 11 | 5.7% |
| `docs/tutorials/painting_embeddings.ipynb` | 121 | 11 | 9.1% |
| `docs/tutorials/conversation_shape.ipynb` | 176 | 11 | 6.2% |
| `docs/tutorials/morph_shapes_zoo.ipynb` | 46 | 9 | 19.6% |
| **five notebooks, total** | **724** | **53** | **7.3%** |
| `examples/animate_market_forecast.py` | 191 | 11 | 5.8% |
| `examples/animate_weather_decades.py` | 195 | 11 | 5.6% |
| `examples/animate_painting_embeddings.py` | 146 | 11 | 7.5% |
| `examples/animate_conversation.py` | 165 | 9 | 5.5% |
| `examples/animate_morph_zoo.py` | 26 | 6 | 23.1% |
| **five scripts, total** | **723** | **48** | **6.6%** |

### 5c. Plan-quoted baseline vs. actual current measurement

The plan's own "Measured baseline, logical-statement metric" table (plan lines 54–68) and the aggregate
figure quoted in the goal statement and Verification note (plan lines 5, 52) are the only baseline
numbers the plan states in this form. Each is reproduced below next to what the fixed tool measures on
the CURRENT file today.

**Scripts** (never touched the buggy code path — the plan's own tool measured these correctly even
before the fix; divergence below is real file drift, not the defect):

| file | plan states | measured now (fixed) | status |
|-|-|-|-|
| `examples/animate_market_forecast.py` | 191 / 11 / 5.8% | 191 / 11 / 5.8% | MATCHES |
| `examples/animate_weather_decades.py` | 196 / 11 / 5.6% | 195 / 11 / 5.6% | STALE-was-196-now-195 |
| `examples/animate_painting_embeddings.py` | 146 / 11 / 7.5% | 146 / 11 / 7.5% | MATCHES |
| `examples/animate_conversation.py` | 166 / 9 / 5.4% | 165 / 9 / 5.5% | STALE-was-166-now-165 |
| `examples/animate_morph_zoo.py` | 40 / 6 / 15.0% | 26 / 6 / 23.1% | STALE-was-40-now-26 |
| **five scripts, total** | **739 / 48 / 6.5%** | **723 / 48 / 6.6%** | STALE-was-739-now-723 |

The morph script's 40→26 drop is not noise: `git log` shows `d730a085 docs(1.1): … per-segment titles;
simplify examples` already landed a mechanical migration on this file (confirmed by reading it — it
already calls `hyp.plot(..., title=titles)` and contains no `_morph` import, no `shape_title`, no
`ani._func` reach). It is already close to Task 6's prescribed shape; see §6.

**Notebooks** (the plan's own baseline table for these was very likely produced with the very tool being
fixed here, since the plan predates this fix; shown with BOTH the unfixed and fixed re-measurement for
transparency):

| file | plan states | measured now, orig tool | measured now, fixed tool | status (vs. fixed) |
|-|-|-|-|-|
| `docs/tutorials/market_forecast.ipynb` | 192 / 11 / 5.7% | 193 / 12 / 6.2% | 187 / 11 / 5.9% | STALE-was-192-now-187 |
| `docs/tutorials/weather_decades.ipynb` | 206 / 10 / 4.9% | 207 / 11 / 5.3% | 194 / 11 / 5.7% | STALE-was-206-now-194 |
| `docs/tutorials/painting_embeddings.ipynb` | 116 / 10 / 8.6% | 121 / 11 / 9.1% | 121 / 11 / 9.1% | STALE-was-116-now-121 |
| `docs/tutorials/conversation_shape.ipynb` | 186 / 11 / 5.9% | 191 / 12 / 6.3% | 176 / 11 / 6.2% | STALE-was-186-now-176 |
| `docs/tutorials/morph_shapes_zoo.ipynb` | 45 / 8 / 17.8% | 46 / 9 / 19.6% | 46 / 9 / 19.6% | STALE-was-45-now-46 |

All ten plan-quoted per-file baseline numbers are stale relative to the files on disk today (mostly
single-digit drift; morph script is the one large jump). None of the five notebook rows the plan quotes
survive re-measurement unchanged — this independently reproduces `review_plan4_v2.md`'s "Prior finding 5
… NOT FIXED … Five of five still wrong" (lines 57–69), whose own re-measured numbers (191/12, 193/12,
46/9, 121/11, 207/11) match this session's *unfixed*-tool re-measurement exactly, confirming both
sessions read the same files with the same (buggy) tool and got the same answer.

**Other aggregate figures the plan states**, and why they are not re-derivable from this metric:

- "37.9% is defect" (plan line 5) — this comes from the separate audit's manual A/B/C/D line
  classification (class B + class C ÷ total), not from `measure_native_ratio.py`. Out of scope for this
  metric; not re-measured here.
- "reproducing the audit's 6.0% NATIVE-line classification" (plan line 52) — a comparison against a
  different, independent manual audit, not a `measure_native_ratio.py` output. Not re-measured here.

---

## 6. BUDGETS: plan number, current, and prescribed-content, per file

The only enforced per-file budgets are the `BUDGETS` list in `tests/test_examples_are_native.py` (plan
lines 2241–2256) — a `(path, max_code_lines)` pair per file, asserted by
`test_file_is_within_its_size_budget`. **There is no enforced ratio floor**: v2's own Revision note
(plan line 20) states the per-file native-ratio floor was removed as one of the two v1 Fatals, and the
test module's docstring (plan lines 2211–2229) confirms only four things are gated (defect markers, this
code-line budget, semantic artifact checks, exact notebook execution) — ratio is "REPORTED, not gated."
The "AFTER (contracted budget)" prose inside Tasks 2–6 (e.g. plan line 1435: "script ≤ 72 code lines, ≥
25% native") still quotes the old, now-deprecated ratio floors and, in one place, a stale code budget —
see the conversation row below.

**PRESCRIBED final content**, per file: for the five **scripts**, Tasks 2 and 3 "replace
`examples/animate_*.py` entirely" with fully literal code (plan lines 728–922, 1018–1129), and Tasks 4–6
give literal new code plus an explicit instruction to keep a cited line range of the **current** file
verbatim (paintings: "Keep the `PAINTINGS` dict verbatim (lines 43–96 of the current file)", plan line
1220; conversation: "Keep `SPEAKER_COLOR` and the `TURNS` list verbatim (lines 44–85)", plan line 1455;
morph: keep everything before the deleted import / before line 94, plan lines 1827–1842). All five are
therefore fully reconstructable and were rebuilt exactly per those instructions (current-file slices
spliced with the plan's literal new code; scratch files
`.../scratchpad/prescribed_{market_forecast,weather_decades,painting_embeddings,conversation,morph_zoo}.py`)
and measured with the fixed tool.

For the five **notebooks**, the plan gives only a cell-*structure* table (which script section goes in
which cell), not literal cell source — so there is no verbatim "prescribed final content" to measure
exactly. What IS pinned exactly: cell 0 (explicitly "unchanged" from the real current notebook, measured
directly) and every code cell in between (explicitly the prescribed script's own code, verbatim, split at
stated boundaries — summing back to exactly the prescribed script's code-line count, since the cell
table is a complete, non-overlapping partition of it). The only unpinned part is the final display cell,
described only as `HTML(ani.to_jshtml())`, which needs `from IPython.display import HTML` somewhere to
actually run (spelled out as 2 lines only in Task 2's cell table, plan line 978; abbreviated to 1 line in
Tasks 3–6). Reported below as **cell0 (exact) + prescribed-script-code (exact) + display cell (2 lines,
import+call)**, labeled ESTIMATE; the 1-line alternative changes every total by exactly 1 and flips no
verdict.

### Scripts

| file | plan's budget | current (fixed metric) | prescribed final content (fixed metric) | verdict |
|-|-|-|-|-|
| `examples/animate_market_forecast.py` | ≤ 115 | 191 | **109** | ATTAINABLE |
| `examples/animate_weather_decades.py` | ≤ 62 | 195 | **56** | ATTAINABLE |
| `examples/animate_painting_embeddings.py` | ≤ 118 | 146 | **111** | ATTAINABLE |
| `examples/animate_conversation.py` | ≤ 90 (BUDGETS dict; **AFTER-prose at plan:1435 still says ≤72, stale**) | 165 | **88** | ATTAINABLE against 90; **UNATTAINABLE-by-16-lines against the stale 72 still in the prose** |
| `examples/animate_morph_zoo.py` | ≤ 30 | 26 (**already under budget**) | **26** | ATTAINABLE |

The conversation script's own `BUDGETS` entry (plan lines 2245–2249) is self-aware about this: `# 90, not
72: the prescribed rewrite measures 88 code lines (87 at best, with turn_alpha inlined ...)`. That
comment's own number (88) is exactly what this remeasurement independently produced — the plan already
carries the correct figure in a comment next to the dict, it just never propagated it back into Task 5's
"AFTER" prose four hundred lines earlier. `review_plan4_v2.md:111` ("conversation … 88 … ≤72 … EXCEEDS by
16") found the same 88-vs-72 mismatch against the *older* 72 budget, before it was raised to 90 in the
`065c841e` revision; against the now-current 90 it is attainable with 2 lines to spare.

Every prescribed script attains its `BUDGETS` figure. Only `animate_morph_zoo.py` also already attains
it on disk today — see §5c: its script half was already migrated in `d730a085` before this plan's Task 6
runs.

### Notebooks

| file | plan's budget | current (fixed metric) | prescribed final content (ESTIMATE: cell0 + script + 2) | verdict |
|-|-|-|-|-|
| `docs/tutorials/market_forecast.ipynb` | ≤ 120 | 187 | 2 + 109 + 2 = **113** | ATTAINABLE |
| `docs/tutorials/weather_decades.ipynb` | ≤ 66 | 194 | 2 + 56 + 2 = **60** | ATTAINABLE |
| `docs/tutorials/painting_embeddings.ipynb` | ≤ 110 | 121 | 3 + 111 + 2 = **116** | **UNATTAINABLE-by-6-lines** |
| `docs/tutorials/conversation_shape.ipynb` | ≤ 76 | 176 | 3 + 88 + 2 = **93** | **UNATTAINABLE-by-17-lines** |
| `docs/tutorials/morph_shapes_zoo.ipynb` | ≤ 34 | 46 | 2 + 26 + 2 = **30** | ATTAINABLE |

Robustness check on the one soft input (the display cell might be 1 line, not 2, if the `HTML` import is
read as already-implied rather than spelled out): every total above shifts by exactly ∓1, and **no
verdict changes** — paintings would be 115 (still over 110), conversation 92 (still over 76), the other
three still comfortably under. `review_plan4_v2.md:138–156` reached the same two failures by a cruder
route ("notebook code is script code + 4 … paintings (115 > 110) and conversation (91 > 76) still fail"),
which lands within 1–2 lines of this session's more precise cell-by-cell reconstruction — independent
convergence on the same two UNATTAINABLE verdicts.

### Verdict summary

7 of 10 budgets are **ATTAINABLE** by the plan's own prescribed content (all 5 scripts against their
`BUDGETS`-dict figures; the market/weather/morph notebooks). **2 are UNATTAINABLE by the plan's own
prescribed content as written**: `docs/tutorials/painting_embeddings.ipynb` (by 6 lines) and
`docs/tutorials/conversation_shape.ipynb` (by 17 lines) — both because their code cells necessarily carry
a real, non-trivial function docstring (`canvas_color`'s and `recency_fade`'s, both multi-paragraph) that
a correct notebook-side counter must NOT strip for free the way the buggy one used to. This is a plan
defect to report, not a number to round away: per Contract 6 (plan line 96, "budgets are contracts, never
weakened to fit the code … the assertion is never weakened to fit the code"), the fix is to either raise
`docs/tutorials/painting_embeddings.ipynb` and `docs/tutorials/conversation_shape.ipynb`'s budgets in the
`BUDGETS` dict, or trim the prescribed notebook content — not to leave the current figures in place and
hope the metric bug keeps hiding the gap.

One more conversation-script inconsistency, unrelated to the `.ipynb` defect but found while
cross-checking every number in this section: Task 5's own "AFTER (contracted budget)" prose (plan line
1435) still reads "script ≤ 72 code lines," which is the pre-`065c841e` figure; the enforced `BUDGETS`
dict entry (plan line 2249) was already corrected to 90 with an explanatory comment. The prose was never
updated to match.

---

## 7. Files

- `.../scratchpad/measure_native_ratio_original.py` — verbatim extraction of the plan's prescribed script
  (plan lines 2082–2189).
- `.../scratchpad/measure_native_ratio_fixed.py` — the fix (§3), full source above.
- `.../scratchpad/repro_docstring.py`, `.../scratchpad/repro_docstring.ipynb` — the minimal reproduction
  (§2/§4a).
- `.../scratchpad/prescribed_market_forecast.py`, `prescribed_weather_decades.py`,
  `prescribed_painting_embeddings.py`, `prescribed_conversation.py`, `prescribed_morph_zoo.py` — the five
  reconstructed prescribed scripts (§6), built from the plan's literal code blocks plus (for paintings,
  conversation, morph) the cited verbatim line ranges of the real current files.
  `paintings_plan_header.py`/`paintings_plan_footer.py`, `conversation_plan_tail.py`,
  `morph_plan_tail.py` hold the plan-literal pieces used to build them.

No file under `/Users/jmanning/hypertools/scripts/`, `/Users/jmanning/hypertools/tests/`, or the plan
file itself was created or modified by this audit.
