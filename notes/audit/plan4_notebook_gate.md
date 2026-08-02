# Plan 4 Task 8 notebook-output gate: from guessed counts to a measured design

Audited 2026-08-02 against `dev-1.0` HEAD `065c841e` (clean). Plan under audit:
`docs/superpowers/plans/2026-07-28-hypertools-1.1-examples-and-tutorials.md`
("Plan 4"), Task 8 (~line 2077), gated notebooks are the prescribed
replacements for `docs/tutorials/{market_forecast, weather_decades,
painting_embeddings, conversation_shape, morph_shapes_zoo}.ipynb` (Tasks 2-6).
Neither the plan file nor any notebook was modified by this audit. Every
number below comes from a command that was actually run; the exact commands
are in the Appendix.

**Headline finding.** The five notebooks currently committed in
`docs/tutorials/` are **not** the Tasks 2-6 rewrite. They are the pre-rewrite
("BEFORE") content — confirmed by literal strings that Tasks 2-6 explicitly
delete (`FRED_IDS`, a hand-rolled `embed()`, `from hypertools.plot import
morph as _morph`, `ani._func = _wrapped`) — partially executed once
(`git log`: commit `9b94d86f`, 2026-07-30) using a GIF-plus-markdown-image
rendering convention that Tasks 2-6's prescribed text does not use at all
(they end each notebook in `HTML(ani.to_jshtml())`). **Part 1** below
therefore measures reality; **Part 2** independently derives a prediction
from the plan's prescribed text, because the two cannot be cross-checked
against the same files.

---

## Part 1 — measured: the five CURRENT notebooks

Command: real `json.load` of each `docs/tutorials/{stem}.ipynb`, no
`nbformat` abstraction, no mocks (script in the Appendix).

| notebook | total code cells | non-null `execution_count` | non-empty `outputs` | error outputs | output_types seen | MIME types seen |
|-|-|-|-|-|-|-|
| market_forecast | 7 | {1,2,3,4,5,6} | **{2,4,5,6}** | {} | stream | (none — stream only) |
| weather_decades | 7 | {1,2,3,4,5,6} | **{2,6}** | {} | stream | (none — stream only) |
| painting_embeddings | 6 | {1,2,3,4,5} | **{3,5}** | {} | stream, display_data | `application/vnd.jupyter.widget-view+json`, `text/plain` |
| conversation_shape | 6 | {1,2,3,4,5} | **{2,5}** | {} | stream, display_data | `application/vnd.jupyter.widget-view+json`, `text/plain` |
| morph_shapes_zoo | 6 | {1,2,3,4,5} | **{5}** | {} | stream | (none — stream only) |

(Index 0 is the Colab install cell in every notebook; it is never in either
set today — its `execution_count` is `None` in all five files.)

Reproduces, independently, the plan's own "re-measured 2026-08-01" note
(plan line 36 and lines 2402-2405: *"conversation_shape 2/6 ... market_forecast
4/7, morph_shapes_zoo 1/6, painting_embeddings 2/6, weather_decades 2/7"*) —
exact match on every count and, going further than the plan's prose, on the
exact indices.

**No committed error output anywhere** (all five `error outputs` sets are
empty) — `test_no_launch_notebook_committed_an_error_output` would pass
today. **No `image/png` or `text/html` output exists in any of the five
files** — the display_data entries are tqdm progress-bar widgets from
`sentence_transformers` embedding calls (painting/conversation only), not the
animation. The animation itself lives outside any cell's `outputs`
altogether: the last code cell of every notebook calls
`ani.save('<stem>.gif', fps=fps)` and the notebook's final **markdown** cell
embeds it (`![...](market_forecast.gif)`, etc. — `morph_shapes_zoo.ipynb`
oddly points at `morph_zoo.gif`, not `morph_shapes_zoo.gif`; both the
5-cell-notebook-name and the file that exists on disk were checked and the
reference resolves). This convention — companion GIF, referenced from
markdown, `9b94d86f` — matches how `conversation_trajectories.ipynb`,
`streaming_data.ipynb`, and `wikipedia_embeddings.ipynb` already ship, and it
is **not** any of the three options the plan's own "Decisions still needed"
entry ("How the five launch tutorials get a visible figure", plan line 2566)
lists as implemented/alternative.

---

## Part 2 — predicted: classifying the plan's PRESCRIBED replacement cells

Tasks 2-6 each give a complete cell table plus complete code for the new
notebook, so a prediction **is** derivable for all five — with one exception
noted below. Classification rule, verified (not assumed) three separate
ways:

1. **A `print(...)` anywhere in a cell → `stream` output.** print() itself
   returns `None`, so it does not *also* trigger rule 2.
2. **The cell's LAST top-level statement, if a bare (unassigned) expression
   with a non-`None` value → `execute_result` output**, regardless of what
   ran earlier in the cell (loops, assignments) and regardless of matplotlib
   figure-registration state. Verified for real, in a real `IPython`
   shell (`IPython.testing.globalipapp.get_ipython()`, real `ip.run_cell`,
   not a mock): a bare call after a `for` loop echoes `Out[]:`; an
   assignment or a `print()` does not. Then re-verified with the plan's
   **exact, verbatim** market side-panel code (see cell 6 below) —
   `Out[0]: Text(0.66, 0.015, 'next-day direction, last 30 sessions (50% =
   coin flip)')`.
3. **`fig, ani = hyp.plot(..., show=False, ...)` never auto-displays**,
   confirmed two ways: (a) source, `hypertools/plot/plot.py:5101-5130` —
   `show=False` explicitly calls `plt.close(fig)` specifically so "Jupyter's
   post-cell `flush_figures()` still displays it" (GH #148) cannot fire; (b)
   real prior execution — every current notebook (Part 1) has a cell that
   both builds such a figure *and* prints in the same cell, and every one of
   those cells shows `stream` output only, never an image.

### market_forecast (Task 2, plan lines 726-978 — 8 prescribed code cells)

| code-idx | content | class | reason |
|-|-|-|-|
| 0 | Colab install (unchanged) | **EMITS** | not silent — see note below |
| 1 | imports + `CACHE`/`MARKET`/`RANGE`/`SECTORS`/`COLUMN_NAMES` | SILENT | import-only + bare assignments, last stmt is an assignment |
| 2 | `fetch_prices`/`synthetic_prices`/`_framed` + dispatch + `print(f'market data: ...')` | **EMITS** | trailing `print()` |
| 3 | `sector_index = [...]` comprehension | SILENT | bare assignment |
| 4 | `duration, fps = 8, 20` / `fig, ani = hyp.plot(..., show=False, ...)` | SILENT | bare assignment; `show=False` deregisters the figure (rule 3) |
| 5 | `WINDOW, N_SCORED`/`directional_accuracy`/`scores` + `print('next-day direction correct: ' ...)` | **EMITS** | trailing `print()` |
| 6 | side panel: `ax.set_position(...)`, a `for` loop of `fig.text(...)`, then **two more bare `fig.text(...)` calls after the loop** | **EMITS** | rule 2 — verified verbatim, `Out[0]: Text(0.66, 0.015, 'next-day direction, last 30 sessions (50% = coin flip)')` |
| 7 | `from IPython.display import HTML` / `HTML(ani.to_jshtml())` | **EMITS** | bare trailing expression, `HTML._repr_html_` |

Predicted `{0, 2, 5, 6, 7}` → **5 of 8**.

### weather_decades (Task 3, plan lines 1014-1176 — 5 prescribed code cells)

| code-idx | content | class | reason |
|-|-|-|-|
| 0 | Colab install (unchanged) | **EMITS** | see note below |
| 1 | imports + `CACHE`/`BASE` | SILENT | last stmt is an assignment |
| 2 | `fetch_temperatures`/`synthetic_temperatures` + dispatch + `print(f'weather: ...')` | **EMITS** | trailing `print()` |
| 3 | `duration, fps = 8, 20` / `fig, ani = hyp.plot(..., show=False, ...)` | SILENT | bare assignment; rule 3 |
| 4 | `HTML(ani.to_jshtml())` | **EMITS** | bare trailing expression |

Predicted `{0, 2, 4}` → **3 of 5**. This is a real, previously-run
measurement, not just this audit's derivation: `notes/audit/review_plan4_v2.md`
(2026-08-01) built this exact notebook from Task 3's own cell table and ran
it with `nbclient` against a kernel from this repo's venv:
`code cell 0: 2 outputs ['stream', 'stream']` / `cell 1: 0` / `cell 2: 1
['stream']` / `cell 3: 0` / `cell 4: 1 ['execute_result']` → with-output =
`{0, 2, 4}`, count 3 — exact agreement with this audit's independently-reasoned
derivation.

**On cell 0.** That same real run is also the reason this audit does not
call the install cell silent: a real `%pip install -q "hypertools[...] @
git+...@dev-1.0"` (even quiet) printed **two** stream chunks in that
execution. This audit did not re-run a real `pip install` itself (it would
mutate this repo's own editable dev install), so cell 0's status rests on
that one prior real measurement. Generalizing it across notebooks needed a
check, and the check (see Appendix) found the install cell is **not**
byte-identical in all five files: market_forecast, weather_decades and
morph_shapes_zoo share the plain 2-line install (matching, exactly, the
variant the prior review actually executed); painting_embeddings and
conversation_shape carry one extra line, `%pip install -q
sentence-transformers` (both need it for the text-embedding path). The
plain-variant generalization (market/weather/morph) rests on a real
measurement; painting/conversation's exact output was not independently
re-run, so "EMITS" is carried over for them on the weaker basis that a
second `%pip install` line is not going to make the cell *more* silent than
the one already-measured line was — not on a re-confirmed count. This also
directly contradicts the plan's own repeated assumption, quoted five times,
that "cell 0's Colab install cell produces none" — **that assumption itself
needs a real, disposable-environment verification**, not another assertion.

### painting_embeddings (Task 4, plan lines 1218-1410 — 6 prescribed code cells)

| code-idx | content | class | reason |
|-|-|-|-|
| 0 | Colab install (unchanged) | **EMITS** | see note above |
| 1 | imports, `CACHE`/`FILEPATH`, `PAINTINGS` dict, `WINDOW, STEP = 10, 1` | SILENT | last stmt is an assignment |
| 2 | `windows`/`canvas_color` + `names`/`descriptions`/`colors`/`labels` + `print(f'paintings: ...')` | **EMITS** | trailing `print()` |
| 3 | `duration, fps = 12, 20` / `fig, ani = hyp.plot(..., show=False, ...)` | SILENT | bare assignment; rule 3 |
| 4 | side panel: `ax.set_position(...)`, then a `for i, name in enumerate(names): fig.text(...); fig.text(...)` — **the loop IS the last statement, nothing bare follows it** | SILENT | rule 2 does not fire — no trailing bare expression (contrast with market cell 6, which has two extra calls *after* its loop) |
| 5 | `HTML(ani.to_jshtml())` | **EMITS** | bare trailing expression |

Predicted `{0, 2, 5}` → **3 of 6**.

### conversation_shape (Task 5, plan lines 1453-1782 — 6 prescribed code cells) — **does not execute as written**

| code-idx | content | class | reason |
|-|-|-|-|
| 0 | Colab install (unchanged) | **EMITS** | see note above |
| 1 | imports + `SPEAKER_COLOR` + `TURNS` (verified verbatim against `examples/animate_conversation.py:44-85`, the file the plan cites) | SILENT | last stmt (`TURNS = [...]`) is a plain assignment |
| 2 | `WINDOW, STEP, MIN_WINDOWS`/`windows`/`turns`/`speakers` + `print(f'conversation: ...')` | **EMITS** | trailing `print()` |
| 3 | `duration, fps = 12, 16` / `FLOOR, DECAY = ...` / `fig, ani = hyp.plot(..., show=False, ...)` | SILENT | bare assignment; rule 3 |
| 4 | `turn_alpha`/`recency_fade` defs + **`ani.on_frame(recency_fade)`** | **CRASHES** | verified — see below |
| 5 | `HTML(ani.to_jshtml())` | **UNREACHED** | never executes; cell 4 raises first, and `nbclient`'s default `.execute()` halts the notebook on the first uncaught cell error |

**Verified, real, reproducible bug.** `hyp.plot(..., animate=True, ...)`
returns a `HyperAnimation` (`hypertools/plot/hyper_animation.py:45`), a
`(figure, animation)` **tuple subclass** whose `.on_frame()` method
(returns `self`, so calls chain — `hyper_animation.py:79-96`) lives on the
**wrapper**, not on the plain `matplotlib.animation.FuncAnimation` inside it.
Every prescribed plot call in every one of the five notebooks unpacks with
`fig, ani = hyp.plot(...)`, which — because `HyperAnimation` is a 2-tuple —
binds `ani` to element `[1]`, the **raw** `FuncAnimation`, discarding the
wrapper. Reproduced directly:

```
$ .venv/bin/python -c "
import numpy as np, hypertools as hyp
data = np.cumsum(np.random.default_rng(0).standard_normal((60,5)), axis=0)
fig, ani = hyp.plot(data, '-', animate=True, show=False, duration=1, frame_rate=5, size=(3,3))
print(type(ani), hasattr(ani, 'on_frame'))
ani.on_frame(lambda ctx: None)
"
<class 'matplotlib.animation.FuncAnimation'> False
AttributeError: 'FuncAnimation' object has no attribute 'on_frame'
```

Only `conversation_shape` calls `.on_frame()` post-hoc this way (market,
weather, painting, morph never call it; morph passes `title=` as a plot
kwarg directly), so this is the only notebook this bug touches — but it
means **no valid "N of 6" prediction can be derived for conversation_shape
from the plan's text as written**: it never gets that far. Under the
maintainer's own gate design this is doubly caught — the crashed cell 4
*does* have a non-empty `outputs` list (one `error`-type output, which
`cell.get("outputs")` alone cannot distinguish from success), so `Part 3`'s
"no error output" check exists precisely for cases like this one. Consistent
with the CLAUDE.md instruction against treating any encountered defect as
out of scope: this is flagged here, not fixed (the task instructs not to
edit the plan). Checked against every other Plan-4 audit document in
`notes/audit/` (`review_plan4_v2.md`, `plan4_landed_state.md`) — this
specific defect (the unpack discarding `.on_frame()`) is not previously
documented; it is new. Note also that the **actual, already-landed**
`examples/animate_conversation.py` on disk today (post commit `d730a085`,
independent of this plan) already avoids it, by binding
`anim = hyp.plot(...)` (no unpacking) and calling `anim.on_frame(decorate)`
— so applying Task 5's prescribed text verbatim would not just fail to add
serial-reveal titles, it would regress already-working code.

### morph_shapes_zoo (Task 6, plan lines 1825-1894 — 5 prescribed code cells)

Task 6 only rewrites the *tail* of the file; the plan explicitly defers to
"the current file" for imports/loading (kept verbatim except deleting one
import line). Pulled that kept content from the real
`examples/animate_morph_zoo.py` / current notebook, exactly as the plan's
own citations point there, so this is not fabricated — it is the plan's own
by-reference content.

| code-idx | content | class | reason |
|-|-|-|-|
| 0 | Colab install (unchanged) | **EMITS** | see note above |
| 1 | `import numpy as np` / `import hypertools as hyp` (the `from hypertools.plot import morph as _morph` line is the one thing Task 6 deletes here) | SILENT | import-only |
| 2 | `normalize`/`SHAPES`/`TITLES`/`N`/`CUBE_SCALE`/`rng`/`load`/`clouds`/`titles` | SILENT | last stmt (`titles = TITLES + [TITLES[0]]`) is an assignment |
| 3 | `rotations = [...]` / `duration, fps = 12, 20` / `fig, ani = hyp.plot(..., title=titles, show=False, ...)` | SILENT | bare assignment; rule 3 |
| 4 | `HTML(ani.to_jshtml())` (or the commented `ani.save(...)` alternative) | **EMITS** | bare trailing expression |

Predicted `{0, 4}` → **2 of 5**. (Morph has **zero** `print()` calls
anywhere in its prescribed rewrite — the only cell-table row that never
mentions "print".)

### Summary: where the plan's numbers disagree with this derivation

| notebook | `EXPECTED_OUTPUT_CELLS` (plan Task 8, line 2381) | plan's own per-task Step-N claim | this audit's derived set | count |
|-|-|-|-|-|
| market_forecast | 7 | "7/8" (Task 2 Step 6, plan:990) | {0,2,5,6,7} | **5** |
| weather_decades | 3 | "4/5, cells 3,5,7,9" (Task 3 Step 5, plan:1186) | {0,2,4} | **3** (right total, wrong cells) |
| painting_embeddings | 5 | "5/6" (Task 4 Step 5, plan:1420) | {0,2,5} | **3** |
| conversation_shape | 5 | "5/6" (Task 5 Step 5, plan:1792) | *crashes — not derivable* | **n/a** |
| morph_shapes_zoo | 4 | "4/5" (Task 6 Step 4, plan:1904) | {0,4} | **2** |

**All five of `EXPECTED_OUTPUT_CELLS`'s numbers are wrong**, and — beyond the
constants Task 8 hardcodes — **every one of Tasks 2-6's own per-task "Execute
and measure" expectations is also wrong**, for the identical reason: each
one assumes every non-install cell produces output, when several are bare
imports, bare assignments, or a `fig, ani = hyp.plot(..., show=False, ...)`
call. Weather is the instructive case: its total (3) happens to match
`EXPECTED_OUTPUT_CELLS`, but for entirely the wrong reason — the plan
assumed the imports cell (idx 1) and the plot-call cell (idx 3) contribute
while the install cell (idx 0) does not; the real cells with output are
exactly the complement, `{0, 2, 4}`. A count-only gate cannot tell these
apart; an index-set gate fails immediately and names the exact cell.

---

## Part 3 — the gate (real, runnable)

Written to match repo convention (`tests/test_colab_install_cell.py`,
`tests/test_notebook_install_gate.py`: plain `json`, a `REPO`-relative path,
`pytest.mark.parametrize`, a rationale-first module docstring). Kept in the
audit scratch area, not added to `tests/`, since adding it to the suite was
not requested and the underlying `EXPECTED_VISIBLE_OUTPUTS` values are only
valid for the CURRENT (pre-rewrite) notebooks (Part 1) — wiring it into CI
today would gate the wrong artifact.

```python
# -*- coding: utf-8 -*-
"""Real, runnable implementation of the maintainer's four-part notebook gate
(2026-08-01 audit of Plan 4 / docs/superpowers/plans/2026-07-28-hypertools-
1.1-examples-and-tutorials.md, Task 8).

Two variants are provided so the audit can show its work:

* ``test_every_code_cell_ran__LITERAL`` implements the prescription exactly
  as given -- "every code cell has a non-null execution_count" -- with no
  exceptions. Run alone, it FAILS on all five current notebooks, because the
  Colab install cell (index 0) is a `%pip install` + `%matplotlib inline`
  cell that is deliberately never executed by the tooling that produced
  these files (verified: `execution_count` is `None` for cell 0 in all five
  committed notebooks, and a `%pip install` from a plotting/test harness
  would mutate the environment it runs in). A gate that cannot pass on a
  cell nobody intends to run is not a usable gate.

* ``test_every_non_install_code_cell_ran`` is the corrected version: it
  scopes the same check to code cells other than a recognized Colab install
  cell (detected the same way `tests/test_notebook_install_gate.py` already
  detects one -- a `pip install` line, not a hardcoded index), which is
  exactly what the plan's own prose already assumes in five separate places
  ("cell 0's Colab install cell produces none").

Both the count-hiding failure mode this whole audit exists to fix, and the
maintainer's fix for it, are exercised for real here:

* ``EXPECTED_VISIBLE_OUTPUTS`` is an INDEX SET per notebook, not a count.
* ``test_no_error_outputs`` rejects any committed ``output_type == 'error'``.
* ``test_notebook_has_its_rendered_artifact`` asserts the notebook actually
  references, from a markdown cell, an animation file that exists on disk --
  today that is a companion ``.gif`` (see the module docstring note below);
  the assertion is a real filesystem check, not a cell-output inspection,
  because none of the five notebooks embeds the animation as a cell output
  at all (Part 1 of the audit: every code-cell output today is either
  `stream` text or, twice, a tqdm widget -- never `image/png`, never
  `text/html`).

No network, no mocks: real files, real json parsing.
"""
import json
import os
import re

import pytest

REPO = '/Users/jmanning/hypertools'
TUT_DIR = os.path.join(REPO, 'docs', 'tutorials')

STEMS = ['market_forecast', 'weather_decades', 'painting_embeddings',
         'conversation_shape', 'morph_shapes_zoo']

# Same detector `tests/test_notebook_install_gate.py` already uses for a
# real install line (not a hardcoded "index 0" assumption, so a notebook
# whose install cell moves, or that has none, is still handled correctly).
_INSTALL_LINE_RE = re.compile(
    r'^[%!]?\s*(?:pip[0-9]*|pipx|uv\s+pip|conda|mamba|python[0-9.]*\s+-m\s+pip)'
    r'\s+install\b', re.IGNORECASE)

#: Measured 2026-08-01 directly from the CURRENT committed notebooks (real
#: json parsing, see the audit report for the exact command). These are
#: NOT the Tasks 2-6 prescribed rewrite -- independently confirmed (Part 2
#: of the audit) that none of the five notebooks has been rewritten yet;
#: they are the pre-rewrite content, partially executed. Re-derive by
#: running this file's `_measure` over each notebook whenever a notebook's
#: cells change (that is the whole point of an index set over a guess).
EXPECTED_VISIBLE_OUTPUTS_CURRENT = {
    'market_forecast': {2, 4, 5, 6},
    'weather_decades': {2, 6},
    'painting_embeddings': {3, 5},
    'conversation_shape': {2, 5},
    'morph_shapes_zoo': {5},
}


def _read_nb(stem):
    path = os.path.join(TUT_DIR, f'{stem}.ipynb')
    with open(path, encoding='utf-8') as fh:
        return json.load(fh), path


def _code_cells(nb):
    return [c for c in nb['cells'] if c.get('cell_type') == 'code']


def _is_install_cell(cell):
    src = ''.join(cell.get('source', []))
    return any(_INSTALL_LINE_RE.match(line.strip())
               for line in src.splitlines())


# ---------------------------------------------------------------------------
# Part 1: every code cell actually ran.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('stem', STEMS)
def test_every_code_cell_ran__LITERAL(stem):
    """The maintainer's prescription taken literally: ALL code cells,
    no exceptions. Expected (and confirmed below, Part 4 of the audit)
    to FAIL on every current notebook because of cell 0."""
    nb, path = _read_nb(stem)
    code_cells = _code_cells(nb)
    unexecuted = [i for i, c in enumerate(code_cells)
                  if c.get('execution_count') is None]
    assert not unexecuted, (
        f'{stem}.ipynb: code cell(s) at index {unexecuted} have no '
        f'execution_count (partially executed notebook)')


@pytest.mark.parametrize('stem', STEMS)
def test_every_non_install_code_cell_ran(stem):
    """Corrected scope: exclude the Colab install cell, which this exact
    tooling deliberately never runs (a live `%pip install` inside a test/
    docs-build harness would mutate the environment running it). Every
    OTHER code cell must show a real execution_count."""
    nb, path = _read_nb(stem)
    code_cells = _code_cells(nb)
    unexecuted = [i for i, c in enumerate(code_cells)
                  if c.get('execution_count') is None
                  and not _is_install_cell(c)]
    assert not unexecuted, (
        f'{stem}.ipynb: non-install code cell(s) at index {unexecuted} '
        f'have no execution_count (partially executed notebook)')


# ---------------------------------------------------------------------------
# Part 2: visible outputs are an INDEX SET, not a count.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('stem', STEMS)
def test_visible_outputs_match_the_measured_index_set(stem):
    """A stray print() in the wrong cell shifts WHICH index carries output
    without changing the total -- a count-only gate cannot see that; an
    index-set gate fails immediately, pointing at the exact cell."""
    nb, path = _read_nb(stem)
    code_cells = _code_cells(nb)
    actual = {i for i, c in enumerate(code_cells) if c.get('outputs')}
    expected = EXPECTED_VISIBLE_OUTPUTS_CURRENT[stem]
    assert actual == expected, (
        f'{stem}.ipynb: cells with visible output are {sorted(actual)}, '
        f'expected {sorted(expected)} (added: {sorted(actual - expected)}, '
        f'missing: {sorted(expected - actual)}); re-run the measurement '
        f'script and re-derive EXPECTED_VISIBLE_OUTPUTS_CURRENT -- never '
        f'hand-edit the set to make this pass')


# ---------------------------------------------------------------------------
# Part 3: no committed error output.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('stem', STEMS)
def test_no_error_outputs(stem):
    nb, path = _read_nb(stem)
    for i, cell in enumerate(_code_cells(nb)):
        for out in cell.get('outputs', []):
            assert out.get('output_type') != 'error', (
                f"{stem}.ipynb: code cell {i} committed a traceback "
                f"({out.get('ename')}: {out.get('evalue')})")


# ---------------------------------------------------------------------------
# Part 4: the notebook actually contains its intended rendered artifact.
# ---------------------------------------------------------------------------
# Measured (Part 1 of the audit): none of the five notebooks embeds the
# animation as a CELL output -- there is no `image/png` or `text/html`
# output anywhere in any of the five files today. The current convention
# (commit 9b94d86f, 2026-07-30) is a companion GIF saved by the final code
# cell and referenced from the LAST markdown cell, exactly like
# conversation_trajectories.ipynb / streaming_data.ipynb / wikipedia_
# embeddings.ipynb already do. `EXPECTED_VISIBLE_OUTPUTS` alone is
# therefore not sufficient to prove "the reader sees the animation" -- a
# notebook could pass Parts 1-3 while its GIF reference is a typo or the
# file was never committed. This check is independent of cell outputs by
# design.
_IMG_MD_RE = re.compile(r'!\[[^\]]*\]\(([^)\s]+\.gif)\)')


@pytest.mark.parametrize('stem', STEMS)
def test_notebook_has_its_rendered_artifact(stem):
    nb, path = _read_nb(stem)
    nb_dir = os.path.dirname(path)
    refs = []
    for cell in nb['cells']:
        if cell.get('cell_type') != 'markdown':
            continue
        src = ''.join(cell.get('source', []))
        refs.extend(_IMG_MD_RE.findall(src))
    assert refs, (
        f'{stem}.ipynb: no markdown cell references a .gif -- the notebook '
        f'never shows its animation to a reader (nbsphinx_execute is '
        f"'never', docs/conf.py:131, so a bare code cell renders as code "
        f'and nothing else)')
    missing = [r for r in refs if not os.path.isfile(os.path.join(nb_dir, r))]
    assert not missing, (
        f'{stem}.ipynb: markdown references {missing} but the file does '
        f'not exist next to the notebook')
```

---

## Part 4 — validation: running the gate against the five CURRENT notebooks

```
$ .venv/bin/python -m pytest test_notebook_output_gate.py -v
...
5 failed, 20 passed in 0.06s
```

| test | result | why |
|-|-|-|
| `test_every_code_cell_ran__LITERAL` (×5) | **FAILED**, all 5 | cell 0 (Colab install) has `execution_count = None` in every notebook — the maintainer's prescription taken literally ("every code cell") cannot pass today, and would not pass even after Tasks 2-6 land, because nothing in the design ever executes the install cell |
| `test_every_non_install_code_cell_ran` (×5) | **passed**, all 5 | once the install cell is excluded (the same way the repo's own `test_notebook_install_gate.py` already detects one), every other code cell in all five notebooks has a real `execution_count` |
| `test_visible_outputs_match_the_measured_index_set` (×5) | **passed**, all 5 | `EXPECTED_VISIBLE_OUTPUTS_CURRENT` was populated *from* the Part 1 measurement, so this is a tautological check today by construction — its value is entirely in catching future drift, not in today's pass |
| `test_no_error_outputs` (×5) | **passed**, all 5 | confirmed independently in Part 1: no notebook has committed a traceback |
| `test_notebook_has_its_rendered_artifact` (×5) | **passed**, all 5 | all five markdown-referenced GIFs exist on disk (`market_forecast.gif`, `weather_decades.gif`, `painting_embeddings.gif`, `conversation_shape.gif`, `morph_zoo.gif`) |

**A gate that cannot even describe today's notebooks is not ready** — and
the maintainer's design, taken 100% literally, is exactly that gate: Part 1
of every check would fail forever on the install cell, for all five
notebooks, independent of anything Tasks 2-6 do. The fix is narrow and
already implicit in the plan's own prose (repeated five times: "cell 0's
Colab install cell produces none") — scope the execution_count check to
non-install cells, using a real detector rather than a hardcoded index (a
detector survives the install cell moving or a notebook adding/losing one;
an index does not). With that one, already-plan-endorsed exception, the
remaining three parts of the design are sound and pass cleanly against real
files today.

---

## Part 5 — recommendation: how the plan should carry these constants

**Verdict on the proposed workflow (build → measure → record): adopt it, but
as three steps, not two, and make the middle step a nameable one.** Writing
`EXPECTED_VISIBLE_OUTPUTS` by hand before the notebook exists is exactly how
the plan got here — the current `EXPECTED_OUTPUT_CELLS` and every one of
Tasks 2-6's own "N/M code cells produced output" expectations were written
before the code they describe, and Part 2 above shows **all of them** are
wrong, most off by more than one, one of them (conversation) not even
executable. There is no way to have the right index set without first
having the artifact. Concretely:

1. **Build.** Tasks 2-6 already do this (Step 1/2 of each).
2. **Measure.** Tasks 2-6 already run `scripts/execute_tutorial.py` right
   after building (Task 2 Step 6, Task 3 Step 5, Task 4 Step 5, Task 5 Step
   5, Task 6 Step 4) — but that script only prints a **count**
   (`f'{path}: {executed}/{total} code cells produced output'`,
   plan:699-702). One line fixes this:
   `print('  visible-output indices:', sorted(i for i, c in enumerate(code_cells) if c.get("outputs")))`.
   With that, each task's own existing step becomes the single, authoritative
   place the set for *that* notebook is produced — Task 8 should copy five
   printed sets, not invent them centrally after the fact (which is what
   produced the current wrong numbers).
3. **Record, reviewed.** This is the step the two-step version of the
   proposal skips, and skipping it is exactly how a build→measure→record
   gate degrades into a snapshot/regression test that can no longer catch
   its own bugs — a test that pins whatever the code currently does, rather
   than what it should do. This audit produced two concrete, real examples
   of what an unreviewed recording would have silently enshrined:
   - **market_forecast cell 6** verified to emit a stray `Text(0.66, 0.015,
     'next-day direction, last 30 sessions (50% = coin flip)')` — almost
     certainly not the intended output of that cell (a `fig.text(...)`
     side-panel block), and a blind "record whatever the measure script
     printed" step would have pinned this cosmetic leak forever, since
     nothing about it *looks* wrong from an index number alone.
   - **conversation_shape cell 4**, if executed under a harness that
     swallows cell errors (e.g. `allow_errors=True`, sometimes reached for
     under time pressure to "get past" a failure), would show up as "has
     outputs" — an index-only recording step cannot tell a `stream` success
     from a committed `error` output. This is precisely why Part 3 of the
     gate (`test_no_error_outputs`) must remain a permanent, always-on
     assertion **independent of whatever gets recorded** — it is the one
     check in the four-part design that a bad recording cannot defeat.

   **Concrete mitigation:** require a human to look at a per-cell *preview*
   (output type + first line of stream text, or "execute_result:
   `Text(...)`") before the constant is committed — not just the bare index
   list — and require a one-line justification comment on the constant for
   any index that is not an obvious `print()` (matching how `DEFECT_MARKERS`
   entries already carry a "the native API that replaced it" justification,
   Task 8 Step 2). Doing the recording **after** Task 8 Step 8's doc-build
   check ("verify the five tutorial pages actually show something") rather
   than before it also helps: a stray `Text(...)` repr is far more obvious
   staring out of a rendered HTML page than as a bare integer in a Python
   set literal.

4. **Keep checks 1/3/4 invariant-based, not measurement-based.** They should
   never be *derived from* a recording run the way the index set is — "every
   non-install cell executed", "no error output", "the artifact file exists"
   are properties the notebook must have regardless of what any one build
   happened to produce, which is exactly why they survive even if
   `EXPECTED_VISIBLE_OUTPUTS` itself is ever rubber-stamped.

---

## Appendix — commands run

Notebook measurement (Part 1):

```bash
.venv/bin/python /path/to/scratchpad/measure_notebooks.py     # real json.load over the 5 notebooks
.venv/bin/python /path/to/scratchpad/dump_full.py              # full cell source + full output text/data
```

Provenance check (current notebooks are pre-rewrite):

```bash
find /Users/jmanning/hypertools/scripts -name "execute_tutorial.py" -o -name "measure_native_ratio.py"
# (no output -- neither script exists yet; both are prescribed, not built)
git log --oneline -5 -- docs/tutorials/market_forecast.ipynb
# 9b94d86f fix(docs): execute the five new tutorials; repair make html at the source
# 4d1d2223 docs(examples): add five animated gallery demos and refresh the tutorials
ls docs/tutorials/*.gif   # market_forecast.gif, weather_decades.gif, painting_embeddings.gif,
                           # conversation_shape.gif, morph_zoo.gif all present
```

Cell-0 text check (corrected after first draft — see below):

```bash
.venv/bin/python -c "
import json
srcs = [''.join(json.load(open(f'docs/tutorials/{s}.ipynb'))['cells'][0]['source'])
        for s in ['market_forecast','weather_decades','painting_embeddings',
                   'conversation_shape','morph_shapes_zoo']]
print('distinct cell-0 sources:', len(set(srcs)))"
# -> distinct cell-0 sources: 2  (NOT 1 -- first draft of this report wrongly
#    assumed identical text without running this check; corrected here)
.venv/bin/python -c "
import json
for s in ['market_forecast','weather_decades','painting_embeddings',
          'conversation_shape','morph_shapes_zoo']:
    src = ''.join(json.load(open(f'docs/tutorials/{s}.ipynb'))['cells'][0]['source'])
    print(s, 'WITH sentence-transformers line' if 'sentence-transformers' in src else 'plain')"
# market_forecast        plain
# weather_decades        plain
# painting_embeddings    WITH sentence-transformers line
# conversation_shape     WITH sentence-transformers line
# morph_shapes_zoo       plain
```

Rule verification (empirical, real `IPython` shell, no mocks):

```bash
.venv/bin/python -c "
from IPython.testing.globalipapp import get_ipython
ip = get_ipython()
ip.run_cell('def f(x):\n    return x*2\n')
ip.run_cell('for i in range(3):\n    f(i)\nf(99)\n')"   # -> prints 'Out[0]: 198'
```

`show=False` non-display (source + reproduction):

```bash
grep -n 'flush_figures\|plt.close(fig)' hypertools/plot/plot.py   # :5101-5130, GH #148
```

`HyperAnimation` unpack bug (real reproduction):

```bash
MPLBACKEND=Agg .venv/bin/python -c "
import numpy as np, hypertools as hyp
data = np.cumsum(np.random.default_rng(0).standard_normal((60,5)), axis=0)
fig, ani = hyp.plot(data, '-', animate=True, show=False, duration=1, frame_rate=5, size=(3,3))
print(type(ani), hasattr(ani, 'on_frame'))
ani.on_frame(lambda ctx: None)"
# <class 'matplotlib.animation.FuncAnimation'> False
# AttributeError: 'FuncAnimation' object has no attribute 'on_frame'
```

market cell 6 trailing-expression reproduction (plan's verbatim code): see
Part 2, market_forecast row 6 for the full script; result
`Out[0]: Text(0.66, 0.015, 'next-day direction, last 30 sessions (50% = coin flip)')`.

Gate validation (Part 4):

```bash
.venv/bin/python -m pytest test_notebook_output_gate.py -v
# 5 failed, 20 passed in 0.06s
```

Prior-review cross-check (no duplicate discovery claimed without checking):

```bash
grep -n "EXPECTED_OUTPUT_CELLS\|on_frame\|unattainable" notes/audit/review_plan4_v2.md
grep -rn "AttributeError" notes/audit/*.md | grep -i "on_frame\|FuncAnimation"   # -> no hits; the
                                                                                  # unpack bug is new
```
