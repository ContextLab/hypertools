"""The gallery examples and their notebooks must SHOWCASE hypertools.

Measured on 2026-07-26/28, before the 1.1 examples plan: 48 of 739 code
lines across the five launch examples belonged to a hypertools call (6.5%;
the same five measure 48/723 = 6.6% since `d730a085` shrank morph),
and 37.9% of the code either re-implemented something native or worked
around a gap. This module makes the fix permanent -- it fails if a defect
marker comes back, or if a file drifts back above its size budget.

**The native-code ratio is REPORTED, not gated.** v1 of this plan asserted a
per-file minimum ratio and picked the floors before the rewrites existed;
measured against the plan's own proposed code, four of the five missed their
own floors (market 14.7% vs 26, paintings 12.5% vs 20, conversation 18.9% vs
25, morph 22.2% vs 26), so the gate could not have gone green no matter how
good the rewrite was. Raising the floors to whatever the code happens to
measure would make the gate tautological, and the ratio is trivially gamed
in the wrong direction anyway -- splitting one `hyp.plot(...)` call across
six lines "improves" it, and so does deleting a comment. What the ratio is
genuinely good for is watching a trend, so this module PRINTS it and asserts
only things that cannot be satisfied by reformatting:

1. no private API or named defect pattern (`DEFECT_MARKERS`);
2. a maximum code-line budget per file;
3. executable semantic checks -- the example actually produces the artifact
   it claims (`test_examples_produce_their_stated_artifact`);
4. exact notebook execution success (Step 2c).

No network, no mocks: it reads the committed files.
"""
import ast
import contextlib
import os
import re

import numpy as np
import pytest

from scripts.measure_native_ratio import measure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A notebook holds its script's code, plus a Colab install cell, plus a
#: display cell.
#:
#: MEASURED: the largest install cell across the five is 3 code lines
#: (paintings and conversation are 3, the other three are 2).
#: NOT MEASURED -- a design decision: the 2-line display cell
#: (`from IPython.display import HTML` + `HTML(ani.to_jshtml())`). No
#: current notebook has one (`grep -l "to_jshtml\|IPython.display"` over
#: the five returns nothing), so this is a budget for a cell that does not
#: exist yet, and it is LOAD-BEARING: conversation has only 2 lines of
#: headroom. If the display cell turns out to be 3 lines, or a second one
#: is added, raise NOTEBOOK_OVERHEAD here -- in the plan -- rather than
#: letting a task quietly exceed its budget.
#:
#: This is why the notebook budgets are DERIVED rather than written down. v2
#: wrote them down and set two of them BELOW their own script's -- paintings
#: 110 against a script of 118, conversation 76 against 90 -- which no
#: correct notebook can satisfy, whatever the metric does. Deriving makes
#: that class of mistake impossible and means only ONE number per example is
#: ever chosen by hand.
NOTEBOOK_OVERHEAD = 5

#: script path -> max code lines. EVERY figure here is MEASURED, on
#: 2026-08-04, against the exact code this plan prescribes: each task's
#: rewrite block was transcribed to a file, the Step 0b loader/builder split
#: was applied to it, and both were run through `measure()` below. No entry is
#: projected, and none is weather's overhead carried across -- doing that is
#: what made two of them unattainable.
#:
#:      file            rewrite   split   overhead
#:      market            124      145      +21   (re-measured 2026-08-17)
#:      weather            56       73      +17
#:      paintings         112      135      +23   (two fetch sites)
#:      conversation       88      106      +18
#:      morph              26       43      +17
#:
#: Each budget is the SPLIT figure rounded up to the next multiple of 5. That
#: rounding is the entire allowance: at most 4 lines, for wording differences
#: an implementer will legitimately introduce, and nothing else. A file that
#: exceeds its budget has grown beyond what the plan prescribes.
#:
#: The two that changed and why: paintings' placeholder 133 was UNATTAINABLE
#: (the prescribed file measures 135) and conversation's 105 was short by one
#: (it measures 106). Weather's 77 came from a +15 measured on the file as it
#: stands TODAY (195 -> 210), but Task 3 REPLACES that file; re-measured
#: against the rewrite the overhead is +17 on a 56-line base, so 75.
#: `test_file_is_within_its_size_budget` failing is still an instruction, not
#: a licence to trim the split.
SCRIPT_BUDGETS = {
    # market's was PROJECTED (140), then MEASURED at 145 on 2026-08-17. The
    # +16 split overhead the projection inherited was wrong twice over --
    # measured +18 on the v4 rewrite as first landed, and +21 once the
    # rewrite gained the loader's offline degrade, the atomic cache write
    # and the per-measure display scaling.
    #
    # RE-MEASURED 2026-08-18: **148**, so this gate is RED and correctly so.
    # Threading `sectors=` through the five sites that had closed over the
    # module-level SECTORS cost the three lines, and the earlier comment
    # here still claimed the file sat exactly on 145 -- a measurement that
    # had stopped being true. The budget is deliberately NOT raised to
    # cover it: the representation is still under review, and moving a
    # limit to fit an interim design is how a budget stops meaning
    # anything. Re-measure and reset this number once the rewrite settles.
    'examples/animate_market_forecast.py': 145,   # 148 measured -> OVER by 3
    'examples/animate_weather_decades.py': 75,    # 73 measured -> 75
    'examples/animate_painting_embeddings.py': 140,  # 135 measured -> 140
    'examples/animate_conversation.py': 110,      # 106 measured -> 110
    'examples/animate_morph_zoo.py': 45,          # 43 measured -> 45
}

#: script stem -> notebook, so the derivation below has something to pair.
NOTEBOOKS = {
    'examples/animate_market_forecast.py': 'docs/tutorials/market_forecast.ipynb',
    'examples/animate_weather_decades.py': 'docs/tutorials/weather_decades.ipynb',
    'examples/animate_painting_embeddings.py': 'docs/tutorials/painting_embeddings.ipynb',
    'examples/animate_conversation.py': 'docs/tutorials/conversation_shape.ipynb',
    'examples/animate_morph_zoo.py': 'docs/tutorials/morph_shapes_zoo.ipynb',
}

#: (path, max_code_lines) for every gated file -- scripts as chosen,
#: notebooks as derived.
BUDGETS = ([(p, n) for p, n in SCRIPT_BUDGETS.items()]
           + [(NOTEBOOKS[p], n + NOTEBOOK_OVERHEAD)
              for p, n in SCRIPT_BUDGETS.items()])


def test_notebook_budgets_are_derived_not_written_down():
    """The v2 defect, pinned so it cannot return.

    Asserts the DERIVATION is still in force -- each notebook limit equals
    its script's plus exactly `NOTEBOOK_OVERHEAD` -- not merely that it is
    larger. `>= limits[script]` would be `n + 5 >= n`, true for every `n`,
    a comment wearing a test's clothes and the same inert-assertion defect
    this plan has now hit twice (`_save_count >= 1`, `'morph' in 'morph'`).

    Equality CAN fail, and fails on the thing actually worth catching:
    someone replacing the comprehension with hand-written numbers, which is
    how paintings ended up at 110 against a script of 118.
    """
    limits = dict(BUDGETS)
    for script, nb in NOTEBOOKS.items():
        assert limits[nb] == limits[script] + NOTEBOOK_OVERHEAD, (
            f'{nb} is budgeted at {limits[nb]}, but the derivation says '
            f'{limits[script]} + {NOTEBOOK_OVERHEAD} = '
            f'{limits[script] + NOTEBOOK_OVERHEAD}. Change the SCRIPT budget '
            f'and let the notebook follow; do not hand-write this one.')


#: Private reaches that are DELIBERATELY retained, with the reason. Contract
#: 3 bans private API only where a public equivalent exists; these two have
#: none, are one-time setup rather than per-frame work, and each carries an
#: inline rationale in the source. Anything NOT listed here still fails, so
#: a new reach cannot creep in, and each of these was reviewed rather than
#: assumed. Landed in `d730a085` with measurements.
#:
#: **It is now EMPTY, which is the goal state.** Both entries belonged to
#: `examples/animate_market_forecast.py`, and Task 2's rewrite removed the
#: code they covered: the `ani._args` readback of the drawn line existed only
#: to recover plot's own reduce->drawn affine by hand, which `predict=`/`t=`
#: replaced, and the `hypertools._shared` import pulled in `antialias_line`
#: for a hand-drawn forecast fan that `forecast_trail=16` replaced. Verified
#: 2026-08-17: neither pattern appears in `examples/animate_market_forecast.py`
#: any more, and the empty case is asserted rather than skipped by
#: `test_every_allowlisted_reach_is_still_present_and_still_explained`.
#:
#: **That test is RED right now, and correctly so.** Emptying this dict turns
#: on its empty-case branch, which scans EVERY gated file -- and
#: `docs/tutorials/conversation_shape.ipynb` still does `lines = ani._args[1]`
#: and `[... for a in ani._args[0]]` (cell 9), with no inline rationale within
#: `RATIONALE_WINDOW` lines (measured 2026-08-17). Its green before market's
#: rewrite was a FALSE green: the two market entries made the test take the
#: allowlist branch, which only ever looked at the market file, so
#: conversation's reach was never examined. The failure names Task 4's file
#: and is its to-do, not something to be silenced by re-adding a dead entry.
PRIVATE_API_EXCEPTIONS = {}

#: Every one of these was found in the launch examples or the older
#: tutorials and removed. Each maps to the native API that replaced it.
DEFECT_MARKERS = {
    r'\bSentenceTransformer\b': "use vectorizer='<hf-model-id>', semantic=None, corpus=None",
    r'ani\._func': 'use on_frame= / HyperAnimation.on_frame()',
    r'ani\._args': 'use the FrameContext passed to on_frame=',
    r'hypertools\._shared': 'private module; use a documented kwarg',
    r'from hypertools\.plot import morph': "use title=[...] for per-segment names",
    r'\bantialias_line\b': 'plot() antialiases every drawn line already',
    r'\bffmpeg\b': ("save_path='*.gif' needs no ffmpeg -- .gif/.png/.apng "
                    "go through PillowWriter; see animate.py's writer "
                    "dispatch and plot()'s save_path docstring"),
    r'morph_schedule|frame_to_segment': 'the morph schedule is the library\'s business',
}

def _read(path):
    full = os.path.join(REPO, path)
    with open(full, encoding='utf-8') as handle:
        return handle.read()


def _code_text(path):
    """Code only -- and DOCSTRINGS ARE NOT CODE here.

    Two reasons, both load-bearing:

    1. Markdown/prose may still discuss a removed workaround, so notebook
       markdown cells are excluded.
    2. `d730a085` documented each migration by NAMING the pattern it
       removed -- `animate_weather_decades.py` and `animate_conversation.py`
       both contain the string ``ani._func`` inside a docstring explaining
       that the monkeypatch is gone. Scanning raw source would fail those
       files for their own documentation.

    This shares `strip_docstrings` with `scripts/measure_native_ratio.py`
    rather than re-implementing it: the two counters previously disagreed
    (one stripped, one did not, so identical source measured differently as
    .py and .ipynb), and a shared callee cannot drift from itself.
    """
    from scripts.measure_native_ratio import strip_docstrings
    if path.endswith('.ipynb'):
        import json
        nb = json.loads(_read(path))
        # PER CELL, exactly as `_code_lines_nb` does -- not concatenated
        # first. Concatenating makes the two disagree on the same file: a
        # notebook whose first cell holds an unclosed bare `"""` note
        # measured 5 code lines under the budget test while this function
        # returned '' and the defect-marker ban passed unconditionally on an
        # empty string. That is the F2 defect class -- two counters
        # disagreeing on identical input -- relocated into the code written
        # to eliminate it.
        kept = []
        for cell in nb['cells']:
            if cell.get('cell_type') != 'code':
                continue
            kept.extend(strip_docstrings(
                ''.join(cell['source']).split('\n')))
        return '\n'.join(kept)
    return '\n'.join(strip_docstrings(_read(path).split('\n')))


def test_a_docstring_naming_a_removed_pattern_is_not_a_defect():
    """Pins the above. `d730a085` explains each migration by naming what it
    removed; that is documentation, not a reach. Red before the docstring
    strip: weather and conversation both failed the marker scan for their
    own prose.

    **The migration docstrings are KEPT deliberately, by Tasks 3 and 5.**
    They are the record of why these files no longer monkeypatch
    `ani._func`, and this test is what stops the docstring-stripping in
    `_code_text` from being "fixed" in a way that starts counting prose as
    a reach again. If a rewrite drops the prose, this fails -- and the
    right response is to put the sentence back, not to delete the test.
    """
    for path in ('examples/animate_weather_decades.py',
                 'examples/animate_conversation.py'):
        assert 'ani._func' in _read(path), (
            f'{path}: expected the migration docstring to still name the '
            f'pattern it replaced')
        assert 'ani._func' not in _code_text(path), (
            f'{path}: the docstring mention leaked into the scanned code')


def _docstring_lines(path):
    """1-based line numbers occupied by docstrings in a .py file.

    Used to tell a real private reach from a docstring that merely NAMES
    one while explaining why it was removed (or why it has to stay).

    Unparseable input RAISES here, unlike `strip_docstrings`, which keeps
    every line. The polarity is opposite and so is the safe default:
    `strip_docstrings` *removes* lines, so failing open keeps code in the
    scan; this function *excludes* lines from a search, so failing open
    silently converts every docstring mention into a reported reach --
    which is precisely what happened when this was tested with the file
    made unparseable. It blamed line 34 (prose in the market example's
    Coordinate note) for a defect that did not exist, while the real
    problem -- a file that does not parse -- went unnamed. An example that
    does not parse is a hard failure in its own right; say so.
    """
    if not path.endswith('.py'):
        return set()
    try:
        tree = ast.parse(_read(path))
    except SyntaxError as exc:
        raise AssertionError(
            f'{path} does not parse ({exc.msg} at line {exc.lineno}), so '
            f'docstring spans cannot be computed. Fix the file: every '
            f'shipped example must be importable and executable.') from exc
    spans = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return spans


def _parsable_code(path):
    """Code text that `ast.parse` will accept.

    Notebook cells legitimately contain IPython magics (`%pip install`,
    `%matplotlib inline`) and shell escapes (`!cmd`), which are not Python
    and raise SyntaxError. Commenting them out preserves line numbering,
    which keeps any reported position meaningful.
    """
    return '\n'.join(('# ' + line) if line.lstrip()[:1] in ('%', '!') else line
                      for line in _code_text(path).split('\n'))


@pytest.mark.parametrize('path,max_code', BUDGETS)
def test_file_is_within_its_size_budget(path, max_code):
    code, _native = measure(os.path.join(REPO, path))
    assert code <= max_code, (
        f'{path}: {code} code lines exceeds the {max_code}-line budget')


def test_native_ratio_is_reported(capsys):
    """REPORTED, not gated -- see the module docstring. Fails only if a file
    is missing or unparseable, so the number can never be met by
    reformatting. Read it with `pytest -s` or in the CI log."""
    rows = []
    for path, _max_code in BUDGETS:
        full = os.path.join(REPO, path)
        # `measure()` raises FileNotFoundError on a moved/renamed file long
        # before any assert here could report it, so check existence first
        # -- otherwise the "moved or renamed?" message is unreachable.
        assert os.path.exists(full), f'{path}: moved or renamed?'
        code, native = measure(full)
        assert code > 0, f'{path}: parsed to zero code lines'
        rows.append((path, code, native, 100.0 * native / code))
    with capsys.disabled():
        print('\nnative-code ratio (reported, not gated):')
        for path, code, native, ratio in rows:
            print(f'  {ratio:5.1f}%  {native:3d}/{code:3d}  {path}')


@pytest.mark.parametrize('path,_max', BUDGETS)
@pytest.mark.parametrize('marker,fix', sorted(DEFECT_MARKERS.items()))
def test_no_defect_marker_in_the_launch_examples(path, _max, marker, fix):
    if (path, marker) in PRIVATE_API_EXCEPTIONS:
        pytest.skip(f'allowlisted: {PRIVATE_API_EXCEPTIONS[(path, marker)]}')
    text = _code_text(path)
    assert not re.search(marker, text), (
        f'{path} contains {marker!r} again -- {fix}')


#: How far from an allowlisted private reach its rationale may sit. 15 lines
#: is the size of a comment block plus the statement it explains -- close
#: enough that a reader who lands on the reach sees the reason without
#: scrolling.
RATIONALE_WINDOW = 15


def test_every_allowlisted_reach_is_still_present_and_still_explained():
    """An allowlist entry that no longer matches anything is dead weight --
    it would silently permit a pattern nobody uses. And an allowlisted reach
    with no inline rationale is exactly the 'private API taught as normal'
    that Contract 3 exists to prevent.

    The rationale must sit WITHIN `RATIONALE_WINDOW` lines of the reach, not
    merely somewhere in the file. An earlier version searched the whole file
    for the words 'deliberately' or 'no public', which a 380-line example
    satisfies by accident -- it would have passed even if the explanation
    were 200 lines from the code it explains, or explained something else
    entirely.

    **An EMPTY allowlist is the GOAL state, not a failure** -- Task 2's
    rewrite is supposed to remove market's two reaches, at which point
    Contract 3's ban is absolute and there is nothing left to allowlist.
    But "empty" must mean *the reaches are gone*, not *someone deleted the
    entries*, and a `for` loop over an empty dict asserts nothing at all --
    the vacuous-gate class this plan has shipped repeatedly. So the empty
    case gets its own real assertion.
    """
    if not PRIVATE_API_EXCEPTIONS:
        for path, _max in BUDGETS:
            for marker in (r'ani\._args', r'hypertools\._shared'):
                assert not re.search(marker, _code_text(path)), (
                    f'{path} reaches {marker!r} but PRIVATE_API_EXCEPTIONS '
                    f'is empty. Either restore the allowlist entry (with '
                    f'its recorded reason and MAINTAINER SIGN-OFF) or '
                    f'remove the reach; an empty allowlist must mean the '
                    f'reaches are gone.')
        return
    for (path, marker), reason in PRIVATE_API_EXCEPTIONS.items():
        lines = _read(path).split('\n')
        # Skip matches inside docstrings: `animate_market_forecast.py`'s
        # module docstring explains the reach by NAMING it
        # (`ani._args[1][0]`, in the "Coordinate note" paragraph), and that
        # prose is documentation, not a second reach. Same reason
        # `_code_text` strips docstrings before the marker scan.
        doc_lines = _docstring_lines(path)
        hits = [i for i, line in enumerate(lines)
                if re.search(marker, line) and (i + 1) not in doc_lines]
        assert hits, (
            f'{path} no longer contains {marker!r}; drop the '
            f'PRIVATE_API_EXCEPTIONS entry rather than leaving it to permit '
            f'a pattern that is gone')
        for i in hits:
            # +1 on the upper bound: a slice end is exclusive, so without it
            # the window reaches 15 lines back and only 14 forward.
            window = '\n'.join(lines[max(0, i - RATIONALE_WINDOW):
                                     i + RATIONALE_WINDOW + 1])
            explained = ('deliberately' in window or 'no public' in window
                         or 'no publicly' in window)
            assert explained, (
                f'{path}:{i + 1} reaches {marker!r} with no rationale within '
                f'{RATIONALE_WINDOW} lines. Contract 3 allowlists it only '
                f'because the source explains itself where a reader will '
                f'find it (reason on record: {reason})')


#: Members that live ONLY on the `HyperAnimation` wrapper. Unpacking or
#: indexing a plot result throws the wrapper away, so reaching any of these
#: on a name that came out of a tuple is an `AttributeError`.
#:
#: `figure`/`animation` are included because they are wrapper properties
#: too, and `n_frames`/`n_segments`/`draw_frame` because Step 0 ADDS them --
#: a guard that greps only for `.on_frame(` would widen the trap and leave
#: the check where it was.
WRAPPER_ONLY = ('on_frame', 'n_frames', 'n_segments', 'draw_frame',
                'figure', 'animation')


#: Properties that hand back the RAW FuncAnimation, discarding the wrapper.
UNWRAPPING_ATTRS = ('animation',)


def _hypertools_names(tree):
    """(module aliases, bare names) in this file that refer to hypertools.

    Learned from the file's OWN imports. An earlier version matched any
    attribute call named `plot`, which made matplotlib's `ax.plot` and
    pandas' `df.plot` collateral -- and `Line2D.figure`/`Axes.figure` are
    real public attributes, so `WRAPPER_ONLY` containing `figure` turned
    them into false positives with a factually wrong message. The trigger
    was already present in a gated file:
    `examples/animate_market_forecast.py` does
    `fc_line, = ax.plot([], [], [], '--', ...)`, so `fc_line` already
    entered the unpacked set, and the test passed only because nothing
    happened to read `fc_line.figure`.
    """
    mods, bare = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == 'hypertools':
                    mods.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if (node.module or '').split('.')[0] == 'hypertools':
                for alias in node.names:
                    if alias.name == 'plot':
                        bare.add(alias.asname or alias.name)
    return mods, bare


def _bindings(node):
    """`(targets, value)` for every node kind that binds a NAME.

    `ast.Assign` is not the only one, and a guard that walks only `Assign`
    is evaded by three ordinary spellings -- measured, not imagined:
    `ani: object = anim[1]` (`AnnAssign`), `fig, ani = (anim := hyp.plot(d))`
    (`NamedExpr`), and `for fig, ani in [...]` (handled separately, since
    its value is an ITERABLE of wrappers rather than a wrapper).
    """
    if isinstance(node, ast.Assign):
        return node.targets, node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return [node.target], node.value
    if isinstance(node, ast.NamedExpr):
        return [node.target], node.value
    return (), None


def _unpacked_wrapper_uses(source):
    """[(name, attr), ...] for names holding an UNPACKED plot result that
    then reach a wrapper-only member.

    Raises `SyntaxError` if `source` will not parse -- the caller must
    handle it. Returning `[]` there would make this silently vacuous on
    exactly the files most likely to be odd, which is the
    assertion-that-cannot-fail class this plan has shipped four revisions
    running.

    **What this deliberately does NOT catch**, verified by
    `test_the_contract_8_guard_actually_detects` so the list cannot rot:
    `getattr(hyp, 'plot')(d)` (dynamic dispatch); a wrapper stored in and
    read back out of a dict or list; `fig, *rest = hyp.plot(d)` then
    `rest[0]`; `with hyp.plot(d) as (fig, ani)` (which cannot run at all --
    `HyperAnimation` is not a context manager); an unpacked name passed
    into a function and used there (interprocedural); a nested target
    (`(fig, ani), z = hyp.plot(d), 1`); and `for fig, ani in R` where `R`
    is a NAME bound to a list of results rather than a literal. No AST pass
    is complete against arbitrary Python; the honest move is to name the
    holes rather than imply there are none.

    That list was itself wrong once, which is worth recording: it named
    five exotic spellings while omitting the plainest of all -- a bare
    `b = ani` after an unpack. The guard propagated aliases in the WRAPPER
    direction (`a = hyp.plot(d)`, `b = a`) but not in the UNPACKED
    direction, so a reader who assumed the obvious symmetry was wrong, and
    the docstring implied a completeness the code did not have. Both
    directions now propagate, to a real fixed point rather than a fixed
    three passes.
    """
    tree = ast.parse(source)
    mods, bare = _hypertools_names(tree)

    def is_plot_call(node):
        if not isinstance(node, ast.Call):
            return False
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == 'plot':
            return isinstance(fn.value, ast.Name) and fn.value.id in mods
        return isinstance(fn, ast.Name) and fn.id in bare

    wrappers, unpacked = set(), set()

    def yields_wrapper(value):
        """`value` evaluates to the `HyperAnimation` wrapper itself."""
        if is_plot_call(value):
            return True
        if isinstance(value, ast.Name) and value.id in wrappers:
            return True
        if isinstance(value, ast.NamedExpr):        # (anim := hyp.plot(d))
            return yields_wrapper(value.value)
        return False

    def unwraps(value):
        """`hyp.plot(...).animation` or `<wrapper>.animation` -- the
        documented property that hands back the raw FuncAnimation, and so
        the most plausible form of this bug after direct unpacking."""
        return (isinstance(value, ast.Attribute)
                and value.attr in UNWRAPPING_ATTRS
                and yields_wrapper(value.value))

    def note_unpack(target):
        """Record the elements that LOSE the wrapper -- index >= 1 only.

        `HyperAnimation` is `(figure, animation)`, so element 0 of an
        unpack is a genuine `matplotlib.figure.Figure`. Recording it makes
        `fig.figure` -- real, public matplotlib API -- a violation, and
        reports it as "`fig` ... is a raw FuncAnimation and has no
        .figure", which is wrong twice over. That is the same
        factually-wrong-message class `_hypertools_names` was written to
        kill for `ax.plot`/`df.plot`; it survived here because
        `WRAPPER_ONLY` contains `figure` and this function did not care
        which slot a name came from. It cares now.
        """
        if isinstance(target, (ast.Tuple, ast.List)):
            unpacked.update(e.id for e in target.elts[1:]
                            if isinstance(e, ast.Name))

    # Iterate to a real fixed point rather than a fixed number of passes: a
    # long alias chain needs one pass per hop, and `range(3)` silently
    # stopped resolving at the fourth.
    changed = True
    while changed:
        before = (len(wrappers), len(unpacked))
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                it = node.iter
                elts = it.elts if isinstance(
                    it, (ast.List, ast.Tuple, ast.Set)) else []
                if any(yields_wrapper(e) for e in elts):
                    note_unpack(node.target)
                continue
            targets, value = _bindings(node)
            for target in targets:
                if isinstance(target, ast.Name):
                    if is_plot_call(value) \
                            or (isinstance(value, ast.Name)
                                and value.id in wrappers) \
                            or (isinstance(value, ast.NamedExpr)
                                and yields_wrapper(value)):
                        wrappers.add(target.id)
                    elif unwraps(value):
                        unpacked.add(target.id)
                    elif isinstance(value, ast.Subscript) \
                            and yields_wrapper(value.value):
                        unpacked.add(target.id)
                    elif isinstance(value, ast.Name) and value.id in unpacked:
                        # `b = ani` -- the UNPACKED direction of the alias
                        # propagation above. The plainest evasion there is,
                        # and the one an earlier version missed while
                        # handling the wrapper direction.
                        unpacked.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    if yields_wrapper(value):
                        note_unpack(target)
        if (len(wrappers), len(unpacked)) == before:
            changed = False
    return sorted({(n.value.id, n.attr) for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute)
                   and n.attr in WRAPPER_ONLY
                   and isinstance(n.value, ast.Name)
                   and n.value.id in unpacked})


#: Sources the guard MUST flag, MUST leave alone, and is DOCUMENTED not to
#: catch. Every entry was constructed and run; `note` records why it exists.
GUARD_MUST_FLAG = {
    'direct unpack then on_frame': 'fig, ani = hyp.plot(d)\nani.on_frame(cb)\n',
    'unpack from a wrapper variable':
        'anim = hyp.plot(d)\nfig, ani = anim\nani.on_frame(cb)\n',
    'draw_frame on an unpacked name': 'fig, ani = hyp.plot(d)\nani.draw_frame(0)\n',
    'n_frames on an unpacked name': 'fig, ani = hyp.plot(d)\nprint(ani.n_frames)\n',
    'index instead of unpack': 'res = hyp.plot(d)\nani = res[1]\nani.on_frame(cb)\n',
    'chained assignment': 'a = b = hyp.plot(d)\nfig, ani = b\nani.on_frame(cb)\n',
    'alias chain': 'a = hyp.plot(d)\nb = a\nfig, ani = b\nani.on_frame(cb)\n',
    '.animation property then on_frame':
        'ani = hyp.plot(d).animation\nani.on_frame(cb)\n',
    'walrus inside the tuple target':
        'fig, ani = (anim := hyp.plot(d))\nani.on_frame(cb)\n',
    'walrus in an if test':
        'if (anim := hyp.plot(d)):\n    fig, ani = anim\n    ani.on_frame(cb)\n',
    'annotated assignment':
        'anim = hyp.plot(d)\nani: object = anim[1]\nani.on_frame(cb)\n',
    'for-loop unpack': 'for fig, ani in [hyp.plot(d)]:\n    ani.on_frame(cb)\n',
    'use appears BEFORE the unpack':
        'def go():\n    ani.on_frame(cb)\nfig, ani = hyp.plot(d)\n',
    'alias OF an unpacked name':
        'fig, ani = hyp.plot(d)\nb = ani\nb.on_frame(cb)\n',
    'alias of a subscript result':
        'anim = hyp.plot(d)\nani = anim[1]\nb = ani\nb.on_frame(cb)\n',
    'alias of .animation':
        'ani = hyp.plot(d).animation\nb = ani\nb.on_frame(cb)\n',
    'five-hop alias chain written bottom-up':
        'e.on_frame(cb)\ne = d_\nd_ = c\nc = b\nb = ani\n'
        'fig, ani = hyp.plot(dat)\n',
}
GUARD_MUST_IGNORE = {
    'the blessed idiom':
        'anim = hyp.plot(d)\nfig, ani = anim\nanim.on_frame(cb)\n',
    'wrapper used without unpacking': 'anim = hyp.plot(d)\nanim.draw_frame(0)\n',
    'FuncAnimation.save on an unpacked name (legal)':
        "fig, ani = hyp.plot(d)\nani.save('x.gif')\n",
    # element 0 of the unpack is a genuine Figure, and Figure.figure is
    # real public matplotlib API -- see note_unpack
    'Figure.figure on the FIGURE half of an unpack':
        'fig, ani = hyp.plot(d)\nprint(fig.figure)\n',
    'savefig on the figure half': "fig, ani = hyp.plot(d)\nfig.savefig('x.png')\n",
}
#: matplotlib's `Line2D.figure` and pandas' `.plot` are real public API.
GUARD_MUST_IGNORE_FOREIGN = {
    'matplotlib ax.plot then .figure':
        'import matplotlib.pyplot as plt\nln, = ax.plot(x, y)\nprint(ln.figure)\n',
    'pandas df.plot then .figure':
        'import pandas as pd\nfig, axx = df.plot(subplots=True)\nprint(axx.figure)\n',
}
GUARD_KNOWN_UNCAUGHT = {
    'dynamic getattr dispatch':
        "fig, ani = getattr(hyp, 'plot')(d)\nani.on_frame(cb)\n",
    'stored in and read back from a dict':
        "d2 = {'a': hyp.plot(d)}\nfig, ani = d2['a']\nani.on_frame(cb)\n",
    'stored in and read back from a list':
        'L = [hyp.plot(d)]\nfig, ani = L[0]\nani.on_frame(cb)\n',
    'starred remainder': 'fig, *rest = hyp.plot(d)\nrest[0].on_frame(cb)\n',
    'with-as unpack (cannot run: not a context manager)':
        'with hyp.plot(d) as (fig, ani):\n    ani.on_frame(cb)\n',
    'passed into a function and used there (interprocedural)':
        'def use(a):\n    a.on_frame(cb)\nfig, ani = hyp.plot(d)\nuse(ani)\n',
    'nested tuple target':
        '(fig, ani), z = hyp.plot(d), 1\nani.on_frame(cb)\n',
    'for over a NAME bound to a list, not a literal':
        'R = [hyp.plot(d)]\nfor fig, ani in R:\n    ani.on_frame(cb)\n',
}


def test_the_contract_8_guard_actually_detects():
    """A detector that has never detected anything is indistinguishable
    from one that cannot.

    `test_no_example_or_notebook_unpacks_then_uses_the_wrapper` passes on
    all ten shipped files, so on its own it proves nothing about the
    guard. These constructed sources do: each MUST-FLAG case is the bug,
    each MUST-IGNORE case is correct code the guard must not punish, and
    the KNOWN-UNCAUGHT set pins the documented limits so that
    strengthening the guard forces its docstring to be updated in the same
    commit.
    """
    hyp_import = 'import hypertools as hyp\n'
    for note, body in GUARD_MUST_FLAG.items():
        assert _unpacked_wrapper_uses(hyp_import + body), (
            f'guard missed a real Contract 8 violation ({note}):\n{body}')
    # the alias spelling has to bring its own import
    assert _unpacked_wrapper_uses(
        'import hypertools as ht\nfig, ani = ht.plot(d)\nani.on_frame(cb)\n')
    assert _unpacked_wrapper_uses(
        'from hypertools import plot as p\nfig, ani = p(d)\nani.on_frame(cb)\n')

    for note, body in GUARD_MUST_IGNORE.items():
        hits = _unpacked_wrapper_uses(hyp_import + body)
        assert not hits, f'guard false-positived on {note}: {hits}'
    for note, source in GUARD_MUST_IGNORE_FOREIGN.items():
        hits = _unpacked_wrapper_uses(source)
        assert not hits, (
            f'guard flagged non-hypertools API ({note}): {hits}. '
            f'`Line2D.figure` and `DataFrame.plot` are public; a guard that '
            f'reports them gives a factually wrong reason and cannot be '
            f'relaxed without weakening the real check.')
    # a `hyp` call in one function must not launder an `ax.plot` unpack in
    # another
    assert not _unpacked_wrapper_uses(
        hyp_import + 'def a():\n    fig, ani = ax.plot(x)\n'
                     'def b():\n    anim = hyp.plot(d)\n    anim.on_frame(cb)\n')

    for note, body in GUARD_KNOWN_UNCAUGHT.items():
        assert not _unpacked_wrapper_uses(hyp_import + body), (
            f'the guard now catches {note!r}, which its docstring lists as '
            f'a known limitation. Good -- update the docstring and move this '
            f'case into GUARD_MUST_FLAG.')


@pytest.mark.parametrize('path,_max', BUDGETS)
def test_no_example_or_notebook_unpacks_then_uses_the_wrapper(path, _max):
    """Contract 8. `fig, ani = hyp.plot(...)` binds `ani` to the raw
    FuncAnimation, so every wrapper member raises AttributeError -- while
    `_save_count` SURVIVES the unpack, so a gate written against that
    attribute passes as the public API is discarded.

    PASSES on all ten files today: it is a CONTROL against regression, not
    coverage of a present defect. It would have caught v2's prescribed
    conversation notebook, which unpacked and then called `.on_frame()`.
    """
    try:
        hits = _unpacked_wrapper_uses(_parsable_code(path))
    except SyntaxError as exc:
        pytest.fail(
            f'{path}: could not be parsed ({exc}), so this guard would be '
            f'silently vacuous on it. `_parsable_code` comments out magics '
            f'that start a line; a cell magic (%%bash), an INDENTED magic '
            f'inside a block, or a `hyp.plot?` help suffix still defeats it. '
            f'Fix the notebook or extend `_parsable_code` -- do not let the '
            f'file through unchecked.')
    assert not hits, (
        f'{path}: ' + '; '.join(
            f'`{name}` comes from unpacking a hyp.plot() result, so it is a '
            f'raw FuncAnimation and has no .{attr}' for name, attr in hits)
        + '. Bind the HyperAnimation first (`anim = hyp.plot(...)`), then '
          '`fig, ani = anim` if the parts are wanted.')


#: Measured 2026-08-02 against the committed notebooks, so a reader can
#: tell coverage from controls at a glance. RED today (Task 7 turns them
#: green): conversation_trajectories, hugging_face_embeddings and
#: wikipedia_embeddings fail BOTH assertions; modern_sklearn_dynamics fails
#: on ffmpeg only. ALREADY GREEN, and therefore CONTROLS rather than
#: coverage: stock_forecasting and projectile_kalman -- they are here to
#: prove Task 7 does not REGRESS a clean notebook, not to prove it fixed
#: one. Do not read six passing IDs as six notebooks repaired.
@pytest.mark.parametrize('nb', [
    'conversation_trajectories', 'hugging_face_embeddings',
    'wikipedia_embeddings', 'modern_sklearn_dynamics',
    'stock_forecasting',        # control -- already clean
    'projectile_kalman',        # control -- already clean
])
def test_older_tutorials_dropped_their_hand_rolled_helpers(nb):
    text = _code_text(f'docs/tutorials/{nb}.ipynb')
    assert 'SentenceTransformer' not in text
    assert 'ffmpeg' not in text


def test_analyze_tutorial_actually_plots():
    """A pipeline tutorial that never calls hyp.plot never shows why the
    pipeline exists (audit: analyze.ipynb, 20.0% hypertools, 0 hyp.plot)."""
    assert 'hyp.plot' in _code_text('docs/tutorials/analyze.ipynb')


def test_reduce_tutorial_mentions_describe():
    assert 'hyp.describe' in _code_text('docs/tutorials/reduce.ipynb')


#: The artifact each example exists to produce. These are the SEMANTIC
#: gates that replaced the native-ratio floor: unlike a line-count ratio,
#: none of them can be satisfied by reformatting, and each fails loudly if
#: the rewrite drops the thing the example is for.
STATED_ARTIFACT = {
    # min_frames is a real floor per example (frame_rate x duration), not
    # `>= 1`, which every animation satisfies by construction.
    'animate_market_forecast': dict(min_frames=100, predicts=True),
    'animate_weather_decades': dict(min_frames=100, axes=2),
    'animate_painting_embeddings': dict(min_frames=60, palette=True),
    'animate_conversation': dict(min_frames=100, on_frame=True),
    # 5 shapes, plus `clouds.append(clouds[0])` to close the loop = 6
    # clouds -> 2*6 - 1 = 11 segments, matching the example's own
    # 11-entry `rotations` list. (Measured; NOT 10 -- the schedule has no
    # implicit closing transition, and the example's inline comment
    # comment used to say "for the 5 clouds = 9 segments", counting the
    # shapes rather than what the call receives; corrected in the file.)
    'animate_morph_zoo': dict(min_frames=200, morph=11),
}


def _import_example_without_fetching(stem):
    """Import an example as a module, and prove the import fetched nothing.

    **This depends on Step 0b having been done, and fails loudly if it has
    not.** Measured 2026-08-02: NO example currently has a
    `if __name__ == '__main__':` guard (`grep -c __main__ examples/animate_*.py`
    -> 0 for all ten), so today every loader runs at module scope --
    `animate_morph_zoo.py:74` and `animate_market_forecast.py:113` fetch
    during import. `runpy.run_path` (v2) had the same problem for the same
    reason.

    Step 0b is what makes the premise true: it moves every loader call
    behind the `__main__` guard so the module body only DEFINES things, and
    it makes each fetcher honour `HYPERTOOLS_OFFLINE` by raising instead of
    silently substituting. Until then this helper is not merely ineffective
    -- it would download Dropbox shape files, FRED CSVs and HuggingFace
    models inside the default suite.

    The guard below turns that from a silent regression into a failure that
    names the file.
    """
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    source = _read(f'examples/{stem}.py')
    # Refuse to import an example that has not been split yet, rather than
    # letting it fetch. Checked BEFORE exec, because after exec the damage
    # is done.
    # Match the guard STRUCTURALLY, not as a literal string: `__name__ ==
    # "__main__"` with double quotes is the same guard, and a literal
    # substring test would reject it while blaming the missing split -- a
    # wrong reason for a real-looking failure.
    _guarded = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == '__name__'
        and any(isinstance(c, ast.Constant) and c.value == '__main__'
                for c in node.test.comparators)
        for node in ast.walk(ast.parse(source)))
    assert _guarded, (
        f'examples/{stem}.py has no __main__ guard, so importing it would '
        f'run its loaders and hit the network (Step 0b). Do the loader / '
        f'construct_artifact split before enabling this gate.')
    os.environ['HYPERTOOLS_OFFLINE'] = '1'
    try:
        spec = importlib.util.spec_from_file_location(stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for required in ('construct_artifact', 'fixture_data'):
            assert hasattr(module, required), (
                f'examples/{stem}.py does not define {required}() (Step 0b)')
        return module
    finally:
        # NOTE: the caller runs `construct_artifact(fixture_data())` AFTER
        # this returns, so leaving the variable set only for the import
        # would guard the cheap half and not the expensive one. The caller
        # re-arms it around the construction; see `_offline`.
        os.environ.pop('HYPERTOOLS_OFFLINE', None)


@contextlib.contextmanager
def _offline():
    """Assert no fetch happens inside the block, not merely at import.

    Belt AND braces: `_import_example_without_fetching` covers module
    import; this covers `fixture_data()` + `construct_artifact()`, which is
    where the loaders would actually be called from if the split were done
    wrong.
    """
    os.environ['HYPERTOOLS_OFFLINE'] = '1'
    try:
        yield
    finally:
        os.environ.pop('HYPERTOOLS_OFFLINE', None)


def _drive(anim, frame):
    """Render one frame, by index, through the public animation object.

    `HyperAnimation.draw_frame(i)` (Task 8 Step 0) is the supported way to
    do this. v2 reached for matplotlib's private `ani._func(i, *ani._args)`,
    which Contract 3 forbids and `DEFECT_MARKERS` lists.
    """
    anim.draw_frame(frame)


@pytest.mark.parametrize('stem', sorted(STATED_ARTIFACT))
def test_examples_produce_their_stated_artifact(stem):
    """Executable semantics, not source-shape -- driven by a FIXTURE.

    v2 ran each example with `runpy`. Measured: **all five are
    network-coupled** (weather 6 blocked connections, paintings 7, morph 4,
    conversation 2, market 1), so that version put model downloads and
    remote fetches in the default suite, contradicting Contract 4 and making
    CI nondeterministic. Morph does not even degrade -- `hyp.load()` has no
    offline path and takes the example down with `HypertoolsIOError`.

    So each example splits its loader from its figure builder (Contract 4),
    and this drives `construct_artifact(data)` with the example's own seeded
    synthetic data. Four of the five need ZERO committed fixture bytes,
    because their existing offline fallbacks already ARE deterministic
    fixtures; paintings ships one 1.7 KB thumbnail. Importing an example
    must not fetch. The whole-example run survives as an opt-in smoke test
    (`HYPERTOOLS_EXAMPLE_SMOKE=1`), never in the default suite.

    Every assertion below had to be rewritten, because v2's could not fail:

    * `_save_count >= 1` is a TAUTOLOGY -- hyp.plot always passes
      `max(1, round(frame_rate * duration))`, so it holds for a zero-length,
      zero-dataset animation. Measured at duration=0.01/frame_rate=1: 1.
      It is also a private matplotlib field, which Contract 3 forbids and
      `DEFECT_MARKERS` lists ten lines above the gate that used it.
    * `'morph' in str(ns.get('ANIMATE', 'morph'))` is a TAUTOLOGY -- no
      example binds `ANIMATE`, so the default makes it `'morph' in 'morph'`.
    * `ns['ani']` does not exist in weather or conversation, which bind
      `anim` (Contract 8), so 2 of 5 parametrisations failed on day one for
      a reason unrelated to what they gate.

    An unsatisfiable gate and a vacuous one are the same defect wearing
    opposite clothes: neither discriminates. v1 shipped the first, v2
    replaced it with the second.
    """
    module = _import_example_without_fetching(stem)
    want = STATED_ARTIFACT[stem]
    with _offline():
        anim = module.construct_artifact(module.fixture_data())
    fig = anim.figure

    assert anim.n_frames >= want['min_frames'], (
        f'{stem}: {anim.n_frames} frames, expected at least '
        f"{want['min_frames']}")
    if want.get('axes'):
        assert len(fig.axes) >= want['axes']
    if want.get('predicts'):
        _drive(anim, frame=anim.n_frames - 1)
        live = [a for a in fig.axes[0].lines
                if getattr(a, '_hyp_forecast_role', None) == 'live']
        assert live, 'no live forecast artist after driving a frame'
        pts = np.asarray(live[0].get_data_3d()
                         if hasattr(live[0], 'get_data_3d')
                         else live[0].get_data())
        assert pts.size and np.isfinite(pts).all(), (
            'the forecast artist exists but its geometry is empty or '
            'non-finite -- artists can be created and never filled')
    if want.get('on_frame'):
        _drive(anim, frame=0)
        first = [a.get_alpha() for a in fig.axes[0].lines]
        _drive(anim, frame=anim.n_frames - 1)
        last = [a.get_alpha() for a in fig.axes[0].lines]
        assert first != last, (
            'the per-frame hook never changed any alpha, so the recency '
            'fade is not actually running')
    if want.get('morph'):
        assert anim.n_segments == want['morph'], (
            f'{stem}: {anim.n_segments} morph segments, expected '
            f"{want['morph']}")
        _drive(anim, frame=anim.n_frames // 2)
        assert fig.axes[0].collections or fig.axes[0].lines, (
            'a driven mid-morph frame drew nothing')
    if want.get('palette'):
        # Paintings is the ONLY example whose whole point is Task 1's native
        # `palette='image:<path>'`, and the only one that costs a committed
        # fixture -- and for one revision this key was in STATED_ARTIFACT
        # with no branch reading it, so the example was gated by nothing but
        # its frame count. A key that no code reads is a claim with no gate
        # behind it.
        from hypertools.plot.colors import image_palette
        anchors = image_palette(module.PALETTE_FIXTURE, n_colors=6)
        drawn = {tuple(np.round(a.get_color()[:3], 3))
                 if not isinstance(a.get_color(), str) else a.get_color()
                 for a in fig.axes[0].lines}
        assert drawn, 'no line artists to take a colour from'
        # every drawn colour must be one of the image's extracted anchors
        # (or a blend of them) -- i.e. inside the anchors' convex range per
        # channel, with a small tolerance for blending
        arr = np.asarray([np.asarray(a.get_color()[:3], dtype=float)
                          for a in fig.axes[0].lines
                          if not isinstance(a.get_color(), str)])
        if arr.size:
            lo = np.asarray(anchors, dtype=float).min(0) - 0.02
            hi = np.asarray(anchors, dtype=float).max(0) + 0.02
            assert ((arr >= lo) & (arr <= hi)).all(), (
                f'{stem}: drawn colours fall outside the palette extracted '
                f'from {module.PALETTE_FIXTURE} -- the example is not '
                f'actually using the image palette it exists to demonstrate')


LAUNCH_NOTEBOOKS = ('market_forecast', 'weather_decades',
                    'painting_embeddings', 'conversation_shape',
                    'morph_shapes_zoo')


def _is_install_cell(source):
    """Detected by CONTENT, never by index.

    Two measurements forced this. First, the install cell is NOT uniformly
    unexecuted: 9 of the 20 notebooks in docs/tutorials/ ship it executed,
    the five launch notebooks do not -- so a gate asserting either polarity
    fails on half the repo, and it must simply be EXEMPT. Second, indexing
    by position breaks the moment a cell is inserted above it, or a notebook
    has no install cell at all.
    """
    return 'pip install' in source


def _code_cells(stem):
    import json
    nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
    return [c for c in nb['cells'] if c.get('cell_type') == 'code']


#: Per notebook, the INDEX SET of code cells that carry a visible output --
#: not a count. Recorded from a real nbclient run in each task's "Execute
#: and measure" step; do not write a number here before the notebook exists.
#:
#: Why an index set and not a total: v2 hardcoded five counts and ALL FIVE
#: were wrong, as was every per-task prediction in Tasks 2-6, because each
#: assumed every non-install cell emits when several are bare imports, bare
#: assignments, or `fig, ani = hyp.plot(..., show=False)`. Weather is the
#: instructive one -- its TOTAL happened to be right while naming entirely
#: the wrong cells. A count cannot tell those apart and is satisfied by a
#: stray print() landing anywhere; an index set fails immediately and names
#: the cell.
#:
#: Install-cell indices are filtered out of both sides before comparing.
EXPECTED_VISIBLE_OUTPUTS = {
    # MEASURED 2026-08-17 from a real `scripts/execute_tutorial.py` run,
    # and RE-MEASURED from a second one the same day that reported
    # `6/9 code cells produced output` (indices are into the CODE cells,
    # install cell included as 0 and filtered out on both sides below).
    # 3/5/6 are the three `print`s -- `471 weekly bars x 6 sectors x 3
    # measures`, `120 frames`, and the accuracy line -- 7 is the side-panel
    # cell, whose last expression is `fig` so the fully-revealed frame
    # renders inline as an `execute_result` carrying `image/png`, and 8 is
    # the `saved market_forecast.gif` line. 1/2/4 are pure definitions and
    # assignments and legitimately emit nothing.
    #
    # The 6th emitting cell in that count is the INSTALL cell (index 0),
    # which nbclient does run: `%pip install ... git+https://...` would
    # replace this repo's EDITABLE hypertools with a snapshot of the remote
    # branch, so the re-measurement ran with `PIP_DRY_RUN=1` and the cell
    # was restored to the unexecuted state every launch notebook ships it
    # in. `_is_install_cell` filters index 0 out of both sides here, so the
    # recorded set is unaffected either way.
    'market_forecast': {3, 5, 6, 7, 8},
}


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_every_launch_notebook_ran_every_cell_it_should(stem):
    """`nbsphinx_execute = 'never'` (docs/conf.py:131) renders the COMMITTED
    outputs, so a half-executed notebook is a figure-less docs page.

    Gates EXECUTION, which is a different property from OUTPUT: a cell can
    run perfectly and legitimately emit nothing. v1 allowed `len(code) - 2`
    unexecuted cells, which would pass a notebook whose only two code cells
    both failed; v2 demanded every code cell carry output, which no notebook
    can satisfy. This asserts what is actually required -- every cell ran --
    and leaves what each cell EMITS to the index-set test below.

    **CONTROL, not coverage.** It PASSES on all five notebooks TODAY:
    measured, every non-install code cell already carries an
    `execution_count` (market and weather `[None,2,3,4,5,6,7]`,
    paintings/conversation/morph `[None,2,3,4,5,6]`), and
    `_is_install_cell` exempts the one `None`. The defect this plan exists
    to fix is missing OUTPUTS, not missing execution -- so this test cannot
    detect it, and two places in this document previously claimed it could.
    It is here to stop a FUTURE half-executed notebook from shipping, which
    is worth having; it is not the gate on the present defect. That is
    `test_the_right_cells_carry_visible_output` plus
    `test_each_notebook_ships_its_rendered_artifact`.
    """
    cells = _code_cells(stem)
    unrun = [i for i, c in enumerate(cells)
             if c.get('execution_count') is None
             and not _is_install_cell(''.join(c['source']))]
    assert not unrun, (
        f'{stem}.ipynb: code cells {unrun} were never executed; re-run '
        f'scripts/execute_tutorial.py')


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_the_right_cells_carry_visible_output(stem):
    """Which cells emit, not how many."""
    if stem not in EXPECTED_VISIBLE_OUTPUTS:
        pytest.fail(
            f'{stem}: no measured index set recorded. Execute the notebook '
            f'and paste the measured set into EXPECTED_VISIBLE_OUTPUTS -- '
            f'do not guess it ahead of the artifact (v2 guessed five and got '
            f'all five wrong)')
    cells = _code_cells(stem)
    installs = {i for i, c in enumerate(cells)
                if _is_install_cell(''.join(c['source']))}
    got = {i for i, c in enumerate(cells) if c.get('outputs')} - installs
    want = set(EXPECTED_VISIBLE_OUTPUTS[stem]) - installs
    assert got == want, (
        f'{stem}.ipynb: cells {sorted(got)} carry output, expected '
        f'{sorted(want)} (missing {sorted(want - got)}, unexpected '
        f'{sorted(got - want)})')


@pytest.mark.parametrize('stem', LAUNCH_NOTEBOOKS)
def test_each_notebook_ships_its_rendered_artifact(stem):
    """The artifact assertion, keyed to how these notebooks ACTUALLY ship.

    Measured: there is no `image/png` and no `text/html` output anywhere in
    any of the five -- the display_data entries are tqdm progress widgets
    from sentence_transformers. The convention (commit 9b94d86f), shared
    with conversation_trajectories/streaming_data/wikipedia_embeddings, is a
    companion GIF written by the last code cell and embedded from a MARKDOWN
    cell. So "did a figure render" is not answerable from cell outputs, and
    a rule like "a cell calling hyp.plot must emit something" is satisfied
    by an unrelated print() in the same cell.

    This asserts the artifact that actually exists, and that its reference
    resolves.

    **It PASSES today, on all five -- it is a CONTROL, not coverage.**
    Measured 2026-08-02: every reference resolves, `morph_zoo.gif` included
    (4.5 MB, present). An earlier draft of this plan claimed the test
    "catches morph_shapes_zoo.ipynb embedding morph_zoo.gif"; it does not,
    because that file exists -- the stem mismatch is a naming
    inconsistency, not a broken link. What this test does is stop a rewrite
    from DROPPING the GIF or breaking its reference, which is worth having
    and is why it stays. Do not read five green IDs as five things fixed.
    """
    import json
    import os
    import re as _re
    nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
    md = '\n'.join(''.join(c['source']) for c in nb['cells']
                   if c.get('cell_type') == 'markdown')
    refs = _re.findall(r'!\[[^\]]*\]\(([^)]+\.gif)\)', md)
    assert refs, f'{stem}.ipynb: no rendered artifact is embedded'
    for ref in refs:
        target = os.path.join(REPO, 'docs', 'tutorials', ref)
        assert os.path.exists(target), (
            f'{stem}.ipynb embeds {ref!r}, which does not exist')


@pytest.mark.skipif(not os.environ.get('HYPERTOOLS_EXAMPLE_SMOKE'),
                    reason='set HYPERTOOLS_EXAMPLE_SMOKE=1 to run the '
                           'examples end to end (network + model downloads)')
@pytest.mark.parametrize('stem', sorted(STATED_ARTIFACT))
def test_example_runs_end_to_end(stem):
    """The whole-example run, OPT-IN.

    v2 ran every example in the default suite via `runpy`, which put model
    downloads and remote fetches on every CI run. v3 moved the default gate
    onto `construct_artifact(fixture_data())`, and this is what replaces the
    coverage that removed -- the loaders, the `__main__` guard, and the real
    data path, exercised on demand rather than never.

    Enable with `HYPERTOOLS_EXAMPLE_SMOKE=1 pytest -k end_to_end`. Run it
    before a release and whenever a loader changes; a failure here means the
    example is broken for a user even though the fixture-driven gate is
    green.
    """
    import subprocess
    import sys as _sys
    path = os.path.join(REPO, 'examples', f'{stem}.py')
    env = dict(os.environ, MPLBACKEND='Agg')
    env.pop('HYPERTOOLS_OFFLINE', None)
    proc = subprocess.run([_sys.executable, path], env=env, cwd=REPO,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f'examples/{stem}.py exited {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout[-2000:]}\n'
        f'--- stderr ---\n{proc.stderr[-2000:]}')


def test_no_launch_notebook_committed_an_error_output():
    """A notebook can be fully executed and still be broken."""
    import json
    for stem in ('market_forecast', 'weather_decades', 'painting_embeddings',
                 'conversation_shape', 'morph_shapes_zoo'):
        nb = json.loads(_read(f'docs/tutorials/{stem}.ipynb'))
        for cell in nb['cells']:
            for out in cell.get('outputs', []):
                assert out.get('output_type') != 'error', (
                    f"{stem}.ipynb: committed a traceback "
                    f"({out.get('ename')})")
