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
classification, this metric gave 48/739 = 6.5% for the five launch scripts
on 2026-07-28 (48/723 = 6.6% today, after `d730a085`)
where the audit reported 6.0% -- i.e. the two agree.

    .venv/bin/python scripts/measure_native_ratio.py examples/animate_*.py
    .venv/bin/python scripts/measure_native_ratio.py docs/tutorials/*.ipynb
"""

import ast
import json
import re
import sys

HYP = re.compile(r'\bhyp\.|\bhypertools\b')


def _docstring_line_numbers(source):
    """1-based line numbers occupied by REAL docstrings.

    `ast` is what makes this correct, and a heuristic cannot be. A docstring
    is the FIRST statement of a module/class/function and is a bare string
    expression. A line-scanner that keys on "the stripped line starts with a
    triple quote" cannot tell that from the CLOSING quote of an ordinary
    multi-line string -- it flips into docstring mode there and silently
    drops everything after it.

    That is not hypothetical. The first version of this function did exactly
    that, and measured against the real repo it dropped 171 code lines from
    `tests/test_density.py`, 123 from `tests/test_backend_state_safety.py`
    and 121 from `tests/test_surface.py` -- 8 files in all. Every dropped
    line is invisible to BOTH the size budget and the defect-marker ban, so
    a private reach sitting after an ordinary multi-line string would have
    passed the gate. A scan that silently drops code is worse than no scan,
    because it reports green.
    """
    # IPython magics and shell escapes are not Python; comment them out so a
    # notebook cell still parses. Line numbering is preserved.
    prepared = '\n'.join(
        ('# ' + line) if line.lstrip()[:1] in ('%', '!') else line
        for line in source.split('\n'))
    try:
        tree = ast.parse(prepared)
    except SyntaxError:
        # Unparseable: KEEP EVERY LINE. A spurious marker hit fails loudly
        # and gets investigated; a silently dropped line hides a defect for
        # good. When in doubt, keep the line.
        return set()
    drop = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) \
                and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            drop.update(range(first.lineno,
                              (first.end_lineno or first.lineno) + 1))
    return drop


def strip_docstrings(lines):
    """Yield the CODE lines from an iterable of source lines.

    Drops blank lines, comment-only lines, and real docstrings. This is the
    ONE place that logic lives -- shared by both counters below AND by
    `tests/test_examples_are_native.py`'s `_code_text`, so none of the three
    can drift out of sync.

    Public (no leading underscore) precisely because the test module imports
    it. Before this was shared, `_code_lines_py` and `_code_lines_nb` carried
    two INDEPENDENT copies and the notebook one was never written, so
    identical source measured (code=3, native=2) as `.py` but
    (code=11, native=2) as `.ipynb`.
    """
    lines = list(lines)
    drop = _docstring_line_numbers('\n'.join(lines))
    for n, line in enumerate(lines, 1):
        if n in drop:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        yield line


def _code_lines_py(path):
    # `with`, not a bare `open(...).read()`: this module is imported by the
    # gate, which measures 30 files in one run, and CPython's refcount is
    # what would close those handles. Under `-W error` the leak is a
    # ResourceWarning that turns budget checks into errors, and on a
    # non-refcounting interpreter it is a real descriptor leak. (The plan
    # prescribes the bare form; deviating from it here, deliberately.)
    with open(path, encoding='utf-8') as fh:
        return list(strip_docstrings(fh.read().splitlines()))


def _code_lines_nb(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        cells = json.load(fh)['cells']
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        # Reset per cell: a bare docstring cannot span a cell boundary (each
        # cell is parsed and executed independently), so carrying in_doc /
        # delim across cells would be wrong, not merely unnecessary.
        out.extend(strip_docstrings(
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
