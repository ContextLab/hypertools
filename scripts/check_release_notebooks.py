#!/usr/bin/env python
"""Validate that PUBLISHED notebooks install the RELEASED PyPI package.

A notebook passes ONLY if it has at least one hypertools pip-install and every
hypertools install is the plain PyPI spec (``hypertools[...]``, no ``git+`` /
``@<branch>``) with no leftover preview note in its setup cell. This closes the
weaker "reject only git+/@branch" check, which passed notebooks that had no
install, a non-``hypertools[...]`` install, or a stale preview note next to a
PyPI-form command.

Used by the ``docs-clean`` CI job on the GENERATED gallery
(``docs/auto_examples/*.ipynb``, which is gitignored and rebuilt each run) at
release, and unit-tested in ``tests/test_release_notebook_check.py`` so the
validation logic is covered independently of the workflow.

    python scripts/check_release_notebooks.py [--min N] <dir-or-glob> ...
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys


def _code_cell_texts(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            yield ''.join(cell.get('source', []))


# A line that actually EXECUTES a package install. Anchored to the start of the
# (stripped) line at a shell/magic prefix or a bare tool token, so it matches
# `%pip`/`!pip`/`!pip3`/`pipx`/`pip<TAB>install`/`uv pip`/`conda`/... but NOT a
# comment (`# ... pip install`) or a documentation STRING (`print("pip install
# git+...")`), which lack that prefix. Keyed on the literal `pip install` before,
# it both MISSED `!pip3 install ...@branch` (a live dev install slipping through)
# and FLAGGED `print("pip install git+...")` (a false positive). Release review.
_INSTALL_LINE_RE = re.compile(
    r'^[%!]?\s*(?:pip[0-9]*|pipx|uv\s+pip|conda|mamba|python[0-9.]*\s+-m\s+pip)'
    r'\s+install\b', re.IGNORECASE)


def _hyp_install_lines(nb):
    """Every code-cell LINE that installs hypertools, paired with the full text
    of the cell it lives in (for the preview-note check). Working per-LINE (not
    per-cell) is what lets a SECOND malformed install alongside a valid one be
    rejected."""
    out = []
    for text in _code_cell_texts(nb):
        for line in text.splitlines():
            if 'hypertools' in line and _INSTALL_LINE_RE.match(line.lstrip()):
                out.append((line, text))
    return out


def classify_notebooks(paths):
    """Classify each notebook by basename into three problem buckets.

    Returns ``(missing, branch_installs, stale_notes)`` where a name appears in:

    * ``missing`` -- no hypertools pip-install at all, OR ANY hypertools install
      line that is not in the required ``hypertools[...]`` PyPI-extras form;
    * ``branch_installs`` -- ANY hypertools install line still using ``git+`` /
      ``hypertools.git@<branch>``;
    * ``stale_notes`` -- a hypertools install cell that still carries a
      ``preview`` / "On release this becomes" note.

    EVERY hypertools install must be valid: a notebook with a good
    ``hypertools[...]`` install AND a second malformed ``pip install hypertools``
    is rejected (into ``missing``). A notebook that passes appears in none of
    the three.
    """
    missing, branch_installs, stale_notes = [], [], []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            nb = json.load(f)
        lines = _hyp_install_lines(nb)
        name = os.path.basename(p)
        if not lines:
            missing.append(name)                       # no hypertools install
        elif any('hypertools[' not in ln for ln, _ in lines):
            missing.append(name)                       # a non-`hypertools[...]` install
        elif any('git+' in ln or 'hypertools.git@' in ln for ln, _ in lines):
            branch_installs.append(name)               # any branch install
        elif any('preview' in ct or 'On release this becomes' in ct
                 for _, ct in lines):
            stale_notes.append(name)
    return missing, branch_installs, stale_notes


def _collect(dirs_or_globs):
    paths = []
    for d in dirs_or_globs:
        if os.path.isdir(d):
            paths += sorted(glob.glob(os.path.join(d, '*.ipynb')))
        else:
            paths += sorted(glob.glob(d))
    return paths


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    min_count, targets = 1, []
    i = 0
    while i < len(argv):
        if argv[i] == '--min':
            min_count = int(argv[i + 1])
            i += 2
        else:
            targets.append(argv[i])
            i += 1
    if not targets:
        print('usage: check_release_notebooks.py [--min N] <dir-or-glob> ...',
              file=sys.stderr)
        return 2
    paths = _collect(targets)
    if len(paths) < min_count:
        print(f'FAIL: expected >= {min_count} notebooks, found {len(paths)}',
              file=sys.stderr)
        return 1
    missing, branch, stale = classify_notebooks(paths)
    if not (missing or branch or stale):
        print(f'release notebook check OK: {len(paths)} notebooks, '
              'all install the PyPI package')
        return 0
    if missing:
        print(f'FAIL no valid hypertools[...] install: {missing}', file=sys.stderr)
    if branch:
        print(f'FAIL git+/@branch install: {branch}', file=sys.stderr)
    if stale:
        print(f'FAIL leftover preview note: {stale}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
