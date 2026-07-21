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
import sys


def _code_cell_texts(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            yield ''.join(cell.get('source', []))


def classify_notebooks(paths):
    """Classify each notebook by basename into three problem buckets.

    Returns ``(missing, branch_installs, stale_notes)`` where a name appears in:

    * ``missing`` -- no hypertools pip-install at all, OR a hypertools install
      that is not in the required ``hypertools[...]`` PyPI-extras form;
    * ``branch_installs`` -- a hypertools install still using ``git+`` /
      ``hypertools.git@<branch>``;
    * ``stale_notes`` -- a hypertools install cell that still carries a
      ``preview`` / "On release this becomes" note.

    A notebook that passes appears in none of the three.
    """
    missing, branch_installs, stale_notes = [], [], []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            nb = json.load(f)
        hyp_cells = [t for t in _code_cell_texts(nb)
                     if 'pip install' in t and 'hypertools' in t]
        name = os.path.basename(p)
        if not hyp_cells or not any('hypertools[' in t for t in hyp_cells):
            missing.append(name)
        elif any('git+' in t or 'hypertools.git@' in t for t in hyp_cells):
            branch_installs.append(name)
        elif any('preview' in t or 'On release this becomes' in t
                 for t in hyp_cells):
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
