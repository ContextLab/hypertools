"""Ensure every committed docs notebook (gallery + tutorials) starts with a
cell that installs hypertools, so it runs standalone when opened in Google
Colab.

The install line is branch-aware:

* on ``master`` it installs the RELEASED package
  (``%pip install -q "hypertools[interactive]"``);
* on any other branch it installs THAT branch from GitHub, so the dev-1.0
  preview notebooks install the matching dev build rather than the older
  PyPI release.

The script is idempotent AND self-correcting: a notebook that already has a
hypertools install cell is not skipped -- its install target is RE-TARGETED to
match the current branch (2026-07 release review: the old ``has_install`` guard
merely detected the words "pip install" and skipped, so a stale
``@dev-1.0-refactor`` target could never be migrated to ``@dev-1.0`` or, at
release, to the plain PyPI spec). Only the hypertools ``... @ git+...@<branch>``
spec is rewritten; the notebook's own extras (``[interactive]``,
``[interactive,predict]``, ``[interactive,lsl]``, ...) and any other install
lines (e.g. a tutorial's ``%pip install -q convokit``) are preserved verbatim.

Run after (re)generating notebooks, then commit:

    .venv/bin/python scripts/add_colab_install_cell.py

RELEASE NOTE: the PyPI form only resolves to 1.0 once 1.0 is published to
PyPI. Run this on ``master`` AFTER the PyPI upload (or the notebooks would
install the previous PyPI release), then commit the migrated notebooks. The
``notebook-install-gate`` CI job enforces that no ``git+``/``@dev`` install
survives on a release build.
"""

import glob
import json
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = (glob.glob(os.path.join(REPO, 'docs', 'auto_examples', '*.ipynb'))
             + glob.glob(os.path.join(REPO, 'docs', 'tutorials', '*.ipynb')))

MARKER = 'pip install'          # a code cell with any pip install line
INLINE = '%matplotlib inline'
_GIT_URL = 'git+https://github.com/ContextLab/hypertools.git'

# A hypertools GitHub-branch install spec, capturing the extras so they are
# preserved across a re-target: e.g. `hypertools[interactive,predict] @
# git+https://github.com/ContextLab/hypertools.git@dev-1.0-refactor`.
_BRANCH_SPEC_RE = re.compile(
    r'hypertools\[([^\]]*)\]\s*@\s*'
    + re.escape(_GIT_URL) + r'@[\w./\-]+')

# The `(<branch> preview)` token inside the "# Install hypertools (...)" note.
_PREVIEW_NOTE_RE = re.compile(r'\(([\w.\-]+) preview\)')


def current_branch():
    branch = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', '')
    if not branch:
        try:
            branch = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, cwd=REPO,
                timeout=10).stdout.strip()
        except Exception:
            branch = ''
    return branch or 'master'


def hyp_spec(extras, branch):
    """The canonical install spec for `hypertools[<extras>]` on `branch`."""
    if branch == 'master':
        return f'hypertools[{extras}]'
    return f'hypertools[{extras}] @ {_GIT_URL}@{branch}'


def install_lines(branch):
    if branch == 'master':
        pip = '%pip install -q "hypertools[interactive]"'
        note = '# Install hypertools (run this first on Colab)'
    else:
        pip = f'%pip install -q "{hyp_spec("interactive", branch)}"'
        note = (f'# Install hypertools ({branch} preview) -- run this first '
                'on Colab.\n# On release this becomes: '
                '%pip install hypertools')
    return note, pip


def retarget_text(text, branch):
    """Rewrite any hypertools branch-install spec in `text` to `branch`.

    Preserves the notebook's extras and every other line. Also updates the
    ``(<branch> preview)`` token in the install note so the comment stays
    consistent with the install line. Returns the (possibly unchanged) text.
    """
    new = _BRANCH_SPEC_RE.sub(
        lambda m: hyp_spec(m.group(1), branch), text)
    if branch != 'master':
        new = _PREVIEW_NOTE_RE.sub(f'({branch} preview)', new)
    return new


def has_install(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code' and \
                MARKER in ''.join(cell.get('source', [])):
            return True
    return False


def retarget_notebook(nb, branch):
    """Re-target every code cell's hypertools branch install. True if changed."""
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new = retarget_text(src, branch)
        if new != src:
            cell['source'] = new.splitlines(keepends=True)
            changed = True
    return changed


def new_code_cell(source_text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_text.splitlines(keepends=True),
    }


def main():
    branch = current_branch()
    note, pip = install_lines(branch)
    retargeted = added = 0
    for path in sorted(NOTEBOOKS):
        with open(path) as f:
            nb = json.load(f)
        if has_install(nb):
            # already has an install cell -> re-target it to this branch
            if retarget_notebook(nb, branch):
                with open(path, 'w') as f:
                    # ensure_ascii=False keeps literal UTF-8 (matching nbformat)
                    # so re-targeting doesn't churn every non-ASCII glyph into a
                    # \\uXXXX escape and bloat the diff.
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                    f.write('\n')
                retargeted += 1
            continue
        cells = nb.setdefault('cells', [])
        # if the notebook opens with a lone `%matplotlib inline` cell (the
        # sphinx-gallery gallery notebooks), fold the install line into it so
        # there is a single tidy setup cell; otherwise prepend a new cell
        first = cells[0] if cells else None
        if (first and first.get('cell_type') == 'code'
                and INLINE in ''.join(first.get('source', []))
                and len(''.join(first.get('source', [])).strip()) < 40):
            first['source'] = (f'{note}\n{pip}\n\n{INLINE}').splitlines(
                keepends=True)
        else:
            cells.insert(0, new_code_cell(f'{note}\n{pip}'))
        with open(path, 'w') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        added += 1
    print(f'branch: {branch}')
    print(f'install line: {pip}')
    print(f'added install cell to {added} notebook(s); '
          f're-targeted {retargeted}; '
          f'{len(NOTEBOOKS) - added - retargeted} already correct '
          f'(of {len(NOTEBOOKS)})')


if __name__ == '__main__':
    main()
