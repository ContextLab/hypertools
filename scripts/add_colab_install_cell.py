"""Ensure every committed docs notebook (gallery + tutorials) starts with a
cell that installs hypertools, so it runs standalone when opened in Google
Colab.

The install line is branch-aware: on `master` it installs the released
package (``%pip install hypertools[interactive]``); on any other branch it
installs THAT branch from GitHub, so the dev-2.0 preview notebooks install
the matching dev build rather than the older PyPI release.

Idempotent: notebooks that already have the install line are left alone.
Run after (re)generating notebooks, then commit:

    .venv/bin/python scripts/add_colab_install_cell.py
"""

import glob
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = (glob.glob(os.path.join(REPO, 'docs', 'auto_examples', '*.ipynb'))
             + glob.glob(os.path.join(REPO, 'docs', 'tutorials', '*.ipynb')))

MARKER = 'pip install'          # idempotency check (any pip install line)
INLINE = '%matplotlib inline'


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


def install_lines(branch):
    if branch == 'master':
        pip = '%pip install -q "hypertools[interactive]"'
        note = '# Install hypertools (run this first on Colab)'
    else:
        url = ('git+https://github.com/ContextLab/hypertools.git@' + branch)
        pip = f'%pip install -q "hypertools[interactive] @ {url}"'
        note = (f'# Install hypertools ({branch} preview) -- run this first '
                'on Colab.\n# On release this becomes: '
                '%pip install hypertools')
    return note, pip


def has_install(nb):
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code' and \
                MARKER in ''.join(cell.get('source', [])):
            return True
    return False


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
    changed = 0
    for path in sorted(NOTEBOOKS):
        with open(path) as f:
            nb = json.load(f)
        if has_install(nb):
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
            json.dump(nb, f, indent=1)
            f.write('\n')
        changed += 1
    print(f'branch: {branch}')
    print(f'install line: {pip}')
    print(f'updated {changed} / {len(NOTEBOOKS)} notebooks '
          f'({len(NOTEBOOKS) - changed} already had it)')


if __name__ == '__main__':
    main()
