"""Execute a tutorial notebook in place with THIS repo's venv.

The user-level `python3` kernelspec points at an unrelated project's venv
(verified 2026-07-28 in `~/Library/Jupyter/kernels/python3/kernel.json`),
so `nbconvert --execute` with the default kernel does not run hypertools at
all. Register this repo's kernel once:

    .venv/bin/python -m ipykernel install --user --name hypertools-venv \
        --display-name "hypertools (.venv)"

then:

    .venv/bin/python scripts/execute_tutorial.py docs/tutorials/<name>.ipynb

Outputs are written back into the notebook (`nbsphinx_execute = 'never'`,
docs/conf.py, means the committed outputs are what the docs render), and
`metadata.kernelspec` is restored to the neutral python3 entry the committed
notebooks carry, so Colab is unaffected.

``--out-dir DIR`` writes the executed copy into `DIR` instead, leaving the
tracked notebook untouched. That exists so a smoke test of this script does
not have to be undone afterwards: `git checkout -- <notebook>` discards a
file change and cannot tell an unwanted execution from a wanted edit made in
the same window. Execution still resolves relative paths against the
notebook's ORIGINAL directory, so a redirected run reads the same data.

**The Colab install cell is skipped, and this is not optional.** Every launch
notebook opens with ``%pip install "hypertools[...] @ git+...@dev-1.0"`` for
Colab. Executed locally, that cell installs the REMOTE branch over this venv's
editable checkout, mid-run, so every later cell runs against whatever was
last pushed rather than the code being documented. Measured 2026-09-03: the
market notebook failed in its own kernel with "48 dimensions ... static plots
support at most 2" -- the column-MultiIndex support that lands in 1.1 was
gone -- and ``pip show hypertools`` afterwards reported the git install, not
the editable one. The committed notebooks carry execution timestamps on that
cell from 2026-07-30, when local and remote happened to agree, which is why
nothing noticed. So cells whose source contains ``pip install`` are tagged
``skip-execution`` in memory for the run (nbclient honours that tag), and the
tag is stripped before writing, so the committed cell is byte-identical.
The example gate already exempts install cells from having executed.
"""

import json
import os
import sys

import nbformat
from nbclient import NotebookClient

NEUTRAL_KERNELSPEC = {'display_name': 'Python 3', 'language': 'python',
                      'name': 'python3'}
KERNEL = 'hypertools-venv'
# The kernel inherits this process's environment. Model loading (transformers
# via sentence-transformers) emits tqdm progress bars as ipywidgets, which
# would be committed as `widget-view` outputs with no saved state and render
# in the docs as a stuck "Loading weights: 0%" line. Measured 2026-09-03 on
# painting_embeddings.ipynb. huggingface_hub and transformers both honour
# this variable.
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
SKIP_TAG = 'skip-execution'         # what nbclient honours
TIMEOUT = 1800


def execute(path, out=None):
    """Execute `path`, writing the result to `out` (default: in place)."""
    nb = nbformat.read(path, as_version=4)
    original = json.loads(json.dumps(nb.metadata.get('kernelspec',
                                                     NEUTRAL_KERNELSPEC)))
    installs = [c for c in nb.cells
                if c.cell_type == 'code' and 'pip install' in c.source]
    for cell in installs:
        cell.metadata.setdefault('tags', []).append(SKIP_TAG)
    # the notebook's OWN directory is the cwd it runs in, so its relative
    # data paths resolve -- `or '.'` because a bare filename has no dirname
    # (`'reduce.ipynb'.rsplit('/', 1)[0]` is the filename itself, which would
    # make the kernel's cwd a nonexistent directory)
    NotebookClient(nb, timeout=TIMEOUT, kernel_name=KERNEL,
                   resources={'metadata': {'path': os.path.dirname(path)
                                           or '.'}}).execute()
    nb.metadata['kernelspec'] = original
    for cell in installs:
        cell.metadata['tags'].remove(SKIP_TAG)
        if not cell.metadata['tags']:
            del cell.metadata['tags']
    nbformat.write(nb, out or path)
    executed = sum(1 for c in nb.cells
                   if c.cell_type == 'code' and c.get('outputs'))
    total = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f'{out or path}: {executed}/{total} code cells produced output')


if __name__ == '__main__':
    args = sys.argv[1:]
    out_dir = None
    if '--out-dir' in args:
        k = args.index('--out-dir')
        out_dir = args[k + 1]
        del args[k:k + 2]
    if not args:
        raise SystemExit(
            'usage: execute_tutorial.py [--out-dir DIR] <notebook> [...]')
    for target in args:
        execute(target, os.path.join(out_dir, os.path.basename(target))
                if out_dir else None)
