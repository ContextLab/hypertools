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

**HyperTools installation cells are skipped during local verification.**
Current tutorials use a version-aware PyPI installer; candidate verification
must retain the selected checkout. Only explicitly tagged HyperTools installers
(or legacy cells containing a live HyperTools pip command) are skipped.
Configuration and independent prerequisites remain executable. Successful
setup-only prerequisite cells tagged ``prerequisite-install`` have their pip
chatter cleared before saving; failures still abort execution. Mixed legacy
install/work cells must be split before verification; a comment mentioning pip
is not an installation command. For the feature tour, configuration and its
Colab-only installer are separate cells. Use --out-dir to preserve source files.

**The executing user's home directory is rewritten to ``~`` in the outputs.**
Warnings and tracebacks carry absolute paths (``/Users/<name>/hypertools/
hypertools/tools/format_data.py:495: UserWarning: ...``), so an executed
notebook committed as-is publishes whoever ran it. After execution, every
stream output, error traceback and ``text/plain`` result has
``os.path.expanduser('~')`` replaced by ``~`` (see `scrub_home`); nothing
else in an output is touched.
"""

import json
import os
import re
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


def scrub_home(nb, home=None):
    """Rewrite `home` (default: this user's home directory) to ``~`` in every
    text output of `nb`, in place: stream text, error tracebacks and
    ``evalue``, and ``text/plain`` display/execute-result data. Returns the
    number of outputs changed."""
    home = home or os.path.expanduser('~')
    changed = 0
    for cell in nb.cells:
        for output in cell.get('outputs', []):
            before = json.dumps(output, sort_keys=True)
            kind = output.get('output_type')
            if kind == 'stream':
                output['text'] = output['text'].replace(home, '~')
            elif kind == 'error':
                output['traceback'] = [line.replace(home, '~')
                                       for line in output['traceback']]
                output['evalue'] = output['evalue'].replace(home, '~')
            elif kind in ('display_data', 'execute_result'):
                text = output.get('data', {}).get('text/plain')
                if isinstance(text, str):
                    output['data']['text/plain'] = text.replace(home, '~')
            changed += json.dumps(output, sort_keys=True) != before
    return changed


def skip_install_cells(nb):
    """Tag HyperTools installation cells of `nb` skip-execution (in memory)
    and drop the outputs it carried; return those cells for
    `restore_install_cells`.

    nbclient leaves a skipped cell exactly as the file had it, outputs
    included: two 1.0.0 tutorials shipped a pip upgrade notice naming a
    local interpreter path, from a run that DID execute the install cell.
    A cell that did not run here has no output.
    """
    installs = [c for c in nb.cells if c.cell_type == 'code' and (
        'hypertools-install' in c.metadata.get('tags', []) or
        re.search(r'^\s*[%!]pip\s+install[^\n]*hypertools', c.source, re.M))]
    for cell in installs:
        cell.metadata.setdefault('tags', []).append(SKIP_TAG)
        cell.outputs = []
        cell.execution_count = None
    return installs


def restore_install_cells(installs):
    """Remove the in-memory skip tag `skip_install_cells` added."""
    for cell in installs:
        cell.metadata['tags'].remove(SKIP_TAG)
        if not cell.metadata['tags']:
            del cell.metadata['tags']


def clear_prerequisite_install_outputs(nb):
    """Execute prerequisites normally, then omit successful pip chatter from docs.

    Called only after NotebookClient succeeds, so installation failures still
    propagate with their traceback. These tagged cells contain setup only.
    """
    for cell in nb.cells:
        if 'prerequisite-install' in cell.metadata.get('tags', []):
            cell.outputs = []
            cell.execution_count = None


def execute(path, out=None):
    """Execute `path`, writing the result to `out` (default: in place)."""
    nb = nbformat.read(path, as_version=4)
    original = json.loads(json.dumps(nb.metadata.get('kernelspec',
                                                     NEUTRAL_KERNELSPEC)))
    installs = skip_install_cells(nb)
    # the notebook's OWN directory is the cwd it runs in, so its relative
    # data paths resolve -- `or '.'` because a bare filename has no dirname
    # (`'reduce.ipynb'.rsplit('/', 1)[0]` is the filename itself, which would
    # make the kernel's cwd a nonexistent directory)
    NotebookClient(nb, timeout=TIMEOUT, kernel_name=KERNEL,
                   resources={'metadata': {'path': os.path.dirname(path)
                                           or '.'}}).execute()
    nb.metadata['kernelspec'] = original
    scrub_home(nb)
    restore_install_cells(installs)
    clear_prerequisite_install_outputs(nb)
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
