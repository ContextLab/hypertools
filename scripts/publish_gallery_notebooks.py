#!/usr/bin/env python
"""Publish the GENERATED sphinx-gallery notebooks to the ``docs-notebooks``
branch so the docs' "Open in Colab" badges resolve.

The gallery notebooks (``docs/auto_examples/*.ipynb``) are gitignored build
products -- they are never committed to the main tree (the ``docs-clean`` CI job
fails if they are). But the Colab badge injected by ``docs/post_build.py`` opens
a notebook via ``github.com/.../blob/<ref>/...``, which requires the file to
exist on GitHub. This script publishes the built notebooks to a dedicated
``docs-notebooks`` orphan branch, laid out by docs ref::

    docs-notebooks
    ├── dev-1.0/auto_examples/plot_basic.ipynb
    ├── v1.0.0/auto_examples/plot_basic.ipynb
    └── ...

so post_build.py's ``blob/docs-notebooks/<ref>/auto_examples/<stem>.ipynb`` links
resolve. The `publish-gallery-notebooks` CI job runs this on master/tag builds;
run it manually with the same effect. Nothing on the main branch changes.

    python scripts/publish_gallery_notebooks.py --ref v1.0.0 \
        --notebooks-dir docs/auto_examples [--push]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_URL = 'https://github.com/ContextLab/hypertools.git'
PUBLISH_BRANCH = 'docs-notebooks'
_REF_RE = re.compile(r'^[\w.][\w./\-]*$')      # a git branch/tag identifier


def target_paths(notebook_paths, ref):
    """Map each source notebook to its path within the publish branch:
    ``<ref>/auto_examples/<basename>``. Pure/​unit-testable."""
    if not _REF_RE.match(ref or '') or '..' in ref:
        raise ValueError(f'unsafe ref: {ref!r}')
    prefix = f'{ref}/auto_examples'
    return {p: f'{prefix}/{os.path.basename(p)}'
            for p in notebook_paths if p.endswith('.ipynb')}


def _run(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def publish(notebooks_dir, ref, push=False, remote=REPO_URL):
    nbs = sorted(glob.glob(os.path.join(notebooks_dir, '*.ipynb')))
    if not nbs:
        print(f'no notebooks in {notebooks_dir}', file=sys.stderr)
        return 1
    layout = target_paths(nbs, ref)
    work = tempfile.mkdtemp(prefix='docs-notebooks-')
    try:
        # start from the existing publish branch if it exists, else a fresh
        # orphan branch (first publish / branch bootstrap)
        rc = subprocess.run(
            ['git', 'clone', '--depth', '1', '--branch', PUBLISH_BRANCH,
             remote, work], capture_output=True, text=True).returncode
        if rc != 0:
            _run(['git', 'clone', '--depth', '1', remote, work])
            _run(['git', 'checkout', '--orphan', PUBLISH_BRANCH], cwd=work)
            _run(['git', 'rm', '-rf', '--quiet', '.'], cwd=work)
        # replace this ref's notebooks wholesale so removed examples don't linger
        ref_dir = os.path.join(work, ref, 'auto_examples')
        shutil.rmtree(os.path.join(work, ref), ignore_errors=True)
        os.makedirs(ref_dir, exist_ok=True)
        for src, rel in layout.items():
            shutil.copy2(src, os.path.join(work, rel))
        _run(['git', 'add', '-A'], cwd=work)
        # nothing to commit -> already up to date
        if subprocess.run(['git', 'diff', '--cached', '--quiet'],
                          cwd=work).returncode == 0:
            print(f'docs-notebooks already up to date for {ref}')
            return 0
        _run(['git', 'commit', '-m',
              f'docs: publish {len(layout)} gallery notebooks for {ref}'],
             cwd=work)
        if push:
            _run(['git', 'push', 'origin', PUBLISH_BRANCH], cwd=work)
            print(f'pushed {len(layout)} notebooks to {PUBLISH_BRANCH}/{ref}/')
        else:
            print(f'staged {len(layout)} notebooks for {PUBLISH_BRANCH}/{ref}/ '
                  '(dry run; pass --push to publish)')
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ref', required=True,
                    help='docs ref (e.g. v1.0.0, dev-1.0, master)')
    ap.add_argument('--notebooks-dir', default='docs/auto_examples')
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--remote', default=REPO_URL)
    args = ap.parse_args(argv)
    return publish(args.notebooks_dir, args.ref, push=args.push,
                   remote=args.remote)


if __name__ == '__main__':
    raise SystemExit(main())
