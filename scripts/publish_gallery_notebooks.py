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
resolve. This is run MANUALLY as a release step (see RELEASE_CHECKLIST.md) --
there is no CI job for it yet; a ``contents: write`` ``publish-gallery-notebooks``
job could automate it once token/environment handling is decided. Nothing on the
main branch changes.

Alongside the notebooks it writes a ``<ref>/manifest.json`` recording the ref,
the source commit, and the full notebook inventory, so the release gate can
verify the complete published set in one request (not one probe per notebook).

    python scripts/publish_gallery_notebooks.py --ref v1.0.0 \
        --notebooks-dir docs/auto_examples [--push]
"""

from __future__ import annotations

import argparse
import glob
import json
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


def manifest_content(notebook_paths, ref, source_commit=None):
    """The ``<ref>/manifest.json`` contents: the ref, the source commit the
    notebooks were generated from (or ``None``), the count, and the sorted
    notebook stems. Written FROM the actual copied files, so ``count`` always
    equals ``len(notebooks)`` and cannot over-claim -- the release gate reads
    this one file to verify the whole published inventory (rather than probing
    each notebook). Pure/unit-testable."""
    stems = sorted(os.path.basename(p)[:-len('.ipynb')]
                   for p in notebook_paths if p.endswith('.ipynb'))
    return {
        'ref': ref,
        'source_commit': source_commit,
        'count': len(stems),
        'notebooks': stems,
    }


def _head_commit():
    """The hypertools repo commit these notebooks were generated from (best
    effort; ``None`` if git is unavailable)."""
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'],
                             capture_output=True, text=True).stdout.strip()
        return out or None
    except Exception:
        return None


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
            # clear the default branch's files from the orphan's index;
            # --ignore-unmatch keeps this a no-op (not an error) when the
            # default branch is empty, so first-bootstrap can't wedge here
            _run(['git', 'rm', '-rf', '--quiet', '--ignore-unmatch', '.'],
                 cwd=work)
        # replace this ref's notebooks wholesale so removed examples don't linger
        ref_dir = os.path.join(work, ref, 'auto_examples')
        shutil.rmtree(os.path.join(work, ref), ignore_errors=True)
        os.makedirs(ref_dir, exist_ok=True)
        for src, rel in layout.items():
            shutil.copy2(src, os.path.join(work, rel))
        # manifest sits one level up from auto_examples, at <ref>/manifest.json,
        # written from the files just copied so it can't over-claim the set
        manifest = manifest_content(nbs, ref, _head_commit())
        with open(os.path.join(work, ref, 'manifest.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')
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
            # the commit was made in a throwaway clone that is deleted below --
            # nothing leaves this machine; "validated locally" is the honest
            # description (not a "dry run" that "staged" anything remotely)
            print(f'validated {len(layout)} notebooks for {PUBLISH_BRANCH}/{ref}/ '
                  'locally; no push performed (pass --push to publish)')
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
