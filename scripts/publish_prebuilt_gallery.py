#!/usr/bin/env python
"""Publish the BUILT example gallery (``docs/auto_examples``) to the
``docs-gallery-v<version>`` branch, for Read the Docs to reuse.

Read the Docs ends a build after 15 minutes and executing the gallery takes
about 25, so its build fetches this tree instead
(``docs/fetch_prebuilt_gallery.py``, run from ``.readthedocs.yaml``) and
sphinx-gallery skips every example whose source is unchanged.

The branch is one orphan commit holding ``auto_examples/`` and a
``manifest.json`` (version, source commit, the md5 of every example's
source). Publishing again REPLACES that commit (a forced push): the tree is
about 100 MB of images and video, and keeping every republish would grow the
repository by that much each time. Nothing on the main branch changes.

It refuses to publish a gallery that is not a complete build of this
checkout: the tracked tree must be clean, every example must have a current
md5 in the gallery, and no page may contain this machine's path to the
checkout.

Run MANUALLY as a release step, after ``make html`` from the release commit
(see RELEASE_CHECKLIST.md):

    python scripts/publish_prebuilt_gallery.py [--gallery-dir docs/auto_examples] [--push]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fetch_module():
    path = os.path.join(_REPO, 'docs', 'fetch_prebuilt_gallery.py')
    spec = importlib.util.spec_from_file_location('fetch_prebuilt_gallery',
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fpg = _load_fetch_module()

# files a reader's browser or sphinx reads as text; scanned for local paths
_TEXT_SUFFIXES = ('.rst', '.py', '.ipynb', '.json', '.txt', '.html', '.md5')


def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, capture_output=True,
                          text=True)


def head_commit(repo_root):
    out = _git(['rev-parse', 'HEAD'], repo_root)
    return out.stdout.strip() if out.returncode == 0 else None


def tracked_tree_is_clean(repo_root):
    out = _git(['status', '--porcelain', '--untracked-files=no'], repo_root)
    return out.returncode == 0 and not out.stdout.strip()


def files_naming(gallery_dir, needle):
    """Text files under ``gallery_dir`` that contain ``needle``."""
    hits = []
    for base, _dirs, names in os.walk(gallery_dir):
        for name in names:
            if not name.endswith(_TEXT_SUFFIXES):
                continue
            path = os.path.join(base, name)
            with open(path, encoding='utf-8', errors='replace') as f:
                if needle in f.read():
                    hits.append(os.path.relpath(path, gallery_dir))
    return sorted(hits)


def manifest_content(repo_root, version, source_commit):
    """The published ``manifest.json``: what the release gate checks."""
    return {
        'version': version,
        'branch': fpg.branch_for(version),
        'source_commit': source_commit,
        'examples': fpg.example_md5s(os.path.join(repo_root, 'examples')),
    }


def problems(repo_root, gallery_dir):
    """Reasons this gallery must not be published (empty when it may be)."""
    found = []
    if not os.path.isdir(gallery_dir) or not os.listdir(gallery_dir):
        return [f'no built gallery in {gallery_dir}; run `make html` in docs/']
    if head_commit(repo_root) is None:
        found.append(f'{repo_root} is not a git checkout')
    elif not tracked_tree_is_clean(repo_root):
        found.append('the tracked tree has uncommitted changes, so the '
                     'gallery cannot be tied to a commit')
    stale = fpg.stale_examples(os.path.join(repo_root, 'examples'),
                               gallery_dir)
    if stale:
        found.append('the gallery is not a complete build of these examples '
                     '(no current md5 for: ' + ', '.join(stale) + ')')
    leaked = files_naming(gallery_dir, os.path.abspath(repo_root))
    if leaked:
        found.append("pages contain this machine's path to the checkout: "
                     + ', '.join(leaked))
    return found


def publish(repo_root=_REPO, gallery_dir=None, push=False,
            remote=fpg.REPO_URL):
    gallery_dir = gallery_dir or os.path.join(repo_root, 'docs',
                                              fpg.GALLERY_DIRNAME)
    found = problems(repo_root, gallery_dir)
    if found:
        for reason in found:
            print(f'refusing to publish: {reason}', file=sys.stderr)
        return 1
    version = fpg.project_version(repo_root)
    branch = fpg.branch_for(version)
    manifest = manifest_content(repo_root, version, head_commit(repo_root))
    work = tempfile.mkdtemp(prefix='docs-gallery-')
    try:
        shutil.copytree(gallery_dir, os.path.join(work, fpg.GALLERY_DIRNAME),
                        ignore=shutil.ignore_patterns('__pycache__',
                                                      '.DS_Store'))
        with open(os.path.join(work, fpg.MANIFEST_NAME), 'w',
                  encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write('\n')
        n_files = sum(len(names) for _b, _d, names in os.walk(work))
        steps = [
            ['init', '--quiet', '--initial-branch', branch],
            ['add', '-A'],
            ['commit', '--quiet', '-m', f'docs: pre-built gallery for v{version} '
                   f'(from {manifest["source_commit"]})'],
        ]
        if push:
            steps.append(['push', '--quiet', '--force', remote,
                          f'HEAD:refs/heads/{branch}'])
        for step in steps:
            done = _git(step, work)
            if done.returncode != 0:
                print(f'git {step[0]} failed: {done.stderr.strip()}',
                      file=sys.stderr)
                return 1
        if push:
            print(f'pushed {n_files} files to {branch} '
                  f'(from {manifest["source_commit"]})')
        else:
            # the commit lives in a throwaway directory deleted below
            print(f'validated {n_files} files for {branch} locally; no push '
                  'performed (pass --push to publish)')
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--gallery-dir', default=None)
    ap.add_argument('--remote', default=fpg.REPO_URL)
    ap.add_argument('--push', action='store_true')
    args = ap.parse_args(argv)
    return publish(gallery_dir=args.gallery_dir, push=args.push,
                   remote=args.remote)


if __name__ == '__main__':
    sys.exit(main())
