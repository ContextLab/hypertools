#!/usr/bin/env python
"""Put the release's PRE-BUILT example gallery into ``docs/auto_examples``
before sphinx runs, so a Read the Docs build does not execute the gallery.

Executing the 51 gallery examples takes about 25 minutes; Read the Docs ends a
build after 15. The release pipeline already builds the gallery from the
release commit (RELEASE_CHECKLIST.md), and
``scripts/publish_prebuilt_gallery.py`` publishes that tree to the
``docs-gallery-v<version>`` branch. This script fetches it. sphinx-gallery then
finds, for every example, a ``<name>.py.md5`` that matches the example's
source and skips executing it; an example whose source has changed since the
gallery was published has no matching md5 and is executed as usual.

Run from ``.readthedocs.yaml`` (``pre_build``). It never fails the build: with
no published gallery for this version it says so and sphinx executes
everything. ``--require`` turns "missing" and "stale" into a non-zero exit,
for checking a published gallery by hand.

    python docs/fetch_prebuilt_gallery.py [--require] [--remote URL]

Trust: the gallery branch is written by whoever can push to this repository,
the same people who can change ``docs/conf.py``. A fetched tree is still
checked before it is used (``unsafe_entries``): it may hold only regular
files, and its pages may only read files inside the gallery. A symlink, or a
page that includes a path outside ``auto_examples``, would make the docs
build copy a file from the build machine into the published site; such a
tree is rejected and sphinx executes the examples instead.

Standard library only: it runs before the docs requirements matter and is
imported by the publish script and the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_URL = 'https://github.com/ContextLab/hypertools.git'
BRANCH_PREFIX = 'docs-gallery-v'
GALLERY_DIRNAME = 'auto_examples'
MANIFEST_NAME = 'manifest.json'
_VERSION_RE = re.compile(r'^[0-9][0-9A-Za-z.+\-]*$')


def branch_for(version):
    """The branch that holds the pre-built gallery for ``version``."""
    if not _VERSION_RE.match(version or '') or '..' in version:
        raise ValueError(f'unsafe version: {version!r}')
    return f'{BRANCH_PREFIX}{version}'


def project_version(repo_root):
    """The release version, from ``pyproject.toml``."""
    with open(os.path.join(repo_root, 'pyproject.toml'),
              encoding='utf-8') as f:
        m = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']', f.read())
    if not m:
        raise ValueError('pyproject.toml has no version')
    return m.group(1)


def source_md5(path):
    """md5 of an example's source, computed the way sphinx-gallery computes
    the one it compares against ``<name>.py.md5`` (text mode, so the hash is
    the same on every platform's line endings)."""
    with open(path, 'rt', errors='surrogateescape', encoding='utf-8') as f:
        content = f.read().encode(errors='surrogateescape', encoding='utf-8')
    return hashlib.md5(content).hexdigest()


def example_md5s(examples_dir):
    """``{'plot_basic.py': md5, ...}`` for every example script."""
    return {name: source_md5(os.path.join(examples_dir, name))
            for name in sorted(os.listdir(examples_dir))
            if name.endswith('.py')}


def stale_examples(examples_dir, gallery_dir):
    """Examples sphinx-gallery would execute given ``gallery_dir``: those with
    no recorded md5 there, or one that differs from the example's source."""
    stale = []
    for name, md5 in example_md5s(examples_dir).items():
        recorded = os.path.join(gallery_dir, name + '.md5')
        try:
            with open(recorded, encoding='utf-8') as f:
                current = f.read().strip() == md5
        except OSError:
            current = False
        if not current:
            stale.append(name)
    return stale


# RST constructs that make sphinx read a file named in the page
_FILE_DIRECTIVE_RE = re.compile(
    r'^\s*\.\.\s+(?:include|literalinclude|image|image-sg|figure|video|'
    r'csv-table|raw|parsed-literal)::[ \t]*(\S*)', re.M)
_FILE_OPTION_RE = re.compile(r'^\s*:(?:file|srcset):[ \t]*(.+)$', re.M)
_DOWNLOAD_ROLE_RE = re.compile(r':download:`[^`<]*<([^`>]+)>`|:download:`([^`<>]+)`')


def _path_stays_in_gallery(path):
    path = path.strip()
    if not path or re.match(r'^[a-z][a-z0-9+.\-]*://', path, re.I):
        return True                               # nothing local is read
    if '..' in path.replace('\\', '/').split('/'):
        return False
    if path.startswith('/'):                      # sphinx: relative to srcdir
        return path.startswith(f'/{GALLERY_DIRNAME}/')
    return not os.path.isabs(path) and not re.match(r'^[A-Za-z]:', path)


def unsafe_entries(tree):
    """Reasons ``tree`` must not be used as a gallery: entries that are not
    regular files or directories (a symlink would be followed when copied or
    read), and pages that read a file outside the gallery."""
    found = []
    for base, dirs, names in os.walk(tree):
        for name in dirs + names:
            path = os.path.join(base, name)
            rel = os.path.relpath(path, tree)
            if os.path.islink(path) or not (os.path.isdir(path)
                                            or os.path.isfile(path)):
                found.append(f'{rel}: not a regular file or directory')
        for name in names:
            path = os.path.join(base, name)
            if not name.endswith(('.rst', '.txt')) or os.path.islink(path):
                continue
            with open(path, encoding='utf-8', errors='replace') as f:
                text = f.read()
            named = [m.group(1) for m in _FILE_DIRECTIVE_RE.finditer(text)]
            for m in _FILE_OPTION_RE.finditer(text):
                # srcset holds several "path [1.5x]" entries
                named += [part.split()[0] for part in m.group(1).split(',')
                          if part.split()]
            named += [m.group(1) or m.group(2)
                      for m in _DOWNLOAD_ROLE_RE.finditer(text)]
            for target in named:
                if not _path_stays_in_gallery(target):
                    found.append(f'{os.path.relpath(path, tree)}: reads '
                                 f'{target!r}, outside the gallery')
    return sorted(set(found))


def fetch(repo_root, remote=REPO_URL, version=None, out=print):
    """Copy the published gallery for ``version`` into
    ``<repo_root>/docs/auto_examples``.

    Returns ``(status, stale)``: status is ``'fetched'``, ``'missing'`` (no
    branch for this version), ``'rejected'`` (the published tree failed
    ``unsafe_entries``) or ``'present'`` (a gallery is already there and is
    left alone); ``stale`` lists the examples sphinx will still execute
    (``None`` when nothing was fetched).
    """
    version = version or project_version(repo_root)
    branch = branch_for(version)
    examples_dir = os.path.join(repo_root, 'examples')
    target = os.path.join(repo_root, 'docs', GALLERY_DIRNAME)
    if os.path.isdir(target) and os.listdir(target):
        out(f'pre-built gallery: {target} already exists; leaving it alone')
        return 'present', stale_examples(examples_dir, target)

    work = tempfile.mkdtemp(prefix='docs-gallery-')
    try:
        clone = subprocess.run(
            ['git', 'clone', '--quiet', '--depth', '1', '--branch', branch,
             remote, work], capture_output=True, text=True)
        source = os.path.join(work, GALLERY_DIRNAME)
        if clone.returncode != 0 or not os.path.isdir(source):
            out(f'pre-built gallery: none published for {version} (branch '
                f'{branch} of {remote}); sphinx will execute every example.\n'
                f'{clone.stderr.strip()}')
            return 'missing', None
        unsafe = unsafe_entries(source)
        if unsafe:
            out(f'pre-built gallery: REJECTED {branch}; sphinx will execute '
                'every example.\n  ' + '\n  '.join(unsafe))
            return 'rejected', None
        manifest_path = os.path.join(work, MANIFEST_NAME)
        commit = None
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                commit = json.load(f).get('source_commit')
        if os.path.isdir(target):
            os.rmdir(target)                      # empty, checked above
        shutil.copytree(source, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    stale = stale_examples(examples_dir, target)
    n = len(example_md5s(examples_dir))
    out(f'pre-built gallery: fetched {branch} (built from {commit}); '
        f'{n - len(stale)} of {n} examples are current')
    if stale:
        out('pre-built gallery: sphinx will execute the examples that changed '
            'since it was published: ' + ', '.join(stale))
    return 'fetched', stale


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo-root', default=os.path.dirname(here))
    ap.add_argument('--remote', default=REPO_URL)
    ap.add_argument('--require', action='store_true',
                    help='exit 1 unless a gallery was fetched and every '
                         'example in it is current')
    args = ap.parse_args(argv)
    status, stale = fetch(args.repo_root, remote=args.remote)
    if args.require and (status in ('missing', 'rejected') or stale):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
