#!/usr/bin/env python
"""Put the release's PRE-BUILT example gallery into ``docs/auto_examples``
before sphinx runs, so a Read the Docs build does not execute the gallery.

Executing the 51 gallery examples takes about 25 minutes; Read the Docs ends a
build after 15. The release pipeline already builds the gallery from the
release commit (RELEASE_CHECKLIST.md), and
``scripts/publish_prebuilt_gallery.py`` publishes that tree as one commit on
the ``docs-gallery-v<version>`` branch and records that commit's id in the
tracked file ``docs/prebuilt_gallery.json``. This script fetches exactly that
commit. sphinx-gallery then finds, for every example, a ``<name>.py.md5`` that
matches the example's source and skips executing it; an example whose source
has changed since the gallery was published has no matching md5 and is
executed as usual.

Trust: the gallery is fetched BY COMMIT ID, taken from the checkout being
built. A commit id is a hash of the content, so the build uses the tree the
release recorded or nothing: overwriting the gallery branch cannot change
what a build of this checkout reads. The recorded commit must ALSO be the
one the ``docs-gallery-v<version>`` branch points at, so a record cannot
name some other commit the host happens to serve (one from a fork's pull
request, say); and its manifest must name the same version and source
commit as the record. The tree may hold only regular files (a symlink would
be followed when copied).

What this does not do is inspect the gallery's pages. Whoever can change the
record in a checkout can change ``docs/conf.py`` in it too, which runs as
code in the docs build; the record is trusted exactly as far as the checkout
it is read from.

Run from ``.readthedocs.yaml`` (``pre_build``). It never fails the build: with
no usable gallery it says why and sphinx executes everything. ``--require``
turns "missing", "rejected" and "stale" into a non-zero exit, for checking a
published gallery by hand.

    python docs/fetch_prebuilt_gallery.py [--require] [--remote URL]

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
PIN_RELPATH = os.path.join('docs', 'prebuilt_gallery.json')
_VERSION_RE = re.compile(r'^[0-9][0-9A-Za-z.+\-]*$')
_SHA40_RE = re.compile(r'^[0-9a-f]{40}$')


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


def read_pin(repo_root):
    """The recorded gallery commit for this checkout:
    ``{'version', 'commit', 'source_commit'}``, or ``None`` with no (or an
    unusable) ``docs/prebuilt_gallery.json``."""
    try:
        with open(os.path.join(repo_root, PIN_RELPATH),
                  encoding='utf-8') as f:
            pin = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(pin, dict):
        return None
    if not _SHA40_RE.match(str(pin.get('commit', ''))):
        return None
    return pin


def irregular_entries(tree):
    """Entries of ``tree`` that are not regular files or directories. A
    symlink would be followed when the tree is copied or read, pulling a file
    from the build machine into the published site."""
    found = []
    for base, dirs, names in os.walk(tree):
        for name in dirs + names:
            path = os.path.join(base, name)
            if os.path.islink(path) or not (os.path.isdir(path)
                                            or os.path.isfile(path)):
                found.append(os.path.relpath(path, tree))
    return sorted(found)


def _git(args, cwd):
    return subprocess.run(['git'] + args, cwd=cwd, capture_output=True,
                          text=True)


def fetch(repo_root, remote=REPO_URL, out=print):
    """Copy the recorded gallery commit into
    ``<repo_root>/docs/auto_examples``.

    Returns ``(status, stale)``: status is ``'fetched'``, ``'missing'`` (no
    gallery recorded for this version, no gallery branch, or the commit
    cannot be fetched), ``'rejected'`` (the branch does not point at the
    recorded commit, the manifest disagrees with the record, or the tree
    holds a symlink) or
    ``'present'`` (a gallery is already there and is left alone); ``stale``
    lists the examples sphinx will still execute (``None`` when nothing was
    fetched).
    """
    version = project_version(repo_root)
    examples_dir = os.path.join(repo_root, 'examples')
    target = os.path.join(repo_root, 'docs', GALLERY_DIRNAME)
    if os.path.isdir(target) and os.listdir(target):
        out(f'pre-built gallery: {target} already exists; leaving it alone')
        return 'present', stale_examples(examples_dir, target)

    pin = read_pin(repo_root)
    if pin is None or pin.get('version') != version:
        out(f'pre-built gallery: none recorded for {version} in '
            f'{PIN_RELPATH}; sphinx will execute every example.')
        return 'missing', None
    commit = pin['commit']
    branch = branch_for(version)

    work = tempfile.mkdtemp(prefix='docs-gallery-')
    try:
        listed = _git(['ls-remote', remote, f'refs/heads/{branch}'], work)
        tip = listed.stdout.split()[0] if listed.stdout.split() else None
        if listed.returncode != 0 or tip is None:
            out(f'pre-built gallery: no {branch} branch on {remote}; sphinx '
                f'will execute every example.\n{listed.stderr.strip()}')
            return 'missing', None
        if tip != commit:
            out(f'pre-built gallery: REJECTED; {branch} points at {tip}, not '
                f'the recorded {commit}. sphinx will execute every example.')
            return 'rejected', None
        steps = (['init', '--quiet'],
                 ['fetch', '--quiet', '--depth', '1', remote, commit],
                 ['checkout', '--quiet', '--detach', 'FETCH_HEAD'])
        for step in steps:
            done = _git(step, work)
            if done.returncode != 0:
                out(f'pre-built gallery: cannot fetch commit {commit} from '
                    f'{remote}; sphinx will execute every example.\n'
                    f'{done.stderr.strip()}')
                return 'missing', None
        got = _git(['rev-parse', 'HEAD'], work).stdout.strip()
        source = os.path.join(work, GALLERY_DIRNAME)
        problems = []
        if got != commit:
            problems.append(f'fetched {got}, not the recorded {commit}')
        try:
            with open(os.path.join(work, MANIFEST_NAME),
                      encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            manifest = None
        if not isinstance(manifest, dict):
            problems.append(f'no readable {MANIFEST_NAME} in the commit')
        else:
            for key in ('version', 'source_commit'):
                if manifest.get(key) != pin.get(key):
                    problems.append(
                        f'{MANIFEST_NAME} {key} is {manifest.get(key)!r}, '
                        f'the record says {pin.get(key)!r}')
        if not os.path.isdir(source):
            problems.append(f'no {GALLERY_DIRNAME}/ in the commit')
        else:
            problems += [f'{rel}: not a regular file or directory'
                         for rel in irregular_entries(source)]
        if problems:
            out('pre-built gallery: REJECTED; sphinx will execute every '
                'example.\n  ' + '\n  '.join(problems))
            return 'rejected', None
        if os.path.isdir(target):
            os.rmdir(target)                      # empty, checked above
        shutil.copytree(source, target, symlinks=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    stale = stale_examples(examples_dir, target)
    n = len(example_md5s(examples_dir))
    out(f'pre-built gallery: fetched commit {commit} (built from '
        f'{pin.get("source_commit")}); {n - len(stale)} of {n} examples are '
        'current')
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
