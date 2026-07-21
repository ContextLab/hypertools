"""Guard the Colab install cells in the committed tutorial notebooks.

Two layers (2026-07 release review, notebook-install finding):

* ALWAYS ON -- the tracked notebooks must not drift back to the defunct
  ``dev-1.0-refactor`` branch (whose head is no longer an ancestor of the
  release line, so it installs code missing recent fixes), and every hypertools
  GitHub-branch install must point at ONE consistent branch. This would have
  caught the stale-branch bug and prevents it regressing on any branch.

* RELEASE GATE (``HYPERTOOLS_REQUIRE_RELEASE=1``; the dedicated
  ``release-gate`` CI job sets it on master/tag builds) -- every
  hypertools install must be the plain PyPI spec, i.e. NO ``git+`` / ``@<branch>``
  preview install may survive into a release. This cannot pass by skipping.
"""

import glob
import json
import os
import re
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TUT_DIR = os.path.join(_REPO, 'docs', 'tutorials')

# docs/ is not shipped in the wheel/sdist (MANIFEST grafts only tests/), so when
# the suite runs from an installed package these notebook checks have nothing to
# inspect -- skip the whole module rather than fail.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(_TUT_DIR),
    reason='requires a source checkout (docs/tutorials/ absent in wheel/sdist)')

# a %pip/!pip install line that mentions hypertools
_PIP_HYP_RE = re.compile(r'(?:%|!)?\s*pip install\b.*hypertools', re.IGNORECASE)
# a hypertools GitHub-branch install spec, capturing the branch
_HYP_BRANCH_RE = re.compile(
    r'hypertools\[[^\]]*\]\s*@\s*'
    r'git\+https://github\.com/ContextLab/hypertools\.git@([\w./\-]+)')

REQUIRE_RELEASE = os.environ.get('HYPERTOOLS_REQUIRE_RELEASE') == '1'


def _git_ls(pattern):
    try:
        out = subprocess.run(['git', 'ls-files', pattern],
                             capture_output=True, text=True, cwd=_REPO,
                             timeout=30).stdout.split()
        return [os.path.join(_REPO, p) for p in out]
    except Exception:
        return []


def _tracked_tutorials():
    """The hand-authored tutorial notebooks (always git-tracked)."""
    got = _git_ls('docs/tutorials/*.ipynb')
    return got or sorted(glob.glob(os.path.join(_TUT_DIR, '*.ipynb')))


def _tracked_published_notebooks():
    """Every git-tracked published notebook. Today this is the 15 tutorials:
    docs/auto_examples/*.ipynb are GITIGNORED and regenerated at build time
    from docs/conf.py's branch-aware install cell, so they are not shipped and
    do not exist in a bare checkout (the release-gate CI job runs on a bare
    checkout). The GENERATED gallery is release-gated separately in the
    docs-clean CI job. Scanning the union keeps this gate correct if a gallery
    notebook is ever committed. (2026-07 release review, finding #2.)"""
    got = _git_ls('docs/tutorials/*.ipynb') + _git_ls('docs/auto_examples/*.ipynb')
    if got:
        return got
    # no .git -> source-archive fallback (both dirs, if present)
    out = []
    for d in (_TUT_DIR, os.path.join(_REPO, 'docs', 'auto_examples')):
        out += sorted(glob.glob(os.path.join(d, '*.ipynb')))
    return out


def _hyp_install_lines(path):
    """Every code-cell line that pip-installs hypertools, across all cells."""
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    lines = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        for line in ''.join(cell.get('source', [])).splitlines():
            if _PIP_HYP_RE.search(line):
                lines.append(line)
    return lines


def test_there_are_tracked_published_notebooks():
    # guards against the scan silently passing because it found nothing
    assert len(_tracked_tutorials()) >= 15
    # the published-notebook union is at least the tutorials
    assert len(_tracked_published_notebooks()) >= len(_tracked_tutorials())


def test_no_notebook_installs_the_defunct_refactor_branch():
    offenders = []
    branches = set()
    for path in _tracked_published_notebooks():
        for line in _hyp_install_lines(path):
            for br in _HYP_BRANCH_RE.findall(line):
                branches.add(br)
                if br == 'dev-1.0-refactor':
                    offenders.append((os.path.basename(path), line.strip()))
    assert not offenders, (
        'tutorial notebooks still install the defunct dev-1.0-refactor '
        f'branch (run scripts/add_colab_install_cell.py): {offenders}')
    # and every branch install points at ONE branch (no stale-vs-current mix)
    assert len(branches) <= 1, (
        f'tutorial notebooks install from mixed branches {sorted(branches)}; '
        'run scripts/add_colab_install_cell.py to unify them')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the '
           'release-gate CI job does on master/tag builds)')
def test_release_gate_no_branch_installs_in_published_notebooks():
    # finding: a release must not ship notebooks that install a GitHub branch
    # (they would install code that omits release fixes, or 404 once the branch
    # is deleted post-merge). Every hypertools install must be the PyPI spec.
    offenders = []
    for path in _tracked_published_notebooks():
        for line in _hyp_install_lines(path):
            if 'git+' in line or _HYP_BRANCH_RE.search(line):
                offenders.append((os.path.basename(path), line.strip()))
    assert not offenders, (
        'RELEASE GATE: published notebooks still contain preview/branch '
        'hypertools installs; run `python scripts/add_colab_install_cell.py` '
        f'on master AFTER the PyPI upload and commit: {offenders}')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the '
           'release-gate CI job does on master/tag builds)')
def test_release_gate_no_preview_note_in_published_notebooks():
    # release review, GAP #1: the migration flips the install LINE but must also
    # strip the standard "(<x> preview) / On release this becomes ..." note, or
    # the released notebooks ship saying "preview". Scans install cells only.
    offenders = []
    for path in _tracked_published_notebooks():
        with open(path, encoding='utf-8') as f:
            nb = json.load(f)
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell.get('source', []))
            if 'pip install' not in src:
                continue
            for marker in ('On release this becomes', ' preview)'):
                if marker in src:
                    offenders.append((os.path.basename(path), marker))
    assert not offenders, (
        'RELEASE GATE: published notebooks still carry a preview install note; '
        'run `python scripts/add_colab_install_cell.py` on master: '
        f'{offenders}')
