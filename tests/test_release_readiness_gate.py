"""Release-readiness gate for the non-notebook release-facing references
(2026-07 release review). Companion to test_notebook_install_gate.py.

Some references are deliberately in DEV form on dev branches and MUST flip at
publish. This module pins two of them:

* README image URLs are pinned to a commit SHA today; at release they must be
  the ``v1.0.0`` git tag (immutable, survives the dev-1.0 branch deletion).
* the CHANGELOG heading is ``(unreleased)`` today; at release it must carry a
  real date.

ALWAYS-ON checks (safe on any branch) verify internal consistency and that the
referenced image files actually exist, so the tag will contain them. The
RELEASE-GATED checks (``HYPERTOOLS_REQUIRE_RELEASE=1``; the ``release-gate`` CI
job sets it on master/tag builds) enforce the flipped form and cannot pass by
skipping. See RELEASE_CHECKLIST.md.
"""

import datetime
import glob
import os
import re
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_README = os.path.join(_REPO, 'readme.md')
_CHANGELOG = os.path.join(_REPO, 'CHANGELOG.md')
_PYPROJECT = os.path.join(_REPO, 'pyproject.toml')

# readme.md / CHANGELOG.md / pyproject.toml ship in the sdist but NOT the wheel,
# so skip when the suite runs from an installed package that lacks them.
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_README) and os.path.isfile(_CHANGELOG)
         and os.path.isfile(_PYPROJECT)),
    reason='requires a source checkout (readme.md / CHANGELOG.md / pyproject '
           'absent)')

REQUIRE_RELEASE = os.environ.get('HYPERTOOLS_REQUIRE_RELEASE') == '1'

# an image URL served from OUR repo, capturing the ref (commit SHA now, tag at
# release): .../ContextLab/hypertools/<ref>/images/<name>
_OUR_IMG_RE = re.compile(
    r'raw\.githubusercontent\.com/ContextLab/hypertools/([^/]+)/(images/[^\s")]+)')
_SHA40_RE = re.compile(r'^[0-9a-f]{40}$')
# capture a PEP440-ish version (accepts pre/post-releases like 1.0.0rc1) so an
# rc/beta cut is compared against pyproject rather than hard-failing the regex.
_CHANGELOG_HEADING_RE = re.compile(
    r'^##\s*(\d+\.\d+\.\d+[\w.+!-]*)\s*\(([^)]*)\)', re.M)


def _project_version():
    """The single source of truth for the release version (pyproject.toml)."""
    m = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']',
                  open(_PYPROJECT, encoding='utf-8').read())
    assert m, 'pyproject.toml has no version'
    return m.group(1)


def _readme():
    with open(_README, encoding='utf-8') as f:
        return f.read()


def _our_image_refs():
    """(ref, image_path) for every OUR-repo raw image URL in the README."""
    return _OUR_IMG_RE.findall(_readme())


# the generated gallery has ~53 notebooks; a floor (not the exact count, which
# drifts as examples are added/removed) catches a truncated/partial publish
# (e.g. 3 of 53) while tolerating normal gallery evolution. Only used remotely,
# where the gallery isn't built to compare against exactly.
_GALLERY_MIN = 40
_CORE_STEMS = {'plot_basic', 'plot_clusters', 'plot_align'}


def _manifest_is_complete(manifest, version, expected_commit=None,
                          expected_stems=None, min_count=_GALLERY_MIN):
    """(ok, reason) for the published docs-notebooks/v{version}/manifest.json.

    Proves the manifest describes THIS release's complete, self-consistent
    gallery inventory -- not just some set with the right filenames:
      * ref == v{version};
      * source_commit is a real 40-hex sha AND, when ``expected_commit`` is
        given, equals it -- so same-named notebooks from an older release
        candidate cannot pass;
      * the notebooks list is string-typed, duplicate-free, and count-consistent;
      * when ``expected_stems`` is given (the gallery was built locally, i.e.
        the pre-push gate) the inventory must EXACTLY equal it -- catching
        extra/renamed/missing examples; otherwise (remote CI, no built gallery)
        it must clear the floor and contain the core examples.
    Pure/unit-testable (no network)."""
    if not isinstance(manifest, dict):
        return False, f'manifest is not an object: {type(manifest).__name__}'
    want_ref = 'v' + version
    if manifest.get('ref') != want_ref:
        return False, f"manifest ref {manifest.get('ref')!r} != {want_ref!r}"
    sha = manifest.get('source_commit')
    if not (isinstance(sha, str) and _SHA40_RE.match(sha)):
        return False, f'manifest source_commit is not a 40-hex sha: {sha!r}'
    if expected_commit is not None and sha != expected_commit:
        return False, (f'manifest source_commit {sha} != release HEAD '
                       f'{expected_commit} -- notebooks are from a different '
                       'commit; rebuild the gallery and re-publish')
    nbs = manifest.get('notebooks')
    if not isinstance(nbs, list):
        return False, 'manifest has no notebooks list'
    if not all(isinstance(s, str) for s in nbs):
        return False, 'manifest notebooks list has non-string entries'
    if len(set(nbs)) != len(nbs):
        dups = sorted({s for s in nbs if nbs.count(s) > 1})
        return False, f'manifest lists duplicate notebooks: {dups}'
    if manifest.get('count') != len(nbs):
        return False, (f"manifest count {manifest.get('count')!r} != "
                       f'len(notebooks) {len(nbs)}')
    if expected_stems is not None:
        got, want = set(nbs), set(expected_stems)
        if got != want:
            return False, (f'manifest inventory != generated gallery -- missing '
                           f'{sorted(want - got)}, extra {sorted(got - want)}')
    else:
        if len(nbs) < min_count:
            return False, (f'manifest lists only {len(nbs)} notebooks (< '
                           f'{min_count}) -- a partial/truncated publish?')
        missing_core = sorted(_CORE_STEMS - set(nbs))
        if missing_core:
            return False, (f'manifest missing core gallery notebooks: '
                           f'{missing_core}')
    return True, 'ok'


def _published_matches_manifest(published_stems, manifest_stems):
    """(ok, reason): the .ipynb stems ACTUALLY present on the docs-notebooks
    branch must exactly equal the manifest's inventory -- so a manifest that
    lists 53 while only a few files were pushed (or extra/renamed files linger)
    fails, even though only the manifest and the branch tree were fetched.
    Pure/unit-testable."""
    got, want = set(published_stems), set(manifest_stems)
    if got != want:
        return False, (f'published notebooks != manifest inventory -- missing '
                       f'on branch {sorted(want - got)}, unexpected on branch '
                       f'{sorted(got - want)}')
    return True, 'ok'


def _repo_head():
    """The release checkout's HEAD sha (40-hex lowercase), or None if this is
    not a git tree (e.g. an unpacked sdist)."""
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_REPO,
                             capture_output=True, text=True).stdout.strip()
        return out if _SHA40_RE.match(out) else None
    except Exception:
        return None


def _local_gallery_stems():
    """Stems of the locally built gallery (docs/auto_examples/*.ipynb), or None
    when it isn't built -- e.g. the remote release-gate CI job, which doesn't
    build docs, so there the gate falls back to the floor + core check."""
    d = os.path.join(_REPO, 'docs', 'auto_examples')
    if not os.path.isdir(d):
        return None
    stems = [os.path.basename(p)[:-len('.ipynb')]
             for p in glob.glob(os.path.join(d, '*.ipynb'))]
    return stems or None


def _published_gallery_stems(version):
    """Stems actually present under v{version}/auto_examples/ on the
    docs-notebooks branch, via ONE GitHub trees API request (not one request
    per notebook). Fails closed: a release integrity gate must not pass on a
    tree it could not read."""
    import json
    import urllib.request
    url = ('https://api.github.com/repos/ContextLab/hypertools/git/trees/'
           'docs-notebooks?recursive=1')
    req = urllib.request.Request(url, headers={
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'hypertools-release-gate'})
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode('utf-8'))
    except Exception as e:                                # HTTPError/URLError
        pytest.fail('RELEASE GATE: could not read the docs-notebooks tree '
                    f'({getattr(e, "code", e)}); cannot verify the published '
                    'gallery set.')
    if data.get('truncated'):
        pytest.fail('RELEASE GATE: docs-notebooks tree response was truncated; '
                    'cannot verify the full gallery set.')
    return _gallery_stems_from_tree(data, version)


def _gallery_stems_from_tree(tree_data, version):
    """Extract the v{version}/auto_examples/<stem>.ipynb stems from a GitHub
    trees API response body (only blobs directly under that prefix). Pure /
    unit-testable -- separated from the network fetch above."""
    prefix = f'v{version}/auto_examples/'
    stems = []
    for e in tree_data.get('tree', []):
        path = e.get('path', '')
        if (e.get('type') == 'blob' and path.startswith(prefix)
                and path.endswith('.ipynb')):
            rest = path[len(prefix):-len('.ipynb')]
            if '/' not in rest:            # flat auto_examples/, no subdirs
                stems.append(rest)
    return stems


# --------------------------------------------------------------- always on

def test_readme_has_our_repo_images():
    refs = _our_image_refs()
    assert len(refs) >= 5, f'expected several README images, found {len(refs)}'


def test_readme_images_share_one_ref_and_exist_in_tree():
    refs = _our_image_refs()
    pins = {ref for ref, _ in refs}
    # a single pin (all SHA now, all tag at release) -- no stale-vs-current mix
    assert len(pins) == 1, f'README images point at mixed refs {sorted(pins)}'
    # every referenced image exists in the tree, so the release tag contains it
    missing = sorted({img for _, img in refs
                      if not os.path.isfile(os.path.join(_REPO, img))})
    assert not missing, f'README references images absent from the tree: {missing}'


def test_changelog_top_version_matches_pyproject():
    m = _CHANGELOG_HEADING_RE.search(open(_CHANGELOG, encoding='utf-8').read())
    assert m, 'CHANGELOG.md has no "## X.Y.Z (...)" heading'
    assert m.group(1) == _project_version(), (
        f'CHANGELOG top version {m.group(1)!r} != pyproject version '
        f'{_project_version()!r}')


def test_manifest_is_complete_validator():
    """The pure gallery-manifest validator the release gate relies on, incl.
    the round-4 provenance checks (source_commit, exact inventory)."""
    head = 'a' * 40
    stems = ['plot_basic', 'plot_clusters', 'plot_align'] \
        + [f'ex_{i}' for i in range(37)]
    good = {'ref': 'v1.0.0', 'source_commit': head, 'count': 40,
            'notebooks': stems}
    # happy paths: remote (floor+core), with expected_commit, and exact-match
    assert _manifest_is_complete(good, '1.0.0') == (True, 'ok')
    assert _manifest_is_complete(good, '1.0.0', expected_commit=head)[0]
    assert _manifest_is_complete(good, '1.0.0', expected_commit=head,
                                 expected_stems=stems)[0]
    # correct inventory but STALE source_commit -- must fail
    assert not _manifest_is_complete(good, '1.0.0', expected_commit='b' * 40)[0]
    # missing / malformed source_commit -- must fail
    assert not _manifest_is_complete({**good, 'source_commit': None}, '1.0.0')[0]
    assert not _manifest_is_complete({**good, 'source_commit': 'abc'}, '1.0.0')[0]
    assert not _manifest_is_complete({**good, 'source_commit': 'A' * 40},
                                     '1.0.0')[0]              # uppercase != hex
    # wrong ref (published under a different version's namespace)
    assert not _manifest_is_complete({**good, 'ref': 'v0.9.0'}, '1.0.0')[0]
    # count disagrees with the actual list
    assert not _manifest_is_complete({**good, 'count': 99}, '1.0.0')[0]
    # duplicate notebook names
    dup = {**good, 'notebooks': stems[:-1] + [stems[0]]}
    assert not _manifest_is_complete(dup, '1.0.0')[0]
    # truncated publish -- only the 3 core notebooks (remote path, below floor)
    truncated = {'ref': 'v1.0.0', 'source_commit': head, 'count': 3,
                 'notebooks': ['plot_basic', 'plot_clusters', 'plot_align']}
    assert not _manifest_is_complete(truncated, '1.0.0')[0]
    # complete count but a core example missing (remote path)
    no_core = {'ref': 'v1.0.0', 'source_commit': head, 'count': 40,
               'notebooks': [f'ex_{i}' for i in range(40)]}
    assert not _manifest_is_complete(no_core, '1.0.0')[0]
    # extra / renamed vs the locally built gallery (exact-match path)
    assert not _manifest_is_complete(good, '1.0.0', expected_commit=head,
                                     expected_stems=stems + ['sneaky'])[0]
    assert not _manifest_is_complete(good, '1.0.0', expected_commit=head,
                                     expected_stems=stems[:-1] + ['renamed'])[0]
    # not even an object
    assert not _manifest_is_complete(['nope'], '1.0.0')[0]


def test_published_matches_manifest_validator():
    """The branch-tree-vs-manifest check the gate runs after the tree fetch."""
    stems = [f'ex_{i}' for i in range(5)]
    assert _published_matches_manifest(stems, stems) == (True, 'ok')
    assert _published_matches_manifest(list(reversed(stems)), stems)[0]  # order
    # manifest claims 5 but a file is missing on the branch -- must fail
    assert not _published_matches_manifest(stems[:-1], stems)[0]
    # an unexpected/renamed file lingers on the branch -- must fail
    assert not _published_matches_manifest(stems + ['stale'], stems)[0]


def test_gallery_stems_from_tree_parses_only_this_versions_notebooks():
    """The GitHub-tree parser picks exactly the v{version}/auto_examples/*.ipynb
    blobs (not other versions, other dirs, the manifest, or tree entries)."""
    tree = {'truncated': False, 'tree': [
        {'path': 'v1.0.0/auto_examples/plot_basic.ipynb', 'type': 'blob'},
        {'path': 'v1.0.0/auto_examples/animate_spin.ipynb', 'type': 'blob'},
        {'path': 'v1.0.0/auto_examples', 'type': 'tree'},        # dir entry
        {'path': 'v1.0.0/manifest.json', 'type': 'blob'},        # not a nb
        {'path': 'v0.9.0/auto_examples/plot_old.ipynb', 'type': 'blob'},  # other ver
        {'path': 'v1.0.0/auto_examples/nested/x.ipynb', 'type': 'blob'},  # subdir
        {'path': 'dev-1.0/auto_examples/plot_basic.ipynb', 'type': 'blob'},  # dev ns
    ]}
    assert sorted(_gallery_stems_from_tree(tree, '1.0.0')) == [
        'animate_spin', 'plot_basic']


# --------------------------------------------------------------- release gate

@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the release-gate '
           'CI job does on master/tag builds)')
def test_release_gate_readme_images_use_the_version_tag_not_commit_sha():
    # finding: README images pinned to a commit SHA must become the release tag
    # (a SHA can be garbage-collected / is opaque; the tag is stable). The ref
    # must be EXACTLY v<pyproject.version>, not merely any semver-looking tag.
    want = 'v' + _project_version()
    bad = sorted({ref for ref, _ in _our_image_refs() if ref != want})
    assert not bad, (
        f'RELEASE GATE: README image URLs must point at the {want} tag, not '
        f'{bad}. Re-point .../ContextLab/hypertools/<ref>/images/... to '
        f'/{want}/ (see RELEASE_CHECKLIST.md).')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the release-gate '
           'CI job does on master/tag builds)')
def test_release_gate_changelog_is_dated_not_unreleased():
    text = open(_CHANGELOG, encoding='utf-8').read()
    m = _CHANGELOG_HEADING_RE.search(text)
    assert m, 'CHANGELOG.md has no "## X.Y.Z (...)" heading'
    assert m.group(1) == _project_version(), (
        f'RELEASE GATE: CHANGELOG top version {m.group(1)!r} != pyproject '
        f'version {_project_version()!r}')
    date = m.group(2).strip()
    # require the canonical YYYY-MM-DD form AND a real calendar date: the regex
    # rejects fromisoformat-accepted-but-non-canonical values (20260721,
    # 2026-07-21T00:00, 2026-7-21), and fromisoformat rejects impossible dates
    # (2026-99-99, 2026-02-30) the regex alone would accept.
    canonical = re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is not None
    real = True
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        real = False
    assert canonical and real, (
        'RELEASE GATE: the top CHANGELOG heading must carry a real release '
        f'date in YYYY-MM-DD form, not {date!r} (see RELEASE_CHECKLIST.md).')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the release-gate '
           'CI job does on master/tag builds)')
def test_release_gate_readme_has_no_dev_branch_reference():
    # the README must not ship pointing at a dev preview branch
    offenders = [ln.strip() for ln in _readme().splitlines()
                 if 'dev-1.0-refactor' in ln
                 or re.search(r'hypertools\.git@dev', ln)]
    assert not offenders, (
        f'RELEASE GATE: README still references a dev branch: {offenders}')


@pytest.mark.skipif(
    not REQUIRE_RELEASE,
    reason='release gate; set HYPERTOOLS_REQUIRE_RELEASE=1 (the release-gate '
           'CI job does on master/tag builds)')
def test_release_gate_gallery_colab_notebooks_are_published():
    # blocker 2 + review rounds 3/4: the gallery "Open in Colab" badges point at
    # github.com/.../blob/docs-notebooks/v<version>/auto_examples/<stem>.ipynb
    # (master 'latest' AND the v<version> tag 'stable' docs both resolve to this
    # one namespace -- docs/post_build.py _publish_ref). Prove the published set
    # BELONGS to this release, from two requests (manifest + branch tree), not
    # one-per-notebook: (a) the manifest's source_commit == this checkout's
    # HEAD; (b) its inventory == the gallery built here, exactly, when available;
    # (c) the branch's ACTUAL .ipynb set == the manifest inventory.
    import json
    import urllib.request
    version = _project_version()
    head = _repo_head()
    if head is None:
        pytest.fail('RELEASE GATE: cannot determine the release HEAD commit; '
                    'run the gate from the git checkout being released.')
    base = ('https://raw.githubusercontent.com/ContextLab/hypertools/'
            f'docs-notebooks/v{version}/')

    # 1) fetch the single manifest published alongside the notebooks
    try:
        with urllib.request.urlopen(base + 'manifest.json', timeout=30) as r:
            manifest = json.loads(r.read().decode('utf-8'))
    except Exception as e:                            # HTTPError(404) etc.
        pytest.fail(
            f'RELEASE GATE: no gallery manifest at docs-notebooks/v{version}/'
            f'manifest.json ({getattr(e, "code", e)}). Run '
            f'scripts/publish_gallery_notebooks.py --ref v{version} --push '
            'after building the docs from the release commit.')

    # 2) the manifest must describe THIS release's complete, self-consistent
    #    set: source_commit == HEAD, and (when the gallery is built here) an
    #    exact inventory match to it
    ok, reason = _manifest_is_complete(manifest, version, expected_commit=head,
                                       expected_stems=_local_gallery_stems())
    assert ok, f'RELEASE GATE: gallery manifest for v{version} invalid: {reason}'

    # 3) the branch's ACTUAL notebook set must equal the manifest inventory --
    #    catches a manifest that lists files never (or no longer) on the branch
    ok, reason = _published_matches_manifest(
        _published_gallery_stems(version), manifest['notebooks'])
    assert ok, f'RELEASE GATE: docs-notebooks/v{version}/ {reason}'
