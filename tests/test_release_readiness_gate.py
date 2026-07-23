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
import os
import re

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
# (e.g. 3 of 53) while tolerating normal gallery evolution.
_GALLERY_MIN = 40


def _manifest_is_complete(manifest, version, min_count=_GALLERY_MIN):
    """(ok, reason) for the published docs-notebooks/v{version}/manifest.json.

    Verifies the manifest describes the release version's COMPLETE,
    self-consistent gallery inventory in one shot (so the gate need not probe
    each notebook): right ref, internally consistent count, a non-truncated
    set, and the core examples present. Pure/unit-testable (no network)."""
    if not isinstance(manifest, dict):
        return False, f'manifest is not an object: {type(manifest).__name__}'
    want_ref = 'v' + version
    if manifest.get('ref') != want_ref:
        return False, f"manifest ref {manifest.get('ref')!r} != {want_ref!r}"
    nbs = manifest.get('notebooks')
    if not isinstance(nbs, list):
        return False, 'manifest has no notebooks list'
    if manifest.get('count') != len(nbs):
        return False, (f"manifest count {manifest.get('count')!r} != "
                       f'len(notebooks) {len(nbs)}')
    if len(nbs) < min_count:
        return False, (f'manifest lists only {len(nbs)} notebooks (< '
                       f'{min_count}) -- a partial/truncated publish?')
    missing_core = sorted({'plot_basic', 'plot_clusters', 'plot_align'}
                          - set(nbs))
    if missing_core:
        return False, f'manifest missing core gallery notebooks: {missing_core}'
    return True, 'ok'


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
    """The pure gallery-manifest validator the release gate relies on."""
    good = {'ref': 'v1.0.0', 'count': 40,
            'notebooks': ['plot_basic', 'plot_clusters', 'plot_align']
                         + [f'ex_{i}' for i in range(37)]}
    assert _manifest_is_complete(good, '1.0.0') == (True, 'ok')
    # wrong ref (published under a different version's namespace)
    assert not _manifest_is_complete({**good, 'ref': 'v0.9.0'}, '1.0.0')[0]
    # count disagrees with the actual list (partial/hand-edited)
    assert not _manifest_is_complete({**good, 'count': 99}, '1.0.0')[0]
    # truncated publish -- only the 3 core notebooks
    truncated = {'ref': 'v1.0.0', 'count': 3,
                 'notebooks': ['plot_basic', 'plot_clusters', 'plot_align']}
    assert not _manifest_is_complete(truncated, '1.0.0')[0]
    # complete count but a core example missing
    no_core = {'ref': 'v1.0.0', 'count': 40,
               'notebooks': [f'ex_{i}' for i in range(40)]}
    assert not _manifest_is_complete(no_core, '1.0.0')[0]
    # not even an object
    assert not _manifest_is_complete(['nope'], '1.0.0')[0]


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
    # blocker 2 + review Med #1: the gallery "Open in Colab" badges point at
    # github.com/.../blob/docs-notebooks/v<version>/auto_examples/<stem>.ipynb
    # (master 'latest' AND the v<version> tag 'stable' docs both resolve to
    # this one namespace -- see docs/post_build.py _publish_ref). Verify the
    # WHOLE published set from a single manifest fetch (not one probe per
    # notebook), then spot-check that a sample of the listed notebooks resolve.
    import json
    import urllib.request
    version = _project_version()
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
            'after building the docs.')

    # 2) the manifest must describe the complete, self-consistent inventory
    ok, reason = _manifest_is_complete(manifest, version)
    assert ok, f'RELEASE GATE: gallery manifest for v{version} invalid: {reason}'

    # 3) spot-check that a sample of the listed notebooks actually resolve
    #    (a manifest can't vouch for files that were never pushed)
    nbs = manifest['notebooks']
    sample = sorted({nbs[0], nbs[len(nbs) // 2], nbs[-1]})
    missing = []
    for stem in sample:
        url = base + 'auto_examples/' + stem + '.ipynb'
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                if r.status != 200:
                    missing.append((stem, r.status))
        except Exception as e:                        # HTTPError(404) etc.
            missing.append((stem, getattr(e, 'code', repr(e))))
    assert not missing, (
        f'RELEASE GATE: manifest lists notebooks that do not resolve on '
        f'docs-notebooks/v{version}/ (partial publish?): {missing}')
