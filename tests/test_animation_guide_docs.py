"""The animation guide (docs/animation.rst) exists, is reachable, and
covers every animation feature 1.1 documents."""
import pathlib
import re

import pytest

DOCS = pathlib.Path(__file__).resolve().parents[1] / 'docs'
GUIDE = DOCS / 'animation.rst'


def test_animation_guide_exists():
    assert GUIDE.is_file(), 'docs/animation.rst is a Task 9 deliverable'


def test_animation_guide_is_in_the_toctree():
    """Not decorative: an unreferenced .rst makes Sphinx warn, and the repo
    holds a zero-warning build standard."""
    index = (DOCS / 'index.rst').read_text()
    toctree = index.split('.. toctree::', 1)[1]
    entries = [ln.strip() for ln in toctree.split('\n\n')[1].splitlines()]
    assert 'animation' in entries, f'not in the toctree: {entries}'


@pytest.mark.parametrize('topic', [
    "order='serial'",      # ordering as its own axis
    'chemtrails',          # trails
    'title=',              # per-segment title sequences
    'simplify=',           # morph tractability
    'alpha=',              # per-dataset styling
    'on_frame=',           # the hook itself
    'FrameContext',        # the public context type
    'revealed_counts',     # the serial schedule it exposes
    'segment_kind',        # morph segment structure
    '_func',               # migration away from the private internals
])
def test_animation_guide_covers(topic):
    assert topic in GUIDE.read_text(), f'guide does not mention {topic}'


def test_animation_guide_documents_both_backend_schedules():
    text = GUIDE.read_text()
    assert 'matplotlib' in text and 'plotly' in text
    assert 'render time' in text
    assert 'build time' in text


def test_animation_guide_states_the_callback_contract_verbatim():
    """The one sentence that has to be identical in the guide, the plot()
    docstring and the CHANGELOG."""
    text = ' '.join(GUIDE.read_text().split())
    assert ('Callbacks must be deterministic and idempotent for a given '
            'frame context.') in text
    assert ('must not depend on call count, call order, wall-clock time, '
            'or accumulated external state.') in text


def test_animation_guide_does_not_call_the_contract_purity():
    """Regression guard. Callbacks mutate artists by design -- calling the
    contract 'purity' is the misstatement this plan's v4 removed, and the
    guide's own example sets a title every frame."""
    text = GUIDE.read_text().lower()
    assert 'pure function' not in text


def test_animation_guide_marks_post_construction_registration_matplotlib_only():
    """`HyperAnimation.on_frame()` cannot exist on plotly: animated plotly
    returns a plain go.Figure whose frames are already built when plot()
    returns (plot.py:4605-4612 -- only animated matplotlib sets line_ani).
    The guide must not present post-construction registration as portable."""
    text = ' '.join(GUIDE.read_text().split())
    assert 'Registering after construction is matplotlib-only' in text
    assert 'This is not available on plotly, and cannot be.' in text
    # and it must say what to do instead
    assert 'pass the callback to' in text.lower()


def test_animation_guide_labels_its_backend_specific_examples():
    """ctx.axes is None on plotly and ctx.artists are traces, so neither
    example is portable. Each must be labelled rather than sitting
    unmarked in a backend-general section."""
    text = GUIDE.read_text()
    assert '# MATPLOTLIB ONLY' in text
    assert '# PLOTLY ONLY' in text


def test_animation_guide_documents_artist_lifetime_for_both_backends():
    """Artists are SHARED on matplotlib (FuncAnimation mutates the same
    Line2D objects every render) and on plotly spin (camera-only frames).
    Only plotly's reveal/morph styles hand out per-frame trace payloads. A
    caller who assumes per-frame artists writes a conditional mutation that
    silently applies to the whole animation."""
    text = ' '.join(GUIDE.read_text().split()).lower()
    assert 'artist lifetime' in text
    assert 'matplotlib, **all** styles' in text or 'matplotlib, all' in text
    assert 'whole animation' in text or 'figure-wide' in text
    assert 'spin' in text
    # the corrected claim must not come back
    assert 'every style on matplotlib, is per-frame' not in text


#: Every way the guide dates a behaviour change. The capture group is the
#: version the sentence claims; `1.0` in ``alpha=[1.0, 0.4]`` or "matplotlib's
#: opaque, i.e. 1.0" is deliberately NOT reachable by these prefixes.
_VERSION_CLAIM = re.compile(
    r'(?:new in|As of|Before|versionadded::|versionchanged::)\s+'
    r'(\d+\.\d+(?:\.\d+)?)', re.IGNORECASE)


def test_animation_guide_version_claims_match_the_package_version():
    """Every "new in X" in the guide must name the version that ships it.

    History: the guide first said 'new in 1.1' while the package declared
    1.0.1, so this test pinned it to 1.0.1. That premise is now retired --
    `pyproject.toml` declares 1.1.0, `hypertools.__version__` reports 1.1.0,
    and CHANGELOG.md states outright that "1.0.1 was never published on its
    own ... now ship as part of 1.1.0". A guide dating features to 1.0.1
    therefore points users at a release that will never exist.

    So the assertion is no longer against a hard-coded string: it is against
    `hypertools.__version__`, which is the one number a reader can check with
    `pip show`, and against the releases CHANGELOG.md records.

    1.1.1 (a patch release) retired a second premise: that the guide's
    features are all dated to the CURRENT version. A feature that arrived in
    1.1.0 stays "new in 1.1.0" under 1.1.1. What must hold is that every
    version the guide names is one a user can install: a dated release in
    CHANGELOG.md, no newer than the package, and never the unpublished
    1.0.1. The feature release of the current line must be among them, so
    the guide cannot fall a whole feature release behind unnoticed."""
    import hypertools as hyp

    def _parse(version):
        # leading X.Y.Z only: a pre-release suffix (`1.2.0rc1`) is not
        # part of the release a guide can name
        parts = [int(p) for p in
                 re.match(r'\d+(?:\.\d+){0,2}', version).group(0).split('.')]
        return tuple(parts + [0] * (3 - len(parts)))

    text = GUIDE.read_text()
    claimed = sorted(set(_VERSION_CLAIM.findall(text)))
    assert claimed, (
        'sanity check: the guide must date at least one behaviour change, '
        'or this test asserts nothing')
    changelog = (DOCS.parent / 'CHANGELOG.md').read_text()
    current = _parse(hyp.__version__)
    for version in claimed:
        assert re.fullmatch(r'\d+\.\d+\.\d+', version), (
            f'docs/animation.rst dates a feature to {version!r}; name the '
            'full release (X.Y.Z) so a reader can install exactly it')
        assert version != '1.0.1', (
            'docs/animation.rst dates a feature to 1.0.1, which was never '
            'published')
        assert _parse(version) <= current, (
            f'docs/animation.rst dates a feature to {version}, but the '
            f'package is {hyp.__version__}. Retarget the guide (or the '
            'version); a version a user cannot install is not a useful date.')
        assert re.search(
            r'(?m)^## %s \(\d{4}-\d{2}-\d{2}\)$' % re.escape(version),
            changelog), (
            f'docs/animation.rst dates a feature to {version}, which has no '
            'dated section in CHANGELOG.md')
    feature_release = '%d.%d.0' % current[:2]
    assert feature_release in claimed, (
        f'docs/animation.rst dates features to {claimed} but none to '
        f'{feature_release}, the feature release of the {hyp.__version__} '
        'line')

    assert f'## {hyp.__version__}' in changelog, (
        "sanity check: CHANGELOG.md must carry a section for the declared "
        "version for the assertion above to mean anything")
    if '## 1.0.1' in changelog:
        assert '1.0.1 was never published' in changelog, (
            'CHANGELOG.md still keeps a `## 1.0.1` section, so it must keep '
            'saying that 1.0.1 never shipped -- otherwise a reader who finds '
            'these features there is never told which release carries them')


def test_no_shipped_module_or_guide_dates_a_behaviour_to_the_patch_line():
    """The retired patch number must not leak back into what users read.

    `1.0.1` was developed as a patch release and folded into 1.1.0 without
    ever being published, so a docstring or guide saying "new in 1.0.1" (or
    "pre-1.0.1", or "since 1.0.1") sends a reader to a version they cannot
    install and cannot diff against. CHANGELOG.md is deliberately exempt: it
    keeps a `## 1.0.1` section precisely to say that the release never
    happened.

    The forecast-style contract is what made this worth pinning -- it was
    dated 1.0.1 in `plot.forecast`, `plot.plot`, `plot.plotly_backend` and
    `docs/animation.rst` at once, so a single stale site is easy to leave
    behind.
    """
    repo = DOCS.parent
    shipped = sorted(repo.glob('hypertools/**/*.py')) + sorted(
        DOCS.glob('*.rst'))
    offenders = []
    for path in shipped:
        rel = path.relative_to(repo)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if '1.0.1' in line:
                offenders.append(f'{rel}:{number}: {line.strip()}')
    assert not offenders, (
        '1.0.1 was never published (see CHANGELOG.md); date these to the '
        'release that actually carries them:\n' + '\n'.join(offenders))


def test_animation_guide_gives_both_failure_modes_not_just_persistence():
    """The guide must not say persistence applies to both backends.

    Measured 2026-07-30: plotly's parallel/serial/window/morph frames are
    INDEPENDENT payloads (`fig.frames[0].data[0] is not
    fig.frames[1].data[0]`), so a frame-0-only mutation there affects only
    frame 0 -- the opposite of matplotlib and plotly spin, where it affects
    everything. An earlier draft stated the shared behaviour as universal.
    Both modes must be present, and the persistence claim must be scoped.

    The rule must also stay stated as an ASSIGNMENT rule, not a ban on
    per-frame decisions: v4.3 said "never write a mutation that fires on
    one frame only", which forbids highlighting a single frame -- a
    legitimate thing to want, and portable when the condition sits in the
    value rather than around the call.
    """
    raw = GUIDE.read_text()
    # collapse whitespace AND strip rst emphasis, so the assertions below
    # survive `**shared**` being bolded or re-wrapped
    text = ' '.join(raw.replace('*', '').split()).lower()
    # both failure modes are described, not just the shared one
    assert 'whole animation' in text
    assert 'only frame 0' in text
    # and persistence is scoped to shared artists rather than to "both backends"
    assert 'where artists are shared' in text
    # the plotly example uses a real plotly API, not matplotlib's set_color
    assert '.line.color' in raw
    # the rule is about assigning every invocation, and single-frame
    # highlighting is shown as supported rather than forbidden
    assert 'assign the complete value' in text
    assert 'highlighting exactly one frame' in text
