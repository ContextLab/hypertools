"""The animation guide (docs/animation.rst) exists, is reachable, and
covers every animation feature 1.1 documents."""
import pathlib

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


def test_animation_guide_version_claims_match_the_package_version():
    """Minor finding (whole-branch review): the guide said 'new in 1.1'
    (x2), 'As of 1.1' and 'Before 1.1' -- but CHANGELOG.md's own section
    for this work is "## 1.0.1 (unreleased)" and `hypertools.__version__`
    / pyproject.toml agree on 1.0.1. There is no 1.1 release on this
    branch; these features shipped in 1.0.1, so the guide -- not the
    version -- was wrong."""
    import hypertools as hyp

    text = GUIDE.read_text()
    stale = [s for s in ('new in 1.1', 'As of 1.1', 'Before 1.1') if s in text]
    assert not stale, (
        f"docs/animation.rst still claims {stale}, but CHANGELOG.md and "
        f"hypertools.__version__ ({hyp.__version__}) agree these features "
        "are 1.0.1, not 1.1")

    changelog = (DOCS.parent / 'CHANGELOG.md').read_text()
    assert '## 1.0.1' in changelog, (
        "sanity check: CHANGELOG.md's own unreleased section must still "
        "say 1.0.1 for the assertion above to mean anything")


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
