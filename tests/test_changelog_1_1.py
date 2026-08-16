"""The 1.1.0 section exists, is on top, and records the behaviour changes.

A validation change that only appears under "New features" is a change users
will meet as a crash. `## 1.1.0 (unreleased)` did not exist when the plan was
written -- the top section was `## 1.0.1 (unreleased)` (measured at
59405545, `CHANGELOG.md:3`).

Three of the plan's six prescribed tests were strengthened before the section
was written, because as written they could not detect what they claim:

* `_section()` bounded a section at the next `\\n## `, which does NOT match
  `\\n### `, so `_section(text, '### Changed / validation')` swallowed the
  `### Documented limitations` subsection that follows it and every
  "the Changed section says X" assertion could be satisfied by text in
  Limitations. It now stops at the next heading of the same or higher level.
* `assert 'list' in changed.lower()` is satisfied by "listed", "listing" or
  any of a dozen unrelated words. It asserts the actual claim now.
* Nothing tested the FOURTH compatibility change (duplicate timestamps are
  rejected for FLAT inputs too -- `resolve_t` owns the check, so it is not
  hierarchy-only; see the Task 7 commit c51d274d, which flagged it for this
  task). It is both documented and EXECUTED here, so the entry cannot drift
  away from the code it describes.
"""
import os
import re

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _changelog():
    with open(os.path.join(REPO, 'CHANGELOG.md'), encoding='utf-8') as handle:
        return handle.read()


def _section(text, heading):
    """The body under `heading`, stopping at the next same-or-higher heading.

    The obvious `re.split(r'\\n## ')` does not bound a `###` subsection at
    all (`\\n## ` cannot match `\\n### `), which would let a `### Changed`
    assertion pass on text that lives under a later `### Limitations`.
    """
    level = len(heading) - len(heading.lstrip('#'))
    start = text.index(heading) + len(heading)
    stop = re.search(r'\n#{1,%d} ' % level, text[start:])
    return text[start:start + stop.start()] if stop else text[start:]


def test_changelog_has_a_1_1_0_unreleased_section():
    assert '## 1.1.0 (unreleased)' in _changelog()


def test_1_1_0_precedes_1_0_1():
    text = _changelog()
    assert text.index('## 1.1.0') < text.index('## 1.0.1')


def test_the_section_has_added_changed_and_limitations_headings():
    section = _section(_changelog(), '## 1.1.0 (unreleased)')
    for heading in ('### Added', '### Changed / validation',
                    '### Documented limitations'):
        assert heading in section, f'missing {heading}'


def test_changed_validation_documents_dual_axis_rejection():
    changed = _section(_changelog(), '### Changed / validation')
    assert 'both a row and a column MultiIndex' in changed


def test_changed_validation_documents_list_and_predict_changes():
    changed = _section(_changelog(), '### Changed / validation')
    # "list" alone is satisfied by "listed"; the claim is about a
    # hierarchical frame nested INSIDE a list, on the two entry points.
    assert 'inside a list' in changed
    assert '`hyp.plot`' in changed and '`hyp.predict`' in changed
    assert 'predict=' in changed


def test_changed_validation_documents_the_global_duplicate_time_rejection():
    """The fourth compatibility change: it is NOT hierarchy-only.

    `resolve_t` runs for flat inputs too, so this reaches callers who never
    touch a MultiIndex. The plan's own *Compatibility changes* table listed
    only three changes when Task 8 was written; c51d274d flagged this one
    for the CHANGELOG.
    """
    changed = _section(_changelog(), '### Changed / validation')
    assert 'duplicated' in changed
    assert 'flat' in changed.lower()
    for spelling in ('DatetimeIndex', 'TimedeltaIndex', 'PeriodIndex'):
        assert spelling in changed, f'missing {spelling}'


def test_the_documented_duplicate_time_rejection_actually_happens():
    """Execute the entry, so the prose cannot drift away from the code."""
    idx = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-02',
                          '2020-01-04', '2020-01-05'])
    frame = pd.DataFrame(np.arange(15, dtype=float).reshape(5, 3),
                         index=idx, columns=list('abc'))
    assert frame.index.nlevels == 1 and frame.columns.nlevels == 1
    with pytest.raises(ValueError, match='duplicated entr'):
        hyp.predict(frame, model='Kalman', t=1)


def test_added_documents_every_new_capability():
    added = _section(_changelog(), '### Added')
    for phrase in ('column MultiIndex', 'hue', 'hyp.predict', 'trace_data',
                   'plotly'):
        assert phrase in added, f'missing {phrase!r}'


def test_no_shipped_release_is_still_labelled_unreleased():
    """1.0.0 shipped to master on 2026-07-24; dev-1.0 never picked that up.

    `git show master:CHANGELOG.md` carries `## 1.0.0 (2026-07-24)`, and the
    two sections were otherwise byte-identical, so the heading on this branch
    was simply stale.
    """
    text = _changelog()
    assert '## 1.0.0 (2026-07-24)' in text
    assert '## 1.0.0 (unreleased)' not in text
