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


# The 1.1.0 heading reads `## 1.1.0 (unreleased)` on the development branch
# and `## 1.1.0 (YYYY-MM-DD)` once RELEASE_CHECKLIST.md step 2 dates it on
# master (the release-gate REQUIRES the date), so the section is located by
# either form. Measured 2026-09-04: the first master run of the release commit
# failed all 12 matrix jobs on the hard-coded `(unreleased)` string.
_HEADING_1_1_0_RE = re.compile(r'^## 1\.1\.0 \((unreleased|\d{4}-\d{2}-\d{2})\)$',
                               re.MULTILINE)


def _heading_1_1_0(text):
    match = _HEADING_1_1_0_RE.search(text)
    assert match, 'no `## 1.1.0 (unreleased|YYYY-MM-DD)` heading in CHANGELOG.md'
    return match.group(0)


def test_changelog_has_a_1_1_0_section():
    text = _changelog()
    heading = _heading_1_1_0(text)
    # exactly one 1.1.0 heading, and it is the first version section
    assert len(_HEADING_1_1_0_RE.findall(text)) == 1
    assert text.index(heading) == text.index('\n## ') + 1


def test_1_1_0_precedes_1_0_1():
    text = _changelog()
    assert text.index('## 1.1.0') < text.index('## 1.0.1')


def test_the_section_has_added_changed_and_limitations_headings():
    text = _changelog()
    section = _section(text, _heading_1_1_0(text))
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


# --------------------------------------------------------------------------
# the release-review fixes (2026-09-06) live INSIDE 1.1.0, and are true
# --------------------------------------------------------------------------

def test_the_release_review_fixes_are_a_subsection_of_1_1_0():
    """1.1.0 was reviewed against 1.0.0 before publication; the fixes ship in
    1.1.0, so they must sit under its heading and not under a new version."""
    text = _changelog()
    section = _section(text, _heading_1_1_0(text))
    assert '### Fixed during the release review' in section
    assert text.index('### Fixed during the release review') < text.index(
        '## 1.0.1')


def test_the_release_review_subsection_documents_its_user_visible_fixes():
    review = _section(_changelog(), '### Fixed during the release review')
    for phrase in ('metrics=', 'holdout=True', 'return_score=True',
                   'HypertoolsOfflineError', 'palette=', 'title_wrap=',
                   'offline=True', 'gmtoffset', 'streaming=True',
                   'text_windows', 'text2mat'):
        assert phrase in review, f'missing {phrase!r}'


def test_added_names_the_scoring_and_source_kwargs():
    """The Added section describes backtests, imputer scoring and the
    synthetic/web loaders, so it has to name the keywords that drive them."""
    added = _section(_changelog(), '### Added')
    for phrase in ('`metrics=`', '`return_imputed=True`', '`**source_kwargs`'):
        assert phrase in added, f'missing {phrase!r}'


def test_the_documented_duplicate_metric_rejection_actually_happens():
    """Execute the entry: a repeated metric is a ValueError that says which."""
    x = np.random.RandomState(0).randn(30, 3)
    with pytest.raises(ValueError, match='MAE'):
        hyp.predict(x, model=['Kalman'], holdout=3, metrics=['mae', 'MAE'])


def test_the_documented_holdout_t0_error_actually_names_t():
    x = np.random.RandomState(0).randn(30, 3)
    with pytest.raises(ValueError, match='t=0'):
        hyp.predict(x, model=['Kalman'], holdout=True, t=0)


def test_the_documented_empty_palette_rejection_actually_happens():
    x = np.random.RandomState(0).randn(30, 3)
    with pytest.raises(ValueError, match='palette'):
        hyp.plot([x, x], palette=[])


def test_the_documented_streaming_rejection_actually_happens():
    with pytest.raises(ValueError, match='streaming=True'):
        hyp.load('iris', streaming=True)


def test_the_documented_offline_error_import_paths_actually_work():
    from hypertools import HypertoolsOfflineError as top
    from hypertools.io import HypertoolsOfflineError as via_io
    assert top is via_io
    assert issubclass(top, hyp.HypertoolsIOError)
