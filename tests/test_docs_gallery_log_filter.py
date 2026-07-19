# -*- coding: utf-8 -*-
"""Unit test for the docs -W-gate logging filter (2026-07 review, finding #2).

The docs-clean CI gate builds with -W and would otherwise fail on a TRANSIENT
third-party doc-site outage (sphinx-gallery fetching an external searchindex.js
gets e.g. an HTTP 503). docs/_gallery_log_filter.py drops ONLY that specific
fetch-failure warning. That filter matches on a fixed substring of
sphinx-gallery's message, so this test both documents the intended behavior and
guards it: if sphinx-gallery ever changes the wording, the guard here fails
instead of the filter silently ceasing to work.
"""
import importlib.util
import logging
from pathlib import Path

import pytest

# docs/ is not an importable package; load the side-effect-free filter module
# directly by path
_MOD_PATH = Path(__file__).resolve().parent.parent / 'docs' / '_gallery_log_filter.py'
_spec = importlib.util.spec_from_file_location('_gallery_log_filter', _MOD_PATH)
glf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glf)


def _record(msg):
    return logging.LogRecord(glf.GALLERY_LOGGER, logging.WARNING, __file__, 0,
                             msg, None, None)


# the exact messages sphinx-gallery's _handle_http_url_error builds
_FETCH_FAILURES = [
    'The following HTTPError has occurred fetching '
    'https://scikit-learn.org/stable//searchindex.js: 503 (Service Unavailable)',
    'The following URLError has occurred fetching: <urlopen error timed out>',
]
# warnings that MUST still fail the -W build (never filtered)
_REAL_WARNINGS = [
    'py:class reference target not found: hypertools.Foo',
    'Example plot_foo.py failed to execute: Traceback ...',
    'image file not readable: foo.png',
    'the following documents are not included in any toctree',
]


@pytest.mark.parametrize('msg', _FETCH_FAILURES)
def test_transient_external_fetch_warning_is_dropped(msg):
    assert glf.TransientDocLinkFetchFilter().filter(_record(msg)) is False


@pytest.mark.parametrize('msg', _REAL_WARNINGS)
def test_real_warnings_are_kept(msg):
    assert glf.TransientDocLinkFetchFilter().filter(_record(msg)) is True


def test_install_attaches_filter_to_the_gallery_logger():
    name = 'sphinx.sphinx-gallery.__test__'          # isolated logger
    glf.install(name)
    logger = logging.getLogger(name)
    try:
        assert any(isinstance(f, glf.TransientDocLinkFetchFilter)
                   for f in logger.filters)
        # a fetch-failure record is rejected by the logger's filter chain,
        # a genuine warning passes it
        assert logger.filter(_record(_FETCH_FAILURES[0])) is False
        assert logger.filter(_record(_REAL_WARNINGS[0]))  # truthy (record/True)
    finally:
        logger.filters.clear()


def test_marker_still_matches_sphinx_gallery_source():
    # the guard finding #2 asks for: if sphinx-gallery changes the wording its
    # resolver emits, this fails loudly so the filter is updated rather than
    # silently ceasing to work
    dr = pytest.importorskip('sphinx_gallery.docs_resolv')
    import inspect
    src = inspect.getsource(dr._handle_http_url_error)
    assert 'has occurred' in src and 'fetching' in src, (
        'sphinx-gallery changed its fetch-failure message wording; update '
        'docs/_gallery_log_filter.py._TRANSIENT_FETCH_MARKER (and this test)')
