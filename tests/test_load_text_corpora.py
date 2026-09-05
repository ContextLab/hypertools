# -*- coding: utf-8 -*-
"""GH #285: ``hyp.load('wiki')`` / ``hyp.load('nips')`` used to come back as
a list holding a single ``(n, 1)`` numpy object array, forcing every
consumer to do ``[str(p) for p in x[0].ravel()]`` before it could treat the
corpus as text -- inconsistent with ``hyp.load('sotus')``, which already
returned a flat list of strings. ``hypertools/io/load.py``'s
``_parse_rehosted`` (the ``jsongz_text``/``jsongz_strlist`` branch) now
returns a flat list of strings for all three hosted text corpora.

Real hosted loads (no mocks): these download the actual Dropbox-hosted
corpora, same as the rest of the ``test_load*`` suite.
"""
import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools.io.load import load


def _assert_text_corpus(data, expected_len=None):
    assert isinstance(data, list)
    if expected_len is not None:
        assert len(data) == expected_len
    assert len(data) > 0
    assert all(isinstance(doc, str) for doc in data)
    # first few entries are real, non-empty documents
    assert all(len(doc.strip()) > 0 for doc in data[:5])


def test_load_wiki_is_list_of_strings():
    data = load('wiki')
    _assert_text_corpus(data, expected_len=3136)


def test_load_nips_is_list_of_strings():
    data = load('nips')
    # ~7,241 papers per the rehosting manifest (tests/data/
    # rehosted_conversion_manifest.json); assert the exact count so a
    # future re-host that silently drops/duplicates documents is caught.
    _assert_text_corpus(data, expected_len=7241)


def test_load_sotus_is_list_of_strings():
    # sotus already returned a flat list of strings before GH #285; kept
    # here so all three hosted text corpora are asserted consistent side
    # by side.
    data = load('sotus')
    _assert_text_corpus(data, expected_len=29)


def test_wiki_documents_are_distinct():
    # a regression where every entry collapsed to the same string (e.g. a
    # broken reshape) would still pass a bare "list of str" check
    data = load('wiki')
    assert len(set(data[:50])) > 1


def test_plot_accepts_wiki_text_list():
    # end-to-end: the text pipeline (text2mat's default vectorizer +
    # semantic model) must accept the new flat list-of-strings shape.
    docs = load('wiki')[:20]
    fig = hyp.plot(docs, '.', show=False)
    assert fig is not None
