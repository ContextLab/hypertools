# -*- coding: utf-8 -*-
"""A Hugging Face vectorizer with `text2mat`'s DEFAULT semantic=/corpus=.

The trap (found in the 1.1.0 release review): `hyp.plot(docs,
vectorizer='all-MiniLM-L6-v2')` -- the call docs/optional_dependencies.rst
advertises -- first embedded the entire hosted 'wiki' corpus (3,136
documents, ~13 s) with MiniLM and then crashed with sklearn's internal
"ValueError: Negative values in data passed to LatentDirichletAllocation.fit",
because `text2mat`'s auto-skip of the topic-model semantic stage only
covered the gensim vectorizer names, not the Hugging Face tier.

Now (hypertools/tools/text2mat.py, `_is_pretrained_vectorizer`): the
vectorizer name is resolved BEFORE the corpus block, a pretrained
embedding vectorizer with the default LDA semantic silently resolves to
`semantic=None`, and the corpus is dropped (nothing is left to fit), so
no hosted corpus is ever loaded or embedded. An explicit non-default
topic model ('NMF') raises a clear hypertools ValueError before any
corpus work.

Every test here makes REAL sentence-transformers calls (the model is
downloaded from the Hub on first use and cached); nothing is mocked. The
`text` extra is optional and CI's dev install leaves it out, so the whole
module skips without `sentence_transformers` (the same `importorskip`
convention as the other HF-guarded tests).
"""

import os
import time
import warnings

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

pytest.importorskip('sentence_transformers')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import hypertools as hyp  # noqa: E402
from hypertools.tools.text2mat import (  # noqa: E402
    _hf_fallback_model, _is_pretrained_vectorizer, _resolve_registry_name,
    text2mat, vectorizer_models)

HF_MODEL = 'all-MiniLM-L6-v2'
HF_DIM = 384  # all-MiniLM-L6-v2's embedding width

DOCS = [
    'the cat sat on the mat',
    'dogs chase cats around the yard',
    'stock markets fell sharply today',
    'the central bank raised interest rates again',
    'a simple recipe for chocolate cake',
    'bake the bread at a high heat for an hour',
    'the spacecraft entered lunar orbit',
    'astronomers found a new exoplanet',
]

# Well under the ~13 s the trap cost (embedding 3,136 wiki documents), and
# well over a warm 8-document MiniLM call (measured ~2 s on 2026-09-05,
# dominated by re-instantiating the SentenceTransformer from disk cache).
FAST_ENOUGH_S = 6.0


def _hypertools_warnings(records):
    """The recorded warnings raised from hypertools' own code (torch /
    transformers deprecation chatter is not what these tests are about)."""
    return [w for w in records
            if 'hypertools' in w.filename.replace(os.sep, '/')
            and 'site-packages' not in w.filename]


@pytest.fixture(scope='module')
def explicit_none_embeddings():
    """The launch-example spelling (explicit semantic=None, corpus=None),
    which must be unchanged; also serves as the model warm-up so the timing
    assertions below measure the corpus decision, not a cold model load."""
    out = text2mat([DOCS], vectorizer=HF_MODEL, semantic=None, corpus=None)
    assert len(out) == 1
    assert out[0].shape == (len(DOCS), HF_DIM)
    return out[0]


def test_hf_model_class_is_marked_pretrained():
    cls = _hf_fallback_model(HF_MODEL)
    assert cls._hypertools_pretrained is True
    # the helper reads the live registry, so it answers only once the name
    # has been resolved into it (text2mat does this before calling it)
    _resolve_registry_name(HF_MODEL, vectorizer_models, 'vectorizer')
    assert vectorizer_models[HF_MODEL]._hypertools_pretrained is True
    assert _is_pretrained_vectorizer(HF_MODEL, HF_MODEL) is True
    assert _is_pretrained_vectorizer({'model': HF_MODEL}, HF_MODEL) is True
    # instances and bare classes carry the mark too
    assert _is_pretrained_vectorizer(cls(), None) is True
    assert _is_pretrained_vectorizer(cls, None) is True
    # count vectorizers are not pretrained
    assert _is_pretrained_vectorizer('CountVectorizer', 'CountVectorizer') is False
    assert _is_pretrained_vectorizer('TfidfVectorizer', 'TfidfVectorizer') is False


def test_default_semantic_and_corpus_return_embeddings_without_warning(
        explicit_none_embeddings):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        out = text2mat([DOCS], vectorizer=HF_MODEL)  # semantic/corpus defaults
        elapsed = time.perf_counter() - t0
    assert len(out) == 1
    assert out[0].shape == (len(DOCS), HF_DIM)
    # silent: this is the resolved default for an embedding vectorizer, not
    # a user mistake (the gensim path keeps its warning; see
    # tests/test_gensim_text.py)
    assert _hypertools_warnings(rec) == [], \
        [str(w.message) for w in _hypertools_warnings(rec)]
    assert not [w for w in rec if 'semantic' in str(w.message)], \
        [str(w.message) for w in rec]
    # identical to the explicit semantic=None, corpus=None spelling
    np.testing.assert_allclose(out[0], explicit_none_embeddings)
    # and the hosted 'wiki' corpus was NOT embedded on the way (that alone
    # took ~13 s before the fix)
    assert elapsed < FAST_ENOUGH_S, \
        f'default HF call took {elapsed:.1f}s -- did it embed the wiki corpus?'


def test_default_semantic_with_hosted_corpus_name_is_ignored(
        explicit_none_embeddings):
    # an explicit hosted-corpus name is equally unused: no load, no embed,
    # no warning, same output
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        out = text2mat([DOCS], vectorizer=HF_MODEL, corpus='nips')
        elapsed = time.perf_counter() - t0
    assert _hypertools_warnings(rec) == []
    np.testing.assert_allclose(out[0], explicit_none_embeddings)
    assert elapsed < FAST_ENOUGH_S


def test_plot_with_hf_vectorizer_and_defaults_returns_figure(
        explicit_none_embeddings):
    # the exact call docs/optional_dependencies.rst advertises
    t0 = time.perf_counter()
    fig = hyp.plot(DOCS, vectorizer=HF_MODEL, show=False)
    elapsed = time.perf_counter() - t0
    try:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 1
    finally:
        plt.close(fig)
    assert elapsed < FAST_ENOUGH_S


@pytest.mark.parametrize('semantic', ['NMF', {'model': 'NMF'},
                                      {'model': 'NMF', 'kwargs': {'n_components': 5}}])
def test_explicit_nmf_with_hf_vectorizer_raises_clear_valueerror(
        explicit_none_embeddings, semantic):
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match=r"semantic='NMF'.*vectorizer='all-MiniLM-L6-v2'"
                                         r".*embeddings.*not the word counts") as excinfo:
        text2mat([DOCS], vectorizer=HF_MODEL, semantic=semantic)
    elapsed = time.perf_counter() - t0
    # hypertools' own message, not sklearn's "Negative values in data
    # passed to ..." internal error
    assert 'Negative values' not in str(excinfo.value)
    assert 'semantic=None' in str(excinfo.value)
    # raised before any corpus work (no wiki download / embedding)
    assert elapsed < 1.0, f'{elapsed:.1f}s: was the corpus embedded first?'


def test_explicit_nmf_with_hf_vectorizer_raises_through_plot(
        explicit_none_embeddings):
    with pytest.raises(ValueError, match=r"semantic='NMF'.*embeddings"):
        hyp.plot(DOCS, vectorizer=HF_MODEL, semantic='NMF', show=False)


def test_explicit_lda_dict_spec_with_hf_vectorizer_raises(
        explicit_none_embeddings):
    # a configured LDA is an explicit request, not the bare default
    with pytest.raises(ValueError,
                       match=r"semantic='LatentDirichletAllocation'.*embeddings"):
        text2mat([DOCS], vectorizer=HF_MODEL,
                 semantic={'model': 'LatentDirichletAllocation',
                           'kwargs': {'n_components': 5}})


def test_count_vectorizer_defaults_unchanged():
    # the default CountVectorizer + pretrained hosted wiki LDA path is
    # untouched: 50 topics, no embedding-vectorizer logic involved
    out = text2mat([DOCS])
    assert len(out) == 1
    assert out[0].shape == (len(DOCS), 50)
    assert np.all(out[0] >= 0)
    assert not _is_pretrained_vectorizer('CountVectorizer', 'CountVectorizer')
