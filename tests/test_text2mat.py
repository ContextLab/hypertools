# -*- coding: utf-8 -*-

import numpy as np
from hypertools.tools import text2mat
from sklearn.decomposition import LatentDirichletAllocation

data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]

def test_transform_text():
    assert isinstance(text2mat(data)[0], np.ndarray)

def test_count_LDA():
    # GH #244: this test previously called `isinstance(...)` with no
    # `assert`, so it always passed regardless of the result. LDA's
    # transform output is a per-document topic-probability distribution,
    # so each row must sum to 1 -- a real, non-tautological invariant.
    out = text2mat(data, vectorizer='CountVectorizer',
                    semantic='LatentDirichletAllocation', corpus=data)
    assert isinstance(out[0], np.ndarray)
    assert all(o.shape == (3, 20) for o in out)
    for o in out:
        assert np.allclose(o.sum(axis=1), 1.0, atol=1e-6)

def test_tfidf_LDA():
    out = text2mat(data, vectorizer='TfidfVectorizer',
                    semantic='LatentDirichletAllocation', corpus=data)
    assert isinstance(out[0], np.ndarray)
    assert all(o.shape == (3, 20) for o in out)
    for o in out:
        assert np.allclose(o.sum(axis=1), 1.0, atol=1e-6)

def test_count_NMF():
    # GH #244: same missing-`assert` bug as above. NMF factors are
    # non-negative by construction, and the fitted model should explain
    # some real signal in the data (not an all-zero degenerate fit).
    out = text2mat(data, vectorizer='CountVectorizer', semantic='NMF', corpus=data)
    assert isinstance(out[0], np.ndarray)
    assert all(o.shape == (3, 20) for o in out)
    assert all((o >= 0).all() for o in out)
    assert max(o.max() for o in out) > 0

def test_tfidf_NMF():
    out = text2mat(data, vectorizer='TfidfVectorizer', semantic='NMF', corpus=data)
    assert isinstance(out[0], np.ndarray)
    assert all(o.shape == (3, 20) for o in out)
    assert all((o >= 0).all() for o in out)
    assert max(o.max() for o in out) > 0

def test_transform_no_text_model():
    assert isinstance(text2mat(data, semantic=None, corpus=data)[0], np.ndarray)

def test_text_model_params():
    assert isinstance(text2mat(data, semantic={
        'model' : 'LatentDirichletAllocation',
        'params' : {
            'learning_method' : 'batch'
            }}
        , corpus=data)[0], np.ndarray)

def test_vectorizer_params():
    assert text2mat(data, vectorizer={
        'model' : 'CountVectorizer',
        'params': {
        'max_features' : 2
        }}, corpus=data)[0].shape[1]==20

def test_LDA_class():
    assert text2mat(data, semantic=LatentDirichletAllocation, corpus=data)[0].shape[1]==10

def test_LDA_class_instance():
    user_model = LatentDirichletAllocation(n_components=15)
    assert text2mat(data, semantic=user_model, corpus=data)[0].shape[1]==15

def test_corpus():
    assert text2mat(data, corpus=data)[0].shape[1]==20


# -------------------------- flat list of strings is ONE dataset (1.1, X1)

import pytest  # noqa: E402

DOCS = ['cats like milk', 'dogs like bones', 'birds like seeds']


def test_flat_list_of_strings_is_one_dataset():
    # before 1.1 a flat list split by each string's CHARACTER length and
    # returned [(N, d), (0, d), (0, d), ...]
    out = text2mat(DOCS, vectorizer='CountVectorizer', semantic=None,
                   corpus=None)
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (3, 7)          # 7 distinct words
    assert out[0].sum() == 9               # 3 words per document


def test_nested_list_matches_the_flat_form():
    flat = text2mat(DOCS, semantic=None, corpus=None)
    nested = text2mat([DOCS], semantic=None, corpus=None)
    assert len(nested) == 1
    np.testing.assert_array_equal(flat[0], nested[0])


def test_ragged_list_of_lists_is_one_dataset_per_inner_list():
    out = text2mat([DOCS, DOCS[:2]], semantic=None, corpus=None)
    assert [o.shape for o in out] == [(3, 7), (2, 7)]
    np.testing.assert_array_equal(out[0][:2], out[1])
    # the same flat/nested rule applies to corpus=
    with_corpus = text2mat([DOCS, DOCS[:2]], semantic=None,
                           corpus=[DOCS, DOCS[:2]])
    assert [o.shape for o in with_corpus] == [(3, 7), (2, 7)]
    flat_corpus = text2mat([DOCS, DOCS[:2]], semantic=None, corpus=DOCS)
    assert [o.shape for o in flat_corpus] == [(3, 7), (2, 7)]


def test_flat_list_through_the_default_topic_model():
    out = text2mat(DOCS, corpus=DOCS)     # CountVectorizer -> LDA
    assert len(out) == 1 and out[0].shape == (3, 20)
    assert np.allclose(out[0].sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize('argname', ['data', 'corpus'])
def test_mixed_strings_and_lists_raise(argname):
    mixed = [DOCS[0], DOCS[1:]]
    kwargs = {'data': mixed, 'semantic': None, 'corpus': None} \
        if argname == 'data' else \
        {'data': DOCS, 'semantic': None, 'corpus': mixed}
    with pytest.raises(ValueError, match=f'{argname}= mixes strings and '
                                         'lists'):
        text2mat(**kwargs)


def test_single_string_is_one_dataset_of_one_document():
    out = text2mat(DOCS[0], semantic=None, corpus=None)
    assert len(out) == 1 and out[0].shape == (1, 3)
