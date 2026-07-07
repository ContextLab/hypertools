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
