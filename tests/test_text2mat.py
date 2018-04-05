# -*- coding: utf-8 -*-

import pytest
import numpy as np
from hypertools.tools import text2mat
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]

def test_transform_text():
    assert isinstance(text2mat(data)[0], np.ndarray)

def test_count_LDA():
    isinstance(text2mat(data, vectorizer='CountVectorizer',
                        semantic='LatentDirichletAllocation', corpus=data)[0], np.ndarray)

def test_tfidf_LDA():
    isinstance(text2mat(data, vectorizer='TfidfVectorizer',
                        semantic='LatentDirichletAllocation', corpus=data)[0], np.ndarray)

def test_count_NMF():
    isinstance(text2mat(data, vectorizer='CountVectorizer', semantic='NMF', corpus=data)[0], np.ndarray)

def test_tfidf_NMF():
    isinstance(text2mat(data, vectorizer='TfidfVectorizer', semantic='NMF', corpus=data)[0], np.ndarray)

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
