# -*- coding: utf-8 -*-

import pytest
import numpy as np
from hypertools.tools import text2mat

data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
        ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]

def test_transform_text():
    assert isinstance(text2mat(data)[0], np.ndarray)

def test_count_LDA():
    isinstance(text2mat(data, vectorizer='count', text_model='LDA')[0], np.ndarray)

def test_tfidf_LDA():
    isinstance(text2mat(data, vectorizer='tfidf', text_model='LDA')[0], np.ndarray)

def test_count_NMF():
    isinstance(text2mat(data, vectorizer='count', text_model='NMF')[0], np.ndarray)

def test_tfidf_NMF():
    isinstance(text2mat(data, vectorizer='tfidf', text_model='NMF')[0], np.ndarray)

def test_transform_ndims():
    assert text2mat(data, ndims=10)[0].shape[1]==10

def test_transform_no_text_model():
    assert isinstance(text2mat(data, text_model=None)[0], np.ndarray)

def test_text_model_params():
    assert isinstance(text2mat(data, text_params={'learning_method' : 'batch'})[0], np.ndarray)

def test_vectorizer_params():
    assert text2mat(data, vectorizer_params={'max_features' : 2}, text_model=None)[0].shape[1]==2
