# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

import hypertools as hyp

models = ['DictionaryLearning', 'FactorAnalysis', 'FastICA', 'IncrementalPCA', 'KernelPCA', 'LatentDirichletAllocation',
          'MiniBatchDictionaryLearning', 'MiniBatchSparsePCA', 'NMF', 'PCA', 'SparsePCA', 'SparseCoder', 'TruncatedSVD',
          'Isomap', 'LocallyLinearEmbedding', 'MDS', 'SpectralEmbedding', 'TSNE', 'UMAP', 'PPCA']


normalized_weights = hyp.manip(hyp.load('weights_sample'), 'Normalize')


def test_reduce():
    n_components = 10
    for m in models:
        if m == 'SparseCoder':
            dictionary = dw.stack(hyp.reduce(normalized_weights, 'PCA', n_components=n_components)).values
            next_model = {'model': m, 'args': [], 'kwargs': {'dictionary': dictionary}}
        else:
            next_model = {'model': m, 'args': [], 'kwargs': {'n_components': n_components}}

        reduced_weights = hyp.reduce(normalized_weights, model=next_model)
        assert type(reduced_weights) is list
        assert len(reduced_weights) == len(normalized_weights)
        assert all([r.shape[0] == w.shape[0] for r, w in zip(reduced_weights, normalized_weights)])
        assert all([r.shape[1] == n_components for r in reduced_weights])

        x = hyp.reduce(normalized_weights[0], model=next_model)
        assert type(x) is pd.DataFrame
        assert x.shape[0] == normalized_weights[0].shape[0]
        assert x.shape[1] == n_components


test_reduce()
