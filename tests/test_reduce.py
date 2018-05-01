# -*- coding: utf-8 -*-

from builtins import range

import numpy as np

from hypertools.tools.reduce import reduce as reducer
from hypertools.plot.plot import plot

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=10) for i in range(2)]
reduced_data_2d = reducer(data,ndims=2)
reduced_data_1d = reducer(data,ndims=1)


def test_reduce_is_list():
    reduced_data_3d = reducer(data)
    assert type(reduced_data_3d) is list


def test_reduce_is_array():
    reduced_data_3d = reducer(data, ndims=3)
    assert isinstance(reduced_data_3d[0],np.ndarray)


def test_reduce_dims_3d():
    reduced_data_3d = reducer(data, ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_dims_2d():
    reduced_data_2d = reducer(data, ndims=2)
    assert reduced_data_2d[0].shape==(10,2)


def test_reduce_dims_1d():
    reduced_data_1d = reducer(data, ndims=1)
    assert reduced_data_1d[0].shape==(10,1)


def test_reduce_geo():
    geo = plot(data, show=False)
    reduced_data_3d = reducer(geo, ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_PCA():
    reduced_data_3d = reducer(data, reduce='PCA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_IncrementalPCA():
    reduced_data_3d = reducer(data, reduce='IncrementalPCA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_SparsePCA():
    reduced_data_3d = reducer(data, reduce='SparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_MiniBatchSparsePCA():
    reduced_data_3d = reducer(data, reduce='MiniBatchSparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_KernelPCA():
    reduced_data_3d = reducer(data, reduce='KernelPCA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_FastICA():
    reduced_data_3d = reducer(data, reduce='FastICA', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_FactorAnalysis():
    reduced_data_3d = reducer(data, reduce='FactorAnalysis', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_TruncatedSVD():
    reduced_data_3d = reducer(data, reduce='TruncatedSVD', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_DictionaryLearning():
    reduced_data_3d = reducer(data, reduce='DictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_MiniBatchDictionaryLearning():
    reduced_data_3d = reducer(data, reduce='MiniBatchDictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_TSNE():
    reduced_data_3d = reducer(data, reduce='TSNE', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_Isomap():
    reduced_data_3d = reducer(data, reduce='Isomap', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_SpectralEmbedding():
    reduced_data_3d = reducer(data, reduce='SpectralEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_LocallyLinearEmbedding():
    reduced_data_3d = reducer(data, reduce='LocallyLinearEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_MDS():
    reduced_data_3d = reducer(data, reduce='MDS', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_UMAP():
    reduced_data_3d = reducer(data, reduce='UMAP', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)
