# -*- coding: utf-8 -*-

from builtins import range
import pytest

import numpy as np
import scipy

from hypertools.tools.reduce import reduce as reducer

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i in range(2)]
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
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_dims_2d():
    reduced_data_2d = reducer(data, ndims=2)
    assert reduced_data_2d[0].shape==(100,2)

def test_reduce_dims_1d():
    reduced_data_1d = reducer(data, ndims=1)
    assert reduced_data_1d[0].shape==(100,1)

def test_reduce_dims_3d_align():
    # Should return aligned data that is reduced to 3 dims
    reduced_aligned_data_3d = reduc(data, align=True)
    assert all(rad.shape == (100, 3) for rad in reduced_aligned_data_3d)

def test_reduce_dims_2d_align():
    # Should return aligned data that is reduced to 2 dims
    reduced_aligned_data_2d = reduc(data, ndims=2, align=True)
    assert all(rad.shape == (100, 2) for rad in reduced_aligned_data_2d)

def test_reduce_dims_1d_align():
    # Should return aligned data that is reduced to 1 dim
    reduced_aligned_data_1d = reduc(data, ndims=1, align=True)
    assert all(rad.shape == (100, 1) for rad in reduced_aligned_data_1d)
    
def test_reduce_assert_exception():
    with pytest.raises(Exception) as e_info:
        reduc(data,ndims=4)

def test_reduce_PCA():
    reduced_data_3d = reducer(data, model='PCA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_IncrementalPCA():
    reduced_data_3d = reducer(data, model='IncrementalPCA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_SparsePCA():
    reduced_data_3d = reducer(data, model='SparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_MiniBatchSparsePCA():
    reduced_data_3d = reducer(data, model='MiniBatchSparsePCA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_KernelPCA():
    reduced_data_3d = reducer(data, model='KernelPCA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_FastICA():
    reduced_data_3d = reducer(data, model='FastICA', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_FactorAnalysis():
    reduced_data_3d = reducer(data, model='FactorAnalysis', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_TruncatedSVD():
    reduced_data_3d = reducer(data, model='TruncatedSVD', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_DictionaryLearning():
    reduced_data_3d = reducer(data, model='DictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_MiniBatchDictionaryLearning():
    reduced_data_3d = reducer(data, model='MiniBatchDictionaryLearning', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_TSNE():
    reduced_data_3d = reducer(data, model='TSNE', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_Isomap():
    reduced_data_3d = reducer(data, model='Isomap', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_SpectralEmbedding():
    reduced_data_3d = reducer(data, model='SpectralEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_LocallyLinearEmbedding():
    reduced_data_3d = reducer(data, model='LocallyLinearEmbedding', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)

def test_reduce_MDS():
    reduced_data_3d = reducer(data, model='MDS', ndims=3)
    assert reduced_data_3d[0].shape==(100,3)
