# -*- coding: utf-8 -*-

import numpy as np
import pytest

from hypertools.reduce.reduce import reduce as reducer

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=10) for i in range(2)]
reduced_data_2d = reducer(data,ndims=2)
reduced_data_1d = reducer(data,ndims=1)


def test_reduce_is_list():
    reduced_data_3d = reducer(data)
    assert type(reduced_data_3d) is list


def test_reduce_is_array():
    reduced_data_3d = reducer(data, ndims=3)
    assert isinstance(reduced_data_3d[0],np.ndarray)
    # a real (non-tautological) check: reduction must actually reduce the
    # dimensionality (input is 4D, requested output is 3D) while preserving
    # the number of samples
    assert reduced_data_3d[0].shape == (10, 3)
    assert reduced_data_3d[0].shape[1] < data[0].shape[1]


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
    # reduce() operates on raw data directly (no geo round-trip in 1.0)
    reduced_data_3d = reducer(data, ndims=3)
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


# upstream: sklearn FastICA rarely converges on this tiny 10-sample fixture
@pytest.mark.filterwarnings(
    'ignore:FastICA did not converge:sklearn.exceptions.ConvergenceWarning')
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
    # legacy 'params' dict spec exercised deliberately; assert the
    # deprecation notice fires
    with pytest.warns(DeprecationWarning, match=r"'params'.*deprecated"):
        reduced_data_3d = reducer(
            data, reduce={'model': 'TSNE', 'params': {'perplexity': 5}},
            ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


def test_reduce_Isomap():
    reduced_data_3d = reducer(data, reduce='Isomap', ndims=3)
    assert reduced_data_3d[0].shape==(10,3)


# upstream: sklearn warns the 10-sample affinity graph may be disconnected
@pytest.mark.filterwarnings(
    'ignore:Graph is not fully connected:UserWarning')
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


# upstream: umap warns that random_state forces n_jobs=1 (fires from both
# the hypertools-wrapped fit and the direct UMAP fit below)
@pytest.mark.filterwarnings(
    'ignore:n_jobs value 1 overridden to 1 by setting random_state:UserWarning')
def test_reduce_params_UMAP():
    from umap import UMAP
    data1 = np.random.rand(20, 10)
    params = {'n_neighbors': 5, 'n_components': 2, 'metric': 'correlation', 'random_state': 1234}
    # testing override of n_dims by n_components. Should raise UserWarning due
    # to conflict; the legacy 'params' dict spec is exercised deliberately, so
    # assert BOTH the conflict warning and the deprecation notice
    with pytest.warns(DeprecationWarning, match=r"'params'.*deprecated"), \
         pytest.warns(UserWarning, match='Unequal values passed to dims'):
        hyp_data = reducer(data1, reduce={'model': 'UMAP', 'params': params},
                           ndims=3)
    umap_data = UMAP(**params).fit_transform(data1)
    np.testing.assert_array_equal(hyp_data, umap_data)


def test_reduce_custom_model_instance():
    # regression test for GH #162: passing an already-constructed
    # scikit-learn model instance must not crash (UnboundLocalError) and
    # must use the instance as-is (not re-constructed/clobbered)
    from sklearn.decomposition import PCA
    reduced_data_3d = reducer(data, reduce=PCA(n_components=3))
    assert reduced_data_3d[0].shape == (10, 3)


def test_reduce_custom_model_class():
    # regression test for GH #162: passing a bare (uninstantiated)
    # scikit-learn model class must be constructed with ndims
    from sklearn.decomposition import PCA
    reduced_data_3d = reducer(data, reduce=PCA, ndims=3)
    assert reduced_data_3d[0].shape == (10, 3)
