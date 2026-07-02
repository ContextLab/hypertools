# -*- coding: utf-8 -*-

import numpy as np
import pytest

import hypertools as hyp
from hypertools.tools.apply_model import apply_model, supported_models


rng = np.random.default_rng(0)
data1 = np.cumsum(rng.standard_normal((60, 8)), axis=0)
data2 = np.cumsum(np.random.default_rng(1).standard_normal((60, 8)), axis=0)
clusters = np.vstack([rng.standard_normal((40, 5)) + 6 * i for i in range(2)])


def test_apply_model_string_reduce():
    result = apply_model(data1, 'PCA', ndims=3)
    assert result.shape == (60, 3)


def test_apply_model_list_input_stacked():
    result = apply_model([data1, data2], 'PCA', ndims=3)
    assert isinstance(result, list) and len(result) == 2
    assert all(r.shape == (60, 3) for r in result)
    # stacked fit -> results live in ONE shared embedding: refitting only
    # the first dataset gives different coordinates than the shared fit
    solo = apply_model(data1, 'PCA', ndims=3)
    assert not np.allclose(result[0], solo)


def test_apply_model_stack_false_fits_per_dataset():
    result = apply_model([data1, data2], 'PCA', ndims=3, stack=False)
    solo = apply_model(data1, 'PCA', ndims=3)
    assert np.allclose(result[0], solo)


def test_apply_model_dict_spec():
    result = apply_model(data1, {'model': 'PCA',
                                 'params': {'n_components': 2}})
    assert result.shape == (60, 2)


def test_apply_model_instance():
    from sklearn.decomposition import PCA
    result = apply_model(data1, PCA(n_components=4))
    assert result.shape == (60, 4)


def test_apply_model_cluster_labels():
    labels = apply_model(clusters, 'KMeans', mode='fit_predict',
                         ndims=None, format_data=True)
    # KMeans wants n_clusters, not n_components
    labels = apply_model(clusters, {'model': 'KMeans',
                                    'params': {'n_clusters': 2}},
                         mode='fit_predict')
    assert len(np.unique(labels)) == 2


def test_apply_model_mixture_proportions_auto():
    props = apply_model(clusters, {'model': 'GaussianMixture',
                                   'params': {'n_components': 2}})
    assert props.shape == (80, 2)
    assert np.allclose(props.sum(axis=1), 1)


def test_apply_model_pipeline():
    result, fitted = apply_model(
        [data1, data2],
        [{'model': 'PCA', 'params': {'n_components': 5}},
         {'model': 'PCA', 'params': {'n_components': 2}}],
        return_model=True)
    assert all(r.shape == (60, 2) for r in result)
    assert len(fitted) == 2


def test_apply_model_return_model_reusable():
    result, fitted = apply_model(data1, 'PCA', ndims=3, return_model=True)
    held_out = fitted.transform(np.asarray(data2, dtype=np.float64))
    assert held_out.shape == (60, 3)


def test_apply_model_unknown_string_raises():
    with pytest.raises(ValueError, match='unknown model'):
        apply_model(data1, 'NotARealModel')


def test_apply_model_umap_lazy():
    result = apply_model(clusters, 'UMAP', ndims=2)
    assert result.shape == (80, 2)


def test_apply_model_public_api():
    assert hyp.apply_model is apply_model
    assert 'PCA' in supported_models()
    assert 'GaussianMixture' in supported_models()
