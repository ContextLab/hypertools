"""Tests for the 1.0-pattern migration of hypertools.reduce (round17 Task 2):
the `Reducer` base class, mixture-model support (GH #174), cross-module
kwargs (GH #138), `return_model`, and full backward compatibility with
every pre-1.0 `reduce=` call form. All data is real (small) numeric arrays
-- no mocks.
"""
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

import hypertools as hyp
from hypertools.reduce.reduce import reduce as reducer
from hypertools.reduce.common import Reducer, REDUCERS, models, resolve_reducer
from hypertools.core.pipeline import Pipeline


def _rng():
    return np.random.RandomState(0)


def _data():
    return [_rng().rand(20, 6) for _ in range(2)]


# --- REDUCERS registry --------------------------------------------------

def test_reducers_registry_covers_models_dict_names():
    assert set(models).issubset(set(REDUCERS))


def test_reducers_registry_includes_mixture_models():
    for name in ('GaussianMixture', 'BayesianGaussianMixture',
                 'LatentDirichletAllocation', 'NMF'):
        assert name in REDUCERS


def test_resolve_reducer_umap_lazy_import():
    from umap import UMAP
    assert resolve_reducer('UMAP') is UMAP


def test_reduce_reduce_module_still_exposes_models_dict():
    # core.model._build_registry and core.pipeline._resolve_step_class both
    # import `from ..reduce.reduce import models` -- must keep working.
    from hypertools.reduce.reduce import models as reduce_models
    assert 'PCA' in reduce_models and 'IncrementalPCA' in reduce_models
    assert 'GaussianMixture' not in reduce_models  # unchanged, classic-only


# --- mixture models (#174) -----------------------------------------------

@pytest.mark.parametrize('name', ['GaussianMixture', 'BayesianGaussianMixture'])
def test_mixture_gaussian_returns_proportions_summing_to_one(name):
    x = np.abs(_rng().rand(30, 5))
    out = reducer(x, reduce=name, ndims=3)
    assert out.shape == (30, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert (out >= 0).all()


@pytest.mark.parametrize('name', ['LatentDirichletAllocation', 'NMF'])
def test_mixture_lda_nmf_returns_normalized_proportions(name):
    x = np.abs(_rng().rand(30, 5))  # LDA/NMF require non-negative data
    out = reducer(x, reduce=name, ndims=3)
    assert out.shape == (30, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_mixture_reuses_cluster_module_logic():
    # hyp.reduce(..., reduce='GaussianMixture', ...) must produce the exact
    # same style of output as hyp.cluster's mixture-model path (both call
    # hypertools.cluster.cluster.mixture_proportions).
    from hypertools.cluster.cluster import cluster as clusterer
    x = np.abs(_rng().rand(25, 4))
    reduced = reducer(x, reduce='GaussianMixture', ndims=2)
    clustered = clusterer(x, cluster='GaussianMixture', n_clusters=2)
    assert np.asarray(reduced).shape == np.asarray(clustered).shape


def test_mixture_list_input_stacks_and_splits():
    data = [np.abs(_rng().rand(10, 4)), np.abs(_rng().rand(15, 4))]
    out = reducer(data, reduce='NMF', ndims=2)
    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape == (10, 2)
    assert out[1].shape == (15, 2)


# --- return_model: single-stage returns a fitted Reducer -----------------

def test_return_model_single_stage_returns_reducer():
    x = _data()
    out, fitted = reducer(x, reduce='PCA', ndims=3, return_model=True)
    assert isinstance(fitted, Reducer)
    assert fitted.is_fitted
    assert np.asarray(out[0]).shape == (20, 3)


def test_return_model_reuse_no_refit(monkeypatch):
    x = _data()
    _, fitted = reducer(x, reduce='PCA', ndims=3, return_model=True)
    original_pca = fitted.model_

    def _boom(self, *args, **kwargs):
        raise AssertionError('fit_transform must not be called during transform-reuse')

    monkeypatch.setattr(PCA, 'fit_transform', _boom)

    new_x = [_rng().rand(12, 6)]
    out2 = reducer(new_x, reduce=fitted)
    assert np.asarray(out2).shape == (12, 3)
    assert fitted.model_ is original_pca


def test_return_model_mixture_wraps_reducer():
    x = np.abs(_rng().rand(20, 5))
    out, fitted = reducer(x, reduce='GaussianMixture', ndims=2, return_model=True)
    assert isinstance(fitted, Reducer)
    assert out.shape == (20, 2)


# --- cross-module kwargs (#138): multi-stage returns a fitted Pipeline ---

def test_cross_kwargs_manip_reduce_returns_pipeline():
    x = _data()
    out, fitted = reducer(x, reduce='PCA', ndims=3, manip='ZScore', return_model=True)
    assert isinstance(fitted, Pipeline)
    assert [name for name, _ in fitted.steps] == ['manip', 'reduce']
    assert np.asarray(out[0]).shape == (20, 3)


def test_cross_kwargs_canonical_order():
    x = _data()
    out, fitted = reducer(x, reduce='PCA', ndims=3, manip='ZScore',
                          normalize='within', cluster='KMeans', return_model=True)
    assert [name for name, _ in fitted.steps] == ['manip', 'normalize', 'reduce', 'cluster']
    assert len(out) == 40  # stacked across the two 20-row datasets for clustering


def test_cross_kwargs_align_only():
    x = [_rng().rand(15, 4), _rng().rand(15, 4)]
    out, fitted = reducer(x, reduce='PCA', ndims=3, align='HyperAlign', return_model=True)
    assert [name for name, _ in fitted.steps] == ['reduce', 'align']
    assert len(out) == 2


def test_no_cross_kwargs_matches_legacy_single_stage_call():
    # byte-identical behavior check: with only legacy kwargs, output must
    # match calling reduce() the old way (no manip/normalize/align/cluster).
    x = _data()
    out_legacy = reducer(x, reduce='PCA', ndims=3)
    out_new = reducer(x, reduce='PCA', ndims=3, manip=None, normalize=None,
                      align=None, cluster=None)
    for a, b in zip(out_legacy, out_new):
        np.testing.assert_array_equal(a, b)


# --- backward-compatible call forms ---------------------------------------

def test_legacy_params_dict_still_works_with_deprecation_warning():
    x = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = reducer(x, reduce={'model': 'PCA', 'params': {'whiten': True}}, ndims=3)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert np.asarray(out[0]).shape == (20, 3)


def test_canonical_dict_args_kwargs_form_no_warning():
    x = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = reducer(x, reduce={'model': 'PCA', 'args': [], 'kwargs': {'n_components': 3}})
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert np.asarray(out[0]).shape == (20, 3)


def test_bare_class_form():
    x = _data()
    out = reducer(x, reduce=PCA, ndims=3)
    assert np.asarray(out[0]).shape == (20, 3)


def test_instance_form():
    x = _data()
    out = reducer(x, reduce=PCA(n_components=3))
    assert np.asarray(out[0]).shape == (20, 3)


def test_string_form_default():
    x = _data()
    out = reducer(x, ndims=3)
    assert np.asarray(out[0]).shape == (20, 3)


def test_hyp_reduce_exposed_at_top_level():
    x = _data()
    out = hyp.reduce(x, reduce='PCA', ndims=2)
    assert np.asarray(out[0]).shape == (20, 2)


def test_fitted_reducer_reuse_warns_on_ndims_mismatch():
    x = _data()
    _, fitted = reducer(x, reduce='PCA', ndims=3, return_model=True)
    with pytest.warns(UserWarning, match='Unequal values passed to dims and n_components'):
        out = reducer([_rng().rand(12, 6)], reduce=fitted, ndims=5)
    # the already-fitted model wins: output keeps the fit-time dimensionality
    assert np.asarray(out).shape == (12, 3)


def test_fitted_reducer_reuse_matching_ndims_no_warning():
    x = _data()
    _, fitted = reducer(x, reduce='PCA', ndims=3, return_model=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        out = reducer([_rng().rand(12, 6)], reduce=fitted, ndims=3)
    assert np.asarray(out).shape == (12, 3)


def test_bare_mixture_class_duck_typing():
    from sklearn.mixture import GaussianMixture
    x = _data()
    out = reducer(x, reduce=GaussianMixture, ndims=2)
    assert len(out) == 2
    for piece in out:
        piece = np.asarray(piece)
        assert piece.shape == (20, 2)
        assert np.allclose(piece.sum(axis=1), 1.0)
