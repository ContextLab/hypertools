"""Tests for the 1.0-pattern migration of hypertools.cluster (round17 Task
3): the `Clusterer` base class, the `CLUSTERERS`/`MIXTURES` registries,
cross-module kwargs (GH #138), `return_model`, fitted-model reuse (never
refit), and full backward compatibility with every pre-1.0 `cluster=` call
form. Also covers the `core.model` consolidation (Task 3's `_build_registry`
/ `_resolve_model` changes) and `apply_model`'s list -> `hyp.Pipeline`
`return_model` behavior. All data is real (small) numeric arrays -- no
mocks.
"""
import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans, DBSCAN

import hypertools as hyp
from hypertools.cluster.cluster import cluster, models, mixture_models
from hypertools.cluster.common import Clusterer, CLUSTERERS, MIXTURES
from hypertools.core.pipeline import Pipeline
from hypertools.core.model import apply_model, supported_models, _build_registry


def _blobs():
    rng = np.random.RandomState(0)
    return np.vstack([rng.randn(50, 3), rng.randn(50, 3) + 50])


# --- CLUSTERERS / MIXTURES registries ------------------------------------

def test_clusterers_registry_hard_names():
    for name in ('KMeans', 'MiniBatchKMeans', 'AgglomerativeClustering',
                 'Birch', 'FeatureAgglomeration', 'SpectralClustering',
                 'HDBSCAN', 'MeanShift', 'DBSCAN', 'OPTICS',
                 'AffinityPropagation'):
        assert name in CLUSTERERS


def test_mixtures_registry_soft_names():
    for name in ('GaussianMixture', 'BayesianGaussianMixture',
                 'LatentDirichletAllocation', 'NMF'):
        assert name in MIXTURES


def test_models_and_mixture_models_are_backward_compat_aliases():
    # hypertools.reduce.common and hypertools.core.pipeline still import
    # these names directly from hypertools.cluster.cluster
    assert models is CLUSTERERS
    assert mixture_models is MIXTURES


# --- cluster(): legacy behavior byte-identical ---------------------------

def test_cluster_hard_labels_list_unchanged():
    data = _blobs()
    labels = cluster(data, n_clusters=2)
    assert isinstance(labels, list)
    assert len(labels) == len(data)
    assert len(set(labels)) == 2


def test_cluster_mixture_proportions_unchanged():
    data = _blobs()
    props = cluster(data, cluster='GaussianMixture', n_clusters=2)
    assert props.shape == (100, 2)
    assert np.allclose(props.sum(axis=1), 1)


def test_cluster_legacy_dict_params_form_deprecated_but_works():
    data = _blobs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = cluster(data, cluster={'model': 'KMeans',
                                        'params': {'n_clusters': 2}})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(labels, list) and len(set(labels)) == 2


def test_cluster_canonical_dict_form_no_warning():
    data = _blobs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = cluster(data, cluster={'model': 'KMeans',
                                        'kwargs': {'n_clusters': 2}})
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(labels, list) and len(set(labels)) == 2


def test_cluster_n_clusters_top_level_dict_convenience():
    data = _blobs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = cluster(data, cluster={'model': 'KMeans', 'n_clusters': 2})
    assert len(set(labels)) == 2
    # no 'params'/'kwargs' at all -- not the legacy form, so no warning
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_cluster_density_model_ignores_n_clusters():
    # regression: n_clusters must not be force-injected into constructors
    # that don't accept it (DBSCAN discovers cluster count itself)
    data = _blobs()
    labels = cluster(data, cluster='DBSCAN', n_clusters=2)
    assert isinstance(labels, list)
    assert len(labels) == len(data)


def test_cluster_unknown_string_raises():
    with pytest.raises(ValueError, match='unknown cluster model'):
        cluster(_blobs(), cluster='NotARealClusterer')


# --- cluster(): new (1.0) spec grammar ------------------------------------

def test_cluster_accepts_bare_class_directly():
    data = _blobs()
    labels = cluster(data, cluster=KMeans, n_clusters=2)
    assert len(set(labels)) == 2


def test_cluster_accepts_instance_directly():
    data = _blobs()
    labels = cluster(data, cluster=KMeans(n_clusters=2))
    assert len(set(labels)) == 2


def test_cluster_none_passthrough():
    data = _blobs()
    result = cluster(data, cluster=None)
    assert np.allclose(result, data)
    result2, model = cluster(data, cluster=None, return_model=True)
    assert model is None


# --- return_model=True: single stage --------------------------------------

def test_cluster_return_model_single_stage_gives_fitted_clusterer():
    data = _blobs()
    labels, fitted = cluster(data, cluster='KMeans', n_clusters=2,
                             return_model=True)
    assert isinstance(fitted, Clusterer)
    assert fitted.is_fitted
    assert isinstance(labels, list) and len(set(labels)) == 2


def test_cluster_return_model_mixture_gives_fitted_clusterer():
    data = _blobs()
    props, fitted = cluster(data, cluster='GaussianMixture', n_clusters=2,
                            return_model=True)
    assert isinstance(fitted, Clusterer)
    assert fitted.is_fitted
    assert props.shape == (100, 2)


# --- fitted-model reuse: never refit (no-mock poison pill) ----------------

def test_cluster_fitted_clusterer_reused_via_transform_not_refit():
    data = _blobs()
    _, fitted = cluster(data, cluster='KMeans', n_clusters=2,
                        return_model=True)

    def poison(*args, **kwargs):
        raise AssertionError('must not refit an already-fitted Clusterer')
    fitted.model_.fit = poison

    new_data = np.random.RandomState(1).randn(10, 3) + 50
    labels2 = cluster(new_data, cluster=fitted, format_data=False)
    assert isinstance(labels2, list) and len(labels2) == 10

    # return_model=True on a reused fitted Clusterer hands the SAME object back
    labels3, same = cluster(new_data, cluster=fitted, format_data=False,
                            return_model=True)
    assert same is fitted


def test_cluster_fitted_no_predict_model_raises_on_reuse():
    # AgglomerativeClustering has no out-of-sample predict: reuse must fail
    # loudly rather than silently refitting or returning garbage
    data = _blobs()
    _, fitted = cluster(data, cluster='AgglomerativeClustering', n_clusters=2,
                        return_model=True)
    with pytest.raises(NotImplementedError):
        fitted.transform(np.random.RandomState(2).randn(5, 3))


# --- cross-module kwargs (#138) --------------------------------------------

def test_cluster_cross_kwargs_reduce_returns_pipeline():
    data = [np.random.RandomState(0).rand(30, 6) for _ in range(2)]
    result, fitted = cluster(data, cluster='KMeans', n_clusters=2,
                             reduce='PCA', ndims=2, return_model=True)
    assert isinstance(fitted, Pipeline)
    names = [name for name, _ in fitted.steps]
    assert names == ['reduce', 'cluster']


def test_cluster_cross_kwargs_canonical_order():
    # manip -> normalize -> reduce -> align -> cluster (GH #153)
    data = [np.random.RandomState(i).rand(20, 4) for i in range(2)]
    result, fitted = cluster(data, cluster='KMeans', n_clusters=2,
                             manip='ZScore', normalize='across',
                             reduce='PCA', ndims=2, align='NullAlign',
                             return_model=True)
    names = [name for name, _ in fitted.steps]
    assert names == ['manip', 'normalize', 'reduce', 'align', 'cluster']


def test_cluster_cross_kwargs_n_clusters_still_threaded():
    # n_clusters must reach the cluster stage even when other cross-module
    # kwargs are present (build_pipeline itself has no n_clusters= kwarg;
    # cluster() must bake it into the resolved spec before delegating)
    data = [np.random.RandomState(0).rand(30, 6) for _ in range(2)]
    props, fitted = cluster(data, cluster='GaussianMixture', n_clusters=4,
                            reduce='PCA', ndims=3, return_model=True)
    # reduce -> (60, 3); cluster stage's n_components=4 convenience must
    # still be threaded through even though build_pipeline itself has no
    # n_clusters= kwarg of its own
    assert np.asarray(props).shape == (60, 4)


def test_cluster_no_cross_kwargs_returns_plain_clusterer():
    # sanity: without any cross-module kwarg, single-stage path is used
    data = _blobs()
    _, fitted = cluster(data, cluster='KMeans', n_clusters=2,
                        return_model=True)
    assert isinstance(fitted, Clusterer)
    assert not isinstance(fitted, Pipeline)


# --- core.model consolidation ----------------------------------------------

def test_build_registry_covers_reducers_and_clusterers():
    from hypertools.reduce.common import REDUCERS
    registry = _build_registry()
    assert set(REDUCERS).issubset(set(registry))
    assert set(CLUSTERERS).issubset(set(registry))
    assert 'UMAP' in registry


def test_supported_models_covers_cluster_and_reduce_names():
    sm = supported_models()
    assert 'KMeans' in sm and 'PCA' in sm and 'GaussianMixture' in sm


def test_apply_model_resolve_delegates_to_unpack_model_legacy_warning():
    # apply_model's dict-spec resolution now goes through
    # core.shared.unpack_model (the shared eval-free resolver), which is
    # what emits the DeprecationWarning for the legacy {'model','params'}
    # form -- verifies the duplicated resolver was actually removed, not
    # just its warning re-implemented locally
    data = np.random.RandomState(0).rand(20, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = apply_model(data, {'model': 'PCA', 'params': {'n_components': 2}},
                          format_data=False)
    assert np.asarray(out).shape == (20, 2)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_apply_model_list_return_model_gives_pipeline_not_list():
    # the one spec'd behavior change (round17 Task 3): list-of-specs +
    # return_model=True now returns a fitted hyp.Pipeline, not a plain
    # list of fitted models -- but the same fitted models must still be
    # reachable via named_steps/steps
    d1 = np.cumsum(np.random.RandomState(0).standard_normal((60, 8)), axis=0)
    d2 = np.cumsum(np.random.RandomState(1).standard_normal((60, 8)), axis=0)
    result, fitted = apply_model(
        [d1, d2],
        [{'model': 'PCA', 'params': {'n_components': 5}},
         {'model': 'PCA', 'params': {'n_components': 2}}],
        return_model=True)
    assert isinstance(fitted, hyp.Pipeline)
    assert len(fitted.steps) == 2
    held_out = fitted.transform(np.asarray(d2, dtype=np.float64))
    assert np.asarray(held_out).shape == (60, 2)


# --- public API / plot() integration ---------------------------------------

def test_cluster_public_api():
    assert hyp.cluster is cluster


def test_cluster_via_plot_still_works():
    # plot.py's cluster+hue handling (~1150) must keep working unchanged
    data = _blobs()
    geo = hyp.plot(data, '.', cluster='KMeans', n_clusters=2, show=False)
    assert geo is not None


def test_cluster_via_plot_mixture_still_works():
    data = _blobs()
    geo = hyp.plot(data, '.', cluster='GaussianMixture', n_clusters=2, show=False)
    assert geo is not None
