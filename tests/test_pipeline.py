"""Tests for hypertools.core.pipeline: unified spec resolution, Pipeline,
and build_pipeline. All data is real (small) numeric arrays/DataFrames --
no mocks.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

import hypertools as hyp
from hypertools.core.pipeline import Pipeline, build_pipeline, CANONICAL_ORDER
from hypertools.core.shared import unpack_model


def _rng():
    return np.random.RandomState(0)


# --- Pipeline export -------------------------------------------------

def test_pipeline_exported_from_top_level():
    assert hyp.Pipeline is Pipeline


# --- chain fit/transform round-trips ----------------------------------

def test_chain_fit_transform_roundtrip_matches_fit_then_transform():
    x = _rng().randn(20, 5)
    pipe = Pipeline([('scale', PCA(n_components=3))])
    out_fit_transform = pipe.fit_transform(x)
    assert out_fit_transform.shape == (20, 3)

    pipe.fit(x)
    out_via_fit_then_transform = pipe.transform(x)
    assert np.allclose(out_fit_transform, out_via_fit_then_transform)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_multi_step_chain_threads_output_to_next_input():
    x = _rng().randn(20, 5)
    pipe = Pipeline(['ZScore', PCA(n_components=2)])
    out = pipe.fit_transform(x)
    assert np.asarray(out).shape == (20, 2)
    assert [name for name, _ in pipe.steps] == ['zscore', 'pca']


def test_transform_before_fit_raises_not_fitted():
    pipe = Pipeline([PCA(n_components=2)])
    with pytest.raises(NotFittedError):
        pipe.transform(_rng().randn(10, 4))


# --- refit-vs-reuse distinction ----------------------------------------

def test_refit_vs_reuse_distinction_for_kmeans():
    # KMeans with n_init=1 and no fixed random_state: independent fits on
    # the same data can land on different cluster-label permutations, but
    # reusing a FITTED model via transform-equivalent predict is stable.
    # We exercise this through fit_transform (refit) vs fit+transform
    # (reuse) using PCA, whose fitted components are deterministic given
    # the same data, so two INDEPENDENT refits are identical only because
    # the input is unchanged -- the real distinction is that `transform`
    # on new data reuses the ORIGINAL fit rather than re-fitting to it.
    x = np.random.RandomState(0).randn(30, 6)
    y = np.random.RandomState(1).randn(30, 6)

    pipe = Pipeline([('pca', PCA(n_components=2))])
    pipe.fit(x)
    reused = pipe.transform(y)

    # a fresh pipeline refit directly on y sees y's own variance structure
    fresh = Pipeline([('pca', PCA(n_components=2))])
    refitted = fresh.fit_transform(y)

    # both are valid 2D projections of y, but they come from different
    # fitted components (fit on x vs fit on y), so they generally disagree
    assert reused.shape == refitted.shape == (30, 2)
    assert not np.allclose(reused, refitted)

    # applying the SAME fitted pipeline to the SAME data it was fit on is
    # reproducible (fit_transform == fit().transform() on the same input)
    again = pipe.transform(x)
    assert np.allclose(pipe.fit_transform(x), again)


def test_named_steps_and_is_fitted_flag():
    pipe = Pipeline([('pca', PCA(n_components=2))])
    assert pipe.is_fitted is False
    assert pipe.named_steps['pca'].__class__ is PCA
    pipe.fit(_rng().randn(10, 4))
    assert pipe.is_fitted is True


# --- inverse_transform through PCA --------------------------------------

def test_inverse_transform_through_pca_reconstructs_reduced_space():
    x = _rng().randn(25, 4)
    pipe = Pipeline([('pca', PCA(n_components=4))])  # no info loss
    out = pipe.fit_transform(x)
    back = pipe.inverse_transform(out)
    assert np.allclose(back, x, atol=1e-8)


# upstream: datawrangler calls pd.concat(copy=...), deprecated in pandas 4.
# the filter names the DeprecationWarning BASE class on purpose:
# pandas.errors.Pandas4Warning subclasses it but does not exist on
# pandas 2.x, and pytest aborts (UsageError, exit 4) on filter
# categories it cannot import
@pytest.mark.filterwarnings(
    'ignore:The copy keyword is deprecated:DeprecationWarning')
def test_inverse_transform_succeeds_through_invertible_zscore():
    # ZScore is invertible (QC P1-1): a Pipeline can inverse_transform
    # through a leading ZScore step (it used to raise). PCA at full rank is
    # lossless, so the round-trip recovers the original data.
    x = _rng().randn(15, 3)
    pipe = Pipeline(['ZScore', ('pca', PCA(n_components=3))])
    out = pipe.fit_transform(x)
    rec = np.asarray(pipe.inverse_transform(out))
    assert rec.shape == (15, 3) and np.allclose(rec, x, atol=1e-6)


def test_inverse_transform_raises_through_lossy_step():
    # a genuinely lossy manipulator (Smooth) is NOT invertible, so
    # inverse_transform through it still raises a clear NotImplementedError.
    x = _rng().randn(15, 3)
    pipe = Pipeline(['Smooth', ('pca', PCA(n_components=2))])
    pipe.fit_transform(x)
    with pytest.raises(NotImplementedError, match='not invertible'):
        pipe.inverse_transform(pipe.fit_transform(x))


# --- align shape validation error --------------------------------------

def test_align_step_transform_reuses_fit_on_matching_shape():
    d1 = pd.DataFrame(_rng().rand(12, 3))
    d2 = pd.DataFrame(_rng().rand(12, 3))
    d3 = pd.DataFrame(_rng().rand(12, 3))

    pipe = Pipeline(['HyperAlign'])
    pipe.fit([d1, d2])
    out = pipe.transform([d1, d3])
    assert len(out) == 2
    assert np.asarray(out[0]).shape == (12, 3)


def test_align_step_transform_raises_on_dataset_count_mismatch():
    d1 = pd.DataFrame(_rng().rand(10, 3))
    d2 = pd.DataFrame(_rng().rand(10, 3))

    pipe = Pipeline(['HyperAlign'])
    pipe.fit([d1, d2])
    with pytest.raises(ValueError, match=r'2 dataset'):
        pipe.transform([d1])


def test_align_step_transform_raises_on_column_count_mismatch():
    d1 = pd.DataFrame(_rng().rand(10, 3))
    d2 = pd.DataFrame(_rng().rand(10, 3))
    d_bad = pd.DataFrame(_rng().rand(10, 5))

    pipe = Pipeline(['HyperAlign'])
    pipe.fit([d1, d2])
    with pytest.raises(ValueError, match=r'column'):
        pipe.transform([d1, d_bad])


# --- legacy params deprecation warning fired exactly once ---------------

def test_legacy_params_dict_warns_exactly_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = unpack_model({'model': 'PCA', 'params': {'n_components': 2}}, valid=[PCA])

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
    assert out == {'model': PCA, 'args': [], 'kwargs': {'n_components': 2}}


def test_legacy_params_dict_step_builds_correctly_configured_model():
    x = _rng().randn(15, 5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipe = Pipeline([{'model': 'PCA', 'params': {'n_components': 2}}])
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    out = pipe.fit_transform(x)
    assert out.shape == (15, 2)


def test_canonical_dict_spec_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        unpack_model({'model': 'PCA', 'args': [], 'kwargs': {'n_components': 2}}, valid=[PCA])
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


# --- unpack_model: fitted instances pass through unchanged -------------

def test_unpack_model_passes_through_fitted_instance_unchanged():
    fitted = PCA(n_components=2).fit(_rng().randn(10, 4))
    assert unpack_model(fitted, valid=[], parent_class=None) is fitted


def test_unpack_model_wrong_type_instance_with_parent_class_raises():
    from sklearn.cluster import KMeans
    from hypertools.align.common import Aligner
    with pytest.raises(ValueError, match='unknown model'):
        unpack_model(KMeans(), valid=[], parent_class=Aligner)


def test_pipeline_still_accepts_fitted_pca_instance_as_step():
    x = _rng().randn(20, 5)
    fitted = PCA(n_components=3).fit(x)
    pipe = Pipeline([fitted])
    assert pipe.named_steps['pca'] is fitted
    out = pipe.fit_transform(x)
    assert out.shape == (20, 3)


# --- auto-naming ---------------------------------------------------------

def test_auto_naming_suffixes_on_collision():
    pipe = Pipeline(['HyperAlign', 'HyperAlign', 'HyperAlign'])
    names = [name for name, _ in pipe.steps]
    assert names == ['hyperalign', 'hyperalign-1', 'hyperalign-2']


def test_explicit_and_auto_names_can_mix_without_collision():
    pipe = Pipeline([('first', PCA(n_components=2)), PCA(n_components=2)])
    names = [name for name, _ in pipe.steps]
    assert names == ['first', 'pca']


def test_repr_lists_steps():
    pipe = Pipeline([('pca', PCA(n_components=2))])
    r = repr(pipe)
    assert r.startswith('Pipeline([') and 'pca=' in r


# --- build_pipeline --------------------------------------------------

def test_build_pipeline_assembles_canonical_order():
    pipe = build_pipeline(manip='ZScore', normalize='within', reduce='PCA',
                          ndims=2, align='HyperAlign', cluster='KMeans')
    assert [name for name, _ in pipe.steps] == list(CANONICAL_ORDER)


def test_build_pipeline_skips_none_stages():
    pipe = build_pipeline(reduce='PCA', ndims=2)
    assert [name for name, _ in pipe.steps] == ['reduce']


def test_build_pipeline_manip_normalize_reduce_cluster_end_to_end():
    x = [_rng().randn(20, 6), _rng().randn(20, 6)]
    pipe = build_pipeline(manip='ZScore', normalize='within', reduce='PCA',
                          ndims=3, cluster='KMeans')
    labels = pipe.fit_transform(x)
    assert len(labels) == 40  # two 20-row datasets stacked for clustering


def test_build_pipeline_align_stage_returns_list_of_aligned_datasets():
    x = [_rng().randn(15, 4), _rng().randn(15, 4)]
    pipe = build_pipeline(align='HyperAlign')
    out = pipe.fit_transform(x)
    assert len(out) == 2
    assert np.asarray(out[0]).shape == (15, 4)


def test_build_pipeline_respects_custom_order():
    pipe = build_pipeline(reduce='PCA', ndims=2, cluster='KMeans',
                          order=('cluster', 'reduce'))
    assert [name for name, _ in pipe.steps] == ['cluster', 'reduce']
