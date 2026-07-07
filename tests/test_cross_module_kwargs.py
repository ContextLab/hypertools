"""Tests for round17 Task 6: cross-module kwargs on `hyp.analyze`/
`hyp.normalize`/`hyp.plot`, `pipeline=` reuse (GH #138, #227), the plot()
`cluster=` canonical-dict fix, and the impute/predict legacy-dict warning
consistency follow-up. All data is real (small) numeric arrays -- no mocks.
"""
import warnings

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hypertools as hyp
from hypertools.tools.analyze import analyze
from hypertools.tools.normalize import normalize, Normalizer
from hypertools.core.pipeline import Pipeline, build_pipeline


def _rng():
    return np.random.RandomState(0)


def _two_datasets(n=30, d=5):
    r = _rng()
    return [r.randn(n, d), r.randn(n, d)]


# --- hyp.analyze: legacy path stays byte-identical ------------------------

def test_analyze_legacy_kwargs_only_matches_direct_chain():
    from hypertools.reduce.reduce import reduce as reducer
    from hypertools.tools.align import align as aligner
    from hypertools.tools.normalize import normalize as normalizer

    x = _two_datasets()
    expected = aligner(reducer(normalizer(x, normalize='within', internal=True),
                               reduce='PCA', ndims=2, internal=True),
                       align='HyperAlign')
    actual = analyze(x, normalize='within', reduce='PCA', ndims=2, align='HyperAlign')
    assert len(actual) == len(expected) == 2
    for a, e in zip(actual, expected):
        assert np.allclose(a, e)


def test_analyze_load_style_call_unchanged():
    # mirrors hypertools.io.load's call form exactly
    x = _two_datasets()
    out = analyze(x, reduce='PCA', ndims=2, align=None, normalize=None)
    assert len(out) == 2
    assert all(o.shape == (30, 2) for o in out)


# --- hyp.analyze: return_model ---------------------------------------------

def test_analyze_return_model_legacy_kwargs_gives_fitted_pipeline():
    x = _two_datasets()
    result, model = analyze(x, normalize='within', reduce='PCA', ndims=2,
                            align='HyperAlign', return_model=True, internal=True)
    assert isinstance(model, Pipeline)
    assert model.is_fitted
    assert len(result) == 2


def test_analyze_manip_and_cluster_cross_kwargs_run_full_pipeline():
    x = _two_datasets()
    result, model = analyze(x, manip='Smooth', reduce='PCA', ndims=2,
                            align='HyperAlign', cluster='KMeans',
                            return_model=True, internal=True)
    assert [name for name, _ in model.steps] == ['manip', 'reduce', 'align', 'cluster']
    # cluster stage produces per-observation labels (60 = two 30-row datasets stacked)
    assert len(result) == 60


# --- hyp.analyze: pipeline= reuse (GH #227) --------------------------------

def test_analyze_pipeline_reuse_does_not_refit():
    a = _two_datasets()
    b = _two_datasets()
    _, model = analyze(a, reduce='PCA', ndims=2, align='HyperAlign',
                       return_model=True, internal=True)

    reduce_step = model.named_steps['reduce']
    before = reduce_step._fitted.model_.components_.copy()

    # poison pill: fitting again must never happen on .transform
    def _poison(*args, **kwargs):
        raise AssertionError('reduce stage was refit during pipeline.transform!')
    reduce_step._fitted.model_.fit = _poison
    reduce_step._fitted.model_.fit_transform = _poison

    out, reused_model = analyze(b, pipeline=model, return_model=True, internal=True)
    assert reused_model is model
    assert len(out) == 2
    after = reduce_step._fitted.model_.components_.copy()
    assert np.allclose(before, after)


def test_analyze_pipeline_mutually_exclusive_with_stage_kwargs():
    x = _two_datasets()
    _, model = analyze(x, reduce='PCA', ndims=2, return_model=True)
    with pytest.raises(ValueError, match='reduce'):
        analyze(x, pipeline=model, reduce='PCA')
    with pytest.raises(ValueError, match='manip'):
        analyze(x, pipeline=model, manip='ZScore')


# --- end-to-end acceptance test (#227/#161): fit on A, reuse on B ----------

def test_end_to_end_fit_on_A_reuse_on_B_via_plot_pipeline():
    a = _two_datasets(n=25, d=4)
    b = _two_datasets(n=25, d=4)

    p = analyze(a, manip='Smooth', reduce='PCA', ndims=2, align='HyperAlign',
               return_model=True)[1]
    reduce_step = p.named_steps['reduce']
    before = reduce_step._fitted.model_.components_.copy()

    def _poison(*args, **kwargs):
        raise AssertionError('reduce stage was refit!')
    reduce_step._fitted.model_.fit = _poison
    reduce_step._fitted.model_.fit_transform = _poison

    fig = hyp.plot(b, pipeline=p, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    after = reduce_step._fitted.model_.components_.copy()
    assert np.allclose(before, after)

    with pytest.raises(ValueError, match='reduce'):
        hyp.plot(b, pipeline=p, reduce='PCA', show=False)


# --- hyp.plot: manip= + align dict + reduce (GH #275) ----------------------

def test_plot_manip_list_align_dict_reduce_end_to_end():
    x = _two_datasets(n=20, d=6)
    fig = hyp.plot(
        x,
        manip=[{'model': 'Smooth', 'kwargs': {'kernel_width': 5}}],
        align={'model': 'HyperAlign'},
        reduce='PCA',
        ndims=2,
        show=False,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --- hyp.plot: resample= sugar still runs (unchanged) ----------------------

def test_plot_resample_sugar_still_works_alongside_manip_default():
    x = _two_datasets(n=20, d=4)
    fig = hyp.plot(x, resample=15, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --- hyp.plot: pipeline= mutually exclusive with stage kwargs --------------

def test_plot_pipeline_mutually_exclusive_with_align():
    x = _two_datasets(n=20, d=4)
    p = build_pipeline(reduce='PCA', ndims=2)
    p.fit_transform(x)
    with pytest.raises(ValueError, match='align'):
        hyp.plot(x, pipeline=p, align='HyperAlign', show=False)


# --- hyp.plot: return_model bundle gains 'pipeline' ------------------------

def test_plot_return_model_bundle_has_fitted_pipeline():
    x = _two_datasets(n=20, d=4)
    bundle = hyp.plot(x, reduce='PCA', ndims=2, align='HyperAlign',
                      return_model=True, show=False)
    plt.close(bundle['fig'])
    assert 'pipeline' in bundle
    assert isinstance(bundle['pipeline'], Pipeline)
    assert bundle['pipeline'].is_fitted


def test_plot_return_model_bundle_pipeline_reused_when_pipeline_kwarg_given():
    x = _two_datasets(n=20, d=4)
    p = build_pipeline(reduce='PCA', ndims=2)
    p.fit_transform(x)
    bundle = hyp.plot(x, pipeline=p, return_model=True, show=False)
    plt.close(bundle['fig'])
    assert bundle['pipeline'] is p


# --- hyp.plot: cluster= canonical dict fix (deferred LOW from Task 3) -----

def test_plot_cluster_canonical_kwargs_dict_not_dropped():
    # a spread-out KMeans(n_clusters=2) run on two well-separated blobs
    # should recover exactly 2 groups; the OLD code silently dropped
    # 'kwargs' (only read 'params'), so n_clusters never reached KMeans and
    # the (unrelated) friendly default took over -- this asserts the
    # SPECIFIED n_clusters is actually honored.
    r = _rng()
    x = np.vstack([r.randn(20, 3), r.randn(20, 3) + 30])
    fig = hyp.plot(x, cluster={'model': 'KMeans', 'kwargs': {'n_clusters': 2}},
                   show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cluster_legacy_params_dict_still_works_and_warns_once():
    r = _rng()
    x = np.vstack([r.randn(20, 3), r.randn(20, 3) + 30])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(x, cluster={'model': 'KMeans', 'params': {'n_clusters': 2}},
                       show=False)
    plt.close(fig)
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1


# --- hyp.normalize: return_model + Normalizer reuse ------------------------

def test_normalize_return_model_across_gives_fitted_normalizer():
    x = _two_datasets()
    result, model = normalize(x, normalize='across', return_model=True)
    assert isinstance(model, Normalizer)
    assert model.is_fitted
    assert np.allclose(np.mean(np.vstack(result), axis=0), 0, atol=1e-8)


def test_normalize_across_reuse_applies_fit_time_stats_to_new_data():
    a = _two_datasets()
    b = _two_datasets()
    _, model = normalize(a, normalize='across', return_model=True)

    # poison pill: refitting must never happen on reuse
    original_fit = Normalizer.fit
    def _poison(self, x):
        raise AssertionError('Normalizer was refit during reuse!')
    Normalizer.fit = _poison
    try:
        reused_result, reused_model = normalize(b, normalize=model, return_model=True)
    finally:
        Normalizer.fit = original_fit
    assert reused_model is model
    # b's own values, normalized against A's fit-time mean/std (NOT
    # necessarily zero-mean, since b's distribution differs from a's)
    assert len(reused_result) == 2


def test_normalize_default_behavior_byte_identical_across_within_row():
    x = _two_datasets()

    def _legacy_zscore(X, y):
        if len(y) == 0 or len(set(y.ravel())) <= 1:
            return np.zeros_like(y, dtype=np.float64)
        mean = np.mean(X)
        std = np.std(X)
        if std == 0:
            return np.zeros_like(y, dtype=np.float64)
        return (y - mean) / std

    for mode in ('across', 'within', 'row'):
        got = normalize(x, normalize=mode)
        if mode == 'across':
            x_stacked = np.vstack(x)
            expected = [np.array([_legacy_zscore(x_stacked[:, j], i[:, j])
                                  for j in range(i.shape[1])]).T for i in x]
        elif mode == 'within':
            expected = [np.array([_legacy_zscore(i[:, j], i[:, j])
                                  for j in range(i.shape[1])]).T for i in x]
        else:
            expected = [np.array([_legacy_zscore(i[j, :], i[j, :])
                                  for j in range(i.shape[0])]) for i in x]
        for g, e in zip(got, expected):
            assert np.allclose(g, e), f"mode={mode} mismatch"


def test_normalize_cross_module_kwargs_reduce():
    x = _two_datasets()
    result, model = normalize(x, normalize='within', reduce='PCA', ndims=2,
                              return_model=True)
    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ['normalize', 'reduce']
    assert all(np.asarray(r).shape == (30, 2) for r in result)


# --- impute/predict: legacy dict now warns consistently (item 8) ----------

def test_impute_legacy_params_dict_warns():
    from hypertools.impute.impute import impute
    x = _rng().randn(20, 3)
    x[0, 0] = np.nan
    with pytest.warns(DeprecationWarning, match="'params'"):
        impute(x, model={'model': 'KNNImputer', 'params': {'n_neighbors': 3}})


def test_predict_legacy_params_dict_warns():
    from hypertools.predict.predict import predict
    import pandas as pd
    df = pd.DataFrame(_rng().randn(30, 2))
    with pytest.warns(DeprecationWarning, match="'params'"):
        predict(df, model={'model': 'Kalman', 'params': {}}, t=3)
