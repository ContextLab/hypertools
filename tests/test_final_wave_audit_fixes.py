# noinspection PyPackageRequirements
"""Regression tests for the FINAL fix wave of the 2026-07 release-1.0 audit.

One test (or small group) per item from the adversarial re-audit:

1.  align with duplicate index labels must return SAME-length datasets
2.  minimal canonical dict spec {'model': ...} works in every dispatcher
3.  cluster dict specs honor their 'args' key
4.  cluster dict spec wrapping an INSTANCE warns about dropped args/kwargs
5.  reduce warns (like cluster) when 'params' rides along with 'kwargs'
6.  describe() gives ragged lists a curated error
7.  manip(model=False/None) skips, like align/cluster/reduce
8.  manip asserts are real ValueErrors (survive python -O)
9.  None input raises ONE unified TypeError everywhere; analyze(None)/
    analyze([]) raise instead of silently returning the input
10. empty input uses ONE unified 'no observations' phrasing everywhere
11. Pipeline rejects duplicate step names
12. AutoRegressor validates lags (0 / negative / float / bool)
13. impute distinguishes empty datasets from all-NaN ones
14. predict raises for a datetime horizon BEFORE the first observation
15. tuples of datasets are accepted exactly like lists everywhere
16. Smooth raises the same clear NaN error for all three kernels
"""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.tools.analyze import analyze


def _rng():
    return np.random.default_rng(17)


def _two(n=20, k=4):
    rng = _rng()
    return [rng.standard_normal((n, k)), rng.standard_normal((n, k))]


# --- item 1: duplicate index labels no longer misalign -----------------------

def test_align_duplicate_datetimeindex_labels_same_lengths():
    rng = _rng()
    dup_idx = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-02',
                                '2024-01-03', '2024-01-04'])
    a = pd.DataFrame(rng.standard_normal((5, 3)), index=dup_idx)
    b = pd.DataFrame(rng.standard_normal((5, 3)),
                     index=pd.date_range('2024-01-01', periods=5))
    # two expected warnings from one call: the duplicated row-index notice
    # AND the row-trim notice (4 unique labels < 5 rows), so record both
    # explicitly instead of pytest.warns (which would re-emit the unmatched
    # one as a leaked warning)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = hyp.align([a, b])
    msgs = [str(w.message) for w in rec]
    assert any('duplicated row-index' in m for m in msgs), msgs
    assert any('common to all datasets' in m for m in msgs), msgs
    # 4 unique common labels; EVERY dataset must come back with exactly them
    assert [o.shape for o in out] == [(4, 3), (4, 3)]


def test_align_duplicate_labels_keeps_first_occurrence():
    rng = _rng()
    dup_idx = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-02',
                                '2024-01-03'])
    a = pd.DataFrame(rng.standard_normal((4, 2)), index=dup_idx)
    b = pd.DataFrame(rng.standard_normal((4, 2)),
                     index=pd.date_range('2024-01-01', periods=4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = hyp.align([a, b], model='NullAlign')
    # NullAlign returns trimmed data unchanged: row for the duplicated label
    # must be its FIRST occurrence (a.iloc[1], not a.iloc[2])
    assert np.allclose(out[0][1], a.iloc[1].to_numpy())


# --- item 2: minimal canonical dict spec everywhere ---------------------------

def _dispatch_minimal_dict(name):
    rng = _rng()
    x = rng.standard_normal((20, 5))
    if name == 'manip':
        return hyp.manip(x, model={'model': 'ZScore'})
    if name == 'align':
        y = x @ np.linalg.qr(rng.standard_normal((5, 5)))[0]
        return hyp.align([x, y], model={'model': 'HyperAlign'})
    if name == 'cluster':
        return hyp.cluster(x, cluster={'model': 'KMeans'})
    if name == 'reduce':
        return hyp.reduce(x, reduce={'model': 'PCA'}, ndims=2)
    if name == 'analyze':
        return analyze(x, reduce={'model': 'PCA'}, ndims=2)
    if name == 'plot':
        import matplotlib.pyplot as plt
        fig = hyp.plot(x, reduce={'model': 'PCA'}, ndims=3, show=False)
        plt.close('all')
        return fig
    raise ValueError(name)


@pytest.mark.parametrize('dispatcher', ['manip', 'align', 'cluster', 'reduce',
                                        'analyze', 'plot'])
def test_minimal_canonical_dict_spec_works_everywhere(dispatcher):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _dispatch_minimal_dict(dispatcher)
    assert result is not None


def test_minimal_dict_reduce_actually_reduces():
    x = _rng().standard_normal((20, 5))
    out = hyp.reduce(x, reduce={'model': 'PCA'}, ndims=2)
    assert np.asarray(out).shape == (20, 2)


# --- item 3: cluster dict 'args' honored --------------------------------------

def test_cluster_dict_args_honored():
    rng = _rng()
    x = np.vstack([rng.standard_normal((10, 3)) + 10.0 * i for i in range(5)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels = hyp.cluster(x, cluster={'model': 'KMeans', 'args': [5]},
                             random_state=0)
    assert len(set(labels)) == 5  # asked for 5 via args -- silently got 3 before


def test_cluster_dict_args_conflict_with_explicit_n_clusters_warns_spec_wins():
    rng = _rng()
    x = np.vstack([rng.standard_normal((10, 3)) + 10.0 * i for i in range(5)])
    with pytest.warns(UserWarning, match='conflicts'):
        labels = hyp.cluster(x, cluster={'model': 'KMeans', 'args': [5],
                                         'kwargs': {'n_init': 10}},
                             n_clusters=7, random_state=0)
    assert len(set(labels)) == 5  # documented rule: the spec's value wins


def test_cluster_dict_kwargs_win_over_args_with_warning():
    rng = _rng()
    x = np.vstack([rng.standard_normal((10, 3)) + 10.0 * i for i in range(4)])
    with pytest.warns(UserWarning, match="using the 'kwargs' value"):
        labels = hyp.cluster(x, cluster={'model': 'KMeans', 'args': [5],
                                         'kwargs': {'n_clusters': 4,
                                                    'n_init': 10}},
                             random_state=0)
    assert len(set(labels)) == 4


def test_cluster_dict_too_many_args_clear_error():
    x = _rng().standard_normal((20, 3))
    with pytest.raises(TypeError, match="positional argument"):
        hyp.cluster(x, cluster={'model': 'KMeans',
                                'args': list(range(30))})


# --- item 4: instance inside a cluster dict spec warns about dropped params ---

def test_cluster_dict_instance_with_kwargs_warns():
    from sklearn.cluster import KMeans
    rng = _rng()
    x = np.vstack([rng.standard_normal((10, 3)),
                   rng.standard_normal((10, 3)) + 10.0])
    with pytest.warns(UserWarning, match='used as-is'):
        labels = hyp.cluster(x, cluster={'model': KMeans(n_clusters=2,
                                                         n_init=10,
                                                         random_state=0),
                                         'kwargs': {'n_clusters': 5}})
    assert len(set(labels)) == 2  # the instance's own setting wins


# --- item 5: reduce warns like cluster when params rides along ---------------

def test_reduce_params_alongside_kwargs_warns_like_cluster():
    rng = _rng()
    x = rng.standard_normal((20, 5))

    with pytest.warns(UserWarning) as w_reduce:
        hyp.reduce(x, reduce={'model': 'PCA', 'kwargs': {},
                              'params': {'whiten': True}}, ndims=2)
    reduce_msgs = [str(w.message) for w in w_reduce
                   if "ignoring 'params'" in str(w.message)]
    assert reduce_msgs, 'reduce did not warn about the ignored params key'

    with pytest.warns(UserWarning) as w_cluster:
        hyp.cluster(x, cluster={'model': 'KMeans', 'kwargs': {'n_init': 10},
                                'params': {'max_iter': 5}}, random_state=0)
    cluster_msgs = [str(w.message) for w in w_cluster
                    if "ignoring 'params'" in str(w.message)]
    assert cluster_msgs, 'cluster did not warn about the ignored params key'

    # byte-identical wording, modulo the module name
    assert reduce_msgs[0] == cluster_msgs[0].replace('cluster spec',
                                                     'reduce spec')


# --- item 6: describe() ragged columns get a curated error -------------------

def test_describe_ragged_columns_clear_error():
    rng = _rng()
    with pytest.raises(ValueError, match='different numbers of columns'):
        hyp.describe([rng.standard_normal((10, 3)),
                      rng.standard_normal((10, 5))], show=False)


# --- item 7: manip False/None skip contract ----------------------------------

@pytest.mark.parametrize('spec', [False, None])
def test_manip_false_and_none_skip(spec):
    x = _rng().standard_normal((10, 3))
    out = hyp.manip(x, model=spec)
    assert out is x  # unchanged, like reduce(reduce=None)/cluster(cluster=None)


def test_manip_false_skip_with_return_model():
    x = _rng().standard_normal((10, 3))
    out, model = hyp.manip(x, model=False, return_model=True)
    assert out is x and model is None


def test_manip_false_model_with_downstream_stage_still_runs_stage():
    x = _rng().standard_normal((20, 5))
    out = hyp.manip(x, model=False, reduce='PCA', ndims=2)
    assert np.asarray(out).shape == (20, 2)


def test_manip_stage_kwargs_accept_false_as_skip():
    x = _rng().standard_normal((10, 3))
    out = hyp.manip(x, model='ZScore', normalize=False, reduce=False,
                    align=False, cluster=False)
    assert np.allclose(np.asarray(out).mean(axis=0), 0.0)


# --- item 8: real raises in manip (no asserts) --------------------------------

def test_manipulator_fit_none_raises_valueerror_not_assertionerror():
    from hypertools.manip.common import Manipulator
    with pytest.raises(ValueError, match='no observations'):
        Manipulator().fit(None)


def test_manip_transformers_missing_axis_raise_valueerror():
    from hypertools.manip import zscore, normalize as mnormalize, resample
    df = pd.DataFrame(np.arange(12, dtype=float).reshape(4, 3))
    with pytest.raises(ValueError, match='axis='):
        zscore.transformer(df, mean=df.mean(), std=df.std())
    with pytest.raises(ValueError, match='axis='):
        mnormalize.transformer(df, baseline=df.min(), peak=df.max(),
                               min=0, max=1)
    with pytest.raises(ValueError, match='axis='):
        resample.transformer(df, n_samples=10)


def test_manip_invalid_axis_messages_name_kwarg_and_value():
    x = _rng().standard_normal((10, 3))
    with pytest.raises(ValueError, match=r'invalid Smooth axis 2'):
        hyp.manip(x, model='Smooth', axis=2, kernel_width=5)
    with pytest.raises(ValueError, match='axis must be either 0 or 1'):
        hyp.manip(x, model='ZScore', axis=2)


def test_no_assert_statements_left_in_manip_package():
    import ast
    import importlib
    import pathlib
    manip_pkg = importlib.import_module('hypertools.manip')
    pkg_dir = pathlib.Path(manip_pkg.__file__).parent
    offenders = []
    for path in pkg_dir.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f'{path.name}:{node.lineno}')
    assert not offenders, f'assert statements remain in manip/: {offenders}'


# --- item 9: unified None-input handling --------------------------------------

@pytest.mark.parametrize('name,call', [
    ('manip', lambda: hyp.manip(None)),
    ('align', lambda: hyp.align(None)),
    ('cluster', lambda: hyp.cluster(None)),
    ('reduce', lambda: hyp.reduce(None)),
    ('predict', lambda: hyp.predict(None, t=2)),
    ('impute', lambda: hyp.impute(None)),
    ('analyze', lambda: analyze(None)),
    ('describe', lambda: hyp.describe(None, show=False)),
])
def test_none_input_unified_typeerror(name, call):
    with pytest.raises(TypeError) as err:
        call()
    assert f'Unsupported data type passed to {name}: None' in str(err.value)


def test_analyze_empty_list_raises_even_with_no_stage_kwargs():
    # analyze([]) with NO stage kwargs used to silently return []
    with pytest.raises(ValueError, match='no observations'):
        analyze([])
    with pytest.raises(TypeError, match='Unsupported data type'):
        analyze(None)


# --- item 10: unified empty-input phrasing -------------------------------------

@pytest.mark.parametrize('call', [
    lambda: hyp.manip([], model='ZScore'),
    lambda: hyp.align([]),
    lambda: hyp.impute([]),
    lambda: hyp.predict([], model='GaussianProcess', t=2),
    lambda: analyze([]),
])
def test_empty_list_unified_no_observations_phrasing(call):
    with pytest.raises(ValueError) as err:
        call()
    assert re.search(r'input has no observations \(.*\); there is nothing to',
                     str(err.value))


def test_aligner_fit_empty_uses_unified_phrasing():
    from hypertools.align import HyperAlign
    with pytest.raises(ValueError, match='no observations'):
        HyperAlign().fit(None)
    with pytest.raises(ValueError, match='no observations'):
        HyperAlign().fit([])


def test_forecaster_fit_empty_uses_unified_phrasing():
    with pytest.raises(ValueError, match='no observations'):
        hyp.predict(pd.DataFrame(np.empty((0, 2))), model='GaussianProcess',
                    t=2)


# --- item 11: Pipeline rejects duplicate step names ----------------------------

def test_pipeline_duplicate_step_names_raise():
    with pytest.raises(ValueError, match='must be unique'):
        hyp.Pipeline([('a', 'ZScore'), ('a', 'PCA')])


def test_pipeline_auto_names_still_deduplicate():
    pipe = hyp.Pipeline(['ZScore', 'ZScore'])
    names = [name for name, _ in pipe.steps]
    assert len(names) == len(set(names)) == 2


# --- item 12: AutoRegressor lags validation ------------------------------------

@pytest.mark.parametrize('lags', [0, -1, 2.5, True])
def test_autoregressor_invalid_lags_clear_error(lags):
    x = np.cumsum(_rng().standard_normal((30, 2)), axis=0)
    with pytest.raises(ValueError, match='lags must be a positive integer'):
        hyp.predict(x, model='AutoRegressor', t=3, lags=lags)


def test_autoregressor_valid_lags_still_fit():
    x = np.cumsum(_rng().standard_normal((30, 2)), axis=0)
    out = hyp.predict(x, model='AutoRegressor', t=3, lags=5)
    assert out.shape == (3, 2)


# --- item 13: impute distinguishes empty from all-NaN --------------------------

def test_impute_empty_dataset_in_list_not_reported_as_all_nan():
    with pytest.raises(ValueError) as err:
        hyp.impute([pd.DataFrame()])
    assert 'no observations' in str(err.value)
    assert 'entirely missing' not in str(err.value)


def test_impute_mixed_list_with_empty_dataset_clear_error():
    rng = _rng()
    good = pd.DataFrame(rng.standard_normal((5, 2)))
    with pytest.raises(ValueError, match=r'dataset 1 has shape \(0, 2\)'):
        hyp.impute([good, pd.DataFrame(np.empty((0, 2)))])


def test_impute_all_nan_still_reported_as_entirely_missing():
    with pytest.raises(ValueError, match='entirely missing'):
        hyp.impute(np.full((10, 3), np.nan))


# --- item 14: datetime horizon before the first observation --------------------

# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_predict_datetime_before_first_observation_raises():
    df = pd.DataFrame({'a': np.arange(30.0)},
                      index=pd.date_range('2024-06-01', periods=30))
    with pytest.raises(ValueError, match='before the first observation'):
        hyp.predict(df, model='GaussianProcess', t=pd.Timestamp('2024-01-01'))


# upstream: sklearn GP pins the noise-level kernel bound on tiny fixtures
@pytest.mark.filterwarnings(
    'ignore:The optimal value found for dimension 0 of parameter'
    ':sklearn.exceptions.ConvergenceWarning')
def test_predict_datetime_truncation_and_forecast_still_work():
    df = pd.DataFrame({'a': np.arange(30.0)},
                      index=pd.date_range('2024-06-01', periods=30))
    truncated = hyp.predict(df, model='GaussianProcess',
                            t=pd.Timestamp('2024-06-10'))
    assert truncated.shape == (10, 1)  # truncation contract unchanged
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forecast = hyp.predict(df, model='GaussianProcess',
                               t=pd.Timestamp('2024-07-03'))
    assert forecast.shape[0] == 3


# --- item 15: tuples accepted like lists everywhere ----------------------------

def _dispatch_tuple(name):
    rng = _rng()
    pair = (rng.standard_normal((12, 3)), rng.standard_normal((12, 3)))
    if name == 'manip':
        return hyp.manip(tuple(pd.DataFrame(p) for p in pair))
    if name == 'align':
        return hyp.align(pair)
    if name == 'cluster':
        return hyp.cluster(pair)
    if name == 'reduce':
        return hyp.reduce(pair, ndims=2)
    if name == 'predict':
        return hyp.predict((np.cumsum(rng.standard_normal((20, 2)), axis=0),),
                           model='GaussianProcess', t=2)
    if name == 'impute':
        return hyp.impute((rng.standard_normal((10, 3)),))
    if name == 'analyze':
        return analyze(pair, normalize='within')
    if name == 'describe':
        return hyp.describe(pair, show=False, max_dims=3)
    raise ValueError(name)


@pytest.mark.parametrize('dispatcher', ['manip', 'align', 'cluster', 'reduce',
                                        'predict', 'impute', 'analyze',
                                        'describe'])
def test_tuple_of_datasets_accepted_like_list(dispatcher):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = _dispatch_tuple(dispatcher)
    assert result is not None
    if dispatcher in ('manip', 'align', 'reduce', 'analyze', 'predict',
                      'impute'):
        assert isinstance(result, list)


def test_tuple_and_list_give_identical_align_output():
    rng = _rng()
    a, b = rng.standard_normal((15, 3)), rng.standard_normal((15, 3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from_list = hyp.align([a, b], model='NullAlign')
        from_tuple = hyp.align((a, b), model='NullAlign')
    for u, v in zip(from_list, from_tuple):
        assert np.allclose(u, v)


# --- item 16: Smooth raises the same clear NaN error for every kernel ----------

@pytest.mark.parametrize('kernel', ['savgol', 'gaussian', 'boxcar'])
def test_smooth_nan_raises_clear_error_all_kernels(kernel):
    x = _rng().standard_normal((30, 2))
    x[5, 0] = np.nan
    with pytest.raises(ValueError, match=r'cannot smooth data containing '
                                         r'NaN.*hyp\.impute'):
        hyp.manip(x, model='Smooth', kernel=kernel, kernel_width=5)


def test_smooth_nan_free_data_still_smooths():
    x = _rng().standard_normal((30, 2))
    out = hyp.manip(x, model='Smooth', kernel_width=5)
    assert out.shape == (30, 2)
    assert not np.isnan(out.to_numpy()).any()
