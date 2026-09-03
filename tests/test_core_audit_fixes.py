"""Regression tests for the A6-core-packaging audit batch (2026-07).

Covers CONFIRMED findings from F23-core-config-exceptions and
F21-apply-model-pipeline plus the cross-unit numpy-error-state finding
(X7-code-org-rest-002). All tests use real data and real calls -- no mocks.
"""
import os
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

import hypertools as hyp
from hypertools.core.configurator import get_default_options, apply_defaults
from hypertools.core.exceptions import HypertoolsIOError
from hypertools.core.model import apply_model
from hypertools.core.pipeline import Pipeline
from hypertools.core.shared import RobustDict, get, unpack_model


rng = np.random.default_rng(1)
data1 = np.cumsum(rng.standard_normal((80, 8)), axis=0)
data2 = np.cumsum(np.random.default_rng(2).standard_normal((50, 8)), axis=0)


# --------------------------------------------------------------------------
# F23-001 / F23-002: config.ini mirrors the real dispatcher defaults, typed
# --------------------------------------------------------------------------

def test_config_defaults_match_dispatcher_signatures():
    import inspect
    opts = get_default_options()
    assert opts['reduce']['reduce'] == \
        inspect.signature(hyp.reduce).parameters['reduce'].default
    assert opts['reduce']['ndims'] == \
        inspect.signature(hyp.reduce).parameters['ndims'].default
    assert opts['cluster']['cluster'] == \
        inspect.signature(hyp.cluster).parameters['cluster'].default
    # hyp.cluster's signature default is a None sentinel that means 3
    # (hypertools/cluster/cluster.py: "n_clusters=None ... means 3"); the
    # published default records the EFFECTIVE value
    assert opts['cluster']['n_clusters'] == 3
    assert opts['align']['model'] == \
        inspect.signature(hyp.align).parameters['model'].default
    assert opts['plot']['fmt'] == \
        inspect.signature(hyp.plot).parameters['fmt'].default


def test_config_values_are_typed_python_values():
    merged = apply_defaults('cluster')
    assert merged['n_clusters'] == 3 and isinstance(merged['n_clusters'], int)
    opts = get_default_options()
    assert opts['reduce']['ndims'] is None


# --------------------------------------------------------------------------
# F23-003: custom fname layers on top of the shipped defaults, deep-merged;
# missing files fail loudly; no configparser 'DEFAULT' artifact
# --------------------------------------------------------------------------

def test_custom_config_layers_on_top_of_shipped_defaults(tmp_path):
    p = tmp_path / 'custom.ini'
    p.write_text('[cluster]\nn_clusters = 7\n')
    opts = get_default_options(fname=str(p))
    # the custom value wins ...
    assert opts['cluster']['n_clusters'] == 7
    # ... but the rest of the shipped section and other sections survive
    assert opts['cluster']['cluster'] == 'KMeans'
    assert opts['plot'] != {}
    assert opts['reduce']['reduce'] == 'IncrementalPCA'


def test_missing_custom_config_raises(tmp_path):
    missing = str(tmp_path / 'nope' / 'config.ini')
    with pytest.raises(HypertoolsIOError, match='config'):
        get_default_options(fname=missing)


def test_no_default_artifact_section():
    assert 'DEFAULT' not in get_default_options()


# --------------------------------------------------------------------------
# F23-004: RobustDict consistency (.get(), .copy(), no default aliasing)
# --------------------------------------------------------------------------

def test_robustdict_get_honors_default():
    rd = RobustDict({'a': 1}, __default_value__='DV')
    assert rd['zzz'] == 'DV'
    assert rd.get('zzz') == 'DV'
    assert rd.get('zzz', 'explicit') == 'explicit'
    assert rd.get('a') == 1


def test_robustdict_copy_preserves_robustness():
    rd = RobustDict({'a': 1}, __default_value__={})
    c = rd.copy()
    assert isinstance(c, RobustDict)
    assert c['missing'] == {}
    assert c['a'] == 1


def test_robustdict_mutable_default_not_aliased():
    rd = RobustDict({'a': 1}, __default_value__={})
    rd['missing1']['polluted'] = True
    assert rd['missing2'] == {}


# --------------------------------------------------------------------------
# F23-007 / F21-016(b): 'params' alongside 'args'/'kwargs' warns (and is
# dropped) instead of being silently ignored
# --------------------------------------------------------------------------

def test_unpack_model_params_plus_kwargs_warns_and_drops_params():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = unpack_model({'model': 'PCA', 'params': {'n_components': 4},
                            'kwargs': {'n_components': 9}}, valid=[PCA])
    assert any(issubclass(w.category, DeprecationWarning) and
               'params' in str(w.message) for w in caught)
    assert 'params' not in out
    assert out['kwargs'] == {'n_components': 9}


# --------------------------------------------------------------------------
# F23-008 / F21-007: bare classes pass through (parent_class=None) and are
# usable everywhere; error messages name the problem, not a memory address
# --------------------------------------------------------------------------

def test_unpack_model_bare_class_passthrough_without_parent_class():
    assert unpack_model(PCA) is PCA


def test_apply_model_accepts_bare_class():
    out = apply_model(data1, PCA, ndims=3)
    assert np.asarray(out).shape == (80, 3)


def test_apply_model_accepts_dict_with_class():
    out = apply_model(data1, {'model': PCA, 'kwargs': {'n_components': 3}})
    assert np.asarray(out).shape == (80, 3)


def test_pipeline_accepts_dict_with_class():
    out = Pipeline([{'model': PCA, 'kwargs': {'n_components': 3}}]) \
        .fit_transform(np.asarray(data1, dtype=np.float64))
    assert np.asarray(out).shape == (80, 3)


def test_unpack_model_error_names_types_not_addresses():
    from hypertools.align.common import Aligner

    class NotAModel:
        pass

    with pytest.raises(ValueError) as err:
        unpack_model(NotAModel(), valid=[], parent_class=Aligner)
    msg = str(err.value)
    assert '0x' not in msg
    assert 'NotAModel' in msg and 'Aligner' in msg


def test_unpack_model_wrong_class_with_parent_names_both_classes():
    from hypertools.align.common import Aligner
    with pytest.raises(ValueError) as err:
        unpack_model(PCA, valid=[], parent_class=Aligner)
    msg = str(err.value)
    assert 'PCA' in msg and 'Aligner' in msg


# --------------------------------------------------------------------------
# F23-012: shared.get() supports negative indices and warns on short lists
# --------------------------------------------------------------------------

def test_get_negative_index_is_python_conventional():
    assert get([10, 20, 30], -1) == 30
    assert get([10, 20, 30], -3) == 10


def test_get_out_of_range_warns_and_returns_value():
    with pytest.warns(UserWarning, match='no entry for dataset index'):
        assert get([10, 20, 30], 5) == [10, 20, 30]


# --------------------------------------------------------------------------
# F23-013: complete hypertools.core export surface
# --------------------------------------------------------------------------

def test_is_reused_pipeline_exported_from_core():
    from hypertools.core import is_reused_pipeline  # noqa: F401


# --------------------------------------------------------------------------
# F23-010: one default cluster count across entry points
# --------------------------------------------------------------------------

def test_default_cluster_count_consistent_across_entry_points():
    from hypertools._shared.params import parameters
    # hyp.cluster's effective default is 3 (its signature's None sentinel
    # means 3); the _shared/params defaults consumed by hyp.plot's cluster=
    # path must agree, so both entry points partition identically
    effective_default = 3
    assert parameters['KMeans']['n_clusters'] == effective_default
    assert parameters['GaussianMixture']['n_components'] == effective_default
    # functional check on real data: hyp.cluster's default yields 3 groups
    # on well-separated blobs
    blobs = np.vstack([np.random.default_rng(3).normal(6 * i, 0.2, size=(30, 4))
                       for i in range(6)])
    labels = hyp.cluster(blobs)
    assert len(np.unique(labels)) == effective_default


# --------------------------------------------------------------------------
# F21-001: dict-spec 'args' honored by apply_model
# --------------------------------------------------------------------------

def test_apply_model_honors_dict_args():
    out = apply_model(data1, {'model': 'PCA', 'args': [3]})
    assert np.asarray(out).shape == (80, 3)


def test_apply_model_args_on_instance_raises_clear_error():
    with pytest.raises(ValueError, match='args'):
        apply_model(data1, {'model': PCA(n_components=2), 'args': [3]})


# --------------------------------------------------------------------------
# F21-002: dispatcher-built (multi-stage return_model) Pipelines pickle
# --------------------------------------------------------------------------

def test_dispatcher_built_pipeline_pickles_and_reproduces_transform():
    # pickle here round-trips an object we just built in-process (the
    # documented hyp.save workflow) -- no untrusted data is ever loaded.
    train = np.cumsum(np.random.default_rng(4).standard_normal((100, 8)), axis=0)
    test = np.cumsum(np.random.default_rng(5).standard_normal((40, 8)), axis=0)
    red, bp = hyp.reduce(train, reduce='PCA', ndims=3, normalize='across',
                         return_model=True)
    expected = np.asarray(bp.transform(test))
    reloaded = pickle.loads(pickle.dumps(bp))
    got = np.asarray(reloaded.transform(test))
    assert got.shape == expected.shape
    assert np.allclose(got, expected)


# --------------------------------------------------------------------------
# F21-003: stack=False + list pipeline + return_model returns usable
# per-dataset fitted Pipelines
# --------------------------------------------------------------------------

def test_stack_false_list_pipeline_return_model_usable():
    res, fitted = apply_model(
        [data1, data2],
        [{'model': 'PCA', 'kwargs': {'n_components': 4}},
         {'model': 'PCA', 'kwargs': {'n_components': 2}}],
        stack=False, return_model=True)
    assert isinstance(fitted, list) and len(fitted) == 2
    for pipe, d, r in zip(fitted, [data1, data2], res):
        assert isinstance(pipe, Pipeline) and pipe.is_fitted
        out = np.asarray(pipe.transform(np.asarray(d, dtype=np.float64)))
        assert out.shape == np.asarray(r).shape
        assert np.allclose(out, np.asarray(r))


# --------------------------------------------------------------------------
# F21-004: mode= applies to the FINAL stage of a list pipeline; unsupported
# modes raise a clear ValueError instead of a raw AttributeError
# --------------------------------------------------------------------------

def test_list_pipeline_mode_fit_predict_returns_final_stage_labels():
    labels = apply_model([data1, data1],
                         ['PCA', {'model': 'KMeans',
                                  'kwargs': {'n_clusters': 2,
                                             'random_state': 0,
                                             'n_init': 10}}],
                         mode='fit_predict')
    assert isinstance(labels, list) and len(labels) == 2
    assert all(np.asarray(lab).shape == (80,) for lab in labels)
    assert len(np.unique(np.concatenate([np.asarray(lab) for lab in labels]))) == 2


def test_mode_mismatch_raises_clear_value_error():
    with pytest.raises(ValueError, match="PCA.*predict_proba|predict_proba.*PCA"):
        apply_model(data1, 'PCA', mode='predict_proba')
    with pytest.raises(ValueError, match="DBSCAN.*fit_transform|fit_transform.*DBSCAN"):
        apply_model(data1, 'DBSCAN', mode='fit_transform')


# --------------------------------------------------------------------------
# F21-008: Pipeline.transform warns when its only option is a re-fit
# --------------------------------------------------------------------------

def test_pipeline_transform_warns_on_fit_predict_refit():
    train = np.cumsum(np.random.default_rng(6).standard_normal((120, 8)), axis=0)
    test = np.cumsum(np.random.default_rng(7).standard_normal((50, 8)), axis=0)
    p = Pipeline(['DBSCAN'])
    p.fit_transform(train)
    with pytest.warns(UserWarning, match='fit_predict'):
        out = p.transform(test)
    assert np.asarray(out).shape == (50,)


# --------------------------------------------------------------------------
# F21-009: models with no out-of-sample path (TSNE) raise a clear error on
# .transform instead of a raw AttributeError
# --------------------------------------------------------------------------

def test_pipeline_transform_tsne_raises_clear_error():
    res, p = apply_model(
        data1,
        [{'model': 'PCA', 'kwargs': {'n_components': 5}},
         {'model': 'TSNE', 'kwargs': {'n_components': 2,
                                      'perplexity': 10.0}}],
        return_model=True)
    with pytest.raises(TypeError, match='TSNE'):
        p.transform(np.asarray(data2, dtype=np.float64))


# --------------------------------------------------------------------------
# F21-010: Pipeline constructor validates its input up front
# --------------------------------------------------------------------------

def test_pipeline_rejects_string_steps_argument():
    with pytest.raises(TypeError, match='list of step'):
        Pipeline('PCA')


def test_pipeline_rejects_wrong_arity_tuple_step():
    with pytest.raises(TypeError, match='tuple'):
        Pipeline([('a', 'PCA', 'extra')])


# --------------------------------------------------------------------------
# F21-012: clear error when a raw-estimator step is fed a list of datasets
# --------------------------------------------------------------------------

def test_pipeline_list_input_to_raw_step_raises_clear_error():
    x = np.cumsum(np.random.default_rng(8).standard_normal((90, 6)), axis=0)
    y = np.cumsum(np.random.default_rng(9).standard_normal((40, 6)), axis=0)
    pipe = Pipeline([{'model': 'PCA', 'kwargs': {'n_components': 3}}])
    with pytest.raises(TypeError, match='apply_model'):
        pipe.fit_transform([x, y])


def test_pipeline_aligner_step_ndarray_list_hint():
    x = np.random.default_rng(10).standard_normal((40, 6))
    y = np.random.default_rng(11).standard_normal((40, 6))
    pipe = Pipeline(['HyperAlign'])
    with pytest.raises(TypeError, match='DataFrame'):
        pipe.fit_transform([x, y])


# --------------------------------------------------------------------------
# F21-016(a): dict specs without a 'model' key fail fast with guidance
# --------------------------------------------------------------------------

def test_apply_model_dict_missing_model_key_clear_error():
    with pytest.raises(ValueError, match="'model' key"):
        apply_model(data1, {'mode': 'PCA'})


# --------------------------------------------------------------------------
# X7-code-org-rest-002: importing hypertools must not change numpy's
# process-wide error state
# --------------------------------------------------------------------------

def test_import_hypertools_leaves_numpy_err_state_unchanged():
    code = (
        "import warnings, numpy as np\n"
        "before = np.geterr()\n"
        "import hypertools\n"
        "after = np.geterr()\n"
        "assert before == after, f'np.geterr changed: {before} -> {after}'\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    np.array([1.0]) / np.array([0.0])\n"
        "assert any(issubclass(x.category, RuntimeWarning) for x in w), \\\n"
        "    'divide-by-zero RuntimeWarning was silenced'\n"
        "print('OK')\n"
    )
    env = dict(os.environ, MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, '-c', code], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert 'OK' in result.stdout


def test_helpers_scale_constant_data_no_runtime_warning():
    from hypertools._shared.helpers import scale
    const = [np.ones((10, 3))]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = scale(const)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert out[0].shape == (10, 3)
