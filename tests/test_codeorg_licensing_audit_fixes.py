# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release-audit batch H2-codeorg-licensing.

Covers: the deduplicated Procrustes implementation (X7-006), the now-working
Procrustes ``index=`` parameter and its real (non-assert) errors (X7-025),
vendored-code license headers (X7-015), consolidated cross-module helpers
(X7-019), removal of the dead ``parse_args``/star-imports (X7-010), the
subpackage base-class exports and exception import migration (X7-021), and
the top-level ``__all__`` (X1-014).
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.align.procrustes import (Procrustes, fitter, procrustes,
                                         transformer)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rotation(rng, m):
    rot, _ = np.linalg.qr(rng.rand(m, m))
    if np.linalg.det(rot) < 0:
        rot[:, 0] *= -1
    return rot


# ---------------------------------------------------------------------------
# X7-006: single Procrustes implementation, no np.matrix deprecation warnings
# ---------------------------------------------------------------------------

def test_procrustes_emits_zero_warnings():
    # the old nested transform() used np.asmatrix/.A, so EVERY procrustes()
    # call emitted PendingDeprecationWarning twice; the deduplicated
    # implementation must be warning-free
    rng = np.random.RandomState(0)
    target = rng.rand(20, 3)
    source = target @ _rotation(rng, 3)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # ANY warning fails the test
        out = procrustes(source, target)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, target, atol=1e-6)


def test_procrustes_source_has_no_nested_duplicate_implementation():
    # the module must contain exactly ONE copy of the fitting algorithm
    # (module-level align()); the nested fit()/transform() duplicates and
    # their np.matrix usage are gone
    path = os.path.join(REPO_ROOT, 'hypertools', 'align', 'procrustes.py')
    with open(path) as f:
        src = f.read()
    assert src.count('np.linalg.svd') == 1
    assert 'asmatrix' not in src
    assert 'XXX' not in src


def test_procrustes_function_agrees_with_aligner_class():
    # the function and the Procrustes Aligner now share one implementation:
    # aligning [source] to an explicit target must reproduce the function
    rng = np.random.RandomState(1)
    target = rng.rand(25, 3)
    source = target @ _rotation(rng, 3)
    out_func = procrustes(source, target, format_data=False)
    # index=1 makes data[1] (the target) the default alignment target
    out_class = hyp.align([source, target], model=Procrustes(index=1))
    # dataset 1 IS the target (identity projection); dataset 0 maps onto it
    assert np.allclose(np.asarray(out_class[0]), out_func, atol=1e-6)
    assert np.allclose(np.asarray(out_class[1]), target, atol=1e-8)


# ---------------------------------------------------------------------------
# X7-025: index= selects the default alignment target (and real errors)
# ---------------------------------------------------------------------------

def test_procrustes_index_selects_default_target():
    rng = np.random.RandomState(2)
    data = [rng.rand(20, 3) for _ in range(3)]

    out1 = hyp.align(data, model=Procrustes(index=1))
    # the target dataset is mapped onto itself (identity projection)
    assert np.allclose(np.asarray(out1[1]), data[1], atol=1e-8)

    out0 = hyp.align(data, model=Procrustes(index=0))
    assert np.allclose(np.asarray(out0[0]), data[0], atol=1e-8)

    # index now has a real effect: different targets, different alignments
    assert not all(np.allclose(np.asarray(a), np.asarray(b))
                   for a, b in zip(out0, out1))

    # negative indices follow the usual Python convention
    out_neg = hyp.align(data, model=Procrustes(index=-1))
    out2 = hyp.align(data, model=Procrustes(index=2))
    assert all(np.allclose(np.asarray(a), np.asarray(b))
               for a, b in zip(out_neg, out2))


def test_procrustes_explicit_target_overrides_index():
    rng = np.random.RandomState(3)
    data = [rng.rand(20, 3) for _ in range(2)]
    target = pd.DataFrame(rng.rand(20, 3))
    out_a = hyp.align(data, model=Procrustes(target=target, index=0))
    out_b = hyp.align(data, model=Procrustes(target=target, index=1))
    assert all(np.allclose(np.asarray(a), np.asarray(b))
               for a, b in zip(out_a, out_b))


def test_procrustes_index_out_of_range_raises_indexerror():
    rng = np.random.RandomState(4)
    data = [rng.rand(20, 3) for _ in range(3)]
    with pytest.raises(IndexError, match='out of range'):
        hyp.align(data, model=Procrustes(index=7))
    with pytest.raises(IndexError, match='out of range'):
        hyp.align(data, model=Procrustes(index=-4))


def test_procrustes_index_non_integer_raises_typeerror():
    rng = np.random.RandomState(5)
    data = [rng.rand(20, 3) for _ in range(2)]
    for bad in ('a', 1.5, True):
        with pytest.raises(TypeError, match='index= must be an integer'):
            hyp.align(data, model=Procrustes(index=bad))


def test_procrustes_transformer_raises_real_errors_not_asserts():
    rng = np.random.RandomState(6)
    df = pd.DataFrame(rng.rand(10, 3))
    proj = np.eye(3)

    # unfitted: RuntimeError (was a bare assert)
    with pytest.raises(RuntimeError, match='fit model before transforming'):
        transformer(df, proj=None)

    # list-length mismatch: ValueError (was a bare assert)
    with pytest.raises(ValueError, match='fitted list of projections'):
        transformer([df, df.copy()], proj=[proj])

    # index outside the fitted projection list: a real IndexError (was
    # `assert ..., IndexError(...)`, which raised AssertionError with a
    # malformed f-string message)
    with pytest.raises(IndexError, match='outside the range'):
        transformer(df, proj=[proj, proj], index=5)

    # the happy paths still work
    out = transformer(df, proj=[proj, proj], index=1)
    assert np.allclose(np.asarray(out), df.to_numpy())
    outs = transformer([df, df.copy()], proj=[proj, proj])
    assert all(np.allclose(np.asarray(o), df.to_numpy()) for o in outs)


def test_procrustes_fitter_empty_list_keeps_index_for_lookups():
    fitted = fitter([], index=0)
    assert fitted['proj'] == []
    assert fitted['index'] == 0


# ---------------------------------------------------------------------------
# X7-015: vendored-code license headers
# ---------------------------------------------------------------------------

def test_brainiak_header_says_apache_2():
    path = os.path.join(REPO_ROOT, 'hypertools', 'external', 'brainiak.py')
    with open(path) as f:
        head = f.read(2000)
    assert 'Apache License, Version 2.0' in head
    assert 'Version 1.0' not in head


def test_ppca_header_carries_upstream_license_and_provenance():
    path = os.path.join(REPO_ROOT, 'hypertools', 'external', 'ppca.py')
    with open(path) as f:
        head = f.read(2000)
    assert 'pca-magic' in head
    assert 'github.com/allentran/pca-magic' in head
    assert 'Apache License, Version 2.0' in head
    assert 'Copyright 2015 Allen Tran' in head


# ---------------------------------------------------------------------------
# X7-019: cross-module helpers consolidated into core.shared
# ---------------------------------------------------------------------------

def test_as_dataframe_shared_between_predict_and_impute():
    from hypertools.core.shared import as_dataframe
    from hypertools.predict import common as predict_common
    from hypertools.impute import common as impute_common

    assert predict_common._as_dataframe is as_dataframe
    assert impute_common._as_dataframe is as_dataframe

    df = pd.DataFrame(np.arange(6.0).reshape(3, 2))
    assert as_dataframe(df) is df  # passthrough, no copy
    out = as_dataframe(np.arange(6.0).reshape(3, 2))
    assert isinstance(out, pd.DataFrame)
    assert np.allclose(out.to_numpy(), df.to_numpy())


def test_import_kalman_filter_shared_and_returns_real_class():
    from hypertools.core.shared import import_kalman_filter
    from pykalman import KalmanFilter

    assert import_kalman_filter('forecaster') is KalmanFilter
    assert import_kalman_filter('imputer') is KalmanFilter


def test_supported_names_shared_across_dispatchers():
    from hypertools.core.shared import supported_names
    from hypertools.predict.predict import FORECASTERS
    from hypertools.impute.impute import IMPUTERS

    assert supported_names(FORECASTERS) == [f.__name__ for f in FORECASTERS]
    assert 'Kalman' in supported_names(FORECASTERS)
    assert supported_names(IMPUTERS) == [m.__name__ for m in IMPUTERS]
    assert 'PPCA' in supported_names(IMPUTERS)
    # the dispatchers still name their models in unknown-model errors
    with pytest.raises(ValueError, match='PPCA'):
        hyp.impute(np.array([[1.0, np.nan], [2.0, 3.0]]), model='NopeModel')


# ---------------------------------------------------------------------------
# X7-010: parse_args removed; reduce/cluster no longer star-import helpers
# ---------------------------------------------------------------------------

def test_parse_args_removed_from_helpers():
    from hypertools._shared import helpers
    assert not hasattr(helpers, 'parse_args')
    # the production broadcaster survives
    assert hasattr(helpers, 'parse_kwargs')


def test_reduce_and_cluster_no_longer_leak_helper_names():
    from hypertools.reduce import reduce as reduce_mod
    from hypertools.cluster import cluster as cluster_mod
    for mod in (reduce_mod, cluster_mod):
        for name in ('parse_kwargs', 'vals2colors', 'group_by_category',
                     'interp_array', 'is_line'):
            assert not hasattr(mod, name), (mod.__name__, name)


# ---------------------------------------------------------------------------
# X7-021: base-class exports + canonical exception imports
# ---------------------------------------------------------------------------

def test_subpackages_export_their_base_classes():
    from hypertools.reduce import Reducer
    from hypertools.cluster import Clusterer
    from hypertools.align import Aligner
    from hypertools.reduce.common import Reducer as Reducer_impl
    from hypertools.cluster.common import Clusterer as Clusterer_impl
    assert Reducer is Reducer_impl
    assert Clusterer is Clusterer_impl
    assert Aligner is not None


def test_first_party_modules_import_exceptions_from_core():
    for rel in ('hypertools/io/sources.py', 'hypertools/io/load.py',
                'hypertools/io/lsl.py', 'hypertools/io/save.py',
                'hypertools/plot/backend.py'):
        with open(os.path.join(REPO_ROOT, rel)) as f:
            src = f.read()
        assert '_shared.exceptions' not in src, rel
    # the back-compat shim itself still works for external users
    from hypertools._shared.exceptions import HypertoolsError
    from hypertools.core.exceptions import HypertoolsError as canonical
    assert HypertoolsError is canonical


# ---------------------------------------------------------------------------
# X1-014: top-level __all__ defines the star-import surface exactly
# ---------------------------------------------------------------------------

def test_star_import_yields_exactly_all():
    ns = {}
    exec('from hypertools import *', ns)
    got = {k for k in ns if k != '__builtins__'}
    assert got == set(hyp.__all__)


def test_star_import_does_not_leak_internal_submodules():
    ns = {}
    exec('from hypertools import *', ns)
    for internal in ('config', 'core', 'datageometry', 'external', 'tools',
                     'plot', 'manip'):
        # the FUNCTIONS plot/manip are public; the SUBMODULE names config/
        # core/datageometry/external/tools must not leak
        if internal in ('plot', 'manip'):
            assert callable(ns[internal])
        else:
            assert internal not in ns, internal


def test_all_names_resolve_and_cover_documented_api():
    documented = {'plot', 'analyze', 'reduce', 'align', 'normalize',
                  'describe', 'cluster', 'manip', 'predict', 'impute',
                  'load', 'save', 'apply_model', 'supported_models',
                  'Pipeline', 'set_interactive_backend', 'HyperAnimation',
                  'FrameContext',
                  'io', 'HypertoolsError', 'HypertoolsBackendError',
                  'HypertoolsIOError',
                  # 1.1.0 (GH #285): the hand-written helpers folded in
                  'text_windows', 'damage', 'stack'}
    assert set(hyp.__all__) == documented
    for name in hyp.__all__:
        assert getattr(hyp, name) is not None
