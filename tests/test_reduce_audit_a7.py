# -*- coding: utf-8 -*-
"""Regression tests for the 2026-07 release-audit fixes to hypertools.reduce
and hypertools.describe (audit unit F11-reduce-describe, fix batch
A7-cluster-reduce).

Real data and real scikit-learn models throughout -- no mocks.
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hypertools.reduce.reduce import reduce as reducer
from hypertools.reduce.describe import describe


# --- F11-reduce-describe-001: describe(reduce='TSNE') at default max_dims ---

def test_describe_tsne_default_max_dims_runs():
    x = np.random.default_rng(3).normal(size=(50, 15))
    with pytest.warns(UserWarning, match='barnes_hut'):
        result = describe(x, reduce='TSNE', show=False)
    # barnes_hut TSNE supports n_components <= 3: dims 2 and 3 are evaluated
    assert len(result['average']) == 2
    assert all(len(t) == 2 for t in result['individual'])


# --- F11-reduce-describe-002: TSNE gets a workable perplexity on small data -

def test_reduce_tsne_small_dataset_clamps_perplexity():
    x = np.random.default_rng(2).normal(size=(30, 8))
    with pytest.warns(UserWarning, match='perplexity'):
        out = reducer(x, 'TSNE', ndims=2)
    assert np.asarray(out).shape == (30, 2)


def test_reduce_tsne_user_perplexity_not_overridden():
    x = np.random.default_rng(2).normal(size=(30, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = reducer(x, reduce={'model': 'TSNE',
                                 'kwargs': {'perplexity': 5}}, ndims=2)
    assert np.asarray(out).shape == (30, 2)
    assert not any('perplexity' in str(w.message) for w in caught)


# --- F11-reduce-describe-005: invalid spec errors name the offending value --

def test_unknown_reduce_string_names_value_and_suggests():
    x = np.random.default_rng(0).normal(size=(30, 8))
    with pytest.raises(ValueError) as err:
        reducer(x, 'umap', ndims=2)
    msg = str(err.value)
    assert "'umap'" in msg           # the offending value is named
    assert 'UMAP' in msg             # did-you-mean / supported list
    assert 'PCA' in msg              # supported models listed
    assert 'readthedocs' not in msg  # dead 0.8-docs link removed
    assert '  ' not in msg           # no run-on continuation whitespace


def test_invalid_reduce_object_names_value():
    x = np.random.default_rng(0).normal(size=(30, 8))
    with pytest.raises(ValueError, match='42'):
        reducer(x, 42, ndims=2)


# --- F11-reduce-describe-006: dict-spec error teaches the canonical form ----

def test_dict_spec_missing_model_error_recommends_canonical_form():
    x = np.random.default_rng(0).normal(size=(30, 8))
    with pytest.raises(ValueError) as err:
        reducer(x, {'kwargs': {'whiten': True}}, ndims=3)
    msg = str(err.value)
    assert "'model'" in msg and "'kwargs'" in msg
    assert '  ' not in msg  # old message had run-on continuation whitespace


# --- F11-reduce-describe-007: ragged lists get a hypertools error ------------

def test_ragged_list_raises_hypertools_error():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match='column'):
        reducer([rng.standard_normal((30, 8)),
                 rng.standard_normal((30, 5))], 'PCA', ndims=3)


# --- F11-reduce-describe-008: transformless fitted-Reducer reuse -------------

def test_fitted_tsne_reducer_reuse_raises_clear_error():
    rng = np.random.default_rng(2)
    _, fitted = reducer(rng.normal(size=(40, 8)), 'TSNE', ndims=2,
                        return_model=True, random_state=0)
    with pytest.raises(NotImplementedError, match='TSNE'):
        reducer(rng.normal(size=(5, 8)), reduce=fitted)


def test_fitted_mds_reducer_reuse_raises_clear_error():
    rng = np.random.default_rng(2)
    _, fitted = reducer(rng.normal(size=(40, 8)), 'MDS', ndims=2,
                        return_model=True, random_state=0)
    with pytest.raises(NotImplementedError, match='MDS'):
        reducer(rng.normal(size=(5, 8)), reduce=fitted)


# --- F11-reduce-describe-011: large-data warning only when data is large ----

def test_describe_small_data_no_large_warning():
    x = np.random.default_rng(0).normal(size=(20, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        describe(x, reduce='PCA', max_dims=5, show=False)
    assert not any('long time' in str(w.message) for w in caught)


def test_describe_large_data_warns():
    x = np.random.default_rng(0).normal(size=(1200, 6))
    with pytest.warns(UserWarning, match='long time'):
        describe(x, reduce='PCA', max_dims=3, show=False)


# --- F11-reduce-describe-012: multi-dataset matplotlib figure ----------------

def test_describe_matplotlib_multi_dataset_colors_legend_average():
    plt.close('all')
    rng = np.random.default_rng(3)
    data = [rng.normal(size=(30, 8)), rng.normal(size=(30, 8))]
    result = describe(data, reduce='PCA', max_dims=6, show=True,
                      backend='matplotlib')
    ax = result['fig'].axes[0]
    lines = ax.get_lines()
    # two individual traces plus the documented average overlay
    assert len(lines) >= 3
    colors = {tuple(np.round(matplotlib.colors.to_rgba(line.get_color()), 3))
              for line in lines}
    assert len(colors) >= 3  # per-dataset colors are distinguishable
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert 'average' in texts
    assert any('dataset' in t for t in texts)
    plt.close('all')


def test_describe_matplotlib_single_dataset_no_legend():
    plt.close('all')
    x = np.random.default_rng(0).normal(size=(30, 8))
    result = describe(x, reduce='PCA', max_dims=6, show=True,
                      backend='matplotlib')
    assert result['fig'].axes[0].get_legend() is None
    plt.close('all')


# --- F11-reduce-describe-015: describe returns its figure handle -------------

def test_describe_returns_fig_key():
    x = np.random.default_rng(0).normal(size=(30, 8))
    result = describe(x, reduce='PCA', max_dims=5, show=False)
    assert set(result.keys()) == {'average', 'individual', 'fig'}
    assert result['fig'] is None  # no figure drawn when show=False
    plt.close('all')
    shown = describe(x, reduce='PCA', max_dims=5, show=True,
                     backend='matplotlib')
    assert isinstance(shown['fig'], plt.Figure)
    plt.close('all')


# --- F11-reduce-describe-016: MDS defaults pinned (stable across sklearn) ---

def test_reduce_mds_no_future_warnings():
    x = np.random.default_rng(0).normal(size=(40, 10))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = reducer(x, 'MDS', ndims=2, random_state=0)
    assert np.asarray(out).shape == (40, 2)
    assert not any(issubclass(w.category, FutureWarning) for w in caught)


# --- binding contract: reduce=False means "skip stage, return input" --------

def test_reduce_false_returns_input_unchanged():
    x = np.random.default_rng(0).standard_normal((20, 6))
    out = reducer(x, reduce=False, ndims=3)
    assert np.array_equal(np.asarray(out), x)
    out2, model = reducer(x, reduce=False, ndims=3, return_model=True)
    assert model is None
    assert np.array_equal(np.asarray(out2), x)


# --- doc-accuracy fixes (F11-003 / F11-004 / F11-009 / F11-010) --------------

def test_reduce_docstring_list_return_claim_corrected():
    doc = reducer.__doc__
    # the blanket (false) claim is gone; single-dataset unwrapping documented
    assert 'If the input is a list, a list is returned.' not in doc
    assert 'single' in doc


def test_reduce_docstring_documents_skip_behavior():
    # ndims=None / ndims >= n_features returning the input unchanged is
    # documented (F11-reduce-describe-004)
    assert 'unchanged' in reducer.__doc__


def test_describe_docstring_distance_not_covariance():
    doc = describe.__doc__
    assert 'covariance' not in doc
    assert 'distance' in doc


def test_describe_docstring_max_dims_exclusive():
    doc = describe.__doc__
    assert ('max_dims - 1' in doc) or ('exclusive' in doc)
