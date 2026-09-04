"""Reduce-dispatch correctness (QC 2026-07 release hardening).

- The canonical dict spec {'model':...,'kwargs':{...}} must honor ndims (it
  silently returned full-dimensional data).
- reduce() must return a bare array for a single-array input regardless of ndims
  (it flipped between ndarray and list).
- ndims must be validated (non-int / <1 gave cryptic errors or wrong output).
- describe(max_dims > n_features) must not crash; describe(max_dims < 3, show=True)
  must warn+skip instead of crashing seaborn.

Real data, sklearn cross-check, no mocks.
"""
import contextlib

import numpy as np
import pytest
from sklearn.decomposition import PCA

import hypertools as hyp
from hypertools.reduce.reduce import reduce as reducer


def test_canonical_dict_spec_honors_ndims_and_kwargs():
    x = np.random.default_rng(0).normal(size=(120, 6))
    out = np.asarray(reducer(x, reduce={'model': 'PCA', 'kwargs': {'whiten': True}},
                             ndims=2))
    assert out.shape == (120, 2)  # was (120, 6) -- ndims ignored
    manual = PCA(whiten=True, n_components=2).fit_transform(x)
    assert np.allclose(np.abs(out), np.abs(manual), atol=1e-6)  # whiten applied


def test_user_supplied_n_components_in_kwargs_is_not_overridden():
    x = np.random.default_rng(0).normal(size=(120, 6))
    out = np.asarray(reducer(x, reduce={'model': 'PCA', 'kwargs': {'n_components': 3}}))
    assert out.shape == (120, 3)


@pytest.mark.parametrize('ndims', [3, None, 20])
def test_single_array_returns_bare_array_for_any_ndims(ndims):
    x = np.random.default_rng(0).normal(size=(40, 6))
    if ndims == 20:
        # ndims beyond the 6 features deliberately provokes the
        # no-reduction-performed notice
        ctx = pytest.warns(UserWarning, match='no reduction was performed')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        out = reducer(x, reduce='PCA', ndims=ndims)
    assert not isinstance(out, list)
    assert np.asarray(out).ndim == 2


def test_list_input_still_returns_list():
    x = np.random.default_rng(0).normal(size=(30, 5))
    out = reducer([x, x + 1], reduce='PCA', ndims=3)
    assert isinstance(out, list) and len(out) == 2


@pytest.mark.parametrize('bad', ['two', 0, -1, 2.5, True])
def test_invalid_ndims_raises_clear_error(bad):
    x = np.random.default_rng(0).normal(size=(30, 5))
    with pytest.raises(ValueError, match='ndims'):
        reducer(x, reduce='PCA', ndims=bad)


def test_describe_max_dims_gt_features_does_not_crash():
    x = np.random.default_rng(0).normal(size=(30, 4))
    # max_dims beyond the 4 features deliberately provokes the
    # capped-evaluation notice
    with pytest.warns(UserWarning, match='exceeds the data dimensionality'):
        result = hyp.describe(x, reduce='PCA', max_dims=8, show=False)
    # 'fig' added by the 2026-07 release audit (F11-reduce-describe-015)
    assert set(result.keys()) == {'average', 'individual', 'fig'}


def test_describe_empty_component_range_raises_clear_error():
    # H1 polish wave (X2-error-quality-017): a max_dims that leaves NO
    # component range (range(2, max_dims) empty) used to silently return
    # empty results and merely warn at figure time; it now fails fast with
    # a ValueError naming the kwarg and its accepted domain.
    import matplotlib
    matplotlib.use('Agg')
    x = np.random.default_rng(0).normal(size=(30, 4))
    with pytest.raises(ValueError, match='max_dims must be an integer >= 3'):
        hyp.describe(x, reduce='PCA', max_dims=2, show=True)
