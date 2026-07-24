# -*- coding: utf-8 -*-
"""Regression tests for GH #209: `type(x) is T` / `type(x) is not T` object-type
checks must be replaced with `isinstance(x, T)` so that subclasses of builtin
types (list, dict, ...) are handled correctly instead of crashing or silently
mis-dispatching.

The canonical repro (from the issue audit) is a `list` subclass passed through
`hypertools.align.common.trim_and_pad`: with a brittle `type(data) is not list`
check, the subclass fails the identity check, gets double-wrapped in another
list, and then crashes trying to call `.index.values` on the subclass itself
instead of on a DataFrame.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from collections import OrderedDict

from hypertools.align.common import trim_and_pad


class ListSubclass(list):
    """Trivial list subclass with identical content/behavior to `list`."""
    pass


def test_trim_and_pad_accepts_list_subclass():
    """CL_209.md repro: a list subclass must be recognized as a list by
    trim_and_pad, not double-wrapped into `[MyList(...)]`."""
    df1 = pd.DataFrame(np.random.RandomState(0).randn(5, 3))
    df2 = pd.DataFrame(np.random.RandomState(1).randn(5, 3))
    plain = [df1, df2]
    sub = ListSubclass([df1, df2])

    plain_result = trim_and_pad(plain)
    assert len(plain_result) == 2

    # Before the fix: `type(data) is not list` is True for the subclass,
    # so `data = [data]` wraps the whole ListSubclass in another list, and
    # `.loc[rows]`/`.index.values` is then attempted on the ListSubclass
    # itself instead of on a DataFrame -> AttributeError.
    sub_result = trim_and_pad(sub)
    assert len(sub_result) == 2
    for a, b in zip(plain_result, sub_result):
        assert np.allclose(a.values, b.values)


def test_align_hyper_accepts_list_subclass_of_arrays():
    """hyp.align (public API) on a list-subclass of arrays must align
    correctly, exercising the isinstance conversions in align/common.py,
    align/hyperalign.py, and align/procrustes.py along the way."""
    import hypertools as hyp

    rng = np.random.RandomState(0)
    a1 = rng.randn(20, 3)
    rot = np.array([[-0.89433495, -0.44719485, -0.01348182],
                    [-0.43426149, 0.87492975, -0.21427761],
                    [-0.10761949, 0.18578133, 0.97667976]])
    a2 = a1 @ rot

    sub = ListSubclass([a1, a2])
    # one call deliberately provokes TWO deprecation notices (legacy align=
    # kwarg AND the 'hyper' alias), so record and assert both explicitly
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        result = hyp.align(sub, align='hyper')
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, DeprecationWarning)]
    assert any('align= is deprecated' in m for m in msgs), msgs
    assert any("'hyper' is a deprecated alias" in m for m in msgs), msgs
    assert len(result) == 2
    assert np.allclose(result[0], result[1], rtol=1)


def test_cluster_accepts_ordereddict_spec():
    """hyp.cluster's `cluster=` argument accepts a dict-subclass spec
    (cluster/cluster.py:106 used to require `type(cluster) is dict`, which
    rejects OrderedDict even though it's a fully-conforming Mapping)."""
    import hypertools as hyp

    data = np.random.RandomState(0).randn(50, 5)
    spec = OrderedDict([('model', 'KMeans'), ('params', {'n_clusters': 3})])
    # the 'params' dict form is the deliberately-exercised legacy spec;
    # assert its deprecation notice fires
    with pytest.warns(DeprecationWarning, match=r"'params'.*deprecated"):
        labels = hyp.cluster(data, cluster=spec)
    assert len(labels) == 50
    assert len(set(labels)) == 3


def test_describe_accepts_list_subclass():
    """hyp.describe's internal `summary()` helper (reduce/describe.py:62)
    must stack a list-subclass input via np.vstack rather than skipping the
    stack step and crashing on `.shape` for a bare list."""
    import hypertools as hyp

    rng = np.random.RandomState(0)
    sub = ListSubclass([rng.randn(20, 5), rng.randn(20, 5)])
    out = hyp.describe(sub, format_data=False, show=False)
    assert 'average' in out and 'individual' in out
    assert len(out['individual']) == 2


def test_resample_accepts_list_subclass_of_dataframes():
    """manip/resample.py's fitter/transformer (`type(data) is list`) must
    treat a list-subclass of per-dataset DataFrames the same as a plain
    list, resampling each DataFrame independently."""
    from hypertools.manip.resample import Resample

    df1 = pd.DataFrame({'x': np.linspace(0, 1, 50), 'y': np.linspace(1, 2, 50)})
    df2 = pd.DataFrame({'x': np.linspace(0, 1, 50), 'y': np.linspace(2, 3, 50)})
    sub = ListSubclass([df1, df2])
    out = Resample(n_samples=17).fit_transform(sub)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape[0] == 17
    assert out[1].shape[0] == 17
