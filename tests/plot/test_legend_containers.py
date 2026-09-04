# -*- coding: utf-8 -*-
"""`legend=` accepts four label CONTAINERS; all four must mean the same.

`plot()`'s type check accepts `list`, `tuple`, `np.ndarray`, `pd.Series` and
`pd.Index`, but every downstream reader tested `isinstance(legend, (list,
tuple))` -- or, in `_build_colorbar_info`, just `list`. So on a
NON-hierarchical input the three array-likes skipped the per-trace length
check entirely and were handed to matplotlib WHOLE as each artist's label.
Measured on two (10, 3) datasets:

    legend=np.array(['a', 'b'])   -> both traces named "['a' 'b']"
    legend=pd.Index(['a', 'b'])   -> both named "Index(['a', 'b'], ...)"
    legend=pd.Series(['a', 'b'])  -> both named "0    a\\n1    b\\ndtype: str"

each with a matplotlib "Passing label as a length 2 sequence when plotting a
single ..." UserWarning. The HIERARCHY path handled all four containers
correctly, so the two paths disagreed about the same accepted input type.

`plot()` now normalises every accepted container to a plain list at the type
check, so one rule covers them all.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


CONTAINERS = [
    pytest.param(list, id='list'),
    pytest.param(tuple, id='tuple'),
    pytest.param(np.array, id='ndarray'),
    pytest.param(pd.Series, id='Series'),
    pytest.param(pd.Index, id='Index'),
]


def two_datasets(seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((10, 3)).cumsum(axis=0) for _ in range(2)]


def _legend_texts(fig):
    leg = fig.axes[0].get_legend()
    return None if leg is None else [t.get_text() for t in leg.get_texts()]


def hierarchy_frame(seed=1):
    """2 top-level groups, so the same 2-label legend applies to both paths."""
    rng = np.random.default_rng(seed)
    columns = pd.MultiIndex.from_tuples(
        [(group, measure) for group in ('P', 'Q')
         for measure in ('m1', 'm2', 'm3')],
        names=['group', 'measure'])
    return pd.DataFrame(rng.normal(size=(20, 6)).cumsum(axis=0),
                        columns=columns)


@pytest.mark.parametrize('container', CONTAINERS)
def test_every_accepted_container_labels_the_traces_the_same(container):
    """One label per trace, not the whole container on every trace."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(two_datasets(), '-', legend=container(['a', 'b']),
                       show=False)
    assert _legend_texts(fig) == ['a', 'b']
    # matplotlib's own complaint about a sequence label must be gone too
    assert [str(w.message) for w in caught
            if 'Passing label as a length' in str(w.message)] == []


@pytest.mark.parametrize('container', CONTAINERS)
def test_every_accepted_container_gets_the_same_length_check(container):
    """The per-trace length check names `legend=`, the kwarg the caller
    passed -- the array-likes used to bypass it and mislabel silently."""
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(two_datasets(), '-', legend=container(['a', 'b', 'c']),
                 show=False)
    assert 'legend= was given as a list of length 3' in str(excinfo.value)
    assert 'there are 2 dataset(s)/group(s)' in str(excinfo.value)


@pytest.mark.parametrize('container', CONTAINERS)
def test_the_hierarchy_path_agrees_with_the_flat_path(container):
    """The path that was already right must stay right, on every container:
    the two paths now give the same answer for the same input."""
    fig = hyp.plot(hierarchy_frame(), '-', legend=container(['X', 'Y']),
                   show=False)
    assert _legend_texts(fig) == ['X', 'Y']


def test_a_zero_dimensional_array_is_one_label_like_a_bare_string():
    """`np.array('a')` is not iterable, so it cannot be a label LIST. It is
    treated as the single label `legend='a'` is, which then reports the
    length mismatch instead of silently broadcasting 'a' over both traces
    (measured before the fix: legend texts ['a', 'a'])."""
    with pytest.raises(ValueError, match='length 1'):
        hyp.plot(two_datasets(), '-', legend=np.array('a'), show=False)
    with pytest.raises(ValueError, match='length 1'):
        hyp.plot(two_datasets(), '-', legend='a', show=False)


def test_the_type_check_still_refuses_a_non_container():
    """Normalising the accepted containers must not widen what is accepted."""
    with pytest.raises(TypeError,
                       match='legend= must be True/False, a label string'):
        hyp.plot(two_datasets(), '-', legend=7, show=False)


@pytest.mark.parametrize('container', CONTAINERS)
def test_names_and_a_container_legend_still_conflict(container):
    """`_legend_user_list` must keep its meaning through the normalisation:
    a caller-supplied legend of ANY container type still conflicts with
    `names=`."""
    with pytest.raises(ValueError, match='not both'):
        hyp.plot(two_datasets(), '-', names=['a', 'b'],
                 legend=container(['c', 'd']), show=False)


@pytest.mark.parametrize('container', CONTAINERS)
def test_plotly_names_its_traces_from_every_container_too(container):
    hyp.set_interactive_backend('plotly')
    try:
        fig = hyp.plot(two_datasets(), '-', legend=container(['a', 'b']),
                       show=False)
    finally:
        hyp.set_interactive_backend('matplotlib')
    named = [t for t in fig.data
             if (t.meta or {}).get('hyp_trace_index') is not None]
    assert [t.name for t in named] == ['a', 'b']
