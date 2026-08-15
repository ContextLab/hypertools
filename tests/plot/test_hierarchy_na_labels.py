"""Missing hierarchy LABELS must form exactly one group, at every level.

`dropna=False` in `hypertools.core.hierarchy` promises that a group whose
hierarchy label is missing survives grouping. That promise is only half the
story: the surviving group then has to be *built* and *styled* as one group.
Ordinary tuple/dict equality cannot do that, because NaN != NaN.

Measured on the pandas in this repo's venv:

  * `np.nan`, `None` and `pd.NA` in a MultiIndex level all normalise to
    plain `float('nan')`, and `groupby` mints a SEPARATE nan object per
    group key -- so `(nan, 'Tech')[:1] == (nan, 'Energy')[:1]` is False and
    a dict keyed on the prefix sees two groups where there is one.
  * `pd.NaT` is a singleton, so identity short-circuiting hides the bug for
    that spelling alone.
  * `expand_multiindex` (the ROW rule) reads its keys straight off
    `df.index`, which returns the SAME nan object each time, so the row axis
    happens to be correct today -- by accident, not by construction.

Every test below therefore states the group count it expects rather than
relying on either accident.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from hypertools.core.hierarchy import group_columns
from hypertools.plot.hierarchy import build_hierarchy_styles, build_hierarchy_traces
from hypertools.plot.multiindex import expand_multiindex

MISSING = [np.nan, None, pd.NA]
MISSING_IDS = ['np.nan', 'None', 'pd.NA']


def _col_frame_missing_top(na, T=6):
    """(Market, Sector, Ticker) with the TOP level missing for every column.

    Two sectors, so the top-level prefix is shared by two leaves -- which is
    what makes a duplicated mean visible.
    """
    cols = pd.MultiIndex.from_tuples(
        [(na, 'Tech', 'AAPL'), (na, 'Tech', 'MSFT'),
         (na, 'Energy', 'XOM'), (na, 'Energy', 'CVX')],
        names=['Market', 'Sector', 'Ticker'])
    return pd.DataFrame(np.zeros((T, 4)), columns=cols)


def _leaves_like(meta, values=None, rows=5, cols=2):
    n = len(meta['leaf_keys'])
    if values is None:
        return [np.zeros((rows, cols))] * n
    return [np.full((rows, cols), float(v)) for v in values]


@pytest.mark.parametrize('na', MISSING, ids=MISSING_IDS)
def test_missing_top_level_label_makes_ONE_mean_not_one_per_leaf(na):
    leaves, meta = group_columns(_col_frame_missing_top(na))
    assert len(leaves) == 2, 'grouping already keeps both sectors (dropna=False)'

    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    assert sum(ft.is_mean) == 1, 'one missing top-level label = ONE mean'
    assert len(ft.arrays) == 3
    assert [bool(pd.isna(k[0])) for k in ft.keys] == [True, True, True]
    assert len(ft.keys[-1]) == 1, 'the mean is the top-level prefix'


@pytest.mark.parametrize('na', MISSING, ids=MISSING_IDS)
def test_missing_label_group_gets_one_colour_and_one_legend_entry(na):
    leaves, meta = group_columns(_col_frame_missing_top(na))
    style = build_hierarchy_styles(build_hierarchy_traces(_leaves_like(meta), meta))
    assert style['n_top'] == 1, 'one missing label = one top-level group'
    assert len(style['unique_top']) == 1
    assert len(set(style['colors'])) == 1, 'one colour, not one per leaf'
    labelled = [lbl for lbl in style['labels'] if lbl != '_nolegend_']
    assert len(labelled) == 1, 'exactly one legend entry'


def test_mean_membership_is_exact_for_a_missing_label_group():
    """Not just the COUNT: the surviving mean must average BOTH members."""
    leaves, meta = group_columns(_col_frame_missing_top(np.nan))
    ft = build_hierarchy_traces(_leaves_like(meta, values=[1.0, 3.0]), meta)
    assert sum(ft.is_mean) == 1
    assert np.allclose(ft.arrays[-1], 2.0), 'mean of 1.0 and 3.0 over BOTH leaves'


def test_missing_intermediate_level_label_makes_one_mean_per_group():
    """A missing label at an INTERMEDIATE level, where the prefix that
    contains it is what gets deduplicated. Four column levels, so grouping
    is by the first three and prefixes of length 2 carry the NaN."""
    cols = pd.MultiIndex.from_tuples(
        [('M', np.nan, 'g1', 'AAPL'), ('M', np.nan, 'g2', 'MSFT'),
         ('M', 'Energy', 'g3', 'XOM')],
        names=['Market', 'Sector', 'Sub', 'Ticker'])
    df = pd.DataFrame(np.zeros((6, 3)), columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 3 and meta['n_levels'] == 3

    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    # level 1 prefixes: ('M', nan) ONCE + ('M', 'Energy'); level 0: ('M',)
    assert sum(ft.is_mean) == 3, '2 intermediate means + 1 top mean'
    assert len(ft.arrays) == 6
    mean_keys = [k for k, m in zip(ft.keys, ft.is_mean) if m]
    assert [len(k) for k in mean_keys] == [2, 2, 1]
    assert bool(pd.isna(mean_keys[0][1])) and mean_keys[1][1] == 'Energy'


def test_distinct_nan_objects_are_the_same_group():
    """The identity trap, stated directly. Two nan objects that are `==`-
    unequal AND `is`-distinct must still form one group."""
    n1, n2 = float('nan'), float('nan')
    assert n1 is not n2 and not (n1 == n2)
    meta = {'n_levels': 2, 'axis': 'columns', 'level_names': ['top', 'sub'],
            'leaf_keys': [(n1, 'a'), (n2, 'b')]}
    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    assert sum(ft.is_mean) == 1
    assert build_hierarchy_styles(ft)['n_top'] == 1


def test_pd_NaT_labels_also_collapse():
    """NaT is a singleton, so this spelling passed even before the fix --
    pinned so it keeps passing for the right reason."""
    cols = pd.MultiIndex.from_tuples(
        [(pd.NaT, 'Tech', 'AAPL'), (pd.NaT, 'Energy', 'XOM')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.zeros((6, 2)), columns=cols)
    leaves, meta = group_columns(df)
    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    assert sum(ft.is_mean) == 1
    assert build_hierarchy_styles(ft)['n_top'] == 1


def test_original_label_values_are_preserved_not_replaced_by_a_sentinel():
    """Canonicalisation is for COMPARISON only. The keys and labels users
    see must still hold the original value, not an internal marker."""
    leaves, meta = group_columns(_col_frame_missing_top(np.nan))
    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    top_key = ft.keys[-1][0]
    assert isinstance(top_key, float) and np.isnan(top_key)
    style = build_hierarchy_styles(ft)
    assert bool(pd.isna(style['unique_top'][0]))
    labelled = [lbl for lbl in style['labels'] if lbl != '_nolegend_']
    assert labelled == ['nan'], 'the label renders the original value'


def test_first_appearance_order_is_kept_when_a_missing_label_is_interleaved():
    """Ordering is part of the contract: colours and leaf_keys both depend
    on first-appearance order, so a missing label must not jump position."""
    cols = pd.MultiIndex.from_tuples(
        [('A', 'x', 't1'), (np.nan, 'y', 't2'), ('B', 'z', 't3'),
         (np.nan, 'w', 't4')],
        names=['Top', 'Mid', 'Leaf'])
    df = pd.DataFrame(np.zeros((6, 4)), columns=cols)
    leaves, meta = group_columns(df)
    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    style = build_hierarchy_styles(ft)
    assert style['n_top'] == 3, 'A, missing, B -- the two missing are ONE'
    tops = style['unique_top']
    assert tops[0] == 'A' and bool(pd.isna(tops[1])) and tops[2] == 'B'


def test_row_hierarchy_metadata_with_a_missing_label_behaves_identically():
    """The ROW axis, through `expand_multiindex`'s metadata shape. This path
    is correct today only because `df.index` hands back the same nan object
    every time; after the fix it is correct by construction."""
    idx = pd.MultiIndex.from_tuples(
        [(np.nan, 'S0')] * 3 + [(np.nan, 'S1')] * 3 + [('condB', 'S0')] * 3,
        names=['cond', 'subj'])
    df = pd.DataFrame(np.arange(27.0).reshape(9, 3), index=idx)
    leaves, meta = expand_multiindex(df)
    assert len(leaves) == 3

    ft = build_hierarchy_traces(leaves, meta)
    assert sum(ft.is_mean) == 2, 'one mean for the missing cond, one for condB'
    style = build_hierarchy_styles(ft)
    assert style['n_top'] == 2
    assert len([lbl for lbl in style['labels'] if lbl != '_nolegend_']) == 2


def test_present_labels_are_completely_unaffected():
    """Regression guard: canonicalisation must be a no-op without NA."""
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', 'AAPL'), ('M', 'Tech', 'MSFT'),
         ('M', 'Energy', 'XOM'), ('M', 'Energy', 'CVX')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.zeros((6, 4)), columns=cols)
    leaves, meta = group_columns(df)
    ft = build_hierarchy_traces(_leaves_like(meta), meta)
    assert ft.keys == [('M', 'Tech'), ('M', 'Energy'), ('M',)]
    assert ft.is_mean == [False, False, True]
    style = build_hierarchy_styles(ft)
    assert style['n_top'] == 1 and style['unique_top'] == ['M']
    assert style['labels'] == ['_nolegend_', '_nolegend_', 'M']
