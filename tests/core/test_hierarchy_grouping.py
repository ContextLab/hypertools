"""Axis-agnostic hierarchy grouping (1.1, plan 2 Task 1).

Three DIFFERENT rules live here deliberately; conflating them was the main
defect in v1 of this plan. Every expectation below was measured on the
pandas 3.0.3 / numpy 2.3.5 in this repo's venv before it was written.
"""
import numpy as np
import pandas as pd
import pytest

from hypertools.core.hierarchy import (group_columns, group_rows_for_forecast,
                                       reject_dual_axis,
                                       reject_hierarchical_in_list)


def market_frame(T=120, seed=0):
    """rows = trading days, columns = (Market, Sector, Ticker)."""
    rng = np.random.default_rng(seed)
    tuples = ([('Market', 'Tech', t) for t in ('AAPL', 'MSFT', 'NVDA')]
              + [('Market', 'Financials', t) for t in ('JPM', 'BAC', 'GS')]
              + [('Market', 'Energy', t) for t in ('XOM', 'CVX', 'COP')])
    cols = pd.MultiIndex.from_tuples(tuples, names=['Market', 'Sector', 'Ticker'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def row_frame(T=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], range(T)],
                                     names=['Sector', 'day'])
    return pd.DataFrame(rng.normal(size=(2 * T, 3)).cumsum(axis=0), index=idx)


def dated_row_frame(T=30):
    days = pd.date_range('2020-01-01', periods=T)
    idx = pd.MultiIndex.from_product([['Tech', 'Energy'], days],
                                     names=['Sector', 'date'])
    return pd.DataFrame(np.arange(2 * T * 3).reshape(2 * T, 3) * 1.0, index=idx)


# --- column rule: group by every level ABOVE the innermost -------------------

def test_columns_group_by_every_level_above_the_innermost():
    leaves, meta = group_columns(market_frame())
    assert len(leaves) == 3
    assert [k[-1] for k in meta['leaf_keys']] == ['Tech', 'Financials', 'Energy']
    assert all(leaf.shape == (120, 3) for leaf in leaves)


def test_column_leaves_are_flattened_to_the_innermost_feature_level():
    """Contract 11. `sub.T` alone keeps the FULL column MultiIndex, which
    (a) contradicts the feature-axis rule and (b) makes hyp.predict recurse
    without bound (Revision note (v6) D1). Measured before the fix:
    [('Market','Tech','AAPL'), ...], names ['Market','Sector','Ticker']."""
    leaves, _ = group_columns(market_frame())
    assert all(not isinstance(leaf.columns, pd.MultiIndex) for leaf in leaves)
    assert leaves[0].columns.tolist() == ['AAPL', 'MSFT', 'NVDA']
    assert leaves[0].columns.name == 'Ticker'
    assert all(leaf.columns.nlevels == 1 for leaf in leaves)


def test_group_columns_does_not_mutate_the_callers_frame():
    """`df.T` / `sub.T` may return a VIEW depending on the pandas version and
    copy-on-write state, so the leaf is built explicitly rather than by
    assigning `.columns` onto a possibly-aliasing transpose. An input-mutation
    bug here would be silent."""
    df = market_frame()
    before = df.columns.copy()
    before_values = df.to_numpy(copy=True)
    group_columns(df)
    assert df.columns.equals(before)
    assert list(df.columns.names) == ['Market', 'Sector', 'Ticker']
    assert isinstance(df.columns, pd.MultiIndex)
    assert np.array_equal(df.to_numpy(), before_values)


def test_column_meta_matches_the_expand_multiindex_contract():
    _, meta = group_columns(market_frame())
    assert set(meta) >= {'leaf_keys', 'level_names', 'n_levels', 'axis'}
    assert meta['level_names'] == ['Market', 'Sector']
    assert meta['n_levels'] == 2
    assert meta['axis'] == 'columns'


def test_column_groups_keep_first_appearance_order_when_unsorted():
    df = market_frame()
    shuffled = df.iloc[:, [8, 0, 4, 1, 5, 2, 6, 3, 7]]
    _, meta = group_columns(shuffled)
    assert meta['leaf_keys'][0] == ('Market', 'Energy')


def test_single_level_columns_are_rejected():
    df = pd.DataFrame(np.zeros((10, 3)), columns=['a', 'b', 'c'])
    with pytest.raises(ValueError, match='2 or more'):
        group_columns(df)


def test_duplicate_tickers_in_different_sectors_are_kept_separate():
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', 'X'), ('M', 'Tech', 'Y'),
         ('M', 'Energy', 'X'), ('M', 'Energy', 'Y')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.zeros((10, 4)), columns=cols)
    leaves, _ = group_columns(df)
    assert len(leaves) == 2 and all(leaf.shape[1] == 2 for leaf in leaves)


def test_duplicate_innermost_feature_names_are_kept_positionally():
    """DECIDED (Revision note (v6) D3): duplicates WITHIN a group are
    permitted. Flattening can collide two tickers (two share classes, a
    repeated sensor); everything downstream is positional, so nothing is
    dropped and rejecting would break legitimate frames. Measured: widths
    [3, 3], np.asarray -> (20, 3), predict -> (1, 3), plot -> Figure."""
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', 'AAPL'), ('M', 'Tech', 'AAPL'), ('M', 'Tech', 'NVDA'),
         ('M', 'Fin', 'JPM'), ('M', 'Fin', 'GS'), ('M', 'Fin', 'BAC')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 6)),
                      columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 2, 'duplicate names must not merge groups'
    assert [leaf.shape[1] for leaf in leaves] == [3, 3]
    assert leaves[0].columns.tolist() == ['AAPL', 'AAPL', 'NVDA']
    assert not leaves[0].columns.is_unique
    assert np.asarray(leaves[0]).shape == (20, 3), 'no column was dropped'
    assert np.allclose(np.asarray(leaves[0]), df.to_numpy()[:, :3])
    assert meta['leaf_keys'] == [('M', 'Tech'), ('M', 'Fin')]


def test_unnamed_levels_are_tolerated():
    cols = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'z')])
    df = pd.DataFrame(np.zeros((5, 3)), columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 2
    assert meta['level_names'] == [None]


def test_two_level_columns_give_one_leaf_per_group_and_no_mean_level():
    """(Group, Feature) -> n_levels == 1, so the style layer must NOT expect
    an aggregate mean (see Task 3)."""
    cols = pd.MultiIndex.from_tuples(
        [('A', 'f0'), ('A', 'f1'), ('B', 'f0'), ('B', 'f1'),
         ('C', 'f0'), ('C', 'f1')], names=['Group', 'Feature'])
    df = pd.DataFrame(np.zeros((10, 6)), columns=cols)
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert meta['n_levels'] == 1
    assert meta['leaf_keys'] == [('A',), ('B',), ('C',)]


def test_nan_in_an_outer_column_label_does_not_drop_the_group():
    """Measured: the pandas default drops it (3 groups -> 2)."""
    df = market_frame(T=20)
    tuples = [(np.nan if s == 'Energy' else m, s, t)
              for m, s, t in df.columns]
    df.columns = pd.MultiIndex.from_tuples(tuples,
                                           names=['Market', 'Sector', 'Ticker'])
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert any(isinstance(k[0], float) and np.isnan(k[0])
               for k in meta['leaf_keys'])


def test_nan_in_an_intermediate_column_label_does_not_drop_the_group():
    df = market_frame(T=20)
    tuples = [(m, np.nan if s == 'Energy' else s, t) for m, s, t in df.columns]
    df.columns = pd.MultiIndex.from_tuples(tuples,
                                           names=['Market', 'Sector', 'Ticker'])
    leaves, meta = group_columns(df)
    assert len(leaves) == 3
    assert any(isinstance(k[1], float) and np.isnan(k[1])
               for k in meta['leaf_keys'])


# --- forecasting rule: innermost level is TIME, and it SURVIVES --------------

def test_row_forecast_grouping_treats_the_innermost_level_as_time():
    groups, keys = group_rows_for_forecast(row_frame())
    assert len(groups) == 2, 'one group per sector, NOT one per (sector, day)'
    assert [k[0] for k in keys] == ['Tech', 'Energy']
    assert all(g.shape == (60, 3) for g in groups)


def test_row_forecast_grouping_preserves_the_time_index():
    """reset_index(drop=True) would discard this; hyp.predict needs it."""
    groups, _ = group_rows_for_forecast(row_frame())
    assert groups[0].index.name == 'day'
    assert list(groups[0].index[:5]) == [0, 1, 2, 3, 4]
    assert groups[0].index.nlevels == 1


def test_row_forecast_grouping_preserves_a_datetime_index():
    groups, _ = group_rows_for_forecast(dated_row_frame())
    idx = groups[0].index
    assert idx.name == 'date'
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx[0] == pd.Timestamp('2020-01-01')
    assert idx.is_monotonic_increasing and idx.is_unique


def test_row_groups_are_flat_and_keep_their_datetime_identity():
    """Contract 11 on the ROW axis, together with the F5 promise. `droplevel`
    delivers both at once: the grouping levels go, the innermost level stays
    as a FLAT index with its own name and dtype. Measured: DatetimeIndex,
    name 'date', datetime64[us], monotonic, unique, nlevels == 1."""
    groups, _ = group_rows_for_forecast(dated_row_frame())
    for g in groups:
        assert not isinstance(g.index, pd.MultiIndex)
        assert g.index.nlevels == 1
        assert isinstance(g.index, pd.DatetimeIndex)
        assert g.index.name == 'date'
        assert g.index.is_monotonic_increasing and g.index.is_unique


def test_regrouping_a_leaf_is_refused_on_both_axes():
    """The fixed point is gone. A leaf that still carried its grouping levels
    would be re-detected as hierarchical and regrouped forever -- measured on
    the plot rule, `expand_multiindex(leaf0)` returns leaf0 itself. Neither
    core helper does that: each refuses, because there is nothing left to
    group. This is the property hyp.predict's recursion relies on."""
    col_leaf = group_columns(market_frame())[0][0]
    assert col_leaf.columns.nlevels == 1
    with pytest.raises(ValueError, match='2 or more'):
        group_columns(col_leaf)

    row_leaf = group_rows_for_forecast(row_frame())[0][0]
    assert row_leaf.index.nlevels == 1
    with pytest.raises(ValueError, match='2 or more'):
        group_rows_for_forecast(row_leaf)


def test_row_forecast_grouping_differs_from_plot_expansion():
    """Documented divergence: plot leaves are full tuples, forecast groups
    drop the innermost (time) level."""
    from hypertools.plot.multiindex import expand_multiindex
    df = row_frame()
    plot_leaves, _ = expand_multiindex(df)
    forecast_groups, _ = group_rows_for_forecast(df)
    assert len(plot_leaves) == 120
    assert len(forecast_groups) == 2


def test_three_level_row_forecast_grouping():
    idx = pd.MultiIndex.from_product([['M'], ['Tech', 'Energy'], range(30)],
                                     names=['Market', 'Sector', 'day'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, keys = group_rows_for_forecast(df)
    assert len(groups) == 2 and keys[0] == ('M', 'Tech')
    assert groups[0].index.name == 'day'


def test_nan_in_an_outer_row_label_does_not_drop_the_group():
    df = row_frame(T=10)
    tuples = [(np.nan if s == 'Energy' else s, d) for s, d in df.index]
    df.index = pd.MultiIndex.from_tuples(tuples, names=['Sector', 'day'])
    groups, _ = group_rows_for_forecast(df)
    assert len(groups) == 2


def test_unsorted_times_are_detectable_on_the_returned_index():
    """Preserving the index is what makes predict/common.py:103-109's
    'not sorted in ascending order' warning fire per group (F8)."""
    days = pd.date_range('2020-01-01', periods=30)
    perm = np.random.default_rng(0).permutation(30)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 30 + ['Energy'] * 30, list(days[perm]) + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, _ = group_rows_for_forecast(df)
    assert not groups[0].index.is_monotonic_increasing
    assert groups[1].index.is_monotonic_increasing


def test_duplicate_times_are_detectable_on_the_returned_index():
    days = pd.date_range('2020-01-01', periods=30)
    idx = pd.MultiIndex.from_arrays(
        [['Tech'] * 30 + ['Energy'] * 30, list(days[:15]) * 2 + list(days)],
        names=['Sector', 'date'])
    df = pd.DataFrame(np.zeros((60, 3)), index=idx)
    groups, _ = group_rows_for_forecast(df)
    assert not groups[0].index.is_unique
    assert groups[1].index.is_unique


# --- rejections -------------------------------------------------------------

def test_dual_axis_frames_are_rejected():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(5)])
    cols = pd.MultiIndex.from_tuples([('M', 'Tech'), ('M', 'Energy')])
    df = pd.DataFrame(np.zeros((10, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        reject_dual_axis(df)


def test_single_axis_frames_pass_the_dual_axis_check():
    reject_dual_axis(market_frame())
    reject_dual_axis(row_frame())


def test_column_hierarchical_frame_in_a_list_is_rejected():
    """Both callers reject a COLUMN hierarchy nested in a list."""
    with pytest.raises(ValueError, match='element 1'):
        reject_hierarchical_in_list([np.zeros((5, 3)), market_frame()],
                                    caller='hyp.plot', axes='columns')
    with pytest.raises(ValueError, match='hyp.predict'):
        reject_hierarchical_in_list([market_frame()], caller='hyp.predict',
                                    axes='both')


def test_row_hierarchical_frame_in_a_list_is_rejected_for_predict_only():
    """The deliberate asymmetry (Decisions (resolved) #1): `hyp.plot` keeps
    today's warn-and-flatten for the ROW axis, pinned by
    tests/test_multiindex.py:453, so the check must let it through; for
    `hyp.predict` today's behaviour is an opaque pandas TypeError, so
    rejecting it is additive."""
    reject_hierarchical_in_list([row_frame()], caller='hyp.plot',
                                axes='columns')          # must NOT raise
    with pytest.raises(ValueError, match='hyp.predict'):
        reject_hierarchical_in_list([row_frame()], caller='hyp.predict',
                                    axes='both')


def test_flat_frames_in_a_list_pass():
    reject_hierarchical_in_list(
        [np.zeros((5, 3)), pd.DataFrame(np.zeros((5, 3)))], caller='hyp.plot',
        axes='columns')
    reject_hierarchical_in_list(market_frame(), caller='hyp.plot',
                                axes='columns')
