"""Column-hierarchy expansion end to end, INCLUDING its return bundle.

Rule: the innermost column level is the FEATURE axis; every level above it
groups. (Market, Sector, Ticker) -> 3 sector leaves + 1 market mean.
Every count below is exact -- measured, not bounded.

The hierarchical half of the Task 4 bundle contract lives here rather than
in tests/plot/test_hierarchy_bundle.py: the bundle needs the column path,
and the column path's assertions read the bundle, so putting them in one
task makes each task's verification step pass standalone (4 -> 5).
"""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


def market_frame(T=120, seed=0):
    rng = np.random.default_rng(seed)
    tuples = ([('Market', 'Tech', t) for t in ('AAPL', 'MSFT', 'NVDA')]
              + [('Market', 'Financials', t) for t in ('JPM', 'BAC', 'GS')]
              + [('Market', 'Energy', t) for t in ('XOM', 'CVX', 'COP')])
    cols = pd.MultiIndex.from_tuples(tuples, names=['Market', 'Sector', 'Ticker'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0) + 100.0, columns=cols)


def two_level_frame(T=60, seed=0):
    """(Group, Feature): n_levels == 1, one leaf per group, NO mean."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples(
        [(g, f) for g in ('A', 'B', 'C') for f in ('f0', 'f1', 'f2')],
        names=['Group', 'Feature'])
    return pd.DataFrame(rng.normal(size=(T, 9)).cumsum(axis=0), columns=cols)


def _ax(fig):
    return [a for a in fig.axes if hasattr(a, 'zaxis')][0]


def _labelled(ax):
    return [ln for ln in ax.lines
            if ln.get_label() and not ln.get_label().startswith('_')]


def test_draws_one_trace_per_sector_plus_a_market_mean():
    fig = hyp.plot(market_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_three_level_column_hierarchy_exact_styles():
    """linewidth = 1 + (L-1-level_idx), alpha = min(1, 1/(level_idx+1)+0.2),
    with L = n_levels = 2 for a (Market, Sector, Ticker) frame."""
    fig = hyp.plot(market_frame(), '-', show=False)
    lines = _ax(fig).lines
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == \
        [1.0, 1.0, 1.0, 2.0]
    assert [ln.get_alpha() for ln in lines] == pytest.approx(
        [0.7, 0.7, 0.7, 1.0])
    assert [ln.get_label() for ln in lines] == \
        ['_nolegend_'] * 3 + ['Market']
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in lines}
    assert len(colours) == 1, 'colour comes from the single top level'


def test_two_level_column_hierarchy_draws_one_trace_per_group():
    fig = hyp.plot(two_level_frame(), '-', show=False)
    assert len(_ax(fig).lines) == 3


def test_two_level_column_hierarchy_labels_every_trace():
    """F11: with no mean, unlabelled traces would leave an empty legend."""
    fig = hyp.plot(two_level_frame(), '-', show=False)
    assert [ln.get_label() for ln in _ax(fig).lines] == ['A', 'B', 'C']
    assert len(_labelled(_ax(fig))) == 3


def test_two_level_column_hierarchy_colours_widths_and_opacities():
    fig = hyp.plot(two_level_frame(), '-', show=False)
    lines = _ax(fig).lines
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == [1.0] * 3
    assert [ln.get_alpha() for ln in lines] == pytest.approx([1.0] * 3)
    colours = {tuple(np.round(matplotlib.colors.to_rgb(ln.get_color()), 4))
               for ln in lines}
    assert len(colours) == 3, 'each leaf is its own top-level group'


def test_ragged_groups_raise_the_existing_width_error():
    """Expansion accepts ragged groups; the analysis pipeline does not
    (plot.py:2750-2751)."""
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_tuples(
        [('M', 'Tech', t) for t in ('AAPL', 'MSFT', 'NVDA', 'ORCL')]
        + [('M', 'Energy', t) for t in ('XOM', 'CVX')],
        names=['Market', 'Sector', 'Ticker'])
    df = pd.DataFrame(rng.normal(size=(60, 6)), columns=cols)
    with pytest.raises(ValueError, match='same number of columns'):
        hyp.plot(df, '-', show=False)


def test_dual_axis_frame_is_rejected_by_plot():
    idx = pd.MultiIndex.from_product([['a', 'b'], range(30)])
    cols = pd.MultiIndex.from_tuples([('M', 'Tech'), ('M', 'Energy')])
    df = pd.DataFrame(np.zeros((60, 2)), index=idx, columns=cols)
    with pytest.raises(ValueError, match='both a row and a column MultiIndex'):
        hyp.plot(df, '-', show=False)


def test_nan_hierarchy_label_does_not_silently_drop_a_group():
    """A NaN LABEL, not a NaN value. Measured: the pandas default gives 2
    groups instead of 3 (see tests/core/test_hierarchy_grouping.py)."""
    df = market_frame()
    df.columns = pd.MultiIndex.from_tuples(
        [(m, np.nan if s == 'Energy' else s, t) for m, s, t in df.columns],
        names=['Market', 'Sector', 'Ticker'])
    fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_nan_data_values_still_plot():
    """NaN VALUES, in contrast to the NaN LABEL above: these are imputed at
    format time (and say so), not grouped."""
    df = market_frame()
    df.iloc[:, 0] = np.nan
    with pytest.warns(UserWarning, match='Missing data'):
        fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_row_multiindex_behaviour_is_unchanged():
    """Exact, not `>= 6`: 6 leaves (lw 1.0, alpha 0.7, unlabelled) + 2 means
    (lw 2.0, alpha 1.0, labelled) -- measured on dev-1.0."""
    idx = pd.MultiIndex.from_tuples(
        [('cond1', s) for s in range(3)] + [('cond2', s) for s in range(3)],
        names=['cond', 'subj'])
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 4)), index=idx)
    fig = hyp.plot(df, '-', show=False)
    lines = _ax(fig).lines
    assert len(lines) == 8
    assert [round(float(ln.get_linewidth()), 3) for ln in lines] == \
        [1.0] * 6 + [2.0] * 2
    assert [ln.get_alpha() for ln in lines] == pytest.approx(
        [0.7] * 6 + [1.0] * 2)
    assert [ln.get_label() for ln in lines] == \
        ['_nolegend_'] * 6 + ['cond1', 'cond2']


def test_datetime_row_index_with_column_hierarchy():
    df = market_frame(T=60)
    df.index = pd.date_range('2020-01-01', periods=60)
    fig = hyp.plot(df, '-', show=False)
    assert len(_ax(fig).lines) == 4


def test_column_hierarchy_inside_a_list_is_rejected():
    """Before 1.1 this silently flattened to ONE line, with no warning."""
    with pytest.raises(ValueError, match='element 0'):
        hyp.plot([market_frame()], '-', show=False)


def test_colorbar_shows_one_segment_per_top_level_group():
    """The GH #100/#95 invariant, now on the column axis: one segment per
    top-level value, never '_nolegend_'."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fig = hyp.plot(two_level_frame(), '-', colorbar=True, show=False)
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert '_nolegend_' not in labels
    assert [lbl for lbl in labels if lbl] == ['A', 'B', 'C']


# --- the hierarchical half of the Task 4 bundle contract --------------------

def test_xform_data_holds_only_analysed_leaves():
    """3 sectors in, 3 analysed leaves out -- the market mean is NOT one."""
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['xform_data']) == 3


def test_trace_data_holds_every_plotted_trajectory():
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['trace_data']) == 4
    assert np.allclose(np.asarray(out['trace_data'][3]),
                       np.mean(np.stack([np.asarray(a)
                                         for a in out['trace_data'][:3]]),
                               axis=0))


def test_trace_data_length_matches_the_drawn_artists():
    """One artist per plotted trajectory. The artists' VERTEX counts differ
    (centering, scaling and antialiasing come after `trace_data`); only the
    counts of traces are compared."""
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    assert len(out['trace_data']) == len(_ax(out['fig']).lines) == 4


def test_trace_metadata_describes_every_trace():
    out = hyp.plot(market_frame(), '-', return_model=True, show=False)
    meta = out['trace_metadata']
    assert meta['axis'] == 'columns'
    assert meta['level_names'] == ['Market', 'Sector']
    assert meta['is_mean'] == [False, False, False, True]
    assert meta['level_idx'] == [1, 1, 1, 0]
    assert meta['keys'][-1] == ('Market',)
    assert meta['aux'] is None, 'no hue was passed'
