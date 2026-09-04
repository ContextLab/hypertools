# -*- coding: utf-8 -*-
"""Three user-facing API defects on the hierarchy path, all one site apart.

The MultiIndex branch of `plot()` used to end with an UNCONDITIONAL
``legend = _mi_style["labels"]``. Three things followed from it, each
reproduced below before being fixed:

A. ``legend=[...]`` was silently discarded -- the drawn legend showed the
   index values. Every SIBLING kwarg the hierarchy overrides
   (``color``/``colors``, ``linewidth``, ``alpha``) warns; ``legend``
   alone vanished without a word. ``legend=False`` was ignored the same
   way, so opting out still drew a legend.
B. ``names=`` ALONE raised "pass dataset names via names= OR a legend=
   list, not both" -- factually false, since the assignment above had put
   the hierarchy's labels into `legend` before the conflict check read it.
C. `'_nolegend_'` is MATPLOTLIB's "keep this artist out of the legend"
   sentinel. plotly has no such convention, so passing it through made it
   the literal trace NAME of every hierarchy leaf -- shown in hover labels
   and written into exported HTML, where a plain list of arrays yields
   ``name=None``.

Trap notes that shaped these assertions:

* the drawn legend is read from ``ax.get_legend().get_texts()``, not from
  artist labels: the hierarchy labels every artist and relies on the
  sentinel to keep leaves out, so "the labels" and "the legend" differ.
* on plotly, ``fig.data`` also carries the wireframe cube and a colorbar
  phantom; data traces are found by ``meta['hyp_trace_index']``.
"""
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp
from hypertools.plot.hierarchy import (build_hierarchy_styles,
                                       build_hierarchy_traces)


def row_frame(n_time=6, seed=0):
    """2 sectors x 2 tickers x `n_time` rows: 4 leaves + 2 sector means."""
    rng = np.random.default_rng(seed)
    tuples, rows = [], []
    for si, sector in enumerate(['tech', 'fin']):
        for ticker in ('t0', 't1'):
            rows.append(rng.standard_normal((n_time, 3)).cumsum(axis=0)
                        + si * 5.0)
            tuples.extend([(sector, ticker)] * n_time)
    idx = pd.MultiIndex.from_tuples(tuples, names=['sector', 'ticker'])
    return pd.DataFrame(np.vstack(rows), index=idx, columns=['x', 'y', 'z'])


def col_frame(T=20, seed=1):
    """A (Market, Sector, Measure) column hierarchy: 2 leaves + 1 mean."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples(
        [('Market', sector, m)
         for sector in ('Tech', 'Energy')
         for m in ('return', 'volatility', 'momentum')],
        names=['Market', 'Sector', 'Measure'])
    return pd.DataFrame(rng.normal(size=(T, 6)).cumsum(axis=0) + 100.0,
                        columns=cols)


def _legend_texts(fig):
    """What the READER sees: the drawn legend's entries, or None."""
    leg = fig.axes[0].get_legend()
    return None if leg is None else [t.get_text() for t in leg.get_texts()]


def _plotly(*args, **kwargs):
    hyp.set_interactive_backend('plotly')
    try:
        return hyp.plot(*args, **kwargs)
    finally:
        hyp.set_interactive_backend('matplotlib')


def _data_traces(fig):
    return [t for t in fig.data
            if (t.meta or {}).get('hyp_trace_index') is not None]


# ---------------------------------------------------------------- Finding A

def test_caller_supplied_legend_renames_the_top_level_groups():
    """The reported repro: the drawn legend said 'tech'/'fin' instead."""
    fig = hyp.plot(row_frame(), '-',
                   legend=['Technology sector', 'Financial sector'],
                   show=False)
    assert _legend_texts(fig) == ['Technology sector', 'Financial sector']


def test_caller_supplied_legend_does_not_warn_and_keeps_leaves_unlabelled():
    """Honouring it means no warning AND no extra legend entries: the
    leaves and intermediate means keep the sentinel, so a 6-trace figure
    still shows exactly the 2 top-level entries the caller passed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(row_frame(), '-', legend=['Tech', 'Fin'], show=False)
    assert [str(w.message) for w in caught
            if 'legend' in str(w.message)] == []
    assert _legend_texts(fig) == ['Tech', 'Fin']


def test_legend_list_sized_per_top_level_group_not_per_drawn_trace():
    """One entry per unique TOP-LEVEL value (2), not per drawn trace (6).

    The message must say which of the two it wants -- the generic per-trace
    check would have reported "there are 6 dataset(s)/group(s) to plot".
    """
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(row_frame(), '-', legend=['a', 'b', 'c'], show=False)
    message = str(excinfo.value)
    assert '3 entries' in message
    assert '2 unique top-level MultiIndex value(s)' in message


def test_legend_false_suppresses_the_hierarchys_automatic_legend():
    """`legend=False` is an explicit opt-out; the unconditional assignment
    overwrote it, so a legend was drawn anyway."""
    assert _legend_texts(hyp.plot(row_frame(), '-', legend=False,
                                  show=False)) is None
    # ... while the default and legend=True still label by index value
    assert _legend_texts(hyp.plot(row_frame(), '-', show=False)) \
        == ['tech', 'fin']
    assert _legend_texts(hyp.plot(row_frame(), '-', legend=True,
                                  show=False)) == ['tech', 'fin']


def test_caller_supplied_legend_on_a_column_hierarchy():
    """Same rule on the other axis: 2 leaves + 1 market mean, ONE top-level
    value, so one entry renames it."""
    fig = hyp.plot(col_frame(), '-', legend=['All markets'], show=False)
    assert _legend_texts(fig) == ['All markets']


def test_build_hierarchy_styles_legend_labels_only_touches_labelled_traces():
    """The styling half, in isolation: colours/widths/alphas are untouched
    and only the top-level mean's label changes."""
    meta = {'n_levels': 2, 'axis': 'columns',
            'level_names': ['Market', 'Sector'],
            'leaf_keys': [('M', 'Tech'), ('M', 'Fin')]}
    traces = build_hierarchy_traces([np.zeros((5, 2))] * 2, meta)
    plain = build_hierarchy_styles(traces)
    renamed = build_hierarchy_styles(traces, legend_labels=['Everything'])
    assert plain['labels'] == ['_nolegend_', '_nolegend_', 'M']
    assert renamed['labels'] == ['_nolegend_', '_nolegend_', 'Everything']
    for key in ('colors', 'linewidths', 'alphas'):
        assert renamed[key] == plain[key]


def test_continuous_hue_drops_a_legend_list_with_the_same_warning_as_true():
    """The column-hierarchy continuous-hue path colours by VALUE, so it has
    no groups to name. It already warned for `legend=True`; a list used to
    survive to the per-trace length check and be rejected for its length,
    which blames the caller for a legend that path cannot draw at all."""
    frame = col_frame()
    hue = np.linspace(0.0, 1.0, len(frame))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(frame, '-', hue=hue, legend=['a', 'b'], show=False)
    assert any('legend is not supported for continuous or matrix-valued hue'
               in str(w.message) for w in caught)
    assert _legend_texts(fig) is None


# ---------------------------------------------------------------- Finding B

def test_names_alone_no_longer_claims_a_legend_list_was_also_passed():
    """The reported repro. No `legend=` was passed, so the "not both"
    message was simply untrue -- and it fired only because the hierarchy
    had already written its own labels into `legend`."""
    with pytest.raises(ValueError) as excinfo:
        hyp.plot(row_frame(), names=['a', 'b', 'c'], show=False)
    message = str(excinfo.value)
    assert 'not both' not in message
    # says something true about what IS drawn, and where to go instead
    assert 'names= assigns one name per input dataset' in message
    assert '4 leaf trajectory/ies + 2 derived per-level mean(s)' in message
    assert 'legend=[...]' in message
    # the flatten remedy is AXIS-specific, like the cluster= messages:
    # reset_index does not touch a COLUMN MultiIndex
    assert 'df.reset_index(drop=True)' in message
    with pytest.raises(ValueError) as col_excinfo:
        hyp.plot(col_frame(), names=['a'], show=False)
    assert "df.columns = df.columns.map('_'.join)" in str(col_excinfo.value)
    assert 'reset_index' not in str(col_excinfo.value)


def test_names_under_a_hierarchy_raises_whatever_its_length():
    """Not a length quibble: 6 names match the 6 drawn traces exactly and
    are still refused, because none of those traces is an input dataset."""
    with pytest.raises(ValueError, match='names= assigns one name per '
                                         'input dataset'):
        hyp.plot(row_frame(), names=list('abcdef'), show=False)


def test_names_under_a_hierarchy_raises_on_the_continuous_hue_path_too():
    """The one path where `names=` used to slip through and label the drawn
    traces (column hierarchy + continuous hue: that branch leaves `legend`
    None, so the "not both" check never fired). It labelled leaves and
    derived means with per-DATASET names -- the same category error the
    categorical-hue guard rejects -- and the identical call without `hue=`
    raised. Refused on every hierarchy path now, with one message."""
    frame = col_frame()
    hue = np.linspace(0.0, 1.0, len(frame))
    with pytest.raises(ValueError, match='names= assigns one name per '
                                         'input dataset'):
        hyp.plot(frame, '-', hue=hue, names=['a', 'b', 'c'], show=False)


def test_names_and_legend_list_together_still_raise_not_both():
    """The real conflict must survive the fix -- on a flat input, where
    `names=` applies at all."""
    data = [np.random.default_rng(s).standard_normal((8, 3)) for s in (0, 1)]
    with pytest.raises(ValueError, match='not both'):
        hyp.plot(data, names=['a', 'b'], legend=['c', 'd'], show=False)


def test_names_alone_still_works_on_a_flat_input():
    """The fix must not make `names=` harder to use where it does apply."""
    data = [np.random.default_rng(s).standard_normal((8, 3)) for s in (0, 1)]
    fig = hyp.plot(data, names=['first', 'second'], show=False)
    assert _legend_texts(fig) == ['first', 'second']


# ---------------------------------------------------------------- Finding C

@pytest.mark.parametrize('frame_fn', [row_frame, col_frame])
def test_plotly_never_names_a_trace_with_matplotlibs_sentinel(frame_fn):
    fig = _plotly(frame_fn(), '-', show=False)
    names = [t.name for t in _data_traces(fig)]
    assert '_nolegend_' not in names
    # unlabelled traces look exactly like a plain list's traces do
    assert all(n is None or not n.startswith('_') for n in names)


def test_plotly_hierarchy_unlabelled_traces_match_a_plain_lists_name():
    """The baseline the sentinel deviated from: plotly's own "no name"."""
    plain = _plotly([np.random.default_rng(s).standard_normal((8, 3))
                     for s in (0, 1)], '-', show=False)
    assert {t.name for t in _data_traces(plain)} == {None}
    hier = _plotly(row_frame(), '-', show=False)
    leaves = [t for t in _data_traces(hier) if not t.showlegend]
    assert len(leaves) == 4
    assert {t.name for t in leaves} == {None}


def test_plotly_sentinel_does_not_reach_exported_html():
    """Where a user actually meets it: hover labels and saved HTML."""
    fig = _plotly(row_frame(), '-', show=False)
    assert '_nolegend_' not in fig.to_html(include_plotlyjs=False)


def test_plotly_showlegend_still_follows_the_sentinel():
    """Normalising the NAME must not change WHICH traces are in the legend:
    exactly the two top-level means, named by their index values."""
    fig = _plotly(row_frame(), '-', show=False)
    shown = [t for t in _data_traces(fig) if t.showlegend]
    assert [t.name for t in shown] == ['tech', 'fin']
    assert len(_data_traces(fig)) == 6


def test_plotly_honours_a_caller_supplied_legend_too():
    """Findings A and C meet here: renamed groups, and still no sentinel."""
    fig = _plotly(row_frame(), '-', legend=['Tech sector', 'Fin sector'],
                  show=False)
    shown = [t for t in _data_traces(fig) if t.showlegend]
    assert [t.name for t in shown] == ['Tech sector', 'Fin sector']
    assert '_nolegend_' not in fig.to_html(include_plotlyjs=False)


def test_plotly_legend_false_hides_every_hierarchy_trace_from_the_legend():
    fig = _plotly(row_frame(), '-', legend=False, show=False)
    assert not any(t.showlegend for t in _data_traces(fig))
