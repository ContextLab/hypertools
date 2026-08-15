"""Styling consumes trace METADATA only, and never builds a trace.

Formulas (multiindex.py:14-38): linewidth = 1 + (L - 1 - level_idx);
alpha = min(1.0, 1 / (level_idx + 1) + 0.2); colour by the TOP level; only
the top-level mean is labelled -- EXCEPT when n_levels == 1, where no mean
exists and every leaf is itself top-level.
"""
import matplotlib
matplotlib.use("Agg")

import inspect

import numpy as np
import pytest

from hypertools.plot.hierarchy import build_hierarchy_styles, build_hierarchy_traces

COL_META = {'n_levels': 2, 'axis': 'columns', 'level_names': ['Market', 'Sector'],
            'leaf_keys': [('M', 'Tech'), ('M', 'Fin'), ('M', 'Energy')]}
ROW_META = {'n_levels': 3, 'axis': 'rows', 'level_names': ['grp', 'cond', 'subj'],
            'leaf_keys': [('X', 'A', 'S0'), ('X', 'A', 'S1'),
                          ('X', 'B', 'S0'), ('X', 'B', 'S1'),
                          ('Y', 'A', 'S0'), ('Y', 'A', 'S1'),
                          ('Y', 'B', 'S0'), ('Y', 'B', 'S1')]}
ONE_META = {'n_levels': 1, 'axis': 'columns', 'level_names': ['Group'],
            'leaf_keys': [('A',), ('B',), ('C',)]}


def _traces(meta, n):
    return build_hierarchy_traces([np.zeros((5, 2))] * n, meta)


def test_style_formulas_match_the_documented_contract_two_levels():
    style = build_hierarchy_styles(_traces(COL_META, 3))
    assert style['linewidths'] == [1.0, 1.0, 1.0, 2.0]
    assert style['alphas'] == pytest.approx([0.7, 0.7, 0.7, 1.0])


def test_style_formulas_match_the_documented_contract_three_levels():
    style = build_hierarchy_styles(_traces(ROW_META, 8))
    assert style['linewidths'] == [1.0] * 8 + [2.0] * 4 + [3.0] * 2
    assert style['alphas'] == pytest.approx(
        [1 / 3 + 0.2] * 8 + [0.7] * 4 + [1.0] * 2)


def test_only_the_top_level_mean_is_labelled_when_means_exist():
    style = build_hierarchy_styles(_traces(ROW_META, 8))
    assert style['labels'] == ['_nolegend_'] * 12 + ['X', 'Y']


def test_one_level_hierarchy_labels_every_leaf():
    """F11: with no mean, three unlabelled traces was the bug."""
    style = build_hierarchy_styles(_traces(ONE_META, 3))
    assert style['labels'] == ['A', 'B', 'C']
    assert style['linewidths'] == [1.0, 1.0, 1.0]
    assert style['alphas'] == pytest.approx([1.0, 1.0, 1.0])


def test_one_level_hierarchy_gives_each_leaf_its_own_colour():
    style = build_hierarchy_styles(_traces(ONE_META, 3))
    assert len(set(style['colors'])) == 3
    assert style['unique_top'] == ['A', 'B', 'C']


def test_styles_take_metadata_not_leaf_arrays():
    """Structural guarantee for F1: the styler has no leaves to average."""
    params = list(inspect.signature(build_hierarchy_styles).parameters)
    assert params[0] == 'traces'
    assert 'leaf_arrays' not in params
    ft = _traces(COL_META, 3)
    ft.arrays = None                      # styling must not need them
    assert build_hierarchy_styles(ft)['linewidths'] == [1.0, 1.0, 1.0, 2.0]
