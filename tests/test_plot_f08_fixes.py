# -*- coding: utf-8 -*-
"""Release-1.0 audit, final wave: plot-side F08 items.

Covers:
- F08-016: DataFrame column names become default axis labels when the drawn
  axes correspond 1:1 to the columns (2-D/3-D, no real reduction); explicit
  xlabel=/ylabel=/zlabel= win
- plotly title font size follows the module's PT_TO_PX (100/72) rule
  (12pt -> 17px), not the abandoned CSS 96/72 factor
- R1: the "Unequal values passed to dims and n_components" UserWarning is
  issued ONCE per plot() call with a reduce instance + return_model=True
"""

import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import hypertools as hyp


rng = np.random.RandomState(11)


# --- F08-016: DataFrame column names as default axis labels --------------------

def test_two_col_dataframe_labels_axes_2d():
    df = pd.DataFrame({'temperature': rng.randn(50),
                       'pressure': rng.randn(50)})
    fig = hyp.plot(df, '.', ndims=2, show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'temperature'
    assert ax.get_ylabel() == 'pressure'


def test_three_col_dataframe_labels_axes_3d():
    df = pd.DataFrame({'a': rng.randn(50), 'b': rng.randn(50),
                       'c': rng.randn(50)})
    fig = hyp.plot(df, '.', show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'a'
    assert ax.get_ylabel() == 'b'
    assert ax.get_zlabel() == 'c'


def test_user_passed_labels_win_over_inference():
    df = pd.DataFrame({'a': rng.randn(50), 'b': rng.randn(50)})
    fig = hyp.plot(df, '.', ndims=2, xlabel='MINE', ylabel='', show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'MINE'
    assert ax.get_ylabel() == ''  # explicit empty string suppresses


def test_reduced_dataframe_axes_not_labeled():
    # 3 columns drawn in 2-D: axes are reduced components, not columns
    df = pd.DataFrame({'a': rng.randn(50), 'b': rng.randn(50),
                       'c': rng.randn(50)})
    fig = hyp.plot(df, '.', ndims=2, show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == ''
    assert ax.get_ylabel() == ''


def test_default_integer_columns_not_used_as_labels():
    fig = hyp.plot(pd.DataFrame(rng.randn(50, 2)), '.', ndims=2, show=False)
    assert fig.axes[0].get_xlabel() == ''


def test_plain_array_axes_not_labeled():
    fig = hyp.plot(rng.randn(50, 2), '.', ndims=2, show=False)
    assert fig.axes[0].get_xlabel() == ''


def test_single_element_list_of_dataframe_labels_axes():
    df = pd.DataFrame({'x1': rng.randn(40), 'x2': rng.randn(40)})
    fig = hyp.plot([df], '.', ndims=2, show=False)
    ax = fig.axes[0]
    assert ax.get_xlabel() == 'x1'
    assert ax.get_ylabel() == 'x2'


def test_two_dataframes_axes_not_labeled():
    df1 = pd.DataFrame({'a': rng.randn(30), 'b': rng.randn(30)})
    df2 = pd.DataFrame({'a': rng.randn(30), 'b': rng.randn(30)})
    fig = hyp.plot([df1, df2], '.', ndims=2, show=False)
    assert fig.axes[0].get_xlabel() == ''


def test_plotly_backend_gets_inferred_labels_too():
    df = pd.DataFrame({'left': rng.randn(30), 'right': rng.randn(30)})
    with hyp.set_interactive_backend('plotly'):
        pfig = hyp.plot(df, '.', ndims=2, show=False)
    assert pfig.layout.xaxis.title.text == 'left'
    assert pfig.layout.yaxis.title.text == 'right'


# --- plotly title font size (PT_TO_PX consistency) ------------------------------

def test_plotly_title_font_size_matches_pt_to_px_rule():
    from hypertools.plot import plotly_backend as pb
    expected = round(12 * pb.PT_TO_PX)
    assert expected == 17
    with hyp.set_interactive_backend('plotly'):
        pfig = hyp.plot(rng.randn(30, 3), '.', title='Hello', show=False)
    assert pfig.layout.title.font.size == expected


# --- R1: duplicate dims/n_components warning ------------------------------------

def test_reduce_instance_return_model_warns_once():
    from sklearn.decomposition import PCA
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = hyp.plot(rng.randn(20, 5), '.', reduce=PCA(n_components=2),
                       return_model=True, show=False)
    msgs = [str(x.message) for x in w
            if 'Unequal values passed to dims and n_components'
            in str(x.message)]
    assert len(msgs) == 1
    # the bundle pipeline is still fitted and reusable
    assert res['pipeline'] is not None
    out = hyp.plot(rng.randn(20, 5), '.', pipeline=res['pipeline'],
                   show=False)
    assert out is not None


def test_reduce_instance_without_return_model_still_warns():
    from sklearn.decomposition import PCA
    with pytest.warns(UserWarning,
                      match='Unequal values passed to dims and n_components'):
        hyp.plot(rng.randn(20, 5), '.', reduce=PCA(n_components=2),
                 show=False)
