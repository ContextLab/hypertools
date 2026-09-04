# -*- coding: utf-8 -*-
"""hyp.load() gains two resolvers, inserted right after built-in example
dataset names and before local-file resolution: scikit-learn's small
bundled datasets (sklearn.datasets.load_*) and seaborn's named datasets
(seaborn.load_dataset). scikit-learn wins over seaborn for names both
define (e.g. 'iris'); built-in hypertools names always win over both.

All tests use real function calls -- no mocks. The seaborn tests hit the
real seaborn-data GitHub repo over the network.
"""

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError
from tests._netskip import skip_on_transient_network


def test_load_iris_is_sklearn_iris():
    # both scikit-learn and seaborn ship an 'iris' dataset with different
    # column names -- this proves scikit-learn wins the precedence order
    df = hyp.load('iris')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (150, 5)
    assert 'target' in df.columns
    for col in ('sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)'):
        assert col in df.columns
    # seaborn's iris uses snake_case column names instead
    assert 'sepal_length' not in df.columns


def test_load_digits_shape():
    df = hyp.load('digits')
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1797, 65)  # 64 pixel features + 'target'
    assert 'target' in df.columns


def test_load_linnerud_multioutput_targets():
    df = hyp.load('linnerud')
    assert isinstance(df, pd.DataFrame)
    for col in ('Weight', 'Waist', 'Pulse'):
        assert col in df.columns
    # multi-output targets get their own column names, not 'target'
    assert 'target' not in df.columns


def test_load_penguins_is_seaborn():
    # real fetch of seaborn-data from GitHub -- skip (never pass) if that host
    # is transiently unreachable; a genuine defect still fails. See
    # tests/_netskip.py for why guarding is not weakening.
    with skip_on_transient_network("loading seaborn's penguins"):
        df = hyp.load('penguins')
    assert isinstance(df, pd.DataFrame)
    for col in ('species', 'island', 'bill_length_mm'):
        assert col in df.columns


def test_load_weights_still_builtin():
    # precedence regression: a hypertools built-in name is resolved before
    # scikit-learn/seaborn are ever consulted
    data = hyp.load('weights')
    assert isinstance(data, list)
    assert len(data) > 1
    assert all(isinstance(arr, np.ndarray) for arr in data)


def test_load_unknown_name_mentions_new_resolvers():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('definitely_not_a_dataset_xyz')
    message = str(excinfo.value)
    assert 'scikit-learn' in message
    assert 'seaborn' in message


def test_load_iris_plot_end_to_end():
    fig = hyp.plot(hyp.load('iris').drop(columns=['target']), show=False)
    assert fig is not None


def test_load_iris_with_reduce_composes():
    # load()'s reduce/ndims kwargs must compose with the new resolvers,
    # exactly as they already do with built-in/local/remote sources
    out = hyp.load('iris', reduce='PCA', ndims=3)
    assert isinstance(out, np.ndarray)
    assert out.shape == (150, 3)
