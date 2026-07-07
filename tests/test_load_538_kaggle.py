# -*- coding: utf-8 -*-
"""hyp.load() gains two explicit-prefix resolvers (GH #116), inserted right
after the seaborn resolver and before local-file resolution:

- 'fivethirtyeight/<slug>' -- a dataset from
  https://github.com/fivethirtyeight/data, loaded by listing the slug's
  folder via the GitHub contents API and downloading every CSV it
  contains from raw.githubusercontent.com.
- 'kaggle/<owner>/<dataset>' -- a Kaggle dataset, downloaded anonymously
  via kagglehub.dataset_download.

Both are explicit, unambiguous prefixes: no shared index is maintained
(Jeremy's directive on GH #116), and a matching-but-failing name raises
HypertoolsIOError directly instead of falling through to the rest of the
resolution chain.

All tests use real function calls -- no mocks. They hit the real GitHub
API / raw.githubusercontent.com and the real Kaggle Hub over the network,
using small, public datasets chosen to keep downloads tiny and avoid
GitHub's unauthenticated 60 requests/hour rate limit (the fivethirtyeight
folder listing is cached per-process in hypertools.io.sources, so repeated
loads of the same slug within a test run don't re-hit the API).
"""

import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')

import hypertools as hyp
from hypertools._shared.exceptions import HypertoolsIOError

kagglehub = pytest.importorskip(
    'kagglehub',
    reason="kagglehub is required to exercise hyp.load('kaggle/...') -- "
          "install it with `pip install hypertools[kaggle]`")


def test_load_538_bechdel_single_csv():
    # bechdel's folder has exactly one CSV (movies.csv) -> a DataFrame
    df = hyp.load('fivethirtyeight/bechdel')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    assert 'binary' in df.columns


def test_load_538_multi_csv_returns_dict():
    # college-majors' folder has five (small) CSVs -> a dict of DataFrames
    data = hyp.load('fivethirtyeight/college-majors')
    assert isinstance(data, dict)
    assert len(data) > 1
    expected = {'all-ages', 'grad-students', 'majors-list', 'recent-grads',
               'women-stem'}
    assert expected.issubset(data.keys())
    for df in data.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_load_538_bad_slug_raises_immediately():
    # an explicit prefix that doesn't resolve raises directly -- it does
    # NOT silently fall through to local-file/HF/URL resolution
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('fivethirtyeight/definitely-not-real-xyz')
    message = str(excinfo.value)
    assert 'fivethirtyeight' in message


def test_load_kaggle_tiny_dataset():
    # uciml/iris is a tiny (~3.6KB) public dataset that downloads
    # anonymously (no Kaggle credentials needed) via kagglehub
    df = hyp.load('kaggle/uciml/iris')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.shape[1] > 1


def test_load_kaggle_malformed_name_raises_clear_error():
    with pytest.raises(HypertoolsIOError) as excinfo:
        hyp.load('kaggle/no-slash')
    message = str(excinfo.value)
    assert 'kaggle' in message.lower()
    assert '<owner>' in message or 'owner' in message.lower()


def test_load_538_bechdel_plot_end_to_end():
    df = hyp.load('fivethirtyeight/bechdel')
    numeric = df.select_dtypes(include='number').dropna()
    fig = hyp.plot(numeric, show=False)
    assert fig is not None
