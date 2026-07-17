# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import hypertools as hyp
from hypertools.reduce.describe import describe

data = np.random.multivariate_normal(np.zeros(10), np.eye(10), size=100)


def test_describe_data_is_dict():
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict


def test_describe_geo():
    # describe() operates on raw data directly (no geo round-trip in 1.0)
    result = describe(data, reduce='PCA', show=False)
    assert type(result) is dict


def test_describe_exposed_at_top_level():
    # QC 2026-07: confirm hyp.describe is the public entry point
    assert hyp.describe is describe


def test_describe_matplotlib_removes_top_and_right_spines():
    # QC 2026-07 (Jeremy): the describe plot should drop the top/right spines
    # (seaborn sns.despine(top=True, right=True)).
    plt.close('all')
    describe(data, reduce='PCA', max_dims=6, show=True, backend='matplotlib')
    ax = plt.gca()
    assert ax.spines['top'].get_visible() is False
    assert ax.spines['right'].get_visible() is False
    # the data axes stay
    assert ax.spines['left'].get_visible() is True
    assert ax.spines['bottom'].get_visible() is True
    plt.close('all')


def test_describe_plotly_backend_runs_and_returns_dict(monkeypatch):
    # the plotly backend renders an interactive go.Figure (Jeremy's "also
    # support plotly") and still returns the same dict. Suppress the actual
    # display so the test is headless.
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    monkeypatch.setattr(go.Figure, 'show', lambda self, *a, **k: None)
    result = describe(data, reduce='PCA', max_dims=6, show=True,
                      backend='plotly')
    assert type(result) is dict
    # 'fig' added by the 2026-07 release audit (F11-reduce-describe-015):
    # describe() now hands back its figure so it can be saved/embedded
    assert set(result.keys()) == {'average', 'individual', 'fig'}
    assert isinstance(result['fig'], go.Figure)
