# -*- coding: utf-8 -*-
"""Regression tests for the same bug pattern fixed for ``colors=`` in GH #142
(commit 8d424e08): ``linestyles=`` and ``markers=`` (the plural aliases) were
silent no-ops in ``hyp.plot`` unless the corresponding singular kwarg
(``linestyle=``/``marker=``) was ALSO passed, because the hoisted blocks in
``plot.py`` only ever consulted ``linestyles``/``markers`` from *inside*
``if linestyle is not None:`` / ``if marker is not None:``.

These tests exercise the public ``hyp.plot`` API with real renders (no
mocks) and assert on real matplotlib/plotly artist properties.
"""

import warnings

import matplotlib as mpl
import pytest

import numpy as np

import hypertools as hyp
from hypertools.plot.plotly_backend import _MARKER_SYMBOLS, _SYMBOL_3D_FALLBACK, \
    _SYMBOLS_3D, _LINESTYLE_NAMES

mpl.rcParams['figure.max_open_warning'] = 25

LINESTYLES = ['-', '--', ':']
MARKERS = ['o', 's', '^']


def _make_data(n=3, size=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(size, 3)) + i for i in range(n)]


def _expected_3d_symbol(mpl_marker):
    symbol = _MARKER_SYMBOLS[mpl_marker]
    if symbol not in _SYMBOLS_3D:
        symbol = _SYMBOL_3D_FALLBACK.get(symbol, 'circle')
    return symbol


# ---------------------------------------------------------------------------
# linestyles= (plural) alone
# ---------------------------------------------------------------------------

def test_linestyles_kwarg_alone_sets_linestyle_mpl():
    data = _make_data()
    fig = hyp.plot(data, linestyles=LINESTYLES, legend=['a', 'b', 'c'],
                   show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, LINESTYLES):
        assert line.get_linestyle() == expected


def test_linestyle_kwarg_alone_still_sets_linestyle_mpl():
    # singular alias must keep working
    data = _make_data()
    fig = hyp.plot(data, linestyle=LINESTYLES, legend=['a', 'b', 'c'],
                   show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, LINESTYLES):
        assert line.get_linestyle() == expected


def test_linestyles_kwarg_alone_sets_trace_dash_plotly():
    pytest.importorskip('plotly')
    data = _make_data()
    fig = hyp.plot(data, linestyles=LINESTYLES, legend=['a', 'b', 'c'],
                   backend='plotly', show=False)
    data_traces = [t for t in fig.data
                   if t.type == 'scatter3d' and t.name is not None]
    assert len(data_traces) == 3
    for trace, expected in zip(data_traces, LINESTYLES):
        assert trace.line.dash == _LINESTYLE_NAMES[expected]


def test_linestyle_and_linestyles_both_given_linestyles_wins_and_warns():
    data = _make_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(data, linestyle=['-', '-', '-'],
                       linestyles=LINESTYLES, legend=['a', 'b', 'c'],
                       show=False)
    assert any('linestyle' in str(w.message).lower() for w in caught)
    ax = fig.axes[0]
    lines = ax.get_lines()
    for line, expected in zip(lines, LINESTYLES):
        assert line.get_linestyle() == expected


# ---------------------------------------------------------------------------
# markers= (plural) alone
# ---------------------------------------------------------------------------

def test_markers_kwarg_alone_sets_marker_mpl():
    data = _make_data()
    fig = hyp.plot(data, markers=MARKERS, legend=['a', 'b', 'c'], show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, MARKERS):
        assert line.get_marker() == expected


def test_marker_kwarg_alone_still_sets_marker_mpl():
    # singular alias must keep working
    data = _make_data()
    fig = hyp.plot(data, marker=MARKERS, legend=['a', 'b', 'c'], show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, MARKERS):
        assert line.get_marker() == expected


def test_markers_kwarg_alone_sets_trace_symbol_plotly():
    pytest.importorskip('plotly')
    data = _make_data()
    fig = hyp.plot(data, markers=MARKERS, legend=['a', 'b', 'c'],
                   backend='plotly', show=False)
    data_traces = [t for t in fig.data
                   if t.type == 'scatter3d' and t.name is not None]
    assert len(data_traces) == 3
    for trace, expected in zip(data_traces, MARKERS):
        assert trace.marker.symbol == _expected_3d_symbol(expected)


def test_marker_and_markers_both_given_markers_wins_and_warns():
    data = _make_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(data, marker=['o', 'o', 'o'], markers=MARKERS,
                       legend=['a', 'b', 'c'], show=False)
    assert any('marker' in str(w.message).lower() for w in caught)
    ax = fig.axes[0]
    lines = ax.get_lines()
    for line, expected in zip(lines, MARKERS):
        assert line.get_marker() == expected
