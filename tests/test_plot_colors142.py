# -*- coding: utf-8 -*-
"""Regression tests for GH #142 follow-up: ``colors=`` (plural) alone was a
silent no-op in ``hyp.plot`` because ``plot.py``'s color-handling block only
ever looked at ``colors`` when ``color`` was ALSO not None. ``color=`` alone
worked; ``colors=`` alone silently fell back to the default hls palette.

These tests exercise the public ``hyp.plot`` API with real renders (no
mocks) and assert on real matplotlib/plotly artist properties.
"""

import warnings

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pytest

import hypertools as hyp

mpl.rcParams['figure.max_open_warning'] = 25

RED_GREEN_BLUE = ['red', 'green', 'blue']


def _make_data(n=3, size=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(size, 3)) + i for i in range(n)]


def _rgb(name):
    return mcolors.to_rgb(name)


# (a) mpl static: colors= alone must be honored, both on the lines AND on
# the legend swatches.
def test_colors_kwarg_alone_sets_line_and_legend_colors_mpl():
    data = _make_data()
    fig = hyp.plot(data, colors=RED_GREEN_BLUE, legend=['a', 'b', 'c'],
                   show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, RED_GREEN_BLUE):
        assert mcolors.to_rgb(line.get_color()) == _rgb(expected)

    handles, labels = ax.get_legend_handles_labels()
    assert labels == ['a', 'b', 'c']
    for handle, expected in zip(handles, RED_GREEN_BLUE):
        assert mcolors.to_rgb(handle.get_color()) == _rgb(expected)


# (b) same, but via the singular `color=` alias -- must keep working.
def test_color_kwarg_alone_sets_line_and_legend_colors_mpl():
    data = _make_data()
    fig = hyp.plot(data, color=RED_GREEN_BLUE, legend=['a', 'b', 'c'],
                   show=False)
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, expected in zip(lines, RED_GREEN_BLUE):
        assert mcolors.to_rgb(line.get_color()) == _rgb(expected)

    handles, labels = ax.get_legend_handles_labels()
    for handle, expected in zip(handles, RED_GREEN_BLUE):
        assert mcolors.to_rgb(handle.get_color()) == _rgb(expected)


# (c) plotly backend: colors= alone must be honored on trace line colors.
def test_colors_kwarg_alone_sets_trace_colors_plotly():
    pytest.importorskip('plotly')
    data = _make_data()
    fig = hyp.plot(data, colors=RED_GREEN_BLUE, legend=['a', 'b', 'c'],
                   backend='plotly', show=False)
    data_traces = [t for t in fig.data
                   if t.type == 'scatter3d' and t.name is not None]
    assert len(data_traces) == 3
    for trace, expected in zip(data_traces, RED_GREEN_BLUE):
        er, eg, eb = _rgb(expected)
        expected_str = (f'rgba({int(er * 255)},{int(eg * 255)},'
                        f'{int(eb * 255)},1.0)')
        assert trace.line.color == expected_str


# (d) color and colors both given: colors wins (matching the existing
# conflict-warning style at plot.py's linestyle/marker blocks), and a
# warning is raised.
def test_color_and_colors_both_given_colors_wins_and_warns():
    data = _make_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hyp.plot(data, color=['black', 'black', 'black'],
                       colors=RED_GREEN_BLUE, legend=['a', 'b', 'c'],
                       show=False)
    assert any('color' in str(w.message).lower() for w in caught)
    ax = fig.axes[0]
    lines = ax.get_lines()
    for line, expected in zip(lines, RED_GREEN_BLUE):
        assert mcolors.to_rgb(line.get_color()) == _rgb(expected)


# (e) animated mpl: colors= must be honored on the animated line artists.
def test_colors_kwarg_alone_sets_line_colors_animated_mpl():
    data = _make_data()
    fig, line_ani = hyp.plot(data, colors=RED_GREEN_BLUE, animate=True,
                             duration=1, frame_rate=2, show=False)
    ax = fig.axes[0]
    # animated lines carry the requested color even before any frames are
    # drawn (set at ax.plot(...) construction time in animate_plot3D)
    focus_lines = [ln for ln in ax.lines if ln.get_label() != '_nolegend_']
    assert len(focus_lines) == 3
    for line, expected in zip(focus_lines, RED_GREEN_BLUE):
        assert mcolors.to_rgb(line.get_color()) == _rgb(expected)
