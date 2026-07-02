# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hypertools.plot.plot import plot, _flatten_nested


a = np.cumsum(np.random.default_rng(0).standard_normal((50, 4)), axis=0)
b = np.cumsum(np.random.default_rng(1).standard_normal((50, 4)), axis=0)
c = np.cumsum(np.random.default_rng(2).standard_normal((50, 4)), axis=0)


def test_flatten_nested_two_level():
    leaves, groups, depths = _flatten_nested([[a, b], [c]])
    assert len(leaves) == 3
    assert groups == [0, 0, 1]
    assert depths == [2, 2, 2]


def test_flatten_nested_mixed_depth():
    leaves, groups, depths = _flatten_nested([[a, [b]], c])
    assert len(leaves) == 3
    assert groups == [0, 0, 1]
    assert depths == [2, 3, 1]


def test_flatten_nested_skips_text():
    x = [['doc one', 'doc two'], ['doc three']]
    result, groups, depths = _flatten_nested(x)
    assert result is x and groups is None and depths is None


def test_plot_nested_colors_by_outer_group():
    geo = plot([[a, b], [c]], show=False)
    ax = geo.ax
    lines = ax.get_lines()
    assert len(lines) == 3
    # leaves under the same outer group share a color; other group differs
    assert lines[0].get_color() == lines[1].get_color()
    assert lines[0].get_color() != lines[2].get_color()
    plt.close('all')


def test_plot_nested_depth_styling():
    geo = plot([[a, [b]], c], show=False)
    lines = geo.ax.get_lines()
    widths = [line.get_linewidth() for line in lines]
    # deeper leaves render thinner
    assert widths[1] < widths[0] < widths[2] or widths[1] < widths[2]
    plt.close('all')


def test_plot_flat_list_unchanged():
    geo = plot([a, b], show=False)
    lines = geo.ax.get_lines()
    assert len(lines) == 2
    assert lines[0].get_color() != lines[1].get_color()
    plt.close('all')
