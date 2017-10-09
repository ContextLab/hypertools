# -*- coding: utf-8 -*-

from builtins import range
import pytest

import numpy as np
import matplotlib as mpl

from hypertools.plot import plot
from hypertools.tools.reduce import reduce as reducer
from hypertools.tools.load import load

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i
        in range(2)]
weights = load('weights_avg')

# To prevent warning about 20+ figs being open
mpl.rcParams['figure.max_open_warning'] = 25

## STATIC ##
def test_plot_1d():
    data_reduced_1d = reducer(data, ndims=1)
    geo = plot.plot(data_reduced_1d, show=False)
    assert all([i.shape[1]==1 for i in geo.data])
#
def test_plot_1dim():
    geo  = plot.plot(np.array([1,2,3,4]), show=False)
    assert geo.data[0].ndim==2
#
def test_plot_2d():
    data_reduced_2d = reducer(data, ndims=2)
    geo = plot.plot(data_reduced_2d, show=False)
    assert all([i.shape[1]==2 for i in geo.data])
#
def test_plot_3d():
    data_reduced_3d = reducer(data, ndims=3)
    geo = plot.plot(data_reduced_3d, show=False)
    assert all([i.shape[1]==3 for i in geo.data])
#
def test_plot_reduce_none():
    # Should return same dimensional data if ndims is None
    geo = plot.plot(data, show=False)
    assert all([i.shape[1] == d.shape[1] for i, d in zip(geo.data, data)])

def test_plot_reduce3d():
    # should return 3d data since ndims=3
    geo = plot.plot(data, ndims=3, show=False)
    assert all([i.shape[1] == 3 for i in geo.xform_data])

def test_plot_reduce2d():
    # should return 2d data since ndims=2
    geo = plot.plot(data, ndims=2, show=False)
    assert all([i.shape[1] == 2 for i in geo.xform_data])

def test_plot_reduce1d():
    # should return 1d data since ndims=1
    geo = plot.plot(data, ndims=1, show=False)
    assert all([i.shape[1] == 1 for i in geo.xform_data])
#
def test_plot_reduce_align5d():
    # should return 5d data since ndims=5
    geo = plot.plot(weights, ndims=5, align=True, show=False)
    assert all([i.shape[1] == 5 for i in geo.xform_data])

def test_plot_reduce10d():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, ndims=10, show=False)
    assert all([i.shape[1] == 10 for i in geo.xform_data])

def test_plot_nd():
    geo  = plot.plot(data, show=False)
    assert all([i.shape[1]==d.shape[1] for i, d in zip(geo.data, data)])

def test_plot_data_is_list():
    geo  = plot.plot(data, show=False)
    assert type(geo.data) is list
#
def test_plot_check_fig():
    geo  = plot.plot(data, show=False)
    assert isinstance(geo.fig, mpl.figure.Figure)

def test_plot_check_ax():
    geo  = plot.plot(data, show=False)
    assert isinstance(geo.ax, mpl.axes._axes.Axes)

# ## ANIMATED ##

def test_plot_1d_animate():
    d = reducer(data, ndims=1)
    with pytest.raises(Exception) as e_info:
        plot.plot(d, animate=True, show=False)

def test_plot_2d_animate():
    data_reduced_2d = reducer(data, ndims=2)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_2d, animate=True, show=False)

def test_plot_3d_animate():
    data_reduced_3d = reducer(data,ndims=3)
    geo = plot.plot(data_reduced_3d, animate=True, show=False)
    assert all([i.shape[1]==3 for i in geo.data])

def test_plot_nd_animate():
    geo = plot.plot(data, animate=True, show=False)
    assert all([i.shape[1]==d.shape[1] for i, d in zip(geo.data, data)])

def test_plot_data_animate_is_list():
    geo = plot.plot(data, animate=True, show=False)
    assert type(geo.data) is list

def test_plot_animate_check_fig():
    geo = plot.plot(data, animate=True, show=False)
    assert isinstance(geo.fig, mpl.figure.Figure)

def test_plot_animate_check_ax():
    geo = plot.plot(data, animate=True, show=False)
    assert isinstance(geo.ax, mpl.axes._axes.Axes)

def test_plot_animate_check_line_ani():
    geo = plot.plot(data, animate=True, show=False)
    assert isinstance(geo.line_ani, mpl.animation.FuncAnimation)
