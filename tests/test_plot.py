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
weights = load('weights')

# To prevent warning about 20+ figs being open
mpl.rcParams['figure.max_open_warning'] = 25

## STATIC ##
def test_plot_1d():
    data_reduced_1d = reducer(data,ndims=1)
    _, _, data_1d, _ = plot.plot(data_reduced_1d, show=False)
    assert all([i.shape[1]==1 for i in data_1d])

def test_plot_1dim():
    _, _, data_1dim, _  = plot.plot(np.array([1,2,3,4]), show=False)
    assert data_1dim[0].ndim==2

def test_plot_2d():
    data_reduced_2d = reducer(data,ndims=2)
    _, _, data_2d, _  = plot.plot(data_reduced_2d, show=False)
    assert all([i.shape[1]==2 for i in data_2d])

def test_plot_3d():
    data_reduced_3d = reducer(data,ndims=3)
    _, _, data_3d, _  = plot.plot(data_reduced_3d, show=False)
    assert all([i.shape[1]==3 for i in data_3d])

def test_plot_reduce_none():
    # Should return same dimensional data if ndims is None
    _, _, data_new, _ = plot.plot(data, show=False)
    assert all([i.shape[1] == d.shape[1] for i, d in zip(data_new, data)])

def test_plot_reduce3d():
    # should return 3d data since ndims=3
    _, _, data_3d, _ = plot.plot(data, ndims=3, show=False)
    assert all([i.shape[1] == 3 for i in data_3d])

def test_plot_reduce2d():
    # should return 2d data since ndims=2
    _, _, data_2d, _ = plot.plot(data, ndims=2, show=False)
    assert all([i.shape[1] == 2 for i in data_2d])

def test_plot_reduce1d():
    # should return 1d data since ndims=1
    _, _, data_1d, _ = plot.plot(data, ndims=1, show=False)
    assert all([i.shape[1] == 1 for i in data_1d])

def test_plot_reduce_align5d():
    # should return 5d data since ndims=5
    _, _, weights_5d, _ = plot.plot(weights, ndims=5, align=True, show=False)
    assert all([i.shape[1] == 5 for i in weights_5d])

def test_plot_reduce10d():
    # should return 10d data since ndims=10
    _, _, weights_10d, _ = plot.plot(weights, ndims=10, show=False)
    assert all([i.shape[1] == 10 for i in weights_10d])

def test_plot_nd():
    _, _, data_nd, _  = plot.plot(data, show=False)
    assert all([i.shape[1]==d.shape[1] for i, d in zip(data_nd, data)])

def test_plot_data_is_list():
    _, _, data_nd, _  = plot.plot(data, show=False)
    assert type(data_nd) is list

def test_plot_check_fig():
    fig, _, _, _  = plot.plot(data, show=False)
    assert isinstance(fig,mpl.figure.Figure)

def test_plot_check_ax():
    _, ax, _, _  = plot.plot(data, show=False)
    assert isinstance(ax,mpl.axes._axes.Axes)

## ANIMATED ##

def test_plot_1d_animate():
    data_reduced_1d = reducer(data,ndims=1)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_1d, animate=True, show=False)

def test_plot_2d_animate():
    data_reduced_2d = reducer(data,ndims=2)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_2d, animate=True, show=False)

def test_plot_3d_animate():
    data_reduced_3d = reducer(data,ndims=3)
    _,_,data_3d,_ = plot.plot(data_reduced_3d, animate=True, show=False)
    assert all([i.shape[1]==3 for i in data_3d])

def test_plot_nd_animate():
    _,_,data_nd,_ = plot.plot(data, animate=True, show=False)
    assert all([i.shape[1]==d.shape[1] for i, d in zip(data_nd, data)])

def test_plot_data_animate_is_list():
    _,_,data_nd,_ = plot.plot(data, animate=True, show=False)
    assert type(data_nd) is list

def test_plot_animate_check_fig():
    fig,_,_,_ = plot.plot(data, animate=True, show=False)
    assert isinstance(fig,mpl.figure.Figure)

def test_plot_animate_check_ax():
    _,ax,_,_ = plot.plot(data, animate=True, show=False)
    assert isinstance(ax,mpl.axes._axes.Axes)

def test_plot_animate_check_line_ani():
    _,_,_,line_ani = plot.plot(data, animate=True, show=False)
    assert isinstance(line_ani,mpl.animation.FuncAnimation)

def test_plot_mpl_kwargs():
    _, _, data_new, _  = plot.plot(data, colors=['b','r'], linestyles=['--',':'], markers=['o','*'], show=False)
    print([i.shape for i in data_new], [i.shape for i in data])
    assert all([i.shape[1]==d.shape[1] for i, d in zip(data_new, data)])
