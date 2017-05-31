# -*- coding: utf-8 -*-

from builtins import range
import pytest

import numpy as np
import matplotlib as mpl

from hypertools.plot import plot
from hypertools.tools.reduce import reduce as reduc

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i in range(2)]

## STATIC ##
def test_plot_1d():
    data_reduced_1d = reduc(data,ndims=1)
    _, _, data_1d, _ = plot.plot(data_reduced_1d, show=False)
    assert all([i.shape[1]==1 for i in data_1d])

def test_plot_1dim():
    _, _, data_1dim, _  = plot.plot(np.array([1,2,3,4]), show=False)
    assert data_1dim[0].ndim==2

def test_plot_2d():
    data_reduced_2d = reduc(data,ndims=2)
    _, _, data_2d, _  = plot.plot(data_reduced_2d, show=False)
    assert all([i.shape[1]==2 for i in data_2d])

def test_plot_3d():
    data_reduced_3d = reduc(data,ndims=3)
    _, _, data_3d, _  = plot.plot(data_reduced_3d, show=False)
    assert all([i.shape[1]==3 for i in data_3d])

def test_plot_nd():
    _, _, data_nd, _  = plot.plot(data, show=False)
    assert all([i.shape[1]==3 for i in data_nd])

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
    data_reduced_1d = reduc(data,ndims=1)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_1d, animate=True, show=False)

def test_plot_2d_animate():
    data_reduced_2d = reduc(data,ndims=2)
    with pytest.raises(Exception) as e_info:
        plot.plot(data_reduced_2d, animate=True, show=False)

def test_plot_3d_animate():
    data_reduced_3d = reduc(data,ndims=3)
    _,_,data_3d,_ = plot.plot(data_reduced_3d, animate=True, show=False)
    assert all([i.shape[1]==3 for i in data_3d])

def test_plot_nd_animate():
    _,_,data_nd,_ = plot.plot(data, animate=True, show=False)
    assert all([i.shape[1]==3 for i in data_nd])

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
    _, _, data_3d, _  = plot.plot(data, colors=['b','r'], linestyles=['--',':'], markers=['o','*'], show=False)
    assert all([i.shape[1]==3 for i in data_3d])
