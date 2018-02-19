# -*- coding: utf-8 -*-

from builtins import range
import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from hypertools.plot import plot
from hypertools.tools.reduce import reduce as reducer
from hypertools.tools.load import load
from hypertools.datageometry import DataGeometry

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

def test_plot_model_dict():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, reduce={'model' : 'PCA', 'params' : {'whiten' : True}}, show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_cluster_str():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, cluster='KMeans', show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_cluster_dict():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, cluster={'model' : 'KMeans', 'params' : {'n_clusters' : 3}}, show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_cluster_n_clusters():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, n_clusters=3, show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_cluster_HDBSCAN():
    # should return 10d data since ndims=10
    geo = plot.plot(weights, cluster='HDBSCAN', show=False)
    assert isinstance(geo, DataGeometry)

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

def test_plot_text():
    text_data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
            ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]
    geo = plot.plot(text_data, show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    geo = plot.plot(data, ax=ax, show=False)
    assert isinstance(geo, DataGeometry)

def test_plot_ax_2d():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    geo = plot.plot(data, ax=ax, show=False, ndims=2)
    assert isinstance(geo, DataGeometry)

def test_plot_ax_error():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError) as e_info:
        geo = plot.plot(data, ax=ax, show=False)

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
