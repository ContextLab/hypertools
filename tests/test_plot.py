# -*- coding: utf-8 -*-

import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from hypertools.plot import plot
from hypertools.reduce.reduce import reduce as reducer
from hypertools.io.load import load

data = [np.random.multivariate_normal(np.zeros(4), np.eye(4), size=100) for i
        in range(2)]
weights = load('weights_avg').get_data()

# To prevent warning about 20+ figs being open
mpl.rcParams['figure.max_open_warning'] = 25

## STATIC ##


def test_plot_1d():
    data_reduced_1d = reducer(data, ndims=1)
    fig = plot.plot(data_reduced_1d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==1 for i in data_reduced_1d])


def test_plot_2d():
    data_reduced_2d = reducer(data, ndims=2)
    fig = plot.plot(data_reduced_2d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==2 for i in data_reduced_2d])


def test_plot_3d():
    data_reduced_3d = reducer(data, ndims=3)
    fig = plot.plot(data_reduced_3d, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==3 for i in data_reduced_3d])


def test_plot_reduce_none():
    # Should return same dimensional data if ndims is None
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_reduce3d():
    # should return 3d data since ndims=3
    result = plot.plot(data, ndims=3, show=False, return_model=True)
    assert all([i.shape[1] == 3 for i in result['xform_data']])


def test_plot_reduce2d():
    # should return 2d data since ndims=2
    result = plot.plot(data, ndims=2, show=False, return_model=True)
    assert all([i.shape[1] == 2 for i in result['xform_data']])


def test_plot_reduce1d():
    # should return 1d data since ndims=1
    result = plot.plot(data, ndims=1, show=False, return_model=True)
    assert all([i.shape[1] == 1 for i in result['xform_data']])


def test_plot_reduce_align5d():
    # should return 5d data since ndims=5
    result = plot.plot(weights, ndims=5, align='hyper', show=False,
                       return_model=True)
    assert all([i.shape[1] == 5 for i in result['xform_data']])


def test_plot_reduce10d():
    # should return 10d data since ndims=10
    result = plot.plot(weights, ndims=10, show=False, return_model=True)
    assert all([i.shape[1] == 10 for i in result['xform_data']])


def test_plot_model_dict():
    fig = plot.plot(weights, reduce={'model' : 'PCA', 'params' : {'whiten' : True}}, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_str():
    fig = plot.plot(weights, cluster='KMeans', show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_dict():
    fig = plot.plot(weights, cluster={'model' : 'KMeans', 'params' : {'n_clusters' : 3}}, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_cluster_n_clusters():
    fig = plot.plot(weights, n_clusters=3, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_nd():
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_data_is_list():
    # list input still plots and returns a figure
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_check_fig():
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_check_ax():
    fig = plot.plot(data, show=False)
    assert isinstance(fig.axes[0], mpl.axes._axes.Axes)


def test_plot_text():
    text_data = [['i like cats alot', 'cats r pretty cool', 'cats are better than dogs'],
            ['dogs rule the haus', 'dogs are my jam', 'dogs are a mans best friend']]
    fig = plot.plot(text_data, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_ax():
    parent = plt.figure()
    ax = parent.add_subplot(111, projection='3d')
    fig = plot.plot(data, ax=ax, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig is ax.figure


def test_plot_ax_2d():
    parent = plt.figure()
    ax = parent.add_subplot(111)
    fig = plot.plot(data, ax=ax, show=False, ndims=2)
    assert isinstance(fig, mpl.figure.Figure)
    assert fig is ax.figure


def test_plot_ax_error():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError) as e_info:
        plot.plot(data, ax=ax, show=False)


def test_plot_geo():
    # re-plotting the same raw data twice both succeed (geo replay retired)
    fig = plot.plot(data, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    fig2 = plot.plot(data, show=False)
    assert isinstance(fig2, mpl.figure.Figure)


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
    fig, ani = plot.plot(data_reduced_3d, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert all([i.shape[1]==3 for i in data_reduced_3d])


def test_plot_nd_animate():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_data_animate_is_list():
    # list input still animates and returns a (fig, animation) tuple
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_animate_check_fig():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig, mpl.figure.Figure)


def test_plot_animate_check_ax():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(fig.axes[0], mpl.axes._axes.Axes)


def test_plot_animate_check_line_ani():
    fig, ani = plot.plot(data, animate=True, show=False)
    assert isinstance(ani, mpl.animation.FuncAnimation)
