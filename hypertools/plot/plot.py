#!/usr/bin/env python

##PACKAGES##
from __future__ import division
import sys
import warnings
import re
import itertools
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from .._shared.helpers import *
from ..tools.cluster import cluster
from ..tools.df2mat import df2mat
from ..tools.reduce import reduce as reduceD
from ..tools.normalize import normalize as normalizer
from .draw import draw

def plot(x, format_string='-', marker=None, markers=None, linestyle=None,
         linestyles=None, color=None, colors=None, style='whitegrid',
         palette='hls', group=None, labels=None, legend=None, ndims=3,
         normalize=False, n_clusters=None, animate=False, show=True,
         save_path=None, explore=False, duration=30, tail_duration=2,
         rotations=2, zoom=0, chemtrails=False, return_data=False,
         frame_rate=50, **kwargs):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame or list of arrays/dfs
        Data for the plot. The form should be samples (rows) by features (cols).

    linestyle : str or list of str
        A list of line styles

    marker : str or list of str
        A list of marker types

    color : str or list of str
        A list of marker types

    palette : str
        A matplotlib or seaborn color palette

    group : str/int/float or list
        A list of group labels. Length must match the number of rows in your
        dataset. If the data type is numerical, the values will be mapped to
        rgb values in the specified palette. If the data type is strings,
        the points will be labeled categorically. To label a subset of points,
        use None (i.e. ['a', None, 'b','a']).

    labels : list
        A list of labels for each point. Must be dimensionality of data (x).
        If no label is wanted for a particular point, input None.

    legend : list
        A list of string labels to be plotted in a legend (one for each list
        item).

    ndims : int
        An `int` representing the number of dims to plot in. Must be 1,2, or 3.
        NOTE: Currently only works with static plots.

    normalize : str or False
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). If set to 'within', the columns will be
        z-scored within each list that is passed. If set to 'row', each row of
        the input data will be z-scored. If set to False, the input data will
        be returned (default is False).

    n_clusters : int
        If n_clusters is passed, HyperTools will perform k-means clustering
        with the k parameter set to n_clusters. The resulting clusters will
        be plotted in different colors according to the color palette.

    animate : bool
        If True, plots the data as an animated trajectory (default: False).

    show : bool
        If set to False, the figure will not be displayed, but the figure,
        axis and data objects will still be returned (see Outputs)
        (default: True).

    save_path str :
        Path to save the image/movie. Must include the file extension in the
        save path (i.e. save_path='/path/to/file/image.png'). NOTE: If saving
        an animation, FFMPEG must be installed (this is a matplotlib req).
        FFMPEG can be easily installed on a mac via homebrew brew install
        ffmpeg or linux via apt-get apt-get install ffmpeg. If you don't
        have homebrew (mac only), you can install it like this:
        /usr/bin/ruby -e "$(curl -fsSL
        https://raw.githubusercontent.com/Homebrew/install/master/install)".

    explore : bool
        Displays user defined labels will appear on hover. If no labels are
        passed, the point index and coordinate will be plotted. To use,
        set explore=True. Note: Explore more is currently only supported
        for 3D static plots.

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float
        Number of rotations around the box (default: 2)

    zoom (animation only) : float
        Zoom, positive numbers will zoom in (default: 0)

    chemtrails (animation only) : bool
        Added trail with change in opacity (default: False)

    Returns
    ----------
    fig, ax, data : Matplotlib.Figure.figure, Matplotlib.Axes.axis, Numpy array
        By default, the plot function outputs a figure handle
        (matplotlib.figure.Figure), axis handle (matplotlib.axes._axes.Axes)
        and data (list of numpy arrays), e.g. fig,axis,data = hyp.plot(x)

        If animate=True, the plot function additionally outputs an animation
        handle (matplotlib.animation.FuncAnimation)
        e.g. fig,axis,data,line_ani = hyp.plot(x, animate=True).

    """

    # turn data into common format - a list of arrays
    x = format_data(x)

    # handle styling and palette with seaborn
    sns.set_style(style=style)
    sns.set_palette(palette=palette, n_colors=len(x))

    # catch all non-hypertools kwargs here to pass on to matplotlib
    mpl_kwargs = kwargs

    # handle color (to be passed onto matplotlib)
    if color is not None:
        mpl_kwargs['color'] = color
    if colors is not None:
        mpl_kwargs['color'] = colors

    # handle linestyle (to be passed onto matplotlib)
    if linestyle is not None:
        mpl_kwargs['linestyle'] = linestyle
    if linestyles is not None:
        mpl_kwargs['linestyle'] = linestyles

    # handle marker (to be passed onto matplotlib)
    if marker is not None:
        mpl_kwargs['marker'] = marker
    if markers is not None:
        mpl_kwargs['marker'] = markers

    # handle marker (to be passed onto matplotlib)
    if legend is not False:
        mpl_kwargs['label'] = legend

    # normalize
    x = normalizer(x, normalize=normalize, internal=True)

    # reduce data
    if x[0].shape[1]>3:
        x = reduceD(x, ndims, internal=True)

    # find cluster and reshape if n_clusters
    if n_clusters is not None:
        cluster_labels = cluster(x, n_clusters=n_clusters, ndims=ndims)
        x = reshape_data(x, cluster_labels)
        if group:
            warnings.warn('n_clusters overrides group, ignoring group.')

    # group data if there is a grouping var
    if group is not None:

        if color is not None:
            warnings.warn("Using group, color keyword will be ignored.")

        # if list of lists, unpack
        if any(isinstance(el, list) for el in group):
            group = list(itertools.chain(*group))

        # if all of the elements are numbers, map them to colors
        if all(isinstance(el, int) or isinstance(el, float) for el in group):
            group = vals2bins(group)
        elif all(isinstance(el, str) for el in group):
            group = group_by_category(group)

        # reshape the data according to group
        x = reshape_data(x, group)

        # interpolate lines if they are grouped
        if all([symbol is not format_string for symbol in Line2D.markers.keys()]):
            x = patch_lines(x)

    # interpolate
    if format_string is '-':
        interp_val = frame_rate*duration/(x[0].shape[0] - 1)
        x = interp_array_list(x, interp_val=interp_val)

    # center
    x = center(x)

    # scale
    x = scale(x)

    # draw the plot
    fig, ax, data, line_ani = draw(x, format_string=format_string,
                            mpl_kwargs=mpl_kwargs,
                            labels=labels,
                            explore=explore,
                            legend=legend,
                            animate=animate)

    if save_path is not None:
        if animate:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=frame_rate, bitrate=1800)
            line_ani.save(save_path, writer=writer)
        else:
            plt.savefig(save_path)
    if show:
        plt.show()

    return fig, ax, data, line_ani
