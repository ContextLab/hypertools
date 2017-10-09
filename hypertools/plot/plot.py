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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from .._shared.helpers import *
from ..tools.analyze import analyze
from ..tools.cluster import cluster as clusterer
from ..tools.df2mat import df2mat
from ..tools.reduce import reduce as reducer
from ..tools.normalize import normalize as normalizer
from ..tools.align import align as aligner
from .draw import draw
from ..datageometry import DataGeometry

def plot(x, fmt=None, marker=None, markers=None, linestyle=None,
         linestyles=None, color=None, colors=None, palette='hls', group=None,
         labels=None, legend=None, title=None, elev=10, azim=-60, ndims=None,
         model=None, model_params=None, reduce='IncrementalPCA', cluster='KMeans',
         align=None, normalize=None,
         n_clusters=None, save_path=None, animate=False, duration=30, tail_duration=2,
         rotations=2, zoom=1, chemtrails=False, precog=False, bullettime=False,
         frame_rate=50, explore=False, show=True, transform=True):
    """
    Plots dimensionality reduced data and parses plot arguments

    Parameters
    ----------
    x : Numpy array, DataFrame or list of arrays/dfs
        Data for the plot. The form should be samples (rows) by features (cols).

    fmt : str or list of strings
        A list of format strings.  All matplotlib format strings are supported.

    linestyle(s) : str or list of str
        A list of line styles

    marker(s) : str or list of str
        A list of marker types

    color(s) : str or list of str
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

    legend : list or bool
        If set to True, legend is implicitly computed from data. Passing a
        list will add string labels to the legend (one for each list item).

    title : str
        A title for the plot

    ndims : int
        An `int` representing the number of dims to reduce the data x
        to. If ndims > 3, will plot in 3 dimensions but return the higher
        dimensional data. Default is None, which will plot data in 3
        dimensions and return the data with the same number of dimensions
        possibly normalized and/or aligned according to normalize/align
        kwargs.

    model : str
        Decomposition/manifold learning model to use.  Models supported: PCA,
        IncrementalPCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, FastICA,
        FactorAnalysis, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
        TSNE, Isomap, SpectralEmbedding, LocallyLinearEmbedding, and MDS.

    model_params : dict
        Optional dictionary of scikit-learn parameters to pass to reduction model.
        See scikit-learn specific model docs for details.

    align : bool
        If set to True, data will be run through the ``hyperalignment''
        algorithm implemented in hypertools.tools.align (default: False).

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

    save_path str :
        Path to save the image/movie. Must include the file extension in the
        save path (i.e. save_path='/path/to/file/image.png'). NOTE: If saving
        an animation, FFMPEG must be installed (this is a matplotlib req).
        FFMPEG can be easily installed on a mac via homebrew brew install
        ffmpeg or linux via apt-get apt-get install ffmpeg. If you don't
        have homebrew (mac only), you can install it like this:
        /usr/bin/ruby -e "$(curl -fsSL
        https://raw.githubusercontent.com/Homebrew/install/master/install)".

    animate : bool, 'parallel' or 'spin'
        If True or 'parallel', plots the data as an animated trajectory, with
        each dataset plotted simultaneously. If 'spin', all the data is plotted
        at once but the camera spins around the plot (default: False).

    duration (animation only) : float
        Length of the animation in seconds (default: 30 seconds)

    tail_duration (animation only) : float
        Sets the length of the tail of the data (default: 2 seconds)

    rotations (animation only) : float
        Number of rotations around the box (default: 2)

    zoom (animation only) : float
        How far to zoom into the plot, positive numbers will zoom in (default: 0)

    chemtrails (animation only) : bool
        A low-opacity trail is left behind the trajectory (default: False).

    precog (animation only) : bool
        A low-opacity trail is plotted ahead of the trajectory (default: False).

    bullettime (animation only) : bool
        A low-opacity trail is plotted ahead and behind the trajectory
        (default: False).

    frame_rate (animation only) : int or float
        Frame rate for animation (default: 50)

    explore : bool
        Displays user defined labels will appear on hover. If no labels are
        passed, the point index and coordinate will be plotted. To use,
        set explore=True. Note: Explore mode is currently only supported
        for 3D static plots, and is an experimental feature (i.e it may not yet
        work properly).

    show : bool
        If set to False, the figure will not be displayed, but the figure,
        axis and data objects will still be returned (default: True).

    Returns
    ----------
    fig, ax, data, line_ani : matplotlib.figure.figure, matplotlib.axis.axes, numpy.array, matplotlib.animation.funcanimation
        The plot function outputs a figure handle, axis handle, data, and line
        animation object.  The line animation object is None if animation=False.

    """

    # warnings for deprecated API args
    if (model is not None) or (model_params is not None):
        warnings.warn('Model and model_params arguments are deprecated. Please use \
                      reduce')
        reduce = {}
        reduce['model'] = model
        reduce['params'] = model_params


    if align is True:
        warnings.warn("Setting align=True will be deprecated.  Please specify the \
                      type of alignment, i.e. align='hyper'")
        align = 'hyper'

    # put into common format
    raw = format_data(x)

    # analyze the data
    if transform is True:
        x = analyze(raw, ndims=ndims, normalize=normalize, reduce=reduce,
                    align=align, internal=True)
    else:
        x = raw

    # Return data that has been normalized and possibly reduced and/or aligned
    xform_data = x

    # catch all matplotlib kwargs here to pass on
    mpl_kwargs = {}

    # handle color (to be passed onto matplotlib)
    if color is not None:
        mpl_kwargs['color'] = color
        if colors is not None:
            mpl_kwargs['color'] = colors
            warnings.warn('Both color and colors defined: color will be ignored \
                          in favor of colors.')

    # handle linestyle (to be passed onto matplotlib)
    if linestyle is not None:
        mpl_kwargs['linestyle'] = linestyle
        if linestyles is not None:
            mpl_kwargs['linestyle'] = linestyles
            warnings.warn('Both linestyle and linestyles defined: linestyle  \
                          will be ignored in favor of linestyles.')

    # handle marker (to be passed onto matplotlib)
    if marker is not None:
        mpl_kwargs['marker'] = marker
        if markers is not None:
            mpl_kwargs['marker'] = markers
            warnings.warn('Both marker and markers defined: marker will be \
                          ignored in favor of markers.')

    # reduce data to 3 dims for plotting, if ndims is None, return this
    if (ndims and ndims > 3):
        x = reducer(x, ndims=3, reduce=reduce, internal=True)
    elif ndims is None:
        x = reducer(x, ndims=3, reduce=reduce, internal=True)
        xform_data = x

    # find cluster and reshape if n_clusters
    if n_clusters is not None:
        cluster_labels = clusterer(x, cluster=cluster, n_clusters=n_clusters)
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
        if is_line(fmt):
            x = patch_lines(x)

    # handle legend
    if legend is not None:
        if legend is False:
            legend = None
        elif legend is True and group is not None:
            legend = [item for item in sorted(set(group), key=list(group).index)]
        elif legend is True and group is None:
            legend = [i + 1 for i in range(len(x))]

        mpl_kwargs['label'] = legend

    # interpolate if its a line plot
    if fmt is None or type(fmt) is str:
        if is_line(fmt):
            if x[0].shape[0] > 1:
                x = interp_array_list(x, interp_val=frame_rate*duration/(x[0].shape[0] - 1))
    elif type(fmt) is list:
        for idx, xi in enumerate(x):
            if is_line(fmt[idx]):
                if xi.shape[0] > 1:
                    x[idx] = interp_array_list(xi, interp_val=frame_rate*duration/(xi.shape[0] - 1))

    # handle explore flag
    if explore:
        assert x[0].shape[1] is 3, "Explore mode is currently only supported for 3D plots."
        mpl_kwargs['picker']=True

    # center
    x = center(x)

    # scale
    x = scale(x)

    # handle palette with seaborn
    sns.set_palette(palette=palette, n_colors=len(x))
    sns.set_style(style='whitegrid')

    # turn kwargs into a list
    kwargs_list = parse_kwargs(x, mpl_kwargs)

    # handle format strings
    if fmt is not None:
        if type(fmt) is not list:
            draw_fmt = [fmt for i in x]
        else:
            draw_fmt = fmt
    else:
        draw_fmt = fmt

    # draw the plot
    fig, ax, data, line_ani = draw(x, fmt=draw_fmt,
                            kwargs_list=kwargs_list,
                            labels=labels,
                            legend=legend,
                            title=title,
                            animate=animate,
                            duration=duration,
                            tail_duration=tail_duration,
                            rotations=rotations,
                            zoom=zoom,
                            chemtrails=chemtrails,
                            precog=precog,
                            bullettime=bullettime,
                            frame_rate=frame_rate,
                            elev=elev,
                            azim=azim,
                            explore=explore,
                            show=show)

    # tighten layout
    plt.tight_layout()

    # save
    if save_path is not None:
        if animate:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=frame_rate, bitrate=1800)
            line_ani.save(save_path, writer=writer)

        else:
            plt.savefig(save_path)

    # show the plot
    if show:
        plt.show()
    else:
        # safely closes the plot so it doesn't pop up in another call to this function
        plt.close()

    # gather reduce params
    if isinstance(reduce, dict):
        reduce_dict = reduce
    else:
        reduce_dict = {
            'model' : reduce,
            'params' : {
                'n_components' : ndims
            },
        }

    # gather align params
    if isinstance(align, dict):
        align_dict = align
    else:
        align_dict = {
            'model' : align,
            'params' : {}
        }

    # gather all other kwargs
    kwargs = {
        'fmt' : fmt,
        'marker': marker,
        'markers' : markers,
        'linestyle' : linestyle,
        'linestyles' : linestyles,
        'color' : color,
        'colors' : colors,
        'palette' : palette,
        'labels' : labels,
        'legend' : legend,
        'title' : title,
        'animate' : animate,
        'duration' : duration,
        'tail_duration' : tail_duration,
        'rotations' : rotations,
        'zoom' : zoom,
        'chemtrails' : chemtrails,
        'precog' : precog,
        'bullettime' : bullettime,
        'frame_rate' : frame_rate,
        'elev' : elev,
        'azim' : azim,
        'explore' : explore,
        'n_clusters' : n_clusters,
        'cluster' : cluster,
        'reduce' : reduce,
        'align' : align,
        'normalize' : normalize,
    }

    return DataGeometry(fig=fig, ax=ax, data=raw, xform_data=xform_data,
                        line_ani=line_ani, reduce=reduce_dict, align=align_dict,
                        normalize=normalize, kwargs=kwargs)
