#!/usr/bin/env python

"""
Wrapper function that parses plot styling arguments and calls plotting functions

INPUTS:
-numpy array(s)
-list of numpy arrays
-pandas dataframe

OUTPUTS:
-None
"""

##PACKAGES##
from __future__ import division
import sys
import warnings
import re
import itertools
import seaborn as sns
import pandas as pd
from .._shared.helpers import *
from .static import static_plot
from .animate import animated_plot
from ..tools.cluster import cluster
from ..tools.df2mat import df2mat
from ..tools.reduce import reduce as reduceD
from ..tools.normalize import normalize as normalizer

## MAIN FUNCTION ##
def plot(x,*args,**kwargs):

    # turn data into common format - a list of arrays
    x = format_data(x)

    ## HYPERTOOLS-SPECIFIC ARG PARSING ##

    if 'normalize' in kwargs:
        normalize = kwargs['normalize']
        x = normalizer(x, normalize=normalize)
        del kwargs['normalize']
    else:
        x = normalizer(x, normalize='across')

    # reduce dimensionality of the data
    if 'ndims' in kwargs:
        ndims=kwargs['ndims']
        x = reduceD(x,ndims)
        del kwargs['ndims']
    elif x[0].shape[1]>3:
        x = reduceD(x,3)
        ndims=3
    else:
        ndims=x[0].shape[1]

    if 'n_clusters' in kwargs:
        n_clusters=kwargs['n_clusters']

        cluster_labels = cluster(x, n_clusters=n_clusters, ndims=ndims)
        x = reshape_data(x,cluster_labels)
        del kwargs['n_clusters']

        if 'point_colors' in kwargs:
            warnings.warn('n_clusters overrides point_colors, ignoring point_colors.')
            del kwargs['point_colors']

    if 'point_colors' in kwargs:
        point_colors=kwargs['point_colors']
        del kwargs['point_colors']

        if 'color' in kwargs:
            warnings.warn("Using point_colors, color keyword will be ignored.")
            del kwargs['color']

        # if list of lists, unpack
        if any(isinstance(el, list) for el in point_colors):
            point_colors = list(itertools.chain(*point_colors))

        # if all of the elements are numbers, map them to colors
        if all(isinstance(el, int) or isinstance(el, float) for el in point_colors):
            point_colors = vals2bins(point_colors)
        elif all(isinstance(el, str) for el in point_colors):
            point_colors = group_by_category(point_colors)

        # reshape the data according to point_colors
        x = reshape_data(x,point_colors)

    if 'style' in kwargs:
        sns.set(style=kwargs['style'])
        del kwargs['style']
    else:
        sns.set(style="whitegrid")

    if 'palette' in kwargs:
        sns.set_palette(palette=kwargs['palette'], n_colors=len(x))
        palette = sns.color_palette(palette=kwargs['palette'], n_colors=len(x))
        del kwargs['palette']
    else:
        sns.set_palette(palette="hls", n_colors=len(x))
        palette=sns.color_palette(palette="hls", n_colors=len(x))

    if 'animate' in kwargs:
        animate=kwargs['animate']
        del kwargs['animate']

        # if animate mode, pass the color palette via kwargs so we can build a legend
        kwargs['color_palette']=palette

    else:
        animate=False

    if animate:
        return animated_plot(x,*args,**kwargs)
    else:
        return static_plot(x,*args,**kwargs)
