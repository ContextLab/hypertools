#!/usr/bin/env python

"""
Wrapper function that parses plot styling arguments and calls plotting functions

INPUTS:
-numpy array(s)
-list of numpy arrays

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
from ..util.cluster import cluster

##MAIN FUNCTION##
def plot(x,*args,**kwargs):

    ##HANDLE TEXT VARIABLES FOR PANDAS DF##
    if 'text_vars' in kwargs:
        text_vars = kwargs['text_vars']
        del kwargs['text_vars']
    else:
        text_vars = 'dummy'

    ##CHECK DATA FORMAT##
    if isinstance(x, pd.DataFrame):
        x = pandas_to_list(x, text_vars=text_vars)

    ##HYPERTOOLS-SPECIFIC ARG PARSING##

    if 'n_clusters' in kwargs:
        n_clusters=kwargs['n_clusters']

        if 'ndims' in kwargs:
            ndims = kwargs['ndims']
        else:
            ndims = 3

        cluster_labels = cluster(x, n_clusters=n_clusters, ndims=ndims)
        x = reshape_data(x,cluster_labels)
        del kwargs['n_clusters']

        if 'point_colors' in kwargs:
            warnings.warn('n_clusters overrides point_colors, ignoring point_colors.')
            del kwargs['point_colors']

    ##STYLING##

    # handle point_colors flag
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

    # if x is not a list, make it one
    if type(x) is not list:
        x = [x]

    if animate:
        animated_plot(x,*args,**kwargs)
    else:
        static_plot(x,*args,**kwargs)
