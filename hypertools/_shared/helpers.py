#!/usr/bin/env python

"""
Helper functions
"""

##PACKAGES##
from __future__ import division
from __future__ import print_function
import sys
import warnings
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import seaborn as sns
import itertools
import pandas as pd
from matplotlib.lines import Line2D
from .._externals.ppca import PPCA
##HELPER FUNCTIONS##
def center(x):
    assert type(x) is list, "Input data to center must be list"
    x_stacked = np.vstack(x)
    return [i - np.mean(x_stacked, 0) for i in x]

def scale(x):
    assert type(x) is list, "Input data to scale must be list"
    x_stacked = np.vstack(x)
    m1 = np.min(x_stacked)
    m2 = np.max(x_stacked - m1)
    f = lambda x: 2*((x - m1) / m2) - 1
    return [f(i) for i in x]

def group_by_category(vals):
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))
    val_set = list(sorted(set(vals), key=list(vals).index))
    return [val_set.index(val) for val in vals]

def vals2colors(vals,cmap='GnBu_d',res=100):
    """Maps values to colors
    Args:
    values (list or list of lists) - list of values to map to colors
    cmap (str) - color map (default is 'husl')
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of rgb tuples
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))

    # get palette from seaborn
    palette = np.array(sns.color_palette(cmap, res))
    ranks = np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1
    return [tuple(i) for i in palette[ranks, :]]

def vals2bins(vals,res=100):
    """Maps values to bins
    Args:
    values (list or list of lists) - list of values to map to colors
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of numbers representing bins
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))
    return list(np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1)

def interp_array(arr,interp_val=10):
    x=np.arange(0, len(arr), 1)
    xx=np.arange(0, len(arr)-1, 1/interp_val)
    q=pchip(x,arr)
    return q(xx)

def interp_array_list(arr_list,interp_val=10):
    smoothed= [np.zeros(arr_list[0].shape) for item in arr_list]
    for idx,arr in enumerate(arr_list):
        smoothed[idx] = interp_array(arr,interp_val)
    return smoothed

def check_data(data):
    if type(data) is list:
        if all([isinstance(x, np.ndarray) for x in data]):
            return 'list'
        elif all([isinstance(x, pd.DataFrame) for x in data]):
            return 'dflist'
        else:
            raise ValueError("Data must be numpy array, list of numpy array, pandas dataframe or list of pandas dataframes.")
    elif isinstance(data, np.ndarray):
        return 'array'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    else:
        raise ValueError("Data must be numpy array, list of numpy array, pandas dataframe or list of pandas dataframes.")

def parse_args(x,args):
    args_list = []
    for i,item in enumerate(x):
        tmp = []
        for ii,arg in enumerate(args):
            if type(arg) is tuple or type(arg) is list:
                if len(arg) == len(x):
                    tmp.append(arg[i])
                else:
                    print('Error: arguments must be a list of the same length as x')
                    sys.exit(1)
            else:
                tmp.append(arg)
        args_list.append(tuple(tmp))
    return args_list

def parse_kwargs(x, kwargs):
    kwargs_list = []
    for i,item in enumerate(x):
        tmp = {}
        for kwarg in kwargs:
            if type(kwargs[kwarg]) is tuple or type(kwargs[kwarg]) is list:
                if len(kwargs[kwarg]) == len(x):
                    tmp[kwarg]=kwargs[kwarg][i]
                else:
                    # print('Error: keyword arguments must be a list of the same length as x')
                    # sys.exit(1)
                    tmp[kwarg]=None
            else:
                tmp[kwarg]=kwargs[kwarg]
        kwargs_list.append(tmp)
    return kwargs_list

def reshape_data(x,labels,comparison_key=None):
    if comparison_key==None:
        comparison_key=list(labels).index
    categories = list(sorted(set(labels), key=comparison_key))
    x_stacked = np.vstack(x)
    x_reshaped = [[] for i in categories]
    for idx,point in enumerate(labels):
        x_reshaped[categories.index(point)].append(x_stacked[idx])
    return [np.vstack(i) for i in x_reshaped]

def is_label_probabilistic(labels):
    """
        Tells Whether the Assigned Label is a Scalar or Not[Case for GaussianMixture models]
    """
    if type(labels[0]) is np.ndarray or type(labels) is pd.DataFrame or type(labels[0]) is list:
        return True
    return False
def reshape_labels(labels):
    """
        Since, Most of the functionalities in Hypertool work with Scalar(To be Precise, Categorical in nature) Value of Label.
        This method assigns the label/cluster to which the data point(s) most probably belong.  
    """
    if labels==None or len(labels)<1:
        return labels
    return [label.argmax() for label in labels]

def labels2RGB(n_clusters,cluster_colors,cluster_discrete_labels,cluster_labels):
    clustered_labels=[ [] for _ in range(n_clusters)]
    for i in range(len(cluster_labels)):
        clustered_labels[cluster_discrete_labels[i]].append(cluster_labels[i])#Breaks the cluster_lables into n_cluster Groups
    rgbLabels=[]
    #Below rgbLabels is weighted sum by considering the confidence which point belongs to various clusters and the RGB color of those clusters.
    for i in range(n_clusters):
        rgbLabels.append(list(map(lambda x, c=cluster_colors: np.dot(x[:n_clusters], c)/np.sum(x[:n_clusters]), clustered_labels[i])))
    return rgbLabels

def bin_cluster_colors(x,labels):
    categories = list(sorted(set(labels)))
    x_stacked = np.vstack(x)
    x_reshaped = [[] for i in categories]
    for idx,point in enumerate(labels):
        x_reshaped[categories.index(point)].append(x_stacked[idx])
    return {ind:np.apply_over_axes(np.average,np.vstack(i),[0])[0] for ind,i in enumerate(x_reshaped)}

def compute_bins_resolution(resolution,clustered_x):
    cluster_cnts=np.array([len(cluster) for cluster in clustered_x])
    return np.minimum(np.ceil(cluster_cnts*(resolution/np.sum(cluster_cnts))),cluster_cnts)

def format_data(x, ppca=False):
    """
    Formats data into a list of numpy arrays

    This function is useful to identify rows of your array that contain missing
    data or nans.  The returned indices can be used to remove the rows with
    missing data, or label the missing data points that are interpolated
    using PPCA.

    Parameters
    ----------

    x : numpy array, dataframe or (mixed) list
        The data to convert

    ppca : bool
        Performs PPCA to fill in missing values (default: False)

    Returns
    ----------
    data : list of numpy arrays
        A list of formatted arrays
    """

    def fill_missing(x):

        # ppca if missing data
        m = PPCA()
        m.fit(data=np.vstack(x))
        x_pca = m.transform()

        # if the whole row is missing, return nans
        all_missing = [idx for idx,a in enumerate(np.vstack(x)) if all([type(b)==np.nan for b in a])]
        if len(all_missing)>0:
            for i in all_missing:
                x_pca[i,:]=np.nan

        # get the original lists back
        if len(x)>1:
            x_split = np.cumsum([i.shape[0] for i in x][:-1])
            return list(np.split(x_pca,x_split,axis=0))
        else:
            return [x_pca]

    # not sure why i needed to import here, but its the only way I could get it to work
    from ..tools.df2mat import df2mat

    if type(x) is not list:
        x = [x]

    formatted_data = []

    for xi in x:

        data_type = check_data(xi)

        if data_type=='df':
            xi = df2mat(xi)

        formatted_data.append(xi)

    if any([i.ndim==1 for i in formatted_data]):
        formatted_data = [np.reshape(i,(i.shape[0],1)) if i.ndim==1 else i for i in formatted_data]

    # if there are any nans in any of the lists, use ppca
    if ppca is True:
        if np.isnan(np.vstack(formatted_data)).any():
            warnings.warn('Missing data: Inexact solution computed with PPCA (see https://github.com/allentran/pca-magic for details)')
            formatted_data = fill_missing(formatted_data)

    return formatted_data

def patch_lines(x):
    """
    Draw lines between groups
    """
    for idx in range(len(x)-1):
        x[idx] = np.vstack([x[idx], x[idx+1][0,:]])
    return x

def is_line(format_str):
    return (format_str is None) or (all([str(symbol) not in format_str for symbol in Line2D.markers.keys()]))

import collections
import functools

def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer
