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
import six
import copy
from scipy.interpolate import PchipInterpolator as pchip
import seaborn as sns
import itertools
import pandas as pd
from matplotlib.lines import Line2D
from .._externals.ppca import PPCA
np.seterr(divide='ignore', invalid='ignore')

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
    f = lambda x: 2*(np.divide(x - m1, m2)) - 1
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

# def check_data(data):
#     if type(data) is list:
#         if all([isinstance(x, np.ndarray) for x in data]):
#             return 'list'
#         elif all([isinstance(x, pd.DataFrame) for x in data]):
#             return 'dflist'
#         elif all([isinstance(x, str) for x in data]):
#                 return 'text'
#         elif isinstance(data[0], collections.Iterable):
#             if all([isinstance(x, str) for x in data[0]]):
#                     return 'text'
#         else:
#             raise ValueError("Data must be numpy array, list of numpy array, pandas dataframe or list of pandas dataframes.")
#     elif isinstance(data, np.ndarray):
#         return 'array'
#     elif isinstance(data, pd.DataFrame):
#         return 'df'
#     else:
#         raise ValueError("Data must be numpy array, list of numpy array, pandas dataframe or list of pandas dataframes.")

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

def reshape_data(x,labels):
    categories = list(sorted(set(labels), key=list(labels).index))
    x_stacked = np.vstack(x)
    x_reshaped = [[] for i in categories]
    for idx,point in enumerate(labels):
        x_reshaped[categories.index(point)].append(x_stacked[idx])
    return [np.vstack(i) for i in x_reshaped]

def patch_lines(x):
    """
    Draw lines between groups
    """
    for idx in range(len(x)-1):
        x[idx] = np.vstack([x[idx], x[idx+1][0,:]])
    return x

def is_line(format_str):
    if isinstance(format_str, np.bytes_):
        format_str = format_str.decode('utf-8')
    markers = list(map(lambda x: str(x), Line2D.markers.keys()))

    return (format_str is None) or (all([str(symbol) not in format_str for symbol in markers]))

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

def get_type(data):
    """
    Checks what the data type is and returns it as a string label
    """
    import six
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        if isinstance(data[0], (six.string_types, six.text_type, six.binary_type)):
            return 'list_str'
        elif isinstance(data[0], (int, float)):
            return 'list_num'
        elif isinstance(data[0], np.ndarray):
            return 'list_arr'
        else:
            raise TypeError('Unsupported data type passed. Supported types: '
                            'Numpy Array, Pandas DataFrame, String, List of strings'
                            ', List of numbers')
    elif isinstance(data, np.ndarray):
        if isinstance(data[0][0], (six.string_types, six.text_type, six.binary_type)):
            return 'arr_str'
        else:
            return 'arr_num'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (six.string_types, six.text_type, six.binary_type)):
        return 'str'
    elif isinstance(data, DataGeometry):
        return 'geo'
    else:
        raise TypeError('Unsupported data type passed. Supported types: '
                        'Numpy Array, Pandas DataFrame, String, List of strings'
                        ', List of numbers')

def convert_text(data):
    dtype = get_type(data)
    if dtype in ['list_str', 'str']:
        data = np.array(data).reshape(-1, 1)
    return data

def check_geo(geo):
    """ Checks a geo and makes sure the text fields are not binary """
    geo = copy.copy(geo)
    def fix_item(item):
        if isinstance(item, six.binary_type):
            return item.decode()
        return item
    def fix_list(lst):
        return [fix_item(i) for i in lst]
    if isinstance(geo.reduce, six.binary_type):
        geo.reduce = geo.reduce.decode()
    for key in geo.kwargs.keys():
        if geo.kwargs[key] is not None:
            if isinstance(geo.kwargs[key], (list, np.ndarray)):
                geo.kwargs[key] = fix_list(geo.kwargs[key])
            elif isinstance(geo.kwargs[key], six.binary_type):
                geo.kwargs[key] = fix_item(geo.kwargs[key])
    return geo

def get_dtype(data):
    """
    Checks what the data type is and returns it as a string label
    """
    import six
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        return 'list'
    elif isinstance(data, np.ndarray):
        return 'arr'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (six.string_types, six.text_type, six.binary_type)):
        return 'str'
    elif isinstance(data, DataGeometry):
        return 'geo'
    else:
        raise TypeError('Unsupported data type passed. Supported types: '
                        'Numpy Array, Pandas DataFrame, String, List of strings'
                        ', List of numbers')
