#!/usr/bin/env python

"""
Helper functions
"""

##PACKAGES##
import functools
import sys
import numpy as np
import copy
from scipy.interpolate import PchipInterpolator as pchip
import seaborn as sns
import itertools
import pandas as pd
from matplotlib.lines import Line2D
np.seterr(divide='ignore', invalid='ignore')


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


def vals2colors(vals, cmap='GnBu',res=100):
    """Maps values to colors
    Args:
    values (list or list of lists) - list of values to map to colors
    cmap (str) - color map (default is 'GnBu')
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


def parse_args(x,args):
    args_list = []
    for i,item in enumerate(x):
        tmp = []
        for ii, arg in enumerate(args):
            if isinstance(arg, (tuple, list)):
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
            if isinstance(kwargs[kwarg], (tuple, list)):
                if len(kwargs[kwarg]) == len(x):
                    tmp[kwarg]=kwargs[kwarg][i]
                else:
                    tmp[kwarg] = None
            else:
                tmp[kwarg]=kwargs[kwarg]
        kwargs_list.append(tmp)
    return kwargs_list


def reshape_data(x, hue, labels):
    categories = list(sorted(set(hue), key=list(hue).index))
    x_stacked = np.vstack(x)
    x_reshaped = [[] for _ in categories]
    labels_reshaped = [[] for _ in categories]
    if labels is None:
        labels = [None]*len(hue)
    for idx, (point, label) in enumerate(zip(hue, labels)):
        x_reshaped[categories.index(point)].append(x_stacked[idx])
        labels_reshaped[categories.index(point)].append(labels[idx])
    return [np.vstack(i) for i in x_reshaped], labels_reshaped


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
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        if isinstance(data[0], (str, bytes)):
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
        if isinstance(data[0][0], (str, bytes)):
            return 'arr_str'
        else:
            return 'arr_num'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (str, bytes)):
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
        if isinstance(item, bytes):
            return item.decode()
        return item

    def fix_list(lst):
        return [fix_item(i) for i in lst]
    if isinstance(geo.reduce, bytes):
        geo.reduce = geo.reduce.decode()
    for key in geo.kwargs.keys():
        if geo.kwargs[key] is not None:
            if isinstance(geo.kwargs[key], (list, np.ndarray)):
                geo.kwargs[key] = fix_list(geo.kwargs[key])
            elif isinstance(geo.kwargs[key], bytes):
                geo.kwargs[key] = fix_item(geo.kwargs[key])
    return geo


def get_dtype(data):
    """
    Checks what the data type is and returns it as a string label
    """
    from ..datageometry import DataGeometry

    if isinstance(data, list):
        return 'list'
    elif isinstance(data, np.ndarray):
        return 'arr'
    elif isinstance(data, pd.DataFrame):
        return 'df'
    elif isinstance(data, (str, bytes)):
        return 'str'
    elif isinstance(data, DataGeometry):
        return 'geo'
    else:
        raise TypeError('Unsupported data type passed. Supported types: '
                        'Numpy Array, Pandas DataFrame, String, List of strings'
                        ', List of numbers')
