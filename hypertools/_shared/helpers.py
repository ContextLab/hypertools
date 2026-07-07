#!/usr/bin/env python

"""
Helper functions
"""

##PACKAGES##
import sys
import numpy as np
import itertools
import pandas as pd
from matplotlib.lines import Line2D

# NOTE: seaborn and scipy.interpolate are imported lazily inside the functions
# that use them -- together they added ~1s to `import hypertools`.
np.seterr(divide='ignore', invalid='ignore')


def center(x):
    assert isinstance(x, list), "Input data to center must be list"
    x_stacked = np.vstack(x)
    return [i - np.mean(x_stacked, 0) for i in x]


def scale(x):
    assert isinstance(x, list), "Input data to scale must be list"
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
    import seaborn as sns
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
    from scipy.interpolate import PchipInterpolator as pchip
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
    """True if the format string draws pure lines (no markers).

    Notes: linestyle tokens are stripped BEFORE checking for marker
    characters so that '-.' (dash-dot) is recognized as a line rather than
    a '.' marker, mirroring matplotlib's own fmt grammar. The "no marker"
    sentinel keys ('', ' ', 'None', 'none') are excluded from the marker
    set -- '' is a substring of every string, which previously made this
    function return False for ALL format strings (silently disabling line
    interpolation on matplotlib versions whose Line2D.markers includes '').
    """
    if isinstance(format_str, np.bytes_):
        format_str = format_str.decode('utf-8')
    if format_str is None:
        return True
    if isinstance(format_str, (list, tuple, np.ndarray)):
        return all(is_line(f) for f in format_str)
    remainder = format_str
    for linestyle in ('-.', '--', '-', ':'):  # two-char styles first
        remainder = remainder.replace(linestyle, '')
    markers = [str(symbol) for symbol in Line2D.markers.keys()
               if str(symbol) not in ('', ' ', 'None', 'none')]

    return all(symbol not in remainder for symbol in markers)


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
