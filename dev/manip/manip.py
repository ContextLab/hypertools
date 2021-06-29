import warnings
import numpy as np
import pandas as pd

from ..decorate import funnel, apply_defaults
from ..data.formats.dataframe import is_multiindex_dataframe, is_dataframe

# import all hypertools functions supported by the pipeline

# manipulators
from normalize import normalize
from resample import resample
from smooth import smooth
from zscore import zscore
from ztransform import ztransform

# align, reduce, cluster,
from ..align.align import align
from ..reduce.reduce import reduce
from ..cluster.cluster import cluster


def unstack_namer(names, grouper='ID'):
    if not (grouper in names):
        names[0] = grouper
    elif not (names[0] == grouper):
        # trying n things other than 'ID'; at least one of them must be outside of the n-1 remaining names
        for i in np.arange(
                len(names)):
            next_grouper = f'{grouper}{i}'
            if not (next_grouper in names):
                names[0] = next_grouper
                grouper = next_grouper
                break
    assert names[0] == grouper, 'Unstacking error'
    return names, grouper


def pandas_unstack(x):
    if not is_multiindex_dataframe(x):
        if is_dataframe(x):
            return x
        else:
            raise Exception(f'Unsupported datatype: {type(x)}')

    names, grouper = unstack_namer(list(x.index.names))
    x.index.rename(names, inplace=True)
    unstacked = [d[1].set_index(d[1].index.get_level_values(1)) for d in list(x.groupby(grouper))]
    if len(unstacked) == 1:
        return unstacked[0]
    else:
        return unstacked


# noinspection PyUnusedLocal
@funnel
def pandas_stack(data, names=None, keys=None, verify_integrity=False, sort=False, copy=True, ignore_index=False,
                 levels=None, **kwargs):
    """
    Take a list of DataFrames with the same number of columns and (optionally)
    a list of names (of the same length as the original list; default:
    range(len(x))).  Return a single MultiIndex DataFrame where the original
    DataFrames are stacked vertically, with the data names as their level 1
    indices and their original indices as their level 2 indices.

    INPUTS
    data: data in any format (text, numpy arrays, pandas dataframes, or a mixed list (or nested lists) of those types)
    text_vectorizer: function that takes a string (or list of strings) and returns a numpy array or dataframe.  If
    force is False, must pass in a list of DataFrames.

    force: if True, use format_data to coerce everything into a list of pandas dataframes.

    text_vectorizer: function for turning text data into DataFrames, used if force is True

    Also takes all keyword arguments from pandas.concat except axis, join, join_axes

    All other keyword arguments (if any) are passed to text_vectorizer

    OUTPUTS
    a single MultiIndex DataFrame
    """
    if is_multiindex_dataframe(data):
        return data
    elif is_dataframe(data):
        data = [data]
    elif len(data) == 0:
        return None

    assert len(np.unique([d.shape[1] for d in data])) == 1, 'All DataFrames must have the same number of columns'
    for i, d1 in enumerate(data):
        template = d1.columns.values
        for d2 in data[(i + 1):]:
            assert np.all([(c in template) for c in d2.columns.values]), 'All DataFrames must have the same columns'

    if keys is None:
        keys = np.arange(len(data), dtype=int)

    assert is_array(keys) or (type(keys) == list), f'keys must be None or a list or array of length len(data)'
    assert len(keys) == len(data), f'keys must be None or a list or array of length len(data)'

    if names is None:
        names = ['ID', *[f'ID{i}' for i in range(1, len(data[0].index.names))], None]

    return pd.concat(data, axis=0, join='outer', names=names, keys=keys,
                     verify_integrity=verify_integrity, sort=sort, copy=copy,
                     ignore_index=ignore_index, levels=levels)


def pandas_flatten(x):
    while len(x.index.names) > 1:
        x[x.index.names[0]] = x.index.get_level_values(0)

        if len(x.index.names) > 2:
            index = pd.MultiIndex.from_arrays([x.index.get_level_values(i) for i in range(1, len(x.index.levels))],
                                              names=x.index.names[1:])
        else:
            index = pd.Index(data=x.index.get_level_values(1), name=x.index.names[1])

        x.index = index
    return x


@funnel
def pipeline(x, ops=None):
    def str2fun(f):
        if callable(f):
            return f
        elif type(f) is str:
            try:
                return eval(f)
            except NameError:
                warning.warn(f'skipping unknown manipulation: {op}')

    if ops is None:
        ops = []

    if type(ops) is not list:
        ops = [ops]

    for op in ops:
        if callable(op) or (type(op) is str):
            op = str2fun(op)

            if op is None:
                continue

            x = apply_defaults(op(x))
        elif type(op) is dict:
            # noinspection PyArgumentList
            model = str2fun(op.pop('model', None))
            # noinspection PyArgumentList
            args = op.pop('args', None)
            # noinspection PyArgumentList
            kwargs = op.pop('kwargs', None)

            if args is None:
                args = []

            if kwargs is None:
                kwargs = {}

            if model is None:
                warning.warn(f'skipping unknown manipulation: {{**op, "args": args, "kwargs": kwargs}}')
                continue

            x = apply_defaults(model(x, *args, **kwargs))
    return x
