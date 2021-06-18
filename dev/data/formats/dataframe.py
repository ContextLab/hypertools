import numpy as np
import pandas as pd
import modin
from array import is_array


def df_like(x):
    required_attributes = ['values', 'index', 'columns', 'shape', 'stack', 'unstack', 'loc', 'iloc', 'size', 'copy',
                           'head', 'tail', 'lat', 'at', 'items', 'iteritems', 'keys', 'iterrows', 'itertuples',
                           'get', 'isin', 'where', 'query', 'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod',
                           'pow', 'dot', 'radd', 'rsub', 'rmul', 'rdiv', 'rtruediv', 'rfloordiv', 'rmod', 'rpow',
                           'lt', 'gt', 'le', 'ge', 'ne', 'eq', 'apply', 'groupby', 'rolling', 'expanding', 'abs',
                           'filter', 'drop', 'drop_duplicates', 'backfill', 'bfill', 'ffill', 'fillna', 'interpolate',
                           'pad', 'droplevel', 'pivot', 'pivot_table', 'squeeze', 'melt', 'join', 'merge']
    for r in required_attributes:
        if not hasattr(x, r):
            print(f'missing method: {r}')
            return False
    return True


def array_like(x):
    return is_array(x) or is_dataframe(x) or (type(x) in [list, np.array, np.ndarray, pd.Series, modin.Series])


def is_dataframe(x):
    if type(x).__module__ in ['pandas.core.frame', 'modin.pandas.dataframe']:
        return True
    elif df_like(x):
        return True
    else:
        return False


def is_multiindex_dataframe(x):
    return is_dataframe(x) and ('indexes.multi' in type(x.index).__module__)


def wrangle_dataframe(data, return_model=False, **kwargs):
    if return_model:
        return pd.DataFrame(data, **kwargs), {'model': pd.DataFrame, 'args': [], 'kwargs': kwargs}
    return pd.DataFrame(data, **kwargs)
