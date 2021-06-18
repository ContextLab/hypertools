import numpy as np
import pandas as pd

from ..core.configurator import __version__
from ..decorate import interpolate, list_generalizer


def pandas_unstack(x):
    if not is_multiindex_dataframe(x):
        if is_dataframe(x):
            return x
        else:
            raise Exception(f'Unsupported datatype: {type(x)}')

    names = list(x.index.names)
    grouper = 'ID'
    if not (grouper in names):
        names[0] = grouper
    elif not (names[0] == grouper):
        for i in np.arange(
                len(names)):  # trying n things other than 'ID'; at least one of them must be outside of the n-1 remaining names
            next_grouper = f'{grouper}{i}'
            if not (next_grouper in names):
                names[0] = next_grouper
                grouper = next_grouper
                break
    assert names[0] == grouper, 'Unstacking error'

    x.index.rename(names, inplace=True)
    unstacked = [d[1].set_index(d[1].index.get_level_values(1)) for d in list(x.groupby(grouper))]
    if len(unstacked) == 1:
        return unstacked[0]
    else:
        return unstacked


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


@interpolate
def format_interp_stack_extract(data, keys=None, **kwargs):
    stacked_data = pandas_stack(data, keys=keys)
    vals = stacked_data.values
    return vals, stacked_data

class HyperData(pd.DataFrame):
    def __init__(self, data, wrangler=None, dtype=None, index=None, columns=None, copy=False, **kwargs):
        for k, v in kwargs.items():
            assert k not in ['df', '__version__', 'stacked'], RuntimeError(f'Cannot set reserved property: {k}')
            self.k = v

        self.df = None
        self.dtype = None

        if wrangler is not None:
            self.df = wrangler(data)
            self.dtype = dtype
        else:
            for fc in format_checkers:
                if eval(f'is_{fc}(data)'):
                    self.dtype = fc
                    self.df = eval(f'{wrangle_{fc}(data, **kwargs)')

        if is_multiindex_dataframe(self.df):
            self.stacked = True
        else:
            self.stacked = False

        self.version = __version__





    def unstack(self, inplace=False):
        if not self.stacked:
            return self.df
        elif not is_multiindex_dataframe(self.df):
            if is_dataframe(self.df):
                return self.df
            else:
                raise Exception(f'Unsupported datatype: {type(x)}')

        names = list(self.df.index.names)
        grouper = 'ID'
        if not (grouper in names):
            names[0] = grouper
        elif not (names[0] == grouper):
            for i in np.arange(
                    len(names)):  # trying n things other than 'ID'; at least one of them must be outside of the n-1 remaining names
                next_grouper = f'{grouper}{i}'
                if not (next_grouper in names):
                    names[0] = next_grouper
                    grouper = next_grouper
                    break
        assert names[0] == grouper, 'Unstacking error'

        self.df.index.rename(names, inplace=True)
        unstacked = [d[1].set_index(d[1].index.get_level_values(1)) for d in list(x.groupby(grouper))]
        if len(unstacked) == 1:
            data = unstacked[0]
        else:
            data = unstacked

        if inplace:
            self.df = data
            self.stacked = False
        else:
            return HyperData(data)

    def stack(self, newdata=None, inplace=False, names=None, keys=None, verify_integrity=False, sort=False, copy=True, ignore_index=False, levels=None):
        if self.stacked:
            if (newdata is None) and (not inplace):
                return self.df
        else:
            if newdata is None:
                data = self.df
            else:
                data = [*self.df *newdata]

        if is_multiindex_dataframe(data):
            if inplace:
                return data
            else:
                self.stacked = True
                return None

        elif is_dataframe(data):
            data = [data]
        elif len(data) == 0:
            return None

        assert len(
            np.unique([d.shape[1] for d in data])) == 1, 'All DataFrames must have the same number of columns'

        for i, d1 in enumerate(data):
            template = d1.columns.values
            for d2 in data[(i + 1):]:
                assert np.all(
                    [(c in template) for c in d2.columns.values]), 'All DataFrames must have the same columns'

        if keys is None:
            keys = np.arange(len(data), dtype=int)

        assert is_array(keys) or (type(keys) == list), f'keys must be None or a list or array of length len(data)'
        assert len(keys) == len(data), f'keys must be None or a list or array of length len(data)'

        if names is None:
            names = ['ID', *[f'ID{i}' for i in range(1, len(data[0].index.names))], None]

        stacked = pd.concat(data, axis=0, join='outer', names=names, keys=keys,
                            verify_integrity=verify_integrity, sort=sort, copy=copy,
                            ignore_index=ignore_index, levels=levels)

        if inplace:
            self.df = stacked
            self.stacked = True
        else:
            return stacked

    def trajectorize(self,  window_length=0.1, samplerate=None, inplace=False):
        pass #convert to sliding windows

    def align(self, template=None, inplace=False, **kwargs):
        pass

    def manip(self, inplace=False, **kwargs):
        pass

    def reduce(self, inplace=False, **kwargs):
        pass

    def cluster(self, **kwargs):
        pass

    def plot(self, **kwargs):
        pass

    def save(self, fname, **kwargs):
        pass

