import numpy as np
import pandas as pd
from format import is_multiindex_dataframe
from formats.dataframe import is_dataframe, wrangle_dataframe
from formats.array import is_array, wrangle_array
from formats.image import is_image, wrangle_image
from formats.sound import is_sound, wrangle_sound
from formats.nifti import is_nifti, wrangle_nifti
from formats.text import is_text, wrangle_text
from formats.null import is_null, wrangle_null

#the order matters: if earlier checks pass, later checks will not run.
#the list specifies the priority of converting to the given datatypes.
format_checkers = ['pandas', 'numpy', 'image', 'sound', 'nifti', 'text', 'null']

def HyperData(pd.DataFrame):
    def __init__(self, data, wrangler=None, dtype=None, **kwargs):
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
    
