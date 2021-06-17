import pandas as pd
from format import is_multiindex_dataframe, is_dataframe
from formats.dataframe import is_pandas, wrangle_pandas
from formats.array import is_numpy, wrangle_numpy
from formats.image import is_image, wrangle_image
from formats.sound import is_sound, wrangle_sound
from formats.nifti import is_nifti, wrangle_nifti
from formats.text import is_text, wrangle_text
from formats.null import is_null, wrangle_null

#the order matters: if earlier checks pass, later checks will not run.
#the list specifies the priority of converting to the given datatypes.
format_checkers = ['pandas', 'numpy', 'image', 'sound', 'nifti', 'text', 'null']

def HyperData(pd.DataFrame):
    def __init__(self, data, **kwargs):
        for fc in format_checkers:
            if eval(f'is_{fc}(data)'):
                self.dtype = fc
                self.df = eval(f'{wrangle_{fc}(data, **kwargs)')
        self.dtype = None
        self.df = None

    def unstack(self, inplace=False):
        if not is_multiindex_dataframe(self.df):
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
        else:
            return HyperData(data)

    def stack(self, inplace=False):
        pass

    def trajectorize(self,  window_length=0.1, samplerate=None, inplace=False):
        pass