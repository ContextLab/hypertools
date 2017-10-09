from .reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer
from .._shared.helpers import format_data

def analyze(data, reduce=None, align=None, normalize=None, ndims=None, internal=False):
    """
    Wrapper function for normalize -> reduce -> align transformations.

    Parameters
    ----------

    data : numpy array, pandas df, or list of arrays/dfs
        The data to analyze

    reduce : str or dict
    """


    # return processed data
    return aligner(reducer(normalizer(data, normalize=normalize, internal=internal),
                   reduce=reduce, ndims=ndims, internal=internal), align=align)
