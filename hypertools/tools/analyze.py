from .reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer
from .._shared.helpers import format_data

def analyze(data, reduce=None, align=None, normalize=None, ndims=None, internal=False):

    # put data into common format
    data = format_data(data)

    # return processed data
    return aligner(reducer(normalizer(data, normalize=normalize, internal=internal),
                   reduce=reduce, ndims=ndims, internal=internal), align=align)
