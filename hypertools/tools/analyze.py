from .reduce import reduce as reducer
from .align import align as aligner
from .normalize import normalize as normalizer
from .._shared.helpers import format_data

def analyze(data, reduce_model=None, reduce_params=None, align_model=None,
            align_params=None, normalize=False, ndims=None):

    # put data into common format
    data = format_data(data)

    # return processed data
    return aligner(reducer(normalizer(data, normalize=normalize),
                   model=reduce_model, model_params=reduce_params, ndims=ndims),
                   model=align_model, model_params=align_params)
