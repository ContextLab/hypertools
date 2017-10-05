from .tools.reduce import reduce as reducer
from .tools.align import align
from .tools.normalize import normalize
from ._shared.helpers import format_data

def analyze(data, reduce_model=None, reduce_params=None, align_model=None,
            align_params=None, normalize=False, ndims=None):

    # put data into common format
    data = format_data(data)

    return reducer(aligner(normalize(data, normalize=normalize),
                 model=align_model, model_params=align_params),
                 model=reduce_model, model_params=reduce_params, ndims=ndims)
