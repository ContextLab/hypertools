from .common import Aligner, reject_unknown_kwargs


def fitter(*args, **kwargs):
    """No-op fitter: returns an empty parameter dict (nothing to fit)."""
    return {}


def transformer(data, **kwargs):
    """No-op transformer: returns `data` unchanged."""
    return data


class NullAlign(Aligner):
    """Returns the (trimmed + padded) dataset unchanged. Takes no parameters."""
    def __init__(self, **kwargs):
        reject_unknown_kwargs('NullAlign', kwargs, [])
        super().__init__(required=[], fitter=fitter, transformer=transformer,
                         data=None)
