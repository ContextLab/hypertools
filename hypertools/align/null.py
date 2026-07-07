from .common import Aligner


def fitter(*args, **kwargs):
    """No-op fitter: returns an empty parameter dict (nothing to fit)."""
    return {}


def transformer(data, **kwargs):
    """No-op transformer: returns `data` unchanged."""
    return data


class NullAlign(Aligner):
    """Returns the (trimmed + padded) dataset unchanged."""
    def __init__(self, **kwargs):
        super().__init__(required=[], fitter=fitter, transformer=transformer,
                         data=None, **kwargs)
