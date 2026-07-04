from .common import Aligner


def fitter(*args, **kwargs):
    return {}


def transformer(data, **kwargs):
    return data


class NullAlign(Aligner):
    """Returns the (trimmed + padded) dataset unchanged."""
    def __init__(self, **kwargs):
        super().__init__(required=[], fitter=fitter, transformer=transformer,
                         data=None, **kwargs)
