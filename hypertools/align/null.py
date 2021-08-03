from .common import Aligner


def fitter(data):
    return {}


def transformer(data, **kwargs):
    return data


class NullAlign(Aligner):
    """
    Base class for NullAlign objects.  Returns the original (unmodified) dataset after
    trimming and padding it.
    """
    def __init__(self):
        super().__init__(nrequired=[], fitter=fitter, transformer=transformer, data=None)
