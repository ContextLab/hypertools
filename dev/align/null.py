from .common import Aligner


def fitter(data):
    return {}


def transformer(data, **kwargs):
    return data


class NullAlign(Aligner):
    def __init__(self):
        super().__init__(nrequired=[], fitter=fitter, transformer=transformer, data=None)
