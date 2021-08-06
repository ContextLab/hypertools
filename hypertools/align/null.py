# noinspection PyPackageRequirements
import datawrangler as dw

from .common import Aligner

from ..core import get_default_options, eval_dict


# noinspection PyUnusedLocal
def fitter(*args, **kwargs):
    return {}


# noinspection PyUnusedLocal
def transformer(data, **kwargs):
    return data


class NullAlign(Aligner):
    """
    Base class for NullAlign objects.  Returns the original (unmodified) dataset after
    trimming and padding it.
    """
    def __init__(self, **kwargs):
        opts = dw.core.update_dict(eval_dict(get_default_options()['NullAlign']), kwargs)
        required = []
        super().__init__(required=required, fitter=fitter, transformer=transformer, data=None)

        for k, v in opts.items():
            setattr(self, k, v)
        self.required = required
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
