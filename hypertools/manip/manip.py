from .normalize import Normalize
from .resample import Resample
from .smooth import Smooth
from .zscore import ZScore
from .common import Manipulator

from ..core import get_default_options


def manip(data, model='ZScore', **kwargs):
    manipulators = [Normalize, Resample, Smooth, ZScore]
    return apply_model(data, unpack_model(model, valid=manipulators, parent_class=Manipulator),
                       **dw.core.update_dict(get_default_options()['manip'], kwargs))
