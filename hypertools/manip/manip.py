# noinspection PyPackageRequirements
import datawrangler as dw

from .normalize import Normalize
from .resample import Resample
from .smooth import Smooth
from .zscore import ZScore
from .common import Manipulator

from ..core import get_default_options, get_model, apply_model
from ..core.shared import unpack_model


def manip(data, model='ZScore', **kwargs):
    manipulators = [Normalize, Resample, Smooth, ZScore]
    opts = dw.core.update_dict(get_default_options()['manip'], kwargs)
    opts['search'] = ['sklearn.preprocessing']

    model = unpack_model(model, valid=manipulators, parent_class=Manipulator)
    return apply_model(data, model, **opts)
