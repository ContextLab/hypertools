# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np

from .srm import SharedResponseModel, DeterministicSharedResponseModel, RobustSharedResponseModel
from .procrustes import Procrustes
from .hyperalign import Hyperalign
from .null import NullAlign

from ..core.model import apply_model


# TODO: consider limiting to only alignment (or custom) models?
@dw.decorate.apply_unstacked
def align(data, model='HyperAlign', **kwargs):
    return apply_model(data, model, **kwargs)  # FIXME: may need to import align models into model.py for this to work
