# noinspection PyPackageRequirements
import datawrangler as dw

from .hyperalign import HyperAlign
from .null import NullAlign
from .procrustes import Procrustes
from .srm import SharedResponseModel, DeterministicSharedResponseModel, RobustSharedResponseModel
from .common import Aligner

from ..core import apply_model, has_all_attributes, unpack_model, get_default_options


@dw.decorate.funnel
def align(data, model='HyperAlign', **kwargs):
    """
    Align a datasets to itself or a supplied template, using the procrustean transformation, hyperalignment, or
    the shared response model

    Parameters
    ----------
    :param data: a hypertools-compatible dataset
    :param model: one of: 'HyperAlign' (default), 'SharedResponseModel', 'RobustSharedResponseModel',
      'DeterministicSharedResponseModel', or 'Procrustes'.  Aligner objects are also supported.  Models may also be
      supplied in dictionary form to modify their behaviors.  Lists of models (to be applied in sequence) are also
      supported.
    :param kwargs: keyword arguments are first passed to datawrangler.decorate.funnel, and any remaining arguments are
      passed to the appropriate Aligner object.

    Returns
    -------
    :returns: aligned data (as a DataFrame or a list of DataFrames)
    """
    aligners = [HyperAlign, SharedResponseModel, RobustSharedResponseModel,
                DeterministicSharedResponseModel, Procrustes]
    return apply_model(data, unpack_model(model, valid=aligners, parent_class=Aligner),
                       **dw.core.update_dict(get_default_options()['align'], kwargs))
