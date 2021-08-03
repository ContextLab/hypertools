# noinspection PyPackageRequirements
import datawrangler as dw

from ..core.model import apply_model


@dw.decorate.apply_unstacked
def align(data, model='HyperAlign', **kwargs):
    """
    Align a datasets to itself or a supplied template, using the procrustean transformation, hyperalignment, or
    the shared response model

    Parameters
    ----------
    :param data: a hypertools-compatible dataset
    :param model: one of: 'HyperAlign' (default), 'SharedResponseModel', 'RobustSharedResponseModel',
      'DeterministicSharedResponseModel', or 'Procrustes'.  Aligner objects are also supported.  Models may also be
      supplied in dictionary form to modify their behaviors.
    :param kwargs: keyword arguments are first passed to datawrangler.decorate.funnel, and any remaining arguments are
      passed to the appropriate Aligner object.

    Returns
    -------
    :returns: aligned data (as a DataFrame or a list of DataFrames)
    """
    return apply_model(data, model, search=['aligners'], **kwargs)
