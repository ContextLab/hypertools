"""Shared Response Model aligners (Chen et al., 2015) as :class:`Aligner` children.

Adapters over the vendored ``hypertools.external.brainiak.{SRM, DetSRM, RSRM}``.
"""
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .common import Aligner

from ..external.brainiak import SRM, DetSRM, RSRM


def fitter(data, align_type, **kwargs):
    """Fit a Shared Response Model variant (SRM, DetSRM, or RSRM) on `data`.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Dataset(s) to fit the shared response model on (coerced to a
        list if a single DataFrame is given).
    align_type : type
        The vendored brainiak model class to instantiate (`SRM`,
        `DetSRM`, or `RSRM`).
    **kwargs
        `features` : int, optional number of shared features (default:
        the minimum column count across `data`). Remaining kwargs are
        ignored by this function.

    Returns
    -------
    dict
        `{'model': <fitted align_type instance>, 'features': features,
        'indices': [d.index for d in data]}`.
    """
    if not isinstance(data, list):
        data = [data]

    features = kwargs.pop('features', None)
    if features is None:
        features = np.min([d.shape[1] for d in data])

    model = align_type(features=features)
    model.fit([d.values.T for d in data])
    indices = [d.index for d in data]
    return {'model': model, 'features': features, 'indices': indices}


def transformer(data, **kwargs):
    """Apply a fitted Shared Response Model to `data`.

    Parameters
    ----------
    data : list of DataFrame
        Held-out (or fit-time) dataset(s) to project into the shared
        response space.
    **kwargs
        `model` : the fitted brainiak model instance (from `fitter`).

    Returns
    -------
    list of pandas.DataFrame
        The transformed dataset(s), one per entry in `data`, each
        indexed by the corresponding entry's own index (not the
        fit-time index -- see GH #227).

    Raises
    ------
    sklearn.exceptions.NotFittedError
        If `model` is missing/`None`.
    """
    model = kwargs.pop('model', None)
    if model is None:
        raise NotFittedError('aligner model must be fit before data can be transformed')

    # Build the output index from the INCOMING data (`data`), not the
    # fit-time `indices` -- the latter mislabels (or, on a row-count
    # mismatch, raises on) held-out data passed to `.transform()` (GH #227).
    return [pd.DataFrame(j.T, index=i.index) for i, j in zip(data, model.transform([i.values.T for i in data]))]


def srm_fitter(data, **kwargs):
    """Fit a standard `SRM` model on `data`. See `fitter`."""
    return fitter(data, SRM, **kwargs)


def detsrm_fitter(data, **kwargs):
    """Fit a `DetSRM` (deterministic SRM) model on `data`. See `fitter`."""
    return fitter(data, DetSRM, **kwargs)


def rsrm_fitter(data, **kwargs):
    """Fit an `RSRM` (robust SRM) model on `data`. See `fitter`."""
    return fitter(data, RSRM, **kwargs)


class SharedResponseModel(Aligner):
    """Shared Response Model (Chen et al., 2015).

    :param features: number of shared features (default: minimum number of
        columns across datasets).
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(required=['model', 'features', 'indices'],
                         fitter=srm_fitter, transformer=transformer,
                         data=None, features=features, **kwargs)


class DeterministicSharedResponseModel(Aligner):
    """Deterministic Shared Response Model (Chen et al., 2015).

    :param features: number of shared features (default: minimum number of
        columns across datasets).
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(required=['model', 'features', 'indices'],
                         fitter=detsrm_fitter, transformer=transformer,
                         data=None, features=features, **kwargs)


class RobustSharedResponseModel(Aligner):
    """Robust Shared Response Model (Turek et al., 2017).

    :param features: number of shared features (default: minimum number of
        columns across datasets).
    """
    def __init__(self, features=None, **kwargs):
        super().__init__(required=['model', 'features', 'indices'],
                         fitter=rsrm_fitter, transformer=transformer,
                         data=None, features=features, **kwargs)
