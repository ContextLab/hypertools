"""Shared Response Model aligners (Chen et al., 2015) as :class:`Aligner` children.

Adapters over the vendored ``hypertools.external.brainiak.{SRM, DetSRM, RSRM}``.
"""
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .common import Aligner

from ..external.brainiak import SRM, DetSRM, RSRM


def fitter(data, align_type, **kwargs):
    if type(data) is not list:
        data = [data]

    features = kwargs.pop('features', None)
    if features is None:
        features = np.min([d.shape[1] for d in data])

    model = align_type(features=features)
    model.fit([d.values.T for d in data])
    indices = [d.index for d in data]
    return {'model': model, 'features': features, 'indices': indices}


def transformer(data, **kwargs):
    model = kwargs.pop('model', None)
    if model is None:
        raise NotFittedError('aligner model must be fit before data can be transformed')

    return [pd.DataFrame(j.T, index=i) for i, j in zip(kwargs['indices'], model.transform([i.values.T for i in data]))]


def srm_fitter(data, **kwargs):
    return fitter(data, SRM, **kwargs)


def detsrm_fitter(data, **kwargs):
    return fitter(data, DetSRM, **kwargs)


def rsrm_fitter(data, **kwargs):
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
