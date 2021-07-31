import numpy as np
import scipy
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError

from ..external.brainiak import SRM, DetSRM, RSRM
from .common import Aligner


def fitter(data, align_type, **kwargs):
    if type(data) is not list:
        data = [data]

    features = kwargs.pop('features', None)
    if features is None:
        features = np.min([d.shape[1] for d in data])

    model = align_type(features=features)
    model.fit([d.T for d in data])
    return {'model': model, 'features': features}


def transformer(data, **kwargs):
    model = kwargs.pop('model', None)
    if model is None:
        raise NotFittedError('aligner model must be fit before data can be transformed')

    return [j.T for j in model.transform([i.T for i in data])]


def srm_fitter(data, **kwargs):
    return fitter(data, SRM, **kwargs)


def detsrm_fitter(data, **kwargs):
    return fitter(data, DetSRM, **kwargs)


def rsrm_fitter(data, **kwargs):
    return fitter(data, RSRM, **kwargs)


class SharedResponseModel(Aligner):
    def __init__(self, **kwargs):
        super().__init__(required=['model', 'features'], fitter=srm_fitter, transformer=transformer, **kwargs)


class DeterministicSharedResponseModel(Aligner):
    def __init__(self, **kwargs):
        super().__init__(required=['model', 'features'], fitter=detsrm_fitter, transformer=transformer, **kwargs)


class RobustSharedResponseModel(Aligner):
    def __init__(self, **kwargs):
        super().__init__(required=['model', 'features'], fitter=rsrm_fitter, transformer=transformer, **kwargs)
