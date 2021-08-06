# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError

from .common import Aligner

from ..external.brainiak import SRM, DetSRM, RSRM
from ..core import get_default_options, eval_dict


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
    """
    Base class for SharedResponseModel objects (no parameters).
    """
    def __init__(self, **kwargs):
        opts = dw.core.update_dict(eval_dict(get_default_options()['SharedResponseModel']), kwargs)
        required = ['model', 'features', 'indices']
        super().__init__(required=required, **opts,
                         fitter=srm_fitter, transformer=transformer, data=None)

        for k, v in opts.items():
            setattr(self, k, v)
        self.required = required
        self.fitter = srm_fitter
        self.transformer = transformer
        self.data = None


class DeterministicSharedResponseModel(Aligner):
    """
    Base class for DeterministicSharedResponseModel objects (no parameters).
    """
    def __init__(self, **kwargs):
        opts = dw.core.update_dict(eval_dict(get_default_options()['DeterministicSharedResponseModel']), kwargs)
        required = ['model', 'features', 'indices']
        super().__init__(required=required, **opts,
                         fitter=detsrm_fitter, transformer=transformer, data=None)

        for k, v in opts.items():
            setattr(self, k, v)
        self.required = required
        self.fitter = detsrm_fitter
        self.transformer = transformer
        self.data = None


class RobustSharedResponseModel(Aligner):
    """
    Base class for RobustSharedResponseModel objects (no parameters).
    """

    def __init__(self, **kwargs):
        opts = dw.core.update_dict(eval_dict(get_default_options()['RobustSharedResponseModel']), kwargs)
        required = ['model', 'features', 'indices']
        super().__init__(required=required, **opts,
                         fitter=rsrm_fitter, transformer=transformer, data=None)

        for k, v in opts.items():
            setattr(self, k, v)
        self.required = required
        self.fitter = rsrm_fitter
        self.transformer = transformer
        self.data = None
