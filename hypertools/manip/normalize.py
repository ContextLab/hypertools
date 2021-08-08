# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.apply_stacked
def fitter(data, axis=0, min=0, max=1):
    assert min < max, ValueError('minimum must be strictly less than maximum')

    if axis == 1:
        return dw.core.update_dict(fitter(data.T, axis=0, min=min, max=max), {'transpose': True})
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

    baseline = pd.Series(index=data.columns)
    peak = pd.Series(index=data.columns)

    z = data.copy()
    for c in z.columns:
        baseline[c] = z[c].min(axis=0)
        z[c] -= baseline[c]

        peak[c] = z[c].max(axis=0)

    return {'baseline': baseline, 'peak': peak, 'axis': axis, 'transpose': False, 'min': min, 'max': max}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    z = data.copy()
    for c in z.columns:
        z[c] -= kwargs['baseline'][c]
        z[c] /= kwargs['peak'][c]

    z *= (kwargs['max'] - kwargs['min'])
    z += kwargs['min']
    return z


class Normalize(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, min=0, max=1, axis=0):
        required = ['min', 'max', 'transpose', 'baseline', 'peak', 'axis']
        super().__init__(min=min, max=max, axis=axis, fitter=fitter, transformer=transformer, data=None,
                         required=required)

        self.min = min
        self.max = max
        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
