# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter

import warnings

from .common import Manipulator


@dw.decorate.apply_stacked
def fitter(data, **kwargs):
    data_max = data.max(axis=kwargs['axis'])
    data_min = data.min(axis=kwargs['axis'])

    return {'axis': kwargs['axis'], 'kernel_width': kwargs['kernel_width'], 'order': kwargs['order'], 'max': data_max,
            'min': data_min, 'maintain_bounds': kwargs['maintain_bounds']}


@dw.decorate.apply_stacked
def transformer(data, **kwargs):
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')
    axis = kwargs.pop('axis', None)

    transpose = False
    if axis == 1:
        transpose = not transpose
        axis = int(not axis)
    elif axis != 0:
        raise ValueError(f'Invalid smoothing axis: {axis}')

    if kwargs['kernel_width'] != int(np.round(kwargs['kernel_width'])):
        warnings.warn('Rounding smoothing kernel width to the nearest integer')
        kwargs['kernel_width'] = int(kwargs['kernel_width'])
    if kwargs['kernel_width'] % 2 != 1:
        warnings.warn('Increasing smoothing kernel width by 1 (must be odd)')
        kwargs['kernel_width'] += 1
    assert kwargs['kernel_width'] > 0, ValueError('smoothing kernel width must be a positive odd integer')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': axis})).T

    assert axis == 0, ValueError('invalid transformation')

    smoothed = data.copy()
    for c in data.columns:
        smoothed[c] = savgol_filter(data[c].values, kwargs['kernel_width'], kwargs['order'])

        if kwargs['maintain_bounds']:
            smoothed[c].loc[smoothed[c] > kwargs['max'][c]] = kwargs['max'][c]
            smoothed[c].loc[smoothed[c] < kwargs['min'][c]] = kwargs['min'][c]

    return smoothed


class Smooth(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, kernel_width=11, order=3, maintain_bounds=True):
        required = ['axis', 'min', 'max', 'kernel_width', 'order', 'maintain_bounds']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, kernel_width=kernel_width,
                         order=order, maintain_bounds=maintain_bounds,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.kernel_width = kernel_width
        self.order = order
        self.maintain_bounds = maintain_bounds
        self.required = required
