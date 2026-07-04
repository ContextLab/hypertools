# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import warnings

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, **kwargs):
    if isinstance(data, list):
        data = pd.concat(data, axis=0, ignore_index=True)

    data_max = data.max(axis=kwargs['axis'])
    data_min = data.min(axis=kwargs['axis'])

    return {'axis': kwargs['axis'], 'kernel_width': kwargs['kernel_width'], 'order': kwargs['order'],
            'mode': kwargs['mode'], 'var': kwargs['var'], 'max': data_max,
            'min': data_min, 'maintain_bounds': kwargs['maintain_bounds']}


@dw.decorate.apply_stacked
def _transform_stacked(data, **kwargs):
    smoothed = data.copy()
    for c in data.columns:
        if kwargs['mode'] == 'gaussian':
            smoothed[c] = gaussian_filter1d(np.asarray(data[c], dtype=float), sigma=np.sqrt(kwargs['var']))
        else:
            smoothed[c] = savgol_filter(data[c].values, kwargs['kernel_width'], kwargs['order'])

        if kwargs['maintain_bounds']:
            smoothed[c] = np.clip(smoothed[c].to_numpy(), kwargs['min'][c], kwargs['max'][c])

    return smoothed


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
        # NOTE: recurse into the (undecorated) *transformer* itself, not into
        # _transform_stacked. _transform_stacked is decorated with
        # dw.decorate.apply_stacked, which vertically re-stacks whatever data it is
        # given (adding a synthetic 'ID' level to the row index) before doing any
        # work. If we transposed data that had already been through that decorator,
        # the synthetic ID level would leak into the columns, and the fitted
        # min/max (keyed by the ORIGINAL, pre-stacking row labels) could no longer
        # be looked up -- raising "key of type tuple not found and not a
        # MultiIndex". Transposing before the data ever reaches the decorated
        # function keeps the stacking machinery isolated to the (always axis==0)
        # base case, where it is harmless.
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': axis})).T

    assert axis == 0, ValueError('invalid transformation')
    return _transform_stacked(data, **kwargs, axis=axis)


class Smooth(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, mode='savgol', kernel_width=11, order=3, var=300, maintain_bounds=True):
        required = ['axis', 'min', 'max', 'mode', 'kernel_width', 'order', 'var', 'maintain_bounds']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, mode=mode,
                         kernel_width=kernel_width, order=order, var=var, maintain_bounds=maintain_bounds,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.mode = mode
        self.kernel_width = kernel_width
        self.order = order
        self.var = var
        self.maintain_bounds = maintain_bounds
        self.required = required
