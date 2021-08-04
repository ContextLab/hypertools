# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter

from .common import Manipulator


def resample_and_smooth(traj, kernel_width, N=500, order=3, min_val=0):
    if traj is None or traj.shape[0] <= 3:
        return None

    try:
        r = np.zeros([N, traj.shape[1]])
        x = traj.index.values
        xx = np.linspace(np.min(x), np.max(x), num=N)

        for i in range(traj.shape[1]):
            r[:, i] = signal.savgol_filter(sp.interpolate.pchip(x, traj.values[:, i])(xx),
                                           kernel_width, order)
            r[:, i][r[:, i] < min_val] = min_val

        return pd.DataFrame(data=r, index=xx, columns=traj.columns)
    except:
        return None


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def fitter(data, transpose=False, axis=0, kernel_width=10, order=3, maintain_bounds=True):
    if transpose:
        return fitter(data.T, axis=int(not axis), kernel_width=kernel_width, order=order,
                      maintain_bounds=maintain_bounds)
    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    data_max = data.max()
    data_min = data.min()

    return {'transpose': transpose, 'axis': axis, 'kernel_width': kernel_width, 'order': order, 'max': data_max,
            'min': data_min, 'maintain_bounds': maintain_bounds}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    smoothed = data.copy()
    for c in data.columns:
        smoothed[c] = savgol_filter(data[c].values, kwargs['kernel_width'], kwargs['order'])

    if kwargs['maintain_bounds']:
        smoothed.loc[smoothed > kwargs['max']] = kwargs['max']
        smoothed.loc[smoothed < kwargs['min']] = kwargs['min']

    return smoothed


class Smooth(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, kernel_width=10, order=3, maintain_bounds=True):
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, kernel_width=kernel_width,
                         order=order, maintain_bounds=maintain_bounds,
                         required=['transpose', 'axis', 'min', 'max', 'kernel_width', 'order', 'maintain_bounds'])
