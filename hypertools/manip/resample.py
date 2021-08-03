# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from .manip import Manipulator


@dw.decorate.apply_stacked
def fitter(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return fitter(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    x = data.index.values
    resampled_x = np.linspace(np.min(x), np.max(x), num=kwargs['n_samples'])
    pchip = pd.Series(index=data.columns)
    for c in data.columns:
        pchip[c] = interpolate.pchip(x, data[c].values)

    return {'x': x, 'resampled_x': resampled_x, 'pchip': pchip, 'transpose': transpose, 'axis': axis,
            'n_samples': n_samples}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')
    resampled = pd.DataFrame(index=kwargs['resampled_x'], columns=data.columns)

    for c in data.columns:
        resampled[c] = kwargs['pchip'](kwargs['resampled_x'])
    return resampled


class Resample(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, n_samples=100):
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, n_samples=n_samples,
                         required=['transpose', 'axis', 'n_samples', 'x', 'resampled_x', 'pchip'])
