# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from .common import Manipulator

from ..core import get


def fitter(data, **kwargs):
    def listify_dicts(dicts):
        if len(dicts) == 0:
            return {}
        ld = {}
        for d in dicts:
            for k in d.keys():
                if k not in ld.keys():
                    ld[k] = [d[k]]
                else:
                    ld[k].append(d[k])
        return ld

    if dw.zoo.is_multiindex_dataframe(data):
        return listify_dicts([fitter(d, **kwargs) for d in dw.unstack(data)])
    elif type(data) is list:
        return listify_dicts([fitter(d, **kwargs) for d in data])

    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if kwargs['axis'] == 1:
        return fitter(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis']), 'transpose': True}))

    assert kwargs['axis'] == 0, ValueError('invalid transformation')

    if dw.zoo.is_multiindex_dataframe(data):
        x = np.array(data.index.levels[-1])
    else:
        x = data.index.values

    resampled_x = np.linspace(np.min(x), np.max(x), num=kwargs['n_samples'])
    pchip = pd.Series(index=data.columns)
    for c in data.columns:
        pchip[c] = interpolate.pchip(x, data[c].values)

    return {'x': x, 'resampled_x': resampled_x, 'pchip': pchip, 'transpose': transpose, 'axis': kwargs['axis'],
            'n_samples': kwargs['n_samples']}


def transformer(data, **kwargs):
    if dw.zoo.is_multiindex_dataframe(data):
        stack_result = True
        data = dw.unstack(data)
    else:
        stack_result = False

    if type(data) is list:
        transformed_data = []
        for i, d in enumerate(data):
            next_kwargs = {k: get(v, i) for k, v in kwargs.items()}
            transformed_data.append(transformer(d, **next_kwargs))
        if stack_result:
            return dw.stack(transformed_data)
        else:
            return transformed_data

    # noinspection DuplicatedCode
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])})).T

    assert kwargs['axis'] == 0, ValueError('invalid transformation')
    resampled = pd.DataFrame(index=kwargs['resampled_x'], columns=data.columns)

    for c in data.columns:
        try:
            resampled[c] = kwargs['pchip'][c](kwargs['resampled_x'])
        except IndexError:
            resampled[c] = kwargs['pchip'][int(c)](kwargs['resampled_x'])
    return resampled


class Resample(Manipulator):
    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, n_samples=100):
        required = ['transpose', 'axis', 'n_samples', 'x', 'resampled_x', 'pchip']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, n_samples=n_samples,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.n_samples = n_samples
        self.required = required
