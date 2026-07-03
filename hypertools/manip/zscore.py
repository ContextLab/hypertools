# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, axis=0):
    if isinstance(data, list):
        data = pd.concat(data, axis=0, ignore_index=True)

    if axis == 1:
        return dw.core.update_dict(fitter(data.T, axis=0), {'transpose': True})
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

    mean = pd.Series(index=data.columns, dtype=float)
    std = pd.Series(index=data.columns, dtype=float)

    for c in data.columns:
        mean[c] = data[c].mean(axis=0)
        std[c] = data[c].std(axis=0)

    return {'mean': mean, 'std': std, 'axis': axis, 'transpose': False}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def _transform_stacked(data, **kwargs):
    z = data.copy()
    for c in z.columns:
        z[c] -= kwargs['mean'][c]
        z[c] /= kwargs['std'][c]
    return z


def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        # NOTE: this recurses into the (undecorated) *transformer* itself, not into
        # _transform_stacked. _transform_stacked is decorated with
        # dw.decorate.apply_stacked, which vertically re-stacks whatever data it is
        # given (adding a synthetic 'ID' level to the row index) before doing any
        # work. If we transposed data that had already been through that decorator,
        # the synthetic ID level would leak into the columns, and the fitted
        # mean/std (keyed by the ORIGINAL, pre-stacking row labels) could no longer
        # be looked up -- raising "key of type tuple not found and not a
        # MultiIndex". Transposing before the data ever reaches the decorated
        # function keeps the stacking machinery isolated to the (always axis==0)
        # base case, where it is harmless.
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])})).T

    assert kwargs['axis'] == 0, ValueError('invalid transformation')
    return _transform_stacked(data, **kwargs)


class ZScore(Manipulator):
    def __init__(self, axis=0):
        required = ['transpose', 'mean', 'std', 'axis']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None,
                          required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
