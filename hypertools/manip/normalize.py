# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, axis=0, min=0, max=1):
    assert min < max, ValueError('minimum must be strictly less than maximum')

    if isinstance(data, list):
        data = pd.concat(data, axis=0, ignore_index=True)

    if axis == 1:
        return dw.core.update_dict(fitter(data.T, axis=0, min=min, max=max), {'transpose': True})
    elif axis != 0:
        raise ValueError('axis must be either 0 or 1')

    baseline = pd.Series(index=data.columns, dtype=float)
    peak = pd.Series(index=data.columns, dtype=float)

    z = data.copy()
    for c in z.columns:
        baseline[c] = z[c].min(axis=0)
        z[c] -= baseline[c]

        peak[c] = z[c].max(axis=0)

    return {'baseline': baseline, 'peak': peak, 'axis': axis, 'transpose': False, 'min': min, 'max': max}


# noinspection DuplicatedCode
@dw.decorate.apply_stacked
def _transform_stacked(data, **kwargs):
    z = data.copy()
    for c in z.columns:
        z[c] -= kwargs['baseline'][c]
        z[c] /= kwargs['peak'][c]

    z *= (kwargs['max'] - kwargs['min'])
    z += kwargs['min']
    return z


def transformer(data, **kwargs):
    transpose = kwargs.pop('transpose', False)
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')

    if transpose:
        # NOTE: recurse into the (undecorated) *transformer* itself, not into
        # _transform_stacked. _transform_stacked is decorated with
        # dw.decorate.apply_stacked, which vertically re-stacks whatever data it is
        # given (adding a synthetic 'ID' level to the row index) before doing any
        # work. If we transposed data that had already been through that decorator,
        # the synthetic ID level would leak into the columns, and the fitted
        # baseline/peak (keyed by the ORIGINAL, pre-stacking row labels) could no
        # longer be looked up -- raising "key of type tuple not found and not a
        # MultiIndex". Transposing before the data ever reaches the decorated
        # function keeps the stacking machinery isolated to the (always axis==0)
        # base case, where it is harmless.
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])})).T

    assert kwargs['axis'] == 0, ValueError('invalid transformation')
    return _transform_stacked(data, **kwargs)


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
