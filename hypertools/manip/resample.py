# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from .common import Manipulator

from ..core.shared import get


def fitter(data, **kwargs):
    """Fit PCHIP resampling interpolators for the `Resample` manipulator.

    Recurses over multi-index/list data (one fit per dataset, combined
    via `listify_dicts`); for a single DataFrame, builds a PCHIP
    interpolator per column against the original index values.

    Parameters
    ----------
    data : DataFrame, multi-index DataFrame, or list of DataFrame
        Data to fit resampling interpolators on.
    **kwargs
        `axis` : int, 0 to resample along rows (default), 1 to
        transpose and resample along columns instead. `n_samples` : int,
        number of resampled points to target.

    Returns
    -------
    dict
        `{'x': <original index values>, 'resampled_x': <target index
        values>, 'pchip': <per-column PCHIP interpolators>, 'transpose':
        bool, 'axis': axis, 'n_samples': n_samples}`, or (for
        multi-index/list input) a dict of lists of these values, one
        entry per dataset (see `listify_dicts`).
    """
    def listify_dicts(dicts):
        """Merge a list of same-keyed dicts into one dict of lists (one list per key)."""
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
    elif isinstance(data, list):
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
    pchip = pd.Series(index=data.columns, dtype=object)
    for c in data.columns:
        pchip[c] = interpolate.pchip(x, data[c].values)

    return {'x': x, 'resampled_x': resampled_x, 'pchip': pchip, 'transpose': transpose, 'axis': kwargs['axis'],
            'n_samples': kwargs['n_samples']}


def transformer(data, **kwargs):
    """Resample `data` to `n_samples` evenly-spaced points via PCHIP interpolation.

    Recurses over multi-index/list data (`kwargs` broadcast per dataset
    via `core.shared.get`). For a single DataFrame, PCHIP interpolators
    are rebuilt from `data`'s OWN index/values (not the fit-time data)
    and evaluated at `n_samples` evenly-spaced points spanning `data`'s
    own index range -- so a fitted `Resample` resamples new data using
    that new data's own range, not a replay of the fit-time values.

    Parameters
    ----------
    data : DataFrame, multi-index DataFrame, or list of DataFrame
        Data to resample.
    **kwargs
        `axis` : int, 0 (default) or 1 (transpose first). `n_samples` :
        int, target number of resampled points. `transpose` : bool,
        whether to transpose, recurse with `axis` flipped, and transpose
        back.

    Returns
    -------
    The resampled data (DataFrame, multi-index DataFrame, or list,
    matching the input structure), with `n_samples` rows.

    Raises
    ------
    ValueError
        If `axis` is missing from `kwargs`, or (after resolving
        `transpose`) is not 0.
    """
    if dw.zoo.is_multiindex_dataframe(data):
        stack_result = True
        data = dw.unstack(data)
    else:
        stack_result = False

    if isinstance(data, list):
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

    # Build the interpolators from THE DATA BEING TRANSFORMED, not from the
    # fit-time data: Resample's fitted state is only its `n_samples` target
    # (and interpolation settings) -- applying a fitted Resample to new data
    # must resample the new data's own values/x-index, not replay the
    # fit-time values (round17 fix wave 1, finding 1).
    if dw.zoo.is_multiindex_dataframe(data):
        x = np.array(data.index.levels[-1])
    else:
        x = data.index.values

    resampled_x = np.linspace(np.min(x), np.max(x), num=kwargs['n_samples'])
    resampled = pd.DataFrame(index=resampled_x, columns=data.columns, dtype=float)

    for c in data.columns:
        resampled[c] = interpolate.pchip(x, data[c].values)(resampled_x)
    return resampled


class Resample(Manipulator):
    """Resample data to a fixed number of evenly-spaced points via PCHIP interpolation.

    Parameters
    ----------
    axis : int, optional
        0 to resample each column along the row (index) axis (default),
        1 to resample along columns instead (transposed internally).
    n_samples : int, optional
        Number of evenly-spaced output points (default: 100).
    """
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
