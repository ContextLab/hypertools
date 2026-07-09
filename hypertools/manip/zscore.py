# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, axis=0):
    """Fit z-score parameters (mean/std) for the `ZScore` manipulator.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data to fit on. If a list, datasets are concatenated row-wise
        before fitting (one shared mean/std across all of them).
    axis : int, optional
        0 to z-score each column (default), 1 to z-score each row
        (implemented by transposing, fitting on axis=0, and flagging
        `transpose=True` for the transformer).

    Returns
    -------
    dict
        `{'mean': <per-column mean>, 'std': <per-column std>, 'axis':
        axis, 'transpose': bool}`.

    Raises
    ------
    ValueError
        If `axis` is not 0 or 1.
    """
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
    import numpy as np
    z = data.copy()
    # Key the fitted mean/std POSITIONALLY (by column order), not by column
    # LABEL -- matching this manipulator's `inverter`, which is already
    # positional. Label keying broke reuse of a fitted ZScore on data whose
    # column labels differ from the fit-time data (e.g. fit on an ndarray ->
    # 'c0'.. then reused on a DataFrame with 'a'.. -> KeyError, QC 2026-07). In
    # the normal (non-reuse) path the labels are identical, so this is a no-op.
    mean = np.asarray(kwargs['mean'], dtype=float)
    std = np.asarray(kwargs['std'], dtype=float)
    if z.shape[1] != mean.shape[0]:
        raise ValueError(
            f'ZScore was fit on {mean.shape[0]} column(s) but got {z.shape[1]}')
    for i, c in enumerate(z.columns):
        z[c] = (z[c] - mean[i]) / std[i]
    return z


def transformer(data, **kwargs):
    """Apply fitted z-score parameters for the `ZScore` manipulator.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data to z-score.
    **kwargs
        `mean`, `std`, `axis` : parameters from `fitter`. `transpose` :
        bool, whether to transpose, recurse with `axis` flipped, and
        transpose back (the `axis=1` row-wise path).

    Returns
    -------
    The z-scored data, `(data - mean) / std` per the fitted parameters,
    in the same shape as `data`.

    Raises
    ------
    ValueError
        If `axis` is missing from `kwargs`, or (after resolving
        `transpose`) is not 0.
    """
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


def inverter(data, **kwargs):
    """Invert the z-score: reconstruct ``data * std + mean`` from the fitted
    parameters. Column-wise (``axis=0``) only -- inverting a row-wise
    (``axis=1``) z-score on held-out data is ill-defined, so it raises.

    Operates at the numpy level (so it works on the plain arrays a
    `hypertools.Pipeline` passes between inverse-transform steps as well as
    on DataFrames), broadcasting the fitted per-column ``mean``/``std``.
    """
    import numpy as np
    if kwargs.get('transpose', False) or kwargs.get('axis', 0) != 0:
        raise NotImplementedError(
            'ZScore.inverse_transform is only supported for axis=0 (column-wise)')
    mean = np.asarray(kwargs['mean'], dtype=float)
    std = np.asarray(kwargs['std'], dtype=float)
    arr = np.asarray(data, dtype=float)
    return arr * std + mean


class ZScore(Manipulator):
    """Z-score (mean-center, unit-variance-scale) data, per column or per row.

    Parameters
    ----------
    axis : int, optional
        0 to z-score each column independently (default), 1 to z-score
        each row independently.
    """
    def __init__(self, axis=0):
        required = ['transpose', 'mean', 'std', 'axis']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer,
                          inverter=inverter, data=None, required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
