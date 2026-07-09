# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd

from .common import Manipulator


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, axis=0, min=0, max=1):
    """Fit min-max normalization parameters for the `Normalize` manipulator.

    Computes, per column (or per row if `axis=1`, via transposed
    recursion), the minimum (`baseline`) and post-baseline maximum
    (`peak`) needed to rescale values into `[min, max]`.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data to fit on. If a list, datasets are concatenated row-wise
        before fitting (one shared baseline/peak across all of them).
    axis : int, optional
        0 to normalize each column (default), 1 to normalize each row
        (implemented by transposing, fitting on axis=0, and flagging
        `transpose=True` for the transformer).
    min : float, optional
        Lower bound of the target range (default: 0).
    max : float, optional
        Upper bound of the target range (default: 1).

    Returns
    -------
    dict
        `{'baseline': <per-column min>, 'peak': <per-column max after
        baseline subtraction>, 'axis': axis, 'transpose': bool, 'min':
        min, 'max': max}`.

    Raises
    ------
    ValueError
        If `min >= max`, or `axis` is not 0 or 1.
    """
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
    import numpy as np
    z = data.copy()
    # Key the fitted baseline/peak POSITIONALLY (by column order), not by column
    # LABEL -- matching this manipulator's `inverter`, which is already
    # positional. Label keying broke reuse of a fitted Normalize on data whose
    # column labels differ from the fit-time data (QC 2026-07). In the normal
    # (non-reuse) path the labels are identical, so this is a no-op.
    baseline = np.asarray(kwargs['baseline'], dtype=float)
    peak = np.asarray(kwargs['peak'], dtype=float)
    if z.shape[1] != baseline.shape[0]:
        raise ValueError(
            f'Normalize was fit on {baseline.shape[0]} column(s) but got '
            f'{z.shape[1]}')
    # guard zero-range (constant) columns: dividing by peak=0 turned the whole
    # column into NaN (QC 2026-07). A constant column is 0 after subtracting its
    # baseline, so scaling by 1 leaves it 0.
    peak_safe = np.where(peak == 0, 1.0, peak)
    for i, c in enumerate(z.columns):
        z[c] = (z[c] - baseline[i]) / peak_safe[i]

    z *= (kwargs['max'] - kwargs['min'])
    z += kwargs['min']
    return z


def transformer(data, **kwargs):
    """Apply fitted min-max normalization parameters for the `Normalize` manipulator.

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data to normalize.
    **kwargs
        `baseline`, `peak`, `min`, `max`, `axis` : parameters from
        `fitter`. `transpose` : bool, whether to transpose, recurse with
        `axis` flipped, and transpose back (the `axis=1` row-wise path).

    Returns
    -------
    The normalized data, rescaled into `[min, max]` per the fitted
    `baseline`/`peak`, in the same shape as `data`.

    Raises
    ------
    ValueError
        If `axis` is missing from `kwargs`, or (after resolving
        `transpose`) is not 0.
    """
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


def inverter(data, **kwargs):
    """Invert the min-max normalization: reconstruct the pre-normalization
    values from the fitted ``baseline``/``peak`` and target ``[min, max]``.
    Column-wise (``axis=0``) only.

    Forward is ``z = (x - baseline) / peak * (max - min) + min``; this
    computes ``x = (z - min) / (max - min) * peak + baseline``. Operates at
    the numpy level so it works on the plain arrays a `hypertools.Pipeline`
    passes between inverse-transform steps as well as on DataFrames.
    """
    import numpy as np
    if kwargs.get('transpose', False) or kwargs.get('axis', 0) != 0:
        raise NotImplementedError(
            'Normalize.inverse_transform is only supported for axis=0 (column-wise)')
    baseline = np.asarray(kwargs['baseline'], dtype=float)
    peak = np.asarray(kwargs['peak'], dtype=float)
    lo, hi = kwargs['min'], kwargs['max']
    arr = np.asarray(data, dtype=float)
    return (arr - lo) / (hi - lo) * peak + baseline


class Normalize(Manipulator):
    """Min-max normalize data into a `[min, max]` range, per column or per row.

    Parameters
    ----------
    min : float, optional
        Lower bound of the target range (default: 0).
    max : float, optional
        Upper bound of the target range (default: 1).
    axis : int, optional
        0 to normalize each column independently (default), 1 to
        normalize each row independently.
    """
    # noinspection PyShadowingBuiltins
    def __init__(self, min=0, max=1, axis=0):
        required = ['min', 'max', 'transpose', 'baseline', 'peak', 'axis']
        super().__init__(min=min, max=max, axis=axis, fitter=fitter, transformer=transformer,
                          inverter=inverter, data=None, required=required)

        self.min = min
        self.max = max
        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
