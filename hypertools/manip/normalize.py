# noinspection PyPackageRequirements
import datawrangler as dw
import pandas as pd

from .common import Manipulator


MODES = ('minmax', 'isotropic')


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, axis=0, min=0, max=1, mode='minmax'):
    """Fit normalization parameters for the `Normalize` manipulator.

    In the default ``mode='minmax'``, computes, per column (or per row if
    `axis=1`, via transposed recursion), the minimum (`baseline`) and
    post-baseline maximum (`peak`) needed to rescale values into
    `[min, max]`. In ``mode='isotropic'``, computes the per-column
    centroid (`baseline`, the column means) and ONE scalar `peak` (the
    largest absolute deviation from the centroid over every entry) shared
    by all columns, so the transform preserves the data's shape.

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
    mode : {'minmax', 'isotropic'}, optional
        `'minmax'` (default) rescales each column (or row) independently;
        `'isotropic'` centres on the centroid and divides every column by
        one shared scalar (see `Normalize`).

    Returns
    -------
    dict
        `{'baseline': <per-column min>, 'peak': <per-column max after
        baseline subtraction>, 'axis': axis, 'transpose': bool, 'min':
        min, 'max': max, 'mode': mode}`. In `'isotropic'` mode `baseline`
        is the per-column centroid and `peak` is a single float.

    Raises
    ------
    ValueError
        If `min >= max`, `axis` is not 0 or 1, `mode` is not one of
        `MODES`, or ``mode='isotropic'`` is combined with ``axis=1``.
    """
    # a real ValueError (as documented in Raises), not "assert cond,
    # ValueError(...)" -- the assert idiom raised AssertionError and was
    # silently stripped under `python -O` (audit F14-009)
    if min >= max:
        raise ValueError(
            f'minimum must be strictly less than maximum; got min={min!r}, '
            f'max={max!r}')

    if mode not in MODES:
        raise ValueError(
            f"invalid Normalize mode {mode!r}; mode must be one of "
            f"{', '.join(repr(m) for m in MODES)}")

    if isinstance(data, list):
        data = pd.concat(data, axis=0, ignore_index=True)

    if mode == 'isotropic':
        # one shared centre + scale for the whole table (and, for a list,
        # for every dataset in it): subtract the per-column centroid, then
        # divide EVERY column by the same scalar -- the largest absolute
        # deviation from the centroid across all entries. Row-wise fitting
        # is meaningless here (a row-wise centroid would be per-row, i.e.
        # not one point), so refuse it rather than silently ignore `axis`.
        if axis != 0:
            raise ValueError(
                "Normalize(mode='isotropic') centres and rescales the whole "
                'table with one scalar, so it only supports axis=0; got '
                f'axis={axis!r}')
        baseline = data.mean(axis=0).astype(float)
        peak = float((data - baseline).abs().max().max())
        if not peak > 0:
            # a single point, or all rows identical: leave the (already
            # zero) deviations alone rather than dividing by 0 -> NaN
            peak = 1.0
        return {'baseline': baseline, 'peak': peak, 'axis': 0,
                'transpose': False, 'min': min, 'max': max, 'mode': mode}

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

    return {'baseline': baseline, 'peak': peak, 'axis': axis, 'transpose': False, 'min': min, 'max': max,
            'mode': mode}


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
    if kwargs.get('mode', 'minmax') == 'isotropic':
        # one scalar `peak` for every column: (x - centroid) / peak lies in
        # [-1, 1], which the affine below maps onto [min, max] with the
        # centroid landing at the midpoint (min + max) / 2
        half = (kwargs['max'] - kwargs['min']) / 2.0
        mid = (kwargs['max'] + kwargs['min']) / 2.0
        for i, c in enumerate(z.columns):
            z[c] = (z[c] - baseline[i]) / float(peak) * half + mid
        return z
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
        `baseline`, `peak`, `min`, `max`, `axis`, `mode` : parameters from
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
    # real raises (not `assert ..., ValueError(...)`, which raised
    # AssertionError and was stripped under `python -O`) -- 2026-07 release
    # audit, final wave item 8
    if 'axis' not in kwargs:
        raise ValueError(
            "Normalize's transformer requires an axis= parameter; pass "
            'axis=0 (normalize each column, the default) or axis=1 '
            '(normalize each row).')

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

    if kwargs['axis'] != 0:
        raise ValueError(
            f"invalid Normalize axis {kwargs['axis']!r}; axis must be 0 "
            '(normalize each column, the default) or 1 (normalize each '
            'row).')
    return _transform_stacked(data, **kwargs)


def inverter(data, **kwargs):
    """Invert the min-max normalization: reconstruct the pre-normalization
    values from the fitted ``baseline``/``peak`` and target ``[min, max]``.
    Column-wise (``axis=0``) only.

    Forward is ``z = (x - baseline) / peak * (max - min) + min``; this
    computes ``x = (z - min) / (max - min) * peak + baseline``. In
    ``mode='isotropic'`` the forward map is
    ``z = (x - centroid) / peak * (max - min) / 2 + (max + min) / 2`` with a
    scalar ``peak``, and this computes
    ``x = (z - (max + min) / 2) / ((max - min) / 2) * peak + centroid``.
    Operates at the numpy level so it works on the plain arrays a
    `hypertools.Pipeline` passes between inverse-transform steps as well as
    on DataFrames.
    """
    import numpy as np
    if kwargs.get('transpose', False) or kwargs.get('axis', 0) != 0:
        raise NotImplementedError(
            'Normalize.inverse_transform is only supported for axis=0 (column-wise)')
    baseline = np.asarray(kwargs['baseline'], dtype=float)
    peak = np.asarray(kwargs['peak'], dtype=float)
    lo, hi = kwargs['min'], kwargs['max']
    arr = np.asarray(data, dtype=float)
    if kwargs.get('mode', 'minmax') == 'isotropic':
        return (arr - (hi + lo) / 2.0) / ((hi - lo) / 2.0) * peak + baseline
    return (arr - lo) / (hi - lo) * peak + baseline


class Normalize(Manipulator):
    """Normalize data into a `[min, max]` range: min-max per column or per
    row (default), or isotropically (one shared centre and scale for the
    whole table, preserving its shape).

    Parameters
    ----------
    min : float, optional
        Lower bound of the target range (default: 0).
    max : float, optional
        Upper bound of the target range (default: 1).
    axis : int, optional
        0 to normalize each column independently (default), 1 to
        normalize each row independently. Only ``axis=0`` is valid with
        ``mode='isotropic'``.
    mode : {'minmax', 'isotropic'}, optional
        ``'minmax'`` (default): each column (or row) is rescaled
        independently so that it spans exactly `[min, max]` -- every
        feature gets its own offset and scale, so the data's shape is
        (deliberately) distorted.

        ``'isotropic'``: the whole table is centred on its CENTROID (the
        per-column mean, ``data.mean(axis=0)``) and then every column is
        divided by the SAME scalar -- the largest absolute deviation from
        the centroid over all entries, ``abs(data - centroid).max()`` --
        so that the result lies in the `[-1, 1]` cube with its farthest
        coordinate exactly on a face; that cube is then mapped affinely
        onto `[min, max]`. All pairwise distances are scaled by one
        constant, so the data's shape (angles, distance ratios) is
        preserved and a rotated copy is rescaled by the same scalar. The
        centroid lands at the midpoint ``(min + max) / 2`` and every
        coordinate lies in `[min, max]`, with at least one coordinate
        touching `min` or `max`. With ``min=-1, max=1`` this is exactly
        ``(x - x.mean(axis=0)) / abs(x - x.mean(axis=0)).max()`` -- the
        "centre and scale a point cloud into the unit cube" recipe. A
        degenerate table (a single point, or all rows identical) is only
        centred (its zero deviations are not divided).

    Notes
    -----
    For a LIST of datasets, ONE shared baseline/peak is fit across all of
    them (like ``normalize='across'``): in ``'minmax'`` mode the shared
    per-column min/max, in ``'isotropic'`` mode the shared centroid and
    the single scalar scale of the concatenated data, so every dataset in
    the list is moved and rescaled identically. Constant (zero-range)
    columns normalize to `min` rather than NaN in ``'minmax'`` mode.

    `inverse_transform` is supported for ``axis=0`` in both modes.

    Raises
    ------
    ValueError
        If `min >= max`, `mode` is not ``'minmax'`` or ``'isotropic'``,
        or ``mode='isotropic'`` is combined with ``axis=1``.

    Examples
    --------
    >>> import numpy as np
    >>> from hypertools.manip import Normalize
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1., 2., 3.], 'b': [10., 20., 30.]})
    >>> out = Normalize().fit_transform(df)
    >>> float(out['a'].min()), float(out['a'].max())
    (0.0, 1.0)

    Isotropic mode moves the centroid to the origin (with ``min=-1,
    max=1``) and divides both columns by the same scalar (here 10, the
    farthest deviation of column ``b`` from its mean), so column ``a``
    keeps its narrow spread relative to ``b``:

    >>> iso = Normalize(mode='isotropic', min=-1, max=1).fit_transform(df)
    >>> iso['a'].round(2).tolist(), iso['b'].round(2).tolist()
    ([-0.1, 0.0, 0.1], [-1.0, 0.0, 1.0])
    """
    # noinspection PyShadowingBuiltins
    def __init__(self, min=0, max=1, axis=0, mode='minmax'):
        required = ['min', 'max', 'transpose', 'baseline', 'peak', 'axis',
                    'mode']
        super().__init__(min=min, max=max, axis=axis, mode=mode, fitter=fitter,
                          transformer=transformer, inverter=inverter, data=None,
                          required=required)

        self.min = min
        self.max = max
        self.axis = axis
        self.mode = mode
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.required = required
