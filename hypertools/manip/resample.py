# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

from .common import Manipulator

from ..core.shared import get


def _resampling_x(data):
    """Derive the interpolation x-axis for one dataset from its index.

    Non-numeric index values (e.g. the string column labels that become the
    index after an ``axis=1`` transpose -- audit F14-002) fall back to
    positions (``0..n-1``) instead of crashing inside numpy with a
    ``DTypePromotionError``. Numeric indexes must be strictly increasing
    (PCHIP interpolation requires it); duplicated or decreasing values get
    a clear error instead of scipy's bare "`x` must be strictly increasing
    sequence." (audit F14-018).
    """
    if dw.zoo.is_multiindex_dataframe(data):
        x = np.array(data.index.levels[-1])
    else:
        x = data.index.values

    if not np.issubdtype(np.asarray(x).dtype, np.number):
        # non-numeric labels: interpolate over positions instead
        return np.arange(len(x), dtype=float)

    x = np.asarray(x, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError(
            'Resample interpolates each column against the DataFrame index, '
            'which must be strictly increasing; this index contains '
            'duplicate or decreasing values. Sort the index (sort_index()) '
            'or replace it (reset_index(drop=True)) first.')
    return x


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
    # real raises (not `assert ..., ValueError(...)`, which raised
    # AssertionError and was stripped under `python -O`) -- 2026-07 release
    # audit, final wave item 8
    if 'axis' not in kwargs:
        raise ValueError(
            "Resample's fitter requires an axis= parameter; pass axis=0 "
            '(resample along rows, the default) or axis=1 (resample along '
            'columns).')

    if kwargs['axis'] == 1:
        return fitter(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis']), 'transpose': True}))

    if kwargs['axis'] != 0:
        raise ValueError(
            f"invalid Resample axis {kwargs['axis']!r}; axis must be 0 "
            '(resample along rows, the default) or 1 (resample along '
            'columns).')

    x = _resampling_x(data)

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
    # real raises (see `fitter` above; 2026-07 release audit, final wave
    # item 8)
    if 'axis' not in kwargs:
        raise ValueError(
            "Resample's transformer requires an axis= parameter; pass "
            'axis=0 (resample along rows, the default) or axis=1 (resample '
            'along columns).')

    if transpose:
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': int(not kwargs['axis'])})).T

    if kwargs['axis'] != 0:
        raise ValueError(
            f"invalid Resample axis {kwargs['axis']!r}; axis must be 0 "
            '(resample along rows, the default) or 1 (resample along '
            'columns).')

    # Build the interpolators from THE DATA BEING TRANSFORMED, not from the
    # fit-time data: Resample's fitted state is only its `n_samples` target
    # (and interpolation settings) -- applying a fitted Resample to new data
    # must resample the new data's own values/x-index, not replay the
    # fit-time values (round17 fix wave 1, finding 1).
    x = _resampling_x(data)

    resampled_x = np.linspace(np.min(x), np.max(x), num=kwargs['n_samples'])
    resampled = pd.DataFrame(index=resampled_x, columns=data.columns, dtype=float)

    for c in data.columns:
        resampled[c] = interpolate.pchip(x, data[c].values)(resampled_x)
    return resampled


class Resample(Manipulator):
    """Resample data to a fixed number of evenly-spaced points via PCHIP interpolation.

    Each dataset is resampled independently, against its OWN index (which
    must be strictly increasing when numeric; a non-numeric index -- e.g.
    string column labels after an ``axis=1`` transpose -- is interpolated
    over positions ``0..n-1`` instead).

    Parameters
    ----------
    axis : int, optional
        0 to resample each column along the row (index) axis (default),
        1 to resample along columns instead (transposed internally).
    n_samples : int, optional
        Number of evenly-spaced output points (default: 100). Must be a
        positive integer; 0, negative, or non-integer values raise
        `ValueError`.

    Raises
    ------
    ValueError
        If `n_samples` is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from hypertools.manip import Resample
    >>> df = pd.DataFrame({'y': np.sin(np.linspace(0, 6, 30))})
    >>> out = Resample(n_samples=100).fit_transform(df)
    >>> out.shape
    (100, 1)
    """
    # transform re-derives everything (interpolators, x-axis) from the data
    # being transformed, so reusing a fitted Resample on new data -- even one
    # fit with axis=1 -- is always well-defined (see Manipulator.transform's
    # row-wise reuse guard, audit F14-012).
    _stateless_transform = True

    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, n_samples=100):
        # validate up front (audit X2-error-quality-007): n_samples=0 used to
        # silently return an EMPTY dataset, and negative/float values leaked
        # numpy internals ("Number of samples, -10, must be non-negative." /
        # "'float' object cannot be interpreted as an integer") from
        # np.linspace at transform time.
        if (isinstance(n_samples, bool)
                or not isinstance(n_samples, (int, np.integer))
                or n_samples < 1):
            raise ValueError(
                'n_samples must be a positive integer (the number of '
                f'resampled points per dataset); got {n_samples!r}')
        n_samples = int(n_samples)
        required = ['transpose', 'axis', 'n_samples', 'x', 'resampled_x', 'pchip']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, n_samples=n_samples,
                         required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.n_samples = n_samples
        self.required = required
