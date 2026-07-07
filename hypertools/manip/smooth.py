# noinspection PyPackageRequirements
import datawrangler as dw
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

import warnings

from .common import Manipulator


#: valid values for `Smooth`'s `kernel=` kwarg (GH #274/#153, round17 Task 5).
KERNELS = ('savgol', 'gaussian', 'boxcar')


# noinspection PyShadowingBuiltins
@dw.decorate.funnel
def fitter(data, **kwargs):
    if isinstance(data, list):
        data = pd.concat(data, axis=0, ignore_index=True)

    data_max = data.max(axis=kwargs['axis'])
    data_min = data.min(axis=kwargs['axis'])

    return {'axis': kwargs['axis'], 'kernel_width': kwargs['kernel_width'], 'order': kwargs['order'],
            'mode': kwargs['mode'], 'var': kwargs['var'], 'max': data_max,
            'min': data_min, 'maintain_bounds': kwargs['maintain_bounds']}


def _resolve_kernel(kwargs):
    """Decide which smoothing branch to run and, for `'gaussian'`, which
    sigma mapping to use.

    `kernel=` (round17 Task 5, GH #274/#153) is the new, preferred kwarg:
    `'savgol'`, `'gaussian'` (`scipy.ndimage.gaussian_filter1d`,
    `sigma = kernel_width / 4` -- a sensible width-to-sigma mapping so
    `kernel='gaussian'` needs no separate `var=` kwarg), or `'boxcar'`
    (`scipy.ndimage.uniform_filter1d`, `size = kernel_width`).

    Internally, `kernel` defaults to `None` ("unspecified") rather than
    `'savgol'`, so this function can tell "kernel left at its default"
    apart from "the user explicitly passed `kernel='savgol'`" (round17 fix
    wave 1, finding 2). Any EXPLICIT `kernel=` string (including
    `'savgol'`) always takes precedence over `mode=`/`var=`. Only when
    `kernel` is left unspecified (`None`) does the older `mode=`/`var=`
    kwargs (added for the weights-trajectory recipe, GH #153 plan 6) kick
    in, byte-identical to their original behavior: `mode='gaussian'` uses
    `sigma = sqrt(var)`; otherwise (the true default, neither `kernel` nor
    `mode='gaussian'` given) smoothing is `'savgol'`.

    Returns
    -------
    (branch, use_legacy_var) : (str, bool)
        `branch` is one of `'savgol'`, `'gaussian'`, `'boxcar'`;
        `use_legacy_var` is True only for the `mode='gaussian'` backward
        -compat path (sigma from `var`, not `kernel_width`), which can only
        happen when `kernel` was left unspecified.
    """
    kernel = kwargs.get('kernel', None)
    if kernel is None:
        legacy_mode = kwargs.get('mode', 'savgol')
        if legacy_mode == 'gaussian':
            return 'gaussian', True
        return 'savgol', False
    if kernel not in KERNELS:
        raise ValueError(f"invalid Smooth kernel {kernel!r}; must be one of {KERNELS}")
    return kernel, False


@dw.decorate.apply_stacked
def _transform_stacked(data, **kwargs):
    smoothed = data.copy()
    branch, use_legacy_var = _resolve_kernel(kwargs)
    for c in data.columns:
        values = np.asarray(data[c], dtype=float)
        if branch == 'gaussian':
            sigma = np.sqrt(kwargs['var']) if use_legacy_var else kwargs['kernel_width'] / 4
            smoothed[c] = gaussian_filter1d(values, sigma=sigma)
        elif branch == 'boxcar':
            smoothed[c] = uniform_filter1d(values, size=kwargs['kernel_width'])
        else:
            smoothed[c] = savgol_filter(data[c].values, kwargs['kernel_width'], kwargs['order'])

        if kwargs['maintain_bounds']:
            smoothed[c] = np.clip(smoothed[c].to_numpy(), kwargs['min'][c], kwargs['max'][c])

    return smoothed


def transformer(data, **kwargs):
    assert 'axis' in kwargs.keys(), ValueError('Must specify axis')
    axis = kwargs.pop('axis', None)

    transpose = False
    if axis == 1:
        transpose = not transpose
        axis = int(not axis)
    elif axis != 0:
        raise ValueError(f'Invalid smoothing axis: {axis}')

    if kwargs['kernel_width'] != int(np.round(kwargs['kernel_width'])):
        warnings.warn('Rounding smoothing kernel width to the nearest integer')
        kwargs['kernel_width'] = int(kwargs['kernel_width'])
    if kwargs['kernel_width'] % 2 != 1:
        warnings.warn('Increasing smoothing kernel width by 1 (must be odd)')
        kwargs['kernel_width'] += 1
    assert kwargs['kernel_width'] > 0, ValueError('smoothing kernel width must be a positive odd integer')

    if transpose:
        # NOTE: recurse into the (undecorated) *transformer* itself, not into
        # _transform_stacked. _transform_stacked is decorated with
        # dw.decorate.apply_stacked, which vertically re-stacks whatever data it is
        # given (adding a synthetic 'ID' level to the row index) before doing any
        # work. If we transposed data that had already been through that decorator,
        # the synthetic ID level would leak into the columns, and the fitted
        # min/max (keyed by the ORIGINAL, pre-stacking row labels) could no longer
        # be looked up -- raising "key of type tuple not found and not a
        # MultiIndex". Transposing before the data ever reaches the decorated
        # function keeps the stacking machinery isolated to the (always axis==0)
        # base case, where it is harmless.
        return transformer(data.T, **dw.core.update_dict(kwargs, {'axis': axis})).T

    assert axis == 0, ValueError('invalid transformation')
    return _transform_stacked(data, **kwargs, axis=axis)


class Smooth(Manipulator):
    """Smooth each dataset (per-column) along `axis`.

    Parameters
    ----------
    axis : int
        `0` smooths down each column (the default: time along rows);
        `1` smooths across each row instead.

    kernel : {'savgol', 'gaussian', 'boxcar', None}
        Which smoothing kernel to apply (GH #274/#153, round17 Task 5).
        Defaults to `None` ("unspecified"), which distinguishes "left at
        its default" from an explicit `kernel='savgol'` -- an EXPLICIT
        `kernel=` (any of the three strings below, including `'savgol'`)
        ALWAYS takes precedence over `mode=`/`var=` below. When left
        unspecified (`None`), the legacy `mode=`/`var=` kwargs decide the
        branch instead (see `mode` below); with neither `kernel` nor
        `mode='gaussian'` given, smoothing is `'savgol'`.

        - `'savgol'`: `scipy.signal.savgol_filter` with window length
          `kernel_width` and polynomial `order` -- unchanged from
          pre-round17 behavior.
        - `'gaussian'`: `scipy.ndimage.gaussian_filter1d` with
          `sigma = kernel_width / 4` (a sensible width-to-sigma mapping;
          e.g. `kernel_width=25` gives `sigma=6.25`).
        - `'boxcar'`: `scipy.ndimage.uniform_filter1d` with
          `size = kernel_width` (a moving average).

        Invalid values raise `ValueError` listing the supported options.

    mode : {'savgol', 'gaussian'}
        LEGACY kwarg (predates `kernel=`), kept for backward compatibility:
        when `kernel` is left UNSPECIFIED (`None`, the default) and `mode`
        is explicitly `'gaussian'`, gaussian smoothing uses
        `sigma = sqrt(var)` instead of the `kernel_width`-based mapping
        above -- byte-identical to the original weights-trajectory recipe
        behavior. An explicit `kernel=` (including `kernel='savgol'`)
        always takes precedence over `mode=`.

    kernel_width : int
        Smoothing window width for `'savgol'`/`'boxcar'` (and, via the
        mapping above, `'gaussian'` when using `kernel=`). Must be a
        positive odd integer; non-integer/even values are rounded up with a
        warning.

    order : int
        Polynomial order for the `'savgol'` kernel (ignored otherwise).

    var : float
        Variance for the LEGACY `mode='gaussian'` path (`sigma = sqrt(var)`,
        default 300, matching the original weights-trajectory recipe).
        Ignored by `kernel='gaussian'`.

    maintain_bounds : bool
        If True (default), clip the smoothed output to each column's
        original (pre-smoothing) min/max.
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, kernel=None, mode='savgol', kernel_width=11, order=3, var=300,
                 maintain_bounds=True):
        required = ['axis', 'min', 'max', 'mode', 'kernel_width', 'order', 'var', 'maintain_bounds']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, mode=mode,
                         kernel=kernel, kernel_width=kernel_width, order=order, var=var,
                         maintain_bounds=maintain_bounds, required=required)

        self.axis = axis
        self.fitter = fitter
        self.transformer = transformer
        self.data = None
        self.mode = mode
        self.kernel = kernel
        self.kernel_width = kernel_width
        self.order = order
        self.var = var
        self.maintain_bounds = maintain_bounds
        self.required = required
