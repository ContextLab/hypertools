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
    """Record the smoothing parameters for the `Smooth` manipulator.

    Smoothing is stateless: nothing is estimated from the fit-time data.
    (Earlier versions recorded fit-time per-column min/max here for the
    `maintain_bounds` clip; those bounds are now derived by `transformer`
    from the data actually being transformed, so a fitted `Smooth` reused
    on NEW data -- via ``return_model=True`` -- no longer replays the
    fit-time range onto it, and column labels/counts may differ freely.)

    Parameters
    ----------
    data : DataFrame or list of DataFrame
        Data being fit on (unused beyond marking the manipulator fitted).
    **kwargs
        `axis`, `kernel_width`, `order`, `mode`, `var`,
        `maintain_bounds` : the `Smooth` constructor parameters, passed
        through unchanged.

    Returns
    -------
    dict
        `{'axis', 'kernel_width', 'order', 'mode', 'var',
        'maintain_bounds'}`.
    """
    return {'axis': kwargs['axis'], 'kernel_width': kwargs['kernel_width'], 'order': kwargs['order'],
            'mode': kwargs['mode'], 'var': kwargs['var'], 'maintain_bounds': kwargs['maintain_bounds']}


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
        if legacy_mode != 'savgol':
            raise ValueError(
                f"invalid Smooth mode {legacy_mode!r}; must be 'savgol' or "
                "'gaussian' (or use the kernel= kwarg: one of "
                f"{KERNELS})")
        return 'savgol', False
    if kernel not in KERNELS:
        raise ValueError(f"invalid Smooth kernel {kernel!r}; must be one of {KERNELS}")
    return kernel, False


def _smooth_dataset(data, **kwargs):
    """Smooth ONE DataFrame (axis=0 base case) column by column.

    When `maintain_bounds` is True, each smoothed column is clipped to
    THAT column's own pre-smoothing min/max, derived from `data` itself
    -- never from fit-time state -- so reusing a fitted `Smooth` on new
    data (different values, labels, or column count) behaves exactly
    like smoothing that data fresh.
    """
    smoothed = data.copy()
    branch, use_legacy_var = _resolve_kernel(kwargs)
    # clear, actionable errors for kernel_width edge cases (QC 2026-07): scipy
    # otherwise raises opaque "window_length must be <= size of x" /
    # "polyorder must be less than window_length" from deep in savgol_filter.
    kw = kwargs.get('kernel_width')
    n_rows = data.shape[0]
    if kw is not None and kw > n_rows:
        raise ValueError(
            f"kernel_width ({kw}) is larger than the number of samples "
            f"({n_rows}); use a kernel_width <= {n_rows}.")
    if branch == 'savgol' and kw is not None and kw <= kwargs.get('order', 3):
        raise ValueError(
            f"the savgol kernel needs kernel_width ({kw}) > order "
            f"({kwargs.get('order', 3)}); increase kernel_width or lower order "
            "(or use kernel='gaussian'/'boxcar').")
    for c in data.columns:
        values = np.asarray(data[c], dtype=float)
        if branch == 'gaussian':
            sigma = np.sqrt(kwargs['var']) if use_legacy_var else kwargs['kernel_width'] / 4
            smoothed_values = gaussian_filter1d(values, sigma=sigma)
        elif branch == 'boxcar':
            smoothed_values = uniform_filter1d(values, size=kwargs['kernel_width'])
        else:
            smoothed_values = savgol_filter(values, kwargs['kernel_width'], kwargs['order'])

        if kwargs['maintain_bounds']:
            smoothed_values = np.clip(smoothed_values, np.nanmin(values), np.nanmax(values))
        smoothed[c] = smoothed_values

    return smoothed


def _transform(data, **kwargs):
    """Apply smoothing PER DATASET (audit F14-001/D01-001 fix).

    Lists and stacked (multiindex) DataFrames are smoothed one dataset at
    a time (mirroring `resample.transformer`); smoothing a row-stacked
    list as one continuous timeseries silently bled each dataset's edge
    samples into its neighbors' (about kernel_width/2 samples per side of
    every dataset boundary).
    """
    if dw.zoo.is_multiindex_dataframe(data):
        return dw.stack([_transform(d, **kwargs) for d in dw.unstack(data)])
    if isinstance(data, list):
        return [_transform(d, **kwargs) for d in data]
    if not isinstance(data, pd.DataFrame):
        # e.g. a bare array passed between hypertools.Pipeline steps -- the
        # old apply_stacked decorator wrangled these to DataFrames implicitly
        data = pd.DataFrame(data)

    axis = kwargs['axis']
    if axis == 1:
        return _transform(data.T, **dw.core.update_dict(kwargs, {'axis': 0})).T
    if axis != 0:
        raise ValueError(f'Invalid smoothing axis: {axis}')
    return _smooth_dataset(data, **kwargs)


def transformer(data, **kwargs):
    """Apply the smoothing kernel for the `Smooth` manipulator.

    Selects the smoothing branch via `_resolve_kernel` (savgol/gaussian/
    boxcar), validates/coerces `kernel_width` to a positive odd integer
    (rounding to the nearest integer and/or incrementing by 1 with a
    warning if needed), then smooths PER DATASET: each element of a list
    (or each dataset in a stacked multiindex DataFrame) is smoothed
    independently, so data never bleeds across dataset boundaries. For
    `axis=1`, each dataset is transposed, smoothed along axis 0, and
    transposed back.

    Parameters
    ----------
    data : DataFrame, multiindex DataFrame, or list of DataFrame
        Data to smooth.
    **kwargs
        `axis`, `kernel`, `kernel_width`, `order`, `mode`, `var`,
        `maintain_bounds` : parameters from `fitter` (plus `kernel`,
        passed through from the `Smooth` constructor).

    Returns
    -------
    The smoothed data, in the same shape/structure as `data`, with each
    column optionally clipped to its own pre-smoothing min/max
    (`maintain_bounds=True`); the bounds are derived from the data being
    transformed, not from fit-time state.

    Raises
    ------
    ValueError
        If `axis` is missing from `kwargs`, is not 0 or 1, or
        `kernel_width` resolves to a non-positive value.
    """
    if 'axis' not in kwargs:
        raise ValueError('Must specify axis')

    # coerce kernel_width ONCE, up front, so warnings fire once per call even
    # for list input, and so the effective width matches the warning text
    # (audit F14-008: the old code warned "Rounding ... to the nearest
    # integer" but then TRUNCATED, e.g. 11.7 -> 11 instead of 12 -> odd 13).
    kw = kwargs.get('kernel_width')
    if kw is not None:
        if kw != int(np.round(kw)):
            warnings.warn('Rounding smoothing kernel width to the nearest integer')
        kw = int(np.round(kw))
        if kw % 2 != 1:
            warnings.warn('Increasing smoothing kernel width by 1 (must be odd)')
            kw += 1
        if kw <= 0:
            raise ValueError(
                'kernel_width must be a positive odd integer; got '
                f'{kwargs["kernel_width"]!r}')
        kwargs = dw.core.update_dict(kwargs, {'kernel_width': kw})

    return _transform(data, **kwargs)


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
        always takes precedence over `mode=`. Values outside
        `{'savgol', 'gaussian'}` raise `ValueError`.

    kernel_width : int
        Smoothing window width for `'savgol'`/`'boxcar'` (and, via the
        mapping above, `'gaussian'` when using `kernel=`). Must be a
        positive odd integer; non-integer values are rounded to the
        nearest integer, and even values incremented by 1, each with a
        warning.

    order : int
        Polynomial order for the `'savgol'` kernel (ignored otherwise).

    var : float
        Variance for the LEGACY `mode='gaussian'` path (`sigma = sqrt(var)`,
        default 300, matching the original weights-trajectory recipe).
        Ignored by `kernel='gaussian'`.

    maintain_bounds : bool
        If True (default), clip the smoothed output to each column's
        original (pre-smoothing) min/max. The bounds are derived from the
        data being transformed -- so a fitted `Smooth` reused on new data
        (via ``return_model=True``) clips to the NEW data's own range,
        never the fit-time range.

    Notes
    -----
    Smoothing is applied PER DATASET: each element of a list input is
    smoothed independently, so data never bleeds across dataset
    boundaries (audit F14-001/D01-001 fix).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from hypertools.manip import Smooth
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'y': np.sin(np.linspace(0, 6, 50))
    ...                         + 0.1 * rng.standard_normal(50)})
    >>> out = Smooth(kernel_width=11).fit_transform(df)
    >>> out.shape
    (50, 1)
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, kernel=None, mode='savgol', kernel_width=11, order=3, var=300,
                 maintain_bounds=True):
        required = ['axis', 'mode', 'kernel_width', 'order', 'var', 'maintain_bounds']
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
