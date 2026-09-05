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


def _resolve_center(kwargs):
    """Validate `center=` and `min_periods=` up front, and enforce which
    kernels support `center=False` (GH #285).

    `center=`/`min_periods=` deliberately reuse pandas' own
    `rolling(...)` vocabulary (rather than an `align='center'|'trailing'`
    kwarg) since `hypertools`' cross-module API already has an unrelated
    top-level `align=` (the alignment STAGE, e.g. `HyperAlign`); a
    same-named `Smooth` kwarg would collide with it in
    `hyp.manip`/`hyp.plot`/`hyp.analyze`.

    Only `kernel='boxcar'` has an unambiguous, well-established trailing
    (causal) definition: a plain moving average, exactly
    `pd.Series(x).rolling(kernel_width, min_periods=...).mean()`. Neither
    `scipy.signal.savgol_filter` nor `scipy.ndimage.gaussian_filter1d`
    ships a causal/one-sided variant, and there is no single canonical way
    to build one (a truncated one-sided Gaussian and a causal Savitzky-
    Golay fit are both used in the literature, with different tradeoffs) --
    shipping an ad hoc version here would be a silent, unreviewable
    correctness risk. So `center=False` is refused for `'savgol'`/
    `'gaussian'` with a clear `ValueError` pointing at `kernel='boxcar'`
    (or `center=True`) instead, rather than guessing.

    `min_periods=` (pandas-style: the number of non-NaN values required
    before a window produces a value) is only meaningful for the trailing
    (`center=False`) boxcar path; passing it with `center=True` raises
    `ValueError` rather than silently ignoring it.
    """
    center = kwargs.get('center', True)
    if not isinstance(center, bool):
        raise ValueError(f'invalid Smooth center {center!r}; must be True or False')
    min_periods = kwargs.get('min_periods', None)
    if center and min_periods is not None:
        raise ValueError(
            "min_periods is only meaningful with center=False (a centered "
            "kernel has no partial-window/warm-up region); pass "
            "center=False or drop min_periods.")
    return center, min_periods


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
    center, min_periods = _resolve_center(kwargs)
    if not center and branch != 'boxcar':
        raise ValueError(
            f"center=False is only supported for kernel='boxcar' (got "
            f"kernel={branch!r}); savgol/gaussian have no unambiguous "
            "causal (trailing) definition. Use kernel='boxcar', or "
            "center=True (the default).")
    # NaN handling (2026-07 release audit, final wave item 16): missing
    # values used to behave differently per kernel -- savgol propagated or
    # crashed with a raw scipy error, while gaussian/boxcar silently SPREAD
    # each NaN into its neighbors -- so all three kernels now raise the
    # same clear error up front.
    try:
        all_values = data.to_numpy(dtype=float)
    except (TypeError, ValueError):
        all_values = None
    if all_values is not None and np.isnan(all_values).any():
        n_missing = int(np.isnan(all_values).sum())
        raise ValueError(
            f'Smooth cannot smooth data containing NaN ({n_missing} '
            'missing value(s) found): the savgol, gaussian, and boxcar '
            'kernels would either fail or silently spread missing values '
            'into neighboring samples. Fill missing values first (e.g. '
            'hyp.impute(data)).')
    # clear, actionable errors for kernel_width edge cases (QC 2026-07): scipy
    # otherwise raises opaque "window_length must be <= size of x" /
    # "polyorder must be less than window_length" from deep in savgol_filter.
    # A trailing boxcar has no such requirement -- pandas rolling handles a
    # window wider than the data gracefully via min_periods (and min_periods=1
    # is exactly the documented expanding-start use case), so this guard is
    # centered-kernel only.
    kw = kwargs.get('kernel_width')
    n_rows = data.shape[0]
    if center and kw is not None and kw > n_rows:
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
        if not center:
            # byte-identical to pd.Series(values).rolling(kernel_width,
            # min_periods=...).mean() (GH #285): min_periods defaults to
            # kernel_width (pandas' own default), giving NaN for the first
            # kernel_width - 1 rows; min_periods=1 gives an expanding start.
            effective_min_periods = kwargs['kernel_width'] if min_periods is None else min_periods
            smoothed_values = pd.Series(values).rolling(
                window=kwargs['kernel_width'], min_periods=effective_min_periods).mean().to_numpy()
        elif branch == 'gaussian':
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
        raise ValueError(
            f'invalid Smooth axis {axis!r}; axis must be 0 (smooth down '
            'each column, the default) or 1 (smooth across each row).')
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
        `kernel_width` resolves to a non-positive value; or if the data
        contain NaN (fill missing values first, e.g. `hyp.impute(data)` --
        identical behavior for all three kernels).
    """
    if 'axis' not in kwargs:
        raise ValueError(
            "Smooth's transformer requires an axis= parameter; pass axis=0 "
            '(smooth down each column, the default) or axis=1 (smooth '
            'across each row).')

    # coerce kernel_width ONCE, up front, so warnings fire once per call even
    # for list input, and so the effective width matches the warning text
    # (audit F14-008: the old code warned "Rounding ... to the nearest
    # integer" but then TRUNCATED, e.g. 11.7 -> 11 instead of 12 -> odd 13).
    # center=False (GH #285) skips the ODD-integer bump: a trailing
    # (causal) boxcar window has no symmetry requirement -- pandas rolling
    # accepts any positive integer window, and forcing it odd would make
    # `center=False, kernel_width=12` (the weather-tutorial's
    # `rolling(12).mean()` recipe) silently smooth over a different width.
    center = kwargs.get('center', True)
    kw = kwargs.get('kernel_width')
    if kw is not None:
        if kw != int(np.round(kw)):
            warnings.warn('Rounding smoothing kernel width to the nearest integer')
        kw = int(np.round(kw))
        if center and kw % 2 != 1:
            warnings.warn('Increasing smoothing kernel width by 1 (must be odd)')
            kw += 1
        if kw <= 0:
            requirement = 'a positive odd integer' if center else 'a positive integer'
            raise ValueError(
                f'kernel_width must be {requirement}; got {kwargs["kernel_width"]!r}')
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

    center : bool
        Where the smoothing window sits relative to each output sample
        (GH #285, pandas' own `rolling(...)` vocabulary -- NOT
        `align='center'|'trailing'`, since `hypertools`'s cross-module
        API already has an unrelated top-level `align=` for the
        alignment STAGE, e.g. `HyperAlign`; a same-named `Smooth` kwarg
        would collide with it in `hyp.manip`/`hyp.plot`/`hyp.analyze`).
        Defaults to `True` (unchanged pre-existing behavior for all
        three kernels: a symmetric window straddling each sample, no
        NaNs introduced).

        - `False`: a CAUSAL (backward-looking, trailing) window ending
          at each sample -- only supported for `kernel='boxcar'`, where
          it is byte-identical to
          ``pd.Series(x).rolling(kernel_width, min_periods=min_periods).mean()``
          (NaN for the first `kernel_width - 1` rows with the default
          `min_periods`). `kernel_width` is NOT forced odd for
          `center=False` (a causal window has no symmetry requirement).
          Requesting `center=False` with `kernel='savgol'`/`'gaussian'`
          raises `ValueError`: neither `scipy.signal.savgol_filter` nor
          `scipy.ndimage.gaussian_filter1d` has a causal/one-sided
          variant, and there is no single canonical way to build one (a
          truncated one-sided Gaussian and a causal Savitzky-Golay fit
          are both used in the literature, with different tradeoffs) --
          shipping an ad hoc version would be a silent correctness risk,
          so this is refused rather than guessed. Use `kernel='boxcar'`
          for a trailing smoother, or `center=True` (the default) for
          savgol/gaussian.

    min_periods : int or None
        Only meaningful for `center=False` (pandas-rolling-style): the
        minimum number of non-NaN values in the trailing window before
        it produces a value. `None` (default) uses `kernel_width` itself
        (pandas' own default: NaN for the first `kernel_width - 1`
        rows). `min_periods=1` gives an expanding start (the first
        output sample is the input itself, the second is a 2-sample
        average, etc., with no leading NaNs). Passing a non-`None` value
        together with `center=True` raises `ValueError` (a centered
        kernel has no partial-window/warm-up region for it to control).

    Notes
    -----
    Smoothing is applied PER DATASET: each element of a list input is
    smoothed independently, so data never bleeds across dataset
    boundaries (audit F14-001/D01-001 fix).

    Data containing NaN raises a `ValueError` (identical for all three
    kernels): smoothing would either fail or silently spread missing
    values into neighboring samples. Fill missing values first (e.g.
    ``hyp.impute(data)``) -- 2026-07 release audit, final wave item 16.

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

    A trailing (causal) boxcar moving average, matching
    ``pd.Series(x).rolling(12).mean()`` exactly (NaN for the first 11
    rows):

    >>> df2 = pd.DataFrame({'y': np.arange(20, dtype=float)})
    >>> out2 = Smooth(kernel='boxcar', kernel_width=12,
    ...                center=False).fit_transform(df2)
    >>> bool(out2['y'].iloc[:11].isna().all())
    True
    >>> import pandas as pd
    >>> expected = df2['y'].rolling(12).mean()
    >>> bool(np.allclose(out2['y'].to_numpy(), expected.to_numpy(),
    ...                   equal_nan=True))
    True
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, axis=0, kernel=None, mode='savgol', kernel_width=11, order=3, var=300,
                 maintain_bounds=True, center=True, min_periods=None):
        required = ['axis', 'mode', 'kernel_width', 'order', 'var', 'maintain_bounds']
        super().__init__(axis=axis, fitter=fitter, transformer=transformer, data=None, mode=mode,
                         kernel=kernel, kernel_width=kernel_width, order=order, var=var,
                         maintain_bounds=maintain_bounds, center=center, min_periods=min_periods,
                         required=required)

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
        self.center = center
        self.min_periods = min_periods
        self.required = required
