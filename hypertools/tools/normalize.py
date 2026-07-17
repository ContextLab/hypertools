#!/usr/bin/env python
"""hyp.normalize: z-score the columns/rows of an array (or list of arrays).

Adds `return_model=True` (a fitted `Normalizer` wrapper -- round17 Task 6,
GH #138) and the same cross-module stage kwargs (`manip=`, `reduce=`,
`ndims=`, `align=`, `cluster=`) every other 1.0 dispatcher accepts, on top
of the classic z-scoring behavior (kept byte-identical by default).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .format_data import format_data as formatter


def _as_list_2d(x):
    """Coerce `x` to a list of 2-D float arrays, tracking single-vs-list input.

    The classic ``normalize()`` entry point runs ``format_data`` first, so its
    ``Normalizer`` only ever sees a list of 2-D arrays. But a fitted
    ``Normalizer`` returned by ``return_model=True`` is documented to be
    reapplied directly via ``.transform(new_data)`` -- where ``new_data`` is
    typically a bare 2-D array. This coerces either shape (a single array/
    DataFrame, or a list of them) to a list of 2-D float arrays so the
    per-column z-scoring below is well-defined.

    Returns
    -------
    (list of numpy.ndarray, bool)
        The 2-D arrays, and whether the original input was a single array
        (so callers can return single-in -> single-out).
    """
    if isinstance(x, (list, tuple)):
        return [np.atleast_2d(np.asarray(a, dtype=np.float64)) for a in x], False
    return [np.atleast_2d(np.asarray(x, dtype=np.float64))], True


def _check_column_counts(arrs):
    """`'across'`-mode z-scoring stacks every dataset row-wise, so they must
    all have the same number of columns; raise a clear ValueError naming the
    mismatched counts instead of leaking numpy's vstack internals (audit
    F14-018)."""
    cols = [a.shape[1] for a in arrs]
    if len(set(cols)) > 1:
        raise ValueError(
            "normalize='across' requires every dataset to have the same "
            "number of columns (z-scores are computed per column across all "
            f"datasets); got column counts {cols}. Use normalize='within' "
            "or normalize='row' to z-score each dataset independently.")


def _zscore_column(mean, std, y):
    """Z-score `y` against a given `mean`/`std`, matching the classic
    `normalize()` degenerate-input handling: an empty or constant-valued
    `y`, or a zero-variance `std`, returns zeros rather than dividing by
    zero."""
    y = np.asarray(y)
    if len(y) == 0 or len(set(y.ravel())) <= 1:
        return np.zeros_like(y, dtype=np.float64)
    if std == 0:
        return np.zeros_like(y, dtype=np.float64)
    return (y - mean) / std


class Normalizer(BaseEstimator):
    """Fit/transform wrapper around `normalize`'s z-scoring, capturing
    fit-time statistics so a `return_model=True` result can be reapplied to
    NEW data via `.transform` without re-estimating them -- mirrors
    `hypertools.reduce.common.Reducer`/`hypertools.cluster.common.Clusterer`/
    `hypertools.align.common.Aligner`/`hypertools.manip.common.Manipulator`'s
    already-fitted-instance reuse contract (a fitted `Normalizer` passed
    back in as `normalize=` is applied via `.transform`, never refit).

    Only `normalize='across'` mode has cross-call state worth reusing: its
    per-column mean/std are computed once, by stacking every dataset in the
    fit-time list, and then frozen. `'within'`/`'row'` are inherently
    self-referential (each dataset's/row's own values determine its own
    z-score), so `.transform` on either simply re-runs the same
    self-referential z-score on whatever data it is given -- byte-identical
    to calling `normalize()` fresh on that data, regardless of what it was
    fit on.

    Parameters
    ----------
    normalize : {'across', 'within', 'row'}
        Which z-scoring mode to use (see `normalize`'s docstring).

    Attributes
    ----------
    mean_, std_ : numpy.ndarray or None
        The fit-time per-column mean/std (`'across'` mode only); `None`
        until `fit`/`fit_transform` runs, and always `None` for
        `'within'`/`'row'` (no state to capture).
    """

    def __init__(self, normalize='across'):
        self.normalize = normalize
        self.mean_ = None
        self.std_ = None

    @property
    def is_fitted(self):
        """Whether `fit`/`fit_transform` has already been run.

        `'within'`/`'row'` modes have no fit-time state to reuse, so they
        report fitted immediately -- their `.transform` always re-derives
        statistics from whatever data it is given, matching `normalize()`'s
        classic self-referential behavior for those modes.
        """
        if self.normalize == 'across':
            return self.mean_ is not None
        return True

    def fit(self, x):
        """Compute per-column mean/std across the stacked fit-time data
        (`'across'` mode only; a no-op for `'within'`/`'row'`).

        Accepts either a single 2-D array or a list of them.
        """
        if self.normalize == 'across':
            arrs, _ = _as_list_2d(x)
            _check_column_counts(arrs)
            x_stacked = np.vstack(arrs)
            self.mean_ = np.mean(x_stacked, axis=0)
            self.std_ = np.std(x_stacked, axis=0)
        return self

    def transform(self, x):
        """Apply this normalizer's z-scoring to new data.

        `x` may be a single 2-D array (or DataFrame) or a list of them; the
        result mirrors the input (single array in -> single array out, list
        in -> list out), matching `normalize()`'s own convention so a fitted
        `Normalizer` can be reused directly on held-out data.
        """
        arrs, single = _as_list_2d(x)
        if self.normalize == 'across':
            if self.mean_ is None:
                raise NotFittedError('must fit Normalizer before transforming data')
            _check_column_counts(arrs)
            if arrs[0].shape[1] != self.mean_.shape[0]:
                raise ValueError(
                    f'Normalizer was fit on {self.mean_.shape[0]} column(s) '
                    f'but got {arrs[0].shape[1]}')
            out = [
                np.array([_zscore_column(self.mean_[j], self.std_[j], i[:, j])
                          for j in range(i.shape[1])]).T
                for i in arrs
            ]
        elif self.normalize == 'within':
            out = [
                np.array([_zscore_column(np.mean(i[:, j]), np.std(i[:, j]), i[:, j])
                          for j in range(i.shape[1])]).T
                for i in arrs
            ]
        elif self.normalize == 'row':
            out = [
                np.array([_zscore_column(np.mean(i[j, :]), np.std(i[j, :]), i[j, :])
                          for j in range(i.shape[0])])
                for i in arrs
            ]
        else:
            raise ValueError(
                f"normalize must be 'across', 'within', or 'row'; got {self.normalize!r}")
        return out[0] if single else out

    def fit_transform(self, x):
        """Fit then transform `x` (equivalent to `fit(x)` followed by
        `transform(x)`)."""
        self.fit(x)
        return self.transform(x)


def normalize(x, normalize='across', internal=False, format_data=True, impute=None,
             return_model=False, manip=None, reduce=None, ndims=None, align=None,
             cluster=None):
    """
    Z-transform the columns or rows of an array, or list of arrays

    This function normalizes the rows or columns of the input array(s).  This
    can be useful because data reduction and machine learning techniques are
    sensitive to scaling differences between features. By default, the function
    is set to normalize 'across' the columns of all lists, but it can also
    normalize the columns 'within' each individual list, or alternatively, for
    each row in the array.

    Parameters
    ----------
    x : Numpy array or list of arrays
        This can either be a single array, or list of arrays

    normalize : str, False, None, or fitted Normalizer
        If set to 'across', the columns of the input data will be z-scored
        across lists (default). That is, the z-scores will be computed with
        respect to column n across all arrays passed in the list. If set
        to 'within', the columns will be z-scored within each list that is
        passed. If set to 'row', each row of the input data will be z-scored.
        If set to False or None, the input data will be returned with no
        z-scoring.
        A previously-fitted `Normalizer` (as returned by `return_model=True`)
        is applied via `.transform` instead of being refit. Any other value
        raises a `ValueError`.

    internal : bool
        If True, ALWAYS return a list (one array per input dataset), even
        for single-dataset input -- used by hypertools' own pipeline
        plumbing (default: False).

    format_data : bool
        Whether or not to first call the format_data function (default: True).

    impute : str, dict, class, class instance or None
        Overrides the default PPCA missing-data fill applied during the
        `format_data` stage with a different `hypertools.impute` model
        (default: None, i.e. PPCA). Only used when `format_data` is True.

    return_model : bool
        If True, also return the fitted model: the fitted `Normalizer` when
        only the `normalize` stage ran, or a fitted `hypertools.Pipeline`
        when `manip=`/`reduce=`/`align=`/`cluster=` made multiple stages run
        (default: False).

    manip, reduce, align, cluster : model spec or None
        Cross-module stage kwargs (GH #138): when any of these is given,
        the other stages also run (via
        `hypertools.core.pipeline.build_pipeline`), in the canonical order
        `manip -> normalize -> reduce -> align -> cluster` (GH #153), with
        this function's own `normalize=` slotted in at the normalize stage
        (default: None for all four, i.e. only `normalize` runs).

    ndims : int or None
        Passed through to the `reduce` stage (as `ndims=`) when `reduce=`
        is also given.

    Returns
    ----------
    normalized_x : Numpy array or list of arrays
        An array or list of arrays where the columns or rows are z-scored. If
        the input was a list with more than one element, a list is returned;
        a single array -- or a single-element list -- returns a bare array.
        DataFrame inputs are converted to arrays (index/column metadata is
        not preserved; use `hypertools.manip` with ``model='ZScore'`` to keep
        it). If `return_model=True`, a `(normalized_x, model)` tuple is
        returned instead.

    Notes
    -----
    Standard deviations are POPULATION standard deviations (``ddof=0``, the
    ``np.std``/``scipy.stats.zscore`` convention). `hypertools.manip`'s
    `ZScore` manipulator uses the SAMPLE standard deviation (``ddof=1``, the
    pandas convention), so the two z-scoring entry points differ by a factor
    of ``sqrt(n / (n - 1))``. Missing values (NaN) are PPCA-imputed during
    the `format_data` stage (when `format_data=True`); `hypertools.manip`
    propagates NaNs unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from hypertools import normalize
    >>> x = np.array([[1., 0.], [2., 10.], [3., 20.], [4., 30.]])
    >>> z = normalize(x, normalize='within')
    >>> np.allclose(z.mean(axis=0), 0.0) and np.allclose(z.std(axis=0), 1.0)
    True
    >>> a, b = np.zeros((5, 2)), np.ones((5, 2))
    >>> z_across = normalize([a, b], normalize='across')
    >>> np.allclose(np.vstack(z_across).mean(axis=0), 0.0)
    True
    """
    # cross-module kwargs (#138): assemble and run a Pipeline (in canonical
    # order, #153) instead of the single-stage path below whenever another
    # stage is requested. Lazy import avoids a normalize<->core.pipeline
    # cycle (core.pipeline itself lazily imports tools.normalize).
    if any(stage is not None for stage in (manip, reduce, align, cluster)):
        from ..core.pipeline import build_pipeline
        if impute is not None and normalize not in (False, None):
            # thread impute= through the same way the legacy (single-stage)
            # path below does (impute at format time, BEFORE any pipeline
            # stage runs): build_pipeline's normalize stage re-enters this
            # function with return_model=True but no impute=, so without
            # this it always falls back to PPCA regardless of what impute=
            # was passed here. Gated on `normalize not in (False, None)` to
            # match the legacy gating a few lines down (`if normalize in
            # [False, None]: return x` -- format_data, and therefore
            # impute=, only runs when normalization is actually requested).
            x = formatter(x, ppca=True, impute=impute)
        pipeline = build_pipeline(manip=manip, normalize=normalize, reduce=reduce,
                                  ndims=ndims, align=align, cluster=cluster)
        result = pipeline.fit_transform(x)
        return (result, pipeline) if return_model else result

    # a real ValueError naming the actual parameter (audit F14-009: this was
    # an assert -- stripped under `python -O` -- whose message referenced
    # 'scale_type', a parameter that does not exist in the 1.0 API)
    if not (normalize in ['across', 'within', 'row', False, None]
            or isinstance(normalize, Normalizer)):
        raise ValueError(
            "normalize must be one of 'across', 'within', 'row', False/None "
            f"(to skip normalization), or a fitted Normalizer; got "
            f"{normalize!r}")

    if normalize in [False, None]:
        return (x, None) if return_model else x

    if format_data:
        x = formatter(x, ppca=True, impute=impute)

    if isinstance(normalize, Normalizer):
        normalizer_ = normalize
        normalized_x = normalizer_.transform(x) if normalizer_.is_fitted \
            else normalizer_.fit_transform(x)
    else:
        normalizer_ = Normalizer(normalize)
        normalized_x = normalizer_.fit_transform(x)

    if internal or len(normalized_x) > 1:
        result = normalized_x
    else:
        result = normalized_x[0]

    return (result, normalizer_) if return_model else result
