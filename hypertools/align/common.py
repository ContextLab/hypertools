"""Base class + helpers for hypertools aligners (scikit-learn compatible).

An Aligner wraps a (fitter, transformer, required-params) triple operating on
a *list* of DataFrames: `fit` unstacks the stored data into that list, trims to
common rows and pads to common columns, runs the fitter, and stores the returned
dict as attributes; `transform` re-derives the list and runs the transformer with
those params. Child classes (HyperAlign, Procrustes, SharedResponseModel, ...)
supply the three pieces plus their defaults.
"""
import warnings

import datawrangler as dw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


def reject_unknown_kwargs(cls_name, kwargs, supported):
    """Raise a `TypeError` naming any unexpected constructor kwargs.

    Aligner constructors accept `**kwargs` (so the dispatcher can forward
    user options), which used to silently SWALLOW misspelled parameters --
    e.g. `hyp.align(data, n_itr=5)` (typo for `n_iter`) ran with defaults
    and returned results identical to an un-parameterized call
    (QC 2026-07, X2-error-quality-003). Every Aligner child calls this
    with its leftover `**kwargs` so typos fail loudly, matching the
    `TypeError` that `hyp.predict`/`hyp.manip` already raise.

    Parameters
    ----------
    cls_name : str
        Name of the Aligner class being constructed (for the message).
    kwargs : dict
        The leftover (unrecognized) keyword arguments; no-op if empty.
    supported : list of str
        The keyword argument names the class DOES accept.
    """
    if not kwargs:
        return
    supported_txt = ', '.join(supported) if supported else '(none)'
    raise TypeError(
        f"{cls_name}() got unexpected keyword argument(s): "
        f"{', '.join(sorted(kwargs))}. Supported keyword argument(s): "
        f"{supported_txt}. Check for typos; to pass parameters to a "
        "different pipeline stage, use that stage's dict spec (e.g. "
        "cluster={'model': 'KMeans', 'kwargs': {'n_clusters': 3}}).")


def pad(x, c=None):
    """Horizontally zero-pad a DataFrame (or list of DataFrames) to `c` columns."""
    if isinstance(x, list):
        if c is None:
            c = np.max([d.shape[1] for d in x])
        return [pad(d, c) for d in x]
    if c is None:
        return x
    y = np.zeros([x.shape[0], c])
    n = np.min([c, x.shape[1]])
    y[:, :n] = x.iloc[:, :n]
    return pd.DataFrame(data=y, index=x.index.copy())


def trim_and_pad(data, warn=True):
    """Select the common rows across a list of DataFrames and pad to common columns.

    Rows are matched across datasets by index VALUE, and the kept rows are
    returned in the FIRST dataset's index order (deterministic). An earlier
    implementation collected the common rows via ``list(set(...))``, whose
    hash-bucket iteration order silently SCRAMBLED the observation/time
    order of every output dataset for any non-RangeIndex index --
    DatetimeIndex timeseries, string labels, shuffled integers
    (QC 2026-07, F12-align-001).

    Parameters
    ----------
    data : DataFrame or list of DataFrames
        The dataset(s) to trim/pad.
    warn : bool
        Whether to emit the data-loss UserWarning when rows are trimmed
        (default: True). `Aligner.transform` passes False when replaying
        the fit-time data, so a single `fit_transform` warns exactly once.

    Raises
    ------
    ValueError
        If the datasets share NO common row-index values (there is nothing
        to align; returning empty output silently hid mismatched inputs).
    """
    if len(data) == 0:
        return data
    if not isinstance(data, list):
        data = [data]
    common = set(data[0].index.values)
    for d in data[1:]:
        common = common.intersection(set(d.index.values))
    c = np.max([x.shape[1] for x in data])
    # preserve the FIRST dataset's observation order (dropping duplicate
    # index values after their first occurrence, as the set() form did)
    rows = []
    for r in data[0].index.values:
        if r in common:
            rows.append(r)
            common.discard(r)
    if len(rows) == 0:
        raise ValueError(
            "datasets share no common row-index values, so there is "
            "nothing to align. Alignment matches observations across "
            "datasets by their row index; reindex the datasets so "
            "corresponding observations share index values (e.g. "
            "df.reset_index(drop=True) to match rows by position).")
    # warn on data loss: alignment keeps only the rows COMMON to every dataset
    # (matched observation-by-observation), so datasets with different row
    # counts / indices are trimmed. This used to happen silently (QC 2026-07).
    if warn and any(len(rows) < d.shape[0] for d in data):
        warnings.warn(
            f"alignment keeps only the {len(rows)} row(s) common to all "
            "datasets; datasets with more rows were trimmed. Align datasets "
            "with matching numbers of observations to avoid dropping data.")
    return [pad(d.loc[rows], c) for d in data]


class Aligner(BaseEstimator):
    """Scikit-learn-compatible base class for hypertools aligners.

    Wraps a `(fitter, transformer, required)` triple that operates on a
    *list* of DataFrames: `fit` unstacks the stored data into that list,
    trims to common rows and zero-pads to common columns (see
    `trim_and_pad`), runs `fitter` on it, and stores each key of the
    returned dict as an attribute on `self`; `transform` re-derives the
    list the same way and runs `transformer` with those fitted params.
    Child classes (e.g. HyperAlign, Procrustes, SharedResponseModel)
    supply `fitter`, `transformer`, and `required` (the list of
    attribute names `fitter` must return) via `**kwargs` to `__init__`.

    Parameters
    ----------
    **kwargs
        `data` : the dataset(s) to align (may be `None` until `fit` is
        called). `fitter` : callable that fits the alignment and returns
        a dict of parameters. `transformer` : callable that applies a
        fitted alignment. `required` : list of parameter names `fitter`
        must return. Any remaining kwargs are forwarded to `fitter`/
        `transformer` on every call.
    """

    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        # Remaining kwargs are the aligner's configuration parameters (e.g.
        # Procrustes's `target`/`scaling`, HyperAlign's `n_iter`, SRM's
        # `features`). Store each as its OWN attribute -- the scikit-learn
        # estimator convention that `get_params`/`set_params`/`clone` rely
        # on (every __init__ parameter must be readable as `self.<param>`).
        # Folding them into a single `self.kwargs` dict left e.g.
        # `Procrustes().get_params()` raising `AttributeError: 'Procrustes'
        # object has no attribute 'target'` (QC 2026-07). Remember their
        # names so the `kwargs` property below can rebuild the dict that
        # `fit`/`transform` forward to the fitter/transformer.
        self._param_names = list(kwargs.keys())
        for name, value in kwargs.items():
            setattr(self, name, value)
        self._fit_shape = None

    @property
    def kwargs(self):
        """Configuration kwargs forwarded to `fitter`/`transformer`, rebuilt
        from the individually-stored parameter attributes (see `__init__`).
        Reading it always reflects the current attribute values, so
        `set_params(...)` correctly changes what `fit`/`transform` use."""
        return {name: getattr(self, name) for name in self._param_names}

    @property
    def is_fitted(self):
        """Whether `fit`/`fit_transform` has already been run.

        Lets a fitted `Aligner` returned from an earlier
        `hypertools.align.align.align(..., return_model=True)` call be
        passed back in as `model=` on NEW data and reuse its learned
        alignment via `transform`, without re-fitting.
        """
        return self.data is not None

    @staticmethod
    def _shape_of(data):
        """`(n_datasets, [n_columns_per_dataset])` for `data` (a single
        DataFrame/array, or a list of them) -- the shape `transform`
        validates new data against (GH #227)."""
        items = data if isinstance(data, list) else [data]
        return len(items), [np.asarray(d).shape[1] for d in items]

    def fit(self, data):
        """Fit the alignment on `data` and store the fitted parameters.

        Records `data` and its shape (for later validation in
        `transform`), then -- if `self.fitter` is set -- unstacks,
        trims, and pads `data` (see `trim_and_pad`) and calls
        `self.fitter` on it, setting each key of the returned dict as an
        attribute on `self`.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit the alignment on.

        Raises
        ------
        ValueError
            If `data` is `None` or an empty list, if `self.fitter` does
            not return a dict, or if any name in `self.required` is
            missing from the returned dict.
        """
        if data is None or (isinstance(data, list) and len(data) == 0):
            raise ValueError(
                'cannot align an empty dataset: no data provided. Pass one '
                'or more numeric arrays/DataFrames to fit the aligner on.')
        self.data = data
        self._fit_shape = self._shape_of(data)
        if self.fitter is None:
            return
        data = trim_and_pad(dw.unstack(self.data))
        params = self.fitter(data, **self.kwargs)
        if not isinstance(params, dict):
            raise ValueError(
                f'aligner fit function must return a dictionary of fitted '
                f'parameters; got {type(params).__name__}')
        missing = [r for r in self.required if r not in params]
        if missing:
            raise ValueError(
                f"aligner fit function did not return required field(s): "
                f"{', '.join(missing)}")
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, new_data=None):
        """Apply the fitted alignment to `new_data`.

        Parameters
        ----------
        new_data : DataFrame, array, list of these, or None
            Held-out data to project into the fitted common space. Must
            have the same number of datasets, each with the same number of
            columns, as the data `fit` was called with (GH #227) --
            raises `ValueError` naming the fit-time shape otherwise.
            `None` (default) replays the fit-time data itself (no shape
            check needed, since it trivially matches).

        Returns
        -------
        The aligned `new_data` (or the aligned fit-time data, when
        `new_data` is `None`), in the same list/single-item shape as the
        input.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet, or a
            required fitted attribute is missing.
        ValueError
            If `new_data`'s shape (dataset count or any per-dataset
            column count) does not match the fit-time shape.
        """
        if self.data is None:
            raise NotFittedError('must fit aligner before transforming data')
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f'missing fitted attribute: {r}')

        if new_data is None:
            data_to_use = self.data
        else:
            # `new_data` may arrive as raw array(s) rather than
            # DataFrame(s) -- e.g. `model.transform(...)` called directly
            # (bypassing the `@dw.decorate.funnel`/`format_data` coercion
            # `align()` applies before `fit`). Coerce here (single
            # array|DataFrame, or list of these) to the same DataFrame(s)
            # format `fit` uses, BEFORE shape validation/`dw.unstack` below
            # -- `dw.wrangle` preserves each DataFrame's index and the
            # single-vs-list shape of the input, matching the funnel path.
            new_data = dw.wrangle(new_data)
            n_datasets, n_columns = self._shape_of(new_data)
            fit_n_datasets, fit_n_columns = self._fit_shape
            if n_datasets != fit_n_datasets or n_columns != fit_n_columns:
                raise ValueError(
                    f"aligner was fit on {fit_n_datasets} dataset(s) with "
                    f"{fit_n_columns} column(s) each; got {n_datasets} "
                    f"dataset(s) with {n_columns} column(s) (fit-time "
                    f"shape: {fit_n_datasets} datasets x {fit_n_columns} "
                    f"columns)")
            data_to_use = new_data

        if self.transformer is None:
            return data_to_use
        # only warn about trimmed rows for genuinely NEW data -- when
        # replaying the fit-time data (new_data=None, e.g. from
        # fit_transform), fit's own trim_and_pad already warned, and
        # repeating the identical warning misled users into thinking two
        # separate trims happened (QC 2026-07, F12-align-003)
        data = trim_and_pad(dw.unstack(data_to_use), warn=new_data is not None)
        required_params = {r: getattr(self, r) for r in self.required}
        return self.transformer(data, **dw.core.update_dict(required_params, self.kwargs))

    def fit_transform(self, data):
        """Fit the alignment on `data`, then immediately transform it.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit and align.

        Returns
        -------
        The aligned `data`, in the same list/single-item shape as the
        input (see `transform`).
        """
        self.fit(data)
        # replay the just-fit data (rather than re-passing `data`) so the
        # trim warning fires once per call and the redundant shape
        # re-validation is skipped -- `transform(None)` uses `self.data`,
        # which `fit` just stored (F12-align-003)
        return self.transform()
