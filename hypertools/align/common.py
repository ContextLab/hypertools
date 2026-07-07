"""Base class + helpers for hypertools aligners (scikit-learn compatible).

An Aligner wraps a (fitter, transformer, required-params) triple operating on
a *list* of DataFrames: `fit` unstacks the stored data into that list, trims to
common rows and pads to common columns, runs the fitter, and stores the returned
dict as attributes; `transform` re-derives the list and runs the transformer with
those params. Child classes (HyperAlign, Procrustes, SharedResponseModel, ...)
supply the three pieces plus their defaults.
"""
import datawrangler as dw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


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


def trim_and_pad(data):
    """Select the common rows across a list of DataFrames and pad to common columns."""
    if len(data) == 0:
        return data
    if not isinstance(data, list):
        data = [data]
    rows = set(data[0].index.values)
    for d in data[1:]:
        rows = rows.intersection(set(d.index.values))
    c = np.max([x.shape[1] for x in data])
    rows = list(rows)
    return [pad(d.loc[rows], c) for d in data]


class Aligner(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs
        self._fit_shape = None

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
        assert data is not None, ValueError('cannot align empty dataset')
        self.data = data
        self._fit_shape = self._shape_of(data)
        if self.fitter is None:
            return
        data = trim_and_pad(dw.unstack(self.data))
        params = self.fitter(data, **self.kwargs)
        assert isinstance(params, dict), ValueError('fit function must return a dictionary')
        assert all([r in params.keys() for r in self.required]), \
            ValueError('one or more required fields not returned')
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
        data = trim_and_pad(dw.unstack(data_to_use))
        required_params = {r: getattr(self, r) for r in self.required}
        return self.transformer(data, **dw.core.update_dict(required_params, self.kwargs))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
