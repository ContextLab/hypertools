"""Base class for hypertools imputers (scikit-learn compatible).

An Imputer wraps a (fitter, transformer, required-params) triple, mirroring
`hypertools.manip.common.Manipulator`, but with SAME-SHAPE, list-aware
fit/transform semantics: `fit(data)` stacks a list of datasets (which must
share columns) into one array and fits ONE set of imputation parameters
jointly across them -- mirroring the pre-1.0 `format_data.fill_missing`
behavior of pooling every dataset for a single PPCA fit -- storing the
fitted params in `models_`. `transform(data=None)` applies those params: to
the ORIGINAL fitted data by default, or to brand-new data passed in (the
`return_model=True` reuse path -- see `hypertools.impute.impute`: a
previously-fit Imputer's learned parameters, e.g. PPCA's components, a fit
KNNImputer, or a Kalman filter's transition/observation matrices, are
applied to new data WITHOUT re-fitting). List-in/list-out: if `fit` (or
`transform`) receives a list, the per-dataset row boundaries are recorded
and the (jointly-imputed) result is split back into a list matching the
input structure.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..core.shared import as_dataframe as _as_dataframe


def _shared_columns(datasets):
    """Whether every dataset in `datasets` has the same columns (in order)."""
    first = list(datasets[0].columns)
    return all(list(d.columns) == first for d in datasets[1:])


def _stack(datasets):
    if len(datasets) == 1:
        return datasets[0], [len(datasets[0])], [datasets[0].index]
    # joint imputation requires shared columns: pd.concat of mismatched
    # columns silently WIDENED every dataset to the column union, "filling"
    # the invented all-NaN blocks with the other datasets' statistics (QC
    # 2026-07 red-team F17-impute-001). The `hypertools.impute.impute`
    # dispatcher falls back to per-dataset imputation (with a warning)
    # before reaching this point; direct Imputer use fails clearly instead.
    if not _shared_columns(datasets):
        raise ValueError(
            'datasets in a list must share columns to be imputed jointly; '
            f'got columns {[list(d.columns) for d in datasets]}. Impute '
            'each dataset separately (or rename/align the columns first).')
    stacked = pd.concat(datasets, axis=0, ignore_index=True)
    return stacked, [len(d) for d in datasets], [d.index for d in datasets]


def _split(stacked, boundaries, indexes=None):
    if len(boundaries) == 1:
        return [stacked]
    bounds = np.cumsum(boundaries[:-1])
    parts = np.split(stacked.to_numpy(), bounds, axis=0)
    if indexes is None:
        indexes = [None] * len(parts)
    # restore each dataset's own index (it used to be reset to a RangeIndex
    # on the list path only -- QC 2026-07 red-team F17-impute-012)
    return [pd.DataFrame(p, columns=stacked.columns, index=idx)
            for p, idx in zip(parts, indexes)]


class Imputer(BaseEstimator):
    """Scikit-learn-compatible base class for hypertools imputers.

    Wraps a `(fitter, transformer, required)` triple with same-shape,
    list-aware fit/transform semantics: `fit(data)` stacks a list of
    datasets (which must share columns) into one array and fits ONE set
    of imputation parameters jointly across them, storing the fitted
    params in `models_`. `transform(data=None)` applies those params to
    the original fitted data by default, or to new data passed in (the
    `return_model=True` reuse path). If `fit`/`transform` receives a
    list, the per-dataset row boundaries (and indexes) are recorded and
    the result is split back into a list matching the input structure.
    Output values are always float64 (imputation coerces to float
    internally), even on a NaN-free passthrough.

    Parameters
    ----------
    **kwargs
        `data` : the dataset(s) to impute (may be `None` until `fit` is
        called). `fitter` : callable that fits the imputation and
        returns a dict of parameters. `transformer` : callable that
        applies fitted imputation parameters. `required` : list of
        parameter names `fitter` must return. Any remaining kwargs are
        forwarded to `fitter`/`transformer` on every call.
    """

    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        """Fit the imputer on `data` and store the fitted parameters.

        Stacks `data` (a single dataset or a list of datasets, coerced
        to DataFrames) into one array, recording the per-dataset row
        boundaries, then -- if `self.fitter` is set -- calls it on the
        stacked data to jointly fit imputation parameters, storing the
        result in `self.models_`.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit the imputer on.

        Returns
        -------
        Imputer
            `self`, for chaining.

        Raises
        ------
        ValueError
            If `data` is `None` or empty, if the datasets in a list do
            not share columns, if `self.fitter` does not return a dict,
            or if any name in `self.required` is missing from the
            returned dict.
        """
        # real raises (not `assert ..., ValueError(...)`, which raises
        # AssertionError and is stripped under `python -O`) -- QC 2026-07.
        from ..core.shared import no_observations_message
        if data is None:
            raise ValueError(
                no_observations_message('impute', 'data is None'))
        single = not isinstance(data, list)
        if not single and len(data) == 0:
            raise ValueError(
                no_observations_message('impute', 'got an empty list'))
        datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]
        stacked, boundaries, indexes = _stack(datasets)
        if stacked.shape[0] == 0 or stacked.shape[1] == 0:
            raise ValueError(
                no_observations_message(
                    'impute', f'got shape {tuple(stacked.shape)}')
                + ' Pass at least 1 observation (row) of at least 1 '
                'feature (column).')

        self._single = single
        self._boundaries = boundaries
        self._indexes = indexes
        self.data = stacked

        if self.fitter is None:
            self.models_ = {}
            return self

        params = self.fitter(stacked, **self.kwargs)
        if not isinstance(params, dict):
            raise ValueError('fit function must return a dictionary')
        if not all(r in params for r in self.required):
            raise ValueError('one or more required fields not returned')
        self.models_ = params
        return self

    def transform(self, data=None):
        """Apply the fitted imputation parameters to `data`.

        Parameters
        ----------
        data : DataFrame, array, list of these, or None, optional
            Data to impute. If `None` (default), re-transforms the data
            `fit` was called with. Otherwise, imputes new data using the
            already-fitted parameters (without re-fitting).

        Returns
        -------
        The imputed `data` (or the imputed fit-time data, when `data` is
        `None`), split back into a list if the input was a list.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit` has not been called yet, or a required fitted
            attribute is missing.
        """
        if self.data is None or not hasattr(self, 'models_'):
            raise NotFittedError('must fit imputer before transforming data')
        for r in self.required:
            if r not in self.models_:
                raise NotFittedError(f'missing fitted attribute: {r}')

        is_original = data is None
        if is_original:
            stacked, boundaries, single = self.data, self._boundaries, self._single
            indexes = getattr(self, '_indexes', None)
        else:
            single = not isinstance(data, list)
            datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]
            stacked, boundaries, indexes = _stack(datasets)

        if self.transformer is None:
            result = stacked
        else:
            # `_is_original_fit_data` lets a child's transformer distinguish
            # "transform the data fit() was called with" (used by
            # `fit_transform`) from "apply learned params to brand-new data"
            # (the `return_model=True` reuse path). Most children ignore it
            # (their transformer works identically either way); PPCA uses it
            # to stay byte-identical to the legacy `format_data` behavior on
            # the default path while still supporting reuse.
            merged = {**self.models_, **self.kwargs, '_is_original_fit_data': is_original}
            result = self.transformer(stacked, **merged)

        if single:
            return result
        return _split(result, boundaries, indexes)

    def fit_transform(self, data):
        """Fit the imputer on `data`, then immediately transform it.

        Parameters
        ----------
        data : DataFrame, array, or list of these
            The dataset(s) to fit and impute.

        Returns
        -------
        The imputed `data`, in the same list/single-item shape as the
        input (see `transform`).
        """
        self.fit(data)
        return self.transform()

    @property
    def is_fitted(self):
        """Whether `fit` has already been run (so a fitted instance can be
        passed back as `model=` on new data and reuse its learned
        parameters via `transform(new_data)` without re-fitting)."""
        return self.data is not None and hasattr(self, 'models_')
