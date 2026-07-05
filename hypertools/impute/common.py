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


def _as_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(np.asarray(data))


def _stack(datasets):
    if len(datasets) == 1:
        return datasets[0], [len(datasets[0])]
    stacked = pd.concat(datasets, axis=0, ignore_index=True)
    return stacked, [len(d) for d in datasets]


def _split(stacked, boundaries):
    if len(boundaries) == 1:
        return [stacked]
    bounds = np.cumsum(boundaries[:-1])
    parts = np.split(stacked.to_numpy(), bounds, axis=0)
    return [pd.DataFrame(p, columns=stacked.columns) for p in parts]


class Imputer(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop('data', None)
        self.fitter = kwargs.pop('fitter', None)
        self.transformer = kwargs.pop('transformer', None)
        self.required = kwargs.pop('required', [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError('cannot impute an empty dataset')
        single = not isinstance(data, list)
        datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]
        stacked, boundaries = _stack(datasets)

        self._single = single
        self._boundaries = boundaries
        self.data = stacked

        if self.fitter is None:
            self.models_ = {}
            return self

        params = self.fitter(stacked, **self.kwargs)
        assert isinstance(params, dict), ValueError('fit function must return a dictionary')
        assert all(r in params for r in self.required), \
            ValueError('one or more required fields not returned')
        self.models_ = params
        return self

    def transform(self, data=None):
        if self.data is None or not hasattr(self, 'models_'):
            raise NotFittedError('must fit imputer before transforming data')
        for r in self.required:
            if r not in self.models_:
                raise NotFittedError(f'missing fitted attribute: {r}')

        is_original = data is None
        if is_original:
            stacked, boundaries, single = self.data, self._boundaries, self._single
        else:
            single = not isinstance(data, list)
            datasets = [_as_dataframe(data)] if single else [_as_dataframe(d) for d in data]
            stacked, boundaries = _stack(datasets)

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
        return _split(result, boundaries)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()

    @property
    def is_fitted(self):
        """Whether `fit` has already been run (so a fitted instance can be
        passed back as `model=` on new data and reuse its learned
        parameters via `transform(new_data)` without re-fitting)."""
        return self.data is not None and hasattr(self, 'models_')
