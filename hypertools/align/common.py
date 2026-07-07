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

    def fit(self, data):
        assert data is not None, ValueError('cannot align empty dataset')
        self.data = data
        if self.fitter is None:
            return
        data = trim_and_pad(dw.unstack(self.data))
        params = self.fitter(data, **self.kwargs)
        assert isinstance(params, dict), ValueError('fit function must return a dictionary')
        assert all([r in params.keys() for r in self.required]), \
            ValueError('one or more required fields not returned')
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        if self.data is None:
            raise NotFittedError('must fit aligner before transforming data')
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f'missing fitted attribute: {r}')
        if self.transformer is None:
            return self.data
        data = trim_and_pad(dw.unstack(self.data))
        required_params = {r: getattr(self, r) for r in self.required}
        return self.transformer(data, **dw.core.update_dict(required_params, self.kwargs))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
