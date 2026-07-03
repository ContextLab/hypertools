"""Base class for hypertools manipulators (scikit-learn compatible).

A Manipulator wraps a (fitter, transformer, required-params) triple: `fit`
runs the fitter and stores the returned dict as attributes; `transform` runs
the transformer with those params. Child classes (Normalize, ZScore, Smooth,
Resample) supply the three pieces plus their defaults.
"""
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Manipulator(BaseEstimator):
    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        self.fitter = kwargs.pop("fitter", None)
        self.transformer = kwargs.pop("transformer", None)
        self.required = kwargs.pop("required", [])
        self.kwargs = kwargs

    def fit(self, data):
        assert data is not None, ValueError("cannot manipulate an empty dataset")
        self.data = data
        if self.fitter is None:
            return
        params = self.fitter(data, **self.kwargs)
        assert isinstance(params, dict), ValueError("fit function must return a dictionary")
        assert all(r in params for r in self.required), \
            ValueError("one or more required fields not returned")
        for k, v in params.items():
            setattr(self, k, v)

    def transform(self, *_):
        if self.data is None:
            raise NotFittedError("must fit manipulator before transforming data")
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f"missing fitted attribute: {r}")
        if self.transformer is None:
            return self.data
        required_params = {r: getattr(self, r) for r in self.required}
        merged = {**required_params, **self.kwargs}
        return self.transformer(self.data, **merged)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform()
