"""Base class for hypertools manipulators (scikit-learn compatible).

A Manipulator wraps a (fitter, transformer, required-params) triple: `fit`
runs the fitter and stores the returned dict as attributes; `transform` runs
the transformer with those params. Child classes (Normalize, ZScore, Smooth,
Resample) supply the three pieces plus their defaults.
"""
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Manipulator(BaseEstimator):
    """Base class for `Normalize`/`ZScore`/`Smooth`/`Resample`.

    Parameters
    ----------
    **kwargs
        `data`, `fitter`, `transformer`, `required` are popped off for the
        base class's own bookkeeping; everything else is stored in
        `self.kwargs` and passed to `fitter`/`transformer` on every call.
    """

    def __init__(self, **kwargs):
        self.data = kwargs.pop("data", None)
        self.fitter = kwargs.pop("fitter", None)
        self.transformer = kwargs.pop("transformer", None)
        self.required = kwargs.pop("required", [])
        self.kwargs = kwargs

    @property
    def is_fitted(self):
        """Whether `fit`/`fit_transform` has already been run.

        Lets a fitted `Manipulator` (returned from an earlier
        `hypertools.manip.manip.manip(..., return_model=True)` call) be
        passed back in as `model=` on NEW data and reuse its learned
        parameters (e.g. `ZScore`'s fitted mean/std) via `transform`,
        without re-fitting.
        """
        return self.data is not None

    def fit(self, data):
        """Fit this manipulator's parameters on `data`; stores them as
        attributes (named by `self.required`)."""
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

    def transform(self, new_data=None):
        """Apply the fitted parameters to `new_data`.

        Parameters
        ----------
        new_data : DataFrame, array, list of these, or None
            Data to transform with the already-fitted parameters. `None`
            (default) replays the fit-time data itself (`fit_transform`'s
            behavior).

        Returns
        -------
        The transformed `new_data` (or the transformed fit-time data, when
        `new_data` is `None`).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet, or a
            required fitted attribute is missing.
        """
        if self.data is None:
            raise NotFittedError("must fit manipulator before transforming data")
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f"missing fitted attribute: {r}")
        data_to_use = self.data if new_data is None else new_data
        if self.transformer is None:
            return data_to_use
        required_params = {r: getattr(self, r) for r in self.required}
        merged = {**required_params, **self.kwargs}
        return self.transformer(data_to_use, **merged)

    def fit_transform(self, data):
        """Fit on `data`, then transform it (equivalent to `fit(data)`
        followed by `transform(data)`)."""
        self.fit(data)
        return self.transform(data)
