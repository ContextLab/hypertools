"""Base class for hypertools manipulators (scikit-learn compatible).

A Manipulator wraps a (fitter, transformer, required-params) triple: `fit`
runs the fitter and stores the returned dict as attributes; `transform` runs
the transformer with those params. Child classes (Normalize, ZScore, Smooth,
Resample, Delay) supply the three pieces plus their defaults.
"""
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class Manipulator(BaseEstimator):
    """Base class for `Normalize`/`ZScore`/`Smooth`/`Resample`/`Delay`.

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
        self.inverter = kwargs.pop("inverter", None)
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
        attributes (named by `self.required`).

        Raises
        ------
        ValueError
            If `data` is `None`, if the fitter does not return a dict, or
            if any name in `self.required` is missing from the returned
            dict. (Real raises -- these used to be ``assert cond,
            ValueError(...)``, which raised `AssertionError` and was
            stripped entirely under ``python -O``; 2026-07 release audit,
            final wave item 8.)
        """
        if data is None:
            from ..core.shared import no_observations_message
            raise ValueError(
                no_observations_message('manipulate', 'data is None'))
        self.data = data
        if self.fitter is None:
            return
        params = self.fitter(data, **self.kwargs)
        if not isinstance(params, dict):
            raise ValueError(
                f'{type(self).__name__} fit function must return a '
                f'dictionary of fitted parameters; got '
                f'{type(params).__name__}')
        missing = [r for r in self.required if r not in params]
        if missing:
            raise ValueError(
                f'{type(self).__name__} fit function did not return '
                f"required field(s): {', '.join(missing)}")
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
        NotImplementedError
            If this manipulator was fit row-wise (``axis=1``) and
            `new_data` is different data than it was fit on: the fitted
            statistics are per-ROW of the FIT-time data, so applying them
            positionally to unrelated rows is ill-defined (mirroring the
            `inverse_transform` restriction; audit F14-012).
        """
        if self.data is None:
            raise NotFittedError("must fit manipulator before transforming data")
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f"missing fitted attribute: {r}")
        # refuse to replay row-wise (axis=1) fit-time statistics onto NEW data
        # (audit F14-012): a fitted axis=1 ZScore/Normalize stores one
        # statistic per fit-time ROW, and silently broadcasting those onto a
        # different dataset's rows corrupts it without warning. Manipulators
        # whose transform re-derives everything from the data being
        # transformed (e.g. Resample) set `_stateless_transform = True` and
        # are exempt. Replaying the fit-time data itself (new_data=None, or
        # fit_transform's internal call) is always fine.
        if (new_data is not None and new_data is not self.data
                and getattr(self, 'transpose', False)
                and not getattr(self, '_stateless_transform', False)):
            raise NotImplementedError(
                f"applying a fitted row-wise (axis=1) {type(self).__name__} "
                "to new data is ill-defined: its fitted statistics are "
                "per-row of the fit-time data. Re-fit on the new data "
                f"instead, e.g. hyp.manip(new_data, "
                f"model='{type(self).__name__}', axis=1).")
        data_to_use = self.data if new_data is None else new_data
        if self.transformer is None:
            return data_to_use
        required_params = {r: getattr(self, r) for r in self.required}
        merged = {**required_params, **self.kwargs}
        return self.transformer(data_to_use, **merged)

    def inverse_transform(self, data):
        """Undo this manipulator's transform on `data`, when it is invertible.

        Invertible manipulators (`ZScore`, `Normalize`) supply an `inverter`
        that reconstructs the pre-transform values from their fitted
        parameters (mean/std, or min-max baseline/peak). Lossy ones
        (`Smooth`, `Resample`, `Delay`) have no inverter and raise
        `NotImplementedError`. Lets a `hypertools.Pipeline` round-trip
        `inverse_transform` through a leading `ZScore`/`Normalize` step.

        Parameters
        ----------
        data : DataFrame or array
            Data in this manipulator's OUTPUT space, to map back to its
            input space.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If `fit`/`fit_transform` has not been called yet.
        NotImplementedError
            If this manipulator is not invertible.
        """
        if self.data is None:
            raise NotFittedError("must fit manipulator before inverse-transforming data")
        if self.inverter is None:
            raise NotImplementedError(
                f"{type(self).__name__} is not invertible (no inverse_transform)")
        for r in self.required:
            if not hasattr(self, r):
                raise NotFittedError(f"missing fitted attribute: {r}")
        required_params = {r: getattr(self, r) for r in self.required}
        merged = {**required_params, **self.kwargs}
        return self.inverter(data, **merged)

    def fit_transform(self, data):
        """Fit on `data`, then transform it (equivalent to `fit(data)`
        followed by `transform(data)`)."""
        self.fit(data)
        return self.transform(data)
