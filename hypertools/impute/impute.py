"""hyp.impute dispatcher: resolve an imputer spec and fit_transform it.

Mirrors `hypertools.manip.manip`'s dispatcher shape (and, for `return_model`,
`hypertools.predict.predict`'s convention) but for same-shape missing-data
imputation. Wrapped by datawrangler's funnel so any input (array / DataFrame
/ list / text / polars) arrives as DataFrame(s); the resolved Imputer
(DataFrame-based) is applied directly.

Model specs may be: a registered name (`IMPUTERS`' `__name__`\\ s), a dict in
either `{'model': ..., 'params': {...}}` or the fork-style
`{'model': ..., 'args': [...], 'kwargs': {...}}` form, an Imputer subclass,
or an Imputer instance. An instance that has already been fit
(`instance.is_fitted`) is routed to `Imputer.transform` instead of
`fit_transform` -- the no-re-fitting path behind `return_model=True`: a
fitted imputer returned by an earlier `impute(..., return_model=True)` call
can be passed back as `model=` on NEW data without re-estimating its learned
parameters.
"""
import warnings

import datawrangler as dw

from .common import Imputer
from .ppca import PPCA
from .sklearn_imputers import SimpleImputer, KNNImputer, IterativeImputer
from .kalman import Kalman
from ..core.shared import unpack_model


IMPUTERS = [PPCA, SimpleImputer, KNNImputer, IterativeImputer, Kalman]


def _supported_names():
    return [m.__name__ for m in IMPUTERS]


@dw.decorate.funnel
def impute(data, model='PPCA', return_model=False, **kwargs):
    """Fill missing (NaN) values in `data`, preserving its shape.

    Parameters
    ----------
    data : DataFrame/array or list of these
        Dataset(s) to impute. A list is stacked (row-wise) and imputed
        jointly, then split back into a list matching the input structure.

    model : str, dict, class, or Imputer instance
        Which imputer to use (default: 'PPCA', matching the pre-1.0
        `format_data` default). A string is one of `IMPUTERS`' names (PPCA,
        SimpleImputer, KNNImputer, IterativeImputer, Kalman). A dict may be
        `{'model': ..., 'params': {...}}` or
        `{'model': ..., 'args': [...], 'kwargs': {...}}`. A class or an
        already-constructed (unfitted) instance is used directly. An
        ALREADY-FITTED Imputer instance (returned from a previous
        `return_model=True` call) is applied to `data` via `transform`
        (its learned parameters are reused -- not re-estimated).

    return_model : bool
        If True, also return the fitted (or reused) Imputer instance, so it
        can be passed back as `model=` on future calls with new data
        (default: False).

    **kwargs
        Passed through to the imputer's constructor when `model` resolves
        to a class (ignored when `model` is already an instance).

    Returns
    -------
    The imputed data (and the fitted Imputer if return_model=True). Lists
    in, lists out: a single input dataset returns a single imputed
    DataFrame.
    """
    args = []
    if isinstance(model, dict) and 'kwargs' not in model and 'args' not in model:
        # {'model': ..., 'params': {...}} form: unpack before handing the
        # inner model spec to unpack_model (which only auto-unpacks the
        # fork-style {'model', 'args', 'kwargs'} triple). Warn here (round17
        # Task 6 fix): this shortcut used to bypass unpack_model entirely
        # for this dict shape, silently swallowing its DeprecationWarning.
        if 'params' in model:
            warnings.warn(
                "{'model': ..., 'params': {...}} is deprecated; use "
                "{'model': ..., 'args': [...], 'kwargs': {...}} instead",
                DeprecationWarning, stacklevel=2)
        kwargs = {**dict(model.get('params', {})), **kwargs}
        model = model['model']

    resolved = unpack_model(model, valid=IMPUTERS, parent_class=Imputer)

    if isinstance(resolved, type):
        resolved = resolved(*args, **kwargs)
    elif isinstance(resolved, dict):
        cls = resolved['model']
        resolved = cls(*resolved.get('args', []), **resolved.get('kwargs', {}))
    elif isinstance(resolved, str):
        raise ValueError(
            f'unknown impute model {resolved!r}; supported names: '
            f'{", ".join(_supported_names())} (or pass a dict '
            "{'model': ..., 'params': {...}}, an Imputer subclass, or an "
            'Imputer instance directly)')

    if isinstance(resolved, Imputer) and resolved.is_fitted:
        result = resolved.transform(data)
    else:
        result = resolved.fit_transform(data)

    return (result, resolved) if return_model else result
